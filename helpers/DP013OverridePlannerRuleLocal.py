import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")

col_mapping = {"System Assigned Algorithm List": str}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    GlobalAlgoList,
    RuleNos,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in RuleNos[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                Grains=Grains,
                GlobalAlgoList=GlobalAlgoList,
                RuleNos=RuleNos,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    Grains,
    GlobalAlgoList,
    RuleNos,
    df_keys,
):
    plugin_name = "DP013OverridePlannerRuleLocal"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    version_col = "Version.[Version Name]"
    stat_rule_col = "Stat Rule.[Stat Rule]"
    assigned_rule_col = "Assigned Rule"
    system_algo_list_col = "Stat Rule.[System Algorithm List]"
    planner_algo_list_col = "Planner Algorithm List"
    system_assigned_algo_list_col = "System Assigned Algorithm List"

    # split on delimiter and obtain grains
    forecast_level = Grains.split(",")

    # remove leading/trailing spaces if any
    forecast_level = [x.strip() for x in forecast_level]

    logger.debug(f"forecast_level : {forecast_level}")

    LocalAlgoList_cols = [version_col] + forecast_level + [system_assigned_algo_list_col]
    LocalAlgoList = pd.DataFrame(columns=LocalAlgoList_cols)
    try:
        req_cols = [version_col] + forecast_level + [assigned_rule_col]
        RuleNos = RuleNos[req_cols]

        # filter out NAs
        RuleNos = RuleNos[RuleNos[assigned_rule_col].notna()]

        if len(RuleNos) == 0:
            logger.warning("RuleNos is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return LocalAlgoList

        req_cols = [stat_rule_col, planner_algo_list_col, system_algo_list_col]
        GlobalAlgoList = GlobalAlgoList[req_cols]

        # consider system algorithm list if planner is null
        GlobalAlgoList[planner_algo_list_col] = np.where(
            GlobalAlgoList[planner_algo_list_col].isna(),
            GlobalAlgoList[system_algo_list_col],
            GlobalAlgoList[planner_algo_list_col],
        )

        if len(GlobalAlgoList) == 0:
            logger.warning("GlobalAlgoList is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return LocalAlgoList

        logger.debug("Joining RuleNos with GlobalAlgoList ...")

        # join rule dataframe with global algo list
        LocalAlgoList = pd.merge(
            RuleNos,
            GlobalAlgoList,
            how="left",
            left_on=assigned_rule_col,
            right_on=stat_rule_col,
        )

        LocalAlgoList.drop(columns=[assigned_rule_col, stat_rule_col], inplace=True)
        LocalAlgoList.drop_duplicates(inplace=True)
        LocalAlgoList.rename(
            columns={planner_algo_list_col: system_assigned_algo_list_col},
            inplace=True,
        )

        # filter relevant columns
        LocalAlgoList = LocalAlgoList[LocalAlgoList_cols]
        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        LocalAlgoList = pd.DataFrame(columns=LocalAlgoList_cols)
    return LocalAlgoList

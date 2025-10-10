import logging

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

col_mapping = {
    "Stat Algorithm Parameter Association": float,
    "System Stat Algorithm Association": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Parameters,
    OverrideFlags,
    AlgoList,
    Grains,
    df_keys,
):
    try:
        AlgoAssociationList = list()
        ParameterAssociationList = list()
        for the_iteration in AlgoList[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_algo_assoc, the_parameter_assoc = decorated_func(
                Parameters=Parameters,
                OverrideFlags=OverrideFlags,
                AlgoList=AlgoList,
                Grains=Grains,
                df_keys=df_keys,
            )

            AlgoAssociationList.append(the_algo_assoc)
            ParameterAssociationList.append(the_parameter_assoc)

        AlgoAssociation = concat_to_dataframe(AlgoAssociationList)
        ParameterAssociation = concat_to_dataframe(ParameterAssociationList)

    except Exception as e:
        logger.exception(e)
        AlgoAssociation, ParameterAssociation = None
    return AlgoAssociation, ParameterAssociation


def processIteration(
    Parameters,
    OverrideFlags,
    AlgoList,
    Grains,
    df_keys,
):
    plugin_name = "DP012SetupAlgoAssociation"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    assigned_algo_list_col = "Assigned Algorithm List"
    system_stat_algo_association_col = "System Stat Algorithm Association"
    stat_algorithm_col = "Stat Algorithm.[Stat Algorithm]"
    version_col = "Version.[Version Name]"
    stat_parameter_col = "Stat Parameter.[Stat Parameter]"
    stat_algo_parameter_association_col = "Stat Algorithm Parameter Association"

    # split on delimiter and obtain grains
    forecast_level = Grains.split(",")

    # remove leading/trailing spaces if any
    forecast_level = [x.strip() for x in forecast_level]

    logger.info(f"forecast_level : {forecast_level}")

    AlgoAssociation_cols = (
        [version_col] + forecast_level + [stat_algorithm_col, system_stat_algo_association_col]
    )

    ParameterAssociation_cols = (
        [version_col]
        + forecast_level
        + [
            stat_algorithm_col,
            stat_parameter_col,
            stat_algo_parameter_association_col,
        ]
    )

    AlgoAssociation = pd.DataFrame(columns=AlgoAssociation_cols)
    ParameterAssociation = pd.DataFrame(columns=ParameterAssociation_cols)
    try:
        # filter out NAs
        AlgoList = AlgoList[AlgoList[assigned_algo_list_col].notna()]

        if len(AlgoList) == 0 or len(OverrideFlags) == 0:
            logger.warning("AlgoList/OverrideFlags is empty, returning empty dataframe")
            return AlgoAssociation, ParameterAssociation

        logger.debug("Merging AlgoList with OverrideFlags ...")

        # merge AlgoList with OverrideFlags to filter out the intersections which need to be processed
        AlgoList = AlgoList.merge(OverrideFlags, on=[version_col] + forecast_level, how="inner")
        if len(AlgoList) == 0:
            logger.warning(
                "No records left after joining AlgoList with OverrideFlags, returning empty dataframe"
            )
            return AlgoAssociation, ParameterAssociation

        all_algo_association = list()

        logger.info(f"Number of intersections : {AlgoList.groupby(forecast_level).ngroups}")

        # for every intersection, split the assigned algo list column and form the parameter association
        for the_name, the_group in AlgoList.groupby([version_col] + forecast_level):
            logger.debug(f"the_name : {the_name}")

            try:
                # get algo list
                the_algo_list = the_group[assigned_algo_list_col].unique()[0]

                # split on delimiter and form list
                the_algo_list = [x.strip() for x in the_algo_list.split(",")]

                # repeat rows as many as algos
                the_algo_association = pd.concat([the_group] * len(the_algo_list))

                # add the algo column and association col
                the_algo_association[stat_algorithm_col] = the_algo_list
                the_algo_association[system_stat_algo_association_col] = 1.0

                # append to master list
                all_algo_association.append(the_algo_association)
            except Exception as e:
                logger.exception(e)

        # convert to dataframe
        AlgoAssociation = concat_to_dataframe(all_algo_association)

        AlgoAssociation = AlgoAssociation[AlgoAssociation_cols]

        logger.debug(f"AlgoAssociation.shape : {AlgoAssociation.shape}")

        req_cols = [
            stat_algorithm_col,
            stat_parameter_col,
            stat_algo_parameter_association_col,
        ]
        Parameters = Parameters[req_cols]

        logger.debug("Merging AlgoAssociation with ParameterAssociation ...")
        ParameterAssociation = AlgoAssociation.merge(Parameters, on=stat_algorithm_col, how="inner")

        # filter relevant columns
        ParameterAssociation = ParameterAssociation[ParameterAssociation_cols]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        AlgoAssociation = pd.DataFrame(columns=AlgoAssociation_cols)
        ParameterAssociation = pd.DataFrame(columns=ParameterAssociation_cols)
    return AlgoAssociation, ParameterAssociation

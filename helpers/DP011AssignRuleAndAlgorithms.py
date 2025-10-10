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

from helpers.assign_forecast_rule import assign_rules
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
    "System Assigned Rule": str,
    "System Assigned Algorithm List": str,
}


def sort_rule_description(rule: str, delimiter: str):
    try:
        desc_list = [x.strip() for x in rule.split(delimiter)]

        # sort the newly generated list
        desc_list.sort()
        # join it back to string format
        sorted_rule = "|".join(desc_list)
    except Exception as e:
        logger.exception(f"Exception {e} occured during sort rule function")
        sorted_rule = rule
    return sorted_rule


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    SegmentedData,
    Rules,
    ForecastGenTimeBucket,
    df_keys,
):
    try:
        ForecastRuleList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_forecast_rule = decorated_func(
                Grains=Grains,
                SegmentedData=SegmentedData,
                Rules=Rules,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                df_keys=df_keys,
            )

            ForecastRuleList.append(the_forecast_rule)

        ForecastRule = concat_to_dataframe(ForecastRuleList)
    except Exception as e:
        logger.exception(e)
        ForecastRule = None

    return ForecastRule


def processIteration(
    Grains,
    SegmentedData,
    Rules,
    ForecastGenTimeBucket,
    df_keys,
):
    plugin_name = "DP011AssignRuleAndAlgorithms"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    stat_rule_col = "Stat Rule.[Stat Rule]"
    planner_vol_segment_col = "Planner Volume Segment"
    planner_cov_segment_col = "Planner COV Segment"
    planner_intermittency_col = "Planner Intermittency"
    planner_plc_col = "Planner PLC"
    planner_trend_col = "Planner Trend"
    planner_seasonality_col = "Planner Seasonality"
    planner_algo_list_col = "Planner Algorithm List"
    planner_los_col = "Planner Length of Series"
    stat_rule_sys_rule_desc_col = "Stat Rule.[System Rule Description]"
    stat_rule_sys_algo_list = "Stat Rule.[System Algorithm List]"
    forecast_gen_time_bucket_col = "Forecast Generation Time Bucket"

    intermittent_l1_col = "Intermittent L1"
    plc_status_l1_col = "PLC Status L1"
    vol_segment_l1_col = "Volume Segment L1"
    cov_segment_l1_col = "COV Segment L1"
    trend_l1_col = "Trend L1"
    seasonality_l1_col = "Seasonality L1"
    los_l1_col = "Length of Series L1"

    # col names for sys default
    sys_intermittent_col = "Intermittency"
    sys_plc_col = "PLC"
    sys_los_col = "Length of Series"
    sys_cov_segment_col = "COV Segment"
    sys_vol_segment_col = "Volume Segment"
    sys_trend_col = "Trend"
    sys_seasonality_col = "Seasonality"

    # output measures
    sys_assigned_rule_col = "System Assigned Rule"
    sys_assigned_algo_list_col = "System Assigned Algorithm List"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("forecast_level : {}".format(forecast_level))

    cols_required_in_output = (
        [version_col] + forecast_level + [sys_assigned_rule_col, sys_assigned_algo_list_col]
    )
    ForecastRule = pd.DataFrame(columns=cols_required_in_output)
    try:
        # empty check
        if len(SegmentedData) == 0:
            logger.warning("Segmentation input is empty for slice {}".format(df_keys))
            logger.warning("Returning without further execution ...")
            return ForecastRule

        # select rows where the required columns are not null
        req_cols = [
            vol_segment_l1_col,
            cov_segment_l1_col,
            trend_l1_col,
            seasonality_l1_col,
            intermittent_l1_col,
            plc_status_l1_col,
        ]

        logger.debug(f"SegmentedData, shape : {SegmentedData.shape}")

        # drop rows if any of the above columns contain null
        SegmentedData.dropna(subset=req_cols, how="any", inplace=True)

        logger.debug(f"After dropping nulls, SegmentedData, shape : {SegmentedData.shape}")

        if len(SegmentedData) == 0:
            logger.warning(
                "No rows left in SegmentedData after dropping NAs, kindly check for NANs in input measures .."
            )
            return ForecastRule

        if len(Rules) == 0:
            logger.warning(
                "There are no values in Rules, please populate it via the starter dataset .."
            )
            return ForecastRule

        # converting np.nan and "nan" to N/A if present
        Rules.fillna("nan", inplace=True)

        logger.info("Assigning YES/NO values to trend/seasonality ...")

        # assign yes/no to trend/seasonality/intermittency
        SegmentedData[trend_l1_col] = np.where(
            SegmentedData[trend_l1_col].isin(["UPWARD", "DOWNWARD"]),
            "YES",
            "NO",
        )

        SegmentedData[seasonality_l1_col] = np.where(
            SegmentedData[seasonality_l1_col].isin(["Exists"]), "YES", "NO"
        )

        logger.info(
            "Extracting segmentwise conditions from system defined rule description string ..."
        )

        # sorting the rules dataframe desc column
        Rules[stat_rule_sys_rule_desc_col] = Rules[stat_rule_sys_rule_desc_col].apply(
            lambda x: sort_rule_description(x, "|")
        )

        # Extract segmentwise clauses from system rule description
        system_default_rule_conditions = Rules[stat_rule_sys_rule_desc_col].str.split(
            "|", expand=True
        )

        # obtaining rule order via the first row
        rule_value = Rules[stat_rule_sys_rule_desc_col].iloc[0]

        # splitting on | and : to get the criteria, Ex. "Intermittent: YES" becomes "Intermittent"
        sys_default_col_names = rule_value.split("|")
        sys_default_col_names = [x.split(":")[0] for x in sys_default_col_names]

        # removing trailing whitespaces
        sys_default_col_names = [x.strip() for x in sys_default_col_names]
        system_default_rule_conditions.columns = sys_default_col_names

        # Remove colname and colon prefix - Example 'Intermittent: YES' becomes 'YES'
        for the_col in sys_default_col_names:
            char_to_replace = the_col + ":"
            system_default_rule_conditions[the_col] = system_default_rule_conditions[
                the_col
            ].str.replace(char_to_replace, "")

            # strip leading and trailing zeros if any
            system_default_rule_conditions[the_col] = system_default_rule_conditions[
                the_col
            ].str.strip()

        # concat dataframes
        system_default_rule_df = pd.concat(
            [
                Rules[[stat_rule_col, stat_rule_sys_algo_list]],
                system_default_rule_conditions,
            ],
            axis=1,
        )

        # sort in rule order (so that intermittency can be evaluated first)
        system_default_rule_df.sort_values(stat_rule_col, inplace=True)

        logger.info("---------- system_default_rule_df ---------")
        logger.info(system_default_rule_df)

        # get planner rule df
        planner_rule_df = Rules[
            [
                stat_rule_col,
                planner_algo_list_col,
                planner_intermittency_col,
                planner_plc_col,
                planner_los_col,
                planner_cov_segment_col,
                planner_vol_segment_col,
                planner_trend_col,
                planner_seasonality_col,
            ]
        ]

        # sort in rule order (so that intermittency can be evaluated first)
        planner_rule_df.sort_values(stat_rule_col, inplace=True)

        logger.info("---------- planner_rule_df ---------")
        logger.info(planner_rule_df)

        # merge system and planner rules into a single dataframe
        rule_df = system_default_rule_df.merge(planner_rule_df, on=stat_rule_col)

        # define the columns to be overridden in case of null
        sys_to_planner_col_mapping = {
            stat_rule_sys_algo_list: planner_algo_list_col,
            sys_intermittent_col: planner_intermittency_col,
            sys_plc_col: planner_plc_col,
            sys_los_col: planner_los_col,
            sys_cov_segment_col: planner_cov_segment_col,
            sys_vol_segment_col: planner_vol_segment_col,
            sys_trend_col: planner_trend_col,
            sys_seasonality_col: planner_seasonality_col,
        }

        logger.info("Filling nulls in planner measures with system default values ...")

        # loop through system and planner measure, fill NAs if exist
        for (
            the_sys_measure,
            the_planner_measure,
        ) in sys_to_planner_col_mapping.items():

            logger.info(
                f"-- the_sys_measure : {the_sys_measure}, the_planner_measure : {the_planner_measure}"
            )

            # if planner measure is null, populate it from system measures
            rule_df[the_planner_measure] = np.where(
                rule_df[the_planner_measure] == "nan",
                rule_df[the_sys_measure],
                rule_df[the_planner_measure],
            )

        # drop the system columns from rule df
        rule_df.drop(list(sys_to_planner_col_mapping.keys()), axis=1, inplace=True)

        # assert no nulls are present before passing to rule assignment function
        assert ~(
            rule_df.isnull().values.any()
        ), "Nulls present in rule df, kindly inspect data/code ..."

        forecast_gen_time_bucket = ForecastGenTimeBucket[forecast_gen_time_bucket_col].unique()[0]
        logger.debug(f"forecast_gen_time_bucket : {forecast_gen_time_bucket}")

        # Convert Length of Series value into categorical value
        if "Week" in forecast_gen_time_bucket:
            los_for_2_cycles = 104
        elif "Month" in forecast_gen_time_bucket:
            los_for_2_cycles = 26
        elif "Quarter" in forecast_gen_time_bucket:
            los_for_2_cycles = 8
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {forecast_gen_time_bucket}, returning empty df"
            )
            return ForecastRule
        condition = SegmentedData[los_l1_col] < los_for_2_cycles
        logger.debug(f"Converting {los_l1_col} into categorical values ...")
        SegmentedData[los_l1_col] = np.where(condition, "< 2 Cycles", ">= 2 Cycles")

        logger.info("Shape of intersections : {}".format(SegmentedData.shape))

        logger.info("Assigning rules ...")

        seg_to_planner_col_mapping = {
            intermittent_l1_col: planner_intermittency_col,
            plc_status_l1_col: planner_plc_col,
            los_l1_col: planner_los_col,
            cov_segment_l1_col: planner_cov_segment_col,
            vol_segment_l1_col: planner_vol_segment_col,
            trend_l1_col: planner_trend_col,
            seasonality_l1_col: planner_seasonality_col,
        }

        # assign rules
        ForecastRule = assign_rules(
            segmentation_output=SegmentedData.copy(),
            rule_df=rule_df,
            column_mapping=seg_to_planner_col_mapping,
            rule_col=stat_rule_col,
            algo_col=planner_algo_list_col,
            intermittent_col=intermittent_l1_col,
            plc_col=plc_status_l1_col,
            los_col=los_l1_col,
            cov_segment_col=cov_segment_l1_col,
            vol_segment_col=vol_segment_l1_col,
            trend_col=trend_l1_col,
            seasonality_col=seasonality_l1_col,
        )
        # rename columns
        ForecastRule.rename(
            columns={
                stat_rule_col: sys_assigned_rule_col,
                planner_algo_list_col: sys_assigned_algo_list_col,
            },
            inplace=True,
        )

        # re order columns
        ForecastRule = ForecastRule[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        ForecastRule = pd.DataFrame(columns=cols_required_in_output)

    return ForecastRule

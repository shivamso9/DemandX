import logging

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import get_abs_error, get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {
    "Outlook Abs Error": float,
    "Uncons Outlook Abs Error": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    PlanningActual,
    OutlookLag1,
    UnconsOutlookLag1,
    CurrentTimePeriod,
    TimeDimension,
    PlanningGrains,
    df_keys={},
):
    plugin_name = "DP215CollabIBPAbsError"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)
    OutlookAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
        ]
        + planning_grains
        + [o9Constants.OUTLOOK_ABS_ERROR]
    )
    OutlookAbsError = pd.DataFrame(columns=OutlookAbsError_cols)
    UnconsOutlookAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
        ]
        + planning_grains
        + [o9Constants.UNCONS_OUTLOOK_ABS_ERROR]
    )
    UnconsOutlookAbsError = pd.DataFrame(columns=UnconsOutlookAbsError_cols)

    try:

        if len(OutlookLag1) != 0:
            OutlookAbsError = get_abs_error(
                source_df=OutlookLag1,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                source_measure=o9Constants.OUTLOOK_LAG1,
                actual_measure=o9Constants.ACTUAL_L0,
                output_measure=o9Constants.OUTLOOK_ABS_ERROR,
                output_cols=OutlookAbsError_cols,
            )
        if len(UnconsOutlookLag1) != 0:
            UnconsOutlookAbsError = get_abs_error(
                source_df=UnconsOutlookLag1,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                source_measure=o9Constants.UNCONS_OUTLOOK_LAG1,
                actual_measure=o9Constants.ACTUAL_L0,
                output_measure=o9Constants.UNCONS_OUTLOOK_ABS_ERROR,
                output_cols=UnconsOutlookAbsError_cols,
            )

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return OutlookAbsError, UnconsOutlookAbsError

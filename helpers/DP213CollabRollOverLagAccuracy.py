import logging

import numpy as np
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
    "Consensus Fcst W Lag Abs Error": float,
    "Published Fcst W Lag Abs Error": float,
    "Consensus Fcst M Lag Abs Error": float,
    "Published Fcst M Lag Abs Error": float,
    "Consensus Fcst PM Lag Abs Error": float,
    "Published Fcst PM Lag Abs Error": float,
    "Last 3M Accuracy": float,
    "Last 6M Accuracy": float,
    "Last 3M Accuracy LT 50% Count": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    PlanningActual,
    ConsensusFcstWLag,
    ConsensusFcstMLag,
    ConsensusFcstPMLag,
    PublishedFcstWLag,
    PublishedFcstMLag,
    PublishedFcstPMLag,
    CurrentTimePeriod,
    TimeDimension,
    PlanningGrains,
    TimeLevel="Week",
    df_keys={},
):
    plugin_name = "DP213CollabRollOverLagAccuracy"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)

    ConsensusFcstWLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.CONSENSUS_FCST_W_LAG_ABS_ERROR]
    )
    ConsensusFcstWLagAbsError = pd.DataFrame(columns=ConsensusFcstWLagAbsError_cols)
    PublishedFcstWLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.PUBLISHED_FCST_W_LAG_ABS_ERROR]
    )
    PublishedFcstWLagAbsError = pd.DataFrame(columns=PublishedFcstWLagAbsError_cols)
    ConsensusFcstMLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.CONSENSUS_FCST_M_LAG_ABS_ERROR]
    )
    ConsensusFcstMLagAbsError = pd.DataFrame(columns=ConsensusFcstMLagAbsError_cols)
    PublishedFcstMLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.PUBLISHED_FCST_M_LAG_ABS_ERROR]
    )
    PublishedFcstMLagAbsError = pd.DataFrame(columns=PublishedFcstMLagAbsError_cols)
    LastMonthsAccuracy_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.LAST_3M_ACCURACY,
            o9Constants.LAST_6M_ACCURACY,
            o9Constants.LAST_3M_ACCURACY_LT_HALF_COUNT,
        ]
    )
    LastMonthsAccuracy = pd.DataFrame(columns=LastMonthsAccuracy_cols)
    ConsensusFcstPMLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.CONSENSUS_FCST_PM_LAG_ABS_ERROR]
    )
    ConsensusFcstPMLagAbsError = pd.DataFrame(columns=ConsensusFcstPMLagAbsError_cols)
    PublishedFcstPMLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.PUBLISHED_FCST_PM_LAG_ABS_ERROR]
    )
    PublishedFcstPMLagAbsError = pd.DataFrame(columns=PublishedFcstPMLagAbsError_cols)

    try:
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.lower() == "planning month"
        if week:
            if len(ConsensusFcstWLag) != 0:
                ConsensusFcstWLagAbsError = get_abs_error(
                    source_df=ConsensusFcstWLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.WEEK,
                    time_key=o9Constants.WEEK_KEY,
                    source_measure=o9Constants.CONSENSUS_FCST_W_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.CONSENSUS_FCST_W_LAG_ABS_ERROR,
                    output_cols=ConsensusFcstWLagAbsError_cols,
                    lags=",".join(ConsensusFcstWLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )
            if len(PublishedFcstWLag) != 0:
                PublishedFcstWLagAbsError = get_abs_error(
                    source_df=PublishedFcstWLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.WEEK,
                    time_key=o9Constants.WEEK_KEY,
                    source_measure=o9Constants.PUBLISHED_FCST_W_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.PUBLISHED_FCST_W_LAG_ABS_ERROR,
                    output_cols=PublishedFcstWLagAbsError_cols,
                    lags=",".join(PublishedFcstWLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )
        if month:
            if len(ConsensusFcstMLag) != 0:
                ConsensusFcstMLagAbsError = get_abs_error(
                    source_df=ConsensusFcstMLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.MONTH,
                    time_key=o9Constants.MONTH_KEY,
                    source_measure=o9Constants.CONSENSUS_FCST_M_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.CONSENSUS_FCST_M_LAG_ABS_ERROR,
                    output_cols=ConsensusFcstMLagAbsError_cols,
                    lags=",".join(ConsensusFcstMLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )
            if len(PublishedFcstMLag) != 0:
                PublishedFcstMLagAbsError = get_abs_error(
                    source_df=PublishedFcstMLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.MONTH,
                    time_key=o9Constants.MONTH_KEY,
                    source_measure=o9Constants.PUBLISHED_FCST_M_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.PUBLISHED_FCST_M_LAG_ABS_ERROR,
                    output_cols=PublishedFcstMLagAbsError_cols,
                    lags=",".join(PublishedFcstMLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )
            LastMonthsAccuracy = ConsensusFcstMLagAbsError.copy()
            PlanningActual = PlanningActual.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, o9Constants.MONTH]],
                on=[o9Constants.PARTIAL_WEEK],
                how="inner",
            )
            PlanningActual = PlanningActual.groupby(
                [o9Constants.VERSION_NAME, o9Constants.MONTH] + planning_grains,
                as_index=False,
            ).agg({o9Constants.ACTUAL_L0: "sum"})
            LastMonthsAccuracy = LastMonthsAccuracy.merge(
                PlanningActual,
                on=[o9Constants.VERSION_NAME, o9Constants.MONTH] + planning_grains,
                how="left",
            )

            # Calculate WMAPE using np.where to handle division by zero
            LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_WMAPE] = np.where(
                (LastMonthsAccuracy[o9Constants.ACTUAL_L0] != 0)
                | (LastMonthsAccuracy[o9Constants.ACTUAL_L0].notna()),
                LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_ABS_ERROR]
                / LastMonthsAccuracy[o9Constants.ACTUAL_L0],
                0,
            )

            LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY] = np.where(
                LastMonthsAccuracy[o9Constants.ACTUAL_L0].notna(),
                np.where(
                    LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_WMAPE] <= 0,
                    0,
                    np.where(
                        LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_WMAPE] >= 1,
                        1,
                        (1 - LastMonthsAccuracy[o9Constants.CONSENSUS_FCST_M_LAG_WMAPE]),
                    ),
                ),
                np.nan,
            )

            LastMonthsAccuracy = LastMonthsAccuracy.merge(
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                on=o9Constants.MONTH,
                how="inner",
            )
            LastMonthsAccuracy = LastMonthsAccuracy.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.MONTH_KEY]
            )
            rolling_3M = (
                LastMonthsAccuracy.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.LAG]
                )[o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY]
                .rolling(window=3, min_periods=3)
                .mean()
                .reset_index()
            )
            rolling_6M = (
                LastMonthsAccuracy.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.LAG]
                )[o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY]
                .rolling(window=6, min_periods=6)
                .mean()
                .reset_index()
            )
            Last3MonthsAccuracy = LastMonthsAccuracy.merge(
                rolling_3M, left_index=True, right_on="level_9"
            )
            Last3MonthsAccuracy = Last3MonthsAccuracy.rename(
                columns={
                    o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY + "_y": o9Constants.LAST_3M_ACCURACY
                }
            )
            Last3MonthsAccuracy.columns = Last3MonthsAccuracy.columns.str.replace(
                "_x", ""
            ).str.replace("_y", "")
            Last3MonthsAccuracy = Last3MonthsAccuracy.drop(columns="level_9")
            Last3MonthsAccuracy = Last3MonthsAccuracy.loc[
                :, ~Last3MonthsAccuracy.columns.duplicated()
            ]

            Last6MonthsAccuracy = LastMonthsAccuracy.merge(
                rolling_6M, left_index=True, right_on="level_9"
            )
            Last6MonthsAccuracy = Last6MonthsAccuracy.rename(
                columns={
                    o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY + "_y": o9Constants.LAST_6M_ACCURACY
                }
            )
            Last6MonthsAccuracy.columns = Last6MonthsAccuracy.columns.str.replace(
                "_x", ""
            ).str.replace("_y", "")
            Last6MonthsAccuracy = Last6MonthsAccuracy.drop(columns="level_9")
            Last6MonthsAccuracy = Last6MonthsAccuracy.loc[
                :, ~Last6MonthsAccuracy.columns.duplicated()
            ]
            Last3MonthsAccuracy[o9Constants.LAST_3M_ACCURACY] = Last3MonthsAccuracy.groupby(
                [o9Constants.VERSION_NAME, o9Constants.LAG] + planning_grains
            )[o9Constants.LAST_3M_ACCURACY].shift()
            Last6MonthsAccuracy[o9Constants.LAST_6M_ACCURACY] = Last6MonthsAccuracy.groupby(
                [o9Constants.VERSION_NAME, o9Constants.LAG] + planning_grains
            )[o9Constants.LAST_6M_ACCURACY].shift()

            LastMonthsAccuracy = Last3MonthsAccuracy.merge(
                Last6MonthsAccuracy,
                on=[
                    o9Constants.VERSION_NAME,
                    o9Constants.LAG,
                    o9Constants.CONSENSUS_FCST_M_LAG_ABS_ERROR,
                    o9Constants.CONSENSUS_FCST_M_LAG_ACCURACY,
                    o9Constants.PLANNING_CYCLE_DATE,
                    o9Constants.MONTH,
                    o9Constants.MONTH_KEY,
                    o9Constants.CONSENSUS_FCST_M_LAG_WMAPE,
                    o9Constants.ACTUAL_L0,
                ]
                + planning_grains,
                how="outer",
            )
            LastMonthsAccuracy[o9Constants.LAST_3M_ACCURACY_LT_HALF_COUNT] = np.where(
                (
                    (LastMonthsAccuracy[o9Constants.LAST_3M_ACCURACY] < 0.5)
                    & (LastMonthsAccuracy[o9Constants.LAG].astype(int) == 1)
                ),
                1,
                np.nan,
            )
            LastMonthsAccuracy = LastMonthsAccuracy[LastMonthsAccuracy_cols]

        if pl_month:
            if len(ConsensusFcstPMLag) != 0:
                ConsensusFcstPMLagAbsError = get_abs_error(
                    source_df=ConsensusFcstPMLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.PLANNING_MONTH,
                    time_key=o9Constants.PLANNING_MONTH_KEY,
                    source_measure=o9Constants.CONSENSUS_FCST_PM_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.CONSENSUS_FCST_PM_LAG_ABS_ERROR,
                    output_cols=ConsensusFcstPMLagAbsError_cols,
                    lags=",".join(ConsensusFcstPMLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )
            if len(PublishedFcstPMLag) != 0:
                PublishedFcstPMLagAbsError = get_abs_error(
                    source_df=PublishedFcstPMLag,
                    Actuals=PlanningActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.MONTH,
                    time_key=o9Constants.MONTH_KEY,
                    source_measure=o9Constants.PUBLISHED_FCST_PM_LAG,
                    actual_measure=o9Constants.ACTUAL_L0,
                    output_measure=o9Constants.PUBLISHED_FCST_PM_LAG_ABS_ERROR,
                    output_cols=PublishedFcstPMLagAbsError_cols,
                    lags=",".join(PublishedFcstPMLag[o9Constants.LAG].unique().astype(str)),
                    df_keys=df_keys,
                )

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        ConsensusFcstWLagAbsError = pd.DataFrame(columns=ConsensusFcstWLagAbsError_cols)
        ConsensusFcstMLagAbsError = pd.DataFrame(columns=ConsensusFcstMLagAbsError_cols)
        ConsensusFcstPMLagAbsError = pd.DataFrame(columns=ConsensusFcstPMLagAbsError_cols)
        PublishedFcstWLagAbsError = pd.DataFrame(columns=PublishedFcstWLagAbsError_cols)
        PublishedFcstMLagAbsError = pd.DataFrame(columns=PublishedFcstMLagAbsError_cols)
        PublishedFcstPMLagAbsError = pd.DataFrame(columns=PublishedFcstPMLagAbsError_cols)
        LastMonthsAccuracy = pd.DataFrame(columns=LastMonthsAccuracy_cols)

    return (
        ConsensusFcstWLagAbsError,
        PublishedFcstWLagAbsError,
        ConsensusFcstMLagAbsError,
        PublishedFcstMLagAbsError,
        ConsensusFcstPMLagAbsError,
        PublishedFcstPMLagAbsError,
        LastMonthsAccuracy,
    )

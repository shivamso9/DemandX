import logging

import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data

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
    "Stat Fcst Abs Error": float,
    "Stat Fcst PL W Lag Abs Error": float,
    "Stat Fcst PL M Lag Abs Error": float,
    "Stat Fcst PL PM Lag Abs Error": float,
    "Stat Fcst PL Q Lag Abs Error": float,
    "Stat Fcst PL PQ Lag Abs Error": float,
}


def get_relevant_statfcst_lag(
    relevant_time_name,
    relevant_time_key,
    lag_df,
    TimeDimension,
    CurrentTimePeriod,
    AccuracyWindow,
):
    """
    Return Stat Fcst Lag measures for the accuracy windows, for the relevant time bucket.

    relevant_time_name : Time.[Week]
    relevant_time_key : Time.[WeekKey]
    lag_df : StatFcstPLWLag
    AccuracyWindow : 1
    """
    relevant_time_cols = [
        o9Constants.PARTIAL_WEEK,
        relevant_time_name,
        relevant_time_key,
    ]
    time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()
    time_attribute_dict = {relevant_time_name: relevant_time_key}
    current_planning_cycle = CurrentTimePeriod[relevant_time_name].values[0]
    logger.info(f"CURRENT PLANNING CYCLE : {current_planning_cycle}")
    logger.info(f"ACCURACY WINDOW SIZE : {AccuracyWindow} periods")
    lag_df = lag_df.merge(
        time_mapping.rename(columns={relevant_time_name: "Planning Cycle"}),
        left_on="Planning Cycle.[PlanningCycleDateKey]",
        right_on=relevant_time_key,
        how="inner",
    )
    accuracy_window_periods = get_n_time_periods(
        current_planning_cycle,
        -AccuracyWindow,
        time_mapping,
        time_attribute_dict,
        include_latest_value=False,
    )
    logger.info(f"ACCURACY WINDOW : {accuracy_window_periods}")
    lag_df = lag_df[lag_df[relevant_time_name].isin(accuracy_window_periods)]
    return lag_df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    PlanningActual,
    StatFcstPLWLag,
    StatFcstPLMLag,
    StatFcstPLPMLag,
    StatFcstPLQLag,
    StatFcstPLPQLag,
    CurrentTimePeriod,
    TimeDimension,
    PlanningGrains,
    StatBucketWeight,
    TimeLevel="Week",
    AccuracyWindow="1",
    df_keys={},
):
    """
    Calculate the Abs errors between the Lag measures and the Actual values for the time range of Accuracy Windows.

    AccuracyWindow : n - gives n buckets prior to the current planning cycle
    TimeLevel :  Forecast Generation Time Bucket
    """
    plugin_name = "DP213CalculateRollOverAccuracy"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)

    StatFcstLag1_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_ABS_ERROR]
    )
    StatFcstAbsError = pd.DataFrame(columns=StatFcstLag1_cols)

    StatFcstWLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR,
        ]
    )
    StatFcstWLagAbsError = pd.DataFrame(columns=StatFcstWLag_cols)
    StatFcstMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR,
        ]
    )
    StatFcstMLagAbsError = pd.DataFrame(columns=StatFcstMLag_cols)
    StatFcstPMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR,
        ]
    )
    StatFcstPMLagAbsError = pd.DataFrame(columns=StatFcstPMLag_cols)
    StatFcstQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR,
        ]
    )
    StatFcstQLagAbsError = pd.DataFrame(columns=StatFcstQLag_cols)
    StatFcstPQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR,
        ]
    )
    StatFcstPQLagAbsError = pd.DataFrame(columns=StatFcstPQLag_cols)

    try:
        if PlanningActual.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return (
                StatFcstAbsError,
                StatFcstWLagAbsError,
                StatFcstMLagAbsError,
                StatFcstPMLagAbsError,
                StatFcstQLagAbsError,
                StatFcstPQLagAbsError,
            )
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.replace(" ", "").lower() == "planningmonth"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.replace(" ", "").lower() == "planningquarter"
        AccuracyWindow = int(AccuracyWindow)

        # merge StatBucketWeight with TimeDimension to get all relevant columns
        StatBucketWeight = StatBucketWeight.merge(
            TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
        )

        if len(StatFcstPLWLag) != 0 and week:
            current_week = CurrentTimePeriod[o9Constants.WEEK_KEY].unique()[0]
            StatFcstPLWLag = StatFcstPLWLag.merge(
                TimeDimension[[o9Constants.WEEK, o9Constants.WEEK_KEY]].drop_duplicates(),
                on=o9Constants.WEEK,
                how="inner",
            )
            StatFcstPLWLag = StatFcstPLWLag[StatFcstPLWLag[o9Constants.WEEK_KEY] <= current_week]
            logger.info(
                f"Maximum Lag for which Stat Fcst PL W Lag is available : {max(StatFcstPLWLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstPLWLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.WEEK
            relevant_time_key = o9Constants.WEEK_KEY
            StatFcstPLWLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstPLWLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstPLWLag) == 0:
                logger.warning(
                    "Stat Fcst PL W Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstAbsError,
                    StatFcstWLagAbsError,
                    StatFcstMLagAbsError,
                    StatFcstPMLagAbsError,
                    StatFcstQLagAbsError,
                    StatFcstPQLagAbsError,
                )

            StatFcstWLagAbsError = get_abs_error(
                source_df=StatFcstPLWLag,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.WEEK,
                time_key=o9Constants.WEEK_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PL_W_LAG,
                actual_measure=o9Constants.ACTUAL,
                output_measure=o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR,
                output_cols=StatFcstWLag_cols,
            )
            if len(StatFcstWLagAbsError) != 0:
                StatFcstAbsError = StatFcstWLagAbsError[
                    (StatFcstWLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]
                StatFcstAbsError = StatFcstAbsError.rename(
                    columns={
                        o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR: o9Constants.STAT_FCST_ABS_ERROR
                    }
                )

        if len(StatFcstPLMLag) != 0 and month:
            current_month = CurrentTimePeriod[o9Constants.MONTH_KEY].unique()[0]
            StatFcstPLMLag = StatFcstPLMLag.merge(
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                on=o9Constants.MONTH,
                how="inner",
            )
            StatFcstPLMLag = StatFcstPLMLag[StatFcstPLMLag[o9Constants.MONTH_KEY] <= current_month]
            logger.info(
                f"Maximum Lag for which Stat Fcst PL M Lag is available : {max(StatFcstPLMLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstPLMLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.MONTH
            relevant_time_key = o9Constants.MONTH_KEY
            StatFcstPLMLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstPLMLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstPLMLag) == 0:
                logger.warning(
                    "Stat Fcst PL M Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstAbsError,
                    StatFcstWLagAbsError,
                    StatFcstMLagAbsError,
                    StatFcstPMLagAbsError,
                    StatFcstQLagAbsError,
                    StatFcstPQLagAbsError,
                )
            StatFcstMLagAbsError = get_abs_error(
                source_df=StatFcstPLMLag,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PL_M_LAG,
                actual_measure=o9Constants.ACTUAL,
                output_measure=o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR,
                output_cols=StatFcstMLag_cols,
            )
            if len(StatFcstMLagAbsError) != 0:
                StatFcstAbsError = StatFcstMLagAbsError[
                    (StatFcstMLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]
                StatFcstAbsError = disaggregate_data(
                    source_df=StatFcstAbsError,
                    source_grain=o9Constants.MONTH,
                    target_grain=o9Constants.WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR,
                    ],
                )
                StatFcstAbsError = StatFcstAbsError.rename(
                    columns={
                        o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR: o9Constants.STAT_FCST_ABS_ERROR
                    }
                )

        if len(StatFcstPLPMLag) != 0 and pl_month:
            current_pl_month = CurrentTimePeriod[o9Constants.PLANNING_MONTH_KEY].unique()[0]
            StatFcstPLPMLag = StatFcstPLPMLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_MONTH, o9Constants.PLANNING_MONTH_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_MONTH,
                how="inner",
            )
            StatFcstPLPMLag = StatFcstPLPMLag[
                StatFcstPLPMLag[o9Constants.PLANNING_MONTH_KEY] <= current_pl_month
            ]
            logger.info(
                f"Maximum Lag for which Stat Fcst PL PM Lag is available : {max(StatFcstPLPMLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstPLPMLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.PLANNING_MONTH
            relevant_time_key = o9Constants.PLANNING_MONTH_KEY
            StatFcstPLPMLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstPLPMLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstPLPMLag) == 0:
                logger.warning(
                    "Stat Fcst PL PM Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstAbsError,
                    StatFcstWLagAbsError,
                    StatFcstMLagAbsError,
                    StatFcstPMLagAbsError,
                    StatFcstQLagAbsError,
                    StatFcstPQLagAbsError,
                )
            StatFcstPMLagAbsError = get_abs_error(
                source_df=StatFcstPLPMLag,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_MONTH,
                time_key=o9Constants.PLANNING_MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PL_PM_LAG,
                actual_measure=o9Constants.ACTUAL,
                output_measure=o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR,
                output_cols=StatFcstPMLag_cols,
            )
            if len(StatFcstPMLagAbsError) != 0:
                StatFcstAbsError = StatFcstPMLagAbsError[
                    (StatFcstPMLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]
                StatFcstAbsError = disaggregate_data(
                    source_df=StatFcstAbsError,
                    source_grain=o9Constants.PLANNING_MONTH,
                    target_grain=o9Constants.WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR,
                    ],
                )
                StatFcstAbsError = StatFcstAbsError.rename(
                    columns={
                        o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR: o9Constants.STAT_FCST_ABS_ERROR
                    }
                )

        if len(StatFcstPLQLag) != 0 and quarter:
            current_quarter = CurrentTimePeriod[o9Constants.QUARTER_KEY].unique()[0]
            StatFcstPLQLag = StatFcstPLQLag.merge(
                TimeDimension[[o9Constants.QUARTER, o9Constants.QUARTER_KEY]].drop_duplicates(),
                on=o9Constants.QUARTER,
                how="inner",
            )
            StatFcstPLQLag = StatFcstPLQLag[
                StatFcstPLQLag[o9Constants.QUARTER_KEY] <= current_quarter
            ]
            logger.info(
                f"Maximum Lag for which Stat Fcst PL Q Lag is available : {max(StatFcstPLQLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstPLQLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.QUARTER
            relevant_time_key = o9Constants.QUARTER_KEY
            StatFcstPLQLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstPLQLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstPLQLag) == 0:
                logger.warning(
                    "Stat Fcst PL Q Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstAbsError,
                    StatFcstWLagAbsError,
                    StatFcstMLagAbsError,
                    StatFcstPMLagAbsError,
                    StatFcstQLagAbsError,
                    StatFcstPQLagAbsError,
                )
            StatFcstQLagAbsError = get_abs_error(
                source_df=StatFcstPLQLag,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.QUARTER,
                time_key=o9Constants.QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PL_Q_LAG,
                actual_measure=o9Constants.ACTUAL,
                output_measure=o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR,
                output_cols=StatFcstQLag_cols,
            )
            if len(StatFcstQLagAbsError) != 0:
                StatFcstAbsError = StatFcstQLagAbsError[
                    (StatFcstQLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]
                StatFcstAbsError = disaggregate_data(
                    source_df=StatFcstAbsError,
                    source_grain=o9Constants.QUARTER,
                    target_grain=o9Constants.WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR,
                    ],
                )
                StatFcstAbsError = StatFcstAbsError.rename(
                    columns={
                        o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR: o9Constants.STAT_FCST_ABS_ERROR
                    }
                )

        if len(StatFcstPLPQLag) != 0 and pl_quarter:
            current_pl_quarter = CurrentTimePeriod[o9Constants.PLANNING_QUARTER_KEY].unique()[0]
            StatFcstPLPQLag = StatFcstPLPQLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_QUARTER, o9Constants.PLANNING_QUARTER_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_QUARTER,
                how="inner",
            )
            StatFcstPLPQLag = StatFcstPLPQLag[
                StatFcstPLPQLag[o9Constants.PLANNING_QUARTER_KEY] <= current_pl_quarter
            ]
            logger.info(
                f"Maximum Lag for which Stat Fcst PL PQ Lag is available : {max(StatFcstPLPQLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstPLPQLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.PLANNING_QUARTER
            relevant_time_key = o9Constants.PLANNING_QUARTER_KEY
            StatFcstPLPQLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstPLPQLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstPLPQLag) == 0:
                logger.warning(
                    "Stat Fcst PL PQ Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstAbsError,
                    StatFcstWLagAbsError,
                    StatFcstMLagAbsError,
                    StatFcstPMLagAbsError,
                    StatFcstQLagAbsError,
                    StatFcstPQLagAbsError,
                )
            StatFcstPQLagAbsError = get_abs_error(
                source_df=StatFcstPLPQLag,
                Actuals=PlanningActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_QUARTER,
                time_key=o9Constants.PLANNING_QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PL_PQ_LAG,
                actual_measure=o9Constants.ACTUAL,
                output_measure=o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR,
                output_cols=StatFcstPQLag_cols,
            )
            if len(StatFcstPQLagAbsError) != 0:
                StatFcstAbsError = StatFcstPQLagAbsError[
                    (StatFcstPQLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]
                StatFcstAbsError = disaggregate_data(
                    source_df=StatFcstAbsError,
                    source_grain=o9Constants.PLANNING_QUARTER,
                    target_grain=o9Constants.WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR,
                    ],
                )
                StatFcstAbsError = StatFcstAbsError.rename(
                    columns={
                        o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR: o9Constants.STAT_FCST_ABS_ERROR
                    }
                )
        StatFcstAbsError = StatFcstAbsError[StatFcstLag1_cols]
        StatFcstWLagAbsError = StatFcstWLagAbsError[StatFcstWLag_cols]
        StatFcstMLagAbsError = StatFcstMLagAbsError[StatFcstMLag_cols]
        StatFcstPMLagAbsError = StatFcstPMLagAbsError[StatFcstPMLag_cols]
        StatFcstQLagAbsError = StatFcstQLagAbsError[StatFcstQLag_cols]
        StatFcstPQLagAbsError = StatFcstPQLagAbsError[StatFcstPQLag_cols]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        StatFcstAbsError = pd.DataFrame(columns=StatFcstLag1_cols)
        StatFcstWLagAbsError = pd.DataFrame(columns=StatFcstWLag_cols)
        StatFcstMLagAbsError = pd.DataFrame(columns=StatFcstMLag_cols)
        StatFcstPMLagAbsError = pd.DataFrame(columns=StatFcstPMLag_cols)
        StatFcstQLagAbsError = pd.DataFrame(columns=StatFcstQLag_cols)
        StatFcstPQLagAbsError = pd.DataFrame(columns=StatFcstPQLag_cols)

    return (
        StatFcstAbsError,
        StatFcstWLagAbsError,
        StatFcstMLagAbsError,
        StatFcstPMLagAbsError,
        StatFcstQLagAbsError,
        StatFcstPQLagAbsError,
    )

import logging

import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data

from helpers.o9Constants import o9Constants
from helpers.utils import (
    filter_for_iteration,
    get_abs_error,
    get_list_of_grains_from_string,
)

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Stat Fcst L1 Abs Error": float,
    "Stat Fcst L1 Lag": float,
    "Stat Fcst L1 Lag Abs Error": float,
    "Stat Fcst L1 W Lag Abs Error": float,
    "Stat Fcst L1 PM Lag Abs Error": float,
    "Stat Fcst L1 Q Lag Abs Error": float,
    "Stat Fcst L1 PQ Lag Abs Error": float,
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
    lag_df : StatFcstL1WLag
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
    StatActual,
    StatFcstL1WLag,
    StatFcstL1MLag,
    StatFcstL1PMLag,
    StatFcstL1QLag,
    StatFcstL1PQLag,
    ForecastGenTimeBucket,
    CurrentTimePeriod,
    TimeDimension,
    StatBucketWeight,
    StatGrains,
    AccuracyWindow="1",
    df_keys={},
):
    try:
        StatFcstAbsError_list = list()
        StatFcstL1LagAbsError_list = list()
        StatFcstL1WLagAbsError_list = list()
        StatFcstL1MLagAbsError_list = list()
        StatFcstL1PMLagAbsError_list = list()
        StatFcstL1QLagAbsError_list = list()
        StatFcstL1PQLagAbsError_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_l1_error,
                all_lag_error,
                all_w_lag_error,
                all_m_lag_error,
                all_pm_lag_error,
                all_q_lag_error,
                all_pq_lag_error,
            ) = decorated_func(
                StatActual=StatActual,
                StatFcstL1WLag=StatFcstL1WLag,
                StatFcstL1MLag=StatFcstL1MLag,
                StatFcstL1PMLag=StatFcstL1PMLag,
                StatFcstL1QLag=StatFcstL1QLag,
                StatFcstL1PQLag=StatFcstL1PQLag,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                StatGrains=StatGrains,
                StatBucketWeight=StatBucketWeight,
                AccuracyWindow=AccuracyWindow,
                df_keys=df_keys,
            )

            StatFcstAbsError_list.append(all_l1_error)
            StatFcstL1LagAbsError_list.append(all_lag_error)
            StatFcstL1WLagAbsError_list.append(all_w_lag_error)
            StatFcstL1MLagAbsError_list.append(all_m_lag_error)
            StatFcstL1PMLagAbsError_list.append(all_pm_lag_error)
            StatFcstL1QLagAbsError_list.append(all_q_lag_error)
            StatFcstL1PQLagAbsError_list.append(all_pq_lag_error)

        StatFcstL1AbsError = concat_to_dataframe(StatFcstAbsError_list)
        StatFcstL1LagAbsError = concat_to_dataframe(StatFcstL1LagAbsError_list)
        StatFcstL1WLagAbsError = concat_to_dataframe(StatFcstL1WLagAbsError_list)
        StatFcstL1MLagAbsError = concat_to_dataframe(StatFcstL1MLagAbsError_list)
        StatFcstL1PMLagAbsError = concat_to_dataframe(StatFcstL1PMLagAbsError_list)
        StatFcstL1QLagAbsError = concat_to_dataframe(StatFcstL1QLagAbsError_list)
        StatFcstL1PQLagAbsError = concat_to_dataframe(StatFcstL1PQLagAbsError_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstL1AbsError,
            StatFcstL1LagAbsError,
            StatFcstL1WLagAbsError,
            StatFcstL1MLagAbsError,
            StatFcstL1PMLagAbsError,
            StatFcstL1QLagAbsError,
            StatFcstL1PQLagAbsError,
        ) = (None, None, None, None, None, None, None)
    return (
        StatFcstL1AbsError,
        StatFcstL1LagAbsError,
        StatFcstL1WLagAbsError,
        StatFcstL1MLagAbsError,
        StatFcstL1PMLagAbsError,
        StatFcstL1QLagAbsError,
        StatFcstL1PQLagAbsError,
    )


def processIteration(
    StatActual,
    StatFcstL1WLag,
    StatFcstL1MLag,
    StatFcstL1PMLag,
    StatFcstL1QLag,
    StatFcstL1PQLag,
    ForecastGenTimeBucket,
    CurrentTimePeriod,
    TimeDimension,
    StatBucketWeight,
    StatGrains,
    AccuracyWindow="1",
    df_keys={},
):
    plugin_name = "DP212CalculateRollOverAccuracyIteration"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    stat_fcst_l1_lag = "Stat Fcst L1 Lag"
    stat_fcst_l1_lag_abs_error = "Stat Fcst L1 Lag Abs Error"

    stat_grains = get_list_of_grains_from_string(input=StatGrains)

    StatFcstL1Lag1_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_ABS_ERROR]
    )
    StatFcstL1AbsError = pd.DataFrame(columns=StatFcstL1Lag1_cols)

    StatFcstL1LagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [stat_fcst_l1_lag, stat_fcst_l1_lag_abs_error]
    )
    StatFcstL1LagAbsError = pd.DataFrame(columns=StatFcstL1LagAbsError_cols)

    StatFcstL1WLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR,
        ]
    )
    StatFcstL1WLagAbsError = pd.DataFrame(columns=StatFcstL1WLag_cols)

    StatFcstL1MLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR,
        ]
    )
    StatFcstL1MLagAbsError = pd.DataFrame(columns=StatFcstL1MLag_cols)

    StatFcstL1PMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR,
        ]
    )
    StatFcstL1PMLagAbsError = pd.DataFrame(columns=StatFcstL1PMLag_cols)

    StatFcstL1QLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR,
        ]
    )
    StatFcstL1QLagAbsError = pd.DataFrame(columns=StatFcstL1QLag_cols)

    StatFcstL1PQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR,
        ]
    )
    StatFcstL1PQLagAbsError = pd.DataFrame(columns=StatFcstL1PQLag_cols)

    try:
        TimeLevel = ForecastGenTimeBucket[o9Constants.FORECAST_GEN_TIME_BUCKET].values[0]
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.replace(" ", "").lower() == "planningmonth"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.replace(" ", "").lower() == "planningquarter"
        AccuracyWindow = int(AccuracyWindow)

        # merge with TimeDimension to get all relevant columns
        StatBucketWeight = StatBucketWeight.merge(
            TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
        )

        if len(StatFcstL1WLag) != 0 and week:
            current_week = CurrentTimePeriod[o9Constants.WEEK_KEY].values[0]
            StatFcstL1WLag = StatFcstL1WLag.merge(
                TimeDimension[[o9Constants.WEEK, o9Constants.WEEK_KEY]].drop_duplicates(),
                on=o9Constants.WEEK,
                how="inner",
            )
            StatFcstL1WLag = StatFcstL1WLag[StatFcstL1WLag[o9Constants.WEEK_KEY] <= current_week]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst L1 W Lag is available : {max(StatFcstL1WLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstL1WLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.WEEK
            relevant_time_key = o9Constants.WEEK_KEY
            StatFcstL1WLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstL1WLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstL1WLag) == 0:
                logger.warning(
                    "Stat Fcst L1 W Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstL1AbsError,
                    StatFcstL1LagAbsError,
                    StatFcstL1WLagAbsError,
                    StatFcstL1MLagAbsError,
                    StatFcstL1PMLagAbsError,
                    StatFcstL1QLagAbsError,
                    StatFcstL1PQLagAbsError,
                )

            StatFcstL1WLagAbsError = get_abs_error(
                source_df=StatFcstL1WLag,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.WEEK,
                time_key=o9Constants.WEEK_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_L1_W_LAG,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR,
                output_cols=StatFcstL1WLag_cols + [o9Constants.STAT_FCST_L1_W_LAG],
            )
            if len(StatFcstL1WLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_W_LAG: stat_fcst_l1_lag,
                    o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR: stat_fcst_l1_lag_abs_error,
                }
                StatFcstL1LagAbsError = disaggregate_data(
                    source_df=StatFcstL1WLagAbsError,
                    source_grain=o9Constants.WEEK,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_W_LAG,
                        o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR,
                    ],
                )
                StatFcstL1LagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstL1AbsError = StatFcstL1LagAbsError[
                    (StatFcstL1LagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstL1MLag) != 0 and month:
            current_month = CurrentTimePeriod[o9Constants.MONTH_KEY].values[0]
            StatFcstL1MLag = StatFcstL1MLag.merge(
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                on=o9Constants.MONTH,
                how="inner",
            )
            StatFcstL1MLag = StatFcstL1MLag[StatFcstL1MLag[o9Constants.MONTH_KEY] <= current_month]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst L1 M Lag is available : {max(StatFcstL1MLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstL1MLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.MONTH
            relevant_time_key = o9Constants.MONTH_KEY
            StatFcstL1MLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstL1MLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstL1MLag) == 0:
                logger.warning(
                    "Stat Fcst L1 M Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstL1AbsError,
                    StatFcstL1LagAbsError,
                    StatFcstL1WLagAbsError,
                    StatFcstL1MLagAbsError,
                    StatFcstL1PMLagAbsError,
                    StatFcstL1QLagAbsError,
                    StatFcstL1PQLagAbsError,
                )
            StatFcstL1MLagAbsError = get_abs_error(
                source_df=StatFcstL1MLag,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_L1_M_LAG,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR,
                output_cols=StatFcstL1MLag_cols + [o9Constants.STAT_FCST_L1_M_LAG],
            )
            if len(StatFcstL1MLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_M_LAG: stat_fcst_l1_lag,
                    o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR: stat_fcst_l1_lag_abs_error,
                }
                StatFcstL1LagAbsError = disaggregate_data(
                    source_df=StatFcstL1MLagAbsError,
                    source_grain=o9Constants.MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_M_LAG,
                        o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR,
                    ],
                )
                StatFcstL1LagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstL1AbsError = StatFcstL1LagAbsError[
                    (StatFcstL1LagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstL1PMLag) != 0 and pl_month:
            current_pl_month = CurrentTimePeriod[o9Constants.PLANNING_MONTH_KEY].values[0]
            StatFcstL1PMLag = StatFcstL1PMLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_MONTH, o9Constants.PLANNING_MONTH_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_MONTH,
                how="inner",
            )
            StatFcstL1PMLag = StatFcstL1PMLag[
                StatFcstL1PMLag[o9Constants.PLANNING_MONTH_KEY] <= current_pl_month
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst L1 PM Lag is available : {max(StatFcstL1PMLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstL1PMLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.PLANNING_MONTH
            relevant_time_key = o9Constants.PLANNING_MONTH_KEY
            StatFcstL1PMLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstL1PMLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstL1PMLag) == 0:
                logger.warning(
                    "Stat Fcst L1 PM Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstL1AbsError,
                    StatFcstL1LagAbsError,
                    StatFcstL1WLagAbsError,
                    StatFcstL1MLagAbsError,
                    StatFcstL1PMLagAbsError,
                    StatFcstL1QLagAbsError,
                    StatFcstL1PQLagAbsError,
                )
            StatFcstL1PMLagAbsError = get_abs_error(
                source_df=StatFcstL1PMLag,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_MONTH,
                time_key=o9Constants.PLANNING_MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_L1_PM_LAG,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR,
                output_cols=StatFcstL1PMLag_cols + [o9Constants.STAT_FCST_L1_PM_LAG],
            )
            if len(StatFcstL1PMLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_PM_LAG: stat_fcst_l1_lag,
                    o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR: stat_fcst_l1_lag_abs_error,
                }
                StatFcstL1LagAbsError = disaggregate_data(
                    source_df=StatFcstL1PMLagAbsError,
                    source_grain=o9Constants.PLANNING_MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_PM_LAG,
                        o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR,
                    ],
                )
                StatFcstL1LagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstL1AbsError = StatFcstL1LagAbsError[
                    (StatFcstL1LagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstL1QLag) != 0 and quarter:
            current_quarter = CurrentTimePeriod[o9Constants.QUARTER_KEY].values[0]
            StatFcstL1QLag = StatFcstL1QLag.merge(
                TimeDimension[[o9Constants.QUARTER, o9Constants.QUARTER_KEY]].drop_duplicates(),
                on=o9Constants.QUARTER,
                how="inner",
            )
            StatFcstL1QLag = StatFcstL1QLag[
                StatFcstL1QLag[o9Constants.QUARTER_KEY] <= current_quarter
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst L1 Q Lag is available : {max(StatFcstL1QLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstL1QLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.QUARTER
            relevant_time_key = o9Constants.QUARTER_KEY
            StatFcstL1QLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstL1QLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstL1QLag) == 0:
                logger.warning(
                    "Stat Fcst L1 Q Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstL1AbsError,
                    StatFcstL1LagAbsError,
                    StatFcstL1WLagAbsError,
                    StatFcstL1MLagAbsError,
                    StatFcstL1PMLagAbsError,
                    StatFcstL1QLagAbsError,
                    StatFcstL1PQLagAbsError,
                )

            StatFcstL1QLagAbsError = get_abs_error(
                source_df=StatFcstL1QLag,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.QUARTER,
                time_key=o9Constants.QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_L1_Q_LAG,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR,
                output_cols=StatFcstL1QLag_cols + [o9Constants.STAT_FCST_L1_Q_LAG],
            )
            if len(StatFcstL1QLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_Q_LAG: stat_fcst_l1_lag,
                    o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR: stat_fcst_l1_lag_abs_error,
                }
                StatFcstL1LagAbsError = disaggregate_data(
                    source_df=StatFcstL1QLagAbsError,
                    source_grain=o9Constants.QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_Q_LAG,
                        o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR,
                    ],
                )
                StatFcstL1LagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstL1AbsError = StatFcstL1LagAbsError[
                    (StatFcstL1LagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstL1PQLag) != 0 and pl_quarter:
            current_pl_quarter = CurrentTimePeriod[o9Constants.PLANNING_QUARTER_KEY].values[0]
            StatFcstL1PQLag = StatFcstL1PQLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_QUARTER, o9Constants.PLANNING_QUARTER_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_QUARTER,
                how="inner",
            )
            StatFcstL1PQLag = StatFcstL1PQLag[
                StatFcstL1PQLag[o9Constants.PLANNING_QUARTER_KEY] <= current_pl_quarter
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst L1 PQ Lag is available : {max(StatFcstL1PQLag[o9Constants.LAG].astype(int))}"
            )
            lags = [i for i in range(max(StatFcstL1PQLag[o9Constants.LAG].astype(int)) + 1)]
            lags = ",".join(map(str, lags))
            relevant_time_name = o9Constants.PLANNING_QUARTER
            relevant_time_key = o9Constants.PLANNING_QUARTER_KEY
            StatFcstL1PQLag = get_relevant_statfcst_lag(
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                lag_df=StatFcstL1PQLag,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AccuracyWindow=AccuracyWindow,
            )
            if len(StatFcstL1PQLag) == 0:
                logger.warning(
                    "Stat Fcst L1 PQ Lag not present for the accuracy window periods, returning empty outputs ..."
                )
                return (
                    StatFcstL1AbsError,
                    StatFcstL1LagAbsError,
                    StatFcstL1WLagAbsError,
                    StatFcstL1MLagAbsError,
                    StatFcstL1PMLagAbsError,
                    StatFcstL1QLagAbsError,
                    StatFcstL1PQLagAbsError,
                )

            StatFcstL1PQLagAbsError = get_abs_error(
                source_df=StatFcstL1PQLag,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_QUARTER,
                time_key=o9Constants.PLANNING_QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_L1_PQ_LAG,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR,
                output_cols=StatFcstL1PQLag_cols + [o9Constants.STAT_FCST_L1_PQ_LAG],
            )
            if len(StatFcstL1PQLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_PQ_LAG: stat_fcst_l1_lag,
                    o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR: stat_fcst_l1_lag_abs_error,
                }
                StatFcstL1LagAbsError = disaggregate_data(
                    source_df=StatFcstL1PQLagAbsError,
                    source_grain=o9Constants.PLANNING_QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_PQ_LAG,
                        o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR,
                    ],
                )
                StatFcstL1LagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstL1AbsError = StatFcstL1LagAbsError[
                    (StatFcstL1LagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        StatFcstL1LagAbsError = StatFcstL1LagAbsError[StatFcstL1LagAbsError_cols]
        StatFcstL1AbsError = StatFcstL1AbsError.rename(
            columns={stat_fcst_l1_lag_abs_error: o9Constants.STAT_FCST_L1_ABS_ERROR}
        )
        StatFcstL1AbsError = StatFcstL1AbsError[StatFcstL1Lag1_cols]
        StatFcstL1WLagAbsError = StatFcstL1WLagAbsError[StatFcstL1WLag_cols]
        StatFcstL1MLagAbsError = StatFcstL1MLagAbsError[StatFcstL1MLag_cols]
        StatFcstL1PMLagAbsError = StatFcstL1PMLagAbsError[StatFcstL1PMLag_cols]
        StatFcstL1QLagAbsError = StatFcstL1QLagAbsError[StatFcstL1QLag_cols]
        StatFcstL1PQLagAbsError = StatFcstL1PQLagAbsError[StatFcstL1PQLag_cols]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        logger.info("")

    return (
        StatFcstL1AbsError,
        StatFcstL1LagAbsError,
        StatFcstL1WLagAbsError,
        StatFcstL1MLagAbsError,
        StatFcstL1PMLagAbsError,
        StatFcstL1QLagAbsError,
        StatFcstL1PQLagAbsError,
    )

import logging

import pandas as pd
from joblib import Parallel, delayed
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
    "Stat Fcst L1 Abs Error Backtest": float,
    "Stat Fcst L1 Lag Abs Error Backtest": float,
    "Stat Fcst L1 LagN Abs Error Backtest": float,
    "Stat Fcst L1 W Lag Abs Error Backtest": float,
    "Stat Fcst L1 M Lag Abs Error Backtest": float,
    "Stat Fcst L1 PM Lag Abs Error Backtest": float,
    "Stat Fcst L1 Q Lag Abs Error Backtest": float,
    "Stat Fcst L1 PQ Lag Abs Error Backtest": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatActual,
    StatFcstL1Lag1Backtest,
    StatFcstL1LagNBacktest,
    StatFcstL1WLagBacktest,
    StatFcstL1MLagBacktest,
    StatFcstL1PMLagBacktest,
    StatFcstL1QLagBacktest,
    StatFcstL1PQLagBacktest,
    CurrentTimePeriod,
    TimeDimension,
    StatGrains,
    StatBucketWeight,
    BackTestCyclePeriod,
    ForecastGenTimeBucket,
    PlanningCycleDates,
    df_keys={},
):
    try:
        StatFcstL1Lag1AbsErrorBacktest_list = list()
        StatFcstL1LagAbsErrorBacktest_list = list()
        StatFcstL1LagNAbsErrorBacktest_list = list()
        StatFcstL1WLagAbsErrorBacktest_list = list()
        StatFcstL1MLagAbsErrorBacktest_list = list()
        StatFcstL1PMLagAbsErrorBacktest_list = list()
        StatFcstL1QLagAbsErrorBacktest_list = list()
        StatFcstL1PQLagAbsErrorBacktest_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_l1_error,
                all_l1_lag_error,
                all_l1_lagn_error,
                all_w_lag_error,
                all_m_lag_error,
                all_pm_lag_error,
                all_q_lag_error,
                all_pq_lag_error,
            ) = decorated_func(
                StatActual=StatActual,
                StatFcstL1Lag1Backtest=StatFcstL1Lag1Backtest,
                StatFcstL1LagNBacktest=StatFcstL1LagNBacktest,
                StatFcstL1WLagBacktest=StatFcstL1WLagBacktest,
                StatFcstL1QLagBacktest=StatFcstL1QLagBacktest,
                StatFcstL1PQLagBacktest=StatFcstL1PQLagBacktest,
                StatFcstL1MLagBacktest=StatFcstL1MLagBacktest,
                StatFcstL1PMLagBacktest=StatFcstL1PMLagBacktest,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                StatGrains=StatGrains,
                StatBucketWeight=StatBucketWeight,
                BackTestCyclePeriod=BackTestCyclePeriod,
                PlanningCycleDates=PlanningCycleDates,
                TimeLevel=ForecastGenTimeBucket[
                    ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] == the_iteration
                ][o9Constants.FORECAST_GEN_TIME_BUCKET].values[0],
                df_keys=df_keys,
            )

            StatFcstL1Lag1AbsErrorBacktest_list.append(all_l1_error)
            StatFcstL1LagAbsErrorBacktest_list.append(all_l1_lag_error)
            StatFcstL1LagNAbsErrorBacktest_list.append(all_l1_lagn_error)
            StatFcstL1WLagAbsErrorBacktest_list.append(all_w_lag_error)
            StatFcstL1MLagAbsErrorBacktest_list.append(all_m_lag_error)
            StatFcstL1PMLagAbsErrorBacktest_list.append(all_pm_lag_error)
            StatFcstL1QLagAbsErrorBacktest_list.append(all_q_lag_error)
            StatFcstL1PQLagAbsErrorBacktest_list.append(all_pq_lag_error)

        StatFcstL1Lag1AbsErrorBacktest = concat_to_dataframe(StatFcstL1Lag1AbsErrorBacktest_list)
        StatFcstL1LagAbsErrorBacktest = concat_to_dataframe(StatFcstL1LagAbsErrorBacktest_list)
        StatFcstL1LagNAbsErrorBacktest = concat_to_dataframe(StatFcstL1LagNAbsErrorBacktest_list)
        StatFcstL1WLagAbsErrorBacktest = concat_to_dataframe(StatFcstL1WLagAbsErrorBacktest_list)
        StatFcstL1MLagAbsErrorBacktest = concat_to_dataframe(StatFcstL1MLagAbsErrorBacktest_list)
        StatFcstL1PMLagAbsErrorBacktest = concat_to_dataframe(StatFcstL1PMLagAbsErrorBacktest_list)
        StatFcstL1QLagAbsErrorBacktest = concat_to_dataframe(StatFcstL1QLagAbsErrorBacktest_list)
        StatFcstL1PQLagAbsErrorBacktest = concat_to_dataframe(StatFcstL1PQLagAbsErrorBacktest_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstL1Lag1AbsErrorBacktest,
            StatFcstL1LagAbsErrorBacktest,
            StatFcstL1LagNAbsErrorBacktest,
            StatFcstL1WLagAbsErrorBacktest,
            StatFcstL1MLagAbsErrorBacktest,
            StatFcstL1PMLagAbsErrorBacktest,
            StatFcstL1QLagAbsErrorBacktest,
            StatFcstL1PQLagAbsErrorBacktest,
        ) = (None, None, None, None, None, None, None)
    return (
        StatFcstL1Lag1AbsErrorBacktest,
        StatFcstL1LagAbsErrorBacktest,
        StatFcstL1LagNAbsErrorBacktest,
        StatFcstL1WLagAbsErrorBacktest,
        StatFcstL1MLagAbsErrorBacktest,
        StatFcstL1PMLagAbsErrorBacktest,
        StatFcstL1QLagAbsErrorBacktest,
        StatFcstL1PQLagAbsErrorBacktest,
    )


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def processIteration(
    StatActual,
    StatFcstL1Lag1Backtest,
    StatFcstL1LagNBacktest,
    StatFcstL1WLagBacktest,
    StatFcstL1MLagBacktest,
    StatFcstL1PMLagBacktest,
    StatFcstL1QLagBacktest,
    StatFcstL1PQLagBacktest,
    CurrentTimePeriod,
    TimeDimension,
    StatGrains,
    StatBucketWeight,
    BackTestCyclePeriod,
    PlanningCycleDates,
    TimeLevel="Week",
    df_keys={},
):
    plugin_name = "DP210CalculateBacktestStatLevelAccuracy"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # Configurables
    stat_actual_lag_backtest_col = "Stat Actual Lag Backtest"
    PLANNING_CYCLE_DATE_KEY = "Planning Cycle.[PlanningCycleDateKey]"
    stat_grains = get_list_of_grains_from_string(input=StatGrains)

    StatFcstL1AbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_ABS_ERROR_BACKTEST]
    )
    StatFcstL1AbsErrorBacktest = pd.DataFrame(columns=StatFcstL1AbsErrorBacktest_cols)
    StatFcstL1LagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            stat_actual_lag_backtest_col,
            o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
        ]
    )
    StatFcstL1LagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1LagAbsErrorBacktest_cols)
    StatFcstL1LagNAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_LAGN_ABS_ERROR_BACKTEST]
    )
    StatFcstL1LagNAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1LagNAbsErrorBacktest_cols)

    StatFcstL1WLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstL1WLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1WLagAbsErrorBacktest_cols)

    StatFcstL1MLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstL1MLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1MLagAbsErrorBacktest_cols)

    StatFcstL1PMLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstL1PMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1PMLagAbsErrorBacktest_cols)
    StatFcstL1QLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstL1QLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1QLagAbsErrorBacktest_cols)
    StatFcstL1PQLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstL1PQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1PQLagAbsErrorBacktest_cols)

    try:
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.lower() == "planning month"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.lower() == "planning quarter"

        StatBucketWeight = StatBucketWeight.merge(
            TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
        )

        time_grain = (
            o9Constants.WEEK
            if week
            else (
                o9Constants.MONTH
                if month
                else (
                    o9Constants.PLANNING_MONTH
                    if pl_month
                    else (
                        o9Constants.QUARTER
                        if quarter
                        else (o9Constants.PLANNING_QUARTER if pl_quarter else None)
                    )
                )
            )
        )
        time_key = (
            o9Constants.WEEK_KEY
            if week
            else (
                o9Constants.MONTH_KEY
                if month
                else (
                    o9Constants.PLANNING_MONTH_KEY
                    if pl_month
                    else (
                        o9Constants.QUARTER_KEY
                        if quarter
                        else (o9Constants.PLANNING_QUARTER_KEY if pl_quarter else None)
                    )
                )
            )
        )
        latest_time_period = CurrentTimePeriod[time_grain][0]
        relevant_time_mapping = TimeDimension[[time_grain, time_key]].drop_duplicates()
        time_attribute_dict = {time_grain: time_key}
        relevant_time_grains = [
            o9Constants.PARTIAL_WEEK,
            o9Constants.PARTIAL_WEEK_KEY,
        ]
        actual_cols = StatActual.columns
        StatActual = StatActual[actual_cols].drop_duplicates()

        if time_grain != o9Constants.PARTIAL_WEEK:
            relevant_time_grains = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PARTIAL_WEEK_KEY,
                time_grain,
                time_key,
            ]
        StatActual = StatActual.merge(
            TimeDimension[relevant_time_grains].drop_duplicates(),
            on=o9Constants.PARTIAL_WEEK,
            how="inner",
        )
        last_n_time_periods = get_n_time_periods(
            latest_time_period,
            -6,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )
        if len(StatFcstL1Lag1Backtest) != 0:
            StatFcstL1Lag1Backtest = StatFcstL1Lag1Backtest.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, time_grain]],
                on=o9Constants.PARTIAL_WEEK,
                how="inner",
            )
            Actuals = StatActual[StatActual[time_grain].isin(last_n_time_periods)]
            Actuals = Actuals[actual_cols].drop_duplicates()
            StatActual = StatActual[actual_cols].drop_duplicates()
            StatFcstL1Lag1Backtest = StatFcstL1Lag1Backtest[
                StatFcstL1Lag1Backtest[time_grain].isin(last_n_time_periods)
            ]
            output_cols = [
                col for col in StatFcstL1AbsErrorBacktest_cols if col != o9Constants.PARTIAL_WEEK
            ] + [time_grain]
            StatFcstL1AbsErrorBacktest = get_abs_error(
                source_df=StatFcstL1Lag1Backtest,
                Actuals=Actuals,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=time_grain,
                time_key=time_key,
                source_measure=o9Constants.STAT_FCST_L1_LAG1_BACKTEST,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_ABS_ERROR_BACKTEST,
                output_cols=output_cols,
                cycle_period=BackTestCyclePeriod,
            )
            StatFcstL1AbsErrorBacktest = disaggregate_data(
                source_df=StatFcstL1AbsErrorBacktest,
                source_grain=time_grain,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_L1_ABS_ERROR_BACKTEST],
            )
            if len(StatFcstL1AbsErrorBacktest) > 0:
                StatFcstL1AbsErrorBacktest = StatFcstL1AbsErrorBacktest[
                    StatFcstL1AbsErrorBacktest_cols
                ]
            else:
                StatFcstL1AbsErrorBacktest = pd.DataFrame(columns=StatFcstL1AbsErrorBacktest_cols)

        if len(StatFcstL1LagNBacktest) != 0:
            StatFcstL1LagNBacktest = StatFcstL1LagNBacktest.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, time_grain]],
                on=o9Constants.PARTIAL_WEEK,
                how="inner",
            )
            StatFcstL1LagNBacktest = StatFcstL1LagNBacktest[
                StatFcstL1LagNBacktest[time_grain].isin(last_n_time_periods)
            ]
            output_cols = [
                col
                for col in StatFcstL1LagNAbsErrorBacktest_cols
                if col != o9Constants.PARTIAL_WEEK
            ] + [time_grain]
            StatFcstL1LagNAbsErrorBacktest = get_abs_error(
                source_df=StatFcstL1LagNBacktest,
                Actuals=StatActual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                time_grain=time_grain,
                time_key=time_key,
                source_measure=o9Constants.STAT_FCST_L1_LAGN_BACKTEST,
                actual_measure=o9Constants.STAT_ACTUAL,
                output_measure=o9Constants.STAT_FCST_L1_LAGN_ABS_ERROR_BACKTEST,
                output_cols=output_cols,
                cycle_period=BackTestCyclePeriod,
            )

            StatFcstL1LagNAbsErrorBacktest = disaggregate_data(
                source_df=StatFcstL1LagNAbsErrorBacktest,
                source_grain=time_grain,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_L1_LAGN_ABS_ERROR_BACKTEST],
            )

        PlanningCycleDates = PlanningCycleDates.merge(
            TimeDimension,
            left_on=PLANNING_CYCLE_DATE_KEY,
            right_on=o9Constants.PARTIAL_WEEK_KEY,
            how="inner",
        )
        if len(PlanningCycleDates) == 0:
            logger.exception(
                "No matching PlanningCycleDates found in TimeDimension, please check the Planning Cycle Date Key formatting...\nReturning empty W/M/PM/Q/PQ results..."
            )
            return (
                StatFcstL1AbsErrorBacktest,
                StatFcstL1LagAbsErrorBacktest,
                StatFcstL1LagNAbsErrorBacktest,
                StatFcstL1WLagAbsErrorBacktest,
                StatFcstL1MLagAbsErrorBacktest,
                StatFcstL1PMLagAbsErrorBacktest,
                StatFcstL1QLagAbsErrorBacktest,
                StatFcstL1PQLagAbsErrorBacktest,
            )

        if len(StatFcstL1WLagBacktest) != 0 and week:
            # Get unique planning cycle dates for Week
            planning_cycle_dates = StatFcstL1WLagBacktest[o9Constants.PLANNING_CYCLE_DATE].unique()

            def process_cycle_date_w(cycle_date):
                StatFcstL1WLagBacktest_cycle = StatFcstL1WLagBacktest[
                    StatFcstL1WLagBacktest[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ]
                # Get the week corresponding to the cycle date
                cycle_current_date = PlanningCycleDates[
                    PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ][o9Constants.WEEK].values[0]
                pcd_cycle_dates = get_n_time_periods(
                    cycle_current_date,
                    156,
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=True,
                )
                lag_mapping = pd.DataFrame(
                    {
                        o9Constants.WEEK: pcd_cycle_dates,
                        o9Constants.LAG: list(range(len(pcd_cycle_dates))),
                    }
                )
                lag_mapping[o9Constants.LAG] = lag_mapping[o9Constants.LAG].astype(int).astype(str)
                result = get_abs_error(
                    source_df=StatFcstL1WLagBacktest_cycle,
                    Actuals=StatActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.WEEK,
                    time_key=o9Constants.WEEK_KEY,
                    source_measure=o9Constants.STAT_FCST_L1_W_LAG_BACKTEST,
                    actual_measure=o9Constants.STAT_ACTUAL,
                    output_measure=o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR_BACKTEST,
                    output_cols=StatFcstL1WLagAbsErrorBacktest_cols
                    + [o9Constants.STAT_FCST_L1_W_LAG_BACKTEST, o9Constants.STAT_ACTUAL],
                    cycle_period=BackTestCyclePeriod,
                    df_keys=df_keys,
                )
                result[o9Constants.PLANNING_CYCLE_DATE] = cycle_date
                result.drop(columns=[o9Constants.LAG], inplace=True)
                result = result.merge(lag_mapping, on=o9Constants.WEEK, how="inner")
                return result

            StatFcstL1WLagAbsErrorBacktest_list = Parallel(n_jobs=len(planning_cycle_dates))(
                delayed(process_cycle_date_w)(cycle_date) for cycle_date in planning_cycle_dates
            )
            StatFcstL1WLagAbsErrorBacktest = pd.concat(
                StatFcstL1WLagAbsErrorBacktest_list, ignore_index=True
            )
            if len(StatFcstL1WLagAbsErrorBacktest) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_W_LAG_BACKTEST: o9Constants.STAT_FCST_L1_LAG_BACKTEST,
                    o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR_BACKTEST: o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
                }
                StatFcstL1LagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstL1WLagAbsErrorBacktest,
                    source_grain=o9Constants.WEEK,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_W_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.STAT_ACTUAL,
                    ],
                )
                StatFcstL1LagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstL1MLagBacktest) != 0 and month:
            # Get unique planning cycle dates
            planning_cycle_dates = StatFcstL1MLagBacktest[o9Constants.PLANNING_CYCLE_DATE].unique()

            def process_cycle_date(cycle_date):
                StatFcstL1MLagBacktest_cycle = StatFcstL1MLagBacktest[
                    StatFcstL1MLagBacktest[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ]
                # TODO: Include PlanningCycleDates to get current cycle instead of PW
                cycle_current_date = PlanningCycleDates[
                    PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ][o9Constants.MONTH].values[0]
                pcd_cycle_dates = get_n_time_periods(
                    cycle_current_date,
                    36,
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=True,
                )
                lag_mapping = pd.DataFrame(
                    {
                        o9Constants.MONTH: pcd_cycle_dates,
                        o9Constants.LAG: list(range(len(pcd_cycle_dates))),
                    }
                )
                lag_mapping[o9Constants.LAG] = lag_mapping[o9Constants.LAG].astype(int).astype(str)
                result = get_abs_error(
                    source_df=StatFcstL1MLagBacktest_cycle,
                    Actuals=StatActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.MONTH,
                    time_key=o9Constants.MONTH_KEY,
                    source_measure=o9Constants.STAT_FCST_L1_M_LAG_BACKTEST,
                    actual_measure=o9Constants.STAT_ACTUAL,
                    output_measure=o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR_BACKTEST,
                    output_cols=StatFcstL1MLagAbsErrorBacktest_cols
                    + [o9Constants.STAT_FCST_L1_M_LAG_BACKTEST, o9Constants.STAT_ACTUAL],
                    cycle_period=BackTestCyclePeriod,
                    df_keys=df_keys,
                )
                result[o9Constants.PLANNING_CYCLE_DATE] = cycle_date
                result.drop(columns=[o9Constants.LAG], inplace=True)

                result = result.merge(lag_mapping, on=o9Constants.MONTH, how="inner")
                return result

            StatFcstL1MLagAbsErrorBacktest_list = Parallel(n_jobs=len(planning_cycle_dates))(
                delayed(process_cycle_date)(cycle_date) for cycle_date in planning_cycle_dates
            )
            StatFcstL1MLagAbsErrorBacktest = pd.concat(
                StatFcstL1MLagAbsErrorBacktest_list, ignore_index=True
            )
            if len(StatFcstL1MLagAbsErrorBacktest) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_M_LAG_BACKTEST: o9Constants.STAT_FCST_L1_LAG_BACKTEST,
                    o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR_BACKTEST: o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
                }
                StatFcstL1LagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstL1MLagAbsErrorBacktest,
                    source_grain=o9Constants.MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_M_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.STAT_ACTUAL,
                    ],
                )
                StatFcstL1LagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstL1PMLagBacktest) != 0 and pl_month:
            # Get unique planning cycle dates for Planning Month
            planning_cycle_dates = StatFcstL1PMLagBacktest[o9Constants.PLANNING_CYCLE_DATE].unique()

            def process_cycle_date_pm(cycle_date):
                StatFcstL1PMLagBacktest_cycle = StatFcstL1PMLagBacktest[
                    StatFcstL1PMLagBacktest[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ]
                # Get the planning month corresponding to the cycle date
                cycle_current_date = PlanningCycleDates[
                    PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ][o9Constants.PLANNING_MONTH].values[0]
                pcd_cycle_dates = get_n_time_periods(
                    cycle_current_date,
                    36,
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=True,
                )
                lag_mapping = pd.DataFrame(
                    {
                        o9Constants.PLANNING_MONTH: pcd_cycle_dates,
                        o9Constants.LAG: list(range(len(pcd_cycle_dates))),
                    }
                )
                lag_mapping[o9Constants.LAG] = lag_mapping[o9Constants.LAG].astype(int).astype(str)
                result = get_abs_error(
                    source_df=StatFcstL1PMLagBacktest_cycle,
                    Actuals=StatActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.PLANNING_MONTH,
                    time_key=o9Constants.PLANNING_MONTH_KEY,
                    source_measure=o9Constants.STAT_FCST_L1_PM_LAG_BACKTEST,
                    actual_measure=o9Constants.STAT_ACTUAL,
                    output_measure=o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR_BACKTEST,
                    output_cols=StatFcstL1PMLagAbsErrorBacktest_cols
                    + [o9Constants.STAT_FCST_L1_PM_LAG_BACKTEST, o9Constants.STAT_ACTUAL],
                    cycle_period=BackTestCyclePeriod,
                    df_keys=df_keys,
                )
                result[o9Constants.PLANNING_CYCLE_DATE] = cycle_date
                result.drop(columns=[o9Constants.LAG], inplace=True)
                result = result.merge(lag_mapping, on=o9Constants.PLANNING_MONTH, how="inner")
                result.dropna(subset=[o9Constants.LAG], inplace=True)
                return result

            StatFcstL1PMLagAbsErrorBacktest_list = Parallel(n_jobs=len(planning_cycle_dates))(
                delayed(process_cycle_date_pm)(cycle_date) for cycle_date in planning_cycle_dates
            )
            StatFcstL1PMLagAbsErrorBacktest = pd.concat(
                StatFcstL1PMLagAbsErrorBacktest_list, ignore_index=True
            )
            if len(StatFcstL1PMLagAbsErrorBacktest) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_PM_LAG_BACKTEST: o9Constants.STAT_FCST_L1_LAG_BACKTEST,
                    o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR_BACKTEST: o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
                }
                StatFcstL1LagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstL1PMLagAbsErrorBacktest,
                    source_grain=o9Constants.PLANNING_MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_PM_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.STAT_ACTUAL,
                    ],
                )
                StatFcstL1LagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstL1QLagBacktest) != 0 and quarter:
            # Get unique planning cycle dates for Quarter
            planning_cycle_dates = StatFcstL1QLagBacktest[o9Constants.PLANNING_CYCLE_DATE].unique()

            def process_cycle_date_q(cycle_date):
                StatFcstL1QLagBacktest_cycle = StatFcstL1QLagBacktest[
                    StatFcstL1QLagBacktest[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ]
                # Get the quarter corresponding to the cycle date
                cycle_current_date = PlanningCycleDates[
                    PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ][o9Constants.QUARTER].values[0]
                pcd_cycle_dates = get_n_time_periods(
                    cycle_current_date,
                    12,
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=True,
                )
                lag_mapping = pd.DataFrame(
                    {
                        o9Constants.QUARTER: pcd_cycle_dates,
                        o9Constants.LAG: list(range(len(pcd_cycle_dates))),
                    }
                )
                lag_mapping[o9Constants.LAG] = lag_mapping[o9Constants.LAG].astype(int).astype(str)
                result = get_abs_error(
                    source_df=StatFcstL1QLagBacktest_cycle,
                    Actuals=StatActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.QUARTER,
                    time_key=o9Constants.QUARTER_KEY,
                    source_measure=o9Constants.STAT_FCST_L1_Q_LAG_BACKTEST,
                    actual_measure=o9Constants.STAT_ACTUAL,
                    output_measure=o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR_BACKTEST,
                    output_cols=StatFcstL1QLagAbsErrorBacktest_cols
                    + [o9Constants.STAT_FCST_L1_Q_LAG_BACKTEST, o9Constants.STAT_ACTUAL],
                    cycle_period=BackTestCyclePeriod,
                    df_keys=df_keys,
                )
                result[o9Constants.PLANNING_CYCLE_DATE] = cycle_date
                result.drop(columns=[o9Constants.LAG], inplace=True)
                result = result.merge(lag_mapping, on=o9Constants.QUARTER, how="inner")
                return result

            StatFcstL1QLagAbsErrorBacktest_list = Parallel(n_jobs=len(planning_cycle_dates))(
                delayed(process_cycle_date_q)(cycle_date) for cycle_date in planning_cycle_dates
            )
            StatFcstL1QLagAbsErrorBacktest = pd.concat(
                StatFcstL1QLagAbsErrorBacktest_list, ignore_index=True
            )
            if len(StatFcstL1QLagAbsErrorBacktest) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_Q_LAG_BACKTEST: o9Constants.STAT_FCST_L1_LAG_BACKTEST,
                    o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR_BACKTEST: o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
                }
                StatFcstL1LagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstL1MLagAbsErrorBacktest,
                    source_grain=o9Constants.QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_Q_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.STAT_ACTUAL,
                    ],
                )
                StatFcstL1LagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstL1PQLagBacktest) != 0 and pl_quarter:
            # Get unique planning cycle dates for Planning Quarter
            planning_cycle_dates = StatFcstL1PQLagBacktest[o9Constants.PLANNING_CYCLE_DATE].unique()

            def process_cycle_date_pq(cycle_date):
                StatFcstL1PQLagBacktest_cycle = StatFcstL1PQLagBacktest[
                    StatFcstL1PQLagBacktest[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ]
                # Get the planning quarter corresponding to the cycle date
                cycle_current_date = PlanningCycleDates[
                    PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE] == cycle_date
                ][o9Constants.PLANNING_QUARTER].values[0]
                pcd_cycle_dates = get_n_time_periods(
                    cycle_current_date,
                    12,
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=True,
                )
                lag_mapping = pd.DataFrame(
                    {
                        o9Constants.PLANNING_QUARTER: pcd_cycle_dates,
                        o9Constants.LAG: list(range(len(pcd_cycle_dates))),
                    }
                )
                lag_mapping[o9Constants.LAG] = lag_mapping[o9Constants.LAG].astype(int).astype(str)
                result = get_abs_error(
                    source_df=StatFcstL1PQLagBacktest_cycle,
                    Actuals=StatActual,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    merge_grains=stat_grains + [o9Constants.VERSION_NAME],
                    time_grain=o9Constants.PLANNING_QUARTER,
                    time_key=o9Constants.PLANNING_QUARTER_KEY,
                    source_measure=o9Constants.STAT_FCST_L1_PQ_LAG_BACKTEST,
                    actual_measure=o9Constants.STAT_ACTUAL,
                    output_measure=o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR_BACKTEST,
                    output_cols=StatFcstL1PQLagAbsErrorBacktest_cols
                    + [o9Constants.STAT_FCST_L1_PQ_LAG_BACKTEST, o9Constants.STAT_ACTUAL],
                    cycle_period=BackTestCyclePeriod,
                    df_keys=df_keys,
                )
                result[o9Constants.PLANNING_CYCLE_DATE] = cycle_date
                result.drop(columns=[o9Constants.LAG], inplace=True)
                result = result.merge(lag_mapping, on=o9Constants.PLANNING_QUARTER, how="inner")
                return result

            StatFcstL1PQLagAbsErrorBacktest_list = Parallel(n_jobs=len(planning_cycle_dates))(
                delayed(process_cycle_date_pq)(cycle_date) for cycle_date in planning_cycle_dates
            )
            StatFcstL1PQLagAbsErrorBacktest = pd.concat(
                StatFcstL1PQLagAbsErrorBacktest_list, ignore_index=True
            )
            if len(StatFcstL1PQLagAbsErrorBacktest) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_L1_PQ_LAG_BACKTEST: o9Constants.STAT_FCST_L1_LAG_BACKTEST,
                    o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR_BACKTEST: o9Constants.STAT_FCST_L1_LAG_ABS_ERROR_BACKTEST,
                }
                StatFcstL1LagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstL1PQLagAbsErrorBacktest,
                    source_grain=o9Constants.PLANNING_QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_L1_PQ_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.STAT_ACTUAL,
                    ],
                )
                StatFcstL1LagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstL1LagAbsErrorBacktest) > 0:
            StatFcstL1LagAbsErrorBacktest = StatFcstL1LagAbsErrorBacktest.rename(
                columns={o9Constants.STAT_ACTUAL: stat_actual_lag_backtest_col}
            )[StatFcstL1LagAbsErrorBacktest_cols]
        else:
            StatFcstL1LagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1LagAbsErrorBacktest_cols)
        if len(StatFcstL1LagNAbsErrorBacktest) > 0:
            StatFcstL1LagNAbsErrorBacktest = StatFcstL1LagNAbsErrorBacktest[
                StatFcstL1LagNAbsErrorBacktest_cols
            ]
        else:
            StatFcstL1LagNAbsErrorBacktest = pd.DataFrame(
                columns=StatFcstL1LagNAbsErrorBacktest_cols
            )

        StatFcstL1WLagAbsErrorBacktest = StatFcstL1WLagAbsErrorBacktest[
            StatFcstL1WLagAbsErrorBacktest_cols
        ]
        StatFcstL1MLagAbsErrorBacktest = StatFcstL1MLagAbsErrorBacktest[
            StatFcstL1MLagAbsErrorBacktest_cols
        ]
        StatFcstL1PMLagAbsErrorBacktest = StatFcstL1PMLagAbsErrorBacktest[
            StatFcstL1PMLagAbsErrorBacktest_cols
        ]
        StatFcstL1QLagAbsErrorBacktest = StatFcstL1QLagAbsErrorBacktest[
            StatFcstL1QLagAbsErrorBacktest_cols
        ]
        StatFcstL1PQLagAbsErrorBacktest = StatFcstL1PQLagAbsErrorBacktest[
            StatFcstL1PQLagAbsErrorBacktest_cols
        ]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        StatFcstL1AbsErrorBacktest = pd.DataFrame(columns=StatFcstL1AbsErrorBacktest_cols)
        StatFcstL1LagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1LagAbsErrorBacktest_cols)
        StatFcstL1LagNAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1LagNAbsErrorBacktest_cols)
        StatFcstL1WLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1WLagAbsErrorBacktest_cols)
        StatFcstL1MLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1MLagAbsErrorBacktest_cols)
        StatFcstL1PMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1PMLagAbsErrorBacktest_cols)
        StatFcstL1QLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1QLagAbsErrorBacktest_cols)
        StatFcstL1PQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstL1PQLagAbsErrorBacktest_cols)
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return (
        StatFcstL1AbsErrorBacktest,
        StatFcstL1LagAbsErrorBacktest,
        StatFcstL1LagNAbsErrorBacktest,
        StatFcstL1WLagAbsErrorBacktest,
        StatFcstL1MLagAbsErrorBacktest,
        StatFcstL1PMLagAbsErrorBacktest,
        StatFcstL1QLagAbsErrorBacktest,
        StatFcstL1PQLagAbsErrorBacktest,
    )

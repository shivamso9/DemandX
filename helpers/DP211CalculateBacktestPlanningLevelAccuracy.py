import logging

import pandas as pd
from o9Reference.common_utils.common_utils import (
    get_n_time_periods,
    get_seasonal_periods,
)
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
    "Stat Fcst PL Abs Error Backtest": float,
    "Stat Fcst PL Lag Backtest": float,
    "Stat Actual PL Lag Backtest": float,
    "Stat Fcst PL Lag Abs Error Backtest": float,
    "Stat Fcst PL W Lag Abs Error Backtest": float,
    "Stat Fcst PL M Lag Abs Error Backtest": float,
    "Stat Fcst PL PM Lag Abs Error Backtest": float,
    "Stat Fcst PL Q Lag Abs Error Backtest": float,
    "Stat Fcst PL PQ Lag Abs Error Backtest": float,
    "Stat Fcst PL Lag Backtest COCC": float,
    "Stat Fcst PL Lag Backtest LC": float,
    "Actual Last N Buckets PL Backtest": float,
    "Fcst Next N Buckets PL Backtest": float,
}


def get_lag_mapping(
    df, ReasonabilityPeriods, time_mapping, time_attribute_dict, planning_cycles_list
):
    time_grain = list(time_attribute_dict.keys())[0]
    result = {}
    try:
        for cycle in planning_cycles_list:
            time_bucket = df[df[o9Constants.PLANNING_CYCLE_DATE] == cycle][time_grain].values[0]
            last_n_periods = get_n_time_periods(
                time_bucket,
                -ReasonabilityPeriods,
                time_mapping,
                time_attribute_dict,
                include_latest_value=True,
            )
            last_n_periods = last_n_periods[::-1]
            last_n_periods = pd.DataFrame({time_grain: last_n_periods})
            last_n_periods[o9Constants.LAG] = last_n_periods.index
            last_n_periods[o9Constants.PLANNING_CYCLE_DATE] = cycle
            result[cycle] = last_n_periods
        final_df = concat_to_dataframe(list(result.values()))
    except Exception as e:
        logger.exception(f"Error in get_lag_mapping: {e}")
        final_df = pd.DataFrame(
            columns=[time_grain, o9Constants.LAG, o9Constants.PLANNING_CYCLE_DATE]
        )
    return final_df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    PlanningActual,
    StatFcstPLLag1Backtest,
    StatFcstPLWLagBacktest,
    StatFcstPLMLagBacktest,
    StatFcstPLPMLagBacktest,
    StatFcstPLQLagBacktest,
    StatFcstPLPQLagBacktest,
    CurrentTimePeriod,
    TimeDimension,
    PlanningGrains,
    StatBucketWeight,
    BackTestCyclePeriod,
    ForecastGenTimeBucket,
    ReasonabilityCycles=1,
    PlanningCycleDates=pd.DataFrame(),
    ForecastIterationMasterData=pd.DataFrame(),
    default_mapping={},
    df_keys={},
):
    try:
        StatFcstPLLagBacktest_list = list()
        StatFcstPLAbsErrorBacktest_list = list()
        StatFcstPLLagAbsErrorBacktest_list = list()
        StatFcstPLWLagAbsErrorBacktest_list = list()
        StatFcstPLMLagAbsErrorBacktest_list = list()
        StatFcstPLPMLagAbsErrorBacktest_list = list()
        StatFcstPLQLagAbsErrorBacktest_list = list()
        StatFcstPLPQLagAbsErrorBacktest_list = list()
        StabilityOutput_list = list()
        FcstNextNBucketsBacktest_list = list()
        ActualsLastNBucketsBacktest_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_lag_backtest,
                all_l1_error,
                all_lag_error,
                all_w_lag_error,
                all_m_lag_error,
                all_pm_lag_error,
                all_q_lag_error,
                all_pq_lag_error,
                the_stat_fcst_l1_lag_backtest_cocc,
                fcst_next_n_buckets_backtest,
                actuals_last_n_buckets_backtest,
            ) = decorated_func(
                PlanningActual=PlanningActual,
                StatFcstPLLag1Backtest=StatFcstPLLag1Backtest,
                StatFcstPLWLagBacktest=StatFcstPLWLagBacktest,
                StatFcstPLMLagBacktest=StatFcstPLMLagBacktest,
                StatFcstPLPMLagBacktest=StatFcstPLPMLagBacktest,
                StatFcstPLQLagBacktest=StatFcstPLQLagBacktest,
                StatFcstPLPQLagBacktest=StatFcstPLPQLagBacktest,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                PlanningGrains=PlanningGrains,
                ReasonabilityCycles=ReasonabilityCycles,
                StatBucketWeight=StatBucketWeight,
                BackTestCyclePeriod=BackTestCyclePeriod,
                TimeLevel=ForecastGenTimeBucket[
                    ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] == the_iteration
                ][o9Constants.FORECAST_GEN_TIME_BUCKET].values[0],
                PlanningCycleDates=PlanningCycleDates,
                ForecastIterationMasterData=ForecastIterationMasterData,
                default_mapping=default_mapping,
                df_keys=df_keys,
            )
            StatFcstPLLagBacktest_list.append(all_lag_backtest)
            StatFcstPLAbsErrorBacktest_list.append(all_l1_error)
            StatFcstPLLagAbsErrorBacktest_list.append(all_lag_error)
            StatFcstPLWLagAbsErrorBacktest_list.append(all_w_lag_error)
            StatFcstPLMLagAbsErrorBacktest_list.append(all_m_lag_error)
            StatFcstPLPMLagAbsErrorBacktest_list.append(all_pm_lag_error)
            StatFcstPLQLagAbsErrorBacktest_list.append(all_q_lag_error)
            StatFcstPLPQLagAbsErrorBacktest_list.append(all_pq_lag_error)
            StabilityOutput_list.append(the_stat_fcst_l1_lag_backtest_cocc)
            ActualsLastNBucketsBacktest_list.append(actuals_last_n_buckets_backtest)
            FcstNextNBucketsBacktest_list.append(fcst_next_n_buckets_backtest)

        StatFcstPLLagBacktest = concat_to_dataframe(StatFcstPLLagBacktest_list)
        StatFcstPLAbsErrorBacktest = concat_to_dataframe(StatFcstPLAbsErrorBacktest_list)
        StatFcstPLLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLLagAbsErrorBacktest_list)
        StatFcstPLWLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLWLagAbsErrorBacktest_list)
        StatFcstPLMLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLMLagAbsErrorBacktest_list)
        StatFcstPLPMLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLPMLagAbsErrorBacktest_list)
        StatFcstPLQLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLQLagAbsErrorBacktest_list)
        StatFcstPLPQLagAbsErrorBacktest = concat_to_dataframe(StatFcstPLPQLagAbsErrorBacktest_list)
        StabilityOutput = concat_to_dataframe(StabilityOutput_list)
        FcstNextNBucketsBacktest = concat_to_dataframe(FcstNextNBucketsBacktest_list)
        ActualsLastNBucketsBacktest = concat_to_dataframe(ActualsLastNBucketsBacktest_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstPLLagBacktest,
            StatFcstPLAbsErrorBacktest,
            StatFcstPLLagAbsErrorBacktest,
            StatFcstPLWLagAbsErrorBacktest,
            StatFcstPLMLagAbsErrorBacktest,
            StatFcstPLPMLagAbsErrorBacktest,
            StatFcstPLQLagAbsErrorBacktest,
            StatFcstPLPQLagAbsErrorBacktest,
            StabilityOutput,
            FcstNextNBucketsBacktest,
            ActualsLastNBucketsBacktest,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    return (
        StatFcstPLLagBacktest,
        StatFcstPLAbsErrorBacktest,
        StatFcstPLLagAbsErrorBacktest,
        StatFcstPLWLagAbsErrorBacktest,
        StatFcstPLMLagAbsErrorBacktest,
        StatFcstPLPMLagAbsErrorBacktest,
        StatFcstPLQLagAbsErrorBacktest,
        StatFcstPLPQLagAbsErrorBacktest,
        StabilityOutput,
        FcstNextNBucketsBacktest,
        ActualsLastNBucketsBacktest,
    )


def check_all_dimensions(df, grains, default_mapping):
    # checks if all 7 dim are present, if not, adds the dimension with member "All"
    # Renames input stream to Actual as well
    df_copy = df.copy()
    dims = {}
    for x in grains:
        dims[x] = x.strip().split(".")[0]
    try:
        for i in dims:
            if i not in df_copy.columns:
                if dims[i] in default_mapping:
                    df_copy[i] = default_mapping[dims[i]]
                else:
                    logger.warning(
                        f'Column {i} not found in default_mapping dictionary, adding the member "All"'
                    )
                    df_copy[i] = "All"
    except Exception as e:
        logger.exception(f"Error in check_all_dimensions\nError:-{e}")
        return df
    return df_copy


def processIteration(
    PlanningActual,
    StatFcstPLLag1Backtest,
    StatFcstPLWLagBacktest,
    StatFcstPLMLagBacktest,
    StatFcstPLPMLagBacktest,
    StatFcstPLQLagBacktest,
    StatFcstPLPQLagBacktest,
    CurrentTimePeriod,
    TimeDimension,
    PlanningGrains,
    StatBucketWeight,
    BackTestCyclePeriod,
    TimeLevel="Week",
    ReasonabilityCycles=1,
    PlanningCycleDates=pd.DataFrame(),
    ForecastIterationMasterData=pd.DataFrame(),
    default_mapping={},
    df_keys={},
):
    plugin_name = "DP211CalculateBacktestPlanningLevelAccuracy"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    STAT_FCST_PL_LAG_BACKTEST = "Stat Fcst PL Lag Backtest"
    STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST = "Stat Fcst PL Lag Abs Error Backtest"
    STAT_ACTUAL_PL_LAG_BACKTEST = "Stat Actual PL Lag Backtest"
    cocc_measure = "Stat Fcst PL Lag Backtest COCC"
    lc_measure = "Stat Fcst PL Lag Backtest LC"
    actual_last_n_buckets_backtest = "Actual Last N Buckets PL Backtest"
    fcst_next_n_buckets_backtest = "Fcst Next N Buckets PL Backtest"
    PLANNING_CYCLE_DATE_KEY = "Planning Cycle.[PlanningCycleDateKey]"

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)
    StatFcstPLAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_ABS_ERROR_BACKTEST]
    )
    StatFcstPLAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLAbsErrorBacktest_cols)
    StatFcstPLLagBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [STAT_FCST_PL_LAG_BACKTEST]
    )
    StatFcstPLLagBacktest = pd.DataFrame(columns=StatFcstPLLagBacktest_cols)
    StatFcstPLLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST, STAT_ACTUAL_PL_LAG_BACKTEST]
    )
    StatFcstPLLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLLagAbsErrorBacktest_cols)
    StatFcstPLWLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstPLWLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLWLagAbsErrorBacktest_cols)

    StatFcstPLMLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstPLMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLMLagAbsErrorBacktest_cols)

    StatFcstPLPMLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstPLPMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLPMLagAbsErrorBacktest_cols)

    StatFcstPLQLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstPLQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLQLagAbsErrorBacktest_cols)

    StatFcstPLPQLagAbsErrorBacktest_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR_BACKTEST]
    )
    StatFcstPLPQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLPQLagAbsErrorBacktest_cols)
    StabilityOutput_cols = (
        [o9Constants.VERSION_NAME]
        + planning_grains
        + [o9Constants.PLANNING_CYCLE_DATE, o9Constants.PARTIAL_WEEK, cocc_measure, lc_measure]
    )
    StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)
    FcstNextNBucketsBacktest_cols = (
        [o9Constants.VERSION_NAME]
        + planning_grains
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            fcst_next_n_buckets_backtest,
        ]
    )
    FcstNextNBucketsBacktest = pd.DataFrame(columns=FcstNextNBucketsBacktest_cols)
    ActualsLastNBucketsBacktest_cols = (
        [o9Constants.VERSION_NAME]
        + planning_grains
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            actual_last_n_buckets_backtest,
        ]
    )
    ActualsLastNBucketsBacktest = pd.DataFrame(columns=ActualsLastNBucketsBacktest_cols)

    try:
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return (
                StatFcstPLLagBacktest,
                StatFcstPLAbsErrorBacktest,
                StatFcstPLLagAbsErrorBacktest,
                StatFcstPLWLagAbsErrorBacktest,
                StatFcstPLMLagAbsErrorBacktest,
                StatFcstPLPMLagAbsErrorBacktest,
                StatFcstPLQLagAbsErrorBacktest,
                StatFcstPLPQLagAbsErrorBacktest,
                StabilityOutput,
                FcstNextNBucketsBacktest,
                ActualsLastNBucketsBacktest,
            )

        history_measure = input_stream
        if isinstance(PlanningActual, dict):
            df = next(
                (df for key, df in PlanningActual.items() if input_stream in df.columns),
                None,
            )
            if df is None:
                logger.warning(
                    f"Input Stream '{input_stream}' not found, returning empty dataframe..."
                )
                return (
                    StatFcstPLLagBacktest,
                    StatFcstPLAbsErrorBacktest,
                    StatFcstPLLagAbsErrorBacktest,
                    StatFcstPLWLagAbsErrorBacktest,
                    StatFcstPLMLagAbsErrorBacktest,
                    StatFcstPLPMLagAbsErrorBacktest,
                    StatFcstPLQLagAbsErrorBacktest,
                    StatFcstPLPQLagAbsErrorBacktest,
                    StabilityOutput,
                    FcstNextNBucketsBacktest,
                    ActualsLastNBucketsBacktest,
                )

            PlanningActual_df = df
        else:
            PlanningActual_df = PlanningActual
        if PlanningActual_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return (
                StatFcstPLLagBacktest,
                StatFcstPLAbsErrorBacktest,
                StatFcstPLLagAbsErrorBacktest,
                StatFcstPLWLagAbsErrorBacktest,
                StatFcstPLMLagAbsErrorBacktest,
                StatFcstPLPMLagAbsErrorBacktest,
                StatFcstPLQLagAbsErrorBacktest,
                StatFcstPLPQLagAbsErrorBacktest,
                StabilityOutput,
                FcstNextNBucketsBacktest,
                ActualsLastNBucketsBacktest,
            )

        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.lower() == "planning month"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.lower() == "planning quarter"
        StatBucketWeight = StatBucketWeight.merge(
            TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
        )
        freq_mapping = {
            "week": "Weekly",
            "month": "Monthly",
            "planning month": "Monthly",
            "quarter": "Quarterly",
            "planning quarter": "Quarterly",
        }
        seasonalPeriod = get_seasonal_periods(frequency=freq_mapping[TimeLevel.lower()])

        # round off reasonability cycles to 2 decimal places - 0.0083 should become 0.01
        ReasonabilityCycles = round(int(ReasonabilityCycles), 2)

        # multiply with seasonal periods to get reasonability periods
        ReasonabilityPeriods = ReasonabilityCycles * seasonalPeriod

        if len(StatFcstPLLag1Backtest) != 0:
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
            StatFcstPLLag1Backtest = StatFcstPLLag1Backtest.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, time_grain]],
                on=o9Constants.PARTIAL_WEEK,
                how="inner",
            )
            StatFcstPLLag1Backtest = (
                StatFcstPLLag1Backtest.groupby(
                    planning_grains + [o9Constants.VERSION_NAME, time_grain]
                )
                .agg({o9Constants.STAT_FCST_PL_LAG1_BACKTEST: "sum"})
                .reset_index()
            )
            latest_time_period = CurrentTimePeriod[time_grain][0]
            relevant_time_mapping = TimeDimension[[time_grain, time_key]].drop_duplicates()
            time_attribute_dict = {time_grain: time_key}
            relevant_time_grains = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PARTIAL_WEEK_KEY,
            ]
            actual_cols = PlanningActual_df.columns

            if time_grain != o9Constants.PARTIAL_WEEK:
                relevant_time_grains = [
                    o9Constants.PARTIAL_WEEK,
                    o9Constants.PARTIAL_WEEK_KEY,
                    time_grain,
                    time_key,
                ]
            PlanningActual_df = PlanningActual_df.merge(
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
            PlanningActual_df_abs_error = PlanningActual_df[
                PlanningActual_df[time_grain].isin(last_n_time_periods)
            ]
            PlanningActual_df_abs_error = PlanningActual_df_abs_error[actual_cols].drop_duplicates()
            StatFcstPLLag1Backtest = StatFcstPLLag1Backtest[
                StatFcstPLLag1Backtest[time_grain].isin(last_n_time_periods)
            ]
            StatFcstPLAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLLag1Backtest,
                Actuals=PlanningActual_df_abs_error,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=time_grain,
                time_key=time_key,
                source_measure=o9Constants.STAT_FCST_PL_LAG1_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLAbsErrorBacktest_cols + [time_grain],
                cycle_period=BackTestCyclePeriod,
            )
            StatFcstPLAbsErrorBacktest = StatFcstPLAbsErrorBacktest.drop(
                columns=[o9Constants.PARTIAL_WEEK]
            )
            StatFcstPLAbsErrorBacktest = disaggregate_data(
                source_df=StatFcstPLAbsErrorBacktest,
                source_grain=time_grain,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_ABS_ERROR_BACKTEST],
            )
        if len(StatFcstPLWLagBacktest) != 0 and week:
            col_mapping = {o9Constants.STAT_FCST_PL_W_LAG_BACKTEST: STAT_FCST_PL_LAG_BACKTEST}
            StatFcstPLLagBacktest = disaggregate_data(
                source_df=StatFcstPLWLagBacktest,
                source_grain=o9Constants.WEEK,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[
                    o9Constants.STAT_FCST_PL_W_LAG_BACKTEST,
                ],
            )
            StatFcstPLLagBacktest = StatFcstPLLagBacktest.merge(
                PlanningActual_df[
                    [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                    + planning_grains
                ].drop_duplicates(),
                on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + planning_grains,
                how="left",
            )
            StatFcstPLLagBacktest.rename(columns=col_mapping, inplace=True)
            CurrentPlanningCycleDate = CurrentTimePeriod[o9Constants.WEEK][0]
            last_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                -int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.WEEK, o9Constants.WEEK_KEY]].drop_duplicates(),
                {o9Constants.WEEK: o9Constants.WEEK_KEY},
                include_latest_value=True,
            )
            next_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.WEEK, o9Constants.WEEK_KEY]].drop_duplicates(),
                {o9Constants.WEEK: o9Constants.WEEK_KEY},
                include_latest_value=True,
            )
            relevant_partial_weeks = TimeDimension[
                TimeDimension[o9Constants.WEEK].isin(last_n_time_periods + next_n_time_periods)
            ][o9Constants.PARTIAL_WEEK].drop_duplicates()
            StatFcstPLLagBacktest_PW = disaggregate_data(
                source_df=StatFcstPLWLagBacktest,
                source_grain=o9Constants.WEEK,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_W_LAG_BACKTEST],
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW[
                StatFcstPLLagBacktest_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]

            StatFcstPLLagBacktest_PW.rename(
                columns={o9Constants.STAT_FCST_PL_W_LAG_BACKTEST: fcst_next_n_buckets_backtest},
                inplace=True,
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({fcst_next_n_buckets_backtest: "sum"})
            FcstNextNBucketsBacktest = StatFcstPLLagBacktest_PW[
                FcstNextNBucketsBacktest_cols
            ].drop_duplicates()
            PlanningActual_df_week = PlanningActual_df.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PARTIAL_WEEK_KEY]
            )
            PlanningActual_df_week = PlanningActual_df_week[
                PlanningActual_df_week[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            PlanningActual_df_week_merge = PlanningActual_df_week.merge(
                PlanningCycleDates,
                left_on=o9Constants.PARTIAL_WEEK_KEY,
                right_on=PLANNING_CYCLE_DATE_KEY,
                how="inner",
            )
            lag_df = get_lag_mapping(
                df=PlanningActual_df_week_merge,
                ReasonabilityPeriods=ReasonabilityPeriods,
                time_mapping=TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.WEEK, o9Constants.WEEK_KEY]
                ].drop_duplicates(),
                time_attribute_dict={o9Constants.WEEK: o9Constants.WEEK_KEY},
                planning_cycles_list=list(
                    set(StatFcstPLWLagBacktest[o9Constants.PLANNING_CYCLE_DATE].values)
                ),
            )
            PlanningActual_df_week = PlanningActual_df_week.merge(
                lag_df, on=[o9Constants.WEEK], how="left"
            )
            PlanningActual_df_week.rename(
                columns={o9Constants.ACTUAL: actual_last_n_buckets_backtest}, inplace=True
            )
            PlanningActual_df_week = PlanningActual_df_week.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({actual_last_n_buckets_backtest: "sum"})
            ActualsLastNBucketsBacktest = PlanningActual_df_week[
                ActualsLastNBucketsBacktest_cols
            ].drop_duplicates()
            StabilityOutput = StatFcstPLWLagBacktest.rename(
                columns={o9Constants.STAT_FCST_PL_W_LAG_BACKTEST: lc_measure}
            )
            StabilityOutput.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.WEEK], inplace=True
            )
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.WEEK]
                )[lc_measure]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=o9Constants.WEEK,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
            PlanningActual_df = PlanningActual_df[
                [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                + planning_grains
            ].drop_duplicates()
            StatFcstPLWLagAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLWLagBacktest,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.WEEK,
                time_key=o9Constants.WEEK_KEY,
                source_measure=o9Constants.STAT_FCST_PL_W_LAG_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLWLagAbsErrorBacktest_cols
                + [o9Constants.STAT_FCST_PL_W_LAG_BACKTEST, o9Constants.ACTUAL],
                cycle_period=BackTestCyclePeriod,
                df_keys=df_keys,
            )

            if len(StatFcstPLWLagAbsErrorBacktest) != 0:
                StatFcstPLWLagAbsErrorBacktest = StatFcstPLWLagAbsErrorBacktest.dropna(
                    subset=[o9Constants.PLANNING_CYCLE_DATE, o9Constants.LAG]
                )
                col_mapping = {
                    o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR_BACKTEST: STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST,
                    o9Constants.ACTUAL: STAT_ACTUAL_PL_LAG_BACKTEST,
                }
                StatFcstPLLagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstPLWLagAbsErrorBacktest,
                    source_grain=o9Constants.WEEK,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_W_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.ACTUAL,
                    ],
                )
                StatFcstPLLagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)

        if len(StatFcstPLMLagBacktest) != 0 and month:
            col_mapping = {
                o9Constants.STAT_FCST_PL_M_LAG_BACKTEST: STAT_FCST_PL_LAG_BACKTEST,
            }
            StatFcstPLLagBacktest = disaggregate_data(
                source_df=StatFcstPLMLagBacktest,
                source_grain=o9Constants.MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[
                    o9Constants.STAT_FCST_PL_M_LAG_BACKTEST,
                    o9Constants.ACTUAL,
                ],
            )
            StatFcstPLLagBacktest = StatFcstPLLagBacktest.merge(
                PlanningActual_df[
                    [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                    + planning_grains
                ].drop_duplicates(),
                on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + planning_grains,
                how="left",
            )
            StatFcstPLLagBacktest.rename(columns=col_mapping, inplace=True)
            CurrentPlanningCycleDate = CurrentTimePeriod[o9Constants.MONTH][0]
            last_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                -int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                {o9Constants.MONTH: o9Constants.MONTH_KEY},
                include_latest_value=True,
            )
            next_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                {o9Constants.MONTH: o9Constants.MONTH_KEY},
                include_latest_value=True,
            )
            relevant_partial_weeks = TimeDimension[
                TimeDimension[o9Constants.MONTH].isin(last_n_time_periods + next_n_time_periods)
            ][o9Constants.PARTIAL_WEEK].drop_duplicates()
            StatFcstPLLagBacktest_PW = disaggregate_data(
                source_df=StatFcstPLMLagBacktest,
                source_grain=o9Constants.MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_M_LAG_BACKTEST],
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW[
                StatFcstPLLagBacktest_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            StatFcstPLLagBacktest_PW.rename(
                columns={o9Constants.STAT_FCST_PL_M_LAG_BACKTEST: fcst_next_n_buckets_backtest},
                inplace=True,
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({fcst_next_n_buckets_backtest: "sum"})
            FcstNextNBucketsBacktest = StatFcstPLLagBacktest_PW[
                FcstNextNBucketsBacktest_cols
            ].drop_duplicates()
            PlanningActual_df_month = PlanningActual_df.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PARTIAL_WEEK_KEY]
            )
            PlanningActual_df_month = PlanningActual_df_month[
                PlanningActual_df_month[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            PlanningActual_df_month_merge = PlanningActual_df_month.merge(
                PlanningCycleDates,
                left_on=o9Constants.PARTIAL_WEEK_KEY,
                right_on=PLANNING_CYCLE_DATE_KEY,
                how="inner",
            )
            lag_df = get_lag_mapping(
                df=PlanningActual_df_month_merge,
                ReasonabilityPeriods=ReasonabilityPeriods,
                time_mapping=TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.MONTH, o9Constants.MONTH_KEY]
                ].drop_duplicates(),
                time_attribute_dict={o9Constants.MONTH: o9Constants.MONTH_KEY},
                planning_cycles_list=list(
                    set(StatFcstPLMLagBacktest[o9Constants.PLANNING_CYCLE_DATE].values)
                ),
            )
            PlanningActual_df_month = PlanningActual_df_month.merge(
                lag_df, on=[o9Constants.MONTH], how="left"
            )
            PlanningActual_df_month.rename(
                columns={o9Constants.ACTUAL: actual_last_n_buckets_backtest}, inplace=True
            )
            PlanningActual_df_month = PlanningActual_df_month.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({actual_last_n_buckets_backtest: "sum"})
            ActualsLastNBucketsBacktest = PlanningActual_df_month[
                ActualsLastNBucketsBacktest_cols
            ].drop_duplicates()
            StabilityOutput = StatFcstPLMLagBacktest.rename(
                columns={o9Constants.STAT_FCST_PL_M_LAG_BACKTEST: lc_measure}
            )
            StabilityOutput.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.MONTH], inplace=True
            )
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.MONTH]
                )[lc_measure]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=o9Constants.MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
            PlanningActual_df = PlanningActual_df[
                [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                + planning_grains
            ].drop_duplicates()
            StatFcstPLMLagAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLMLagBacktest,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                source_measure=o9Constants.STAT_FCST_PL_M_LAG_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLMLagAbsErrorBacktest_cols
                + [o9Constants.STAT_FCST_PL_M_LAG_BACKTEST, o9Constants.ACTUAL],
                cycle_period=BackTestCyclePeriod,
                df_keys=df_keys,
            )
            if len(StatFcstPLMLagAbsErrorBacktest) != 0:
                StatFcstPLMLagAbsErrorBacktest = StatFcstPLMLagAbsErrorBacktest.dropna(
                    subset=[o9Constants.PLANNING_CYCLE_DATE, o9Constants.LAG]
                )
                col_mapping = {
                    o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR_BACKTEST: STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST,
                    o9Constants.ACTUAL: STAT_ACTUAL_PL_LAG_BACKTEST,
                }
                StatFcstPLLagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstPLMLagAbsErrorBacktest,
                    source_grain=o9Constants.MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_M_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.ACTUAL,
                    ],
                )
                StatFcstPLLagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstPLPMLagBacktest) != 0 and pl_month:
            col_mapping = {
                o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST: STAT_FCST_PL_LAG_BACKTEST,
            }
            # Disaggregate planning month lag backtest to partial week
            StatFcstPLLagBacktest = disaggregate_data(
                source_df=StatFcstPLPMLagBacktest,
                source_grain=o9Constants.PLANNING_MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[
                    o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST,
                    o9Constants.ACTUAL,
                ],
            )
            StatFcstPLLagBacktest = StatFcstPLLagBacktest.merge(
                PlanningActual_df[
                    [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                    + planning_grains
                ].drop_duplicates(),
                on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + planning_grains,
                how="left",
            )
            StatFcstPLLagBacktest.rename(columns=col_mapping, inplace=True)
            CurrentPlanningCycleDate = CurrentTimePeriod[o9Constants.PLANNING_MONTH][0]
            last_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                -int(ReasonabilityPeriods),
                TimeDimension[
                    [o9Constants.PLANNING_MONTH, o9Constants.PLANNING_MONTH_KEY]
                ].drop_duplicates(),
                {o9Constants.PLANNING_MONTH: o9Constants.PLANNING_MONTH_KEY},
                include_latest_value=True,
            )
            next_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                int(ReasonabilityPeriods),
                TimeDimension[
                    [o9Constants.PLANNING_MONTH, o9Constants.PLANNING_MONTH_KEY]
                ].drop_duplicates(),
                {o9Constants.PLANNING_MONTH: o9Constants.PLANNING_MONTH_KEY},
                include_latest_value=True,
            )
            relevant_partial_weeks = TimeDimension[
                TimeDimension[o9Constants.PLANNING_MONTH].isin(
                    last_n_time_periods + next_n_time_periods
                )
            ][o9Constants.PARTIAL_WEEK].drop_duplicates()
            StatFcstPLLagBacktest_PW = disaggregate_data(
                source_df=StatFcstPLPMLagBacktest,
                source_grain=o9Constants.PLANNING_MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST],
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW[
                StatFcstPLLagBacktest_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            StatFcstPLLagBacktest_PW.rename(
                columns={o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST: fcst_next_n_buckets_backtest},
                inplace=True,
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({fcst_next_n_buckets_backtest: "sum"})
            FcstNextNBucketsBacktest = StatFcstPLLagBacktest_PW[
                FcstNextNBucketsBacktest_cols
            ].drop_duplicates()
            PlanningActual_df_pm = PlanningActual_df.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PARTIAL_WEEK_KEY]
            )
            PlanningActual_df_pm = PlanningActual_df_pm[
                PlanningActual_df_pm[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]

            PlanningActual_df_pm_merge = PlanningActual_df_pm.merge(
                PlanningCycleDates,
                left_on=o9Constants.PARTIAL_WEEK_KEY,
                right_on=PLANNING_CYCLE_DATE_KEY,
                how="inner",
            )
            lag_df = get_lag_mapping(
                df=PlanningActual_df_pm_merge,
                ReasonabilityPeriods=ReasonabilityPeriods,
                time_mapping=TimeDimension[
                    [
                        o9Constants.PARTIAL_WEEK,
                        o9Constants.PLANNING_MONTH,
                        o9Constants.PLANNING_MONTH_KEY,
                    ]
                ].drop_duplicates(),
                time_attribute_dict={o9Constants.PLANNING_MONTH: o9Constants.PLANNING_MONTH_KEY},
                planning_cycles_list=list(
                    set(StatFcstPLPMLagBacktest[o9Constants.PLANNING_CYCLE_DATE].values)
                ),
            )
            PlanningActual_df_pm = PlanningActual_df_pm.merge(
                lag_df, on=[o9Constants.PLANNING_MONTH], how="left"
            )
            PlanningActual_df_pm.rename(
                columns={o9Constants.ACTUAL: actual_last_n_buckets_backtest}, inplace=True
            )
            PlanningActual_df_pm = PlanningActual_df_pm.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({actual_last_n_buckets_backtest: "sum"})
            ActualsLastNBucketsBacktest = PlanningActual_df_pm[
                ActualsLastNBucketsBacktest_cols
            ].drop_duplicates()
            StabilityOutput = StatFcstPLPMLagBacktest.rename(
                columns={o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST: lc_measure}
            )
            StabilityOutput.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PLANNING_MONTH],
                inplace=True,
            )
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PLANNING_MONTH]
                )[lc_measure]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=o9Constants.PLANNING_MONTH,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
            PlanningActual_df = PlanningActual_df[
                [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                + planning_grains
            ].drop_duplicates()
            StatFcstPLPMLagAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLPMLagBacktest,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_MONTH,
                time_key=o9Constants.PLANNING_MONTH_KEY,
                source_measure=o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLPMLagAbsErrorBacktest_cols
                + [o9Constants.STAT_FCST_PL_PM_LAG_BACKTEST, o9Constants.ACTUAL],
                cycle_period=BackTestCyclePeriod,
                df_keys=df_keys,
            )
            if len(StatFcstPLPMLagAbsErrorBacktest) != 0:
                StatFcstPLPMLagAbsErrorBacktest = StatFcstPLPMLagAbsErrorBacktest.dropna(
                    subset=[o9Constants.PLANNING_CYCLE_DATE, o9Constants.LAG]
                )
                col_mapping = {
                    o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR_BACKTEST: STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST,
                    o9Constants.ACTUAL: STAT_ACTUAL_PL_LAG_BACKTEST,
                }
                StatFcstPLLagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstPLPMLagAbsErrorBacktest,
                    source_grain=o9Constants.PLANNING_MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_PM_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.ACTUAL,
                    ],
                )
                StatFcstPLLagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)

        if len(StatFcstPLQLagBacktest) != 0 and quarter:
            col_mapping = {
                o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST: STAT_FCST_PL_LAG_BACKTEST,
            }
            StatFcstPLLagBacktest = disaggregate_data(
                source_df=StatFcstPLQLagBacktest,
                source_grain=o9Constants.QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[
                    o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST,
                ],
            )
            StatFcstPLLagBacktest = StatFcstPLLagBacktest.merge(
                PlanningActual_df[
                    [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                    + planning_grains
                ].drop_duplicates(),
                on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + planning_grains,
                how="left",
            )
            StatFcstPLLagBacktest.rename(columns=col_mapping, inplace=True)
            CurrentPlanningCycleDate = CurrentTimePeriod[o9Constants.QUARTER][0]
            last_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                -int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.QUARTER, o9Constants.QUARTER_KEY]].drop_duplicates(),
                {o9Constants.QUARTER: o9Constants.QUARTER_KEY},
                include_latest_value=True,
            )
            next_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                int(ReasonabilityPeriods),
                TimeDimension[[o9Constants.QUARTER, o9Constants.QUARTER_KEY]].drop_duplicates(),
                {o9Constants.QUARTER: o9Constants.QUARTER_KEY},
                include_latest_value=True,
            )
            relevant_partial_weeks = TimeDimension[
                TimeDimension[o9Constants.QUARTER].isin(last_n_time_periods + next_n_time_periods)
            ][o9Constants.PARTIAL_WEEK].drop_duplicates()
            StatFcstPLLagBacktest_PW = disaggregate_data(
                source_df=StatFcstPLQLagBacktest,
                source_grain=o9Constants.QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST],
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW[
                StatFcstPLLagBacktest_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            StatFcstPLLagBacktest_PW.rename(
                columns={o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST: fcst_next_n_buckets_backtest},
                inplace=True,
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({fcst_next_n_buckets_backtest: "sum"})
            FcstNextNBucketsBacktest = StatFcstPLLagBacktest_PW[
                FcstNextNBucketsBacktest_cols
            ].drop_duplicates()
            PlanningActual_df_q = PlanningActual_df.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PARTIAL_WEEK_KEY]
            )
            PlanningActual_df_q = PlanningActual_df_q[
                PlanningActual_df_q[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            PlanningActual_df_q_merge = PlanningActual_df_q.merge(
                PlanningCycleDates,
                left_on=o9Constants.PARTIAL_WEEK_KEY,
                right_on=PLANNING_CYCLE_DATE_KEY,
                how="inner",
            )
            lag_df = get_lag_mapping(
                df=PlanningActual_df_q_merge,
                ReasonabilityPeriods=ReasonabilityPeriods,
                time_mapping=TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.QUARTER, o9Constants.QUARTER_KEY]
                ].drop_duplicates(),
                time_attribute_dict={o9Constants.QUARTER: o9Constants.QUARTER_KEY},
                planning_cycles_list=list(
                    set(StatFcstPLQLagBacktest[o9Constants.PLANNING_CYCLE_DATE].values)
                ),
            )
            PlanningActual_df_q = PlanningActual_df_q.merge(
                lag_df, on=[o9Constants.QUARTER], how="left"
            )
            PlanningActual_df_q.rename(
                columns={o9Constants.ACTUAL: actual_last_n_buckets_backtest}, inplace=True
            )
            PlanningActual_df_q = PlanningActual_df_q.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({actual_last_n_buckets_backtest: "sum"})
            ActualsLastNBucketsBacktest = PlanningActual_df_q[
                ActualsLastNBucketsBacktest_cols
            ].drop_duplicates()
            StabilityOutput = StatFcstPLQLagBacktest.rename(
                columns={o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST: lc_measure}
            )
            StabilityOutput.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.QUARTER],
                inplace=True,
            )
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.QUARTER]
                )[lc_measure]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=o9Constants.QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
            PlanningActual_df = PlanningActual_df[
                [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                + planning_grains
            ].drop_duplicates()
            StatFcstPLQLagAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLQLagBacktest,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.QUARTER,
                time_key=o9Constants.QUARTER_KEY,
                source_measure=o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLQLagAbsErrorBacktest_cols
                + [o9Constants.STAT_FCST_PL_Q_LAG_BACKTEST, o9Constants.ACTUAL],
                cycle_period=BackTestCyclePeriod,
                df_keys=df_keys,
            )
            if len(StatFcstPLQLagAbsErrorBacktest) != 0:
                StatFcstPLQLagAbsErrorBacktest = StatFcstPLQLagAbsErrorBacktest.dropna(
                    subset=[o9Constants.PLANNING_CYCLE_DATE, o9Constants.LAG]
                )
                col_mapping = {
                    o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR_BACKTEST: STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST,
                    o9Constants.ACTUAL: STAT_ACTUAL_PL_LAG_BACKTEST,
                }
                StatFcstPLLagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstPLQLagAbsErrorBacktest,
                    source_grain=o9Constants.QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_Q_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.ACTUAL,
                    ],
                )
                StatFcstPLLagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)
        if len(StatFcstPLPQLagBacktest) != 0 and pl_quarter:
            col_mapping = {
                o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST: STAT_FCST_PL_LAG_BACKTEST,
            }
            # Disaggregate planning quarter lag backtest to partial week
            StatFcstPLLagBacktest = disaggregate_data(
                source_df=StatFcstPLPQLagBacktest,
                source_grain=o9Constants.PLANNING_QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[
                    o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST,
                ],
            )
            StatFcstPLLagBacktest = StatFcstPLLagBacktest.merge(
                PlanningActual_df[
                    [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                    + planning_grains
                ].drop_duplicates(),
                on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + planning_grains,
                how="left",
            )
            StatFcstPLLagBacktest.rename(columns=col_mapping, inplace=True)
            CurrentPlanningCycleDate = CurrentTimePeriod[o9Constants.PLANNING_QUARTER][0]
            last_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                -int(ReasonabilityPeriods),
                TimeDimension[
                    [o9Constants.PLANNING_QUARTER, o9Constants.PLANNING_QUARTER_KEY]
                ].drop_duplicates(),
                {o9Constants.PLANNING_QUARTER: o9Constants.PLANNING_QUARTER_KEY},
                include_latest_value=True,
            )
            next_n_time_periods = get_n_time_periods(
                CurrentPlanningCycleDate,
                int(ReasonabilityPeriods),
                TimeDimension[
                    [o9Constants.PLANNING_QUARTER, o9Constants.PLANNING_QUARTER_KEY]
                ].drop_duplicates(),
                {o9Constants.PLANNING_QUARTER: o9Constants.PLANNING_QUARTER_KEY},
                include_latest_value=True,
            )
            relevant_partial_weeks = TimeDimension[
                TimeDimension[o9Constants.PLANNING_QUARTER].isin(
                    last_n_time_periods + next_n_time_periods
                )
            ][o9Constants.PARTIAL_WEEK].drop_duplicates()
            StatFcstPLLagBacktest_PW = disaggregate_data(
                source_df=StatFcstPLPQLagBacktest,
                source_grain=o9Constants.PLANNING_QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST],
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW[
                StatFcstPLLagBacktest_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            StatFcstPLLagBacktest_PW.rename(
                columns={o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST: fcst_next_n_buckets_backtest},
                inplace=True,
            )
            StatFcstPLLagBacktest_PW = StatFcstPLLagBacktest_PW.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({fcst_next_n_buckets_backtest: "sum"})
            FcstNextNBucketsBacktest = StatFcstPLLagBacktest_PW[
                FcstNextNBucketsBacktest_cols
            ].drop_duplicates()
            PlanningActual_df_pq = PlanningActual_df.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PARTIAL_WEEK_KEY]
            )
            PlanningActual_df_pq = PlanningActual_df_pq[
                PlanningActual_df_pq[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
            ]
            PlanningActual_df_pq_merge = PlanningActual_df_pq.merge(
                PlanningCycleDates,
                left_on=o9Constants.PARTIAL_WEEK_KEY,
                right_on=PLANNING_CYCLE_DATE_KEY,
                how="inner",
            )
            lag_df = get_lag_mapping(
                df=PlanningActual_df_pq_merge,
                ReasonabilityPeriods=ReasonabilityPeriods,
                time_mapping=TimeDimension[
                    [
                        o9Constants.PARTIAL_WEEK,
                        o9Constants.PLANNING_QUARTER,
                        o9Constants.PLANNING_QUARTER_KEY,
                    ]
                ].drop_duplicates(),
                time_attribute_dict={
                    o9Constants.PLANNING_QUARTER: o9Constants.PLANNING_QUARTER_KEY
                },
                planning_cycles_list=list(
                    set(StatFcstPLMLagBacktest[o9Constants.PLANNING_CYCLE_DATE].values)
                ),
            )
            PlanningActual_df_pq = PlanningActual_df_pq.merge(
                lag_df, on=[o9Constants.PLANNING_QUARTER], how="left"
            )
            PlanningActual_df_pq.rename(
                columns={o9Constants.ACTUAL: actual_last_n_buckets_backtest}, inplace=True
            )
            PlanningActual_df_pq = PlanningActual_df_pq.groupby(
                planning_grains + [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE],
                as_index=False,
            ).agg({actual_last_n_buckets_backtest: "sum"})
            ActualsLastNBucketsBacktest = PlanningActual_df_pq[
                ActualsLastNBucketsBacktest_cols
            ].drop_duplicates()
            StabilityOutput = StatFcstPLPQLagBacktest.rename(
                columns={o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST: lc_measure}
            )
            StabilityOutput.sort_values(
                by=[o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PLANNING_QUARTER],
                inplace=True,
            )
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby(
                    [o9Constants.VERSION_NAME] + planning_grains + [o9Constants.PLANNING_QUARTER]
                )[lc_measure]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=o9Constants.PLANNING_QUARTER,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
            PlanningActual_df = PlanningActual_df[
                [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, o9Constants.ACTUAL]
                + planning_grains
            ].drop_duplicates()
            StatFcstPLPQLagAbsErrorBacktest = get_abs_error(
                source_df=StatFcstPLPQLagBacktest,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_QUARTER,
                time_key=o9Constants.PLANNING_QUARTER_KEY,
                source_measure=o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST,
                actual_measure=history_measure,
                output_measure=o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR_BACKTEST,
                output_cols=StatFcstPLPQLagAbsErrorBacktest_cols
                + [o9Constants.STAT_FCST_PL_PQ_LAG_BACKTEST, o9Constants.ACTUAL],
                cycle_period=BackTestCyclePeriod,
                df_keys=df_keys,
            )
            if len(StatFcstPLPQLagAbsErrorBacktest) != 0:
                StatFcstPLPQLagAbsErrorBacktest = StatFcstPLPQLagAbsErrorBacktest.dropna(
                    subset=[o9Constants.PLANNING_CYCLE_DATE, o9Constants.LAG]
                )
                col_mapping = {
                    o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR_BACKTEST: STAT_FCST_PL_LAG_ABS_ERROR_BACKTEST,
                    o9Constants.ACTUAL: STAT_ACTUAL_PL_LAG_BACKTEST,
                }
                StatFcstPLLagAbsErrorBacktest = disaggregate_data(
                    source_df=StatFcstPLPQLagAbsErrorBacktest,
                    source_grain=o9Constants.PLANNING_QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PL_PQ_LAG_ABS_ERROR_BACKTEST,
                        o9Constants.ACTUAL,
                    ],
                )
                StatFcstPLLagAbsErrorBacktest.rename(columns=col_mapping, inplace=True)

        StatFcstPLAbsErrorBacktest = StatFcstPLAbsErrorBacktest[StatFcstPLAbsErrorBacktest_cols]
        StatFcstPLLagAbsErrorBacktest = StatFcstPLLagAbsErrorBacktest[
            StatFcstPLLagAbsErrorBacktest_cols
        ]
        StatFcstPLWLagAbsErrorBacktest = StatFcstPLWLagAbsErrorBacktest[
            StatFcstPLWLagAbsErrorBacktest_cols
        ]
        StatFcstPLMLagAbsErrorBacktest = StatFcstPLMLagAbsErrorBacktest[
            StatFcstPLMLagAbsErrorBacktest_cols
        ]
        StatFcstPLPMLagAbsErrorBacktest = StatFcstPLPMLagAbsErrorBacktest[
            StatFcstPLPMLagAbsErrorBacktest_cols
        ]
        StatFcstPLQLagAbsErrorBacktest = StatFcstPLQLagAbsErrorBacktest[
            StatFcstPLQLagAbsErrorBacktest_cols
        ]
        StatFcstPLPQLagAbsErrorBacktest = StatFcstPLPQLagAbsErrorBacktest[
            StatFcstPLPQLagAbsErrorBacktest_cols
        ]
        FcstNextNBucketsBacktest = FcstNextNBucketsBacktest[FcstNextNBucketsBacktest_cols]
        ActualsLastNBucketsBacktest = ActualsLastNBucketsBacktest[ActualsLastNBucketsBacktest_cols]
        StabilityOutput = StabilityOutput[StabilityOutput_cols]
        StatFcstPLLagBacktest = StatFcstPLLagBacktest[StatFcstPLLagBacktest_cols]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        StatFcstPLAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLAbsErrorBacktest_cols)
        StatFcstPLLagBacktest = pd.DataFrame(columns=StatFcstPLLagBacktest_cols)
        StatFcstPLLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLLagAbsErrorBacktest_cols)
        StatFcstPLWLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLWLagAbsErrorBacktest_cols)
        StatFcstPLMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLMLagAbsErrorBacktest_cols)
        StatFcstPLPMLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLPMLagAbsErrorBacktest_cols)
        StatFcstPLQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLQLagAbsErrorBacktest_cols)
        StatFcstPLPQLagAbsErrorBacktest = pd.DataFrame(columns=StatFcstPLPQLagAbsErrorBacktest_cols)
        StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)
        FcstNextNBucketsBacktest = pd.DataFrame(columns=FcstNextNBucketsBacktest_cols)
        ActualsLastNBucketsBacktest = pd.DataFrame(columns=ActualsLastNBucketsBacktest_cols)

    return (
        StatFcstPLLagBacktest,
        StatFcstPLAbsErrorBacktest,
        StatFcstPLLagAbsErrorBacktest,
        StatFcstPLWLagAbsErrorBacktest,
        StatFcstPLMLagAbsErrorBacktest,
        StatFcstPLPMLagAbsErrorBacktest,
        StatFcstPLQLagAbsErrorBacktest,
        StatFcstPLPQLagAbsErrorBacktest,
        StabilityOutput,
        FcstNextNBucketsBacktest,
        ActualsLastNBucketsBacktest,
    )

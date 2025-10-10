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
    "Stat Fcst PL Abs Error": float,
    "Stat Fcst PL Lag": float,
    "Stat Fcst PL Lag Abs Error": float,
    "Stat Fcst W Lag Abs Error": float,
    "Stat Fcst M Lag Abs Error": float,
    "Stat Fcst PM Lag Abs Error": float,
    "Stat Fcst Q Lag Abs Error": float,
    "Stat Fcst PQ Lag Abs Error": float,
    "Actual Lag Backtest": float,
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
    ForecastGenTimeBucket,
    ForecastIterationMasterData,
    CurrentTimePeriod,
    TimeDimension,
    StatBucketWeight,
    PlanningGrains,
    AccuracyWindow="1",
    df_keys={},
):
    try:
        StatFcstAbsError_list = list()
        StatFcstPLLagAbsError_list = list()
        StatFcstPLWLagAbsError_list = list()
        StatFcstPLMLagAbsError_list = list()
        StatFcstPLPMLagAbsError_list = list()
        StatFcstPLQLagAbsError_list = list()
        StatFcstPLPQLagAbsError_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_PL_error,
                all_lag_error,
                all_w_lag_error,
                all_m_lag_error,
                all_pm_lag_error,
                all_q_lag_error,
                all_pq_lag_error,
            ) = decorated_func(
                PlanningActual=PlanningActual,
                StatFcstPLWLag=StatFcstPLWLag,
                StatFcstPLMLag=StatFcstPLMLag,
                StatFcstPLPMLag=StatFcstPLPMLag,
                StatFcstPLQLag=StatFcstPLQLag,
                StatFcstPLPQLag=StatFcstPLPQLag,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                ForecastIterationMasterData=ForecastIterationMasterData,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                PlanningGrains=PlanningGrains,
                StatBucketWeight=StatBucketWeight,
                AccuracyWindow=AccuracyWindow,
                df_keys=df_keys,
            )

            StatFcstAbsError_list.append(all_PL_error)
            StatFcstPLLagAbsError_list.append(all_lag_error)
            StatFcstPLWLagAbsError_list.append(all_w_lag_error)
            StatFcstPLMLagAbsError_list.append(all_m_lag_error)
            StatFcstPLPMLagAbsError_list.append(all_pm_lag_error)
            StatFcstPLQLagAbsError_list.append(all_q_lag_error)
            StatFcstPLPQLagAbsError_list.append(all_pq_lag_error)

        StatFcstPLAbsError = concat_to_dataframe(StatFcstAbsError_list)
        StatFcstPLLagAbsError = concat_to_dataframe(StatFcstPLLagAbsError_list)
        StatFcstPLWLagAbsError = concat_to_dataframe(StatFcstPLWLagAbsError_list)
        StatFcstPLMLagAbsError = concat_to_dataframe(StatFcstPLMLagAbsError_list)
        StatFcstPLPMLagAbsError = concat_to_dataframe(StatFcstPLPMLagAbsError_list)
        StatFcstPLQLagAbsError = concat_to_dataframe(StatFcstPLQLagAbsError_list)
        StatFcstPLPQLagAbsError = concat_to_dataframe(StatFcstPLPQLagAbsError_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstPLAbsError,
            StatFcstPLLagAbsError,
            StatFcstPLWLagAbsError,
            StatFcstPLMLagAbsError,
            StatFcstPLPMLagAbsError,
            StatFcstPLQLagAbsError,
            StatFcstPLPQLagAbsError,
        ) = (None, None, None, None, None, None, None)
    return (
        StatFcstPLAbsError,
        StatFcstPLLagAbsError,
        StatFcstPLWLagAbsError,
        StatFcstPLMLagAbsError,
        StatFcstPLPMLagAbsError,
        StatFcstPLQLagAbsError,
        StatFcstPLPQLagAbsError,
    )


def processIteration(
    PlanningActual,
    StatFcstPLWLag,
    StatFcstPLMLag,
    StatFcstPLPMLag,
    StatFcstPLQLag,
    StatFcstPLPQLag,
    ForecastGenTimeBucket,
    ForecastIterationMasterData,
    CurrentTimePeriod,
    TimeDimension,
    StatBucketWeight,
    PlanningGrains,
    AccuracyWindow="1",
    df_keys={},
):
    plugin_name = "DP214CalculateRollOverAccuracyPL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    stat_fcst_pl_lag = "Stat Fcst PL Lag"
    stat_fcst_pl_lag_abs_error = "Stat Fcst PL Lag Abs Error"
    actual_lag_backtest_col = "Actual Lag Backtest"

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)

    StatFcstPLLag1_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + planning_grains
        + [o9Constants.STAT_FCST_PL_ABS_ERROR]
    )
    StatFcstPLAbsError = pd.DataFrame(columns=StatFcstPLLag1_cols)

    StatFcstPLLagAbsError_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [stat_fcst_pl_lag, stat_fcst_pl_lag_abs_error, actual_lag_backtest_col]
    )
    StatFcstPLLagAbsError = pd.DataFrame(columns=StatFcstPLLagAbsError_cols)

    StatFcstPLWLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_W_LAG_ABS_ERROR,
        ]
    )
    StatFcstPLWLagAbsError = pd.DataFrame(columns=StatFcstPLWLag_cols)

    StatFcstPLMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_M_LAG_ABS_ERROR,
        ]
    )
    StatFcstPLMLagAbsError = pd.DataFrame(columns=StatFcstPLMLag_cols)

    StatFcstPLPMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PM_LAG_ABS_ERROR,
        ]
    )
    StatFcstPLPMLagAbsError = pd.DataFrame(columns=StatFcstPLPMLag_cols)

    StatFcstPLQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_Q_LAG_ABS_ERROR,
        ]
    )
    StatFcstPLQLagAbsError = pd.DataFrame(columns=StatFcstPLQLag_cols)

    StatFcstPLPQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PQ_LAG_ABS_ERROR,
        ]
    )
    StatFcstPLPQLagAbsError = pd.DataFrame(columns=StatFcstPLPQLag_cols)

    try:
        TimeLevel = ForecastGenTimeBucket[o9Constants.FORECAST_GEN_TIME_BUCKET].values[0]
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.replace(" ", "").lower() == "planningmonth"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.replace(" ", "").lower() == "planningquarter"
        AccuracyWindow = int(AccuracyWindow)

        # SellOut inputs
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return (
                StatFcstPLAbsError,
                StatFcstPLLagAbsError,
                StatFcstPLWLagAbsError,
                StatFcstPLMLagAbsError,
                StatFcstPLPMLagAbsError,
                StatFcstPLQLagAbsError,
                StatFcstPLPQLagAbsError,
            )

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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )
            PlanningActual_df = df
        else:
            PlanningActual_df = PlanningActual

        # merge with TimeDimension to get all relevant columns
        StatBucketWeight = StatBucketWeight.merge(
            TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
        )

        if len(StatFcstPLWLag) != 0 and week:
            current_week = CurrentTimePeriod[o9Constants.WEEK_KEY].values[0]
            StatFcstPLWLag = StatFcstPLWLag.merge(
                TimeDimension[[o9Constants.WEEK, o9Constants.WEEK_KEY]].drop_duplicates(),
                on=o9Constants.WEEK,
                how="inner",
            )
            StatFcstPLWLag = StatFcstPLWLag[
                StatFcstPLWLag[o9Constants.WEEK_KEY].dt.tz_localize(None) <= current_week
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst PL W Lag is available : {max(StatFcstPLWLag[o9Constants.LAG].astype(int))}"
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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )

            StatFcstPLWLagAbsError = get_abs_error(
                source_df=StatFcstPLWLag,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.WEEK,
                time_key=o9Constants.WEEK_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_W_LAG,
                actual_measure=input_stream,
                output_measure=o9Constants.STAT_FCST_W_LAG_ABS_ERROR,
                output_cols=StatFcstPLWLag_cols + [o9Constants.STAT_FCST_W_LAG],
            )
            if len(StatFcstPLWLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_W_LAG: stat_fcst_pl_lag,
                    o9Constants.STAT_FCST_W_LAG_ABS_ERROR: stat_fcst_pl_lag_abs_error,
                }
                StatFcstPLLagAbsError = disaggregate_data(
                    source_df=StatFcstPLWLagAbsError,
                    source_grain=o9Constants.WEEK,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_W_LAG,
                        o9Constants.STAT_FCST_W_LAG_ABS_ERROR,
                    ],
                )
                StatFcstPLLagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstPLAbsError = StatFcstPLLagAbsError[
                    (StatFcstPLLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstPLMLag) != 0 and month:
            current_month = CurrentTimePeriod[o9Constants.MONTH_KEY].values[0]
            StatFcstPLMLag = StatFcstPLMLag.merge(
                TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
                on=o9Constants.MONTH,
                how="inner",
            )
            StatFcstPLMLag = StatFcstPLMLag[
                StatFcstPLMLag[o9Constants.MONTH_KEY].dt.tz_localize(None) <= current_month
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst PL M Lag is available : {max(StatFcstPLMLag[o9Constants.LAG].astype(int))}"
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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )
            StatFcstPLMLagAbsError = get_abs_error(
                source_df=StatFcstPLMLag,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.MONTH,
                time_key=o9Constants.MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_M_LAG,
                actual_measure=input_stream,
                output_measure=o9Constants.STAT_FCST_M_LAG_ABS_ERROR,
                output_cols=StatFcstPLMLag_cols + [o9Constants.STAT_FCST_M_LAG],
            )
            if len(StatFcstPLMLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_M_LAG: stat_fcst_pl_lag,
                    o9Constants.STAT_FCST_M_LAG_ABS_ERROR: stat_fcst_pl_lag_abs_error,
                }
                StatFcstPLLagAbsError = disaggregate_data(
                    source_df=StatFcstPLMLagAbsError,
                    source_grain=o9Constants.MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_M_LAG,
                        o9Constants.STAT_FCST_M_LAG_ABS_ERROR,
                    ],
                )
                StatFcstPLLagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstPLAbsError = StatFcstPLLagAbsError[
                    (StatFcstPLLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstPLPMLag) != 0 and pl_month:
            current_pl_month = CurrentTimePeriod[o9Constants.PLANNING_MONTH_KEY].values[0]
            StatFcstPLPMLag = StatFcstPLPMLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_MONTH, o9Constants.PLANNING_MONTH_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_MONTH,
                how="inner",
            )
            StatFcstPLPMLag = StatFcstPLPMLag[
                StatFcstPLPMLag[o9Constants.PLANNING_MONTH_KEY].dt.tz_localize(None)
                <= current_pl_month
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst PL PM Lag is available : {max(StatFcstPLPMLag[o9Constants.LAG].astype(int))}"
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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )
            StatFcstPLPMLagAbsError = get_abs_error(
                source_df=StatFcstPLPMLag,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_MONTH,
                time_key=o9Constants.PLANNING_MONTH_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PM_LAG,
                actual_measure=input_stream,
                output_measure=o9Constants.STAT_FCST_PM_LAG_ABS_ERROR,
                output_cols=StatFcstPLPMLag_cols + [o9Constants.STAT_FCST_PM_LAG],
            )
            if len(StatFcstPLPMLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_PM_LAG: stat_fcst_pl_lag,
                    o9Constants.STAT_FCST_PM_LAG_ABS_ERROR: stat_fcst_pl_lag_abs_error,
                }
                StatFcstPLLagAbsError = disaggregate_data(
                    source_df=StatFcstPLPMLagAbsError,
                    source_grain=o9Constants.PLANNING_MONTH,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PM_LAG,
                        o9Constants.STAT_FCST_PM_LAG_ABS_ERROR,
                    ],
                )
                StatFcstPLLagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstPLAbsError = StatFcstPLLagAbsError[
                    (StatFcstPLLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstPLQLag) != 0 and quarter:
            current_quarter = CurrentTimePeriod[o9Constants.QUARTER_KEY].values[0]
            StatFcstPLQLag = StatFcstPLQLag.merge(
                TimeDimension[[o9Constants.QUARTER, o9Constants.QUARTER_KEY]].drop_duplicates(),
                on=o9Constants.QUARTER,
                how="inner",
            )
            StatFcstPLQLag = StatFcstPLQLag[
                StatFcstPLQLag[o9Constants.QUARTER_KEY].dt.tz_localize(None) <= current_quarter
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst PL Q Lag is available : {max(StatFcstPLQLag[o9Constants.LAG].astype(int))}"
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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )

            StatFcstPLQLagAbsError = get_abs_error(
                source_df=StatFcstPLQLag,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.QUARTER,
                time_key=o9Constants.QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_Q_LAG,
                actual_measure=input_stream,
                output_measure=o9Constants.STAT_FCST_Q_LAG_ABS_ERROR,
                output_cols=StatFcstPLQLag_cols + [o9Constants.STAT_FCST_Q_LAG],
            )
            if len(StatFcstPLQLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_Q_LAG: stat_fcst_pl_lag,
                    o9Constants.STAT_FCST_Q_LAG_ABS_ERROR: stat_fcst_pl_lag_abs_error,
                }
                StatFcstPLLagAbsError = disaggregate_data(
                    source_df=StatFcstPLQLagAbsError,
                    source_grain=o9Constants.QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_Q_LAG,
                        o9Constants.STAT_FCST_Q_LAG_ABS_ERROR,
                    ],
                )
                StatFcstPLLagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstPLAbsError = StatFcstPLLagAbsError[
                    (StatFcstPLLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        if len(StatFcstPLPQLag) != 0 and pl_quarter:
            current_pl_quarter = CurrentTimePeriod[o9Constants.PLANNING_QUARTER_KEY].values[0]
            StatFcstPLPQLag = StatFcstPLPQLag.merge(
                TimeDimension[
                    [o9Constants.PLANNING_QUARTER, o9Constants.PLANNING_QUARTER_KEY]
                ].drop_duplicates(),
                on=o9Constants.PLANNING_QUARTER,
                how="inner",
            )
            StatFcstPLPQLag = StatFcstPLPQLag[
                StatFcstPLPQLag[o9Constants.PLANNING_QUARTER_KEY].dt.tz_localize(None)
                <= current_pl_quarter
            ]
            logger.info(
                f"Maximum Lag in history for which Stat Fcst PL PQ Lag is available : {max(StatFcstPLPQLag[o9Constants.LAG].astype(int))}"
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
                    StatFcstPLAbsError,
                    StatFcstPLLagAbsError,
                    StatFcstPLWLagAbsError,
                    StatFcstPLMLagAbsError,
                    StatFcstPLPMLagAbsError,
                    StatFcstPLQLagAbsError,
                    StatFcstPLPQLagAbsError,
                )

            StatFcstPLPQLagAbsError = get_abs_error(
                source_df=StatFcstPLPQLag,
                Actuals=PlanningActual_df,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                merge_grains=planning_grains + [o9Constants.VERSION_NAME],
                time_grain=o9Constants.PLANNING_QUARTER,
                time_key=o9Constants.PLANNING_QUARTER_KEY,
                lags=lags,
                source_measure=o9Constants.STAT_FCST_PQ_LAG,
                actual_measure=input_stream,
                output_measure=o9Constants.STAT_FCST_PQ_LAG_ABS_ERROR,
                output_cols=StatFcstPLPQLag_cols + [o9Constants.STAT_FCST_PQ_LAG],
            )
            if len(StatFcstPLPQLagAbsError) != 0:
                col_mapping = {
                    o9Constants.STAT_FCST_PQ_LAG: stat_fcst_pl_lag,
                    o9Constants.STAT_FCST_PQ_LAG_ABS_ERROR: stat_fcst_pl_lag_abs_error,
                }
                StatFcstPLLagAbsError = disaggregate_data(
                    source_df=StatFcstPLPQLagAbsError,
                    source_grain=o9Constants.PLANNING_QUARTER,
                    target_grain=o9Constants.PARTIAL_WEEK,
                    profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
                    profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                    cols_to_disaggregate=[
                        o9Constants.STAT_FCST_PQ_LAG,
                        o9Constants.STAT_FCST_PQ_LAG_ABS_ERROR,
                    ],
                )
                StatFcstPLLagAbsError.rename(columns=col_mapping, inplace=True)
                StatFcstPLAbsError = StatFcstPLLagAbsError[
                    (StatFcstPLLagAbsError[o9Constants.LAG].astype(int) == 1)
                ]

        StatFcstPLAbsError = StatFcstPLAbsError.rename(
            columns={
                stat_fcst_pl_lag_abs_error: o9Constants.STAT_FCST_PL_ABS_ERROR,
            }
        )
        if len(StatFcstPLLagAbsError) > 0:
            StatFcstPLLagAbsError = StatFcstPLLagAbsError.merge(
                PlanningActual_df,
                on=planning_grains + [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK],
                how="inner",
            )
            StatFcstPLLagAbsError = StatFcstPLLagAbsError.rename(
                columns={
                    input_stream: actual_lag_backtest_col,
                }
            )
        StatFcstPLLagAbsError = StatFcstPLLagAbsError[StatFcstPLLagAbsError_cols]
        StatFcstPLAbsError = StatFcstPLAbsError[StatFcstPLLag1_cols]
        StatFcstPLWLagAbsError = StatFcstPLWLagAbsError[StatFcstPLWLag_cols]
        StatFcstPLMLagAbsError = StatFcstPLMLagAbsError[StatFcstPLMLag_cols]
        StatFcstPLPMLagAbsError = StatFcstPLPMLagAbsError[StatFcstPLPMLag_cols]
        StatFcstPLQLagAbsError = StatFcstPLQLagAbsError[StatFcstPLQLag_cols]
        StatFcstPLPQLagAbsError = StatFcstPLPQLagAbsError[StatFcstPLPQLag_cols]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        StatFcstPLLagAbsError = pd.DataFrame(columns=StatFcstPLLagAbsError_cols)
        StatFcstPLAbsError = pd.DataFrame(columns=StatFcstPLLag1_cols)
        StatFcstPLWLagAbsError = pd.DataFrame(columns=StatFcstPLWLag_cols)
        StatFcstPLMLagAbsError = pd.DataFrame(columns=StatFcstPLMLag_cols)
        StatFcstPLPMLagAbsError = pd.DataFrame(columns=StatFcstPLPMLag_cols)
        StatFcstPLQLagAbsError = pd.DataFrame(columns=StatFcstPLQLag_cols)
        StatFcstPLPQLagAbsError = pd.DataFrame(columns=StatFcstPLPQLag_cols)

    return (
        StatFcstPLAbsError,
        StatFcstPLLagAbsError,
        StatFcstPLWLagAbsError,
        StatFcstPLMLagAbsError,
        StatFcstPLPMLagAbsError,
        StatFcstPLQLagAbsError,
        StatFcstPLPQLagAbsError,
    )

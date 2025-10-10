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
from helpers.utils import filter_for_iteration, get_first_day_in_time_bucket

logger = logging.getLogger("o9_logger")


pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def create_lag_assoc(
    TimeDimension,
    relevant_time_name,
    relevant_time_key,
    assoc_col,
    forecast_periods,
    lag_col,
    partial_week_col,
    partial_week_key_col,
):
    # M to PW Lag Association
    lag_mapping = TimeDimension[[relevant_time_name, relevant_time_key]].drop_duplicates()
    lag_mapping.sort_values(relevant_time_key, inplace=True)
    lag_mapping.reset_index(drop=True, inplace=True)
    lag_mapping[lag_col] = lag_mapping.index

    # restrict number of lags based on forecast periods
    lag_mapping = lag_mapping.head(forecast_periods)

    # join to get PW
    req_time_cols = [
        relevant_time_name,
        partial_week_col,
        partial_week_key_col,
    ]
    assoc_df = lag_mapping.merge(
        TimeDimension[req_time_cols].drop_duplicates(),
        on=relevant_time_name,
        how="inner",
    )
    assoc_df[assoc_col] = 1.0

    # select relevant columns
    output_cols = [lag_col, partial_week_col, assoc_col]
    return assoc_df[output_cols]


col_mapping = {
    "M to PW Lag Association": float,
    "W to PW Lag Association": float,
    "PM to PW Lag Association": float,
    "Q to PW Lag Association": float,
    "PQ to PW Lag Association": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ForecastTimeBucket,
    CurrentTimePeriod,
    TimeDimension,
    ForecastParameters,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in ForecastTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                ForecastTimeBucket=ForecastTimeBucket,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    ForecastTimeBucket,
    CurrentTimePeriod,
    TimeDimension,
    ForecastParameters,
    df_keys,
):
    plugin_name = "DP048PartialWeekLagAssociation"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    lag_col = "Lag.[Lag]"
    fcst_storage_time_bucket_col = "Forecast Storage Time Bucket"
    forecast_period_col = "Forecast Period"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"
    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"
    partial_week_col = "Time.[Partial Week]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    partial_week_key_col = "Time.[PartialWeekKey]"
    planning_cycle_col = "Planning Cycle.[Planning Cycle Date]"

    # Output measures
    m_to_pw_lag_assoc_col = "M to PW Lag Association"
    w_to_pw_lag_assoc_col = "W to PW Lag Association"
    pm_to_pw_lag_assoc_col = "PM to PW Lag Association"
    q_to_pw_lag_assoc_col = "Q to PW Lag Association"
    pq_to_pw_lag_assoc_col = "PQ to PW Lag Association"

    cols_required_in_output = [
        version_col,
        lag_col,
        partial_week_col,
        planning_cycle_col,
        m_to_pw_lag_assoc_col,
        w_to_pw_lag_assoc_col,
        pm_to_pw_lag_assoc_col,
        q_to_pw_lag_assoc_col,
        pq_to_pw_lag_assoc_col,
    ]
    PWLagAssociation = pd.DataFrame(columns=cols_required_in_output)
    try:
        if ForecastTimeBucket.empty:
            logger.warning("ForecastTimeBucket is empty")
            return PWLagAssociation

        # Extract the parameters
        fcst_gen_time_bucket = ForecastTimeBucket[fcst_gen_time_bucket_col].unique()[0]
        fcst_storage_time_bucket = ForecastTimeBucket[fcst_storage_time_bucket_col].unique()[0]

        if ForecastParameters.empty:
            logger.warning("ForecastTimeBucket is empty")
            return PWLagAssociation

        forecast_periods = int(ForecastParameters[forecast_period_col].unique()[0])
        input_version = ForecastParameters[version_col].unique()[0]

        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")
        logger.debug(f"fcst_storage_time_bucket : {fcst_storage_time_bucket}")
        logger.debug(f"forecast_periods : {forecast_periods}")

        # Filter time dimension data according to last time period
        current_partial_week_key = CurrentTimePeriod[partial_week_key_col].unique()[0]

        current_week = CurrentTimePeriod[o9Constants.WEEK].unique()[0]

        current_week_key = CurrentTimePeriod[o9Constants.WEEK_KEY].unique()[0]

        current_month = CurrentTimePeriod[o9Constants.MONTH].unique()[0]

        current_month_key = CurrentTimePeriod[o9Constants.MONTH_KEY].unique()[0]

        current_planning_month = CurrentTimePeriod[o9Constants.PLANNING_MONTH].unique()[0]

        current_planning_month_key = CurrentTimePeriod[o9Constants.PLANNING_MONTH_KEY].unique()[0]

        current_quarter = CurrentTimePeriod[o9Constants.QUARTER].unique()[0]

        current_quarter_key = CurrentTimePeriod[o9Constants.QUARTER_KEY].unique()[0]

        current_planning_quarter = CurrentTimePeriod[o9Constants.PLANNING_QUARTER].unique()[0]

        current_planning_quarter_key = CurrentTimePeriod[o9Constants.PLANNING_QUARTER_KEY].unique()[
            0
        ]

        logger.debug(f"current_partial_week_key : {current_partial_week_key}")

        grain_key_mapping = {
            "Partial Week": partial_week_key_col,
            "Week": week_key_col,
            "Month": month_key_col,
            "Planning Month": planning_month_key_col,
            "Quarter": quarter_key_col,
            "Planning Quarter": planning_quarter_key_col,
        }

        relevant_grain = ForecastTimeBucket[fcst_gen_time_bucket_col].values[0]
        relevant_key = grain_key_mapping[relevant_grain]
        if relevant_grain == "Week":
            filter_clause = TimeDimension[relevant_key] >= current_week_key
            TimeDimension = TimeDimension[filter_clause]

            logger.debug("Creating w_to_pw_lag_assoc ...")
            PWLagAssociation = create_lag_assoc(
                TimeDimension=TimeDimension,
                relevant_time_name=week_col,
                relevant_time_key=week_key_col,
                assoc_col=w_to_pw_lag_assoc_col,
                forecast_periods=forecast_periods,
                lag_col=lag_col,
                partial_week_col=partial_week_col,
                partial_week_key_col=partial_week_key_col,
            )
            PWLagAssociation[planning_cycle_col] = get_first_day_in_time_bucket(
                time_bucket_value=current_week,
                relevant_time_name=week_col,
                time_dimension=TimeDimension,
            )
            logger.debug(f"w_to_pw_lag_assoc, shape : {PWLagAssociation.shape}")
            PWLagAssociation[
                [
                    m_to_pw_lag_assoc_col,
                    pm_to_pw_lag_assoc_col,
                    q_to_pw_lag_assoc_col,
                    pq_to_pw_lag_assoc_col,
                ]
            ] = [None, None, None, None]
        elif relevant_grain == "Month":
            filter_clause = (TimeDimension[relevant_key] >= current_month_key) & (
                TimeDimension[partial_week_key_col] >= current_month_key
            )
            TimeDimension = TimeDimension[filter_clause]

            logger.debug("Creating m_to_pw_lag_assoc ...")
            PWLagAssociation = create_lag_assoc(
                TimeDimension=TimeDimension,
                relevant_time_name=month_col,
                relevant_time_key=month_key_col,
                assoc_col=m_to_pw_lag_assoc_col,
                forecast_periods=forecast_periods,
                lag_col=lag_col,
                partial_week_col=partial_week_col,
                partial_week_key_col=partial_week_key_col,
            )
            PWLagAssociation[planning_cycle_col] = get_first_day_in_time_bucket(
                time_bucket_value=current_month,
                relevant_time_name=month_col,
                time_dimension=TimeDimension,
            )
            logger.debug(f"m_to_pw_lag_assoc, shape : {PWLagAssociation.shape}")
            PWLagAssociation[
                [
                    w_to_pw_lag_assoc_col,
                    pm_to_pw_lag_assoc_col,
                    q_to_pw_lag_assoc_col,
                    pq_to_pw_lag_assoc_col,
                ]
            ] = [None, None, None, None]
        elif relevant_grain == "Planning Month":
            filter_clause = (TimeDimension[relevant_key] >= current_planning_month_key) & (
                TimeDimension[partial_week_key_col] >= current_planning_month_key
            )
            TimeDimension = TimeDimension[filter_clause]
            logger.debug("Creating pm_to_pw_lag_assoc ...")
            PWLagAssociation = create_lag_assoc(
                TimeDimension=TimeDimension,
                relevant_time_name=planning_month_col,
                relevant_time_key=planning_month_key_col,
                assoc_col=pm_to_pw_lag_assoc_col,
                forecast_periods=forecast_periods,
                lag_col=lag_col,
                partial_week_col=partial_week_col,
                partial_week_key_col=partial_week_key_col,
            )
            PWLagAssociation[planning_cycle_col] = get_first_day_in_time_bucket(
                time_bucket_value=current_planning_month,
                relevant_time_name=planning_month_col,
                time_dimension=TimeDimension,
            )
            logger.debug(f"pm_to_pw_lag_assoc, shape : {PWLagAssociation.shape}")
            PWLagAssociation[
                [
                    m_to_pw_lag_assoc_col,
                    w_to_pw_lag_assoc_col,
                    q_to_pw_lag_assoc_col,
                    pq_to_pw_lag_assoc_col,
                ]
            ] = [None, None, None, None]
        elif relevant_grain == "Quarter":
            filter_clause = (TimeDimension[relevant_key] >= current_quarter_key) & (
                TimeDimension[partial_week_key_col] >= current_quarter_key
            )
            TimeDimension = TimeDimension[filter_clause]
            logger.debug("Creating q_to_pw_lag_assoc ...")
            PWLagAssociation = create_lag_assoc(
                TimeDimension=TimeDimension,
                relevant_time_name=quarter_col,
                relevant_time_key=quarter_key_col,
                assoc_col=q_to_pw_lag_assoc_col,
                forecast_periods=forecast_periods,
                lag_col=lag_col,
                partial_week_col=partial_week_col,
                partial_week_key_col=partial_week_key_col,
            )
            PWLagAssociation[planning_cycle_col] = get_first_day_in_time_bucket(
                time_bucket_value=current_quarter,
                relevant_time_name=quarter_col,
                time_dimension=TimeDimension,
            )
            logger.debug(f"q_to_pw_lag_assoc, shape : {PWLagAssociation.shape}")
            PWLagAssociation[
                [
                    m_to_pw_lag_assoc_col,
                    pm_to_pw_lag_assoc_col,
                    w_to_pw_lag_assoc_col,
                    pq_to_pw_lag_assoc_col,
                ]
            ] = [None, None, None, None]
        elif relevant_grain == "Planning Quarter":
            filter_clause = (TimeDimension[relevant_key] >= current_planning_quarter_key) & (
                TimeDimension[partial_week_key_col] >= current_planning_quarter_key
            )
            TimeDimension = TimeDimension[filter_clause]
            logger.debug("Creating pq_to_pw_lag_assoc ...")
            PWLagAssociation = create_lag_assoc(
                TimeDimension=TimeDimension,
                relevant_time_name=planning_quarter_col,
                relevant_time_key=planning_quarter_key_col,
                assoc_col=pq_to_pw_lag_assoc_col,
                forecast_periods=forecast_periods,
                lag_col=lag_col,
                partial_week_col=partial_week_col,
                partial_week_key_col=partial_week_key_col,
            )
            PWLagAssociation[planning_cycle_col] = get_first_day_in_time_bucket(
                time_bucket_value=current_planning_quarter,
                relevant_time_name=planning_quarter_col,
                time_dimension=TimeDimension,
            )
            logger.debug(f"pq_to_pw_lag_assoc, shape : {PWLagAssociation.shape}")
            PWLagAssociation[
                [
                    m_to_pw_lag_assoc_col,
                    pm_to_pw_lag_assoc_col,
                    q_to_pw_lag_assoc_col,
                    w_to_pw_lag_assoc_col,
                ]
            ] = [None, None, None, None]
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return PWLagAssociation

        PWLagAssociation.insert(0, version_col, input_version)
        logger.debug(f"PWLagAssociation, shape : {PWLagAssociation.shape}")

        PWLagAssociation = PWLagAssociation[cols_required_in_output]

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        PWLagAssociation = pd.DataFrame(columns=cols_required_in_output)
    return PWLagAssociation

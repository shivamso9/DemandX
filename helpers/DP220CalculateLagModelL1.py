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

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration, get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Stat Fcst L1 W Lag": float,
    "Stat Fcst L1 M Lag": float,
    "Stat Fcst L1 PM Lag": float,
    "Stat Fcst L1 Q Lag": float,
    "Stat Fcst L1 PQ Lag": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatFcstL1,
    ForecastGenTimeBucket,
    SelectedPlanningCycle,
    PlanningCycles,
    TimeDimension,
    StatGrains,
    LagWindow="All",
    df_keys={},
):
    try:
        StatFcstL1WLag_list = list()
        StatFcstL1MLag_list = list()
        StatFcstL1PMLag_list = list()
        StatFcstL1QLag_list = list()
        StatFcstL1PQLag_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_w_lag,
                all_m_lag,
                all_pm_lag,
                all_q_lag,
                all_pq_lag,
            ) = decorated_func(
                StatFcstL1=StatFcstL1,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                SelectedPlanningCycle=SelectedPlanningCycle,
                PlanningCycles=PlanningCycles,
                TimeDimension=TimeDimension,
                StatGrains=StatGrains,
                LagWindow=LagWindow,
                the_iteration=the_iteration,
                df_keys=df_keys,
            )

            StatFcstL1WLag_list.append(all_w_lag)
            StatFcstL1MLag_list.append(all_m_lag)
            StatFcstL1PMLag_list.append(all_pm_lag)
            StatFcstL1QLag_list.append(all_q_lag)
            StatFcstL1PQLag_list.append(all_pq_lag)

        StatFcstL1WLag = concat_to_dataframe(StatFcstL1WLag_list)
        StatFcstL1MLag = concat_to_dataframe(StatFcstL1MLag_list)
        StatFcstL1PMLag = concat_to_dataframe(StatFcstL1PMLag_list)
        StatFcstL1QLag = concat_to_dataframe(StatFcstL1QLag_list)
        StatFcstL1PQLag = concat_to_dataframe(StatFcstL1PQLag_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstL1WLag,
            StatFcstL1MLag,
            StatFcstL1PMLag,
            StatFcstL1QLag,
            StatFcstL1PQLag,
        ) = (None, None, None, None, None)
    return (
        StatFcstL1WLag,
        StatFcstL1MLag,
        StatFcstL1PMLag,
        StatFcstL1QLag,
        StatFcstL1PQLag,
    )


def processIteration(
    StatFcstL1,
    ForecastGenTimeBucket,
    SelectedPlanningCycle,
    PlanningCycles,
    TimeDimension,
    StatGrains,
    the_iteration,
    LagWindow="All",
    df_keys={},
):
    plugin_name = "DP220CalculateLagModelL1"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # configurables
    planning_cycle_date_key = "Planning Cycle.[PlanningCycleDateKey]"

    stat_grains = get_list_of_grains_from_string(input=StatGrains)

    StatFcstL1WLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_W_LAG,
        ]
    )
    StatFcstL1WLag = pd.DataFrame(columns=StatFcstL1WLag_cols)

    StatFcstL1MLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_M_LAG,
        ]
    )
    StatFcstL1MLag = pd.DataFrame(columns=StatFcstL1MLag_cols)

    StatFcstL1PMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_PM_LAG,
        ]
    )
    StatFcstL1PMLag = pd.DataFrame(columns=StatFcstL1PMLag_cols)
    StatFcstL1QLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_Q_LAG,
        ]
    )
    StatFcstL1QLag = pd.DataFrame(columns=StatFcstL1QLag_cols)
    StatFcstL1PQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + stat_grains
        + [
            o9Constants.STAT_FCST_L1_PQ_LAG,
        ]
    )
    StatFcstL1PQLag = pd.DataFrame(columns=StatFcstL1PQLag_cols)

    try:
        fcst_gen_time_bucket = ForecastGenTimeBucket[o9Constants.FORECAST_GEN_TIME_BUCKET].unique()[
            0
        ]
        lags = LagWindow.lower()
        if fcst_gen_time_bucket == "Week":
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.WEEK,
                o9Constants.WEEK_KEY,
            ]
            relevant_time_name = o9Constants.WEEK
            relevant_time_key = o9Constants.WEEK_KEY
            if lags == "all":
                LagWindow = 52
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif fcst_gen_time_bucket == "Planning Month":
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PLANNING_MONTH,
                o9Constants.PLANNING_MONTH_KEY,
            ]
            relevant_time_name = o9Constants.PLANNING_MONTH
            relevant_time_key = o9Constants.PLANNING_MONTH_KEY
            if lags == "all":
                LagWindow = 12
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif fcst_gen_time_bucket == "Month":
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.MONTH,
                o9Constants.MONTH_KEY,
            ]
            relevant_time_name = o9Constants.MONTH
            relevant_time_key = o9Constants.MONTH_KEY
            if lags == "all":
                LagWindow = 12
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif fcst_gen_time_bucket == "Planning Quarter":
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PLANNING_QUARTER,
                o9Constants.PLANNING_QUARTER_KEY,
            ]
            relevant_time_name = o9Constants.PLANNING_QUARTER
            relevant_time_key = o9Constants.PLANNING_QUARTER_KEY
            if lags == "all":
                LagWindow = 4
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif fcst_gen_time_bucket == "Quarter":
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.QUARTER,
                o9Constants.QUARTER_KEY,
            ]
            relevant_time_name = o9Constants.QUARTER
            relevant_time_key = o9Constants.QUARTER_KEY
            if lags == "all":
                LagWindow = 4
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty outputs"
            )
            return (
                StatFcstL1WLag,
                StatFcstL1MLag,
                StatFcstL1PMLag,
                StatFcstL1QLag,
                StatFcstL1PQLag,
            )
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}
        SelectedPlanningCycle = SelectedPlanningCycle[relevant_time_cols]
        selected_cycle = SelectedPlanningCycle[relevant_time_cols].merge(
            PlanningCycles,
            left_on=relevant_time_key,
            right_on=planning_cycle_date_key,
            how="inner",
        )
        selected_cycle_name = SelectedPlanningCycle[relevant_time_name].values[0]
        if len(selected_cycle) == 0:
            logger.warning(
                f"Planning Cycle {selected_cycle_name} not present in list of Planning Cycles, returning empty outputs for iteration {the_iteration}..."
            )
            return (
                StatFcstL1WLag,
                StatFcstL1MLag,
                StatFcstL1PMLag,
                StatFcstL1QLag,
                StatFcstL1PQLag,
            )
        # need to seperate planning cycle date and current cycle to match the time mapping formats(planning cycle date - 01-Jul-23 but Planning month - M07-23)
        current_planning_cycle = selected_cycle[o9Constants.PLANNING_CYCLE_DATE].values[0]
        if len(StatFcstL1) == 0:
            logger.warning(
                f"Stat Fcst L1 empty for the forecast iteration {the_iteration}, returning empty outputs..."
            )
            return (
                StatFcstL1WLag,
                StatFcstL1MLag,
                StatFcstL1PMLag,
                StatFcstL1QLag,
                StatFcstL1PQLag,
            )
        next_n_periods = get_n_time_periods(
            selected_cycle_name,
            LagWindow,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )
        lag_data = pd.DataFrame({relevant_time_name: next_n_periods}).assign(
            **{
                o9Constants.LAG: lambda df: df.index,
            }
        )
        if lags != "all":
            lag_data = lag_data.merge(lag_df, on=o9Constants.LAG, how="inner")
            logger.debug(f"Lags Data : \n{lag_data}")
        lag_data = lag_data.merge(
            TimeDimension[relevant_time_cols].drop_duplicates(),
            on=relevant_time_name,
            how="inner",
        )
        lag_data[o9Constants.PLANNING_CYCLE_DATE] = current_planning_cycle
        StatFcstwithLag_PW = StatFcstL1.merge(lag_data, on=o9Constants.PARTIAL_WEEK, how="inner")

        StatFcstLagOutput = (
            StatFcstwithLag_PW.groupby(
                stat_grains
                + [
                    o9Constants.VERSION_NAME,
                    o9Constants.LAG,
                    o9Constants.PLANNING_CYCLE_DATE,
                    relevant_time_name,
                ]
            )
            .agg({o9Constants.STAT_FCST_L1: "sum"})
            .reset_index()
        )
        if relevant_time_name == o9Constants.WEEK:
            StatFcstL1WLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_L1_W_LAG}
            )
            StatFcstL1WLag = StatFcstL1WLag[StatFcstL1WLag_cols]
        elif relevant_time_name == o9Constants.MONTH:
            StatFcstL1MLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_L1_M_LAG}
            )
            StatFcstL1MLag = StatFcstL1MLag[StatFcstL1MLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_MONTH:
            StatFcstL1PMLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_L1_PM_LAG}
            )
            StatFcstL1PMLag = StatFcstL1PMLag[StatFcstL1PMLag_cols]
        elif relevant_time_name == o9Constants.QUARTER:
            StatFcstL1QLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_L1_Q_LAG}
            )
            StatFcstL1QLag = StatFcstL1QLag[StatFcstL1QLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_QUARTER:
            StatFcstL1PQLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_L1_PQ_LAG}
            )
            StatFcstL1PQLag = StatFcstL1PQLag[StatFcstL1PQLag_cols]
        else:
            logger.warning(
                f"Invalid time bucket {relevant_time_name}, returning empty outputs for iteration {the_iteration}..."
            )

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return (
        StatFcstL1WLag,
        StatFcstL1MLag,
        StatFcstL1PMLag,
        StatFcstL1QLag,
        StatFcstL1PQLag,
    )

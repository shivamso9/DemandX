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
    "Stat Fcst W Lag": float,
    "Stat Fcst M Lag": float,
    "Stat Fcst PM Lag": float,
    "Stat Fcst Q Lag": float,
    "Stat Fcst PQ Lag": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatFcstPL,
    ForecastGenTimeBucket,
    SelectedPlanningCycle,
    PlanningCycles,
    TimeDimension,
    PlanningGrains,
    LagWindow="All",
    NBucketsinMonths="12",
    df_keys={},
):
    try:
        StatFcstPLLC_list = list()
        StatFcstPLWLag_list = list()
        StatFcstPLMLag_list = list()
        StatFcstPLPMLag_list = list()
        StatFcstPLQLag_list = list()
        StatFcstPLPQLag_list = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                all_lc_output,
                all_w_lag,
                all_m_lag,
                all_pm_lag,
                all_q_lag,
                all_pq_lag,
            ) = decorated_func(
                StatFcstPL=StatFcstPL,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                SelectedPlanningCycle=SelectedPlanningCycle,
                PlanningCycles=PlanningCycles,
                TimeDimension=TimeDimension,
                PlanningGrains=PlanningGrains,
                LagWindow=LagWindow,
                NBucketsinMonths=NBucketsinMonths,
                the_iteration=the_iteration,
                df_keys=df_keys,
            )

            StatFcstPLLC_list.append(all_lc_output)
            StatFcstPLWLag_list.append(all_w_lag)
            StatFcstPLMLag_list.append(all_m_lag)
            StatFcstPLPMLag_list.append(all_pm_lag)
            StatFcstPLQLag_list.append(all_q_lag)
            StatFcstPLPQLag_list.append(all_pq_lag)

        StatFcstPLLC = concat_to_dataframe(StatFcstPLLC_list)
        StatFcstPLWLag = concat_to_dataframe(StatFcstPLWLag_list)
        StatFcstPLMLag = concat_to_dataframe(StatFcstPLMLag_list)
        StatFcstPLPMLag = concat_to_dataframe(StatFcstPLPMLag_list)
        StatFcstPLQLag = concat_to_dataframe(StatFcstPLQLag_list)
        StatFcstPLPQLag = concat_to_dataframe(StatFcstPLPQLag_list)
    except Exception as e:
        logger.exception(e)
        (
            StatFcstPLLC,
            StatFcstPLWLag,
            StatFcstPLMLag,
            StatFcstPLPMLag,
            StatFcstPLQLag,
            StatFcstPLPQLag,
        ) = (None, None, None, None, None, None)
    return (
        StatFcstPLLC,
        StatFcstPLWLag,
        StatFcstPLMLag,
        StatFcstPLPMLag,
        StatFcstPLQLag,
        StatFcstPLPQLag,
    )


def processIteration(
    StatFcstPL,
    ForecastGenTimeBucket,
    SelectedPlanningCycle,
    PlanningCycles,
    TimeDimension,
    PlanningGrains,
    the_iteration,
    LagWindow="All",
    NBucketsinMonths="12",
    df_keys={},
):
    plugin_name = "DP222CalculateLagModelPL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # configurables
    planning_cycle_date_key = "Planning Cycle.[PlanningCycleDateKey]"
    STAT_FCST_PL_LC = "Stat Fcst PL LC"

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)

    StatFcstPLLC_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PARTIAL_WEEK,
        ]
        + planning_grains
        + [
            STAT_FCST_PL_LC,
        ]
    )
    StatFcstPLLC = pd.DataFrame(columns=StatFcstPLLC_cols)
    StatFcstPLWLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_W_LAG,
        ]
    )
    StatFcstPLWLag = pd.DataFrame(columns=StatFcstPLWLag_cols)

    StatFcstPLMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_M_LAG,
        ]
    )
    StatFcstPLMLag = pd.DataFrame(columns=StatFcstPLMLag_cols)

    StatFcstPLPMLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_MONTH,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PM_LAG,
        ]
    )
    StatFcstPLPMLag = pd.DataFrame(columns=StatFcstPLPMLag_cols)
    StatFcstPLQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_Q_LAG,
        ]
    )
    StatFcstPLQLag = pd.DataFrame(columns=StatFcstPLQLag_cols)
    StatFcstPLPQLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_QUARTER,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PQ_LAG,
        ]
    )
    StatFcstPLPQLag = pd.DataFrame(columns=StatFcstPLPQLag_cols)

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
            NBucketsinMonths = int(NBucketsinMonths) // 4 * 4
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
            NBucketsinMonths = int(NBucketsinMonths) // 4 * 4
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
                StatFcstPLLC,
                StatFcstPLWLag,
                StatFcstPLMLag,
                StatFcstPLPMLag,
                StatFcstPLQLag,
                StatFcstPLPQLag,
            )
        LagWindow = int(LagWindow)
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}
        current_month = SelectedPlanningCycle[o9Constants.MONTH].values[0]
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
                StatFcstPLLC,
                StatFcstPLWLag,
                StatFcstPLMLag,
                StatFcstPLPMLag,
                StatFcstPLQLag,
                StatFcstPLPQLag,
            )
        # need to seperate planning cycle date and current cycle to match the time mapping formats(planning cycle date - 01-Jul-23 but Planning month - M07-23)
        current_planning_cycle = selected_cycle[o9Constants.PLANNING_CYCLE_DATE].values[0]
        if len(StatFcstPL) == 0:
            logger.warning(
                f"Stat Fcst PL empty for the forecast iteration {the_iteration}, returning empty outputs..."
            )
            return (
                StatFcstPLLC,
                StatFcstPLWLag,
                StatFcstPLMLag,
                StatFcstPLPMLag,
                StatFcstPLQLag,
                StatFcstPLPQLag,
            )
        next_n_periods = get_n_time_periods(
            selected_cycle_name,
            LagWindow,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        n_bucket_months = get_n_time_periods(
            current_month,
            int(NBucketsinMonths),
            TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
            {o9Constants.MONTH: o9Constants.MONTH_KEY},
            include_latest_value=True,
        )
        relevant_partial_weeks = TimeDimension[
            TimeDimension[o9Constants.MONTH].isin(n_bucket_months)
        ][o9Constants.PARTIAL_WEEK]
        StatFcstPLLC = StatFcstPL[StatFcstPL[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)]
        StatFcstPLLC.rename(columns={o9Constants.STAT_FCST_PL: STAT_FCST_PL_LC}, inplace=True)
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
        StatFcstwithLag_PW = StatFcstPL.merge(lag_data, on=o9Constants.PARTIAL_WEEK, how="inner")

        StatFcstLagOutput = (
            StatFcstwithLag_PW.groupby(
                planning_grains
                + [
                    o9Constants.VERSION_NAME,
                    o9Constants.LAG,
                    o9Constants.PLANNING_CYCLE_DATE,
                    relevant_time_name,
                ]
            )
            .agg({o9Constants.STAT_FCST_PL: "sum"})
            .reset_index()
        )
        if relevant_time_name == o9Constants.WEEK:
            StatFcstPLWLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_PL: o9Constants.STAT_FCST_W_LAG}
            )
            StatFcstPLWLag = StatFcstPLWLag[StatFcstPLWLag_cols]
        elif relevant_time_name == o9Constants.MONTH:
            StatFcstPLMLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_PL: o9Constants.STAT_FCST_M_LAG}
            )
            StatFcstPLMLag = StatFcstPLMLag[StatFcstPLMLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_MONTH:
            StatFcstPLPMLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_PL: o9Constants.STAT_FCST_PM_LAG}
            )
            StatFcstPLPMLag = StatFcstPLPMLag[StatFcstPLPMLag_cols]
        elif relevant_time_name == o9Constants.QUARTER:
            StatFcstPLQLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_PL: o9Constants.STAT_FCST_Q_LAG}
            )
            StatFcstPLQLag = StatFcstPLQLag[StatFcstPLQLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_QUARTER:
            StatFcstPLPQLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST_PL: o9Constants.STAT_FCST_PQ_LAG}
            )
            StatFcstPLPQLag = StatFcstPLPQLag[StatFcstPLPQLag_cols]
        else:
            logger.warning(
                f"Invalid time bucket {relevant_time_name}, returning empty outputs for iteration {the_iteration}..."
            )

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return (
        StatFcstPLLC,
        StatFcstPLWLag,
        StatFcstPLMLag,
        StatFcstPLPMLag,
        StatFcstPLQLag,
        StatFcstPLPQLag,
    )

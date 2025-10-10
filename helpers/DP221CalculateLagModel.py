import logging

import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Stat Fcst PL W Lag": float,
    "Stat Fcst PL M Lag": float,
    "Stat Fcst PL PM Lag": float,
    "Stat Fcst PL Q Lag": float,
    "Stat Fcst PL PQ Lag": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatFcst,
    SelectedPlanningCycle,
    PlanningCycles,
    TimeDimension,
    PlanningGrains,
    LagWindow="All",
    TimeLevel="Week",
    df_keys={},
):
    plugin_name = "DP221CalculateLagModel"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # configurables
    planning_cycle_date_key = "Planning Cycle.[PlanningCycleDateKey]"

    planning_grains = get_list_of_grains_from_string(input=PlanningGrains)

    StatFcstPLWLag_cols = (
        [
            o9Constants.VERSION_NAME,
            o9Constants.WEEK,
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.LAG,
        ]
        + planning_grains
        + [
            o9Constants.STAT_FCST_PL_W_LAG,
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
            o9Constants.STAT_FCST_PL_M_LAG,
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
            o9Constants.STAT_FCST_PL_PM_LAG,
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
            o9Constants.STAT_FCST_PL_Q_LAG,
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
            o9Constants.STAT_FCST_PL_PQ_LAG,
        ]
    )
    StatFcstPLPQLag = pd.DataFrame(columns=StatFcstPLPQLag_cols)

    try:
        week = TimeLevel.lower() == "week"
        month = TimeLevel.lower() == "month"
        pl_month = TimeLevel.lower() == "planning month"
        quarter = TimeLevel.lower() == "quarter"
        pl_quarter = TimeLevel.lower() == "planning quarter"

        lags = LagWindow.lower()

        if week:
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
        elif pl_month:
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PLANNING_MONTH,
                o9Constants.PLANNING_MONTH_KEY,
            ]
            relevant_time_name = o9Constants.PLANNING_MONTH
            relevant_time_key = o9Constants.PLANNING_MONTH_KEY
            if lags == "all":
                LagWindow = 13
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif month:
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.MONTH,
                o9Constants.MONTH_KEY,
            ]
            relevant_time_name = o9Constants.MONTH
            relevant_time_key = o9Constants.MONTH_KEY
            if lags == "all":
                LagWindow = 13
            else:
                lags = LagWindow.strip().split(",")
                lags = [int(lag) for lag in lags]
                LagWindow = max(lags) + 1
                lag_df = pd.DataFrame({o9Constants.LAG: lags})
        elif pl_quarter:
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
        elif quarter:
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
            logger.warning(f"Unknown Time Level {TimeLevel}, returning empty outputs")
            return (
                StatFcstPLWLag,
                StatFcstPLMLag,
                StatFcstPLPMLag,
                StatFcstPLQLag,
                StatFcstPLPQLag,
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
                f"Planning Cycle {selected_cycle_name} not present in list of Planning Cycles, returning empty outputs for iteration..."
            )
            return (
                StatFcstPLWLag,
                StatFcstPLMLag,
                StatFcstPLPMLag,
                StatFcstPLQLag,
                StatFcstPLPQLag,
            )
        # need to seperate planning cycle date and current cycle to match the time mapping formats(planning cycle date - 01-Jul-23 but Planning month - M07-23)
        current_planning_cycle = selected_cycle[o9Constants.PLANNING_CYCLE_DATE].values[0]
        if len(StatFcst) == 0:
            logger.warning("Stat Fcst empty, returning empty outputs...")
            return (
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
        StatFcstwithLag_PW = StatFcst.merge(lag_data, on=o9Constants.PARTIAL_WEEK, how="inner")
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
            .agg({o9Constants.STAT_FCST: "sum"})
            .reset_index()
        )
        if relevant_time_name == o9Constants.WEEK:
            StatFcstPLWLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST: o9Constants.STAT_FCST_PL_W_LAG}
            )
            StatFcstPLWLag = StatFcstPLWLag[StatFcstPLWLag_cols]
        elif relevant_time_name == o9Constants.MONTH:
            StatFcstPLMLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST: o9Constants.STAT_FCST_PL_M_LAG}
            )
            StatFcstPLMLag = StatFcstPLMLag[StatFcstPLMLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_MONTH:
            StatFcstPLPMLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST: o9Constants.STAT_FCST_PL_PM_LAG}
            )
            StatFcstPLPMLag = StatFcstPLPMLag[StatFcstPLPMLag_cols]
        elif relevant_time_name == o9Constants.QUARTER:
            StatFcstPLQLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST: o9Constants.STAT_FCST_PL_Q_LAG}
            )
            StatFcstPLQLag = StatFcstPLQLag[StatFcstPLQLag_cols]
        elif relevant_time_name == o9Constants.PLANNING_QUARTER:
            StatFcstPLPQLag = StatFcstLagOutput.rename(
                columns={o9Constants.STAT_FCST: o9Constants.STAT_FCST_PL_PQ_LAG}
            )
            StatFcstPLPQLag = StatFcstPLPQLag[StatFcstPLPQLag_cols]
        else:
            logger.warning(f"Invalid time bucket {relevant_time_name}, returning empty outputs...")

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        StatFcstPLWLag = pd.DataFrame(columns=StatFcstPLWLag_cols)
        StatFcstPLMLag = pd.DataFrame(columns=StatFcstPLMLag_cols)
        StatFcstPLPMLag = pd.DataFrame(columns=StatFcstPLPMLag_cols)
        StatFcstPLQLag = pd.DataFrame(columns=StatFcstPLQLag_cols)
        StatFcstPLPQLag = pd.DataFrame(columns=StatFcstPLPQLag_cols)

    return (
        StatFcstPLMLag,
        StatFcstPLWLag,
        StatFcstPLPMLag,
        StatFcstPLQLag,
        StatFcstPLPQLag,
    )

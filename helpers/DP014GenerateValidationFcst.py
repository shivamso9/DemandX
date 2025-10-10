import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import polars as pl
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.DP015SystemStat import main as forecasting_main
from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration
from helpers.utils_polars import create_algo_list

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Stat Fcst Growth Snaive Planning Cycle": float,
    "Stat Fcst Moving Average Planning Cycle": float,
    "Stat Fcst SCHM Planning Cycle": float,
    "Stat Fcst TBATS Planning Cycle": float,
    "Stat Fcst Weighted AOA Planning Cycle": float,
    "Stat Fcst sARIMA Planning Cycle": float,
    "Stat Fcst Growth AOA Planning Cycle": float,
    "Stat Fcst SES Planning Cycle": float,
    "Stat Fcst AR-NNET Planning Cycle": float,
    "Stat Fcst TES Planning Cycle": float,
    "Stat Fcst Theta Planning Cycle": float,
    "Stat Fcst Simple Snaive Planning Cycle": float,
    "Stat Fcst Seasonal Naive YoY Planning Cycle": float,
    "Stat Fcst STLF Planning Cycle": float,
    "Stat Fcst Croston Planning Cycle": float,
    "Stat Fcst Weighted Snaive Planning Cycle": float,
    "Stat Fcst ETS Planning Cycle": float,
    "Stat Fcst Auto ARIMA Planning Cycle": float,
    "Stat Fcst Naive Random Walk Planning Cycle": float,
    "Stat Fcst Simple AOA Planning Cycle": float,
    "Stat Fcst DES Planning Cycle": float,
    "Stat Fcst Prophet Planning Cycle": float,
}


@contextmanager
def suppress_logging():
    """A context manager to temporarily suppress logging for a specific logger."""
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def create_planning_cycles_from_vp_vf_vs(vp, vf, vs):
    try:
        vp_int = int(vp)
        vf_int = int(vf)
        vs_int = int(vs)
        cycles = [str(x) for x in range(vp_int, vp_int + vf_int * vs_int, vs_int)]
        return cycles
    except Exception as e:
        logger.warning(f"Error creating planning cycles from VP, VF, VS: {e}")
        return None


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actual,
    Parameters,
    AlgoParameters,
    TimeDimension,
    CurrentTimePeriod,
    AlgoList,
    ForecastGenTimeBucket,
    StatBucketWeight,
    RUN_VALIDATION_EVERY_FOLD,
    Grains,
    IncludeDiscIntersections,
    MasterAlgoList,
    DefaultAlgoParameters,
    PlannerOverrideCycles="None",
    SeasonalIndices=None,
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    multiprocessing_num_cores=4,
    PlanningCycleDates=pd.DataFrame(),
    ValidationParameters=pd.DataFrame(),
    StatSegment=pd.DataFrame(),
    history_measure="Actual Cleansed System",
    the_iteration=None,
    df_keys={},
):
    try:
        if o9Constants.FORECAST_ITERATION in ForecastGenTimeBucket.columns:
            AllForecastWithPC_list = list()

            for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
                logger.warning(f"--- Processing iteration {the_iteration}")

                decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)
                the_all_forecast_with_lag_dim = decorated_func(
                    Grains=Grains,
                    IncludeDiscIntersections=IncludeDiscIntersections,
                    RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
                    Actual=Actual,
                    StatSegment=StatSegment,
                    Parameters=Parameters,
                    AlgoParameters=AlgoParameters,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    AlgoList=AlgoList,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatBucketWeight=StatBucketWeight,
                    MasterAlgoList=MasterAlgoList,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                    PlannerOverrideCycles=PlannerOverrideCycles,
                    SeasonalIndices=SeasonalIndices,
                    ForecastEngine=ForecastEngine,
                    SellOutOffset=SellOutOffset,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    PlanningCycleDates=PlanningCycleDates,
                    ValidationParameters=ValidationParameters,
                    history_measure=history_measure,
                    the_iteration=the_iteration,
                    df_keys=df_keys,
                )

                AllForecastWithPC_list.append(the_all_forecast_with_lag_dim)

            AllForecastWithPC = concat_to_dataframe(AllForecastWithPC_list)

        else:
            AllForecastWithPC = processIteration(
                Grains=Grains,
                IncludeDiscIntersections=IncludeDiscIntersections,
                RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
                Actual=Actual,
                StatSegment=StatSegment,
                Parameters=Parameters,
                AlgoParameters=AlgoParameters,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                AlgoList=AlgoList,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                MasterAlgoList=MasterAlgoList,
                DefaultAlgoParameters=DefaultAlgoParameters,
                PlannerOverrideCycles=PlannerOverrideCycles,
                SeasonalIndices=SeasonalIndices,
                ForecastEngine=ForecastEngine,
                SellOutOffset=SellOutOffset,
                multiprocessing_num_cores=multiprocessing_num_cores,
                PlanningCycleDates=PlanningCycleDates,
                ValidationParameters=ValidationParameters,
                history_measure=history_measure,
                the_iteration=None,
                df_keys=df_keys,
            )

    except Exception as e:
        logger.exception(e)
        AllForecastWithPC = None

    return AllForecastWithPC


def processIteration(
    Actual,
    Parameters,
    AlgoParameters,
    TimeDimension,
    CurrentTimePeriod,
    AlgoList,
    ForecastGenTimeBucket,
    StatBucketWeight,
    RUN_VALIDATION_EVERY_FOLD,
    Grains,
    IncludeDiscIntersections,
    MasterAlgoList,
    DefaultAlgoParameters,
    PlannerOverrideCycles="None",
    SeasonalIndices=None,
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    multiprocessing_num_cores=4,
    PlanningCycleDates=pd.DataFrame(),
    ValidationParameters=pd.DataFrame(),
    StatSegment=pd.DataFrame(),
    history_measure="Actual Cleansed System",
    the_iteration=None,
    df_keys={},
):
    plugin_name = "DP014GenerateValidationFcst"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    forecast_period_col = "Forecast Period"
    validation_period_col = "Validation Period"
    assigned_algo_list_col = "Assigned Algorithm List"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    partial_week_col = "Time.[Partial Week]"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"
    planning_cycle_date_key = "Planning Cycle.[PlanningCycleDateKey]"
    sell_out_offset_col = "Offset Period"

    # SCHM
    seasonal_index = "SCHM Seasonal Index"
    validation_seasonal_index = "SCHM Validation Seasonal Index"

    seasonal_index_backtest = "SCHM Seasonal Index Backtest"
    validation_seasonal_index_backtest = "SCHM Validation Seasonal Index Backtest"

    # add all algos to MasterAlgoList to avoid discrepancy with planner algo list/different slices
    MasterAlgoList = pd.DataFrame(
        {
            assigned_algo_list_col: "Seasonal Naive YoY,AR-NNET,Moving Average,Naive Random Walk,STLF,Auto ARIMA,sARIMA,Prophet,SES,DES,TES,ETS,Croston,Theta,TBATS,Simple Snaive,Weighted Snaive,Growth Snaive,Simple AOA,Weighted AOA,Growth AOA,SCHM"
        },
        index=[0],
    )

    # Add names of new algorithms here
    ALGO_MASTER_LIST = create_algo_list(
        AlgoDF=pl.from_pandas(MasterAlgoList), algo_list_col=assigned_algo_list_col
    )

    ALL_STAT_FORECAST_COLUMNS = ["Stat Fcst " + x for x in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_PLANNINGCYCLE_COLUMNS = [
        "Stat Fcst " + x + " Planning Cycle" for x in ALGO_MASTER_LIST
    ]

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    logger.info("forecast_level : {}".format(forecast_level))

    # infer time related attributes from forecast gen time bucket
    fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
    logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

    # Default is planning month
    relevant_time_cols = [
        partial_week_col,
        partial_week_key_col,
        planning_month_col,
        planning_month_key_col,
    ]
    relevant_time_name = planning_month_col
    relevant_time_key = planning_month_key_col

    if fcst_gen_time_bucket == "Week":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            week_col,
            week_key_col,
        ]
        relevant_time_name = week_col
        relevant_time_key = week_key_col
    elif fcst_gen_time_bucket == "Month":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            month_col,
            month_key_col,
        ]
        relevant_time_name = month_col
        relevant_time_key = month_key_col
    elif fcst_gen_time_bucket == "Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            quarter_col,
            quarter_key_col,
        ]
        relevant_time_name = quarter_col
        relevant_time_key = quarter_key_col
    elif fcst_gen_time_bucket == "Planning Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            planning_quarter_col,
            planning_quarter_key_col,
        ]
        relevant_time_name = planning_quarter_col
        relevant_time_key = planning_quarter_key_col

    base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

    AllForecastWithPC_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.PARTIAL_WEEK,
        ]
        + ALL_STAT_FORECAST_PLANNINGCYCLE_COLUMNS
    )
    AllForecastWithPC = pd.DataFrame(columns=AllForecastWithPC_cols)

    try:

        # Filter relevant columns from time mapping
        time_attribute_dict = {relevant_time_name: relevant_time_key}

        input_version = ForecastGenTimeBucket[version_col].unique()[0]
        forecast_engine = (
            ForecastEngine[o9Constants.FORECAST_ENGINE].iloc[0]
            if len(ForecastEngine) > 0
            else "Stat"
        )
        if len(ForecastEngine) == 0:
            logger.warning("Forecast Engine measure not populated, considering Stat as default ...")
        if forecast_engine == "ML":
            logger.warning("Forecast Engine is ML, returning empty Stat Validation Fcst outputs...")
            return AllForecastWithPC

        if PlannerOverrideCycles.replace(" ", "").lower() == "none":
            if len(ValidationParameters) > 0:
                ValidationPeriod = (
                    ValidationParameters["Validation Period"].values[0]
                    if "Validation Period" in ValidationParameters.columns
                    else None
                )
                ValidationFold = (
                    ValidationParameters["Validation Fold"].values[0]
                    if "Validation Fold" in ValidationParameters.columns
                    else None
                )
                ValidationStep = (
                    ValidationParameters["Validation Step Size"].values[0]
                    if "Validation Step Size" in ValidationParameters.columns
                    else None
                )
            else:
                logger.warning(
                    "Validation Parameters empty, please populate Validation parameters and rerun the plugin ..."
                )
                return AllForecastWithPC
            if ValidationPeriod is None:
                logger.warning(
                    "Validation Period input is empty, please populate Validation parameters and rerun the plugin ..."
                )
                return AllForecastWithPC
            ValidationFold = ValidationFold if ValidationFold is not None else 1
            ValidationStep = ValidationStep if ValidationStep is not None else 1
            cycles = [
                int(x)
                for x in create_planning_cycles_from_vp_vf_vs(
                    ValidationPeriod, ValidationFold, ValidationStep
                )
            ]
            if not cycles:
                logger.warning(
                    "No validation cycles available - please review the ValidationPeriod, ValidationFold, ValidationStep parameters..."
                )
                validation_cycle = np.array([])
            else:
                if RUN_VALIDATION_EVERY_FOLD:
                    validation_cycle = np.array(list(set(cycles)))
                else:
                    validation_cycle = np.array([max(int(c) for c in cycles)])
            planning_cycles = validation_cycle

            if len(SellOutOffset) == 0:
                logger.warning(
                    f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
                )
                SellOutOffset = pd.DataFrame(
                    {
                        o9Constants.VERSION_NAME: [
                            ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                        ],
                        sell_out_offset_col: [0],
                    }
                )
            offset_period = SellOutOffset[sell_out_offset_col].values[0]
            planning_cycle_dates = []
            for cycle in planning_cycles:
                the_cycle_date = get_n_time_periods(
                    CurrentTimePeriod[relevant_time_name].values[0],
                    -int(cycle + offset_period),
                    TimeDimension[[relevant_time_name, relevant_time_key]].drop_duplicates(),
                    time_attribute_dict,
                    include_latest_value=False,
                )[0]
                planning_cycle_dates.append(the_cycle_date)
            planning_cycles = planning_cycle_dates
            logger.info(f"Planning Cycles : {planning_cycles}")

            PlanningCycleDates = (
                PlanningCycleDates.merge(
                    TimeDimension,
                    left_on="Planning Cycle.[PlanningCycleDateKey]",
                    right_on=o9Constants.PARTIAL_WEEK_KEY,
                    how="inner",
                )
                .drop_duplicates(subset="Planning Cycle.[PlanningCycleDateKey]")
                .sort_values(o9Constants.PARTIAL_WEEK_KEY)
                .reset_index(drop=True)
            )
            PlanningCycleDates = PlanningCycleDates[
                PlanningCycleDates[relevant_time_name].isin(planning_cycles)
            ]

        else:
            planning_cycles = list(set(PlannerOverrideCycles.replace(" ", "").split(",")))
            missing_cycles = list(
                set(planning_cycles).difference(
                    set(PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE].unique())
                )
            )
            if len(missing_cycles) > 0:
                logger.warning(
                    f"Input cycles {missing_cycles} not in Planning Cycle Date dimension, please review the values, returning empty outputs..."
                )
                return AllForecastWithPC
            PlanningCycleDates = PlanningCycleDates.merge(
                pd.DataFrame({o9Constants.PLANNING_CYCLE_DATE: planning_cycles}),
                on=o9Constants.PLANNING_CYCLE_DATE,
                how="inner",
            )
            if o9Constants.DAY in TimeDimension.columns:
                PlanningCycleDates = PlanningCycleDates.merge(
                    TimeDimension,
                    left_on=planning_cycle_date_key,
                    right_on=o9Constants.DAY_KEY,
                    how="inner",
                )
            else:
                PlanningCycleDates = PlanningCycleDates.merge(
                    TimeDimension,
                    left_on=planning_cycle_date_key,
                    right_on=o9Constants.PARTIAL_WEEK_KEY,
                    how="inner",
                )
            if len(PlanningCycleDates) == 0:
                logger.warning(
                    "Cannot find correct time records for the planning cycles, please add Time.[Day], Time.[DayKey] to the Time dimension input..."
                )
                return AllForecastWithPC
        planning_cycle_df = pd.DataFrame({relevant_time_name: planning_cycles})
        planning_cycle_df.insert(0, version_col, input_version)

        # initialize pw col
        planning_cycle_df[partial_week_col] = None
        planning_cycle_df[partial_week_key_col] = None

        Actual = Actual.merge(base_time_mapping, on=partial_week_col, how="inner")

        # Initialize variables to avoid referenced before assigned warning in else clause
        AllForecast = pd.DataFrame()

        all_forecast_with_planning_cycle = []

        for the_idx, the_cycle in enumerate(planning_cycles):
            logger.info(f"Running Validation run for {the_cycle}")
            the_cycle_time_period = TimeDimension[
                TimeDimension[relevant_time_name] == the_cycle
            ].sort_values(
                o9Constants.DAY_KEY
                if o9Constants.DAY_KEY in TimeDimension.columns
                else o9Constants.PARTIAL_WEEK_KEY
            )
            the_current_time_period = the_cycle_time_period.head(1)
            logger.debug(f"the_current_time_period\n{the_current_time_period}")

            current_partial_week_key = the_current_time_period[o9Constants.PARTIAL_WEEK_KEY].values[
                0
            ]
            current_partial_week = the_current_time_period[o9Constants.PARTIAL_WEEK].values[0]

            the_planning_cycle_date = PlanningCycleDates[
                PlanningCycleDates[relevant_time_name] == the_cycle
            ][o9Constants.PLANNING_CYCLE_DATE].values[0]
            logger.debug(f"the_planning_cycle_date : {the_planning_cycle_date}")

            filter_clause = planning_cycle_df[relevant_time_name] == the_cycle
            planning_cycle_df.loc[filter_clause, [partial_week_key_col, partial_week_col]] = [
                current_partial_week_key,
                current_partial_week,
            ]

            cleansed_actuals = Actual[
                Actual[partial_week_key_col].dt.tz_localize(None) < current_partial_week_key
            ]

            new_intersections = Actual[
                (Actual[partial_week_key_col].dt.tz_localize(None) >= current_partial_week_key)
                & (
                    Actual[partial_week_key_col].dt.tz_localize(None)
                    < CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY].dt.tz_localize(None).values[0]
                )
            ]

            # Remove rows from new_intersections that have the same combination of grains as cleansed_actuals
            if len(cleansed_actuals) > 0 and len(new_intersections) > 0:
                cleansed_keys = cleansed_actuals[forecast_level].drop_duplicates()
                new_intersections = new_intersections.merge(
                    cleansed_keys, on=forecast_level, how="left", indicator=True
                )
                new_intersections = new_intersections[
                    new_intersections["_merge"] == "left_only"
                ].drop(columns=["_merge"])
            # SCHM Logic for backtesting
            # Renaming columns as per backtesting logic
            SeasonalIndices.rename(
                columns={
                    validation_seasonal_index_backtest: validation_seasonal_index,
                    seasonal_index_backtest: seasonal_index,
                },
                inplace=True,
            )
            logger.info("SeasonalIndices : \n{}".format(SeasonalIndices))

            Parameters[forecast_period_col] = Parameters[validation_period_col]
            logger.info(
                f"Generating Validation forecast for {int(Parameters[validation_period_col].values[0])} periods..."
            )
            ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] = the_iteration
            if len(cleansed_actuals) > 0:
                logger.warning(
                    f"Generating validation forecasts for all the following mature intersections \n{cleansed_actuals[forecast_level].drop_duplicates()}"
                )
                with suppress_logging():
                    AllForecast, ForecastModel = forecasting_main(
                        Grains=Grains,
                        history_measure=history_measure,
                        AlgoList=AlgoList,
                        Actual=cleansed_actuals,
                        TimeDimension=TimeDimension,
                        ForecastParameters=Parameters,
                        AlgoParameters=AlgoParameters,
                        CurrentTimePeriod=the_current_time_period,
                        non_disc_intersections=StatSegment,
                        ForecastGenTimeBucket=ForecastGenTimeBucket,
                        StatBucketWeight=StatBucketWeight,
                        SellOutOffset=SellOutOffset,
                        IncludeDiscIntersections=IncludeDiscIntersections,
                        multiprocessing_num_cores=multiprocessing_num_cores,
                        MasterAlgoList=MasterAlgoList,
                        DefaultAlgoParameters=DefaultAlgoParameters,
                        SeasonalIndices=SeasonalIndices,
                        model_params=False,
                        df_keys=df_keys,
                    )
            if len(new_intersections) > 0:
                logger.warning(
                    f"Relevant actuals not found for the intersections below, doing in sample prediction for the intersections:\n{new_intersections[forecast_level].drop_duplicates()}"
                )
                Parameters_insample = Parameters.copy()
                Parameters_insample[o9Constants.BESTFIT_METHOD] = "In Sample"
                Parameters_insample[validation_period_col] = 1
                Parameters_insample[forecast_period_col] = 1

                # current period of the validation cycle
                prev_bucket_key = TimeDimension[
                    TimeDimension[relevant_time_key].dt.tz_localize(None)
                    < CurrentTimePeriod[relevant_time_key].values[0]
                ][relevant_time_key].values[-1]
                current_time_period_insample = TimeDimension[
                    TimeDimension[relevant_time_key].dt.tz_localize(None) == prev_bucket_key
                ].head(1)

                # previous period of the validation cycle
                prev_bucket_key_insample = TimeDimension[
                    TimeDimension[relevant_time_key].dt.tz_localize(None) < prev_bucket_key
                ][relevant_time_key].values[-1]
                prev_time_period_insample = TimeDimension[
                    TimeDimension[relevant_time_key].dt.tz_localize(None)
                    == prev_bucket_key_insample
                ].head(1)

                # Get count of history points available for each intersection
                new_intersections = new_intersections[
                    [
                        o9Constants.VERSION_NAME,
                    ]
                    + forecast_level
                    + [relevant_time_name, history_measure]
                ].drop_duplicates()
                new_intersections["count"] = new_intersections.groupby(
                    [
                        o9Constants.VERSION_NAME,
                    ]
                    + forecast_level
                )[history_measure].transform("count")

                # Identify intersections with only 1 datapoint
                one_datapoint_intersections = new_intersections[
                    new_intersections["count"] == 1
                ].copy()
                if len(one_datapoint_intersections) > 0:
                    # Add a dummy row with the previous period for each such intersections
                    unique_combinations = one_datapoint_intersections[
                        [
                            o9Constants.VERSION_NAME,
                        ]
                        + forecast_level
                        + [relevant_time_name]
                    ].drop_duplicates()
                    dummy_rows = unique_combinations.copy()
                    dummy_rows[relevant_time_name] = prev_time_period_insample[
                        relevant_time_name
                    ].values[0]
                    dummy_rows[history_measure] = np.nan  # or set to a default value if needed
                    # Concatenate dummy rows to new_intersections
                    new_intersections = pd.concat(
                        [new_intersections, dummy_rows], ignore_index=True
                    )

                # add relevant time key and sort by time before shifting
                new_intersections = new_intersections.merge(
                    base_time_mapping[[relevant_time_name, relevant_time_key]].drop_duplicates(),
                    on=relevant_time_name,
                    how="inner",
                )
                new_intersections = new_intersections.sort_values(
                    [
                        o9Constants.VERSION_NAME,
                    ]
                    + forecast_level
                    + [relevant_time_key]
                )
                new_intersections.drop(columns=[relevant_time_key], inplace=True)

                # shift the history measure 1 period back to get the "forecast" for the current validation period
                new_intersections["updated fcst"] = new_intersections.groupby(
                    forecast_level
                    + [
                        o9Constants.VERSION_NAME,
                    ]
                )[history_measure].shift(-1)
                new_intersections = new_intersections.merge(
                    base_time_mapping, on=relevant_time_name, how="inner"
                )
                new_intersections[history_measure] = new_intersections["updated fcst"]
                new_intersections.drop(columns=["updated fcst", "count"], inplace=True)
                with suppress_logging():
                    AllForecast_insample, ForecastModel_insample = forecasting_main(
                        Grains=Grains,
                        history_measure=history_measure,
                        AlgoList=AlgoList,
                        Actual=new_intersections,
                        TimeDimension=TimeDimension,
                        ForecastParameters=Parameters_insample,
                        AlgoParameters=AlgoParameters,
                        CurrentTimePeriod=current_time_period_insample,
                        non_disc_intersections=StatSegment,
                        ForecastGenTimeBucket=ForecastGenTimeBucket,
                        StatBucketWeight=StatBucketWeight,
                        SellOutOffset=SellOutOffset,
                        IncludeDiscIntersections=IncludeDiscIntersections,
                        multiprocessing_num_cores=multiprocessing_num_cores,
                        MasterAlgoList=MasterAlgoList,
                        DefaultAlgoParameters=DefaultAlgoParameters,
                        SeasonalIndices=SeasonalIndices,
                        model_params=False,
                        df_keys=df_keys,
                    )
                if len(AllForecast_insample) > 0:
                    AllForecast = concat_to_dataframe([AllForecast, AllForecast_insample])

            if len(AllForecast) == 0:
                logger.warning(
                    "AllForecast is empty for the cycle : {}, slice : {}".format(the_cycle, df_keys)
                )
                continue

            AllForecast[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            all_forecast_with_planning_cycle += [AllForecast]
        AllForecastWithPC = concat_to_dataframe(all_forecast_with_planning_cycle)
        rename_mapping = {
            old: new
            for old, new in zip(ALL_STAT_FORECAST_COLUMNS, ALL_STAT_FORECAST_PLANNINGCYCLE_COLUMNS)
        }
        AllForecastWithPC.rename(columns=rename_mapping, inplace=True)
        if len(AllForecastWithPC.columns) == 0:
            AllForecastWithPC = pd.DataFrame(columns=AllForecastWithPC_cols)
        AllForecastWithPC = AllForecastWithPC[AllForecastWithPC_cols].drop_duplicates()
    except Exception as e:
        logger.exception(
            f"Exception {e} for slice : {df_keys}, returning empty dataframe as output ..."
        )
        AllForecastWithPC = pd.DataFrame(columns=AllForecastWithPC_cols)

    return AllForecastWithPC

import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.DP015IdentifyBestFitModel import (
    processIteration as identify_best_fit,  # type: ignore
)
from helpers.DP015PopulateBestFitForecast import (
    processIteration as populate_best_fit,  # type: ignore
)
from helpers.DP015SystemStat import processIteration as system_stat  # type: ignore
from helpers.o9Constants import o9Constants
from helpers.utils import create_algo_list, get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")
# col_mapping = {'Stat Fcst HML': float}


@log_inputs_and_outputs
@timed
# @map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    history_measure,
    AlgoList,
    Actual,
    TimeDimension,
    ForecastParameters,
    AlgoParameters,
    CurrentTimePeriod,
    HolidayData,
    StatRegionMapping,
    ForecastGenTimeBucket,
    StatBucketWeight,
    MasterAlgoList,
    DefaultAlgoParameters,
    UseHolidays,
    IncludeDiscIntersections,
    multiprocessing_num_cores,
    AlgoListMeasure,
    df_keys,
    PlanningGrains,
    MasterAssignedAlgorithmList,
    ForecastEngine,
):
    plugin_name = "DP204SystemStatRetail"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # combine grains to get forecast level
    stat_level = get_list_of_grains_from_string(input=Grains)

    planning_level = get_list_of_grains_from_string(input=PlanningGrains)

    # Add names of new algorithms here
    # Extract only the algos that were assigned to intersections
    ALGO_MASTER_LIST = create_algo_list(
        AlgoDF=MasterAlgoList,
        algo_list_col=o9Constants.ASSIGNED_ALGORITHM_LIST,
        use_all_algos=False,
    )

    logger.info(f"Master Algo List: {ALGO_MASTER_LIST}")

    ALL_STAT_FORECAST_COLUMNS = [" ".join(["Stat Fcst", x]) for x in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_LB_COLUMNS = [" ".join(["Stat Fcst", x, "80% LB"]) for x in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_UB_COLUMNS = [" ".join(["Stat Fcst", x, "80% UB"]) for x in ALGO_MASTER_LIST]
    ALL_STAT_FCST_BOUND_COLUMNS = ALL_STAT_FORECAST_LB_COLUMNS + ALL_STAT_FORECAST_UB_COLUMNS
    ALL_FORECAST_COLUMNS = ALL_STAT_FORECAST_COLUMNS
    # Collect forecast bound columns
    ForecastBounds_req_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [o9Constants.PARTIAL_WEEK]
        + ALL_STAT_FCST_BOUND_COLUMNS
    )
    logger.info("Extracting forecast level ...")

    SystemStatFcstL1_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [o9Constants.FORECAST_ITERATION, o9Constants.PARTIAL_WEEK]
        + ALL_FORECAST_COLUMNS
        + [o9Constants.SYSTEM_STAT_FCST_L1]
    )
    SystemStatFcstL1 = pd.DataFrame(columns=SystemStatFcstL1_cols)

    StatFcstL1_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [o9Constants.FORECAST_ITERATION, o9Constants.PARTIAL_WEEK]
        + [o9Constants.STAT_FCST_L1]
    )
    StatFcstL1 = pd.DataFrame(columns=StatFcstL1_cols)

    ModelParameters_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [
            o9Constants.FORECAST_ITERATION,
            o9Constants.STAT_ALGORITHM,
            o9Constants.STAT_RULE,
            o9Constants.ALGORITHM_PARAMETERS,
            o9Constants.VALIDATION_METHOD,
            o9Constants.RUN_TIME,
            o9Constants.VALIDATION_ERROR,
        ]
    )
    ModelParameters = pd.DataFrame(columns=ModelParameters_cols)

    SystemBestfitAlgorithm_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [
            o9Constants.FORECAST_ITERATION,
            o9Constants.SYSTEM_BESTFIT_ALGORITHM,
        ]
    )
    SystemBestfitAlgorithm = pd.DataFrame(columns=SystemBestfitAlgorithm_cols)

    BestfitAlgorithm_cols = (
        [o9Constants.VERSION_NAME]
        + stat_level
        + [
            o9Constants.FORECAST_ITERATION,
            o9Constants.BESTFIT_ALGORITHM,
        ]
    )
    BestfitAlgorithm = pd.DataFrame(columns=BestfitAlgorithm_cols)

    StatFcstPL_cols = (
        [o9Constants.VERSION_NAME]
        + planning_level
        + [
            o9Constants.FORECAST_ITERATION,
            o9Constants.PARTIAL_WEEK,
            o9Constants.STAT_FCST_PL,
        ]
    )
    StatFcstPL = pd.DataFrame(columns=StatFcstPL_cols)
    try:
        AlgoList = AlgoList.merge(
            MasterAssignedAlgorithmList,
            on=[
                "Version.[Version Name]",
                "Forecast Iteration.[Forecast Iteration]",
            ],
            how="left",
        )
        AlgoList["Assigned Algorithm List"] = AlgoList["Master Assigned Algorithm List"]

        # Drop the extra `Master Assigned Algorithm List` column
        AlgoList.drop(columns=["Master Assigned Algorithm List"], inplace=True)

        if Actual.empty:
            logger.warning(f"Actual is empty for slice {df_keys}")
            return (
                SystemStatFcstL1,
                StatFcstL1,
                ModelParameters,
                SystemBestfitAlgorithm,
                BestfitAlgorithm,
                StatFcstPL,
            )

        # segmentation won't be populated, so create an empty dataframe
        non_disc_intersections = pd.DataFrame(
            columns=[o9Constants.VERSION_NAME, o9Constants.CLASS]
            + stat_level
            + [o9Constants.PRODUCT_CUSTOMER_L1_SEGMENT]
        )

        # Assume trend and seasonality for all intersections
        AlgoList[o9Constants.TREND_L1] = "YES"
        AlgoList[o9Constants.SEASONALITY_L1] = "YES"
        AlgoList[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST] = AlgoList[
            o9Constants.ASSIGNED_ALGORITHM_LIST
        ]
        the_iteration = "FI-Stat"
        # get system stat
        AllForecast, ForecastModel = system_stat(
            Grains=Grains,
            history_measure=history_measure,
            AlgoList=AlgoList,
            Actual=Actual,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            AlgoParameters=AlgoParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            non_disc_intersections=non_disc_intersections,
            HolidayData=HolidayData,
            StatRegionMapping=StatRegionMapping,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            StatBucketWeight=StatBucketWeight,
            MasterAlgoList=MasterAlgoList,
            DefaultAlgoParameters=DefaultAlgoParameters,
            the_iteration=the_iteration,
            UseHolidays=UseHolidays,
            IncludeDiscIntersections=IncludeDiscIntersections,
            multiprocessing_num_cores=multiprocessing_num_cores,
            AlgoListMeasure=AlgoListMeasure,
            df_keys=df_keys,
        )
        if len(AllForecast) == 0:
            raise ValueError(f"AllForecast is empty for {df_keys}")

        logger.info("Generating best fit ...")

        # Collect actuals and forecasts in the same dataframe as required by best fit
        join_cols = [o9Constants.VERSION_NAME] + stat_level + [o9Constants.PARTIAL_WEEK]

        ActualsAndForecastData = AllForecast.drop(
            ALL_STAT_FCST_BOUND_COLUMNS, axis=1, errors="ignore"
        ).merge(Actual, on=join_cols, how="inner")
        # To make sure AllForecast contains all the ForecastBoundsreq_cols
        for col in ForecastBounds_req_cols:
            if col not in AllForecast.columns:
                AllForecast[col] = np.nan

        ForecastBounds = AllForecast[ForecastBounds_req_cols]

        input_version = AllForecast[o9Constants.VERSION_NAME].unique()[0]

        # hard code selection criteria
        SelectionCriteria = pd.DataFrame(
            {
                o9Constants.VERSION_NAME: input_version,
                o9Constants.BESTFIT_SELECTION_CRITERIA: "Validation Error",
            },
            index=[0],
        )

        # identify best fit
        BestFitAlgo, ValidationError = identify_best_fit(
            Grains=Grains,
            HistoryMeasure=history_measure,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            ActualsAndForecastData=ActualsAndForecastData,
            OverrideFlatLineForecasts="False",
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            AssignedAlgoList=AlgoList,
            SelectionCriteria=SelectionCriteria,
            ForecastEngine=ForecastEngine,
            MasterAlgoList=MasterAlgoList,
            multiprocessing_num_cores=multiprocessing_num_cores,
            the_iteration=the_iteration,
            df_keys=df_keys,
            Weights=None,
            Violations=None,
        )

        # Assign mandatory column
        BestFitAlgo[o9Constants.PLANNER_BESTFIT_ALGORITHM] = BestFitAlgo[
            o9Constants.SYSTEM_BESTFIT_ALGORITHM
        ]

        # populate best fit
        (
            SystemStatFcstL1,
            BestFitAlgorithmCandidateOutput,
            BestFitViolationOutput,
        ) = populate_best_fit(
            Grains=Grains,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            ForecastData=AllForecast,
            ForecastBounds=ForecastBounds,
            BestFitAlgo=BestFitAlgo,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            StatBucketWeight=StatBucketWeight,
            Violations=None,
            df_keys={},
        )

        SystemStatFcstL1 = SystemStatFcstL1.merge(
            AllForecast,
            on=[
                o9Constants.VERSION_NAME,
                o9Constants.PARTIAL_WEEK,
            ]
            + stat_level,
            how="left",
        )

        # add iteration
        input_iteration = AlgoList[o9Constants.FORECAST_ITERATION].unique()[0]
        SystemStatFcstL1[o9Constants.FORECAST_ITERATION] = input_iteration
        ForecastModel[o9Constants.FORECAST_ITERATION] = input_iteration
        ValidationError[o9Constants.FORECAST_ITERATION] = input_iteration
        BestFitAlgo[o9Constants.FORECAST_ITERATION] = input_iteration

        # To make sure SystemStatFcstL1 contains SystemStatFcstL1_cols
        for col in SystemStatFcstL1_cols:
            if col not in SystemStatFcstL1.columns:
                SystemStatFcstL1[col] = np.nan
        SystemStatFcstL1 = SystemStatFcstL1[SystemStatFcstL1_cols]

        StatFcstL1 = SystemStatFcstL1.rename(
            columns={o9Constants.SYSTEM_STAT_FCST_L1: o9Constants.STAT_FCST_L1}
        )

        StatFcstL1 = StatFcstL1[StatFcstL1_cols]

        ModelParameters = ForecastModel.merge(
            ValidationError,
            on=[
                o9Constants.VERSION_NAME,
                o9Constants.FORECAST_ITERATION,
                o9Constants.STAT_ALGORITHM,
                o9Constants.STAT_RULE,
            ]
            + stat_level,
            how="outer",
        )
        ModelParameters = ModelParameters[ModelParameters_cols]

        SystemBestfitAlgorithm = BestFitAlgo.copy()
        SystemBestfitAlgorithm = SystemBestfitAlgorithm[SystemBestfitAlgorithm_cols]
        BestfitAlgorithm = BestFitAlgo.rename(
            columns={o9Constants.SYSTEM_BESTFIT_ALGORITHM: o9Constants.BESTFIT_ALGORITHM}
        )
        BestfitAlgorithm = BestfitAlgorithm[BestfitAlgorithm_cols]

        StatFcstPL = StatFcstL1.rename(columns={o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_PL})
        grain_rename_mapping = dict(zip(stat_level, planning_level))
        StatFcstPL.rename(columns=grain_rename_mapping, inplace=True)
        StatFcstPL = StatFcstPL[StatFcstPL_cols]
        # Removing stat fcst HML from systemstatfcstl1 because it wont be generated in this plugin
        SystemStatFcstL1 = SystemStatFcstL1.drop(columns=["Stat Fcst HML"])
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return (
        SystemStatFcstL1,
        StatFcstL1,
        ModelParameters,
        SystemBestfitAlgorithm,
        BestfitAlgorithm,
        StatFcstPL,
    )

import logging

import pandas as pd
import polars as pl
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.DP015IdentifyBestFitModel import run_best_fit_analysis_pl
from helpers.DP015PopulateBestFitForecast import processIteration as bestfitmain
from helpers.o9Constants import o9Constants
from helpers.utils_polars import build_iter_inputs_from_dict

logger = logging.getLogger("o9_logger")


def processIteration(
    Grains,
    HistoryMeasure,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    Actuals,
    ValidationForecastData,
    ValidationParameters,
    PlanningCycleDates,
    ForecastData,
    ForecastBounds,
    PlannerBestfitAlgo,
    OverrideFlatLineForecasts,
    ForecastGenTimeBucket,
    AssignedAlgoList,
    SelectionCriteria,
    MasterAlgoList,
    multiprocessing_num_cores,
    df_keys,
    StatBucketWeight,
    Weights=None,
    Violations=None,
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
):
    plugin_name = "DP015IdentifyBestFitandPopulateBestfit"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Prepare forecast level (grain columns)
    forecast_level = [x.strip() for x in Grains.split(",") if x.strip() not in ["", "NA"]]

    BestFitAlgo = pd.DataFrame()
    ValidationError = pd.DataFrame()
    BestFitAlgorithmCandidateOutput = pd.DataFrame()
    BestFitViolationOutput = pd.DataFrame()
    BestFitForecast = pd.DataFrame()

    # Configurables
    try:

        # Prepare forecast level (grain columns)
        forecast_level = [x.strip() for x in Grains.split(",") if x.strip() not in ["", "NA"]]
        # Add Logic here for combined Plugin

        # Calling the identifybestfit
        logger.info("______________________________________")
        logger.info("Executing the IdentifyBestfit")

        BestFitAlgo, ValidationError, ValidationErrorPlanningCycle, ValidationForecast = (
            run_best_fit_analysis_pl(
                Grains_str=Grains,
                HistoryMeasure_str=HistoryMeasure,
                TimeDimension_pl=TimeDimension,
                ForecastParameters_pl=ForecastParameters,
                CurrentTimePeriod_pl=CurrentTimePeriod,
                ForecastEngine_pl=ForecastEngine,
                Actuals_pl=Actuals,
                ForecastData_pl=ValidationForecastData,
                ValidationParameters_pl=ValidationParameters,
                PlanningCycleDates_pl=PlanningCycleDates,
                OverrideFlatLineForecasts_str=OverrideFlatLineForecasts,
                ForecastGenTimeBucket_pl=ForecastGenTimeBucket,
                AssignedAlgoList_pl=AssignedAlgoList,
                SelectionCriteria_pl=SelectionCriteria,
                MasterAlgoList_pl=MasterAlgoList,
                Weights_pl=Weights,
                Violations_pl=Violations,
            )
        )

        logger.info("IdentifyBestfit Execution is complete")
        logger.info(f"BestFitAlgo Shape {BestFitAlgo.shape}")
        logger.info(f"ValidationError Shape {ValidationError.shape}")
        logger.info("______________________________________")

        # Adding the PlannerBestfitAlgo to the Plugin

        if BestFitAlgo.is_empty() and ValidationError.is_empty():
            return (
                BestFitAlgo,
                ValidationError,
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        PlannerBestfitAlgo = BestFitAlgo.join(
            PlannerBestfitAlgo, on=[o9Constants.VERSION_NAME] + forecast_level, how="left"
        ).rename({o9Constants.SYSTEM_BESTFIT_ALGORITHM: o9Constants.SYSTEM_BESTFIT_ALGORITHM_FINAL})

        # Calling the populatebestfit
        logger.info("______________________________________")
        logger.info("Executing the PopulateBestfit")

        BestFitForecast, BestFitAlgorithmCandidateOutput, BestFitViolationOutput = bestfitmain(
            Grains=Grains,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            ForecastData=ForecastData,
            ForecastBounds=ForecastBounds,
            BestFitAlgo=PlannerBestfitAlgo,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            StatBucketWeight=StatBucketWeight,
            Violations=Violations,
            df_keys=df_keys,
        )

        logger.info("IdentifyBestfit Execution is complete")
        logger.info(f"BestFitAlgo Shape {BestFitForecast.shape}")
        logger.info(f"BestFitAlgo Shape {BestFitAlgorithmCandidateOutput.shape}")
        logger.info(f"BestFitAlgo Shape {BestFitViolationOutput.shape}")
        logger.info("______________________________________")

        # Converting back to pandas

        BestFitAlgo = BestFitAlgo.to_pandas()
        ValidationError = ValidationError.to_pandas()
        ValidationErrorPlanningCycle = ValidationErrorPlanningCycle.to_pandas()
        ValidationForecast = ValidationForecast.to_pandas()
        BestFitForecast = BestFitForecast.to_pandas()
        BestFitAlgorithmCandidateOutput = BestFitAlgorithmCandidateOutput.to_pandas()
        BestFitViolationOutput = BestFitViolationOutput.to_pandas()

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")

    return (
        BestFitAlgo,
        ValidationError,
        BestFitForecast,
        BestFitAlgorithmCandidateOutput,
        BestFitViolationOutput,
        ValidationErrorPlanningCycle,
        ValidationForecast,
    )


col_mapping = {
    "Validation Error": float,
    "Validation Error Planning Cycle": float,
    "Composite Error": float,
    "System Bestfit Algorithm": str,
    "Validation Actual": float,
    "Validation Fcst": float,
    "Validation Fcst Abs Error": float,
    "Stat Fcst SES": float,
    "Stat Fcst DES": float,
    "Stat Fcst TES": float,
    "Stat Fcst ETS": float,
    "Stat Fcst Auto ARIMA": float,
    "Stat Fcst sARIMA": float,
    "Stat Fcst Prophet": float,
    "Stat Fcst STLF": float,
    "Stat Fcst Theta": float,
    "Stat Fcst Croston": float,
    "Stat Fcst TBATS": float,
    "Stat Fcst AR-NNET": float,
    "Stat Fcst Simple Snaive": float,
    "Stat Fcst Weighted Snaive": float,
    "Stat Fcst Growth Snaive": float,
    "Stat Fcst Naive Random Walk": float,
    "Stat Fcst Seasonal Naive YoY": float,
    "Stat Fcst Moving Average": float,
    "Stat Fcst Simple AOA": float,
    "Stat Fcst Growth AOA": float,
    "Stat Fcst Weighted AOA": float,
    "Stat Fcst SCHM": float,
    "Stat Fcst CML": float,
    "Stat Fcst Ensemble": float,
    "System Stat Fcst L1": float,
    "System Stat Fcst L1 80% UB": float,
    "System Stat Fcst L1 80% LB": float,
    "Straight Line": float,
    "Trend Violation": float,
    "Level Violation": float,
    "Seasonal Violation": float,
    "Range Violation": float,
    "Run Count": float,
    "No Alerts": float,
    "Is Bestfit": float,
    "Algorithm Parameters": str,
    "Fcst Next N Buckets": float,
    "Run Time": float,
    "Validation Error": float,
    "Validation Method": str,
    "Validation Actual": float,
    "Validation Fcst Abs Error": float,
    "Validation Fcst": float,
    "Bestfit Straight Line": float,
    "Bestfit Trend Violation": float,
    "Bestfit Level Violation": float,
    "Bestfit Seasonal Violation": float,
    "Bestfit Range Violation": float,
    "Missing Bestfit": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    HistoryMeasure,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    Actuals,
    ValidationForecastData,
    ValidationParameters,
    PlanningCycleDates,
    ForecastData,
    ForecastBounds,
    PlannerBestfitAlgo,
    OverrideFlatLineForecasts,
    ForecastGenTimeBucket,
    AssignedAlgoList,
    SelectionCriteria,
    MasterAlgoList,
    multiprocessing_num_cores,
    df_keys,
    StatBucketWeight,
    Weights=None,
    Violations=None,
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
):
    try:
        BestFitAlgoList = list()
        ValidationErrorList = list()
        BestFitForecastList = list()
        BestFitAlgorithmCandidateOutputList = list()
        BestFitViolationOutputList = list()
        ValidationErrorPlanningCycleList = list()
        ValidationForecastList = list()

        # Converting in polars (as a dict, similar to DP015IdentifyBestFitModel.py)
        base_inputs = {
            "Grains": Grains,
            "HistoryMeasure": HistoryMeasure,
            "TimeDimension": pl.from_pandas(TimeDimension),
            "ForecastParameters": pl.from_pandas(ForecastParameters),
            "CurrentTimePeriod": pl.from_pandas(CurrentTimePeriod),
            "Actuals": pl.from_pandas(Actuals),
            "ValidationForecastData": pl.from_pandas(ValidationForecastData),
            "ValidationParameters": pl.from_pandas(ValidationParameters),
            "PlanningCycleDates": pl.from_pandas(PlanningCycleDates),
            "ForecastData": pl.from_pandas(ForecastData),
            "ForecastBounds": pl.from_pandas(ForecastBounds),
            "PlannerBestfitAlgo": pl.from_pandas(PlannerBestfitAlgo),
            "OverrideFlatLineForecasts": OverrideFlatLineForecasts,
            "ForecastGenTimeBucket": pl.from_pandas(ForecastGenTimeBucket),
            "AssignedAlgoList": pl.from_pandas(AssignedAlgoList),
            "SelectionCriteria": pl.from_pandas(SelectionCriteria),
            "MasterAlgoList": pl.from_pandas(MasterAlgoList),
            "SellOutOffset": pl.from_pandas(SellOutOffset) if SellOutOffset is not None else None,
            "multiprocessing_num_cores": multiprocessing_num_cores,
            "df_keys": df_keys,
            "StatBucketWeight": pl.from_pandas(StatBucketWeight),
            "Weights": pl.from_pandas(Weights) if Weights is not None else None,
            "Violations": pl.from_pandas(Violations) if Violations is not None else None,
            "ForecastEngine": (
                pl.from_pandas(ForecastEngine) if ForecastEngine is not None else None
            ),
        }
        for iter_val in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {iter_val}")

            try:
                iter_inputs = build_iter_inputs_from_dict(base_inputs, iter_val)
                (
                    the_best_fit_algo,
                    the_validation_error,
                    the_best_fit_forecast,
                    the_bestfit_algorithm_candidate,
                    the_bestfit_violationoutput,
                    the_validation_error_planningcycle,
                    the_validation_forecast,
                ) = processIteration(**iter_inputs)

                logger.info(f"--- Processing iteration: {iter_val} for slice: {df_keys} ---")

                if the_best_fit_algo is not None and not the_best_fit_algo.empty:
                    the_best_fit_algo.insert(0, o9Constants.FORECAST_ITERATION, iter_val)
                    BestFitAlgoList.append(the_best_fit_algo)

                if the_validation_error is not None and not the_validation_error.empty:
                    the_validation_error.insert(0, o9Constants.FORECAST_ITERATION, iter_val)
                    ValidationErrorList.append(the_validation_error)

                if the_best_fit_forecast is not None and not the_best_fit_forecast.empty:
                    the_best_fit_forecast.insert(0, o9Constants.FORECAST_ITERATION, iter_val)
                    BestFitForecastList.append(the_best_fit_forecast)

                if (
                    the_bestfit_algorithm_candidate is not None
                    and not the_bestfit_algorithm_candidate.empty
                ):
                    the_bestfit_algorithm_candidate.insert(
                        0, o9Constants.FORECAST_ITERATION, iter_val
                    )
                    BestFitAlgorithmCandidateOutputList.append(the_bestfit_algorithm_candidate)

                if (
                    the_bestfit_violationoutput is not None
                    and not the_bestfit_violationoutput.empty
                ):
                    the_bestfit_violationoutput.insert(0, o9Constants.FORECAST_ITERATION, iter_val)
                    BestFitViolationOutputList.append(the_bestfit_violationoutput)

                if (
                    the_validation_error_planningcycle is not None
                    and not the_validation_error_planningcycle.empty
                ):
                    the_validation_error_planningcycle.insert(
                        0, o9Constants.FORECAST_ITERATION, iter_val
                    )
                    ValidationErrorPlanningCycleList.append(the_validation_error_planningcycle)

                if the_validation_forecast is not None and not the_validation_forecast.empty:
                    the_validation_forecast.insert(0, o9Constants.FORECAST_ITERATION, iter_val)
                    ValidationForecastList.append(the_validation_forecast)

            except Exception as e:
                logger.exception(
                    f"Failed to process iteration '{iter_val}' for slice {df_keys}: {e}. Skipping."
                )
                continue

        BestFitAlgo = (
            pd.concat(BestFitAlgoList, ignore_index=True) if BestFitAlgoList else pd.DataFrame()
        )
        ValidationError = (
            pd.concat(ValidationErrorList, ignore_index=True)
            if ValidationErrorList
            else pd.DataFrame()
        )
        BestFitForecast = (
            pd.concat(BestFitForecastList, ignore_index=True)
            if BestFitForecastList
            else pd.DataFrame()
        )
        BestFitAlgorithmCandidateOutput = (
            pd.concat(BestFitAlgorithmCandidateOutputList, ignore_index=True)
            if BestFitAlgorithmCandidateOutputList
            else pd.DataFrame()
        )
        BestFitViolationOutput = (
            pd.concat(BestFitViolationOutputList, ignore_index=True)
            if BestFitViolationOutputList
            else pd.DataFrame()
        )
        ValidationErrorPlanningCycle = (
            pd.concat(ValidationErrorPlanningCycleList, ignore_index=True)
            if ValidationErrorPlanningCycleList
            else pd.DataFrame()
        )
        ValidationForecast = (
            pd.concat(ValidationForecastList, ignore_index=True)
            if ValidationForecastList
            else pd.DataFrame()
        )

    except Exception as e:
        logger.exception(e)
        (
            BestFitAlgo,
            ValidationError,
            BestFitForecast,
            BestFitAlgorithmCandidateOutput,
            BestFitViolationOutput,
            ValidationErrorPlanningCycle,
            ValidationForecast,
        ) = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    return (
        BestFitAlgo,
        ValidationError,
        BestFitForecast,
        BestFitAlgorithmCandidateOutput,
        BestFitViolationOutput,
        ValidationErrorPlanningCycle,
        ValidationForecast,
    )

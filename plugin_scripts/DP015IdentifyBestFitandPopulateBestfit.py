"""
    Plugin : DP015IdentifyBestFitModel
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Stat Actual
    OverrideFlatLineForecasts - False
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    MultiprocessingNumCores - 1

Input Queries:
    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({  Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets]  }) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Actuals : Select ([Sequence].[Sequence] *  [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * &AllPastPartialWeeks ) on row,  ({Measure.[Stat Actual], Measure.[Actual Cleansed], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Actual Cleansed]), (Measure.[Slice Association Stat] == 1)};

    ValidationForecastData : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Planning Cycle].[Planning Cycle Date] * &AllPastPartialWeeks * [Account].[Stat Account] ) on row,  ({Measure.[Stat Fcst AR-NNET Planning Cycle], Measure.[Stat Fcst Auto ARIMA Planning Cycle], Measure.[Stat Fcst CML Planning Cycle], Measure.[Stat Fcst Croston Planning Cycle], Measure.[Stat Fcst DES Planning Cycle], Measure.[Stat Fcst ETS Planning Cycle], Measure.[Stat Fcst Ensemble Planning Cycle], Measure.[Stat Fcst Growth AOA Planning Cycle], Measure.[Stat Fcst Growth Snaive Planning Cycle], Measure.[Stat Fcst ML L1 Planning Cycle], Measure.[Stat Fcst Moving Average Planning Cycle], Measure.[Stat Fcst Naive Random Walk Planning Cycle], Measure.[Stat Fcst Prophet Planning Cycle], Measure.[Stat Fcst SCHM Planning Cycle], Measure.[Stat Fcst SES Planning Cycle], Measure.[Stat Fcst STLF Planning Cycle], Measure.[Stat Fcst Seasonal Naive YoY Planning Cycle], Measure.[Stat Fcst Simple AOA Planning Cycle], Measure.[Stat Fcst Simple Snaive Planning Cycle], Measure.[Stat Fcst TBATS Planning Cycle], Measure.[Stat Fcst TES Planning Cycle], Measure.[Stat Fcst Theta Planning Cycle], Measure.[Stat Fcst Weighted AOA Planning Cycle], Measure.[Stat Fcst Weighted Snaive Planning Cycle], Measure.[Stat Fcst sARIMA Planning Cycle], Measure.[System Stat Fcst L1 Planning Cycle], Measure.[Slice Association Stat]}) on column where {(Measure.[Slice Association Stat] == 1)};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    AssignedAlgoList :  Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Item].[Stat Item] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] ) on row, ({Measure.[Assigned Algorithm List], Measure.[Planner Bestfit Algorithm], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[System Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Assigned Algorithm List]), (Measure.[Slice Association Stat] == 1)};

    Weights: Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Item].[Segmentation LOB] ) on row,({Measure.[Straight Line Weight],Measure.[Trend Weight],Measure.[Seasonality Weight],Measure.[Level Weight],Measure.[Range Weight], Measure.[COCC Weight]}) on column;

    SelectionCriteria: Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,({Measure.[Bestfit Selection Criteria]}) on column;

    Violations: Select ([Sequence].[Sequence] * [Version].[Version Name] *  [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Item].[Segmentation LOB] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Stat Algorithm].[Stat Algorithm] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Stat Rule].[Stat Rule] ) on row, ({Measure.[Straight Line], Measure.[Trend Violation], Measure.[Level Violation], Measure.[Seasonal Violation], Measure.[Range Violation], Measure.[COCC Violation],  Measure.[Run Count], Measure.[No Alerts], Measure.[Validation Fcst], Measure.[Run Time], Measure.[Algorithm Parameters],Measure.[Is Bestfit], Measure.[Fcst Next N Buckets], Measure.[Validation Error], Measure.[Validation Method], Measure.[Composite Error], Measure.[Validation Actual], Measure.[Validation Fcst Abs Error], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Validation Fcst]), (Measure.[Slice Association Stat] == 1)};

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

    ForecastEngine : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Engine]}) on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

Output Variables:
    BestFitAlgo
    ValidationError

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP015IdentifyBestFitandPopulateBestfit import main

logger = logging.getLogger("o9_logger")

# Function Calls
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
Actuals = O9DataLake.get("Actuals")
ValidationForecastData = O9DataLake.get("ValidationForecastData")
ForecastData = O9DataLake.get("ForecastData")
ForecastBounds = O9DataLake.get("ForecastBounds")
PlannerBestfitAlgo = O9DataLake.get("PlannerBestfitAlgo")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
AssignedAlgoList = O9DataLake.get("AssignedAlgoList")
Weights = O9DataLake.get("Weights")
Violations = O9DataLake.get("Violations")
SelectionCriteria = O9DataLake.get("SelectionCriteria")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
ForecastEngine = O9DataLake.get("ForecastEngine")
SellOutOffset = O9DataLake.get("SellOutOffset")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
ValidationParameters = O9DataLake.get("ValidationParameters")
PlanningCycleDates = O9DataLake.get("PlanningCycleDates")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

(
    BestFitAlgo,
    ValidationError,
    BestFitForecast,
    BestFitAlgorithmCandidateOutput,
    BestFitViolationOutput,
    ValidationErrorPlanningCycle,
    ValidationForecast,
) = main(
    Grains=Grains,
    HistoryMeasure=HistoryMeasure,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    Actuals=Actuals,
    ValidationForecastData=ValidationForecastData,
    ValidationParameters=ValidationParameters,
    PlanningCycleDates=PlanningCycleDates,
    ForecastData=ForecastData,
    ForecastBounds=ForecastBounds,
    PlannerBestfitAlgo=PlannerBestfitAlgo,
    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    AssignedAlgoList=AssignedAlgoList,
    Weights=Weights,
    Violations=Violations,
    SelectionCriteria=SelectionCriteria,
    ForecastEngine=ForecastEngine,
    SellOutOffset=SellOutOffset,
    MasterAlgoList=MasterAlgoList,
    StatBucketWeight=StatBucketWeight,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    df_keys=df_keys,
)


O9DataLake.put("BestFitAlgo", BestFitAlgo)
O9DataLake.put("ValidationError", ValidationError)
O9DataLake.put("BestFitForecast", BestFitForecast)
O9DataLake.put("BestFitAlgorithmCandidateOutput", BestFitAlgorithmCandidateOutput)
O9DataLake.put("BestFitViolationOutput", BestFitViolationOutput)
O9DataLake.put("ValidationErrorPlanningCycle", ValidationErrorPlanningCycle)
O9DataLake.put("ValidationForecast", ValidationForecast)

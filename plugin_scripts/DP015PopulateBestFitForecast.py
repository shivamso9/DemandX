"""
    Plugin : DP015PopulateBestFitForecast
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive - False
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]

Input Queries:
    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({  Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets]  }) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastData : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * &CurrentAndFuturePartialWeeks ) on row,  ({Measure.[Stat Fcst AR-NNET], Measure.[Stat Fcst Auto ARIMA], Measure.[Stat Fcst Croston], Measure.[Stat Fcst DES], Measure.[Stat Fcst ETS], Measure.[Stat Fcst Moving Average], Measure.[Stat Fcst Naive Random Walk], Measure.[Stat Fcst Prophet], Measure.[Stat Fcst SES], Measure.[Stat Fcst STLF], Measure.[Stat Fcst Seasonal Naive YoY], Measure.[Stat Fcst TBATS], Measure.[Stat Fcst TES], Measure.[Stat Fcst Theta], Measure.[Stat Fcst sARIMA], Measure.[Stat Fcst Simple Snaive], Measure.[Stat Fcst Weighted Snaive], Measure.[Stat Fcst Growth Snaive], Measure.[Stat Fcst Simple AOA], Measure.[Stat Fcst Weighted AOA], Measure.[Stat Fcst Growth AOA],Measure.[Stat Fcst SCHM],Measure.[Stat Fcst Ensemble],Measure.[Stat Fcst CML], Measure.[Stat Bucket Weight], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Stat Bucket Weight]), (Measure.[Slice Association Stat] == 1)};  

    ForecastBounds : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * &CurrentAndFuturePartialWeeks ) on row,  ({Measure.[Stat Fcst AR-NNET 80% LB], Measure.[Stat Fcst AR-NNET 80% UB], Measure.[Stat Fcst Auto ARIMA 80% LB], Measure.[Stat Fcst Auto ARIMA 80% UB], Measure.[Stat Fcst Croston 80% LB], Measure.[Stat Fcst Croston 80% UB], Measure.[Stat Fcst DES 80% LB], Measure.[Stat Fcst DES 80% UB], Measure.[Stat Fcst ETS 80% LB], Measure.[Stat Fcst ETS 80% UB], Measure.[Stat Fcst Moving Average 80% LB], Measure.[Stat Fcst Moving Average 80% UB], Measure.[Stat Fcst Naive Random Walk 80% LB], Measure.[Stat Fcst Naive Random Walk 80% UB], Measure.[Stat Fcst Prophet 80% LB], Measure.[Stat Fcst Prophet 80% UB], Measure.[Stat Fcst STLF 80% LB], Measure.[Stat Fcst STLF 80% UB], Measure.[Stat Fcst Seasonal Naive YoY 80% LB], Measure.[Stat Fcst Seasonal Naive YoY 80% UB], Measure.[Stat Fcst TBATS 80% LB], Measure.[Stat Fcst TBATS 80% UB], Measure.[Stat Fcst TES 80% LB], Measure.[Stat Fcst TES 80% UB], Measure.[Stat Fcst Theta 80% LB], Measure.[Stat Fcst Theta 80% UB], Measure.[Stat Fcst sARIMA 80% LB], Measure.[Stat Fcst sARIMA 80% UB], Measure.[Stat Fcst SES 80% LB], Measure.[Stat Fcst SES 80% UB], Measure.[Stat Fcst Simple Snaive 80% LB], Measure.[Stat Fcst Simple Snaive 80% UB], Measure.[Stat Fcst Weighted Snaive 80% LB], Measure.[Stat Fcst Weighted Snaive 80% UB], Measure.[Stat Fcst Growth Snaive 80% LB], Measure.[Stat Fcst Growth Snaive 80% UB], Measure.[Stat Fcst Simple AOA 80% LB], Measure.[Stat Fcst Simple AOA 80% UB], Measure.[Stat Fcst Weighted AOA 80% LB], Measure.[Stat Fcst Weighted AOA 80% UB], Measure.[Stat Fcst Growth AOA 80% LB], Measure.[Stat Fcst Growth AOA 80% UB],Measure.[Stat Fcst SCHM 80% LB],Measure.[Stat Fcst SCHM 80% UB], Measure.[Stat Bucket Weight], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Stat Bucket Weight]), (Measure.[Slice Association Stat] == 1)};

    BestFitAlgo : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] ) on row,  ({Measure.[System Bestfit Algorithm Final], Measure.[Planner Bestfit Algorithm], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[System Bestfit Algorithm Final]), (Measure.[Slice Association Stat] == 1)};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    Violations : Select ([Sequence].[Sequence] * [Version].[Version Name] *  [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Stat Algorithm].[Stat Algorithm] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Stat Rule].[Stat Rule] ) on row, ({Measure.[Straight Line], Measure.[Trend Violation], Measure.[Level Violation], Measure.[Seasonal Violation], Measure.[Range Violation], Measure.[COCC Violation],  Measure.[Run Count], Measure.[No Alerts], Measure.[Validation Fcst], Measure.[Run Time], Measure.[Algorithm Parameters],Measure.[Is Bestfit], Measure.[Fcst Next N Buckets], Measure.[Validation Error], Measure.[Validation Method], Measure.[Composite Error], Measure.[Validation Actual], Measure.[Validation Fcst Abs Error], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Validation Fcst]), (Measure.[Slice Association Stat] == 1)};

Output Variables:
    BestFitForecast
    BestFitAlgorithmCandidateOutput
    BestFitViolationOutput

Slice Dimension Attributes:
    Sequence.[Sequence]

"""
import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP015PopulateBestFitForecast import main

logger = logging.getLogger("o9_logger")

# Function Calls
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastData = O9DataLake.get("ForecastData")
ForecastBounds = O9DataLake.get("ForecastBounds")
BestFitAlgo = O9DataLake.get("BestFitAlgo")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
Violations = O9DataLake.get("Violations")

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
    BestFitForecast,
    BestFitAlgorithmCandidateOutput,
    BestFitViolationOutput,
) = main(
    Grains=Grains,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastData=ForecastData,
    ForecastBounds=ForecastBounds,
    BestFitAlgo=BestFitAlgo,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    Violations=Violations,
    df_keys=df_keys,
)

O9DataLake.put("BestFitForecast", BestFitForecast)
O9DataLake.put(
    "BestFitAlgorithmCandidateOutput", BestFitAlgorithmCandidateOutput
)
O9DataLake.put("BestFitViolationOutput", BestFitViolationOutput)

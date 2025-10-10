"""
    Plugin : DP056OutputChecks
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

    Script Params:
        TrendVariationThreshold - 0.6
        LevelVariationThreshold - 0.6
        RangeVariationThreshold - 0.6
        SeasonalVariationThreshold - 0.6
        SeasonalVariationCountThreshold - 0.6
        MinimumIndicePercentage - 0.01
        AbsTolerance - 50
        Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
        ReasonabilityCycles - 1
        COCCVariationThreshold - 0.25

    Input Queries:
        Actual :  Select ([Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] * [Sequence].[Sequence]) on row, ({Measure.[Slice Association Stat], Measure.[Actual Cleansed], Measure.[Stat Actual]}) on column where {~isnull(Measure.[Actual Cleansed]), (Measure.[Slice Association Stat] == 1)};
        ForecastData :  Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Item].[Stat Item] ) on row, ({Measure.[Planner Stat Fcst L1], Measure.[Stat Fcst AR-NNET], Measure.[Stat Fcst Auto ARIMA], Measure.[Stat Fcst Croston], Measure.[Stat Fcst DES], Measure.[Stat Fcst ETS], Measure.[Stat Fcst Growth AOA], Measure.[Stat Fcst Growth Snaive], Measure.[Stat Fcst Moving Average], Measure.[Stat Fcst Naive Random Walk], Measure.[Stat Fcst Prophet], Measure.[Stat Fcst SES], Measure.[Stat Fcst STLF], Measure.[Stat Fcst Seasonal Naive YoY], Measure.[Stat Fcst Simple AOA], Measure.[Stat Fcst Simple Snaive], Measure.[Stat Fcst TBATS], Measure.[Stat Fcst TES], Measure.[Stat Fcst Theta], Measure.[Stat Fcst Weighted AOA], Measure.[Stat Fcst Weighted Snaive], Measure.[Stat Fcst sARIMA],Measure.[Stat Fcst Ensemble],Measure.[Stat Fcst CML],Measure.[Stat Fcst SCHM] ,Measure.[Stat Fcst ML L1], Measure.[Stat Fcst L1], Measure.[Slice Association Stat], Measure.[Stat Bucket Weight], Measure.[Stat Fcst L1 LC]}) on column where {~isnull(Measure.[Stat Bucket Weight]), (Measure.[Slice Association Stat] == 1)};
        SegmentationOutput : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row, ({Measure.[Intermittent L1], Measure.[PLC Status L1], Measure.[Seasonality L1], Measure.[Length of Series L1], Measure.[Bestfit Algorithm], Measure.[Planner Bestfit Algorithm], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[System Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[PLC Status L1]), (Measure.[Slice Association Stat] == 1)};
        TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};
        CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};
        ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket], Measure.[Forecast Storage Time Bucket]}) on column;
        ForecastSetupConfiguration: Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Forecast Period],Measure.[History Period]}) on column;
        AlgoStats : Select ([Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Stat Algorithm].[Stat Algorithm] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Stat Rule].[Stat Rule] ) on row, ({Measure.[Algorithm Parameters], Measure.[Run Time], Measure.[Validation Error], Measure.[Composite Error], Measure.[Validation Method]}) on column;

        SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

    Output Variables:
        OutputAllAlgo
        OutputBestFit
        ActualLastNBuckets
        FcstNextNBuckets
        AlgoStatsForBestFitMembers

    Slice Dimension Attributes:
        Sequence
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP056OutputChecks import main

logger = logging.getLogger("o9_logger")


# Function Calls
# Get all input dataframes from o9DataLake
logger.debug("Reading Inputs.")
actual = O9DataLake.get("Actual")
forecastData = O9DataLake.get("ForecastData")
segmentationOutput = O9DataLake.get("SegmentationOutput")
timeDimension = O9DataLake.get("TimeDimension")
currentTimePeriod = O9DataLake.get("CurrentTimePeriod")
forecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
forecastSetupConfiguration = O9DataLake.get("ForecastSetupConfiguration")
AlgoStats = O9DataLake.get("AlgoStats")
SellOutOffset = O9DataLake.get("SellOutOffset")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.debug("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.debug("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.debug("Starting background thread for memory profiling ...")
back_thread.start()

(
    OutputAllAlgo,
    OutputBestFit,
    ActualLastNBuckets,
    FcstNextNBuckets,
    AlgoStatsForBestFitMembers,
) = main(
    Actual=actual,
    ForecastData=forecastData,
    SegmentationOutput=segmentationOutput,
    TimeDimension=timeDimension,
    CurrentTimePeriod=currentTimePeriod,
    ForecastGenTimeBucket=forecastGenTimeBucket,
    ForecastSetupConfiguration=forecastSetupConfiguration,
    AlgoStats=AlgoStats,
    SellOutOffset=SellOutOffset,
    TrendVariationThreshold=float(TrendVariationThreshold),
    LevelVariationThreshold=float(LevelVariationThreshold),
    RangeVariationThreshold=float(RangeVariationThreshold),
    SeasonalVariationThreshold=float(SeasonalVariationThreshold),
    SeasonalVariationCountThreshold=float(SeasonalVariationCountThreshold),
    ReasonabilityCycles=float(ReasonabilityCycles),
    MinimumIndicePercentage=float(MinimumIndicePercentage),
    AbsTolerance=float(AbsTolerance),
    Grains=Grains,
    df_keys=df_keys,
    COCCVariationThreshold=float(COCCVariationThreshold),
)

O9DataLake.put("OutputAllAlgo", OutputAllAlgo)
O9DataLake.put("OutputBestFit", OutputBestFit)
O9DataLake.put("ActualLastNBuckets", ActualLastNBuckets)
O9DataLake.put("FcstNextNBuckets", FcstNextNBuckets)
O9DataLake.put("AlgoStatsForBestFitMembers", AlgoStatsForBestFitMembers)

"""
Plugin : DP037SeasonalityDetection
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    ReadFromHive - False
    AlphaACF - 0.05
    MultiprocessingNumCores - 4
    ACFLowerThreshold - -0.10
    ACFUpperThreshold - 0.90
    ACFSkipLags - 11
    ACFDiff - 0
    RequiredACFLagsInWeeks - 12
    TrendThreshold - 20

Input Queries:
    ActualData : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Item].[Stat Item] * [Item].[Segmentation LOB] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Time].[Partial Week]) on row, ({Measure.[Stat Actual], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Stat Actual]), (Measure.[Slice Association Stat] == 1)};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Parameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[History Measure], Measure.[History Period]}) on column;

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    SegmentationOutput : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row, ({Measure.[Intermittent L1], Measure.[Length of Series L1], Measure.[PLC Status L1], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[PLC Status L1]), (Measure.[Slice Association Stat] == 1)};

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

Output Variables:
    SeasonalDetectionData

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP037SeasonalityDetection import main

logger = logging.getLogger("o9_logger")

# Function Calls
ActualData = O9DataLake.get("ActualData")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
Parameters = O9DataLake.get("Parameters")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
SegmentationOutput = O9DataLake.get("SegmentationOutput")
SellOutOffset = O9DataLake.get("SellOutOffset")

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

SeasonalDetectionData = main(
    Grains=Grains,
    Actual=ActualData,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    Parameters=Parameters,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    SegmentationOutput=SegmentationOutput,
    SellOutOffset=SellOutOffset,
    ReadFromHive=ReadFromHive,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    alpha=AlphaACF,
    lower_ci_threshold=float(ACFLowerThreshold),
    upper_ci_threshold=float(ACFUpperThreshold),
    skip_lags=int(ACFSkipLags),
    diff=int(ACFDiff),
    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
    TrendThreshold=int(TrendThreshold),
    df_keys=df_keys,
)

O9DataLake.put("SeasonalDetectionData", SeasonalDetectionData)

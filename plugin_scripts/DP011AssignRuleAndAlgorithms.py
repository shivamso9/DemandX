"""
Plugin : DP011AssignRuleAndAlgorithms
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive - False
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]

Input Queries:
    SegmentedData : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row, ({Measure.[Volume Segment L1], Measure.[COV Segment L1], Measure.[Trend L1], Measure.[Seasonality L1], Measure.[Intermittent L1], Measure.[PLC Status L1], Measure.[Length of Series L1], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Length of Series L1]), (Measure.[Slice Association Stat] == 1)};

    Rules : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Stat Rule].[Stat Rule] *{Measure.[Planner Volume Segment],Measure.[Planner COV Segment],Measure.[Planner Intermittency],Measure.[Planner PLC],Measure.[Planner Trend],Measure.[Planner Seasonality],Measure.[Planner Length of Series], Measure.[Planner Algorithm List]} ) on row,  () on column include memberproperties {[Stat Rule].[Stat Rule],[System Rule Description],[System Algorithm List]};

    ForecastGenTimeBucket : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

Output Variables:
    ForecastRule

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP011AssignRuleAndAlgorithms import main

logger = logging.getLogger("o9_logger")


# Function Calls
SegmentedData = O9DataLake.get("SegmentedData")
Rules = O9DataLake.get("Rules")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")

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

ForecastRule = main(
    Grains=Grains,
    SegmentedData=SegmentedData,
    Rules=Rules,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    df_keys=df_keys,
)
O9DataLake.put("ForecastRule", ForecastRule)

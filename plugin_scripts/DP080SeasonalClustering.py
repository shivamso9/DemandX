"""
Plugin : DP080SeasonalClustering
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    IncludeDiscIntersections - False
    item_level_to_select - Item.[L6]
    weights - 0.1, 0.8, 0.1
    min_data_points_for_clustering - 24
    distance_threshold - 0.8
    normalize_clustering_input - False
    number_of_clusters - sqrt
    req_data_points_for_hl_seasonality - 1
    minimum_points_for_assigning_seasonality - 8
    min_ts_for_seasonality_borrowing - 5
    MultiprocessingNumCores - 4

Input Queries:

    Actual : Select ([Sequence].[Sequence]* [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] ) on row,  ({Measure.[Actual Cleansed], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Actual Cleansed]), (Measure.[Slice Association Stat] == 1)};

    ItemMasterData : Select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[Stat Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6]) on row,  () on column;

    ForecastParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({  Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets]  }) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    SegmentationOutput : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row, ({Measure.[Intermittent L1], Measure.[PLC Status L1], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[PLC Status L1]), (Measure.[Slice Association Stat] == 1)}

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    StatLevel :  Select ([Forecast Iteration].[Forecast Iteration]* [Version].[Version Name]) on row,  ({Measure.[Item Level]}) on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;


Output Variables:
    ClusterOutput

Slice Dimension Attributes:

Pseudocode :


"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

from helpers.DP080SeasonalClustering import main

# Function Calls
Actual = O9DataLake.get("Actual")
ItemMasterData = O9DataLake.get("ItemMasterData")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
SegmentationOutput = O9DataLake.get("SegmentationOutput")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
StatLevel = O9DataLake.get("StatLevel")
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

ClusterOutput = main(
    Grains=Grains,
    min_data_points_for_clustering=int(min_data_points_for_clustering),
    min_ts_for_seasonality_borrowing=int(min_ts_for_seasonality_borrowing),
    minimum_points_for_assigning_seasonality=int(minimum_points_for_assigning_seasonality),
    normalize_clustering_input=normalize_clustering_input,
    number_of_clusters=number_of_clusters,
    req_data_points_for_hl_seasonality=int(req_data_points_for_hl_seasonality),
    weights=weights,
    distance_threshold=float(distance_threshold),
    item_level_to_select=item_level_to_select,
    IncludeDiscIntersections=IncludeDiscIntersections,
    Actual=Actual,
    ItemMasterData=ItemMasterData,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    SegmentationOutput=SegmentationOutput,
    StatBucketWeight=StatBucketWeight,
    StatLevel=StatLevel,
    SellOutOffset=SellOutOffset,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    df_keys=df_keys,
)

O9DataLake.put("ClusterOutput", ClusterOutput)

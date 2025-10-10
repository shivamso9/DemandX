"""
Plugin : DP005StatSegment
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    DimClass - Class.[Class]
    VolSegLvl - A,B
    COVSegLvl - X,Y
    ReadFromHive - False
    Grains - Item.[Planning Item]
    SegmentationVolumeCalcGrain - Item.[Segmentation LOB]
    VolumeThreshold - 0.8
    COVThreshold - 0.5
    IntermittencyThreshold - 0.5
    NewLaunchPeriodInWeeks - 52
    DiscoPeriodInWeeks - 52
    HistoryMeasure - Actual
    VolumeCOVHistoryPeriodInWeeks - 52
    HistoryPeriodInWeeks - 156
    ForecastGenerationTimeBucket - Week
    AlphaACF - 0.05
    ACFLowerThreshold - -0.10
    ACFUpperThreshold - 0.90
    ACFSkipLags - 11
    ACFDiff - 0
    RequiredACFLags - 12
    useACFForSeasonality - True
    OffsetPeriods - 0
    TrendThreshold - 20


Input Queries:
    Actual : Select ([Time].[Partial Week] * [Version].[Version Name] * [Item].[Planning Item]) on row, ({Measure.[Actual], Measure.[Billing],Measure.[Orders],Measure.[Shipments], Measure.[Assortment Stat]}) on column where {Measure.[Assortment Stat]>0};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    SegmentationLOBGroupFilter : Select ([Version].[Version Name]) on row,  ({Measure.[Segmentation LOB Group Filter]}) on column;

    ItemMasterData : select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[All Item]) on row, () on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

Output Variables:
    StatSegmentation
    ProductSegmentation
    ItemClass

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP005StatSegment import main

logger = logging.getLogger("o9_logger")

# Function Calls
Actual = O9DataLake.get("Actual")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
SegmentationLOBGroupFilter = O9DataLake.get("SegmentationLOBGroupFilter")
ItemMasterData = O9DataLake.get("ItemMasterData")

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


StatSegmentation, ProductSegmentation, ItemClass = main(
    COVSegLvl=COVSegLvl,
    Actual=Actual,
    Grains=Grains,
    DimClass=DimClass,
    TimeDimension=TimeDimension,
    VolSegLvl=VolSegLvl,
    CurrentTimePeriod=CurrentTimePeriod,
    ReadFromHive=ReadFromHive,
    SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
    df_keys=df_keys,
    VolumeThreshold=VolumeThreshold,
    COVThreshold=COVThreshold,
    IntermittencyThreshold=IntermittencyThreshold,
    NewLaunchPeriodInWeeks=NewLaunchPeriodInWeeks,
    DiscoPeriodInWeeks=DiscoPeriodInWeeks,
    HistoryMeasure=HistoryMeasure,
    VolumeCOVHistoryPeriodInWeeks=VolumeCOVHistoryPeriodInWeeks,
    HistoryPeriodInWeeks=HistoryPeriodInWeeks,
    ForecastGenerationTimeBucket=ForecastGenerationTimeBucket,
    SegmentationLOBGroupFilter=SegmentationLOBGroupFilter,
    ItemMasterData=ItemMasterData,
    AlphaACF=AlphaACF,
    ACFLowerThreshold=ACFLowerThreshold,
    ACFUpperThreshold=ACFUpperThreshold,
    ACFSkipLags=ACFSkipLags,
    ACFDiff=ACFDiff,
    RequiredACFLags=RequiredACFLags,
    useACFForSeasonality=useACFForSeasonality,
    OffsetPeriods=int(OffsetPeriods),
    TrendThreshold=int(TrendThreshold),
)
O9DataLake.put("StatSegmentation", StatSegmentation)
O9DataLake.put("ProductSegmentation", ProductSegmentation)
O9DataLake.put("ItemClass", ItemClass)

"""
Plugin : DP016TransitionLevelStat
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Transition Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Transition Demand Domain],Location.[Planning Location]
    UseMovingAverage - True
    MultiprocessingNumCores - 4
    StatGrains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    HistoryPeriodsInWeeks - 13,52,104,156
    InputTables - History,SellOutActual
    DefaultMapping - Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL
Input Queries:
    History : Select ([Sequence].[Sequence] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain] * [Location].[Location] * [Location].[Planning Location] * [Version].[Version Name] * [Item].[Planning Item] * [Item].[Transition Item] * [Time].[Partial Week]) on row,  ({Measure.[Actual], Measure.[Slice Association TL]}) on column where {~isnull(Measure.[Actual]), (Measure.[Slice Association TL] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Time].[Partial Week] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    ForecastParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[History Period], Measure.[Forecast Period]}) on column;

    DisaggregationType : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Disaggregation Type]}) on column;

    StatFcstL1ForFIPLIteration : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Forecast Iteration].[Forecast Iteration].[FI-PL] ) on row, ({Measure.[Stat Fcst L1]}) on column;

    ForecastIterationSelectionAtTransitionLevel : Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Transition Demand Domain] * [Location].[Location] * [Version].[Version Name] * [Item].[Transition Item]) on row,  ({Measure.[Forecast Iteration Selection]}) on column;

    StatActual : Select ([Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] ) on row, ({Measure.[Stat Actual]}) on column;

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level], Measure.[Location Level]}) on column;

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Transition Demand Domain] *  [Demand Domain].[Planning Demand Domain]) on row, () on column;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column;

    TItemDates : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Transition Demand Domain] * [Item].[Transition Item]) on row, ({ Measure.[Disco Date],Measure.[Intro Date]}) on column;

    ForecastIterationMasterData : Select (Version.[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    SellOutActual :  Select ([Sequence].[Sequence] *[Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] *  [Item].[Transition Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual], Measure.[Slice Association TL]}) on column where {(Measure.[Slice Association TL] == 1)};

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;


Output Variables:
    output

Slice Dimension Attributes:
    Sequence.[Sequence]
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.dataframe_utils import convert_category_to_str
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP016TransitionLevelStat import main

logger = logging.getLogger("o9_logger")

# Function Calls
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
ForecastParameters = O9DataLake.get("ForecastParameters")
DisaggregationType = O9DataLake.get("DisaggregationType")
StatFcstL1ForFIPLIteration = O9DataLake.get("StatFcstL1ForFIPLIteration")
ForecastIterationSelectionAtTransitionLevel = O9DataLake.get(
    "ForecastIterationSelectionAtTransitionLevel"
)
StatActual = O9DataLake.get("StatActual")
ForecastLevelData = O9DataLake.get("ForecastLevelData")
ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
TItemDates = O9DataLake.get("TItemDates")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
SellOutOffset = O9DataLake.get("SellOutOffset")

DefaultMapping_list = DefaultMapping.strip().split(",")
default_mapping = {}
for mapping in DefaultMapping_list:
    key, value = mapping.split("-")
    default_mapping[key] = value

# configurable inputs
try:
    CombinedActual = {}
    AllInputTables = InputTables.strip().split(",")
    for i in AllInputTables:
        Input = O9DataLake.get(i)
        if Input is None:
            logger.exception(f"Input '{i}' is None, please recheck the table name...")
            continue
        Input = convert_category_to_str(Input)
        CombinedActual[i] = Input
        logger.warning(f"Reading input {i}...")
except Exception as e:
    logger.exception(f"Error while reading inputs\nError:-\n{e}")

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

output = main(
    History=CombinedActual,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    Grains=Grains,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    DisaggregationType=DisaggregationType,
    StatFcstL1ForFIPLIteration=StatFcstL1ForFIPLIteration,
    ForecastIterationSelectionAtTransitionLevel=ForecastIterationSelectionAtTransitionLevel,
    StatActual=StatActual,
    ForecastLevelData=ForecastLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    TItemDates=TItemDates,
    StatGrains=StatGrains,
    HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
    UseMovingAverage=UseMovingAverage,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    ForecastIterationMasterData=ForecastIterationMasterData,
    SellOutOffset=SellOutOffset,
    default_mapping=default_mapping,
    df_keys=df_keys,
)
O9DataLake.put("output", output)

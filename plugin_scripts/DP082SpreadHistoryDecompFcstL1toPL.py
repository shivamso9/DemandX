"""
Plugin : DP082SpreadHistoryDecompFcstL1toPL
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    StatGrains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    HistoryWindowinWeeks - 52

Input Queries:
    Actual : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Time].[Partial Week] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain]  * [Location].[Planning Location] * [Item].[Planning Item] ) on row, ({Measure.[Actual], Measure.[Slice Association PL]}) on column where {~isnull(Measure.[Actual]), (Measure.[Slice Association PL] == 1)};

    PItemDates : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Item].[Planning Item]) on row, ({Measure.[Intro Date], Measure.[Disco Date], Measure.[Stat History Start Date]}) on column;

    StatActual : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Account].[Stat Account] ) on row,  ({Measure.[Stat Actual]}) on column;

    CMLFcstL1 : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Region].[Stat Region] * [Location].[Stat Location] * [Time].[Partial Week] * [Item].[Stat Item] ) on row, ({Measure.[ML Fcst L1 CML Baseline], Measure.[ML Fcst L1 CML Holiday], Measure.[ML Fcst L1 CML Marketing], Measure.[ML Fcst L1 CML Price], Measure.[ML Fcst L1 CML Promo], Measure.[ML Fcst L1 CML Residual], Measure.[ML Fcst L1 CML Weather], Measure.[ML Fcst L1 CML External Driver]}) on column;

    TimeDimension : select ([Time].[Day] *[Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Day], Key}{Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentDay * Time.[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level], Measure.[Location Level]}) on column;

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Transition Demand Domain] *  [Demand Domain].[Planning Demand Domain]) on row, () on column;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column;

Output Variables:
    StatFcstPLOutput

Slice Dimension Attributes: None
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP082SpreadHistoryDecompFcstL1toPL import main

logger = logging.getLogger("o9_logger")

import threading

from o9_common_utils.O9DataLake import O9DataLake

# Function Calls
Actual = O9DataLake.get("Actual")
PItemDates = O9DataLake.get("PItemDates")
StatActual = O9DataLake.get("StatActual")
CMLFcstL1 = O9DataLake.get("CMLFcstL1")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
ForecastLevelData = O9DataLake.get("ForecastLevelData")
ItemMasterData = O9DataLake.get("ItemMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")


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

StatFcstPLOutput = main(
    Actual=Actual,
    PItemDates=PItemDates,
    StatActual=StatActual,
    CMLFcstL1=CMLFcstL1,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    ForecastLevelData=ForecastLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    LocationMasterData=LocationMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    StatGrains=StatGrains,
    PlanningGrains=PlanningGrains,
    HistoryWindowinWeeks=HistoryWindowinWeeks,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstPLOutput", StatFcstPLOutput)

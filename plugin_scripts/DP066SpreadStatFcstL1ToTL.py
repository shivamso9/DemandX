"""
Plugin : DP066SpreadStatFcstL1ToTL
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Region.[Planning Region],Item.[Transition Item],PnL.[Planning PnL],Location.[Planning Location],Demand Domain.[Transition Demand Domain],Account.[Planning Account],Channel.[Planning Channel]
    DefaultMapping - Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL

Input Queries:
    StatFcstFinalProfileTL : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Transition Item] * [PnL].[Planning PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Planning Location] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row,  ({Measure.[Stat Fcst Final Profile TL]}) on column;

    StatFcstL1 : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] ) on row,  ({Measure.[Stat Fcst L1]}) on column;

    CMLFcstL1 : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Region].[Stat Region] * [Location].[Stat Location] * [Time].[Partial Week] * [Item].[Stat Item] ) on row, ({Measure.[ML Fcst L1 CML Baseline], Measure.[ML Fcst L1 CML Holiday], Measure.[ML Fcst L1 CML Marketing], Measure.[ML Fcst L1 CML Price], Measure.[ML Fcst L1 CML Promo], Measure.[ML Fcst L1 CML Residual], Measure.[ML Fcst L1 CML Weather], Measure.[ML Fcst L1 CML External Driver]}) on column;

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level], Measure.[Location Level]}) on column;

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]) on row, () on column;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column;

    MLDecompositionFlag : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({Measure.[CML Iteration Decomposition]}) on column;

    ItemDates: Select ([Version].[Version] * [Region].[Stat Region] * [Location].[Stat Location]  * [Channel].[Stat Channel] * [PnL].[Stat PnL]  * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, 
 ({Measure.[Intro Date], Measure.[Disco Date]}) on column;
                
    CurrentDay: select (&CurrentPartialWeek * [Time].[Week]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key};

Output Variables:
    Output
    CMLOutput
    Output_volume_loss_flag

Slice Dimension Attributes : None
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

import threading

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP066SpreadStatFcstL1ToTL import main

# Function Calls
StatFcstFinalProfileTL = O9DataLake.get("StatFcstFinalProfileTL")
StatFcstL1 = O9DataLake.get("StatFcstL1")
CMLFcstL1 = O9DataLake.get("CMLFcstL1")
ForecastLevelData = O9DataLake.get("ForecastLevelData")
ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
MLDecompositionFlag = O9DataLake.get("MLDecompositionFlag")
ItemDates = O9DataLake.get("ItemDates")
CurrentDay = O9DataLake.get("CurrentDay")
ForecastGenerationTimeBucket = O9DataLake.get("ForecastGenerationTimeBucket")


DefaultMapping_list = DefaultMapping.strip().split(",")
default_mapping = {}
for mapping in DefaultMapping_list:
    key, value = mapping.split("-")
    default_mapping[key] = value


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

Output, CMLOutput, Output_volume_loss_flag = main(
    Grains=Grains,
    StatFcstFinalProfileTL=StatFcstFinalProfileTL,
    StatFcstL1=StatFcstL1,
    CMLFcstL1=CMLFcstL1,
    ForecastLevelData=ForecastLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    ItemDates=ItemDates,
    CurrentDay=CurrentDay,
    MLDecompositionFlag=MLDecompositionFlag,
    default_mapping=default_mapping,
    ForecastGenerationTimeBucket=ForecastGenerationTimeBucket,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
O9DataLake.put("CMLOutput", CMLOutput)
O9DataLake.put("Output_volume_loss_flag", Output_volume_loss_flag)
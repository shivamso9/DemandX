"""

Plugin : DP062CreateStatMembers
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:None

Input Queries:
    ItemMasterData :  select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column include memberproperties {[Item].[Planning Item], DisplayName} {[Item].[L1], DisplayName} {[Item].[L2], DisplayName} {[Item].[L3], DisplayName} {[Item].[L4], DisplayName} {[Item].[L5], DisplayName} {[Item].[L6], DisplayName} {[Item].[Item Class], DisplayName} {[Item].[PLC Status], DisplayName} {[Item].[All Item], DisplayName} {[Item].[Item Type], DisplayName} {[Item].[Segmentation LOB], DisplayName};

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName};

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include memberproperties {[Account].[Account L1], DisplayName} {[Account].[Account L2], DisplayName} {[Account].[Account L3], DisplayName} {[Account].[Account L4], DisplayName} {[Account].[All Account], DisplayName} {[Account].[Planning Account], DisplayName};

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column include memberproperties {[Channel].[Channel L1], DisplayName} {[Channel].[Channel L2], DisplayName} {[Channel].[Planning Channel], DisplayName} {[Channel].[All Channel], DisplayName};

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include memberproperties {[PnL].[All PnL], DisplayName} {[PnL].[Planning PnL], DisplayName} {[PnL].[PnL L1], DisplayName} {[PnL].[PnL L2], DisplayName} {[PnL].[PnL L3], DisplayName} {[PnL].[PnL L4], DisplayName};

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName};

    LocationMasterData :  select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column include memberproperties {[Location].[All Location], DisplayName} {[Location].[Location Type], DisplayName} {[Location].[Location], DisplayName} {[Location].[Location Region], DisplayName} {[Location].[Location Country], DisplayName} {[Location].[Planning Location], DisplayName};

    ForecastItemLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row,  ({Measure.[Item Level]}) on column;

    ForecastRegionLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row,  ({Measure.[Region Level]}) on column;

    ForecastAccountLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row,  ({Measure.[Account Level]}) on column;

    ForecastChannelLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({Measure.[Channel Level]}) on column;

    ForecastPnLLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row,  ({Measure.[PnL Level]}) on column;

    ForecastDemandDomainLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({Measure.[Demand Domain Level]}) on column;

    ForecastLocationLevelData : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row,  ({Measure.[Location Level]}) on column;

    ForecastIterationSelectionData : Select ([Channel].[Planning Channel] * [Forecast Iteration].[Forecast Iteration Type] * [Demand Domain].[Planning Demand Domain] * [Version].[Version Name] * [Region].[Planning Region] * [Account].[Planning Account] * [PnL].[Planning PnL] * [Location].[Location] * [Item].[Planning Item] ) on row,  ({Measure.[Forecast Iteration Selection]}) on column;

    StatItem : Select ([Item].[Stat Item]);

    StatRegion : Select ([Region].[Stat Region]);

    StatAccount : Select ([Account].[Stat Account]);

    StatChannel : Select ([Channel].[Stat Channel]);

    StatPnL : Select ([PnL].[Stat PnL]);

    StatDemandDomain : Select ([Demand Domain].[Stat Demand Domain]);

    StatLocation : Select ([Location].[Stat Location]);

    GroupFilter : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Stat Item Group Filter], Measure.[Stat Region Group Filter],Measure.[Stat Account Group Filter],Measure.[Stat Channel Group Filter],Measure.[Stat PnL Group Filter],Measure.[Stat Demand Domain Group Filter],Measure.[Stat Location Group Filter],Measure.[Segmentation LOB Group Filter]}) on column;

Output Variables:
    ItemMasterDataOut
    RegionMasterDataOut
    ChannelMasterDataOut
    AccountMasterDataOut
    PnLMasterDataOut
    DemandDomainMasterDataOut
    LocationMasterDataOut

Slice Dimension Attributes: None
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import threading

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP062CreateStatMembers import main

# Function Calls
ForecastItemLevelData = O9DataLake.get("ForecastItemLevelData")
ForecastRegionLevelData = O9DataLake.get("ForecastRegionLevelData")
ForecastAccountLevelData = O9DataLake.get("ForecastAccountLevelData")
ForecastChannelLevelData = O9DataLake.get("ForecastChannelLevelData")
ForecastPnLLevelData = O9DataLake.get("ForecastPnLLevelData")
ForecastDemandDomainLevelData = O9DataLake.get("ForecastDemandDomainLevelData")
ForecastLocationLevelData = O9DataLake.get("ForecastLocationLevelData")

ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")

StatItem = O9DataLake.get("StatItem")
StatRegion = O9DataLake.get("StatRegion")
StatAccount = O9DataLake.get("StatAccount")
StatChannel = O9DataLake.get("StatChannel")
StatPnL = O9DataLake.get("StatPnL")
StatDemandDomain = O9DataLake.get("StatDemandDomain")
StatLocation = O9DataLake.get("StatLocation")

ForecastIterationSelectionData = O9DataLake.get("ForecastIterationSelectionData")
GroupFilter = O9DataLake.get("GroupFilter")

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
    ItemMasterDataOut,
    RegionMasterDataOut,
    AccountMasterDataOut,
    ChannelMasterDataOut,
    PnLMasterDataOut,
    DemandDomainMasterDataOut,
    LocationMasterDataOut,
) = main(
    ForecastItemLevelData=ForecastItemLevelData,
    ForecastRegionLevelData=ForecastRegionLevelData,
    ForecastAccountLevelData=ForecastAccountLevelData,
    ForecastChannelLevelData=ForecastChannelLevelData,
    ForecastPnLLevelData=ForecastPnLLevelData,
    ForecastDemandDomainLevelData=ForecastDemandDomainLevelData,
    ForecastLocationLevelData=ForecastLocationLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    StatItemData=StatItem,
    StatRegionData=StatRegion,
    StatAccountData=StatAccount,
    StatChannelData=StatChannel,
    StatPnLData=StatPnL,
    StatDemandDomainData=StatDemandDomain,
    StatLocationData=StatLocation,
    ForecastIterationSelectionData=ForecastIterationSelectionData,
    GroupFilter=GroupFilter,
    df_keys=df_keys,
)

O9DataLake.put("ItemMasterDataOut", ItemMasterDataOut)
O9DataLake.put("RegionMasterDataOut", RegionMasterDataOut)
O9DataLake.put("AccountMasterDataOut", AccountMasterDataOut)
O9DataLake.put("ChannelMasterDataOut", ChannelMasterDataOut)
O9DataLake.put("PnLMasterDataOut", PnLMasterDataOut)
O9DataLake.put("DemandDomainMasterDataOut", DemandDomainMasterDataOut)
O9DataLake.put("LocationMasterDataOut", LocationMasterDataOut)

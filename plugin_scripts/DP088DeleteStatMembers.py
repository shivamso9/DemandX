"""
Plugin : DP088DeleteStatMembers
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    SelectedIterations - FI-2

Input Queries:
    StatLevels : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[Region Level], Measure.[Location Level], Measure.[PnL Level], Measure.[Demand Domain Level]}) on column;

    AccountMasterDate : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account] *[Account].[Stat Account]) on row, () on column;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel]*[Channel].[Stat Channel] ) on row, () on column;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region] * [Region].[Stat Region])  on row, () on column;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] *[PnL].[Stat PnL]) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]*[Demand Domain].[Stat Demand Domain]) on row, () on column;

    ItemMasterData : select ([Item].[Planning Item] *  [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item]  * [Item].[Segmentation LOB] * [Item].[Stat Item]) on row, () on column;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location] * [Location].[Reporting Location] *[Location].[Stat Location]) on row, () on column;

    StatItem : Select([Item].[Stat Item]);

    StatChannel : Select([Channel].[Stat Channel]);

    StatRegion : Select([Region].[Stat Region]);

    StatLocation : Select([Location].[Stat Location]);

    StatAccount : Select([Account].[Stat Account]);

    StatPnL : Select([PnL].[Stat PnL]);

    StatDemandDomain : Select([Demand Domain].[Stat Demand Domain]);

Output Variables:
    AccountMembers
    ChannelMembers
    RegionMembers
    PnLMembers
    ItemMembers
    LocationMembers
    DemandDomainMembers

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

from helpers.DP088DeleteStatMembers import main

# Function Calls
StatLevels = O9DataLake.get("StatLevels")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
StatItem = O9DataLake.get("StatItem")
StatChannel = O9DataLake.get("StatChannel")
StatAccount = O9DataLake.get("StatAccount")
StatRegion = O9DataLake.get("StatRegion")
StatPnL = O9DataLake.get("StatPnL")
StatLocation = O9DataLake.get("StatLocation")
StatDemandDomain = O9DataLake.get("StatDemandDomain")

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
    ItemMembers,
    LocationMembers,
    ChannelMembers,
    RegionMembers,
    AccountMembers,
    PnLMembers,
    DemandDomainMembers,
) = main(
    SelectedIterations=SelectedIterations,
    StatLevels=StatLevels,
    ItemMasterData=ItemMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    LocationMasterData=LocationMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    StatPnL=StatPnL,
    StatItem=StatItem,
    StatRegion=StatRegion,
    StatLocation=StatLocation,
    StatDemandDomain=StatDemandDomain,
    StatAccount=StatAccount,
    StatChannel=StatChannel,
    df_keys=df_keys,
)

O9DataLake.put("ItemMembers", ItemMembers)
O9DataLake.put("LocationMembers", LocationMembers)
O9DataLake.put("ChannelMembers", ChannelMembers)
O9DataLake.put("RegionMembers", RegionMembers)
O9DataLake.put("AccountMembers", AccountMembers)
O9DataLake.put("PnLMembers", PnLMembers)
O9DataLake.put("DemandDomainMembers", DemandDomainMembers)

"""
    Plugin Information:
    --------------------
		Plugin : DP121CreateNPIMember
		Version : 0.0.0
		Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------

    Input Queries:
    --------------------
        LevelInput :  Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row, ({Measure.[Global NPI Account Level], Measure.[Global NPI Channel Level], Measure.[Global NPI Demand Domain Level], Measure.[Global NPI Item Level], Measure.[Global NPI Location Level], Measure.[Global NPI PnL Level], Measure.[Global NPI Region Level]}) on column;

		ItemMaster : select ([Item].[Item] * [Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6] * [Item].[All Item]) on row, () on column include memberproperties {Item.[Item], DisplayName} {Item.[Planning Item], DisplayName} {Item.[L1], DisplayName} {Item.[L2], DisplayName} {Item.[L3], DisplayName} {Item.[L4], DisplayName} {Item.[L5], DisplayName} {Item.[L6], DisplayName} {Item.[All Item], DisplayName}  include_nullmembers Where {&DPAllFinishedGoodsItemGroups};

		AccountMaster : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include memberproperties {[Account].[Account L1], DisplayName} {[Account].[Account L2], DisplayName} {[Account].[Account L3], DisplayName} {[Account].[Account L4], DisplayName} {[Account].[All Account], DisplayName} {[Account].[Planning Account], DisplayName} include_nullmembers;

		ChannelMaster : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column include memberproperties {[Channel].[Channel L1], DisplayName} {[Channel].[Channel L2], DisplayName} {[Channel].[Planning Channel], DisplayName} {[Channel].[All Channel], DisplayName} include_nullmembers;

		RegionMaster : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName}  {[Region].[Region L3], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName} include_nullmembers;

		PnLMaster : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include memberproperties {[PnL].[All PnL], DisplayName} {[PnL].[Planning PnL], DisplayName} {[PnL].[PnL L1], DisplayName} {[PnL].[PnL L2], DisplayName} {[PnL].[PnL L3], DisplayName} {[PnL].[PnL L4], DisplayName} include_nullmembers;

		DemandDomainMaster : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName} {[Demand Domain].[Transition Demand Domain], DisplayName} include_nullmembers;

		LocationMaster : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location] * [Location].[Reporting Location]) on row, () on column include memberproperties {Location.[Planning Location], DisplayName} {Location.[Location Country], DisplayName} {Location.[Location Region], DisplayName} {Location.[Location Type], DisplayName} {Location.[All Location], DisplayName} {Location.[Location], DisplayName }  {[Location].[Reporting Location], DisplayName} include_nullmembers;

		NPIItemMaster : select([Item].[NPI Item]);

		NPIAccountMaster : select([Account].[NPI Account]);

		NPIChannelMaster : select([Channel].[NPI Channel]);

		NPIRegionMaster : select([Region].[NPI Region]);

		NPIPnLMaster : select([PnL].[NPI PnL]);

		NPIDemandDomainMaster : select([DemandDomain].[NPI DemandDomain]);

		NPILocationMaster: select([Location].[NPI Location]);

    Output Variables:
    --------------------
        NPIItem
        NPIAccount
        NPIChannel
        NPIRegion
        NPIPnL
        NPIDemandDomain
        NPILocation

    Slice Dimension Attributes:
    -----------------------------
"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP121CreateNPIMember import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


# Load Data from o9DataLake
LevelInput = O9DataLake.get("LevelInput")
ItemMaster = O9DataLake.get("ItemMaster")
AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
RegionMaster = O9DataLake.get("RegionMaster")
PnLMaster = O9DataLake.get("PnLMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
LocationMaster = O9DataLake.get("LocationMaster")
NPIItemMaster = O9DataLake.get("NPIItemMaster")
NPIAccountMaster = O9DataLake.get("NPIAccountMaster")
NPIChannelMaster = O9DataLake.get("NPIChannelMaster")
NPIRegionMaster = O9DataLake.get("NPIRegionMaster")
NPIPnLMaster = O9DataLake.get("NPIPnLMaster")
NPILocationMaster = O9DataLake.get("NPILocationMaster")
NPIDemandDomainMaster = O9DataLake.get("NPIDemandDomainMaster")


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


# Execute Main
(NPIItem, NPIAccount, NPIChannel, NPIRegion, NPIPnL, NPIDemandDomain, NPILocation) = main(
    # Data
    LevelInput=LevelInput,
    # Master data
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    LocationMaster=LocationMaster,
    NPIItemMaster=NPIItemMaster,
    NPIAccountMaster=NPIAccountMaster,
    NPIRegionMaster=NPIRegionMaster,
    NPIChannelMaster=NPIChannelMaster,
    NPIPnLMaster=NPIPnLMaster,
    NPILocationMaster=NPILocationMaster,
    NPIDemandDomainMaster=NPIDemandDomainMaster,
    # Others
    df_keys=df_keys,
)

# Save Output Data
O9DataLake.put("NPIItem", NPIItem)
O9DataLake.put("NPIAccount", NPIAccount)
O9DataLake.put("NPIChannel", NPIChannel)
O9DataLake.put("NPIRegion", NPIRegion)
O9DataLake.put("NPIPnL", NPIPnL)
O9DataLake.put("NPIDemandDomain", NPIDemandDomain)
O9DataLake.put("NPILocation", NPILocation)

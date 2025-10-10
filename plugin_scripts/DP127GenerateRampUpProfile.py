"""
    Plugin Information:
    --------------------
        Plugin : DP127GenerateRampUpProfile
        Version : 2025.08.0
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
    ExecutionScope: 'LikeItemInfo' or 'RampUpInfo' or 'All'.

    Input Queries:
    --------------------
        SelectedCombinations: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[Generate Ramp Up Profile Assortment L0]}) on column;

        Parameters: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[NPI Ramp Up Period L0],Measure.[NPI Ramp Up Bucket L0], Measure.[Like Item Fcst Method L0]}) on column;

        LifecycleVolume: Select ([Version].[Version Name] * [Channel].[Planning Channel] * [Item].[Planning Item] * [Region].[Planning Region] * [Location].[Planning Location] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Lifecycle Time].[Day] ) on row, ({Measure.[Lifecycle Volume]}) on column;

        LifecycleTime: Select ([Lifecycle Time].[Day] * [Lifecycle Time].[Week] * [Lifecycle Time].[Partial Week] * [Lifecycle Time].[Month] * [Lifecycle Time].[Planning Month] * [Lifecycle Time].[Lifecycle Bucket]) on row, () on column include memberproperties {[Lifecycle Time].[Day], Key} {[Lifecycle Time].[Week], Key} {[Lifecycle Time].[Partial Week], Key} {[Lifecycle Time].[Month], Key} {[Lifecycle Time].[Planning Month], Key} {[Lifecycle Time].[Lifecycle Bucket], Key};

        LikeItem: Select (FROM.[Initiative].[Initiative] * FROM.[Data Object].[Data Object] * FROM.[Item].[NPI Item] * FROM.[Account].[NPI Account] * FROM.[Channel].[NPI Channel] * FROM.[Region].[NPI Region] * FROM.[PnL].[NPI PnL] * FROM.[Demand Domain].[NPI Demand Domain] * FROM.[Location].[NPI Location] * TO.[Item].[NPI Item] * TO.[Account].[NPI Account] * TO.[Channel].[NPI Channel] * TO.[Region].[NPI Region] * TO.[PnL].[NPI PnL] * TO.[Demand Domain].[NPI Demand Domain] * TO.[Location].[NPI Location]) on row, ({Edge.[630 Initiative Like Assortment Match].[Like Assortment Rank L0] , Edge.[630 Initiative Like Assortment Match].[User Override Like Assortment L0] , Edge.[630 Initiative Like Assortment Match].[Final Like Assortment L0] , Edge.[630 Initiative Like Assortment Match].[User Override Like Assortment Weight L0]}) on column where {RelationshipType.[630 Initiative Like Assortment Match], Version.[Version Name]};

        InitiativeLevel: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] ) on row, ({Measure.[NPI Item Level],Measure.[NPI Account Level], Measure.[NPI Channel Level], Measure.[NPI Region Level], Measure.[NPI PnL Level], Measure.[NPI Demand Domain Level], Measure.[NPI Location Level]}) on column;

        ItemMaster: Select ([Item].[Planning Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]) on row,() on column Where {&DPAllFinishedGoodsItemGroups};

        AccountMaster: Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account]);

        ChannelMaster: Select ( [Channel].[All Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel]);

        RegionMaster: Select ([Region].[All Region]*[Region].[Region L4]*[Region].[Region L3]*[Region].[Region L2]*[Region].[Region L1]*[Region].[Planning Region]);

        PnLMaster: Select ([PnL].[All PnL]*[PnL].[PnL L4]*[PnL].[PnL L3]*[PnL].[PnL L2]*[PnL].[PnL L1]*[PnL].[Planning PnL]);

        DemandDomainMaster: Select ([Demand Domain].[All Demand Domain]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Planning Demand Domain]);

        LocationMaster: Select ([Location].[All Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]*[Location].[Reporting Location]);


    Output Variables:
    --------------------
        LikeItemInfo
        SystemDefinedRampUpVolume
        RampUpProfile

    Slice Dimension Attributes:

"""

# Library imports
from helpers.o9helpers.o9logger import O9Logger
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP127GenerateRampUpProfile import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = O9Logger()

# Load Data from o9DataLake
SelectedCombinations = O9DataLake.get("SelectedCombinations")
Parameters = O9DataLake.get("Parameters")
LifecycleVolume = O9DataLake.get("LifecycleVolume")
LifecycleTime = O9DataLake.get("LifecycleTime")
LikeItem = O9DataLake.get("LikeItem")
InitiativeLevel = O9DataLake.get("InitiativeLevel")
ItemMaster = O9DataLake.get("ItemMaster")
AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
RegionMaster = O9DataLake.get("RegionMaster")
PnLMaster = O9DataLake.get("PnLMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
LocationMaster = O9DataLake.get("LocationMaster")


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


(LikeItemInfo, SystemDefinedRampUpVolume, RampUpProfile) = main(
    # Data
    SelectedCombinations=SelectedCombinations,
    Parameters=Parameters,
    LifecycleVolume=LifecycleVolume,
    LifecycleTime=LifecycleTime,
    LikeItem=LikeItem,
    InitiativeLevel=InitiativeLevel,
    # Master data
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    LocationMaster=LocationMaster,
    ExecutionScope=ExecutionScope,
    df_keys=df_keys,
)


# Save Output Data
O9DataLake.put("LikeItemInfo", LikeItemInfo)
O9DataLake.put("SystemDefinedRampUpVolume", SystemDefinedRampUpVolume)
O9DataLake.put("RampUpProfile", RampUpProfile)

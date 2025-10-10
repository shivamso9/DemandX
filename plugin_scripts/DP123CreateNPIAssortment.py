"""
    Plugin Information:
    --------------------
        Plugin : DP123CreateNPIAssortment
        Version : 2025.08.00
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
        Initiative: NA

        Level: NA

        AssortmentType: NA

        NPIItemScope: NA

        NPIAccountScope: NA

        NPIChannelScope: NA

        NPIRegionScope: NA

        NPIPnLScope: NA

        NPIDemandDomainScope: NA

        NPILocationScope: NA

        AssortmentSequence: Item.[Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Location]

        Delimiter: ',' #Separator of NPIItemScope and others.

        ReadFromHive: False

    Input Queries:
    --------------------
        InitiativeLevel: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] ) on row, ({Measure.[NPI Item Level],Measure.[NPI Account Level], Measure.[NPI Channel Level], Measure.[NPI Region Level], Measure.[NPI PnL Level], Measure.[NPI Demand Domain Level], Measure.[NPI Location Level], Measure.[NPI Level Sequence L1]}) on column;

        NPIAssortmentFinal: Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Initiative].[Initiative] * [Item].[Item] * [Location].[Location] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row, ({Measure.[NPI Assortment Final]}) on column;

        NPIAssortmentFinalByLevel: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[Item] * [Location].[Location] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[NPI Assortment Final by Level]}) on column;

        AssortmentFinal: Select ([Version].[Version Name] * [Item].[Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Assortment Final]}) on column;

        NPIAssortmentFileUpload: Select ([Version].[Version Name]  * [Personnel].[Email] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row,  ({Measure.[NPI Data Accepted Flag]}) on column where{Measure.[NPI Data Accepted Flag] == True && coalesce(Measure.[Is NPI Assortment Created],0) == 0};
        
        ItemMaster: Select ([Item].[Item]*[Item].[Planning Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]) on row,() on column Where {&DPAllFinishedGoodsItemGroups};

        AccountMaster: Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account]);

        ChannelMaster: Select ( [Channel].[All Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel]);

        RegionMaster: Select ([Region].[All Region]*[Region].[Region L4]*[Region].[Region L3]*[Region].[Region L2]*[Region].[Region L1]*[Region].[Planning Region]);

        PnLMaster: Select ([PnL].[All PnL]*[PnL].[PnL L4]*[PnL].[PnL L3]*[PnL].[PnL L2]*[PnL].[PnL L1]*[PnL].[Planning PnL]);

        DemandDomainMaster: Select ([Demand Domain].[All Demand Domain]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Planning Demand Domain]);

        LocationMaster: Select ([Location].[All Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]*[Location].[Reporting Location]*[Location].[Location]);

    Output Variables:
    --------------------
        NPIAssociationL0Output
        NPIAssortmentFinalByLevelOutput
        NPIAssortmentFinalOutput
        PlLevelAssortmentOutput

    Slice Dimension Attributes:

"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP123CreateNPIAssortment import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
InitiativeLevel = O9DataLake.get("InitiativeLevel")
NPIAssortmentFinal = O9DataLake.get("NPIAssortmentFinal")
NPIAssortmentFinalByLevel = O9DataLake.get("NPIAssortmentFinalByLevel")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
NPIAssortmentFileUpload = O9DataLake.get("NPIAssortmentFileUpload")
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

(
    NPIAssociationL0Output,
    NPIAssortmentFinalByLevelOutput,
    NPIAssortmentFinalOutput,
    PlLevelAssortmentOutput,
) = main(
    # Params
    Initiative=Initiative,
    Level=Level,
    AssortmentType=AssortmentType,
    AssortmentSequence=AssortmentSequence,
    # NPI Scope
    NPIItemScope=NPIItemScope,
    NPIAccountScope=NPIAccountScope,
    NPIChannelScope=NPIChannelScope,
    NPIRegionScope=NPIRegionScope,
    NPIPnLScope=NPIPnLScope,
    NPIDemandDomainScope=NPIDemandDomainScope,
    NPILocationScope=NPILocationScope,
    # Data
    AssortmentFinal=AssortmentFinal,
    NPIAssortmentFileUpload=NPIAssortmentFileUpload,
    InitiativeLevel=InitiativeLevel,
    NPIAssortmentFinal=NPIAssortmentFinal,
    NPIAssortmentFinalByLevel=NPIAssortmentFinalByLevel,
    # Master data
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    LocationMaster=LocationMaster,
    # Others
    Delimiter=Delimiter,
    ReadFromHive=ReadFromHive,
    df_keys=df_keys,
)

# Save Output Data
O9DataLake.put("NPIAssociationL0Output", NPIAssociationL0Output)
O9DataLake.put("NPIAssortmentFinalByLevelOutput", NPIAssortmentFinalByLevelOutput)
O9DataLake.put("NPIAssortmentFinalOutput", NPIAssortmentFinalOutput)
O9DataLake.put("PlLevelAssortmentOutput", PlLevelAssortmentOutput)

"""
    Plugin Information:
    --------------------
        Plugin : DP136ValidateNPIAssortmentFromFileUpload
        Version : 2025.08.00
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------

    Input Queries:
    --------------------
        GlobalNPILevels :  Select ([Version].[Version Name] * &AllNPILevel ) on row,
        ({ Measure.[Global NPI Item Level], Measure.[Global NPI Account Level], Measure.[Global NPI Channel Level], Measure.[Global NPI Region Level], Measure.[Global NPI PnL Level], Measure.[Global NPI Demand Domain Level], Measure.[Global NPI Location Level], Measure.[NPI Level Sequence], Measure.[NPI Forecast Generation Method], Measure.[NPI Ramp Up Bucket], Measure.[NPI Ramp Up Period]}) on column;

        ItemMaster : select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6] * [Item].[All Item]) on row, () on column include memberproperties {Item.[Planning Item], DisplayName} {Item.[L1], DisplayName} {Item.[L2], DisplayName} {Item.[L3], DisplayName} {Item.[L4], DisplayName} {Item.[L5], DisplayName} {Item.[L6], DisplayName} {Item.[All Item], DisplayName}  include_nullmembers Where {&DPAllFinishedGoodsItemGroups};

        AccountMaster :  select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include memberproperties {[Account].[Account L1], DisplayName} {[Account].[Account L2], DisplayName} {[Account].[Account L3], DisplayName} {[Account].[Account L4], DisplayName} {[Account].[All Account], DisplayName} {[Account].[Planning Account], DisplayName} include_nullmembers;

        ChannelMaster : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column include memberproperties {[Channel].[Channel L1], DisplayName} {[Channel].[Channel L2], DisplayName} {[Channel].[Planning Channel], DisplayName} {[Channel].[All Channel], DisplayName} include_nullmembers;

        RegionMaster : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName}  {[Region].[Region L3], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName} include_nullmembers;

        PnLMaster : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include memberproperties {[PnL].[All PnL], DisplayName} {[PnL].[Planning PnL], DisplayName} {[PnL].[PnL L1], DisplayName} {[PnL].[PnL L2], DisplayName} {[PnL].[PnL L3], DisplayName} {[PnL].[PnL L4], DisplayName} include_nullmembers;

        DemandDomainMaster : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName} {[Demand Domain].[Transition Demand Domain], DisplayName} include_nullmembers;

        LocationMaster : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location] * [Location].[Reporting Location]) on row, () on column include memberproperties {Location.[Planning Location], DisplayName} {Location.[Location Country], DisplayName} {Location.[Location Region], DisplayName} {Location.[Location Type], DisplayName} {Location.[All Location], DisplayName} {Location.[Location], DisplayName }  {[Location].[Reporting Location], DisplayName} include_nullmembers;

        Fileupload : Select ([Version].[Version Name] * [Personnel].[Email] * [Sequence].[Sequence] ) on row,
        ({Measure.[NPI Initiative Name], Measure.[NPI Initiative Description], Measure.[NPI Level Name], Measure.[Assortment Group Number], Measure.[NPI Item Name], Measure.[NPI Account Name], Measure.[NPI Channel Name], Measure.[NPI Region Name], Measure.[NPI PnL Name], Measure.[NPI Demand Domain Name], Measure.[NPI Location Name], Measure.[NPI Product Launch Date], Measure.[NPI Product EOL Date]}) on column;

        NPIInitiative : select (&AllNPIInitiative ) on row, () on column include memberproperties {[Initiative].[Initiative], DisplayName} include_nullmembers;

        NPILevel : select (&AllNPILevel ) on row, () on column include memberproperties {[Data Object].[Data Object], DisplayName} include_nullmembers;

        NPIInitiativeLevel : Select ([Version].[Version Name] * &AllNPIInitiative * &AllNPILevel) on row,
        ({Measure.[NPI Item Level], Measure.[NPI Account Level], Measure.[NPI Channel Level], Measure.[NPI Region Level], Measure.[NPI PnL Level], Measure.[NPI Demand Domain Level], Measure.[NPI Location Level], Measure.[NPI Initiative Level Association], Measure.[NPI Initiative Level Status], Measure.[NPI Level Sequence L1]}) on column;

        NPIInitiativeStatus : Select ([Version].[Version Name] * &AllNPIInitiative ) on row, ({Measure.[NPI Initiative Status]}) on column;

        NPIAssortments : Select ([Version].[Version Name] * &AllNPIInitiative * &AllNPILevel * &AllNPIItem * &AllNPIAccount * &AllNPIChannel * &AllNPIRegion * &AllNPIPnL * &AllNPIDemandDomain * &AllNPILocation ) on row, ({Measure.[NPI Association L0]}) on column;

        NPIInitiativeIDSequence(to be execute in plugin): select NextLabel([NPIInitiativeID]);

        NPIDataValidation: select([Data Validation].[Data Validation Type].[NPI Project].relatedmembers([Data Validation]));

    Output Variables:
    --------------------
        Invalid_Entries 
        Initiative_Fact_Data 
        Initiative_Level_Fact_Data 
        valid_entries 
        initiative_stage_fact_file
        initiative_dim_data: uploaded via WebAPI using CreateMember

    Slice Dimension Attributes:
    -----------------------------
"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP136ValidateNPIAssortmentFromFileUpload  import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


# Load Data from o9DataLake
GlobalNPILevels = O9DataLake.get("GlobalNPILevels")
ItemMaster = O9DataLake.get("ItemMaster")
AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
RegionMaster = O9DataLake.get("RegionMaster")
PnLMaster = O9DataLake.get("PnLMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
LocationMaster = O9DataLake.get("LocationMaster")
FileUpload = O9DataLake.get("FileUpload")
NPIInitiative = O9DataLake.get("NPIInitiative")
NPILevel = O9DataLake.get("NPILevel")
NPIInitiativeLevel = O9DataLake.get("NPIInitiativeLevel")
NPIInitiativeStatus = O9DataLake.get("NPIInitiativeStatus")
NPIAssortments = O9DataLake.get("NPIAssortments")
DataValidation = O9DataLake.get("DataValidation")


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
Invalid_Entries, Initiative_Fact_Data, Initiative_Level_Fact_Data, valid_entries, initiative_stage_fact_file = main(
    # Data
    GlobalNPILevels=GlobalNPILevels,
    # Master data
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    LocationMaster=LocationMaster,
    FileUpload=FileUpload,
    NPIInitiative=NPIInitiative,
    NPILevel=NPILevel,
    NPIInitiativeLevel=NPIInitiativeLevel,
    NPIInitiativeStatus=NPIInitiativeStatus,
    NPIAssortments=NPIAssortments,
    DataValidation=DataValidation,
    # Others
    df_keys=df_keys
)

# Save Output Data
O9DataLake.put("Invalid_Entries", Invalid_Entries)
O9DataLake.put("Initiative_Fact_Data", Initiative_Fact_Data)
O9DataLake.put("Initiative_Level_Fact_Data", Initiative_Level_Fact_Data)
O9DataLake.put("valid_entries", valid_entries)
O9DataLake.put("initiative_stage_fact_file", initiative_stage_fact_file)
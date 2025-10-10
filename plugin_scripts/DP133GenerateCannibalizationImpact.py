"""
    Plugin Information:
    --------------------
		Plugin : DP133GenerateCannibalizationImpact
		Version : 0.0.0
		Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
        PlanningMonthFormat: M%m-%y

    Input Queries:
    --------------------
        SelectedCombinations :  Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[Generate Cannib Impact Assortment L0]}) on column;

		ItemMaster : Select ([Item].[Item]*[Item].[Planning Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]) on row,() on column Where {&DPAllFinishedGoodsItemGroups};

		AccountMaster : Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account]);

		ChannelMaster : Select ( [Channel].[All Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel]);

		RegionMaster : Select ([Region].[All Region]*[Region].[Region L4]*[Region].[Region L3]*[Region].[Region L2]*[Region].[Region L1]*[Region].[Planning Region]);

		PnLMaster : Select ([PnL].[All PnL]*[PnL].[PnL L4]*[PnL].[PnL L3]*[PnL].[PnL L2]*[PnL].[PnL L1]*[PnL].[Planning PnL]);

		DemandDomainMaster : Select ([Demand Domain].[All Demand Domain]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Planning Demand Domain]);

		LocationMaster : Select ([Location].[All Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]*[Location].[Reporting Location]*[Location].[Location]);

		AssortmentFinal : Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Assortment Final]}) on column;

		CannibItem : Select (FROM.[Initiative].[Initiative] * FROM.[Data Object].[Data Object] * FROM.[Item].[NPI Item] * FROM.[Account].[NPI Account] * FROM.[Channel].[NPI Channel] * FROM.[Region].[NPI Region] * FROM.[PnL].[NPI PnL] * FROM.[Demand Domain].[NPI Demand Domain] * FROM.[Location].[NPI Location] * TO.[Item].[NPI Item] * [Version].[Version Name]) on row, ({Edge.[635 Initiative Cannibalized Item Match].[Final Cannib Item L0]}) on column where {RelationshipType.[635 Initiative Cannibalized Item Match], Edge.[635 Initiative Cannibalized Item Match].[Final Cannib Item L0]==True};

		TimeDim : Select ([Time].[Day] * [Time].[Week] * [Time].[Partial Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {[Time].[Week], Key} {[Time].[Partial Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

		StatFcst : Select ([Version].[Version Name] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] * [Time].[Partial Week] ) on row, ({Measure.[Stat Fcst NPI BB L0]}) on column;

		NPIFcst : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Location].[NPI Location] * [Item].[NPI Item] * [Initiative].[Initiative] * [Demand Domain].[NPI Demand Domain] * [Data Object].[Data Object] * [Channel].[NPI Channel] * [Account].[NPI Account] ) on row, ({Measure.[NPI Fcst Final L0]}) on column;

		DefaultProfile : Select ([PLC Profile].[PLC Profile] * [Version].[Version Name] * [Lifecycle Time].[Lifecycle Bucket] ) on row, ({Measure.[Default Profile]}) on column include memberproperties {[PLC Profile].[PLC Profile], [PLC Time Bucket]} {[PLC Profile].[PLC Profile], [PLC Profile Type]} {[Lifecycle Time].[Lifecycle Bucket], Key} include nulls;

		Parameters: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[Start Date L0],Measure.[End Date L0],Measure.[User Defined Cannibalization Period L0], Measure.[Cannibalization Profile L0], Measure.[Cannibalization Profile Bucket L0]}) on column;

    Output Variables:
    --------------------
        CannibImpact
        IndependenceDate
        Splits

    Slice Dimension Attributes:
    -----------------------------
"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP133GenerateCannibalizationImpact import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
SelectedCombinations = O9DataLake.get("SelectedCombinations")
CannibItem = O9DataLake.get("CannibItem")
StatFcst = O9DataLake.get("StatFcst")
NPIFcst = O9DataLake.get("NPIFcst")
DefaultProfile = O9DataLake.get("DefaultProfile")
Parameters = O9DataLake.get("Parameters")
TimeDim = O9DataLake.get("TimeDim")
RegionMaster = O9DataLake.get("RegionMaster")
PnLMaster = O9DataLake.get("PnLMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
ItemMaster = O9DataLake.get("ItemMaster")
AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
LocationMaster = O9DataLake.get("LocationMaster")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
InitiativeLevels = O9DataLake.get("InitiativeLevels")

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


CannibImpact, IndependenceDate, Splits = main(
    # Params
    PlanningMonthFormat=PlanningMonthFormat,
    # NPI Scope
    SelectedCombinations=SelectedCombinations,
    CannibItem=CannibItem,
    StatFcst=StatFcst,
    NPIFcst=NPIFcst,
    DefaultProfile=DefaultProfile,
    Parameters=Parameters,
    TimeDim=TimeDim,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    LocationMaster=LocationMaster,
    AssortmentFinal=AssortmentFinal,
    InitiativeLevels=InitiativeLevels,
    df_keys=df_keys,
)


# Save Output Data
O9DataLake.put("CannibImpact", CannibImpact)
O9DataLake.put("IndependenceDate", IndependenceDate)
O9DataLake.put("Splits", Splits)

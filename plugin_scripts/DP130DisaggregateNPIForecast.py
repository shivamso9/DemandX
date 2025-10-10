"""
    Plugin Information:
    --------------------
        Plugin : DP130DisaggregateNPIForecast
        Version : 0.0.0
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------

    Input Queries:
    --------------------
        SelectedInitiativeLevel: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] ) on row, ({Measure.[NPI Forecast Disaggregation Flag L1]}) on column;

        InitiativeLevels: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object]) on row, ({Measure.[NPI Item Level], Measure.[NPI Account Level], Measure.[NPI Channel Level], Measure.[NPI Region Level], Measure.[NPI PnL Level], Measure.[NPI Demand Domain Level], Measure.[NPI Location Level], Measure.[NPI Level Sequence L1]}) on column;

        NPIAssociation: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location]) on row, ({Measure.[NPI Association L0]}) on column;

        NPIFcst: Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Location].[NPI Location] * [Item].[NPI Item] * [Initiative].[Initiative] * [Demand Domain].[NPI Demand Domain] * [Data Object].[Data Object] * [Channel].[NPI Channel] * [Account].[NPI Account]) on row, ({Measure.[NPI Fcst Final L0]}) on column;

        NPIPlanningLevelFcst: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week]) on row, ({Measure.[Planning Level NPI Fcst]}) on column;

        NPIPlanningLevelFcstL1: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week]) on row, ({Measure.[Planning Level NPI Fcst L1]}) on column;

        Splits: Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Data Object].[Data Object] * [Demand Domain].[Planning Demand Domain] * [Initiative].[Initiative] * [Item].[Planning Item] * [Location].[Planning Location] * [PnL].[Planning PnL] * [Region].[Planning Region]) on row, ({Measure.[NPI Final Split %]}) on column;

        ItemMaster: Select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6]) on row, () on column where {&DPAllFinishedGoodsItemGroups};

        AccountMaster: Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account]);

        ChannelMaster: Select ([Channel].[All Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel]);

        RegionMaster: Select ([Region].[All Region] * [Region].[Region L4] * [Region].[Region L3] * [Region].[Region L2] * [Region].[Region L1] * [Region].[Planning Region]);

        PnLMaster: Select ([PnL].[All PnL] * [PnL].[PnL L4] * [PnL].[PnL L3] * [PnL].[PnL L2] * [PnL].[PnL L1] * [PnL].[Planning PnL]);

        DemandDomainMaster: Select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L2] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Planning Demand Domain]);

        LocationMaster: Select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location Country] * [Location].[Location Region] * [Location].[Planning Location] * [Location].[Reporting Location]);

    Output Variables:
    --------------------
        PlanningFcstOutput
        PlanningFcstL1Output
        EligibleLevelsOutput

    Slice Dimension Attributes:

"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP130DisaggregateNPIForecast import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
SelectedInitiativeLevel = O9DataLake.get("SelectedInitiativeLevel")
InitiativeLevels = O9DataLake.get("InitiativeLevels")
NPIAssociation = O9DataLake.get("NPIAssociation")
NPIFcst = O9DataLake.get("NPIFcst")
NPIPlanningLevelFcst = O9DataLake.get("NPIPlanningLevelFcst")
NPIPlanningLevelFcstL1 = O9DataLake.get("NPIPlanningLevelFcstL1")
Splits = O9DataLake.get("Splits")
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

(PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput) = main(
    # Data
    SelectedInitiativeLevel=SelectedInitiativeLevel,
    InitiativeLevels=InitiativeLevels,
    NPIAssociation=NPIAssociation,
    NPIFcst=NPIFcst,
    NPIPlanningLevelFcst=NPIPlanningLevelFcst,
    NPIPlanningLevelFcstL1=NPIPlanningLevelFcstL1,
    Splits=Splits,
    # Master data
    ItemMaster=ItemMaster,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    RegionMaster=RegionMaster,
    PnLMaster=PnLMaster,
    DemandDomainMaster=DemandDomainMaster,
    LocationMaster=LocationMaster,
    # Others
    df_keys=df_keys,
)

# Save Output Data
O9DataLake.put("PlanningFcstOutput", PlanningFcstOutput)
O9DataLake.put("PlanningFcstL1Output", PlanningFcstL1Output)
O9DataLake.put("EligibleLevelsOutput", EligibleLevelsOutput)

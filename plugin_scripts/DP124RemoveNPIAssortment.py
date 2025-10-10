"""
    Plugin Information:
    --------------------
        Plugin : DP124RemoveNPIAssortment
        Version : 0.0.0
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
        RemoveAssortmentScope: Partial. # Options: Full or Partial.

    Input Queries:
    --------------------
        NPIRemoveAssortmentFlag: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location]) on row, ({Measure.[Remove NPI Assortment Flag L0]}) on column;

        InitiativeLevel: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object]) on row, ({Measure.[NPI Item Level], Measure.[NPI Account Level], Measure.[NPI Channel Level], Measure.[NPI Region Level], Measure.[NPI PnL Level], Measure.[NPI Demand Domain Level], Measure.[NPI Location Level], Measure.[NPI Level Sequence L1]}) on column;

        NPIFcstPublished: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week]) on row, ({Measure.[NPI Fcst Published]}) on column;

        AssortmentNew: Select ([Version].[Version Name] * [Item].[Item]  * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Assortment New]}) on column;

        InitiativePlanningLevel: Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Data Object].[Data Object] * [Demand Domain].[Planning Demand Domain] * [Initiative].[Initiative] * [Item].[Planning Item] * [Location].[Planning Location] * [PnL].[Planning PnL] * [Region].[Planning Region]) on row, ({Measure.[Cannib Final Split %], Measure.[Cannib Override Split %], Measure.[Cannib System Split %], Measure.[Cannibalization Independence Date Planning Level], Measure.[NPI Final Split %], Measure.[NPI Override Split %], Measure.[NPI Planning Assortment by Level], Measure.[NPI System Split %]}) on column;

        InitiativeLevelAssortment: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[Item]  * [Location].[Location]  * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[NPI Assortment Final by Level]}) on column;

        InitiativeAssortment: Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Initiative].[Initiative] * [Item].[Item] * [Location].[Location] * [PnL].[Planning PnL] * [Region].[Planning Region]) on row, ({Measure.[NPI Assortment Final]}) on column;

        InitiativePlanningLevelNewProductForecast: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week]) on row, ({Measure.[Planning Level Cannibalization Impact], Measure.[Planning Level NPI Fcst]}) on column;

        InitiativeNewProductForecast: Select ([Version].[Version Name] * [Initiative].[Initiative] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location]*Time.[Partial Week]) on row, ({Measure.[Planning Level Cannibalization Impact L1], Measure.[Planning Level NPI Fcst L1]}) on column;

        ItemMaster: Select (Item.[Item]*[Item].[Planning Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]) on row,() on column Where {&DPAllFinishedGoodsItemGroups};

        AccountMaster: Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account]);

        ChannelMaster: Select ([Channel].[All Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel]);

        RegionMaster: Select ([Region].[All Region] * [Region].[Region L4] * [Region].[Region L3] * [Region].[Region L2] * [Region].[Region L1] * [Region].[Planning Region]);

        PnLMaster: Select ([PnL].[All PnL] * [PnL].[PnL L4] * [PnL].[PnL L3] * [PnL].[PnL L2] * [PnL].[PnL L1] * [PnL].[Planning PnL]);

        DemandDomainMaster: Select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L2] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Planning Demand Domain]);

        LocationMaster: Select (Location.[Location]*[Location].[All Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]);

    Output Variables:
    --------------------
        NPIFcstPublishedDeleted
        AssortmentNewDeleted
        InitiativePlanningLevelDeleted
        InitiativeLevelAssortmentDeleted
        InitiativeAssortmentDeleted
        InitiativePlanningLevelNewProdFcstDeleted
        InitiativeNewProductForecastDeleted

    Slice Dimension Attributes:

"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP124RemoveNPIAssortment import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
NPIRemoveAssortmentFlag = O9DataLake.get("NPIRemoveAssortmentFlag")
InitiativeLevel = O9DataLake.get("InitiativeLevel")
NPIFcstPublished = O9DataLake.get("NPIFcstPublished")
AssortmentNew = O9DataLake.get("AssortmentNew")
InitiativePlanningLevel = O9DataLake.get("InitiativePlanningLevel")
InitiativeLevelAssortment = O9DataLake.get("InitiativeLevelAssortment")
InitiativeAssortment = O9DataLake.get("InitiativeAssortment")
InitiativePlanningLevelNewProductForecast = O9DataLake.get(
    "InitiativePlanningLevelNewProductForecast"
)
InitiativeNewProductForecast = O9DataLake.get("InitiativeNewProductForecast")
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
    NPIFcstPublishedDeleted,
    AssortmentNewDeleted,
    InitiativePlanningLevelDeleted,
    InitiativeLevelAssortmentDeleted,
    InitiativeAssortmentDeleted,
    InitiativePlanningLevelNewProdFcstDeleted,
    InitiativeNewProductForecastDeleted,
) = main(
    # Params
    RemoveAssortmentScope=RemoveAssortmentScope,
    # Data
    NPIRemoveAssortmentFlag=NPIRemoveAssortmentFlag,
    InitiativeLevel=InitiativeLevel,
    NPIFcstPublished=NPIFcstPublished,
    AssortmentNew=AssortmentNew,
    InitiativePlanningLevel=InitiativePlanningLevel,
    InitiativeLevelAssortment=InitiativeLevelAssortment,
    InitiativeAssortment=InitiativeAssortment,
    InitiativePlanningLevelNewProductForecast=InitiativePlanningLevelNewProductForecast,
    InitiativeNewProductForecast=InitiativeNewProductForecast,
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
O9DataLake.put("NPIFcstPublishedDeleted", NPIFcstPublishedDeleted)
O9DataLake.put("AssortmentNewDeleted", AssortmentNewDeleted)
O9DataLake.put("InitiativePlanningLevelDeleted", InitiativePlanningLevelDeleted)
O9DataLake.put("InitiativeLevelAssortmentDeleted", InitiativeLevelAssortmentDeleted)
O9DataLake.put("InitiativeAssortmentDeleted", InitiativeAssortmentDeleted)
O9DataLake.put(
    "InitiativePlanningLevelNewProdFcstDeleted", InitiativePlanningLevelNewProdFcstDeleted
)
O9DataLake.put("InitiativeNewProductForecastDeleted", InitiativeNewProductForecastDeleted)

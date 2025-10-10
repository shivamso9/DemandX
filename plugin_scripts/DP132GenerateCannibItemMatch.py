"""
    Plugin Information:
    --------------------
		Plugin : DP132GenerateCannibItemMatch
		Version : 0.0.0
		Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
        IncludeDemandDomain: True

    Input Queries:
    --------------------
        LikeAssortmentMatch :  Select (FROM.[Initiative].[Initiative] * FROM.[Data Object].[Data Object] * FROM.[Item].[NPI Item] * FROM.[Account].[NPI Account] * FROM.[Channel].[NPI Channel] * FROM.[Region].[NPI Region] * FROM.[PnL].[NPI PnL] * FROM.[Demand Domain].[NPI Demand Domain] * FROM.[Location].[NPI Location] * TO.[Item].[NPI Item] * TO.[Account].[NPI Account] * TO.[Channel].[NPI Channel] * TO.[Region].[NPI Region] * TO.[PnL].[NPI PnL] * TO.[Demand Domain].[NPI Demand Domain] * TO.[Location].[NPI Location]) on row, ({Edge.[630 Initiative Like Assortment Match].[Like Assortment Rank L0], Edge.[630 Initiative Like Assortment Match].[Final Like Assortment L0]}) on column where {RelationshipType.[630 Initiative Like Assortment Match], Version.[Version Name]};

		GenSysCannItemMatchAssortment : Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[Generate Cannibalized Item Match AssortmentL0]}) on column where {Measure.[Generate Cannibalized Item Match AssortmentL0]==1};

    Output Variables:
    --------------------
        Suggested_Cann_Item

    Slice Dimension Attributes:
    -----------------------------
"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP132GenerateCannibItemMatch import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


# Load Data from o9DataLake
LikeAssortmentMatch = O9DataLake.get("LikeAssortmentMatch")
GenSysCannItemMatchAssortment = O9DataLake.get("GenSysCannItemMatchAssortment")

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
Suggested_Cann_Item = main(
    # Data
    LikeAssortmentMatch=LikeAssortmentMatch,
    GenSysCannItemMatchAssortment=GenSysCannItemMatchAssortment,
    IncludeDemandDomain=IncludeDemandDomain,
    # Others
    df_keys=df_keys,
)

# Save Output Data
O9DataLake.put("Suggested_Cann_Item", Suggested_Cann_Item)

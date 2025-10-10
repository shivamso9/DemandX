"""
    Plugin : DP024SystemCannibItemMatch
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

    Input Queries:
        LikeAssortmentMatch : Select (FROM.[Item].[Planning Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Location].[Planning Location] *
                                      TO.[Item].[Planning Item] * TO.[Account].[Planning Account] * TO.[Channel].[Planning Channel] * TO.[Region].[Planning Region] * TO.[PnL].[Planning PnL] * TO.[Demand Domain].[Planning Demand Domain] * TO.[Location].[Planning Location]) on row,
                                      ({Edge.[620 Like Assortment Match].[Like Assortment Rank], Edge.[620 Like Assortment Match].[Final Like Assortment Selected]}) on column where {RelationshipType.[620 Like Assortment Match], [Version].[Version Name]};

        GenSysCannItemMatchAssortment : Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] ) on row,
                                        ({Measure.[Generate System Cannibalized Item Match Assortment]}) on column;

        IncludeDemandDomain: "True or False";

    Output Variables:
        System Suggested Cannibalized Item

    Slice Dimension Attributes:

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP024SystemCannibItemMatch import main

logger = logging.getLogger("o9_logger")

# Function Calls
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


Suggested_Cann_Item = main(
    LikeAssortmentMatch=LikeAssortmentMatch,
    GenSysCannItemMatchAssortment=GenSysCannItemMatchAssortment,
    IncludeDemandDomain=IncludeDemandDomain,
    df_keys=df_keys,
)


O9DataLake.put("Suggested_Cann_Item", Suggested_Cann_Item)
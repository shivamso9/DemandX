"""
Plugin : DP068UpdateTransitionItemOnDeleteTransition
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:

Input Queries:
    selectedCombinations : Select ([Version].[Version Name] ) on row, ({Measure.[Transition From Planning Item], Measure.[Transition To Planning Item], Measure.[Transition Selected Planning Account], Measure.[Transition Selected Planning Channel], Measure.[Transition Selected Planning Region], Measure.[Transition Selected Planning PnL], Measure.[Transition Selected Planning Demand Domain], Measure.[Transition Selected Planning Location]}) on column;

    TransitionFlag : Select (Version.[Version Name] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Location].[Planning Location] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Channel].[Planning Channel] * FROM.[Account].[Planning Account] * FROM.[Item].[Planning Item] * TO.[Item].[Planning Item]) on row, ({Edge.[510 Product Transition].[Transition Flag]}) on column where {RelationshipType.[510 Product Transition]};

    ItemDates : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row, ({Measure.[Intro Date], Measure.[Disco Date]}) on column;

    TransitionAttributes : Select ([Version].[Version Name] * [Item].[Planning Item] ) on row, ({Measure.[Transition Item Before Set Product Transition], Measure.[Transition Item After Set Product Transition], Measure.[Final Transition Item]}) on column;

Output Variables:
    TransitionAttributesOutput
    TransitionDates
    TransitionFlagOutput

Slice Dimension Attributes:

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP068UpdateTransitionItemOnDeleteTransition import main

logger = logging.getLogger("o9_logger")

# Function Calls
selectedCombinations = O9DataLake.get("selectedCombinations")
TransitionFlag = O9DataLake.get("TransitionFlag")
ItemDates = O9DataLake.get("ItemDates")
TransitionAttributes = O9DataLake.get("TransitionAttributes")

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

TransitionDates, TransitionAttributesOutput, TransitionFlagOutput = main(
    selectedCombinations=selectedCombinations,
    TransitionFlag=TransitionFlag,
    ItemDates=ItemDates,
    TransitionAttributes=TransitionAttributes,
    df_keys=df_keys,
)
O9DataLake.put("TransitionDates", TransitionDates)
O9DataLake.put("TransitionAttributesOutput", TransitionAttributesOutput)
O9DataLake.put("TransitionFlagOutput", TransitionFlagOutput)

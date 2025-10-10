"""
Plugin : DP067UpdateTransitionItemOnSetTransition
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    IntroDate : 09/01/2023
    DiscoDate : 12/01/2023
    AssortmentOutputGrains : Item.[Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Location.[Location],Demand Domain.[Planning Demand Domain]
    DatesOutputGrains : Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Location.[Planning Location],Demand Domain.[Planning Demand Domain]

Input Queries:
    AssortmentFoundation : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row,  ({Measure.[Assortment Foundation], Measure.[Intro Date], Measure.[Actual]}) on column;

    ItemMasterData : select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]);

    selectedCombinations : Select ([Version].[Version Name] ) on row, ({Measure.[Transition From Planning Item], Measure.[Transition To Planning Item], Measure.[Transition Selected Planning Account], Measure.[Transition Selected Planning Channel], Measure.[Transition Selected Planning Region], Measure.[Transition Selected Planning PnL], Measure.[Transition Selected Planning Demand Domain], Measure.[Transition Selected Planning Location]}) on column;

    TransitionFlag : Select (Version.[Version Name] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Location].[Planning Location] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Channel].[Planning Channel] * FROM.[Account].[Planning Account] * FROM.[Item].[Planning Item] * TO.[Item].[Planning Item]) on row, ({Edge.[510 Product Transition].[Transition Flag]}) on column where {RelationshipType.[510 Product Transition]};

    ExistingOutput : Select ([Version].[Version Name] * [Item].[Planning Item] ) on row,  ({Measure.[Transition Item Before Set Product Transition], Measure.[Transition Item After Set Product Transition], Measure.[Final Transition Item]}) on column;

    AssortmentFinal : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row, ({Measure.[Assortment Final]}) on column where {Measure.[Assortment Final] == 1.0};

Output Variables:
    TransitionAttributes
    IntroDateOutput
    DiscoDateOutput
    Assortment
    TransitionFlagOutput

Slice Dimension Attributes:

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP067UpdateTransitionItemOnSetTransition import main

logger = logging.getLogger("o9_logger")

# Function Calls
AssortmentFoundation = O9DataLake.get("AssortmentFoundation")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
ItemMasterData = O9DataLake.get("ItemMasterData")
selectedCombinations = O9DataLake.get("selectedCombinations")
TransitionFlag = O9DataLake.get("TransitionFlag")
ExistingOutput = O9DataLake.get("ExistingOutput")

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
    Assortment,
    IntroDateOutput,
    DiscoDateOutput,
    TransitionAttributes,
    TransitionFlagOutput,
) = main(
    AssortmentOutputGrains=AssortmentOutputGrains,
    DatesOutputGrains=DatesOutputGrains,
    IntroDate=IntroDate,
    DiscoDate=DiscoDate,
    AssortmentFoundation=AssortmentFoundation,
    AssortmentFinal=AssortmentFinal,
    ItemMasterData=ItemMasterData,
    selectedCombinations=selectedCombinations,
    TransitionFlag=TransitionFlag,
    ExistingOutput=ExistingOutput,
    df_keys=df_keys,
)
O9DataLake.put("Assortment", Assortment)
O9DataLake.put("IntroDateOutput", IntroDateOutput)
O9DataLake.put("DiscoDateOutput", DiscoDateOutput)
O9DataLake.put("TransitionAttributes", TransitionAttributes)
O9DataLake.put("TransitionFlagOutput", TransitionFlagOutput)

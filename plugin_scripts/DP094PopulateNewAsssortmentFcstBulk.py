"""
Plugin : DP094PopulateNewAsssortmentFcstBulk
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:

Input Queries:
    Parameters - Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item]) on row, ({Measure.[Intro Date]}) on column;

    SellOutFcstAdj - Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Region].[Planning Region] ) on row,  ({Measure.[Sell Out Forecast Adjustment 1 FND BB],Measure.[Sell Out Forecast Adjustment 2 FND BB],Measure.[Sell Out Forecast Adjustment 3 FND BB],Measure.[Sell Out Forecast Adjustment 4 FND BB],Measure.[Sell Out Forecast Adjustment 5 FND BB],Measure.[Sell Out Forecast Adjustment 6 FND BB], Measure.[Sell Out Stat Fcst KAF BB New FND BB],Measure.[DA Sell Out - 1 FND BB],Measure.[DA Sell Out - 2 FND BB],Measure.[DA Sell Out - 3 FND BB],Measure.[DA Sell Out - 4 FND BB],Measure.[DA Sell Out - 5 FND BB],Measure.[DA Sell Out - 6 FND BB]}) on column include memberproperties {Time.[Partial Week], Key};

    FcstAdj - Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row,  ({Measure.[Fcst Adjustment 1 FND BB],Measure.[Fcst Adjustment 2 FND BB],Measure.[Fcst Adjustment 3 FND BB],Measure.[Fcst Adjustment 4 FND BB],Measure.[Fcst Adjustment 5 FND BB],Measure.[Fcst Adjustment 6 FND BB],Measure.[Cannibalization Impact FND BB],Measure.[NPI Fcst FND BB],Measure.[DA - 1 FND BB],Measure.[DA - 2 FND BB],Measure.[DA - 3 FND BB],Measure.[DA - 4 FND BB],Measure.[DA - 5 FND BB],Measure.[DA - 6 FND BB]}) on column include memberproperties {Time.[Partial Week], Key};

    likeAssortmentMappings - Select (FROM.[Item].[Planning Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Location].[Planning Location] * TO.[Item].[Planning Item] * TO.[Account].[Planning Account] * TO.[Channel].[Planning Channel] * TO.[Region].[Planning Region] * TO.[PnL].[Planning PnL] * TO.[Demand Domain].[Planning Demand Domain] * TO.[Location].[Planning Location]) on row, ({Edge.[620 Like Assortment Match].[Final Like Assortment Selected],Edge.[620 Like Assortment Match].[User Like Assortment Override]}) on column where {RelationshipType.[620 Like Assortment Match], &CWV , Edge.[620 Like Assortment Match].[Final Like Assortment Selected] == TRUE};

    selectedCombinations - select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * Version.[Version Name]) on row, ({Measure.[Populate Like Item Fcst Assortment]}) on column where {Measure.[Populate Like Item Fcst Assortment]==1};

Output Variables:
    SellInFcstOutput
    SellInNPIFcstOutput
    SellOutFcstOutput
    SellOutNPIFcstOutput

Slice Dimension Attributes:

"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP094PopulateNewAsssortmentFcstBulk import main

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None
logger = logging.getLogger("o9_logger")

# Function Calls
likeAssortmentMappings = O9DataLake.get("likeAssortmentMappings")
SellOutFcstAdj = O9DataLake.get("SellOutFcstAdj")
FcstAdj = O9DataLake.get("FcstAdj")
selectedCombinations = O9DataLake.get("selectedCombinations")
Parameters = O9DataLake.get("Parameters")

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

SellInFcstOutput, SellInNPIFcstOutput, SellOutFcstOutput, SellOutNPIFcstOutput = main(
    like_item_mappings=likeAssortmentMappings,
    SellOutFcst=SellOutFcstAdj,
    SellInFcst=FcstAdj,
    selected_combinations=selectedCombinations,
    Parameters=Parameters,
)
O9DataLake.put("SellInFcstOutput", SellInFcstOutput)
O9DataLake.put("SellInNPIFcstOutput", SellInNPIFcstOutput)
O9DataLake.put("SellOutFcstOutput", SellOutFcstOutput)
O9DataLake.put("SellOutNPIFcstOutput", SellOutNPIFcstOutput)

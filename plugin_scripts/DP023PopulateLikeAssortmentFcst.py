"""
Plugin : DP023PopulateLikeAssortmentFcst
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    TimeLevel - Time.[Partial Week]
    Grains  - Item.[Planning Item],PnL.[Planning PnL],Region.[Planning Region],Account.[Planning Account],Channel.[Planning Channel],Location.[Planning Location],Demand Domain.[Planning Demand Domain]
    ReadFromHive - False

Input Queries:
    likeItemMappings - Select (FROM.[Item].[Planning Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Location].[Planning Location] * TO.[Item].[Planning Item] * TO.[Account].[Planning Account] * TO.[Channel].[Planning Channel] * TO.[Region].[Planning Region] * TO.[PnL].[Planning PnL] * TO.[Demand Domain].[Planning Demand Domain] * TO.[Location].[Planning Location]) on row, ({Edge.[620 Like Assortment Match].[Final Like Assortment Weight], Edge.[620 Like Assortment Match].[Final Like Assortment Selected]}) on column where {RelationshipType.[630 Like Assortment Match], [Version].[Version Name], Edge.[630 Like Assortment Match].[Final Like Assortment Selected] == TRUE};
    forecastData - Select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Time].[Partial Week] * [Version].[Version Name]) on row, ({Measure.[Stat Fcst NPI BB]}) on column;
    selectedCombinations - select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * Version.[Version Name]) on row, ({Measure.[Populate Like Item Fcst Assortment]}) on column where {Measure.[Populate Like Item Fcst Assortment]==1};
    IsAssorted - Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Planning Item] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] ) on row, ({Measure.[Is Assorted]}) on column;
    Parameters - Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Like Item Fcst Method]}) on column;

Output Variables:
    LikeItemForecast

Slice Dimension Attributes:

"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

# Function Calls
likeItemMappings = O9DataLake.get("likeItemMappings")
forecastData = O9DataLake.get("forecastData")
selectedCombinations = O9DataLake.get("selectedCombinations")
IsAssorted = O9DataLake.get("IsAssorted")
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

from helpers.DP023PopulateLikeAssortmentFcst import main

LikeItemForecast = main(
    like_item_mappings=likeItemMappings,
    forecast_data=forecastData,
    selected_combinations=selectedCombinations,
    TimeLevel=TimeLevel,
    Grains=Grains,
    IsAssorted=IsAssorted,
    Parameters=Parameters,
    df_keys=df_keys,
)
O9DataLake.put("LikeItemForecast", LikeItemForecast)

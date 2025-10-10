"""
Plugin : DP023PopulateLikeItemFcst
Version : 0.0.0
Maintained by : pmm_algocoe@o9solutions.com

Script Params:
    TimeLevel - Time.[Partial Week]
    Grains  - Item.[Stat Item],PnL.[Stat PnL],Region.[Stat Region],Account.[Stat Account],Channel.[Stat Channel],Location.[Stat Location],Demand Domain.[Stat Demand Domain]
    ReadFromHive - False

Input Queries:
    likeItemMappings - Select (FROM.[Item].[Stat Item] * FROM.[Account].[Stat Account] * FROM.[Channel].[Stat Channel] * FROM.[Region].[Stat Region] * FROM.[PnL].[Stat PnL]  *  FROM.[Location].[Stat Location] * TO.[Item].[Stat Item] * Version.[Version Name]) on row, ({Edge.[620 Like Item Match].[Final Like Item Selected]}) on column where {RelationshipType.[620 Like Item Match], Edge.[620 Like Item Match].[Final Like Item Selected]==TRUE};
    forecastData - Select ([Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Time].[Partial Week] * [Version].[Version Name]) on row, ({Measure.[Stat Fcst NPI BB]}) on column;
    selectedCombinations - select ([Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * Version.[Version Name]) on row, ({Measure.[Populate Like Item Fcst Assortment]}) on column where {Measure.[Populate Like Item Fcst Assortment]==1};
    IsAssorted - Select ([Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] ) on row, ({Measure.[Is Assorted]}) on column;

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

pd.options.mode.chained_assignment = None

# Function Calls
likeItemMappings = O9DataLake.get("likeItemMappings")
forecastData = O9DataLake.get("forecastData")
selectedCombinations = O9DataLake.get("selectedCombinations")
IsAssorted = O9DataLake.get("IsAssorted")

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

from helpers.DP024PopulateLikeItemFcstLEGO import main

LikeItemForecast = main(
    like_item_mappings=likeItemMappings,
    forecast_data=forecastData,
    selected_combinations=selectedCombinations,
    TimeLevel=TimeLevel,
    Grains=Grains,
    IsAssorted=IsAssorted,
    df_keys=df_keys,
)
O9DataLake.put("LikeItemForecast", LikeItemForecast)

"""
Plugin : DP034DCSplitRatio
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive - False

Input Queries:
    Actuals - Select ([Version].[Version Name] * [Location].[Location] * &AllPastPartialWeeks *[Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Item].[Planning Item] * [Item].[Item] ) on row, ({Measure.[Sell In Stat L0]}) on column;
    ConsensusFcst - Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * &AllPastPartialWeeks* [Item].[Planning Item] ) on row, ({Measure.[Consensus Fcst]}) on column;
    ItemAttribute - Select ([Version].[Version Name] * [Item].[Item] ) on row,  ({Measure.[Collab Attribute Item]}) on column;
    SKUDCSplit - Select ([Version].[Version Name] * [Location].[Location] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] * [Item].[Item] * [Item].[Planning Item] ) on row, ({Measure.[SKU DC Split]}) on column;

Output Variables:
    Output

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

from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

import threading

from o9Reference.common_utils.o9_memory_utils import _get_memory

# Function Calls
Actuals = O9DataLake.get("Actuals")
ConsensusFcst = O9DataLake.get("ConsensusFcst")
ItemAttribute = O9DataLake.get("ItemAttribute")
SKUDCSplit = O9DataLake.get("SKUDCSplit")

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

from helpers.DP034DCSplitRatio import main

Output = main(
    Actuals=Actuals,
    ConsensusFcst=ConsensusFcst,
    ItemAttribute=ItemAttribute,
    SKUDCSplit=SKUDCSplit,
    df_keys=df_keys,
)
O9DataLake.put("Output", Output)

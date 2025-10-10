"""
Plugin to expand transition assortments for sell-in and sell-out.
"""
import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

# Assuming the repo file is named 'DPXXXTransitionAssortmentExpansion.py' and is in the 'helpers' directory
from helpers.DPXXXTransitionAssortmentExpansion import main

logger = logging.getLogger("o9_logger")

# Function Calls to get data from the data lake
ItemMaster = O9DataLake.get("ItemMaster")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
AssortmentSellOut = O9DataLake.get("AssortmentSellOut")
Date = O9DataLake.get("Date")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

# Call the main business logic function from the repo file
(
    AssortmentFinal_output,
    AssortmentSellOut_output,
    TransitionFlag_output,
) = main(
    ItemMaster=ItemMaster,
    AssortmentFinal=AssortmentFinal,
    AssortmentSellOut=AssortmentSellOut,
    Date=Date,
)

# Put the processed dataframes back into the data lake
O9DataLake.put("AssortmentFinal", AssortmentFinal_output)
O9DataLake.put("AssortmentSellOut", AssortmentSellOut_output)
O9DataLake.put("TransitionFlag", TransitionFlag_output)
import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from TransitionAssortmentExpansion_Repo import main

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

# Call the main function from the repo file
AssortmentFinal_Output, AssortmentSellOut_Output, TransitionFlag_Output = main(
    ItemMaster=ItemMaster,
    AssortmentFinal=AssortmentFinal,
    AssortmentSellOut=AssortmentSellOut,
    Date=Date,
)

# Put the output dataframes into the data lake
O9DataLake.put("AssortmentFinal_Output", AssortmentFinal_Output)
O9DataLake.put("AssortmentSellOut_Output", AssortmentSellOut_Output)
O9DataLake.put("TransitionFlag_Output", TransitionFlag_Output)
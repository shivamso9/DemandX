import logging
import threading

from o9.datalake.o9_datalake import O9DataLake
from o9.reference.common.o9_memory_utils import _get_memory

# Assuming the repo file is named 'DPXXXTransitionAssortment'
# and is placed in the 'helpers' directory.
from helpers.DPXXXTransitionAssortment import main

logger = logging.getLogger("o9_logger")

# Function Calls to get data from the data lake
ItemMaster = O9DataLake.get("ItemMaster")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
AssortmentSellOut = O9DataLake.get("AssortmentSellOut")
Date = O9DataLake.get("Date")

# Check if slicing variable is present, default to empty dict if not
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice: %s", df_keys)

# Start a thread to monitor memory usage
back_thread = threading.Thread(
    target=_get_memory,
    kwargs={"max_memory": 0.0, "sleep_seconds": 90, "df_keys": df_keys},
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

# Call the main function from the repo file with the retrieved DataFrames
(
    output_assortment_final,
    output_assortment_sellout,
    output_transition_flag,
) = main(
    ItemMaster=ItemMaster,
    AssortmentFinal=AssortmentFinal,
    AssortmentSellOut=AssortmentSellOut,
    Date=Date,
    df_keys=df_keys,
)

# Put the resulting DataFrames back into the data lake
O9DataLake.put("AssortmentFinal", output_assortment_final)
O9DataLake.put("AssortmentSellOut", output_assortment_sellout)
O9DataLake.put("TransitionFlag", output_transition_flag)
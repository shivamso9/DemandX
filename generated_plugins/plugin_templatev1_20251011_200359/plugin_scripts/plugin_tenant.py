import logging
from o9_common_utils.O9DataLake import O9DataLake
from business_logic import main

logger = logging.getLogger("o9_logger")

# --- Function Calls ---
# Fetching data from the data lake
ItemMaster = O9DataLake.get("ItemMaster")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
AssortmentSellOut = O9DataLake.get("AssortmentSellOut")
Date = O9DataLake.get("Date")


# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# --- Main Business Logic ---
# The main function from the repository file is called with the dataframes.
AssortmentFinal_Output, AssortmentSellOut_Output, TransitionFlag_Output = main(
    ItemMaster=ItemMaster,
    AssortmentFinal=AssortmentFinal,
    AssortmentSellOut=AssortmentSellOut,
    Date=Date
)


# --- Data Upload ---
# The processed dataframes are uploaded back to the data lake.
O9DataLake.put("AssortmentFinal", AssortmentFinal_Output)
O9DataLake.put("AssortmentSellOut", AssortmentSellOut_Output)
O9DataLake.put("TransitionFlag", TransitionFlag_Output)

logger.info("Plugin execution completed successfully.")
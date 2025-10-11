import logging
from o9_common_utils.O9DataLake import O9DataLake

# Assuming the repo file is named 'DPXXX_Transition_Assortment_Repo.py'
# and is placed in the 'helpers' directory.
from helpers.DPXXX_Transition_Assortment_Repo import main

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
logger.info("Starting Transition Assortment Generation...")

# Call the main function from the repo file
Output_AssortmentFinal, Output_AssortmentSellOut, Output_TransitionFlag = main(
    ItemMaster=ItemMaster,
    AssortmentFinal=AssortmentFinal,
    AssortmentSellOut=AssortmentSellOut,
    Date=Date,
)

# Put the output dataframes back to the data lake
O9DataLake.put("AssortmentFinal", Output_AssortmentFinal)
O9DataLake.put("AssortmentSellOut", Output_AssortmentSellOut)
O9DataLake.put("TransitionFlag", Output_TransitionFlag)

logger.info("Transition Assortment Generation Completed.")
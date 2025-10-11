"""
Plugin to generate Sellin Season week association.
This plugin expands item-customer combinations for each week they are active,
based on their introduction and discontinuation dates.

Input Queries:
    PlanningItemCustomerGroup: SELECT 
        ([Version].[Version Name] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * 
         [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * 
         [Region].[Planning Region] * [Location].[Planning Location] * [Item].[Planning Item]) ON ROW, 
        ({Measure.[Intro Date], Measure.[Disco Date]}) ON COLUMN
        WHERE {~ISNULL(Measure.[Intro Date]), ~ISNULL(Measure.[Disco Date])};

    DimTime: SELECT 
        [Time].[Partial Week] ON ROW, 
        () ON COLUMN 
        INCLUDE MEMBERPROPERTIES {[Time].[Partial Week], Key};

Output Variables:
    Sellin Season
"""
import logging

from o9_common_utils.O9DataLake import O9DataLake

# Assuming the repo file is named 'sellin_season_repo.py' and is in the 'helpers' directory
from helpers.sellin_season_repo import main

logger = logging.getLogger("o9_logger")

# Get DataFrames from the data lake
PlanningItemCustomerGroup = O9DataLake.get("PlanningItemCustomerGroup")
DimTime = O9DataLake.get("DimTime")

# Check if slicing variable is present, default to an empty dictionary if not
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice: %s", df_keys)

# Call the main logic function from the repo file
sellin_season_df = main(
    PlanningItemCustomerGroup=PlanningItemCustomerGroup,
    DimTime=DimTime
)

# Put the resulting DataFrame back into the data lake
O9DataLake.put("Sellin Season", sellin_season_df)

logger.info("Successfully generated Sellin Season data.")
"""
Plugin : DP033OpportunityShortlist
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    Opportunities_input : Select ([Version].[Version Name] * &CurrentAndFuturePartialWeeks * &AllPlanningItem * &AllAccount * &AllChannel * &AllRegion * &AllPnL * &AllDemandDomain * &AllPlanningLocation * &AllOpportunity ) on row, ({Measure.[Opportunity Probability], Measure.[Opportunity Stage], Measure.[Opportunity Sales Type], Measure.[Opportunity Must Win]}) on column;
    Parameters : Select ([Version].[Version Name]) on row, ({Measure.[Probability Threshold], Measure.[Must Win Options], Measure.[List of Opportunity Stages], Measure.[List of Opportunity Types]}) on column;

Output Variables:
    output

Slice Dimension Attributes: None

"""

import logging

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP033OpportunityShortlist import main

logger = logging.getLogger("o9_logger")


# Function Calls
opportunities = O9DataLake.get("Opportunities_input")
Parameters = O9DataLake.get("Parameters")


# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

output = main(
    opportunities=opportunities,
    Parameters=Parameters,
    df_keys=df_keys,
)

O9DataLake.put("output", output)

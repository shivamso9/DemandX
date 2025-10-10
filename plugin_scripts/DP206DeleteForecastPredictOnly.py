"""
    Plugin : DP206DeleteForecastPredictOnly
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

    Input Queries:
        Assortment: Externalized table "045 Planning Item Customer Group"
        Forecast : Externalized table "511 HML Fcst"

    Output Variables:
        Output : Externalized table "511 HML Fcst"

    Slice Dimension Attributes: None

"""

import logging

from o9_common_utils.O9DataLake import O9DataLake

from helpers.pyspark.DP206DeleteForecastPredictOnly import main

logger = logging.getLogger("o9_logger")

# Function Calls
Assortment = O9DataLake.get("Assortment")
Forecast = O9DataLake.get("Forecast")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

Output = main(Assortment, Forecast, spark)

O9DataLake.put("Output", Output)

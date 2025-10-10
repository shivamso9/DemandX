"""Pyspark code for DP245DPHandShakeRateofSale_Pyspark."""

import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, StructField, StructType

from helpers.o9Constants import o9Constants
from helpers.o9PySparkConstants import o9PySparkConstants

# Initialize logging
logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    """Constants for DPAP Handshake ROS."""

    ACCOUNT_PLANNINGACCOUNT = o9PySparkConstants.PLANNING_ACCOUNT
    REGION_PLANNINGREGION = o9PySparkConstants.PLANNING_REGION
    PNL_PLANNINGPNL = o9PySparkConstants.PLANNING_PNL
    FORECASTITERATION = o9PySparkConstants.FORECAST_ITERATION
    VERSION_NAME = o9PySparkConstants.VERSION_NAME
    ITEM_PLANNINGITEM = o9PySparkConstants.PLANNING_ITEM
    LOCATION_PLANNINGLOCATION = o9PySparkConstants.PLANNING_LOCATION
    DEMAND_DOMAIN = o9PySparkConstants.PLANNING_DEMAND_DOMAIN
    HML_RATE_OF_SALE = o9Constants.HML_RATE_OF_SALE
    CHANNEL_PLANNINGCHANNEL = o9PySparkConstants.PLANNING_CHANNEL
    Intro_Date = o9Constants.INTRO_DATE
    Disco_Date = o9Constants.DISCO_DATE
    HML_SEASONALITY_ORIGIN = o9Constants.HML_SEASONALITY_ORIGIN
    HML_SEASONALITY_ORIGIN_VALUES = o9Constants.HML_SEASONALITY_ORIGIN_VALUES

    GROUP_COLUMNS = [
        VERSION_NAME,
        ITEM_PLANNINGITEM,
        CHANNEL_PLANNINGCHANNEL,
        LOCATION_PLANNINGLOCATION,
        ACCOUNT_PLANNINGACCOUNT,
        PNL_PLANNINGPNL,
        REGION_PLANNINGREGION,
    ]

    OUTPUT_COLUMNS = GROUP_COLUMNS + [
        FORECASTITERATION,
        DEMAND_DOMAIN,
        HML_RATE_OF_SALE,
        HML_SEASONALITY_ORIGIN,
        HML_SEASONALITY_ORIGIN_VALUES,
    ]


@log_inputs_and_outputs
def main(
    hml: DataFrame,
    item_customer_group: DataFrame,
    spark,
):
    """
    Implemetation for DP245DPHandshakeRateofSale_Pyspark.

    hml (DataFrame): Input DataFrame containing HML data.
    item_customer_group (DataFrame): Input DataFrame for Planning Item Customer Group.

    """
    plugin_name = "DP245DPHandshakeRateofSale_Pyspark"
    logger.info(f"Executing {plugin_name} ...")
    rename_dict = {
        Constants.HML_RATE_OF_SALE + " Input": Constants.HML_RATE_OF_SALE,
        Constants.HML_SEASONALITY_ORIGIN + " Input": Constants.HML_SEASONALITY_ORIGIN,
        Constants.HML_SEASONALITY_ORIGIN_VALUES + " Input": Constants.HML_SEASONALITY_ORIGIN_VALUES,
    }
    try:
        hml_df = col_namer.convert_to_pyspark_cols(hml)
        item_customer_group_df = col_namer.convert_to_pyspark_cols(item_customer_group).drop(
            Constants.Intro_Date, Constants.Disco_Date
        )
        if hml_df.rdd.isEmpty() or item_customer_group_df.rdd.isEmpty():
            raise Exception("Input Cannot be Empty..")

        hml_rate_of_sale = hml_df.join(
            item_customer_group_df, on=Constants.GROUP_COLUMNS, how="left"
        ).dropDuplicates()

        for old, new in rename_dict.items():
            hml_rate_of_sale = hml_rate_of_sale.withColumnRenamed(old, new)
        hml_rate_of_sale = hml_rate_of_sale.select(Constants.OUTPUT_COLUMNS)
        hml_rate_of_sale = col_namer.convert_to_o9_cols(hml_rate_of_sale)
    except Exception as e:
        logger.exception(f"An unexpected error occurred {e}", exc_info=True)
        try:
            schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.OUTPUT_COLUMNS]
            )
            hml_rate_of_sale = col_namer.convert_to_o9_cols(
                spark.createDataFrame([], schema=schema)
            )
            return hml_rate_of_sale
        except Exception as inner_e:
            logger.exception(f"Primary schema creation failed: {inner_e}")
            return spark.createDataFrame([], schema=None)
    return hml_rate_of_sale

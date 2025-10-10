"""Pyspark code for DP245APHandShakeRateofSale_Pyspark."""

import logging
from functools import reduce

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType

from helpers.o9Constants import o9Constants
from helpers.o9PySparkConstants import o9PySparkConstants

# Initialize logging
logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    """Constants for AP Handshake ROS."""

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
    TARGET_VERSION_NAME = o9Constants.VERSION_NAME
    TARGET_PLANNING_ITEM = o9Constants.PLANNING_ITEM
    TARGET_SELLING_SEASON = o9Constants.SELLING_SEASON
    TARGET_LOCATION = o9Constants.LOCATION
    TARGET_AGGREGATED_FORECAST = o9Constants.DP_ROS
    TARGET_SALES_DOMAIN = o9Constants.PLANNING_SALES_DEMAND
    LOCATION = o9PySparkConstants.LOCATION
    CLUSTER = o9PySparkConstants.CLUSTER
    ASSORTED_SC_L = o9PySparkConstants.ASSORTED_SC_L
    SELLING_SEASON = o9PySparkConstants.SELLING_SEASON
    SALES_DOMAIN = o9PySparkConstants.PLANNING_SALES_DEMAND
    TARGET_CLUSTER = o9Constants.CLUSTER
    HML_SEASONALITY_ORIGIN = o9Constants.HML_SEASONALITY_ORIGIN
    HML_SEASONALITY_ORIGIN_VALUES = o9Constants.HML_SEASONALITY_ORIGIN_VALUES

    OUTPUT_COLUMNS = [
        VERSION_NAME,
        ITEM_PLANNINGITEM,
        TARGET_LOCATION,
        TARGET_SELLING_SEASON,
        TARGET_SALES_DOMAIN,
        TARGET_CLUSTER,
        TARGET_AGGREGATED_FORECAST,
    ]

    ROS_OUTPUT_COLUMNS = [
        VERSION_NAME,
        ITEM_PLANNINGITEM,
        DEMAND_DOMAIN,
        LOCATION_PLANNINGLOCATION,
        CHANNEL_PLANNINGCHANNEL,
        FORECASTITERATION,
        ACCOUNT_PLANNINGACCOUNT,
        REGION_PLANNINGREGION,
        PNL_PLANNINGPNL,
        HML_RATE_OF_SALE,
        HML_SEASONALITY_ORIGIN,
        HML_SEASONALITY_ORIGIN_VALUES,
    ]


@log_inputs_and_outputs
def main(
    HML: DataFrame,
    Selling_Season_week_association: DataFrame,
    Ranging: DataFrame,
    spark,
):
    """
    Process input dataframes using PySpark to generate aggregated forecast results.

    Args:
        HML (DataFrame): Input DataFrame containing HML data.
        Selling_Season_week_association (DataFrame): Input DataFrame
        containing selling season associations.

    Returns:
        DataFrame: Processed DataFrame with renamed columns and ordered output.
    """
    plugin_name = "DP245DPAPHandshakeRateofSale_Pyspark"
    logger.info(f"Executing {plugin_name} ...")

    column_rename_mapping = {
        "Version": Constants.TARGET_VERSION_NAME,
        "Item": Constants.TARGET_PLANNING_ITEM,
        "Demand": Constants.TARGET_SELLING_SEASON,
        "Location": Constants.TARGET_LOCATION,
        "Channel": Constants.TARGET_SALES_DOMAIN,
        "Cluster": Constants.TARGET_CLUSTER,
        Constants.HML_RATE_OF_SALE + " Input": Constants.TARGET_AGGREGATED_FORECAST,
    }

    ros_column_rename_mapping = {
        "Version": Constants.VERSION_NAME,
        "Item": Constants.ITEM_PLANNINGITEM,
        "Demand": Constants.DEMAND_DOMAIN,
        "Location": Constants.LOCATION_PLANNINGLOCATION,
        "Channel": Constants.CHANNEL_PLANNINGCHANNEL,
        "Forecast": Constants.FORECASTITERATION,
        "Account": Constants.ACCOUNT_PLANNINGACCOUNT,
        "Region": Constants.REGION_PLANNINGREGION,
        "Pnl": Constants.PNL_PLANNINGPNL,
        Constants.HML_RATE_OF_SALE + " Input": Constants.HML_RATE_OF_SALE,
        Constants.HML_SEASONALITY_ORIGIN + " Input": Constants.HML_SEASONALITY_ORIGIN,
        Constants.HML_SEASONALITY_ORIGIN_VALUES + " Input": Constants.HML_SEASONALITY_ORIGIN_VALUES,
    }

    try:
        hml_df = col_namer.convert_to_pyspark_cols(HML)
        selling_season_df = col_namer.convert_to_pyspark_cols(Selling_Season_week_association)
        ranging_df = col_namer.convert_to_pyspark_cols(Ranging)
        if hml_df.rdd.isEmpty() or selling_season_df.rdd.isEmpty() or ranging_df.rdd.isEmpty():
            raise Exception("Input Cannot be Empty..")

        hml_df = hml_df.select(
            col(Constants.VERSION_NAME).alias("Version"),
            col(Constants.ITEM_PLANNINGITEM).alias("Item"),
            col(Constants.LOCATION_PLANNINGLOCATION).alias("Location"),
            col(Constants.ACCOUNT_PLANNINGACCOUNT).alias("Account"),
            col(Constants.PNL_PLANNINGPNL).alias("Pnl"),
            col(Constants.FORECASTITERATION).alias("Forecast"),
            col(Constants.CHANNEL_PLANNINGCHANNEL).alias("Channel"),
            col(Constants.REGION_PLANNINGREGION).alias("Region"),
            col(Constants.HML_RATE_OF_SALE + " Input").cast("double"),
            col(Constants.HML_SEASONALITY_ORIGIN + " Input"),
            col(Constants.HML_SEASONALITY_ORIGIN_VALUES + " Input"),
        )

        ranging_df = ranging_df.filter(ranging_df[Constants.ASSORTED_SC_L] == "true")

        ranging_df = ranging_df.select(
            col(Constants.VERSION_NAME).alias("Version"),
            col(Constants.ITEM_PLANNINGITEM).alias("Item"),
            col(Constants.LOCATION).alias("Location"),
            col(Constants.SELLING_SEASON).alias("Demand"),
            col(Constants.SALES_DOMAIN).alias("Channel"),
            col(Constants.CLUSTER).alias("Cluster"),
        )

        selling_season_df = selling_season_df.select(
            col(Constants.VERSION_NAME).alias("Version"),
            col(Constants.ITEM_PLANNINGITEM).alias("Item"),
            col(Constants.LOCATION_PLANNINGLOCATION).alias("Location"),
            col(Constants.DEMAND_DOMAIN).alias("Demand"),
            col(Constants.CHANNEL_PLANNINGCHANNEL).alias("Channel"),
            col(Constants.REGION_PLANNINGREGION).alias("Region"),
            col(Constants.PNL_PLANNINGPNL).alias("Pnl"),
            col(Constants.ACCOUNT_PLANNINGACCOUNT).alias("Account"),
        ).dropDuplicates()

        hml_selling_season_assortment = hml_df.join(
            selling_season_df,
            on=["Item", "Location", "Version", "Account", "Channel", "Pnl", "Region"],
            how="left",
        ).dropDuplicates()

        ros = reduce(
            lambda df, col_name: df.withColumnRenamed(
                col_name, ros_column_rename_mapping[col_name]
            ),
            ros_column_rename_mapping.keys(),
            hml_selling_season_assortment,
        )

        hml_selling_season_assortment = hml_selling_season_assortment.join(
            ranging_df, on=["Item", "Location", "Version", "Demand", "Channel"], how="left"
        ).dropDuplicates()
        hml_selling_season_assortment = hml_selling_season_assortment.drop(
            "Account",
            "Pnl",
            "Region",
            "Forecast",
            Constants.HML_SEASONALITY_ORIGIN + " Input",
            Constants.HML_SEASONALITY_ORIGIN_VALUES + " Input",
        )

        column_order = [
            col
            for col in hml_selling_season_assortment.columns
            if col != Constants.HML_RATE_OF_SALE + " Input"
        ] + [Constants.HML_RATE_OF_SALE + " Input"]
        hml_selling_season_assortment = hml_selling_season_assortment.select(column_order)
        aggregated_forecast = reduce(
            lambda df, col_name: df.withColumnRenamed(col_name, column_rename_mapping[col_name]),
            column_rename_mapping.keys(),
            hml_selling_season_assortment,
        )
        ros = ros.select(Constants.ROS_OUTPUT_COLUMNS)
        ros = col_namer.convert_to_o9_cols(df=ros)
        aggregated_hml_forecast = col_namer.convert_to_o9_cols(df=aggregated_forecast)
        logger.info("Data processing completed successfully.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred {e}", exc_info=True)
        try:
            schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.OUTPUT_COLUMNS]
            )
            aggregated_hml_forecast = col_namer.convert_to_o9_cols(
                spark.createDataFrame([], schema=schema)
            )
            ros_schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.ROS_OUTPUT_COLUMNS]
            )
            ros = col_namer.convert_to_o9_cols(spark.createDataFrame([], schema=ros_schema))
            return aggregated_hml_forecast, ros
        except Exception as inner_e:
            logger.exception(f"Primary schema creation failed: {inner_e}")
            return spark.createDataFrame([], schema=None), spark.createDataFrame([], schema=None)
    return aggregated_hml_forecast, ros

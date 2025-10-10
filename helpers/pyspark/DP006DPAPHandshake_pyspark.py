"""Pyspark code for DP006DPAPHandshake."""

import logging
from functools import reduce

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# Initialize logging
logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    """Constants for DPAP Handshake ROS."""

    ACCOUNT_PLANNINGACCOUNT = "Account_PlanningAccount"
    REGION_PLANNINGREGION = "Region_PlanningRegion"
    PNL_PLANNINGPNL = "PnL_PlanningPnL"
    VERSION_NAME = "Version_VersionName"
    TARGET_VERSION_NAME = "Version.[Version Name]"
    TARGET_PLANNING_ITEM = "Item.[Planning Item]"
    LOCATION = "Location_Location"
    SELLING_SEASON = "SellingSeason_SellingSeason"
    SALES_DOMAIN = "SalesDomain_SalesPlanningChannel"
    START_WEEK = "Start Week SC L AP"
    END_WEEK = "End Week SC L AP"
    INTRO_DATE = "Intro Date"
    DISCO_DATE = "Disco Date"
    TARGET_DEMAND_DOMAIN = "Demand Domain.[Planning Demand Domain]"
    TARGET_ACCOUNT_PLANNINGACCOUNT = "Account.[Planning Account]"
    TARGET_REGION_PLANNINGREGION = "Region.[Planning Region]"
    TARGET_PNL_PLANNINGPNL = "PnL.[Planning PnL]"
    TARGET_CHANNEL_PLANNINGCHANNEL = "Channel.[Planning Channel]"
    TARGET_LOCATION_PLANNINGLOCATION = "Location.[Planning Location]"
    CLUSTER = "Cluster_Cluster"
    ITEM_PLANNING_ITEM = "Item_PlanningItem"

    OUTPUT_COLUMNS = [
        TARGET_VERSION_NAME,
        TARGET_LOCATION_PLANNINGLOCATION,
        TARGET_PLANNING_ITEM,
        TARGET_ACCOUNT_PLANNINGACCOUNT,
        TARGET_REGION_PLANNINGREGION,
        TARGET_PNL_PLANNINGPNL,
        TARGET_CHANNEL_PLANNINGCHANNEL,
        TARGET_DEMAND_DOMAIN,
        INTRO_DATE,
        DISCO_DATE,
    ]


@log_inputs_and_outputs
def main(
    Ranging: DataFrame,
    PnlMaster: DataFrame,
    RegionMaster: DataFrame,
    AccountMaster: DataFrame,
    spark,
):
    """
    Process input dataframes using PySpark to generate Planning item customer group results.

    Args:
        `Ranging` (DataFrame): Input DataFrame containing ranging data.
        `PnlMaster` (DataFrame): DataFrame containing PnL master data.
        `RegionMaster` (DataFrame): DataFrame containing region master data.
        `AccountMaster` (DataFrame): DataFrame containing account master data.

    Returns:
        DataFrame: Processed DataFrame with renamed columns and ordered output.
    """
    plugin_name = "DP006DPAPHandshakepyspark"
    logger.info(f"Executing {plugin_name} ...")
    column_rename_mapping = {
        Constants.VERSION_NAME: Constants.TARGET_VERSION_NAME,
        Constants.SALES_DOMAIN: Constants.TARGET_CHANNEL_PLANNINGCHANNEL,
        Constants.SELLING_SEASON: Constants.TARGET_DEMAND_DOMAIN,
        Constants.LOCATION: Constants.TARGET_LOCATION_PLANNINGLOCATION,
        Constants.ACCOUNT_PLANNINGACCOUNT: Constants.TARGET_ACCOUNT_PLANNINGACCOUNT,
        Constants.REGION_PLANNINGREGION: Constants.TARGET_REGION_PLANNINGREGION,
        Constants.PNL_PLANNINGPNL: Constants.TARGET_PNL_PLANNINGPNL,
        Constants.START_WEEK: Constants.INTRO_DATE,
        Constants.END_WEEK: Constants.DISCO_DATE,
        Constants.ITEM_PLANNING_ITEM: Constants.TARGET_PLANNING_ITEM,
    }
    try:
        if (
            Ranging.rdd.isEmpty()
            or PnlMaster.rdd.isEmpty()
            or RegionMaster.rdd.isEmpty()
            or AccountMaster.rdd.isEmpty()
        ):
            raise Exception("Input Cannot be Empty..")

        ranging_df = col_namer.convert_to_pyspark_cols(Ranging)
        ranging_df = ranging_df.drop(Constants.CLUSTER)
        pnl_value = PnlMaster.first()[0]
        account_value = AccountMaster.first()[0]
        region_value = RegionMaster.first()[0]
        planning_item_cutsomer_group = (
            ranging_df.withColumn(Constants.ACCOUNT_PLANNINGACCOUNT, F.lit(account_value))
            .withColumn(Constants.PNL_PLANNINGPNL, F.lit(pnl_value))
            .withColumn(Constants.REGION_PLANNINGREGION, F.lit(region_value))
        )
        column_order = [
            col
            for col in planning_item_cutsomer_group.columns
            if col not in [Constants.START_WEEK, Constants.END_WEEK]
        ] + [Constants.START_WEEK, Constants.END_WEEK]
        planning_item_cutsomer_group = planning_item_cutsomer_group.select(column_order)

        planning_item_cutsomer_group = reduce(
            lambda df, col_name: df.withColumnRenamed(col_name, column_rename_mapping[col_name]),
            column_rename_mapping.keys(),
            planning_item_cutsomer_group,
        )
        logger.info("Successfully executed the plugin.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred {e}", exc_info=True)
        try:
            schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.OUTPUT_COLUMNS]
            )
            planning_item_cutsomer_group = spark.createDataFrame([], schema=schema)

            return planning_item_cutsomer_group
        except Exception as inner_e:
            logger.exception(f"Primary schema creation failed: {inner_e}")
            return spark.createDataFrame([], schema=None)

    return planning_item_cutsomer_group

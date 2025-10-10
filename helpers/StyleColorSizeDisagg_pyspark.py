# For local infoging only
# import logging
# logging.basicConfig(level=logging.INFO)
#
# from pyspark.sql import SparkSession
#
# spark = (
#     SparkSession.builder.master("local[*]")
#     .appName("APDPHandshake_pyspark")
#     .getOrCreate()
# )
#
# For local infoging only
import logging

import pandas as pd
import pyspark
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.functions import col as fcol
from pyspark.sql.functions import sum as fsum
from pyspark.sql.types import (
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


def check_one_to_one_mapping(df, col_name):
    # Check for duplicates in the specified column
    duplicates_df = df.groupBy(fcol(col_name)).count().filter(fcol("count") > 1)

    # Check if there are any duplicates
    is_one_to_one = duplicates_df.count() != 0

    return is_one_to_one, duplicates_df


def log_dataframe(df: pyspark.sql.dataframe) -> None:
    logger.info("------ dataframe head (5) --------")
    logger.info(f"Head : {df.take(5)}")
    logger.info(f"Schema : {df.schema}")
    logger.info(f"Shape : ({df.count()}, {len(df.columns)})")


def main(
    SizeRatioDP,
    LocationMapping,
    ItemMapping,
    ConsensusFcst,
    Grains,
    spark,
    df_keys,
):
    plugin_name = "StyleColorSizeDisagg_pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version_VersionName"
    pl_location_col = "Location_PlanningLocation"
    location_col = "Location_Location"
    pl_item_col = "Item_PlanningItem"
    item_col = "Item_Item"
    pl_channel_col = "Channel_PlanningChannel"
    pl_pnl_col = "PnL_PlanningPnL"
    pl_region_col = "Region_PlanningRegion"
    pl_account_col = "Account_PlanningAccount"
    pl_demand_domain_col = "DemandDomain_PlanningDemandDomain"

    partial_week_col = "Time_PartialWeek"

    size_ratio_dp = "Size Ratio DP"
    size_ratio_dp_sum = "Size Ratio DP Sum"
    consensus_fcst = "Consensus Fcst"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    output_o9_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    output_o9_grains = [x.strip() for x in output_o9_grains]

    # combine grains to get granular level
    output_o9_grains = [str(x) for x in output_o9_grains if x != "NA" and x != ""]

    output_grains = [get_clean_string(grain) for grain in Grains.split(",")]

    # output measures
    final_fcst = "Final Fcst"

    cols_required_in_output = [version_col] + output_grains + [partial_week_col, final_fcst]

    # to get col_mapping for all columns
    output_df = pd.DataFrame(columns=output_o9_grains)
    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    output_df = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        # Convert o9 columns to pyspark columns
        output_df = col_namer.convert_to_pyspark_cols(output_df)  # no use
        SizeRatioDP = col_namer.convert_to_pyspark_cols(SizeRatioDP)
        LocationMapping = col_namer.convert_to_pyspark_cols(LocationMapping)
        ItemMapping = col_namer.convert_to_pyspark_cols(ItemMapping)
        ConsensusFcst = col_namer.convert_to_pyspark_cols(ConsensusFcst)

        req_cols = [
            pl_location_col,
            location_col,
        ]
        LocationMapping = LocationMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            pl_item_col,
            item_col,
        ]
        ItemMapping = ItemMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            pl_channel_col,
            pl_pnl_col,
            pl_region_col,
            pl_account_col,
            pl_demand_domain_col,
            pl_location_col,
            pl_item_col,
            partial_week_col,
            consensus_fcst,
        ]
        ConsensusFcst = ConsensusFcst.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            pl_channel_col,
            pl_account_col,
            pl_region_col,
            pl_pnl_col,
            location_col,
            item_col,
            size_ratio_dp,
        ]
        SizeRatioDP = SizeRatioDP.select(*req_cols).dropDuplicates()

        logger.info("Location Mapping ...")
        log_dataframe(LocationMapping)
        logger.info("Size Ratio DP ...")
        log_dataframe(SizeRatioDP)
        logger.info("Consensus Fcst ...")
        log_dataframe(ConsensusFcst)

        if (
            (ItemMapping.count() == 0)
            or (LocationMapping.count() == 0)
            or (SizeRatioDP.count() == 0)
            or (ConsensusFcst.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        SizeRatioDP = SizeRatioDP.dropna(subset=[size_ratio_dp])
        if SizeRatioDP.count() == 0:
            raise Exception(
                "Size Ratio DP is empty after dropping NAs in Size Ratio DP. Exiting without execution."
            )

        ConsensusFcst = ConsensusFcst.dropna(subset=[consensus_fcst])
        if ConsensusFcst.count() == 0:
            raise Exception(
                "Consensus Fcst is empty after dropping NAs in Consensus Fcst. Exiting without execution."
            )

        # checking duplicates in Location
        is_duplicate, duplicates_df = check_one_to_one_mapping(LocationMapping, pl_location_col)

        if is_duplicate:
            logger.warning("Duplicates found in LocationMapping ...")
            logger.warning("Duplicate values ...")
            logger.warning(duplicates_df)
            raise Exception("Exiting without execution.")

        # get planning item column
        SizeRatioDP = SizeRatioDP.join(
            ItemMapping,
            on=[item_col],
            how="inner",
        )
        if SizeRatioDP.count() == 0:
            raise Exception(
                "No planning item present for item in SizeRatioDP. Exiting without execution."
            )

        # get item and location column
        ConsensusFcst = ConsensusFcst.join(
            ItemMapping,
            on=[pl_item_col],
            how="inner",
        )
        if ConsensusFcst.count() == 0:
            raise Exception(
                "No item present for planning item in ConsensusFcst. Exiting without execution."
            )

        ConsensusFcst = ConsensusFcst.join(
            LocationMapping,
            on=[pl_location_col],
            how="inner",
        )
        if ConsensusFcst.count() == 0:
            raise Exception(
                "No location present for planning location in ConsensusFcst. Exiting without execution."
            )

        # getting column for group by
        group_by_cols = [x for x in SizeRatioDP.columns if ((x != size_ratio_dp) & (x != item_col))]
        window_spec = Window().partitionBy(group_by_cols)
        SizeRatioDP = SizeRatioDP.withColumn(
            size_ratio_dp_sum, fsum(fcol(size_ratio_dp)).over(window_spec)
        )
        SizeRatioDP = SizeRatioDP.withColumn(
            size_ratio_dp, fcol(size_ratio_dp) / fcol(size_ratio_dp_sum)
        )

        # merge SizeRatioDP and ConsensusFcst
        common_cols = list(set(ConsensusFcst.columns).intersection(set(SizeRatioDP.columns)))
        ConsensusFcst = ConsensusFcst.join(
            SizeRatioDP,
            on=common_cols,
            how="inner",
        )
        if ConsensusFcst.count() == 0:
            raise Exception(
                "No common intersections between SizeRatioDP and ConsensusFcst. Exiting without execution."
            )

        ConsensusFcst = ConsensusFcst.withColumn(
            final_fcst, fcol(size_ratio_dp) * fcol(consensus_fcst)
        )

        output = ConsensusFcst.select(*cols_required_in_output).dropDuplicates()
        output = col_namer.convert_to_o9_cols(output)
        log_dataframe(output)

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(df_keys)
        )
        logger.exception(e)
        size_ratio_dp_schema_Output = (
            StructType()
            .add("Version.[Version Name]", StringType(), True)
            .add("Region.[Planning Region]", StringType(), True)
            .add("Item.[Item]", StringType(), True)
            .add("PnL.[Planning PnL]", StringType(), True)
            .add("Location.[Location]", StringType(), True)
            .add("Account.[Planning Account]", StringType(), True)
            .add("Channel.[Planning Channel]", StringType(), True)
            .add("Demand Domain.[Planning Demand Domain]", StringType(), True)
            .add("Time.[Partial Week]", TimestampType(), True)
            .add("Size Ratio DP", FloatType(), True)
        )

        emp_RDD = spark.sparkContext.emptyRDD()
        output = spark.createDataFrame(data=emp_RDD, schema=size_ratio_dp_schema_Output)

    return output

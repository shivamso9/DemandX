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
from pyspark.sql.functions import countDistinct, lit
from pyspark.sql.types import FloatType, StringType, StructField, StructType

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
    SizeRatio,
    ItemMapping,
    ChannelMapping,
    StyleColorPAGAssociation,
    StyleColorSizeRangeAssociation,
    Grains,
    spark,
    df_keys,
):
    plugin_name = "SizeProfilerDP_pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version_VersionName"
    prod_attr_grp_col = "ProductAttributeGroup_ProductAttributeGroup"
    item_l3_col = "Item_L3"
    location_col = "Location_Location"
    sales_planning_channel_col = "SalesDomain_SalesPlanningChannel"
    size_range_col = "SizeRange_SizeRange"
    size_col = "Size_Size"
    pl_item_col = "Item_PlanningItem"
    item_size_col = "Item_Size"
    item_col = "Item_Item"
    planning_channel_col = "Channel_PlanningChannel"

    size_ratio = "Approved Size Ratio"
    style_color_pag_association = "Style Color PAG Association"
    style_color_size_range_association = "Style Color Size Range Association"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    output_o9_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    output_o9_grains = [x.strip() for x in output_o9_grains]

    # combine grains to get granular level
    output_o9_grains = [str(x) for x in output_o9_grains if x != "NA" and x != ""]

    output_grains = [get_clean_string(grain) for grain in Grains.split(",")]

    # output measures
    size_ratio_dp = "Size Ratio DP"

    cols_required_in_output = [version_col] + output_grains + [size_ratio_dp]

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
        SizeRatio = col_namer.convert_to_pyspark_cols(SizeRatio)
        ItemMapping = col_namer.convert_to_pyspark_cols(ItemMapping)
        ChannelMapping = col_namer.convert_to_pyspark_cols(ChannelMapping)
        StyleColorPAGAssociation = col_namer.convert_to_pyspark_cols(StyleColorPAGAssociation)
        StyleColorSizeRangeAssociation = col_namer.convert_to_pyspark_cols(
            StyleColorSizeRangeAssociation
        )

        req_cols = [
            pl_item_col,
            item_col,
            item_size_col,
            item_l3_col,
        ]
        ItemMapping = ItemMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            planning_channel_col,
        ]
        ChannelMapping = ChannelMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            item_l3_col,
            prod_attr_grp_col,
            location_col,
            sales_planning_channel_col,
            size_range_col,
            size_col,
            size_ratio,
        ]
        SizeRatio = SizeRatio.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            pl_item_col,
            prod_attr_grp_col,
            style_color_pag_association,
        ]
        StyleColorPAGAssociation = StyleColorPAGAssociation.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            size_col,
            size_range_col,
            pl_item_col,
            style_color_size_range_association,
        ]
        StyleColorSizeRangeAssociation = StyleColorSizeRangeAssociation.select(
            *req_cols
        ).dropDuplicates()

        logger.info("Item Mapping ...")
        log_dataframe(ItemMapping)
        logger.info("Channel Mapping ...")
        log_dataframe(ChannelMapping)
        logger.info("Size Ratio ...")
        log_dataframe(SizeRatio)
        logger.info("Style Color PAG Association ...")
        log_dataframe(StyleColorPAGAssociation)
        logger.info("Style Color Size Range Association ...")
        log_dataframe(StyleColorSizeRangeAssociation)

        if (
            (SizeRatio.count() == 0)
            or (StyleColorSizeRangeAssociation.count() == 0)
            or (StyleColorPAGAssociation.count() == 0)
            or (ItemMapping.count() == 0)
            or (ChannelMapping.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        SizeRatio = SizeRatio.dropna(subset=[size_ratio])
        if SizeRatio.count() == 0:
            raise Exception(
                "SizeRatio is empty after dropping NAs in Size Ratio. Exiting without execution."
            )

        StyleColorSizeRangeAssociation = StyleColorSizeRangeAssociation.dropna(
            subset=[style_color_size_range_association]
        )
        if StyleColorSizeRangeAssociation.count() == 0:
            raise Exception(
                "StyleColorSizeRangeAssociation is empty after dropping NAs in Style Color Size Range Association. Exiting without execution."
            )

        StyleColorPAGAssociation = StyleColorPAGAssociation.dropna(
            subset=[style_color_pag_association]
        )
        if StyleColorPAGAssociation.count() == 0:
            raise Exception(
                "StyleColorPAGAssociation is empty after dropping NAs in Style Color PAG Association. Exiting without execution."
            )

        # getting item column
        StyleColorSizeRangeAssociation = StyleColorSizeRangeAssociation.withColumnRenamed(
            size_col, item_size_col
        )
        StyleColorSizeRangeAssociation = StyleColorSizeRangeAssociation.join(
            ItemMapping,
            on=[pl_item_col, item_size_col],
            how="inner",
        )
        if StyleColorSizeRangeAssociation.count() == 0:
            raise Exception(
                "No item present for planning item, size combination. Exiting without execution."
            )

        # getting product attribute group
        StyleColorSizeRangeAssociation = StyleColorSizeRangeAssociation.join(
            StyleColorPAGAssociation,
            on=[version_col, pl_item_col],
            how="inner",
        )
        if StyleColorSizeRangeAssociation.count() == 0:
            raise Exception("No product attribute group found. Exiting without execution.")

        SizeRatio = SizeRatio.withColumnRenamed(size_col, item_size_col)

        merge_cols = [
            version_col,
            item_l3_col,
            prod_attr_grp_col,
            size_range_col,
            item_size_col,
        ]
        SizeRatio = SizeRatio.join(
            StyleColorSizeRangeAssociation,
            on=merge_cols,
            how="inner",
        )
        if SizeRatio.count() == 0:
            raise Exception("No common intersections. Exiting without execution.")

        pl_channel_values = (
            ChannelMapping.select(planning_channel_col).toPandas()[planning_channel_col].tolist()
        )

        logger.info("checking for available sales planning channel values ...")
        # get sales planning channel values which are not present
        is_all_sales_pl_channel_values = (
            SizeRatio.filter(~fcol(sales_planning_channel_col).isin(pl_channel_values)).count() == 0
        )

        not_present_count = (
            SizeRatio.filter(~fcol(sales_planning_channel_col).isin(pl_channel_values))
            .agg(countDistinct(sales_planning_channel_col).alias("count_unique_values_not_present"))
            .collect()[0]["count_unique_values_not_present"]
        )

        if not is_all_sales_pl_channel_values:
            logger.warning(
                "Not all values from Sales Planning Channel present in Planning Channel ..."
            )
            logger.warning(
                "Out of {} Sales Planning Channel unique values, {} is/are not present in Planning Channel ...".format(
                    SizeRatio.select(
                        fcol(sales_planning_channel_col).alias(sales_planning_channel_col)
                    )
                    .distinct()
                    .count(),
                    not_present_count,
                )
            )
            raise Exception("Exiting without execution.")

        col_mapping = {
            sales_planning_channel_col: planning_channel_col,
            size_ratio: size_ratio_dp,
        }
        for old_col, new_col in col_mapping.items():
            SizeRatio = SizeRatio.withColumnRenamed(old_col, new_col)

        for col in output_grains:
            if col not in SizeRatio.columns:
                SizeRatio = SizeRatio.withColumn(col, lit("All"))

        output = SizeRatio.select(*cols_required_in_output)
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
            .add("Size Ratio DP", FloatType(), True)
        )

        emp_RDD = spark.sparkContext.emptyRDD()
        output = spark.createDataFrame(data=emp_RDD, schema=size_ratio_dp_schema_Output)

    return output

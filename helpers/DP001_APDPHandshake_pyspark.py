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
from pyspark.sql import Window
from pyspark.sql.functions import col as fcol
from pyspark.sql.functions import countDistinct, lit
from pyspark.sql.functions import max as fmax
from pyspark.sql.functions import min as fmin
from pyspark.sql.functions import to_timestamp, when
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

col_namer = ColumnNamer()

logger = logging.getLogger("o9_logger")


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
    AssortmentOutputGrains,
    DateOutputGrains,
    AssortmentPlanning,
    ItemMapping,
    LocationMapping,
    DemandDomainMapping,
    ChannelMapping,
    StyleColorSizeAssociation,
    spark,
    df_keys,
):
    plugin_name = "APDPHandshake_pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version_VersionName"
    cluster_col = "Cluster_Cluster"
    sales_planning_channel_col = "SalesDomain_SalesPlanningChannel"
    size_col = "Size_Size"
    item_size_col = "Item_Size"
    item_col = "Item_Item"
    size_range_col = "SizeRange_SizeRange"
    selling_season_col = "SellingSeason_SellingSeason"
    pl_item_col = "Item_PlanningItem"
    location_col = "Location_Location"
    pl_location_col = "Location_PlanningLocation"
    planning_channel_col = "Channel_PlanningChannel"
    planning_demand_domain_col = "DemandDomain_PlanningDemandDomain"

    start_week_sc = "Start Week SC L"
    end_week_sc = "End Week SC L"
    assortment = "Assorted SC L"
    size_color_association = "Style Color Size Range Association"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    assortment_output_o9_grains = AssortmentOutputGrains.split(",")
    dates_output_o9_grains = DateOutputGrains.split(",")

    # remove leading/trailing spaces if any
    assortment_output_o9_grains = [x.strip() for x in assortment_output_o9_grains]
    dates_output_o9_grains = [x.strip() for x in dates_output_o9_grains]

    # combine grains to get granular level
    assortment_output_o9_grains = [
        str(x) for x in assortment_output_o9_grains if x != "NA" and x != ""
    ]
    dates_output_o9_grains = [str(x) for x in dates_output_o9_grains if x != "NA" and x != ""]

    assortment_output_grains = [
        get_clean_string(grain) for grain in AssortmentOutputGrains.split(",")
    ]
    date_output_grains = [get_clean_string(grain) for grain in DateOutputGrains.split(",")]

    # output measures
    assortment_final = "Assortment Final"
    intro_date = "Intro Date"
    disco_date = "Disco Date"

    cols_required_in_assortment_output = (
        [version_col] + assortment_output_grains + [assortment_final]
    )
    cols_required_in_dates_output = [version_col] + date_output_grains + [intro_date, disco_date]

    # to get col_mapping for all columns
    assortment_df = pd.DataFrame(
        columns=list(set(assortment_output_o9_grains).union(set(dates_output_o9_grains)))
    )
    assortment_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in assortment_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    assortment_df = spark.createDataFrame(data=emp_RDD, schema=assortment_df_schema)

    try:
        # Convert o9 columns to pyspark columns
        assortment_df = col_namer.convert_to_pyspark_cols(assortment_df)  # no use
        AssortmentPlanning = col_namer.convert_to_pyspark_cols(AssortmentPlanning)
        ItemMapping = col_namer.convert_to_pyspark_cols(ItemMapping)
        LocationMapping = col_namer.convert_to_pyspark_cols(LocationMapping)
        DemandDomainMapping = col_namer.convert_to_pyspark_cols(DemandDomainMapping)
        ChannelMapping = col_namer.convert_to_pyspark_cols(ChannelMapping)
        StyleColorSizeAssociation = col_namer.convert_to_pyspark_cols(StyleColorSizeAssociation)

        req_cols = [pl_item_col, item_col, item_size_col]
        ItemMapping = ItemMapping.select(*req_cols).dropDuplicates()

        req_cols = [location_col, pl_location_col]
        LocationMapping = LocationMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_demand_domain_col]
        DemandDomainMapping = DemandDomainMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_channel_col]
        ChannelMapping = ChannelMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            sales_planning_channel_col,
            pl_item_col,
            location_col,
            cluster_col,
            selling_season_col,
            start_week_sc,
            end_week_sc,
            assortment,
        ]
        AssortmentPlanning = AssortmentPlanning.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            size_col,
            pl_item_col,
            size_range_col,
            size_color_association,
        ]
        StyleColorSizeAssociation = StyleColorSizeAssociation.select(*req_cols).dropDuplicates()

        logger.info("Item Mapping ...")
        log_dataframe(ItemMapping)
        logger.info("Location Mapping ...")
        log_dataframe(LocationMapping)
        logger.info("Channel Mapping ...")
        log_dataframe(ChannelMapping)
        logger.info("Demand Domain Mapping ...")
        log_dataframe(DemandDomainMapping)
        logger.info("Assortment Planning ...")
        log_dataframe(AssortmentPlanning)
        logger.info("Style Color Size Association ...")
        log_dataframe(StyleColorSizeAssociation)

        if AssortmentPlanning.count() == 0:
            raise Exception("AssortmentPlanning input have no data. Exiting without execution.")

        if StyleColorSizeAssociation.count() == 0:
            raise Exception(
                "StyleColorSizeAssociation input have no data. Exiting without execution."
            )

        common_grains = list(
            set(cols_required_in_assortment_output).intersection(set(cols_required_in_dates_output))
        )

        logger.info("Checking for duplicates in LocationMapping")
        # checking duplicates in Location
        is_duplicate, duplicates_df = check_one_to_one_mapping(LocationMapping, pl_location_col)

        if is_duplicate:
            logger.warning("Duplicates found in LocationMapping ...")
            logger.warning("Duplicate values ...")
            logger.warning(duplicates_df)
            raise Exception("Exiting without execution.")

        # getting item column
        StyleColorSizeAssociation = StyleColorSizeAssociation.withColumnRenamed(
            size_col, item_size_col
        )
        StyleColorSizeAssociation = StyleColorSizeAssociation.join(
            ItemMapping,
            on=[pl_item_col, item_size_col],
            how="inner",
        )

        if StyleColorSizeAssociation.count() == 0:
            raise Exception(
                "No item present for planning item, size combination. Exiting without execution."
            )

        # get available selling season and sales planning channel values
        selling_season_values = (
            DemandDomainMapping.select(planning_demand_domain_col)
            .toPandas()[planning_demand_domain_col]
            .tolist()
        )
        pl_channel_values = (
            ChannelMapping.select(planning_channel_col).toPandas()[planning_channel_col].tolist()
        )

        logger.info("checking for available selling season values ...")
        # get selling season values which are not present
        is_all_selling_season_values = (
            AssortmentPlanning.filter(~fcol(selling_season_col).isin(selling_season_values)).count()
            == 0
        )

        not_present_count = (
            AssortmentPlanning.filter(~fcol(selling_season_col).isin(selling_season_values))
            .agg(countDistinct(selling_season_col).alias("count_unique_values_not_present"))
            .collect()[0]["count_unique_values_not_present"]
        )

        if not is_all_selling_season_values:
            logger.warning("Not all values from Selling Season present in Demand Domain ...")
            logger.warning(
                "Out of {} Season unique values, {} is/are not present in Demand Domain ...".format(
                    AssortmentPlanning.select(fcol(selling_season_col).alias(selling_season_col))
                    .distinct()
                    .count(),
                    not_present_count,
                )
            )
            raise Exception("Exiting without execution.")

        logger.info("checking for available sales planning channel values ...")
        # get sales planning channel values which are not present
        is_all_sales_pl_channel_values = (
            AssortmentPlanning.filter(
                ~fcol(sales_planning_channel_col).isin(pl_channel_values)
            ).count()
            == 0
        )

        not_present_count = (
            AssortmentPlanning.filter(~fcol(sales_planning_channel_col).isin(pl_channel_values))
            .agg(countDistinct(sales_planning_channel_col).alias("count_unique_values_not_present"))
            .collect()[0]["count_unique_values_not_present"]
        )

        if not is_all_sales_pl_channel_values:
            logger.warning(
                "Not all values from Sales Planning Channel present in Planning Channel ..."
            )
            logger.warning(
                "Out of {} Sales Planning Channel unique values, {} is/are not present in Planning Channel ...".format(
                    AssortmentPlanning.select(
                        fcol(sales_planning_channel_col).alias(sales_planning_channel_col)
                    )
                    .distinct()
                    .count(),
                    not_present_count,
                )
            )
            raise Exception("Exiting without execution.")

        group_by_cols = [
            pl_item_col,
            sales_planning_channel_col,
            location_col,
            cluster_col,
            selling_season_col,
        ]

        # Define a Window specification based on the group_by_cols
        window_spec = Window.partitionBy(*group_by_cols)

        # getting min of start date
        AssortmentPlanning = AssortmentPlanning.withColumn(
            start_week_sc, fmin(fcol(start_week_sc)).over(window_spec)
        )

        # getting max of end date
        AssortmentPlanning = AssortmentPlanning.withColumn(
            end_week_sc, fmax(fcol(end_week_sc)).over(window_spec)
        )

        # drop cluster column
        AssortmentPlanning.drop(cluster_col)
        AssortmentPlanning.dropDuplicates()

        AssortmentPlanning = AssortmentPlanning.withColumn(
            assortment, when(fcol(assortment), "1").otherwise("0")
        )

        # getting item
        AssortmentPlanning = AssortmentPlanning.join(
            StyleColorSizeAssociation,
            how="inner",
            on=[version_col, pl_item_col],
        )
        if AssortmentPlanning.count() == 0:
            raise Exception(
                "No common planning item present for AssortmentPlanning and StyleColorSizeAssociation. Exiting without execution."
            )

        # getting planning location
        AssortmentPlanning = AssortmentPlanning.join(
            LocationMapping,
            how="inner",
            on=location_col,
        )
        if AssortmentPlanning.count() == 0:
            raise Exception(
                "No common location present for AssortmentPlanning and LocationMapping. Exiting without execution."
            )

        col_mapping = {
            sales_planning_channel_col: planning_channel_col,
            selling_season_col: planning_demand_domain_col,
            start_week_sc: intro_date,
            end_week_sc: disco_date,
            assortment: assortment_final,
        }
        for old_col, new_col in col_mapping.items():
            AssortmentPlanning = AssortmentPlanning.withColumnRenamed(old_col, new_col)

        for col in common_grains:
            if col not in AssortmentPlanning.columns:
                AssortmentPlanning = AssortmentPlanning.withColumn(col, lit("All"))

        Assortment = AssortmentPlanning.select(*cols_required_in_assortment_output).dropDuplicates()
        Dates = AssortmentPlanning.select(*cols_required_in_dates_output).dropDuplicates()

        Assortment = col_namer.convert_to_o9_cols(Assortment)
        Dates = col_namer.convert_to_o9_cols(Dates)

        Assortment = Assortment.withColumn(
            assortment_final, Assortment[assortment_final].cast(DoubleType())
        )

        Dates = Dates.withColumn(
            intro_date, to_timestamp(intro_date, "M/d/yyyy h:mm:ss a")
        ).withColumn(disco_date, to_timestamp(disco_date, "M/d/yyyy h:mm:ss a"))

        log_dataframe(Assortment)
        log_dataframe(Dates)

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(df_keys)
        )
        logger.exception(e)
        assortment_schema_Output = (
            StructType()
            .add("Version.[Version Name]", StringType(), True)
            .add("Region.[Planning Region]", StringType(), True)
            .add("Item.[Item]", StringType(), True)
            .add("PnL.[Planning PnL]", StringType(), True)
            .add("Location.[Location]", StringType(), True)
            .add("Demand Domain.[Planning Demand Domain]", StringType(), True)
            .add("Account.[Planning Account]", StringType(), True)
            .add("Channel.[Planning Channel]", StringType(), True)
            .add("Assortment Final", DoubleType(), True)
        )
        dates_schema_Output = (
            StructType()
            .add("Version.[Version Name]", StringType(), True)
            .add("Region.[Planning Region]", StringType(), True)
            .add("Item.[Planning Item]", StringType(), True)
            .add("PnL.[Planning PnL]", StringType(), True)
            .add("Location.[Planning Location]", StringType(), True)
            .add("Demand Domain.[Planning Demand Domain]", StringType(), True)
            .add("Account.[Planning Account]", StringType(), True)
            .add("Channel.[Planning Channel]", StringType(), True)
            .add("Intro Date", TimestampType(), True)
            .add("Disco Date", TimestampType(), True)
        )
        emp_RDD = spark.sparkContext.emptyRDD()
        Assortment = spark.createDataFrame(data=emp_RDD, schema=assortment_schema_Output)
        Dates = spark.createDataFrame(data=emp_RDD, schema=dates_schema_Output)

    return Assortment, Dates

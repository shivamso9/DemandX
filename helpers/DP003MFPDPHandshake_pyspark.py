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
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

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
    Grains,
    OTB,
    DemandDomainMapping,
    ChannelMapping,
    AccountMapping,
    RegionMapping,
    PnLMapping,
    spark,
    df_keys,
):
    plugin_name = "MFPDPHandshake_pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version_VersionName"
    l3_col = "Item_L3"
    location_country_col = "Location_LocationCountry"
    sales_planning_channel_col = "SalesDomain_SalesPlanningChannel"
    selling_season_col = "SellingSeason_SellingSeason"
    planning_channel_col = "Channel_PlanningChannel"
    planning_demand_domain_col = "DemandDomain_PlanningDemandDomain"
    planning_region_col = "Region_PlanningRegion"
    planning_account_col = "Account_PlanningAccount"
    planning_pnl_col = "PnL_PlanningPnL"

    week_col = "Time_Week"
    otb_sales = "OTB Ttl Sls Unt RP"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    output_o9_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    output_o9_grains = [x.strip() for x in output_o9_grains]

    # combine grains to get granular level
    output_o9_grains = [str(x) for x in output_o9_grains if x != "NA" and x != ""]

    output_grains = [get_clean_string(grain) for grain in Grains.split(",")]

    # output measures
    mfp_units = "MFP Units"

    cols_required_in_output = [version_col] + output_grains + [week_col, mfp_units]

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
        OTB = col_namer.convert_to_pyspark_cols(OTB)
        DemandDomainMapping = col_namer.convert_to_pyspark_cols(DemandDomainMapping)
        ChannelMapping = col_namer.convert_to_pyspark_cols(ChannelMapping)
        AccountMapping = col_namer.convert_to_pyspark_cols(AccountMapping)
        RegionMapping = col_namer.convert_to_pyspark_cols(RegionMapping)
        PnLMapping = col_namer.convert_to_pyspark_cols(PnLMapping)

        req_cols = [planning_demand_domain_col]
        DemandDomainMapping = DemandDomainMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_channel_col]
        ChannelMapping = ChannelMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_account_col]
        AccountMapping = AccountMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_region_col]
        RegionMapping = RegionMapping.select(*req_cols).dropDuplicates()

        req_cols = [planning_pnl_col]
        PnLMapping = PnLMapping.select(*req_cols).dropDuplicates()

        req_cols = [
            version_col,
            l3_col,
            location_country_col,
            sales_planning_channel_col,
            selling_season_col,
            week_col,
            otb_sales,
        ]
        OTB = OTB.select(*req_cols).dropDuplicates()

        logger.info("Channel Mapping ...")
        log_dataframe(ChannelMapping)
        logger.info("Demand Domain Mapping ...")
        log_dataframe(DemandDomainMapping)
        logger.info("OTB ...")
        log_dataframe(OTB)

        if (
            (OTB.count() == 0)
            or (ChannelMapping.count() == 0)
            or (DemandDomainMapping.count() == 0)
            or (AccountMapping.count() == 0)
            or (RegionMapping.count() == 0)
            or (PnLMapping.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        OTB = OTB.dropna(subset=[otb_sales])
        if OTB.count() == 0:
            raise Exception(
                "OTB is empty after dropping NAs in OTB Ttl Sales. Exiting without execution."
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
            OTB.filter(~fcol(selling_season_col).isin(selling_season_values)).count() == 0
        )

        not_present_count = (
            OTB.filter(~fcol(selling_season_col).isin(selling_season_values))
            .agg(countDistinct(selling_season_col).alias("count_unique_values_not_present"))
            .collect()[0]["count_unique_values_not_present"]
        )

        if not is_all_selling_season_values:
            logger.warning("Not all values from Selling Season present in Demand Domain ...")
            logger.warning(
                "Out of {} Season unique values, {} is/are not present in Demand Domain ...".format(
                    OTB.select(fcol(selling_season_col).alias(selling_season_col))
                    .distinct()
                    .count(),
                    not_present_count,
                )
            )
            raise Exception("Exiting without execution.")

        logger.info("checking for available sales planning channel values ...")
        # get sales planning channel values which are not present
        is_all_sales_pl_channel_values = (
            OTB.filter(~fcol(sales_planning_channel_col).isin(pl_channel_values)).count() == 0
        )

        not_present_count = (
            OTB.filter(~fcol(sales_planning_channel_col).isin(pl_channel_values))
            .agg(countDistinct(sales_planning_channel_col).alias("count_unique_values_not_present"))
            .collect()[0]["count_unique_values_not_present"]
        )

        if not is_all_sales_pl_channel_values:
            logger.warning(
                "Not all values from Sales Planning Channel present in Planning Channel ..."
            )
            logger.warning(
                "Out of {} Sales Planning Channel unique values, {} is/are not present in Planning Channel ...".format(
                    OTB.select(fcol(sales_planning_channel_col).alias(sales_planning_channel_col))
                    .distinct()
                    .count(),
                    not_present_count,
                )
            )
            raise Exception("Exiting without execution.")

        # will return empty dataframe if more than 1 values in AccountMapping, RegionMapping and PnLMapping
        if (
            (AccountMapping.count() != 1)
            or (RegionMapping.count() != 1)
            or (PnLMapping.count() != 1)
        ):
            raise Exception(
                "AccountMapping/RegionMapping/PnLMapping has more than 1 values. Please check. Exiting without execution."
            )

        # getting default values for account, region and pnl
        default_account_value = AccountMapping.select(planning_account_col).first()[0]
        default_region_value = RegionMapping.select(planning_region_col).first()[0]
        default_pnl_value = PnLMapping.select(planning_pnl_col).first()[0]

        col_mapping = {
            sales_planning_channel_col: planning_channel_col,
            selling_season_col: planning_demand_domain_col,
            otb_sales: mfp_units,
        }
        for old_col, new_col in col_mapping.items():
            OTB = OTB.withColumnRenamed(old_col, new_col)

        OTB = (
            OTB.withColumn(planning_account_col, lit(default_account_value))
            .withColumn(planning_region_col, lit(default_region_value))
            .withColumn(planning_pnl_col, lit(default_pnl_value))
        )

        Output = OTB.select(*cols_required_in_output).dropDuplicates()

        Output = col_namer.convert_to_o9_cols(Output)

        Output = Output.withColumn(mfp_units, Output[mfp_units].cast(DoubleType()))

        log_dataframe(Output)

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(df_keys)
        )
        logger.exception(e)
        schema_Output = (
            StructType()
            .add("Version.[Version Name]", StringType(), True)
            .add("Region.[Planning Region]", StringType(), True)
            .add("Item.[L3]", StringType(), True)
            .add("PnL.[Planning PnL]", StringType(), True)
            .add("Location.[Location Country]", StringType(), True)
            .add("Demand Domain.[Planning Demand Domain]", StringType(), True)
            .add("Account.[Planning Account]", StringType(), True)
            .add("Channel.[Planning Channel]", StringType(), True)
            .add("Time.[Week]", StringType(), True)
            .add("MFP Units", DoubleType(), True)
        )
        emp_RDD = spark.sparkContext.emptyRDD()
        Output = spark.createDataFrame(data=emp_RDD, schema=schema_Output)

    return Output

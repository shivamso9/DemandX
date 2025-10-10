import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore
from o9Reference.spark_utils.common_utils import get_clean_string  # type: ignore
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.window import Window

from helpers.o9Constants import o9Constants
from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.utils import get_list_of_grains_from_string

# Initialize logging
logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    """Constants for column names."""

    VERSION_NAME = o9PySparkConstants.VERSION_NAME
    PLANNING_ITEM = o9PySparkConstants.PLANNING_ITEM
    PLANNING_CHANNEL = o9PySparkConstants.PLANNING_CHANNEL
    PLANNING_LOCATION = o9PySparkConstants.PLANNING_LOCATION
    PLANNING_PNL = o9PySparkConstants.PLANNING_PNL
    PLANNING_REGION = o9PySparkConstants.PLANNING_REGION
    SEQUENCE = o9PySparkConstants.SEQUENCE
    PLANNING_ACCOUNT = o9PySparkConstants.PLANNING_ACCOUNT
    FORECAST_ITERATION = o9PySparkConstants.FORECAST_ITERATION
    PLANNING_DOMAINDEMAND = o9PySparkConstants.PLANNING_DEMAND_DOMAIN
    TIME_WEEK = o9PySparkConstants.WEEK
    TIME_WEEKKEY = o9PySparkConstants.WEEK_KEY
    TIME_WEEKNAME = o9PySparkConstants.TIME_WEEKNAME
    TIME_WEEKNAMEKEY = o9PySparkConstants.TIME_WEEKNAMEKEY
    TIME_PARTIALWEEK = o9PySparkConstants.PARTIAL_WEEK
    HML_SEASONALITY = o9Constants.HML_SEASONALITY
    SELLING_SEASON = o9PySparkConstants.SELLING_SEASON
    TARGET_SELLING_SEASON = o9Constants.SELLING_SEASON
    SEASONALITY_CURVE = o9PySparkConstants.SEASONALITY_CURVE
    ASSORTED_SC_L = o9PySparkConstants.ASSORTED_SC_L
    LOCATION = o9PySparkConstants.LOCATION
    SALES_DOMAIN = o9PySparkConstants.PLANNING_SALES_DEMAND
    CLUSTER = o9PySparkConstants.CLUSTER
    TARGET_CLUSTER = o9Constants.CLUSTER

    JOIN_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        PLANNING_CHANNEL,
        PLANNING_LOCATION,
        PLANNING_ACCOUNT,
        PLANNING_PNL,
        PLANNING_REGION,
        SEQUENCE,
    ]
    CLUSTER_JOIN_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        LOCATION,
        SELLING_SEASON,
        SALES_DOMAIN,
    ]

    GROUP_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        PLANNING_CHANNEL,
        PLANNING_LOCATION,
        PLANNING_ACCOUNT,
        PLANNING_PNL,
        PLANNING_REGION,
        SEQUENCE,
        FORECAST_ITERATION,
        PLANNING_DOMAINDEMAND,
        TIME_WEEK,
        TIME_WEEKKEY,
        TIME_WEEKNAME,
        TIME_WEEKNAMEKEY,
    ]

    OUTPUT_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        PLANNING_CHANNEL,
        PLANNING_LOCATION,
        PLANNING_ACCOUNT,
        PLANNING_PNL,
        PLANNING_REGION,
        FORECAST_ITERATION,
        PLANNING_DOMAINDEMAND,
        TIME_PARTIALWEEK,
        HML_SEASONALITY,
    ]


@log_inputs_and_outputs
def main(
    HML_Seasonality,
    Selling_Season_week_association,
    MasterDataDict: dict,
    TimeKey,
    TargetGrain,
    Ranging,
    spark,
):
    plugin_name = "D240APDPHandshakeSeasonalitycurve"
    logger.info("Executing {} ...".format(plugin_name))
    is_ap_process = Ranging is not None
    try:
        HML_Seasonality = col_namer.convert_to_pyspark_cols(HML_Seasonality)
        Selling_Season_week_association = col_namer.convert_to_pyspark_cols(
            Selling_Season_week_association
        )
        TimeKey = col_namer.convert_to_pyspark_cols(TimeKey)

        if (
            HML_Seasonality.rdd.isEmpty()
            or Selling_Season_week_association.rdd.isEmpty()
            or TimeKey.rdd.isEmpty()
            or (is_ap_process and Ranging.rdd.isEmpty())
        ):
            raise Exception("Input Cannot be Empty..")

        Selling_Season_week_association = Selling_Season_week_association.join(
            TimeKey, on=Constants.TIME_PARTIALWEEK, how="inner"
        )

        # Extract the week number and create a new column
        Selling_Season_week_association = Selling_Season_week_association.withColumn(
            Constants.SEQUENCE,
            F.regexp_extract(F.col(Constants.TIME_WEEKNAME), r"(\d+)", 0).cast("int"),
        )
        hml_seasonality_df = HML_Seasonality.join(
            Selling_Season_week_association,
            on=Constants.JOIN_COLUMNS,
            how="inner",
        )

        # Define the columns that should be grouped
        # Add a window function to count the duplicates for each group
        window_spec = Window.partitionBy(Constants.GROUP_COLUMNS)
        # Count the number of rows in each group
        hml_seasonality_df = hml_seasonality_df.withColumn(
            "duplicate_count", F.count("*").over(window_spec)
        )
        # Divide the HML_Seasonality Index equally among the duplicates
        hml_seasonality_df = hml_seasonality_df.withColumn(
            Constants.HML_SEASONALITY,
            F.col("HML Seasonality Input") / F.col("duplicate_count"),
        )
        # Drop the temporary 'duplicate_count' column
        hml_seasonality_df = hml_seasonality_df.drop("duplicate_count")

        hml_seasonality_final_output_df = hml_seasonality_df.select(*Constants.OUTPUT_COLUMNS)

        hml_seasonality_df = hml_seasonality_df.withColumnRenamed(
            Constants.PLANNING_DOMAINDEMAND, Constants.SELLING_SEASON
        )
        source_grains = hml_seasonality_df.columns
        logger.debug(f"source_grains : {source_grains}")

        # join input with master data
        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)

            # identify common column if any
            the_common_columns = list(
                set(hml_seasonality_df.columns).intersection(the_master_data.columns)
            )
            logger.debug(f"the_key : {the_key}, common_columns : {the_common_columns}")
            if the_common_columns:
                the_join_col = the_common_columns[0]
                logger.debug(f"the_join_col : {the_join_col}")

                hml_seasonality_df = hml_seasonality_df.join(
                    the_master_data, on=the_join_col, how="inner"
                )

        if is_ap_process:
            target_grains = get_list_of_grains_from_string(input=TargetGrain)
            target_grains = [get_clean_string(x) for x in target_grains]
            logger.debug(f"target_grains : {target_grains}")
            # check if all target grains are present in Input after join
            target_grains = [x for x in target_grains if x != Constants.CLUSTER]
            missing_columns = [x for x in target_grains if x not in hml_seasonality_df.columns]
            logger.debug(f"missing_columns : {missing_columns}")

            if missing_columns:
                raise ValueError(
                    f"missing columns : {missing_columns}, please ensure 'Master' dataframes are"
                    "supplied for all of these ..."
                )

            Output = hml_seasonality_df.groupBy(*target_grains).agg(
                F.sum(Constants.HML_SEASONALITY).alias(Constants.SEASONALITY_CURVE)
            )
            ranging_df = col_namer.convert_to_pyspark_cols(Ranging)
            ranging_df = ranging_df.filter(ranging_df[Constants.ASSORTED_SC_L] == "true")
            ranging_df = ranging_df.drop(Constants.ASSORTED_SC_L)
            Output = Output.join(
                ranging_df, on=Constants.CLUSTER_JOIN_COLUMNS, how="left"
            ).dropDuplicates()
            Output = Output.select(*target_grains, Constants.CLUSTER, Constants.SEASONALITY_CURVE)
            output_seasonality_curve_df = col_namer.convert_to_o9_cols(df=Output)
            output_seasonality_curve_df = output_seasonality_curve_df.withColumnRenamed(
                Constants.SELLING_SEASON, Constants.TARGET_SELLING_SEASON
            )
        hml_seasonality_final_output_df = col_namer.convert_to_o9_cols(
            df=hml_seasonality_final_output_df
        )
        if is_ap_process:
            return hml_seasonality_final_output_df, output_seasonality_curve_df
        else:
            return hml_seasonality_final_output_df
    except Exception as execption:
        logger.error(f"Exception Occured: {execption}")
        try:
            schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.OUTPUT_COLUMNS]
            )
            hml_seasonality_final_output_df = col_namer.convert_to_o9_cols(
                spark.createDataFrame([], schema=schema)
            )
            if is_ap_process:
                schema = StructType(
                    [
                        StructField(col, StringType(), True)
                        for col in get_list_of_grains_from_string(input=TargetGrain)
                        + [Constants.TARGET_CLUSTER, Constants.SEASONALITY_CURVE]
                    ]
                )
                output_seasonality_curve_df = spark.createDataFrame([], schema=schema)
                return hml_seasonality_final_output_df, output_seasonality_curve_df
            else:
                return hml_seasonality_final_output_df
        except Exception as inner_e:
            logger.exception(f"Primary schema creation failed: {inner_e}")
            if is_ap_process:
                return spark.createDataFrame([], schema=None), spark.createDataFrame(
                    [], schema=None
                )
            else:
                return spark.createDataFrame([], schema=None)

import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.types import StringType, StructField, StructType

from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.utils import get_list_of_grains_from_string

# Initialize logging
logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    VERSION_NAME = o9PySparkConstants.VERSION_NAME
    PLANNING_ITEM = o9PySparkConstants.PLANNING_ITEM
    TIME_WEEK = o9PySparkConstants.WEEK
    SELLING_SEASON = o9PySparkConstants.SELLING_SEASON
    ASSORTED_SC_L = o9PySparkConstants.ASSORTED_SC_L
    LOCATION = o9PySparkConstants.LOCATION
    SALES_DOMAIN = o9PySparkConstants.SALES_DOMAIN_SALES_PLANNING_CHANNEL
    CLUSTER = o9PySparkConstants.CLUSTER
    WEEK_FORECAST_QUANTITY = o9PySparkConstants.WEEKLY_FORCAST

    CLUSTER_JOIN_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        LOCATION,
        SELLING_SEASON,
        SALES_DOMAIN,
    ]

    OUTPUT_COLUMNS = [
        VERSION_NAME,
        PLANNING_ITEM,
        LOCATION,
        SELLING_SEASON,
        SALES_DOMAIN,
        TIME_WEEK,
        CLUSTER,
        WEEK_FORECAST_QUANTITY,
    ]


@log_inputs_and_outputs
def main(
    SourceMeasures: str,
    SourceGrain: str,
    TargetMeasures: str,
    TargetGrain: str,
    Input,
    MasterDataDict: dict,
    Ranging,
    spark,
):
    plugin_name = "DP-AP Handshake"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        Input = col_namer.convert_to_pyspark_cols(Input)

        source_grains = get_list_of_grains_from_string(input=SourceGrain)
        source_grains = [get_clean_string(x) for x in source_grains]
        logger.debug(f"source_grains : {source_grains}")

        target_grains = get_list_of_grains_from_string(input=TargetGrain)
        target_grains = [get_clean_string(x) for x in target_grains]
        logger.debug(f"target_grains : {target_grains}")

        # join input with master data
        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)

            # identify common column if any
            the_common_columns = list(set(Input.columns).intersection(the_master_data.columns))
            logger.debug(f"the_key : {the_key}, common_columns : {the_common_columns}")
            if the_common_columns:
                the_join_col = the_common_columns[0]
                logger.debug(f"the_join_col : {the_join_col}")

                Input = Input.join(the_master_data, on=the_join_col, how="inner")

        source_measures = get_list_of_grains_from_string(input=SourceMeasures)
        target_measures = get_list_of_grains_from_string(input=TargetMeasures)

        input_to_output_measure_dict = dict(zip(source_measures, target_measures))

        logger.debug(f"input_to_output_measure_dict : {input_to_output_measure_dict}")

        # check if all target grains are present in Input after join
        target_grains = [x for x in target_grains if x != Constants.CLUSTER]
        missing_columns = [x for x in target_grains if x not in Input.columns]
        logger.debug(f"missing_columns : {missing_columns}")

        if missing_columns:
            raise ValueError(
                f"missing columns : {missing_columns}, please ensure 'Master' dataframes are supplied for all of these ..."
            )

        Input = Input.dropDuplicates()
        Output = Input.groupBy(*target_grains).sum()

        rename_mapping = dict(zip([f"sum({x})" for x in source_measures], target_measures))

        logger.debug(f"rename_mapping : {rename_mapping}")
        for old_name, new_name in rename_mapping.items():
            Output = Output.withColumnRenamed(old_name, new_name)
        ranging_df = col_namer.convert_to_pyspark_cols(Ranging)
        ranging_df = ranging_df.filter(ranging_df[Constants.ASSORTED_SC_L] == "true")
        ranging_df = ranging_df.drop(Constants.ASSORTED_SC_L)
        Output = Output.join(
            ranging_df, on=Constants.CLUSTER_JOIN_COLUMNS, how="left"
        ).dropDuplicates()

        Output = Output.select(Constants.OUTPUT_COLUMNS)
        Output = col_namer.convert_to_o9_cols(df=Output)

    except Exception as e:
        logger.error(f"Exception Occured: {e}")
        try:
            schema = StructType(
                [StructField(col, StringType(), True) for col in Constants.OUTPUT_COLUMNS]
            )
            Output = col_namer.convert_to_o9_cols(spark.createDataFrame([], schema=schema))
            return Output
        except Exception as inner_e:
            logger.exception(f"Primary schema creation failed: {inner_e}")
            return spark.createDataFrame([], schema=None)
    return Output

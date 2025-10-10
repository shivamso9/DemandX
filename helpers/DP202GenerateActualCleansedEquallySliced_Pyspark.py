import logging

from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.functions import col, lit, monotonically_increasing_id, row_number
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window

from helpers.o9Constants import o9Constants
from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.utils import (
    get_list_of_grains_from_string,
    list_of_cols_from_pyspark_schema,
)

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(StatGrains, NumIntersectionsInOneSlice, ActualCleansed, spark):
    plugin_name = "DP202GenerateActualCleansedEquallySliced_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))

    # TODO : Reference this from o9Constants
    ACTUAL_CLEANSED = "Actual Cleansed"
    ACTUAL_CLEANSED_EQUALLY_SLICED = "Actual Cleansed Sliced"

    # TODO : Reference this from StatGrains
    # Define the schema for the empty DataFrame
    ActualCleansedEquallySliced_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.FORECAST_ITERATION,
                StringType(),
                nullable=True,
            ),
            StructField(
                o9PySparkConstants.SEQUENCE,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.STAT_REGION, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_LOCATION, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_CHANNEL, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_PNL, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_ITEM, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.STAT_DEMAND_DOMAIN,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.STAT_ACCOUNT, StringType(), nullable=True),
            StructField(o9PySparkConstants.PARTIAL_WEEK, StringType(), nullable=True),
            StructField(ACTUAL_CLEANSED_EQUALLY_SLICED, DoubleType(), nullable=True),
        ]
    )
    # Create an empty DataFrame with the specified schema
    ActualCleansedEquallySliced = spark.createDataFrame([], ActualCleansedEquallySliced_schema)

    SliceAssociationStat_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.FORECAST_ITERATION,
                StringType(),
                nullable=True,
            ),
            StructField(
                o9PySparkConstants.SEQUENCE,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.STAT_REGION, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_LOCATION, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_CHANNEL, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_PNL, StringType(), nullable=True),
            StructField(o9PySparkConstants.STAT_ITEM, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.STAT_DEMAND_DOMAIN,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.STAT_ACCOUNT, StringType(), nullable=True),
            StructField(o9Constants.SLICE_ASSOCIATION_STAT, DoubleType(), nullable=True),
        ]
    )

    SliceAssociationStat = spark.createDataFrame([], SliceAssociationStat_schema)
    index_col = "index"
    try:
        if ActualCleansed.rdd.isEmpty():
            raise AssertionError("ActualCleansed is not populated, it's a mandatory input ...")

        stat_grains = get_list_of_grains_from_string(input=StatGrains)
        stat_grains = [get_clean_string(x) for x in stat_grains]

        # add version and iteration
        stat_grains = [
            o9PySparkConstants.VERSION_NAME,
            o9PySparkConstants.FORECAST_ITERATION,
        ] + stat_grains

        logger.info(f"stat_grains : {stat_grains}")

        ActualCleansed = col_namer.convert_to_pyspark_cols(ActualCleansed)

        # collect stat intersections from actual cleansed
        stat_intersections = ActualCleansed.select(stat_grains).dropDuplicates()

        num_intersections_in_one_slice = int(NumIntersectionsInOneSlice)

        Totalintersections = stat_intersections.count()
        logger.info(f"Totalintersections : {Totalintersections}")

        if Totalintersections < num_intersections_in_one_slice:
            NumBuckets = 1
        else:
            # determine the number of slice buckets by diving (whole number division) total num of intersections by NumIntersectionsInOneSlice
            # add 1 to ensure that there's atleast one bucket
            NumBuckets = (Totalintersections // num_intersections_in_one_slice) + 1

        logger.info("NumBuckets : {}".format(NumBuckets))

        # add row number
        stat_intersections = stat_intersections.withColumn(
            index_col,
            row_number().over(Window.orderBy(monotonically_increasing_id())),
        )

        # add sequence
        stat_intersections = stat_intersections.withColumn(
            o9PySparkConstants.SEQUENCE, (col(index_col) % NumBuckets) + 1
        )

        # Convert integer column to string
        stat_intersections = stat_intersections.withColumn(
            o9PySparkConstants.SEQUENCE,
            col(o9PySparkConstants.SEQUENCE).cast(StringType()),
        )
        logger.info(f"stat_intersections, schema : {stat_intersections.schema}")

        # join back on original dataframe
        ActualCleansed = ActualCleansed.join(stat_intersections, on=stat_grains, how="inner")

        # rename column
        ActualCleansed = ActualCleansed.withColumnRenamed(
            ACTUAL_CLEANSED, ACTUAL_CLEANSED_EQUALLY_SLICED
        )

        new_column_order = list_of_cols_from_pyspark_schema(
            schema=ActualCleansedEquallySliced_schema
        )

        logger.info(f"new_column_order : {new_column_order}")

        # Reorder the DataFrame columns
        ActualCleansedEquallySliced = ActualCleansed.select(new_column_order)

        # Add slice association stat column
        stat_intersections = stat_intersections.withColumn(
            o9PySparkConstants.SLICE_ASSOCIATION_STAT,
            lit(1.0).cast(DoubleType()),
        )
        slice_association_stat_col_order = list_of_cols_from_pyspark_schema(
            schema=SliceAssociationStat_schema
        )
        SliceAssociationStat = stat_intersections.select(slice_association_stat_col_order)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(e)
    finally:
        ActualCleansedEquallySliced = col_namer.convert_to_o9_cols(df=ActualCleansedEquallySliced)
        ActualCleansedEquallySliced = ActualCleansedEquallySliced.withColumnRenamed(
            o9PySparkConstants.SEQUENCE,
            o9Constants.SEQUENCE,
        )

        SliceAssociationStat = col_namer.convert_to_o9_cols(df=SliceAssociationStat)
        SliceAssociationStat = SliceAssociationStat.withColumnRenamed(
            o9PySparkConstants.SEQUENCE,
            o9Constants.SEQUENCE,
        )
    return ActualCleansedEquallySliced, SliceAssociationStat

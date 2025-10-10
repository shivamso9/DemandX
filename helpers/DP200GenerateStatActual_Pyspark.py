import logging
from datetime import timedelta

from dateutil.relativedelta import relativedelta
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from o9Reference.spark_utils.fill_missing_dates import (
    fill_missing_dates,  # type: ignore
)
from pyspark.sql.functions import col, date_format, lit, lower
from pyspark.sql.functions import max as _max
from pyspark.sql.functions import min as _min
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import to_date
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from helpers.o9Constants import o9Constants
from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.utils import (
    get_list_of_grains_from_string,
    list_of_cols_from_pyspark_schema,
)

# from o9Reference.common_utils.decorators import (
#     map_output_columns_to_dtypes,  # type: ignore
# )

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
# col_mapping = {}


def get_iteration_from_mapping(input: str) -> str:
    # input : StatIteration = FI-Stat
    values = input.split("=")

    iteration = values[1]

    # trim leading/trailing spaces
    return iteration.strip()


@log_inputs_and_outputs
# @map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    spark,
    ForecastEngine,
    AssortmentStat,
    NSIActual,
    Actual,
    ItemMapping,
    AccountMapping,
    LocationMapping,
    RegionMapping,
    DemandDomainMapping,
    PnLMapping,
    ChannelMapping,
    TimeMapping,
    LeafGrains,
    StatGrains,
    IterationMapping,
    CurrentTimePeriod,
    # HistoryPeriod,
    # ForecastTimeBucket,
    ForecastIteration,
):
    plugin_name = "DP200GenerateStatActual_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))

    # TODO : Reference this from StatGrains
    # Define the schema for the empty DataFrame
    StatActual_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.FORECAST_ITERATION,
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
            StructField(o9Constants.STAT_ACTUAL, DoubleType(), nullable=True),
        ]
    )
    # Create an empty DataFrame with the specified schema
    StatActual = spark.createDataFrame([], StatActual_schema)

    ActualHML_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.FORECAST_ITERATION,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.REGION, StringType(), nullable=True),
            StructField(o9PySparkConstants.LOCATION, StringType(), nullable=True),
            StructField(o9PySparkConstants.CHANNEL, StringType(), nullable=True),
            StructField(o9PySparkConstants.PNL, StringType(), nullable=True),
            StructField(o9PySparkConstants.ITEM, StringType(), nullable=True),
            StructField(o9PySparkConstants.DEMAND_DOMAIN, StringType(), nullable=True),
            StructField(o9PySparkConstants.ACCOUNT, StringType(), nullable=True),
            StructField(o9PySparkConstants.DAY, StringType(), nullable=True),
            StructField(o9Constants.ACTUAL_HML, DoubleType(), nullable=True),
        ]
    )
    # Create an empty DataFrame with the specified schema
    ActualHML = spark.createDataFrame([], ActualHML_schema)

    ActualCleansed_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.FORECAST_ITERATION,
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
            StructField(o9Constants.ACTUAL_CLEANSED, DoubleType(), nullable=True),
        ]
    )
    # Create an empty DataFrame with the specified schema
    ActualCleansed = spark.createDataFrame([], ActualCleansed_schema)
    try:
        if ForecastEngine.rdd.isEmpty():
            raise AssertionError("ForecastEngine is not populated, it's a mandatory input ...")

        # if AssortmentStat.rdd.isEmpty():
        #     raise AssertionError(
        #         "AssortmentStat is not populated, it's a mandatory input ..."
        #     )

        if Actual.rdd.isEmpty():
            raise AssertionError("Actual is not populated, it's a mandatory input ...")

        fcst_engine = ForecastEngine.select(o9Constants.FORECAST_ENGINE_CORE_PRODUCTS).first()[
            o9Constants.FORECAST_ENGINE_CORE_PRODUCTS
        ]

        logger.debug(f"fcst_engine : {fcst_engine}")

        assert fcst_engine in [
            "Statistical",
            "HML",
        ], f"invalid entry {fcst_engine}"

        # collect stat and hml iteration
        # IterationMapping = StatIteration = FI-Stat, HMLIteration = FI-HML

        # split on comma, form 2 separate entities
        stat_iteration_mapping, hml_iteration_mapping = IterationMapping.split(",")

        stat_iteration = get_iteration_from_mapping(input=stat_iteration_mapping)
        hml_iteration = get_iteration_from_mapping(input=hml_iteration_mapping)

        logger.info(f"stat_iteration : {stat_iteration}")
        logger.info(f"hml_iteration : {hml_iteration}")

        Actual = col_namer.convert_to_pyspark_cols(Actual)
        NSIActual = col_namer.convert_to_pyspark_cols(NSIActual)
        ItemMapping = col_namer.convert_to_pyspark_cols(ItemMapping)
        AccountMapping = col_namer.convert_to_pyspark_cols(AccountMapping)
        LocationMapping = col_namer.convert_to_pyspark_cols(LocationMapping)
        RegionMapping = col_namer.convert_to_pyspark_cols(RegionMapping)
        DemandDomainMapping = col_namer.convert_to_pyspark_cols(DemandDomainMapping)
        PnLMapping = col_namer.convert_to_pyspark_cols(PnLMapping)
        ChannelMapping = col_namer.convert_to_pyspark_cols(ChannelMapping)
        TimeMapping = col_namer.convert_to_pyspark_cols(TimeMapping)
        AssortmentStat = col_namer.convert_to_pyspark_cols(AssortmentStat)

        if NSIActual.count() > 0:
            logger.debug("Appending NSI Actual to Actual ...")

            # rename column NSI Actual to Actual
            NSIActual = NSIActual.withColumnRenamed(o9Constants.NSI_ACTUAL, o9Constants.ACTUAL)

            # Appending df2 to df1
            # logger.debug(Actual.count())
            logger.debug(Actual.dtypes)
            # logger.debug(NSIActual.count())
            logger.debug(NSIActual.dtypes)
            logger.debug("Sum of Actuals : ")
            logger.debug(Actual.agg(_sum("Actual")).collect())
            logger.debug("Sum of NSI Actuals : ")
            logger.debug(NSIActual.agg(_sum("Actual")).collect())
            All_Actual = Actual.union(NSIActual)
        else:
            All_Actual = Actual

        ActualHML = All_Actual
        # logger.debug(All_Actual.count())
        logger.debug(All_Actual.dtypes)
        logger.debug("Sum of All Actuals : ")
        logger.debug(All_Actual.agg(_sum("Actual")).collect())

        if fcst_engine == "Statistical":
            # filter core products
            ItemMappingCore = ItemMapping.filter(
                lower(col(o9PySparkConstants.PRODUCT_INVENTORY_TYPE)) == "core"
            )

            # filter core products
            # Commenting this as it is not used anywhere now
            # ItemMappingNonCore = ItemMapping.filter(
            #     (col(o9PySparkConstants.PRODUCT_INVENTORY_TYPE) != "Core")
            #     | (col(o9PySparkConstants.PRODUCT_INVENTORY_TYPE).isNull())
            # )

            # Stat has core products or nothing
            # Case 1 : Stat has core, HML will have non core
            # Case 2 : Stat has nothing, HML will have all the products

            if ItemMappingCore.rdd.isEmpty():
                logger.warning("No 'Core' products found in ItemMapping")
            else:
                # join on assortment to filter relevant records
                relevant_assortment_core = (
                    ItemMappingCore.select(o9PySparkConstants.PLANNING_ITEM)
                    .distinct()
                    .join(
                        AssortmentStat,
                        on=o9PySparkConstants.PLANNING_ITEM,
                        how="inner",
                    )
                )

                if relevant_assortment_core.rdd.isEmpty():
                    logger.warning("No common records between ItemMapping and AssortmentStat")

                logger.info("Joining All_Actual with all master data ...")

                # add planning grains to actual
                All_Actual = All_Actual.join(
                    ItemMapping,
                    on=o9PySparkConstants.ITEM,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    AccountMapping,
                    on=o9PySparkConstants.ACCOUNT,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    RegionMapping,
                    on=o9PySparkConstants.REGION,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    DemandDomainMapping,
                    on=o9PySparkConstants.DEMAND_DOMAIN,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    PnLMapping,
                    on=o9PySparkConstants.PNL,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    ChannelMapping,
                    on=o9PySparkConstants.CHANNEL,
                    how="inner",
                )
                All_Actual = All_Actual.join(
                    LocationMapping,
                    on=o9PySparkConstants.LOCATION,
                    how="inner",
                )

                logger.info("Joining All_Actual with relevant_assortment_core ...")

                # join assortment with actual to retain relevant intersections
                relevant_actuals = All_Actual.join(
                    relevant_assortment_core,
                    on=[
                        o9PySparkConstants.VERSION_NAME,
                        o9PySparkConstants.PLANNING_ITEM,
                        o9PySparkConstants.PLANNING_ACCOUNT,
                        o9PySparkConstants.LOCATION,
                        o9PySparkConstants.PLANNING_REGION,
                        o9PySparkConstants.PLANNING_DEMAND_DOMAIN,
                        o9PySparkConstants.PLANNING_PNL,
                        o9PySparkConstants.PLANNING_CHANNEL,
                    ],
                    how="inner",
                )

                if relevant_actuals.rdd.isEmpty():
                    raise ValueError(
                        "No common records between All_Actual and relevant_assortment_core"
                    )

                # Join with time
                relevant_actuals = relevant_actuals.join(
                    TimeMapping, on=o9PySparkConstants.DAY, how="inner"
                )
                if relevant_actuals.rdd.isEmpty():
                    raise ValueError("No common records between relevant_actuals and TimeMapping")

                relevant_actuals = col_namer.convert_to_pyspark_cols(relevant_actuals)

                logger.debug(relevant_actuals.dtypes)
                cols_to_groupby = [
                    o9PySparkConstants.VERSION_NAME,
                    o9PySparkConstants.STAT_ITEM,
                    o9PySparkConstants.STAT_ACCOUNT,
                    o9PySparkConstants.STAT_LOCATION,
                    o9PySparkConstants.STAT_REGION,
                    o9PySparkConstants.STAT_DEMAND_DOMAIN,
                    o9PySparkConstants.STAT_PNL,
                    o9PySparkConstants.STAT_CHANNEL,
                    o9PySparkConstants.PARTIAL_WEEK,
                ]
                logger.info("Grouping by stat grains and aggregating values ...")

                logger.info(relevant_actuals.schema)
                logger.info(relevant_actuals.take(5))

                # Selecting specific columns and performing groupBy operation
                StatActual = relevant_actuals.groupBy(*cols_to_groupby).agg(
                    _sum(o9Constants.ACTUAL).alias(o9Constants.STAT_ACTUAL)
                )

                stat_grains = get_list_of_grains_from_string(input=StatGrains)
                stat_grains = [get_clean_string(x) for x in stat_grains]

                logger.debug(StatActual.dtypes)
                logger.debug("sum of relevant_actuals :")
                logger.debug(relevant_actuals.agg(_sum("Actual")).collect())
                logger.debug("sum of StatActual : ")
                logger.debug(StatActual.agg(_sum("Stat Actual")).collect())

                # create partial week mapping
                partial_week_mapping = TimeMapping.select(
                    o9PySparkConstants.PARTIAL_WEEK,
                    o9PySparkConstants.PARTIAL_WEEK_KEY,
                ).dropDuplicates()

                logger.debug(partial_week_mapping.show(5))

                # Change the datatype of the partial week mapping column before joining
                partial_week_mapping = partial_week_mapping.withColumn(
                    o9PySparkConstants.PARTIAL_WEEK,
                    to_date(col(o9PySparkConstants.PARTIAL_WEEK), "dd-MMM-yy"),
                ).withColumn(
                    o9PySparkConstants.PARTIAL_WEEK_KEY,
                    to_date(
                        col(o9PySparkConstants.PARTIAL_WEEK_KEY),
                        "M/d/yyyy h:mm:ss a",
                    ),
                )

                logger.debug(partial_week_mapping.dtypes)

                # Change the datatype of Partial Week col in StatActual before joining
                StatActual = StatActual.withColumn(
                    o9PySparkConstants.PARTIAL_WEEK,
                    to_date(col(o9PySparkConstants.PARTIAL_WEEK), "dd-MMM-yy"),
                )
                # add partial week key to stat actual
                StatActual = StatActual.join(
                    partial_week_mapping,
                    on=o9PySparkConstants.PARTIAL_WEEK,
                    how="inner",
                )
                logger.info(f"StatActual schema : {StatActual.schema}")

                # get min and max partial week key
                # Calculate the min and max date
                result_df = StatActual.agg(
                    _min(o9PySparkConstants.PARTIAL_WEEK_KEY).alias("min_date"),
                    _max(o9PySparkConstants.PARTIAL_WEEK_KEY).alias("max_date"),
                )

                # Filter for stat iteration
                ForecastIteration = col_namer.convert_to_pyspark_cols(ForecastIteration)
                StatForecastIteration = ForecastIteration.filter(
                    col(o9PySparkConstants.FORECAST_ITERATION) == stat_iteration
                )

                logger.debug(StatForecastIteration.show())

                ForecastBucket = StatForecastIteration.select(
                    o9PySparkConstants.FORECAST_GEN_TIME_BUCKET
                ).collect()[0][0]
                logger.debug(ForecastBucket)

                # Get current date
                current_date = (
                    col_namer.convert_to_pyspark_cols(CurrentTimePeriod)
                    .withColumn(
                        o9PySparkConstants.WEEK,
                        to_date(col(o9PySparkConstants.WEEK), "dd-MMM-yy"),
                    )
                    .select(o9PySparkConstants.WEEK)
                    .collect()[0][0]
                )
                logger.debug(current_date)

                # Get the value for Current date
                if ForecastBucket.lower() == "week":
                    HistoryPeriod = StatForecastIteration.select(
                        o9PySparkConstants.HISTORY_PERIOD
                    ).collect()[0][0]
                    date_n_weeks_ago = current_date - timedelta(weeks=int(HistoryPeriod))
                elif ForecastBucket.lower() == "month":
                    HistoryPeriod = StatForecastIteration.select(
                        o9PySparkConstants.HISTORY_PERIOD
                    ).collect()[0][0]
                    date_n_weeks_ago = current_date - relativedelta(months=int(HistoryPeriod))
                # Get Current Month
                elif ForecastBucket.lower() == "planning month":
                    HistoryPeriod = StatForecastIteration.select(
                        o9PySparkConstants.HISTORY_PERIOD
                    ).collect()[0][0]
                    date_n_weeks_ago = current_date - relativedelta(months=int(HistoryPeriod))
                logger.debug(date_n_weeks_ago)

                # remove key column
                StatActual = StatActual.drop(o9PySparkConstants.PARTIAL_WEEK_KEY)

                # Collect the results into two variables
                result = result_df.collect()[0]
                min_partial_week = result["min_date"]
                max_partial_week = result["max_date"]

                logger.info(f"min_partial_week : {min_partial_week}")
                logger.info(f"max_partial_week : {max_partial_week}")

                # collect all weeks between min and max into a list
                # Filter dates between min_date and max_date
                filtered_dates_df = partial_week_mapping.filter(
                    (col(o9PySparkConstants.PARTIAL_WEEK_KEY) >= date_n_weeks_ago)
                    & (col(o9PySparkConstants.PARTIAL_WEEK_KEY) < current_date)
                )

                # Collect the filtered dates into a Python list
                filtered_dates = [
                    row[o9PySparkConstants.PARTIAL_WEEK] for row in filtered_dates_df.collect()
                ]

                logger.debug(StatActual.dtypes)
                StatActual = StatActual.withColumn(
                    o9PySparkConstants.PARTIAL_WEEK,
                    to_date(col(o9PySparkConstants.PARTIAL_WEEK), "dd-MMM-yy"),
                )
                StatActual = StatActual.filter(
                    (col(o9PySparkConstants.PARTIAL_WEEK) >= date_n_weeks_ago)
                    & (col(o9PySparkConstants.PARTIAL_WEEK) < current_date)
                )
                logger.info(f"filtered_dates ({len(filtered_dates)}): {filtered_dates}")

                logger.info(f"shape before fill missing dates: {StatActual.count()}")
                # fill missing dates
                StatActual_nas_filled = fill_missing_dates(
                    spark_session=spark,
                    actual=StatActual,
                    forecast_level=stat_grains,
                    history_measure=o9Constants.STAT_ACTUAL,
                    relevant_time_name=o9PySparkConstants.PARTIAL_WEEK,
                    relevant_time_key=o9PySparkConstants.PARTIAL_WEEK_KEY,
                    relevant_time_periods=filtered_dates,
                    time_mapping=partial_week_mapping,
                    fill_nulls_with_zero=True,
                    filter_from_start_date=True,
                )
                logger.info(f"shape after fill missing dates: {StatActual_nas_filled.count()}")
                # convert StatActual to 'dd-LLL-YY' format
                StatActual = StatActual.withColumn(
                    o9PySparkConstants.PARTIAL_WEEK,
                    date_format(
                        col(o9PySparkConstants.PARTIAL_WEEK),
                        "dd-LLL-yy",
                    ),
                )

                StatActual_nas_filled = StatActual_nas_filled.withColumn(
                    o9PySparkConstants.PARTIAL_WEEK,
                    date_format(
                        col(o9PySparkConstants.PARTIAL_WEEK),
                        "dd-LLL-yy",
                    ),
                )
                ActualCleansed = StatActual_nas_filled.withColumnRenamed(
                    o9Constants.STAT_ACTUAL, o9Constants.ACTUAL_CLEANSED
                )
                # Add a new column iteration with value
                StatActual = StatActual.withColumn(
                    o9PySparkConstants.FORECAST_ITERATION, lit(stat_iteration)
                )
                ActualCleansed = ActualCleansed.withColumn(
                    o9PySparkConstants.FORECAST_ITERATION, lit(stat_iteration)
                )

            # join on non core to get the relevant items
            # ActualHML = All_Actual.join(
            #     ItemMappingNonCore, on=o9PySparkConstants.ITEM, how="inner"
            # )
            # push all records to Actual HML
            # ActualHML = All_Actual.withColumnRenamed(
            #    o9Constants.ACTUAL, o9Constants.ACTUAL_HML
            # )

        else:
            # HML
            # when engine is HML, stat Actual and Actual cleansed will be null

            # push all records to Actual HML
            logger.debug("Pass")
            # ActualHML = All_Actual.withColumnRenamed(
            #    o9Constants.ACTUAL, o9Constants.ACTUAL_HML
            # )

        ActualHML = ActualHML.withColumnRenamed(o9Constants.ACTUAL, o9Constants.ACTUAL_HML)

        # Add a new column iteration with value
        ActualHML = ActualHML.withColumn(o9PySparkConstants.FORECAST_ITERATION, lit(hml_iteration))

        # Extract column names into a list
        StatActual_cols = list_of_cols_from_pyspark_schema(schema=StatActual_schema)
        logger.info(f"StatActual_cols : {StatActual_cols}")
        ActualHML_cols = list_of_cols_from_pyspark_schema(schema=ActualHML_schema)
        logger.info(f"ActualHML_cols : {ActualHML_cols}")
        ActualCleansed_cols = list_of_cols_from_pyspark_schema(schema=ActualCleansed_schema)
        logger.info(f"ActualCleansed_cols : {ActualCleansed_cols}")

        # reorder columns
        StatActual = StatActual.select(StatActual_cols)
        ActualHML = ActualHML.select(ActualHML_cols)
        ActualCleansed = ActualCleansed.select(ActualCleansed_cols)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(e)
    finally:
        # logger.debug(StatActual.agg(_sum("Stat Actual")).collect())

        # logger.debug(ActualCleansed.agg(_sum("Actual Cleansed")).collect())
        # logger.debug(ActualHML.agg(_sum("Actual HML")).collect())

        StatActual = col_namer.convert_to_o9_cols(df=StatActual)
        ActualHML = col_namer.convert_to_o9_cols(df=ActualHML)
        ActualCleansed = col_namer.convert_to_o9_cols(df=ActualCleansed)

        # Fix Forecast Iteration manually since this is not present among input dataframe
        StatActual = StatActual.withColumnRenamed(
            o9PySparkConstants.FORECAST_ITERATION,
            o9Constants.FORECAST_ITERATION,
        )
        ActualHML = ActualHML.withColumnRenamed(
            o9PySparkConstants.FORECAST_ITERATION,
            o9Constants.FORECAST_ITERATION,
        )
        ActualCleansed = ActualCleansed.withColumnRenamed(
            o9PySparkConstants.FORECAST_ITERATION,
            o9Constants.FORECAST_ITERATION,
        )

    return StatActual, ActualHML, ActualCleansed

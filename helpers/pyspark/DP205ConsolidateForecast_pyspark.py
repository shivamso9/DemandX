import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.functions import col, sum, to_date
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from helpers.o9PySparkConstants import o9PySparkConstants

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()


# Define main function
@log_inputs_and_outputs
def main(
    spark,
    statactual_df,
    hml_df,
    HML,
    PL,
    InputMeasure,
    DiscoDate,
    IntroDate,
    HMLInputMeasure,
    TargetMeasure,
    Grains,
    assortment_df,
    ActualHML,
    engine,
    CurrentTimePeriod,
    MasterDataDict,
    ForecastEngineStat,
    Core,
    TimeMaster,
):
    # Define the variable for the column name

    column_mapping = {
        o9PySparkConstants.STAT_ITEM: o9PySparkConstants.PLANNING_ITEM,
        o9PySparkConstants.STAT_LOCATION: o9PySparkConstants.PLANNING_LOCATION,
        o9PySparkConstants.STAT_REGION: o9PySparkConstants.PLANNING_REGION,
        o9PySparkConstants.STAT_DEMAND_DOMAIN: o9PySparkConstants.PLANNING_DEMAND_DOMAIN,
        o9PySparkConstants.STAT_CHANNEL: o9PySparkConstants.PLANNING_CHANNEL,
        o9PySparkConstants.STAT_PNL: o9PySparkConstants.PLANNING_PNL,
        o9PySparkConstants.STAT_ACCOUNT: o9PySparkConstants.PLANNING_ACCOUNT,
    }

    o9_grains = [get_clean_string(grain) for grain in Grains.split(",")]

    # Convert column names
    HML = col_namer.convert_to_pyspark_cols(HML)
    PL = col_namer.convert_to_pyspark_cols(PL)
    hml_df = col_namer.convert_to_pyspark_cols(df=hml_df)
    statactual_df = col_namer.convert_to_pyspark_cols(df=statactual_df)
    assortment_df = col_namer.convert_to_pyspark_cols(df=assortment_df)
    CurrentTimePeriod = col_namer.convert_to_pyspark_cols(df=CurrentTimePeriod)

    cols_required_filter = (
        o9_grains + [o9PySparkConstants.PARTIAL_WEEK] + [o9PySparkConstants.PARTIAL_WEEK_KEY]
    )
    cols_required_final = cols_required_filter + [TargetMeasure]
    partial_week_filter = [o9PySparkConstants.PARTIAL_WEEK] + [o9PySparkConstants.PARTIAL_WEEK_KEY]

    empty_schema = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(o9PySparkConstants.PLANNING_REGION, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.PLANNING_LOCATION,
                StringType(),
                nullable=True,
            ),
            StructField(
                o9PySparkConstants.PLANNING_CHANNEL,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.PLANNING_PNL, StringType(), nullable=True),
            StructField(o9PySparkConstants.PLANNING_ITEM, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.PLANNING_DEMAND_DOMAIN,
                StringType(),
                nullable=True,
            ),
            StructField(
                o9PySparkConstants.PLANNING_ACCOUNT,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.PARTIAL_WEEK, StringType(), nullable=True),
            StructField(TargetMeasure, DoubleType(), nullable=True),
        ]
    )

    combined_table = spark.createDataFrame([], schema=empty_schema)

    try:
        # Identifying Forecast Engine
        ifstat = engine.filter(engine["Forecast Engine Core Products"] == ForecastEngineStat)
        # If ENgine is Statistical Filter HML Actual for Non-Core
        if ifstat.count() > 0:
            hml_df = hml_df.filter(hml_df[o9PySparkConstants.DEMAND_DOMAIN] != Core)
            HML = HML.filter(col(HMLInputMeasure).isNotNull())
        else:
            PL = PL.filter(col(InputMeasure).isNotNull())
        logger.info(f"HML Actual count , {hml_df.count()}")
        for old_col, new_col in column_mapping.items():
            statactual_df = statactual_df.withColumnRenamed(old_col, new_col)

        # Join HML Actual dataframe with Master Data Dict for aggregation at Planning Level
        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)
            # identify common column if any in Actual
            the_common_columns_actual = list(
                set(hml_df.columns).intersection(the_master_data.columns)
            )

            if the_common_columns_actual:
                the_join_col = the_common_columns_actual[0]
                logger.debug(f"the_join_col_actual : {the_join_col}")

                hml_df = hml_df.join(the_master_data, on=the_join_col, how="inner")
        # Check if either ML Fcst HML or Stat Fcst PL is null
        if HML.count() == 0 and PL.count() == 0:
            logger.warning(
                "Both ML Fcst HML, Stat Fcst PL tables are null. Returning empty DataFrame."
            )

        if HML.count() == 0:
            logger.warning("ML Fcst HML table is null.")

        if PL.count() == 0:
            logger.warning("Stat Fcst PL table is null.")

        logger.info("Executing main function ...")
        logger.info(f"ML Fcst HML Dataframe row count, {HML.count()}")

        logger.info(f"Stat Fcst PL Dataframe row count, {PL.count()}")

        # Aggregate Actual HML to Planning Level for Actualization
        hml_df = hml_df.groupBy(*cols_required_filter).agg(sum(ActualHML).alias(ActualHML))

        # Rename Column Name before UnionByName
        HML = HML.withColumnRenamed(HMLInputMeasure, TargetMeasure)
        PL = PL.withColumnRenamed(InputMeasure, TargetMeasure)
        hml_df = hml_df.withColumnRenamed(ActualHML, TargetMeasure)
        statactual_df = statactual_df.withColumnRenamed(
            o9PySparkConstants.STAT_ACTUAL, TargetMeasure
        )
        statactual_df = statactual_df.drop(o9PySparkConstants.FORECAST_ITERATION)
        hml_df = hml_df.drop(o9PySparkConstants.FORECAST_ITERATION)

        # Combine Both ML Fcst HML And Stat Fcst PL and Drop Duplicate
        combined_table = HML.unionByName(PL).dropDuplicates()

        logger.debug(f"Combined Table schema , {combined_table}")
        logger.info(f"Combined Table count , {combined_table.count()}")

        # Drop Forecast Iteration
        combined_table = combined_table.drop(o9PySparkConstants.FORECAST_ITERATION)

        # Adding Partial Week Key to Consolidate Fcst Table and Stat Actual table
        TimeMasterFinal = col_namer.convert_to_pyspark_cols(df=TimeMaster)
        the_common_columns_combined_table = list(
            set(combined_table.columns).intersection(TimeMasterFinal.columns)
        )
        the_common_columns_statactualdf = list(
            set(statactual_df.columns).intersection(TimeMasterFinal.columns)
        )
        partialweek_Time_df = TimeMasterFinal.select(partial_week_filter).dropDuplicates()

        if the_common_columns_combined_table:
            the_join_col_combined_table = the_common_columns_combined_table[0]

            combined_table = combined_table.join(
                partialweek_Time_df,
                on=the_join_col_combined_table,
                how="inner",
            )

        if the_common_columns_statactualdf:
            the_join_col_statactual_table = the_common_columns_statactualdf[0]

            statactual_df = statactual_df.join(
                partialweek_Time_df,
                on=the_join_col_statactual_table,
                how="inner",
            )

        combined_table_final = combined_table.select(cols_required_final).dropDuplicates()
        statactual_df = statactual_df.select(cols_required_final).dropDuplicates()

        # Remove Fcst before Intro Date and after Disco Date in Consolidate Fcst dataframe
        consolidate_fcst = combined_table_final.join(assortment_df, on=o9_grains, how="inner")
        consolidate_fcst = consolidate_fcst.withColumn(
            o9PySparkConstants.PARTIAL_WEEK_KEY,
            to_date(col(o9PySparkConstants.PARTIAL_WEEK_KEY), "M/d/yyyy h:mm:ss a"),
        )

        consolidate_fcst = consolidate_fcst.withColumn(
            DiscoDate, to_date(col(DiscoDate), "M/d/yyyy h:mm:ss a")
        )
        consolidate_fcst = consolidate_fcst.withColumn(
            IntroDate, to_date(col(IntroDate), "M/d/yyyy h:mm:ss a")
        )

        filtered_df = consolidate_fcst.filter(
            (to_date(col(o9PySparkConstants.PARTIAL_WEEK_KEY)) >= to_date(col(IntroDate)))
            & (to_date(col(o9PySparkConstants.PARTIAL_WEEK_KEY)) < to_date(col(DiscoDate)))
        )

        # Only filter out only specific column
        final_stat_fcst = filtered_df.select(cols_required_final).dropDuplicates()

        # Get current date
        current_date = (
            col_namer.convert_to_pyspark_cols(CurrentTimePeriod)
            .withColumn(
                o9PySparkConstants.PARTIAL_WEEK,
                to_date(col(o9PySparkConstants.PARTIAL_WEEK), "dd-MMM-yy"),
            )
            .select(o9PySparkConstants.PARTIAL_WEEK)
            .collect()[0][0]
        )
        logger.info(f"current_date, {current_date}")

        # Join actual table with stat fcst
        aggregatedcombined_df = statactual_df.unionByName(hml_df)

        # Case when we have Actual for future dates also
        aggregatedcombined_df = aggregatedcombined_df.withColumn(
            o9PySparkConstants.PARTIAL_WEEK,
            to_date(col(o9PySparkConstants.PARTIAL_WEEK), "dd-MMM-yy"),
        )

        aggregatedcombined_df = aggregatedcombined_df.filter(
            col(o9PySparkConstants.PARTIAL_WEEK) < current_date
        )

        # Drop Partial Week Key Column
        aggregatedcombined_df = aggregatedcombined_df.drop(
            o9PySparkConstants.PARTIAL_WEEK_KEY
        ).dropDuplicates()
        final_stat_fcst = final_stat_fcst.drop(o9PySparkConstants.PARTIAL_WEEK_KEY).dropDuplicates()

        # Union Fcst with Actualised Table
        stat_fcst = aggregatedcombined_df.unionByName(final_stat_fcst).dropDuplicates()

        logger.info(f"Stat Fcst Table Count: ,{stat_fcst.count()}")

        stat_fcst = col_namer.convert_to_o9_cols(df=stat_fcst)

        logger.info("Successfully executed main function ...")
    except Exception as e:
        logger.exception("Exception occurred in main function: {}".format(e))

    return stat_fcst

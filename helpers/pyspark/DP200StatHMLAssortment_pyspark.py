import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.functions import concat_ws, lit, max, sum, when
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()


# Define main function
@log_inputs_and_outputs
def main(
    spark,
    actual_df,
    nsi_df,
    assortment_df,
    engine,
    MasterDataDict,
    SourceGrain,
    TargetGrain,
    HMLAssortment,
    SourceGrainAssortment,
    TargetGrainAssortment,
    ForecastEngineStat,
    Core,
    AssortmentFinalcol,
    AssortmentStatcol,
    AssortmentHMLcol,
    Actual,
    NSIActual,
):

    plugin_name = "DP200StatHMLAssortment_pyspark"
    logger.info("Executing {} ...".format(plugin_name))

    actual_df = col_namer.convert_to_pyspark_cols(df=actual_df)
    nsi_df = col_namer.convert_to_pyspark_cols(df=nsi_df)
    assortment_df = col_namer.convert_to_pyspark_cols(df=assortment_df)
    engine = col_namer.convert_to_pyspark_cols(df=engine)
    empty_schema_stat = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(o9PySparkConstants.PLANNING_REGION, StringType(), nullable=True),
            StructField(o9PySparkConstants.LOCATION, StringType(), nullable=True),
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
            StructField(AssortmentStatcol, DoubleType(), nullable=True),
        ]
    )

    empty_schema_hml = StructType(
        [
            StructField(o9PySparkConstants.VERSION_NAME, StringType(), nullable=True),
            StructField(o9PySparkConstants.REGION, StringType(), nullable=True),
            StructField(o9PySparkConstants.LOCATION, StringType(), nullable=True),
            StructField(o9PySparkConstants.CHANNEL, StringType(), nullable=True),
            StructField(o9PySparkConstants.PNL, StringType(), nullable=True),
            StructField(o9PySparkConstants.ITEM, StringType(), nullable=True),
            StructField(
                o9PySparkConstants.DEMAND_DOMAIN,
                StringType(),
                nullable=True,
            ),
            StructField(o9PySparkConstants.ACCOUNT, StringType(), nullable=True),
            StructField(AssortmentHMLcol, DoubleType(), nullable=True),
        ]
    )

    AssortmentStatFinal = spark.createDataFrame([], schema=empty_schema_stat)

    AssortmentHMLFinal = spark.createDataFrame([], schema=empty_schema_hml)

    try:

        # Identify empty DataFrames
        if actual_df.count() == 0 and nsi_df.count() == 0:
            logger.warning(
                "Both Actual and NSI Actual tables are empty. Returning empty DataFrame."
            )
        elif actual_df.count() == 0:
            logger.warning("Actual table is empty.")

        elif nsi_df.count() == 0:
            logger.warning("NSI Actual table is empty")

        # Script Params with DL Compatible
        source_grains = get_list_of_grains_from_string(input=SourceGrain)
        source_grains = [get_clean_string(x) for x in source_grains]
        sourcegrainassortment = get_list_of_grains_from_string(input=SourceGrainAssortment)
        sourcegrainassortment = [get_clean_string(x) for x in sourcegrainassortment]

        target_grains = get_list_of_grains_from_string(input=TargetGrain)
        target_grains = [get_clean_string(x) for x in target_grains]
        targetgrainassortment = get_list_of_grains_from_string(input=TargetGrainAssortment)
        targetgrainassortment = [get_clean_string(x) for x in targetgrainassortment]
        all_grains_hml = get_list_of_grains_from_string(input=HMLAssortment)
        all_grains_hml = [get_clean_string(x) for x in all_grains_hml]

        # Aggregating Actuals without Time
        actual_df = actual_df.groupBy(*all_grains_hml).agg(sum(Actual).alias(Actual))

        aggregated_nsi_df = nsi_df.groupBy(*all_grains_hml).agg(sum(NSIActual).alias(NSIActual))

        # Concat Item Location Channel Valid Intersection
        aggregated_nsi_df = aggregated_nsi_df.withColumn(
            "Item_Location_Channel",
            concat_ws(
                "-",
                o9PySparkConstants.ITEM,
                o9PySparkConstants.LOCATION,
                o9PySparkConstants.CHANNEL,
            ),
        ).dropDuplicates()

        actual_df = actual_df.withColumn(
            "Item_Location_Channel",
            concat_ws(
                "-",
                o9PySparkConstants.ITEM,
                o9PySparkConstants.LOCATION,
                o9PySparkConstants.CHANNEL,
            ),
        ).dropDuplicates()

        actual_df = actual_df.select("Item_Location_Channel").dropDuplicates()
        aggregated_nsi_df = aggregated_nsi_df.select("Item_Location_Channel").dropDuplicates()

        # Union Actual and NSI Actual
        combined_actual_df = actual_df.union(aggregated_nsi_df)

        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)

            # identify common column in nsi actual
            the_common_columns_nsi = list(set(nsi_df.columns).intersection(the_master_data.columns))
            logger.debug(f"the_key : {the_key}, common_columns_nsi : {the_common_columns_nsi}")
            # identify common column in assortment
            the_common_columns_assort = list(
                set(assortment_df.columns).intersection(the_master_data.columns)
            )
            logger.debug(
                f"the_key : {the_key}, common_columns_assort : {the_common_columns_assort}"
            )

            if the_common_columns_nsi:
                the_join_col = the_common_columns_nsi[0]
                logger.debug(f"the_join_col_nsi : {the_join_col}")

                nsi_df = nsi_df.join(the_master_data, on=the_join_col, how="inner")
            if the_common_columns_assort:
                the_join_col = the_common_columns_assort[0]
                logger.debug(f"the_join_col_assortment : {the_join_col}")

                assortment_df = assortment_df.join(the_master_data, on=the_join_col, how="inner")

        # Aggregating Assormtment Final to Assortment Stat using Max function
        aggassortment_df = assortment_df.groupBy(*targetgrainassortment).agg(
            max(AssortmentFinalcol).alias(AssortmentStatcol)
        )

        logger.info(f"Count of Stat Assortment with Core and Non-Core : {aggassortment_df.count()}")

        logger.debug(f"Assortment Stat: {aggassortment_df.take(5)}")

        # Aggregating NSI Actual to Planning Level using Sum function with Time
        aggnsi_df = nsi_df.groupBy(*targetgrainassortment).agg(sum(NSIActual).alias(NSIActual))
        logger.debug(f"Aggregate NSI Actual to Planning Level schema: {aggnsi_df}")

        # If Actual > 0 at planning Level then Assortment HML = 1
        assortmentagg_nsi = aggnsi_df.withColumn(
            AssortmentStatcol,
            when(aggnsi_df[NSIActual] > 0, 1).otherwise(lit(None)),
        )

        assortmentagg_nsi = assortmentagg_nsi.drop(NSIActual)

        # Identify Forecast Engine
        ifstat = engine.filter(engine["Forecast Engine Core Products"] == ForecastEngineStat)
        logger.info(f"Forecast Engine Core Product: {engine}")
        # Filter for Core and Non Core Product
        if ifstat.count() > 0:
            aggstatassortment_df = aggassortment_df.filter(
                aggassortment_df[o9PySparkConstants.PLANNING_DEMAND_DOMAIN] == Core
            )
            logger.info(
                f"Count of Stat Assortment with Core Item without NSI : {aggstatassortment_df.count()}"
            )
            aggstatassortment_nsi_df = assortmentagg_nsi.filter(
                assortmentagg_nsi[o9PySparkConstants.PLANNING_DEMAND_DOMAIN] == Core
            )
            logger.info(
                f"Count of Stat Assortment with Core Item for NSI : {aggstatassortment_nsi_df.count()}"
            )
            finalstatassortment_df = aggstatassortment_df.unionByName(
                aggstatassortment_nsi_df
            ).dropDuplicates()
            agghmlassortment_df = aggassortment_df.filter(
                aggassortment_df[o9PySparkConstants.PLANNING_DEMAND_DOMAIN] != Core
            )
            logger.info(
                f"Count of HML Assortment with Non-Core Item without NSI at Planning Level: {agghmlassortment_df.count()}"
            )
            agghmlassortment_nsi_df = assortmentagg_nsi.filter(
                assortmentagg_nsi[o9PySparkConstants.PLANNING_DEMAND_DOMAIN] != Core
            )
            logger.info(
                f"Count of HML Assortment with Non-Core Item for NSI at Planning Level: {agghmlassortment_nsi_df.count()}"
            )
            finalhmlassortment_df = agghmlassortment_df.unionByName(
                agghmlassortment_nsi_df
            ).dropDuplicates()
        else:
            finalstatassortment_df = AssortmentStatFinal
            finalhmlassortment_df = aggassortment_df.unionByName(assortmentagg_nsi)

        # Disaggregate Assortment at Planning Level to Lower Level
        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)

            # identify common column if any
            the_common_columns = list(
                set(finalhmlassortment_df.columns).intersection(the_master_data.columns)
            )
            logger.debug(f"the_key : {the_key}, common_columns : {the_common_columns}")
            if the_common_columns:
                the_join_col = the_common_columns[0]
                logger.debug(f"the_join_col : {the_join_col}")

                finalhmlassortment_df = finalhmlassortment_df.join(
                    the_master_data, on=the_join_col, how="inner"
                )
        # Concat Assortment Table
        concat_assortment_df = finalhmlassortment_df.withColumn(
            "Item_Location_Channel",
            concat_ws(
                "-",
                o9PySparkConstants.ITEM,
                o9PySparkConstants.LOCATION,
                o9PySparkConstants.CHANNEL,
            ),
        )

        # HML Actual Assortment at granular level
        concat_lower_level_grain = source_grains + ["Item_Location_Channel"]
        final_grains = source_grains + [AssortmentHMLcol]

        concat_assortment_df = concat_assortment_df.select(
            concat_lower_level_grain
        ).dropDuplicates()

        filtered_assort_df = concat_assortment_df.join(
            combined_actual_df, on="Item_Location_Channel", how="inner"
        ).dropDuplicates()

        filtered_assort_df = filtered_assort_df.withColumn(AssortmentHMLcol, lit(1))
        filtered_assort_df = filtered_assort_df.select(final_grains).dropDuplicates()

        AssortmentStatFinal = col_namer.convert_to_o9_cols(df=finalstatassortment_df)
        AssortmentStatFinal = AssortmentStatFinal.withColumn(
            AssortmentStatcol,
            AssortmentStatFinal[AssortmentStatcol].cast(DoubleType()),
        )
        AssortmentHMLFinal = col_namer.convert_to_o9_cols(df=filtered_assort_df)
        AssortmentHMLFinal = AssortmentHMLFinal.withColumn(
            AssortmentHMLcol,
            AssortmentHMLFinal[AssortmentHMLcol].cast(DoubleType()),
        )

        logger.info(f"Final Assortment HML table row count: {AssortmentHMLFinal.count()}")
        logger.info(f"Final Assortment Stat table row count: {AssortmentStatFinal.count()}")
    except Exception as e:
        logger.exception(e)
    return AssortmentStatFinal, AssortmentHMLFinal

import logging

import pandas as pd
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql import functions as F
from pyspark.sql.functions import col, concat_ws, to_date, trim
from pyspark.sql.types import StringType, StructField, StructType

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """A class to store constant values used throughout the module."""

    assortment_col = "Assortment Final"
    intro_date_col = "Intro Date"
    like_item_col = "Like Item Selected"
    item_col = "Item_PlanningItem"
    version_column = "Version_VersionName"
    planning_channel = "Channel_PlanningChannel"
    planning_region = "Region_PlanningRegion"
    planning_location = "Location_PlanningLocation"
    planning_pnl = "PnL_PlanningPnL"
    item = "Item_Item"
    planning_account = "Account_PlanningAccount"
    planning_demand = "DemandDomain_PlanningDemandDomain"
    channel = "Channel_Channel"
    time_Day = "Time_Day"
    location = "Location_Location"
    account = "Account_Account"
    demand = "DemandDomain_DemandDomain"
    pnl = "PnL_PnL"
    region = "Region_Region"
    partial_week = "Time_PartialWeek"
    reference_actual = "Reference Actual"
    week_name = "Time_WeekName"
    year = "Time_Year"
    actual = "Actual"
    like_item_selected = "Like Item Selected"
    week_year = "WeekYear"

    common_merge_cols = [
        version_column,
        planning_region,
        planning_location,
        planning_channel,
        planning_pnl,
        like_item_selected,
        planning_account,
    ]
    groupby_cols = [
        week_year,
        version_column,
        like_item_selected,
        channel,
        time_Day,
        location,
        account,
        demand,
        pnl,
        region,
    ]
    final_output_columns = [
        version_column,
        item_col,
        planning_account,
        planning_channel,
        planning_region,
        planning_pnl,
        planning_location,
        planning_demand,
        partial_week,
        reference_actual,
    ]

    output_columns = [
        "Version.[Version Name]",
        "Item.[Planning Item]",
        "Account.[Planning Account]",
        "Channel.[Planning Channel]",
        "Region.[Planning Region]",
        "PnL.[Planning PnL]",
        "Location.[Planning Location]",
        "Demand Domain.[Planning Demand Domain]",
        "Time.[Partial Week]",
        reference_actual,
    ]


col_namer = ColumnNamer()


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    actual,
    LikeItemSelected,
    IntroDate,
    AssortmentFinal,
    TimeMaster,
    TimeGrains,
    TimeJoinkey,
    OutputColumns,
    ItemMapping,
    spark,
    df_keys,
):
    plugin_name = "DP121PopulateReferenceActuals_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        if any(
            df.rdd.isEmpty()
            for df in [
                AssortmentFinal,
                LikeItemSelected,
                IntroDate,
                actual,
                TimeMaster,
                ItemMapping,
            ]
        ):
            raise Exception(
                "One or more required inputs are empty: AssortmentFinal, LikeItemSelected, "
                "IntroDate, Actual, TimeMaster or ItemMapping. Returning empty outputs ..."
            )

        AssortmentFinal = col_namer.convert_to_pyspark_cols(AssortmentFinal)
        LikeItemSelected = col_namer.convert_to_pyspark_cols(LikeItemSelected)
        IntroDate = col_namer.convert_to_pyspark_cols(IntroDate)
        actual = col_namer.convert_to_pyspark_cols(actual)
        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)
        ItemMapping = col_namer.convert_to_pyspark_cols(ItemMapping)

        output_o9_grains = OutputColumns.split(",")
        output_o9_grains = [x.strip() for x in output_o9_grains]
        output_o9_grains = [str(x) for x in output_o9_grains if x != "NA" and x != ""]
        output_o9_grains = [get_clean_string(x) for x in output_o9_grains]

        time_dim_cols = get_list_of_grains_from_string(TimeGrains)
        time_dim_cols = [get_clean_string(x) for x in time_dim_cols]
        time_join_key = get_clean_string(TimeJoinkey)

        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)[time_dim_cols]

        AssortmentFinal = AssortmentFinal.join(
            ItemMapping.select(
                col(Constants.item).alias("ItemMapping_Item"), col(Constants.item_col)
            ),
            AssortmentFinal[Constants.item] == col("ItemMapping_Item"),
            how="left",
        ).drop("ItemMapping_Item")

        AssortmentFinal = AssortmentFinal.withColumnRenamed(
            Constants.location, Constants.planning_location
        )
        AssortmentFinal = AssortmentFinal.filter(col(Constants.assortment_col) == 1)

        merged_df = AssortmentFinal.join(LikeItemSelected, on=output_o9_grains, how="outer")
        merged_df = merged_df.join(IntroDate, on=output_o9_grains, how="outer")
        merged_df = col_namer.convert_to_pyspark_cols(merged_df)
        merged_df = merged_df.dropna()

        logger.info(f"Datatypes of the Time grain:{TimeMaster.dtypes}")
        actual_day_dtype = [dtype for name, dtype in actual.dtypes if name == time_join_key][0]
        dim_time_day_dtype = [dtype for name, dtype in TimeMaster.dtypes if name == time_join_key][
            0
        ]
        assert (
            actual_day_dtype == dim_time_day_dtype
        ), f"Data Type mistmatch of the key columns: Actuals.[Day] {actual_day_dtype} and Time.[Day] {dim_time_day_dtype}"
        combined_df = actual.join(TimeMaster, on=time_join_key, how="left")
        combined_df = combined_df.join(
            ItemMapping.select(
                col(Constants.item).alias("Item"),
                col(Constants.item_col).alias(Constants.like_item_selected),
            ),
            combined_df[Constants.item] == col("Item"),
            how="left",
        ).drop(Constants.item)
        merged_df = merged_df.withColumn(
            Constants.intro_date_col,
            F.to_date(F.col(Constants.intro_date_col), "M/d/yyyy h:mm:ss a"),
        )

        combined_df = combined_df.withColumn(
            Constants.time_Day, F.to_date(F.col(Constants.time_Day), "dd-MMM-yy")
        )
        combined_df = combined_df.withColumn(
            Constants.week_year,
            concat_ws(
                "-",
                col(Constants.week_name).cast(StringType()),
                col(Constants.year).cast(StringType()),
            ),
        )
        original_combined_df = combined_df
        groupby_cols_without_time_day = [
            col for col in Constants.groupby_cols if col != Constants.time_Day
        ]

        aggregated_df = combined_df.groupBy(groupby_cols_without_time_day).agg(
            F.sum(Constants.actual).alias(Constants.actual)
        )
        combined_df = aggregated_df.join(
            original_combined_df.select(
                *groupby_cols_without_time_day, Constants.time_Day
            ).dropDuplicates(),
            on=groupby_cols_without_time_day,
            how="left",
        )
        partial_week_df = original_combined_df.select(
            *Constants.groupby_cols, Constants.partial_week
        ).dropDuplicates()
        combined_df = combined_df.join(partial_week_df, on=Constants.groupby_cols, how="left")

        for old_col, new_col in {
            Constants.region: Constants.planning_region,
            Constants.location: Constants.planning_location,
            Constants.channel: Constants.planning_channel,
            Constants.pnl: Constants.planning_pnl,
            Constants.account: Constants.planning_account,
            Constants.actual: Constants.reference_actual,
        }.items():
            combined_df = combined_df.withColumnRenamed(old_col, new_col)
        for col_name in Constants.common_merge_cols:
            combined_df = combined_df.withColumn(col_name, trim(col(col_name)).cast("string"))
        merged_df = merged_df.withColumn(col_name, trim(col(col_name)).cast("string"))
        final_df = combined_df.join(merged_df, on=Constants.common_merge_cols, how="left")

        final_df = final_df.withColumn(
            Constants.time_Day, to_date(col(Constants.time_Day), "dd-MM-yyyy")
        )
        final_df = final_df.withColumn(
            Constants.intro_date_col,
            to_date(col(Constants.intro_date_col), "dd-MM-yyyy"),
        )
        final_df = final_df.filter(col(Constants.time_Day) < col(Constants.intro_date_col))
        final_df = final_df.withColumn(Constants.item_col, col(Constants.item_col))
        final_df = final_df.select(Constants.final_output_columns)
        final_df = col_namer.convert_to_o9_cols(final_df)
        logger.info(f"Successfully executed {plugin_name} ...")
    except Exception as e:
        logger.exception(f"Exception occurred: {e}")
        empty_schema = StructType(
            [StructField(col_name, StringType(), True) for col_name in Constants.output_columns]
        )
        final_df = spark.createDataFrame([], schema=empty_schema)
        logger.info("Returning an empty final DataFrame due to an error.")
    return final_df

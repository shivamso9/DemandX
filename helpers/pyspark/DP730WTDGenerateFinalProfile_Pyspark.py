import datetime
import logging
from datetime import timedelta

import pandas as pd
import pyspark
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    date_format,
    datediff,
    dayofweek,
    floor,
    lit,
    max,
    min,
    sum,
    to_date,
    when,
)
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

col_namer = ColumnNamer()
logger = logging.getLogger("o9_logger")


def log_dataframe(df: pyspark.sql.dataframe) -> None:
    logger.info("------ dataframe head (5) --------")
    logger.info(f"Head : {df.take(5)}")
    logger.info(f"Schema : {df.schema}")
    logger.info(f"Shape : ({df.count()}, {len(df.columns)})")


def main(
    SelfOutput,
    BaseProfileOP,
    HolidayType,
    FinalLiftOP,
    ForecastPeriodWTD,
    CurrentDay,
    TimeMaster,
    spark,
    df_keys,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    plugin_name = "DP730WTDGenerateFinalProfile_Pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version_VersionName"
    week_col = "Time_Week"
    day_col = "Time_Day"
    FinalProfile_WTD = "Final Profile WTD"
    EventLiftProf_WTD = "Event Lift WTD"
    BaseProf_WTD = "Base Profile WTD"
    ActualProf_WTD = "Actual Profile WTD"
    WTDEventType = "Event Type WTD"

    logger.info("Extracting dimension cols ...")

    output_o9_grains = SelfOutput.columns

    # to get col_mapping for all columns
    output_df = pd.DataFrame(columns=output_o9_grains)
    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    Plugin3OP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        BaseProfileOP = col_namer.convert_to_pyspark_cols(BaseProfileOP)
        HolidayType = col_namer.convert_to_pyspark_cols(HolidayType)
        FinalLiftOP = col_namer.convert_to_pyspark_cols(FinalLiftOP)
        ForecastPeriodWTD = col_namer.convert_to_pyspark_cols(ForecastPeriodWTD)
        CurrentDay = col_namer.convert_to_pyspark_cols(CurrentDay)
        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)
        SelfOutput = col_namer.convert_to_pyspark_cols(SelfOutput)

        if (
            (BaseProfileOP.count() == 0)
            or (FinalLiftOP.count() == 0)
            or (ForecastPeriodWTD.count() == 0)
            or (CurrentDay.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        log_dataframe(BaseProfileOP)
        log_dataframe(HolidayType)
        log_dataframe(FinalLiftOP)
        log_dataframe(ForecastPeriodWTD)
        log_dataframe(CurrentDay)

        # Dynamic Graining
        SelfColumns = SelfOutput.columns
        accountOP = next(column for column in SelfColumns if column.startswith("Account"))
        regionOP = next(column for column in SelfColumns if column.startswith("Region"))
        channelOP = next(column for column in SelfColumns if column.startswith("Channel"))
        demand_domainOP = next(
            column for column in SelfColumns if column.startswith("DemandDomain")
        )
        pnlOP = next(column for column in SelfColumns if column.startswith("PnL"))
        itemOP = next(column for column in SelfColumns if column.startswith("Item"))
        locationOP = next(column for column in SelfColumns if column.startswith("Location"))

        today = CurrentDay.select(to_date(col(day_col), "dd-MMM-yy")).first()[0]
        # Calculate the difference in days between today and the previous Sunday
        days_since_sunday = today.weekday()
        days_to_subtract = (
            days_since_sunday + 1
        ) % 7  # 0 represents Monday, so we need to add 1 to get the previous Sunday

        # Calculate the date of the previous Sunday
        today = today - timedelta(days=days_to_subtract)
        numOfDays = ForecastPeriodWTD.select(
            col("Forecast Period WTD Profile").cast("int")
        ).first()[0]
        end_date = today + datetime.timedelta(days=numOfDays - 1)

        FinalDays = spark.sql(
            f"SELECT explode(sequence(to_date('{today.strftime('%Y-%m-%d')}'), to_date('{end_date.strftime('%Y-%m-%d')}'), interval 1 day)) as date"
        ).select(
            to_date(col("date")).alias("date")
        )  # Convert back to DateType

        columns_to_check = [
            version_col,
            regionOP,
            channelOP,
            pnlOP,
            demand_domainOP,
            accountOP,
            locationOP,
            itemOP,
        ]
        # Drop duplicates based on the specified columns
        unique_combinations = BaseProfileOP.select(columns_to_check).dropDuplicates()

        # Create a dataframe for FinalDays with an extra key to facilitate the merge
        final_days_df = (
            FinalDays.withColumn(day_col, FinalDays["date"]).withColumn("key", lit(1)).drop("date")
        )

        # Add the same key to unique combinations and merge to align FinalDays with each combination
        unique_combinations = unique_combinations.withColumn("key", lit(1))
        expanded_df = unique_combinations.join(final_days_df, on="key", how="inner").drop("key")
        expanded_df = expanded_df.withColumn("Week Day", dayofweek(col(day_col)) % 7)

        # Add Week Col to BaseProfile
        BaseProfileOP = BaseProfileOP.join(TimeMaster, on=day_col, how="inner")
        columns_to_drop = ["Time_WeekKey", "Time_DayKey"]
        BaseProfileOP = BaseProfileOP.drop(*columns_to_drop)

        BaseProfileOP = BaseProfileOP.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        BaseProfileOP = BaseProfileOP.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        HolidayType = HolidayType.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        window_spec = Window.partitionBy(
            version_col,
            regionOP,
            locationOP,
            channelOP,
            pnlOP,
            itemOP,
            demand_domainOP,
            accountOP,
        )
        BaseProfileOP = BaseProfileOP.withColumn("Latest_Week", max(week_col).over(window_spec))
        BaseProfileOP = BaseProfileOP.filter(col(week_col) == col("Latest_Week")).drop(
            "Latest_Week"
        )
        BaseProfileOP = BaseProfileOP.withColumn("Week Day", dayofweek(col(day_col)) % 7)
        columns_to_drop = [day_col, week_col]
        BaseProfileOP = BaseProfileOP.drop(*columns_to_drop)

        Plugin3OP = expanded_df.join(
            BaseProfileOP,
            on=[
                version_col,
                regionOP,
                locationOP,
                channelOP,
                pnlOP,
                itemOP,
                demand_domainOP,
                accountOP,
                "Week Day",
            ],
            how="inner",
        )
        Plugin3OP = Plugin3OP.drop("Week Day")

        Plugin3OP = Plugin3OP.join(HolidayType, on=[version_col, locationOP, day_col], how="left")

        FinalLiftOP = FinalLiftOP.drop(day_col)
        FinalLiftOP = FinalLiftOP.dropDuplicates()
        FinalLiftOP = FinalLiftOP.withColumnRenamed(WTDEventType, "Holiday Type")

        Plugin3OP = Plugin3OP.join(
            FinalLiftOP,
            on=[
                version_col,
                regionOP,
                channelOP,
                pnlOP,
                demand_domainOP,
                accountOP,
                locationOP,
                itemOP,
                "Holiday Type",
            ],
            how="left",
        )

        Plugin3OP = Plugin3OP.fillna({EventLiftProf_WTD: 1})
        Plugin3OP = Plugin3OP.withColumn(
            "Adjusted Base Profile", col(EventLiftProf_WTD) * col(BaseProf_WTD)
        )

        # Calculate minimum date
        min_date = Plugin3OP.select(min(day_col).alias("min_date")).collect()[0]["min_date"]

        # Add new column 'Week Number'
        Plugin3OP = Plugin3OP.withColumn(
            "Week Number",
            (floor(datediff(col(day_col), lit(min_date)) / 7) + 1),
        )

        # Step 1: Group by the specified columns
        grouped_df = Plugin3OP.groupBy(
            version_col,
            regionOP,
            channelOP,
            pnlOP,
            demand_domainOP,
            accountOP,
            locationOP,
            itemOP,
            "Week Number",
        )

        # Step 2: Calculate the sum of the 'Adjusted Base Profile' for rows where 'Holiday Type' is not null
        sum_holiday_base_profile = grouped_df.agg(
            sum(
                when(
                    col("Holiday Type").isNotNull(),
                    col("Adjusted Base Profile"),
                ).otherwise(0)
            ).alias("sum_holiday_base_profile")
        )

        # Step 3: Join the sum back to the original dataframe
        Plugin3OP = Plugin3OP.join(
            sum_holiday_base_profile,
            on=[
                version_col,
                regionOP,
                channelOP,
                pnlOP,
                demand_domainOP,
                accountOP,
                locationOP,
                itemOP,
                "Week Number",
            ],
            how="left",
        )

        # Step 4: Calculate the 'OneMinus' column
        Plugin3OP = Plugin3OP.withColumn("OneMinus", lit(1) - col("sum_holiday_base_profile"))

        # Step 1: Group by the specified columns
        # grouped_df = df.groupBy("Version_Name", "Region", "Channel", "PnL", "Demand_Domain", "Account", "Location", "Item", "Week_Number")

        # Step 2: Calculate the sum of the 'Adjusted Base Profile' for rows where 'Holiday Type' is null
        sum_base_profile = grouped_df.agg(
            sum(
                when(col("Holiday Type").isNull(), col("Adjusted Base Profile")).otherwise(0)
            ).alias("Sum")
        )

        # Step 3: Join the sum back to the original dataframe
        Plugin3OP = Plugin3OP.join(
            sum_base_profile,
            on=[
                version_col,
                regionOP,
                channelOP,
                pnlOP,
                demand_domainOP,
                accountOP,
                locationOP,
                itemOP,
                "Week Number",
            ],
            how="left",
        )

        Plugin3OP = Plugin3OP.withColumn(
            "Numerator", col("Adjusted Base Profile") * col("OneMinus")
        )

        Plugin3OP = Plugin3OP.withColumn(FinalProfile_WTD, (col("Numerator") * 1.0) / col("Sum"))

        Plugin3OP = Plugin3OP.withColumn(
            FinalProfile_WTD,
            when(col("Holiday Type").isNotNull(), col("Adjusted Base Profile")).otherwise(
                col(FinalProfile_WTD)
            ),
        )

        columns_to_drop = [
            "Week Number",
            "OneMinus",
            "Sum",
            "Numerator",
            "Adjusted Base Profile",
            "Holiday Type",
            EventLiftProf_WTD,
            "sum_holiday_base_profile",
            BaseProf_WTD,
            ActualProf_WTD,
        ]
        Plugin3OP = Plugin3OP.drop(*columns_to_drop)
        Plugin3OP = Plugin3OP.withColumn(
            day_col,
            date_format(col(day_col), "dd-MMM-yy"),  # Convert to specified format
        )

        Plugin3OP = col_namer.convert_to_o9_cols(Plugin3OP)
        Plugin3OP = Plugin3OP.withColumn(
            FinalProfile_WTD, Plugin3OP[FinalProfile_WTD].cast(DoubleType())
        )
        log_dataframe(Plugin3OP)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(df_keys)
        )
        logger.exception(e)
        output_df = pd.DataFrame(columns=output_o9_grains)
        output_df_schema = StructType(
            [StructField(col_name, StringType(), True) for col_name in output_df.columns]
        )
        emp_RDD = spark.sparkContext.emptyRDD()
        Plugin3OP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)
    return Plugin3OP

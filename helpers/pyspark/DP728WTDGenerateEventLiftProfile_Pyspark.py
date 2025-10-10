import datetime
import logging
from datetime import timedelta

import pandas as pd
import pyspark
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import Window
from pyspark.sql.functions import col, date_format, dayofweek, max, mean, sum, to_date
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
    MinEventLift,
    HistoryPeriodWTDEventLift,
    CurrentDay,
    Actuals,
    ItemMaster,
    RegionMaster,
    AccountMaster,
    ChannelMaster,
    PnLMaster,
    DemandDomainMaster,
    TimeMaster,
    LocationMaster,
    spark,
    df_keys,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    plugin_name = "DP728WTDGenerateEventLiftProfile_Pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version_VersionName"
    location_col = "Location_Location"
    channel_col = "Channel_Channel"
    item_col = "Item_Item"
    demand_domain_col = "DemandDomain_DemandDomain"
    region_col = "Region_Region"
    account_col = "Account_Account"
    pnl_col = "PnL_PnL"
    week_col = "Time_Week"
    day_col = "Time_Day"
    EventLiftProf_WTD = "Event Lift WTD"
    BaseProf_WTD = "Base Profile WTD"
    AProfile = "Actual Profile"
    WTDEventType = "Event Type WTD"

    logger.info("Extracting dimension cols ...")

    output_o9_grains = SelfOutput.columns
    # to get col_mapping for all columns
    output_df = pd.DataFrame(columns=output_o9_grains)
    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    FinalLiftOP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        BaseProfileOP = col_namer.convert_to_pyspark_cols(BaseProfileOP)
        HistoryPeriodWTDEventLift = col_namer.convert_to_pyspark_cols(HistoryPeriodWTDEventLift)
        HolidayType = col_namer.convert_to_pyspark_cols(HolidayType)
        MinEventLift = col_namer.convert_to_pyspark_cols(MinEventLift)
        Actuals = col_namer.convert_to_pyspark_cols(Actuals)
        CurrentDay = col_namer.convert_to_pyspark_cols(CurrentDay)
        ItemMaster = col_namer.convert_to_pyspark_cols(ItemMaster)
        AccountMaster = col_namer.convert_to_pyspark_cols(AccountMaster)
        ChannelMaster = col_namer.convert_to_pyspark_cols(ChannelMaster)
        RegionMaster = col_namer.convert_to_pyspark_cols(RegionMaster)
        PnLMaster = col_namer.convert_to_pyspark_cols(PnLMaster)
        DemandDomainMaster = col_namer.convert_to_pyspark_cols(DemandDomainMaster)
        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)
        SelfOutput = col_namer.convert_to_pyspark_cols(SelfOutput)
        LocationMaster = col_namer.convert_to_pyspark_cols(LocationMaster)

        if (
            (BaseProfileOP.count() == 0)
            or (HistoryPeriodWTDEventLift.count() == 0)
            or (MinEventLift.count() == 0)
            or (Actuals.count() == 0)
            or (CurrentDay.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        if HolidayType.count() == 0:
            logger.info("No Holiday Data present, returning empty dataframe.")
            return FinalLiftOP

        log_dataframe(BaseProfileOP)
        log_dataframe(HistoryPeriodWTDEventLift)
        log_dataframe(MinEventLift)
        log_dataframe(HolidayType)
        log_dataframe(Actuals)
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

        # Mapping Lower Actual Grains to Higher Planner decided grains

        if itemOP != item_col:
            ItemMaster = ItemMaster.select(itemOP, item_col)
            Actuals = Actuals.join(ItemMaster, on=item_col, how="inner")
            columns_to_drop = [item_col]
            item_col = itemOP
            Actuals = Actuals.drop(*columns_to_drop)

            Actuals = Actuals.groupBy(
                version_col,
                region_col,
                location_col,
                channel_col,
                pnl_col,
                demand_domain_col,
                account_col,
                item_col,
                day_col,
            ).agg(sum("Actual").alias("Actual"))

        if accountOP != account_col:
            AccountMaster = AccountMaster.select(accountOP, account_col)
            Actuals = Actuals.join(AccountMaster, on=account_col, how="inner")
            columns_to_drop = [account_col]
            account_col = accountOP
            Actuals = Actuals.drop(*columns_to_drop)

        if channelOP != channel_col:
            ChannelMaster = ChannelMaster.select(channelOP, channel_col)
            Actuals = Actuals.join(ChannelMaster, on=channel_col, how="inner")
            columns_to_drop = [channel_col]
            channel_col = channelOP
            Actuals = Actuals.drop(*columns_to_drop)

        if regionOP != region_col:
            RegionMaster = RegionMaster.select(regionOP, region_col)
            Actuals = Actuals.join(RegionMaster, on=region_col, how="inner")
            columns_to_drop = [region_col]
            region_col = regionOP
            Actuals = Actuals.drop(*columns_to_drop)

        if pnlOP != pnl_col:
            PnLMaster = PnLMaster.select(pnlOP, pnl_col)
            Actuals = Actuals.join(PnLMaster, on=pnl_col, how="inner")
            columns_to_drop = [pnl_col]
            pnl_col = pnlOP
            Actuals = Actuals.drop(*columns_to_drop)

        if demand_domainOP != demand_domain_col:
            DemandDomainMaster = DemandDomainMaster.select(demand_domainOP, demand_domain_col)
            Actuals = Actuals.join(DemandDomainMaster, on=demand_domain_col, how="inner")
            columns_to_drop = [demand_domain_col]
            demand_domain_col = demand_domainOP
            Actuals = Actuals.drop(*columns_to_drop)

        if locationOP != location_col:
            LocationMaster = LocationMaster.select(locationOP, location_col)
            Actuals = Actuals.join(LocationMaster, on=location_col, how="inner")
            columns_to_drop = [location_col]
            location_col = locationOP
            Actuals = Actuals.drop(*columns_to_drop)

        Actuals = Actuals.join(TimeMaster, on=day_col, how="inner")
        columns_to_drop = ["Time_WeekKey", "Time_DayKey"]
        Actuals = Actuals.drop(*columns_to_drop)
        Actuals = Actuals.dropDuplicates()

        # Take the Holiday Type, and filter it for the days mentioned in "History Period WTD Event Lift"
        HolidayType = HolidayType.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        today = CurrentDay.select(to_date(col(day_col), "dd-MMM-yy")).first()[0]
        # Calculate the difference in days between today and the previous Saturday
        days_since_saturday = today.weekday()
        days_to_subtract = (
            days_since_saturday + 2
        ) % 7  # 0 represents Monday, so we need to add 2 to get the previous Saturday
        # Calculate the date of the previous Saturday
        today = today - timedelta(days=days_to_subtract)
        x = HistoryPeriodWTDEventLift.select(
            col("History Period WTD Event Lift").cast("int")
        ).first()[0]
        x = ((x + 6) // 7) * 7
        start_date = today - datetime.timedelta(days=x - 1)
        end_date = today
        HolidayType = HolidayType.filter((col(day_col) >= start_date) & (col(day_col) <= end_date))

        # For each Day in Holiday Type, add a "Week Day" based on the day it falls on in the week, if the week starts from Sunday; Sunday = 1
        HolidayType = HolidayType.withColumn("Week Day", dayofweek(col(day_col)) % 7)

        # Merge Holiday Type and Actual, calculate WeeklySum, Actual  Profile
        Actuals = Actuals.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        Actuals = Actuals.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        FinalLiftOP = Actuals.join(HolidayType, on=[version_col, location_col, day_col], how="left")
        window_spec = Window.partitionBy(
            region_col,
            location_col,
            channel_col,
            pnl_col,
            item_col,
            demand_domain_col,
            account_col,
            week_col,
        )

        FinalLiftOP = FinalLiftOP.withColumn(
            "Weekly Demand",
            sum(col("actual")).over(window_spec),  # Apply the sum over the window
        )

        # Filtering Actual to only consist of Holidays
        FinalLiftOP = FinalLiftOP.filter(col("Holiday Type").isNotNull())

        FinalLiftOP = FinalLiftOP.withColumn(
            AProfile,
            (col("actual") * 1.0) / col("Weekly Demand"),  # Calculate the ratio
        )

        # Add Week Col to BaseProfile
        BaseProfileOP = BaseProfileOP.join(TimeMaster, on=day_col, how="inner")
        columns_to_drop = ["Time_WeekKey", "Time_DayKey"]
        BaseProfileOP = BaseProfileOP.drop(*columns_to_drop)

        # Take the BaseProfileOP, filter it for just one week, and add a cumcount to the week => "Week Day"
        BaseProfileOP = BaseProfileOP.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        BaseProfileOP = BaseProfileOP.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        window_spec = Window.partitionBy(
            version_col,
            region_col,
            location_col,
            channel_col,
            pnl_col,
            item_col,
            demand_domain_col,
            account_col,
        )
        BaseProfileOP = BaseProfileOP.withColumn("Latest_Week", max(week_col).over(window_spec))
        BaseProfileOP = BaseProfileOP.filter(col(week_col) == col("Latest_Week")).drop(
            "Latest_Week"
        )
        BaseProfileOP = BaseProfileOP.withColumn("Week Day", dayofweek(col(day_col)) % 7)
        columns_to_drop = [day_col, week_col]
        BaseProfileOP = BaseProfileOP.drop(*columns_to_drop)

        # Merge the BaseProfileOP and FinalLiftOP on "Week Day", Calculate Event Lift and Mean Lift
        FinalLiftOP = FinalLiftOP.join(
            BaseProfileOP,
            on=[
                version_col,
                region_col,
                location_col,
                channel_col,
                pnl_col,
                item_col,
                demand_domain_col,
                account_col,
                "Week Day",
            ],
            how="inner",
        )

        FinalLiftOP = FinalLiftOP.withColumn(
            "Lift", col(AProfile) / col(BaseProf_WTD)  # Calculate the ratio
        )

        columns_to_drop = [
            BaseProf_WTD,
            AProfile,
            "Week Day",
            "actual",
            "Weekly Demand",
        ]
        FinalLiftOP = FinalLiftOP.drop(*columns_to_drop)

        window_spec = Window.partitionBy(
            version_col,
            region_col,
            location_col,
            channel_col,
            pnl_col,
            item_col,
            demand_domain_col,
            account_col,
            "Holiday Type",
        )

        FinalLiftOP = FinalLiftOP.withColumn(
            EventLiftProf_WTD,
            mean(col("Lift")).over(window_spec),  # Apply the sum over the window
        )

        min_event_lift_value = MinEventLift.select(
            col("Min Event Lift % WTD Profile").cast(DoubleType())
        ).first()[0]
        FinalLiftOP = FinalLiftOP.filter(col(EventLiftProf_WTD) > (min_event_lift_value + 1))

        columns_to_drop = ["Lift", week_col]
        FinalLiftOP = FinalLiftOP.drop(*columns_to_drop)
        FinalLiftOP = FinalLiftOP.withColumnRenamed("Holiday Type", WTDEventType)

        FinalLiftOP = FinalLiftOP.withColumn(
            day_col,
            date_format(col(day_col), "dd-MMM-yy"),  # Convert to specified format
        )

        FinalLiftOP = col_namer.convert_to_o9_cols(FinalLiftOP)
        FinalLiftOP = FinalLiftOP.withColumn(
            EventLiftProf_WTD,
            FinalLiftOP[EventLiftProf_WTD].cast(DoubleType()),
        )
        log_dataframe(FinalLiftOP)

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
        FinalLiftOP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)
    return FinalLiftOP

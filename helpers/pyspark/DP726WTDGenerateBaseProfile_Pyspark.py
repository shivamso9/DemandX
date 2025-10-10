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
    mean,
    median,
    row_number,
    sum,
    to_date,
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
    HistoryPeriodWTDBase,
    MeasureofCentralTendency,
    MinWeeklyDemandWTD,
    Actuals,
    CurrentDay,
    HolidayType,
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
    plugin_name = "DP726WTDGenerateBaseProfile_Pyspark"
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
    BaseProf_WTD = "Base Profile WTD"
    ActualProf_WTD = "Actual Profile WTD"
    MeanMedianBaseProfile = "Central Tendency Profile WTD"

    logger.info("Extracting dimension cols ...")

    output_o9_grains = SelfOutput.columns

    # to get col_mapping for all columns
    output_df = pd.DataFrame(columns=output_o9_grains)
    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    BaseProfileOP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        HistoryPeriodWTDBase = col_namer.convert_to_pyspark_cols(HistoryPeriodWTDBase)
        SelfOutput = col_namer.convert_to_pyspark_cols(SelfOutput)
        MinWeeklyDemandWTD = col_namer.convert_to_pyspark_cols(MinWeeklyDemandWTD)
        MeasureofCentralTendency = col_namer.convert_to_pyspark_cols(MeasureofCentralTendency)
        HolidayType = col_namer.convert_to_pyspark_cols(HolidayType)
        CurrentDay = col_namer.convert_to_pyspark_cols(CurrentDay)
        ItemMaster = col_namer.convert_to_pyspark_cols(ItemMaster)
        AccountMaster = col_namer.convert_to_pyspark_cols(AccountMaster)
        ChannelMaster = col_namer.convert_to_pyspark_cols(ChannelMaster)
        RegionMaster = col_namer.convert_to_pyspark_cols(RegionMaster)
        PnLMaster = col_namer.convert_to_pyspark_cols(PnLMaster)
        DemandDomainMaster = col_namer.convert_to_pyspark_cols(DemandDomainMaster)
        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)
        LocationMaster = col_namer.convert_to_pyspark_cols(LocationMaster)

        if (
            (HistoryPeriodWTDBase.count() == 0)
            or (MeasureofCentralTendency.count() == 0)
            or (Actuals.count() == 0)
            or (CurrentDay.count() == 0)
            or (MinWeeklyDemandWTD.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        log_dataframe(HistoryPeriodWTDBase)
        log_dataframe(MeasureofCentralTendency)
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

        today = CurrentDay.select(to_date(col(day_col), "dd-MMM-yy")).first()[0]
        # Calculate the difference in days between today and the previous Saturday
        days_since_saturday = today.weekday()
        days_to_subtract = (
            days_since_saturday + 2
        ) % 7  # 0 represents Monday, so we need to add 2 to get the previous Saturday

        # Calculate the date of the previous Saturday
        today = today - timedelta(days=days_to_subtract)

        # Calculate limits based on today's day and month
        x = HistoryPeriodWTDBase.select(col("History Period WTD Base Profile").cast("int")).first()[
            0
        ]
        x = ((x + 6) // 7) * 7
        start_date = today - datetime.timedelta(days=x - 1)
        end_date = today

        Actuals = Actuals.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        Actuals = Actuals.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        BaseProfileOP = Actuals.filter((col(day_col) >= start_date) & (col(day_col) <= end_date))

        if HolidayType.count() > 0:
            HolidayType = HolidayType.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
            # Remove any week that has an event in it
            common_days = BaseProfileOP.select(day_col).intersect(HolidayType.select(day_col))
            # Get the weeks corresponding to the common days
            common_weeks = BaseProfileOP.join(common_days, day_col).select(week_col).distinct()
            # Filter out rows from df1 where the "Week" is in the common weeks
            BaseProfileOP = BaseProfileOP.join(common_weeks, week_col, "left_anti")

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

        # Calculate Weekly Demand by grouping and summing Actual values within the window
        BaseProfileOP = BaseProfileOP.withColumn(
            "Weekly Demand",
            sum(col("Actual")).over(window_spec),  # Apply the sum over the window
        )

        # Remove any week that has Weekly Demand < MinWeeklyDemandWTD
        MinDemand = MinWeeklyDemandWTD.select(
            col("Min Weekly Demand WTD Profile").cast("int")
        ).first()[0]
        BaseProfileOP = BaseProfileOP.filter(col("Weekly Demand") >= MinDemand)

        BaseProfileOP = BaseProfileOP.withColumn(
            ActualProf_WTD,
            (col("Actual") * 1.0) / col("Weekly Demand"),  # Calculate the ratio
        )

        # Group by 'Week Number' and calculate the median of 'Actual' for each occurrence number of all Week Numbers
        window_spec1 = Window.partitionBy(
            version_col,
            region_col,
            location_col,
            channel_col,
            pnl_col,
            item_col,
            demand_domain_col,
            account_col,
            week_col,
        ).orderBy(
            col(day_col)
        )  # Ensure consistent ordering within weeks

        # Add a row number column within each group
        BaseProfileOP = BaseProfileOP.withColumn(
            "day_number",
            row_number().over(window_spec1) - 1,  # Subtract 1 to start from 0
        )

        # Define another window for calculating the median
        ct_window_spec = Window.partitionBy(
            version_col,
            region_col,
            location_col,
            channel_col,
            pnl_col,
            item_col,
            demand_domain_col,
            account_col,
            "day_number",
        )

        # Calculate the median for each group and add as a new column

        if MeasureofCentralTendency.select(col("Measure of Central Tendency") == "Median").first()[
            0
        ]:
            BaseProfileOP = BaseProfileOP.withColumn(
                MeanMedianBaseProfile,
                median(col(ActualProf_WTD)).over(ct_window_spec),
            )
        else:
            BaseProfileOP = BaseProfileOP.withColumn(
                MeanMedianBaseProfile,
                mean(col(ActualProf_WTD)).over(ct_window_spec),
            )

        # Group by 'Week Number' and calculate the sum of 'Central Tendency' for each Week Number
        BaseProfileOP = BaseProfileOP.withColumn(
            "WeekCTSum",
            sum(col(MeanMedianBaseProfile)).over(window_spec),  # Apply the sum over the window
        )

        # Add a new column to store the ratio of 'CT' to 'WeekCTSum'
        BaseProfileOP = BaseProfileOP.withColumn(
            BaseProf_WTD,
            (col(MeanMedianBaseProfile) * 1.0) / col("WeekCTSum"),  # Calculate the ratio
        )

        columns_to_drop = [
            "Weekly Demand",
            "Actual",
            "WeekCTSum",
            "day_number",
            week_col,
        ]
        BaseProfileOP = BaseProfileOP.drop(*columns_to_drop)

        BaseProfileOP = BaseProfileOP.withColumn(
            day_col,
            date_format(col(day_col), "dd-MMM-yy"),  # Convert to specified format
        )

        BaseProfileOP = col_namer.convert_to_o9_cols(BaseProfileOP)

        BaseProfileOP = BaseProfileOP.withColumn(
            BaseProf_WTD, BaseProfileOP[BaseProf_WTD].cast(DoubleType())
        )
        BaseProfileOP = BaseProfileOP.dropDuplicates()
        log_dataframe(BaseProfileOP)
        # Your code ends here
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
        BaseProfileOP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)
    return BaseProfileOP

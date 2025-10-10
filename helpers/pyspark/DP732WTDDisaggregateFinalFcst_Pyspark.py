import datetime
import logging
from datetime import timedelta

import pandas as pd
import pyspark
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import Window
from pyspark.sql.functions import col, count, date_format, sum, to_date
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
    Actuals,
    FinalFcst,
    FcstSplitMech,
    CurrentDay,
    ForecastPeriodWTD,
    FinalProfile,
    ItemMaster,
    RegionMaster,
    AccountMaster,
    ChannelMaster,
    PnLMaster,
    DemandDomainMaster,
    TimeMaster,
    spark,
    df_keys,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    plugin_name = "DP732WTDDisaggregateFinalFcst_Pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version_VersionName"
    location_col = "Location_Location"
    account_actual = "Account_Account"
    channel_actual = "Channel_Channel"
    region_actual = "Region_Region"
    pnl_actual = "PnL_PnL"
    demdom_actual = "DemandDomain_DemandDomain"
    item_col = "Item_Item"
    week_col = "Time_Week"
    day_col = "Time_Day"
    plachannel = "Channel_PlanningChannel"
    plademdom = "DemandDomain_PlanningDemandDomain"
    plaregion = "Region_PlanningRegion"
    plaaccount = "Account_PlanningAccount"
    plapnl = "PnL_PlanningPnL"
    FinalProfile_WTD = "Final Profile WTD"
    FinalFcstDay = "Final Fcst Day"
    FinalFcstWeek = "FinalFcstWeek"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    output_o9_grains = SelfOutput.columns

    # to get col_mapping for all columns
    output_df = pd.DataFrame(columns=output_o9_grains)
    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_df.columns]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    Plugin4OP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        HistoryPeriodWTDBase = col_namer.convert_to_pyspark_cols(HistoryPeriodWTDBase)
        FinalFcst = col_namer.convert_to_pyspark_cols(FinalFcst)
        FcstSplitMech = col_namer.convert_to_pyspark_cols(FcstSplitMech)
        ForecastPeriodWTD = col_namer.convert_to_pyspark_cols(ForecastPeriodWTD)
        FinalProfile = col_namer.convert_to_pyspark_cols(FinalProfile)
        CurrentDay = col_namer.convert_to_pyspark_cols(CurrentDay)
        ItemMaster = col_namer.convert_to_pyspark_cols(ItemMaster)
        AccountMaster = col_namer.convert_to_pyspark_cols(AccountMaster)
        ChannelMaster = col_namer.convert_to_pyspark_cols(ChannelMaster)
        RegionMaster = col_namer.convert_to_pyspark_cols(RegionMaster)
        PnLMaster = col_namer.convert_to_pyspark_cols(PnLMaster)
        DemandDomainMaster = col_namer.convert_to_pyspark_cols(DemandDomainMaster)
        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)

        if (
            (FinalFcst.count() == 0)
            or (FcstSplitMech.count() == 0)
            or (ForecastPeriodWTD.count() == 0)
            or (CurrentDay.count() == 0)
            or (Actuals.count() == 0)
            or (HistoryPeriodWTDBase.count() == 0)
        ):
            raise Exception(
                "One or more input/s have no data. Please check. Exiting without execution."
            )

        log_dataframe(FinalFcst)
        log_dataframe(FcstSplitMech)
        log_dataframe(ForecastPeriodWTD)
        log_dataframe(Actuals)
        log_dataframe(CurrentDay)
        log_dataframe(HistoryPeriodWTDBase)

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

        # Add week to FinalFcst
        TimeMaster_unique = TimeMaster.select("Time_PartialWeek", week_col)
        TimeMaster_unique = TimeMaster_unique.dropDuplicates()
        FinalFcst = FinalFcst.join(TimeMaster_unique, on="Time_PartialWeek", how="inner")
        FinalFcst = FinalFcst.drop("Time_PartialWeek")
        FinalFcst = FinalFcst.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        FinalFcst = FinalFcst.filter((col(week_col) >= today) & (col(week_col) <= end_date))
        columns_to_drop = [
            "Time_WeekKey",
            "Time_DayKey",
            "Time_PartialWeek",
            "Time_PartialWeekKey",
        ]
        TimeMaster = TimeMaster.drop(*columns_to_drop)
        TimeMaster = TimeMaster.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
        TimeMaster = TimeMaster.withColumn(week_col, to_date(col(week_col), "dd-MMM-yy"))
        TimeMaster = TimeMaster.filter((col(day_col) >= today) & (col(day_col) <= end_date))

        window_spec = Window.partitionBy(
            version_col,
            plaregion,
            location_col,
            plachannel,
            plapnl,
            item_col,
            plademdom,
            plaaccount,
            week_col,
        )

        Plugin4OP = FinalFcst.withColumn(
            FinalFcstWeek,
            sum(col("Final Fcst")).over(window_spec),  # Apply the sum over the window
        )

        columns_to_drop = ["Final Fcst"]
        Plugin4OP = Plugin4OP.drop(*columns_to_drop)
        Plugin4OP = Plugin4OP.dropDuplicates()

        SplitMech = FcstSplitMech.select(col("Forecast Split Mechanism")).first()[0]

        if SplitMech or (FinalProfile.count() == 0):
            Plugin4OP = TimeMaster.join(Plugin4OP, on=week_col, how="left")
            Plugin4OP = Plugin4OP.withColumn(FinalFcstDay, col(FinalFcstWeek) / 7)
            columns_to_drop = [week_col, FinalFcstWeek]
            Plugin4OP = Plugin4OP.drop(*columns_to_drop)

        else:
            # Dynamic Grains
            columns = FinalProfile.columns
            account_col = next(column for column in columns if column.startswith("Account"))
            region_col = next(column for column in columns if column.startswith("Region"))
            channel_col = next(column for column in columns if column.startswith("Channel"))
            demand_domain_col = next(
                column for column in columns if column.startswith("DemandDomain")
            )
            pnl_col = next(column for column in columns if column.startswith("PnL"))
            ItemName = next(column for column in columns if column.startswith("Item"))

            FinalProfile = FinalProfile.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
            FinalProfile = TimeMaster.join(FinalProfile, on=day_col, how="left")

            # Mapping Higher Grain to Lower Grain in Final Profile
            if ItemName != item_col:
                ItemMaster = ItemMaster.select(ItemName, item_col)
                FinalProfile = FinalProfile.join(ItemMaster, on=ItemName, how="inner")
                FinalProfile = FinalProfile.drop(ItemName)

            if plaaccount != account_col:
                AccountMaster = AccountMaster.select(plaaccount, account_col)
                FinalProfile = FinalProfile.join(AccountMaster, on=account_col, how="inner")
                FinalProfile = FinalProfile.drop(account_col)

            if plachannel != channel_col:
                ChannelMaster = ChannelMaster.select(plachannel, channel_col)
                FinalProfile = FinalProfile.join(ChannelMaster, on=channel_col, how="inner")
                FinalProfile = FinalProfile.drop(channel_col)

            if plaregion != region_col:
                RegionMaster = RegionMaster.select(plaregion, region_col)
                FinalProfile = FinalProfile.join(RegionMaster, on=region_col, how="inner")
                FinalProfile = FinalProfile.drop(region_col)

            if plapnl != pnl_col:
                PnLMaster = PnLMaster.select(plapnl, pnl_col)
                FinalProfile = FinalProfile.join(PnLMaster, on=pnl_col, how="inner")
                FinalProfile = FinalProfile.drop(pnl_col)

            if plademdom != demand_domain_col:
                DemandDomainMaster = DemandDomainMaster.select(plademdom, demand_domain_col)
                FinalProfile = FinalProfile.join(
                    DemandDomainMaster, on=demand_domain_col, how="inner"
                )
                FinalProfile = FinalProfile.drop(demand_domain_col)

            Plugin4OP = FinalProfile.join(
                Plugin4OP,
                on=[
                    version_col,
                    plaregion,
                    location_col,
                    plachannel,
                    plapnl,
                    item_col,
                    plademdom,
                    plaaccount,
                    week_col,
                ],
                how="left",
            )

            # Test case: If History present < Base Period, then do equal split
            # Take Actual and filter for those items that have days < HistoryPeriodWTDBase
            x = HistoryPeriodWTDBase.select(
                col("History Period WTD Base Profile").cast("int")
            ).first()[0]
            window_spec = Window.partitionBy(
                version_col,
                account_actual,
                channel_actual,
                region_actual,
                pnl_actual,
                demdom_actual,
                location_col,
                item_col,
            )
            grouped_Actual = Actuals.withColumn("count", count("*").over(window_spec))
            # grouped_Actual = Actuals.groupBy(version_col, account_actual, channel_actual, region_actual, pnl_actual, demdom_actual, location_col, item_col).agg(count("*").alias("count"))
            filteredActual = grouped_Actual.filter(col("count") <= x).drop("count")
            filteredActual = filteredActual.withColumn(day_col, to_date(col(day_col), "dd-MMM-yy"))
            filteredActual = TimeMaster.join(filteredActual, on=day_col, how="left")
            filteredActual = (
                filteredActual.withColumnRenamed(account_actual, plaaccount)
                .withColumnRenamed(channel_actual, plachannel)
                .withColumnRenamed(region_actual, plaregion)
                .withColumnRenamed(pnl_actual, plapnl)
                .withColumnRenamed(demdom_actual, plademdom)
                .withColumnRenamed(day_col, "Day_Actual")
            )

            # Split Plugin4OP into two dfs: First df will have equal split, second will have profile based split
            EqualSplitDF = Plugin4OP.join(
                filteredActual,
                on=[
                    version_col,
                    plaregion,
                    location_col,
                    plachannel,
                    plapnl,
                    item_col,
                    plademdom,
                    plaaccount,
                    week_col,
                ],
                how="inner",
            )
            Plugin4OP = Plugin4OP.join(
                filteredActual,
                on=[
                    version_col,
                    plaregion,
                    location_col,
                    plachannel,
                    plapnl,
                    item_col,
                    plademdom,
                    plaaccount,
                ],
                how="anti",
            )
            columns_to_drop = ["Actual", "Day_Actual"]
            EqualSplitDF = EqualSplitDF.drop(*columns_to_drop)

            EqualSplitDF = EqualSplitDF.withColumn(FinalFcstDay, col(FinalFcstWeek) / 7)

            Plugin4OP = Plugin4OP.withColumn(
                FinalFcstDay, col(FinalFcstWeek) * col(FinalProfile_WTD)
            )

            Plugin4OP = Plugin4OP.unionByName(EqualSplitDF)

            columns_to_drop = [week_col, FinalFcstWeek, FinalProfile_WTD]
            Plugin4OP = Plugin4OP.drop(*columns_to_drop)

        Plugin4OP = Plugin4OP.withColumn(
            day_col,
            date_format(col(day_col), "dd-MMM-yy"),  # Convert to specified format
        )

        Plugin4OP = col_namer.convert_to_o9_cols(Plugin4OP)

        Plugin4OP = Plugin4OP.withColumn(FinalFcstDay, Plugin4OP[FinalFcstDay].cast(DoubleType()))
        log_dataframe(Plugin4OP)

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
        Plugin4OP = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)
    return Plugin4OP

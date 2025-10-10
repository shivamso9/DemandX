import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


def rename_cols_to_data_object_lvl(
    dataframe,
    item_col,
    channel_col,
    location_col,
    item_scope_col,
    channel_scope_col,
    location_scope_col,
    data_validation_col,
    exception_type_col,
):
    dataframe = dataframe.withColumnRenamed(item_scope_col, item_col)
    dataframe = dataframe.withColumnRenamed(channel_scope_col, channel_col)
    dataframe = dataframe.withColumnRenamed(location_scope_col, location_col)
    dataframe = dataframe.withColumnRenamed(exception_type_col, data_validation_col)

    return dataframe


def convert_cols_to_data_object_lvl(
    dataframe,
    item_master,
    channel_master,
    location_master,
    item_col,
    channel_col,
    location_col,
    pl_item_col,
    pl_channel_col,
    pl_location_col,
    item_lvl_col,
    channel_lvl_col,
    location_lvl_col,
    data_object_col,
):
    if pl_item_col != item_col:
        dataframe = dataframe.join(item_master, on=item_col, how="left")
        dataframe = dataframe.drop(item_col)

    if pl_channel_col != channel_col:
        dataframe = dataframe.join(channel_master, on=channel_col, how="left")
        dataframe = dataframe.drop(channel_col)

    if pl_location_col != location_col:
        dataframe = dataframe.join(location_master, on=location_col, how="left")
        dataframe = dataframe.drop(location_col)

    dataframe = dataframe.drop(
        item_lvl_col,
        channel_lvl_col,
        location_lvl_col,
        data_object_col,
    )

    return dataframe


@timed
def filter_on_process_order(
    intermediate_df,
    ProcessOrder,
    ItemMaster,
    ChannelMaster,
    LocationMaster,
    common_cols,
    pl_level_cols,
    scope_lvl_cols,
    data_validation_col,
    exception_type_col,
    dm_rule_col,
    process_order_col,
):
    version_col, data_object_col, item_lvl_col, channel_lvl_col, location_lvl_col = common_cols
    pl_item_col, pl_channel_col, pl_location_col = pl_level_cols
    item_scope_col, channel_scope_col, location_scope_col = scope_lvl_cols

    # extract the numeric part of 'dm_rule_col'
    intermediate_df = intermediate_df.withColumn(
        "dm_rule_num", F.split(dm_rule_col, "Rule_")[1].cast("int")
    )

    # group by the columns and get the maximum dm_rule_num for each group
    max_dm_rule_df = intermediate_df.groupBy(*common_cols, *scope_lvl_cols, exception_type_col).agg(
        F.max("dm_rule_num").alias("max_dm_rule_num")
    )

    # join back to keep the rows with the highest dm_rule_col
    intermediate_df = intermediate_df.join(
        max_dm_rule_df, on=[*common_cols, *scope_lvl_cols, exception_type_col], how="inner"
    ).filter(F.col("dm_rule_num") == F.col("max_dm_rule_num"))

    intermediate_df = intermediate_df.drop("max_dm_rule_num", "dm_rule_num", dm_rule_col)

    intermediate_df = intermediate_df.join(
        ProcessOrder,
        on=[version_col, data_object_col, item_lvl_col, channel_lvl_col, location_lvl_col],
        how="left",
    )

    # Group by 'Data Object' and create a separate DataFrame for each unique value of 'Data Object'
    dataframes_dict = {
        data_object: intermediate_df.filter(F.col(data_object_col) == data_object)
        for data_object in intermediate_df.select(data_object_col)
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    }

    merged_df = None
    for data_object, dataframe in dataframes_dict.items():
        # Split the data_object by ' x ' and remove spaces
        parts = data_object.split(" x ")

        # Extract the parts and remove spaces from each part
        item_col = "Item_" + parts[0].replace(" ", "")  # Extract 'L2' and create like 'Item_L2'
        channel_col = "Channel_" + parts[1].replace(
            " ", ""
        )  # Extract 'Channel' and create like 'Channel_Channel'
        location_col = "Location_" + parts[2].replace(
            " ", ""
        )  # Extract 'Location Country' and create like 'Location_LocationCountry'

        # rename dp scope cols to data object level cols
        dataframe = rename_cols_to_data_object_lvl(
            dataframe,
            item_col,
            channel_col,
            location_col,
            item_scope_col,
            channel_scope_col,
            location_scope_col,
            data_validation_col,
            exception_type_col,
        )

        # select relevant data object level cols
        item_master = ItemMaster.select(item_col, pl_item_col).dropDuplicates()
        channel_master = ChannelMaster.select(channel_col, pl_channel_col).dropDuplicates()
        location_master = LocationMaster.select(location_col, pl_location_col).dropDuplicates()

        dataframe = convert_cols_to_data_object_lvl(
            dataframe,
            item_master,
            channel_master,
            location_master,
            item_col,
            channel_col,
            location_col,
            pl_item_col,
            pl_channel_col,
            pl_location_col,
            item_lvl_col,
            channel_lvl_col,
            location_lvl_col,
            data_object_col,
        )

        if merged_df is None:
            merged_df = dataframe
        else:
            merged_df = merged_df.unionByName(dataframe)

    group_cols = [pl_item_col, pl_channel_col, pl_location_col]

    # for rows with the same group, select the row with the highest process order.
    max_process_order_df = merged_df.groupBy(group_cols).agg(
        F.max(process_order_col).alias("max_process_order")
    )

    # join to keep the rows with max process order
    merged_df = merged_df.join(max_process_order_df, on=group_cols, how="left").filter(
        F.col(process_order_col) == F.col("max_process_order")
    )

    merged_df = merged_df.drop(process_order_col, "max_process_order")

    return merged_df


@timed
def convert_to_planning_lvl(
    intermediate_df,
    ChannelMaster,
    LocationMaster,
    RegionMaster,
    AccountMaster,
    DemandDomainMaster,
    PnLMaster,
    account_col,
    channel_col,
    demand_domain_col,
    location_col,
    pnl_col,
    region_col,
    pl_channel_col,
    pl_location_col,
):
    intermediate_df = intermediate_df.join(AccountMaster, on=account_col, how="inner")
    intermediate_df = intermediate_df.drop(account_col)

    intermediate_df = intermediate_df.join(
        ChannelMaster.select(channel_col, pl_channel_col).dropDuplicates(),
        on=channel_col,
        how="inner",
    )
    intermediate_df = intermediate_df.drop(channel_col)

    intermediate_df = intermediate_df.join(DemandDomainMaster, on=demand_domain_col, how="inner")
    intermediate_df = intermediate_df.drop(demand_domain_col)

    intermediate_df = intermediate_df.join(
        LocationMaster.select(location_col, pl_location_col).dropDuplicates(),
        on=location_col,
        how="inner",
    )
    intermediate_df = intermediate_df.drop(location_col)

    intermediate_df = intermediate_df.join(PnLMaster, on=pnl_col, how="inner")
    intermediate_df = intermediate_df.drop(pnl_col)

    intermediate_df = intermediate_df.join(RegionMaster, on=region_col, how="inner")
    intermediate_df = intermediate_df.drop(region_col)

    return intermediate_df


@timed
def calculate_past_promo_periods(
    intermediate_df,
    PromoDates,
    IntroandDiscoDate,
    LYcurrWeekDate,
    dimensions,
    window_length_col,
    pl_demand_domain_col,
    currDayKey,
    intro_date_col,
    disco_date_col,
    intiative_col,
    promo_start_date_col,
    promo_end_date_col,
    past_promo_periods_col,
):
    if PromoDates.isEmpty():
        intermediate_df = intermediate_df.withColumn(past_promo_periods_col, F.lit(0))
        return intermediate_df

    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        intro_date_col, F.to_date(F.col(intro_date_col), "MM/dd/yyyy hh:mm:ss a")
    )
    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        disco_date_col, F.to_date(F.col(disco_date_col), "MM/dd/yyyy hh:mm:ss a")
    )

    # join the IntroandDiscoDate and PromoDates
    intermediate_df = intermediate_df.join(IntroandDiscoDate, on=dimensions, how="inner")
    intermediate_df = intermediate_df.join(PromoDates, on=dimensions, how="inner")

    intermediate_df = intermediate_df.drop(intiative_col)

    # add the current week day column
    intermediate_df = intermediate_df.withColumn(
        "currWeekDay", F.to_date(F.lit(currDayKey), "dd-MMM-yy")
    )

    # add the this year last week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "TYlastWeekDay",
        F.to_date(F.date_sub(F.col("currWeekDay"), (F.col(window_length_col) * 7).cast("int"))),
    )

    # add the last year current week day
    intermediate_df = intermediate_df.withColumn(
        "LYcurrWeekDate", F.to_date(F.lit(LYcurrWeekDate), "dd-MMM-yy")
    )

    # add the last year next week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "LYnextWeekDate",
        F.to_date(
            F.date_add(
                F.col("LYcurrWeekDate"),
                (F.col(window_length_col) * 7).cast("int"),
            )
        ),
    )

    intermediate_df = intermediate_df.withColumn("currWeekDay", F.date_sub(F.col("currWeekDay"), 1))
    intermediate_df = intermediate_df.withColumn(
        "LYnextWeekDate", F.date_sub(F.col("LYnextWeekDate"), 1)
    )

    # compute the new start date (max of TYlastWeekDay, Intro Date and Promo Start Date)
    intermediate_df = intermediate_df.withColumn(
        "TYStartDate",
        F.to_date(
            F.greatest(
                F.col("TYlastWeekDay"),
                F.to_date(F.col(intro_date_col)),
                F.to_date(F.col(promo_start_date_col)),
            )
        ),
    )

    # compute the new end date (max of currWeekDay, Disco Date and Promo End Date)
    intermediate_df = intermediate_df.withColumn(
        "TYEndDate",
        F.to_date(
            F.least(
                F.col("currWeekDay"),
                F.to_date(F.col(disco_date_col)),
                F.to_date(F.col(promo_end_date_col)),
            )
        ),
    )

    # compute the new start date (max of LYcurrWeekDate, Intro Date and Promo Start Date)
    intermediate_df = intermediate_df.withColumn(
        "LYStartDate",
        F.to_date(
            F.greatest(
                F.col("LYcurrWeekDate"),
                F.to_date(F.col(intro_date_col)),
                F.to_date(F.col(promo_start_date_col)),
            )
        ),
    )

    # compute the new end date (max of LYnextWeekDate, Disco Date and Promo End Date))
    intermediate_df = intermediate_df.withColumn(
        "LYEndDate",
        F.to_date(
            F.least(
                F.col("LYnextWeekDate"),
                F.to_date(F.col(disco_date_col)),
                F.to_date(F.col(promo_end_date_col)),
            )
        ),
    )

    # count the number of promo days
    intermediate_df = intermediate_df.withColumn(
        past_promo_periods_col,
        F.when(
            F.col(pl_demand_domain_col) == "Core",
            F.greatest(F.datediff(F.col("LYEndDate"), F.col("LYStartDate")) + 1, F.lit(0))
            + F.greatest(F.datediff(F.col("TYEndDate"), F.col("TYStartDate")) + 1, F.lit(0)),
        ).otherwise(F.greatest(F.datediff(F.col("TYEndDate"), F.col("TYStartDate")) + 1, F.lit(0))),
    )

    # Drop intermediate columns
    intermediate_df = intermediate_df.drop(
        "currWeekDay", "TYlastWeekDay", "LYcurrWeekDate", "LYnextWeekDate"
    )
    intermediate_df = intermediate_df.drop(promo_start_date_col, promo_end_date_col)
    intermediate_df = intermediate_df.drop("TYStartDate", "TYEndDate", "LYStartDate", "LYEndDate")
    intermediate_df = intermediate_df.drop(intro_date_col, disco_date_col)

    req_cols = intermediate_df.columns
    req_cols.remove(past_promo_periods_col)

    intermediate_df = intermediate_df.groupBy(req_cols).agg(
        F.sum(past_promo_periods_col).alias("past_promo_periods")
    )

    intermediate_df = intermediate_df.withColumnRenamed(
        "past_promo_periods", past_promo_periods_col
    )

    intermediate_df = intermediate_df.withColumn(
        past_promo_periods_col, F.when(F.col(past_promo_periods_col) > 0, 1).otherwise(0)
    )

    intermediate_df = intermediate_df.dropDuplicates()

    return intermediate_df


@timed
def calculate_past_holiday_periods(
    intermediate_df,
    IntroandDiscoDate,
    HolidayDates,
    dimensions,
    common_cols,
    LYcurrWeekDate,
    pl_demand_domain_col,
    window_length_col,
    currDayKey,
    intro_date_col,
    disco_date_col,
    holiday_col,
    past_holiday_periods_col,
    day_col,
    version_col,
    pl_location_col,
):
    if HolidayDates.isEmpty():
        intermediate_df = intermediate_df.withColumn(past_holiday_periods_col, F.lit(0))
        return intermediate_df

    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        intro_date_col, F.to_date(F.col(intro_date_col), "MM/dd/yyyy hh:mm:ss a")
    )
    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        disco_date_col, F.to_date(F.col(disco_date_col), "MM/dd/yyyy hh:mm:ss a")
    )

    # join the IntroandDiscoDate and PromoDates
    intermediate_df = intermediate_df.join(IntroandDiscoDate, on=dimensions, how="inner")

    # add the current week day column
    intermediate_df = intermediate_df.withColumn(
        "currWeekDay", F.to_date(F.lit(currDayKey), "dd-MMM-yy")
    )

    # add the this year last week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "TYlastWeekDay",
        F.to_date(F.date_sub(F.col("currWeekDay"), (F.col(window_length_col) * 7).cast("int"))),
    )

    # add the last year current week day
    intermediate_df = intermediate_df.withColumn(
        "LYcurrWeekDate", F.to_date(F.lit(LYcurrWeekDate), "dd-MMM-yy")
    )

    # add the last year next week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "LYnextWeekDate",
        F.to_date(
            F.date_add(
                F.col("LYcurrWeekDate"),
                (F.col(window_length_col) * 7).cast("int"),
            )
        ),
    )

    intermediate_df = intermediate_df.withColumn("currWeekDay", F.date_sub(F.col("currWeekDay"), 1))
    intermediate_df = intermediate_df.withColumn(
        "LYnextWeekDate", F.date_sub(F.col("LYnextWeekDate"), 1)
    )

    # compute the new start date (max of TYlastWeekDay, Intro Date)
    intermediate_df = intermediate_df.withColumn(
        "TYStartDate",
        F.to_date(F.greatest(F.col("TYlastWeekDay"), F.to_date(F.col(intro_date_col)))),
    )

    # compute the new end date (max of currWeekDay, Disco Date)
    intermediate_df = intermediate_df.withColumn(
        "TYEndDate", F.to_date(F.least(F.col("currWeekDay"), F.to_date(F.col(disco_date_col))))
    )

    # compute the new start date (max of LYcurrWeekDate, Intro Date)
    intermediate_df = intermediate_df.withColumn(
        "LYStartDate",
        F.to_date(F.greatest(F.col("LYcurrWeekDate"), F.to_date(F.col(intro_date_col)))),
    )

    # compute the new end date (max of LYnextWeekDate, Disco Date)
    intermediate_df = intermediate_df.withColumn(
        "LYEndDate", F.to_date(F.least(F.col("LYnextWeekDate"), F.to_date(F.col(disco_date_col))))
    )

    # join HolidayDates
    intermediate_df = intermediate_df.join(
        HolidayDates, on=[version_col, pl_location_col], how="inner"
    )
    intermediate_df = intermediate_df.withColumn(day_col, F.to_date(day_col, "dd-MMM-yy"))

    # filter days in range [TYStartDate, TYEndDate]
    intermediate_df1 = intermediate_df.filter(
        (F.col(day_col) >= F.col("TYStartDate")) & (F.col(day_col) <= F.col("TYEndDate"))
    )

    # filter days in range [LYStartDate, LYEndDate]
    intermediate_df2 = intermediate_df.filter(
        (F.col(day_col) >= F.col("LYStartDate")) & (F.col(day_col) <= F.col("LYEndDate"))
    )

    # count the number of holidays this year
    intermediate_df1 = intermediate_df1.groupBy(
        dimensions + common_cols + [window_length_col, "TYStartDate", "TYEndDate"]
    ).agg(F.count(day_col).alias("TY_holiday_periods"))

    # count the number of holidays last year
    intermediate_df2 = intermediate_df2.groupBy(
        dimensions + common_cols + [window_length_col, "LYStartDate", "LYEndDate"]
    ).agg(F.count(day_col).alias("LY_holiday_periods"))

    # join both dataframes
    intermediate_df = intermediate_df1.join(
        intermediate_df2, on=dimensions + common_cols + [window_length_col], how="outer"
    )

    intermediate_df = intermediate_df.fillna(0)

    # calculate promo periods for core and non-core
    intermediate_df = intermediate_df.withColumn(
        past_holiday_periods_col,
        F.when(
            F.col(pl_demand_domain_col) == "Core",
            F.col("TY_holiday_periods") + F.col("LY_holiday_periods"),
        ).otherwise(F.col("TY_holiday_periods")),
    )

    intermediate_df = intermediate_df.withColumn(
        past_holiday_periods_col, F.when(F.col(past_holiday_periods_col) > 0, 1).otherwise(0)
    )

    # Drop intermediate columns
    intermediate_df = intermediate_df.drop(
        "currWeekDay", "TYlastWeekDay", "LYcurrWeekDate", "LYnextWeekDate"
    )
    intermediate_df = intermediate_df.drop("TYStartDate", "TYEndDate", "LYStartDate", "LYEndDate")
    intermediate_df = intermediate_df.drop("TY_holiday_periods", "LY_holiday_periods")

    intermediate_df = intermediate_df.dropDuplicates()

    return intermediate_df


@timed
def calculate_future_promo_periods(
    intermediate_df,
    PromoDates,
    IntroandDiscoDate,
    dimensions,
    window_length_col,
    currDayKey,
    intro_date_col,
    disco_date_col,
    intiative_col,
    promo_start_date_col,
    promo_end_date_col,
    future_promo_periods_col,
):
    if PromoDates.isEmpty():
        intermediate_df = intermediate_df.withColumn(future_promo_periods_col, F.lit(0))
        return intermediate_df

    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        intro_date_col, F.to_date(F.col(intro_date_col), "MM/dd/yyyy hh:mm:ss a")
    )
    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        disco_date_col, F.to_date(F.col(disco_date_col), "MM/dd/yyyy hh:mm:ss a")
    )

    # join the IntroandDiscoDate and PromoDates
    intermediate_df = intermediate_df.join(IntroandDiscoDate, on=dimensions, how="inner")
    intermediate_df = intermediate_df.join(PromoDates, on=dimensions, how="inner")
    intermediate_df = intermediate_df.drop(intiative_col)

    # add the current week day column
    intermediate_df = intermediate_df.withColumn(
        "currWeekDay", F.to_date(F.lit(currDayKey), "dd-MMM-yy")
    )

    # add the next week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "nextWeekDay",
        F.to_date(F.date_add(F.col("currWeekDay"), (F.col(window_length_col) * 7).cast("int"))),
    )
    intermediate_df = intermediate_df.withColumn("nextWeekDay", F.date_sub(F.col("nextWeekDay"), 1))

    # compute the new start date (max of currWeekDay, Intro Date and Promo Start Date)
    intermediate_df = intermediate_df.withColumn(
        "newStartDate",
        F.to_date(
            F.greatest(
                F.col("currWeekDay"),
                F.to_date(F.col(intro_date_col), "dd-MMM-yy"),
                F.to_date(F.col(promo_start_date_col), "dd-MMM-yy"),
            )
        ),
    )

    # compute the new end date (max of nextWeekDay, Disco Date and Promo End Date)
    intermediate_df = intermediate_df.withColumn(
        "newEndDate",
        F.to_date(
            F.least(
                F.col("nextWeekDay"),
                F.to_date(F.col(disco_date_col)),
                F.to_date(F.col(promo_end_date_col)),
            )
        ),
    )

    # count the number of promo days
    intermediate_df = intermediate_df.withColumn(
        future_promo_periods_col,
        F.greatest(F.datediff(F.col("newEndDate"), F.col("newStartDate")) + 1, F.lit(0)),
    )

    # Drop intermediate columns
    intermediate_df = intermediate_df.drop("currWeekDay", "nextWeekDay")
    intermediate_df = intermediate_df.drop(
        promo_start_date_col, promo_end_date_col, "newStartDate", "newEndDate"
    )
    intermediate_df = intermediate_df.drop(intro_date_col, disco_date_col)

    req_cols = intermediate_df.columns
    req_cols.remove(future_promo_periods_col)

    intermediate_df = intermediate_df.groupBy(req_cols).agg(
        F.sum(future_promo_periods_col).alias("future_promo_periods")
    )

    intermediate_df = intermediate_df.withColumnRenamed(
        "future_promo_periods", future_promo_periods_col
    )

    intermediate_df = intermediate_df.withColumn(
        future_promo_periods_col, F.when(F.col(future_promo_periods_col) > 0, 1).otherwise(0)
    )

    intermediate_df = intermediate_df.dropDuplicates()

    return intermediate_df


@timed
def calculate_future_holiday_periods(
    intermediate_df,
    IntroandDiscoDate,
    HolidayDates,
    dimensions,
    common_cols,
    window_length_col,
    currDayKey,
    intro_date_col,
    disco_date_col,
    holiday_col,
    future_holiday_periods_col,
    day_col,
    version_col,
    pl_location_col,
):
    if HolidayDates.isEmpty():
        intermediate_df = intermediate_df.withColumn(future_holiday_periods_col, F.lit(0))
        return intermediate_df

    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        intro_date_col, F.to_date(F.col(intro_date_col), "MM/dd/yyyy hh:mm:ss a")
    )
    IntroandDiscoDate = IntroandDiscoDate.withColumn(
        disco_date_col, F.to_date(F.col(disco_date_col), "MM/dd/yyyy hh:mm:ss a")
    )

    # join the IntroandDiscoDate and PromoDates
    intermediate_df = intermediate_df.join(IntroandDiscoDate, on=dimensions, how="inner")

    # add the current week day column
    intermediate_df = intermediate_df.withColumn(
        "currWeekDay", F.to_date(F.lit(currDayKey), "dd-MMM-yy")
    )

    # add the next week day based on window length
    intermediate_df = intermediate_df.withColumn(
        "nextWeekDay",
        F.to_date(F.date_add(F.col("currWeekDay"), (F.col(window_length_col) * 7).cast("int"))),
    )
    intermediate_df = intermediate_df.withColumn("nextWeekDay", F.date_sub(F.col("nextWeekDay"), 1))

    # compute the new start date (max of Intro Date and currWeekDay)
    intermediate_df = intermediate_df.withColumn(
        "newStartDate",
        F.to_date(F.greatest(F.col("currWeekDay"), F.to_date(F.col(intro_date_col)))),
    )

    # compute the new end date (min of nextWeekDay and Disco Date)
    intermediate_df = intermediate_df.withColumn(
        "newEndDate", F.to_date(F.least(F.col("nextWeekDay"), F.to_date(F.col(disco_date_col))))
    )

    # join HolidayDates
    intermediate_df = intermediate_df.join(
        HolidayDates, on=[version_col, pl_location_col], how="inner"
    )
    intermediate_df = intermediate_df.withColumn(day_col, F.to_date(day_col, "dd-MMM-yy"))

    # filter days in range [newStartDate, newEndDate]
    intermediate_df = intermediate_df.filter(
        (F.col(day_col) >= F.col("newStartDate")) & (F.col(day_col) <= F.col("newEndDate"))
    )

    # count the number of holidays
    intermediate_df = intermediate_df.groupBy(
        dimensions + common_cols + [window_length_col, "newStartDate", "newEndDate"]
    ).agg(F.count(day_col).alias(future_holiday_periods_col))

    intermediate_df = intermediate_df.withColumn(
        future_holiday_periods_col, F.when(F.col(future_holiday_periods_col) > 0, 1).otherwise(0)
    )

    # Drop intermediate columns
    intermediate_df = intermediate_df.drop("currWeekDay", "nextWeekDay")
    intermediate_df = intermediate_df.drop("newStartDate", "newEndDate")

    intermediate_df = intermediate_df.dropDuplicates()

    return intermediate_df


# Define main function
@timed
@log_inputs_and_outputs
def main(
    ConsensusFcst,
    ConsensusFcstLC,
    FinalFcst,
    Actual,
    StoreInventoryActual,
    TimeMaster,
    ItemMaster,
    RegionMaster,
    LocationMaster,
    ChannelMaster,
    PnLMaster,
    DemandDomainMaster,
    AccountMaster,
    CurrentTimePeriod,
    IntroandDiscoDate,
    WindowLength,
    Threshold,
    MinHistoryPeriods,
    EOLPeriods,
    NPeriods,
    Weight,
    UntilDate,
    PromoDates,
    HolidayDates,
    ProcessOrder,
    WindowLength_param,
    Threshold_param,
    NPeriods_param,
    MinHistory_param,
    EOL_param,
    SizeProfilerThreshold_param,
    coc_exception_col,
    zero_fcst_exception_col,
    system_vs_naive_exception_col,
    size_profiler_exception_col,
    spark,
    df_keys,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    plugin_name = "DP850ExceptionsWeek_Pyspark"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # input cols
    version_col = "Version_VersionName"
    pl_location_col = "Location_PlanningLocation"
    pl_account_col = "Account_PlanningAccount"
    pl_channel_col = "Channel_PlanningChannel"
    pl_region_col = "Region_PlanningRegion"
    pl_pnl_col = "PnL_PlanningPnL"
    pl_demand_domain_col = "DemandDomain_PlanningDemandDomain"
    pl_item_col = "Item_PlanningItem"

    location_col = "Location_Location"
    account_col = "Account_Account"
    channel_col = "Channel_Channel"
    region_col = "Region_Region"
    pnl_col = "PnL_PnL"
    demand_domain_col = "DemandDomain_DemandDomain"
    item_col = "Item_Item"

    data_object_col = "DataObject_DataObject"
    data_validation_col = "DataValidation_DataValidation"
    exception_type_col = "DP Exception Type"
    dm_rule_col = "DMRule_Rule"
    intiative_col = "Initiative_Initiative"

    do_item_lvl_col = "Data Object Item Level"
    do_location_lvl_col = "Data Object Location Level"
    do_channel_lvl_col = "Data Object Channel Level"

    item_lvl_col = "DP Item Level"
    location_lvl_col = "DP Location Level"
    channel_lvl_col = "DP Channel Level"

    item_scope_col = "DP Item Scope"
    location_scope_col = "DP Location Scope"
    channel_scope_col = "DP Channel Scope"

    partial_week_col = "Time_PartialWeek"
    week_col = "Time_Week"
    day_col = "Time_Day"
    day_key_col = "Time_DayKey"
    intro_date_col = "Intro Date"
    disco_date_col = "Disco Date"

    actual_col = "Actual"
    store_inventory_actual_col = "Store Inventory Actual"
    final_fcst_col = "Final Fcst"
    consFcst_col = "Consensus Fcst"
    consFcstLC_col = "Consensus Fcst LC"

    promo_start_date_col = "Promo Start Date LL"
    promo_end_date_col = "Promo End Date LL"
    holiday_col = "Is Holiday"
    process_order_col = "Data Object Process Order"

    window_length_col = "DP Exception Calculation Window"
    threshold_col = "DP Exception Tolerance"
    min_hist_periods_col = "DP Exception Min History Period"
    eol_periods_col = "DP Exception Min Forecast Period"
    n_periods_col = "DP Exception Display Window"
    weight_col = "DP Exception Type Priority"

    # output cols
    coc_consFcst_col = "DP Exception Type 1 Input 1"
    coc_consFcstLC_col = "DP Exception Type 1 Input 2"
    coc_tolerance_col = "DP Exception Type 1 Tolerance"
    coc_tolerance_deviation_col = "DP Exception Type 1 Tolerance Deviation"
    coc_future_promo_periods_col = "DP Exception Type 1 Future Promo Periods"
    coc_future_promo_periods_LC_col = "DP Exception Type 1 Promo Periods LC"
    coc_future_holiday_periods_col = "DP Exception Type 1 Future Holiday Periods"
    coc_future_holiday_periods_LC_col = "DP Exception Type 1 Holiday Periods LC"
    coc_flag_col = "DP Exception Flag 1"

    size_profiler_consFcst_col = "DP Exception Type 2 Input 1"
    size_profiler_finalFcst_col = "DP Exception Type 2 Input 2"
    size_profiler_tolerance_col = "DP Exception Type 2 Tolerance"
    size_profiler_tolerance_devaition_col = "DP Exception Type 2 Tolerance Deviation"
    size_profiler_flag_col = "DP Exception Flag 2"

    system_vs_naive_consFcst_col = "DP Exception Type 3 Input 1"
    system_vs_naive_actualAvg_col = "DP Exception Type 3 Input 2"
    system_vs_naive_tolerance_col = "DP Exception Type 3 Tolerance"
    system_vs_naive_tolerance_devaition_col = "DP Exception Type 3 Tolerance Deviation"
    system_vs_naive_past_promo_periods_col = "DP Exception Type 3 Past Promo Periods"
    system_vs_naive_future_promo_periods_col = "DP Exception Type 3 Future Promo Periods"
    system_vs_naive_past_holiday_periods_col = "DP Exception Type 3 Past Holiday Periods"
    system_vs_naive_future_holiday_periods_col = "DP Exception Type 3 Future Holiday Periods"
    system_vs_naive_out_of_stock_col = "DP Exception Type 3 Out of Stock Periods"
    system_vs_naive_flag_col = "DP Exception Flag 3"

    zero_fcst_flag_col = "DP Exception Flag 4"

    total_weight_col = "DP Exception Weight"
    consFcst_Nperiods_col = "DP Exception Forecast"
    actual_Nperiods_col = "DP Exception Actuals"

    dimensions = [
        version_col,
        pl_item_col,
        pl_channel_col,
        pl_location_col,
        pl_account_col,
        pl_region_col,
        pl_pnl_col,
        pl_demand_domain_col,
    ]

    # create output_df
    output_col = dimensions + [data_validation_col]

    output_df_schema = StructType(
        [StructField(col_name, StringType(), True) for col_name in output_col]
    )
    emp_RDD = spark.sparkContext.emptyRDD()
    Output = spark.createDataFrame(data=emp_RDD, schema=output_df_schema)

    try:
        ConsensusFcst = col_namer.convert_to_pyspark_cols(ConsensusFcst)
        ConsensusFcstLC = col_namer.convert_to_pyspark_cols(ConsensusFcstLC)
        Time = col_namer.convert_to_pyspark_cols(TimeMaster)
        WindowLength = col_namer.convert_to_pyspark_cols(WindowLength)
        Threshold = col_namer.convert_to_pyspark_cols(Threshold)
        CurrentTimePeriod = col_namer.convert_to_pyspark_cols(CurrentTimePeriod)
        Actual = col_namer.convert_to_pyspark_cols(Actual)
        FinalFcst = col_namer.convert_to_pyspark_cols(FinalFcst)
        IntroandDiscoDate = col_namer.convert_to_pyspark_cols(IntroandDiscoDate)
        MinHistoryPeriods = col_namer.convert_to_pyspark_cols(MinHistoryPeriods)
        EOLPeriods = col_namer.convert_to_pyspark_cols(EOLPeriods)
        NPeriods = col_namer.convert_to_pyspark_cols(NPeriods)
        Weight = col_namer.convert_to_pyspark_cols(Weight)
        PromoDates = col_namer.convert_to_pyspark_cols(PromoDates)
        StoreInventoryActual = col_namer.convert_to_pyspark_cols(StoreInventoryActual)
        HolidayDates = col_namer.convert_to_pyspark_cols(HolidayDates)
        ProcessOrder = col_namer.convert_to_pyspark_cols(ProcessOrder)
        UntilDate = col_namer.convert_to_pyspark_cols(UntilDate)
        ItemMaster = col_namer.convert_to_pyspark_cols(ItemMaster)
        RegionMaster = col_namer.convert_to_pyspark_cols(RegionMaster)
        LocationMaster = col_namer.convert_to_pyspark_cols(LocationMaster)
        ChannelMaster = col_namer.convert_to_pyspark_cols(ChannelMaster)
        PnLMaster = col_namer.convert_to_pyspark_cols(PnLMaster)
        DemandDomainMaster = col_namer.convert_to_pyspark_cols(DemandDomainMaster)
        AccountMaster = col_namer.convert_to_pyspark_cols(AccountMaster)

        # Extracting Current Week
        currWeek = CurrentTimePeriod.select(week_col).first()[0]
        currDayKey = CurrentTimePeriod.select(day_col).first()[0]

        # Validate input DataFrames
        if ConsensusFcst.isEmpty():
            raise ValueError(
                "ConsensusFcst dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif ConsensusFcstLC.isEmpty():
            raise ValueError(
                "ConsensusFcstLC dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif Time.isEmpty():
            raise ValueError("Time dataframe cannot be empty! Check logs/ inputs for error!")
        elif Actual.isEmpty():
            raise ValueError("Actual dataframe cannot be empty! Check logs/ inputs for error!")
        elif FinalFcst.isEmpty():
            raise ValueError("FinalFcst dataframe cannot be empty! Check logs/ inputs for error!")
        elif ItemMaster.isEmpty():
            raise ValueError("ItemMaster dataframe cannot be empty! Check logs/ inputs for error!")
        elif RegionMaster.isEmpty():
            raise ValueError(
                "RegionMaster dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif LocationMaster.isEmpty():
            raise ValueError(
                "LocationMaster dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif ChannelMaster.isEmpty():
            raise ValueError(
                "ChannelMaster dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif PnLMaster.isEmpty():
            raise ValueError("PnLMaster dataframe cannot be empty! Check logs/ inputs for error!")
        elif DemandDomainMaster.isEmpty():
            raise ValueError(
                "DemandDomainMaster dataframe cannot be empty! Check logs/ inputs for error!"
            )
        elif AccountMaster.isEmpty():
            raise ValueError(
                "AccountMaster dataframe cannot be empty! Check logs/ inputs for error!"
            )

        logger.info("Converting grains from Higher level to Planning Level ...")

        # -------------------- Convert Higher Level into Planning Levels --------------------------------------------
        ProcessOrder = (
            ProcessOrder.withColumnRenamed(do_item_lvl_col, item_lvl_col)
            .withColumnRenamed(do_channel_lvl_col, channel_lvl_col)
            .withColumnRenamed(do_location_lvl_col, location_lvl_col)
        )

        # Check if there are any null values in 'process_order'
        null_rows = ProcessOrder.filter(F.col(process_order_col).isNull())

        # Check if the DataFrame has more than one row
        row_count = len(ProcessOrder.head(2)) > 1

        # If there are null values in 'process_order' and more than one row, raise an error
        if not null_rows.isEmpty() > 0 and row_count:
            raise ValueError(
                "Found missing Data Object Process Orders, and the ProcessOrder DataFrame has more than one row."
            )

        ProcessOrder = ProcessOrder.fillna({process_order_col: 1})

        common_cols = [
            version_col,
            data_object_col,
            item_lvl_col,
            channel_lvl_col,
            location_lvl_col,
        ]
        pl_level_cols = [pl_item_col, pl_channel_col, pl_location_col]
        scope_lvl_cols = [item_scope_col, channel_scope_col, location_scope_col]

        WindowLength = filter_on_process_order(
            intermediate_df=WindowLength,
            ProcessOrder=ProcessOrder,
            ItemMaster=ItemMaster,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            common_cols=common_cols,
            pl_level_cols=pl_level_cols,
            scope_lvl_cols=scope_lvl_cols,
            data_validation_col=data_validation_col,
            exception_type_col=exception_type_col,
            dm_rule_col=dm_rule_col,
            process_order_col=process_order_col,
        )

        Threshold = filter_on_process_order(
            intermediate_df=Threshold,
            ProcessOrder=ProcessOrder,
            ItemMaster=ItemMaster,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            common_cols=common_cols,
            pl_level_cols=pl_level_cols,
            scope_lvl_cols=scope_lvl_cols,
            data_validation_col=data_validation_col,
            exception_type_col=exception_type_col,
            dm_rule_col=dm_rule_col,
            process_order_col=process_order_col,
        )

        MinHistoryPeriods = filter_on_process_order(
            intermediate_df=MinHistoryPeriods,
            ProcessOrder=ProcessOrder,
            ItemMaster=ItemMaster,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            common_cols=common_cols,
            pl_level_cols=pl_level_cols,
            scope_lvl_cols=scope_lvl_cols,
            data_validation_col=data_validation_col,
            exception_type_col=exception_type_col,
            dm_rule_col=dm_rule_col,
            process_order_col=process_order_col,
        )

        EOLPeriods = filter_on_process_order(
            intermediate_df=EOLPeriods,
            ProcessOrder=ProcessOrder,
            ItemMaster=ItemMaster,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            common_cols=common_cols,
            pl_level_cols=pl_level_cols,
            scope_lvl_cols=scope_lvl_cols,
            data_validation_col=data_validation_col,
            exception_type_col=exception_type_col,
            dm_rule_col=dm_rule_col,
            process_order_col=process_order_col,
        )

        NPeriods = filter_on_process_order(
            intermediate_df=NPeriods,
            ProcessOrder=ProcessOrder,
            ItemMaster=ItemMaster,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            common_cols=common_cols,
            pl_level_cols=pl_level_cols,
            scope_lvl_cols=scope_lvl_cols,
            data_validation_col=data_validation_col,
            exception_type_col=exception_type_col,
            dm_rule_col=dm_rule_col,
            process_order_col=process_order_col,
        )
        logger.info("Completed conversion of grains...")

        # Mapping PromoDates Item to Planning Item
        PromoDates = convert_to_planning_lvl(
            intermediate_df=PromoDates,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            RegionMaster=RegionMaster,
            AccountMaster=AccountMaster,
            DemandDomainMaster=DemandDomainMaster,
            PnLMaster=PnLMaster,
            account_col=account_col,
            channel_col=channel_col,
            demand_domain_col=demand_domain_col,
            location_col=location_col,
            pnl_col=pnl_col,
            region_col=region_col,
            pl_channel_col=pl_channel_col,
            pl_location_col=pl_location_col,
        )

        PromoDates = PromoDates.withColumn(
            promo_start_date_col, F.to_date(promo_start_date_col, "MM/dd/yyyy H:mm")
        )
        PromoDates = PromoDates.withColumn(
            promo_end_date_col, F.to_date(promo_end_date_col, "MM/dd/yyyy H:mm")
        )

        PromoDates = PromoDates.dropDuplicates()

        # Mapping HolidayDates Location to Planning Location
        HolidayDates = HolidayDates.join(
            LocationMaster.select(location_col, pl_location_col).dropDuplicates(),
            on=location_col,
            how="inner",
        )
        HolidayDates = HolidayDates.drop(location_col)

        HolidayDates = HolidayDates.join(
            Time.select(day_col, day_key_col).dropDuplicates(), on=day_col, how="inner"
        )
        HolidayDates = HolidayDates.drop(day_key_col)

        # filter to have only holiday as 1
        HolidayDates = HolidayDates.filter(F.col(holiday_col) == 1)
        HolidayDates = HolidayDates.drop(holiday_col)

        # -------------------------------Aggregate IntroandDiscoDate to week level---------------------

        time_mapping3 = Time.select(day_key_col, week_col).dropDuplicates()
        Assortment = IntroandDiscoDate.select("*")

        Assortment = Assortment.withColumnRenamed(intro_date_col, day_key_col)
        Assortment = Assortment.join(time_mapping3, on=day_key_col, how="left")
        Assortment = Assortment.drop(day_key_col)
        Assortment = Assortment.withColumnRenamed(week_col, intro_date_col)

        Assortment = Assortment.withColumnRenamed(disco_date_col, day_key_col)
        Assortment = Assortment.join(time_mapping3, on=day_key_col, how="left")
        Assortment = Assortment.drop(day_key_col)
        Assortment = Assortment.withColumnRenamed(week_col, disco_date_col)
        Assortment = Assortment.withColumn(intro_date_col, F.to_date(intro_date_col, "dd-MMM-yy"))
        Assortment = Assortment.withColumn(disco_date_col, F.to_date(disco_date_col, "dd-MMM-yy"))

        Assortment = Assortment.dropDuplicates()

        # ------------------------------ CoC Variation -----------------------------------------------

        logger.info("Started computing CoCC Variation flags...")

        # combine ConsensusFcst and ConsensusFcstLC
        statFcst1 = ConsensusFcstLC.join(
            ConsensusFcst, on=dimensions + [partial_week_col], how="inner"
        )

        time_mapping = Time.select(partial_week_col, week_col).dropDuplicates()

        # aggregate to week level and drop partial week col
        statFcst1 = statFcst1.join(time_mapping, on=partial_week_col, how="left")
        statFcst1 = statFcst1.drop(partial_week_col)

        agg_statFcst1 = statFcst1.groupBy(dimensions + [week_col]).agg(
            F.sum(consFcst_col).alias("sum_consFcst"),
            F.sum(consFcstLC_col).alias("sum_consFcstLC"),
        )

        # Adding window length
        filtered_WindowLength1 = WindowLength.filter(
            F.col(data_validation_col) == coc_exception_col
        )

        if filtered_WindowLength1.isEmpty():
            relevant_statFcst1 = agg_statFcst1.withColumn(
                window_length_col, F.lit(WindowLength_param)
            ).withColumn(data_validation_col, F.lit(coc_exception_col))
        else:
            relevant_statFcst1 = agg_statFcst1.join(
                filtered_WindowLength1,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )

            relevant_statFcst1 = relevant_statFcst1.withColumn(
                window_length_col,
                F.coalesce(F.col(window_length_col), F.lit(WindowLength_param)),
            ).withColumn(
                data_validation_col,
                F.coalesce(F.col(data_validation_col), F.lit(coc_exception_col)),
            )

        # Adding threshold
        filtered_Threshold1 = Threshold.filter(F.col(data_validation_col) == coc_exception_col)

        if filtered_Threshold1.isEmpty():
            relevant_statFcst1 = relevant_statFcst1.withColumn(
                threshold_col, F.lit(Threshold_param)
            )
        else:
            relevant_statFcst1 = relevant_statFcst1.join(
                filtered_Threshold1[
                    [
                        version_col,
                        pl_item_col,
                        pl_channel_col,
                        pl_location_col,
                        threshold_col,
                    ]
                ],
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )

            relevant_statFcst1 = relevant_statFcst1.withColumn(
                threshold_col, F.coalesce(F.col(threshold_col), F.lit(Threshold_param))
            )

        # filter based on window length
        relevant_statFcst1 = relevant_statFcst1.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        # add assortment
        relevant_statFcst1 = relevant_statFcst1.join(Assortment, on=dimensions, how="inner")

        # check for pre season product (npi)
        # if npi exists then currWeekDate = max(currWeekDate,IntroDate) and nextWeekDate = min(nextWeekDate,DiscoDate)
        relevant_statFcst1 = relevant_statFcst1.withColumn(
            "currWeekDate", F.greatest(F.col("currWeekDate"), F.col(intro_date_col))
        )

        relevant_statFcst1 = relevant_statFcst1.withColumn(
            "nextWeekDate",
            F.date_add(F.col("currWeekDate"), (F.col(window_length_col) * 7).cast("int")),
        )

        relevant_statFcst1 = relevant_statFcst1.withColumn(
            "nextWeekDate", F.least(F.col("nextWeekDate"), F.col(disco_date_col))
        )

        filtered_df1 = relevant_statFcst1.filter(
            (F.col("currWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("nextWeekDate"))
        )

        filtered_df1 = filtered_df1.drop(
            "currWeekDate", "nextWeekDate", week_col, intro_date_col, disco_date_col
        )

        filtered_df1 = filtered_df1.groupBy(
            dimensions + [data_validation_col, threshold_col, window_length_col]
        ).agg(
            F.round(F.sum("sum_consFcst"), 2).alias(consFcst_col),
            F.round(F.sum("sum_consFcstLC"), 2).alias(consFcstLC_col),
        )

        promo_df = calculate_future_promo_periods(
            intermediate_df=filtered_df1,
            PromoDates=PromoDates,
            IntroandDiscoDate=IntroandDiscoDate,
            dimensions=dimensions,
            window_length_col=window_length_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            intiative_col=intiative_col,
            promo_start_date_col=promo_start_date_col,
            promo_end_date_col=promo_end_date_col,
            future_promo_periods_col=coc_future_promo_periods_col,
        )

        promo_df = promo_df.withColumn(
            coc_future_promo_periods_LC_col, F.col(coc_future_promo_periods_col)
        )

        common_cols = [data_validation_col, threshold_col, consFcst_col, consFcstLC_col]

        holiday_df = calculate_future_holiday_periods(
            intermediate_df=filtered_df1,
            IntroandDiscoDate=IntroandDiscoDate,
            HolidayDates=HolidayDates,
            dimensions=dimensions,
            common_cols=common_cols,
            window_length_col=window_length_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            holiday_col=holiday_col,
            future_holiday_periods_col=coc_future_holiday_periods_col,
            day_col=day_col,
            version_col=version_col,
            pl_location_col=pl_location_col,
        )

        holiday_df = holiday_df.withColumn(
            coc_future_holiday_periods_LC_col, F.col(coc_future_holiday_periods_col)
        )

        filtered_df1 = filtered_df1.join(
            promo_df, on=dimensions + common_cols + [window_length_col], how="left"
        )

        filtered_df1 = filtered_df1.join(
            holiday_df, on=dimensions + common_cols + [window_length_col], how="left"
        )

        filtered_df1 = filtered_df1.withColumn(
            "Abs Deviation",
            F.round(
                F.when(F.col(consFcstLC_col) == 0, 1).otherwise(
                    F.abs(F.col(consFcst_col) - F.col(consFcstLC_col)) / F.col(consFcstLC_col)
                ),
                2,
            ),
        )

        filtered_df1 = filtered_df1.withColumn(
            coc_flag_col,
            F.when(F.col("Abs Deviation") > F.col(threshold_col), 1).otherwise(0),
        )

        filtered_df1 = (
            filtered_df1.withColumnRenamed(consFcst_col, coc_consFcst_col)
            .withColumnRenamed(consFcstLC_col, coc_consFcstLC_col)
            .withColumnRenamed(threshold_col, coc_tolerance_col)
            .withColumnRenamed("Abs Deviation", coc_tolerance_deviation_col)
        )

        CoCVariation = filtered_df1.select(
            dimensions
            + [
                data_validation_col,
                coc_consFcst_col,
                coc_consFcstLC_col,
                coc_tolerance_col,
                coc_tolerance_deviation_col,
                coc_flag_col,
                coc_future_promo_periods_col,
                coc_future_promo_periods_LC_col,
                coc_future_holiday_periods_col,
                coc_future_holiday_periods_LC_col,
            ]
        ).filter(F.col(coc_flag_col) == 1)

        Output = CoCVariation.select("*")

        Output = Output.withColumn(coc_consFcst_col, F.col(coc_consFcst_col).cast(IntegerType()))
        Output = Output.withColumn(
            coc_consFcstLC_col, F.col(coc_consFcstLC_col).cast(IntegerType())
        )
        Output = Output.withColumn(coc_tolerance_col, F.col(coc_tolerance_col).cast(FloatType()))
        Output = Output.withColumn(
            coc_tolerance_deviation_col, F.col(coc_tolerance_deviation_col).cast(FloatType())
        )
        Output = Output.withColumn(
            coc_future_promo_periods_col, F.col(coc_future_promo_periods_col).cast(FloatType())
        )
        Output = Output.withColumn(
            coc_future_promo_periods_LC_col,
            F.col(coc_future_promo_periods_LC_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            coc_future_holiday_periods_col, F.col(coc_future_holiday_periods_col).cast(FloatType())
        )
        Output = Output.withColumn(
            coc_future_holiday_periods_LC_col,
            F.col(coc_future_holiday_periods_LC_col).cast(FloatType()),
        )

        logger.info("Finished computing CoCC Variation flags...")

        # ------------------- Zero Forecast -------------------------------------------
        logger.info("Started computing Zero Forecast flags...")

        # create a copy of ConsensusFcst
        statFcst2 = ConsensusFcst.select("*")

        # aggregate to week level and drop partial week col
        statFcst2 = statFcst2.join(time_mapping, on=partial_week_col, how="left")
        statFcst2 = statFcst2.drop(partial_week_col)

        agg_statFcst2 = statFcst2.groupBy(dimensions + [week_col]).agg(
            F.sum(consFcst_col).alias("sum_consFcst"),
        )

        # Adding window length
        filtered_WindowLength2 = WindowLength.filter(
            F.col(data_validation_col) == zero_fcst_exception_col
        )

        if filtered_WindowLength2.isEmpty():
            relevant_statFcst2 = agg_statFcst2.withColumn(
                window_length_col, F.lit(WindowLength_param)
            ).withColumn(data_validation_col, F.lit(zero_fcst_exception_col))
        else:
            relevant_statFcst2 = agg_statFcst2.join(
                filtered_WindowLength2,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )

            relevant_statFcst2 = relevant_statFcst2.withColumn(
                window_length_col,
                F.coalesce(F.col(window_length_col), F.lit(WindowLength_param)),
            ).withColumn(
                data_validation_col,
                F.coalesce(F.col(data_validation_col), F.lit(zero_fcst_exception_col)),
            )

        # filter based on window length
        relevant_statFcst2 = relevant_statFcst2.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        # add assortment
        relevant_statFcst2 = relevant_statFcst2.join(Assortment, on=dimensions, how="inner")

        # check for pre season product (npi)
        # if npi exists then currWeekDate = max(currWeekDate,IntroDate) and nextWeekDate = min(nextWeekDate,DiscoDate)
        relevant_statFcst2 = relevant_statFcst2.withColumn(
            "currWeekDate", F.greatest(F.col("currWeekDate"), F.col(intro_date_col))
        )

        relevant_statFcst2 = relevant_statFcst2.withColumn(
            "nextWeekDate",
            F.date_add(F.col("currWeekDate"), (F.col(window_length_col) * 7).cast("int")),
        )

        relevant_statFcst2 = relevant_statFcst2.withColumn(
            "nextWeekDate", F.least(F.col("nextWeekDate"), F.col(disco_date_col))
        )

        filtered_df2 = relevant_statFcst2.filter(
            (F.col("currWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("nextWeekDate"))
        )

        filtered_df2 = filtered_df2.drop(
            "currWeekDate", "nextWeekDate", week_col, intro_date_col, disco_date_col
        )

        filtered_df2 = filtered_df2.groupBy(
            dimensions + [data_validation_col, window_length_col]
        ).agg(F.sum("sum_consFcst").alias(consFcst_col))

        filtered_df2 = filtered_df2.withColumn(
            zero_fcst_flag_col, F.when(F.col(consFcst_col) == 0, 1).otherwise(0)
        )

        ZeroFcst = filtered_df2.select(
            dimensions + [data_validation_col, zero_fcst_flag_col]
        ).filter(F.col(zero_fcst_flag_col) == 1)

        Output = Output.join(ZeroFcst, on=dimensions + [data_validation_col], how="outer")

        logger.info("Finished computing Zero Forecast flags...")

        # ------------------ System vs Naive -----------------------------------
        logger.info("Started computing System Vs Naive flags....")

        Actual = Actual.join(
            StoreInventoryActual,
            on=[
                version_col,
                item_col,
                location_col,
                channel_col,
                demand_domain_col,
                pnl_col,
                region_col,
                account_col,
                day_col,
            ],
            how="left",
        )
        Actual = Actual.fillna(0)

        # Calculate Out of Stock Periods
        Actual = Actual.withColumn(
            system_vs_naive_out_of_stock_col,
            F.when(
                (F.col(actual_col) == 0) & (F.col(store_inventory_actual_col) == 0), 1
            ).otherwise(0),
        )

        # Mapping Actual Item to Planning Item
        Actual = convert_to_planning_lvl(
            intermediate_df=Actual,
            ChannelMaster=ChannelMaster,
            LocationMaster=LocationMaster,
            RegionMaster=RegionMaster,
            AccountMaster=AccountMaster,
            DemandDomainMaster=DemandDomainMaster,
            PnLMaster=PnLMaster,
            account_col=account_col,
            channel_col=channel_col,
            demand_domain_col=demand_domain_col,
            location_col=location_col,
            pnl_col=pnl_col,
            region_col=region_col,
            pl_channel_col=pl_channel_col,
            pl_location_col=pl_location_col,
        )

        Actual = Actual.join(
            ItemMaster.select(item_col, pl_item_col).dropDuplicates(), on=item_col, how="inner"
        )
        Actual = Actual.drop(item_col)

        # copy fcst
        statFcst3 = ConsensusFcst.select("*")

        # aggregate to week level and drop partial week col

        statFcst3 = statFcst3.join(time_mapping, on=partial_week_col, how="left")
        statFcst3 = statFcst3.drop(partial_week_col)

        time_mapping2 = Time.select(day_col, week_col).dropDuplicates()

        actual_df = Actual.join(time_mapping2, on=day_col, how="left")
        actual_df = actual_df.drop(day_col)

        agg_statFcst3 = statFcst3.groupBy(dimensions + [week_col]).agg(
            F.sum(consFcst_col).alias("sum_consFcst"),
        )

        agg_actual = actual_df.groupBy(dimensions + [week_col]).agg(
            F.sum(actual_col).alias("sum_actual"),
            F.sum(system_vs_naive_out_of_stock_col).alias("sum_out_of_stock"),
        )

        agg_actual.drop(store_inventory_actual_col)

        # Adding window length
        filtered_WindowLength3 = WindowLength.filter(
            F.col(data_validation_col) == system_vs_naive_exception_col
        )

        if filtered_WindowLength3.isEmpty():
            relevant_statFcst3 = agg_statFcst3.withColumn(
                window_length_col, F.lit(WindowLength_param)
            ).withColumn(data_validation_col, F.lit(system_vs_naive_exception_col))

            relevant_actual = agg_actual.withColumn(
                window_length_col, F.lit(WindowLength_param)
            ).withColumn(data_validation_col, F.lit(system_vs_naive_exception_col))
        else:
            relevant_statFcst3 = agg_statFcst3.join(
                filtered_WindowLength3,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )
            relevant_actual = agg_actual.join(
                filtered_WindowLength3,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )

            relevant_statFcst3 = relevant_statFcst3.withColumn(
                window_length_col,
                F.coalesce(F.col(window_length_col), F.lit(WindowLength_param)),
            ).withColumn(
                data_validation_col,
                F.coalesce(F.col(data_validation_col), F.lit(system_vs_naive_exception_col)),
            )

            relevant_actual = relevant_actual.withColumn(
                window_length_col,
                F.coalesce(F.col(window_length_col), F.lit(WindowLength_param)),
            ).withColumn(
                data_validation_col,
                F.coalesce(F.col(data_validation_col), F.lit(system_vs_naive_exception_col)),
            )

        # add new dates
        relevant_statFcst3 = relevant_statFcst3.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        relevant_statFcst3 = relevant_statFcst3.withColumn(
            "nextWeekDate",
            F.date_add(F.col("currWeekDate"), (F.col(window_length_col) * 7).cast("int")),
        )

        relevant_actual = relevant_actual.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        relevant_actual = relevant_actual.withColumn(
            "TYlastWeekDate",
            F.date_sub(F.col("currWeekDate"), (F.col(window_length_col) * 7).cast("int")),
        )

        time_df = Time.withColumn("WeekDate", F.to_date(F.col(week_col), "dd-MMM-yy"))

        time_df = time_df.withColumn("Year", F.year(F.col("WeekDate"))).withColumn(
            "WeekColumn", F.weekofyear(F.col("WeekDate"))
        )

        currYear = F.year(F.to_date(F.lit(currWeek), "dd-MMM-yy"))
        currWeekNum = F.weekofyear(F.to_date(F.lit(currWeek), "dd-MMM-yy"))

        LYcurrWeekDate = (
            time_df.filter(
                (time_df["Year"] == (currYear - 1)) & (time_df["WeekColumn"] == currWeekNum)
            )
            .select(week_col)
            .first()[0]
        )

        relevant_actual = relevant_actual.withColumn(
            "LYcurrWeekDate", F.to_date(F.lit(LYcurrWeekDate), "dd-MMM-yy")
        )

        relevant_actual = relevant_actual.withColumn(
            "LYnextWeekDate",
            F.date_add(
                F.col("LYcurrWeekDate"),
                (F.col(window_length_col) * 7).cast("int"),
            ),
        )

        # filter based on window length
        filtered_df3 = relevant_statFcst3.filter(
            (F.col("currWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("nextWeekDate"))
        )

        filtered_TY_actual = relevant_actual.filter(
            (F.col("TYlastWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("currWeekDate"))
        )

        filtered_LY_actual = relevant_actual.filter(
            (F.col("LYcurrWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("LYnextWeekDate"))
        )

        # drop time columns
        filtered_df3 = filtered_df3.drop("currWeekDate", "nextWeekDate", week_col)
        filtered_TY_actual = filtered_TY_actual.drop(
            "currWeekDate",
            "TYlastWeekDate",
            "LYcurrWeekDate",
            "LYnextWeekDate",
            week_col,
        )
        filtered_LY_actual = filtered_LY_actual.drop(
            "currWeekDate",
            "TYlastWeekDate",
            "LYcurrWeekDate",
            "LYnextWeekDate",
            week_col,
        )

        # aggregate
        filtered_df3 = filtered_df3.groupBy(
            dimensions + [data_validation_col, window_length_col]
        ).agg(F.sum("sum_consFcst").alias(consFcst_col))

        filtered_TY_actual = filtered_TY_actual.groupBy(
            dimensions + [data_validation_col, window_length_col]
        ).agg(
            F.sum("sum_actual").alias("actual_TY"),
            F.sum("sum_out_of_stock").alias(system_vs_naive_out_of_stock_col),
        )

        filtered_LY_actual = filtered_LY_actual.groupBy(
            dimensions + [data_validation_col, window_length_col]
        ).agg(
            F.sum("sum_actual").alias("actual_LY"),
            F.sum("sum_out_of_stock").alias(system_vs_naive_out_of_stock_col),
        )

        filtered_LY_actual = filtered_LY_actual.filter(F.col(pl_demand_domain_col) == "Core")

        combined_df = filtered_TY_actual.join(
            filtered_LY_actual,
            on=dimensions
            + [data_validation_col, system_vs_naive_out_of_stock_col, window_length_col],
            how="outer",
        ).fillna(0)

        combined_df = combined_df.groupBy(
            dimensions + [data_validation_col, window_length_col]
        ).agg(
            F.sum(system_vs_naive_out_of_stock_col).alias("out_of_stock"),
            F.sum("actual_TY").alias("Actual_TY"),
            F.sum("actual_LY").alias("Actual_LY"),
        )

        combined_df = (
            combined_df.withColumnRenamed("out_of_stock", system_vs_naive_out_of_stock_col)
            .withColumnRenamed("Actual_TY", "actual_TY")
            .withColumnRenamed("Actual_LY", "actual_LY")
        )

        # Filter out rows where columns from filtered_df3 are null (indicating no match)
        rows_with_nan_fcst = combined_df.join(
            filtered_df3, on=dimensions + [data_validation_col], how="left"
        )
        rows_with_nan_fcst = rows_with_nan_fcst.fillna({consFcst_col: -1})
        rows_with_nan_fcst = rows_with_nan_fcst.filter(F.col(consFcst_col) == -1)

        combined_df = filtered_df3.join(
            combined_df, on=dimensions + [data_validation_col, window_length_col], how="outer"
        )
        combined_df = combined_df.fillna(0)

        # Adding threshold
        filtered_Threshold2 = Threshold.filter(
            F.col(data_validation_col) == system_vs_naive_exception_col
        )

        if filtered_Threshold2.isEmpty():
            combined_df1 = combined_df.withColumn(threshold_col, F.lit(Threshold_param))
        else:
            combined_df1 = combined_df.join(
                filtered_Threshold2[
                    [
                        version_col,
                        pl_item_col,
                        pl_channel_col,
                        pl_location_col,
                        threshold_col,
                    ]
                ],
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col],
                how="left",
            )

            combined_df1 = combined_df1.withColumn(
                threshold_col, F.coalesce(F.col(threshold_col), F.lit(Threshold_param))
            )

        promo_df = calculate_future_promo_periods(
            intermediate_df=combined_df1,
            PromoDates=PromoDates,
            IntroandDiscoDate=IntroandDiscoDate,
            dimensions=dimensions,
            window_length_col=window_length_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            intiative_col=intiative_col,
            promo_start_date_col=promo_start_date_col,
            promo_end_date_col=promo_end_date_col,
            future_promo_periods_col=system_vs_naive_future_promo_periods_col,
        )

        common_cols = [
            data_validation_col,
            threshold_col,
            consFcst_col,
            "actual_TY",
            "actual_LY",
            system_vs_naive_out_of_stock_col,
        ]

        holiday_df = calculate_future_holiday_periods(
            intermediate_df=combined_df1,
            IntroandDiscoDate=IntroandDiscoDate,
            HolidayDates=HolidayDates,
            dimensions=dimensions,
            common_cols=common_cols,
            window_length_col=window_length_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            holiday_col=holiday_col,
            future_holiday_periods_col=system_vs_naive_future_holiday_periods_col,
            day_col=day_col,
            version_col=version_col,
            pl_location_col=pl_location_col,
        )

        past_promo_df = calculate_past_promo_periods(
            intermediate_df=combined_df1,
            PromoDates=PromoDates,
            IntroandDiscoDate=IntroandDiscoDate,
            LYcurrWeekDate=LYcurrWeekDate,
            dimensions=dimensions,
            window_length_col=window_length_col,
            pl_demand_domain_col=pl_demand_domain_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            intiative_col=intiative_col,
            promo_start_date_col=promo_start_date_col,
            promo_end_date_col=promo_end_date_col,
            past_promo_periods_col=system_vs_naive_past_promo_periods_col,
        )

        past_holiday_df = calculate_past_holiday_periods(
            intermediate_df=combined_df1,
            IntroandDiscoDate=IntroandDiscoDate,
            HolidayDates=HolidayDates,
            dimensions=dimensions,
            common_cols=common_cols,
            LYcurrWeekDate=LYcurrWeekDate,
            pl_demand_domain_col=pl_demand_domain_col,
            window_length_col=window_length_col,
            currDayKey=currDayKey,
            intro_date_col=intro_date_col,
            disco_date_col=disco_date_col,
            holiday_col=holiday_col,
            past_holiday_periods_col=system_vs_naive_past_holiday_periods_col,
            day_col=day_col,
            version_col=version_col,
            pl_location_col=pl_location_col,
        )

        combined_df1 = combined_df1.join(
            promo_df, on=dimensions + common_cols + [window_length_col], how="left"
        )

        combined_df1 = combined_df1.join(
            holiday_df, on=dimensions + common_cols + [window_length_col], how="left"
        )
        combined_df1 = combined_df1.join(
            past_promo_df, on=dimensions + common_cols + [window_length_col], how="left"
        )
        combined_df1 = combined_df1.join(
            past_holiday_df, on=dimensions + common_cols + [window_length_col], how="left"
        )

        combined_df1 = combined_df1.withColumn(
            "Average",
            F.when(F.col("actual_LY") == 0, F.col("actual_TY")).otherwise(
                (F.col("actual_TY") + F.col("actual_LY")) / 2
            ),
        )

        combined_df1 = combined_df1.withColumn(
            "Percentage Variation",
            F.when(
                F.col(pl_demand_domain_col) == "Core",
                F.when(F.col("Average") == 0, 1).otherwise(
                    F.round(F.abs(F.col("Average") - F.col(consFcst_col)) / F.col("Average"), 2)
                ),
            ).otherwise(
                F.when(F.col(consFcst_col) == 0, 1).otherwise(
                    F.round(
                        F.abs(F.col(consFcst_col) - F.col("actual_TY")) / F.col(consFcst_col), 2
                    )
                )
            ),
        )

        combined_df1 = combined_df1.withColumn(
            system_vs_naive_flag_col,
            F.when(
                F.coalesce(F.col("Percentage Variation"), F.lit(0)) > F.col(threshold_col),
                1,
            ).otherwise(0),
        )

        combined_df1 = combined_df1.withColumn(
            system_vs_naive_out_of_stock_col,
            F.when(F.col(system_vs_naive_out_of_stock_col) > 0, 1).otherwise(0),
        )

        combined_df1 = combined_df1.withColumn(
            system_vs_naive_actualAvg_col,
            F.when(F.col(pl_demand_domain_col) == "Core", F.col("Average")).otherwise(
                F.col("actual_TY")
            ),
        )

        combined_df1 = (
            combined_df1.withColumnRenamed(consFcst_col, system_vs_naive_consFcst_col)
            .withColumnRenamed(threshold_col, system_vs_naive_tolerance_col)
            .withColumnRenamed("Percentage Variation", system_vs_naive_tolerance_devaition_col)
        )

        combined_df1 = combined_df1.withColumn(
            system_vs_naive_consFcst_col, F.round(F.col(system_vs_naive_consFcst_col), 2)
        ).withColumn(
            system_vs_naive_actualAvg_col, F.round(F.col(system_vs_naive_actualAvg_col), 2)
        )

        SystemVsNaive = combined_df1.select(
            dimensions
            + [
                data_validation_col,
                system_vs_naive_consFcst_col,
                system_vs_naive_actualAvg_col,
                system_vs_naive_tolerance_col,
                system_vs_naive_tolerance_devaition_col,
                system_vs_naive_flag_col,
                system_vs_naive_out_of_stock_col,
                system_vs_naive_past_promo_periods_col,
                system_vs_naive_past_holiday_periods_col,
                system_vs_naive_future_promo_periods_col,
                system_vs_naive_future_holiday_periods_col,
            ]
        ).filter(F.col(system_vs_naive_flag_col) == 1)

        # drop pre season products (npi)
        SystemVsNaive = SystemVsNaive.join(Assortment, on=dimensions, how="inner")
        SystemVsNaive = SystemVsNaive.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )
        SystemVsNaive = SystemVsNaive.filter(
            F.col("currWeekDate") > F.to_date(F.col(intro_date_col), "dd-MMM-yy")
        )
        SystemVsNaive = SystemVsNaive.drop("currWeekDate", intro_date_col, disco_date_col)

        # marking zero fcst flag for rows with null fcst but actual are present
        rows_with_nan_fcst = rows_with_nan_fcst.select(dimensions)
        rows_with_nan_fcst = rows_with_nan_fcst.withColumn(zero_fcst_flag_col, F.lit(1)).withColumn(
            data_validation_col, F.lit(zero_fcst_exception_col)
        )
        Output = Output.join(
            rows_with_nan_fcst,
            on=dimensions + [data_validation_col, zero_fcst_flag_col],
            how="outer",
        )

        # join SystemVsNaive flags with Output
        Output = Output.join(SystemVsNaive, on=dimensions + [data_validation_col], how="outer")

        Output = Output.withColumn(
            system_vs_naive_consFcst_col, F.col(system_vs_naive_consFcst_col).cast(IntegerType())
        )
        Output = Output.withColumn(
            system_vs_naive_actualAvg_col, F.col(system_vs_naive_actualAvg_col).cast(IntegerType())
        )
        Output = Output.withColumn(
            system_vs_naive_tolerance_col, F.col(system_vs_naive_tolerance_col).cast(FloatType())
        )
        Output = Output.withColumn(
            system_vs_naive_tolerance_devaition_col,
            F.col(system_vs_naive_tolerance_devaition_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            system_vs_naive_out_of_stock_col,
            F.col(system_vs_naive_out_of_stock_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            system_vs_naive_past_promo_periods_col,
            F.col(system_vs_naive_past_promo_periods_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            system_vs_naive_past_holiday_periods_col,
            F.col(system_vs_naive_past_holiday_periods_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            system_vs_naive_future_promo_periods_col,
            F.col(system_vs_naive_future_promo_periods_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            system_vs_naive_future_holiday_periods_col,
            F.col(system_vs_naive_future_holiday_periods_col).cast(FloatType()),
        )

        logger.info("Finished computing System Vs Naive flags...")

        # -------------------------- Size Profiler ----------------------------------------

        logger.info("Started computing Size Profiler flags...")
        # Mapping Item, Location to Planning Item, Planning Location in Final Fcst
        FinalFcst = FinalFcst.join(ItemMaster, on=item_col, how="inner")
        FinalFcst = FinalFcst.drop(item_col)

        FinalFcst = FinalFcst.join(LocationMaster, on=location_col, how="inner")
        FinalFcst = FinalFcst.drop(location_col)

        statFcst4 = ConsensusFcst.select("*")

        statFcst4 = statFcst4.drop(partial_week_col)
        final_fcst = FinalFcst.drop(partial_week_col)

        agg_statFcst4 = statFcst4.groupBy(dimensions).agg(
            F.sum(F.coalesce(F.col(consFcst_col), F.lit(0))).alias("sum_consFcst")
        )

        agg_final_fcst = final_fcst.groupBy(dimensions).agg(
            F.sum(F.coalesce(F.col(final_fcst_col), F.lit(0))).alias("sum_finalFcst")
        )

        combined_df = agg_statFcst4.join(agg_final_fcst, on=dimensions, how="inner")

        combined_df = combined_df.withColumn(
            data_validation_col, F.lit(size_profiler_exception_col)
        )

        combined_df = combined_df.withColumn(threshold_col, F.lit(SizeProfilerThreshold_param))

        combined_df = combined_df.withColumn(window_length_col, F.lit(WindowLength_param))

        combined_df = combined_df.withColumn(
            "Abs Deviation",
            F.round(
                F.when(F.col("sum_consFcst") == 0, 1).otherwise(
                    F.abs(F.col("sum_consFcst") - F.col("sum_finalFcst")) / F.col("sum_consFcst")
                ),
                2,
            ),
        )

        combined_df = combined_df.withColumn(
            size_profiler_flag_col,
            F.when(F.col("Abs Deviation") > F.col(threshold_col), 1).otherwise(0),
        )

        combined_df = (
            combined_df.withColumnRenamed("sum_consFcst", size_profiler_consFcst_col)
            .withColumnRenamed("sum_finalFcst", size_profiler_finalFcst_col)
            .withColumnRenamed(threshold_col, size_profiler_tolerance_col)
            .withColumnRenamed("Abs Deviation", size_profiler_tolerance_devaition_col)
        )

        combined_df = combined_df.withColumn(
            size_profiler_consFcst_col, F.round(F.col(size_profiler_consFcst_col), 2)
        ).withColumn(size_profiler_finalFcst_col, F.round(F.col(size_profiler_finalFcst_col), 2))

        SizeProfiler = combined_df.select(
            dimensions
            + [
                data_validation_col,
                size_profiler_consFcst_col,
                size_profiler_finalFcst_col,
                size_profiler_tolerance_col,
                size_profiler_tolerance_devaition_col,
                size_profiler_flag_col,
            ]
        ).filter(F.col(size_profiler_flag_col) == 1)

        Output = Output.join(SizeProfiler, on=dimensions + [data_validation_col], how="outer")

        Output = Output.withColumn(
            size_profiler_consFcst_col, F.col(size_profiler_consFcst_col).cast(IntegerType())
        )
        Output = Output.withColumn(
            size_profiler_finalFcst_col, F.col(size_profiler_finalFcst_col).cast(IntegerType())
        )
        Output = Output.withColumn(
            size_profiler_tolerance_col, F.col(size_profiler_tolerance_col).cast(FloatType())
        )
        Output = Output.withColumn(
            size_profiler_tolerance_devaition_col,
            F.col(size_profiler_tolerance_devaition_col).cast(FloatType()),
        )

        logger.info("Finished computing Size Profiler flags...")

        # ----------------- Filtering Based on Min History Periods and EOL Periods
        logger.info("Started filtering based on min history and min forecast periods...")

        # Adding Intro and Disco Date
        Output = Output.join(Assortment, on=dimensions, how="inner")

        Output = Output.join(
            MinHistoryPeriods,
            on=[version_col, pl_item_col, pl_channel_col, pl_location_col] + [data_validation_col],
            how="left",
        )
        Output = Output.join(
            EOLPeriods,
            on=[version_col, pl_item_col, pl_channel_col, pl_location_col] + [data_validation_col],
            how="left",
        )

        Output = Output.withColumn(
            min_hist_periods_col,
            F.coalesce(F.col(min_hist_periods_col), F.lit(MinHistory_param)),
        )

        Output = Output.withColumn(
            eol_periods_col, F.coalesce(F.col(eol_periods_col), F.lit(EOL_param))
        )

        Output = Output.withColumn("currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy"))

        # Adding Min History Periods and EOL (End Of Life) Period
        Output = Output.withColumn(
            "eol_date",
            F.date_add(
                F.to_date(F.col("currWeekDate"), "dd-MMM-yy"),
                (F.col(eol_periods_col) * 7).cast("int"),
            ),
        )

        Output = Output.withColumn(
            "min_history_date",
            F.date_sub(
                F.to_date(F.col("currWeekDate"), "dd-MMM-yy"),
                (F.col(min_hist_periods_col) * 7).cast("int"),
            ),
        )

        # Filter based on Min History Periods and EOL Periods
        Output = Output.filter(
            F.when(
                F.col(intro_date_col) < F.col("currWeekDate"),
                (F.to_date(F.col(intro_date_col), "dd-MMM-yy") <= F.col("min_history_date"))
                & (F.col("eol_date") <= F.to_date(F.col(disco_date_col), "dd-MMM-yy")),
            ).otherwise((F.col("eol_date") <= F.to_date(F.col(disco_date_col), "dd-MMM-yy")))
        )

        Output = Output.drop(
            min_hist_periods_col,
            intro_date_col,
            disco_date_col,
            eol_periods_col,
            "min_history_date",
            "eol_date",
            "currWeekDate",
        )

        logger.info("Finished filtering...")

        # ------------------------ Total Weights Calculation -----------------

        logger.info("Started computing Total Weights...")
        # adding exception types for all rows
        relevant_N_statFcst = agg_statFcst3.withColumn(
            data_validation_col,
            F.explode(
                F.array(
                    F.lit(coc_exception_col),
                    F.lit(zero_fcst_exception_col),
                    F.lit(system_vs_naive_exception_col),
                    F.lit(size_profiler_exception_col),
                )
            ),
        )

        relevant_N_actual = agg_actual.withColumn(
            data_validation_col,
            F.explode(
                F.array(
                    F.lit(coc_exception_col),
                    F.lit(zero_fcst_exception_col),
                    F.lit(system_vs_naive_exception_col),
                    F.lit(size_profiler_exception_col),
                )
            ),
        )

        # Adding N Periods
        if NPeriods.isEmpty():
            relevant_N_statFcst = relevant_N_statFcst.withColumn(
                n_periods_col, F.lit(NPeriods_param)
            )

            relevant_N_actual = relevant_N_actual.withColumn(n_periods_col, F.lit(NPeriods_param))

        else:
            relevant_N_statFcst = relevant_N_statFcst.join(
                NPeriods,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col, data_validation_col],
                how="left",
            )
            relevant_N_actual = relevant_N_actual.join(
                NPeriods,
                on=[version_col, pl_item_col, pl_channel_col, pl_location_col, data_validation_col],
                how="left",
            )

            relevant_N_statFcst = relevant_N_statFcst.withColumn(
                n_periods_col, F.coalesce(F.col(n_periods_col), F.lit(NPeriods_param))
            )

            relevant_N_actual = relevant_N_actual.withColumn(
                n_periods_col, F.coalesce(F.col(n_periods_col), F.lit(NPeriods_param))
            )

        # add new dates
        relevant_N_statFcst = relevant_N_statFcst.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        # add assortment
        relevant_N_statFcst = relevant_N_statFcst.join(Assortment, on=dimensions, how="inner")

        # check for pre season product (npi)
        # if npi exists then currWeekDate = max(currWeekDate,IntroDate) and nextWeekDate = min(nextWeekDate,DiscoDate)
        relevant_N_statFcst = relevant_N_statFcst.withColumn(
            "currWeekDate", F.greatest(F.col("currWeekDate"), F.col(intro_date_col))
        )

        relevant_N_statFcst = relevant_N_statFcst.withColumn(
            "nextWeekDate",
            F.date_add(F.col("currWeekDate"), (F.col(n_periods_col) * 7).cast("int")),
        )

        relevant_N_statFcst = relevant_N_statFcst.withColumn(
            "nextWeekDate", F.least(F.col("nextWeekDate"), F.col(disco_date_col))
        )

        relevant_N_actual = relevant_N_actual.withColumn(
            "currWeekDate", F.to_date(F.lit(currWeek), "dd-MMM-yy")
        )

        relevant_N_actual = relevant_N_actual.withColumn(
            "TYlastWeekDate",
            F.date_sub(F.col("currWeekDate"), (F.col(n_periods_col) * 7).cast("int")),
        )

        # filter based on window length
        relevant_N_statFcst = relevant_N_statFcst.filter(
            (F.col("currWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("nextWeekDate"))
        )

        relevant_N_actual = relevant_N_actual.filter(
            (F.col("TYlastWeekDate") <= F.to_date(F.col(week_col), "dd-MMM-yy"))
            & (F.to_date(F.col(week_col), "dd-MMM-yy") < F.col("currWeekDate"))
        )

        # drop extra columns
        relevant_N_statFcst = relevant_N_statFcst.drop(
            "currWeekDate", "nextWeekDate", week_col, intro_date_col, disco_date_col
        )
        relevant_N_actual = relevant_N_actual.drop("currWeekDate", "TYlastWeekDate", week_col)

        # aggregate
        relevant_N_statFcst = (
            relevant_N_statFcst.groupBy(dimensions + [data_validation_col])
            .agg(F.sum("sum_consFcst").alias(consFcst_Nperiods_col))
            .fillna(0)
        )

        relevant_N_actual = (
            relevant_N_actual.groupBy(dimensions + [data_validation_col])
            .agg(F.sum("sum_actual").alias(actual_Nperiods_col))
            .fillna(0)
        )

        Output = Output.join(
            relevant_N_statFcst, on=dimensions + [data_validation_col], how="left"
        ).join(relevant_N_actual, on=dimensions + [data_validation_col], how="left")

        Output = Output.withColumn(
            consFcst_Nperiods_col, F.coalesce(F.col(consFcst_Nperiods_col), F.lit(0))
        )

        Output = Output.withColumn(
            actual_Nperiods_col, F.coalesce(F.col(actual_Nperiods_col), F.lit(0))
        )

        Output = Output.join(Weight, on=[version_col, data_validation_col], how="left")

        Output = Output.withColumn(weight_col, F.coalesce(F.col(weight_col), F.lit(0)))

        # Calculate Total Weight
        Output = Output.withColumn(
            total_weight_col,
            F.round(
                F.when(
                    F.col(consFcst_Nperiods_col) == 0,
                    F.col(actual_Nperiods_col) * F.col(weight_col),
                ).otherwise(F.col(consFcst_Nperiods_col) * F.col(weight_col)),
                2,
            ),
        )

        Output = Output.drop(weight_col)

        logger.info("Finished computing Total Weights...")

        Output = Output.withColumn(coc_flag_col, F.col(coc_flag_col).cast(FloatType()))
        Output = Output.withColumn(zero_fcst_flag_col, F.col(zero_fcst_flag_col).cast(FloatType()))
        Output = Output.withColumn(
            system_vs_naive_flag_col,
            F.col(system_vs_naive_flag_col).cast(FloatType()),
        )
        Output = Output.withColumn(
            size_profiler_flag_col, F.col(size_profiler_flag_col).cast(FloatType())
        )
        Output = Output.withColumn(
            actual_Nperiods_col, F.col(actual_Nperiods_col).cast(FloatType())
        )
        Output = Output.withColumn(
            consFcst_Nperiods_col, F.col(consFcst_Nperiods_col).cast(FloatType())
        )

        # Convert output DataFrame to O9 column format
        Output = col_namer.convert_to_o9_cols(Output)

        logger.info(
            "Successfully finished Plugin {} for slice: {} ...".format(plugin_name, df_keys)
        )
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output

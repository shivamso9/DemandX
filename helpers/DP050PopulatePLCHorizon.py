import logging

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {"PLC Horizon": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    IntroDate,
    DiscoDate,
    MonthDim,
    PlanningMonthDim,
    FcstStorageTimeBucket,
    ConsensusFcstBuckets,
    df_keys,
):
    plugin_name = "DP050PopulatePLCHorizon"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # configurables
    version_col = "Version.[Version Name]"
    region_col = "Region.[Planning Region]"
    account_col = "Account.[Planning Account]"
    channel_col = "Channel.[Planning Channel]"
    pnl_col = "PnL.[Planning PnL]"
    demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    location_col = "Location.[Planning Location]"
    item_col = "Item.[Planning Item]"
    intro_col = "Intro Date"
    disco_col = "Disco Date"
    partial_week_col = "Time.[Partial Week]"
    partial_weekkey_col = "Time.[PartialWeekKey]"
    key = "key"

    # output measure
    plc_col = "PLC Horizon"
    cols_required_in_output = [
        version_col,
        region_col,
        account_col,
        channel_col,
        pnl_col,
        demand_domain_col,
        location_col,
        item_col,
        partial_week_col,
        plc_col,
    ]

    try:
        TimeDimension = MonthDim
        if FcstStorageTimeBucket.loc[0][1] == "Planning Month":
            TimeDimension = PlanningMonthDim
        relevant_time = "Time.[" + FcstStorageTimeBucket.loc[0][1] + "]"
        relevant_timekey = "Time.[" + FcstStorageTimeBucket.loc[0][1] + "Key]"
        relevant_timekey = relevant_timekey.replace(" ", "")
        relevant_time_cols = [
            relevant_time,
            relevant_timekey,
            partial_week_col,
            partial_weekkey_col,
        ]
        relevant_time_cols = list(set(relevant_time_cols))
        TimeDimension = TimeDimension[relevant_time_cols]
        TimeDimension.drop_duplicates(inplace=True, ignore_index=True)
        Combined_date = pd.merge(IntroDate, DiscoDate, how="outer")
        start_date = ConsensusFcstBuckets.loc[0][0]
        lenbuckets = len(ConsensusFcstBuckets)
        end_date = ConsensusFcstBuckets.loc[lenbuckets - 1][0]
        Combined_date[intro_col].fillna(start_date, inplace=True)
        Combined_date[disco_col].fillna(end_date, inplace=True)
        Combined_date[intro_col] = Combined_date[intro_col].astype("datetime64")
        Combined_date[disco_col] = Combined_date[disco_col].astype("datetime64")
        intro_date_DF = pd.merge_asof(
            Combined_date.sort_values(intro_col),
            TimeDimension,
            left_on=intro_col,
            right_on=relevant_timekey,
            direction="backward",
        )

        Combined_date[key] = 1
        TimeDimension[key] = 1
        AllCombinationDF = pd.merge(Combined_date, TimeDimension, on=key)
        AllCombinationDF.drop(key, axis=1, inplace=True)
        AllCombinationDF[relevant_timekey] = AllCombinationDF[relevant_timekey].astype("datetime64")
        AllCombinationDF[intro_col] = AllCombinationDF[intro_col].astype("datetime64")
        AllCombinationDF[disco_col] = AllCombinationDF[disco_col].astype("datetime64")
        AllCombinationDF.loc[
            (
                (AllCombinationDF[relevant_timekey] >= AllCombinationDF[intro_col])
                & (AllCombinationDF[relevant_timekey] <= AllCombinationDF[disco_col])
            ),
            plc_col,
        ] = 1
        # TODO : AllCombinationDF is going to explode with large datasets, can we avoid looping ?
        for index, row in AllCombinationDF.iterrows():
            # TODO : Is there a way we can avoid hardcoding these grains ? See if you can have the grains coming from a list and use it
            account = row[account_col]
            channel = row[channel_col]
            region = row[region_col]
            pnl = row[pnl_col]
            demand_domain = row[demand_domain_col]
            location = row[location_col]
            item = row[item_col]
            time_instance = intro_date_DF.loc[
                (
                    (intro_date_DF[account_col] == account)
                    & (intro_date_DF[channel_col] == channel)
                    & (intro_date_DF[region_col] == region)
                    & (intro_date_DF[pnl_col] == pnl)
                    & (intro_date_DF[demand_domain_col] == demand_domain)
                    & (intro_date_DF[location_col] == location)
                    & (intro_date_DF[item_col] == item)
                ),
                relevant_time,
            ].iloc[0]
            # TODO : need to check if looping can be removed and vectorization can be used. Intent is to get output at PW
            AllCombinationDF.loc[
                (
                    (AllCombinationDF[account_col] == account)
                    & (AllCombinationDF[channel_col] == channel)
                    & (AllCombinationDF[region_col] == region)
                    & (AllCombinationDF[pnl_col] == pnl)
                    & (AllCombinationDF[demand_domain_col] == demand_domain)
                    & (AllCombinationDF[location_col] == location)
                    & (AllCombinationDF[item_col] == item)
                    & (AllCombinationDF[relevant_time] == time_instance)
                ),
                plc_col,
            ] = 1
        Output = AllCombinationDF[cols_required_in_output]
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
    return Output

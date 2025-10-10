import itertools
import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    create_cartesian_product,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.common_utils.o9_memory_utils import _get_memory
from o9Reference.stat_utils.get_moving_avg_forecast import get_moving_avg_forecast

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


# Function Calls
# Read Inputs
logger.info("Reading data from o9DataLake ...")
AccountAttribute = O9DataLake.get("AccountAttribute")
Actual = O9DataLake.get("Actual")
AssortmentFlag = O9DataLake.get("AssortmentFlag")
ChannelAttribute = O9DataLake.get("ChannelAttribute")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
DemandDomainAttribute = O9DataLake.get("DemandDomainAttribute")
DPRule = O9DataLake.get("DPRule")
FlexibleInput = O9DataLake.get("FlexibleInput")
ForecastBucket = O9DataLake.get("ForecastBucket")
ItemAttribute = O9DataLake.get("ItemAttribute")
ItemConsensusFcst = O9DataLake.get("ItemConsensusFcst")
LocationSplitParameters = O9DataLake.get("LocationSplitParameters")
PnLAttribute = O9DataLake.get("PnLAttribute")
RegionAttribute = O9DataLake.get("RegionAttribute")
TimeDimension = O9DataLake.get("TimeDimension")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def calculate_sliding_window_forecast(
    df,
    MAPeriod,
    FuturePeriods,
    test_dates,
    Grains,
    HistoryMeasure,
    relevant_time_key,
    time_level_col,
) -> pd.DataFrame:
    result = pd.DataFrame()
    try:
        if len(df) == 0:
            return result

        df.sort_values(relevant_time_key, inplace=True)
        # get moving average forecast
        the_forecast = get_moving_avg_forecast(
            data=df[HistoryMeasure].to_numpy(),
            moving_avg_periods=MAPeriod,
            forecast_horizon=FuturePeriods,
        )

        result = pd.DataFrame({time_level_col: test_dates, HistoryMeasure: the_forecast})
        for the_col in Grains:
            result.insert(0, the_col, df[the_col].iloc[0])
    except Exception as e:
        logger.exception("Exception for {}, {}".format(e, df[Grains].iloc[0].values))
    return result


def getting_actual_at_higher_level(
    df,
    ItemAttribute,
    TimeDimension,
    search_level,
    item_col,
    location_col,
    actual_col,
    req_cols,
    relevant_time_name,
    last_n_periods,
    df_keys: dict,
):
    cols_required = req_cols + [
        item_col,
        location_col,
        relevant_time_name,
        actual_col,
    ]
    relevant_actual = pd.DataFrame(columns=cols_required)
    try:
        the_item = df[item_col].iloc[0]
        the_item_df = ItemAttribute[ItemAttribute[item_col] == the_item]

        cols_required = req_cols + [location_col, item_col, actual_col]
        the_level_aggregates = pd.DataFrame(columns=cols_required)

        for the_level in search_level:
            the_level_value = the_item_df[the_level].unique()[0]

            logger.info("--------- {} : {}".format(the_level, the_level_value))

            # check if actuals are present at the level
            filter_clause = df[the_level] == the_level_value
            req_data = df[filter_clause]

            if len(req_data) > 0:
                logger.info(
                    "--------- Actual available at {} level, shape : {} ..".format(
                        the_level, req_data.shape
                    )
                )
                fields_to_group = req_cols + [location_col, the_level]
                the_level_aggregates = (
                    req_data.groupby(fields_to_group)[[actual_col]].sum().reset_index()
                )

                the_level_aggregates[item_col] = the_item
                logger.info(f"the_level_aggregates head\n{the_level_aggregates.head()}")
                break
            else:
                # continue to search the next level in item hierarchy
                continue

        relevant_time_mapping = TimeDimension[
            TimeDimension[relevant_time_name].isin(last_n_periods)
        ]

        if len(the_level_aggregates) != 0:
            relevant_actual = create_cartesian_product(
                the_level_aggregates,
                relevant_time_mapping[[relevant_time_name]].drop_duplicates(),
            )

        else:
            logger.info(f"No actuals were found at search level : {the_level}")

    except Exception as e:
        logger.exception(f"Exception {e}: for slice {df_keys}")

    return relevant_actual


# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains: str,
    AccountAttribute: pd.DataFrame,
    Actual: pd.DataFrame,
    AssortmentFlag: pd.DataFrame,
    ChannelAttribute: pd.DataFrame,
    CurrentTimePeriod: pd.DataFrame,
    DemandDomainAttribute: pd.DataFrame,
    DPRule: pd.DataFrame,
    FlexibleInput: pd.DataFrame,
    ForecastBucket: pd.DataFrame,
    ItemAttribute: pd.DataFrame,
    ItemConsensusFcst: pd.DataFrame,
    LocationSplitParameters: pd.DataFrame,
    PnLAttribute: pd.DataFrame,
    RegionAttribute: pd.DataFrame,
    TimeDimension: pd.DataFrame,
    HistoryMeasure: str,
    df_keys: dict,
    multiprocessing_num_cores=4,
):

    plugin_name = "DP070FlexibleLevelLocationSplit"
    AccountScope_col = "DP Account Scope"
    AccountLevel_col = "DP Account Level"
    ItemLevel_col = "DP Item Level"
    ItemScope_col = "DP Item Scope"
    DemandDomainScope_col = "DP Demand Domain Scope"
    DemandDomainLevel_col = "DP Demand Domain Level"
    RegionLevel_col = "DP Region Level"
    RegionScope_col = "DP Region Scope"
    ChannelScope_col = "DP Channel Scope"
    ChannelLevel_col = "DP Channel Level"
    PnLLevel_col = "DP PnL Level"
    PnLScope_col = "DP PnL Scope"
    Rule_col = "DM Rule.[Rule]"
    DataObject = "Data Object.[Data Object]"
    actual_col = str(HistoryMeasure)
    RuleSequence_col = "DP Rule Sequence"
    item_col = "Item.[Item]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    start_date = "Start Date"
    end_date = "End Date"
    version_col = "Version.[Version Name]"
    finalfcst_col = "Final Fcst"
    normalized_col = "Location Split Flexible Normalized"
    FlexibleOutput = "Location Split Flexible Output"
    FlexibleMethod = "Location Split Flexible Method"
    SplitFlexible = "Location Split Flexible"
    SplitFinal = "Location Split Final"
    location_col = "Location.[Location]"
    pl_location_col = "Location.[Planning Location]"
    created_at = "DP Rule Created Date"
    created_by = "DP Rule Created By"
    RuleSelected = "Location Split Flexible Rule Selected"
    Assortment_location = "Assortment Location Split"
    assort = "Location Split Flexible Assortment"
    flag = "DP Rule Flag"
    Rule_method = "DP Rule Method"
    item_consensus_fcst = "Item Consensus Fcst"
    actual_agg_col = "Actual Agg"
    location_split_history_period = "Location Split History Period"
    location_split_history_time_bucket = "Location Split History Time Bucket"
    cumulative_split_col = "Cumulative Split"
    indicator_col = "_merge"
    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_col = "Time.[Week]"
    week_key_col = "Time.[WeekKey]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"
    pl_month_col = "Time.[Planning Month]"
    pl_month_key_col = "Time.[PlanningMonthKey]"
    Grains = get_list_of_grains_from_string(Grains)
    cols_required_in_output1 = [version_col] + Grains + [Assortment_location]

    cols_required_in_output2 = (
        [version_col]
        + Grains
        + [partial_week_col]
        + [Rule_col]
        + [DataObject]
        + [FlexibleOutput]
        + [finalfcst_col]
        + [SplitFlexible]
        + [SplitFinal]
    )
    dimension = Grains.copy()
    dimension.remove(location_col)
    cols_required_in_output3 = (
        [version_col]
        + dimension
        + [pl_location_col]
        + [partial_week_col]
        + [RuleSelected]
        + [FlexibleMethod]
    )

    Output = pd.DataFrame(columns=cols_required_in_output1)
    Output2 = pd.DataFrame(columns=cols_required_in_output2)
    Output3 = pd.DataFrame(columns=cols_required_in_output3)
    try:
        if (
            ForecastBucket.empty
            or ItemAttribute.empty
            or ItemConsensusFcst.empty
            or TimeDimension.empty
            or FlexibleInput.empty
        ):
            raise ValueError("One or more of the inputs are empty! Check logs/ inputs for error!")
            return (
                Output,
                Output2,
                Output3,
            )

        if len(Actual) == 0:
            logger.warning("Actuals cannot be empty")
            logger.warning("Returning empty dataframe for this slice ...")
        if len(AssortmentFlag) == 0:
            logger.warning("AssortmentFlag cannot be empty")
            logger.warning("Returning empty dataframe for this slice ...")
        input_version = AssortmentFlag[version_col].iloc[0]

        def expand_dataframe(df, cols):
            expanded_rows = []

            for i, row in df.iterrows():
                # Create a list of lists, where each sublist contains the values for one column
                scope_lists = []
                for col in cols:
                    if isinstance(row[col], list):
                        scopes = row[col]
                    elif pd.notna(row[col]):
                        scopes = [scope.strip() for scope in str(row[col]).split(",")]
                    else:
                        scopes = [""]
                    scope_lists.append(scopes)

                # Generate combinations of the values in all specified columns
                combinations = itertools.product(*scope_lists)

                for combo in combinations:
                    new_row = row.copy()
                    for j, col in enumerate(cols):
                        new_row[col] = combo[j]
                    expanded_rows.append(new_row)

            expanded_df = pd.DataFrame(expanded_rows)
            return expanded_df

        cols = [
            AccountScope_col,
            ItemScope_col,
            ChannelScope_col,
            RegionScope_col,
            PnLScope_col,
            DemandDomainScope_col,
        ]

        # Expand the DataFrame
        logger.info("Expanding Dataframe...")
        DPRule = expand_dataframe(DPRule, cols)
        logger.info(f"Input : {DPRule}")

        start_date_key = ForecastBucket[partial_week_key_col].min()

        end_date_key = ForecastBucket[partial_week_key_col].max()
        ForecastBucket[partial_week_col] = pd.to_datetime(ForecastBucket[partial_week_col])
        end_date_col = ForecastBucket[partial_week_col].max()
        end_date_col = end_date_col + pd.Timedelta(days=6)

        FlexibleInput.rename(columns={partial_week_col: start_date}, inplace=True)

        FlexibleInput[start_date] = pd.to_datetime(FlexibleInput[start_date])

        logger.info(f"Flexible Level Input : {FlexibleInput}")
        dfgroup = FlexibleInput.groupby([Rule_col, location_col])
        lista = []

        for a, b in dfgroup:
            b[end_date] = b[start_date].shift(-1)
            lista.append(b)
        FlexibleInput = concat_to_dataframe(lista)

        FlexibleInput[end_date].fillna(end_date_col, inplace=True)

        FlexibleInpputDF = FlexibleInput[[start_date, version_col, Rule_col, end_date, DataObject]]
        index_cols = [start_date, version_col, Rule_col, end_date, DataObject]

        FlexibleInput.rename(columns={Rule_col: RuleSelected}, inplace=True)
        prefix = ".["
        suffix = "]"
        level_cols = [
            AccountLevel_col,
            ItemLevel_col,
            RegionLevel_col,
            ChannelLevel_col,
            PnLLevel_col,
            DemandDomainLevel_col,
        ]
        for col in level_cols:
            list_col = []
            if len(str(col).split()) > 3:
                DPRule[col] = (
                    str(col).split()[1] + " " + str(col).split()[2] + prefix + DPRule[col] + suffix
                )
            else:
                DPRule[col] = str(col).split()[1] + prefix + DPRule[col] + suffix
            list_col.append(col)

        merged_df = pd.merge(
            FlexibleInpputDF,
            DPRule,
            on=[Rule_col, version_col, DataObject],
            how="inner",
        )
        level_scope_pairs = [
            (AccountLevel_col, AccountScope_col),
            (ItemLevel_col, ItemScope_col),
            (PnLLevel_col, PnLScope_col),
            (ChannelLevel_col, ChannelScope_col),
            (RegionLevel_col, RegionScope_col),
            (DemandDomainLevel_col, DemandDomainScope_col),
        ]

        logger.info(
            f"Dataframe after merging with master data and Flexible level input : {merged_df.head()}"
        )
        merged_df = merged_df.drop_duplicates()
        for col in cols:
            merged_df[str(col).split()[1]] = merged_df[col]

        logger.info("Transposing Master Data...")

        for a, b in level_scope_pairs:
            df_1 = merged_df.groupby(index_cols + [a])[b].first().unstack(a)
            df_1 = df_1.reset_index()
            merged_df = pd.merge(merged_df, df_1, on=index_cols, how="left")
        merged_df.drop(columns=level_cols + cols + list_col, inplace=True)
        merged_df.drop(columns=[created_by], inplace=True)
        merged_df.drop(
            columns=["Item", "Account", "Channel", "Region", "PnL", "Demand"],
            inplace=True,
        )

        item_search = [
            "Item.[Planning Item]",
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
            "Item.[All Item]",
        ]

        account_search = [
            "Account.[Planning Account]",
            "Account.[Account L1]",
            "Account.[Account L2]",
            "Account.[Account L3]",
            "Account.[Account L4]",
            "Account.[All Account]",
        ]

        pnl_search = [
            "PnL.[Planning PnL]",
            "PnL.[PnL L1]",
            "PnL.[PnL L2]",
            "PnL.[PnL L3]",
            "PnL.[PnL L4]",
            "PnL.[All PnL]",
        ]
        Channel_search = [
            "Channel.[Planning Channel]",
            "Channel.[Channel L1]",
            "Channel.[Channel L2]",
            "Channel.[All Channel]",
        ]
        Region_search = [
            "Region.[Planning Region]",
            "Region.[Region L1]",
            "Region.[Region L2]",
            "Region.[Region L3]",
            "Region.[Region L4]",
            "Region.[All Region]",
        ]
        DemandDomain_search = [
            "Demand Domain.[Planning Demand Domain]",
            "Demand Domain.[Demand Domain L1]",
            "Demand Domain.[Demand Domain L2]",
            "Demand Domain.[Demand Domain L3]",
            "Demand Domain.[Demand Domain L4]",
            "Demand Domain.[All Demand Domain]",
        ]

        def merge_function(searchlevel, dataframe, merge_col, Attribute):
            search_level_columns = []
            a = dataframe.columns
            Attribute[merge_col] = Attribute[merge_col].astype(str)
            i = 0
            # Searching for higher level columns and merging at lower level
            for col in searchlevel:
                if col in dataframe.columns:
                    Attribute[col] = Attribute[col].astype(str)
                    if col == merge_col:
                        pass
                    else:
                        dataframe = pd.merge(
                            dataframe,
                            Attribute[[col, merge_col]],
                            on=col,
                            how="left",
                        )
                        search_level_columns.append(col)
                    dataframe = dataframe.rename(columns={merge_col: merge_col + str(i)})
                    i = i + 1
                else:
                    pass

            b = dataframe.columns
            diff_col = b.difference(a)
            column = dataframe[diff_col[0]]
            for col in diff_col:
                column = column.combine_first(dataframe[col])

            dataframe[merge_col] = column
            if merge_col in diff_col:
                diff_col = diff_col.drop(merge_col)
            dataframe = dataframe.drop(columns=diff_col)
            dataframe = dataframe.drop(columns=search_level_columns)
            return dataframe

        logger.info("Merging Master Data with all Attributes...")
        MergedData = merge_function(account_search, merged_df, pl_account_col, AccountAttribute)
        MergedData = merge_function(item_search, MergedData, item_col, ItemAttribute)
        MergedData = merge_function(Channel_search, MergedData, pl_channel_col, ChannelAttribute)
        MergedData = merge_function(Region_search, MergedData, pl_region_col, RegionAttribute)
        MergedData = merge_function(pnl_search, MergedData, pl_pnl_col, PnLAttribute)
        MergedData = merge_function(
            DemandDomain_search,
            MergedData,
            pl_demand_domain_col,
            DemandDomainAttribute,
        )

        MergedData[created_at] = pd.to_datetime(MergedData[created_at])
        MergedData[RuleSequence_col] = MergedData[RuleSequence_col].fillna(0)

        # Group by Item and Account to get all available Rules

        max_rule_sequence = (
            MergedData.groupby(
                [
                    item_col,
                    pl_account_col,
                    pl_channel_col,
                    pl_region_col,
                    pl_pnl_col,
                    pl_demand_domain_col,
                    DataObject,
                ]
            )[RuleSequence_col]
            .max()
            .reset_index()
        )

        filtered_df = MergedData.merge(
            max_rule_sequence,
            on=[
                item_col,
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                RuleSequence_col,
                DataObject,
            ],
            how="inner",
        )

        logger.info(f"filtered_df : {filtered_df.isnull().sum()}")
        logger.info(
            f"filtered_df : {filtered_df[[item_col,pl_account_col,pl_channel_col,pl_region_col,pl_pnl_col,pl_demand_domain_col, RuleSequence_col]].drop_duplicates()}"
        )

        latest_created = (
            filtered_df.groupby(
                [
                    item_col,
                    pl_account_col,
                    pl_channel_col,
                    pl_region_col,
                    pl_pnl_col,
                    pl_demand_domain_col,
                    RuleSequence_col,
                    DataObject,
                ]
            )[created_at]
            .max()
            .reset_index()
        )

        rule_picked = MergedData.merge(
            latest_created,
            on=[
                item_col,
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                created_at,
                RuleSequence_col,
                DataObject,
            ],
            how="inner",
        )

        rule_picked.rename(columns={Rule_col: RuleSelected}, inplace=True)

        MergedData = pd.merge(
            MergedData,
            rule_picked,
            on=[
                item_col,
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                start_date,
                flag,
                created_at,
                RuleSequence_col,
                end_date,
                version_col,
                Rule_method,
                DataObject,
            ],
            how="inner",
        )

        MergedData = pd.merge(
            MergedData,
            FlexibleInput,
            on=[version_col, start_date, end_date, RuleSelected, DataObject],
            how="left",
        )

        MergedData = MergedData[MergedData[flag] == 1]
        logger.info(
            f"MergedData : {MergedData[[Rule_col,RuleSelected,location_col,normalized_col,RuleSequence_col,flag]].drop_duplicates()}"
        )

        MergedData[Rule_col] = MergedData[RuleSelected]
        MergedData[Assortment_location] = 1

        # getting min and max dates from ForecastBucket
        # to restrict output for forecast buckets
        start_date_key = ForecastBucket[partial_week_key_col].min()
        end_date_key = ForecastBucket[partial_week_key_col].max()

        ForecastBucket.sort_values(partial_week_key_col, inplace=True)

        # getting min and max dates from TimeDimension
        # to get default intro and disc dates

        MergedData = MergedData.drop_duplicates()

        relevant_data_raw = create_cartesian_product(
            MergedData,
            ForecastBucket[[partial_week_col, partial_week_key_col]].drop_duplicates(),
        )

        relevant_data_raw = relevant_data_raw[
            (relevant_data_raw[partial_week_key_col] >= start_date_key)
            & (relevant_data_raw[partial_week_key_col] <= end_date_key)
        ]

        relevant_data_raw = relevant_data_raw[
            (relevant_data_raw[partial_week_key_col] >= relevant_data_raw[start_date])
            & (relevant_data_raw[partial_week_key_col] < relevant_data_raw[end_date])
        ]
        logger.info(f"Dataframe after merging with time : {relevant_data_raw.head()}")

        ItemConsensusFcst[partial_week_col] = pd.to_datetime(ItemConsensusFcst[partial_week_col])
        ItemConsensusFcst[item_col] = ItemConsensusFcst[item_col].astype(str)
        output_dataframe = pd.merge(
            relevant_data_raw,
            ItemConsensusFcst,
            on=[
                partial_week_col,
                version_col,
                item_col,
                pl_account_col,
                pl_channel_col,
                pl_demand_domain_col,
                pl_region_col,
                pl_pnl_col,
            ],
            how="inner",
        )
        logger.info(f"Dataframe after merging with Item Consensus Fcst : {output_dataframe.head()}")
        if len(output_dataframe) == 0:
            raise ValueError(
                "Dataframe is empty after merging with Item Consensus Fcst! Check Item Consensus Fcst for error!"
            )

        output_dataframe = output_dataframe.drop_duplicates()
        cols_required_final = (
            [version_col]
            + Grains
            + [Assortment_location]
            + [FlexibleOutput]
            + [partial_week_col]
            + [Rule_col]
            + [DataObject]
            + [pl_location_col]
            + [RuleSelected]
            + [SplitFlexible]
            + [FlexibleMethod]
            + [finalfcst_col]
            + [SplitFinal]
        )
        FixedInput = pd.DataFrame(columns=cols_required_final)
        if (output_dataframe[Rule_method] == "Flexible Level Fixed %").any():
            logger.info("Calculating Final Fcst for Flexible Level Fixed %...")
            FixedInput = output_dataframe[output_dataframe[Rule_method] == "Flexible Level Fixed %"]
            FixedInput[SplitFlexible] = FixedInput[assort] * FixedInput[normalized_col]
            FixedInput[FlexibleMethod] = FixedInput[Rule_method]
            FixedInput[finalfcst_col] = FixedInput[SplitFlexible] * FixedInput[item_consensus_fcst]
            FixedInput[FlexibleOutput] = FixedInput[SplitFlexible]
            FixedInput[SplitFinal] = FixedInput[SplitFlexible]

            FixedInput = FixedInput[cols_required_final]
        logger.info(f"Flexible Level Fixed % Output : {FixedInput.head()}")

        MovingAvg = pd.DataFrame(columns=cols_required_final)
        if (output_dataframe[Rule_method] == "Flexible Level Moving Avg").any():
            logger.info("Calculating Flexible Level Moving Avg...")
            ItemConsensusFcst = output_dataframe[
                output_dataframe[Rule_method] == "Flexible Level Moving Avg"
            ]

            key_cols = [
                pl_month_key_col,
                month_key_col,
                week_key_col,
                partial_week_key_col,
            ]
            logger.info("Converting key cols to datetime format ...")

            # convert to datetime
            TimeDimension[key_cols] = TimeDimension[key_cols].apply(
                pd.to_datetime, infer_datetime_format=True
            )

            if len(Grains) == 0:
                logger.warning(
                    "Grains cannot be empty, check Grain configuration for slice : {} ...".format(
                        df_keys
                    )
                )
                logger.warning("Will return empty dataframe for this slice ...")
                return Output, Output2, Output3

            # getting min and max dates from ForecastBucket
            # to restrict output for forecast buckets
            start_date_key = ForecastBucket[partial_week_key_col].min()
            end_date_key = ForecastBucket[partial_week_key_col].max()

            ForecastBucket.sort_values(partial_week_key_col, inplace=True)
            end_date = ForecastBucket.tail(1)[partial_week_col].iloc[0]

            AssortmentFlag.drop(version_col, axis=1, inplace=True)

            # define item hierarchy
            search_level = [
                "Item.[L1]",
                "Item.[L2]",
                "Item.[L3]",
                "Item.[L4]",
                "Item.[L5]",
                "Item.[L6]",
            ]
            logger.info("search_level : {}".format(search_level))

            # Filter relevant columns from Item Attribute
            req_cols = [
                item_col,
                "Item.[L1]",
                "Item.[L2]",
                "Item.[L3]",
                "Item.[L4]",
                "Item.[L5]",
                "Item.[L6]",
            ]
            ItemAttribute = ItemAttribute[req_cols].drop_duplicates()

            Actual = Actual.merge(
                ItemAttribute,
                on=item_col,
                how="inner",
            )

            # identify combinations with consensus fcst but no actual
            cust_grp_item = [
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                item_col,
            ]
            intersections_with_consensus_fcst = ItemConsensusFcst[cust_grp_item].drop_duplicates()

            intersections_with_actual = Actual[cust_grp_item].drop_duplicates()

            # perform a left join, with indicator column
            merged_df = intersections_with_consensus_fcst.merge(
                intersections_with_actual, how="left", indicator=True
            )

            """merged_df = merged_df.merge(
                LocationSplitMethod[cust_grp_item].drop_duplicates(),
                on=cust_grp_item,
                how="inner",
            )"""

            if len(LocationSplitParameters) != 0:

                MovingAvgPeriods = int(LocationSplitParameters[location_split_history_period][0])
                logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))

                if (
                    LocationSplitParameters[location_split_history_time_bucket][0]
                    == "Planning Month"
                ):
                    relevant_time_name = pl_month_col
                    relevant_time_key = pl_month_key_col

                elif LocationSplitParameters[location_split_history_time_bucket][0] == "Month":
                    relevant_time_name = month_col
                    relevant_time_key = month_key_col

                elif LocationSplitParameters[location_split_history_time_bucket][0] == "Week":
                    relevant_time_name = week_col
                    relevant_time_key = week_key_col

                else:
                    logger.warning(
                        "Incorrect history time bucket is provided, check for slice : {} ...".format(
                            df_keys
                        )
                    )
                    logger.warning("Will return empty dataframe for this slice ...")
                    return Output, Output2, Output3

                time_mapping = (
                    TimeDimension[[relevant_time_name, relevant_time_key]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                time_attribute_dict = {relevant_time_name: relevant_time_key}

                latest_time_name = CurrentTimePeriod[relevant_time_name][0]

                # get last n periods dates
                last_n_periods = get_n_time_periods(
                    latest_time_name,
                    -MovingAvgPeriods,
                    time_mapping,
                    time_attribute_dict,
                    include_latest_value=False,
                )
                logger.info("last_n_periods : {} ...".format(last_n_periods))

                Actual = Actual.merge(TimeDimension, on=partial_week_col, how="inner")
                Actual = Actual[Actual[relevant_time_name].isin(last_n_periods)]
                cols_req = (
                    [version_col]
                    + list(set(Grains).union(set(req_cols)))
                    + [partial_week_col, actual_col]
                )
                Actual = Actual[cols_req].drop_duplicates()
                TimeDimension[partial_week_col] = pd.to_datetime(TimeDimension[partial_week_col])
                # to get future periods for relevant time
                relevant_future_periods = ForecastBucket.merge(
                    TimeDimension[[partial_week_col, relevant_time_name]].drop_duplicates(),
                    on=partial_week_col,
                    how="inner",
                )

                # get test period dates
                future_n_periods = list(relevant_future_periods[relevant_time_name].unique())
                logger.info("future_n_periods : {} ...".format(future_n_periods))
                FuturePeriods = int(len(future_n_periods))

                # cap negatives in HistoryMeasure
                Actual[HistoryMeasure] = np.where(
                    Actual[HistoryMeasure] < 0, 0, Actual[HistoryMeasure]
                )

                intersections_with_consensus_fcst_df = merged_df[
                    merged_df[indicator_col] == "left_only"
                ]
                intersections_with_consensus_fcst_and_actual_df = merged_df[
                    merged_df[indicator_col] == "both"
                ]

                req_cols = [
                    pl_account_col,
                    pl_channel_col,
                    pl_region_col,
                    pl_pnl_col,
                    pl_demand_domain_col,
                ]
                cols_required = Grains + [relevant_time_name, actual_col]

                outputData = pd.DataFrame(columns=cols_required)
                if len(intersections_with_consensus_fcst_df) != 0:
                    intersections_with_consensus_fcst_df = (
                        intersections_with_consensus_fcst_df.merge(
                            Actual.drop(columns=[item_col]),
                            on=req_cols,
                            how="inner",
                        )
                    )

                    logger.info("Calculating actual at higher level ...")
                    all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                        delayed(getting_actual_at_higher_level)(
                            df=df,
                            ItemAttribute=ItemAttribute,
                            TimeDimension=TimeDimension,
                            search_level=search_level,
                            item_col=item_col,
                            location_col=location_col,
                            actual_col=actual_col,
                            req_cols=req_cols,
                            relevant_time_name=relevant_time_name,
                            last_n_periods=last_n_periods,
                            df_keys=df_keys,
                        )
                        for name, df in intersections_with_consensus_fcst_df.groupby(cust_grp_item)
                    )

                    # Concat all results to one dataframe
                    outputData = concat_to_dataframe(all_results)

                if len(intersections_with_consensus_fcst_and_actual_df) != 0:
                    intersections_with_consensus_fcst_and_actual_df = (
                        intersections_with_consensus_fcst_and_actual_df.merge(
                            Actual,
                            on=cust_grp_item,
                            how="inner",
                        )
                    )
                    intersections_with_consensus_fcst_and_actual_df[partial_week_col] = (
                        pd.to_datetime(
                            intersections_with_consensus_fcst_and_actual_df[partial_week_col]
                        )
                    )

                    intersections_with_consensus_fcst_and_actual_df = (
                        intersections_with_consensus_fcst_and_actual_df.merge(
                            TimeDimension[[relevant_time_name, partial_week_col]].drop_duplicates(),
                            on=partial_week_col,
                            how="inner",
                        )
                    )
                    intersections_with_consensus_fcst_and_actual_df = (
                        intersections_with_consensus_fcst_and_actual_df[cols_required]
                    )

                else:
                    intersections_with_consensus_fcst_and_actual_df = pd.DataFrame(
                        columns=cols_required
                    )

                relevant_intersections = pd.concat(
                    [
                        outputData[cols_required],
                        intersections_with_consensus_fcst_and_actual_df,
                    ]
                )

                relevant_intersections = (
                    relevant_intersections.groupby(Grains + [relevant_time_name])[[actual_col]]
                    .sum()
                    .reset_index()
                )

                # Fill missing dates
                relevant_actual_nas_filled = fill_missing_dates(
                    actual=relevant_intersections,
                    forecast_level=Grains,
                    history_measure=HistoryMeasure,
                    relevant_time_key=relevant_time_key,
                    relevant_time_name=relevant_time_name,
                    relevant_time_periods=last_n_periods,
                    time_mapping=time_mapping,
                    fill_nulls_with_zero=True,
                    filter_from_start_date=False,
                )

                logger.info("Calculating moving average forecast for all intersections ...")
                all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                    delayed(calculate_sliding_window_forecast)(
                        group,
                        MovingAvgPeriods,
                        FuturePeriods,
                        future_n_periods,
                        Grains,
                        HistoryMeasure,
                        relevant_time_key,
                        relevant_time_name,
                    )
                    for name, group in relevant_actual_nas_filled.groupby(Grains)
                )

                # Concat all results to one dataframe
                outputData = concat_to_dataframe(all_results)
                # validate if output dataframe contains result for all groups present in input
                validate_output(
                    input_df=relevant_actual_nas_filled,
                    output_df=outputData,
                    forecast_level=Grains,
                )

                logger.info("Aggregating actual at item level...")
                filter_clause = cust_grp_item + [relevant_time_name]
                outputData[actual_agg_col] = outputData.groupby(filter_clause, observed=True)[
                    actual_col
                ].transform("sum")

                logger.info("Calculating split % at item level...")
                outputData[FlexibleOutput] = np.where(
                    outputData[actual_agg_col] != 0,
                    outputData[actual_col] / outputData[actual_agg_col],
                    0,
                )

                # get relevant time mapping for copying split values at partial week
                relevant_time_mapping = (
                    TimeDimension[
                        [
                            relevant_time_name,
                            partial_week_col,
                            partial_week_key_col,
                        ]
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                output_df = outputData.merge(
                    relevant_time_mapping,
                    on=relevant_time_name,
                    how="inner",
                )

                output_df = output_df.merge(
                    ItemAttribute,
                    on=item_col,
                    how="inner",
                )

                # get only those partial weeks which satisfy intro and disc dates and are in forecastbucket

                output_df = output_df[
                    (output_df[partial_week_key_col] >= start_date_key)
                    & (output_df[partial_week_key_col] <= end_date_key)
                ]
                output_df[version_col] = input_version

                # filter intersections for which assortment flag is 1
                output_df = output_df.merge(
                    AssortmentFlag,
                    on=Grains,
                    how="inner",
                )

                # normalizing the moving avg split %
                output_df[cumulative_split_col] = output_df.groupby(
                    req_cols + [partial_week_col, item_col],
                    observed=True,
                )[FlexibleOutput].transform("sum")
                output_df[FlexibleOutput] = np.where(
                    output_df[cumulative_split_col] != 0,
                    output_df[FlexibleOutput] / output_df[cumulative_split_col],
                    np.nan,
                )
                if len(output_df) != 0:
                    MovingAvgSplit = output_df

                else:
                    MovingAvgSplit = MovingAvgSplit

            cols_required_icf = (
                [version_col]
                + Grains
                + [partial_week_col]
                + [Rule_col]
                + [DataObject]
                + [pl_location_col]
                + [Rule_method]
                + [RuleSelected]
                + [item_consensus_fcst]
            )
            ICF = ItemConsensusFcst[cols_required_icf]
            cols_required_moving = (
                [version_col] + Grains + [pl_location_col] + [partial_week_col] + [FlexibleOutput]
            )
            merge_cols = [version_col] + Grains + [pl_location_col] + [partial_week_col]
            MovingAvg = MovingAvgSplit[cols_required_moving]
            logger.info("Calculating final fcst for Flexible Level Moving Avg...")
            MovingAvg = ICF.merge(MovingAvg, on=merge_cols, how="left")
            MovingAvg[Assortment_location] = 1
            MovingAvg[SplitFlexible] = MovingAvg[FlexibleOutput]
            MovingAvg[SplitFinal] = MovingAvg[SplitFlexible]
            MovingAvg[FlexibleMethod] = MovingAvg[Rule_method]
            MovingAvg[finalfcst_col] = MovingAvg[SplitFlexible] * MovingAvg[item_consensus_fcst]
            MovingAvg

        logger.info(f"MovingAvg : {MovingAvg.head()}")
        Outputdf = pd.concat(
            [
                FixedInput,
                MovingAvg,
            ]
        )
        logger.info(f"Output : {Outputdf.head()}")

        def convert_date_format(date_str):
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%d-%b-%y")

        # Apply the function to the partial_week_col
        Outputdf[partial_week_col] = Outputdf[partial_week_col].astype(str)
        Outputdf[partial_week_col] = Outputdf[partial_week_col].apply(convert_date_format)
        logger.info(f"Output : {Outputdf.head()}")

        Output = Outputdf[cols_required_in_output1]
        Output2 = Outputdf[cols_required_in_output2]
        Output3 = Outputdf[cols_required_in_output3]

        pass

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output1)
        Output2 = pd.DataFrame(columns=cols_required_in_output2)
        Output3 = pd.DataFrame(columns=cols_required_in_output3)

    return Output, Output2, Output3

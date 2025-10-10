import logging
import re
from datetime import timedelta

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension

from helpers.utils import get_list_of_grains_from_string

# from o9Reference.common_utils.decorators import map_output_columns_to_dtypes

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def remove_brackets(val):
    """
    to remove unnecessary brackets from grains
    [Item].[Item] -> Item.[Item]
    """
    if pd.notnull(val):
        return re.sub(r"^\[([^\]]+)\]", r"\1", val)
    return val


def extract_dimension(element: str, dimensions: list):
    """
    helper function to extract the dimension name from each element
    """
    for dim in dimensions:
        if dim in element:
            return dim
    return None


def rearrange_lists(reference_list: list, target_list: list):
    """
    Rearrange target list based on the order of elements in the reference list
    """
    target_mapping = {extract_dimension(col, reference_list): col for col in target_list}

    # Rearrange based on reference
    reordered_list = [target_mapping[dim] for dim in reference_list if dim in target_mapping]
    return reordered_list


def remove_already_considered_intersections(
    target_df: pd.DataFrame,
    intersections_considered: pd.DataFrame,
    cols_to_consider: list,
):
    """
    function to remove intersections which are already considered for realignment
    """
    cols_to_consider = [x for x in cols_to_consider if x in target_df.columns]
    target_df = target_df.merge(
        intersections_considered,
        on=cols_to_consider,
        how="left",
        indicator=True,
    )
    target_df = target_df[target_df["_merge"] == "left_only"]
    target_df.drop(columns=["_merge"], inplace=True)
    return target_df


def merge_rows(group: pd.DataFrame):
    """
    function to merge date ranges
    """
    merged = []
    current = group.iloc[0].copy()

    logger.info("merging dates to create unique start date and end date intersection ...")
    for i in range(1, len(group)):
        row = group.iloc[i]

        # If dates overlap or are adjacent, retain separate rows with correct percentage
        if row["Transition Start Date"] > current["Transition End Date"] + pd.Timedelta(days=1):
            merged.append(current)
            current = row.copy()
        else:
            # Split overlapping ranges
            current["Transition End Date"] = row["Transition Start Date"] - pd.Timedelta(days=1)
            merged.append(current)
            current = row.copy()

    merged.append(current)
    return pd.DataFrame(merged)


def extract_relevant_data(
    realignment_dims: list,
    planning_dimensions: list,
    leaf_dimensions: list,
    dimension_type: str,
    dimensions: list,
    columns_mapping: dict,
):
    """
    Extract relevant data based on the dimension type (planning, leaf, or other).
    """
    # Identify dimensions based on the type
    dimension_type_mapping = {
        "planning": planning_dimensions,
        "leaf": leaf_dimensions,
        "other": set(realignment_dims) - set(planning_dimensions).union(set(leaf_dimensions)),
    }
    specific_dims = [x for x in realignment_dims if x in dimension_type_mapping[dimension_type]]

    # Get keys and values
    relevant_keys = [dim for dim in dimensions if dim in ",".join(map(str, specific_dims))]
    relevant_values = [columns_mapping[key] for key in relevant_keys if key in columns_mapping]

    # Extract columns and dataframes
    relevant_dims = [value[0] for value in relevant_values]
    relevant_from_cols = [value[2] for value in relevant_values]
    relevant_to_cols = [value[3] for value in relevant_values]

    relevant_dataframes_by_key = {
        key: columns_mapping[key][1]
        for key in relevant_keys
        if key in columns_mapping and isinstance(columns_mapping[key][1], pd.DataFrame)
    }

    return {
        "keys": relevant_keys,
        "dims": relevant_dims,
        "from_cols": relevant_from_cols,
        "to_cols": relevant_to_cols,
        "dataframes_by_key": relevant_dataframes_by_key,
    }


def get_realigned_forecast(
    intersections_considered: pd.DataFrame,
    realignment_leaf_level_dims: list,
    name: str,
    realignment_status: str,
    relevant_keys: list,
    relevant_column_mapping: dict,
    dimensions: list,
    measures: list,
    FcstRaw: pd.DataFrame,
    max_dates: pd.DataFrame,
    planning_level_cols: list,
    from_cols: list,
    to_cols: list,
    group: pd.DataFrame,
    data_object: str,
    planning_dimensions: list,
    version_col: str,
    partial_week_key_col: str,
    partial_week_col: str,
    start_date: str,
    end_date: str,
    contribution_df: pd.DataFrame,
    actual_contribution_per: str,
    realignment_sum: str,
    realignment_percentage: str,
    conversion_factor: str,
):
    """
    function to realign forecast based on the rules
    """

    leaf_flag_col = "Leaf Flag"
    planning_level_cols = [x for x in planning_level_cols if x in dimensions]
    planning_dimensions = [x for x in planning_dimensions if x in dimensions]
    relevant_keys = [x for x in relevant_keys if any(x in value for value in planning_level_cols)]

    output_measures = [value for key, value in relevant_column_mapping.items() if key in measures]
    cols_required_in_output = dimensions + output_measures

    from_cols = [x for x in from_cols if any(value in x for value in relevant_keys)]
    to_cols = [x for x in to_cols if any(value in x for value in relevant_keys)]

    if not "Non Override" in name:
        group = group[group[realignment_status] == 0]

    # filtering FcstRaw to only contain relevant_keys combinations that are in Realignment's from and to
    logger.info("filtering out intersections against for and to intersections ...")
    FcstRaw.set_index(planning_level_cols, inplace=True)
    group.set_index(from_cols, inplace=True)
    filtered_df1 = FcstRaw[FcstRaw.index.isin(group.index)]
    group.reset_index(inplace=True)
    filtered_df1.reset_index(inplace=True)

    group.set_index(to_cols, inplace=True)
    filtered_df2 = FcstRaw[FcstRaw.index.isin(group.index)]
    filtered_df2.reset_index(inplace=True)
    relevant_fcst_raw_df = pd.concat(
        [filtered_df1, filtered_df2], ignore_index=True
    ).drop_duplicates()
    FilStatFcst_copy = relevant_fcst_raw_df.copy()
    group.reset_index(inplace=True)
    FcstRaw.reset_index(inplace=True)

    relevant_cols = planning_dimensions
    if relevant_fcst_raw_df.empty:
        logger.warning("no intersections present for realignment ...")

        return pd.DataFrame(columns=cols_required_in_output), pd.DataFrame(columns=relevant_cols)

    if not intersections_considered.empty:
        logger.info("removing intersection already considered for realignment ...")
        relevant_fcst_raw_df = remove_already_considered_intersections(
            target_df=relevant_fcst_raw_df,
            intersections_considered=intersections_considered,
            cols_to_consider=relevant_cols,
        )
    if data_object in group.columns:
        logger.info("Data Object: {}".format(group[data_object].iloc[0]))
        group.drop(columns=[data_object], inplace=True)

    relevant_fcst_raw_df.drop_duplicates(inplace=True)
    remaining_vars = set(planning_dimensions) - set(planning_level_cols)
    remaining_dims_list = list(x for x in remaining_vars)

    logger.info("getting max dates to fill NAs, if any")
    group = group.merge(
        max_dates,
        left_on=[version_col] + from_cols,
        right_on=[version_col] + planning_level_cols,
        how="left",
    )

    group[end_date] = group[end_date].fillna(group[partial_week_key_col])
    group[start_date] = group[start_date].fillna(group[partial_week_key_col] - timedelta(days=1))
    group.drop(columns=[partial_week_key_col] + planning_level_cols, inplace=True)

    # merge group with relevant_fcst_raw_df on from columns to get data that needs to be realigned
    logger.info("getting relevant data for realignment ...")
    relevant_fcst_raw_df = relevant_fcst_raw_df.merge(
        group,
        left_on=[version_col] + remaining_dims_list + planning_level_cols,
        right_on=[version_col] + remaining_dims_list + from_cols,
        how="inner",
    )

    # getting partial weeks between start date and end date
    logger.info("filtering out relevant partial weeks ...")
    relevant_fcst_raw_df = relevant_fcst_raw_df[
        (relevant_fcst_raw_df[partial_week_key_col] >= relevant_fcst_raw_df[start_date])
        & (relevant_fcst_raw_df[partial_week_key_col] <= relevant_fcst_raw_df[end_date])
    ]

    # getting intersections present in relevant_fcst_raw_df
    # based on this, later will get the extra intersections which were not considered for realignment
    intersections_realigned = relevant_fcst_raw_df[
        remaining_dims_list + from_cols + [partial_week_col]
    ].drop_duplicates()
    intersections_realigned.rename(columns=dict(zip(from_cols, planning_level_cols)), inplace=True)

    logger.info("nulling values for irrelevant intersections between corresponding time frame ...")
    # getting count for each partial week
    group_by_cols = planning_dimensions + [
        partial_week_col,
        start_date,
        end_date,
    ]
    if len(realignment_leaf_level_dims) > 0:
        group_by_cols = group_by_cols + realignment_leaf_level_dims

    relevant_fcst_raw_df["count"] = relevant_fcst_raw_df.groupby(group_by_cols)[
        "Self Transition Flag"
    ].transform("count")

    # nulling out realignment percentage
    condition = (
        (relevant_fcst_raw_df["Self Transition Flag"] == 1)
        & (relevant_fcst_raw_df[realignment_percentage] == 1)
        & (relevant_fcst_raw_df["count"] > 1)
    )
    relevant_fcst_raw_df[realignment_percentage] = np.where(
        condition,
        0,
        relevant_fcst_raw_df[realignment_percentage],
    )
    intersections_not_present = pd.DataFrame(
        columns=planning_level_cols + realignment_leaf_level_dims
    )
    intersections_present_in_relevant_fcst = pd.DataFrame(
        columns=planning_level_cols + realignment_leaf_level_dims + from_cols + to_cols
    )
    relevant_fcst_raw_df.drop(columns=["DM Rule.[Rule]"], inplace=True)
    relevant_fcst_raw_df.drop_duplicates(inplace=True)
    if len(realignment_leaf_level_dims) > 0:
        intersections_not_present = pd.merge(
            relevant_fcst_raw_df[
                planning_level_cols + realignment_leaf_level_dims
            ].drop_duplicates(),
            contribution_df[planning_level_cols + realignment_leaf_level_dims],
            how="right",
            indicator=True,
        )
        intersections_not_present = intersections_not_present[
            intersections_not_present["_merge"] == "right_only"
        ]
        intersections_not_present[leaf_flag_col] = 1

        intersections_present_in_relevant_fcst = relevant_fcst_raw_df[
            planning_level_cols + realignment_leaf_level_dims + from_cols + to_cols
        ].drop_duplicates()
        intersections_present_in_relevant_fcst["Flag"] = 1

        # relevant_fcst_raw_df.drop(
        #     columns=realignment_leaf_level_dims, inplace=True
        # )

    intersections_already_considered = relevant_fcst_raw_df[
        [x for x in relevant_cols if x in relevant_fcst_raw_df.columns]
    ].drop_duplicates()

    # normalizing realignment percentage values
    logger.info("normalizing realignment percentage values ...")
    relevant_fcst_raw_df[realignment_sum] = relevant_fcst_raw_df.groupby(
        planning_dimensions + realignment_leaf_level_dims + from_cols + [partial_week_col],
        observed=True,
    )[realignment_percentage].transform("sum")
    relevant_fcst_raw_df[realignment_percentage] = (
        relevant_fcst_raw_df[realignment_percentage] / relevant_fcst_raw_df[realignment_sum]
    )

    # getting contribution percentage
    logger.info("getting actual contribution values ...")
    if contribution_df.empty:
        relevant_fcst_raw_df[actual_contribution_per] = 1.0
    else:
        raw_intersections = relevant_fcst_raw_df.drop(columns=realignment_leaf_level_dims)
        contribution_df = contribution_df.merge(
            raw_intersections.drop_duplicates(),
            how="left",
        )
        contribution_df = contribution_df.merge(
            intersections_present_in_relevant_fcst,
            how="outer",
        )

        relevant_fcst_raw_df = relevant_fcst_raw_df.merge(
            contribution_df,
            how="outer",
        )
        relevant_fcst_raw_df = relevant_fcst_raw_df.merge(
            intersections_not_present,
            how="left",
        )
        relevant_fcst_raw_df = relevant_fcst_raw_df[
            (relevant_fcst_raw_df["Flag"] == 1) | (relevant_fcst_raw_df["_merge"] == "right_only")
        ].drop_duplicates()

        relevant_fcst_raw_df[to_cols] = np.where(
            relevant_fcst_raw_df[leaf_flag_col].values[:, None] == 1,
            relevant_fcst_raw_df[from_cols].values,
            relevant_fcst_raw_df[to_cols].values,
        )
        relevant_fcst_raw_df["count"] = relevant_fcst_raw_df.groupby(
            from_cols
            + to_cols
            + planning_dimensions
            + realignment_leaf_level_dims
            + [partial_week_col, start_date, end_date, "_merge"],
            observed=True,
        )[realignment_percentage].transform("count")
        relevant_fcst_raw_df = relevant_fcst_raw_df[
            ~(
                (relevant_fcst_raw_df["_merge"] == "right_only")
                & (relevant_fcst_raw_df[realignment_percentage] == 0)
            )
        ].drop_duplicates()

        relevant_fcst_raw_df.drop(columns=["_merge", leaf_flag_col, "Flag"], inplace=True)

        relevant_fcst_raw_df[actual_contribution_per] = relevant_fcst_raw_df[
            actual_contribution_per
        ].fillna(1)

    for measure in measures:
        relevant_fcst_raw_df[measure] = (
            relevant_fcst_raw_df[measure]
            * relevant_fcst_raw_df[realignment_percentage]
            * relevant_fcst_raw_df[conversion_factor]
            * relevant_fcst_raw_df[actual_contribution_per]
        )

    realigned_fcst_base_scope = relevant_fcst_raw_df[
        relevant_fcst_raw_df[planning_level_cols].values != relevant_fcst_raw_df[to_cols].values
    ]
    realigned_fcst_base_scope[to_cols] = realigned_fcst_base_scope[planning_level_cols].values
    realigned_fcst_base_scope[measures] = np.nan

    relevant_fcst_raw_df = pd.concat([relevant_fcst_raw_df, realigned_fcst_base_scope])

    # replacing planning level values with corresponding to column values
    relevant_fcst_raw_df[planning_level_cols] = relevant_fcst_raw_df[to_cols].values

    relevant_fcst_raw_df.rename(columns=relevant_column_mapping, inplace=True)

    relevant_fcst_raw_df = relevant_fcst_raw_df[cols_required_in_output]

    logger.info("getting intersections not considered for realignment ...")
    extra_intersections = FilStatFcst_copy.merge(
        intersections_realigned,
        how="outer",
        indicator=True,
    )
    extra_intersections = extra_intersections[extra_intersections["_merge"] == "left_only"].drop(
        columns="_merge"
    )
    extra_intersections.rename(columns=relevant_column_mapping, inplace=True)
    extra_intersections = extra_intersections[cols_required_in_output]

    output_df = pd.concat([relevant_fcst_raw_df, extra_intersections])

    output_df = output_df.groupby(
        [version_col] + planning_dimensions + [partial_week_col],
        observed=True,
        as_index=False,
    )[output_measures].sum(min_count=1)

    return output_df, intersections_already_considered


def append_input_output_data(
    dimensions: list,
    columns_mapping: dict,
    Forecast_Raw_output: pd.DataFrame,
    Forecast_Raw: pd.DataFrame,
):
    """
    append input and output dataframe after removing common intersections from input
    """
    intersections = Forecast_Raw_output[dimensions].drop_duplicates()

    Forecast_Raw = Forecast_Raw.merge(
        intersections,
        how="left",
        indicator=True,
    )
    Forecast_Raw = Forecast_Raw[Forecast_Raw["_merge"] == "left_only"]
    Forecast_Raw.drop(columns=["_merge"], inplace=True)
    Forecast_Raw = pd.concat(
        [
            Forecast_Raw,
            Forecast_Raw_output.rename(
                columns={value: key for key, value in columns_mapping.items()}
            ),
        ]
    )
    return Forecast_Raw


def get_remaining_intersections(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    planning_dimensions: list,
    column_mapping: dict,
    cols_required_in_output: list,
):
    """
    function to get intersections not considered for realignment
    """
    planning_dimensions = [x for x in planning_dimensions if x in cols_required_in_output]
    remaining_intersections = target_df.merge(
        source_df[planning_dimensions].drop_duplicates(),
        how="outer",
        indicator=True,
    )
    remaining_intersections = remaining_intersections[
        remaining_intersections["_merge"] == "left_only"
    ]
    if len(column_mapping) != 0:
        remaining_intersections.rename(columns=column_mapping, inplace=True)

    source_df = pd.concat([source_df, remaining_intersections[cols_required_in_output]])
    source_df = source_df.replace(0, np.nan)
    return source_df


def get_forecast_data(processed_dataframes, key, dimensions, measures):
    """
    function to extract dataframes from processed_dataframes or create empty ones if the key is not present.
    """
    if key in processed_dataframes.keys():
        forecast_output = processed_dataframes[key][0]
        intersections_already_considered = processed_dataframes[key][1]
    else:
        forecast_output = pd.DataFrame(columns=dimensions + measures)
        intersections_already_considered = pd.DataFrame(columns=dimensions)

    return forecast_output, intersections_already_considered


@log_inputs_and_outputs
@timed
# @map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    AttributeMapping,
    RealignmentRules,
    SellInForecast_Raw,
    KeyFigures,
    Actuals,
    SellOutForecast_Raw,
    AssortmentFinal,
    IsAssorted,
    AccountMapping,
    ChannelMapping,
    PnLMapping,
    DemandDomainMapping,
    LocationMapping,
    TimeDimension,
    ItemMapping,
    RegionMapping,
    ActualContributionParameters,
    CurrentTimePeriod,
    df_keys,
):
    plugin_name = "DP080ForecastRealignment"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    item_col = "Item.[Item]"
    location_col = "Location.[Location]"
    channel_col = "Channel.[Channel]"
    account_col = "Account.[Account]"
    pnl_col = "PnL.[PnL]"
    demand_domain_col = "Demand Domain.[Demand Domain]"
    region_col = "Region.[Region]"

    pl_item_col = "Item.[Planning Item]"
    pl_location_col = "Location.[Planning Location]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_account_col = "Account.[Planning Account]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    pl_region_col = "Region.[Planning Region]"

    data_object = "Data Object.[Data Object]"
    version_col = "Version.[Version Name]"
    dm_rule_col = "DM Rule.[Rule]"

    from_item = "DP From Item Scope"
    from_location = "DP From Location Scope"
    from_channel = "DP From Channel Scope"
    from_account = "DP From Account Scope"
    from_pnl = "DP From PnL Scope"
    from_demand_domain = "DP From Demand Domain Scope"
    from_region = "DP From Region Scope"

    to_item = "DP To Item Scope"
    to_location = "DP To Location Scope"
    to_channel = "DP To Channel Scope"
    to_account = "DP To Account Scope"
    to_pnl = "DP To PnL Scope"
    to_demand_domain = "DP To Demand Domain Scope"
    to_region = "DP To Region Scope"

    do_account = "Data Object Account Level"
    do_channel = "Data Object Channel Level"
    do_item = "Data Object Item Level"
    do_location = "Data Object Location Level"
    do_region = "Data Object Region Level"
    do_pnl = "Data Object PnL Level"
    do_demand_domain = "Data Object Demand Domain Level"

    day_key_col = "Time.[DayKey]"
    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_key_col = "Time.[WeekKey]"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"
    sequence_col = "Sequence.[Sequence]"

    start_date = "Transition Start Date"
    end_date = "Transition End Date"
    realignment_percentage = "DP Realignment Percentage"
    conversion_factor = "DP Conversion Factor"
    realignments_col = "Realignment Types"
    balance_percentage = "Balance Percentage"
    do_process_order = "Data Object Process Order"
    realignment_sum = "Realignment Sum"
    actual_contribution = "Actual"
    actual_contribution_sum = "ActualContributionSum"
    actual_contribution_per = "ActualContributionPercentage"
    include_in_fcst_realignment = "Include in Forecast Realignment"
    assortment_final = "Assortment Final"
    do_planner_input = "Data Object Planner Input"
    actual_contribution_history_bucket = "Actual Contribution History Bucket"
    actual_contribution_history_period = "Actual Contribution History Period"

    fcst_raw_measure_suffix = " FND BB Raw"

    override_measure_list = []
    measures_to_consider = []
    if ~KeyFigures.empty:
        # getting list of measures to be realigned
        measures_to_consider = KeyFigures[KeyFigures[include_in_fcst_realignment]][
            data_object
        ].to_list()

        # getting list of override measures
        override_measure_list = KeyFigures[KeyFigures[do_planner_input] == 1][data_object].to_list()

    # sell in output
    sell_in_dimensions = [x for x in SellInForecast_Raw.columns if is_dimension(x)]

    output_sell_in_override_measures = [
        x
        for x in SellInForecast_Raw.columns
        if (x not in sell_in_dimensions and x in override_measure_list)
    ]
    output_sell_in_override_measures = [
        x for x in output_sell_in_override_measures if x in measures_to_consider
    ]

    output_sell_in_non_override_measures = [
        x.replace(fcst_raw_measure_suffix, "")
        for x in SellInForecast_Raw.columns
        if x not in sell_in_dimensions + output_sell_in_override_measures
    ]
    output_sell_in_non_override_measures = [
        x for x in output_sell_in_non_override_measures if x in measures_to_consider
    ]

    sell_in_output_cols = (
        sell_in_dimensions + output_sell_in_override_measures + output_sell_in_non_override_measures
    )

    sell_in_columns_mapping = {}
    for key in output_sell_in_non_override_measures:
        sell_in_columns_mapping[key] = [
            value for value in SellInForecast_Raw.columns if key + fcst_raw_measure_suffix == value
        ][0]
    for key in output_sell_in_override_measures:
        sell_in_columns_mapping[key] = [
            value
            for value in SellInForecast_Raw.columns
            if (key in value and fcst_raw_measure_suffix not in value)
        ][0]
    sell_in_columns_mapping = {value: key for key, value in sell_in_columns_mapping.items()}

    # sell out output
    sell_out_dimensions = [x for x in SellOutForecast_Raw.columns if is_dimension(x)]

    output_sell_out_override_measures = [
        x
        for x in SellOutForecast_Raw.columns
        if (x not in sell_out_dimensions and x in override_measure_list)
    ]
    output_sell_out_override_measures = [
        x for x in output_sell_out_override_measures if x in measures_to_consider
    ]
    output_sell_out_non_override_measures = [
        x.replace(fcst_raw_measure_suffix, "")
        for x in SellOutForecast_Raw.columns
        if x not in sell_out_dimensions + output_sell_out_override_measures
    ]
    output_sell_out_non_override_measures = [
        x for x in output_sell_out_non_override_measures if x in measures_to_consider
    ]

    sell_out_output_cols = (
        sell_out_dimensions
        + output_sell_out_override_measures
        + output_sell_out_non_override_measures
    )

    sell_out_columns_mapping = {}
    for key in output_sell_out_non_override_measures:
        sell_out_columns_mapping[key] = [
            value for value in SellOutForecast_Raw.columns if key + fcst_raw_measure_suffix in value
        ][0]
    for key in output_sell_out_override_measures:
        sell_out_columns_mapping[key] = [
            value for value in SellOutForecast_Raw.columns if key in value
        ][0]
    sell_out_columns_mapping = {value: key for key, value in sell_out_columns_mapping.items()}

    # override raw output
    sell_in_override_column_mapping = {
        key: key + fcst_raw_measure_suffix for key in output_sell_in_override_measures
    }
    sell_in_non_override_column_mapping = {
        key + fcst_raw_measure_suffix: key for key in output_sell_in_non_override_measures
    }
    sell_out_override_column_mapping = {
        key: key + fcst_raw_measure_suffix for key in output_sell_out_override_measures
    }
    sell_out_non_override_column_mapping = {
        key + fcst_raw_measure_suffix: key for key in output_sell_out_non_override_measures
    }
    RawSellInOverrideOutput = pd.DataFrame(
        columns=sell_in_dimensions + list(sell_in_override_column_mapping.values())
    )
    RawSellOutOverrideOutput = pd.DataFrame(
        columns=sell_out_dimensions + list(sell_out_override_column_mapping.values())
    )

    # assortment output
    assortment_dimensions = [x for x in AssortmentFinal.columns if is_dimension(x)]
    assortment_output = "Assortment Forecast Realignment"

    assortment_output_cols = assortment_dimensions + [assortment_output]
    Assortment = pd.DataFrame(columns=assortment_output_cols)

    # is assorted output
    is_assorted_dimensions = [x for x in IsAssorted.columns if is_dimension(x)]
    is_assorted_output_cols = IsAssorted.columns

    # flag and normalized output
    flag_dimensions = [version_col, data_object, dm_rule_col, sequence_col]
    realignment_status = "Planner Input Realignment Status"  # flag output
    normalized_ouput = "DP Realignment Percentage Weighted"

    flag_and_normalized_values_output_cols = flag_dimensions + [
        realignment_status,
        normalized_ouput,
    ]
    FlagAndNormalizedValues = pd.DataFrame(columns=flag_and_normalized_values_output_cols)

    # actual contribution output
    actual_contribution_per_output = "Realignment Actual Contribution %"
    actual_contribution_dimensions = [
        x
        for x in Actuals.columns
        if (is_dimension(x) and x not in [partial_week_col, partial_week_key_col])
    ]
    actual_contribution_output_cols = actual_contribution_dimensions + [
        actual_contribution_per_output
    ]
    ActualContributionOutput = pd.DataFrame(columns=actual_contribution_output_cols)

    try:
        ActualContribution = Actuals.copy()
        combined_df = pd.concat([SellOutForecast_Raw, SellInForecast_Raw])

        # override and non override measures considered for realignment
        sell_in_measures = [x for x in sell_in_columns_mapping.keys()]
        sell_out_measures = [x for x in sell_out_columns_mapping.keys()]

        sell_in_override_measures = [
            x for x in sell_in_measures if x in output_sell_in_override_measures
        ]
        sell_in_non_override_measures = list(set(sell_in_measures) - set(sell_in_override_measures))

        sell_out_override_measures = [
            x for x in sell_out_measures if x in output_sell_out_override_measures
        ]
        sell_out_non_override_measures = list(
            set(sell_out_measures) - set(sell_out_override_measures)
        )

        SellInForecast_Raw.drop(
            columns=list(
                set(SellInForecast_Raw.columns)
                - set(sell_in_dimensions).union(set(sell_in_measures))
            ),
            inplace=True,
        )
        SellOutForecast_Raw.drop(
            columns=list(
                set(SellOutForecast_Raw.columns)
                - set(sell_out_dimensions).union(set(sell_out_measures))
            ),
            inplace=True,
        )

        # override raw measures output
        sell_in_forecast_raw_output = SellInForecast_Raw.copy()
        sell_out_forecast_raw_output = SellOutForecast_Raw.copy()

        sell_in_forecast_raw_output.rename(columns=sell_in_override_column_mapping, inplace=True)
        sell_out_forecast_raw_output.rename(columns=sell_out_override_column_mapping, inplace=True)

        sell_in_forecast_raw_output = sell_in_forecast_raw_output[
            sell_in_dimensions + list(sell_in_override_column_mapping.values())
        ]
        sell_out_forecast_raw_output = sell_out_forecast_raw_output[
            sell_out_dimensions + list(sell_out_override_column_mapping.values())
        ]

        if (
            len(combined_df) == 0
            or len(AttributeMapping) == 0
            or len(RealignmentRules) == 0
            or len(ActualContribution) == 0
            or len(KeyFigures) == 0
        ):
            logger.warning("One of the inputs is Null. Exiting : {} ...".format(df_keys))
            SellInForecast_Raw.rename(columns=sell_in_columns_mapping, inplace=True)
            RealignedForecastSellIn = SellInForecast_Raw[sell_in_output_cols]

            SellOutForecast_Raw.rename(columns=sell_out_columns_mapping, inplace=True)
            RealignedForecastSellOut = SellOutForecast_Raw[sell_out_output_cols]

            IsAssortedOutput = IsAssorted.copy()
            IsAssortedOutput = IsAssortedOutput[is_assorted_output_cols]

            logger.warning("No further execution of the plugin.")
            return (
                RealignedForecastSellIn,
                RealignedForecastSellOut,
                Assortment,
                IsAssortedOutput,
                FlagAndNormalizedValues,
                ActualContributionOutput,
                RawSellInOverrideOutput,
                RawSellOutOverrideOutput,
            )

        time_key_cols = [
            day_key_col,
            partial_week_key_col,
            week_key_col,
            month_key_col,
        ]
        for col in time_key_cols:
            if col in CurrentTimePeriod.columns:
                CurrentTimePeriod[col] = pd.to_datetime(
                    CurrentTimePeriod[col], utc=True
                ).dt.tz_localize(None)
            if col in TimeDimension.columns:
                TimeDimension[col] = pd.to_datetime(TimeDimension[col], utc=True).dt.tz_localize(
                    None
                )
            if col in ActualContribution.columns:
                ActualContribution[col] = pd.to_datetime(
                    ActualContribution[col], utc=True
                ).dt.tz_localize(None)

        RealignmentRules[start_date] = pd.to_datetime(
            RealignmentRules[start_date], utc=True
        ).dt.tz_localize(None)

        RealignmentRules[end_date] = pd.to_datetime(
            RealignmentRules[end_date], utc=True
        ).dt.tz_localize(None)

        # getting all dimensions and respective grains
        Dimensions = [
            "Item",
            "Location",
            "Channel",
            "Region",
            "Account",
            "PnL",
            "Demand Domain",
        ]
        planning_dimensions = [
            pl_item_col,
            pl_location_col,
            pl_channel_col,
            pl_region_col,
            pl_account_col,
            pl_pnl_col,
            pl_demand_domain_col,
        ]
        # make sure to have same sequence in from_dimensions and to_dimensions
        from_dimensions = [
            from_item,
            from_location,
            from_channel,
            from_region,
            from_account,
            from_pnl,
            from_demand_domain,
        ]
        to_dimensions = [
            to_item,
            to_location,
            to_channel,
            to_region,
            to_account,
            to_pnl,
            to_demand_domain,
        ]
        leaf_dimensions = [
            item_col,
            location_col,
            channel_col,
            region_col,
            account_col,
            pnl_col,
            demand_domain_col,
        ]

        bucket = "Month"
        history_periods = 12
        if not ActualContributionParameters.empty:
            bucket = ActualContributionParameters[actual_contribution_history_bucket].iloc[0]
            history_periods = int(
                ActualContributionParameters[actual_contribution_history_period].iloc[0]
            )

        if bucket == "Month" and history_periods > 12:
            logger.warning(
                f"History periods {history_periods} is greater than 12 for bucket {bucket}. "
                "Setting history_periods to 12."
            )
            history_periods = 12
        if bucket == "Week" and history_periods > 52:
            logger.warning(
                f"History periods {history_periods} is greater than 52 for bucket {bucket}. "
                "Setting history_periods to 52."
            )
            history_periods = 52

        logger.info(f"bucket : {bucket}, history_periods : {history_periods}")

        relevant_time_name = month_col
        relevant_time_key = month_key_col
        if bucket == "Week":
            relevant_time_name = week_col
            relevant_time_key = week_key_col

        pw_mapping = TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates()
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}
        # Dictionary for easier lookups
        relevant_time_mapping_dict = dict(
            zip(
                list(relevant_time_mapping[relevant_time_name]),
                list(relevant_time_mapping[relevant_time_key]),
            )
        )

        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -history_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )
        last_n_period_date = last_n_periods[0]
        cutoff_date = relevant_time_mapping_dict[last_n_period_date]
        logger.info(f"cutoff_date : {cutoff_date}")

        ActualContribution = ActualContribution[
            ActualContribution[partial_week_key_col] >= cutoff_date
        ]
        ActualContribution = ActualContribution.groupby(
            [version_col] + leaf_dimensions, observed=True, as_index=False
        )[actual_contribution].sum(min_count=1)

        # getting contribution at all levels
        logger.info("Calculating actual contribution ...")
        ActualContribution = ActualContribution.merge(
            ItemMapping[[pl_item_col, item_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            LocationMapping[[pl_location_col, location_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            RegionMapping[[pl_region_col, region_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            ChannelMapping[[pl_channel_col, channel_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            AccountMapping[[pl_account_col, account_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            PnLMapping[[pl_pnl_col, pnl_col]],
            how="inner",
        )
        ActualContribution = ActualContribution.merge(
            DemandDomainMapping[[pl_demand_domain_col, demand_domain_col]],
            how="inner",
        )
        AssortmentFinal = AssortmentFinal.merge(
            ItemMapping[[pl_item_col, item_col]],
            how="inner",
        )
        AssortmentFinal = AssortmentFinal.merge(
            LocationMapping[[pl_location_col, location_col]],
            how="inner",
        )

        logger.info("creating column mapping ...")
        columns_mapping = {
            "Item": (
                pl_item_col,
                ItemMapping,
                from_item,
                to_item,
                leaf_dimensions[0],
            ),
            "Location": (
                pl_location_col,
                LocationMapping,
                from_location,
                to_location,
                leaf_dimensions[1],
            ),
            "Channel": (
                pl_channel_col,
                ChannelMapping,
                from_channel,
                to_channel,
                leaf_dimensions[2],
            ),
            "Region": (
                pl_region_col,
                RegionMapping,
                from_region,
                to_region,
                leaf_dimensions[3],
            ),
            "Account": (
                pl_account_col,
                AccountMapping,
                from_account,
                to_account,
                leaf_dimensions[4],
            ),
            "PnL": (
                pl_pnl_col,
                PnLMapping,
                from_pnl,
                to_pnl,
                leaf_dimensions[5],
            ),
            "Demand Domain": (
                pl_demand_domain_col,
                DemandDomainMapping,
                from_demand_domain,
                to_demand_domain,
                leaf_dimensions[6],
            ),
        }

        # separating override and non-override measures dataframe
        logger.info("Separating override and non override measures ...")
        SellInForecast_Raw_override = SellInForecast_Raw[
            sell_in_dimensions + sell_in_override_measures
        ]
        SellInForecast_Raw_non_override = SellInForecast_Raw[
            sell_in_dimensions + sell_in_non_override_measures
        ]

        SellOutForecast_Raw_override = SellOutForecast_Raw[
            sell_out_dimensions + sell_out_override_measures
        ]
        SellOutForecast_Raw_non_override = SellOutForecast_Raw[
            sell_out_dimensions + sell_out_non_override_measures
        ]

        default_start_date = TimeDimension[partial_week_key_col].min()
        default_end_date = TimeDimension[partial_week_key_col].max()

        RealignmentRules[start_date] = RealignmentRules[start_date].fillna(default_start_date)
        RealignmentRules[end_date] = RealignmentRules[end_date].fillna(default_end_date)
        RealignmentRules[conversion_factor] = RealignmentRules[conversion_factor].fillna(1)
        RealignmentRules = RealignmentRules.fillna(0)
        RealignmentRules[end_date] = RealignmentRules[end_date].replace(0, pd.NaT)

        # converting all the times to midnight value to compare the values accurately
        RealignmentRules[start_date] = RealignmentRules[start_date].dt.normalize()
        RealignmentRules[end_date] = RealignmentRules[end_date].dt.normalize()

        cols = [
            do_item,
            do_location,
            do_region,
            do_channel,
            do_account,
            do_pnl,
            do_demand_domain,
        ]
        for col in cols:
            # remove brackets to get dimension name
            AttributeMapping[col] = AttributeMapping[col].map(remove_brackets)

            # Replace NaN with empty strings
            AttributeMapping[col].fillna("", inplace=True)

        # getting start of partial week for start date
        RealignmentRules.rename(columns={start_date: day_key_col}, inplace=True)
        RealignmentRules = (
            RealignmentRules.merge(
                TimeDimension[[partial_week_key_col, day_key_col]],
            )
            .drop(columns=[day_key_col])
            .rename(columns={partial_week_key_col: start_date})
        )

        # getting all dimensions to realign
        AttributeMapping[realignments_col] = (
            AttributeMapping[cols].astype(str).agg(",".join, axis=1)
        )
        AttributeMapping[realignments_col] = AttributeMapping[realignments_col].str.strip(",")

        # Drop unnecessary columns
        AttributeMapping.drop(cols, axis=1, inplace=True)

        # capping negatives if present to 0
        logger.info("capping negative values to 0 ...")
        ActualContribution[actual_contribution] = np.where(
            ActualContribution[actual_contribution] < 0,
            0,
            ActualContribution[actual_contribution],
        )

        # getting sum of actual contribution at planning level
        ActualContribution[actual_contribution_sum] = ActualContribution.groupby(
            [version_col] + planning_dimensions, observed=True
        )[actual_contribution].transform("sum")

        # getting actual contribution percentage
        ActualContribution[actual_contribution_per] = np.where(
            ActualContribution[actual_contribution_sum] != 0,
            ActualContribution[actual_contribution] / ActualContribution[actual_contribution_sum],
            0,
        )
        # drop sum column
        ActualContribution.drop(columns=[actual_contribution_sum], axis=1, inplace=True)
        ActualContributionOutput = ActualContribution.copy()
        ActualContributionOutput.rename(
            columns={actual_contribution_per: actual_contribution_per_output},
            inplace=True,
        )
        ActualContributionOutput = ActualContributionOutput[actual_contribution_output_cols]

        # remove all those intersections where FROM_grains == TO_grains
        # means no realignment
        RealignmentRules = RealignmentRules[
            ~(
                (RealignmentRules[from_item] == RealignmentRules[to_item])
                & (RealignmentRules[from_location] == RealignmentRules[to_location])
                & (RealignmentRules[from_region] == RealignmentRules[to_region])
                & (RealignmentRules[from_channel] == RealignmentRules[to_channel])
                & (RealignmentRules[from_account] == RealignmentRules[to_account])
                & (RealignmentRules[from_pnl] == RealignmentRules[to_pnl])
                & (RealignmentRules[from_demand_domain] == RealignmentRules[to_demand_domain])
            )
        ]
        if RealignmentRules.empty:
            logger.warning("no intersections for realignment : {} ...".format(df_keys))
            SellInForecast_Raw.rename(columns=sell_in_columns_mapping, inplace=True)
            RealignedForecastSellIn = SellInForecast_Raw[sell_in_output_cols]

            SellOutForecast_Raw.rename(columns=sell_out_columns_mapping, inplace=True)
            RealignedForecastSellOut = SellOutForecast_Raw[sell_out_output_cols]

            IsAssortedOutput = IsAssorted.copy()
            IsAssortedOutput = IsAssortedOutput[is_assorted_output_cols]

            logger.warning("No further execution of the plugin.")
            return (
                RealignedForecastSellIn,
                RealignedForecastSellOut,
                Assortment,
                IsAssortedOutput,
                FlagAndNormalizedValues,
                ActualContributionOutput,
                RawSellInOverrideOutput,
                RawSellOutOverrideOutput,
            )

        FlagAndNormalizedValues = RealignmentRules.copy()
        FlagAndNormalizedValues[realignment_status] = 1
        FlagAndNormalizedValues[realignment_sum] = FlagAndNormalizedValues.groupby(
            [
                version_col,
                data_object,
                start_date,
                sequence_col,
            ]
            + from_dimensions
        )[realignment_percentage].transform("sum")

        FlagAndNormalizedValues[normalized_ouput] = np.where(
            FlagAndNormalizedValues[realignment_sum] > 1,
            FlagAndNormalizedValues[realignment_percentage]
            / FlagAndNormalizedValues[realignment_sum],
            FlagAndNormalizedValues[realignment_percentage],
        )
        FlagAndNormalizedValues = FlagAndNormalizedValues[flag_and_normalized_values_output_cols]

        # Calculating Balance Percentage and Normalizing RealignmentPercentage if total Percentage > 1
        RealignmentRules[realignment_sum] = RealignmentRules.groupby(
            [
                version_col,
                data_object,
                start_date,
                end_date,
            ]
            + from_dimensions
        )[realignment_percentage].transform("sum")
        RealignmentRules[balance_percentage] = np.where(
            RealignmentRules[realignment_sum] < 1,
            1 - RealignmentRules[realignment_sum],
            0,
        )

        # getting self transitions with corresponding balance percentage as realignment percentage
        self_transitions = RealignmentRules[RealignmentRules[balance_percentage] >= 0][
            [version_col, data_object, dm_rule_col, sequence_col]
            + from_dimensions
            + [
                balance_percentage,
                conversion_factor,
                start_date,
                end_date,
                realignment_status,
            ]
        ].copy()
        for i in range(0, len(from_dimensions)):
            self_transitions[to_dimensions[i]] = self_transitions[from_dimensions[i]]

        self_transitions.rename(
            columns={
                start_date: start_date + "_realign",
                end_date: end_date + "_realign",
            },
            inplace=True,
        )
        self_transitions[start_date] = default_start_date

        all_columns = set(self_transitions.columns)
        key_columns = list(
            all_columns
            - {
                balance_percentage,
                start_date,
                start_date + "_realign",
                end_date + "_realign",
            }
        )

        rows = []
        for _, row in self_transitions.iterrows():
            base_data = {col: row[col] for col in key_columns}
            balance_percentage_value = row[balance_percentage]
            start_date_value = row[start_date]
            start_date_realign = row[start_date + "_realign"]
            end_date_realign = row[end_date + "_realign"]

            rows.append(
                {
                    **base_data,
                    start_date: start_date_value,
                    end_date: start_date_realign - timedelta(days=1),
                    realignment_percentage: 1.0,
                }
            )

            rows.append(
                {
                    **base_data,
                    start_date: start_date_realign,
                    end_date: end_date_realign,
                    realignment_percentage: balance_percentage_value,
                }
            )

            rows.append(
                {
                    **base_data,
                    start_date: end_date_realign + timedelta(days=1),
                    end_date: default_end_date,
                    realignment_percentage: 1.0,
                }
            )

        self_transitions = pd.DataFrame(rows)

        self_transitions = self_transitions.sort_values(by=[start_date, end_date]).reset_index(
            drop=True
        )

        # merging date ranges
        group_cols = [
            col
            for col in self_transitions.columns
            if col not in [start_date, end_date, realignment_percentage, dm_rule_col]
        ]

        self_transitions = (
            self_transitions.groupby(group_cols, as_index=False)
            .apply(merge_rows)
            .reset_index(drop=True)
        )
        self_transitions = self_transitions[
            (self_transitions[start_date] < self_transitions[end_date])
        ]

        # flag to check whether self transition or not
        self_transitions["Self Transition Flag"] = 1

        RealignmentRules.drop([realignment_sum, balance_percentage], axis=1, inplace=True)

        RealignmentRules["Self Transition Flag"] = 0
        RealignmentRules = pd.concat([RealignmentRules, self_transitions], ignore_index=True)

        # fill nulls with 1
        AttributeMapping[do_process_order] = AttributeMapping[do_process_order].fillna(1)

        # Find the respective Realignment Rules using merging with Attribute Mapping on Data Object
        RealignmentRules = pd.merge(
            RealignmentRules,
            AttributeMapping,
            on=[version_col, data_object],
            how="inner",
        )

        # sort columns in ascending order except process order
        sort_cols = [dm_rule_col] + from_dimensions + [start_date, do_process_order]
        ascending_values = []
        for i in range(0, len(sort_cols)):
            ascending_values.append(True if sort_cols[i] != do_process_order else False)

        RealignmentRules.sort_values(by=sort_cols, ascending=ascending_values, inplace=True)
        RealignmentRules.reset_index(drop=True, inplace=True)

        sell_in_override_output_list = []
        sell_in_non_override_output_list = []
        sell_out_override_output_list = []
        sell_out_non_override_output_list = []
        assortment_intersections_list = []
        is_assorted_intersections_list = []
        sell_in_override_intersections_already_considered_list = []
        sell_in_non_override_intersections_already_considered_list = []
        sell_out_override_intersections_already_considered_list = []
        sell_out_non_override_intersections_already_considered_list = []
        sell_in_override_raw_output_list = []
        sell_out_override_raw_output_list = []
        for name, group in RealignmentRules.groupby(
            [sequence_col, realignments_col], observed=True, sort=False
        ):
            # getting already realigned intersections
            sell_in_override_intersections_considered = concat_to_dataframe(
                sell_in_override_intersections_already_considered_list
            )
            sell_in_non_override_intersections_considered = concat_to_dataframe(
                sell_in_non_override_intersections_already_considered_list
            )
            sell_out_override_intersections_considered = concat_to_dataframe(
                sell_out_override_intersections_already_considered_list
            )
            sell_out_non_override_intersections_considered = concat_to_dataframe(
                sell_out_non_override_intersections_already_considered_list
            )

            # this is being done inside loop because these dataframes gets updated after each loop execution
            # separating override and non-override measures dataframe
            logger.info("creating override and non-override mappings ...")
            sell_in_sell_out_mapping = {
                "Sell In Override": (
                    sell_in_columns_mapping,
                    sell_in_dimensions,
                    sell_in_override_measures,
                    SellInForecast_Raw_override,
                    sell_in_override_intersections_considered,
                ),
                "Sell In Non Override": (
                    sell_in_columns_mapping,
                    sell_in_dimensions,
                    sell_in_non_override_measures,
                    SellInForecast_Raw_non_override,
                    sell_in_non_override_intersections_considered,
                ),
                "Sell Out Override": (
                    sell_out_columns_mapping,
                    sell_out_dimensions,
                    sell_out_override_measures,
                    SellOutForecast_Raw_override,
                    sell_out_override_intersections_considered,
                ),
                "Sell Out Non Override": (
                    sell_out_columns_mapping,
                    sell_out_dimensions,
                    sell_out_non_override_measures,
                    SellOutForecast_Raw_non_override,
                    sell_out_non_override_intersections_considered,
                ),
            }

            sell_in_sell_out_mapping = {
                key: value for key, value in sell_in_sell_out_mapping.items() if not value[3].empty
            }

            # getting partial week key
            sell_in_sell_out_mapping = {
                key: (
                    value[0],
                    value[1],
                    value[2],
                    value[3].merge(pw_mapping, on=partial_week_col, how="inner"),
                    value[4],
                )
                for key, value in sell_in_sell_out_mapping.items()
            }

            # getting max dates from fcst data
            sell_in_sell_out_mapping = {
                key: (
                    value[0],
                    value[1],
                    value[2],
                    value[3],
                    value[4],
                    value[3]
                    .groupby(
                        list(set(value[1]) - set([partial_week_col])),
                        observed=True,
                    )[partial_week_key_col]
                    .max()
                    .reset_index(),
                )
                for key, value in sell_in_sell_out_mapping.items()
            }

            # getting realignment columns
            logger.info("getting realignment columns and other relevant details ...")
            relevant_keys = [dim for dim in Dimensions if dim in name[1]]
            realignment_dims = get_list_of_grains_from_string(name[1])

            realignment_planning_data = extract_relevant_data(
                realignment_dims,
                planning_dimensions,
                leaf_dimensions,
                "planning",
                Dimensions,
                columns_mapping,
            )
            realignment_leaf_level_data = extract_relevant_data(
                realignment_dims,
                planning_dimensions,
                leaf_dimensions,
                "leaf",
                Dimensions,
                columns_mapping,
            )
            realignment_other_level_data = extract_relevant_data(
                realignment_dims,
                planning_dimensions,
                leaf_dimensions,
                "other",
                Dimensions,
                columns_mapping,
            )

            # planning level dimensions
            realignment_planning_dims = [x for x in realignment_dims if x in planning_dimensions]
            realignment_planning_dims = sorted(
                realignment_planning_dims, key=lambda x: Dimensions.index(x.split(".")[0])
            )
            # relevant_planning_level_keys = realignment_planning_data["keys"]
            relevant_planning_level_from_cols = realignment_planning_data["from_cols"]
            relevant_planning_level_to_cols = realignment_planning_data["to_cols"]
            # relevant_planning_level_dataframes_by_key = (
            #     realignment_planning_data["dataframes_by_key"]
            # )

            # leaf level dimensions
            realignment_leaf_level_dims = [x for x in realignment_dims if x in leaf_dimensions]
            realignment_leaf_level_dims = sorted(
                realignment_leaf_level_dims, key=lambda x: Dimensions.index(x.split(".")[0])
            )
            relevant_leaf_level_keys = realignment_leaf_level_data["keys"]
            realignment_planning_leaf_level_dims = realignment_leaf_level_data["dims"]
            relevant_leaf_level_from_cols = realignment_leaf_level_data["from_cols"]
            relevant_leaf_level_to_cols = realignment_leaf_level_data["to_cols"]
            relevant_leaf_level_dataframes_by_key = realignment_leaf_level_data["dataframes_by_key"]

            # other level dimensions
            dims_to_drop = set(realignment_planning_dims).union(set(realignment_leaf_level_dims))
            realignment_other_level_dims = list(set(realignment_dims) - dims_to_drop)
            realignment_other_level_dims = sorted(
                realignment_other_level_dims, key=lambda x: Dimensions.index(x.split(".")[0])
            )
            relevant_other_level_keys = realignment_other_level_data["keys"]
            realignment_planning_other_level_dims = realignment_other_level_data["dims"]
            relevant_other_level_from_cols = realignment_other_level_data["from_cols"]
            relevant_other_level_to_cols = realignment_other_level_data["to_cols"]
            relevant_other_level_dataframes_by_key = realignment_other_level_data[
                "dataframes_by_key"
            ]

            planning_level_cols = (
                realignment_planning_dims
                + realignment_planning_leaf_level_dims
                + realignment_planning_other_level_dims
            )
            from_cols = (
                relevant_planning_level_from_cols
                + relevant_leaf_level_from_cols
                + relevant_other_level_from_cols
            )
            to_cols = (
                relevant_planning_level_to_cols
                + relevant_leaf_level_to_cols
                + relevant_other_level_to_cols
            )

            # rearrange lists to make sure all dimensions are in sequence
            planning_level_cols = rearrange_lists(Dimensions, planning_level_cols)
            from_cols = rearrange_lists(Dimensions, from_cols)
            to_cols = rearrange_lists(Dimensions, to_cols)

            # getting contribution percentage for leaf level and updating leaf level values with planning level values
            contribution_df = pd.DataFrame()
            if len(realignment_leaf_level_dims) > 0:
                logger.info("getting actual contribution percentages ...")

                # replacing from and to leaf level values with corresponding planning level values
                logger.info("getting planning level values corresponding to leaf level values ...")
                for i in range(0, len(relevant_leaf_level_keys)):
                    group = group.merge(
                        relevant_leaf_level_dataframes_by_key[relevant_leaf_level_keys[i]][
                            [
                                realignment_planning_leaf_level_dims[i],
                                realignment_leaf_level_dims[i],
                            ]
                        ],
                        left_on=relevant_leaf_level_from_cols[i],
                        right_on=realignment_leaf_level_dims[i],
                        how="inner",
                    )
                    group = group.merge(
                        relevant_leaf_level_dataframes_by_key[relevant_leaf_level_keys[i]][
                            [
                                realignment_planning_leaf_level_dims[i],
                                realignment_leaf_level_dims[i],
                            ]
                        ],
                        left_on=[relevant_leaf_level_to_cols[i]],
                        right_on=[realignment_leaf_level_dims[i]],
                        how="inner",
                        suffixes=("", "_ac"),
                    )

                    group[relevant_leaf_level_from_cols[i]] = group[
                        realignment_planning_leaf_level_dims[i]
                    ]
                    group[relevant_leaf_level_to_cols[i]] = group[
                        realignment_planning_leaf_level_dims[i] + "_ac"
                    ]

                logger.info("Normalizing actual contribution values ...")
                contribution_df = pd.merge(
                    ActualContribution,
                    group[relevant_leaf_level_from_cols].drop_duplicates(),
                    left_on=realignment_planning_leaf_level_dims,
                    right_on=relevant_leaf_level_from_cols,
                    how="inner",
                )
                contribution_df = contribution_df[
                    realignment_leaf_level_dims + planning_dimensions + [actual_contribution_per]
                ]

                # normalizing the actual contribution per
                contribution_df[actual_contribution_per] = contribution_df.groupby(
                    planning_dimensions + realignment_leaf_level_dims,
                    observed=True,
                )[actual_contribution_per].transform("sum")
                contribution_df.drop_duplicates(inplace=True)

                # normalizing values
                contribution_df["Sum"] = contribution_df.groupby(
                    planning_dimensions, observed=True
                )[actual_contribution_per].transform("sum")
                contribution_df[actual_contribution_per] = (
                    contribution_df[actual_contribution_per] / contribution_df["Sum"]
                )
                contribution_df.drop(columns=["Sum"], inplace=True)

            # updating other level values with planning level values
            if len(relevant_other_level_keys) > 0:
                logger.info(
                    "getting planning level values corresponding to higher level values ..."
                )
                # replacing from and to leaf level values with corresponding planning level values
                for i in range(0, len(relevant_other_level_keys)):
                    col_name = [
                        col
                        for col in realignment_other_level_dims
                        if relevant_other_level_keys[i] in col
                    ][0]
                    group = group.merge(
                        relevant_other_level_dataframes_by_key[relevant_other_level_keys[i]][
                            [
                                realignment_planning_other_level_dims[i],
                                col_name,
                            ]
                        ].drop_duplicates(),
                        left_on=relevant_other_level_from_cols[i],
                        right_on=col_name,
                        how="inner",
                    )
                    group[relevant_other_level_from_cols[i]] = group[
                        realignment_planning_other_level_dims[i]
                    ]
                    group[relevant_other_level_to_cols[i]] = group[
                        realignment_planning_other_level_dims[i]
                    ]
                    group.drop(
                        columns=[
                            realignment_planning_other_level_dims[i],
                            col_name,
                        ],
                        inplace=True,
                    )

            leaf_assortment_cols = realignment_leaf_level_dims + [
                x + "_ac" for x in realignment_leaf_level_dims
            ]
            relevant_cols = list(set(from_cols).union(set(to_cols)))

            logger.info("calculation assortment and is assorted output ...")
            assortment_intersections = group[
                list(set(relevant_cols).union(set(leaf_assortment_cols)))
            ].drop_duplicates()
            assortment_intersections = assortment_intersections[
                assortment_intersections[from_cols].values
                != assortment_intersections[to_cols].values
            ]

            is_assorted_intersections = group[relevant_cols].drop_duplicates()
            is_assorted_intersections = is_assorted_intersections[
                is_assorted_intersections[from_cols].values
                != is_assorted_intersections[to_cols].values
            ]
            if len(leaf_assortment_cols) > 0:
                assortment_intersections = assortment_intersections[
                    ~(
                        assortment_intersections[realignment_leaf_level_dims].values
                        == assortment_intersections[
                            [x + "_ac" for x in realignment_leaf_level_dims]
                        ].values
                    ).all(axis=1)
                ]

            # getting mapping of from, planning and to cols
            assortment_column_mapping = {}
            for i in range(0, len(from_cols)):
                assortment_column_mapping[from_cols[i]] = (
                    planning_level_cols[i],
                    to_cols[i],
                )

            assortment_intersections.rename(
                columns={key: value[0] for key, value in assortment_column_mapping.items()},
                inplace=True,
            )
            is_assorted_intersections.rename(
                columns={key: value[0] for key, value in assortment_column_mapping.items()},
                inplace=True,
            )

            assortment_intersections = assortment_intersections.merge(
                AssortmentFinal,
                how="inner",
            )
            is_assorted_intersections = is_assorted_intersections.merge(
                IsAssorted,
                how="inner",
            )
            # replacing planning value with to value
            for from_col, (
                planning_col,
                to_col,
            ) in assortment_column_mapping.items():
                assortment_intersections[planning_col] = assortment_intersections[to_col]
                is_assorted_intersections[planning_col] = is_assorted_intersections[to_col]

            assortment_intersections.drop(columns=[item_col, location_col], inplace=True)
            assortment_intersections.drop_duplicates(inplace=True)
            assortment_intersections = assortment_intersections.merge(
                ItemMapping[[pl_item_col, item_col]]
            )
            assortment_intersections = assortment_intersections.merge(
                LocationMapping[[pl_location_col, location_col]]
            )

            assortment_intersections.rename(
                columns={assortment_final: assortment_output}, inplace=True
            )

            if len(leaf_assortment_cols) > 0:
                # Update values for all columns in realignment_leaf_level_dims with corresponding _ac values
                assortment_intersections[realignment_leaf_level_dims] = assortment_intersections[
                    [f"{col}_ac" for col in realignment_leaf_level_dims]
                ].values

            assortment_intersections = assortment_intersections[
                assortment_output_cols
            ].drop_duplicates()
            is_assorted_intersections = is_assorted_intersections[IsAssorted.columns]

            assortment_intersections_list.append(assortment_intersections)
            is_assorted_intersections_list.append(is_assorted_intersections)

            # removing extra columns
            if len(realignment_leaf_level_dims) > 0:
                for i in range(0, len(relevant_leaf_level_keys)):
                    group.drop(
                        columns=[
                            realignment_planning_leaf_level_dims[i],
                            realignment_planning_leaf_level_dims[i] + "_ac",
                            realignment_leaf_level_dims[i] + "_ac",
                        ],
                        inplace=True,
                    )

            remaining_vars = set(from_dimensions) - set(from_cols)
            group.drop(columns=list(remaining_vars), inplace=True)
            remaining_vars = set(to_dimensions) - set(to_cols)
            group.drop(columns=list(remaining_vars), inplace=True)
            group.drop(columns=[realignments_col], inplace=True)

            override_from_planning_mapping = {
                key: value for key, value in zip(from_cols, planning_level_cols)
            }
            override_to_planning_mapping = {
                key: value for key, value in zip(to_cols, planning_level_cols)
            }
            override_raw_from_intersections = group[group[realignment_status] != 1][
                from_cols
            ].drop_duplicates()
            override_raw_to_intersections = group[group[realignment_status] != 1][
                to_cols
            ].drop_duplicates()

            override_raw_from_intersections.rename(
                columns=override_from_planning_mapping, inplace=True
            )
            override_raw_to_intersections.rename(columns=override_to_planning_mapping, inplace=True)
            override_raw_intersections = pd.concat(
                [
                    override_raw_from_intersections,
                    override_raw_to_intersections,
                ]
            )

            common_cols = [col for col in planning_level_cols if col in sell_in_dimensions]
            relevant_sell_in_forecast_raw_output = sell_in_forecast_raw_output.merge(
                override_raw_intersections[common_cols].drop_duplicates(),
                on=common_cols,
            )
            sell_in_override_raw_output_list.append(relevant_sell_in_forecast_raw_output)

            common_cols = [col for col in planning_level_cols if col in sell_out_dimensions]
            if len(common_cols) > 0:
                relevant_sell_out_forecast_raw_output = sell_out_forecast_raw_output.merge(
                    override_raw_intersections[common_cols].drop_duplicates(),
                    on=common_cols,
                )
            else:
                relevant_sell_out_forecast_raw_output = pd.DataFrame(
                    columns=sell_out_forecast_raw_output.columns
                )
            sell_out_override_raw_output_list.append(relevant_sell_out_forecast_raw_output)

            # getting realigned forecast
            logger.info("realigning forecast based on given rules ...")
            processed_dataframes = {
                name: get_realigned_forecast(
                    intersections_considered=value[4],
                    realignment_leaf_level_dims=realignment_leaf_level_dims,
                    name=name,
                    realignment_status=realignment_status,
                    relevant_keys=relevant_keys,
                    relevant_column_mapping=value[0],
                    dimensions=value[1],
                    measures=value[2],
                    FcstRaw=value[3],
                    max_dates=value[5],
                    planning_level_cols=planning_level_cols,
                    from_cols=from_cols,
                    to_cols=to_cols,
                    group=group,
                    data_object=data_object,
                    planning_dimensions=planning_dimensions,
                    version_col=version_col,
                    partial_week_key_col=partial_week_key_col,
                    partial_week_col=partial_week_col,
                    start_date=start_date,
                    end_date=end_date,
                    contribution_df=contribution_df,
                    actual_contribution_per=actual_contribution_per,
                    realignment_sum=realignment_sum,
                    realignment_percentage=realignment_percentage,
                    conversion_factor=conversion_factor,
                )
                for name, value in sell_in_sell_out_mapping.items()
            }

            # Sell In Forecast
            (
                SellInForecast_Raw_override_output,
                intersections_already_considered_sell_in_override,
            ) = get_forecast_data(
                processed_dataframes,
                "Sell In Override",
                sell_in_dimensions,
                output_sell_in_override_measures,
            )

            (
                SellInForecast_Raw_non_override_output,
                intersections_already_considered_sell_in_non_override,
            ) = get_forecast_data(
                processed_dataframes,
                "Sell In Non Override",
                sell_in_dimensions,
                output_sell_in_non_override_measures,
            )

            sell_in_override_output_list.append(SellInForecast_Raw_override_output)
            sell_in_non_override_output_list.append(SellInForecast_Raw_non_override_output)

            SellInForecast_Raw_override = append_input_output_data(
                dimensions=list(set(sell_in_dimensions) - set([version_col, partial_week_col])),
                columns_mapping=sell_in_columns_mapping,
                Forecast_Raw=SellInForecast_Raw_override,
                Forecast_Raw_output=SellInForecast_Raw_override_output,
            )
            SellInForecast_Raw_non_override = append_input_output_data(
                dimensions=list(set(sell_in_dimensions) - set([version_col, partial_week_col])),
                columns_mapping=sell_in_columns_mapping,
                Forecast_Raw=SellInForecast_Raw_non_override,
                Forecast_Raw_output=SellInForecast_Raw_non_override_output,
            )

            # Sell Out Forecast
            (
                SellOutForecast_Raw_override_output,
                intersections_already_considered_sell_out_override,
            ) = get_forecast_data(
                processed_dataframes,
                "Sell Out Override",
                sell_out_dimensions,
                output_sell_out_override_measures,
            )

            (
                SellOutForecast_Raw_non_override_output,
                intersections_already_considered_sell_out_non_override,
            ) = get_forecast_data(
                processed_dataframes,
                "Sell Out Non Override",
                sell_out_dimensions,
                output_sell_out_non_override_measures,
            )

            sell_out_override_output_list.append(SellOutForecast_Raw_override_output)
            sell_out_non_override_output_list.append(SellOutForecast_Raw_non_override_output)

            SellOutForecast_Raw_override = append_input_output_data(
                dimensions=list(set(sell_out_dimensions) - set([version_col, partial_week_col])),
                columns_mapping=sell_out_columns_mapping,
                Forecast_Raw=SellOutForecast_Raw_override,
                Forecast_Raw_output=SellOutForecast_Raw_override_output,
            )
            SellOutForecast_Raw_non_override = append_input_output_data(
                dimensions=list(set(sell_out_dimensions) - set([version_col, partial_week_col])),
                columns_mapping=sell_out_columns_mapping,
                Forecast_Raw=SellOutForecast_Raw_non_override,
                Forecast_Raw_output=SellOutForecast_Raw_non_override_output,
            )

            logger.info("appending output data ...")
            sell_in_override_intersections_already_considered_list.append(
                intersections_already_considered_sell_in_override
            )
            sell_in_non_override_intersections_already_considered_list.append(
                intersections_already_considered_sell_in_non_override
            )
            sell_out_override_intersections_already_considered_list.append(
                intersections_already_considered_sell_out_override
            )
            sell_out_non_override_intersections_already_considered_list.append(
                intersections_already_considered_sell_out_non_override
            )

        RealignedForecastSellIn_override = concat_to_dataframe(sell_in_override_output_list)
        RealignedForecastSellIn_non_override = concat_to_dataframe(sell_in_non_override_output_list)
        RealignedForecastSellOut_override = concat_to_dataframe(sell_out_override_output_list)
        RealignedForecastSellOut_non_override = concat_to_dataframe(
            sell_out_non_override_output_list
        )
        Assortment = concat_to_dataframe(assortment_intersections_list)
        IsAssorted_output = concat_to_dataframe(is_assorted_intersections_list)
        relevant_sell_in_forecast_raw_output = concat_to_dataframe(sell_in_override_raw_output_list)
        relevant_sell_out_forecast_raw_output = concat_to_dataframe(
            sell_out_override_raw_output_list
        )

        is_assorted_output_intersections = IsAssorted_output[
            is_assorted_dimensions
        ].drop_duplicates()
        IsAssorted = IsAssorted.merge(
            is_assorted_output_intersections,
            how="outer",
            indicator=True,
        )
        IsAssorted = IsAssorted[IsAssorted["_merge"] == "left_only"]
        IsAssorted.drop(columns=["_merge"], inplace=True)
        IsAssortedOutput = pd.concat([IsAssorted, IsAssorted_output])
        IsAssortedOutput = IsAssortedOutput[is_assorted_output_cols]

        RealignedForecastSellIn_override = get_remaining_intersections(
            source_df=RealignedForecastSellIn_override,
            target_df=SellInForecast_Raw_override,
            planning_dimensions=planning_dimensions,
            column_mapping={},
            cols_required_in_output=sell_in_dimensions + output_sell_in_override_measures,
        )
        RealignedForecastSellIn_non_override = get_remaining_intersections(
            source_df=RealignedForecastSellIn_non_override,
            target_df=SellInForecast_Raw_non_override.rename(
                columns=sell_in_non_override_column_mapping
            ),
            planning_dimensions=planning_dimensions,
            column_mapping=sell_in_non_override_column_mapping,
            cols_required_in_output=sell_in_dimensions + output_sell_in_non_override_measures,
        )
        RealignedForecastSellOut_override = get_remaining_intersections(
            source_df=RealignedForecastSellOut_override,
            target_df=SellOutForecast_Raw_override,
            planning_dimensions=sell_out_dimensions,
            column_mapping={},
            cols_required_in_output=sell_out_dimensions + output_sell_out_override_measures,
        )
        RealignedForecastSellOut_non_override = get_remaining_intersections(
            source_df=RealignedForecastSellOut_non_override,
            target_df=SellOutForecast_Raw_non_override.rename(
                columns=sell_out_non_override_column_mapping
            ),
            planning_dimensions=sell_out_dimensions,
            column_mapping=sell_out_non_override_column_mapping,
            cols_required_in_output=sell_out_dimensions + output_sell_out_non_override_measures,
        )

        RealignedForecastSellIn = pd.merge(
            RealignedForecastSellIn_override.drop_duplicates(),
            RealignedForecastSellIn_non_override.drop_duplicates(),
            how="outer",
        )
        RealignedForecastSellOut = pd.merge(
            RealignedForecastSellOut_override.drop_duplicates(),
            RealignedForecastSellOut_non_override.drop_duplicates(),
            how="outer",
        )

        RealignedForecastSellIn.drop_duplicates(
            subset=sell_in_dimensions, keep="last", inplace=True
        )
        RealignedForecastSellOut.drop_duplicates(
            subset=sell_out_dimensions, keep="last", inplace=True
        )

        RawSellInOverrideOutput = relevant_sell_in_forecast_raw_output.copy()
        RawSellOutOverrideOutput = relevant_sell_out_forecast_raw_output.copy()

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        SellInForecast_Raw.rename(columns=sell_in_columns_mapping, inplace=True)
        RealignedForecastSellIn = SellInForecast_Raw[sell_in_output_cols]

        SellOutForecast_Raw.rename(columns=sell_out_columns_mapping, inplace=True)
        RealignedForecastSellOut = SellOutForecast_Raw[sell_out_output_cols]

        IsAssortedOutput = IsAssorted.copy()
        IsAssortedOutput = IsAssortedOutput[is_assorted_output_cols]

    return (
        RealignedForecastSellIn,
        RealignedForecastSellOut,
        Assortment,
        IsAssortedOutput,
        FlagAndNormalizedValues,
        ActualContributionOutput,
        RawSellInOverrideOutput,
        RawSellOutOverrideOutput,
    )

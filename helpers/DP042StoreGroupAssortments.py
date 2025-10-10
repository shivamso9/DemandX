import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Assortment Rule Selected": str,
    "Assortment Location Group": str,
    "Assortment System": bool,
    "Intro Date": pd.Timestamp,
    "Disco Date": pd.Timestamp,
}

# Configurables
level_col_item = "Item."
level_col_location = "Location."
level_col_region = "Region."
level_col_channel = "Channel."
level_col_account = "Account."
level_col_PnL = "PnL."
level_col_demand = "Demand Domain."
value_col_item = "DP Item Level"
value_col_location = "DP Location Level"
value_col_region = "DP Region Level"
value_col_channel = "DP Channel Level"
value_col_account = "DP Account Level"
value_col_PnL = "DP PnL Level"
value_col_demand = "DP Demand Domain Level"
scope_col_item = "DP Item Scope"
scope_col_location = "DP Location Scope"
scope_col_region = "DP Region Scope"
scope_col_channel = "DP Channel Scope"
scope_col_account = "DP Account Scope"
scope_col_PnL = "DP PnL Scope"
scope_col_demand = "DP Demand Domain Scope"
Item_L3_col = "Item.[L3]"
Item_L4_col = "Item.[L4]"
Item_L5_col = "Item.[L5]"
Item_L6_col = "Item.[L6]"
Location_goup_scope_col = "DP Location Group Scope"
Rule_sequence_col = "DP Rule Sequence"
Rule_Created_Date_col = "DP Rule Created Date"
Rule_x_col = "DM Rule.[Rule]_x"
Rule_y_col = "DM Rule.[Rule]_y"
Exclude_flag_col = "DP Exclude Flag"
Intro_date_col = "DP Intro Date"
Disco_date_col = "DP Disco Date"
FromDate_col = "Location.[Store Intro Date]"
ToDate_col = "Location.[Store Disc Date]"
Assortment_rule_selected_col = "Assortment Rule Selected"
Assortment_system_col = "Assortment System"
Assortment_location_group_col = "Assortment Location Group"
final_intro_date_col = "Intro Date"
final_disco_date_col = "Disco Date"
object_type_col = "Data Object.[Data Object Type]"
cluster_col = "Cluster.[Cluster]"
rule_col = "DM Rule.[Rule]"
placeholder_value = "no value.."
is_selected = "is_selected"
Rule_Created_by_col = "DP Rule Created By"


def split_dataframe(df, n_chunks):
    """Split DataFrame into n_chunks."""
    chunk_size = len(df) // n_chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks


def get_relevant_mapping(df, AttributeMapping, level_col, value_col, scope_col):
    merge_col = f"{level_col}[" + df[value_col].unique()[0] + "]"
    df = pd.merge(
        df,
        AttributeMapping,
        left_on=[scope_col],
        right_on=[merge_col],
        how="inner",
    )
    return df


def process_dataframe(
    df,
    DimItem,
    DimLocation,
    DimRegion,
    DimChannel,
    DimAccount,
    DimPnL,
    DimDemandDomain,
):
    # Explode the columns

    df[scope_col_item] = df[scope_col_item].str.split(",")
    df[scope_col_location] = df[scope_col_location].str.split(",")
    df[scope_col_region] = df[scope_col_region].str.split(",")

    df = df.explode(scope_col_item)
    df = df.explode(scope_col_location)
    df = df.explode(scope_col_region)

    # Process item mapping
    item_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimItem,
            level_col=level_col_item,
            value_col=value_col_item,
            scope_col=scope_col_item,
        )
        for name, group in df.groupby(value_col_item, observed=True)
    )
    item_df = pd.concat(item_results)

    # Drop unnecessary columns and remove duplicates
    final_item_df = item_df.drop(columns=[value_col_item, scope_col_item], errors="ignore")
    final_item_df = final_item_df.drop_duplicates(keep="first")

    location_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimLocation,
            level_col=level_col_location,
            value_col=value_col_location,
            scope_col=scope_col_location,
        )
        for name, group in final_item_df.groupby(value_col_location, observed=True)
    )
    location_df = pd.concat(location_results)

    region_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimRegion,
            level_col=level_col_region,
            value_col=value_col_region,
            scope_col=scope_col_region,
        )
        for name, group in location_df.groupby(value_col_region, observed=True)
    )
    region_df = pd.concat(region_results)

    channel_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimChannel,
            level_col=level_col_channel,
            value_col=value_col_channel,
            scope_col=scope_col_channel,
        )
        for name, group in region_df.groupby(value_col_channel, observed=True)
    )
    channel_df = pd.concat(channel_results)

    account_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimAccount,
            level_col=level_col_account,
            value_col=value_col_account,
            scope_col=scope_col_account,
        )
        for name, group in channel_df.groupby(value_col_account, observed=True)
    )
    account_df = pd.concat(account_results)

    PnL_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimPnL,
            level_col=level_col_PnL,
            value_col=value_col_PnL,
            scope_col=scope_col_PnL,
        )
        for name, group in account_df.groupby(value_col_PnL, observed=True)
    )
    PnL_df = pd.concat(PnL_results)

    Demand_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_relevant_mapping)(
            df=group,
            AttributeMapping=DimDemandDomain,
            level_col=level_col_demand,
            value_col=value_col_demand,
            scope_col=scope_col_demand,
        )
        for name, group in PnL_df.groupby(value_col_demand, observed=True)
    )
    Demand_df = pd.concat(Demand_results)
    final_demand_df = Demand_df
    return final_demand_df


def get_filtered_dataframe(df):
    # Check if 'DP Location Group Scope' has more than one unique value
    cols_to_drop = [
        Item_L4_col,
        Item_L5_col,
        Item_L6_col,
        value_col_location,
        scope_col_location,
        value_col_channel,
        scope_col_channel,
        value_col_account,
        scope_col_account,
        value_col_region,
        scope_col_region,
        value_col_PnL,
        scope_col_PnL,
        value_col_demand,
        scope_col_demand,
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    if df[Location_goup_scope_col].nunique() > 1:
        # Sort by 'DP Rule Sequence' descending and 'DP Rule Created Date' descending
        df_sorted = df.sort_values(
            by=[Rule_sequence_col, Rule_Created_Date_col],
            ascending=[False, False],
        )
        # Get the first unique value of 'DP Location Group Scope' after sorting
        first_group_value = df_sorted[Location_goup_scope_col].iloc[0]

        # Filter to include only rows with the first 'Location_goup_scope_col' value
        df_filtered = df_sorted[df_sorted[Location_goup_scope_col] == first_group_value]
        df_filtered = df_filtered.drop_duplicates()
        df = df_filtered
        return df
    else:
        # Return the entire group unchanged
        df = df.drop_duplicates()
        return df


def process_intro_disco_date(df, StoreDates):
    df = pd.merge(
        df,
        StoreDates,
        on=[o9Constants.LOCATION, Location_goup_scope_col, Item_L3_col],
        how="left",
    )

    # List of date columns you want to ensure exist
    date_columns = [Intro_date_col, Disco_date_col]

    # Check each column, if it doesn't exist, create it and assign NaT
    for col in date_columns:
        if col not in df.columns:
            df[col] = pd.NaT  # Assign NaT if column doesn't exist

    # Ensure 'DP Intro Date', 'FromDate', 'DP Disco Date', and 'ToDate' are datetime objects
    df[Intro_date_col] = pd.to_datetime(df[Intro_date_col], errors="coerce")
    df[FromDate_col] = pd.to_datetime(df[FromDate_col], errors="coerce")
    df[Disco_date_col] = pd.to_datetime(df[Disco_date_col], errors="coerce")
    df[ToDate_col] = pd.to_datetime(df[ToDate_col], errors="coerce")

    # For 'DP Intro Date': Update to the latest date between 'DP Intro Date' and 'FromDate'
    df[Intro_date_col] = np.where(
        pd.isna(df[Intro_date_col]) & pd.isna(df[FromDate_col]),  # Case where both are NaT
        pd.NaT,  # Result should be NaT
        np.where(
            pd.notna(df[Intro_date_col]) & pd.notna(df[FromDate_col]),
            np.maximum(df[Intro_date_col], df[FromDate_col]),
            df[Intro_date_col].fillna(df[FromDate_col]),  # Result if only one is NaT
        ),
    )

    # For 'DP Disco Date': Update to the oldest date between 'DP Disco Date' and 'ToDate'
    df[Disco_date_col] = np.where(
        pd.isna(df[Disco_date_col]) & pd.isna(df[ToDate_col]),  # Case where both are NaT
        pd.NaT,  # Result should be NaT
        np.where(
            pd.notna(df[Disco_date_col]) & pd.notna(df[ToDate_col]),
            np.minimum(df[Disco_date_col], df[ToDate_col]),
            df[Disco_date_col].fillna(df[ToDate_col]),  # Result if only one is NaT
        ),
    )

    df[Intro_date_col] = pd.to_datetime(df[Intro_date_col], errors="coerce")
    df[Disco_date_col] = pd.to_datetime(df[Disco_date_col], errors="coerce")

    # Format the dates as strings in MM/dd/yyyy format for display purposes
    df[Intro_date_col] = df[Intro_date_col].dt.strftime("%m/%d/%Y")
    df[Disco_date_col] = df[Disco_date_col].dt.strftime("%m/%d/%Y")

    # Convert formatted strings back to datetime to ensure dtype is datetime
    df[Intro_date_col] = pd.to_datetime(df[Intro_date_col], format="%m/%d/%Y", errors="coerce")
    df[Disco_date_col] = pd.to_datetime(df[Disco_date_col], format="%m/%d/%Y", errors="coerce")

    return df


def process_assortment_system(df):

    # List of date columns you want to ensure exist
    date_columns = [Exclude_flag_col]

    # Check each column, if it doesn't exist, create it and assign NaT
    for col in date_columns:
        if col not in df.columns:
            df[col] = np.nan  # Assign NaT if column doesn't exist

    # populate measure assortment system
    df[Exclude_flag_col] = df[Exclude_flag_col].fillna(False)
    df[Assortment_system_col] = np.where(df[Exclude_flag_col], 0, 1)

    # df = df[df[Assortment_system_col] != 0]

    return df


def merge_and_conditionally_filter(
    df,
    value_col_item,
    value_col_location,
    value_col_channel,
    value_col_account,
    value_col_region,
    value_col_PnL,
    value_col_demand,
    level_col_item,
    level_col_location,
    level_col_channel,
    level_col_account,
    level_col_region,
    level_col_PnL,
    level_col_demand,
    scope_col_item,
    scope_col_location,
    scope_col_channel,
    scope_col_account,
    StoreDates,
    scope_col_region,
    scope_col_PnL,
    scope_col_demand,
    chunk,
):

    if df.empty:
        df = df.drop(columns=[Rule_sequence_col, Rule_Created_Date_col])
        chunk = chunk.drop(columns=[Rule_sequence_col, Rule_Created_Date_col])
        df = pd.merge(
            df,
            chunk,
            on=[o9Constants.VERSION_NAME, Location_goup_scope_col],
            how="right",
        )

        updated_chunk = pd.DataFrame()
        updated_chunk = chunk

    else:

        relevant_merge_col_item = level_col_item + "[" + df[value_col_item].unique()[0] + "]"
        relevant_merge_col_location = (
            level_col_location + "[" + df[value_col_location].unique()[0] + "]"
        )
        relevant_merge_col_channel = (
            level_col_channel + "[" + df[value_col_channel].unique()[0] + "]"
        )
        relevant_merge_col_account = (
            level_col_account + "[" + df[value_col_account].unique()[0] + "]"
        )
        relevant_merge_col_region = level_col_region + "[" + df[value_col_region].unique()[0] + "]"
        relevant_merge_col_PnL = level_col_PnL + "[" + df[value_col_PnL].unique()[0] + "]"
        relevant_merge_col_demand = level_col_demand + "[" + df[value_col_demand].unique()[0] + "]"

        df[scope_col_item] = df[scope_col_item].str.split(",")
        # df[scope_col_location] = df[scope_col_location].str.split(',')

        df = df.explode(scope_col_item)
        # df = df.explode(scope_col_location)

        # Columns to be excluded from uniqueness determination
        exclude_columns = [
            rule_col,
            Rule_Created_by_col,
            Intro_date_col,
            Disco_date_col,
            Rule_Created_Date_col,
            Rule_sequence_col,
        ]

        # Define the columns to consider for uniqueness
        unique_columns = [col for col in df.columns if col not in exclude_columns]

        df = df.sort_values(
            by=[Rule_sequence_col, Rule_Created_Date_col],
            ascending=[False, False],
        )

        # Drop duplicates, keeping the first occurrence of each combination of unique_columns
        df = df.drop_duplicates(subset=unique_columns, keep="first")

        left_on_cols = [
            scope_col_item,
            scope_col_location,
            scope_col_channel,
            scope_col_region,
            scope_col_account,
            scope_col_PnL,
            scope_col_demand,
            o9Constants.VERSION_NAME,
            Location_goup_scope_col,
        ]

        right_on_cols = [
            relevant_merge_col_item,
            relevant_merge_col_location,
            relevant_merge_col_channel,
            relevant_merge_col_region,
            relevant_merge_col_account,
            relevant_merge_col_PnL,
            relevant_merge_col_demand,
            o9Constants.VERSION_NAME,
            Location_goup_scope_col,
        ]

        # Remove columns from left_on and right_on if they contain only the placeholder value
        if (df[scope_col_item] == placeholder_value).all():
            left_on_cols.remove(scope_col_item)
            right_on_cols.remove(relevant_merge_col_item)

        if (df[scope_col_location] == placeholder_value).all():
            left_on_cols.remove(scope_col_location)
            right_on_cols.remove(relevant_merge_col_location)

        if (df[scope_col_channel] == placeholder_value).all():
            left_on_cols.remove(scope_col_channel)
            right_on_cols.remove(relevant_merge_col_channel)

        if (df[scope_col_region] == placeholder_value).all():
            left_on_cols.remove(scope_col_region)
            right_on_cols.remove(relevant_merge_col_region)

        if (df[scope_col_account] == placeholder_value).all():
            left_on_cols.remove(scope_col_account)
            right_on_cols.remove(relevant_merge_col_account)

        if (df[scope_col_PnL] == placeholder_value).all():
            left_on_cols.remove(scope_col_PnL)
            right_on_cols.remove(relevant_merge_col_PnL)

        if (df[scope_col_demand] == placeholder_value).all():
            left_on_cols.remove(scope_col_demand)
            right_on_cols.remove(relevant_merge_col_demand)

        if (df[Location_goup_scope_col] == placeholder_value).all():
            left_on_cols.remove(Location_goup_scope_col)
            df = df.drop(columns=[Location_goup_scope_col])
            right_on_cols.remove(Location_goup_scope_col)

        temp_df = df

        df = pd.merge(
            df,
            chunk,
            left_on=left_on_cols,
            right_on=right_on_cols,
            how="inner",
        )

        common_rows = pd.merge(
            chunk[right_on_cols],
            temp_df[left_on_cols],
            left_on=right_on_cols,
            right_on=left_on_cols,
            how="inner",
        )
        # updated_chunk = chunk[~chunk.index.isin(common_rows.index)]

        # Extract only the columns used for merging
        common_rows_subset = common_rows[right_on_cols].drop_duplicates()

        # Find rows in chunk that are not in common_rows
        updated_chunk = chunk[
            ~chunk[right_on_cols].apply(tuple, axis=1).isin(common_rows_subset.apply(tuple, axis=1))
        ]

        df = df.drop_duplicates()
        updated_chunk = updated_chunk.drop_duplicates()

    df = df.drop(
        columns=[
            value_col_item,
            scope_col_item,
            value_col_location,
            scope_col_location,
            value_col_channel,
            scope_col_channel,
            value_col_account,
            scope_col_account,
            value_col_region,
            scope_col_region,
            value_col_demand,
            scope_col_demand,
            value_col_PnL,
            scope_col_PnL,
        ]
    )

    # Convert Rule_x_col and Rule_y_col to strings, replacing NaN with empty strings
    df[Rule_x_col] = df[Rule_x_col].astype(str).replace("nan", "")
    df[Rule_y_col] = df[Rule_y_col].astype(str).replace("nan", "")

    # Create conditions for the logic
    conditions = [
        (df[Rule_x_col] == df[Rule_y_col]) & (df[Rule_x_col] != ""),  # Both are equal and not empty
        (df[Rule_x_col] == ""),  # Rule_x_col is empty
        (df[Rule_y_col] == ""),  # Rule_y_col is empty
        (df[Rule_x_col] == "") & (df[Rule_y_col] == ""),  # Both are empty
    ]

    # Corresponding choices for each condition
    choices = [
        df[Rule_x_col],  # If both are equal and not empty, take either one
        df[Rule_y_col],  # If Rule_x_col is empty, take Rule_y_col
        df[Rule_x_col],  # If Rule_y_col is empty, take Rule_x_col
        np.nan,  # If both are empty, result should be NaN
    ]

    # Use np.select to apply the logic
    df[rule_col] = np.select(conditions, choices, default=df[Rule_x_col] + ", " + df[Rule_y_col])

    # Drop the intermediate columns
    df = df.drop(columns=[Rule_x_col, Rule_y_col])

    df_final = process_intro_disco_date(df, StoreDates)
    return df_final, updated_chunk


def get_assortment_system(
    df,
    value_col_item,
    value_col_location,
    value_col_channel,
    value_col_account,
    value_col_region,
    value_col_PnL,
    value_col_demand,
    level_col_item,
    level_col_location,
    level_col_channel,
    level_col_account,
    level_col_region,
    level_col_PnL,
    level_col_demand,
    scope_col_item,
    scope_col_location,
    scope_col_channel,
    scope_col_account,
    scope_col_region,
    scope_col_PnL,
    scope_col_demand,
    chunk,
):

    if df.empty:
        df = df.drop(columns=[Rule_sequence_col, Rule_Created_Date_col])
        df = pd.merge(
            df,
            chunk,
            on=[o9Constants.VERSION_NAME, Location_goup_scope_col],
            how="right",
        )

        updated_chunk = pd.DataFrame()
        updated_chunk = chunk

    else:
        relevant_merge_col_item = level_col_item + "[" + df[value_col_item].unique()[0] + "]"
        relevant_merge_col_location = (
            level_col_location + "[" + df[value_col_location].unique()[0] + "]"
        )
        relevant_merge_col_channel = (
            level_col_channel + "[" + df[value_col_channel].unique()[0] + "]"
        )
        relevant_merge_col_account = (
            level_col_account + "[" + df[value_col_account].unique()[0] + "]"
        )
        relevant_merge_col_region = level_col_region + "[" + df[value_col_region].unique()[0] + "]"
        relevant_merge_col_PnL = level_col_PnL + "[" + df[value_col_PnL].unique()[0] + "]"
        relevant_merge_col_demand = level_col_demand + "[" + df[value_col_demand].unique()[0] + "]"

        # Convert comma-separated strings to lists
        df[scope_col_item] = df[scope_col_item].str.split(",")
        df[scope_col_location] = df[scope_col_location].str.split(",")
        df[scope_col_region] = df[scope_col_region].str.split(",")

        df = df.explode(scope_col_item)
        df = df.explode(scope_col_location)
        df = df.explode(scope_col_region)

        # Columns to be excluded from uniqueness determination
        exclude_columns = [
            rule_col,
            Rule_Created_by_col,
            Exclude_flag_col,
            Rule_Created_Date_col,
            Rule_sequence_col,
        ]

        # Define the columns to consider for uniqueness
        unique_columns = [col for col in df.columns if col not in exclude_columns]

        df = df.sort_values(
            by=[Rule_sequence_col, Rule_Created_Date_col],
            ascending=[False, False],
        )

        # Drop duplicates, keeping the first occurrence of each combination of unique_columns
        df = df.drop_duplicates(subset=unique_columns, keep="first")
        df = df.drop(columns=[Rule_sequence_col, Rule_Created_Date_col])

        left_on_cols = [
            scope_col_item,
            scope_col_location,
            scope_col_channel,
            scope_col_region,
            scope_col_account,
            scope_col_PnL,
            scope_col_demand,
            o9Constants.VERSION_NAME,
            Location_goup_scope_col,
        ]

        right_on_cols = [
            relevant_merge_col_item,
            relevant_merge_col_location,
            relevant_merge_col_channel,
            relevant_merge_col_region,
            relevant_merge_col_account,
            relevant_merge_col_PnL,
            relevant_merge_col_demand,
            o9Constants.VERSION_NAME,
            Location_goup_scope_col,
        ]

        # Remove columns from left_on and right_on if they contain only the placeholder value
        if (df[scope_col_location] == placeholder_value).all():
            left_on_cols.remove(scope_col_location)
            right_on_cols.remove(relevant_merge_col_location)

        if (df[Location_goup_scope_col] == placeholder_value).all():
            left_on_cols.remove(Location_goup_scope_col)
            df = df.drop(columns=[Location_goup_scope_col])
            right_on_cols.remove(Location_goup_scope_col)

        temp_df = df

        df = pd.merge(
            df,
            chunk,
            left_on=left_on_cols,
            right_on=right_on_cols,
            how="inner",
        )

        common_rows = pd.merge(
            chunk[right_on_cols],
            temp_df[left_on_cols],
            left_on=right_on_cols,
            right_on=left_on_cols,
            how="inner",
        )
        # updated_chunk = chunk[~chunk.index.isin(common_rows.index)]

        # Extract only the columns used for merging
        common_rows_subset = common_rows[right_on_cols].drop_duplicates()

        # Find rows in chunk that are not in common_rows
        updated_chunk = chunk[
            ~chunk[right_on_cols].apply(tuple, axis=1).isin(common_rows_subset.apply(tuple, axis=1))
        ]

        df = df.drop_duplicates()
        updated_chunk = updated_chunk.drop_duplicates()

    df = df.drop(
        columns=[
            value_col_item,
            scope_col_item,
            value_col_location,
            scope_col_location,
            value_col_channel,
            scope_col_channel,
            value_col_account,
            scope_col_account,
            value_col_region,
            scope_col_region,
            value_col_demand,
            scope_col_demand,
            value_col_PnL,
            scope_col_PnL,
        ]
    )

    df = process_assortment_system(df)

    # Convert Rule_x_col and Rule_y_col to strings, replacing NaN with empty strings
    df[Rule_x_col] = df[Rule_x_col].astype(str).replace("nan", "")
    df[Rule_y_col] = df[Rule_y_col].astype(str).replace("nan", "")

    # Create conditions for the logic
    conditions = [
        (df[Rule_x_col] == df[Rule_y_col]) & (df[Rule_x_col] != ""),  # Both are equal and not empty
        (df[Rule_x_col] == ""),  # Rule_x_col is empty
        (df[Rule_y_col] == ""),  # Rule_y_col is empty
        (df[Rule_x_col] == "") & (df[Rule_y_col] == ""),  # Both are empty
    ]

    # Corresponding choices for each condition
    choices = [
        df[Rule_x_col],  # If both are equal and not empty, take either one
        df[Rule_y_col],  # If Rule_x_col is empty, take Rule_y_col
        df[Rule_x_col],  # If Rule_y_col is empty, take Rule_x_col
        np.nan,  # If both are empty, result should be NaN
    ]

    # Use np.select to apply the logic
    df[rule_col] = np.select(conditions, choices, default=df[Rule_x_col] + ", " + df[Rule_y_col])

    # Drop the intermediate columns
    df = df.drop(columns=[Rule_x_col, Rule_y_col])

    return df, updated_chunk


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Input1,
    Input2,
    Input3,
    DimItem,
    StoreDates,
    DimLocation,
    DimChannel,
    DimAccount,
    DimRegion,
    DimPnL,
    DimDemandDomain,
):

    # Rename the column name for storeDates
    StoreDates = StoreDates.rename(columns={cluster_col: Location_goup_scope_col})

    # Drop columns from Input1
    Input1 = Input1.drop(columns=[object_type_col])

    # Drop columns from Input2
    Input2 = Input2.drop(columns=[object_type_col])
    # Replace empty strings with NaN first
    Input1[Rule_sequence_col] = Input1[Rule_sequence_col].replace("", pd.NA)

    # Fill NaN values with 1
    Input1[Rule_sequence_col] = Input1[Rule_sequence_col].fillna(1)
    Input2[Rule_sequence_col] = Input2[Rule_sequence_col].fillna(1)

    # List of columns to check and fill
    columns_to_check = [
        value_col_item,
        scope_col_item,
        Location_goup_scope_col,
        value_col_location,
        scope_col_location,
        value_col_channel,
        scope_col_channel,
        value_col_region,
        scope_col_region,
        value_col_account,
        scope_col_account,
        value_col_demand,
        scope_col_demand,
        value_col_PnL,
        scope_col_PnL,
    ]

    # Create missing columns with the placeholder value
    for col in columns_to_check:
        if col not in Input2.columns:
            Input2[col] = placeholder_value

    # Fill missing values in existing columns with the placeholder value
    Input2[columns_to_check] = Input2[columns_to_check].fillna(placeholder_value)

    Input2 = Input2.sort_values(
        by=[Rule_sequence_col, Rule_Created_Date_col], ascending=[False, False]
    )
    Input3 = Input3.sort_values(
        by=[Rule_sequence_col, Rule_Created_Date_col], ascending=[False, False]
    )
    Input3[value_col_location] = Input3[value_col_location].fillna("no value..")
    Input3[scope_col_location] = Input3[scope_col_location].fillna("no value..")
    Input3[Location_goup_scope_col] = Input3[Location_goup_scope_col].fillna("no value..")

    req_cols_assortment = [
        o9Constants.VERSION_NAME,
        o9Constants.ITEM,
        o9Constants.PLANNING_ACCOUNT,
        o9Constants.PLANNING_CHANNEL,
        o9Constants.PLANNING_REGION,
        o9Constants.PLANNING_PNL,
        o9Constants.LOCATION,
        o9Constants.PLANNING_DEMAND_DOMAIN,
        Assortment_system_col,
        Assortment_location_group_col,
        Assortment_rule_selected_col,
    ]

    req_cols_dates = [
        o9Constants.VERSION_NAME,
        o9Constants.PLANNING_ITEM,
        o9Constants.PLANNING_ACCOUNT,
        o9Constants.PLANNING_CHANNEL,
        o9Constants.PLANNING_REGION,
        o9Constants.PLANNING_PNL,
        o9Constants.PLANNING_LOCATION,
        o9Constants.PLANNING_DEMAND_DOMAIN,
        final_intro_date_col,
        final_disco_date_col,
    ]

    # Create an empty DataFrame with the specified columns
    merged_table_assortment = pd.DataFrame(columns=req_cols_assortment)
    merged_table_dates = pd.DataFrame(columns=req_cols_dates)

    if Input1.empty or len(Input1) == 0:
        logger.info("Input1 table is none/empty, Returning empty dataframe...")
        return merged_table_assortment, merged_table_dates

    if DimItem.empty or len(DimItem) == 0:
        logger.info("Item Hierarchy is not present, Returning empty dataframe...")
        return merged_table_assortment, merged_table_dates

    if DimLocation.empty or len(DimLocation) == 0:
        logger.info("Location Hierarchy is not present, Returning empty dataframe...")
        return merged_table_assortment, merged_table_dates

    # Process Input1
    processed_input1 = process_dataframe(
        Input1,
        DimItem,
        DimLocation,
        DimRegion,
        DimChannel,
        DimAccount,
        DimPnL,
        DimDemandDomain,
    )
    processed_input1 = processed_input1.drop_duplicates()

    # Group by 'Item.[L3]' and 'Location.[Location]' and retain the index of rows with the maximum 'DP Rule Sequence'
    final_Input1_chunks = []

    final_results = Parallel(n_jobs=1, verbose=1)(
        delayed(get_filtered_dataframe)(
            df=group,
        )
        for name, group in processed_input1.groupby(
            [Item_L3_col, o9Constants.LOCATION], observed=True
        )
    )
    final_Input1_chunks.append(pd.concat(final_results))

    final_df_input1 = pd.concat(final_Input1_chunks)

    # Split final_df_input1 into 50 chunks
    chunks_final = split_dataframe(final_df_input1, 1)

    final_Input2_chunks = []
    # If Input2 is empty, create a placeholder DataFrame with the required columns
    if Input2.empty:
        # Process the chunks from chunks_final without any groupings from Input2
        for chunk in chunks_final:
            df_final, updated_chunk = merge_and_conditionally_filter(
                df=Input2,
                value_col_item=value_col_item,
                value_col_location=value_col_location,
                value_col_channel=value_col_channel,
                value_col_region=value_col_region,
                value_col_account=value_col_account,
                value_col_PnL=value_col_PnL,
                value_col_demand=value_col_demand,
                level_col_item=level_col_item,
                level_col_location=level_col_location,
                level_col_channel=level_col_channel,
                level_col_region=level_col_region,
                level_col_account=level_col_account,
                level_col_PnL=level_col_PnL,
                level_col_demand=level_col_demand,
                scope_col_item=scope_col_item,
                scope_col_location=scope_col_location,
                scope_col_channel=scope_col_channel,
                scope_col_region=scope_col_region,
                scope_col_account=scope_col_account,
                scope_col_PnL=scope_col_PnL,
                scope_col_demand=scope_col_demand,
                StoreDates=StoreDates,
                chunk=chunk,
            )
            df_final = df_final.rename(
                columns={
                    Intro_date_col: final_intro_date_col,
                    Disco_date_col: final_disco_date_col,
                }
            )
            final_Input2_chunks.append(df_final)

    else:
        # If Input2 is not empty, proceed with grouping
        for chunk in chunks_final:
            updated_chunk = chunk
            for name, group in Input2.groupby(
                [
                    value_col_item,
                    value_col_location,
                    value_col_account,
                    value_col_channel,
                    value_col_demand,
                    value_col_PnL,
                    value_col_region,
                    Location_goup_scope_col,
                ],
                observed=True,
            ):
                df_final, updated_chunk = merge_and_conditionally_filter(
                    df=group,
                    value_col_item=value_col_item,
                    value_col_location=value_col_location,
                    value_col_channel=value_col_channel,
                    value_col_region=value_col_region,
                    value_col_account=value_col_account,
                    value_col_PnL=value_col_PnL,
                    value_col_demand=value_col_demand,
                    level_col_item=level_col_item,
                    level_col_location=level_col_location,
                    level_col_channel=level_col_channel,
                    level_col_region=level_col_region,
                    level_col_account=level_col_account,
                    level_col_PnL=level_col_PnL,
                    level_col_demand=level_col_demand,
                    scope_col_item=scope_col_item,
                    scope_col_location=scope_col_location,
                    scope_col_channel=scope_col_channel,
                    scope_col_region=scope_col_region,
                    scope_col_account=scope_col_account,
                    scope_col_PnL=scope_col_PnL,
                    scope_col_demand=scope_col_demand,
                    StoreDates=StoreDates,
                    chunk=updated_chunk,
                )
                # Update chunk for next iteration
                df_final = df_final.rename(
                    columns={
                        Intro_date_col: final_intro_date_col,
                        Disco_date_col: final_disco_date_col,
                    }
                )
                final_Input2_chunks.append(df_final)

            final_processed_chunk = process_intro_disco_date(updated_chunk, StoreDates)
            final_processed_chunk = final_processed_chunk.rename(
                columns={
                    Intro_date_col: final_intro_date_col,
                    Disco_date_col: final_disco_date_col,
                }
            )
            final_Input2_chunks.append(final_processed_chunk)

    final_Input3_chunks = []
    if Input3.empty:
        # Process the chunks from chunks_final without any groupings from Input3
        for chunk in final_Input2_chunks:
            df_final, temp_chunk = get_assortment_system(
                df=Input3,
                value_col_item=value_col_item,
                value_col_location=value_col_location,
                value_col_channel=value_col_channel,
                value_col_region=value_col_region,
                value_col_account=value_col_account,
                value_col_PnL=value_col_PnL,
                value_col_demand=value_col_demand,
                level_col_item=level_col_item,
                level_col_location=level_col_location,
                level_col_channel=level_col_channel,
                level_col_region=level_col_region,
                level_col_account=level_col_account,
                level_col_PnL=level_col_PnL,
                level_col_demand=level_col_demand,
                scope_col_item=scope_col_item,
                scope_col_location=scope_col_location,
                scope_col_channel=scope_col_channel,
                scope_col_region=scope_col_region,
                scope_col_account=scope_col_account,
                scope_col_PnL=scope_col_PnL,
                scope_col_demand=scope_col_demand,
                chunk=chunk,
            )
            df_final = df_final.drop(columns=[Exclude_flag_col])
            df_final = df_final.rename(
                columns={
                    Location_goup_scope_col: Assortment_location_group_col,
                    rule_col: Assortment_rule_selected_col,
                }
            )
            final_Input3_chunks.append(df_final)

    else:
        # If Input3 is not empty, proceed with grouping
        for chunk in final_Input2_chunks:

            temp_chunk = chunk
            for name, group in Input3.groupby(
                [
                    value_col_item,
                    value_col_location,
                    value_col_account,
                    value_col_channel,
                    value_col_demand,
                    value_col_PnL,
                    value_col_region,
                    Location_goup_scope_col,
                ],
                observed=True,
            ):
                df_final, temp_chunk = get_assortment_system(
                    df=group,
                    value_col_item=value_col_item,
                    value_col_location=value_col_location,
                    value_col_channel=value_col_channel,
                    value_col_region=value_col_region,
                    value_col_account=value_col_account,
                    value_col_PnL=value_col_PnL,
                    value_col_demand=value_col_demand,
                    level_col_item=level_col_item,
                    level_col_location=level_col_location,
                    level_col_channel=level_col_channel,
                    level_col_region=level_col_region,
                    level_col_account=level_col_account,
                    level_col_PnL=level_col_PnL,
                    level_col_demand=level_col_demand,
                    scope_col_item=scope_col_item,
                    scope_col_location=scope_col_location,
                    scope_col_channel=scope_col_channel,
                    scope_col_region=scope_col_region,
                    scope_col_account=scope_col_account,
                    scope_col_PnL=scope_col_PnL,
                    scope_col_demand=scope_col_demand,
                    chunk=temp_chunk,
                )
                df_final = df_final.drop(columns=[Exclude_flag_col])
                df_final = df_final.rename(
                    columns={
                        Location_goup_scope_col: Assortment_location_group_col,
                        rule_col: Assortment_rule_selected_col,
                    }
                )
                final_Input3_chunks.append(df_final)

            final_assorted_chunk = process_assortment_system(temp_chunk)
            final_assorted_chunk = final_assorted_chunk.drop(columns=[Exclude_flag_col])
            final_assorted_chunk = final_assorted_chunk.rename(
                columns={
                    Location_goup_scope_col: Assortment_location_group_col,
                    rule_col: Assortment_rule_selected_col,
                }
            )
            final_Input3_chunks.append(final_assorted_chunk)

    # Concatenate the final results
    merged_table_final = concat_to_dataframe(final_Input3_chunks)
    temp_table = merged_table_final[merged_table_final[Assortment_system_col] != 0]
    final_output_assortment = temp_table[req_cols_assortment].drop_duplicates()
    final_output_location = temp_table[req_cols_dates].drop_duplicates()
    return final_output_assortment, final_output_location

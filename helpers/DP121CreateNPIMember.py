"""This plugin careate NPI Members at NPI Level.

1. INPUT GATHERING
    - Get levels from LevelInput.
    - Get the levels for Item, Account, Channel, Region, PnL, Demand Domain, and Location.

2. PROCESSING FLOW
    a) Get respective columns from other dataFrames.
        - For each level, fetch the respective columns from the corresponding DataFrames (ItemMaster, AccountMaster, etc.).

    b) Compare with NPI DataFrames.
        - Compare the values in the fetched columns from ItemMaster, AccountMaster, ChannelMaster, etc. with the corresponding NPI DataFrames (NPIItemMaster, NPIAccountMaster, etc.) and find the missing values.

    c) Concate the missing_values dfs.
        - For each NPI_Master input dfs will compare and concat with the respective planning <dfs> and version.

3. OUTPUT
    - Save the Concated dfs to the final output.
"""

import logging

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


def rename_columns_for_entity(df, entity_type):
    """Rename the last two columns of a DataFrame based on the entity type."""
    if df is None or df.empty:
        logger.warning(f"DataFrame for {entity_type} is empty. Skipping renaming.")
        return df

    # Column name mapping
    column_mapping = {
        "Item": ["NPI Item Member Name", "NPI Item Member Display Name"],
        "Account": ["NPI Account Member Name", "NPI Account Member Display Name"],
        "Channel": ["NPI Channel Member Name", "NPI Channel Member Display Name"],
        "Region": ["NPI Region Member Name", "NPI Region Member Display Name"],
        "PnL": ["NPI PnL Member Name", "NPI PnL Member Display Name"],
        "Location": ["NPI Location Member Name", "NPI Location Member Display Name"],
        "DemandDomain": ["NPI Demand Domain Member Name", "NPI Demand Domain Member Display Name"],
    }

    # Ensure entity type exists in mapping
    if entity_type not in column_mapping:
        logger.warning(
            f"Entity type '{entity_type}' not found in column mapping. Skipping renaming."
        )
        return df

    # Rename only the last two columns
    df.rename(
        columns={
            df.columns[-2]: column_mapping[entity_type][0],
            df.columns[-1]: column_mapping[entity_type][1],
        },
        inplace=True,
    )

    return df


def concate_missing_df(
    master_df, missing_values_df, fetched_version, npi_master_df, planning_column, entity_type
):
    """Concate to the final results."""
    # Check if the master_df is valid
    if master_df is None or master_df.empty:
        logger.error(
            f"Master DataFrame is None or empty. Skipping operation for {planning_column}."
        )
        return master_df

    # Store original column name
    original_col_name = planning_column

    # Check if npi_master_df is empty
    if npi_master_df is None or npi_master_df.empty:
        logger.warning("NPI Master DataFrame is empty. Processing with column renaming.")

        # Step 1: Create the result_df by extracting the planning column
        if planning_column not in master_df.columns:
            logger.error(
                f"{planning_column} column is missing from the master DataFrame. Skipping operation."
            )
            return pd.DataFrame()

        # Copy the relevant column (planning col) from the master DataFrame
        result_df = master_df[[planning_column]].copy()

        # Rename the planning column to just entity_type
        result_df = result_df.rename(columns={planning_column: entity_type})

        # Add the version column (Version.[Version Name]) with fetched_version as the repeating value
        result_df["Version.[Version Name]"] = fetched_version

        # Check if there is any missing values
        if missing_values_df is None or missing_values_df.empty:
            logger.warning("Missing values DataFrame is None or empty. Skipping concatenation.")
        else:
            # Concatenate the missing_values_df to result_df
            missing_values_to_concat = missing_values_df.reset_index(drop=True)
            result_df = pd.concat([result_df, missing_values_to_concat], axis=1, ignore_index=False)

            # Remove rows from result_df that don't have values in the missing_values_df columns
            result_df = result_df.dropna(subset=missing_values_df.columns, how="any")

        # Rename the last two columns dynamically
        result_df = rename_columns_for_entity(result_df, entity_type)

        # Rename back to original column name before returning
        result_df = result_df.rename(columns={entity_type: original_col_name})

        return result_df

    # Normal case when npi_master_df is not empty
    # Check if there is any missing values
    if missing_values_df is None or missing_values_df.empty:
        logger.warning("Missing values DataFrame is None or empty. Skipping concatenation.")
        return missing_values_df

    # Step 1: Create the result_df by extracting the planning column (Item.[Planning Item] from ItemMaster)
    if planning_column not in master_df.columns:
        logger.error(
            f"{planning_column} column is missing from the master DataFrame. Skipping operation."
        )
        return pd.DataFrame()

    # Copy the relevant column (planning col) from the master DataFrame
    result_df = master_df[[planning_column]].copy()

    # Add the version column (Version.[Version Name]) with fetched_version as the repeating value
    result_df["Version.[Version Name]"] = fetched_version

    # Concatenate the missing_values_df to result_df
    missing_values_to_concat = missing_values_df.reset_index(drop=True)
    result_df = pd.concat([result_df, missing_values_to_concat], axis=1, ignore_index=False)

    # Remove rows from result_df that don't have values in the missing_values_df columns
    # Check for rows where the values in missing_values_df are not null
    result_df = result_df.dropna(subset=missing_values_df.columns, how="any")
    # Rename the last two columns dynamically
    result_df = rename_columns_for_entity(result_df, entity_type)

    # Return the final result_df
    return result_df


def find_missing_values(filtered_df, npi_df, empty_resptive_df):
    """Find the missing values or the data not present in the given input."""
    if filtered_df is None or filtered_df.empty:
        return filtered_df

    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    if npi_df is None or npi_df.empty:
        missing_values = filtered_df.copy()
        return missing_values

    # Get the first column name dynamically for both DataFrames
    filter_column = filtered_df.columns[0]  # First column in master_df
    npi_column = npi_df.columns[0]

    # Find values that are in the filtered_df but not in the npi_df
    missing_values = filtered_df[~filtered_df[filter_column].isin(npi_df[npi_column])]

    if missing_values.empty:
        return empty_resptive_df

    missing_values.rename(
        columns={
            missing_values.columns[-2]: "Member Name",
            missing_values.columns[-1]: "Member Display Name",
        },
        inplace=True,
    )

    return missing_values


def create_level_display_df(df, level, prefix):
    """Create the variables for the user defined levels."""
    # Step 1: Check if the DataFrame is None or empty
    if df is None or df.empty:
        logger.error(f"The DataFrame for {prefix} is None or empty. Returning empty df.")
        return df

    # Step 2: Construct the column name for the level and the display name
    level_column = f"{prefix}.[{level}]"
    display_column = f"{prefix}.[{level}$DisplayName]"

    # Step 3: Check if both columns exist in the DataFrame
    if level_column in df.columns and display_column in df.columns:
        # Step 4: Return the DataFrame with the relevant columns
        return df[[level_column, display_column]]
    else:
        # Step 5: Print a message if the columns do not exist
        logger.error(f"Columns '{level_column}' or '{display_column}' not found in the DataFrame.")
        return pd.DataFrame()


col_mapping = {
    "NPIItem": str,
    "NPIAccount": str,
    "NPIChannel": str,
    "NPIRegion": str,
    "NPIPnL": str,
    "NPIDemandDomain": str,
    "NPILocation": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Data
    LevelInput,
    # Master data
    AccountMaster,
    ChannelMaster,
    RegionMaster,
    PnLMaster,
    DemandDomainMaster,
    LocationMaster,
    ItemMaster,
    NPIItemMaster,
    NPIAccountMaster,
    NPIRegionMaster,
    NPIChannelMaster,
    NPIPnLMaster,
    NPILocationMaster,
    NPIDemandDomainMaster,
    # Others
    df_keys,
):
    """DP121CreateNPIMembers starts."""
    plugin_name = "DP121CreateNPIMember"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_account_col = "Account.[Planning Account]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demanddomain_col = "Demand Domain.[Planning Demand Domain]"
    pl_region_col = "Region.[Planning Region]"
    pl_location_col = "Location.[Planning Location]"

    # entities
    Item = "Item"
    PnL = "PnL"
    Account = "Account"
    Channel = "Channel"
    Region = "Region"
    Location = "Location"
    DemandDomain = "DemandDomain"

    version_level_col = LevelInput[version_col]
    account_level_col = LevelInput["Global NPI Account Level"]
    channel_level_col = LevelInput["Global NPI Channel Level"]
    demand_domain_level_col = LevelInput["Global NPI Demand Domain Level"]
    item_level_col = LevelInput["Global NPI Item Level"]
    location_level_col = LevelInput["Global NPI Location Level"]
    pnl_level_col = LevelInput["Global NPI PnL Level"]
    region_level_col = LevelInput["Global NPI Region Level"]

    npi_item_name = "NPI Item Member Name"
    npi_item_disp_name = "NPI Item Member Display Name"
    npi_account_name = "NPI Account Member Name"
    npi_account_disp_name = "NPI Account Member Display Name"
    npi_channel_name = "NPI Channel Member Name"
    npi_channel_disp_name = "NPI Channel Member Display Name"
    npi_pnl_name = "NPI PnL Member Name"
    npi_pnl_disp_name = "NPI PnL Member Display Name"
    npi_location_name = "NPI Location Member Name"
    npi_location_disp_name = "NPI Location Member Display Name"
    npi_region_name = "NPI Region Member Name"
    npi_region_disp_name = "NPI Region Member Display Name"
    npi_demand_domain_name = "NPI Demand Domain Member Name"
    npi_demand_domain_disp_name = "NPI Demand Domain Member Display Name"

    # output columns
    cols_required_in_npi_item = [version_col, pl_item_col, npi_item_name, npi_item_disp_name]

    cols_required_in_npi_account = [
        version_col,
        pl_account_col,
        npi_account_name,
        npi_account_disp_name,
    ]

    cols_required_in_npi_channel = [
        version_col,
        pl_channel_col,
        npi_channel_name,
        npi_channel_disp_name,
    ]

    cols_required_in_npi_location = [
        version_col,
        pl_location_col,
        npi_location_name,
        npi_location_disp_name,
    ]

    cols_required_in_npi_region = [
        version_col,
        pl_region_col,
        npi_region_name,
        npi_region_disp_name,
    ]

    cols_required_in_npi_demanddomain = [
        version_col,
        pl_demanddomain_col,
        npi_demand_domain_name,
        npi_demand_domain_disp_name,
    ]

    cols_required_in_npi_pnl = [version_col, pl_pnl_col, npi_pnl_name, npi_pnl_disp_name]

    # Output empty_dfs
    NPIItem = pd.DataFrame(columns=cols_required_in_npi_item)
    NPIAccount = pd.DataFrame(columns=cols_required_in_npi_account)
    NPIChannel = pd.DataFrame(columns=cols_required_in_npi_channel)
    NPIPnL = pd.DataFrame(columns=cols_required_in_npi_pnl)
    NPIRegion = pd.DataFrame(columns=cols_required_in_npi_region)
    NPILocation = pd.DataFrame(columns=cols_required_in_npi_location)
    NPIDemandDomain = pd.DataFrame(columns=cols_required_in_npi_demanddomain)

    # Combined empty output dataframes
    combined_output_dataframes = [
        NPIItem,
        NPIAccount,
        NPIChannel,
        NPIDemandDomain,
        NPIPnL,
        NPILocation,
        NPIRegion,
    ]

    try:
        # condition to check both the input are not empty
        if LevelInput.empty:
            logger.warning("LevelInput is empty for slice: {} ...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            return combined_output_dataframes

        if LevelInput.shape[0] > 1:
            logger.error("LevelInput has more than 1 values: {} ...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            return combined_output_dataframes

        # Fetch Levels from LevelInput
        # Fetching the level defined
        fetched_version = version_level_col[0]

        item_level = item_level_col[0]
        account_level = account_level_col[0]
        channel_level = channel_level_col[0]
        region_level = region_level_col[0]
        pnl_level = pnl_level_col[0]
        demand_domain_level = demand_domain_level_col[0]
        location_level = location_level_col[0]

        # Fetching the values of the level form  their respective master dataframes i.e, ItemMaster, AccountMaster, PnLMaster
        item_level_display_df = create_level_display_df(ItemMaster, item_level, "Item")
        account_level_display_df = create_level_display_df(AccountMaster, account_level, "Account")
        Channel_level_display_df = create_level_display_df(ChannelMaster, channel_level, "Channel")
        region_level_display_df = create_level_display_df(RegionMaster, region_level, "Region")
        pnl_level_display_df = create_level_display_df(PnLMaster, pnl_level, "PnL")
        demanddomain_level_display_df = create_level_display_df(
            DemandDomainMaster, demand_domain_level, "Demand Domain"
        )
        location_level_display_df = create_level_display_df(
            LocationMaster, location_level, "Location"
        )

        # Finding the missing values
        missing_values_Item_df = find_missing_values(item_level_display_df, NPIItemMaster, NPIItem)
        missing_values_account_df = find_missing_values(
            account_level_display_df, NPIAccountMaster, NPIAccount
        )
        missing_values_channel_df = find_missing_values(
            Channel_level_display_df, NPIChannelMaster, NPIChannel
        )
        missing_values_pnl_df = find_missing_values(pnl_level_display_df, NPIPnLMaster, NPIPnL)
        missing_values_location_df = find_missing_values(
            location_level_display_df, NPILocationMaster, NPILocation
        )
        missing_values_region_df = find_missing_values(
            region_level_display_df, NPIRegionMaster, NPIRegion
        )
        missing_values_demanddomain_df = find_missing_values(
            demanddomain_level_display_df, NPIDemandDomainMaster, NPIDemandDomain
        )

        # Preparig for the output
        final_output_Item_df = concate_missing_df(
            ItemMaster, missing_values_Item_df, fetched_version, NPIItemMaster, pl_item_col, Item
        )
        final_output_Account_df = concate_missing_df(
            AccountMaster,
            missing_values_account_df,
            fetched_version,
            NPIAccountMaster,
            pl_account_col,
            Account,
        )
        final_output_Channel_df = concate_missing_df(
            ChannelMaster,
            missing_values_channel_df,
            fetched_version,
            NPIChannelMaster,
            pl_channel_col,
            Channel,
        )
        final_output_pnl_df = concate_missing_df(
            PnLMaster, missing_values_pnl_df, fetched_version, NPIPnLMaster, pl_pnl_col, PnL
        )
        final_output_demanddomain_df = concate_missing_df(
            DemandDomainMaster,
            missing_values_demanddomain_df,
            fetched_version,
            NPIDemandDomainMaster,
            pl_demanddomain_col,
            DemandDomain,
        )
        final_output_location_df = concate_missing_df(
            LocationMaster,
            missing_values_location_df,
            fetched_version,
            NPILocationMaster,
            pl_location_col,
            Location,
        )
        final_output_region_df = concate_missing_df(
            RegionMaster,
            missing_values_region_df,
            fetched_version,
            NPIRegionMaster,
            pl_region_col,
            Region,
        )

        # Assigning the values to the output dfs
        NPIItem = final_output_Item_df
        NPIAccount = final_output_Account_df
        NPIChannel = final_output_Channel_df
        NPIDemandDomain = final_output_demanddomain_df
        NPILocation = final_output_location_df
        NPIPnL = final_output_pnl_df
        NPIRegion = final_output_region_df

        NPIItem = NPIItem[cols_required_in_npi_item]
        NPIAccount = NPIAccount[cols_required_in_npi_account]
        NPIChannel = NPIChannel[cols_required_in_npi_channel]
        NPIPnL = NPIPnL[cols_required_in_npi_pnl]
        NPIRegion = NPIRegion[cols_required_in_npi_region]
        NPILocation = NPILocation[cols_required_in_npi_location]
        NPIDemandDomain = NPIDemandDomain[cols_required_in_npi_demanddomain]

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return NPIItem, NPIAccount, NPIChannel, NPIRegion, NPIPnL, NPIDemandDomain, NPILocation

    return NPIItem, NPIAccount, NPIChannel, NPIRegion, NPIPnL, NPIDemandDomain, NPILocation

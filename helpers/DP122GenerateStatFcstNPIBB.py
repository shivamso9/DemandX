"""This is DP122GenerateStatFcstNPIBB plugin."""

# Library imports
import logging
from functools import reduce
from typing import Any, Dict, Optional

import pandas as pd
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


def user_selected_level(
    InitiativeLevel_required,
    initiative_col,
    npi_item_level_col,
    npi_account_level_col,
    npi_channel_level_col,
    npi_region_level_col,
    npi_pnl_level_col,
    npi_dd_level_col,
    npi_location_level_col,
):
    """Return formatted columns of user selected level."""
    # Selecting the required columns
    InitiativeLevel_required = InitiativeLevel_required[
        [
            npi_item_level_col,
            npi_account_level_col,
            npi_channel_level_col,
            npi_region_level_col,
            npi_pnl_level_col,
            npi_dd_level_col,
            npi_location_level_col,
        ]
    ].drop_duplicates()

    for col in InitiativeLevel_required.columns:
        if "Global NPI" in col and "Level" in col:
            # Extracting the attribute name dynamically
            attribute = col.replace("Global NPI", "").replace("Level", "").strip()

            # Formatting the column values with the required format
            InitiativeLevel_required[col] = attribute + ".[" + InitiativeLevel_required[col] + "]"

    return list(InitiativeLevel_required.iloc[0])


def merge_two(df1, df2_key):
    """Merge two dfs."""
    key, df2 = df2_key
    df2 = df2.loc[:, ~df2.columns.duplicated()]
    # if user levels are same, that means, the req column is already present
    if len(df2.columns) == 1:
        return df1
    return pd.merge(df1, df2, on=key, how="left")


col_mapping = {"Stat Fcst NPI BB L0": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    # Data
    NPILevels: pd.DataFrame = None,
    StatFcstNPIBB: pd.DataFrame = None,
    # Master data
    ItemMaster=None,
    AccountMaster=None,
    ChannelMaster=None,
    RegionMaster=None,
    PnLMaster=None,
    DemandDomainMaster=None,
    LocationMaster=None,
    # Others
    df_keys: Optional[Dict[Any, Any]] = None,
):
    """Dp122GeerateStatFcstNPIBB starts."""
    plugin_name = "DP122GenerateStatFcstNPIBB"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    pl_item_col = "Item.[Planning Item]"
    pl_loc_col = "Location.[Planning Location]"
    initiative_col = "Initiative.[Initiative]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"
    planning_location_col = "Location.[Planning Location]"
    npi_item = "Item.[NPI Item]"
    npi_account = "Account.[NPI Account]"
    npi_channel = "Channel.[NPI Channel]"
    npi_region = "Region.[NPI Region]"
    npi_pnl = "PnL.[NPI PnL]"
    npi_loc = "Location.[NPI Location]"
    npi_dem_domain = "Demand Domain.[NPI Demand Domain]"
    npi_item_level_col = "Global NPI Item Level"
    npi_account_level_col = "Global NPI Account Level"
    npi_channel_level_col = "Global NPI Channel Level"
    npi_region_level_col = "Global NPI Region Level"
    npi_pnl_level_col = "Global NPI PnL Level"
    npi_location_level_col = "Global NPI Location Level"
    npi_dd_level_col = "Global NPI Demand Domain Level"
    stat_fcst_npi_bb = "Stat Fcst NPI BB"
    partial_week_col = "Time.[Partial Week]"

    # Output Measure
    statfcstL0 = "Stat Fcst NPI BB L0"

    # outputdf
    cols_required_in_output = [
        version_col,
        npi_item,
        npi_account,
        npi_channel,
        npi_region,
        npi_pnl,
        npi_dem_domain,
        npi_loc,
        partial_week_col,
        statfcstL0,
    ]

    all_results = []

    try:
        if len(NPILevels[version_col].unique()) >= 2:
            logger.error(
                "The plugin does not support multiple versions. The user selected versions are: {}".format(
                    NPILevels[version_col].unique()
                )
            )

        # iterating for all the rows in NPILevels
        for index, row in NPILevels.iterrows():
            InitiativeLevel_required = pd.DataFrame([row])  # Convert row to DataFrame

            if InitiativeLevel_required.isnull().any().any():
                logger.warning(
                    f"Skipping row {index} due to missing values:\n{InitiativeLevel_required}"
                )
                continue

            # got the cols for furter fetching
            (
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_dd_level_col,
                user_loc_level_col,
            ) = user_selected_level(
                InitiativeLevel_required,
                initiative_col,
                npi_item_level_col,
                npi_account_level_col,
                npi_channel_level_col,
                npi_region_level_col,
                npi_pnl_level_col,
                npi_dd_level_col,
                npi_location_level_col,
            )

            # Remove duplicate cols
            ItemMaster_req = ItemMaster[[pl_item_col, user_item_level_col]].drop_duplicates()
            AccountMaster_req = AccountMaster[
                [planning_account_col, user_account_level_col]
            ].drop_duplicates()
            ChannelMaster_req = ChannelMaster[
                [planning_channel_col, user_channel_level_col]
            ].drop_duplicates()
            RegionMaster_req = RegionMaster[
                [planning_region_col, user_region_level_col]
            ].drop_duplicates()
            PnLMaster_req = PnLMaster[[planning_pnl_col, user_pnl_level_col]].drop_duplicates()
            DemandDomainMaster_req = DemandDomainMaster[
                [planning_demand_domain_col, user_dd_level_col]
            ].drop_duplicates()
            LocationMaster_req = LocationMaster[
                [planning_location_col, user_loc_level_col]
            ].drop_duplicates()

            # Merge with the assortment to get all the required columns
            Master_list = [
                (pl_item_col, ItemMaster_req),
                (planning_account_col, AccountMaster_req),
                (planning_channel_col, ChannelMaster_req),
                (planning_region_col, RegionMaster_req),
                (planning_pnl_col, PnLMaster_req),
                (planning_demand_domain_col, DemandDomainMaster_req),
                (pl_loc_col, LocationMaster_req),
            ]

            # removing unwanter cols from InitiaiveLevel_required for further processing
            unwanted_columns = [version_col, data_object_col]
            InitiativeLevel_required = InitiativeLevel_required.drop(
                columns=unwanted_columns, errors="ignore"
            )

            StatFcst_Pllevel = reduce(merge_two, Master_list, StatFcstNPIBB)

            # assuring all cols are present
            cols_required = [
                version_col,
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_dd_level_col,
                user_loc_level_col,
                partial_week_col,
                stat_fcst_npi_bb,
            ]

            missing_cols = [col for col in cols_required if col not in StatFcst_Pllevel.columns]
            if missing_cols:
                logger.error(f"Missing columns in StatFcst_Pllevel: {missing_cols}")
                raise KeyError(f"Required columns missing: {missing_cols}")
            StatFcst_Pllevel_filtered = StatFcst_Pllevel[cols_required]

            agg_cols = [
                version_col,
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_dd_level_col,
                user_loc_level_col,
                partial_week_col,
            ]

            # Aggregating the cols
            Statfcst_final_agg = StatFcst_Pllevel_filtered.groupby(agg_cols, as_index=False)[
                stat_fcst_npi_bb
            ].sum()
            output_cols = [
                version_col,
                npi_item,
                npi_account,
                npi_channel,
                npi_region,
                npi_pnl,
                npi_dem_domain,
                npi_loc,
                partial_week_col,
                statfcstL0,
            ]

            rename_mapping = dict(zip(cols_required, output_cols))
            Statfcst_final_agg.rename(columns=rename_mapping, inplace=True)
            all_results.append(Statfcst_final_agg)

        # Combine all processed levels into one DataFrame
        final_output = (
            pd.concat(all_results, ignore_index=True)
            if all_results
            else pd.DataFrame(columns=cols_required_in_output)
        )
        StatFcstNPIBBbyLevel = final_output

    except Exception as e:
        logger.exception(
            "Exception {} for slice: {}. Returning empty dataframe as output ...".format(e, df_keys)
        )
        # Define an empty DataFrame with the required columns in case of an error
        StatFcstNPIBBbyLevel = pd.DataFrame(columns=cols_required_in_output)

    # Ensure that StatFcstNPIBBbyLevel is returned at the end
    return StatFcstNPIBBbyLevel

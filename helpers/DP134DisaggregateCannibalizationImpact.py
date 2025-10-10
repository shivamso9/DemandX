"""Disaggregate Cannibalization Impact Plugin for Flexible NPI."""

# Library imports
import logging
from functools import reduce
from typing import Optional

import numpy as np
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

col_mapping = {
    "Planning Level Cannibalization Impact": float,
}


def merge_two(df1, df2_w_key):
    """Merge two dataframes."""
    key, df2 = df2_w_key
    return pd.merge(df1, df2, on=[key], how="left")


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Data
    SelectedInitiativeLevel: pd.DataFrame = None,
    InitiativeLevels: pd.DataFrame = None,
    CannibImpact: pd.DataFrame = None,
    CannibImpactPlanningLevel: pd.DataFrame = None,
    Splits: pd.DataFrame = None,
    # Master data
    ItemMaster: pd.DataFrame = None,
    AccountMaster: pd.DataFrame = None,
    ChannelMaster: pd.DataFrame = None,
    RegionMaster: pd.DataFrame = None,
    PnLMaster: pd.DataFrame = None,
    DemandDomainMaster: pd.DataFrame = None,
    LocationMaster: pd.DataFrame = None,
    # Others
    df_keys: Optional[dict] = None,
):
    """Entry point of the script."""
    plugin_name = "DP134DisaggregateCannibalizationImpact"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    partial_week_col = "Time.[Partial Week]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"
    planning_location_col = "Location.[Planning Location]"

    npi_item_col = "Item.[NPI Item]"
    npi_account_col = "Account.[NPI Account]"
    npi_channel_col = "Channel.[NPI Channel]"
    npi_region_col = "Region.[NPI Region]"
    npi_pnl_col = "PnL.[NPI PnL]"
    npi_location_col = "Location.[NPI Location]"
    npi_demand_domain_col = "Demand Domain.[NPI Demand Domain]"

    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_location_level_col = "NPI Location Level"
    npi_dd_level_col = "NPI Demand Domain Level"

    cannib_final_split_perc_col = "Cannib Final Split %"

    pl_level_cannib_impact_col = "Planning Level Cannibalization Impact"
    cannib_impact_final_l0_col = "Cannibalization Impact Final L0"

    pl_cannibImpact = pl_level_cannib_impact_col

    # output columns
    cols_required_in_output = [
        version_col,
        initiative_col,
        data_object_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        partial_week_col,
        pl_cannibImpact,
    ]

    # Empty dataframe
    PlanningImpact = pd.DataFrame(cols_required_in_output)

    try:

        if len(SelectedInitiativeLevel) == 0:
            raise ValueError(f"'SelectedInitiativeLevel' can't be empty for slice: {df_keys}")

        if len(InitiativeLevels) == 0:
            raise ValueError(f"'InitiativeLevels' can't be empty for slice: {df_keys}")

        if SelectedInitiativeLevel[version_col].nunique() > 1:
            raise ValueError(f"Multiple versions not supported for slice: {df_keys}")

        if len(CannibImpact) == 0:
            raise ValueError(f"'NPIFcst' is empty for slice: {df_keys}")

        if len(Splits) == 0:
            raise ValueError(f"'Splits' can't be empty for slice: {df_keys}")

        # Values
        version = SelectedInitiativeLevel[version_col].values[0]

        # Drop versions
        SelectedInitiativeLevel.drop(columns=[version_col], inplace=True)
        InitiativeLevels.drop(columns=[version_col], inplace=True)
        CannibImpact.drop(columns=[version_col], inplace=True)
        Splits.drop(columns=[version_col], inplace=True)

        # filtering initiative level from selected initiative level for the cols {data_object_col,initiatie_col}
        InitiativeLevels = pd.merge(
            InitiativeLevels,
            SelectedInitiativeLevel[[data_object_col, initiative_col]],
            on=[data_object_col, initiative_col],
            how="inner",
        )

        # Get all the master data
        master_list = [
            (pl_item_col, ItemMaster),
            (planning_account_col, AccountMaster),
            (planning_channel_col, ChannelMaster),
            (planning_region_col, RegionMaster),
            (planning_pnl_col, PnLMaster),
            (planning_demand_domain_col, DemandDomainMaster),
            (planning_location_col, LocationMaster),
        ]

        Splits_w_masters = reduce(merge_two, master_list, Splits)

        def format_level(dimension, level):
            """Doing Function to format dimension levels."""
            if level in [None, "na", "NA", "None", "", " ", np.nan]:
                return None
            return f"{dimension}.[{str(level).strip()}]"

        # Initialize list to store results
        final_impact_list = []

        # Process each initiative level
        for _, the_selectedinitiative in InitiativeLevels.iterrows():
            # 1. Extract current iteration data
            the_selectedinitiative_df = the_selectedinitiative.to_frame().T

            # 2. Format level information
            item_level = format_level("Item", the_selectedinitiative[npi_item_level_col])
            account_level = format_level("Account", the_selectedinitiative[npi_account_level_col])
            channel_level = format_level("Channel", the_selectedinitiative[npi_channel_level_col])
            region_level = format_level("Region", the_selectedinitiative[npi_region_level_col])
            pnl_level = format_level("PnL", the_selectedinitiative[npi_pnl_level_col])
            dd_level = format_level("Demand Domain", the_selectedinitiative[npi_dd_level_col])
            location_level = format_level(
                "Location", the_selectedinitiative[npi_location_level_col]
            )

            # Skip if any required level is missing
            if any(
                level is None
                for level in [
                    item_level,
                    account_level,
                    channel_level,
                    region_level,
                    pnl_level,
                    dd_level,
                    location_level,
                ]
            ):
                logger.warning(f"One or more levels are empty for slice: {df_keys}")
                continue

            # 3. Get impact data for current initiative
            the_impact = pd.merge(
                CannibImpact,
                the_selectedinitiative_df[[data_object_col, initiative_col]],
                on=[data_object_col, initiative_col],
                how="inner",
            )

            if len(the_impact) == 0:
                logger.warning(f"No impact data found for slice: {df_keys}")
                continue

            # 4. Rename impact columns to match level columns
            column_mapping = {
                npi_item_col: item_level,
                npi_account_col: account_level,
                npi_channel_col: channel_level,
                npi_region_col: region_level,
                npi_pnl_col: pnl_level,
                npi_location_col: location_level,
                npi_demand_domain_col: dd_level,
            }
            the_impact = the_impact.rename(columns=column_mapping)

            # 5. Get and process splits
            # Define required columns for splits
            req_cols = [
                initiative_col,
                data_object_col,
                item_level,
                account_level,
                channel_level,
                region_level,
                pnl_level,
                location_level,
                dd_level,
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
                cannib_final_split_perc_col,
            ]

            # Get the relevant splits
            keys = [
                initiative_col,
                data_object_col,
                item_level,
                account_level,
                channel_level,
                region_level,
                pnl_level,
                location_level,
                dd_level,
            ]
            req_cols = keys + [
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
                cannib_final_split_perc_col,
            ]
            req_cols = list(set(req_cols))

            # Filter relevant splits
            relevant_splits = Splits_w_masters[req_cols].drop_duplicates()
            relevant_splits = pd.merge(
                relevant_splits, the_impact[keys].drop_duplicates(), on=keys, how="inner"
            )

            # 6. Normalize splits within groups
            group_cols = [
                initiative_col,
                data_object_col,
                item_level,
                account_level,
                channel_level,
                region_level,
                pnl_level,
                location_level,
                dd_level,
            ]
            group_sums = relevant_splits.groupby(group_cols)[cannib_final_split_perc_col].transform(
                "sum"
            )

            normalized_splits = relevant_splits.copy()
            # Only normalize where sum is not zero to avoid division by zero
            mask = group_sums != 0
            normalized_splits.loc[mask, cannib_final_split_perc_col] = (
                normalized_splits.loc[mask, cannib_final_split_perc_col] / group_sums[mask]
            )
            # For groups with sum=0, keep the original zeros
            normalized_splits.loc[~mask, cannib_final_split_perc_col] = 0

            # 7. Calculate final impact
            disaggregated_impact = pd.merge(
                normalized_splits,
                the_impact,
                on=[
                    initiative_col,
                    data_object_col,
                    item_level,
                    account_level,
                    channel_level,
                    region_level,
                    pnl_level,
                    location_level,
                    dd_level,
                ],
                how="inner",
            )

            disaggregated_impact[pl_level_cannib_impact_col] = (
                disaggregated_impact[cannib_final_split_perc_col]
                * disaggregated_impact[cannib_impact_final_l0_col]
            )

            # Add version and select required columns
            disaggregated_impact[version_col] = version
            disaggregated_impact = disaggregated_impact[cols_required_in_output]
            final_impact_list.append(disaggregated_impact)

        # 8. Combine all results
        if final_impact_list:
            PlanningImpact = pd.concat(final_impact_list, ignore_index=True)
        else:
            PlanningImpact = pd.DataFrame(columns=cols_required_in_output)

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return PlanningImpact

    return PlanningImpact

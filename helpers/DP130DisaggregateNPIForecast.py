"""Disaggregate NPI Forecast for Flexible NPI.

Pseudocode:
    Version: 0.0.0
    --------------------
        - Calculate Npi Fcst at planning level
            - Disaggregate the NPI Fcst for 'SelectedInitiativeLevel' input
            - For the above object and inititave
                - find levels using 'InitiativeLevels' input and,
                - find the intersection for which fcst would be disaggregated
            - Iterate for each 'NPIAssociation' input record
            - Get the NPI levels in 'InitiativeLevels' input using 'SelectedInitiativeLevel' inputs
            - Based on 'Splits' input % disaggregate 'NPIFcst' input for the 'NPIAssociation'
        - Calculate Npi Fcst at planning level L1
            - For each initiative in 'SelectedInitiativeLevel' input
            - Get the 'NPIPlanningLevelFcst' input for the initiative
            - Get the 'PlanningFcstOutput' for the initiative
            - Get the 'NPIPlanningLevelFcstL1' input for the initiative and null it out
            - Merge 'NPIPlanningLevelFcst' and 'PlanningFcstOutput' for the initiative
            - Calculate latest level data in the merged data which is the 'PlanningFcstL1Output'
            - Calculate Eligible level output which is the mapping of initiative and latest level sequence
"""

import logging
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
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


def merge_two(df1, df2_w_key):
    """Merge two dataframes."""
    key, df2 = df2_w_key
    return pd.merge(df1, df2, on=[key], how="left")


col_mapping = {
    "Planning Level NPI Fcst": float,
    "Planning Level NPI Fcst L1": float,
    "NPI Initiative Latest Level Sequence": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Data
    SelectedInitiativeLevel: pd.DataFrame = None,
    InitiativeLevels: pd.DataFrame = None,
    NPIAssociation: pd.DataFrame = None,
    NPIFcst: pd.DataFrame = None,
    NPIPlanningLevelFcst: pd.DataFrame = None,
    NPIPlanningLevelFcstL1: pd.DataFrame = None,
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
    plugin_name = "DP130DisaggregateNPIForecast"
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

    npi_association_l0_col = "NPI Association L0"
    npi_level_seq_l1_col = "NPI Level Sequence L1"
    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_location_level_col = "NPI Location Level"
    npi_dd_level_col = "NPI Demand Domain Level"
    npi_fcst_disagg_flag_col = "NPI Forecast Disaggregation Flag L1"
    npi_final_split_perc_col = "NPI Final Split %"
    npi_fcst_final_l0_col = "NPI Fcst Final L0"
    pl_level_npi_fcst_col = "Planning Level NPI Fcst"
    pl_level_npi_fcst_l1_col = "Planning Level NPI Fcst L1"
    npi_latest_level_seq_col = "NPI Initiative Latest Level Sequence"

    # output columns
    cols_required_in_output_pl_fcst = [
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
        pl_level_npi_fcst_col,
    ]

    cols_required_in_output_pl_fcst_l1 = [
        version_col,
        initiative_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        partial_week_col,
        pl_level_npi_fcst_l1_col,
    ]

    cols_required_in_output_eligible_levels = [
        version_col,
        initiative_col,
        npi_latest_level_seq_col,
    ]

    # Empty dataframe
    PlanningFcstOutput = pd.DataFrame(columns=cols_required_in_output_pl_fcst)
    PlanningFcstL1Output = pd.DataFrame(columns=cols_required_in_output_pl_fcst_l1)
    EligibleLevelsOutput = pd.DataFrame(columns=cols_required_in_output_eligible_levels)

    try:

        SelectedInitiativeLevel[npi_fcst_disagg_flag_col] = pd.to_numeric(
            SelectedInitiativeLevel[npi_fcst_disagg_flag_col], downcast="integer", errors="coerce"
        )
        SelectedInitiativeLevel = SelectedInitiativeLevel[
            SelectedInitiativeLevel[npi_fcst_disagg_flag_col] >= 1
        ]
        if len(SelectedInitiativeLevel) == 0:
            logger.error(f"'SelectedInitiativeLevel' can't be empty for slice: {df_keys}")
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        if len(InitiativeLevels) == 0:
            logger.error(f"'InitiativeLevels' can't be empty for slice: {df_keys}")
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        NPIAssociation[npi_association_l0_col] = pd.to_numeric(
            NPIAssociation[npi_association_l0_col], downcast="integer", errors="coerce"
        )
        NPIAssociation = NPIAssociation[NPIAssociation[npi_association_l0_col] >= 1]
        if len(NPIAssociation) == 0:
            logger.error(
                f"'NPIAssociation' does not contains active intersection for slice: {df_keys}..."
            )
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        if SelectedInitiativeLevel[version_col].nunique() > 1:
            logger.error(f"Multiple versions not supported for slice: {df_keys}")
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        if len(NPIFcst) == 0:
            logger.error(f"'NPIFcst' is empty for slice: {df_keys}")
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        if len(Splits) == 0:
            logger.error(f"'Splits' can't be empty for slice: {df_keys}")
            logger.error("Returning empty output ...")
            return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

        # Values
        version = SelectedInitiativeLevel[version_col].values[0]

        # Drop versions
        SelectedInitiativeLevel.drop(columns=[version_col], inplace=True)
        InitiativeLevels.drop(columns=[version_col], inplace=True)
        NPIAssociation.drop(columns=[version_col], inplace=True)
        NPIFcst.drop(columns=[version_col], inplace=True)
        Splits.drop(columns=[version_col], inplace=True)

        # Filter NPI Association for the selected initiative and level
        NPIAssociation = pd.merge(
            NPIAssociation,
            SelectedInitiativeLevel,
            on=[initiative_col, data_object_col],
            how="inner",
        )

        # Get the levels
        NPIAssociation = pd.merge(
            NPIAssociation, InitiativeLevels, on=[initiative_col, data_object_col], how="left"
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

        # ---------------------> Calculate NPI Forecast at Planning level
        PlanningFcst_list = []

        for _, the_NPIAssociation in NPIAssociation.iterrows():

            the_NPIAssociation_df = the_NPIAssociation.to_frame().T

            the_initiative = the_NPIAssociation[initiative_col]
            the_data_object = the_NPIAssociation[data_object_col]
            item_level = the_NPIAssociation[npi_item_level_col]
            account_level = the_NPIAssociation[npi_account_level_col]
            channel_level = the_NPIAssociation[npi_channel_level_col]
            region_level = the_NPIAssociation[npi_region_level_col]
            pnl_level = the_NPIAssociation[npi_pnl_level_col]
            dd_level = the_NPIAssociation[npi_dd_level_col]
            location_level = the_NPIAssociation[npi_location_level_col]

            item_level = (
                "Item.[" + str(item_level).strip() + "]"
                if item_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            account_level = (
                "Account.[" + str(account_level).strip() + "]"
                if account_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            channel_level = (
                "Channel.[" + str(channel_level).strip() + "]"
                if channel_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            region_level = (
                "Region.[" + str(region_level).strip() + "]"
                if region_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            pnl_level = (
                "PnL.[" + str(pnl_level).strip() + "]"
                if pnl_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            dd_level = (
                "Demand Domain.[" + str(dd_level).strip() + "]"
                if dd_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )
            location_level = (
                "Location.[" + str(location_level).strip() + "]"
                if location_level not in [None, "na", "NA", "None", "", " ", np.nan]
                else None
            )

            # Null out existing planning level npi fcst
            the_NPIPlanningLevelFcst = NPIPlanningLevelFcst[
                (NPIPlanningLevelFcst[initiative_col] == the_initiative)
                & (NPIPlanningLevelFcst[data_object_col] == the_data_object)
            ].copy(deep=True)
            the_NPIPlanningLevelFcst[pl_level_npi_fcst_col] = np.nan

            if item_level is None:
                logger.warning(
                    f"Item level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if account_level is None:
                logger.warning(
                    f"Account level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if channel_level is None:
                logger.warning(
                    f"Channel level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if region_level is None:
                logger.warning(
                    f"Region level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if pnl_level is None:
                logger.warning(
                    f"PnL level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if dd_level is None:
                logger.warning(
                    f"Demand Domain level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            if location_level is None:
                logger.warning(
                    f"Location level is empty for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue

            # Get the fcst to disaggregate
            keys = [
                initiative_col,
                data_object_col,
                npi_item_col,
                npi_account_col,
                npi_channel_col,
                npi_region_col,
                npi_pnl_col,
                npi_location_col,
                npi_demand_domain_col,
            ]
            the_fcst = pd.merge(NPIFcst, the_NPIAssociation_df[keys], on=keys, how="inner")

            if len(the_fcst) == 0:
                logger.warning(
                    f"No records found in 'NPIFcst' for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                continue

            # Rename to original columns
            the_fcst.rename(
                columns={
                    npi_item_col: item_level,
                    npi_account_col: account_level,
                    npi_channel_col: channel_level,
                    npi_region_col: region_level,
                    npi_pnl_col: pnl_level,
                    npi_location_col: location_level,
                    npi_demand_domain_col: dd_level,
                },
                inplace=True,
            )

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
                npi_final_split_perc_col,
            ]
            req_cols = list(set(req_cols))

            # Filter relevant splits
            relevant_splits = Splits_w_masters[req_cols].drop_duplicates()
            relevant_splits = pd.merge(
                relevant_splits, the_fcst[keys].drop_duplicates(), on=keys, how="inner"
            )

            # Normalize the split %
            normalized_relevant_splits = relevant_splits
            total_sum = normalized_relevant_splits[npi_final_split_perc_col].sum()
            if total_sum == 0:
                logger.warning(
                    f"Total sum of 'NPI System Split %' is zero for slice: {df_keys} for intersection: \n{the_NPIAssociation}..."
                )
                logger.warning("Skipping the intersection ...")
                continue
            normalized_relevant_splits[npi_final_split_perc_col] = (
                normalized_relevant_splits[npi_final_split_perc_col] / total_sum
            )

            # Disaggregate NPI Fcst
            pl_the_fcst = pd.merge(normalized_relevant_splits, the_fcst, on=keys, how="inner")
            pl_the_fcst[pl_level_npi_fcst_col] = (
                pl_the_fcst[npi_final_split_perc_col] * pl_the_fcst[npi_fcst_final_l0_col]
            )

            # Values
            pl_the_fcst[version_col] = version

            # Req cols
            pl_the_fcst = pl_the_fcst[cols_required_in_output_pl_fcst]

            #  Null out old fcst data
            pl_the_fcst = pd.concat([pl_the_fcst, the_NPIPlanningLevelFcst], axis=0)

            keys = list(set(cols_required_in_output_pl_fcst) - set([pl_level_npi_fcst_col]))
            pl_the_fcst = pl_the_fcst.groupby(keys)[pl_level_npi_fcst_col].max().reset_index()

            PlanningFcst_list.append(pl_the_fcst)

        # Concatenate the output dataframes
        PlanningFcstOutput = concat_to_dataframe(PlanningFcst_list)

        # ---------------------> Calculate NPI Forecast at Planning level L1
        PlanningFcstL1_list = []
        EligibleLevels_list = []

        initiatives = SelectedInitiativeLevel[initiative_col].unique()

        for the_initiative in initiatives:

            the_NPIPlanningLevelFcst = NPIPlanningLevelFcst[
                NPIPlanningLevelFcst[initiative_col] == the_initiative
            ]
            the_pl_fcst = PlanningFcstOutput[PlanningFcstOutput[initiative_col] == the_initiative]
            the_NPIPlanningLevelFcstL1 = NPIPlanningLevelFcstL1[
                NPIPlanningLevelFcstL1[initiative_col] == the_initiative
            ]

            the_NPIPlanningLevelFcst.rename(
                columns={pl_level_npi_fcst_col: pl_level_npi_fcst_l1_col}, inplace=True
            )
            the_pl_fcst.rename(
                columns={pl_level_npi_fcst_col: pl_level_npi_fcst_l1_col}, inplace=True
            )

            # null out the_NPIPlanningLevelFcstL1
            the_NPIPlanningLevelFcstL1[pl_level_npi_fcst_l1_col] = np.nan

            # Merge NPIPlanningLevelFcst and PlanningFcstOutput
            keys = [
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
            ]
            the_fcst = pd.concat([the_pl_fcst, the_NPIPlanningLevelFcst], axis=0)

            #  Get the latest level data
            levels = the_fcst[data_object_col].unique()
            levels_seq_mapping = InitiativeLevels[
                [initiative_col, data_object_col, npi_level_seq_l1_col]
            ].drop_duplicates()
            the_levels_seq_mapping = levels_seq_mapping[
                levels_seq_mapping[initiative_col] == the_initiative
            ]
            the_levels_seq_mapping.sort_values(
                by=npi_level_seq_l1_col, ascending=False, inplace=True
            )

            # Get the latest level have data
            the_level = [lvl for lvl in the_levels_seq_mapping[data_object_col] if lvl in levels]

            if len(the_level) == 0:
                logger.warning(
                    f"No levels found which contains data for slice: {df_keys} for initiative: {the_initiative}..."
                )
                logger.warning("Skipping the initiative ...")
                continue

            the_fcst_for_latest_level = the_fcst[the_fcst[data_object_col] == the_level[0]]
            the_fcst_for_latest_level.drop(columns=[data_object_col], inplace=True)

            # Get the nulled out old fcst data
            keys = [
                version_col,
                initiative_col,
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
                partial_week_col,
            ]
            the_fcst_for_latest_level = pd.concat(
                [the_fcst_for_latest_level, the_NPIPlanningLevelFcstL1], axis=0
            )
            the_fcst_for_latest_level = (
                the_fcst_for_latest_level.groupby(keys)[pl_level_npi_fcst_l1_col]
                .first()
                .reset_index()
            )

            # Eligible level
            the_sequence = the_levels_seq_mapping[
                the_levels_seq_mapping[data_object_col] == the_level[0]
            ][npi_level_seq_l1_col].values[0]
            data = {
                version_col: [version],
                initiative_col: [the_initiative],
                npi_latest_level_seq_col: [the_sequence],
            }
            the_EligibleLevels = pd.DataFrame(data=data)

            EligibleLevels_list.append(the_EligibleLevels)

            # Req col
            the_fcst_for_latest_level = the_fcst_for_latest_level[
                cols_required_in_output_pl_fcst_l1
            ]
            PlanningFcstL1_list.append(the_fcst_for_latest_level)

        # Concatenate the output dataframes
        PlanningFcstL1Output = concat_to_dataframe(PlanningFcstL1_list)
        EligibleLevelsOutput = concat_to_dataframe(EligibleLevels_list)

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

    return PlanningFcstOutput, PlanningFcstL1Output, EligibleLevelsOutput

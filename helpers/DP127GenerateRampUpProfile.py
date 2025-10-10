"""Generate Ramp Up profile for Flexible NPI.

Pseudocode:
    Version: 2025.08.0
    --------------------
        - Step 1: Aggregate 'LifeCycleVolume' input at 'Initiative Level' input for the data object in 'Parameters' input.
        - Step 2: Aggregate the 'LifeCycleVolume' by 'NPI Ramp Up Bucket L0' input at time grain
        - Step 3: Filter 'LifeCycleVolume' for the 'LikeItem' input intersections.
        - Step 4: Calculate 'Ramp Up Period Sales L0' which is sum of vol and 'Ramp Up Period Average Sales L0' which is mean of vol at the 'NPI Ramp Up Bucket L0' input, for the ramp up period.
                    -> Output
        - Step 5: Filter 'LifeCycleVolume' for valid 'Final Like Assortment L0' from [Step 3]
        - Step 6: Calculate weighted sum/avg using 'LifeCycleVolume' for the ramp up period.
        - Step 7: Calculte 'System Defined Ramp Up Volume L0'
                    -> Output
        - Step 8: Calculate 'Ramp Up Profile L0' by dividing 'Lifeycle Volume L1' divided by total 'Lifeycle Volume L1' for the ramp up period from [Step 6]
                    -> Output
"""

from functools import reduce
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from o9Reference.common_utils.dataframe_utils import (
    join_two_polars_df_auto as join_two_auto,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str_polars,
    map_output_columns_to_dtypes,
    pandas_polars_df_io_bridge,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9helpers.o9logger import O9Logger

# Assuming the same logger setup as the original script
logger = O9Logger()

# Polars to set this context for more readable tracebacks.
pl.Config.set_tbl_rows(25)
pl.Config.set_tbl_cols(50)
pl.Config.set_fmt_str_lengths(100)

col_mapping = {
    "Ramp Up Profile L0": float,
    "System Suggested Ramp Up Volume L0": float,
    "630 Initiative Like Assortment Match.[Ramp Up Period Sales L0]": float,
    "630 Initiative Like Assortment Match.[Ramp Up Period Average Sales L0]": float,
}


@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@pandas_polars_df_io_bridge
@convert_category_cols_to_str_polars
@log_inputs_and_outputs
def main(
    # Data
    SelectedCombinations: pl.DataFrame = None,
    Parameters: pl.DataFrame = None,
    LifecycleVolume: pl.DataFrame = None,
    LifecycleTime: pl.DataFrame = None,
    LikeItem: pl.DataFrame = None,
    InitiativeLevel: pl.DataFrame = None,
    # Master data
    ItemMaster: pl.DataFrame = None,
    AccountMaster: pl.DataFrame = None,
    ChannelMaster: pl.DataFrame = None,
    RegionMaster: pl.DataFrame = None,
    PnLMaster: pl.DataFrame = None,
    DemandDomainMaster: pl.DataFrame = None,
    LocationMaster: pl.DataFrame = None,
    ExecutionScope: Optional[str] = None,
    df_keys={},
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Entry point of the script. Polars based Ramp Up plugin computation."""
    plugin_name = "DP127GenerateRampUpProfile"
    logger.info("Executing Polars version of {} for slice {}".format(plugin_name, df_keys))

    # --- Column Definitions (remain the same) ---
    version_col = "Version.[Version Name]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    lifecycle_bucket_col = "Lifecycle Time.[Lifecycle Bucket]"
    lifecycle_day_col_col = "Lifecycle Time.[Day]"
    pl_item_col = "Item.[Planning Item]"
    planning_account_col = "Account.[Planning Account]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_region_col = "Region.[Planning Region]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_loc_col = "Location.[Planning Location]"
    planning_dd_col = "Demand Domain.[Planning Demand Domain]"
    npi_item_col = "Item.[NPI Item]"
    npi_account_col = "Account.[NPI Account]"
    npi_channel_col = "Channel.[NPI Channel]"
    npi_region_col = "Region.[NPI Region]"
    npi_pnl_col = "PnL.[NPI PnL]"
    npi_location_col = "Location.[NPI Location]"
    npi_demand_domain_col = "Demand Domain.[NPI Demand Domain]"
    from_initiative_col = "from.[Initiative].[Initiative]"
    from_data_object_col = "from.[Data Object].[Data Object]"
    from_npi_item_col = "from.[Item].[NPI Item]"
    from_npi_account_col = "from.[Account].[NPI Account]"
    from_npi_channel_col = "from.[Channel].[NPI Channel]"
    from_npi_region_col = "from.[Region].[NPI Region]"
    from_npi_pnl_col = "from.[PnL].[NPI PnL]"
    from_npi_demand_domain_col = "from.[Demand Domain].[NPI Demand Domain]"
    from_npi_location_col = "from.[Location].[NPI Location]"
    to_npi_item_col = "to.[Item].[NPI Item]"
    to_npi_account_col = "to.[Account].[NPI Account]"
    to_npi_channel_col = "to.[Channel].[NPI Channel]"
    to_npi_region_col = "to.[Region].[NPI Region]"
    to_npi_pnl_col = "to.[PnL].[NPI PnL]"
    to_npi_demand_domain_col = "to.[Demand Domain].[NPI Demand Domain]"
    to_npi_location_col = "to.[Location].[NPI Location]"
    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_dd_level_col = "NPI Demand Domain Level"
    npi_loc_level_col = "NPI Location Level"
    npi_ramp_up_period_col = "NPI Ramp Up Period L0"
    sys_def_ramp_up_vol_col = "System Suggested Ramp Up Volume L0"
    npi_ramp_up_bucket_col = "NPI Ramp Up Bucket L0"
    ramp_up_profile_l0_col = "Ramp Up Profile L0"
    ramp_up_period_sales_l0_col = "630 Initiative Like Assortment Match.[Ramp Up Period Sales L0]"
    ramp_up_period_avg_sales_l0_col = (
        "630 Initiative Like Assortment Match.[Ramp Up Period Average Sales L0]"
    )
    gen_ramp_up_profile_ass_l0_col = "Generate Ramp Up Profile Assortment L0"
    lifecycle_vol_col = "Lifecycle Volume"
    final_like_ass_l0_col = "630 Initiative Like Assortment Match.[Final Like Assortment L0]"
    like_item_fcst_method_l0_col = "Like Item Fcst Method L0"
    user_like_ass_weight_col = (
        "630 Initiative Like Assortment Match.[User Override Like Assortment Weight L0]"
    )
    lifecycle_vol_col_weighted = f"{lifecycle_vol_col}_weighted"

    # --- [FIXED] Define output column lists ---
    cols_required_in_output_ramp_up_profile = [
        version_col,
        data_object_col,
        lifecycle_bucket_col,
        initiative_col,
        npi_item_col,
        npi_account_col,
        npi_channel_col,
        npi_region_col,
        npi_pnl_col,
        npi_demand_domain_col,
        npi_location_col,
        ramp_up_profile_l0_col,
    ]
    cols_required_in_output_sys_def_ramp_up_vol = [
        version_col,
        data_object_col,
        initiative_col,
        npi_item_col,
        npi_account_col,
        npi_channel_col,
        npi_region_col,
        npi_pnl_col,
        npi_demand_domain_col,
        npi_location_col,
        sys_def_ramp_up_vol_col,
    ]
    cols_required_in_output_like_item_info = [
        version_col,
        from_initiative_col,
        from_data_object_col,
        from_npi_item_col,
        from_npi_account_col,
        from_npi_channel_col,
        from_npi_region_col,
        from_npi_pnl_col,
        from_npi_demand_domain_col,
        from_npi_location_col,
        to_npi_item_col,
        to_npi_account_col,
        to_npi_channel_col,
        to_npi_region_col,
        to_npi_pnl_col,
        to_npi_demand_domain_col,
        to_npi_location_col,
        ramp_up_period_sales_l0_col,
        ramp_up_period_avg_sales_l0_col,
    ]

    RampUpProfileOutput = pd.DataFrame(columns=cols_required_in_output_ramp_up_profile)
    SystemDefinedRampUpVolumeOutput = pd.DataFrame(
        columns=cols_required_in_output_sys_def_ramp_up_vol
    )
    LikeItemInfoOutput = pd.DataFrame(columns=cols_required_in_output_like_item_info)

    try:
        # --- Pre-computation Checks & Setup ---
        if ExecutionScope not in ["LikeItemInfo", "RampUpInfo", "All"]:
            raise ValueError(
                "ExecutionScope is invalid. Allowed values are 'LikeItemInfo', 'RampUpInfo', or 'All'."
            )

        for df_name, df in [
            ("SelectedCombinations", SelectedCombinations),
            ("LikeItem", LikeItem),
            ("InitiativeLevel", InitiativeLevel),
            ("LifecycleVolume", LifecycleVolume),
            ("Parameters", Parameters),
        ]:
            if df is None or len(df) == 0:
                raise ValueError(f"{df_name} input is empty for slice: {df_keys}")

        # DataFrames could modified for lazy evaluation later
        ldf_sc = SelectedCombinations
        ldf_params = Parameters
        ldf_lc_vol = LifecycleVolume
        ldf_lc_time = LifecycleTime
        ldf_like_item = LikeItem
        ldf_init_level = InitiativeLevel
        ldf_item_master = ItemMaster
        ldf_acct_master = AccountMaster
        ldf_chan_master = ChannelMaster
        ldf_reg_master = RegionMaster
        ldf_pnl_master = PnLMaster
        ldf_dd_master = DemandDomainMaster
        ldf_loc_master = LocationMaster

        # --- Prepare Driving DataFrame ---
        ldf_sc_filtered = ldf_sc.with_columns(
            pl.col(gen_ramp_up_profile_ass_l0_col).cast(pl.Int32, strict=False)
        ).filter(pl.col(gen_ramp_up_profile_ass_l0_col) >= 1)

        if ldf_sc_filtered.is_empty():
            raise ValueError(
                f"SelectedCombinations input does not contains valid assortment for slice: {df_keys}"
            )

        if ldf_sc_filtered.select(pl.col(version_col).n_unique()).item() > 1:
            raise ValueError(f"Plugin does not support multiple versions for slice: {df_keys}.")

        if ldf_like_item.is_empty():
            raise ValueError(f"LikeItem input is empty for slice: {df_keys}")

        if ldf_init_level.is_empty():
            raise ValueError(f"InitiativeLevel input can't be empty for slice {df_keys}")

        if ldf_lc_vol.is_empty():
            raise ValueError(f"LifecycleVolume can't be empty for slice {df_keys}")

        if ldf_params.is_empty():
            raise ValueError(f"Parameters input can't be empty for slice {df_keys}")

        version = ldf_sc_filtered.select(version_col).item(0, 0)

        # Drop version column from sources
        ldf_sc_filtered = ldf_sc_filtered.drop(version_col)
        ldf_init_level = ldf_init_level.drop(version_col)
        ldf_lc_vol = ldf_lc_vol.drop(version_col)
        ldf_params = ldf_params.drop(version_col)

        ldf_like_item = ldf_like_item.with_columns(
            pl.col(user_like_ass_weight_col).cast(pl.Float32, strict=False).fill_null(1.0)
        )

        common_cols_sc_params = list(set(ldf_sc_filtered.columns) & set(ldf_params.columns))
        with pl.StringCache():
            ldf_sc_with_params = ldf_sc_filtered.join(
                ldf_params, on=common_cols_sc_params, how="left"
            )

        LikeItemInfoOutput_list = []
        SystemDefinedRampUpVolumeOutput_list = []
        RampUpProfileOutput_list = []

        # Support the plugin for multiple intersections
        # --- Loop over distinct configurations (highly efficient) ---
        for the_SelectedCombinations in ldf_sc_with_params.iter_rows(named=True):

            the_data_object = the_SelectedCombinations[data_object_col]
            the_initiative = the_SelectedCombinations[initiative_col]
            the_npi_item = the_SelectedCombinations[npi_item_col]
            the_npi_account = the_SelectedCombinations[npi_account_col]
            the_npi_channel = the_SelectedCombinations[npi_channel_col]
            the_npi_region = the_SelectedCombinations[npi_region_col]
            the_npi_pnl = the_SelectedCombinations[npi_pnl_col]
            the_npi_dd = the_SelectedCombinations[npi_demand_domain_col]
            the_npi_loc = the_SelectedCombinations[npi_location_col]
            the_ramp_up_bucket = the_SelectedCombinations[npi_ramp_up_bucket_col]
            the_like_item_fcst_method = the_SelectedCombinations[like_item_fcst_method_l0_col]
            npi_ramp_up_period = the_SelectedCombinations[npi_ramp_up_period_col]

            if len(the_ramp_up_bucket) == 0 or the_ramp_up_bucket in ["", np.nan, None, "NA"]:
                logger.warning(
                    f"{npi_ramp_up_bucket_col} is missing for slice: {df_keys} for intersection: \n{the_SelectedCombinations}"
                )
                logger.warning(f"Intersection skipped for slice: {df_keys}...")
                continue

            allowed_val = ["Weighted Average", "Weighted Sum"]
            if not the_like_item_fcst_method in allowed_val:
                logger.warning(
                    f"{like_item_fcst_method_l0_col} is invalid: '{the_like_item_fcst_method}'. Possible values are {allowed_val}: for slice: {df_keys} for intersection: \n{the_SelectedCombinations}"
                )
                logger.warning(
                    f"Considering {like_item_fcst_method_l0_col} as 'Weighted Average' for slice: {df_keys} ..."
                )
                the_like_item_fcst_method = "Weighted Average"

            if (npi_ramp_up_period in ["", np.nan, None]) or (
                int(npi_ramp_up_period) != npi_ramp_up_period
            ):
                logger.warning(
                    f"{npi_ramp_up_period_col} is invalid: '{npi_ramp_up_period}'. Possible values are integer, for slice: {df_keys} for intersection: \n{the_SelectedCombinations}"
                )
                logger.warning(f"Intersection skipped for slice: {df_keys} ...")
                continue

            the_likeitem = ldf_like_item.filter(
                (pl.col(from_data_object_col) == the_data_object)
                & (pl.col(from_initiative_col) == the_initiative)
                & (pl.col(from_npi_item_col) == the_npi_item)
                & (pl.col(from_npi_account_col) == the_npi_account)
                & (pl.col(from_npi_channel_col) == the_npi_channel)
                & (pl.col(from_npi_region_col) == the_npi_region)
                & (pl.col(from_npi_pnl_col) == the_npi_pnl)
                & (pl.col(from_npi_demand_domain_col) == the_npi_dd)
                & (pl.col(from_npi_location_col) == the_npi_loc)
            )

            if len(the_likeitem) == 0:
                logger.warning(
                    f"Like Item does not have data for slice: {df_keys} for intersection: {the_SelectedCombinations} "
                )
                logger.warning("Ignoring the intersection for slice: {df_keys} ...")
                continue

            the_InitiativeLevel = InitiativeLevel.filter(
                (pl.col(initiative_col) == the_initiative)
                & (pl.col(data_object_col) == the_data_object)
            )

            if len(the_InitiativeLevel) == 0:
                logger.warning(
                    f"Initiative Level does not have data for slice: {df_keys} for intersection: {the_SelectedCombinations} "
                )
                logger.warning("Ignoring the intersection ...")
                continue

            # Levels
            the_item_level = the_InitiativeLevel[npi_item_level_col].item(0)
            the_account_level = the_InitiativeLevel[npi_account_level_col].item(0)
            the_channel_level = the_InitiativeLevel[npi_channel_level_col].item(0)
            the_region_level = the_InitiativeLevel[npi_region_level_col].item(0)
            the_pnl_level = the_InitiativeLevel[npi_pnl_level_col].item(0)
            the_dd_level = the_InitiativeLevel[npi_dd_level_col].item(0)
            the_loc_level = the_InitiativeLevel[npi_loc_level_col].item(0)

            the_item_level = "Item.[" + the_item_level + "]"
            the_account_level = "Account.[" + the_account_level + "]"
            the_channel_level = "Channel.[" + the_channel_level + "]"
            the_region_level = "Region.[" + the_region_level + "]"
            the_pnl_level = "PnL.[" + the_pnl_level + "]"
            the_dd_level = "Demand Domain.[" + the_dd_level + "]"
            the_loc_level = "Location.[" + the_loc_level + "]"
            the_lifecycle_time_col = "Lifecycle Time.[" + the_ramp_up_bucket + "]"
            the_lifecycle_time_key_col = (
                "Lifecycle Time.[" + "".join(the_ramp_up_bucket.split()) + "Key]"
            )

            # --- Aggregate LifecycleVolume for this config ---
            master_ldfs = [
                ldf_item_master.select(list(set([pl_item_col, the_item_level]))).unique(),
                ldf_acct_master.select(
                    list(set([planning_account_col, the_account_level]))
                ).unique(),
                ldf_chan_master.select(
                    list(set([planning_channel_col, the_channel_level]))
                ).unique(),
                ldf_reg_master.select(list(set([planning_region_col, the_region_level]))).unique(),
                ldf_pnl_master.select(list(set([planning_pnl_col, the_pnl_level]))).unique(),
                ldf_dd_master.select(list(set([planning_dd_col, the_dd_level]))).unique(),
                ldf_loc_master.select(list(set([planning_loc_col, the_loc_level]))).unique(),
                ldf_lc_time.select(
                    list(
                        set(
                            [
                                lifecycle_day_col_col,
                                the_lifecycle_time_col,
                                the_lifecycle_time_key_col,
                            ]
                        )
                    )
                ).unique(),
            ]
            ldf_lc_vol_with_masters = reduce(
                lambda left, right: join_two_auto(left, right), master_ldfs, ldf_lc_vol
            ).with_columns(pl.col(lifecycle_vol_col).cast(pl.Float32))

            agg_levels = [
                the_item_level,
                the_account_level,
                the_channel_level,
                the_region_level,
                the_pnl_level,
                the_dd_level,
                the_loc_level,
                the_lifecycle_time_col,
                the_lifecycle_time_key_col,
            ]
            ldf_agg_vol = ldf_lc_vol_with_masters.group_by(agg_levels).agg(
                pl.col(lifecycle_vol_col).sum()
            )

            rename_map = {
                the_item_level: to_npi_item_col,
                the_account_level: to_npi_account_col,
                the_channel_level: to_npi_channel_col,
                the_region_level: to_npi_region_col,
                the_pnl_level: to_npi_pnl_col,
                the_dd_level: to_npi_demand_domain_col,
                the_loc_level: to_npi_location_col,
            }

            the_lifecycle_vol = ldf_agg_vol.rename(rename_map)

            # --- Combine aggregated volume with LikeItem data ---
            the_likeitem = the_likeitem.join(
                LifecycleTime.select(the_lifecycle_time_col, the_lifecycle_time_key_col).unique(),
                how="cross",
            )

            # Npi item with their like item and volumns
            key = [
                to_npi_item_col,
                to_npi_account_col,
                to_npi_channel_col,
                to_npi_region_col,
                to_npi_pnl_col,
                to_npi_demand_domain_col,
                to_npi_location_col,
                the_lifecycle_time_col,
                the_lifecycle_time_key_col,
            ]

            # Join configs with LikeItem to get all valid 'from-to' relationships
            the_data = the_likeitem.join(the_lifecycle_vol, on=key, how="left")

            if len(the_data) == 0:
                logger.warning(
                    f"Lifecycle volume is missing for slice: {df_keys} for intersection: \n{the_SelectedCombinations}"
                )
                logger.warning("Skipping the intersection ...")

            # Fill missing sales with 0
            the_data = the_data.with_columns([pl.col(lifecycle_vol_col).fill_null(0)])

            if ExecutionScope in ["LikeItemInfo", "All"]:
                # --- Calculate Ramp Up Period Sales L0 and Avg
                # For the ramp up period
                ramp_up_period_mapping = (
                    the_data.select([the_lifecycle_time_col, the_lifecycle_time_key_col])
                    .unique()
                    .sort(by=the_lifecycle_time_key_col)
                    .head(int(npi_ramp_up_period))
                )

                ramp_up_period_list = ramp_up_period_mapping[the_lifecycle_time_col].to_list()

                the_data_for_ramp_up_period = the_data.filter(
                    pl.col(the_lifecycle_time_col).is_in(ramp_up_period_list)
                )

                # Grouping and aggregation
                grouped = the_data_for_ramp_up_period.group_by(
                    [
                        from_initiative_col,
                        from_data_object_col,
                        from_npi_item_col,
                        from_npi_account_col,
                        from_npi_channel_col,
                        from_npi_region_col,
                        from_npi_pnl_col,
                        from_npi_demand_domain_col,
                        from_npi_location_col,
                        to_npi_item_col,
                        to_npi_account_col,
                        to_npi_channel_col,
                        to_npi_region_col,
                        to_npi_pnl_col,
                        to_npi_demand_domain_col,
                        to_npi_location_col,
                    ]
                ).agg([pl.col(lifecycle_vol_col).sum().alias(ramp_up_period_sales_l0_col)])

                # Add average sales column
                grouped = grouped.with_columns(
                    [
                        (pl.col(ramp_up_period_sales_l0_col) / int(npi_ramp_up_period)).alias(
                            ramp_up_period_avg_sales_l0_col
                        ),
                        pl.lit(version).alias(version_col),
                        pl.lit(the_data_object).alias(data_object_col),
                        pl.lit(the_initiative).alias(initiative_col),
                    ]
                )

                # Select only the required columns
                the_LikeItemInfoOutput = grouped.select(cols_required_in_output_like_item_info)

                # Append result to the output list
                LikeItemInfoOutput_list.append(the_LikeItemInfoOutput)

            if ExecutionScope in ["RampUpInfo", "All"]:
                # --- Calculate weighted values
                # Step 1: Filter out rows where final_like_ass_l0_col is True
                the_data = the_data.with_columns(
                    [pl.col(final_like_ass_l0_col).fill_null(False).cast(bool)]
                ).filter(pl.col(final_like_ass_l0_col))

                if the_data.height == 0:
                    logger.warning(
                        f"Does not found valid {final_like_ass_l0_col} for slice: {df_keys} for intersection: \n{the_SelectedCombinations}"
                    )
                    logger.warning("Skipping Ramp Up Info calculation ...")
                else:
                    # Step 2: Weighted aggregation
                    if the_like_item_fcst_method == "Weighted Average":
                        the_data_weighted = (
                            the_data.group_by(
                                [
                                    from_initiative_col,
                                    from_data_object_col,
                                    from_npi_item_col,
                                    from_npi_account_col,
                                    from_npi_channel_col,
                                    from_npi_region_col,
                                    from_npi_pnl_col,
                                    from_npi_demand_domain_col,
                                    from_npi_location_col,
                                    the_lifecycle_time_col,
                                    the_lifecycle_time_key_col,
                                ]
                            )
                            .agg(
                                [
                                    (pl.col(lifecycle_vol_col) * pl.col(user_like_ass_weight_col))
                                    .sum()
                                    .alias("weighted_numerator"),
                                    pl.col(user_like_ass_weight_col).sum().alias("weights_sum"),
                                ]
                            )
                            .with_columns(
                                [
                                    (pl.col("weighted_numerator") / pl.col("weights_sum")).alias(
                                        lifecycle_vol_col_weighted
                                    )
                                ]
                            )
                            .select(
                                from_initiative_col,
                                from_data_object_col,
                                from_npi_item_col,
                                from_npi_account_col,
                                from_npi_channel_col,
                                from_npi_region_col,
                                from_npi_pnl_col,
                                from_npi_demand_domain_col,
                                from_npi_location_col,
                                the_lifecycle_time_col,
                                the_lifecycle_time_key_col,
                                lifecycle_vol_col_weighted,
                            )
                        )

                    else:  # Weighted Sum
                        the_data_weighted = (
                            the_data.with_columns(
                                [
                                    (
                                        pl.col(lifecycle_vol_col) * pl.col(user_like_ass_weight_col)
                                    ).alias("weighted_vol")
                                ]
                            )
                            .group_by(
                                [
                                    from_initiative_col,
                                    from_data_object_col,
                                    from_npi_item_col,
                                    from_npi_account_col,
                                    from_npi_channel_col,
                                    from_npi_region_col,
                                    from_npi_pnl_col,
                                    from_npi_demand_domain_col,
                                    from_npi_location_col,
                                    the_lifecycle_time_col,
                                    the_lifecycle_time_key_col,
                                ]
                            )
                            .agg([pl.col("weighted_vol").sum().alias(lifecycle_vol_col_weighted)])
                        )

                    # Step 3: Filter ramp-up period
                    the_data_weighted = the_data_weighted.filter(
                        pl.col(the_lifecycle_time_col).is_in(ramp_up_period_list)
                    )

                    # Step 4: System-defined ramp-up volume calculation
                    the_SystemDefinedRampUpVolumeOutput = (
                        the_data_weighted.group_by(
                            [
                                from_initiative_col,
                                from_data_object_col,
                                from_npi_item_col,
                                from_npi_account_col,
                                from_npi_channel_col,
                                from_npi_region_col,
                                from_npi_pnl_col,
                                from_npi_demand_domain_col,
                                from_npi_location_col,
                            ]
                        )
                        .agg(
                            [
                                pl.col(lifecycle_vol_col_weighted)
                                .sum()
                                .alias(sys_def_ramp_up_vol_col)
                            ]
                        )
                        .with_columns(
                            [
                                pl.lit(version).alias(version_col),
                                pl.lit(the_initiative).alias(initiative_col),
                                pl.lit(the_data_object).alias(data_object_col),
                            ]
                        )
                        .rename(
                            {
                                from_npi_item_col: npi_item_col,
                                from_npi_account_col: npi_account_col,
                                from_npi_channel_col: npi_channel_col,
                                from_npi_region_col: npi_region_col,
                                from_npi_pnl_col: npi_pnl_col,
                                from_npi_demand_domain_col: npi_demand_domain_col,
                                from_npi_location_col: npi_location_col,
                            }
                        )
                        .select(cols_required_in_output_sys_def_ramp_up_vol)
                    )

                    SystemDefinedRampUpVolumeOutput_list.append(the_SystemDefinedRampUpVolumeOutput)

                    # Step 5: Ramp up profile calculation
                    total_volume = the_data_weighted[lifecycle_vol_col_weighted].sum()
                    the_RampUpProfileOutput = the_data_weighted.with_columns(
                        [
                            (pl.col(lifecycle_vol_col_weighted) / total_volume).alias(
                                ramp_up_profile_l0_col
                            ),
                            pl.lit(version).alias(version_col),
                        ]
                    )

                    # Rename columns to match output requirements
                    the_RampUpProfileOutput = the_RampUpProfileOutput.rename(
                        {
                            from_initiative_col: initiative_col,
                            from_data_object_col: data_object_col,
                            from_npi_item_col: npi_item_col,
                            from_npi_account_col: npi_account_col,
                            from_npi_channel_col: npi_channel_col,
                            from_npi_region_col: npi_region_col,
                            from_npi_pnl_col: npi_pnl_col,
                            from_npi_demand_domain_col: npi_demand_domain_col,
                            from_npi_location_col: npi_location_col,
                        }
                    )

                    # Step 6: Lifecycle bucket assignment
                    the_RampUpProfileOutput = (
                        the_RampUpProfileOutput.sort(by=the_lifecycle_time_key_col)
                        .with_row_count("row_idx")
                        .with_columns(
                            [
                                pl.Series(
                                    lifecycle_bucket_col,
                                    LifecycleTime[lifecycle_bucket_col].unique(maintain_order=True)[
                                        : the_RampUpProfileOutput.height
                                    ],
                                )
                            ]
                        )
                        .select(cols_required_in_output_ramp_up_profile)
                    )

                    RampUpProfileOutput_list.append(the_RampUpProfileOutput)

        # --- Final Concatenation and Output ---
        if len(LikeItemInfoOutput_list) > 0:
            LikeItemInfoOutput = pl.concat(LikeItemInfoOutput_list, how="vertical")
        if len(SystemDefinedRampUpVolumeOutput_list) > 0:
            SystemDefinedRampUpVolumeOutput = pl.concat(
                SystemDefinedRampUpVolumeOutput_list, how="vertical"
            )
        if len(RampUpProfileOutput_list) > 0:
            RampUpProfileOutput = pl.concat(RampUpProfileOutput_list, how="vertical")

        logger.info("Polars Plugin Execution completed.")

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}, returning empty dataframes...")
        # Return empty dataframes with correct schema on failure
        return LikeItemInfoOutput, SystemDefinedRampUpVolumeOutput, RampUpProfileOutput

    return LikeItemInfoOutput, SystemDefinedRampUpVolumeOutput, RampUpProfileOutput

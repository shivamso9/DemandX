"""This module calculates the consensus baseline Forecast."""

import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import (
    clean_measure_name,
    remove_first_brackets,
)

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {
    "Consensus Baseline Fcst": float,
    "Source Consensus": str,
    "Is Finalized Rule": int,
    "Planning Horizon Period": float,
    "Planning Horizon Candidate": str,
    "Cannibalization Impact Flag": str,
}


# A class to store constant values used throughout the module.
class Constants:
    """Holds constant values used throughout the baseline calculation."""

    version_col = o9Constants.VERSION_NAME
    pl_item_col = o9Constants.PLANNING_ITEM
    pl_account_col = o9Constants.PLANNING_ACCOUNT
    pl_channel_col = o9Constants.PLANNING_CHANNEL
    pl_region_col = o9Constants.PLANNING_REGION
    pl_pnl_col = o9Constants.PLANNING_PNL
    pl_location_col = o9Constants.PLANNING_LOCATION
    pl_demand_domain_col = o9Constants.PLANNING_DEMAND_DOMAIN

    data_object_col = o9Constants.DATA_OBJECT
    data_validation_col = o9Constants.DATA_VALIDATION
    dm_rule_col = o9Constants.DM_RULE
    planning_cycle_key = o9Constants.PLANNING_CYCLE_KEY

    partial_week_col = "Partial Week"
    pw_col = o9Constants.PARTIAL_WEEK
    pw_key_col = o9Constants.PARTIAL_WEEK_KEY
    month_col = o9Constants.MONTH
    month_key_col = o9Constants.MONTH_KEY
    day_key_col = o9Constants.DAY_KEY

    baseline_fcst = "Consensus Baseline Fcst"
    source_candidate = "Source Candidate Fcst"

    do_account_lvl_col = o9Constants.DATA_OBJECT_ACCOUNT_LEVEL
    do_channel_lvl_col = o9Constants.DATA_OBJECT_CHANNEL_LEVEL
    do_demand_domain_lvl_col = o9Constants.DATA_OBJECT_DEMAND_DOMAIN_LEVEL
    do_item_lvl_col = o9Constants.DATA_OBJECT_ITEM_LEVEL
    do_pnl_lvl_col = o9Constants.DATA_OBJECT_PNL_LEVEL
    do_region_lvl_col = o9Constants.DATA_OBJECT_REGION_LEVEL
    do_location_lvl_col = o9Constants.DATA_OBJECT_LOCATION_LEVEL

    account_scope_col = o9Constants.DP_ACCOUNT_SCOPE_UI
    channel_scope_col = o9Constants.DP_CHANNEL_SCOPE_UI
    demand_domain_scope_col = o9Constants.DP_DEMAND_DOMAIN_SCOPE_UI
    item_scope_col = o9Constants.DP_ITEM_SCOPE_UI
    location_scope_col = o9Constants.DP_LOCATION_SCOPE_UI
    pnl_scope_col = o9Constants.DP_PNL_SCOPE_UI
    region_scope_col = o9Constants.DP_REGION_SCOPE_UI

    forecast_type_col = "DP Forecast Type"
    dm_rule_num = "dm_rule_num"
    segmentation_flag_col = "CDP Segmentation Flag"
    is_already_touchless = "Is Already Touchless"
    assortment_flag_col = o9Constants.ASSORTMENT_COLLAB_DP
    cluster_col = o9Constants.CLUSTER
    process_order_col = o9Constants.DATA_OBJECT_PROCESS_ORDER
    stat_fcst_l0 = o9Constants.STAT_FCST_L0

    is_last_horizon = "Is Last Horizon"
    consensus_horizon_period = "Consensus Horizon Period"
    consensus_horizon_bucket = "Consensus Horizon Bucket"

    horizon_candidate_fcst = "Horizon Candidate Fcst"
    candidate_horizon_period_by_rule = "Horizon Period by Rule"
    outside_horizon_candidate_fcst = "Outside Horizon Candidate Fcst"
    reference_fcst_for_cluster = "Reference Fcst for Cluster"
    cannibalization_impact_flag = "Cannibalization Impact Flag"

    is_finalized_rule = "Is Finalized Rule"
    planning_horizon_period = "Planning Horizon Period"
    planning_horizon_fcst = "Planning Candidate Fcst"
    cannibalization_candidate = "Candidate to Include Cannibalization Impact"

    # column lists
    pl_lvl_version_cols = [
        version_col,
        pl_item_col,
        pl_account_col,
        pl_channel_col,
        pl_region_col,
        pl_location_col,
        pl_pnl_col,
        pl_demand_domain_col,
    ]
    do_lvl_cols = [
        do_item_lvl_col,
        do_account_lvl_col,
        do_channel_lvl_col,
        do_region_lvl_col,
        do_location_lvl_col,
        do_pnl_lvl_col,
        do_demand_domain_lvl_col,
    ]
    scope_lvl_cols = [
        item_scope_col,
        account_scope_col,
        channel_scope_col,
        region_scope_col,
        location_scope_col,
        pnl_scope_col,
        demand_domain_scope_col,
    ]

    horizon_grain = [
        version_col,
        data_validation_col,
        dm_rule_col,
        pw_col,
        horizon_candidate_fcst,
        consensus_horizon_bucket,
        candidate_horizon_period_by_rule,
        is_last_horizon,
    ]

    cols_required_in_output1 = pl_lvl_version_cols + [
        pw_col,
        baseline_fcst,
        source_candidate,
        cannibalization_impact_flag,
    ]

    cols_required_in_output2 = pl_lvl_version_cols + [
        dm_rule_col,
        data_validation_col,
        planning_horizon_fcst,
        planning_horizon_period,
        is_finalized_rule,
    ]

    horizon_required_cols = [
        version_col,
        data_object_col,
        dm_rule_col,
        data_validation_col,
        consensus_horizon_bucket,
        candidate_horizon_period_by_rule,
        horizon_candidate_fcst,
        is_last_horizon,
    ]


def generate_horizon_pw_mapping(
    time_level: str,
    time_level_key: str,
    dataframe: pd.DataFrame,
    future_time_master: pd.DataFrame,
    grain: list,
):
    """Generate PW & horizon mapping per rule, up to its allowed horizon period.

    This function:
    1. Sorts future time periods using a key and assigns a sequential horizon number.
    2. Cross joins each rule with future time periods (up to max defined horizon).
    3. Filters valid horizon windows per rule.
    4. Merges with future time and get PW and horizon mapping .
    Args:
        time_level (str): Time Level (e.g., 'Time.[Month]').
        time_level_key (str): Time Level Key column (e.g., 'Time.[MonthKey]').
        dataframe (pd.DataFrame): DataFrame of rules with max horizon per rule.
        future_time_master (pd.DataFrame): Time master for future periods.
        grain (List[str]): Final set of columns to return (e.g., version, rule, PW, etc.).
    """
    try:
        time_master = future_time_master[[time_level, time_level_key]].drop_duplicates()
        time_master_sorted = time_master.sort_values(by=time_level_key).reset_index(drop=True)
        time_master_sorted["Horizon"] = time_master_sorted.index + 1

        max_horizon = dataframe[Constants.candidate_horizon_period_by_rule].max()
        time_slice = time_master_sorted[time_master_sorted["Horizon"] <= max_horizon].copy()

        dataframe_tmp = dataframe.copy()
        dataframe_tmp["__key"] = 1
        time_slice["__key"] = 1

        cross_joined = pd.merge(dataframe_tmp, time_slice, on="__key", how="inner").drop(
            columns="__key"
        )

        valid_horizon_df = cross_joined[
            cross_joined["Horizon"] <= cross_joined[Constants.candidate_horizon_period_by_rule]
        ]

        final_df = valid_horizon_df.merge(
            future_time_master, on=[time_level, time_level_key], how="left"
        )

        output = final_df[grain]
        return output

    except Exception as e:
        raise Exception(f"Error in generate_horizon_pw_mapping: {e}") from e


def convert_cols_to_data_object_lvl(
    dataframe,
    item_master,
    account_master,
    channel_master,
    region_master,
    location_master,
    pnl_master,
    demand_domain_master,
    assortment,
    common_cols,
    grain,
):
    """Transform columns to the planning level.

    Apply mapping and rename columns as per data object standards.
    Steps:
    1. Iterates through each dimensional level (item, account, etc.).
    2. Checks if the raw-level column (e.g., 'Item') differs from the planning-level column (e.g., 'PL_Item').
    3. Joins the appropriate master to map raw column to planning column.
    4. Filters rows to ensure only valid planning intersections remain.

    Args:
        dataframe (pd.DataFrame): Input data containing raw scope-level dimensions and rule metadata
        item_master (pd.DataFrame): Master mapping for items
        account_master (pd.DataFrame): Master mapping for accounts
        channel_master (pd.DataFrame): Master mapping for channels
        region_master (pd.DataFrame): Master mapping for regions
        location_master (pd.DataFrame): Master mapping for locations
        pnl_master (pd.DataFrame): Master mapping for PnL
        demand_domain_master (pd.DataFrame): Master mapping for demand domains
        assortment (pd.DataFrame): Flagged assortment data at planning level
        common_cols (List[str]): List of raw-level columns in order (item, account, etc.)
        grain (List[str]): Final set of columns to retain (typically planning-level grain)
    """
    try:
        merge_col = [
            Constants.data_object_col,
            Constants.dm_rule_col,
            Constants.cluster_col,
            Constants.process_order_col,
        ]

        master_map = [
            (item_master, Constants.pl_item_col, common_cols[0]),
            (account_master, Constants.pl_account_col, common_cols[1]),
            (channel_master, Constants.pl_channel_col, common_cols[2]),
            (region_master, Constants.pl_region_col, common_cols[3]),
            (location_master, Constants.pl_location_col, common_cols[4]),
            (pnl_master, Constants.pl_pnl_col, common_cols[5]),
            (demand_domain_master, Constants.pl_demand_domain_col, common_cols[6]),
        ]

        for master_df, pl_col, raw_col in master_map:
            if pl_col != raw_col:
                temp_df = master_df[[raw_col, pl_col]].drop_duplicates()
                assortment = assortment.merge(temp_df, on=pl_col, how="left").dropna()

                valid_merge_cols = [col for col in merge_col if col in assortment.columns]
                assortment = assortment.merge(
                    dataframe[[raw_col] + merge_col].drop_duplicates(),
                    on=[raw_col] + valid_merge_cols,
                    how="inner",
                ).dropna()

        missing_cols = [col for col in grain if col not in assortment.columns]

        if missing_cols:
            return pd.DataFrame(columns=grain)

        return assortment[grain].drop_duplicates()

    except Exception as e:
        raise Exception(f"Error in convert_cols_to_data_object_lvl: {e}") from e


def convert_cols_to_planning_lvl(
    candidate_touchless_params_df,
    ItemMaster,
    AccountMaster,
    ChannelMaster,
    RegionMaster,
    LocationMaster,
    PnLMaster,
    DemandDomainMaster,
    AssortmentFlag,
    cross_joined_data,
    pl_lvl_version_cols,
    scope_lvl_cols,
    do_lvl_cols,
):
    """Transform a scope-level input DataFrame (candidate_touchless_params_df) into a planning-level.

    # DataFrame by:
    #   1. Grouping the data by each unique 'Data Object'.
    #   2. Extracting relevant level information (e.g., item, account, location) from the first row.
    #   3. Replacing scope-level columns with planning-level columns dynamically.
    #   4. Joining with corresponding master datasets (e.g., ItemMaster, AccountMaster).
    #   5. Delegating transformation logic to `convert_cols_to_data_object_lvl`.
    #   6. Collecting and concatenating results across all data objects.
    #   7. Joining the final result with a planning-level cross-join reference dataset.
    """
    try:
        dataframes_dict = {
            data_object: candidate_touchless_params_df[
                candidate_touchless_params_df[Constants.data_object_col] == data_object
            ]
            for data_object in candidate_touchless_params_df[Constants.data_object_col].unique()
        }

        planning_combination_results_list = []

        for data_object, dataframe in dataframes_dict.items():
            first_row = dataframe.iloc[0]

            account_col = first_row[Constants.do_account_lvl_col].replace("]", "$DisplayName]")
            channel_col = first_row[Constants.do_channel_lvl_col].replace("]", "$DisplayName]")
            demand_domain_col = first_row[Constants.do_demand_domain_lvl_col].replace(
                "]", "$DisplayName]"
            )
            item_col = first_row[Constants.do_item_lvl_col].replace("]", "$DisplayName]")
            location_col = first_row[Constants.do_location_lvl_col].replace("]", "$DisplayName]")
            pnl_col = first_row[Constants.do_pnl_lvl_col].replace("]", "$DisplayName]")
            region_col = first_row[Constants.do_region_lvl_col].replace("]", "$DisplayName]")

            common_cols = [
                item_col,
                account_col,
                channel_col,
                region_col,
                location_col,
                pnl_col,
                demand_domain_col,
            ]

            dataframe = dataframe.drop(do_lvl_cols, axis=1)
            rename_dict = dict(zip(scope_lvl_cols, common_cols))
            dataframe = dataframe.rename(columns=rename_dict)

            item_master = ItemMaster[[item_col, Constants.pl_item_col]].drop_duplicates()
            account_master = AccountMaster[
                [account_col, Constants.pl_account_col]
            ].drop_duplicates()
            channel_master = ChannelMaster[
                [channel_col, Constants.pl_channel_col]
            ].drop_duplicates()
            region_master = RegionMaster[[region_col, Constants.pl_region_col]].drop_duplicates()
            location_master = LocationMaster[
                [location_col, Constants.pl_location_col]
            ].drop_duplicates()
            pnl_master = PnLMaster[[pnl_col, Constants.pl_pnl_col]].drop_duplicates()
            demand_domain_master = DemandDomainMaster[
                [demand_domain_col, Constants.pl_demand_domain_col]
            ].drop_duplicates()

            output_common_cols = pl_lvl_version_cols + [
                Constants.data_object_col,
                Constants.dm_rule_col,
                Constants.cluster_col,
                Constants.process_order_col,
            ]

            dataframe = convert_cols_to_data_object_lvl(
                dataframe,
                item_master,
                account_master,
                channel_master,
                region_master,
                location_master,
                pnl_master,
                demand_domain_master,
                AssortmentFlag,
                common_cols,
                output_common_cols,
            )

            if not dataframe.empty:
                planning_combination_results_list.append(dataframe.dropna())

        if planning_combination_results_list:
            merged_all_df = pd.concat(planning_combination_results_list, ignore_index=True)
        else:
            return pd.DataFrame()

        merged_all_df = merged_all_df.drop(Constants.data_object_col, axis=1)

        output = merged_all_df.merge(
            cross_joined_data, on=pl_lvl_version_cols + [Constants.cluster_col], how="inner"
        )

        return output

    except Exception as e:
        raise Exception(f"Error in convert_cols_to_planning_lvl: {e}") from e


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ProcessOrder,
    RuleScope,
    AssortmentFlag,
    SegmentationInput,
    AccountMaster,
    ChannelMaster,
    DemandDomainMaster,
    ItemMaster,
    LocationMaster,
    PnLMaster,
    RegionMaster,
    CandidateFcst,
    HorizonRule,
    CurrentTimePeriod,
    TimeMaster,
    Horizon,
    ReferenceFcst,
    CannibalizationImpactFlag,
    CurrentPlanningCycle,
    LastDay,
    CannibalizationImpactCandidate,
    df_keys,
):
    """Generate baseline forecasts.

    This module contains logic to generate baseline forecasts in the CDP flow.
    It includes functions for horizon mapping, data object conversions, and baseline calculation.
    """
    plugin_name = "DP087CDPBaselineCalculation"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    baseline_fcst = pd.DataFrame(columns=Constants.cols_required_in_output1)
    final_all_scope_combinations_df = pd.DataFrame(columns=Constants.cols_required_in_output2)

    try:

        # --- Input DataFrames dictionary for validation ---
        required_dfs = {
            "Item Master": ItemMaster,
            "Account Master": AccountMaster,
            "Channel Master": ChannelMaster,
            "Region Master": RegionMaster,
            "Demand Domain Master": DemandDomainMaster,
            "PnL Master": PnLMaster,
            "Time Master": TimeMaster,
            "Horizon": Horizon,
            "Horizon Rule": HorizonRule,
            "Process Order": ProcessOrder,
            "Rule Scope": RuleScope,
            "Reference Fcst": ReferenceFcst,
        }

        for name, df in required_dfs.items():
            if df.empty:
                raise Exception(f"{name} input has no data. Exiting without execution.")

        touch_baseline_fcst = pd.DataFrame()
        finalized_rule_touch = pd.DataFrame()
        all_scope_combinations = pd.DataFrame()
        touchless_baseline_fcst = pd.DataFrame()
        touchless_df = pd.DataFrame()

        # Keep only future time periods from TimeMaster
        if not CurrentPlanningCycle.empty:
            start_key = CurrentPlanningCycle[Constants.planning_cycle_key].iloc[0]
        else:
            start_key = CurrentTimePeriod[Constants.day_key_col].iloc[0]

        end_key = LastDay[Constants.day_key_col].iloc[-1]

        future_time_period = TimeMaster[
            (TimeMaster[Constants.pw_key_col] >= start_key)
            & (TimeMaster[Constants.pw_key_col] <= end_key)
        ]

        total_partial_week = len(future_time_period)

        AssortmentFlag[Constants.is_already_touchless] = AssortmentFlag[
            Constants.is_already_touchless
        ].fillna("N")

        rename_dict = dict(
            zip(
                ReferenceFcst[Constants.reference_fcst_for_cluster],
                ReferenceFcst[Constants.cluster_col],
            )
        )

        existing_fcst_cols = [col for col in rename_dict if col in SegmentationInput.columns]
        selected_cols = Constants.pl_lvl_version_cols + existing_fcst_cols
        SegmentationInput = SegmentationInput[selected_cols].copy()

        # Apply user-defined column names
        segmentation_flag = SegmentationInput.rename(columns=rename_dict)

        # move the column into rows where the fcst number is not null
        stacked = segmentation_flag.set_index(Constants.pl_lvl_version_cols)
        segmented_data = stacked.stack().reset_index()
        segmented_data.columns = Constants.pl_lvl_version_cols + [
            Constants.cluster_col,
            Constants.segmentation_flag_col,
        ]
        segmented_data = segmented_data.dropna(subset=[Constants.segmentation_flag_col])
        segmented_data = segmented_data.drop(Constants.segmentation_flag_col, axis=1)

        # split the values in the column by commas
        RuleScope = RuleScope.rename(columns={Constants.forecast_type_col: Constants.cluster_col})
        RuleScope[Constants.scope_lvl_cols] = RuleScope[Constants.scope_lvl_cols].apply(
            lambda col: col.str.split(",")
        )

        # explode all column into separate rows
        for col in Constants.scope_lvl_cols:
            RuleScope = RuleScope.explode(column=col, ignore_index=True)

        # This applies the function to each element in the Series individually
        for col in Constants.do_lvl_cols:
            ProcessOrder[col] = ProcessOrder[col].map(remove_first_brackets)

        # assign process order to rules for exceptions parameters
        candidate_params_df = pd.merge(
            RuleScope,
            ProcessOrder,
            on=[Constants.version_col, Constants.data_object_col],
            how="left",
        )

        # Drop rows where the 'process_order_col' is null
        candidate_params_df = candidate_params_df.dropna(subset=[Constants.process_order_col])

        # Extract unique dm_rule values as a DataFrame for merging
        list_rule = candidate_params_df[[Constants.dm_rule_col]].drop_duplicates()

        # Merge list_rule with HorizonRule on dm_rule_col, keeping all rules from list_rule
        HorizonRule = list_rule.merge(HorizonRule, on=Constants.dm_rule_col, how="left")

        # Join HorizonRule with Horizon to enrich rules with bucket info and fallback horizon settings
        horizon_bucket = pd.merge(
            HorizonRule,
            Horizon,
            on=[Constants.version_col, Constants.data_validation_col],
            how="left",
        )

        # Adjust horizon_bucket for last horizon rows
        mask_last = horizon_bucket[Constants.is_last_horizon] == 1
        horizon_bucket.loc[mask_last, Constants.horizon_candidate_fcst] = horizon_bucket[
            Constants.outside_horizon_candidate_fcst
        ]
        horizon_bucket.loc[mask_last, Constants.consensus_horizon_bucket] = (
            Constants.partial_week_col
        )
        horizon_bucket.loc[mask_last, Constants.candidate_horizon_period_by_rule] = (
            total_partial_week
        )
        horizon_bucket.loc[
            horizon_bucket[Constants.candidate_horizon_period_by_rule].isna(),
            Constants.candidate_horizon_period_by_rule,
        ] = horizon_bucket[Constants.consensus_horizon_period]

        # keep necessary columns
        horizon_bucket = horizon_bucket.dropna(subset=[Constants.horizon_candidate_fcst])
        horizon_bucket_df = horizon_bucket[Constants.horizon_required_cols].drop_duplicates()
        horizon_bucket_df[Constants.horizon_candidate_fcst] = clean_measure_name(
            horizon_bucket_df[Constants.horizon_candidate_fcst]
        )

        # Group by candidate horizon bucket for horizon mapping generation
        dataframes_horizon_dict = {
            candidate: df
            for candidate, df in horizon_bucket_df.groupby(Constants.consensus_horizon_bucket)
        }
        horizon_results_list = []

        for df in dataframes_horizon_dict.values():

            # Determine time_col and time_col_key
            time_col = f"Time.[{df.iloc[0][Constants.consensus_horizon_bucket]}]"
            time_col_key = time_col.replace("]", "Key]").replace(" ", "")

            # Handle case where time_col is same as Partial Week
            if time_col == Constants.pw_col:
                time_master_cols = [time_col, time_col_key]
            else:
                time_master_cols = [time_col, time_col_key, Constants.pw_col]

            # Disaggregate to Partial Week
            dataframe = generate_horizon_pw_mapping(
                time_col,
                time_col_key,
                df,
                future_time_period[time_master_cols],
                Constants.horizon_grain,
            )

            if not dataframe.empty:
                horizon_results_list.append(dataframe)

        # Retain first horizon per PW per rule
        # A single, fast concatenation after the loop
        if horizon_results_list:
            null_horizon_df = pd.concat(horizon_results_list, ignore_index=True)
        else:
            raise Exception("No Valid Horizon. Exiting without execution......")

        # list of the horizon
        list_of_horizon_df = pd.DataFrame(Horizon[Constants.data_validation_col])

        # create a temporary 'key' column for the cross join
        list_of_horizon_df["key"] = 1
        segmented_data["key"] = 1

        # perform the cross join
        cross_joined_data = pd.merge(list_of_horizon_df, segmented_data, on="key").drop(
            "key", axis=1
        )

        candidate_touchless_params_df = candidate_params_df[
            candidate_params_df[Constants.dm_rule_col].str.contains("CCID_Touchless", na=False)
        ]
        touchless_assortment_flag = AssortmentFlag[
            AssortmentFlag[Constants.is_already_touchless] == "Y"
        ]

        candidate_touch_params_df = candidate_params_df[
            ~(candidate_params_df[Constants.dm_rule_col].str.contains("CCID_Touchless", na=False))
        ]
        touch_assortment_flag = AssortmentFlag[
            AssortmentFlag[Constants.is_already_touchless] == "N"
        ]

        if not candidate_touchless_params_df.empty and not touchless_assortment_flag.empty:
            touchless_df = convert_cols_to_planning_lvl(
                candidate_touchless_params_df,
                ItemMaster,
                AccountMaster,
                ChannelMaster,
                RegionMaster,
                LocationMaster,
                PnLMaster,
                DemandDomainMaster,
                touchless_assortment_flag,
                cross_joined_data,
                Constants.pl_lvl_version_cols,
                Constants.scope_lvl_cols,
                Constants.do_lvl_cols,
            )

            if not touchless_df.empty:
                all_scope_combinations = pd.concat(
                    [all_scope_combinations, touchless_df], ignore_index=True
                )
                # Set 'Is Finalised' flag to 1
                touchless_df[Constants.is_finalized_rule] = 1
                touchless_df.drop(
                    columns=[Constants.process_order_col, Constants.cluster_col], inplace=True
                )

                # Extract source candidate values for intersections and keep necessary columns
                touchless_horizon_df = pd.merge(
                    touchless_df,
                    null_horizon_df,
                    on=[
                        Constants.version_col,
                        Constants.dm_rule_col,
                        Constants.data_validation_col,
                    ],
                    how="left",
                )[
                    Constants.pl_lvl_version_cols
                    + [
                        Constants.dm_rule_col,
                        Constants.data_validation_col,
                        Constants.pw_col,
                        Constants.horizon_candidate_fcst,
                    ]
                ].drop_duplicates()

                # Clean Measure name by removing 'Measure.' prefix and brackets (e.g., 'Measure.[Stat Fcst]' → 'Stat Fcst')
                touchless_horizon_df[Constants.horizon_candidate_fcst] = clean_measure_name(
                    touchless_horizon_df[Constants.horizon_candidate_fcst]
                )

                # Merge with CandidateFcst to get final forecasts
                touchless_baseline_fcst = pd.merge(
                    touchless_horizon_df,
                    CandidateFcst,
                    on=Constants.pl_lvl_version_cols + [Constants.pw_col],
                    how="left",
                )

                # Create baseline forecast by extracting value from candidate horizon fcst column
                touchless_baseline_fcst[Constants.baseline_fcst] = touchless_baseline_fcst[
                    Constants.stat_fcst_l0
                ]

                touchless_baseline_fcst = touchless_baseline_fcst.sort_values(
                    by=[Constants.data_validation_col]
                ).drop_duplicates(
                    subset=Constants.pl_lvl_version_cols
                    + [Constants.pw_col, Constants.dm_rule_col],
                    keep="first",
                )
                touchless_df = touchless_df[
                    Constants.pl_lvl_version_cols
                    + [
                        Constants.data_validation_col,
                        Constants.dm_rule_col,
                        Constants.is_finalized_rule,
                    ]
                ].drop_duplicates()

        if not candidate_touch_params_df.empty and not touch_assortment_flag.empty:
            touch_df = convert_cols_to_planning_lvl(
                candidate_touch_params_df,
                ItemMaster,
                AccountMaster,
                ChannelMaster,
                RegionMaster,
                LocationMaster,
                PnLMaster,
                DemandDomainMaster,
                touch_assortment_flag,
                cross_joined_data,
                Constants.pl_lvl_version_cols,
                Constants.scope_lvl_cols,
                Constants.do_lvl_cols,
            )
            if not touch_df.empty:
                all_scope_combinations = pd.concat(
                    [all_scope_combinations, touch_df], ignore_index=True
                )

                # Merge all necessary data together first
                merged_horizon_df = pd.merge(
                    touch_df,
                    null_horizon_df,
                    on=[
                        Constants.version_col,
                        Constants.dm_rule_col,
                        Constants.data_validation_col,
                    ],
                    how="left",
                )

                prelim_fcst_df = pd.merge(
                    merged_horizon_df,
                    CandidateFcst,
                    on=Constants.pl_lvl_version_cols + [Constants.pw_col],
                )

                # Vectorized lookup using melt - this replaces the slow .apply()
                id_cols = Constants.pl_lvl_version_cols + [
                    Constants.pw_col,
                    Constants.dm_rule_col,
                    Constants.data_validation_col,
                    Constants.process_order_col,
                    Constants.consensus_horizon_bucket,
                    Constants.candidate_horizon_period_by_rule,
                    Constants.is_last_horizon,
                ]
                candidate_source_col = Constants.horizon_candidate_fcst
                forecast_value_cols = [
                    col
                    for col in CandidateFcst.columns
                    if col not in (Constants.pl_lvl_version_cols + [Constants.pw_col])
                ]

                melted_df = prelim_fcst_df.melt(
                    id_vars=id_cols + [candidate_source_col],
                    value_vars=forecast_value_cols,
                    var_name="source_col_name",
                    value_name=Constants.baseline_fcst,
                )

                correct_baselines = melted_df[
                    melted_df[candidate_source_col] == melted_df["source_col_name"]
                ]

                # This DataFrame now has only intersections with a valid forecast
                prelim_fcst_df = pd.merge(
                    prelim_fcst_df.drop(columns=forecast_value_cols),
                    correct_baselines,
                    on=id_cols + [candidate_source_col],
                    how="left",
                ).dropna(subset=[Constants.baseline_fcst])

                # --- Step 2 & 3: Prioritize Rules and Select the Final Forecast ---

                if not prelim_fcst_df.empty:
                    # Find the highest process order for each intersection
                    max_process = prelim_fcst_df.groupby(
                        Constants.pl_lvl_version_cols + [Constants.data_validation_col]
                    )[Constants.process_order_col].transform("max")

                    filtered_df = prelim_fcst_df[
                        prelim_fcst_df[Constants.process_order_col] == max_process
                    ].copy()

                    # Of those, find the highest rule number
                    filtered_df[Constants.dm_rule_num] = (
                        filtered_df[Constants.dm_rule_col].str.split("CCID_").str[1].astype(int)
                    )
                    max_dm_rule_num = filtered_df.groupby(
                        Constants.pl_lvl_version_cols + [Constants.data_validation_col]
                    )[Constants.dm_rule_num].transform("max")

                    # The final result for touched records, with the forecast already calculated
                    finalized_rule_touch = filtered_df[
                        filtered_df[Constants.dm_rule_num] == max_dm_rule_num
                    ].copy()
                    finalized_rule_touch[Constants.is_finalized_rule] = 1

                    # Final cleanup to match expected columns for concatenation later
                    touch_baseline_fcst = finalized_rule_touch.sort_values(
                        by=[Constants.data_validation_col]
                    ).drop_duplicates(
                        subset=Constants.pl_lvl_version_cols + [Constants.pw_col], keep="first"
                    )
                    finalized_rule_touch = finalized_rule_touch[
                        Constants.pl_lvl_version_cols
                        + [
                            Constants.data_validation_col,
                            Constants.dm_rule_col,
                            Constants.is_finalized_rule,
                        ]
                    ].drop_duplicates()

        # Concatenate touchless baseline fcst and touch baseline fcst for baseline fcst
        if touchless_baseline_fcst.empty and touch_baseline_fcst.empty:
            logger.warning("No Valid Forecast. Exiting without execution. ...")
            return baseline_fcst, final_all_scope_combinations_df

        baseline_fcst_intermediate = pd.concat(
            [touchless_baseline_fcst, touch_baseline_fcst], ignore_index=True
        )
        baseline_fcst_intermediate.rename(
            columns={Constants.horizon_candidate_fcst: Constants.source_candidate}, inplace=True
        )

        # Concatenate touchless_df and touch_df for final rule set
        if touchless_df.empty and finalized_rule_touch.empty:
            logger.warning("No Valid Rule. Exiting without execution. ...")
            return baseline_fcst, final_all_scope_combinations_df
        finalized_rule_df = pd.concat([touchless_df, finalized_rule_touch], ignore_index=True)

        # Merge finalized rules with all possible combinations of scope
        final_all_scope_combinations_df_intermediate = pd.merge(
            all_scope_combinations[
                Constants.pl_lvl_version_cols
                + [Constants.data_validation_col, Constants.dm_rule_col]
            ].drop_duplicates(),
            finalized_rule_df,
            on=Constants.pl_lvl_version_cols
            + [Constants.data_validation_col, Constants.dm_rule_col],
            how="left",
        )

        # Merge with horizon_bucket for final horizon info
        final_all_scope_combinations_df_intermediate = pd.merge(
            final_all_scope_combinations_df_intermediate,
            horizon_bucket,
            on=[Constants.version_col, Constants.dm_rule_col, Constants.data_validation_col],
            how="left",
        )

        # Set horizon period to NaN where it is last horizon
        final_all_scope_combinations_df_intermediate.loc[
            final_all_scope_combinations_df_intermediate[Constants.is_last_horizon] == 1,
            Constants.candidate_horizon_period_by_rule,
        ] = np.nan

        # Rename horizon columns
        final_all_scope_combinations_df_intermediate = (
            final_all_scope_combinations_df_intermediate[
                Constants.pl_lvl_version_cols
                + [Constants.data_validation_col, Constants.dm_rule_col]
                + [
                    Constants.is_finalized_rule,
                    Constants.horizon_candidate_fcst,
                    Constants.candidate_horizon_period_by_rule,
                ]
            ]
            .rename(
                columns={
                    Constants.candidate_horizon_period_by_rule: Constants.planning_horizon_period,
                    Constants.horizon_candidate_fcst: Constants.planning_horizon_fcst,
                }
            )
            .drop_duplicates()
        )

        final_all_scope_combinations_df_intermediate[Constants.is_finalized_rule] = (
            final_all_scope_combinations_df_intermediate[Constants.is_finalized_rule]
            .fillna(0)
            .astype(int)
        )

        baseline_fcst_intermediate = AssortmentFlag.merge(
            baseline_fcst_intermediate, on=Constants.pl_lvl_version_cols, how="left"
        )

        if not CannibalizationImpactFlag.empty:
            baseline_fcst_intermediate = baseline_fcst_intermediate.merge(
                CannibalizationImpactFlag,
                on=Constants.pl_lvl_version_cols + [Constants.pw_col],
                how="left",
            )
        else:
            baseline_fcst_intermediate[Constants.cannibalization_impact_flag] = ""

        # ----------------------------------------------------------------
        # Calculate the Cannibalization Impact flag based on user-selected candidate forecast
        # ----------------------------------------------------------------

        # Ensure column exists and has no NaN before .isin()
        source_col = baseline_fcst_intermediate[Constants.source_candidate].fillna("")

        # List of candidates for which cannibalization needs to be included:
        candidate_str = CannibalizationImpactCandidate[Constants.cannibalization_candidate].iloc[0]

        # Convert the string to a list
        list_of_candidate = [x.strip() for x in candidate_str.split(",") if x.strip()]

        # Apply the condition
        condition_1 = source_col.isin(list_of_candidate)

        # Apply the flag assignment using np.where
        baseline_fcst_intermediate[Constants.cannibalization_impact_flag] = np.where(
            condition_1, "Y(System)", "N(System)"
        )

        # Prepare Output1 and Output2 DataFrames with proper columns

        baseline_fcst_intermediate = baseline_fcst_intermediate[
            Constants.cols_required_in_output1
        ].dropna()

        baseline_fcst = baseline_fcst_intermediate[Constants.cols_required_in_output1]
        final_all_scope_combinations_df = final_all_scope_combinations_df_intermediate[
            Constants.cols_required_in_output2
        ]

        logger.info(f"Successfully executed {plugin_name} ...")

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return baseline_fcst, final_all_scope_combinations_df

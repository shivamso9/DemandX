"""This plugin publishes cannibalization impact from planning level to published level.

1. INPUT GATHERING
    - Get combinations of initiatives that are ready to be published
    - Get current date for processing
    - Get cannibalization independence dates for initiatives
    - Get initiative level status (whether approved or not)
    - Get planning level cannibalization impacts
    - Get previously published impacts
    - Get statistical forecast data

2. PROCESSING FLOW
    a) For each initiative combination:
        - Check if initiative is approved for publishing
        - Verify if current date >= cannibalization independence date
        - If both conditions met, proceed to publishing

    b) For approved initiatives:
        - Take planning level cannibalization impacts
        - Aggregate impacts across relevant business dimensions
        - Validate against statistical forecast to ensure reasonable values
        - Compare with previously published impacts to identify changes

    c) Publishing Logic:
        - For new impacts: Add to published level
        - For existing impacts: Update with new values
        - For expired impacts: Remove from published level

3. OUTPUT
    - Save final published cannibalization impacts back to the system
    - These impacts will be used in downstream planning processes
"""

import logging

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


col_mapping = {
    "Cannibalization Impact Published": float,
}


def find_and_merge_uncommon_rows(published_df, final_result, planning_cols):
    """
    Find rows that exist in published_df but not in final_result and merge them.

    Args:
        published_df: DataFrame containing published impact data
        final_result: DataFrame containing final result data
        planning_cols: List of common columns to compare between dataframes

    Returns:
        DataFrame with merged results including uncommon rows
    """
    if published_df.empty or final_result.empty:
        return final_result

    # Find rows in published_df that don't exist in final_result
    uncommon_mask = ~published_df.set_index(planning_cols).index.isin(
        final_result.set_index(planning_cols).index
    )
    uncommon_rows = published_df[uncommon_mask].copy()

    # Concatenate the uncommon rows with final_result
    merged_result = pd.concat([final_result, uncommon_rows], ignore_index=True)

    return merged_result


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    Combinations,
    CannibIndependenceDate,
    InitiativeLevelStatus,
    PlanningLevelImpact,
    StatFcst,
    PublishedImpact,
    CurrentDate,
    df_keys,
):
    """DP135PublishCannibalizationImpact plugin."""
    plugin_name = "DP135PublishCannibalizationImpact"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Initialize PublishedImpact_output with empty DataFrame
    cols_req_in_output = [
        "Version.[Version Name]",
        "Item.[Planning Item]",
        "Account.[Planning Account]",
        "Channel.[Planning Channel]",
        "Region.[Planning Region]",
        "PnL.[Planning PnL]",
        "Location.[Planning Location]",
        "Demand Domain.[Planning Demand Domain]",
        "Time.[Partial Week]",
        "Cannibalization Impact Published",
    ]
    PublishedImpact_output = pd.DataFrame(columns=cols_req_in_output)

    try:
        # Configurables
        version_col = "Version.[Version Name]"
        data_obj = "Data Object.[Data Object]"
        initiative_col = "Initiative.[Initiative]"
        item_col = "Item.[Planning Item]"
        location_col = "Location.[Planning Location]"
        account_col = "Account.[Planning Account]"
        region_col = "Region.[Planning Region]"
        demand_domain_col = "Demand Domain.[Planning Demand Domain]"
        pnl_col = "PnL.[Planning PnL]"
        channel_col = "Channel.[Planning Channel]"
        partial_week_col = "Time.[Partial Week]"
        cannib_impact_published_col = "Cannibalization Impact Published"
        cannib_impact_planning_level_col = "Planning Level Cannibalization Impact"
        initiative_level_status_col = "NPI Initiative Level Status"
        cannib_independence_date_col = "Cannibalization Independence Date Planning Level"
        stat_fcst_col = "Stat Fcst NPI BB"
        current_date_col = "Time.[DayKey]"
        cannib_item_flag_col = "Cannib Impact Publish Flag L1"

        # List to store results from each iteration
        results_list = []

        # Iterate through each row in Combinations
        for _, combination_row in Combinations.iterrows():
            # Check if flag is 1
            if combination_row[cannib_item_flag_col] != 1:
                logger.warning(
                    f"Cannib Impact Publish Flag is not 1 for combination: {combination_row}"
                )
                continue

            # Filter CannibIndependenceDate based on combination values
            filtered_df = CannibIndependenceDate[
                (CannibIndependenceDate[version_col] == combination_row[version_col])
                & (CannibIndependenceDate[data_obj] == combination_row[data_obj])
                & (CannibIndependenceDate[initiative_col] == combination_row[initiative_col])
            ]

            planning_cols = [
                version_col,
                item_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                location_col,
                demand_domain_col,
            ]

            # Filter PlanningLevelImpact using planning level columns from filtered_df
            planning_impact_filter = PlanningLevelImpact.merge(
                filtered_df[planning_cols], on=planning_cols, how="inner"
            )

            # Get unique initiatives and data objects from planning_impact_filter
            unique_initiatives = planning_impact_filter[initiative_col].unique()
            unique_data_objs = planning_impact_filter[data_obj].unique()

            # Filter by InitiativeLevelStatus using the unique initiatives and data objects
            initiative_status_filter = InitiativeLevelStatus[
                (InitiativeLevelStatus[version_col] == combination_row[version_col])
                & (InitiativeLevelStatus[data_obj].isin(unique_data_objs))
                & (InitiativeLevelStatus[initiative_col].isin(unique_initiatives))
                & (
                    InitiativeLevelStatus[initiative_level_status_col].isin(
                        ["Published", "Completed"]
                    )
                )
            ]

            # Create a DataFrame with the valid initiative and data object pairs
            valid_pairs = initiative_status_filter[[initiative_col, data_obj]].drop_duplicates()

            # Filter planning_impact_filter to only include rows where both initiative and data object match valid pairs
            planning_impact_filter = planning_impact_filter.merge(
                valid_pairs, on=[initiative_col, data_obj], how="inner"
            )

            # Get unique combinations from planning_impact_filter
            unique_combinations = planning_impact_filter[
                planning_cols + [data_obj, initiative_col]
            ].drop_duplicates()

            # Filter CannibIndependenceDate using the unique combinations
            filtered_independence = CannibIndependenceDate.merge(
                unique_combinations, on=planning_cols + [data_obj, initiative_col], how="inner"
            )

            # Filter out rows where independence date is not greater than current date
            current_date = CurrentDate[current_date_col].iloc[0]
            filtered_independence = filtered_independence[
                filtered_independence[cannib_independence_date_col] > current_date
            ]

            # Keep only the rows in planning_impact_filter that match the filtered independence data
            planning_impact_filter = planning_impact_filter.merge(
                filtered_independence[planning_cols + [data_obj, initiative_col]],
                on=planning_cols + [data_obj, initiative_col],
                how="inner",
            )

            # Drop unnecessary columns before aggregation
            planning_impact_filter = planning_impact_filter.drop([data_obj, initiative_col], axis=1)

            # Group by all planning level columns AND partial week, then sum the impact values
            group_by_cols = planning_cols + [partial_week_col]
            aggregated_impact = (
                planning_impact_filter.groupby(
                    group_by_cols,
                    observed=True,  # For better performance with categorical data
                    dropna=False,  # Keep NA values in grouping
                )[cannib_impact_planning_level_col]
                .sum()
                .reset_index()
            )

            # Rename column
            aggregated_impact = aggregated_impact.rename(
                columns={cannib_impact_planning_level_col: cannib_impact_published_col}
            )

            # Filter with StatFcst and compare values
            final_result = aggregated_impact.merge(
                StatFcst[group_by_cols + [stat_fcst_col]], on=group_by_cols, how="inner"
            )

            # to do if the merge is empty throw an logger.warning that merge is empty
            if final_result.empty:
                logger.warning(
                    f"Merge is empty, it may be stat fcst is not there for the selceted combination.: {combination_row}"
                )
                continue

            # Replace values where Cannibalization Impact Published > Stat Fcst NPI BB
            mask = final_result[cannib_impact_published_col] > final_result[stat_fcst_col]
            final_result.loc[mask, cannib_impact_published_col] = final_result.loc[
                mask, stat_fcst_col
            ]

            # Drop Stat Fcst NPI BB column
            final_result = final_result.drop(stat_fcst_col, axis=1)

            # Handle PublishedImpact if not empty
            if not PublishedImpact.empty:
                # Make a copy of PublishedImpact
                published_filter = PublishedImpact.copy()

                # Filter based on planning columns intersection
                published_filter = published_filter.merge(
                    filtered_df[planning_cols], on=planning_cols, how="inner"
                )

                # Keep only required columns
                published_filter = published_filter[
                    [*planning_cols, partial_week_col, cannib_impact_published_col]
                ]

                # Set all values of the column 'Cannibalization Impact Published' to null
                published_filter[cannib_impact_published_col] = None

                # Merge uncommon rows between published_filter and final_result
                final_result = find_and_merge_uncommon_rows(
                    published_filter, final_result, planning_cols + [partial_week_col]
                )

                # Append to results list
                results_list.append(final_result)
            else:
                # If PublishedImpact is empty, directly append the final_result
                results_list.append(final_result)

        # After the loop, combine all results
        if results_list:
            PublishedImpact_output = pd.concat(results_list, ignore_index=True)
        else:
            # If no results were processed, return empty DataFrame with required columns
            PublishedImpact_output = pd.DataFrame(columns=cols_req_in_output)

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        PublishedImpact_output = pd.DataFrame(columns=cols_req_in_output)

    return PublishedImpact_output

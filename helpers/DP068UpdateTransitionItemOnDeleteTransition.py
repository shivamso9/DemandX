"""This module contains functions to update transition items on delete transition."""

import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


def get_selected_values(df, column):
    """Extract selected values from a DataFrame column."""
    result = df[column].iloc[0].replace('"', "").split(",")
    result = [x.strip() for x in result]  # remove leading/trailing spaces if any
    return result


def get_selected_values_df(column, data):
    """Create a DataFrame with a single column containing selected values."""
    return pd.DataFrame(columns=[column], data=data)


def get_previous_transition(
    planning_item: str,
    TO_PLANNING_ITEM: str,
    FROM_REGION_GROUP: str,
    FROM_PnL_GROUP: str,
    FROM_DEMAND_DOMAIN_GROUP: str,
    FROM_LOCATION_GROUP: str,
    FROM_CHANNEL_GROUP: str,
    FROM_ACCOUNT_GROUP: str,
    FROM_PLANNING_ITEM: str,
    TransitionFlag,
    parent_transitions=None,
):
    """Recursively find all previous transitions for a given planning item."""
    if parent_transitions is None:
        parent_transitions = list()

    # check if the from planning item was involved in previous transitions
    prev_transition_filter = (TransitionFlag[TO_PLANNING_ITEM] == planning_item) | (
        TransitionFlag[FROM_PLANNING_ITEM] == planning_item
    )
    prev_transition_row = TransitionFlag[prev_transition_filter]

    TransitionFlag = TransitionFlag.merge(
        prev_transition_row,
        how="left",
        indicator=True,
    )

    TransitionFlag = TransitionFlag[TransitionFlag["_merge"] == "left_only"]
    TransitionFlag.drop("_merge", axis=1, inplace=True)

    existing_value_filter = (TransitionFlag[TO_PLANNING_ITEM].isin(parent_transitions)) | (
        TransitionFlag[FROM_PLANNING_ITEM].isin(parent_transitions)
    )
    TransitionFlag = TransitionFlag[~existing_value_filter]
    if prev_transition_row.empty:
        pass
    else:
        for _, row in prev_transition_row.iterrows():
            parent_transitions.append(row[FROM_PLANNING_ITEM])
            get_previous_transition(
                planning_item=row[FROM_PLANNING_ITEM],
                TO_PLANNING_ITEM=TO_PLANNING_ITEM,
                FROM_REGION_GROUP=FROM_REGION_GROUP,
                FROM_PnL_GROUP=FROM_PnL_GROUP,
                FROM_DEMAND_DOMAIN_GROUP=FROM_DEMAND_DOMAIN_GROUP,
                FROM_LOCATION_GROUP=FROM_LOCATION_GROUP,
                FROM_CHANNEL_GROUP=FROM_CHANNEL_GROUP,
                FROM_ACCOUNT_GROUP=FROM_ACCOUNT_GROUP,
                FROM_PLANNING_ITEM=FROM_PLANNING_ITEM,
                TransitionFlag=TransitionFlag,
                parent_transitions=parent_transitions,
            )
    return parent_transitions


def get_previous_transition_main(
    planning_item: str,
    TO_PLANNING_ITEM: str,
    FROM_REGION_GROUP: str,
    FROM_PnL_GROUP: str,
    FROM_DEMAND_DOMAIN_GROUP: str,
    FROM_LOCATION_GROUP: str,
    FROM_CHANNEL_GROUP: str,
    FROM_ACCOUNT_GROUP: str,
    FROM_PLANNING_ITEM: str,
    TransitionFlag,
):
    """Get previous transitions for a given planning item."""
    parent_transitions = get_previous_transition(
        planning_item=planning_item,
        TO_PLANNING_ITEM=TO_PLANNING_ITEM,
        FROM_REGION_GROUP=FROM_REGION_GROUP,
        FROM_PnL_GROUP=FROM_PnL_GROUP,
        FROM_DEMAND_DOMAIN_GROUP=FROM_DEMAND_DOMAIN_GROUP,
        FROM_LOCATION_GROUP=FROM_LOCATION_GROUP,
        FROM_CHANNEL_GROUP=FROM_CHANNEL_GROUP,
        FROM_ACCOUNT_GROUP=FROM_ACCOUNT_GROUP,
        FROM_PLANNING_ITEM=FROM_PLANNING_ITEM,
        TransitionFlag=TransitionFlag,
    )
    return parent_transitions


def identify_associations(df, from_item, to_item, from_selected, to_selected):
    """Identify items that have associations."""
    df = df[(df[from_item].isin(from_selected)) & (~df[to_item].isin(to_selected))]
    associations = df.groupby(from_item)[to_item].nunique()
    return associations[associations > 0].index


col_mapping = {
    "Intro Date": "datetime64[ns]",
    "Disco Date": "datetime64[ns]",
    "Final Transition Item": str,
    "Transition Item After Set": str,
    "Transition Item Before Set": str,
    "510 Product Transition.[Transition Flag]": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    selectedCombinations,
    TransitionFlag,
    ItemDates,
    TransitionAttributes,
    df_keys,
):
    """Update transition items on delete transition."""
    plugin_name = "DP068UpdateTransitionItemOnDeleteTransition"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    pl_item_col = o9Constants.PLANNING_ITEM
    pl_account_col = o9Constants.PLANNING_ACCOUNT
    pl_channel_col = o9Constants.PLANNING_CHANNEL
    pl_region_col = o9Constants.PLANNING_REGION
    pl_demand_domain_col = o9Constants.PLANNING_DEMAND_DOMAIN
    pl_pnl_col = o9Constants.PLANNING_PNL
    pl_location_col = o9Constants.PLANNING_LOCATION
    from_pl_channel_col = "from.[Channel].[Planning Channel]"
    from_pl_region_col = "from.[Region].[Planning Region]"
    from_pl_account_col = "from.[Account].[Planning Account]"
    from_pl_pnl_col = "from.[PnL].[Planning PnL]"
    from_pl_demand_domain_col = "from.[Demand Domain].[Planning Demand Domain]"
    from_pl_location_col = "from.[Location].[Planning Location]"
    from_pl_item_col = "from.[Item].[Planning Item]"
    to_pl_item_col = "to.[Item].[Planning Item]"

    from_pl_item = "Transition From Planning Item"
    to_pl_item = "Transition To Planning Item"
    selected_pl_account = "Transition Selected Planning Account"
    selected_pl_channel = "Transition Selected Planning Channel"
    selected_pl_region = "Transition Selected Planning Region"
    selected_pl_pnl = "Transition Selected Planning PnL"
    selected_pl_demand_domain = "Transition Selected Planning Demand Domain"
    selected_pl_location = "Transition Selected Planning Location"

    # TransitionDates output measures
    intro_date = "Intro Date"
    disco_date = "Disco Date"

    # TransitionAttributes output measures
    final_transition_item = "Final Transition Item"
    transition_item_before_set = "Transition Item Before Set Product Transition"

    # TransitionFlagOutput output measures
    flag = "510 Product Transition.[Transition Flag]"

    # ItemDates output measures
    assortment_phase_out = "Assortment Phase Out"
    product_transition_start_date = "Product Transition Overlap Start Date"
    assortment_phase_in = "Assortment Phase In"
    actuals = "Actual"

    transition_dates_output_cols = ItemDates.columns
    transition_attributes_output_cols = TransitionAttributes.columns
    transition_flag_output_cols = TransitionFlag.columns

    TransitionDates = pd.DataFrame(columns=transition_dates_output_cols)
    TransitionAttributesOutput = pd.DataFrame(columns=transition_attributes_output_cols)
    TransitionFlagOutput = pd.DataFrame(columns=transition_flag_output_cols)

    try:
        if (len(TransitionFlag) == 0) or (len(selectedCombinations) == 0):
            logger.warning("One or more input/s have no data. Please check the input ...")
            logger.warning("Returning empty dataframes ...")
            return (
                TransitionDates,
                TransitionAttributesOutput,
                TransitionFlagOutput,
            )

        # getting all available values from selectedCombinations
        logger.info("getting selected values")
        selected_old_pl_item = get_selected_values(selectedCombinations, from_pl_item)
        selected_new_pl_item = get_selected_values(selectedCombinations, to_pl_item)
        selected_pl_account_values = get_selected_values(selectedCombinations, selected_pl_account)
        selected_pl_channel_values = get_selected_values(selectedCombinations, selected_pl_channel)
        selected_pl_region_values = get_selected_values(selectedCombinations, selected_pl_region)
        selected_pl_demand_domain_values = get_selected_values(
            selectedCombinations, selected_pl_demand_domain
        )
        selected_pl_location_values = get_selected_values(
            selectedCombinations, selected_pl_location
        )
        selected_pl_pnl_values = get_selected_values(selectedCombinations, selected_pl_pnl)

        # create dataframes
        logger.info("creating dataframes of selected values ...")
        old_pl_item_values_df = get_selected_values_df(pl_item_col, selected_old_pl_item)
        new_pl_item_values_df = get_selected_values_df(pl_item_col, selected_new_pl_item)
        selected_pl_account_values_df = get_selected_values_df(
            pl_account_col, selected_pl_account_values
        )
        selected_pl_channel_values_df = get_selected_values_df(
            pl_channel_col, selected_pl_channel_values
        )
        selected_pl_region_values_df = get_selected_values_df(
            pl_region_col, selected_pl_region_values
        )
        selected_pl_demand_domain_values_df = get_selected_values_df(
            pl_demand_domain_col, selected_pl_demand_domain_values
        )
        selected_pl_location_values_df = get_selected_values_df(
            pl_location_col, selected_pl_location_values
        )
        selected_pl_pnl_values_df = get_selected_values_df(pl_pnl_col, selected_pl_pnl_values)

        # cartesian product of new_pl_item_values_df with available selected planning grains (except from pl_item)
        logger.info("getting all possible intersections ...")
        relevant_df = create_cartesian_product(
            selected_pl_pnl_values_df, selected_pl_account_values_df
        )
        relevant_df = create_cartesian_product(relevant_df, selected_pl_channel_values_df)
        relevant_df = create_cartesian_product(relevant_df, selected_pl_region_values_df)
        relevant_df = create_cartesian_product(relevant_df, selected_pl_demand_domain_values_df)
        relevant_df = create_cartesian_product(relevant_df, selected_pl_location_values_df)

        intersections_to_check = create_cartesian_product(relevant_df, new_pl_item_values_df)

        # copying values to use them later
        transition_flag_intersections = intersections_to_check.copy()

        # getting intersections where transition flag will be nulled out
        logger.info("getting intersections where transition flag will be nulled out ...")
        # rename columns
        col_mapping = {
            pl_item_col: to_pl_item_col,
            pl_account_col: from_pl_account_col,
            pl_region_col: from_pl_region_col,
            pl_channel_col: from_pl_channel_col,
            pl_pnl_col: from_pl_pnl_col,
            pl_demand_domain_col: from_pl_demand_domain_col,
            pl_location_col: from_pl_location_col,
        }

        transition_flag_intersections.rename(columns=col_mapping, inplace=True)
        relevant_intersections = create_cartesian_product(
            old_pl_item_values_df, transition_flag_intersections
        )
        relevant_intersections.rename(columns={pl_item_col: from_pl_item_col}, inplace=True)

        # getting relevant intersections to be nulled out
        TransitionFlagOutput = relevant_intersections.merge(
            TransitionFlag,
        )
        # nulling out transition flag for all intersections
        logger.info("nulling measure transition flag ...")
        TransitionFlagOutput = TransitionFlagOutput[transition_flag_output_cols]
        TransitionFlagOutput[flag] = np.nan

        # Identify items that have multiple associations if item not in selected list
        cols_req = [
            from_pl_account_col,
            from_pl_channel_col,
            from_pl_region_col,
            from_pl_demand_domain_col,
            from_pl_pnl_col,
            from_pl_location_col,
        ]

        intersections_to_update = TransitionFlag.merge(relevant_df.rename(columns=col_mapping))
        multi_assoc_from_items = identify_associations(
            intersections_to_update,
            from_pl_item_col,
            to_pl_item_col,
            selected_old_pl_item,
            selected_new_pl_item,
        )
        multi_assoc_to_items = identify_associations(
            intersections_to_update,
            to_pl_item_col,
            from_pl_item_col,
            selected_new_pl_item,
            selected_old_pl_item,
        )

        items_to_keep_from = set(multi_assoc_from_items)
        items_to_keep_to = set(multi_assoc_to_items)

        items_to_keep = items_to_keep_from.union(items_to_keep_to)

        TransitionDates = ItemDates.merge(relevant_df)
        filter_clause = TransitionDates[pl_item_col].isin(selected_new_pl_item) & ~TransitionDates[
            pl_item_col
        ].isin(items_to_keep)

        # TransitionDates[intro_date] = np.where(filter_clause, pd.NaT, TransitionDates[intro_date])
        # TransitionDates[intro_date] = pd.to_datetime(
        #     TransitionDates[intro_date], infer_datetime_format=True
        # )

        # Split into two conditions based on actuals
        null_actuals_mask = filter_clause & TransitionDates[actuals].isna()
        non_null_actuals_mask = filter_clause & ~TransitionDates[actuals].isna()

        # Case 1: When actuals is null - set both intro_date and assortment_phase_in to NaT/nan
        TransitionDates[intro_date] = np.where(
            null_actuals_mask, pd.NaT, TransitionDates[intro_date]
        )
        TransitionDates[assortment_phase_in] = np.where(
            null_actuals_mask, np.nan, TransitionDates[assortment_phase_in]
        )

        # Case 2: When actuals is not null - only set assortment_phase_in to nan
        TransitionDates[assortment_phase_in] = np.where(
            non_null_actuals_mask, np.nan, TransitionDates[assortment_phase_in]
        )

        # Convert intro_date to datetime and handle timezone
        TransitionDates[intro_date] = pd.to_datetime(TransitionDates[intro_date])
        # Convert timezone-aware to timezone-naive if needed
        if (
            hasattr(TransitionDates[intro_date].dtype, "tz")
            and TransitionDates[intro_date].dtype.tz is not None
        ):
            TransitionDates[intro_date] = TransitionDates[intro_date].dt.tz_localize(None)

        filter_clause = TransitionDates[pl_item_col].isin(selected_old_pl_item) & ~TransitionDates[
            pl_item_col
        ].isin(items_to_keep)
        TransitionDates[disco_date] = np.where(filter_clause, pd.NaT, TransitionDates[disco_date])
        TransitionDates[assortment_phase_out] = np.where(
            filter_clause, np.nan, TransitionDates[assortment_phase_out]
        )
        TransitionDates[product_transition_start_date] = np.where(
            filter_clause, pd.NaT, TransitionDates[product_transition_start_date]
        )
        TransitionDates[disco_date] = pd.to_datetime(TransitionDates[disco_date])
        # Convert timezone-aware to timezone-naive if needed
        if (
            hasattr(TransitionDates[disco_date].dtype, "tz")
            and TransitionDates[disco_date].dtype.tz is not None
        ):
            TransitionDates[disco_date] = TransitionDates[disco_date].dt.tz_localize(None)
        TransitionDates[product_transition_start_date] = pd.to_datetime(
            TransitionDates["Product Transition Overlap Start Date"]
        )
        # Convert timezone-aware to timezone-naive if needed
        if (
            hasattr(TransitionDates[product_transition_start_date].dtype, "tz")
            and TransitionDates[product_transition_start_date].dtype.tz is not None
        ):
            TransitionDates[product_transition_start_date] = TransitionDates[
                product_transition_start_date
            ].dt.tz_localize(None)

        TransitionDates = TransitionDates[transition_dates_output_cols]

        # Droping actuals columns from the TransitionDates DataFrame
        TransitionDates.drop(columns=[actuals], inplace=True, axis=1)

        # filter planning item combinations
        filter_clause = (TransitionFlag[from_pl_item_col].isin(selected_old_pl_item)) & (
            TransitionFlag[to_pl_item_col].isin(selected_new_pl_item)
        )

        relevant_transitions = TransitionFlag[filter_clause]

        # getting count of from and to planning item intersections
        count_df = (
            relevant_transitions.groupby([from_pl_item_col, to_pl_item_col])
            .size()
            .reset_index(name="count")
        )
        relevant_transitions = relevant_transitions.merge(count_df)

        relevant_transitions = relevant_transitions.merge(
            relevant_df.rename(columns=col_mapping),
            indicator=True,
        )

        # getting count based on _merge type, from and to planning item
        count_df = (
            relevant_transitions.groupby([from_pl_item_col, to_pl_item_col, "_merge"])
            .size()
            .reset_index(name="merge count")
        )
        relevant_transitions = relevant_transitions.merge(count_df)

        # getting intersections where count and merge count is same
        relevant_transitions = relevant_transitions[
            relevant_transitions["count"] == relevant_transitions["merge count"]
        ]

        # get groups for which transition flag is 1
        transition_customer_groups = relevant_transitions[cols_req].drop_duplicates()
        transition_customer_groups.reset_index(drop=True, inplace=True)

        transition_customer_groups = transition_customer_groups.merge(
            relevant_df.rename(columns=col_mapping)
        )

        if relevant_transitions.empty:
            logger.warning("No relevant transitions found ...")
            TransitionAttributesOutput = TransitionAttributes.copy()
            return (
                TransitionDates,
                TransitionAttributesOutput,
                TransitionFlagOutput,
            )

        if transition_customer_groups.empty:
            logger.warning(
                "Number of selected combinations do not match combinations for which Transition Flag is 1 ..."
            )
            TransitionAttributesOutput = TransitionAttributes.copy()
            return (
                TransitionDates,
                TransitionAttributesOutput,
                TransitionFlagOutput,
            )

        # remove the transition to delete
        TransitionFlag = TransitionFlag[~filter_clause]

        # transition values inverted
        InvertedTransitionFlag = TransitionFlag.copy()
        InvertedTransitionFlag[from_pl_item_col] = TransitionFlag[to_pl_item_col]
        InvertedTransitionFlag[to_pl_item_col] = TransitionFlag[from_pl_item_col]

        TransitionFlag = pd.concat([TransitionFlag, InvertedTransitionFlag])

        TransitionFlag.drop_duplicates(inplace=True)

        # this will always have only one row
        for name, group in relevant_transitions.groupby(
            cols_req + [from_pl_item_col, to_pl_item_col], observed=True
        ):
            the_from_side_intersection = group[from_pl_item_col].iloc[0]
            the_to_side_intersection = group[to_pl_item_col].iloc[0]

            from_pl_item_parent_combinations = get_previous_transition_main(
                planning_item=the_from_side_intersection,
                TO_PLANNING_ITEM=to_pl_item_col,
                FROM_REGION_GROUP=from_pl_region_col,
                FROM_PnL_GROUP=from_pl_pnl_col,
                FROM_DEMAND_DOMAIN_GROUP=from_pl_demand_domain_col,
                FROM_LOCATION_GROUP=from_pl_location_col,
                FROM_CHANNEL_GROUP=from_pl_channel_col,
                FROM_ACCOUNT_GROUP=from_pl_account_col,
                FROM_PLANNING_ITEM=from_pl_item_col,
                TransitionFlag=TransitionFlag,
            )
            from_pl_item_parent_combinations = list(set(from_pl_item_parent_combinations))

            to_pl_item_parent_combinations = get_previous_transition_main(
                planning_item=the_to_side_intersection,
                TO_PLANNING_ITEM=to_pl_item_col,
                FROM_REGION_GROUP=from_pl_region_col,
                FROM_PnL_GROUP=from_pl_pnl_col,
                FROM_DEMAND_DOMAIN_GROUP=from_pl_demand_domain_col,
                FROM_LOCATION_GROUP=from_pl_location_col,
                FROM_CHANNEL_GROUP=from_pl_channel_col,
                FROM_ACCOUNT_GROUP=from_pl_account_col,
                FROM_PLANNING_ITEM=from_pl_item_col,
                TransitionFlag=TransitionFlag,
            )

            to_pl_item_parent_combinations = list(set(to_pl_item_parent_combinations))

            intersection_of_pl_items = [
                item
                for item in from_pl_item_parent_combinations
                if item in to_pl_item_parent_combinations
            ]

            if len(intersection_of_pl_items) != 0:
                logger.warning(
                    "Common planning items were found between transitions, returning without changes"
                )
            else:
                if from_pl_item_parent_combinations is None:
                    from_pl_item_parent_combinations = []
                from_pl_item_parent_combinations.append(the_from_side_intersection)

                if to_pl_item_parent_combinations is None:
                    to_pl_item_parent_combinations = []
                to_pl_item_parent_combinations.append(the_to_side_intersection)

                from_pl_item_considered = the_from_side_intersection

                # get corresponding transition item
                from_transition_item = TransitionAttributes[
                    TransitionAttributes[pl_item_col] == from_pl_item_considered
                ][transition_item_before_set].iloc[0]
                TransitionAttributes[final_transition_item] = np.where(
                    TransitionAttributes[pl_item_col].isin(from_pl_item_parent_combinations),
                    from_transition_item,
                    TransitionAttributes[final_transition_item],
                )

                # perform the same for to planning item
                to_pl_item_considered = the_to_side_intersection

                to_transition_item = TransitionAttributes[
                    TransitionAttributes[pl_item_col] == to_pl_item_considered
                ][transition_item_before_set].iloc[0]
                TransitionAttributes[final_transition_item] = np.where(
                    TransitionAttributes[pl_item_col].isin(to_pl_item_parent_combinations),
                    to_transition_item,
                    TransitionAttributes[final_transition_item],
                )

        TransitionAttributesOutput = TransitionAttributes

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )

    return TransitionDates, TransitionAttributesOutput, TransitionFlagOutput

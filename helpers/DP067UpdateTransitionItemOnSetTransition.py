"""This module handles transition updates for demand planning items."""

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
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


def get_selected_values(df, column):
    """Extract selected values from a DataFrame column, removing quotes and leading/trailing spaces."""
    result = df[column].iloc[0].replace('"', "").split(",")
    result = [x.strip() for x in result]  # remove leading/trailing spaces if any
    return result


def get_selected_values_df(column, data):
    """Create a DataFrame from the selected values."""
    return pd.DataFrame(columns=[column], data=data)


def get_next_transition(
    planning_item: str,
    from_planning_item_selected: str,
    TO_PLANNING_ITEM: str,
    FROM_REGION_GROUP: str,
    FROM_PnL_GROUP: str,
    FROM_DEMAND_DOMAIN_GROUP: str,
    FROM_LOCATION_GROUP: str,
    FROM_CHANNEL_GROUP: str,
    FROM_ACCOUNT_GROUP: str,
    FROM_PLANNING_ITEM: str,
    TransitionFlag,
    child_transitions=None,
):
    """Recursively finds all next transitions for a given planning item."""
    if child_transitions is None:
        child_transitions = list()

    # check if the from planning item was involved in previous transitions
    next_transition_filter_from = TransitionFlag[FROM_PLANNING_ITEM] == planning_item
    new_transition_row_from = TransitionFlag[next_transition_filter_from]

    # check if the from planning item was involved in previous transitions
    next_transition_filter_to = TransitionFlag[TO_PLANNING_ITEM] == planning_item
    new_transition_row_to = TransitionFlag[next_transition_filter_to]

    if not new_transition_row_to.empty:
        new_transition_row_to = new_transition_row_to[
            ~(
                new_transition_row_to[FROM_PLANNING_ITEM].isin(
                    child_transitions + [from_planning_item_selected]
                )
            )
        ]
        if not new_transition_row_to.empty:
            child_transitions.append(new_transition_row_to[FROM_PLANNING_ITEM].iloc[0])
            get_previous_transition(
                planning_item=new_transition_row_to[FROM_PLANNING_ITEM].iloc[0],
                to_planning_item_selected=from_planning_item_selected,
                TO_PLANNING_ITEM=TO_PLANNING_ITEM,
                FROM_REGION_GROUP=FROM_REGION_GROUP,
                FROM_PnL_GROUP=FROM_PnL_GROUP,
                FROM_DEMAND_DOMAIN_GROUP=FROM_DEMAND_DOMAIN_GROUP,
                FROM_LOCATION_GROUP=FROM_LOCATION_GROUP,
                FROM_CHANNEL_GROUP=FROM_CHANNEL_GROUP,
                FROM_ACCOUNT_GROUP=FROM_ACCOUNT_GROUP,
                FROM_PLANNING_ITEM=FROM_PLANNING_ITEM,
                TransitionFlag=TransitionFlag,
                parent_transitions=child_transitions,
            )

    if not new_transition_row_from.empty:
        new_transition_row_from = new_transition_row_from[
            ~(
                new_transition_row_from[TO_PLANNING_ITEM].isin(
                    child_transitions + [from_planning_item_selected]
                )
            )
        ]
        if not new_transition_row_from.empty:
            child_transitions.append(new_transition_row_from[TO_PLANNING_ITEM].iloc[0])
            get_next_transition(
                planning_item=new_transition_row_from[TO_PLANNING_ITEM].iloc[0],
                from_planning_item_selected=from_planning_item_selected,
                TO_PLANNING_ITEM=TO_PLANNING_ITEM,
                FROM_REGION_GROUP=FROM_REGION_GROUP,
                FROM_PnL_GROUP=FROM_PnL_GROUP,
                FROM_DEMAND_DOMAIN_GROUP=FROM_DEMAND_DOMAIN_GROUP,
                FROM_LOCATION_GROUP=FROM_LOCATION_GROUP,
                FROM_CHANNEL_GROUP=FROM_CHANNEL_GROUP,
                FROM_ACCOUNT_GROUP=FROM_ACCOUNT_GROUP,
                FROM_PLANNING_ITEM=FROM_PLANNING_ITEM,
                TransitionFlag=TransitionFlag,
                child_transitions=child_transitions,
            )

    return child_transitions


def get_previous_transition(
    planning_item: str,
    to_planning_item_selected: str,
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
) -> pd.DataFrame:
    """Recursively finds all previous transitions for a given planning item."""
    if parent_transitions is None:
        parent_transitions = list()

    # check if the from planning item was involved in previous transitions
    prev_transition_filter_to = TransitionFlag[TO_PLANNING_ITEM] == planning_item
    prev_transition_row_to = TransitionFlag[prev_transition_filter_to]

    prev_transition_filter_from = TransitionFlag[FROM_PLANNING_ITEM] == planning_item
    prev_transition_row_from = TransitionFlag[prev_transition_filter_from]

    if not prev_transition_row_from.empty:
        prev_transition_row_from = prev_transition_row_from[
            ~(prev_transition_row_from[TO_PLANNING_ITEM]).isin(
                parent_transitions + [to_planning_item_selected]
            )
        ]
        if not prev_transition_row_from.empty:
            parent_transitions.append(prev_transition_row_from[TO_PLANNING_ITEM].iloc[0])
            get_next_transition(
                planning_item=prev_transition_row_from[TO_PLANNING_ITEM].iloc[0],
                from_planning_item_selected=to_planning_item_selected,
                TO_PLANNING_ITEM=TO_PLANNING_ITEM,
                FROM_REGION_GROUP=FROM_REGION_GROUP,
                FROM_PnL_GROUP=FROM_PnL_GROUP,
                FROM_DEMAND_DOMAIN_GROUP=FROM_DEMAND_DOMAIN_GROUP,
                FROM_LOCATION_GROUP=FROM_LOCATION_GROUP,
                FROM_CHANNEL_GROUP=FROM_CHANNEL_GROUP,
                FROM_ACCOUNT_GROUP=FROM_ACCOUNT_GROUP,
                FROM_PLANNING_ITEM=FROM_PLANNING_ITEM,
                TransitionFlag=TransitionFlag,
                child_transitions=parent_transitions,
            )

    if not prev_transition_row_to.empty:
        prev_transition_row_to = prev_transition_row_to[
            ~(prev_transition_row_to[FROM_PLANNING_ITEM]).isin(
                parent_transitions + [to_planning_item_selected]
            )
        ]
        if not prev_transition_row_to.empty:
            parent_transitions.append(prev_transition_row_to[FROM_PLANNING_ITEM].iloc[0])
            get_previous_transition(
                planning_item=prev_transition_row_to[FROM_PLANNING_ITEM].iloc[0],
                to_planning_item_selected=to_planning_item_selected,
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


col_mapping = {
    "Assortment New": float,
    "Intro Date": "datetime64[ns]",
    "Assortment Phase In": float,
    "Product Transition Overlap Start Date": "datetime64[ns]",
    "Assortment Phase Out": float,
    "Disco Date": "datetime64[ns]",
    "Final Transition Item": str,
    "Transition Item After Set Product Transition": str,
    "Transition Item Before Set Product Transition": str,
    "510 Product Transition.[Transition Flag]": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    AssortmentOutputGrains,
    DatesOutputGrains,
    IntroDate,
    DiscoDate,
    AssortmentFoundation,
    AssortmentFinal,
    ItemMasterData,
    selectedCombinations,
    TransitionFlag,
    ExistingOutput,
    df_keys,
):
    """Update transition items."""
    plugin_name = "DP067UpdateTransitionItemOnSetTransition"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    version_col = o9Constants.VERSION_NAME
    location_col = "Location.[Location]"
    pl_item_col = o9Constants.PLANNING_ITEM
    pl_account_col = o9Constants.PLANNING_ACCOUNT
    pl_channel_col = o9Constants.PLANNING_CHANNEL
    pl_region_col = o9Constants.PLANNING_REGION
    pl_demand_domain_col = o9Constants.PLANNING_DEMAND_DOMAIN
    pl_pnl_col = o9Constants.PLANNING_PNL
    pl_location_col = o9Constants.PLANNING_LOCATION
    transition_item = o9Constants.TRANSITION_ITEM
    from_pl_channel_col = "from.[Channel].[Planning Channel]"
    from_pl_region_col = "from.[Region].[Planning Region]"
    from_pl_account_col = "from.[Account].[Planning Account]"
    from_pl_pnl_col = "from.[PnL].[Planning PnL]"
    from_pl_demand_domain_col = "from.[Demand Domain].[Planning Demand Domain]"
    from_pl_location_col = "from.[Location].[Planning Location]"
    from_pl_item_col = "from.[Item].[Planning Item]"
    to_pl_item_col = "to.[Item].[Planning Item]"

    assortment_foundation = "Assortment Foundation"
    from_pl_item = "Transition From Planning Item"
    to_pl_item = "Transition To Planning Item"
    selected_pl_account = "Transition Selected Planning Account"
    selected_pl_channel = "Transition Selected Planning Channel"
    selected_pl_region = "Transition Selected Planning Region"
    selected_pl_pnl = "Transition Selected Planning PnL"
    selected_pl_demand_domain = "Transition Selected Planning Demand Domain"
    selected_pl_location = "Transition Selected Planning Location"
    default_transition_period = "Default Transition Period"
    assortment_phase_in = "Assortment Phase In"
    assortment_phase_out = "Assortment Phase Out"
    product_transition_start_date = "Product Transition Overlap Start Date"
    actual_col = "Actual"

    assortment_grains = get_list_of_grains_from_string(input=AssortmentOutputGrains)
    transition_dates_grains = get_list_of_grains_from_string(input=DatesOutputGrains)

    # Assortment output measures
    assortment = "Assortment New"

    # dates output measures
    intro_date = "Intro Date"
    disco_date = "Disco Date"

    # TransitionAttributes output measures
    final_transition_item = "Final Transition Item"
    transition_item_after_set = "Transition Item After Set Product Transition"
    transition_item_before_set = "Transition Item Before Set Product Transition"

    # TransitionFlagOutput output measures
    flag = "510 Product Transition.[Transition Flag]"

    assortment_output_cols = [version_col] + assortment_grains + [assortment]
    intro_date_output_cols = [version_col] + transition_dates_grains + [intro_date]
    disco_date_output_cols = [version_col] + transition_dates_grains + [disco_date]
    transition_attributes_output_cols = [
        version_col,
        pl_item_col,
        final_transition_item,
        transition_item_after_set,
        transition_item_before_set,
    ]
    transition_flag_output_cols = TransitionFlag.columns

    Assortment = pd.DataFrame(columns=assortment_output_cols)
    IntroDateOutput = pd.DataFrame(columns=intro_date_output_cols)
    DiscoDateOutput = pd.DataFrame(columns=disco_date_output_cols)
    TransitionAttributes = pd.DataFrame(columns=transition_attributes_output_cols)
    TransitionFlagOutput = pd.DataFrame(columns=transition_flag_output_cols)

    # The below logic does not need intro date and actual col in AssortmentFoundation, availability causing issue, dropping off
    AssortmentFoundation_original = AssortmentFoundation.copy(deep=True)
    AssortmentFoundation.drop([intro_date, actual_col], axis=1, inplace=True)

    try:
        if (len(AssortmentFoundation) == 0) or (len(selectedCombinations) == 0):
            logger.warning("One or more input/s have no data. Please check the input ...")
            logger.warning("Returning empty dataframes ...")
            return (
                Assortment,
                IntroDateOutput,
                DiscoDateOutput,
                TransitionAttributes,
                TransitionFlagOutput,
            )

        # Fill missing with 180 D
        selectedCombinations[default_transition_period] = selectedCombinations[
            default_transition_period
        ].fillna(180)

        input_version = selectedCombinations[version_col].iloc[0]
        IntroDate = pd.to_datetime(IntroDate)
        DiscoDate = pd.to_datetime(DiscoDate)

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

        intersections_to_check_from_item = create_cartesian_product(
            relevant_df, old_pl_item_values_df
        )
        intersections_to_check_to_item = create_cartesian_product(
            relevant_df, new_pl_item_values_df
        )

        # checking which intersections are present for selected from item
        relevant_intersections = intersections_to_check_from_item.merge(AssortmentFoundation)
        relevant_intersections.drop([pl_item_col, assortment_foundation], axis=1, inplace=True)
        relevant_intersections.drop_duplicates(inplace=True)

        # copying values to use them later
        transition_flag_intersections = intersections_to_check_to_item.copy()

        # populate 1 in assortment new for all intersections
        logger.info("populating 1 for measure assortment new ...")
        intersections_to_check_to_item[assortment] = 1

        # getting intersections where assortment foundation is not 1 already
        logger.info("Getting intersections where assortment foundation is not 1 already")
        intersections_to_check_to_item = intersections_to_check_to_item.merge(
            AssortmentFoundation,
            how="outer",
            indicator=True,
        )
        intersections_to_check_to_item[version_col] = input_version

        # getting relevant intersections
        intersections_to_check_to_item = intersections_to_check_to_item.merge(
            relevant_df,
        )
        intersections_to_check_to_item = intersections_to_check_to_item[
            (intersections_to_check_to_item[pl_item_col].isin(selected_old_pl_item))
            | (intersections_to_check_to_item[pl_item_col].isin(selected_new_pl_item))
        ]

        assortment_intersections = intersections_to_check_to_item[
            intersections_to_check_to_item["_merge"] == "left_only"
        ]
        intersections_to_check_to_item = intersections_to_check_to_item[
            ~(intersections_to_check_to_item["_merge"] == "left_only")
        ]

        assortment_intersections = assortment_intersections.merge(relevant_intersections)

        # getting item and location columns
        logger.info("getting corresponding item and location values ...")
        Assortment = assortment_intersections.merge(
            ItemMasterData,
            how="inner",
        )
        cols_req = [
            pl_location_col,
            location_col,
            pl_region_col,
            pl_channel_col,
            pl_demand_domain_col,
            pl_account_col,
            pl_pnl_col,
        ]
        LocationMasterData = AssortmentFinal[cols_req].drop_duplicates()
        Assortment = Assortment.merge(
            LocationMasterData,
            how="inner",
        )
        Assortment = Assortment[assortment_output_cols]

        # getting old items intersection present in AssortmentFoundation
        logger.info("getting old items with assortment foundation as 1 ...")
        old_items_with_assortment = AssortmentFoundation.merge(
            old_pl_item_values_df,
            how="inner",
        )

        # rename column planning item -> from planning item
        old_items_with_assortment.rename(columns={pl_item_col: from_pl_item_col}, inplace=True)

        # cartesian product with new items df
        TransitionFlagOutput = create_cartesian_product(
            old_items_with_assortment, new_pl_item_values_df
        )

        # getting relevant intersection
        TransitionFlagOutput = TransitionFlagOutput.merge(
            relevant_df,
        )

        # populate 1 in transition flag for all intersections
        logger.info("populating 1 for measure transition flag ...")
        TransitionFlagOutput[flag] = 1

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
        TransitionFlagOutput.rename(columns=col_mapping, inplace=True)
        TransitionFlagOutput = TransitionFlagOutput[transition_flag_output_cols]

        # getting intro and disco dates
        logger.info("populating intro and disco dates ...")
        transition_dates_intersection = pd.concat(
            [intersections_to_check_to_item, assortment_intersections]
        )
        filter_clause = transition_dates_intersection["_merge"] == "right_only"

        transition_dates_intersection[intro_date] = np.where(
            ~filter_clause, IntroDate, pd.to_datetime(np.nan)
        )
        transition_dates_intersection[disco_date] = np.where(
            filter_clause, DiscoDate, pd.to_datetime(np.nan)
        )

        IntroDateOutput = transition_dates_intersection[intro_date_output_cols]
        DiscoDateOutput = transition_dates_intersection[disco_date_output_cols]

        IntroDateOutput.dropna(inplace=True)
        DiscoDateOutput.dropna(inplace=True)

        # ------------ New logic of intro and disco date
        # Intro date
        # Actual must be present

        AssortmentFoundation_ac = AssortmentFoundation_original[
            AssortmentFoundation_original[actual_col] > 0
        ]
        AssortmentFoundation_ac.drop([assortment_foundation], axis=1, inplace=True)
        AssortmentFoundation_ac.rename(columns={intro_date: intro_date + "_"}, inplace=True)

        IntroDateOutput[assortment_phase_in] = 1

        IntroDateOutput = IntroDateOutput.merge(AssortmentFoundation_ac, how="left")

        IntroDateOutput.loc[~IntroDateOutput[actual_col].isna(), intro_date] = IntroDateOutput.loc[
            ~IntroDateOutput[actual_col].isna(), intro_date + "_"
        ]
        IntroDateOutput.drop([intro_date + "_"], axis=1, inplace=True)

        # Disco Date
        IntroDateOutput_w_Actual_intersection = IntroDateOutput[~IntroDateOutput[actual_col].isna()]

        IntroDateOutput.drop([actual_col], axis=1, inplace=True)

        DiscoDateOutput[product_transition_start_date] = np.nan

        TransitionFlag_tmp = pd.concat(
            [TransitionFlagOutput, TransitionFlag], axis=0, ignore_index=True, sort=False
        )

        DiscoDateOutput[assortment_phase_out] = 1

        for index, row in IntroDateOutput_w_Actual_intersection.iterrows():
            row_df = pd.DataFrame(row).T

            the_TransitionFlag = TransitionFlag_tmp[
                (TransitionFlag_tmp[from_pl_region_col] == row_df[pl_region_col].values[0])
                & (TransitionFlag_tmp[from_pl_pnl_col] == row_df[pl_pnl_col].values[0])
                & (TransitionFlag_tmp[from_pl_location_col] == row_df[pl_location_col].values[0])
                & (
                    TransitionFlag_tmp[from_pl_demand_domain_col]
                    == row_df[pl_demand_domain_col].values[0]
                )
                & (TransitionFlag_tmp[from_pl_channel_col] == row_df[pl_channel_col].values[0])
                & (TransitionFlag_tmp[from_pl_account_col] == row_df[pl_account_col].values[0])
                & (TransitionFlag_tmp[to_pl_item_col] == row_df[pl_item_col].values[0])
            ]

            if the_TransitionFlag.empty:
                continue

            # Expand
            selectedCombinations_expanded = selectedCombinations.copy(deep=True)
            cols_to_expand = [
                from_pl_item,
                to_pl_item,
                selected_pl_account,
                selected_pl_channel,
                selected_pl_region,
                selected_pl_pnl,
                selected_pl_demand_domain,
                selected_pl_location,
            ]
            selectedCombinations_expanded[cols_to_expand] = selectedCombinations_expanded[
                cols_to_expand
            ].apply(lambda x: x.str.split(","))
            for col in cols_to_expand:
                selectedCombinations_expanded = selectedCombinations_expanded.explode(col)

            for the_TransitionFlag_from_pl_item_col in the_TransitionFlag[from_pl_item_col]:

                filter_clause = (
                    (
                        selectedCombinations_expanded[from_pl_item]
                        == the_TransitionFlag_from_pl_item_col
                    )
                    & (selectedCombinations_expanded[to_pl_item] == row_df[pl_item_col].values[0])
                    & (
                        selectedCombinations_expanded[selected_pl_account]
                        == row_df[pl_account_col].values[0]
                    )
                    & (
                        selectedCombinations_expanded[selected_pl_channel]
                        == row_df[pl_channel_col].values[0]
                    )
                    & (
                        selectedCombinations_expanded[selected_pl_region]
                        == row_df[pl_region_col].values[0]
                    )
                    & (
                        selectedCombinations_expanded[selected_pl_pnl]
                        == row_df[pl_pnl_col].values[0]
                    )
                    & (
                        selectedCombinations_expanded[selected_pl_demand_domain]
                        == row_df[pl_demand_domain_col].values[0]
                    )
                    & (
                        selectedCombinations_expanded[selected_pl_location]
                        == row_df[pl_location_col].values[0]
                    )
                )
                the_selectedCombinations = selectedCombinations_expanded[filter_clause]

                if the_selectedCombinations.empty:
                    continue

                transition_period = int(
                    the_selectedCombinations[default_transition_period].values[0]
                )

                filter_clause = (
                    (DiscoDateOutput[pl_region_col] == row_df[pl_region_col].values[0])
                    & (DiscoDateOutput[pl_pnl_col] == row_df[pl_pnl_col].values[0])
                    & (DiscoDateOutput[pl_location_col] == row_df[pl_location_col].values[0])
                    & (
                        DiscoDateOutput[pl_demand_domain_col]
                        == row_df[pl_demand_domain_col].values[0]
                    )
                    & (DiscoDateOutput[pl_channel_col] == row_df[pl_channel_col].values[0])
                    & (DiscoDateOutput[pl_account_col] == row_df[pl_account_col].values[0])
                    & (DiscoDateOutput[pl_item_col] == the_TransitionFlag_from_pl_item_col)
                )

                if DiscoDateOutput[filter_clause].empty:
                    continue

                DiscoDateOutput.loc[filter_clause, product_transition_start_date] = (
                    DiscoDateOutput.loc[filter_clause, disco_date]
                )

                DiscoDateOutput.loc[filter_clause, product_transition_start_date] = pd.to_datetime(
                    DiscoDateOutput[filter_clause][product_transition_start_date]
                ) - pd.Timedelta(days=transition_period)

        # creating dummy input for TransitionFlag
        transition_flag_intersections[version_col] = input_version
        transition_flag_intersections.rename(columns=col_mapping, inplace=True)
        transition_flag_intersections.rename(columns={pl_item_col: to_pl_item_col}, inplace=True)
        transition_flag_intersections = create_cartesian_product(
            transition_flag_intersections, old_pl_item_values_df
        )
        transition_flag_intersections.rename(columns={pl_item_col: from_pl_item_col}, inplace=True)

        transition_flag_intersections[flag] = 1

        relevant_transitions = transition_flag_intersections.copy()

        combinations_to_update = set()

        # merge item mapping with existing output to not overwrite with nan
        ItemMasterData = ItemMasterData.merge(ExistingOutput, how="left")

        common_cols = [
            from_pl_account_col,
            from_pl_channel_col,
            from_pl_region_col,
            from_pl_demand_domain_col,
            from_pl_pnl_col,
            from_pl_location_col,
        ]
        # Iterate over rows
        for name, group in relevant_transitions.groupby(
            common_cols + [from_pl_item_col, to_pl_item_col], observed=True
        ):
            # create the from pl item, customer group combination
            the_from_side_intersection = group[from_pl_item_col].iloc[0]
            combinations_to_update.add(the_from_side_intersection)

            # create the to pl item customer group combination
            the_to_side_intersection = group[to_pl_item_col].iloc[0]
            combinations_to_update.add(the_to_side_intersection)

            # recursively get all previous transitions in which from planning item is involved
            prev_parent_combinations = get_previous_transition(
                planning_item=the_from_side_intersection,
                to_planning_item_selected=the_to_side_intersection,
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
            combinations_to_update.update(prev_parent_combinations)

            next_parent_combinations = get_next_transition(
                planning_item=the_from_side_intersection,
                from_planning_item_selected=the_from_side_intersection,
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

            combinations_to_update.update(next_parent_combinations)

            # get children combinations
            prev_child_combinations = get_previous_transition(
                planning_item=the_to_side_intersection,
                to_planning_item_selected=the_to_side_intersection,
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

            combinations_to_update.update(prev_child_combinations)

            next_child_combinations = get_next_transition(
                planning_item=the_to_side_intersection,
                from_planning_item_selected=the_from_side_intersection,
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

            combinations_to_update.update(next_child_combinations)

            # transition item of To Planning Item
            row_pl_item = the_to_side_intersection
            transition_of_to_item = ItemMasterData[ItemMasterData[pl_item_col] == row_pl_item][
                transition_item
            ].iloc[0]

            # update before set for found combinations
            filter_clause = (ItemMasterData[pl_item_col].isin(list(combinations_to_update))) & (
                ItemMasterData[transition_item_before_set].isna()
            )
            ItemMasterData[transition_item_before_set] = np.where(
                filter_clause,
                ItemMasterData[transition_item],
                ItemMasterData[transition_item_before_set],
            )

            # update transition item for pl items in update combinations
            ItemMasterData[transition_item_after_set] = np.where(
                ItemMasterData[pl_item_col].isin(list(combinations_to_update)),
                transition_of_to_item,
                ItemMasterData[transition_item_after_set],
            )

        ItemMasterData[final_transition_item] = ItemMasterData[transition_item_after_set]
        ItemMasterData[version_col] = input_version

        TransitionAttributes = ItemMasterData[transition_attributes_output_cols].drop_duplicates()
        TransitionAttributes = TransitionAttributes[
            TransitionAttributes[pl_item_col].isin(list(combinations_to_update))
        ]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )

    return (
        Assortment,
        IntroDateOutput,
        DiscoDateOutput,
        TransitionAttributes,
        TransitionFlagOutput,
    )

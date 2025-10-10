"""helper function for DP024CannibItemMatch."""

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


col_mapping = {"610 Cannibalized Item Match.[System Suggested Cannibalized Item]": bool}


def get_combined_condition(
    LikeAssortmentMatch,
    from_pl_account_col,
    to_pl_account_col,
    from_pl_channel_col,
    to_pl_channel_col,
    from_pl_region_col,
    to_pl_region_col,
    from_pl_pnl_col,
    to_pl_pnl_col,
    from_pl_loc_col,
    to_pl_loc_col,
    from_pl_dem_domain_col,
    to_pl_dem_domain_col,
    IncludeDemandDomain,
):
    """Do check for the present of the DemandDomain and return the combinedcombination."""
    if IncludeDemandDomain == "True":
        # Condition with dem_inc domain column
        combined_condition = (
            (LikeAssortmentMatch[from_pl_account_col] == LikeAssortmentMatch[to_pl_account_col])
            & (LikeAssortmentMatch[from_pl_channel_col] == LikeAssortmentMatch[to_pl_channel_col])
            & (LikeAssortmentMatch[from_pl_region_col] == LikeAssortmentMatch[to_pl_region_col])
            & (LikeAssortmentMatch[from_pl_pnl_col] == LikeAssortmentMatch[to_pl_pnl_col])
            & (
                LikeAssortmentMatch[from_pl_dem_domain_col]
                == LikeAssortmentMatch[to_pl_dem_domain_col]
            )
            & (LikeAssortmentMatch[from_pl_loc_col] == LikeAssortmentMatch[to_pl_loc_col])
        )
    else:
        # Condition without dem_inc domain column
        combined_condition = (
            (LikeAssortmentMatch[from_pl_account_col] == LikeAssortmentMatch[to_pl_account_col])
            & (LikeAssortmentMatch[from_pl_channel_col] == LikeAssortmentMatch[to_pl_channel_col])
            & (LikeAssortmentMatch[from_pl_region_col] == LikeAssortmentMatch[to_pl_region_col])
            & (LikeAssortmentMatch[from_pl_pnl_col] == LikeAssortmentMatch[to_pl_pnl_col])
            & (LikeAssortmentMatch[from_pl_loc_col] == LikeAssortmentMatch[to_pl_loc_col])
        )

    return combined_condition


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    LikeAssortmentMatch,
    GenSysCannItemMatchAssortment,
    IncludeDemandDomain,
    df_keys,
):
    """Do run main function and return a list."""
    plugin_name = "DP024SystemCannibItemMatch"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_loc_col = "Location.[Planning Location]"
    pl_dem_domain_col = "Demand Domain.[Planning Demand Domain]"
    from_pl_item_col = "from.[Item].[Planning Item]"
    from_pl_account_col = "from.[Account].[Planning Account]"
    from_pl_channel_col = "from.[Channel].[Planning Channel]"
    from_pl_region_col = "from.[Region].[Planning Region]"
    from_pl_pnl_col = "from.[PnL].[Planning PnL]"
    from_pl_dem_domain_col = "from.[Demand Domain].[Planning Demand Domain]"
    from_pl_loc_col = "from.[Location].[Planning Location]"
    to_pl_item_col = "to.[Item].[Planning Item]"
    to_pl_account_col = "to.[Account].[Planning Account]"
    to_pl_channel_col = "to.[Channel].[Planning Channel]"
    to_pl_region_col = "to.[Region].[Planning Region]"
    to_pl_pnl_col = "to.[PnL].[Planning PnL]"
    to_pl_dem_domain_col = "to.[Demand Domain].[Planning Demand Domain]"
    to_pl_loc_col = "to.[Location].[Planning Location]"

    likeassortmentrank_col = "620 Like Assortment Match.[Like Assortment Rank]"
    finallikeassortmentselected_col = "620 Like Assortment Match.[Final Like Assortment Selected]"
    generatesystemcannibalizeditemmatchassortment_col = (
        "Generate System Cannibalized Item Match Assortment"
    )

    # output measure
    sys_sugg_cann_item_col = "610 Cannibalized Item Match.[System Suggested Cannibalized Item]"

    # output df
    cols_required_in_output = [
        version_col,
        from_pl_item_col,
        from_pl_account_col,
        from_pl_channel_col,
        from_pl_region_col,
        from_pl_pnl_col,
        from_pl_loc_col,
        to_pl_item_col,
        sys_sugg_cann_item_col,
    ]

    dim_cols_required_in_output = [
        x for x in cols_required_in_output if x != sys_sugg_cann_item_col
    ]

    output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)

    try:

        if IncludeDemandDomain not in ["True", "False"]:
            logger.warning(
                f"Possible values for the IncludeDemandDomain is True or False. User input is {IncludeDemandDomain}"
            )
        # condition to check both the input are not empty
        if LikeAssortmentMatch.empty:
            logger.warning("Like Assorment Match is empty for slice: {} ...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        if GenSysCannItemMatchAssortment.empty:
            logger.warning(
                "Generate System Cannibalized Item Match Assortment is empty for slice: {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        # Generate System Cannibalized Item Match Assortment if exists
        GenSysCannItemMatchAssortment = GenSysCannItemMatchAssortment[
            GenSysCannItemMatchAssortment[generatesystemcannibalizeditemmatchassortment_col] >= 1
        ]

        if GenSysCannItemMatchAssortment.empty:
            logger.warning(
                "Generate System Cannibalized Item Match Assortment does not contain the assortment value for the slice: {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        #  Rename columns in GenSysCannItemMatchAssortment for consistency
        GenSysCannItemMatchAssortment.rename(
            columns={
                pl_item_col: from_pl_item_col,
                pl_account_col: from_pl_account_col,
                pl_channel_col: from_pl_channel_col,
                pl_region_col: from_pl_region_col,
                pl_pnl_col: from_pl_pnl_col,
                pl_loc_col: from_pl_loc_col,
                pl_dem_domain_col: from_pl_dem_domain_col,
            },
            inplace=True,
        )

        #  Filter LikeAssortmentMatch on the basis of GenSysCannItemMatchAssortment
        LikeAssortmentMatch = LikeAssortmentMatch.merge(
            GenSysCannItemMatchAssortment,
            on=[
                from_pl_item_col,
                from_pl_account_col,
                from_pl_channel_col,
                from_pl_region_col,
                from_pl_pnl_col,
                from_pl_loc_col,
                from_pl_dem_domain_col,
            ],
            how="inner",
        )

        # condition to check the like item exists after filter
        if LikeAssortmentMatch.empty:
            logger.warning(
                "Like Assortment Match with filter Generate System Cannibalized Item Match Assortment is empty for slice: {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        # Filter LikeAssortmentMatch DataFrame based on multiple conditions

        # Ensure required columns exist
        required_columns = [
            from_pl_account_col,
            to_pl_account_col,
            from_pl_channel_col,
            to_pl_channel_col,
            from_pl_region_col,
            to_pl_region_col,
            from_pl_pnl_col,
            to_pl_pnl_col,
            from_pl_dem_domain_col,
            to_pl_dem_domain_col,
            from_pl_loc_col,
            to_pl_loc_col,
        ]

        missing_columns = [
            col for col in required_columns if col not in LikeAssortmentMatch.columns
        ]

        if missing_columns:
            logger.warning(
                "Missing columns in DataFrame: {} for slice: {}...".format(missing_columns, df_keys)
            )
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        # Combine all conditions into a single boolean mask using '&'
        combined_condition = get_combined_condition(
            LikeAssortmentMatch,
            from_pl_account_col,
            to_pl_account_col,
            from_pl_channel_col,
            to_pl_channel_col,
            from_pl_region_col,
            to_pl_region_col,
            from_pl_pnl_col,
            to_pl_pnl_col,
            from_pl_loc_col,
            to_pl_loc_col,
            from_pl_dem_domain_col,
            to_pl_dem_domain_col,
            IncludeDemandDomain,
        )

        # Apply the combined condition to filter the DataFrame
        LikeAssortmentMatch = LikeAssortmentMatch[combined_condition]

        # condition to check if like item exists after self filter
        if LikeAssortmentMatch.empty:
            logger.warning(
                "Like Assortment with filter on same accounts is empty for slice: {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty df as result for this slice ...")
            output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)
            return output_sugg_cannib

        key = [
            from_pl_item_col,
            from_pl_account_col,
            from_pl_channel_col,
            from_pl_region_col,
            from_pl_pnl_col,
            from_pl_dem_domain_col,
            from_pl_loc_col,
        ]
        LikeAssortmentMatch_group = LikeAssortmentMatch.groupby(key)
        for group_keys, group_df in LikeAssortmentMatch_group:
            if not group_df.empty:
                group_df_selected = group_df[group_df[finallikeassortmentselected_col] == 1]
                # if Generate System Cannibalized Item Match Assortment is not available then select on the basis of least rank
                if group_df_selected.empty:
                    group_df_selected = group_df.sort_values(
                        likeassortmentrank_col, ascending=True
                    ).head(1)

                group_df_selected[sys_sugg_cann_item_col] = 1

                # dropping demand domain
                group_df_selected_agg = group_df_selected.groupby(
                    dim_cols_required_in_output, as_index=False
                )[sys_sugg_cann_item_col].max()
                output_sugg_cannib = pd.concat(
                    [
                        output_sugg_cannib,
                        group_df_selected_agg[cols_required_in_output],
                    ],
                    ignore_index=True,
                )
    except Exception as e:
        logger.exception(
            "Exception {} for slice: {}. Returning empty dataframe as output ...".format(e, df_keys)
        )
        output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)

    return output_sugg_cannib

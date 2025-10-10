"""Generate Cannib Item Match Plugin for Flexible NPI."""

# Library imports
import logging
from typing import Optional

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

col_mapping = {"635 Initiative Cannibalized Item Match.[System Recommended Cannib Item L0]": bool}


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
    """Construct a combined condition for comparing DataFrame columns on the basis of the demand domain column."""
    if IncludeDemandDomain == "True":
        # Condition with dem_inc domain column
        combined_condition = (
            (
                LikeAssortmentMatch[from_pl_account_col].astype(str)
                == LikeAssortmentMatch[to_pl_account_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_channel_col].astype(str)
                == LikeAssortmentMatch[to_pl_channel_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_region_col].astype(str)
                == LikeAssortmentMatch[to_pl_region_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_pnl_col].astype(str)
                == LikeAssortmentMatch[to_pl_pnl_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_dem_domain_col].astype(str)
                == LikeAssortmentMatch[to_pl_dem_domain_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_loc_col].astype(str)
                == LikeAssortmentMatch[to_pl_loc_col].astype(str)
            )
        )
    else:
        # Condition without dem_inc domain column
        combined_condition = (
            (
                LikeAssortmentMatch[from_pl_account_col].astype(str)
                == LikeAssortmentMatch[to_pl_account_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_channel_col].astype(str)
                == LikeAssortmentMatch[to_pl_channel_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_region_col].astype(str)
                == LikeAssortmentMatch[to_pl_region_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_pnl_col].astype(str)
                == LikeAssortmentMatch[to_pl_pnl_col].astype(str)
            )
            & (
                LikeAssortmentMatch[from_pl_loc_col].astype(str)
                == LikeAssortmentMatch[to_pl_loc_col].astype(str)
            )
        )

    return combined_condition


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    # Data
    LikeAssortmentMatch,
    GenSysCannItemMatchAssortment,
    IncludeDemandDomain,
    # Others
    df_keys: Optional[dict] = None,
):
    """Entry point of the script."""
    plugin_name = "DP132GenerateCannibItemMatch"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_column = "Version.[Version Name]"
    data_obj = "Data Object.[Data Object]"
    from_data_obj = "from.[Data Object].[Data Object]"

    initiative = "Initiative.[Initiative]"
    from_initiative = "from.[Initiative].[Initiative]"

    npi_item = "Item.[NPI Item]"
    npi_account = "Account.[NPI Account]"
    npi_channel = "Channel.[NPI Channel]"
    npi_region = "Region.[NPI Region]"
    npi_pnl = "PnL.[NPI PnL]"
    npi_loc = "Location.[NPI Location]"
    npi_dem_domain = "Demand Domain.[NPI Demand Domain]"

    from_item = "from.[Item].[NPI Item]"
    from_account = "from.[Account].[NPI Account]"
    from_channel = "from.[Channel].[NPI Channel]"
    from_region = "from.[Region].[NPI Region]"
    from_pnl = "from.[PnL].[NPI PnL]"
    from_dem_domain = "from.[Demand Domain].[NPI Demand Domain]"
    from_loc = "from.[Location].[NPI Location]"

    to_item = "to.[Item].[NPI Item]"
    to_account = "to.[Account].[NPI Account]"
    to_channel = "to.[Channel].[NPI Channel]"
    to_region = "to.[Region].[NPI Region]"
    to_pnl = "to.[PnL].[NPI PnL]"
    to_dem_domain = "to.[Demand Domain].[NPI Demand Domain]"
    to_loc = "to.[Location].[NPI Location]"

    likeassortmentrank_col = "630 Initiative Like Assortment Match.[Like Assortment Rank L0]"
    finallikeassortmentselected_col = (
        "630 Initiative Like Assortment Match.[Final Like Assortment L0]"
    )
    generatesystemcannibalizeditemmatchassortment_col = (
        "Generate Cannibalized Item Match Assortment L0"
    )

    # Output Measure
    sys_sugg_cann_item_col = (
        "635 Initiative Cannibalized Item Match.[System Recommended Cannib Item L0]"
    )

    # outputdf
    cols_required_in_output = [
        version_column,
        from_initiative,
        from_data_obj,
        from_item,
        from_account,
        from_channel,
        from_region,
        from_dem_domain,
        from_pnl,
        from_loc,
        to_item,
        sys_sugg_cann_item_col,
    ]

    dim_cols_required_in_output = [
        x for x in cols_required_in_output if x != sys_sugg_cann_item_col
    ]

    output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)

    try:
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
                initiative: from_initiative,
                data_obj: from_data_obj,
                npi_item: from_item,
                npi_account: from_account,
                npi_channel: from_channel,
                npi_region: from_region,
                npi_pnl: from_pnl,
                npi_loc: from_loc,
                npi_dem_domain: from_dem_domain,
            },
            inplace=True,
        )

        #  Filter LikeAssortmentMatch on the basis of GenSysCannItemMatchAssortment
        LikeAssortmentMatch = LikeAssortmentMatch.merge(
            GenSysCannItemMatchAssortment,
            on=[
                from_initiative,
                from_data_obj,
                from_item,
                from_account,
                from_channel,
                from_region,
                from_pnl,
                from_loc,
                from_dem_domain,
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
            from_data_obj,
            from_initiative,
            from_account,
            to_account,
            from_channel,
            to_channel,
            from_region,
            to_region,
            from_pnl,
            to_pnl,
            from_dem_domain,
            to_dem_domain,
            from_loc,
            to_loc,
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
            from_account,
            to_account,
            from_channel,
            to_channel,
            from_region,
            to_region,
            from_pnl,
            to_pnl,
            from_loc,
            to_loc,
            from_dem_domain,
            to_dem_domain,
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
            from_item,
            from_account,
            from_channel,
            from_region,
            from_pnl,
            from_dem_domain,
            from_loc,
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
                    [output_sugg_cannib, group_df_selected_agg[cols_required_in_output]],
                    ignore_index=True,
                )
                output_sugg_cannib = output_sugg_cannib[
                    output_sugg_cannib[sys_sugg_cann_item_col].notna()
                ].reset_index(drop=True)
    except Exception as e:
        logger.exception(
            "Exception {} for slice: {}. Returning empty dataframe as output ...".format(e, df_keys)
        )
        output_sugg_cannib = pd.DataFrame(columns=cols_required_in_output)

    return output_sugg_cannib

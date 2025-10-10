import logging

import pandas as pd
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension

logger = logging.getLogger("o9_logger")


def rearrange_lists(reference_list: list, target_list: list):
    """
    Rearrange target list based on the order of elements in the reference list
    """
    # Rearrange based on reference
    reordered_list = [col for col in reference_list if col in target_list]
    return reordered_list


class Constants:
    version_col = "Version.[Version Name]"
    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    pl_item_col = "Item.[Planning Item]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_location_col = "Location.[Planning Location]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"

    from_item_col = "from.[Item].[Planning Item]"
    from_account_col = "from.[Account].[Planning Account]"
    from_channel_col = "from.[Channel].[Planning Channel]"
    from_region_col = "from.[Region].[Planning Region]"
    from_pnl_col = "from.[PnL].[Planning PnL]"
    from_location_col = "from.[Location].[Planning Location]"
    from_demand_domain_col = "from.[Demand Domain].[Planning Demand Domain]"

    to_item_col = "to.[Item].[Planning Item]"
    to_account_col = "to.[Account].[Planning Account]"
    to_channel_col = "to.[Channel].[Planning Channel]"
    to_region_col = "to.[Region].[Planning Region]"
    to_pnl_col = "to.[PnL].[Planning PnL]"
    to_location_col = "to.[Location].[Planning Location]"
    to_demand_domain_col = "to.[Demand Domain].[Planning Demand Domain]"

    intro_date = "Intro Date"
    stat_fcst = "Stat Fcst"
    npi_fcst = "NPI Fcst"
    cannibalization_fcst = "Cannibalization Impact"
    fcst_adj_1 = "Fcst Adjustment 1"
    fcst_adj_2 = "Fcst Adjustment 2"
    fcst_adj_3 = "Fcst Adjustment 3"
    fcst_adj_4 = "Fcst Adjustment 4"
    fcst_adj_5 = "Fcst Adjustment 5"
    fcst_adj_6 = "Fcst Adjustment 6"
    da_1 = "DA - 1"
    da_2 = "DA - 2"
    da_3 = "DA - 3"
    da_4 = "DA - 4"
    da_5 = "DA - 5"
    da_6 = "DA - 6"

    sell_out_fcst_adj_1 = "Sell Out Forecast Adjustment 1"
    sell_out_fcst_adj_2 = "Sell Out Forecast Adjustment 2"
    sell_out_fcst_adj_3 = "Sell Out Forecast Adjustment 3"
    sell_out_fcst_adj_4 = "Sell Out Forecast Adjustment 4"
    sell_out_fcst_adj_5 = "Sell Out Forecast Adjustment 5"
    sell_out_fcst_adj_6 = "Sell Out Forecast Adjustment 6"
    sell_out_stat_fcst_kaf = "Sell Out Stat Fcst KAF BB New"
    sell_out_npi_fcst = "Sell Out NPI Fcst"
    da_sell_out_1 = "DA Sell Out - 1"
    da_sell_out_2 = "DA Sell Out - 2"
    da_sell_out_3 = "DA Sell Out - 3"
    da_sell_out_4 = "DA Sell Out - 4"
    da_sell_out_5 = "DA Sell Out - 5"
    da_sell_out_6 = "DA Sell Out - 6"

    sell_out_column_mapping = {sell_out_stat_fcst_kaf: sell_out_npi_fcst}

    dims = {
        pl_item_col: (from_item_col, to_item_col),
        pl_account_col: (from_account_col, to_account_col),
        pl_channel_col: (from_channel_col, to_channel_col),
        pl_region_col: (from_region_col, to_region_col),
        pl_pnl_col: (from_pnl_col, to_pnl_col),
        pl_location_col: (from_location_col, to_location_col),
        pl_demand_domain_col: (from_demand_domain_col, to_demand_domain_col),
    }

    sell_in_da_mapping = {
        fcst_adj_1: da_1,
        fcst_adj_2: da_2,
        fcst_adj_3: da_3,
        fcst_adj_4: da_4,
        fcst_adj_5: da_5,
        fcst_adj_6: da_6,
    }
    sell_out_da_mapping = {
        sell_out_fcst_adj_1: da_sell_out_1,
        sell_out_fcst_adj_2: da_sell_out_2,
        sell_out_fcst_adj_3: da_sell_out_3,
        sell_out_fcst_adj_4: da_sell_out_4,
        sell_out_fcst_adj_5: da_sell_out_5,
        sell_out_fcst_adj_6: da_sell_out_6,
    }


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping={})
def main(
    like_item_mappings,
    SellOutFcst,
    SellInFcst,
    selected_combinations,
    Parameters,
):
    plugin_name = "DP094PopulateNewAsssortmentFcstBulk"
    logger.info("Executing {} ...".format(plugin_name))

    default_seq = list(Constants.dims.keys())

    # getting dimensions
    sell_in_fcst_dims = [
        col
        for col in SellInFcst.columns
        if is_dimension(col)
        and (
            col != Constants.partial_week_col
            and col != Constants.partial_week_key_col
            and col != Constants.version_col
        )
    ]
    sell_in_fcst_dims = rearrange_lists(default_seq, sell_in_fcst_dims)
    sell_out_fcst_dims = [
        col
        for col in SellOutFcst.columns
        if is_dimension(col)
        and (
            col != Constants.partial_week_col
            and col != Constants.partial_week_key_col
            and col != Constants.version_col
        )
    ]
    sell_out_fcst_dims = rearrange_lists(default_seq, sell_out_fcst_dims)

    sell_in_output_columns = sell_in_fcst_dims + [
        Constants.version_col,
        Constants.partial_week_col,
        Constants.cannibalization_fcst,
        Constants.fcst_adj_1,
        Constants.fcst_adj_2,
        Constants.fcst_adj_3,
        Constants.fcst_adj_4,
        Constants.fcst_adj_5,
        Constants.fcst_adj_6,
    ]
    sell_in_npi_output_columns = sell_in_fcst_dims + [
        Constants.version_col,
        Constants.partial_week_col,
        Constants.npi_fcst,
    ]

    sell_out_output_columns = sell_out_fcst_dims + [
        Constants.version_col,
        Constants.partial_week_col,
        Constants.sell_out_fcst_adj_1,
        Constants.sell_out_fcst_adj_2,
        Constants.sell_out_fcst_adj_3,
        Constants.sell_out_fcst_adj_4,
        Constants.sell_out_fcst_adj_5,
        Constants.sell_out_fcst_adj_6,
    ]
    sell_out_npi_output_columns = sell_out_fcst_dims + [
        Constants.version_col,
        Constants.partial_week_col,
        Constants.sell_out_npi_fcst,
    ]

    SellOutFcstOutput = pd.DataFrame(columns=sell_out_output_columns)
    SellOutNPIFcstOutput = pd.DataFrame(columns=sell_out_npi_output_columns)
    SellInFcstOutput = pd.DataFrame(columns=sell_in_output_columns)
    SellInNPIFcstOutput = pd.DataFrame(columns=sell_in_npi_output_columns)

    try:
        if selected_combinations.empty:
            logger.warning("No selected combinations, returning empty dataframe ...")
            return SellInFcstOutput, SellInNPIFcstOutput, SellOutFcstOutput, SellOutNPIFcstOutput

        if like_item_mappings.empty:
            logger.warning("No like item mappings, returning empty dataframe ...")
            return SellInFcstOutput, SellInNPIFcstOutput, SellOutFcstOutput, SellOutNPIFcstOutput

        if SellOutFcst.empty and SellInFcst.empty:
            logger.warning("Fcst inputs are empty, returning empty dataframes ...")
            return SellInFcstOutput, SellInNPIFcstOutput, SellOutFcstOutput, SellOutNPIFcstOutput

        # key_cols = [Constants.partial_week_key_col, Constants.intro_date]
        SellInFcst[Constants.partial_week_key_col] = pd.to_datetime(
            SellInFcst[Constants.partial_week_key_col], utc=True
        ).dt.tz_localize(None)
        SellOutFcst[Constants.partial_week_key_col] = pd.to_datetime(
            SellOutFcst[Constants.partial_week_key_col], utc=True
        ).dt.tz_localize(None)
        Parameters[Constants.intro_date] = pd.to_datetime(
            Parameters[Constants.intro_date], utc=True
        ).dt.tz_localize(None)

        # get from and to cols
        sell_in_fcst_from_cols = [
            col[0] for dim, col in Constants.dims.items() if dim in sell_in_fcst_dims
        ]
        sell_out_from_cols = [
            col[0] for dim, col in Constants.dims.items() if dim in sell_out_fcst_dims
        ]
        sell_in_fcst_to_cols = [
            col[1] for dim, col in Constants.dims.items() if dim in sell_in_fcst_dims
        ]
        sell_out_to_cols = [
            col[1] for dim, col in Constants.dims.items() if dim in sell_out_fcst_dims
        ]

        # filter like_item_mappings to only include columns that are in the selected_combinations
        sell_in_like_item_mappings = pd.merge(
            like_item_mappings,
            selected_combinations,
            left_on=sell_in_fcst_from_cols,
            right_on=sell_in_fcst_dims,
            how="inner",
        ).drop(columns=sell_in_fcst_dims)

        sell_out_like_item_mappings = pd.merge(
            like_item_mappings,
            selected_combinations,
            left_on=sell_out_from_cols,
            right_on=sell_out_fcst_dims,
            how="inner",
        ).drop(columns=sell_out_fcst_dims)

        # copy forecast based on intro date to new items
        if not sell_in_like_item_mappings.empty and not SellInFcst.empty:
            SellInFcst = SellInFcst.merge(
                sell_in_like_item_mappings[
                    sell_in_fcst_from_cols + sell_in_fcst_to_cols
                ].drop_duplicates(),
                left_on=sell_in_fcst_dims,
                right_on=sell_in_fcst_to_cols,
                how="inner",
            ).drop(columns=sell_in_fcst_dims + sell_in_fcst_to_cols)
            SellInFcst.rename(
                columns={
                    Constants.from_item_col: Constants.pl_item_col,
                    Constants.from_account_col: Constants.pl_account_col,
                    Constants.from_channel_col: Constants.pl_channel_col,
                    Constants.from_region_col: Constants.pl_region_col,
                    Constants.from_pnl_col: Constants.pl_pnl_col,
                    Constants.from_location_col: Constants.pl_location_col,
                    Constants.from_demand_domain_col: Constants.pl_demand_domain_col,
                },
                inplace=True,
            )

            # get intro date
            SellInFcst = SellInFcst.merge(
                Parameters,
                how="left",
            )
            SellInFcst = SellInFcst[
                SellInFcst[Constants.partial_week_key_col] >= SellInFcst[Constants.intro_date]
            ]
            for adj, da in Constants.sell_in_da_mapping.items():
                SellInFcst[adj] = SellInFcst[[adj, da]].sum(axis=1, skipna=True)
            SellInNPIFcstOutput = SellInFcst[sell_in_npi_output_columns]
            SellInFcstOutput = SellInFcst[sell_in_output_columns]

            SellInNPIFcstOutput.dropna(subset=sell_in_npi_output_columns, how="any", inplace=True)

        # copy forecast based on intro date to new items
        if not sell_out_like_item_mappings.empty and not SellOutFcst.empty:
            SellOutFcst = SellOutFcst.merge(
                sell_out_like_item_mappings[
                    sell_out_from_cols + sell_out_to_cols
                ].drop_duplicates(),
                left_on=sell_out_fcst_dims,
                right_on=sell_out_to_cols,
                how="inner",
            ).drop(columns=sell_out_fcst_dims + sell_out_to_cols)
            SellOutFcst.rename(
                columns={
                    Constants.from_item_col: Constants.pl_item_col,
                    Constants.from_account_col: Constants.pl_account_col,
                    Constants.from_channel_col: Constants.pl_channel_col,
                    Constants.from_region_col: Constants.pl_region_col,
                    Constants.from_demand_domain_col: Constants.pl_demand_domain_col,
                },
                inplace=True,
            )

            # get intro date
            SellOutFcst = SellOutFcst.merge(
                Parameters,
                how="left",
            )
            SellOutFcst = SellOutFcst[
                SellOutFcst[Constants.partial_week_key_col] >= SellOutFcst[Constants.intro_date]
            ]
            for adj, da in Constants.sell_out_da_mapping.items():
                SellOutFcst[adj] = SellOutFcst[[adj, da]].sum(axis=1, skipna=True)
            SellOutFcst.rename(columns=Constants.sell_out_column_mapping, inplace=True)
            SellOutNPIFcstOutput = SellOutFcst[sell_out_npi_output_columns]
            SellOutFcstOutput = SellOutFcst[sell_out_output_columns]

            SellOutNPIFcstOutput.dropna(subset=sell_in_npi_output_columns, how="any", inplace=True)

    except Exception as e:
        logger.exception("Exception {}, returning empty dataframe as output ...".format(e))

    return SellInFcstOutput, SellInNPIFcstOutput, SellOutFcstOutput, SellOutNPIFcstOutput

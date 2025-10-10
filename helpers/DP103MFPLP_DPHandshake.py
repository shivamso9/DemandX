"""Helper function for DP103MFPLP_DPHandshake."""

import pandas as pd
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """
    A class to store constant values used throughout the module.
    """

    time_planning_month = o9Constants.PLANNING_MONTH
    planning_channel_column = o9Constants.PLANNING_CHANNEL
    demand_domain_column = o9Constants.PLANNING_DEMAND_DOMAIN
    sales_planning_channel_column = o9Constants.PLANNING_SALES_DEMAND
    selling_season_column = o9Constants.SELLING_SEASON
    version_column = o9Constants.VERSION_NAME
    time_partial_week_column = o9Constants.PARTIAL_WEEK
    lp_ty_asp = o9Constants.LY_TY_ASP
    lp_avg_selling_price = o9Constants.LP_AVG_SELLING_PRICE
    item_l4 = o9Constants.ITEM_L4
    location = o9Constants.LOCATION
    planning_account = o9Constants.PLANNING_ACCOUNT
    planning_region = o9Constants.PLANNING_REGION
    planning_pnl = o9Constants.PLANNING_PNL
    lp_asp = o9Constants.LP_ASP
    time_week_column = o9Constants.WEEK

    OUTPUT_COLUMNS = [
        version_column,
        item_l4,
        location,
        demand_domain_column,
        planning_channel_column,
        time_partial_week_column,
        planning_account,
        planning_pnl,
        planning_region,
        lp_avg_selling_price,
    ]


class Helper:
    """
    A Helper class that provides utility functions.
    """

    def apply_cartesian_products(df, master_datasets):
        """Apply Cartesian product with multiple master datasets."""
        for master_data in master_datasets:
            df = create_cartesian_product(df1=df, df2=master_data)
        return df


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    time_dimension,
    lptyasp_input,
    region_master,
    account_master,
    pnl_master,
    current_time_period,
    df_keys,
    logger,
):
    """
    Function to Process LPTYASP

    Args:
        time_dimension (pd.DataFrame): DataFrame containing time dimension data.
        lptyasp_input (pd.DataFrame): DataFrame containing LPTYASP input data.
        region_master (pd.DataFrame): DataFrame containing region master data.
        account_master (pd.DataFrame): DataFrame containing account master data.
        pnl_master (pd.DataFrame): DataFrame containing PnL master data.
        current_time_period (pd.DataFrame): DataFrame containing the current time period.

    Outputs:
    - tyasp_df: Processed LPTYASP data.
    """
    plugin_name = "DP103MFPLP_DPHandshake"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    try:
        if (
            time_dimension.empty
            or lptyasp_input.empty
            or region_master.empty
            or account_master.empty
            or pnl_master.empty
            or current_time_period.empty
        ):
            raise Exception("One or more required input DataFrames are empty.")

        partial_week_df = time_dimension[
            [
                Constants.time_planning_month,
                Constants.time_partial_week_column,
                Constants.time_week_column,
            ]
        ]
        tyasp_df = lptyasp_input.merge(
            partial_week_df, on=Constants.time_planning_month, how="left"
        ).drop_duplicates()
        current_week = pd.to_datetime(current_time_period[Constants.time_week_column].max())
        tyasp_df[Constants.time_week_column] = pd.to_datetime(
            tyasp_df[Constants.time_week_column], errors="coerce"
        )
        df_past = tyasp_df[tyasp_df[Constants.time_week_column] < current_week].copy()
        df_future = tyasp_df[tyasp_df[Constants.time_week_column] >= current_week].copy()

        df_past = df_past.dropna(subset=[Constants.lp_ty_asp])
        df_future = df_future.dropna(subset=[Constants.lp_asp])

        df_past[Constants.lp_ty_asp] = df_past[Constants.lp_ty_asp]
        df_future[Constants.lp_ty_asp] = df_future[Constants.lp_asp]

        tyasp_df = pd.concat([df_past, df_future], ignore_index=True)
        tyasp_df = tyasp_df.rename(
            columns={
                Constants.sales_planning_channel_column: Constants.planning_channel_column,
                Constants.selling_season_column: Constants.demand_domain_column,
                Constants.lp_ty_asp: Constants.lp_avg_selling_price,
            }
        )
        tyasp_df = Helper.apply_cartesian_products(
            tyasp_df, [account_master, region_master, pnl_master]
        )
        tyasp_df.drop(
            columns=[Constants.time_planning_month, Constants.time_week_column, Constants.lp_asp],
            inplace=True,
        )

        columns_required = [
            col for col in tyasp_df.columns if col != Constants.lp_avg_selling_price
        ] + [Constants.lp_avg_selling_price]
        tyasp_df = tyasp_df[columns_required]
        logger.info("Finished Processing {}...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception occured:")
        logger.exception(e)
        tyasp_df = pd.DataFrame(columns=Constants.OUTPUT_COLUMNS)
    return tyasp_df

"""Helper function for DP101MFPBUASP_DPHanshake."""

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

    time_week_column = o9Constants.WEEK
    time_planning_month = o9Constants.PLANNING_MONTH
    bu_asp_dp = o9Constants.BU_AVG_SELLING_PRICE
    planning_channel_column = o9Constants.PLANNING_CHANNEL
    demand_domain_column = o9Constants.PLANNING_DEMAND_DOMAIN
    sales_planning_channel_column = o9Constants.PLANNING_SALES_DEMAND
    version_column = o9Constants.VERSION_NAME
    bu_asp_mfp = o9Constants.BU_ASP
    product_attribute_group = o9Constants.PRODUCT_ATTRIBUTE_GROUP
    time_partial_week_column = o9Constants.PARTIAL_WEEK
    planning_account = o9Constants.PLANNING_ACCOUNT
    planning_region = o9Constants.PLANNING_REGION
    planning_pnl = o9Constants.PLANNING_PNL
    ty_asp = o9Constants.TY_ASP
    item = o9Constants.ITEM_L3
    location = o9Constants.LOCATION_COUNTRY
    selling_season_column = o9Constants.SELLING_SEASON

    OUTPUT_COLUMNS = [
        version_column,
        item,
        location,
        time_partial_week_column,
        planning_account,
        planning_pnl,
        planning_region,
        planning_channel_column,
        demand_domain_column,
        bu_asp_dp,
    ]


class Helper:
    """
    A Helper class that provides various utility functions for data aggregation and filtering.
    """

    @staticmethod
    def get_header(dataframe, substring):
        """
        Returns a list of column names from the dataframe that contain the specified substring.

        :param dataframe: DataFrame to extract headers from
        :param substring: Substring to search for in column names
        :return: List of matching column names
        """
        return [column for column in dataframe.columns if substring in column]

    @staticmethod
    def is_week(columns):
        """
        Checks if the time week column exists in the given list of columns.

        :param columns: List of column names
        :return: True if time week column is present, otherwise False
        """
        return True if Constants.time_week_column in columns else False

    @staticmethod
    def is_planning_month(columns):
        """
        Checks if the time planning month column exists in the given list of columns.

        :param columns: List of column names
        :return: True if time planning month column is present, otherwise False
        """
        return True if Constants.time_planning_month in columns else False

    @staticmethod
    def get_required_output_columns_buasp_week(columns):
        """
        Returns a list of required output columns for BUASP week, including necessary adjustments.

        :param columns: List of column names
        :return: Updated list with required output columns
        """
        return Helper.get_aggregation_dimensions_buasp(columns) + [
            Constants.time_partial_week_column,
            Constants.planning_account,
            Constants.planning_pnl,
            Constants.planning_region,
            Constants.bu_asp_dp,
        ]

    def apply_cartesian_products(df, master_datasets):
        """Apply Cartesian product with multiple master datasets."""
        for master_data in master_datasets:
            df = create_cartesian_product(df1=df, df2=master_data)
        return df

    @staticmethod
    def get_aggregation_dimensions_buasp(columns):
        """
        Returns a list of aggregation dimensions for BUASP excluding time week
        and product attribute group.

        :param columns: List of column names
        :return: Filtered list of column names
        """
        return [
            column
            for column in columns
            if column
            not in [
                Constants.time_week_column,
                Constants.product_attribute_group,
                Constants.time_planning_month,
            ]
        ]

    @staticmethod
    def get_required_columns_buasp(columns):
        """
        Returns a filtered list of required columns for BUASP

        :param columns: List of column names
        """
        return [column for column in columns]

    @staticmethod
    def get_required_columns_buasp_month(columns):
        """
        Returns a filtered list of required columns for BUASP, excluding specified columns.

        :param columns: List of column names
        :return: Filtered list excluding specific constant columns
        """
        return Helper.get_aggregation_dimensions_buasp(columns) + [
            Constants.time_partial_week_column,
            Constants.planning_account,
            Constants.planning_pnl,
            Constants.planning_region,
            Constants.bu_asp_dp,
        ]


def process_busap(
    current_time_period,
    merged_output_bu,
    account_master_data,
    regional_master_data,
    pnl_master_data,
    required_output_columns_bu,
    column_rename_mapping,
    logger,
):
    try:
        current_week = pd.to_datetime(current_time_period[Constants.time_week_column].max())
        merged_output_bu[Constants.time_week_column] = pd.to_datetime(
            merged_output_bu[Constants.time_week_column], errors="coerce"
        )
        df_past = merged_output_bu[
            merged_output_bu[Constants.time_week_column] < current_week
        ].copy()
        df_future = merged_output_bu[
            merged_output_bu[Constants.time_week_column] >= current_week
        ].copy()
        df_past = df_past.dropna(subset=[Constants.ty_asp])
        df_future = df_future.dropna(subset=[Constants.bu_asp_mfp])
        df_past[Constants.bu_asp_dp] = df_past[Constants.ty_asp]
        df_future[Constants.bu_asp_dp] = df_future[Constants.bu_asp_mfp]
        merged_output_bu = pd.concat([df_past, df_future], ignore_index=True)
        merged_output_bu = Helper.apply_cartesian_products(
            merged_output_bu, [account_master_data, regional_master_data, pnl_master_data]
        )
        buasp_output = merged_output_bu[required_output_columns_bu]
        buasp_output.rename(columns=column_rename_mapping, inplace=True)
        logger.info("Finished Processing")
        return buasp_output
    except Exception as e:
        logger.exception(e)
        raise Exception("Exception Occured in process busap....")


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    time_dimension,
    buasp_input,
    current_time_period,
    account_master_data,
    regional_master_data,
    pnl_master_data,
    df_keys,
    logger,
):
    """
    Main function to process BUASP input data for weekly and planning-month aggregation.

    Args:
        time_dimension (pd.DataFrame): DataFrame containing time dimension data.
        buasp_input (pd.DataFrame): DataFrame containing BUSAP input data.
        region_master (pd.DataFrame): DataFrame containing region master data.
        account_master (pd.DataFrame): DataFrame containing account master data.
        pnl_master (pd.DataFrame): DataFrame containing PnL master data.
        current_time_period (pd.DataFrame): DataFrame containing the current time period.

    Outputs:
    - buasp_output: Processed BUASP data.
    """

    plugin_name = "DP101MFPBUASP_DPHandshake"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    if (
        buasp_input.empty
        or time_dimension.empty
        or current_time_period.empty
        or account_master_data.empty
        or regional_master_data.empty
        or pnl_master_data.empty
    ):
        logger.warning(
            "One or more required input DataFrames are empty for slice {}.".format(df_keys)
        )
        logger.warning("No further execution for this slice.")
        return pd.DataFrame(columns=Constants.OUTPUT_COLUMNS)

    buasp_input_columns = Helper.get_header(buasp_input, ".[")
    column_rename_mapping = {
        Constants.sales_planning_channel_column: Constants.planning_channel_column,
        Constants.selling_season_column: Constants.demand_domain_column,
    }
    if Helper.is_week(columns=buasp_input_columns):
        logger.info("Processing for Week")

        aggregation_dimensions_buasp = Helper.get_aggregation_dimensions_buasp(buasp_input_columns)

        required_output_columns_buasp = Helper.get_required_output_columns_buasp_week(
            buasp_input_columns
        )
        try:
            input_dataframe_buasp = buasp_input[
                aggregation_dimensions_buasp
                + [Constants.time_week_column, Constants.bu_asp_mfp, Constants.ty_asp]
            ]
            partial_week_associations = time_dimension[
                [
                    Constants.time_week_column,
                    Constants.time_partial_week_column,
                    Constants.time_planning_month,
                ]
            ].drop_duplicates()

            merged_output_bu = input_dataframe_buasp.merge(
                partial_week_associations,
                on=[Constants.time_week_column],
                how="inner",
            )
            buasp_output = process_busap(
                current_time_period,
                merged_output_bu,
                account_master_data,
                regional_master_data,
                pnl_master_data,
                required_output_columns_buasp,
                column_rename_mapping,
                logger,
            )
            logger.info("Finished Processing {}...".format(plugin_name))
        except Exception as e:
            logger.exception(e)
    elif Helper.is_planning_month(columns=buasp_input_columns):
        logger.info("Processing for Planning Month")
        required_output_columns_bu = Helper.get_required_columns_buasp_month(buasp_input_columns)
        aggregation_dimensions_buasp = Helper.get_aggregation_dimensions_buasp(buasp_input_columns)
        aggregation_dimensions_buasp = Helper.get_required_columns_buasp(
            aggregation_dimensions_buasp
        )
        try:
            input_dataframe_bu = buasp_input[
                aggregation_dimensions_buasp
                + [Constants.time_planning_month, Constants.bu_asp_mfp, Constants.ty_asp]
            ]
            partial_week_associations = time_dimension[
                [
                    Constants.time_planning_month,
                    Constants.time_partial_week_column,
                    Constants.time_week_column,
                ]
            ].drop_duplicates()

            merged_output_bu = input_dataframe_bu.merge(
                partial_week_associations,
                on=[Constants.time_planning_month],
                how="inner",
            )
            buasp_output = process_busap(
                current_time_period,
                merged_output_bu,
                account_master_data,
                regional_master_data,
                pnl_master_data,
                required_output_columns_bu,
                column_rename_mapping,
                logger,
            )
            logger.info("Finished Processing {}...".format(plugin_name))
        except Exception as e:
            logger.exception(e)
            logger.exception("Exception occured:")
            buasp_output = pd.DataFrame(columns=Constants.OUTPUT_COLUMNS)
    return buasp_output

"""Helper function for DP101MFPBU_DPHanshake."""

import pandas as pd
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """A class to store constant values used throughout the module."""

    time_week_column = "Time.[Week]"
    adjusted_sales_units = "BU Total Sales Unit"
    adjusted_average_selling_price = "TY ASP DP Input"
    time_planning_month = "Time.[Planning Month]"
    bu_asp_dp = "BU Avg Selling Price"
    planning_channel_column = "Channel.[Planning Channel]"
    demand_domain_column = "Demand Domain.[Planning Demand Domain]"
    sales_planning_channel_column = "Sales Domain.[Sales Planning Channel]"
    selling_season_column = "Selling Season.[Selling Season]"
    current_year_sales_units = "TY Ttl Sls Unt"
    version_column = "Version.[Version Name]"
    bu_asp_mfp = "BU ASP"
    product_attribute_group = "Product Attribute Group.[Product Attribute Group]"
    time_partial_week_column = "Time.[Partial Week]"
    planning_account = "Account.[Planning Account]"
    planning_region = "Region.[Planning Region]"
    planning_pnl = "PnL.[Planning PnL]"
    ty_asp = "TY ASP"
    asp = "ASP"
    stat_bucket_weight = "Stat Bucket Weight"


class Helper:
    """A Helper class that provides various utility functions for data aggregation and filtering."""

    @staticmethod
    def get_aggregation_dimensions_mfpbu(columns):
        """
        Return a list of aggregation dimensions for MFPBU excluding time week and product attribute group.

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
            ]
        ]

    @staticmethod
    def get_aggregation_dimensions_for_planning_month_mfpbu(columns):
        """
        Return a list of aggregation dimensions for MFPBU excluding planning month and product attribute group.

        :param columns: List of column names
        :return: Filtered list of column names
        """
        return [
            column
            for column in columns
            if column
            not in [
                Constants.time_planning_month,
                Constants.product_attribute_group,
            ]
        ]

    @staticmethod
    def get_aggregation_dimensions_mfpbu_with_week(columns):
        """
        Return a list of columns including aggregation dimensions and time week column.

        :param columns: List of column names
        :return: Updated list with time week column included
        """
        return columns + [Constants.time_week_column]

    @staticmethod
    def get_required_output_columns_mfpbu_week(columns):
        """
        Return a list of required output columns for MFPBU week, including necessary adjustments.

        :param columns: List of column names
        :return: Updated list with required output columns
        """
        return Helper.get_aggregation_dimensions_mfpbu(columns) + [
            Constants.time_partial_week_column,
            Constants.planning_account,
            Constants.planning_pnl,
            Constants.planning_region,
            Constants.adjusted_sales_units,
        ]

    @staticmethod
    def get_header(dataframe, substring):
        """
        Return a list of column names from the dataframe that contain the specified substring.

        :param dataframe: DataFrame to extract headers from
        :param substring: Substring to search for in column names
        :return: List of matching column names
        """
        return [column for column in dataframe.columns if substring in column]

    @staticmethod
    def is_week(columns):
        """
        Check if the time week column exists in the given list of columns.

        :param columns: List of column names
        :return: True if time week column is present, otherwise False
        """
        return True if Constants.time_week_column in columns else False

    @staticmethod
    def is_planning_month(columns):
        """
        Check if the time planning month column exists in the given list of columns.

        :param columns: List of column names
        :return: True if time planning month column is present, otherwise False
        """
        return True if Constants.time_planning_month in columns else False

    @staticmethod
    def get_required_columns_mfpbu(columns):
        """
        Return a filtered list of required columns for MFPBU, excluding specified columns.

        :param columns: List of column names
        :return: Filtered list excluding specific constant columns
        """
        return [
            column
            for column in columns
            if column
            not in [
                Constants.current_year_sales_units,
            ]
        ]

    @staticmethod
    def get_required_output_columns_buasp_week(columns):
        """
        Return a list of required output columns for BUASP week, including necessary adjustments.

        :param columns: List of column names
        :return: Updated list with required output columns
        """
        return Helper.get_aggregation_dimensions_mfpbu(columns) + [
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


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    mfpbu_input,
    time_dimension,
    stat_bucket_weight,
    buasp_input,
    logger,
    account_master_data,
    regional_master_data,
    pnl_master_data,
    current_time_period,
    tyasp_input,
    df_keys,
):
    """
    To process MFPBU and BUASP input data for weekly and planning-month aggregation.

    Inputs:
    - mfpbu_input: DataFrame containing MFPBU input data.
    - buasp_input: DataFrame containing BUASP input data.
    - tyasp_input: DataFrame containing TYASP input data.
    - stat_bucket_weight: DataFrame containing statistical bucket weight data.
    - time_dimension: DataFrame containing time dimension data.

    Process:
    - If the data contains weekly information, the function:
      - Extracts necessary columns and renames them as needed.
      - Aggregates and merges input data with statistical weights.
      - Computes adjusted sales units and average selling price.
      - Outputs the processed data.
    - If the data is for planning months, the function:
      - Aggregates data at the planning month level.
      - Computes ratios for time-based adjustments.
      - Merges data with statistical weights and computes adjusted sales values.
      - Outputs the processed data.

    Outputs:
    - mfpbu_output: Processed MFPBU data.
    - buasp_output: Processed BUASP data.
    """
    plugin_name = "DP101MFPBU_DPHandshake"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    if len(mfpbu_input) == 0:
        logger.warning("MFPBU Input data is empty.")
    mfpbu_input_columns = Helper.get_header(mfpbu_input, ".[")
    mfpbu_output = None
    buasp_output = None
    column_rename_mapping = {
        Constants.sales_planning_channel_column: Constants.planning_channel_column,
        Constants.selling_season_column: Constants.demand_domain_column,
    }

    if Helper.is_week(columns=mfpbu_input_columns):
        logger.info("Processing for Week")
        aggregation_dimensions_mfpbu = Helper.get_aggregation_dimensions_mfpbu(mfpbu_input_columns)

        aggregation_dimensions_mfpbu_with_week = aggregation_dimensions_mfpbu + [
            Constants.time_week_column
        ]

        required_output_columns_mfpbu = Helper.get_required_output_columns_mfpbu_week(
            mfpbu_input_columns
        )
        required_output_columns_buasp = Helper.get_required_output_columns_buasp_week(
            mfpbu_input_columns
        )
        try:
            input_dataframe_mfpbu = mfpbu_input[
                aggregation_dimensions_mfpbu
                + [
                    Constants.time_week_column,
                    Constants.current_year_sales_units,
                ]
            ]

            input_dataframe_buasp = buasp_input[
                aggregation_dimensions_mfpbu + [Constants.time_week_column, Constants.bu_asp_mfp]
            ]

            input_dataframe_tyasp = tyasp_input[
                aggregation_dimensions_mfpbu + [Constants.time_week_column, Constants.ty_asp]
            ]

            common_columns = [
                col for col in input_dataframe_buasp.columns if col != Constants.bu_asp_mfp
            ]
            input_dataframe_buasp = pd.merge(
                input_dataframe_buasp, input_dataframe_tyasp, on=common_columns, how="outer"
            )
            weekly_aggregated_data = (
                input_dataframe_mfpbu.groupby(aggregation_dimensions_mfpbu_with_week)
                .agg(
                    {
                        Constants.current_year_sales_units: "sum",
                    }
                )
                .reset_index()
            )
            cleaned_stat_bucket_weight = stat_bucket_weight.drop(
                columns=["Forecast Iteration.[Forecast Iteration]"]
            )

            cleaned_stat_bucket_weight = cleaned_stat_bucket_weight.drop_duplicates()
            partial_week_associations = time_dimension[
                [
                    Constants.time_week_column,
                    Constants.time_partial_week_column,
                ]
            ].drop_duplicates()

            stat_bucket_weight_associations = cleaned_stat_bucket_weight.merge(
                partial_week_associations,
                on=[Constants.time_partial_week_column],
                how="inner",
            )

            merged_output = weekly_aggregated_data.merge(
                stat_bucket_weight_associations,
                on=[Constants.time_week_column, Constants.version_column],
                how="inner",
            )

            merged_output[Constants.adjusted_sales_units] = (
                merged_output[Constants.current_year_sales_units]
                * merged_output[Constants.stat_bucket_weight]
            )
            merged_output = Helper.apply_cartesian_products(
                merged_output, [account_master_data, regional_master_data, pnl_master_data]
            )
            mfpbu_output = merged_output[required_output_columns_mfpbu].copy()

            mfpbu_output = mfpbu_output.rename(columns=column_rename_mapping)

            merged_output_bu = input_dataframe_buasp.merge(
                stat_bucket_weight_associations,
                on=[Constants.time_week_column, Constants.version_column],
                how="inner",
            )

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

            df_past[Constants.asp] = df_past[Constants.ty_asp]
            df_future[Constants.asp] = df_future[Constants.bu_asp_mfp]
            merged_output_bu = pd.concat([df_past, df_future], ignore_index=True)
            merged_output_bu[Constants.bu_asp_dp] = (
                merged_output_bu[Constants.asp] * merged_output_bu[Constants.stat_bucket_weight]
            )
            merged_output_bu = Helper.apply_cartesian_products(
                merged_output_bu, [account_master_data, regional_master_data, pnl_master_data]
            )
            buasp_output = merged_output_bu[required_output_columns_buasp]
            buasp_output.rename(columns=column_rename_mapping, inplace=True)

            logger.info("finished processing")
        except Exception as e:
            logger.exception(e)
    elif Helper.is_planning_month(columns=mfpbu_input_columns):
        logger.info("Processing for Planning Month")
        try:
            mfpbu_input_columns = Helper.get_header(mfpbu_input, ".[")
            aggregation_dimensions_mfpbu = (
                Helper.get_aggregation_dimensions_for_planning_month_mfpbu(mfpbu_input_columns)
            )
            aggregation_dimensions = Helper.get_required_columns_mfpbu(
                aggregation_dimensions_mfpbu
            ) + [
                Constants.time_partial_week_column,
                Constants.time_planning_month,
            ]

            input_dataframe_mfpbu = mfpbu_input[
                aggregation_dimensions_mfpbu
                + [
                    Constants.time_planning_month,
                    Constants.current_year_sales_units,
                ]
            ]

            aggregated = (
                time_dimension.groupby(
                    [
                        Constants.time_planning_month,
                        Constants.time_partial_week_column,
                    ]
                )
                .agg(
                    count=(Constants.time_partial_week_column, "size"),
                    partial_week_sum=(
                        Constants.time_partial_week_column,
                        "count",
                    ),
                )
                .reset_index()
            )

            total_days_in_pm = (
                time_dimension.groupby(Constants.time_planning_month)
                .size()
                .reset_index(name="total_days_in_pm")
            )

            combined = aggregated.merge(
                total_days_in_pm, on=Constants.time_planning_month, how="left"
            )

            combined["ratio"] = combined["partial_week_sum"] / combined["total_days_in_pm"]

            final_dataframe = input_dataframe_mfpbu.merge(
                combined,
                left_on=[Constants.time_planning_month],
                right_on=[Constants.time_planning_month],
                how="left",
            )

            final_dataframe = final_dataframe.loc[:, ~final_dataframe.columns.duplicated()]
            final_dataframe[Constants.current_year_sales_units] = (
                final_dataframe[Constants.current_year_sales_units] * final_dataframe["ratio"]
            )

            weekly_aggregated_data = (
                final_dataframe.groupby(aggregation_dimensions)
                .agg(
                    {
                        Constants.current_year_sales_units: "sum",
                    }
                )
                .reset_index()
            )
            cleaned_stat_bucket_weight = stat_bucket_weight.drop(
                columns=["Forecast Iteration.[Forecast Iteration]"]
            )
            cleaned_stat_bucket_weight = cleaned_stat_bucket_weight.drop_duplicates()
            partial_week_associations = time_dimension[
                [
                    Constants.time_planning_month,
                    Constants.time_partial_week_column,
                ]
            ].drop_duplicates()

            stat_bucket_weight_associations = cleaned_stat_bucket_weight.merge(
                partial_week_associations,
                on=[Constants.time_partial_week_column],
                how="inner",
            )
            merged_output = weekly_aggregated_data.merge(
                stat_bucket_weight_associations,
                on=[
                    Constants.time_partial_week_column,
                    Constants.version_column,
                ],
                how="inner",
            )
            merged_output[Constants.adjusted_sales_units] = (
                merged_output[Constants.current_year_sales_units]
                * merged_output[Constants.stat_bucket_weight]
            )
            merged_output = Helper.apply_cartesian_products(
                merged_output, [account_master_data, regional_master_data, pnl_master_data]
            )

            required_output_columns_mfpbu = Helper.get_required_columns_mfpbu(
                aggregation_dimensions_mfpbu
            ) + [
                Constants.time_partial_week_column,
                Constants.adjusted_sales_units,
            ]
            mfpbu_output = merged_output[required_output_columns_mfpbu].copy()

            mfpbu_output = mfpbu_output.rename(columns=column_rename_mapping)

            aggregation_dimensions_buasp = Helper.get_required_columns_mfpbu(
                aggregation_dimensions_mfpbu
            )
            input_dataframe_bu = buasp_input[
                aggregation_dimensions_buasp + [Constants.time_planning_month, Constants.bu_asp_mfp]
            ]
            input_dataframe_tyasp = tyasp_input[
                aggregation_dimensions_buasp + [Constants.time_planning_month, Constants.ty_asp]
            ]
            common_columns = [
                col for col in input_dataframe_bu.columns if col != Constants.bu_asp_mfp
            ]

            input_dataframe_bu = pd.merge(
                input_dataframe_bu, input_dataframe_tyasp, on=common_columns, how="outer"
            )

            partial_week_df = time_dimension[
                [
                    Constants.time_planning_month,
                    Constants.time_partial_week_column,
                ]
            ]

            input_dataframe_bu = input_dataframe_bu.merge(
                partial_week_df, on=Constants.time_planning_month, how="left"
            ).drop_duplicates()

            merged_output_bu = input_dataframe_bu.merge(
                stat_bucket_weight_associations,
                on=[
                    Constants.time_partial_week_column,
                    Constants.version_column,
                ],
                how="inner",
            )
            current_week = pd.to_datetime(current_time_period[Constants.time_week_column].max())
            merged_output_bu[Constants.time_week_column] = pd.to_datetime(
                merged_output_bu[Constants.time_week_column], errors="coerce"
            )
            df_past = merged_output_bu[
                merged_output_bu[Constants.time_partial_week_column] < current_week
            ].copy()
            df_future = merged_output_bu[
                merged_output_bu[Constants.time_partial_week_column] >= current_week
            ].copy()
            df_past = df_past.dropna(subset=[Constants.ty_asp])
            df_future = df_future.dropna(subset=[Constants.bu_asp_mfp])

            df_past[Constants.asp] = df_past[Constants.ty_asp]
            df_future[Constants.asp] = df_future[Constants.bu_asp_mfp]
            merged_output_bu = pd.concat([df_past, df_future], ignore_index=True)
            merged_output_bu[Constants.bu_asp_dp] = (
                merged_output_bu[Constants.asp] * merged_output_bu[Constants.stat_bucket_weight]
            )
            merged_output_bu = Helper.apply_cartesian_products(
                merged_output_bu, [account_master_data, regional_master_data, pnl_master_data]
            )
            required_output_columns_bu = aggregation_dimensions_buasp + [
                Constants.time_partial_week_column,
                Constants.bu_asp_dp,
            ]
            buasp_output = merged_output_bu[required_output_columns_bu]
            buasp_output.rename(
                columns={
                    Constants.sales_planning_channel_column: Constants.planning_channel_column,
                    Constants.selling_season_column: Constants.demand_domain_column,
                },
                inplace=True,
            )
            logger.info("Finished Processing")
        except Exception as e:
            logger.exception(e)
    return mfpbu_output, buasp_output

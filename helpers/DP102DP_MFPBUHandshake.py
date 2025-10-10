"""
Module to process Selling Season, Statistical Forecast, and Time Dimension data.

for generating BU Forecast Unit outputs as part of DP102DP_MFPBUHandshake.
Handles data validation, processing, merging, and output formatting with logging and error handling.
"""

import numpy as np
import pandas as pd
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import ColumnNamer

from helpers.o9Constants import o9Constants

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_namer = ColumnNamer()


class Constants:
    """A class to store constant values used throughout the module."""

    time_week_column = o9Constants.WEEK
    sales_planning_channel_column = o9Constants.PLANNING_SALES_DEMAND
    selling_season_column = o9Constants.SELLING_SEASON
    version_column = o9Constants.VERSION_NAME
    time_partial_week_column = o9Constants.PARTIAL_WEEK
    forecast_iteration = o9Constants.FORECAST_ITERATION
    stat_account = o9Constants.STAT_ACCOUNT
    stat_pnl = o9Constants.STAT_PNL
    stat_region = o9Constants.STAT_REGION
    item_l3 = o9Constants.ITEM_L3
    location_country = o9Constants.LOCATION_COUNTRY
    stat_demand_domain = o9Constants.STAT_DEMAND_DOMAIN
    start_week = o9Constants.START_WEEK_L3_SS
    end_week = o9Constants.END_WEEK_L3_SS
    bu_forecast_unit = o9Constants.BU_FORECAST_UNIT
    stat_item = o9Constants.STAT_ITEM
    stat_location = o9Constants.STAT_LOCATION
    stat_channel = o9Constants.STAT_CHANNEL
    week_name = o9Constants.TIME_WEEKNAME
    time_week_key = o9Constants.WEEK_KEY
    sales_unit = o9Constants.TOTAL_SALES_UNIT
    time_year = o9Constants.TIME_YEAR
    stat_forecast_l1 = o9Constants.STAT_FCST_L1
    sellingseason = o9Constants.SELLING_SEASON

    planning_demand_domain = "planning_demand_domain"
    disaggregation_sum = "disaggregation_sum"
    disaggregation_ratio = "disaggregation_ratio"
    week_Association = "week_Association"
    selling_season = "Selling_Season"
    week_year = "week_year"
    normalized_ratio = "normalized_ratio"
    fi_mfp_bu = "FI-MFP BU"
    core = "Core"
    week = "Week"
    total_sales_sum = "total_sales_sum"

    demand_domain_serach_columns = ["Item", "Demand", "Location", "Channel"]
    selling_season_search_columns = ["Item", "Selling", "Location", "Channel"]


def process_total_sales_unit(
    total_sales_unit, selling_season_df, time_dimension_df, current_time_period, logger
):
    """
    Processes and aggregates total sales unit data by aligning it with selling season and time dimension data,
    calculates disaggregation sums and ratios used for forecasting.
    """
    try:
        item, demand_domain, location, channel = Helper.find_elements(
            total_sales_unit, Constants.demand_domain_serach_columns
        )
        rename_column = {
            channel: Constants.sales_planning_channel_column,
            demand_domain: Constants.sellingseason,
        }
        disagg_group = [
            Constants.version_column,
            item,
            Constants.sales_planning_channel_column,
            location,
            Constants.week_name,
        ]
        current_week = pd.to_datetime(
            current_time_period[Constants.time_partial_week_column].max(), errors="coerce"
        )

        group_by = disagg_group + [Constants.sellingseason]
        disagg_group_with_demand_domain = disagg_group + [Constants.planning_demand_domain]

        total_sales_unit = total_sales_unit[
            [
                Constants.version_column,
                item,
                location,
                channel,
                demand_domain,
                Constants.time_partial_week_column,
                Constants.sales_unit,
            ]
        ]
        partial_week_mapping = time_dimension_df[
            [
                Constants.time_week_column,
                Constants.time_partial_week_column,
                Constants.week_name,
                Constants.week_year,
            ]
        ].drop_duplicates()

        total_sales_unit = total_sales_unit.rename(columns=rename_column)

        total_sales_unit = total_sales_unit.merge(
            partial_week_mapping, on=[Constants.time_partial_week_column], how="inner"
        )
        total_sales_unit[Constants.week] = pd.to_datetime(
            total_sales_unit[Constants.time_partial_week_column], errors="coerce"
        )
        total_sales_unit = total_sales_unit[total_sales_unit[Constants.week] <= current_week]

        selling_season_df = selling_season_df[selling_season_df[Constants.week_Association] == 1]
        total_sales_unit = selling_season_df.merge(
            total_sales_unit, on=group_by + [Constants.time_week_column], how="inner"
        )

        total_sales_unit[Constants.planning_demand_domain] = np.where(
            total_sales_unit[Constants.sellingseason] == Constants.core,
            Constants.core,
            total_sales_unit[Constants.sellingseason].str.replace(r"\d+", "", regex=True),
        )

        exclude_cols = set(group_by + [Constants.planning_demand_domain, Constants.sales_unit])
        other_cols = [col for col in total_sales_unit.columns if col not in exclude_cols]

        agg_dict = {Constants.sales_unit: "sum"}
        agg_dict.update({col: "first" for col in other_cols})

        total_sales_unit = total_sales_unit.groupby(
            group_by + [Constants.planning_demand_domain], as_index=False
        ).agg(agg_dict)

        disagg_sums_df = (
            total_sales_unit.groupby(disagg_group, as_index=False)[Constants.sales_unit]
            .sum()
            .rename(columns={Constants.sales_unit: Constants.disaggregation_sum})
        )
        total_sales_unit = total_sales_unit.merge(disagg_sums_df, on=disagg_group, how="left")
        disagg_sums_df = total_sales_unit.groupby(
            disagg_group_with_demand_domain, as_index=False
        ).agg(total_sales_sum=(Constants.sales_unit, "sum"))
        total_sales_unit = total_sales_unit.merge(
            disagg_sums_df, on=disagg_group_with_demand_domain, how="left"
        )
        total_sales_unit[Constants.disaggregation_ratio] = (
            total_sales_unit[Constants.total_sales_sum]
            / total_sales_unit[Constants.disaggregation_sum]
        )
        logger.info("Successfully processed total_sales_unit...")
        return total_sales_unit
    except Exception as e:
        raise Exception(f"Error Occurred: {e}")


def process_stat_forecast(time_dimension_df, stat_forecast_df, current_time_period, logger):
    """
    Cleans and aggregates statistical forecast data by week dimensions.

    Args:
            time_dimension_df (DataFrame): Time dimension data.
            stat_forecast_df (DataFrame): Statistical forecast data.
            current_Time_period (DataFrame): Current Time period data.
            logger (Logger): Logger instance.

    Returns:
            DataFrame: Aggregated statistical forecast data by item, location, channel, demand domain, week, week_name and version.
    """
    try:
        partial_week_mapping = time_dimension_df[
            [
                Constants.time_week_column,
                Constants.time_partial_week_column,
                Constants.week_name,
            ]
        ].drop_duplicates()
        stat_forecast_df = stat_forecast_df[
            stat_forecast_df[Constants.forecast_iteration] == Constants.fi_mfp_bu
        ]
        stat_forecast_cleaned = stat_forecast_df.drop(
            columns=[
                Constants.forecast_iteration,
                Constants.stat_account,
                Constants.stat_region,
                Constants.stat_pnl,
            ]
        )
        current_week = pd.to_datetime(
            current_time_period[Constants.time_partial_week_column].max(), errors="coerce"
        )
        stat_forecast_cleaned[Constants.week] = pd.to_datetime(
            stat_forecast_cleaned[Constants.time_partial_week_column], errors="coerce"
        )
        stat_forecast_cleaned = stat_forecast_cleaned[
            stat_forecast_cleaned[Constants.week] >= current_week
        ]
        stat_with_week = stat_forecast_cleaned.merge(
            partial_week_mapping, on=[Constants.time_partial_week_column], how="inner"
        )
        aggregated_df = (
            stat_with_week.groupby(
                [
                    Constants.stat_item,
                    Constants.stat_location,
                    Constants.stat_channel,
                    Constants.stat_demand_domain,
                    Constants.week_name,
                    Constants.time_week_column,
                    Constants.version_column,
                ]
            )[Constants.stat_forecast_l1]
            .sum()
            .reset_index()
        )
        logger.info("Successfully processed stat forecast")
        return aggregated_df
    except Exception as e:
        raise Exception(f"Error Occurred: {e}")


def process_demand_domain(stat_forecast_df, selling_season_df, total_sales_unit_df, logger):
    """
    Merges statistical forecast, and MFPBU data to compute BU forecast units.

    Args:
            selling_season_df (DataFrame): Selling season data.
            total_sales_unit_df (DataFrame): Total sales unit data.
            logger (Logger): Logger instance.

    Returns:
            DataFrame: Final DataFrame with BU forecast units.

    Raises:
            Exception: If processing fails.
    """
    try:
        item_col, demand_col, location_col, channel_col = Helper.find_elements(
            stat_forecast_df, Constants.demand_domain_serach_columns
        )
        sales_item_col, sales_demand_col, sales_location_col, sales_channel_col = (
            Helper.find_elements(total_sales_unit_df, Constants.selling_season_search_columns)
        )
        rename_mapping = {
            item_col: sales_item_col,
            demand_col: sales_demand_col,
            location_col: sales_location_col,
            channel_col: sales_channel_col,
        }
        output_columns = [
            sales_item_col,
            sales_location_col,
            Constants.sales_planning_channel_column,
            Constants.sellingseason,
            Constants.time_week_column,
            Constants.version_column,
            Constants.bu_forecast_unit,
        ]

        groupby_cols = [
            Constants.version_column,
            sales_item_col,
            sales_location_col,
            Constants.sales_planning_channel_column,
            Constants.week_name,
            Constants.time_week_column,
        ]

        stat_forecast_df.rename(columns=rename_mapping, inplace=True)
        stat_forecast_df.drop(columns=[Constants.selling_season_column], inplace=True)
        stat_forecast_df = pd.merge(
            stat_forecast_df, selling_season_df, on=groupby_cols, how="left"
        )
        stat_forecast_df = stat_forecast_df.dropna()
        stat_forecast_df = stat_forecast_df.drop_duplicates()
        stat_forecast_df[Constants.planning_demand_domain] = np.where(
            stat_forecast_df[Constants.sellingseason] == Constants.core,
            Constants.core,
            stat_forecast_df[Constants.sellingseason].str.replace(r"\d+", "", regex=True),
        )
        total_sales_unit_df = total_sales_unit_df.drop_duplicates(
            subset=[
                Constants.week_name,
                Constants.planning_demand_domain,
                Constants.location_country,
                sales_item_col,
            ]
        )
        total_sales_unit_df = total_sales_unit_df[
            [
                Constants.version_column,
                sales_item_col,
                sales_location_col,
                sales_demand_col,
                Constants.disaggregation_ratio,
                Constants.disaggregation_sum,
                Constants.sales_unit,
                Constants.week_Association,
                Constants.sales_planning_channel_column,
                Constants.total_sales_sum,
                Constants.week_name,
                Constants.planning_demand_domain,
            ]
        ]
        stat_forecast_df = pd.merge(
            stat_forecast_df,
            total_sales_unit_df,
            on=[
                Constants.version_column,
                sales_item_col,
                Constants.sales_planning_channel_column,
                sales_location_col,
                Constants.week_name,
                Constants.planning_demand_domain,
            ],
            how="left",
        )
        stat_forecast_df = stat_forecast_df.rename(
            columns={Constants.sellingseason + "_x": Constants.sellingseason}
        )
        stat_forecast_df = Helper.normalize_column_by_group(
            stat_forecast_df,
            group_cols=[
                Constants.week_name,
                Constants.version_column,
                sales_location_col,
                sales_item_col,
            ],
            value_col=Constants.disaggregation_ratio,
            normalized_col=Constants.normalized_ratio,
        )
        stat_forecast_df[Constants.bu_forecast_unit] = (
            stat_forecast_df[Constants.normalized_ratio]
            * stat_forecast_df[Constants.week_Association + "_x"]
            * stat_forecast_df[Constants.stat_forecast_l1]
        )
        stat_forecast_df[Constants.bu_forecast_unit].fillna(
            stat_forecast_df[Constants.stat_forecast_l1], inplace=True
        )
        logger.info("Successfully processed demand domain.....")
        return stat_forecast_df[output_columns]
    except Exception as e:
        raise Exception(f"Error Occurred: {e}")


def process_selling_Season(selling_season_df, time_dim_df, logger):
    """
    Processes the selling season DataFrame by associating time weeks within the start and end week range.
    """
    try:
        logger.info("Processing Selling Season DataFrame...")

        item, location = Helper.find_elements(selling_season_df, ["Item", "Location"])

        required_cols = [
            Constants.start_week,
            Constants.end_week,
            item,
            location,
            Constants.sales_planning_channel_column,
            Constants.selling_season_column,
            Constants.version_column,
            "key",
        ]
        selling_season_df["key"] = 1
        time_dim_df["key"] = 1
        merged = pd.merge(
            selling_season_df[required_cols],
            time_dim_df[
                [Constants.time_week_key, Constants.week_name, Constants.time_week_column, "key"]
            ],
            on="key",
        ).drop("key", axis=1)
        selling_season_df = merged[
            (merged[Constants.time_week_key] >= merged[Constants.start_week])
            & (merged[Constants.time_week_key] <= merged[Constants.end_week])
        ].copy()
        selling_season_df[Constants.week_Association] = 1
        logger.info("successfully processed selling season...")
        return selling_season_df
    except Exception as e:
        raise Exception(f"Error Occurred: {e}")


class Helper:

    def find_elements(df, search_terms):
        """
        Finds and returns a list of column names in the DataFrame that correspond to the given search terms.
        """
        return [Helper.find_element(df, term) for term in search_terms]

    def find_element(lst, keyword):
        """
        Finds and returns the first element in the list containing the keyword.

        Parameters:
        - lst: list of strings
        - keyword: partial string to match

        Returns:
        - The matched element (if found), else None
        """
        return next((x for x in lst if keyword in x), None)

    def process_time_dimension(time_dimension_df):
        """
        Combine Year and Week Name
        """
        time_dimension_df[Constants.week_year] = (
            time_dimension_df[Constants.week_name]
            + " "
            + time_dimension_df[Constants.time_year].astype(str)
        )
        return time_dimension_df

    def normalize_column_by_group(df, group_cols, value_col, normalized_col=None):
        """
        Normalize the values in `value_col` so that they sum to 1 within each group defined by `group_cols`.

        Parameters:
        - df: pandas DataFrame
        - group_cols: list of column names to group by (e.g., ['week_year'])
        - value_col: column name with values to normalize (e.g., 'disaggregation_ratio')
        - normalized_col: name for the output normalized column; if None, overwrites value_col

        Returns:
        - DataFrame with new or updated normalized column
        """
        if normalized_col is None:
            normalized_col = value_col

        df[normalized_col] = df.groupby(group_cols)[value_col].transform(lambda x: x / x.sum())
        return df


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    stat_forecast,
    time_dimension_df,
    current_time_period,
    total_sales_unit,
    selling_season,
    logger,
    df_keys,
):
    try:
        plugin_name = "DP102DP_MFPBUHandshake"
        logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

        if (
            stat_forecast.empty
            or time_dimension_df.empty
            or current_time_period.empty
            or total_sales_unit.empty
            or selling_season.empty
        ):
            raise Exception("Input cannot be Empty...")

        time_dimension_df = Helper.process_time_dimension(time_dimension_df)

        selling_season_df = process_selling_Season(selling_season, time_dimension_df, logger)
        stat_forecast_df = process_stat_forecast(
            time_dimension_df, stat_forecast, current_time_period, logger
        )
        total_sales_unit_df = process_total_sales_unit(
            total_sales_unit, selling_season_df, time_dimension_df, current_time_period, logger
        )
        processed_result = process_demand_domain(
            stat_forecast_df, selling_season_df, total_sales_unit_df, logger
        )

        BUForecastUnt = col_namer.convert_to_o9_cols(df=processed_result)
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        columns = [
            Constants.version_column,
            Constants.item_l3,
            Constants.location_country,
            Constants.sales_planning_channel_column,
            Constants.selling_season_column,
            Constants.time_week_column,
            Constants.bu_forecast_unit,
        ]
        BUForecastUnt = pd.DataFrame(columns=columns)
    return BUForecastUnt

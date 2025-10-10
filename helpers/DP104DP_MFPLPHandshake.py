"""Helper function for DP104DP_MFPLPHandshake."""

import pandas as pd
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
    """A class to store constant values used throughout the module."""

    time_week_column = o9Constants.WEEK
    sales_planning_channel_column = o9Constants.PLANNING_SALES_DEMAND
    selling_season_column = o9Constants.SELLING_SEASON
    version_column = o9Constants.VERSION_NAME
    forecast_iteration = o9Constants.FORECAST_ITERATION
    item_l4 = o9Constants.ITEM_L4
    location = o9Constants.LOCATION
    start_week = o9Constants.START_WEEK_L3_SS
    end_week = o9Constants.END_WEEK_L3_SS
    stat_item = o9Constants.STAT_ITEM
    stat_location = o9Constants.STAT_LOCATION
    stat_channel = o9Constants.STAT_CHANNEL
    time_week_key = o9Constants.WEEK_KEY
    stat_forecast_l1 = o9Constants.STAT_FCST_L1
    planning_month = o9Constants.PLANNING_MONTH
    lp_forecast_unit = o9Constants.LP_FORECAST_UNIT

    selling_season = "Selling_Season"
    fi_mfp_lp = "FI-MFP LP"
    core = "Core"
    week = "Week"


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


def process_selling_Season(selling_season_df, time_dim_df, current_time_period, logger):
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
                [
                    Constants.time_week_key,
                    Constants.time_week_column,
                    Constants.planning_month,
                    "key",
                ]
            ],
            on="key",
        ).drop("key", axis=1)
        current_week = current_time_period[Constants.time_week_column].max()

        selling_season_df = merged[
            (merged[Constants.time_week_key] >= merged[Constants.start_week])
            & (merged[Constants.time_week_key] <= merged[Constants.end_week])
            & (merged[Constants.start_week] >= current_week)
            & (merged[Constants.end_week] >= current_week)
        ].copy()
        selling_season_df = selling_season_df.drop_duplicates()
        logger.info("successfully processed selling season...")
        return selling_season_df
    except Exception as e:
        raise Exception(f"Error Occurred: {e}")


def process_stat_forecast(stat_fcst, selling_season_df, current_time_period, logger):
    """
    Processes the StatForecast DataFrame to create a LP Forecast Unit DataFrame.
    """
    try:
        logger.info("Processing Stat Forecast DataFrame...")
        selling_item, selling_location = Helper.find_elements(
            selling_season_df, ["Item", "Location"]
        )
        stat_item, stat_location = Helper.find_elements(stat_fcst, ["Item", "Location"])

        group_keys = [
            Constants.version_column,
            Constants.sales_planning_channel_column,
            Constants.planning_month,
            selling_item,
            selling_location,
        ]

        current_week = pd.to_datetime(
            current_time_period[Constants.time_week_column].max(), errors="coerce"
        )
        stat_fcst[Constants.week] = pd.to_datetime(
            stat_fcst[Constants.time_week_column], errors="coerce"
        )
        stat_fcst = stat_fcst[stat_fcst[Constants.week] >= current_week]
        stat_forecast_df = stat_fcst[stat_fcst[Constants.forecast_iteration] == Constants.fi_mfp_lp]
        stat_forecast_df = stat_forecast_df.drop(columns=[Constants.forecast_iteration])

        stat_forecast_df = stat_forecast_df.rename(
            columns={
                stat_item: selling_item,
                stat_location: selling_location,
                Constants.stat_channel: Constants.sales_planning_channel_column,
            }
        )
        stat_forecast_df = pd.merge(
            stat_forecast_df,
            selling_season_df[group_keys + [Constants.selling_season_column]],
            on=group_keys,
            how="left",
        )
        stat_forecast_df = stat_forecast_df.dropna()

        season_counts = (
            stat_forecast_df.groupby(group_keys)[Constants.selling_season_column]
            .agg(lambda x: set(s for s in x.unique() if s != Constants.core))
            .reset_index(name=Constants.selling_season)
        )
        season_counts["Count"] = season_counts[Constants.selling_season].apply(len)
        stat_forecast_df = stat_forecast_df.merge(
            season_counts[group_keys + ["Count"]], on=group_keys, how="left"
        )
        stat_forecast_df[Constants.lp_forecast_unit] = stat_forecast_df[
            Constants.stat_forecast_l1
        ] / stat_forecast_df["Count"].replace(0, pd.NA)
        required_columns = group_keys + [
            Constants.selling_season_column,
            Constants.lp_forecast_unit,
        ]
        stat_forecast_df = stat_forecast_df.drop_duplicates()
        stat_forecast_df = stat_forecast_df.groupby(
            group_keys + [Constants.selling_season_column], as_index=False
        )[Constants.lp_forecast_unit].sum()
        logger.info("Successfully processed Stat Forecast DataFrame...")
        return stat_forecast_df[required_columns]
    except Exception as e:
        raise Exception(f"Exception Occured {e}")


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(stat_fcst, selling_season, time_df, current_time_period, logger, df_keys):
    """
    Main function for processing DP104DP_MFPLPHandshake.
    """
    try:
        plugin_name = "DP104DP_MFPLPHandshake"
        logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
        selling_season_df = process_selling_Season(
            selling_season_df=selling_season,
            time_dim_df=time_df,
            current_time_period=current_time_period,
            logger=logger,
        )
        lp_forecast_unit = process_stat_forecast(
            stat_fcst=stat_fcst,
            selling_season_df=selling_season_df,
            current_time_period=current_time_period,
            logger=logger,
        )
        logger.info("Successfully executed {} for slice {}".format(plugin_name, df_keys))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        columns = [
            Constants.version_column,
            Constants.item_l4,
            Constants.location,
            Constants.sales_planning_channel_column,
            Constants.selling_season_column,
            Constants.planning_month,
            Constants.lp_forecast_unit,
        ]
        lp_forecast_unit = pd.DataFrame(columns=columns)
    return lp_forecast_unit

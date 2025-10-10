import logging
from typing import Dict, List

import pandas as pd
import polars as pl
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension

from helpers.o9Constants import o9Constants
from helpers.utils import get_list_of_grains_from_string

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

col_mapping = {}

logger = logging.getLogger("o9_logger")


class constants:
    version_col = o9Constants.VERSION_NAME
    forecast_iteration = o9Constants.FORECAST_ITERATION
    planning_item_col = o9Constants.PLANNING_ITEM
    planning_location_col = o9Constants.PLANNING_LOCATION
    planning_channel_col = o9Constants.PLANNING_CHANNEL
    planning_region_col = o9Constants.PLANNING_REGION
    planning_account_col = o9Constants.PLANNING_ACCOUNT
    planning_pnl_col = o9Constants.PLANNING_PNL
    planning_dem_dom_col = o9Constants.PLANNING_DEMAND_DOMAIN

    stat_item_col = o9Constants.STAT_ITEM
    stat_location_col = o9Constants.STAT_LOCATION
    stat_channel_col = o9Constants.STAT_CHANNEL
    stat_region_col = o9Constants.STAT_REGION
    stat_account_col = o9Constants.STAT_ACCOUNT
    stat_pnl_col = o9Constants.STAT_PNL
    stat_dem_dom_col = o9Constants.STAT_DEMAND_DOMAIN

    item_level = "Item Level"
    channel_level = "Channel Level"
    account_level = "Account Level"
    region_level = "Region Level"
    pnl_level = "PnL Level"
    dem_dom_level = "Demand Domain Level"
    location_level = "Location Level"

    # output columns
    item_members = "Stat Item Members"
    location_members = "Stat Location Members"
    channel_members = "Stat Channel Members"
    region_members = "Stat Region Members"
    account_members = "Stat Account Members"
    pnl_members = "Stat PnL Members"
    dem_dom_members = "Stat Demand Domain Members"

    cols_req_for_item_members = [version_col, planning_item_col, item_members]
    cols_req_for_location_members = [version_col, planning_location_col, location_members]
    cols_req_for_channel_members = [version_col, planning_channel_col, channel_members]
    cols_req_for_region_members = [version_col, planning_region_col, region_members]
    cols_req_for_account_members = [version_col, planning_account_col, account_members]
    cols_req_for_pnl_members = [version_col, planning_pnl_col, pnl_members]
    cols_req_for_dem_dom_members = [version_col, planning_dem_dom_col, dem_dom_members]


def get_formatted_uniques(
    df: pl.DataFrame,
    columns_values: List[str],
    master_df_dict: Dict,
    keep: bool = True,  # to check whether checking for deleting (False) or non-deleting values (True)
) -> Dict[str, List[str]]:
    """
    For each column in columns_values, returns a dict with column names as keys
    and formatted unique values as column and their values.

    """
    result_dict = {}
    for col in columns_values:
        values = (
            df.lazy()
            .select(pl.col(col).unique().sort())
            .select(
                pl.col(col)
                .map_elements(
                    lambda x: f"{col.split()[0]}.[{x}]",
                    return_dtype=pl.String,
                )
                .alias(col)
            )
            .collect()
            .to_series()
            .to_list()
        )
        stat_col = master_df_dict.get(col)[5]
        master_df = master_df_dict.get(col)[1]
        actual_values = []
        heirarchy_values = []
        for column in values:
            if column in master_df.columns:
                actual_values.extend(
                    master_df.select(column).unique().drop_nulls().to_series().to_list()
                )
        heirarchy_values.extend(master_df.select(pl.col(stat_col).unique()).to_series().to_list())

        if not keep:
            actual_values = list(set(actual_values).symmetric_difference(set(heirarchy_values)))
        else:
            actual_values = list(set(actual_values).union(set(heirarchy_values)))
        result_dict[col] = [values, actual_values]

    return result_dict


cols_mapping = {
    constants.item_members: "str",
    constants.location_members: "str",
    constants.channel_members: "str",
    constants.region_members: "str",
    constants.account_members: "str",
    constants.pnl_members: "str",
    constants.dem_dom_members: "str",
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    SelectedIterations,
    StatLevels,
    ItemMasterData,
    ChannelMasterData,
    PnLMasterData,
    LocationMasterData,
    DemandDomainMasterData,
    RegionMasterData,
    AccountMasterData,
    StatPnL,
    StatItem,
    StatRegion,
    StatLocation,
    StatDemandDomain,
    StatAccount,
    StatChannel,
    df_keys,
):
    plugin_name = "DP088DeleteStatMembers"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    ItemMembers = pd.DataFrame(columns=constants.cols_req_for_item_members)
    LocationMembers = pd.DataFrame(columns=constants.cols_req_for_location_members)
    ChannelMembers = pd.DataFrame(columns=constants.cols_req_for_channel_members)
    RegionMembers = pd.DataFrame(columns=constants.cols_req_for_region_members)
    AccountMembers = pd.DataFrame(columns=constants.cols_req_for_account_members)
    PnLMembers = pd.DataFrame(columns=constants.cols_req_for_pnl_members)
    DemandDomainMembers = pd.DataFrame(columns=constants.cols_req_for_dem_dom_members)
    try:
        if StatLevels.empty:
            logger.warning("Input StatLevels is empty, pls check ...")
            logger.warning("Will return empty dataframe ...")
            return (
                ItemMembers,
                LocationMembers,
                ChannelMembers,
                RegionMembers,
                AccountMembers,
                PnLMembers,
                DemandDomainMembers,
            )

        # get iterations to process
        iterations_to_process = get_list_of_grains_from_string(SelectedIterations)

        # get all the stat level columns
        stat_level_cols = [col for col in StatLevels.columns if not is_dimension(col)]

        # get all version values
        version_df = pd.DataFrame(
            {constants.version_col: StatLevels[constants.version_col].unique()}
        )

        # convert pandas dataframes to polar dataframes
        StatLevels = pl.from_pandas(StatLevels)
        ItemMasterData = pl.from_pandas(ItemMasterData)
        ChannelMasterData = pl.from_pandas(ChannelMasterData)
        LocationMasterData = pl.from_pandas(LocationMasterData)
        PnLMasterData = pl.from_pandas(PnLMasterData)
        DemandDomainMasterData = pl.from_pandas(DemandDomainMasterData)
        AccountMasterData = pl.from_pandas(AccountMasterData)
        RegionMasterData = pl.from_pandas(RegionMasterData)
        StatItem = pl.from_pandas(StatItem)
        StatRegion = pl.from_pandas(StatRegion)
        StatChannel = pl.from_pandas(StatChannel)
        StatAccount = pl.from_pandas(StatAccount)
        StatDemandDomain = pl.from_pandas(StatDemandDomain)
        StatPnL = pl.from_pandas(StatPnL)
        StatLocation = pl.from_pandas(StatLocation)

        SelectedLevels = StatLevels.filter(
            pl.col(constants.forecast_iteration).is_in(iterations_to_process)
        )
        NonSelectedLevels = StatLevels.filter(
            ~(pl.col(constants.forecast_iteration).is_in(iterations_to_process))
        )

        master_dataframes_dict = {
            constants.item_level: [
                constants.planning_item_col,
                ItemMasterData,
                ItemMembers,
                constants.item_members,
                StatItem,
                constants.stat_item_col,
            ],
            constants.account_level: [
                constants.planning_account_col,
                AccountMasterData,
                AccountMembers,
                constants.account_members,
                StatAccount,
                constants.stat_account_col,
            ],
            constants.channel_level: [
                constants.planning_channel_col,
                ChannelMasterData,
                ChannelMembers,
                constants.channel_members,
                StatChannel,
                constants.stat_channel_col,
            ],
            constants.location_level: [
                constants.planning_location_col,
                LocationMasterData,
                LocationMembers,
                constants.location_members,
                StatLocation,
                constants.stat_location_col,
            ],
            constants.region_level: [
                constants.planning_region_col,
                RegionMasterData,
                RegionMembers,
                constants.region_members,
                StatRegion,
                constants.stat_region_col,
            ],
            constants.pnl_level: [
                constants.planning_pnl_col,
                PnLMasterData,
                PnLMembers,
                constants.pnl_members,
                StatPnL,
                constants.stat_pnl_col,
            ],
            constants.dem_dom_level: [
                constants.planning_dem_dom_col,
                DemandDomainMasterData,
                DemandDomainMembers,
                constants.dem_dom_members,
                StatDemandDomain,
                constants.stat_dem_dom_col,
            ],
        }

        all_non_delete_levels = get_formatted_uniques(
            NonSelectedLevels, stat_level_cols, master_dataframes_dict, keep=True
        )
        all_delete_levels = get_formatted_uniques(
            SelectedLevels, stat_level_cols, master_dataframes_dict, keep=False
        )

        stat_values = set(v for values in all_non_delete_levels.values() for v in values[0])
        filtered_delete_levels = {
            k: [v for v in values[0] if v not in stat_values]
            for k, values in all_delete_levels.items()
            if any(v not in stat_values for v in values[0])  # keep only non-empty lists
        }

        if not len(filtered_delete_levels) > 0:
            logger.warning(
                "No levels to delete after filtering from all iterations and scenarios ..."
            )
            logger.warning("Will return empty dataframe ...")
            return (
                ItemMembers,
                LocationMembers,
                ChannelMembers,
                RegionMembers,
                AccountMembers,
                PnLMembers,
                DemandDomainMembers,
            )

        for level, column_names in filtered_delete_levels.items():
            master_df = master_dataframes_dict.get(level)[1]
            if master_df is None:
                continue

            # Flatten values from multiple columns into a single list
            for col in column_names:
                if col not in master_df.columns:
                    continue
                filtered_df = master_df.select(
                    [
                        pl.col(master_dataframes_dict.get(level)[0]),
                        pl.col(col).alias(master_dataframes_dict.get(level)[3]),
                    ],
                )

                filtered_df = filtered_df.to_pandas()

                filtered_df = create_cartesian_product(filtered_df, version_df)
                retain_stat_values = set(all_non_delete_levels.get(level)[1])
                filtered_df = filtered_df[
                    ~filtered_df[master_dataframes_dict.get(level)[3]].isin(retain_stat_values)
                ]
                master_dataframes_dict.get(level)[2] = pd.concat(
                    [master_dataframes_dict.get(level)[2], filtered_df],
                    ignore_index=True,
                )

        ItemMembers = master_dataframes_dict["Item Level"][2]
        LocationMembers = master_dataframes_dict["Location Level"][2]
        ChannelMembers = master_dataframes_dict["Channel Level"][2]
        RegionMembers = master_dataframes_dict["Region Level"][2]
        AccountMembers = master_dataframes_dict["Account Level"][2]
        PnLMembers = master_dataframes_dict["PnL Level"][2]
        DemandDomainMembers = master_dataframes_dict["Demand Domain Level"][2]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return (
        ItemMembers,
        LocationMembers,
        ChannelMembers,
        RegionMembers,
        AccountMembers,
        PnLMembers,
        DemandDomainMembers,
    )

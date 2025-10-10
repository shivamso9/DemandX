"""
Version : 0.0.0
Maintained by : dpref@o9solutions.com
"""

import logging

import pandas as pd

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")


def get_planning_to_stat_region_mapping(
    RegionLevel: pd.DataFrame,
    RegionMasterData: pd.DataFrame,
):
    # collect region level
    region_level = RegionLevel[o9Constants.REGION_LEVEL].iloc[0]
    logger.info(f"{o9Constants.REGION_LEVEL} : {region_level}")

    # Eg. required_col = "Region.[Region L1]"
    required_col = "Region.[" + region_level + "]"
    logger.debug(f"required_col : {required_col}")

    if required_col == o9Constants.PLANNING_REGION:
        # collect pl item and level members
        RegionMasterData = RegionMasterData[[o9Constants.PLANNING_REGION]].drop_duplicates()

        RegionMasterData[o9Constants.STAT_REGION] = RegionMasterData[o9Constants.PLANNING_REGION]
    else:
        # collect pl item and level members
        RegionMasterData = RegionMasterData[
            [o9Constants.PLANNING_REGION, required_col]
        ].drop_duplicates()

        # rename to stat region
        RegionMasterData.rename(columns={required_col: o9Constants.STAT_REGION}, inplace=True)
    return RegionMasterData


def get_holidays_at_stat_level(
    HolidayData,
    StatCustomerGroupMapping,
    region_col,
    stat_customergroup_col,
    time_day_col,
    holiday_type_col,
    TimeDimension,
    relevant_time_name,
):
    # Holiday Preprocessing
    logger.debug("Joining HolidayData with StatCustomerGroupMapping ...")

    # Join with customer group mapping to get holidays at stat customer group level
    HolidayData = HolidayData.merge(StatCustomerGroupMapping, on=region_col)

    if len(HolidayData) == 0:
        return pd.DataFrame()

    # get holidays at stat customer group, day level
    req_cols = [
        stat_customergroup_col,
        time_day_col,
        holiday_type_col,
    ]
    HolidayData = HolidayData[req_cols].drop_duplicates()

    logger.debug("Joining HolidayData with time_mapping ...")

    # Join with time mapping based on actuals time granularity
    HolidayData = HolidayData.merge(
        TimeDimension[[time_day_col, relevant_time_name]],
        on=time_day_col,
    )

    # group on customer group and timelevel entities to get holidays at relevant time level
    HolidayData[holiday_type_col] = HolidayData.groupby(
        [stat_customergroup_col, relevant_time_name]
    )[holiday_type_col].transform(lambda x: ",".join(x))

    # select relevant columns
    req_cols = [
        stat_customergroup_col,
        relevant_time_name,
        holiday_type_col,
    ]
    HolidayData = HolidayData[req_cols].drop_duplicates()

    return HolidayData


def get_holidays_at_stat_region_level(
    HolidayData,
    StatRegionMapping,
    planning_region_col,
    stat_region_col,
    time_day_col,
    holiday_type_col,
    TimeDimension,
    relevant_time_name,
):
    # Holiday Preprocessing
    logger.debug("Joining HolidayData with StatRegionMapping ...")

    # Join with customer group mapping to get holidays at stat region level
    HolidayData = HolidayData.merge(StatRegionMapping, on=planning_region_col)

    if len(HolidayData) == 0:
        return pd.DataFrame()

    # get holidays at stat customer group, day level
    req_cols = [
        stat_region_col,
        time_day_col,
        holiday_type_col,
    ]
    HolidayData = HolidayData[req_cols].drop_duplicates()

    logger.debug("Joining HolidayData with time_mapping ...")

    # Join with time mapping based on actuals time granularity
    HolidayData = HolidayData.merge(
        TimeDimension[[time_day_col, relevant_time_name]],
        on=time_day_col,
    )

    # group on customer group and timelevel entities to get holidays at relevant time level
    HolidayData[holiday_type_col] = HolidayData.groupby([stat_region_col, relevant_time_name])[
        holiday_type_col
    ].transform(lambda x: ",".join(x))

    # select relevant columns
    req_cols = [
        stat_region_col,
        relevant_time_name,
        holiday_type_col,
    ]
    HolidayData = HolidayData[req_cols].drop_duplicates()

    return HolidayData

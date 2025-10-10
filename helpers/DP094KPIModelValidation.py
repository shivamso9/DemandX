import logging

import numpy as np
import pandas as pd

# from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    # Grains
    VERSION = "Version.[Version Name]"
    PLANNING_ITEM = "Item.[Planning Item]"
    PLANNING_ACCOUNT = "Account.[Planning Account]"
    PLANNING_LOCATION = "Location.[Planning Location]"
    PLANNING_CHANNEL = "Channel.[Planning Channel]"
    PLANNING_REGION = "Region.[Planning Region]"
    PLANNING_PNL = "PnL.[Planning PnL]"
    PLANNING_DEMAND_DOMAIN = "Demand Domain.[Planning Demand Domain]"
    ALL_PLANNING_ITEM = "All Planning Item"
    ALL_PLANNING_ACCOUNT = "All Planning Account"
    ALL_PLANNING_LOCATION = "All Planning Location"
    ALL_PLANNING_CHANNEL = "All Planning Channel"
    ALL_PLANNING_REGION = "All Planning Region"
    ALL_PLANNING_PNL = "All Planning PnL"
    ALL_PLANNING_DEMAND_DOMAIN = "All Planning Demand Domain"
    PLANNING_ITEM_KEY = "Item.[PlanningItemKey]"
    PLANNING_ACCOUNT_KEY = "Account.[PlanningAccountKey]"
    PLANNING_LOCATION_KEY = "Location.[PlanningLocationKey]"
    PLANNING_CHANNEL_KEY = "Channel.[PlanningChannelKey]"
    PLANNING_REGION_KEY = "Region.[PlanningRegionKey]"
    PLANNING_PNL_KEY = "PnL.[PlanningPnLKey]"
    PLANNING_DEMAND_DOMAIN_KEY = "Demand Domain.[PlanningDemandDomainKey]"
    LAG = "Lag.[Lag]"

    # Time
    PARTIAL_WEEK = "Time.[Partial Week]"
    PARTIAL_WEEK_KEY = "Time.[PartialWeekKey]"
    PLANNING_CYCLE = "Planning Cycle.[Planning Cycle]"
    PLANNING_CYCLE_KEY = "Planning Cycle.[PlanningCycleKey]"
    PLANNING_CYCLE_DATE = "Planning Cycle.[Planning Cycle Date]"
    PLANNING_CYCLE_DATE_KEY = "Planning Cycle.[Planning Cycle Date Key]"

    # Data Object
    DATA_OBJECT = "Data Object.[Data Object]"
    DATA_OBJECT_TYPE = "Data Object.[Data Object Type]"
    RULE = "DM Rule.[Rule]"

    # Data Object Measures
    DO_ITEM_LEVEL = "Data Object Item Level"
    DO_ACCOUNT_LEVEL = "Data Object Account Level"
    DO_CHANNEL_LEVEL = "Data Object Channel Level"
    DO_REGION_LEVEL = "Data Object Region Level"
    DO_LOCATION_LEVEL = "Data Object Location Level"
    DO_PNL_LEVEL = "Data Object PnL Level"
    DO_DEMAND_DOMAIN_LEVEL = "Data Object Demand Domain Level"
    DO_TIME_LEVEL = "Data Object Time Level"
    DO_TIME_AGGREGATION = "DP KPI Time Aggregation"
    DO_HORIZON = "DP KPI Rolling Horizon"
    DO_OFFSET = "DP KPI Rolling Offset"
    DO_LAG = "DP KPI Lag"
    FCST_MEASURE_OUTPUT = "DP KPI Fcst Measure"
    FCST_MEASURE_INPUT = "DP KPI Fcst Measure Input"

    # Assortment
    ASSORTMENT_SYSTEM = "Assortment System"
    ASSORTMENT_FINAL = "Assortment Final"


def findAssortedMember(
    group: pd.DataFrame,
    merge_info: list,
    Assortment: pd.DataFrame,
):

    try:
        if Assortment.empty:
            logger.warning("Input 'Assortment' DataFrame is empty.")
            return

        Assortment = Assortment.loc[Assortment[Constants.ASSORTMENT_FINAL] == 1]

        if Assortment.empty:
            logger.warning("No members with 'Assortment Final' is true found.")
            return

        validMembers = Assortment
        validMembers.drop(
            columns=[Constants.ASSORTMENT_SYSTEM, Constants.ASSORTMENT_FINAL], inplace=True
        )

        groupby_col = []
        for master, planning_col, key_col, dim_col, all_col in merge_info:
            grain = group[dim_col].iloc[0]
            if pd.isna(grain):
                validMembers[planning_col] = all_col
            else:
                grain = grain.replace("[", "", 1).replace("]", "", 1)
            if grain != planning_col and validMembers[planning_col].iloc[0] != all_col:
                validMembers = validMembers.merge(
                    master[[planning_col, key_col, grain]], how="left", on=[planning_col]
                )
            groupby_col.append(grain)

        for i, planning_col2, key_col2, dim_col2, all_col2 in merge_info:
            grain2 = group[dim_col2].iloc[0]
            if pd.isna(grain2):
                validMembers[planning_col2] = all_col2
            else:
                grain2 = grain2.replace("[", "", 1).replace("]", "", 1)
            if grain2 != planning_col2 and validMembers[planning_col2].iloc[0] != all_col2:
                min_rows = validMembers.loc[validMembers.groupby(groupby_col)[key_col2].idxmin()][
                    groupby_col + [key_col2]
                ]
                validMembers = validMembers.merge(
                    min_rows, on=groupby_col + [key_col2], how="inner"
                )
                validMembers.drop(columns=[key_col2], inplace=True)

        validMembers.drop_duplicates(inplace=True)

    except Exception as e:
        logger.warning(e)
    return validMembers


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    DataObject: pd.DataFrame,
    FcstOutput: pd.DataFrame,
    AccountMaster: pd.DataFrame,
    ChannelMaster: pd.DataFrame,
    DemandDomainMaster: pd.DataFrame,
    ItemMaster: pd.DataFrame,
    LocationMaster: pd.DataFrame,
    RegionMaster: pd.DataFrame,
    PnLMaster: pd.DataFrame,
    Assortment: pd.DataFrame,
    df_keys: dict = {},
):
    plugin_name = "DP094KPIModelValidation"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    merge_info = [
        (
            ItemMaster,
            Constants.PLANNING_ITEM,
            Constants.PLANNING_ITEM_KEY,
            Constants.DO_ITEM_LEVEL,
            Constants.ALL_PLANNING_ITEM,
        ),
        (
            AccountMaster,
            Constants.PLANNING_ACCOUNT,
            Constants.PLANNING_ACCOUNT_KEY,
            Constants.DO_ACCOUNT_LEVEL,
            Constants.ALL_PLANNING_ACCOUNT,
        ),
        (
            LocationMaster,
            Constants.PLANNING_LOCATION,
            Constants.PLANNING_LOCATION_KEY,
            Constants.DO_LOCATION_LEVEL,
            Constants.ALL_PLANNING_LOCATION,
        ),
        (
            ChannelMaster,
            Constants.PLANNING_CHANNEL,
            Constants.PLANNING_CHANNEL_KEY,
            Constants.DO_CHANNEL_LEVEL,
            Constants.ALL_PLANNING_CHANNEL,
        ),
        (
            RegionMaster,
            Constants.PLANNING_REGION,
            Constants.PLANNING_REGION_KEY,
            Constants.DO_REGION_LEVEL,
            Constants.ALL_PLANNING_REGION,
        ),
        (
            DemandDomainMaster,
            Constants.PLANNING_DEMAND_DOMAIN,
            Constants.PLANNING_DEMAND_DOMAIN_KEY,
            Constants.DO_DEMAND_DOMAIN_LEVEL,
            Constants.ALL_PLANNING_DEMAND_DOMAIN,
        ),
        (
            PnLMaster,
            Constants.PLANNING_PNL,
            Constants.PLANNING_PNL_KEY,
            Constants.DO_PNL_LEVEL,
            Constants.ALL_PLANNING_PNL,
        ),
    ]
    D7cols = [
        Constants.PLANNING_ITEM,
        Constants.PLANNING_ACCOUNT,
        Constants.PLANNING_CHANNEL,
        Constants.PLANNING_REGION,
        Constants.PLANNING_LOCATION,
        Constants.PLANNING_PNL,
        Constants.PLANNING_DEMAND_DOMAIN,
    ]

    try:
        if FcstOutput.empty:
            logger.info("Input 'FcstOutput' is empty. Returning an empty DataFrame.")
            return pd.DataFrame(columns=FcstOutput.columns)
        FcstColumns = FcstOutput.columns
        Fcstgrains = [col for col in FcstOutput.columns if "." in col]
        Fcstdf = FcstOutput.merge(
            Assortment,
            how="left",
            on=[
                col
                for col in Assortment.columns
                if col != Constants.ASSORTMENT_FINAL and col != Constants.ASSORTMENT_SYSTEM
            ],
        )
        Fcstdf.drop(columns=[Constants.ASSORTMENT_SYSTEM], inplace=True)
        Measures = FcstColumns.difference(Fcstgrains)
        Result_df = pd.DataFrame(columns=FcstColumns)

        measures_cols = Measures.tolist()
        cols_to_convert = [col for col in measures_cols if col in Result_df.columns]

        NotAssorted = Fcstdf.loc[Fcstdf[Constants.ASSORTMENT_FINAL] != 1]
        if NotAssorted.empty:
            logger.info("No assortments were changed")
            Result_df[cols_to_convert] = Result_df[cols_to_convert].astype(float)
            Result_df[Constants.LAG] = Result_df[Constants.LAG].astype(str)
            return Result_df

        for _, grouped in NotAssorted.groupby(Constants.DATA_OBJECT):
            group = DataObject[
                DataObject[Constants.DATA_OBJECT] == grouped[Constants.DATA_OBJECT].iloc[0]
            ]
            FcstdfDO = Fcstdf[Fcstdf[Constants.DATA_OBJECT] == group[Constants.DATA_OBJECT].iloc[0]]
            cols_to_null = Fcstdf.columns.difference(Fcstgrains)
            DOValidColumns = [Constants.VERSION]
            for master, planning_col, planning_key, DO_col, all_col in merge_info:
                grain = group[DO_col].iloc[0].replace("[", "", 1).replace("]", "", 1)
                DOValidColumns.append(grain)
                if planning_col not in FcstdfDO.columns:
                    continue
                if grain != planning_col:
                    FcstdfDO = FcstdfDO.merge(
                        master[[planning_col, grain]], how="left", on=[planning_col]
                    )
            FcstdfDO.drop_duplicates(inplace=True)
            newAssorted = findAssortedMember(group, merge_info, Assortment)
            if newAssorted.empty:
                FcstdfDO.loc[FcstdfDO[Constants.ASSORTMENT_FINAL] != 1, cols_to_null] = np.nan
                FcstdfDO = FcstdfDO[FcstColumns]
                Result_df = pd.concat([Result_df, FcstdfDO], ignore_index=True)
                continue
            FcstdfDO = FcstdfDO.merge(
                newAssorted,
                how="left",
                on=[col for col in DOValidColumns if col in FcstdfDO.columns],
            )
            NoAssortments = FcstdfDO[
                (FcstdfDO[Constants.ASSORTMENT_FINAL] != 1)
                & (
                    FcstdfDO[[col + "_y" for col in D7cols if col + "_y" in FcstdfDO.columns]]
                    .isnull()
                    .any(axis=1)
                )
            ]
            ValidAssortments = FcstdfDO[
                (FcstdfDO[Constants.ASSORTMENT_FINAL] != 1)
                & (
                    FcstdfDO[[col + "_y" for col in D7cols if col + "_y" in FcstdfDO.columns]]
                    .notnull()
                    .all(axis=1)
                )
            ]

            if not NoAssortments.empty:
                NoAssortments.loc[:, cols_to_null] = np.nan
                for cols in D7cols:
                    if cols + "_x" in NoAssortments.columns:
                        NoAssortments.rename(columns={cols + "_x": cols}, inplace=True)
                NoAssortments = NoAssortments[FcstColumns]
                Result_df = pd.concat([Result_df, NoAssortments], ignore_index=True)
            if not ValidAssortments.empty:
                Rectifydf = ValidAssortments.copy()
                for planning_col in D7cols:
                    if planning_col + "_x" in ValidAssortments.columns:
                        ValidAssortments[planning_col + "_x"] = ValidAssortments[
                            planning_col + "_y"
                        ]
                ValidAssortments.loc[:, cols_to_null] = Rectifydf.loc[:, cols_to_null]
                Rectifydf.loc[Rectifydf[Constants.ASSORTMENT_FINAL] != 1, cols_to_null] = np.nan
                for cols in D7cols:
                    if cols + "_x" in ValidAssortments.columns:
                        ValidAssortments.rename(columns={cols + "_x": cols}, inplace=True)
                    if cols + "_x" in Rectifydf.columns:
                        Rectifydf.rename(columns={cols + "_x": cols}, inplace=True)
                ValidAssortments = ValidAssortments[FcstColumns]
                Rectifydf = Rectifydf[FcstColumns]
                Result_df = pd.concat([Result_df, Rectifydf], ignore_index=True)
                Result_df = pd.concat([Result_df, ValidAssortments], ignore_index=True)
        Result_df[cols_to_convert] = Result_df[cols_to_convert].astype(float)
        Result_df[Constants.LAG] = Result_df[Constants.LAG].astype(str)
    except Exception as e:
        logger.error(f"Error during validation: {e}")
    return Result_df

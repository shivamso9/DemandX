import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {}


class Constants:
    """Holds constant values used in KPI forecast archival."""

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
    PLANNING_CYCLE_DATE_KEY = "Planning Cycle.[PlanningCycleDateKey]"

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
    """Find assorted member records based on input criteria."""

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


def get_each_DO(
    group: pd.DataFrame,
    CurrentTimePeriod: pd.DataFrame,
    PlanningCycleDate: pd.DataFrame,
    TimeMaster: pd.DataFrame,
    merge_info: list,
    Assortment: pd.DataFrame,
    **dfs: pd.DataFrame,
):
    """Computation for each Data Object"""
    try:
        # if group is empty return empty
        D7cols = [
            Constants.VERSION,
            Constants.PLANNING_ITEM,
            Constants.PLANNING_ACCOUNT,
            Constants.PLANNING_LOCATION,
            Constants.PLANNING_CHANNEL,
            Constants.PLANNING_REGION,
            Constants.PLANNING_DEMAND_DOMAIN,
            Constants.PLANNING_PNL,
            Constants.PARTIAL_WEEK,
        ]
        InputMeasures = group[Constants.FCST_MEASURE_INPUT].tolist()
        OutputMeasures = group[Constants.FCST_MEASURE_OUTPUT].tolist()
        mappingIO = dict(
            (row[Constants.FCST_MEASURE_INPUT], row[Constants.FCST_MEASURE_OUTPUT])
            for _, row in group.iterrows()
        )

        DOTimeLevel = group[Constants.DO_TIME_LEVEL].iloc[0]
        TimeLevelKey = "Time.[" + DOTimeLevel + "Key]"
        TimeLevelKey = TimeLevelKey.replace(" ", "")
        CurrentTimeLevelKey = CurrentTimePeriod[TimeLevelKey].iloc[0]
        PCDate = PlanningCycleDate[
            PlanningCycleDate[Constants.PLANNING_CYCLE_DATE_KEY] == CurrentTimeLevelKey
        ][Constants.PLANNING_CYCLE_DATE].iloc[0]
        PCKey = CurrentTimeLevelKey
        DOValidColumns = [Constants.VERSION]

        DO = group[Constants.DATA_OBJECT].iloc[0]
        DOType = group[Constants.DATA_OBJECT_TYPE].iloc[0]
        if "Lag" in DOType:
            if pd.isna(group[Constants.DO_LAG].iloc[0]):
                logger.warning(f"Data Object {DO} has no Lag defined, skipping.")
                return pd.DataFrame()
            else:
                Lags = group[Constants.DO_LAG].iloc[0].split(",")
                Lags = [int(x) for x in Lags]
        if "Rolling Model" in DOType:
            if (
                pd.isna(group[Constants.DO_HORIZON].iloc[0])
                or pd.isna(group[Constants.DO_OFFSET].iloc[0])
                or pd.isna(group[Constants.DO_TIME_AGGREGATION].iloc[0])
            ):
                logger.warning(
                    f"Data Object {DO} has no Horizon or Offset or Time Aggregation defined, skipping."
                )
                return pd.DataFrame()
            else:
                horizon = group[Constants.DO_HORIZON].iloc[0]
                offset = group[Constants.DO_OFFSET].iloc[0]
                TimeAggregation = group[Constants.DO_TIME_AGGREGATION].iloc[0]

        if DOTimeLevel != "Partial Week":
            req_cols = [
                Constants.PARTIAL_WEEK,
                Constants.PARTIAL_WEEK_KEY,
                "Time.[" + DOTimeLevel + "]",
                TimeLevelKey,
            ]
            exclude_cols = InputMeasures + [Constants.PARTIAL_WEEK, Constants.PARTIAL_WEEK_KEY]
            drop_cols = [Constants.PARTIAL_WEEK_KEY, "Time.[" + DOTimeLevel + "]", TimeLevelKey]
        else:
            req_cols = [Constants.PARTIAL_WEEK, Constants.PARTIAL_WEEK_KEY]
            exclude_cols = InputMeasures + [Constants.PARTIAL_WEEK_KEY]
            drop_cols = [Constants.PARTIAL_WEEK_KEY]

        result_df = pd.DataFrame()
        merged_dfs = None
        # Aggregate Fcst to the DO defined level
        for name, df in dfs.items():
            for measure in InputMeasures:
                if measure in df.columns:
                    cols_to_use = [col for col in D7cols if col in df.columns]
                    if merged_dfs is None:
                        # First time, create merged_dfs with df subset
                        merged_dfs = df.copy()
                    else:
                        # Merge subsequent dfs on shared columns
                        if measure not in merged_dfs.columns:
                            merged_dfs = merged_dfs.merge(df, how="outer", on=cols_to_use)

        if merged_dfs.empty:
            logger.warning(f"No input data found for measures {InputMeasures} in Data Object {DO}")
            return

        merged_dfs = merged_dfs[
            [col for col in D7cols if col in merged_dfs.columns] + InputMeasures
        ]
        merged_dfs = merged_dfs.merge(TimeMaster[req_cols], how="left", on=[Constants.PARTIAL_WEEK])
        merged_dfs = merged_dfs[merged_dfs[Constants.PARTIAL_WEEK_KEY] >= PCKey]

        # Fill missing grain with All Planning Member
        for master, planning_col, planning_key, DO_col, all_col in merge_info:
            grain = group[DO_col].iloc[0].replace("[", "", 1).replace("]", "", 1)
            DOValidColumns.append(grain)
            if planning_col not in merged_dfs.columns:
                # merged_dfs[planning_col] = all_col
                continue
            if grain != planning_col:
                merged_dfs = merged_dfs.merge(
                    master[[planning_col, grain]], how="left", on=[planning_col]
                )
                merged_dfs.drop(columns=[planning_col], inplace=True)
        result_df = merged_dfs.copy()

        result_df = (
            result_df.groupby(
                [col for col in result_df.columns if col not in InputMeasures],
            )[InputMeasures]
            .sum()
            .reset_index()
        )

        # Pick only first Partial Week
        group_cols = [col for col in result_df.columns if col not in exclude_cols]

        # first PartialWeekKey rows per group
        idx = result_df.groupby(group_cols)[Constants.PARTIAL_WEEK_KEY].idxmin()
        filtered_df = result_df.loc[idx].copy()

        # Aggregate Fcst and Actual per group
        FcstSum = result_df.groupby(group_cols, as_index=False)[InputMeasures].sum()

        # Merge sum into the filtered first-week rows
        # Merge with suffix for sum columns
        filtered_df = filtered_df.merge(FcstSum, on=group_cols, suffixes=("", "_sum"))

        # Overwrite each InputMeasure with its corresponding "_sum" version
        for col in InputMeasures:
            filtered_df[col] = filtered_df[col + "_sum"]

        # Drop the "_sum" columns
        filtered_df.drop(columns=[f"{col}_sum" for col in InputMeasures], inplace=True)

        assortedMembers = findAssortedMember(
            group=group,
            merge_info=merge_info,
            Assortment=Assortment,
        )
        if assortedMembers.empty:
            logger.warning(f"No assorted members found for Data Object {DO}.")
            return

        filtered_df = filtered_df.merge(
            assortedMembers,
            how="left",
            on=[col for col in DOValidColumns if col in filtered_df.columns],
        )
        filtered_df.drop(
            columns=[
                col for col in DOValidColumns if col != Constants.VERSION and "Planning" not in col
            ],
            inplace=True,
        )

        # Add Data Object column
        filtered_df[Constants.DATA_OBJECT] = DO
        # Populate Lag and Planning Cycle
        filtered_df[Constants.PLANNING_CYCLE_DATE] = PCDate

        min_date = PCKey
        TimeMaster = TimeMaster.sort_values(by=TimeLevelKey)
        TimeMaster_filtered = TimeMaster[TimeMaster[TimeLevelKey] >= min_date]
        TimeLag = pd.DataFrame()
        TimeLag[TimeLevelKey] = TimeMaster_filtered[TimeLevelKey].unique()
        TimeLag[Constants.LAG] = range(len(TimeLag))
        filtered_df = filtered_df.merge(TimeLag, on=[TimeLevelKey], how="left")
        filtered_df.rename(columns=mappingIO, inplace=True)
        filtered_df.drop(columns=drop_cols, inplace=True)

        if "Lag" in DOType:
            # Filter For Lags
            filtered_df = filtered_df[filtered_df[Constants.LAG].isin(Lags)]

        elif "Rolling Model" in DOType:
            horizon_fcst = filtered_df[
                (filtered_df[Constants.LAG] >= offset)
                & (filtered_df[Constants.LAG] <= offset + horizon - 1)
            ]
            if TimeAggregation == "No":
                horizon_fcst[Constants.LAG] = 0
                filtered_df = horizon_fcst
            else:
                excludeCols = [Constants.PARTIAL_WEEK, Constants.LAG] + OutputMeasures
                groupCols = [col for col in filtered_df.columns if col not in excludeCols]
                horizon_Sum = horizon_fcst.groupby(groupCols)[OutputMeasures].sum().reset_index()
                horizon_Sum[Constants.LAG] = 0
                horizon_Sum[Constants.PARTIAL_WEEK] = TimeMaster[
                    TimeMaster[Constants.PARTIAL_WEEK_KEY] >= PCKey
                ][Constants.PARTIAL_WEEK].iloc[0]
                filtered_df = horizon_Sum

        # Arrange the columns
        cols_to_move = [Constants.PARTIAL_WEEK] + OutputMeasures
        new_cols = [col for col in filtered_df.columns if col not in cols_to_move] + cols_to_move
        filtered_df = filtered_df[new_cols]
        filtered_df[Constants.LAG] = filtered_df[Constants.LAG].astype(str)

        logger.info("get_each_DO executed successfully for Data Object:")

    except Exception as e:
        logger.warning(e)
        return
    return filtered_df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    DataObject: pd.DataFrame,
    Rule: pd.DataFrame,
    CurrentTimePeriod: pd.DataFrame,
    TimeMaster: pd.DataFrame,
    PlanningCycleDate: pd.DataFrame,
    AccountMaster: pd.DataFrame,
    ChannelMaster: pd.DataFrame,
    DemandDomainMaster: pd.DataFrame,
    ItemMaster: pd.DataFrame,
    LocationMaster: pd.DataFrame,
    RegionMaster: pd.DataFrame,
    PnLMaster: pd.DataFrame,
    Assortment: pd.DataFrame,
    df_keys: dict = {},
    **dfs: pd.DataFrame,
):
    """Main entry point for the DP092 archival process."""
    plugin_name = "DP092FlexibleKPIFcstArchival"
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

    OutputDfs = pd.DataFrame(
        columns=[
            Constants.VERSION,
            Constants.PLANNING_ITEM,
            Constants.PLANNING_ACCOUNT,
            Constants.PLANNING_LOCATION,
            Constants.PLANNING_CHANNEL,
            Constants.PLANNING_REGION,
            Constants.PLANNING_DEMAND_DOMAIN,
            Constants.PLANNING_PNL,
            Constants.PARTIAL_WEEK,
            Constants.LAG,
            Constants.PLANNING_CYCLE_DATE,
            Constants.DATA_OBJECT,
        ]
    )

    try:
        if Rule.empty or DataObject.empty:
            logger.warning("Input 'Rule' or 'DataObject' DataFrame is empty. No rules to process.")
            return OutputDfs  # Return the pre-defined empty DataFrame

        DataObject = pd.merge(
            Rule, DataObject, how="inner", on=[Constants.VERSION, Constants.DATA_OBJECT]
        )

        if DataObject.empty:
            logger.warning(
                "No matching rules found in 'DataObject' after merging with 'Rule'. Exiting."
            )
            return OutputDfs

        OutputMeasures = DataObject[Constants.FCST_MEASURE_OUTPUT].unique()
        for col in OutputMeasures:
            OutputDfs[col] = np.nan
            col_mapping[col] = "float64"
        all_results = Parallel(n_jobs=1, verbose=1)(
            delayed(get_each_DO)(
                group=group,
                CurrentTimePeriod=CurrentTimePeriod,
                PlanningCycleDate=PlanningCycleDate,
                merge_info=merge_info,
                TimeMaster=TimeMaster,
                Assortment=Assortment,
                **dfs,
            )
            for name, group in DataObject.groupby(Constants.DATA_OBJECT)
        )
        OutputDfs = pd.concat([OutputDfs] + all_results, ignore_index=True)

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return OutputDfs

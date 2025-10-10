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
    PLANNING_CYCLE_DATE = "Planning Cycle.[Planning Cycle Date]"
    PLANNING_CYCLE_KEY = "Planning Cycle.[PlanningCycleKey]"
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
    ERROR_OUTPUT = "DP KPI Error Measure"
    ABS_ERROR_OUTPUT = "DP KPI Abs Error Measure"
    ACTUAL_MEASURE_INPUT = "DP KPI Actual Measure Input"
    ACTUAL_MEASURE_OUTPUT = "DP KPI Actual Measure Output"

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
        return
    return validMembers


def each_DO_computation(
    group: pd.DataFrame,
    CurrentPlanningCycleDate: pd.DataFrame,
    PlanningCycleDate: pd.DataFrame,
    TimeMaster: pd.DataFrame,
    merge_info: list,
    Assortment: pd.DataFrame,
    PlanningCyclePeriod: pd.DataFrame,
    **dfs: pd.DataFrame,
):

    try:
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
        ActualMeasureInput = group[Constants.ACTUAL_MEASURE_INPUT].unique().tolist()
        FcstMeasure = group[Constants.FCST_MEASURE_OUTPUT].unique().tolist()
        ErrorOutputMeasure = group[Constants.ERROR_OUTPUT].unique().tolist()
        AbsErrorOutputMeasure = group[Constants.ABS_ERROR_OUTPUT].unique().tolist()
        ActualOutputMeasure = group[Constants.ACTUAL_MEASURE_OUTPUT].unique().tolist()
        ActualIOmapping = dict(
            (row[Constants.ACTUAL_MEASURE_INPUT], row[Constants.ACTUAL_MEASURE_OUTPUT])
            for _, row in group.iterrows()
        )

        TimeAggregation = group[Constants.DO_TIME_AGGREGATION].iloc[0]
        DOTimeLevel = group[Constants.DO_TIME_LEVEL].iloc[0]
        TimeLevelKey = "Time.[" + DOTimeLevel + "Key]"
        TimeLevelKey = TimeLevelKey.replace(" ", "")
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
            exclude_cols = ActualMeasureInput + [Constants.PARTIAL_WEEK, Constants.PARTIAL_WEEK_KEY]
            drop_cols = [Constants.PARTIAL_WEEK_KEY, "Time.[" + DOTimeLevel + "]", TimeLevelKey]
        else:
            req_cols = [Constants.PARTIAL_WEEK, Constants.PARTIAL_WEEK_KEY]
            exclude_cols = ActualMeasureInput + [Constants.PARTIAL_WEEK_KEY]
            drop_cols = [Constants.PARTIAL_WEEK_KEY]

        # Aggregate Fcst to the DO defined level
        actual_dfs = None
        KPIFcst_df = pd.DataFrame(
            columns=D7cols + [Constants.PLANNING_CYCLE_DATE, Constants.LAG, Constants.DATA_OBJECT]
        )
        for name, df in dfs.items():
            for measure in ActualMeasureInput:
                if measure in df.columns:
                    cols_to_use = [col for col in D7cols if col in df.columns]
                    if actual_dfs is None:
                        actual_dfs = df.copy()
                    else:
                        if measure not in actual_dfs.columns:
                            actual_dfs = actual_dfs.merge(df, how="outer", on=cols_to_use)
        for name, df in dfs.items():
            for measure in FcstMeasure:
                if measure in df.columns:
                    cols_to_use = [col for col in D7cols if col in df.columns] + [
                        Constants.PLANNING_CYCLE_DATE,
                        Constants.LAG,
                        Constants.DATA_OBJECT,
                    ]
                    if KPIFcst_df is None:
                        KPIFcst_df = df.copy()
                    else:
                        if measure not in KPIFcst_df.columns:
                            KPIFcst_df = KPIFcst_df.merge(
                                df,
                                how="outer",
                                on=cols_to_use,
                            )
        if KPIFcst_df.empty or actual_dfs.empty:
            logger.warning(f"No input data found in Data Object {DO}")
            return
        actual_dfs = actual_dfs[
            [col for col in D7cols if col in actual_dfs.columns] + ActualMeasureInput
        ]
        KPIFcst_df = KPIFcst_df[
            [col for col in D7cols if col in KPIFcst_df.columns]
            + [Constants.PLANNING_CYCLE_DATE, Constants.LAG, Constants.DATA_OBJECT]
            + FcstMeasure
        ]
        KPIFcst_df = KPIFcst_df[KPIFcst_df[Constants.DATA_OBJECT] == DO]
        if KPIFcst_df.empty:
            logger.warning(
                f"No KPIFcst data found for the specific Data Object: {DO}. Skipping accuracy calculation for this DO."
            )
            return

        actual_dfs = actual_dfs.merge(TimeMaster[req_cols], how="left", on=[Constants.PARTIAL_WEEK])

        KPIFcst_df = KPIFcst_df.merge(
            PlanningCycleDate[[Constants.PLANNING_CYCLE_DATE, Constants.PLANNING_CYCLE_DATE_KEY]],
            how="left",
            on=[Constants.PLANNING_CYCLE_DATE],
        )
        KPIFcst_df.sort_values(
            by=[Constants.PLANNING_CYCLE_DATE_KEY], inplace=True, ascending=False
        )
        PCPeriod = int(PlanningCyclePeriod)
        PCPeriodDates = KPIFcst_df[Constants.PLANNING_CYCLE_DATE].drop_duplicates().head(PCPeriod)

        Result_df = pd.DataFrame()
        for PCDate in PCPeriodDates:
            PCDateKey = PlanningCycleDate[
                PlanningCycleDate[Constants.PLANNING_CYCLE_DATE] == PCDate
            ][Constants.PLANNING_CYCLE_DATE_KEY].iloc[0]
            KPIFcst_df2 = KPIFcst_df.drop(columns=[Constants.PLANNING_CYCLE_DATE_KEY])
            KPIFcst_df2 = KPIFcst_df2[KPIFcst_df2[Constants.PLANNING_CYCLE_DATE] == PCDate]
            actual_dfs2 = actual_dfs[
                pd.to_datetime(actual_dfs[Constants.PARTIAL_WEEK_KEY]) >= pd.to_datetime(PCDateKey)
            ]

            # Fill missing grain with All Planning Member
            for master, planning_col, planning_key, DO_col, all_col in merge_info:
                grain = group[DO_col].iloc[0].replace("[", "", 1).replace("]", "", 1)
                DOValidColumns.append(grain)
                if planning_col not in actual_dfs2.columns:
                    # actual_dfs[planning_col] = all_col
                    continue
                if grain != planning_col:
                    actual_dfs2 = actual_dfs2.merge(
                        master[[planning_col, grain]], how="left", on=[planning_col]
                    )
                    actual_dfs2.drop(columns=[planning_col], inplace=True)
            Actual_df = actual_dfs2.copy()
            Actual_df = (
                Actual_df.groupby(
                    [col for col in Actual_df.columns if col not in ActualMeasureInput],
                )[ActualMeasureInput]
                .sum()
                .reset_index()
            )

            # Pick only first Partial Week
            group_cols = [col for col in Actual_df.columns if col not in exclude_cols]

            # first PartialWeekKey rows per group
            idx = Actual_df.groupby(group_cols)[Constants.PARTIAL_WEEK_KEY].idxmin()
            filtered_df = Actual_df.loc[idx].copy()

            # Aggregate Fcst and Actual per group
            ActualSum = Actual_df.groupby(group_cols, as_index=False)[ActualMeasureInput].sum()

            # Merge sum into the filtered first-week rows
            filtered_df = filtered_df.merge(ActualSum, on=group_cols, suffixes=("", "_sum"))
            for col in ActualMeasureInput:
                filtered_df[col] = filtered_df[col + "_sum"]

            # Drop the "_sum" columns
            filtered_df.drop(columns=[f"{col}_sum" for col in ActualMeasureInput], inplace=True)

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
                    col
                    for col in DOValidColumns
                    if col != Constants.VERSION and "Planning" not in col
                ],
                inplace=True,
            )

            # Add Data Object column
            filtered_df[Constants.DATA_OBJECT] = DO
            # Populate Lag and Planning Cycle
            filtered_df[Constants.PLANNING_CYCLE_DATE] = PCDate

            min_date = PCDateKey
            TimeMaster = TimeMaster.sort_values(by=TimeLevelKey)
            TimeMaster_filtered = TimeMaster[TimeMaster[TimeLevelKey] >= min_date]
            TimeLag = pd.DataFrame()
            TimeLag[TimeLevelKey] = TimeMaster_filtered[TimeLevelKey].unique()
            TimeLag[Constants.LAG] = range(len(TimeLag))
            filtered_df = filtered_df.merge(TimeLag, on=[TimeLevelKey], how="left")
            filtered_df.rename(columns=ActualIOmapping, inplace=True)
            filtered_df.drop(columns=drop_cols, inplace=True)

            if "Lag" in DOType:
                # Filter For Lags
                missing_lags = [
                    lag for lag in Lags if lag not in filtered_df[Constants.LAG].unique()
                ]
                if missing_lags:
                    logger.warning(f"Lags {Lags} not found in Data Object {DO}, skipping.")
                    continue
                filtered_df = filtered_df[filtered_df[Constants.LAG].isin(Lags)]
            elif "Rolling Model" in DOType:
                if (offset + horizon - 1) not in filtered_df[Constants.LAG]:
                    logger.warning("Offset + horizon is out of bounds")
                    continue
                horizon_fcst = filtered_df[
                    (filtered_df[Constants.LAG] >= offset)
                    & (filtered_df[Constants.LAG] <= offset + horizon - 1)
                ]
                if TimeAggregation == "No":
                    horizon_fcst[Constants.LAG] = 0
                    filtered_df = horizon_fcst
                else:
                    excludeCols = [Constants.PARTIAL_WEEK, Constants.LAG] + ActualOutputMeasure
                    groupCols = [col for col in filtered_df.columns if col not in excludeCols]
                    horizon_Sum = (
                        horizon_fcst.groupby(groupCols)[ActualOutputMeasure].sum().reset_index()
                    )
                    horizon_Sum[Constants.LAG] = 0
                    horizon_Sum[Constants.PARTIAL_WEEK] = TimeMaster[
                        TimeMaster[Constants.PARTIAL_WEEK_KEY] >= PCDateKey
                    ][Constants.PARTIAL_WEEK].iloc[0]
                    filtered_df = horizon_Sum

            filtered_df[Constants.LAG] = filtered_df[Constants.LAG].astype(str)
            filtered_df = filtered_df.merge(
                KPIFcst_df2,
                how="outer",
                on=[col for col in filtered_df.columns if col not in ActualOutputMeasure],
            )
            filtered_df[ActualOutputMeasure] = filtered_df[ActualOutputMeasure].fillna(0)

            # Error Calculation
            for _, row in group.iterrows():
                fcst_col = row[Constants.FCST_MEASURE_OUTPUT]
                actual_col = row[Constants.ACTUAL_MEASURE_OUTPUT]
                error_col = row[Constants.ERROR_OUTPUT]
                abs_error_col = row[Constants.ABS_ERROR_OUTPUT]
                if error_col not in filtered_df.columns:
                    filtered_df[error_col] = filtered_df[fcst_col] - filtered_df[actual_col]
                    filtered_df[abs_error_col] = filtered_df[error_col].abs()

            filtered_df.drop(columns=FcstMeasure, inplace=True)
            Result_df = pd.concat([Result_df, filtered_df], ignore_index=True)

        # Arrange the columns
        cols_to_move = (
            [Constants.PARTIAL_WEEK]
            + ActualOutputMeasure
            + ErrorOutputMeasure
            + AbsErrorOutputMeasure
        )
        if not Result_df.empty:
            new_cols = [col for col in Result_df.columns if col not in cols_to_move] + cols_to_move
            Result_df = Result_df[new_cols]

        logger.info("get_each_DO executed successfully for Data Object:")

    except Exception as e:
        logger.warning(e)
        return
    return Result_df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    DataObject: pd.DataFrame,
    Rule: pd.DataFrame,
    CurrentPlanningCycleDate: pd.DataFrame,
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
    PlanningCyclePeriod: pd.DataFrame,
    df_keys: dict = {},
    **dfs: pd.DataFrame,
):
    plugin_name = "DP093FlexibleKPIAccuracyCalculation"
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

        DataObject = DataObject.dropna(subset=["DP KPI Actual Measure Input"])

        if DataObject.empty:
            logger.warning(
                "No rules with 'DP KPI Actual Measure Input' were found. Cannot calculate accuracy. Exiting."
            )
            return OutputDfs

        all_outputs = np.concatenate(
            [
                DataObject[Constants.ACTUAL_MEASURE_OUTPUT].unique(),
                DataObject[Constants.ERROR_OUTPUT].unique(),
                DataObject[Constants.ABS_ERROR_OUTPUT].unique(),
            ]
        )

        # Filter out None and NaN before getting unique values
        OutputMeasures = np.unique([x for x in all_outputs if x is not None and pd.notna(x)])

        for col in OutputMeasures:
            OutputDfs[col] = np.nan
            col_mapping[col] = "float64"
        if DataObject.empty:
            logger.warning("DataObject is empty, returning empty DataFrame.")
            return OutputDfs
        all_results = Parallel(n_jobs=1, verbose=1)(
            delayed(each_DO_computation)(
                group=group,
                CurrentPlanningCycleDate=CurrentPlanningCycleDate,
                PlanningCycleDate=PlanningCycleDate,
                merge_info=merge_info,
                TimeMaster=TimeMaster,
                Assortment=Assortment,
                PlanningCyclePeriod=PlanningCyclePeriod,
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

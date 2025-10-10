import logging

import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")

col_mapping = {
    "Stat Actual": float,
    "Stat Actual L0": float,
}


def check_all_dimensions(df, grains, input_stream):
    # checks if all 7 dim are present, if not, adds the dimension with member "All"
    # Renames input stream to Actual as well
    df_copy = df.copy()
    try:
        for i in grains:
            if i not in df_copy.columns:
                df_copy[i] = "All"
        df_copy = df_copy.rename(columns={input_stream: o9Constants.ACTUAL})
    except Exception as e:
        logger.exception(f"Error in check_all_dimensions\nError:-{e}")
        return df
    return df_copy


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actual,
    ForecastIterationSelection,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    Grains,
    PlanningGrains,
    StatHistoryStartDate=pd.DataFrame(),
    ForecastIterationMasterData=pd.DataFrame(),
    df_keys={},
):
    try:
        OutputList = list()
        ActualL0List = list()
        for the_iteration in ForecastLevelData[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output, the_actual_l0 = decorated_func(
                Actual=Actual,
                ForecastIterationSelection=ForecastIterationSelection,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
                StatHistoryStartDate=StatHistoryStartDate,
                ForecastIterationMasterData=ForecastIterationMasterData,
                Grains=Grains,
                PlanningGrains=PlanningGrains,
                df_keys=df_keys,
            )
            # dropping forecast iteration as the grain is not needed for Stat Fcst L0
            the_actual_l0.drop(columns=[o9Constants.FORECAST_ITERATION], inplace=True)

            OutputList.append(the_output)
            ActualL0List.append(the_actual_l0)

        Output = concat_to_dataframe(OutputList)
        ActualL0 = concat_to_dataframe(ActualL0List)
    except Exception as e:
        logger.exception(e)
        Output = None
        ActualL0 = None
    return Output, ActualL0


def processIteration(
    Actual,
    ForecastIterationSelection,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    Grains,
    PlanningGrains,
    StatHistoryStartDate=pd.DataFrame(),
    ForecastIterationMasterData=pd.DataFrame(),
    df_keys={},
):
    plugin_name = "DP063GenerateStatActual"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")
    planning_grains = PlanningGrains.strip().split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    cols_required_in_output = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [o9Constants.PARTIAL_WEEK, o9Constants.STAT_ACTUAL]
    )
    cols_required_in_actual_l0 = (
        [o9Constants.VERSION_NAME, o9Constants.FORECAST_ITERATION_TYPE]
        + planning_grains
        + [o9Constants.STAT_ACTUAL_L0]
    )

    Output = pd.DataFrame(columns=cols_required_in_output)
    ActualL0 = pd.DataFrame(columns=cols_required_in_actual_l0)
    try:

        input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]
        df = next(
            (df for key, df in Actual.items() if input_stream in df.columns),
            None,
        )
        if df is None:
            logger.warning(f"Input Stream '{input_stream}' not found, returning empty dataframe...")
            return Output, ActualL0
        Actual_df = df
        planning_grains_iteration = [col for col in Actual_df.columns if col in planning_grains]

        if Actual_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return Output, ActualL0

        if ForecastIterationSelection.empty:
            logger.warning("ForecastIterationSelection is empty, returning empty dataframe ...")
            return Output, ActualL0

        ForecastIterationSelection = ForecastIterationSelection.merge(
            ForecastIterationMasterData[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
            on=o9Constants.FORECAST_ITERATION_TYPE,
            how="inner",
        )
        # join with actuals to restrict intersections
        Actual_df = Actual_df.merge(
            ForecastIterationSelection.drop_duplicates(subset=planning_grains_iteration),
            on=[o9Constants.VERSION_NAME] + planning_grains_iteration,
            how="inner",
        )

        if len(Actual_df) == 0:
            logger.warning(
                "No data found after merging with ForecastIterationSelection, returning empty dataframe..."
            )
            return Output, ActualL0

        actual_columns = [col for col in Actual_df.columns if col != input_stream]

        # populating Actual L0
        # Logic - find sum of Actuals for all partial weeks on the input dimensions(could be <7D), update output dim to 7D using merge with ForecastIterationSelection, and divide the sum of actuals equally among all the resultant intersections
        ActualL0 = Actual_df.groupby(
            [o9Constants.VERSION_NAME] + planning_grains_iteration,
            as_index=False,
        ).agg({input_stream: "sum"})
        ActualL0 = ActualL0.merge(
            ForecastIterationSelection,
            on=[o9Constants.VERSION_NAME] + planning_grains_iteration,
            how="inner",
        )
        ActualL0["count"] = ActualL0.groupby(
            [o9Constants.VERSION_NAME] + planning_grains_iteration
        )[input_stream].transform("count")
        ActualL0 = ActualL0.drop_duplicates()
        ActualL0[input_stream] /= ActualL0["count"]
        ActualL0.rename(columns={input_stream: o9Constants.STAT_ACTUAL_L0}, inplace=True)
        ActualL0[o9Constants.FORECAST_ITERATION_TYPE] = ForecastIterationMasterData[
            o9Constants.FORECAST_ITERATION_TYPE
        ].values[0]

        # populate the default start date with the earliest partial week in the entire Actuals
        default_start_date = Actual_df.loc[
            Actual_df[o9Constants.PARTIAL_WEEK_KEY].idxmin(), o9Constants.PARTIAL_WEEK
        ]

        # join with actuals to restrict history period
        if len(StatHistoryStartDate) > 0:
            logger.info("Restricting history from the input history start dates ...")

            Actual_df = Actual_df.merge(
                StatHistoryStartDate.drop_duplicates(subset=planning_grains_iteration),
                on=[o9Constants.VERSION_NAME] + planning_grains_iteration,
                how="left",
            )
            Actual_df["Stat History Start Date"].fillna(default_start_date, inplace=True)
            Actual_df = Actual_df[
                Actual_df[o9Constants.PARTIAL_WEEK_KEY] >= Actual_df["Stat History Start Date"]
            ]
        else:
            logger.info("Stat History Start Date empty, executing plugin for the entire history")
        Actual_df = check_all_dimensions(Actual_df, planning_grains, input_stream)
        Actual_df = Actual_df[actual_columns + [o9Constants.ACTUAL]]

        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Channel"] = ChannelMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData
        master_data_dict["Region"] = RegionMasterData
        master_data_dict["Account"] = AccountMasterData
        master_data_dict["PnL"] = PnLMasterData
        master_data_dict["Location"] = LocationMasterData

        level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
        for the_col in level_cols:

            # extract 'Item' from 'Item Level'
            the_dim = the_col.split(" Level")[0]

            logger.debug(f"the_dim : {the_dim}")

            # all dims exception location will be planning location
            if the_dim == "Location":
                the_planning_col = the_dim + ".[" + the_dim + "]"
            else:
                the_planning_col = the_dim + ".[Planning " + the_dim + "]"

            logger.debug(f"the_planning_col : {the_planning_col}")

            # Item.[Stat Item]
            the_stat_col = the_dim + ".[Stat " + the_dim + "]"
            logger.debug(f"the_stat_col : {the_stat_col}")

            the_dim_data = master_data_dict[the_dim]

            # Eg. the_level = Item.[L2]
            the_level = the_dim + ".[" + ForecastLevelData[the_col].iloc[0] + "]"
            logger.debug(f"the_level : {the_level}")

            # copy values from L2 to Stat Item
            the_dim_data[the_stat_col] = the_dim_data[the_level]

            # select only relevant columns
            the_dim_data = the_dim_data[[the_planning_col, the_stat_col]].drop_duplicates()

            # join with Actual
            Actual_df = Actual_df.merge(the_dim_data, on=the_planning_col, how="inner")

            logger.debug("-------------------------")

        # rename columns

        Actual_df.rename(
            columns={
                o9Constants.ACTUAL: o9Constants.STAT_ACTUAL,
            },
            inplace=True,
        )

        # aggregate on stat columns, take sum
        Output = (
            Actual_df.groupby(
                [
                    o9Constants.VERSION_NAME,
                    o9Constants.PARTIAL_WEEK,
                ]
                + forecast_level
            )[o9Constants.STAT_ACTUAL]
            .sum()
            .reset_index()
        )

        Output = Output[cols_required_in_output]
        ActualL0 = ActualL0[cols_required_in_actual_l0]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
        ActualL0 = pd.DataFrame(columns=cols_required_in_actual_l0)
    return Output, ActualL0

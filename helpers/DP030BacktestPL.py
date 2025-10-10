import logging

import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.DP028PlanningLevelProfile import processIteration as planning_level_profile
from helpers.o9Constants import o9Constants
from helpers.utils import get_first_day_in_time_bucket

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Stat Fcst Profile PL Lag1 Backtest": float,
    "Stat Fcst Profile PL W Lag Backtest": float,
    "Stat Fcst Profile PL M Lag Backtest": float,
    "Stat Fcst Profile PL PM Lag Backtest": float,
}


def check_all_dimensions(df, grains, default_mapping):
    # checks if all 7 dim are present, if not, adds the dimension with member "All"
    # Renames input stream to Actual as well
    df_copy = df.copy()
    dims = {}
    for x in grains:
        dims[x] = x.strip().split(".")[0]
    try:
        for i in dims:
            if i not in df_copy.columns:
                if dims[i] in default_mapping:
                    df_copy[i] = default_mapping[dims[i]]
                else:
                    logger.warning(
                        f'Column {i} not found in default_mapping dictionary, adding the member "All"'
                    )
                    df_copy[i] = "All"
    except Exception as e:
        logger.exception(f"Error in check_all_dimensions\nError:-{e}")
        return df
    return df_copy


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Parameters,
    TimeDimension,
    Actual,
    PItemDates,
    CurrentTimePeriod,
    DefaultProfiles,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ItemLevel,
    SalesDomainGrains,
    LocationLevel,
    ReadFromHive,
    OutputMeasure,
    HistoryPeriodsInWeeks,
    MultiprocessingNumCores,
    LagsToStore,
    BackTestCyclePeriod,
    TransitionFlag,
    ForecastIterationMasterData=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    default_mapping={},
    df_keys={},
):
    plugin_name = "DP030BacktestPL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"

    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    partial_week_col = "Time.[Partial Week]"
    version_col = "Version.[Version Name]"
    lag_col = "Lag.[Lag]"
    stat_fcst_profile_lag1_col = " ".join([OutputMeasure, "Lag1 Backtest"])
    # history_measure_col = "History Measure"
    forecast_period_col = "Forecast Period"
    transition_item_col = "Item.[Transition Item]"
    stat_fcst_pl_lag1_r_exception = "Stat Fcst PL Lag1 R Exception"
    sell_out_offset_col = "Offset Period"

    other_grains = [str(x) for x in SalesDomainGrains.split(",") if x != "NA" and x != ""]

    all_grains = [
        ItemLevel,
        LocationLevel,
    ] + other_grains

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get granular level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("forecast_level : {} ...".format(forecast_level))

    # infer time related attributes from forecast gen time bucket
    fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
    logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

    # Default is planning month
    relevant_time_cols = [
        partial_week_col,
        partial_week_key_col,
        planning_month_col,
        planning_month_key_col,
    ]
    relevant_time_name = planning_month_col
    relevant_time_key = planning_month_key_col
    lag_suffix = " PM Lag"

    if fcst_gen_time_bucket == "Week":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            week_col,
            week_key_col,
        ]
        relevant_time_name = week_col
        relevant_time_key = week_key_col
        lag_suffix = " W Lag"

    elif fcst_gen_time_bucket == "Month":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            month_col,
            month_key_col,
        ]
        relevant_time_name = month_col
        relevant_time_key = month_key_col
        lag_suffix = " M Lag"

    elif fcst_gen_time_bucket == "Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            quarter_col,
            quarter_key_col,
        ]
        relevant_time_name = quarter_col
        relevant_time_key = quarter_key_col
        lag_suffix = " Q Lag"

    elif fcst_gen_time_bucket == "Planning Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            planning_quarter_col,
            planning_quarter_key_col,
        ]
        relevant_time_name = planning_quarter_col
        relevant_time_key = planning_quarter_key_col
        lag_suffix = " PQ Lag"

    stat_fcst_profile_lag_col = "".join(["Stat Fcst Profile PL", lag_suffix, " Backtest"])

    PLProfileLag1_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.FORECAST_ITERATION,
            partial_week_col,
            stat_fcst_profile_lag1_col,
        ]
    )
    PLProfileLag1 = pd.DataFrame(columns=PLProfileLag1_cols)

    if OutputMeasure == "ML Fcst Profile PL":
        LagModelOutput_cols = (
            [version_col]
            + forecast_level
            + [
                o9Constants.PLANNING_CYCLE_DATE,
                partial_week_col,
                stat_fcst_pl_lag1_r_exception,
            ]
        )
    else:
        LagModelOutput_cols = (
            [version_col]
            + forecast_level
            + [
                o9Constants.PLANNING_CYCLE_DATE,
                o9Constants.FORECAST_ITERATION,
                lag_col,
                relevant_time_name,
                stat_fcst_profile_lag_col,
            ]
        )

    LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)
    try:
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return PLProfileLag1, LagModelOutput

        history_measure = input_stream
        if isinstance(Actual, dict):
            df = next(
                (df for key, df in Actual.items() if input_stream in df.columns),
                None,
            )
            if df is None:
                logger.warning(
                    f"Input Stream '{input_stream}' not found, returning empty dataframe..."
                )
                return PLProfileLag1, LagModelOutput
            Actuals_df = df
        else:
            Actuals_df = Actual
        history_dims = [col.split(".")[0] for col in Actuals_df]
        target_dims = [dim.split(".")[0] for dim in forecast_level]
        missing_dims = list(set(target_dims) - set(history_dims))
        missing_grains = list(set(forecast_level) - set(Actuals_df.columns))

        missing_dim_grains = []
        if len(missing_dims) > 0:
            for dim in missing_dims:
                missing_dim_grains += [col for col in forecast_level if col.split(".")[0] == dim]
        if len(missing_dim_grains) > 0:
            Actuals_df = check_all_dimensions(
                df=Actuals_df,
                grains=missing_dim_grains,
                default_mapping=default_mapping,
            )

        missing_grains = list(set(missing_grains) - set(missing_dim_grains))
        if len(missing_grains) > 0:
            logger.warning(
                f"Dimensions {missing_grains} missing in the {input_stream} input query, please add the grains and try again"
            )
            return PLProfileLag1, LagModelOutput

        if Actuals_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return PLProfileLag1, LagModelOutput

        the_iteration = ForecastIterationMasterData[o9Constants.FORECAST_ITERATION].values[0]

        if len(SellOutOffset) == 0:
            logger.warning(
                f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        BackTestCyclePeriods = [int(x.strip()) for x in BackTestCyclePeriod.split(",")]

        if not BackTestCyclePeriods:
            logger.warning("BackTestCyclePeriods not populated")
            return PLProfileLag1, LagModelOutput

        if LagsToStore.upper().strip() == "ALL":
            # generate forecasts for one complete cycle
            forecast_horizon = 52 if fcst_gen_time_bucket == "Week" else 12
            LagsToStore = list(range(0, forecast_horizon))
        else:
            # convert lags to store to a list
            LagsToStore = LagsToStore.split(",")
            # trim leading/trailing spaces
            LagsToStore = [int(x.strip()) for x in LagsToStore]

            # say we want to store lags 2, 4, 6 - we should generate forecast for max(2, 4, 6) + 1 - 7 cycles
            forecast_horizon = max(LagsToStore) + 1
            logger.debug(f"forecast_horizon : {forecast_horizon}")

        logger.info(f"LagsToStore : {LagsToStore}")

        # retrieve history measure
        history_measure = input_stream

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}

        current_time_period_in_relevant_bucket = CurrentTimePeriod[relevant_time_name].iloc[0]

        logger.info(
            f"current_time_period_in_relevant_bucket : {current_time_period_in_relevant_bucket}"
        )

        latest_time_name = current_time_period_in_relevant_bucket

        # adjust the latest time according to the forecast iteration's offset before getting n periods for considering history
        offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
        if offset_periods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info(f"latest_time_name after offset {offset_periods} : {latest_time_name} ...")

        # get max value from backtestcycleperiod and supply to get n time periods
        last_n_cycles = get_n_time_periods(
            latest_value=latest_time_name,
            periods=-(max(BackTestCyclePeriods) + 1),
            time_mapping=relevant_time_mapping,
            time_attribute=time_attribute_dict,
            include_latest_value=True,
        )

        backtest_cycles = [last_n_cycles[-(i + 1)] for i in BackTestCyclePeriods]
        logger.info(f"backtest_cycles : {backtest_cycles}")

        # evaluate cycles in oldest first order
        backtest_cycles.reverse()

        # join actuals with pw
        Actuals_df = Actuals_df.merge(base_time_mapping, on=partial_week_col, how="inner")

        input_version = Actuals_df[version_col].unique()[0]

        groupby_cols_at_relevant_time_level = (
            [version_col]
            + forecast_level
            + [
                o9Constants.FORECAST_ITERATION,
                relevant_time_name,
                relevant_time_key,
            ]
        )
        history_req_cols = (
            [version_col]
            + forecast_level
            + [
                transition_item_col,
                o9Constants.TRANSITION_DEMAND_DOMAIN,
                partial_week_col,
                history_measure,
            ]
        )

        # assign forecast horizon
        Parameters[forecast_period_col] = forecast_horizon

        all_PLProfileLag1 = list()
        all_LagModelOutput = list()
        for the_cycle in backtest_cycles:
            logger.info(f"the_cycle : {the_cycle}")

            the_cycle_time_mapping = base_time_mapping[
                base_time_mapping[relevant_time_name] == the_cycle
            ]
            the_cycle_time_mapping.sort_values(partial_week_key_col, inplace=True)

            the_planning_cycle_date = get_first_day_in_time_bucket(
                time_bucket_value=the_cycle,
                relevant_time_name=relevant_time_name,
                time_dimension=TimeDimension,
            )
            logger.debug(f"the_planning_cycle_date : {the_planning_cycle_date}")

            the_current_time_period = (
                TimeDimension[TimeDimension[relevant_time_name] == the_cycle].head(1).reset_index()
            )
            the_current_time_period[version_col] = input_version
            logger.debug(f"the_current_time_period\n{the_current_time_period}")

            the_lag_1_time_period = get_n_time_periods(
                latest_value=the_cycle,
                periods=1,
                time_mapping=base_time_mapping,
                time_attribute=time_attribute_dict,
                include_latest_value=False,
            )[0]

            filter_clause = base_time_mapping[relevant_time_name] == the_lag_1_time_period
            the_lag_1_base_time_mapping = base_time_mapping[filter_clause]
            the_lag_1_partial_weeks = list(the_lag_1_base_time_mapping[partial_week_col].unique())

            # get the first PW
            the_cycle_first_partial_week_key = the_cycle_time_mapping[partial_week_key_col].min()

            # filter relevant data
            filter_clause = Actuals_df[partial_week_key_col] < the_cycle_first_partial_week_key
            the_history = Actuals_df[filter_clause]

            # filter the relevant columns
            the_history = the_history[history_req_cols]
            logger.debug(f"the_history, shape : {the_history.shape}")

            # get the transition profile
            the_pl_profile, actual_last_n_buckets = planning_level_profile(
                LocationLevel=LocationLevel,
                ItemLevel=ItemLevel,
                Actual=the_history,
                TimeDimension=TimeDimension,
                PItemDates=PItemDates,
                Parameters=Parameters,
                SalesDomainLevel=SalesDomainGrains,
                CurrentTimePeriod=the_current_time_period,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                ForecastIterationMasterData=ForecastIterationMasterData,
                StatBucketWeight=StatBucketWeight,
                HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
                ReadFromHive=ReadFromHive,
                OutputMeasure=OutputMeasure,
                multiprocessing_cores=MultiprocessingNumCores,
                default_mapping=default_mapping,
                DefaultProfiles=DefaultProfiles,
                df_keys={},
                TransitionFlag=TransitionFlag,
                SellOutOffset=SellOutOffset,
                the_iteration=the_iteration,
            )

            if len(the_pl_profile) == 0:
                logger.warning(f"the_pl_profile empty for cycle {the_cycle}")
                continue

            the_pl_profile.insert(
                loc=0,
                column=o9Constants.FORECAST_ITERATION,
                value=Actuals_df[o9Constants.FORECAST_ITERATION].unique()[0],
            )

            # select the data corresponding to the lag1 time period
            the_lag_1_output = the_pl_profile[
                the_pl_profile[partial_week_col].isin(the_lag_1_partial_weeks)
            ]

            # rename column
            the_lag_1_output.rename(
                columns={OutputMeasure: stat_fcst_profile_lag1_col},
                inplace=True,
            )

            # select relevant columns and add to master list
            the_lag_1_output = the_lag_1_output[PLProfileLag1_cols]
            all_PLProfileLag1.append(the_lag_1_output)

            # no need to calculate lag output if output measure is 'ML Fcst Profile PL'
            if OutputMeasure == "ML Fcst Profile PL":
                continue

            # result will be at PW level, join time mapping to get data at relevant_time_name
            the_pl_profile = the_pl_profile.merge(
                base_time_mapping, on=partial_week_col, how="inner"
            )

            logger.debug("Creating lag mapping ...")

            # create the lag to relevant time key mapping
            the_lag_mapping = pd.DataFrame(
                {relevant_time_key: the_pl_profile[relevant_time_key].unique()}
            )
            the_lag_mapping.sort_values(relevant_time_key, inplace=True, ignore_index=True)
            the_lag_mapping[lag_col] = the_lag_mapping.index

            # typecasting the column to make sure there are no decimals in the lag col
            the_lag_mapping[lag_col] = the_lag_mapping[lag_col].astype(int)

            # filter the required lags
            the_lag_filter_clause = the_lag_mapping[lag_col].isin(LagsToStore)
            the_lag_mapping = the_lag_mapping[the_lag_filter_clause]

            if len(the_lag_mapping) == 0:
                logger.warning("the_lag_mapping is empty ...")
                continue

            # join to get the partial weeks
            the_lag_mapping = the_lag_mapping.merge(
                base_time_mapping, on=relevant_time_key, how="inner"
            )

            # select the PWs present in the lag mapping
            req_partial_weeks = list(the_lag_mapping[partial_week_key_col].unique())
            filter_clause = the_pl_profile[partial_week_key_col].isin(req_partial_weeks)
            the_pl_profile = the_pl_profile[filter_clause]

            logger.debug(
                f"Grouping at {groupby_cols_at_relevant_time_level} level and aggregating {OutputMeasure} ..."
            )

            # create the lag model output
            the_lag_output = (
                the_pl_profile.groupby(groupby_cols_at_relevant_time_level)
                .sum(min_count=1)[[OutputMeasure]]
                .reset_index()
            )

            # rename measures
            the_lag_output.rename(
                columns={OutputMeasure: stat_fcst_profile_lag_col},
                inplace=True,
            )

            # populate lag dimension - join with lag mapping
            the_lag_mapping = the_lag_mapping[[relevant_time_key, lag_col]].drop_duplicates()
            the_lag_output = the_lag_output.merge(
                the_lag_mapping, on=relevant_time_key, how="inner"
            )

            # sort for easier readability
            the_lag_output.sort_values(relevant_time_key, inplace=True)
            the_lag_output.reset_index(drop=True, inplace=True)

            the_lag_output[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date

            logger.debug(f"the_lag_output, shape : {the_lag_output.shape}")

            # filter relevant columns required in lag output and append to master list
            the_lag_output = the_lag_output[LagModelOutput_cols]
            all_LagModelOutput.append(the_lag_output)

            logger.info(f"{df_keys}, --- {the_cycle} complete ----")

        PLProfileLag1 = concat_to_dataframe(all_PLProfileLag1)
        if len(PLProfileLag1) == 0:
            PLProfileLag1 = pd.DataFrame(columns=PLProfileLag1_cols)

        LagModelOutput = concat_to_dataframe(all_LagModelOutput)
        if len(LagModelOutput) == 0 or OutputMeasure == "ML Fcst Profile PL":
            LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)

        logger.info(f"Succesfully executed {plugin_name} ...")

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        PLProfileLag1 = pd.DataFrame(columns=PLProfileLag1_cols)
        LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)
    return PLProfileLag1, LagModelOutput

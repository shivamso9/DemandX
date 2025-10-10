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

from helpers.DP016TransitionLevelStat import processIteration as transitionlevelstat
from helpers.DP066SpreadStatFcstL1ToTL import processIteration as spread_l1_to_tl
from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration, get_first_day_in_time_bucket

logger = logging.getLogger("o9_logger")


pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {
    "Stat Fcst Profile TL W Lag": float,
    "Stat Fcst Profile TL M Lag": float,
    "Stat Fcst Profile TL PM Lag": float,
    "Stat Fcst Profile TL Lag1": float,
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
    CurrentTimePeriod,
    Actuals,
    DisaggregationType,
    StatFcstL1ForFIPLIteration,
    ForecastIterationSelectionAtTransitionLevel,
    StatActual,
    TimeDimension,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ForecastParameters,
    StatFcstL1Lag1,
    StatFcstL1WLag,
    StatFcstL1MLag,
    StatFcstL1PMLag,
    StatFcstL1QLag,
    StatFcstL1PQLag,
    ForecastIterationMasterData,
    SellOutOffset,
    TItemDates,
    Grains,
    StatGrains,
    HistoryPeriodsInWeeks,
    UseMovingAverage,
    LagsToStore,
    BackTestCyclePeriod,
    MLDecompositionFlag=pd.DataFrame(),
    multiprocessing_num_cores=1,
    default_mapping={},
    df_keys={},
):
    try:
        if o9Constants.FORECAST_ITERATION in ForecastGenTimeBucket.columns:
            TItemFcstLag1_list = list()
            LagModelOutput_list = list()

            for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
                logger.warning(f"--- Processing iteration {the_iteration}")

                decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)
                (
                    the_t_item_fcst_lag1_output,
                    the_lag_model_output,
                ) = decorated_func(
                    CurrentTimePeriod=CurrentTimePeriod,
                    Actuals=Actuals,
                    DisaggregationType=DisaggregationType,
                    StatFcstL1ForFIPLIteration=StatFcstL1ForFIPLIteration,
                    ForecastIterationSelectionAtTransitionLevel=ForecastIterationSelectionAtTransitionLevel,
                    StatActual=StatActual,
                    TimeDimension=TimeDimension,
                    ForecastLevelData=ForecastLevelData,
                    ItemMasterData=ItemMasterData,
                    RegionMasterData=RegionMasterData,
                    AccountMasterData=AccountMasterData,
                    ChannelMasterData=ChannelMasterData,
                    PnLMasterData=PnLMasterData,
                    DemandDomainMasterData=DemandDomainMasterData,
                    LocationMasterData=LocationMasterData,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatBucketWeight=StatBucketWeight,
                    ForecastParameters=ForecastParameters,
                    StatFcstL1Lag1=StatFcstL1Lag1,
                    StatFcstL1WLag=StatFcstL1WLag,
                    StatFcstL1MLag=StatFcstL1MLag,
                    StatFcstL1PMLag=StatFcstL1PMLag,
                    StatFcstL1QLag=StatFcstL1QLag,
                    StatFcstL1PQLag=StatFcstL1PQLag,
                    ForecastIterationMasterData=ForecastIterationMasterData,
                    SellOutOffset=SellOutOffset,
                    TItemDates=TItemDates,
                    Grains=Grains,
                    StatGrains=StatGrains,
                    HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
                    UseMovingAverage=UseMovingAverage,
                    LagsToStore=LagsToStore,
                    BackTestCyclePeriod=BackTestCyclePeriod,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    MLDecompositionFlag=MLDecompositionFlag,
                    default_mapping=default_mapping,
                    the_iteration=the_iteration,
                    df_keys=df_keys,
                )
                TItemFcstLag1_list.append(the_t_item_fcst_lag1_output)
                LagModelOutput_list.append(the_lag_model_output)
            TItemFcstLag1 = concat_to_dataframe(TItemFcstLag1_list)
            LagModelOutput = concat_to_dataframe(LagModelOutput_list)
    except Exception as e:
        logger.exception(e)
        TItemFcstLag1, LagModelOutput = None, None
    return TItemFcstLag1, LagModelOutput


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def processIteration(
    CurrentTimePeriod,
    Actuals,
    DisaggregationType,
    StatFcstL1ForFIPLIteration,
    ForecastIterationSelectionAtTransitionLevel,
    StatActual,
    TimeDimension,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ForecastParameters,
    StatFcstL1Lag1,
    StatFcstL1WLag,
    StatFcstL1MLag,
    StatFcstL1PMLag,
    StatFcstL1QLag,
    StatFcstL1PQLag,
    ForecastIterationMasterData,
    SellOutOffset,
    TItemDates,
    Grains,
    StatGrains,
    HistoryPeriodsInWeeks,
    UseMovingAverage,
    LagsToStore,
    BackTestCyclePeriod,
    MLDecompositionFlag,
    multiprocessing_num_cores,
    default_mapping,
    the_iteration,
    df_keys,
):
    plugin_name = "DP029BacktestL1"
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
    logger.debug(f"multiprocessing_num_cores : {multiprocessing_num_cores}")
    stat_fcst_tl_lag1_col = "Stat Fcst TL Lag1"
    lag_col = "Lag.[Lag]"
    history_period_col = "History Period"
    forecast_period_col = "Forecast Period"
    GROUP_SUM: str = "Group Sum"
    DISAGG_PROPORTION: str = "Disagg Proportion"
    sell_out_offset_col = "Offset Period"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get granular level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("forecast_level : {} ...".format(forecast_level))

    # infer time related attributes from forecast gen time bucket
    fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
    logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

    # Default is Planning Month
    relevant_time_cols = [
        partial_week_col,
        partial_week_key_col,
        planning_month_col,
        planning_month_key_col,
    ]
    relevant_time_name = planning_month_col
    relevant_time_key = planning_month_key_col
    lag_suffix = " PM Lag"
    stat_fcst_l1_lag_df = StatFcstL1PMLag.copy()
    rename_mapping = {o9Constants.STAT_FCST_L1_PM_LAG_BACKTEST: o9Constants.STAT_FCST_L1}

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
        stat_fcst_l1_lag_df = StatFcstL1WLag.copy()
        rename_mapping = {o9Constants.STAT_FCST_L1_W_LAG_BACKTEST: o9Constants.STAT_FCST_L1}
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
        stat_fcst_l1_lag_df = StatFcstL1MLag.copy()
        rename_mapping = {o9Constants.STAT_FCST_L1_M_LAG_BACKTEST: o9Constants.STAT_FCST_L1}
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
        stat_fcst_l1_lag_df = StatFcstL1QLag.copy()
        rename_mapping = {o9Constants.STAT_FCST_L1_Q_LAG_BACKTEST: o9Constants.STAT_FCST_L1}
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
        stat_fcst_l1_lag_df = StatFcstL1PQLag.copy()
        rename_mapping = {o9Constants.STAT_FCST_L1_PQ_LAG_BACKTEST: o9Constants.STAT_FCST_L1}

    stat_fcst_profile_tl_lag_col = "".join(["Stat Fcst TL", lag_suffix])

    TItemFcstLag1_cols = (
        [version_col]
        + forecast_level
        + [
            partial_week_col,
            stat_fcst_tl_lag1_col,
        ]
    )
    TItemFcstLag1 = pd.DataFrame(columns=TItemFcstLag1_cols)
    LagModelOutput_cols = (
        [version_col]
        + forecast_level
        + [
            lag_col,
            o9Constants.PLANNING_CYCLE_DATE,
            relevant_time_name,
            stat_fcst_profile_tl_lag_col,
        ]
    )
    LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)
    try:
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return TItemFcstLag1, LagModelOutput

        # history_measure = input_stream
        if isinstance(Actuals, dict):
            df = next(
                (df for key, df in Actuals.items() if input_stream in df.columns),
                None,
            )
            if df is None:
                logger.warning(
                    f"Input Stream '{input_stream}' not found, returning empty dataframe..."
                )
                return TItemFcstLag1, LagModelOutput
            Actuals_df = df
        else:
            Actuals_df = Actuals
        history_dims = [col.split(".")[0] for col in Actuals_df]
        target_dims = [dim.split(".")[0] for dim in forecast_level]
        missing_dims = list(set(target_dims) - set(history_dims))
        missing_grains = list(set(forecast_level) - set(Actuals_df.columns))

        if Actuals_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return TItemFcstLag1, LagModelOutput

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

        stat_fcst_l1_lag_df[lag_col] = stat_fcst_l1_lag_df[lag_col].astype("int")
        stat_fcst_l1_lag_df.rename(columns=rename_mapping, inplace=True)

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

        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Channel"] = ChannelMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData
        master_data_dict["Region"] = RegionMasterData
        master_data_dict["Account"] = AccountMasterData
        master_data_dict["PnL"] = PnLMasterData
        master_data_dict["Location"] = LocationMasterData

        # if target grain is missing, but there is another level present in input for the same dimension, we go through the master data and find the target grain values
        if len(missing_grains) > 0:
            for grain in missing_grains:
                dim_of_missing_grain = grain.split(".")[0]
                master_df = master_data_dict[dim_of_missing_grain]
                existing_grain = [
                    col for col in Actuals_df.columns if col.split(".")[0] == dim_of_missing_grain
                ][0]
                Actuals_df = Actuals_df.merge(
                    master_df[[existing_grain, grain]],
                    on=existing_grain,
                    how="inner",
                )

        if not BackTestCyclePeriods:
            logger.warning("BackTestCyclePeriods not populated")
            return TItemFcstLag1, LagModelOutput

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

        # rename column, remove Lag so that we can use the spreading function as is
        StatFcstL1Lag1.rename(
            columns={o9Constants.STAT_FCST_L1_LAG1_BACKTEST: o9Constants.STAT_FCST_L1},
            inplace=True,
        )

        HistoryPeriods = int(ForecastParameters[history_period_col].unique()[0])

        logger.debug(f"HistoryPeriods : {HistoryPeriods}")

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

        groupby_cols_at_relevant_time_level = (
            [version_col] + forecast_level + [relevant_time_name, relevant_time_key]
        )

        # assign forecast horizon
        ForecastParameters[forecast_period_col] = forecast_horizon

        input_version = ForecastGenTimeBucket[version_col].unique()[0]

        all_TItemFcstLag = list()
        all_LagModelOutput = list()
        for the_cycle in backtest_cycles:
            logger.debug(f"the_cycle : {the_cycle}")

            the_cycle_time_mapping = base_time_mapping[
                base_time_mapping[relevant_time_name] == the_cycle
            ]
            the_cycle_time_mapping.sort_values(partial_week_key_col, inplace=True)

            the_current_time_period = pd.DataFrame(
                {
                    version_col: input_version,
                    relevant_time_name: the_cycle,
                    relevant_time_key: the_cycle_time_mapping[relevant_time_key].iloc[0],
                    partial_week_col: the_cycle_time_mapping.head(1)[partial_week_col].iloc[0],
                    partial_week_key_col: the_cycle_time_mapping.head(1)[partial_week_key_col].iloc[
                        0
                    ],
                },
                index=[0],
            )
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

            the_planning_cycle_date = get_first_day_in_time_bucket(
                time_bucket_value=the_cycle,
                relevant_time_name=relevant_time_name,
                time_dimension=TimeDimension,
            )
            logger.debug(f"the_planning_cycle_date : {the_planning_cycle_date}")

            # get the first PW
            the_cycle_first_partial_week = the_cycle_time_mapping[partial_week_key_col].min()

            # filter relevant data
            filter_clause = Actuals_df[partial_week_key_col] < the_cycle_first_partial_week
            the_history = Actuals_df[filter_clause]
            if len(the_history) == 0:
                logger.warning(f"No history available prior to {the_cycle_first_partial_week}")
                continue

            logger.debug(f"the_history, shape : {the_history.shape}")

            # get the transition profile
            the_transition_level_profile = transitionlevelstat(
                History=the_history,
                CurrentTimePeriod=the_current_time_period,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                Grains=Grains,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                DisaggregationType=DisaggregationType,
                StatFcstL1ForFIPLIteration=StatFcstL1ForFIPLIteration,
                ForecastIterationSelectionAtTransitionLevel=ForecastIterationSelectionAtTransitionLevel,
                ForecastIterationMasterData=ForecastIterationMasterData,
                SellOutOffset=SellOutOffset,
                StatActual=StatActual,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
                StatGrains=StatGrains,
                TItemDates=TItemDates,
                HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
                UseMovingAverage=UseMovingAverage,
                multiprocessing_num_cores=multiprocessing_num_cores,
                default_mapping=default_mapping,
                the_iteration=the_iteration,
                df_keys=df_keys,
            )

            if len(the_transition_level_profile) == 0:
                logger.warning(f"the_transition_level_forecast empty for cycle {the_cycle}")
                continue

            # drop individual profile columns
            the_transition_level_profile.drop(
                [
                    o9Constants.STAT_FCST_PROFILE1_TL,
                    o9Constants.STAT_FCST_PROFILE2_TL,
                    o9Constants.STAT_FCST_PROFILE3_TL,
                ],
                axis=1,
                inplace=True,
            )

            # select the data corresponding to the lag1 time period
            the_lag_1_output = the_transition_level_profile[
                the_transition_level_profile[partial_week_col].isin(the_lag_1_partial_weeks)
            ]

            the_stat_fcst_l1_lag1 = StatFcstL1Lag1[
                StatFcstL1Lag1[partial_week_col].isin(the_lag_1_partial_weeks)
            ]

            the_stat_fcst_l1_lag1 = the_stat_fcst_l1_lag1.merge(
                TimeDimension[[partial_week_col, partial_week_key_col]],
                on=partial_week_col,
                how="inner",
            )
            TItemDates_copy = TItemDates.copy()
            missing_stat_grains = [
                o9Constants.STAT_ITEM,
                o9Constants.STAT_REGION,
                o9Constants.STAT_ACCOUNT,
                o9Constants.STAT_CHANNEL,
                o9Constants.STAT_PNL,
                o9Constants.STAT_DEMAND_DOMAIN,
                o9Constants.STAT_LOCATION,
            ]
            if len(missing_stat_grains) > 0:
                for grain in missing_stat_grains:
                    dim_of_missing_grain = grain.split(".")[0]
                    master_df = master_data_dict[dim_of_missing_grain]
                    existing_grain = [
                        col
                        for col in TItemDates_copy.columns
                        if col.split(".")[0] == dim_of_missing_grain
                    ][0]
                    TItemDates_copy = TItemDates_copy.merge(
                        master_df[[existing_grain, grain]],
                        on=existing_grain,
                        how="inner",
                    )
                # TItemDates_copy = TItemDates_copy[
                #     missing_stat_grains + [o9Constants.VERSION_NAME, "Intro Date", "Disco Date"]
                # ].drop_duplicates()

            # call the spreading function
            the_stat_fcst_tl_lag1, cml_fcst_tl_lag1, volume_loss_flag = spread_l1_to_tl(
                Grains=Grains,
                StatFcstFinalProfileTL=the_lag_1_output,
                StatFcstL1=the_stat_fcst_l1_lag1,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
                default_mapping=default_mapping,
                ItemDates=TItemDates_copy,
                MLDecompositionFlag=MLDecompositionFlag,
                ForecastGenerationTimeBucket=ForecastGenTimeBucket,
                CurrentDay=CurrentTimePeriod,
                the_iteration=the_iteration,
                df_keys=df_keys,
            )

            # rename column
            the_stat_fcst_tl_lag1.rename(
                columns={o9Constants.STAT_FCST_TL: stat_fcst_tl_lag1_col},
                inplace=True,
            )

            # select relevant columns and add to master list
            the_stat_fcst_tl_lag1 = the_stat_fcst_tl_lag1[TItemFcstLag1_cols]
            all_TItemFcstLag.append(the_stat_fcst_tl_lag1)

            # result will be at PW level, join time mapping to get data at relevant_time_name
            the_transition_level_profile = the_transition_level_profile.merge(
                base_time_mapping, on=partial_week_col, how="inner"
            )

            logger.debug("Creating lag mapping ...")

            # create the lag to relevant time key mapping
            the_lag_mapping = pd.DataFrame(
                {relevant_time_key: the_transition_level_profile[relevant_time_key].unique()}
            )
            the_lag_mapping.sort_values(relevant_time_key, inplace=True)
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
            filter_clause = the_transition_level_profile[partial_week_key_col].isin(
                req_partial_weeks
            )
            the_transition_level_profile = the_transition_level_profile[filter_clause]

            logger.debug(
                f"Grouping at {groupby_cols_at_relevant_time_level} level and aggregating {o9Constants.STAT_FCST_FINAL_PROFILE_TL} ..."
            )

            # create the lag model output
            the_stat_fcst_final_profile_tl_lag_output = (
                the_transition_level_profile.groupby(groupby_cols_at_relevant_time_level)
                .sum(min_count=1)[[o9Constants.STAT_FCST_FINAL_PROFILE_TL]]
                .reset_index()
            )

            # limit L1 data to the planning cycle date
            filter_clause = (
                stat_fcst_l1_lag_df[o9Constants.PLANNING_CYCLE_DATE] == the_planning_cycle_date
            )
            the_stat_fcst_l1_lag_df = stat_fcst_l1_lag_df[filter_clause]

            if the_stat_fcst_l1_lag_df.empty:
                logger.warning(f"No stat_fcst_l1_lag_df data found for {the_planning_cycle_date}")
                continue

            # Limit 1l lag data to selected lags
            the_stat_fcst_l1_lag_df = the_stat_fcst_l1_lag_df.merge(
                the_lag_mapping[[lag_col]].drop_duplicates(),
                on=lag_col,
                how="inner",
            )

            level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
            all_stat_grains = list()
            for the_col in level_cols:

                # extract 'Item' from 'Item Level'
                the_dim = the_col.split(" Level")[0]

                logger.debug(f"the_dim : {the_dim}")

                # transition/planning item
                if the_dim in ["Item", "Demand Domain"]:
                    the_relevant_col = the_dim + ".[Transition " + the_dim + "]"
                else:
                    the_relevant_col = the_dim + ".[Planning " + the_dim + "]"

                logger.debug(f"the_relevant_col : {the_relevant_col}")

                # Item.[Stat Item]
                the_stat_col = the_dim + ".[Stat " + the_dim + "]"
                logger.debug(f"the_stat_col : {the_stat_col}")
                all_stat_grains.append(the_stat_col)

                the_dim_data = master_data_dict[the_dim]

                # Eg. the_level = Item.[L2]
                the_level = the_dim + ".[" + ForecastLevelData[the_col].iloc[0] + "]"
                logger.debug(f"the_level : {the_level}")

                # copy values from L2 to Stat Item
                the_dim_data[the_stat_col] = the_dim_data[the_level]

                # select only relevant columns
                the_dim_data = the_dim_data[[the_relevant_col, the_stat_col]].drop_duplicates()

                if the_level not in [
                    o9Constants.PLANNING_ITEM,
                    o9Constants.PLANNING_DEMAND_DOMAIN,
                ]:
                    if the_dim_data[the_relevant_col].nunique() != len(the_dim_data):
                        duplicates = the_dim_data[the_dim_data[the_relevant_col].duplicated()]
                        logger.warning(
                            f"Erratic Master data:\n{duplicates.head().to_csv(index=False)}"
                        )
                        logger.warning(
                            "Will continue with the clean data, but check the master data for duplicates"
                        )
                        # drop duplicated values
                        the_dim_data = the_dim_data[~the_dim_data[the_relevant_col].duplicated()]

                # join with Final Profile TL
                the_stat_fcst_final_profile_tl_lag_output = (
                    the_stat_fcst_final_profile_tl_lag_output.merge(
                        the_dim_data, on=the_relevant_col, how="inner"
                    )
                )

            # join with stat fcst l1 - to retain relevant intersections
            the_stat_fcst_final_profile_tl_lag_output = (
                the_stat_fcst_final_profile_tl_lag_output.merge(
                    the_stat_fcst_l1_lag_df,
                    on=[o9Constants.VERSION_NAME, relevant_time_name] + all_stat_grains,
                    how="inner",
                )
            )

            # create group sum - at stat level
            the_stat_fcst_final_profile_tl_lag_output[GROUP_SUM] = (
                the_stat_fcst_final_profile_tl_lag_output.groupby(
                    all_stat_grains + [relevant_time_name]
                )[o9Constants.STAT_FCST_FINAL_PROFILE_TL].transform("sum")
            )

            # create proportions
            the_stat_fcst_final_profile_tl_lag_output[DISAGG_PROPORTION] = (
                the_stat_fcst_final_profile_tl_lag_output[o9Constants.STAT_FCST_FINAL_PROFILE_TL]
                / the_stat_fcst_final_profile_tl_lag_output[GROUP_SUM]
            )

            # multiply value with proportion
            the_stat_fcst_final_profile_tl_lag_output[stat_fcst_profile_tl_lag_col] = (
                the_stat_fcst_final_profile_tl_lag_output[o9Constants.STAT_FCST_L1]
                * the_stat_fcst_final_profile_tl_lag_output[DISAGG_PROPORTION]
            )

            # add planning cycle date
            the_stat_fcst_final_profile_tl_lag_output[o9Constants.PLANNING_CYCLE_DATE] = (
                the_planning_cycle_date
            )

            all_LagModelOutput.append(
                the_stat_fcst_final_profile_tl_lag_output[LagModelOutput_cols]
            )

            logger.debug(f"{df_keys}, --- {the_cycle} complete ----")

        TItemFcstLag1 = concat_to_dataframe(all_TItemFcstLag)
        if len(TItemFcstLag1) == 0:
            TItemFcstLag1 = pd.DataFrame(columns=TItemFcstLag1_cols)

        LagModelOutput = concat_to_dataframe(all_LagModelOutput)
        if len(LagModelOutput) == 0:
            LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)

        logger.info(f"Succesfully executed {plugin_name} ...")

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        TItemFcstLag1 = pd.DataFrame(columns=TItemFcstLag1_cols)
        LagModelOutput = pd.DataFrame(columns=LagModelOutput_cols)
    return TItemFcstLag1, LagModelOutput

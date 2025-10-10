import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from sktime.forecasting.ets import AutoETS

from helpers.o9Constants import o9Constants
from helpers.utils import add_dim_suffix, filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")


def get_moving_avg_forecast(
    data: np.ndarray, moving_avg_periods: int, forecast_horizon: int
) -> np.array:
    """
    Calculates sliding window forecast with n periods, uses history and forecast available to produce future values iteratively
    :param data: array containing the historical data to use for forecasting
    :param moving_avg_periods: integer representing the number of periods to use for the moving average calculation
    :param forecast_horizon: integer representing the number of periods to forecast into the future
    :return: returns an array containing the sliding window forecast values.
    """
    assert isinstance(data, np.ndarray), "train should be of type np.array ..."
    assert isinstance(forecast_horizon, int), "forecast_horizon should be of type integer ..."
    assert isinstance(moving_avg_periods, int), "moving_avg_periods should be of type integer ..."

    assert forecast_horizon > 0, "forecast_horizon should be greater than 0 ..."
    assert moving_avg_periods > 0, "moving_avg_periods should be greater than 0 ..."

    # check for empty array
    if data.size == 0:
        return np.array([])

    # Fill nas with zero
    data[np.isnan(data)] = 0

    # Not enough data, take mean of all available points as forecast
    if data.size == 1 or data.size < moving_avg_periods:
        return np.repeat(np.mean(data), forecast_horizon)

    # create copy of array to avoid modification to source data
    history_values = np.array(data, copy=True).tolist()

    forecasts = []
    for the_period in range(0, forecast_horizon):
        # take avg of last n periods
        last_n_periods_avg = np.mean(history_values[-moving_avg_periods:])

        # append to forecasts
        forecasts.append(last_n_periods_avg)

        # append to source list for next iteration
        history_values.append(last_n_periods_avg)

    assert (
        len(forecasts) == forecast_horizon
    ), "mismatch in output size, check source data and calculation ..."

    return np.array(forecasts)


def calculate_weighted_profile(
    df,
    dimensions,
    first_n_periods_future,
    FuturePeriod,
    frequency_period,
    HistoryMeasure,
    OutputMeasure,
    time_level_col,
    L6MAvg_wt,
    LYHistory_wt,
) -> pd.DataFrame:
    result = pd.DataFrame()
    try:
        if len(df) == 0:
            return result

        # fill nas with zero for calculating weighted profile
        df[HistoryMeasure].fillna(0, inplace=True)

        # mean of last n periods, perform whole number division to get rid of float in quotient
        if time_level_col in ["Time.[Planning Month]", "Time.[Month]"]:
            span = frequency_period // 2  # consider last 6 months for monthly level
        elif time_level_col in ["Time.[Planning Quarter]", "Time.[Quarter]"]:
            span = frequency_period // 1  # consider last 4 quarters for quarterly level
        elif time_level_col == "Time.[Week]":
            span = frequency_period // 4  # consider last 13 weeks for Weekly level
        else:
            raise ValueError("Unknown TimeLevel : {}".format(time_level_col))

        L6MAvg = df[HistoryMeasure].tail(span).mean()
        # Last year history
        LYHistory = df[HistoryMeasure].tail(frequency_period).to_numpy()

        # Check the length of the array
        if len(LYHistory) < frequency_period:
            # Calculate the number of leading zeros needed
            num_zeros = frequency_period - len(LYHistory)
            # Add leading zeros to the array
            LYHistory = np.pad(LYHistory, (num_zeros, 0), mode="constant", constant_values=0)

        # generate forecast
        LYProfileForecast = LYHistory * LYHistory_wt + L6MAvg * L6MAvg_wt
        # resize array to match future_periods
        LYProfileForecast = np.resize(LYProfileForecast, FuturePeriod)

        result = pd.DataFrame(
            {
                time_level_col: first_n_periods_future,
                OutputMeasure: LYProfileForecast,
            }
        )
        for the_col in dimensions:
            result.insert(loc=0, column=the_col, value=df[the_col].iloc[0])
    except Exception as e:
        logger.exception("Exception {} for {}".format(e, df[dimensions].iloc[0].values))
        result = pd.DataFrame()
    return result


def __is_constant_time_series(train: pd.Series) -> bool:
    a = train.to_numpy()
    return (a[0] == a).all(0)


def get_ETS_forecast(
    data: pd.Series,
    frequency: int = 12,
    history_period: int = 36,
    future_period: int = 36,
):
    res = np.array([])
    try:
        function_name = "ETS_forecast"
        logger.info("inside function {}...".format(function_name))
        if data is None or len(data) == 0:
            logger.info("inside function {}: data is empty".format(function_name))
            logger.info("inside function {}: returning empty data".format(function_name))
            return res

        # if there's only one datapoint, return that point itself as profile
        if len(data) == 1:
            return np.repeat(data.values[0], future_period)

        # if time series is constant, return same value as prediction - Auto ETS fails in this case
        if __is_constant_time_series(train=data):
            return np.array([data.iloc[0]] * future_period)

        # counting NaN
        nullDataPoints = np.count_nonzero(np.isnan(data))
        percNull = nullDataPoints / history_period

        # If null data points are > 75%, then use average
        if percNull >= 0.75:
            avg = np.nansum(data) / history_period
            res = np.full(future_period, avg)
        else:
            # sktime auto ETS Produces an error if negatives/zeros are present in data, hence replace with a small number
            data[np.isnan(data)] = 0.01
            data[data <= 0] = 0.01
            if len(data) > 2 * frequency:
                forecaster = AutoETS(
                    auto=True,
                    additive_only=False if np.min(data) > 0 else True,
                    n_jobs=1,
                )
            else:
                forecaster = AutoETS(
                    auto=False,
                    n_jobs=1,
                )
            data.reset_index(drop=True, inplace=True)
            logger.info("data: {}".format(data))
            forecaster.fit(data.astype("float64"))
            fh = np.arange(1, future_period + 1)  # forecasting horizon
            res = forecaster.predict(fh=fh)
            res = res.to_numpy()
    except Exception as e:
        logger.exception(e)

    return res


def calculate_stat_profile(
    df,
    dimensions,
    first_n_periods_future,
    relevant_time_key,
    HistoryMeasure,
    OutputMeasure,
    time_level_col,
    frequency_period,
    FuturePeriod,
    UseMovingAverage,
    moving_avg_periods,
) -> pd.DataFrame:
    result = pd.DataFrame()
    logger.info("Executing for {}".format(df[dimensions].iloc[0].values))
    try:
        if len(df) == 0:
            return result
        # preparing input for ETS_forecast
        ETS_input = df[[HistoryMeasure, relevant_time_key]].copy()

        if UseMovingAverage:
            future_points = get_moving_avg_forecast(
                data=ETS_input[HistoryMeasure].to_numpy(),
                moving_avg_periods=moving_avg_periods,
                forecast_horizon=FuturePeriod,
            )
        else:
            future_points = get_ETS_forecast(
                ETS_input[HistoryMeasure],
                frequency=frequency_period,
                history_period=len(ETS_input),
                future_period=FuturePeriod,
            )

        logger.info("future_points : {}".format(future_points))

        result = pd.DataFrame(
            {
                time_level_col: first_n_periods_future,
                OutputMeasure: future_points,
            }
        )
        # set negative values to zero, autoETS produces negative values
        filter_clause = result[OutputMeasure] < 0
        result[OutputMeasure] = np.where(filter_clause, 0, result[OutputMeasure])

        for the_col in dimensions:
            result.insert(loc=0, column=the_col, value=df[the_col].iloc[0])
        logger.info("--- result : head------")
        logger.info(result.head())

    except Exception as e:
        logger.exception("Exception {} for {}".format(e, df[dimensions].iloc[0].values))
        result = pd.DataFrame()
    return result


def calculate_profiles(
    relevant_history,
    transition_dimensions,
    stat_dimensions,
    first_n_periods_future,
    relevant_time_key,
    HistoryMeasure,
    StatProfileMeasure,
    WeightedProfileMeasure,
    time_level_col,
    frequency_period,
    FuturePeriod,
    UseMovingAverage,
    L6MAvg_wt,
    LYHistory_wt,
    history_periods_based_on_gen_bucket,
    latest_time_name,
    relevant_time_mapping,
    time_attribute_dict,
    history_period,
    disagg_type,
):
    cols_required_stat = (
        stat_dimensions + transition_dimensions + [time_level_col, StatProfileMeasure]
    )
    stat_profile = pd.DataFrame(columns=cols_required_stat)

    cols_required_weighted = (
        stat_dimensions
        + transition_dimensions
        + [
            time_level_col,
            WeightedProfileMeasure,
        ]
    )
    weighted_profile = pd.DataFrame(columns=cols_required_weighted)

    if disagg_type == "Profile 1 (Profile Based)":
        # generate stat profile using the History Period mentioned
        stat_profile_list = Parallel(n_jobs=1, verbose=1)(
            delayed(calculate_stat_profile)(
                df=group,
                dimensions=transition_dimensions + stat_dimensions,
                first_n_periods_future=first_n_periods_future,
                relevant_time_key=relevant_time_key,
                HistoryMeasure=HistoryMeasure,
                OutputMeasure=StatProfileMeasure,
                time_level_col=time_level_col,
                frequency_period=frequency_period,
                FuturePeriod=FuturePeriod,
                UseMovingAverage=UseMovingAverage,
                moving_avg_periods=history_period,
            )
            for _, group in relevant_history.groupby(transition_dimensions)
        )

        stat_profile = concat_to_dataframe(stat_profile_list)

        # Check if profile at stat level is zero for any week
        stat_profile_contains_zeros = np.any(
            stat_profile.groupby(time_level_col).sum()[StatProfileMeasure].values == 0
        )

        if stat_profile_contains_zeros:
            logger.warning("Overriding UseMovingAverage to True ...")
            UseMovingAverage = True

            # iteratively increase history horizon and generate moving average forecast
            for the_history_period in history_periods_based_on_gen_bucket:
                logger.debug(
                    "Getting last {} period dates for history period ...".format(the_history_period)
                )
                last_n_periods_history = get_n_time_periods(
                    latest_time_name,
                    -the_history_period,
                    relevant_time_mapping,
                    time_attribute_dict,
                )

                # restrict data for the specified periods
                the_history = relevant_history[
                    relevant_history[time_level_col].isin(last_n_periods_history)
                ]
                if len(the_history) == 0:
                    continue

                # generate stat profile using the iterative history period
                stat_profile_list = Parallel(n_jobs=1, verbose=1)(
                    delayed(calculate_stat_profile)(
                        df=group,
                        dimensions=transition_dimensions + stat_dimensions,
                        first_n_periods_future=first_n_periods_future,
                        relevant_time_key=relevant_time_key,
                        HistoryMeasure=HistoryMeasure,
                        OutputMeasure=StatProfileMeasure,
                        time_level_col=time_level_col,
                        frequency_period=frequency_period,
                        FuturePeriod=FuturePeriod,
                        UseMovingAverage=UseMovingAverage,
                        moving_avg_periods=the_history_period,
                    )
                    for _, group in the_history.groupby(transition_dimensions)
                )

                stat_profile = concat_to_dataframe(stat_profile_list)

                # Check if profile at stat level is zero for any week
                stat_profile_contains_zeros = np.any(
                    stat_profile.groupby(time_level_col).sum()[StatProfileMeasure].values == 0
                )

                if len(stat_profile) == 0:
                    continue
                elif stat_profile_contains_zeros and the_history_period < history_period:
                    # increase the horizon and keep searching
                    continue
                else:
                    break

        stat_profile = stat_profile[cols_required_stat]

    if disagg_type == "Profile 2 (Historical Avg)":
        # calculate weighted profile
        weighted_profile_list = Parallel(n_jobs=1, verbose=1)(
            delayed(calculate_weighted_profile)(
                df=group,
                dimensions=transition_dimensions + stat_dimensions,
                first_n_periods_future=first_n_periods_future,
                FuturePeriod=FuturePeriod,
                frequency_period=frequency_period,
                HistoryMeasure=HistoryMeasure,
                OutputMeasure=WeightedProfileMeasure,
                time_level_col=time_level_col,
                L6MAvg_wt=L6MAvg_wt,
                LYHistory_wt=LYHistory_wt,
            )
            for _, group in relevant_history.groupby(transition_dimensions)
        )

        weighted_profile = concat_to_dataframe(weighted_profile_list)

        # Check if profile at stat level is zero for any week
        weighted_profile_contains_zeros = np.any(
            weighted_profile.groupby(time_level_col).sum()[WeightedProfileMeasure].values == 0
        )

        if weighted_profile_contains_zeros:
            # take values from stat profile itself assuming we have ensured stat profile is non zero
            weighted_profile = stat_profile.rename(
                columns={StatProfileMeasure: WeightedProfileMeasure}
            )

        weighted_profile = weighted_profile[cols_required_weighted]

    return stat_profile, weighted_profile


col_mapping = {
    "Stat Fcst Profile1 TL": float,
    "Stat Fcst Profile2 TL": float,
    "Stat Fcst Profile3 TL": float,
    "Stat Fcst Final Profile TL": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    History,
    CurrentTimePeriod,
    TimeDimension,
    ForecastParameters,
    Grains,
    ForecastGenTimeBucket,
    StatBucketWeight,
    TItemDates,
    DisaggregationType,
    StatFcstL1ForFIPLIteration,
    ForecastIterationSelectionAtTransitionLevel,
    StatActual,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    StatGrains,
    HistoryPeriodsInWeeks,
    SellOutOffset=pd.DataFrame(),
    UseMovingAverage="False",
    multiprocessing_num_cores=4,
    ForecastIterationMasterData=pd.DataFrame(),
    default_mapping={},
    df_keys=None,
):
    try:
        OutputList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                History=History,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                Grains=Grains,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                TItemDates=TItemDates,
                DisaggregationType=DisaggregationType,
                StatFcstL1ForFIPLIteration=StatFcstL1ForFIPLIteration,
                ForecastIterationSelectionAtTransitionLevel=ForecastIterationSelectionAtTransitionLevel,
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
                HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
                UseMovingAverage=UseMovingAverage,
                multiprocessing_num_cores=multiprocessing_num_cores,
                ForecastIterationMasterData=ForecastIterationMasterData,
                SellOutOffset=SellOutOffset,
                the_iteration=the_iteration,
                default_mapping=default_mapping,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def get_relevant_dim_data(
    dimension: str,
    dimension_master: pd.DataFrame,
    user_stat_level: str,
    child_col: str,
    parent_col: str,
) -> pd.DataFrame:
    if dimension_master.empty:
        logger.warning(f"dimension_master is empty for {dimension}")
        return pd.DataFrame()

    # convert 'All' to 'All Location'
    mod_user_stat_level = add_dim_suffix(input=user_stat_level, dim=dimension)

    # create actual column name - 'Location.[All Location]'
    stat_col_name = dimension + ".[" + mod_user_stat_level + "]"

    # filter relevant data
    result = dimension_master[[child_col, stat_col_name]].drop_duplicates()

    # rename header
    result.rename(columns={stat_col_name, parent_col}, inplace=True)

    return result


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


def processIteration(
    History,
    CurrentTimePeriod,
    TimeDimension,
    ForecastParameters,
    Grains,
    ForecastGenTimeBucket,
    StatBucketWeight,
    TItemDates,
    DisaggregationType,
    StatFcstL1ForFIPLIteration,
    ForecastIterationSelectionAtTransitionLevel,
    StatActual,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    StatGrains,
    the_iteration,
    HistoryPeriodsInWeeks,
    UseMovingAverage="False",
    multiprocessing_num_cores=4,
    ForecastIterationMasterData=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    default_mapping={},
    df_keys=None,
):
    if df_keys is None:
        df_keys = {}
    plugin_name = "DP016TransitionLevelStat"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here

    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"

    partial_week_key_col = "Time.[PartialWeekKey]"
    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"

    partial_week_col = "Time.[Partial Week]"
    version_col = "Version.[Version Name]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    stat_bucket_weight_col = "Stat Bucket Weight"
    forecast_period_col = "Forecast Period"
    disco_date_col = "Disco Date"
    sell_out_offset_col = "Offset Period"

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")
    all_stat_grains = StatGrains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]
    all_stat_grains = [x.strip() for x in all_stat_grains]

    # combine grains to get granular level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))
    stat_dimensions = [str(x) for x in all_stat_grains if x != "NA" and x != ""]
    logger.info("stat_dimensions : {} ...".format(stat_dimensions))
    cols_required_in_output_df = (
        [version_col]
        + dimensions
        + [
            partial_week_col,
            o9Constants.STAT_FCST_PROFILE1_TL,
            o9Constants.STAT_FCST_PROFILE2_TL,
            o9Constants.STAT_FCST_PROFILE3_TL,
            o9Constants.STAT_FCST_FINAL_PROFILE_TL,
        ]
    )
    output_df = pd.DataFrame(columns=cols_required_in_output_df)
    try:
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return output_df

        history_measure = input_stream
        if isinstance(History, dict):
            df = next(
                (df for key, df in History.items() if input_stream in df.columns),
                None,
            )
            if df is None:
                logger.warning(
                    f"Input Stream '{input_stream}' not found, returning empty dataframe..."
                )
                return output_df
            History_df = df
        else:
            History_df = History
        if History_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return output_df
        if DisaggregationType.empty:
            logger.warning("DisaggregationType is None/Empty for slice : {}...".format(df_keys))
            return output_df

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

        # drop nas
        History_df = History_df[History_df[history_measure].notna()]

        merge_grains = [
            o9Constants.VERSION_NAME,
            o9Constants.PLANNING_ACCOUNT,
            o9Constants.PLANNING_CHANNEL,
            o9Constants.PLANNING_REGION,
            o9Constants.PLANNING_PNL,
            o9Constants.TRANSITION_DEMAND_DOMAIN,
            o9Constants.LOCATION,
            o9Constants.TRANSITION_ITEM,
        ]
        history_dims = [col.split(".")[0] for col in History_df]
        target_dims = [dim.split(".")[0] for dim in merge_grains]
        missing_dims = list(set(target_dims) - set(history_dims))
        missing_grains = list(set(dimensions) - set(History_df.columns))

        # if the entire dimension is missing from the input data, we fill the dim with the default values
        missing_dim_grains = []
        if len(missing_dims) > 0:
            for dim in missing_dims:
                missing_dim_grains += [col for col in missing_grains if col.split(".")[0] == dim]
        if len(missing_dim_grains) > 0:
            History_df = check_all_dimensions(
                df=History_df,
                grains=missing_dim_grains,
                default_mapping=default_mapping,
            )
        missing_grains = list(set(missing_grains + merge_grains) - set(History_df.columns))

        # add stat grains to history
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
                    col for col in History_df.columns if col.split(".")[0] == dim_of_missing_grain
                ][0]
                History_df = History_df.merge(
                    master_df[[existing_grain, grain]],
                    on=existing_grain,
                    how="inner",
                )

        # join history with forecast iteration selection to filter out relevant intersections
        History_df = History_df.merge(
            ForecastIterationSelectionAtTransitionLevel,
            on=merge_grains,
            how="inner",
        )

        # check for empty data
        if len(History_df) == 0:
            logger.warning("Input is None/Empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return output_df

        level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
        for the_col in level_cols:

            # extract 'Item' from 'Item Level'
            the_dim = the_col.split(" Level")[0]

            logger.debug(f"the_dim : {the_dim}")

            # all dims exception location will be planning location
            if the_dim in ["Item", "Demand Domain"]:
                the_child_col = the_dim + ".[Transition " + the_dim + "]"
            else:
                the_child_col = the_dim + ".[Planning " + the_dim + "]"

            logger.debug(f"the_planning_col : {the_child_col}")

            # Item.[Stat Item]
            the_stat_col = the_dim + ".[Stat " + the_dim + "]"
            logger.debug(f"the_stat_col : {the_stat_col}")

            the_dim_data = master_data_dict[the_dim]

            # Eg. the_level = Item.[L2]
            the_level = (
                the_dim
                + ".["
                + add_dim_suffix(input=ForecastLevelData[the_col].iloc[0], dim=the_dim)
                + "]"
            )
            logger.debug(f"the_level : {the_level}")

            # copy values from L2 to Stat Item
            the_dim_data[the_stat_col] = the_dim_data[the_level]

            # select only relevant columns
            the_dim_data = the_dim_data[[the_child_col, the_stat_col]].drop_duplicates()

            # join with Actual
            History_df = History_df.merge(the_dim_data, on=the_child_col, how="inner")

            logger.debug("-------------------------")

        cols_req_in_History = (
            [version_col] + dimensions + stat_dimensions + [partial_week_col, history_measure]
        )

        History_df = History_df[cols_req_in_History]
        stat_dimensions = [x for x in stat_dimensions if x.split(".")[0] not in missing_dims]

        # restrict intersections to the ones in Stat Actual
        History_df = History_df.merge(
            StatActual[stat_dimensions].drop_duplicates(),
            on=stat_dimensions,
            how="inner",
        )
        if History_df.empty:
            logger.warning("History is empty, cannot process further ...")
            return output_df

        disagg_type = DisaggregationType[o9Constants.DISAGGREGATION_TYPE].iloc[0]
        logger.debug(f"disagg_type : {disagg_type}")

        if disagg_type == "No Disaggregation":
            return output_df

        if disagg_type == "Bottom-up Forecast":

            # Check if values are available
            if StatFcstL1ForFIPLIteration.empty:
                logger.warning(
                    f"StatFcstL1ForFIPLIteration is empty, cannot calculate profiles for {disagg_type}"
                )
                return output_df

            # rename headers to match the output
            rename_mapping = {
                o9Constants.STAT_REGION: o9Constants.PLANNING_REGION,
                o9Constants.STAT_ITEM: o9Constants.TRANSITION_ITEM,
                o9Constants.STAT_PNL: o9Constants.PLANNING_PNL,
                o9Constants.STAT_LOCATION: o9Constants.PLANNING_LOCATION,
                o9Constants.STAT_DEMAND_DOMAIN: o9Constants.TRANSITION_DEMAND_DOMAIN,
                o9Constants.STAT_ACCOUNT: o9Constants.PLANNING_ACCOUNT,
                o9Constants.STAT_CHANNEL: o9Constants.PLANNING_CHANNEL,
                o9Constants.STAT_FCST_L1: o9Constants.STAT_FCST_PROFILE3_TL,
            }

            HistoryCombinations = History_df.copy()

            HistoryCombinations.drop([partial_week_col, history_measure], axis=1, inplace=True)

            HistoryCombinations.drop_duplicates(inplace=True)

            output_df = StatFcstL1ForFIPLIteration.rename(columns=rename_mapping)

            output_df = output_df.merge(
                HistoryCombinations,
                on=(dimensions + [version_col]),
                how="inner",
            )

            output_df[o9Constants.STAT_FCST_FINAL_PROFILE_TL] = output_df[
                o9Constants.STAT_FCST_PROFILE3_TL
            ]

            # Add other profiles with null values
            output_df[o9Constants.STAT_FCST_PROFILE1_TL] = np.nan
            output_df[o9Constants.STAT_FCST_PROFILE2_TL] = np.nan

            output_df = output_df[cols_required_in_output_df]

            return output_df

        # assert and convert string value to boolean
        assert UseMovingAverage in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(UseMovingAverage)
        UseMovingAverage = eval(UseMovingAverage)

        assert len(dimensions) > 0, "dimensions cannot be empty ..."

        # capping negatives for entire history
        filter_history_clause = History_df[history_measure] < 0

        History_df[history_measure] = np.where(
            filter_history_clause, 0, History_df[history_measure]
        )

        # extracting version name
        input_version = History_df[version_col].iloc[0]

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return output_df

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        # Default config - same for Planning Month and Month
        frequency_period = 12  # 'Q' - Quarterly, 'M' - Monthly, 'W' - Weekly
        L6MAvg_wt = 0.2
        LYProfileForecast_wt = 0.8

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col
            frequency_period = 52  # 'Q' - Quarterly, 'M' - Monthly, 'W' - Weekly
            L6MAvg_wt = 0.4
            LYProfileForecast_wt = 0.6

        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col

        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                planning_quarter_col,
                planning_quarter_key_col,
            ]
            relevant_time_name = planning_quarter_col
            relevant_time_key = planning_quarter_key_col
            frequency_period = 4  # 'Q' - Quarterly, 'M' - Monthly, 'W' - Weekly
            L6MAvg_wt = 0.2
            LYProfileForecast_wt = 0.8
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                quarter_col,
                quarter_key_col,
            ]
            relevant_time_name = quarter_col
            relevant_time_key = quarter_key_col
            frequency_period = 4  # 'Q' - Quarterly, 'M' - Monthly, 'W' - Weekly
            L6MAvg_wt = 0.2
            LYProfileForecast_wt = 0.8
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return output_df

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        default_disco_date = TimeDimension[partial_week_key_col].max()

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        partial_week_mapping = TimeDimension[
            [partial_week_col, partial_week_key_col]
        ].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Join Actuals with time mapping
        History_df = History_df.merge(base_time_mapping, on=partial_week_col, how="inner")

        # select the relevant columns, groupby and sum history measure
        History_df = (
            History_df.groupby(dimensions + stat_dimensions + [relevant_time_name])
            .sum()[[history_measure]]
            .reset_index()
        )

        logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

        HistoryPeriods = HistoryPeriodsInWeeks.split(",")
        HistoryPeriods = sorted([int(x) for x in HistoryPeriods])

        if len(HistoryPeriods) == 0:
            logger.warning("HistoryPeriods is empty, check HistoryPeriodsInWeeks input ...")
            logger.warning("using default value 13 ...")
            HistoryPeriods = [13]

        FuturePeriod = int(ForecastParameters[forecast_period_col].unique()[0])
        logger.info(f"FuturePeriod : {FuturePeriod}")

        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        time_attribute_dict = {relevant_time_name: relevant_time_key}

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

        if relevant_time_name == week_col:
            history_periods_based_on_gen_bucket = HistoryPeriods
        elif relevant_time_name == quarter_col or relevant_time_name == planning_quarter_col:
            history_periods_based_on_gen_bucket = [
                int(round(x * 4 / 52, 0)) for x in HistoryPeriods
            ]
        elif relevant_time_name == month_col or relevant_time_name == planning_month_col:
            history_periods_based_on_gen_bucket = [
                int(round(x * 12 / 52, 0)) for x in HistoryPeriods
            ]
        else:
            logger.warning(
                f"Unknown relevant_time_name {relevant_time_name}, using default history periods {HistoryPeriods}"
            )
            history_periods_based_on_gen_bucket = HistoryPeriods

        HistoryPeriod = max(history_periods_based_on_gen_bucket)
        logger.info(f"HistoryPeriod : {HistoryPeriod}")

        logger.info(f"latest_time_name after offset {offset_periods} : {latest_time_name} ...")

        first_n_periods_future = get_n_time_periods(
            latest_time_name,
            FuturePeriod + offset_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        last_n_time_periods = get_n_time_periods(
            latest_time_name,
            -HistoryPeriod,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        # join to get relevant time key
        History_df = History_df.merge(relevant_time_mapping, on=relevant_time_name, how="inner")

        # filter out future data
        filter_clause = History_df[relevant_time_key] < CurrentTimePeriod[relevant_time_key].iloc[0]
        History_df = History_df[filter_clause]

        relevant_history_nas_filled = fill_missing_dates(
            actual=History_df.drop(relevant_time_key, axis=1),
            forecast_level=dimensions + stat_dimensions,
            time_mapping=relevant_time_mapping,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_time_periods,
            fill_nulls_with_zero=True,
        )

        logger.info("Generating forecast for all intersections ...")

        # sort data by relevant time key
        relevant_history_nas_filled.sort_values(
            dimensions + stat_dimensions + [relevant_time_key], inplace=True
        )

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(calculate_profiles)(
                relevant_history=group,
                transition_dimensions=dimensions,
                stat_dimensions=stat_dimensions,
                first_n_periods_future=first_n_periods_future,
                relevant_time_key=relevant_time_key,
                HistoryMeasure=history_measure,
                StatProfileMeasure=o9Constants.STAT_FCST_PROFILE1_TL,
                WeightedProfileMeasure=o9Constants.STAT_FCST_PROFILE2_TL,
                time_level_col=relevant_time_name,
                frequency_period=frequency_period,
                FuturePeriod=FuturePeriod + offset_periods,
                UseMovingAverage=UseMovingAverage,
                L6MAvg_wt=L6MAvg_wt,
                LYHistory_wt=LYProfileForecast_wt,
                history_periods_based_on_gen_bucket=history_periods_based_on_gen_bucket,
                latest_time_name=latest_time_name,
                relevant_time_mapping=relevant_time_mapping,
                time_attribute_dict=time_attribute_dict,
                history_period=HistoryPeriod,
                disagg_type=disagg_type,
            )
            for name, group in relevant_history_nas_filled.groupby(stat_dimensions)
        )

        # collect stat and weighted profiles into separate lists
        output_profile1_list = [x[0] for x in all_results]
        output_profile2_list = [x[1] for x in all_results]

        # Concatenate all results to one dataframe
        output_profile1 = concat_to_dataframe(output_profile1_list)

        if output_profile1.empty:
            output_profile1 = pd.DataFrame(
                columns=dimensions + [relevant_time_name, o9Constants.STAT_FCST_PROFILE1_TL]
            )

        logger.info("------ output_profile1 : head -----------")
        logger.info(output_profile1.head())

        # Concatenate all results to one dataframe
        output_profile2 = concat_to_dataframe(output_profile2_list)

        if output_profile2.empty:
            output_profile2 = pd.DataFrame(
                columns=dimensions + [relevant_time_name, o9Constants.STAT_FCST_PROFILE2_TL]
            )

        logger.info("------ output_profile2 : head -----------")
        logger.info(output_profile2.head())

        # merge output_profile1 and output_profile2
        # LS requires single output for writing in one measure group
        common_cols_for_merge = dimensions + [relevant_time_name]

        output_df = output_profile1.merge(output_profile2, on=common_cols_for_merge, how="outer")

        # Add input version
        output_df.insert(loc=0, column=version_col, value=input_version)

        relevant_dates = first_n_periods_future[offset_periods:]
        relevant_dates = pd.DataFrame({relevant_time_name: relevant_dates})
        output_df = output_df.merge(
            relevant_dates, on=[relevant_time_name], how="inner"
        ).drop_duplicates()

        cols_to_disaggregate = [
            o9Constants.STAT_FCST_PROFILE1_TL,
            o9Constants.STAT_FCST_PROFILE2_TL,
        ]

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )

        # perform disaggregation
        output_df = disaggregate_data(
            source_df=output_df,
            source_grain=relevant_time_name,
            target_grain=partial_week_col,
            profile_df=StatBucketWeight.drop(version_col, axis=1),
            profile_col=stat_bucket_weight_col,
            cols_to_disaggregate=cols_to_disaggregate,
        )
        for the_col in cols_to_disaggregate:
            if output_df[the_col].isnull().all():
                pass
            else:
                output_df[the_col] = output_df[the_col].round(2)
        # getting partial week key
        output_df = output_df.merge(partial_week_mapping)
        output_df = output_df.merge(
            ItemMasterData[
                [o9Constants.TRANSITION_ITEM, o9Constants.PLANNING_ITEM]
            ].drop_duplicates(),
            on=o9Constants.TRANSITION_ITEM,
            how="inner",
        )
        output_df = output_df.merge(TItemDates, how="left")
        output_df[disco_date_col].fillna(default_disco_date, inplace=True)

        output_df.sort_values(by=disco_date_col, ascending=False, inplace=True)
        output_df = output_df.drop_duplicates(
            subset=stat_dimensions + dimensions + [version_col, partial_week_col]
        )

        # drop the dates after disco date
        filter_clause = output_df[partial_week_key_col] > output_df[disco_date_col]
        output_df = output_df[~filter_clause]

        # Copy to final profile
        if disagg_type == "Profile 1 (Profile Based)":
            output_df[o9Constants.STAT_FCST_FINAL_PROFILE_TL] = output_df[
                o9Constants.STAT_FCST_PROFILE1_TL
            ]
        else:
            output_df[o9Constants.STAT_FCST_FINAL_PROFILE_TL] = output_df[
                o9Constants.STAT_FCST_PROFILE2_TL
            ]

        # preserve all profiles to keep schema consistent
        output_df[o9Constants.STAT_FCST_PROFILE3_TL] = np.nan

        output_df = output_df[cols_required_in_output_df]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        output_df = pd.DataFrame(columns=cols_required_in_output_df)

    return output_df

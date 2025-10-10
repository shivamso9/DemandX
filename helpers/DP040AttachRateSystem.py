import logging

import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    filter_relevant_time_mapping,
    get_last_time_period,
    get_n_time_periods,
    get_relevant_time_name_and_key,
)
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.get_moving_avg_forecast import get_moving_avg_forecast

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None
logger = logging.getLogger("o9_logger")


def calculate_sliding_window_forecast(
    df,
    MAPeriod,
    FuturePeriods,
    test_dates,
    dimensions,
    HistoryMeasure,
    relevant_time_key,
    time_level_col,
) -> pd.DataFrame:
    result = pd.DataFrame()
    try:
        if len(df) == 0:
            return result

        df.sort_values(relevant_time_key, inplace=True)
        # get moving average forecast
        the_forecast = get_moving_avg_forecast(
            data=df[HistoryMeasure].to_numpy(),
            moving_avg_periods=MAPeriod,
            forecast_horizon=FuturePeriods,
        )

        result = pd.DataFrame({time_level_col: test_dates, HistoryMeasure: the_forecast})
        for the_col in dimensions:
            result.insert(0, the_col, df[the_col].iloc[0])
    except Exception as e:
        logger.exception(f"Exception {e} for {df[dimensions].iloc[0].values}")
    return result


col_mapping = {"810 Attach Rate Planning.[Attach Rate System]": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    MovingAvgPeriod,
    attach_rate,
    CurrentTimePeriod,
    TimeDimension,
    df_keys,
    multiprocessing_num_cores=4,
):
    plugin_name = "DP040AttachRateSystem"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here
    version_col = "Version.[Version Name]"
    from_item_col = "from.[Item].[Planning Item]"
    to_item_col = "to.[Item].[Planning Item]"
    to_partial_week_col = "to.[Time].[Partial Week]"
    attach_rate_sys_col = "810 Attach Rate Planning.[Attach Rate System]"

    cols_required_in_output = [
        version_col,
        from_item_col,
        to_item_col,
        to_partial_week_col,
        attach_rate_sys_col,
    ]
    MovingAvg = pd.DataFrame(columns=cols_required_in_output)
    try:
        # Configurables - define all column names here
        time_week_col = "Time.[Week]"
        time_week_key_col = "Time.[WeekKey]"
        time_partial_week_col = "Time.[Partial Week]"
        time_partial_week_key_col = "Time.[PartialWeekKey]"
        to_week_col = "to.[Time].[Week]"
        week_name = "WeekName"
        week_key = "WeekKey"
        partial_week_name = "PartialWeekName"
        partial_week_key = "PartialWeekKey"

        frequency = "Weekly"
        logger.debug(f"frequency : {frequency}")
        HistoryPeriods = 52
        logger.debug(f"HistoryPeriods : {HistoryPeriods}")
        MovingAvgPeriod = int(MovingAvgPeriod)

        logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

        # attach rate might not be present for a particular slice, check and return empty dataframe
        if attach_rate is None or len(attach_rate) == 0:
            logger.warning("Attach Rate is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return MovingAvg

        input_version = attach_rate[version_col].iloc[0]

        req_cols = [
            from_item_col,
            to_item_col,
            to_week_col,
            attach_rate_sys_col,
        ]

        attach_rate = attach_rate[req_cols]

        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            time_week_col,
            time_week_key_col,
        )

        dimensions = [from_item_col, to_item_col]

        logger.info("Creating time mapping ...")

        TimeDimension = TimeDimension.rename(
            columns={
                time_week_col: week_name,
                time_week_key_col: week_key,
                time_partial_week_col: partial_week_name,
                time_partial_week_key_col: partial_week_key,
            }
        )
        logger.info("time mapping head :")
        logger.info(TimeDimension.head())

        if len(TimeDimension) == 0:
            logger.warning("Time Dimension cannot be empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return MovingAvg

        # Filter only relevant columns based on frequency
        relevant_time_mapping = filter_relevant_time_mapping(frequency, TimeDimension)

        # Get relevant time name and key based on frequency
        relevant_time_name, relevant_time_key = get_relevant_time_name_and_key(frequency)

        # convert to datetime
        TimeDimension[week_key] = TimeDimension[week_key].apply(
            pd.to_datetime, infer_datetime_format=True
        )

        TimeDimension[partial_week_key] = TimeDimension[partial_week_key].apply(
            pd.to_datetime, infer_datetime_format=True
        )

        relevant_time_mapping.sort_values(relevant_time_key, inplace=True)
        relevant_time_mapping.reset_index(drop=True, inplace=True)

        # get index of latest value
        index_of_latest_value = relevant_time_mapping[
            relevant_time_mapping[relevant_time_name] == latest_time_name
        ].index[0]

        # get forward looking dates from current time
        start_index = index_of_latest_value + 1

        future_time_mapping = relevant_time_mapping[start_index:]

        future_periods = list(future_time_mapping[relevant_time_name])

        future_periods_count = len(future_periods)

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # get last n periods
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -HistoryPeriods,
            TimeDimension,
            time_attribute_dict,
            include_latest_value=True,
        )

        # Copy TimeLevel into relevant_time_name to make fill missing dates work
        attach_rate[relevant_time_name] = attach_rate[to_week_col]

        # Fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=attach_rate,
            forecast_level=dimensions,
            history_measure=attach_rate_sys_col,
            relevant_time_key=relevant_time_key,
            relevant_time_name=relevant_time_name,
            relevant_time_periods=last_n_periods,
            time_mapping=TimeDimension,
            fill_nulls_with_zero=True,
        )

        # NANs don't get filled in TimeLevel, drop the column
        relevant_history_nas_filled.drop(to_week_col, axis=1, inplace=True)

        # rename relevant time name to TimeLevel
        relevant_history_nas_filled.rename(columns={relevant_time_name: to_week_col}, inplace=True)

        logger.info("Calculating moving average for all intersections ...")
        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(calculate_sliding_window_forecast)(
                group,
                MovingAvgPeriod,
                future_periods_count,
                future_periods,
                dimensions,
                attach_rate_sys_col,
                relevant_time_key,
                to_week_col,
            )
            for name, group in relevant_history_nas_filled.groupby(dimensions)
        )

        # Concatenate all results to one dataframe
        MovingAvg = concat_to_dataframe(all_results)

        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=MovingAvg,
            forecast_level=dimensions,
        )

        if len(MovingAvg) == 0:
            logger.info("No data found, returning empty dataframe ...")
            MovingAvg = pd.DataFrame(columns=cols_required_in_output)
        else:
            MovingAvg = MovingAvg.merge(TimeDimension, left_on=to_week_col, right_on=week_name)
            MovingAvg[to_partial_week_col] = MovingAvg[partial_week_name]

            # Add input version
            MovingAvg.insert(0, version_col, input_version)

        MovingAvg = MovingAvg[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(
            f"Exception {e} for slice : {df_keys}, returning empty dataframe as output ..."
        )
        MovingAvg = pd.DataFrame(columns=cols_required_in_output)

    return MovingAvg

from logging import getLogger

from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    filter_relevant_time_mapping,
    get_last_time_period,
    get_n_time_periods,
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
from pandas import DataFrame, options, set_option, to_datetime

options.display.max_rows = 25
options.display.max_columns = 50
options.display.max_colwidth = 100
set_option("display.width", 1000)
options.display.precision = 3
options.mode.chained_assignment = None
logger = getLogger("o9_logger")


def calculate_sliding_window_forecast(
    df,
    MAPeriod,
    FuturePeriods,
    test_dates,
    dimensions,
    HistoryMeasure,
    relevant_time_key,
    time_level_col,
) -> DataFrame:
    result = DataFrame()
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

        result = DataFrame({time_level_col: test_dates, HistoryMeasure: the_forecast})
        for the_col in dimensions:
            result.insert(0, the_col, df[the_col].iloc[0])
    except Exception as e:
        logger.error("Exception for {}".format(df[dimensions].iloc[0].values), "error")
        logger.exception(e)
    return result


col_mapping = {"Sell In Stat L0": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    MovingAvgPeriods,
    FuturePeriods,
    HistoryMeasure,
    Grains,
    TimeLevel,
    all_history_df,
    CurrentTimePeriod,
    TimeDimension,
    df_keys,
    OutputTimeLevel,
    multiprocessing_num_cores=4,
) -> DataFrame:
    plugin_name = "DP019MovingAverageSCPFcst"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # Configurables
    version_col = "Version.[Version Name]"
    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    # output measure
    # the output is written in input measure itself for future dates
    cols_required_in_output_df = [version_col] + dimensions + [TimeLevel, HistoryMeasure]

    # Define an empty output dataframe
    empOutputData = DataFrame(columns=cols_required_in_output_df)
    # Replace the time attribute with the required time attribute in output data
    empOutputData.rename(columns={TimeLevel: OutputTimeLevel}, inplace=True)

    try:
        if TimeLevel == "Time.[Planning Month]":
            relevant_time_name = "MonthName"
            relevant_time_key = "MonthKey"
            TimeLevel_key = "Time.[PlanningMonthKey]"
            frequency = "Monthly"
        elif TimeLevel == "Time.[Week]":
            relevant_time_name = "WeekName"
            relevant_time_key = "WeekKey"
            TimeLevel_key = "Time.[WeekKey]"
            frequency = "Weekly"
        else:
            raise ValueError(
                "Unknown TimeLevel {}, Time.[Planning Month] and Time.[Week] are only supported ...".format(
                    TimeLevel
                )
            )
        logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

        MovingAvgPeriods = int(MovingAvgPeriods)
        FuturePeriods = int(FuturePeriods)

        logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))
        logger.info("FuturePeriods : {} ...".format(FuturePeriods))

        if len(all_history_df) == 0:
            logger.warning("No records present in history for slice : {}...".format(df_keys))
            logger.warning("Will return empty dataframe for this slice ...")
            return empOutputData
        if len(dimensions) == 0:
            logger.warning(
                "Dimensions cannot be empty, check Grain configuration for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning("Will return empty dataframe for this slice ...")
            return empOutputData
        logger.info("Creating time mapping ...")
        # Process Time data to generate time mapping.
        time_mapping = TimeDimension.copy()
        time_mapping.rename(
            columns={
                "Time.[Day]": "DayName",
                "Time.[DayKey]": "DayKey",
                "Time.[Week]": "WeekName",
                "Time.[WeekKey]": "WeekKey",
                "Time.[Month]": "MonthName",
                "Time.[MonthKey]": "MonthKey",
            },
            inplace=True,
        )
        time_mapping = filter_relevant_time_mapping(frequency=frequency, time_mapping=time_mapping)
        time_mapping[relevant_time_key] = to_datetime(time_mapping[relevant_time_key])
        time_attribute_dict = {relevant_time_name: relevant_time_key}
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            TimeLevel,
            TimeLevel_key,
        )

        logger.info("latest_time_name : {} ...".format(latest_time_name))
        # get last n periods
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -MovingAvgPeriods,
            time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )
        # get test period dates
        test_dates = get_n_time_periods(
            latest_time_name,
            FuturePeriods,
            time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )
        logger.info("test_dates : {} ...".format(test_dates))
        # Filter out relevant history
        relevant_history = all_history_df[all_history_df[TimeLevel].isin(last_n_periods)]
        if len(relevant_history) == 0:
            logger.warning(
                "No records present in history, after filtering data for last {} periods for slice {}...".format(
                    MovingAvgPeriods, df_keys
                )
            )
            logger.warning("Will return empty dataframe for this slice ...")
            return empOutputData

        relevant_history.drop(version_col, axis=1, inplace=True)

        # Copy TimeLevel into relevant_time_name to make fill missing dates work
        relevant_history[relevant_time_name] = relevant_history[TimeLevel]

        # Fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=relevant_history,
            forecast_level=dimensions,
            history_measure=HistoryMeasure,
            relevant_time_key=relevant_time_key,
            relevant_time_name=relevant_time_name,
            relevant_time_periods=last_n_periods,
            time_mapping=time_mapping,
            fill_nulls_with_zero=True,
        )
        # NANs don't get filled in TimeLevel, drop the column
        relevant_history_nas_filled.drop(TimeLevel, axis=1, inplace=True)

        # rename relevant time name to TimeLevel
        relevant_history_nas_filled.rename(columns={relevant_time_name: TimeLevel}, inplace=True)

        logger.info("Calculating moving average forecast for all intersections ...")
        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(calculate_sliding_window_forecast)(
                group,
                MovingAvgPeriods,
                FuturePeriods,
                test_dates,
                dimensions,
                HistoryMeasure,
                relevant_time_key,
                TimeLevel,
            )
            for name, group in relevant_history_nas_filled.groupby(dimensions)
        )

        # Concatenate all results to one dataframe
        outputData = concat_to_dataframe(all_results)
        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=outputData,
            forecast_level=dimensions,
        )
        # Add input version
        outputData.insert(0, version_col, all_history_df[version_col].iloc[0])

        # ADD PARTIAL WEEK
        if TimeLevel != OutputTimeLevel:
            outputData = outputData.merge(
                TimeDimension[[TimeLevel, OutputTimeLevel]].drop_duplicates(),
                on=TimeLevel,
                how="left",
            )
            if (len(outputData.index)) > 0:
                # Split HistoryMeasure equally
                outputData[HistoryMeasure] = outputData[HistoryMeasure] / outputData.groupby(
                    cols_required_in_output_df
                )[TimeLevel].transform(len)
            cols_required_in_output_df.remove(TimeLevel)
            cols_required_in_output_df.insert(-1, OutputTimeLevel)
            outputData = outputData[cols_required_in_output_df]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.error("Exception for slice : {}".format(df_keys))
        logger.exception(e)
        outputData = empOutputData

    return outputData

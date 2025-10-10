import logging

import numpy as np
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

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {"Bestfit Calculation Flag": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ForecastGenTimeBucket,
    BestFitCalcFrequency,
    TimeDimension,
    BestFitFrequencyStartDate,
    CurrentDay,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                BestFitCalcFrequency=BestFitCalcFrequency,
                TimeDimension=TimeDimension,
                BestFitFrequencyStartDate=BestFitFrequencyStartDate,
                CurrentDay=CurrentDay,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    ForecastGenTimeBucket,
    BestFitCalcFrequency,
    TimeDimension,
    BestFitFrequencyStartDate,
    CurrentDay,
    df_keys,
):
    plugin_name = "DP058GenerateBestFitCalculationFlag"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    version_col = "Version.[Version Name]"
    time_day_col = "Time.[Day]"
    segmentation_lob_col = "Item.[Segmentation LOB]"
    bestfit_calc_flag_col = "Bestfit Calculation Flag"
    time_day_key_col = "Time.[DayKey]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    first_day_col = "First Day in Bucket"
    best_fit_calc_freq_col = "Bestfit Calculation Frequency"
    remainder_col = "Remainder"
    bestfit_freq_start_date_col = "Bestfit Frequency Start Date"

    cols_required_in_output = [
        version_col,
        segmentation_lob_col,
        time_day_col,
        bestfit_calc_flag_col,
    ]
    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        if len(ForecastGenTimeBucket) == 0:
            logger.warning(
                "ForecastGenTimeBucket is empty, returning without further execution ..."
            )
            return Output

        if len(BestFitCalcFrequency) == 0:
            logger.warning("BestFitCalcFrequency is empty, returning without further execution ...")
            return Output

        if len(CurrentDay) == 0:
            logger.warning("CurrentDay is empty, returning without further execution ...")
            return Output

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning without further execution ...")
            return Output

        # infer forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]

        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        # create column names depending on fcst gen time bucket
        relevant_time_col = "".join(["Time.[", fcst_gen_time_bucket.strip(""), "]"])
        relevant_time_key_col = "".join(["Time.[", fcst_gen_time_bucket.replace(" ", ""), "Key]"])

        logger.debug(f"relevant_time_col : {relevant_time_col}")
        logger.debug(f"relevant_time_key_col : {relevant_time_key_col}")

        TimeDimension = TimeDimension[
            [
                time_day_col,
                time_day_key_col,
                relevant_time_col,
                relevant_time_key_col,
            ]
        ]

        logger.debug(f"BestFitFrequencyStartDate\n{BestFitFrequencyStartDate.to_csv()}")
        logger.debug(f"CurrentDay\n{CurrentDay.to_csv()}")

        BestFitCalcFrequency.drop(version_col, axis=1, inplace=True)
        BestFitCalcFrequency[best_fit_calc_freq_col] = BestFitCalcFrequency[
            best_fit_calc_freq_col
        ].astype("int")

        # join with best fit start dates, left join to retain all records in BestFitCalcFrequency whether start date exists or not
        BestFitCalcFrequency = BestFitCalcFrequency.merge(
            BestFitFrequencyStartDate.drop(version_col, axis=1),
            on=segmentation_lob_col,
            how="left",
        )

        # fill nas with current daykey
        current_day_key = CurrentDay[time_day_key_col].unique()[0]
        BestFitCalcFrequency[bestfit_freq_start_date_col].fillna(current_day_key, inplace=True)

        logger.debug(f"current_day_key : {current_day_key}")
        input_version = ForecastGenTimeBucket[version_col].unique()[0]
        logger.debug(f"input_version : {input_version}")

        all_output = list()
        for (
            the_lob,
            the_best_fit_frequency,
            the_best_fit_start_date,
        ) in BestFitCalcFrequency.itertuples(index=False):
            logger.debug(f"--- the_lob : {the_lob}")
            logger.debug(f"--- the_best_fit_frequency : {the_best_fit_frequency}")
            logger.debug(f"--- the_best_fit_start_date : {the_best_fit_start_date}")

            # get relevant time bucket key corresponding to best fit start date
            the_date_filter = TimeDimension[time_day_key_col] == the_best_fit_start_date
            the_day = TimeDimension[the_date_filter]

            # filter clause
            the_filter_clause = (
                TimeDimension[relevant_time_key_col] >= the_day[relevant_time_key_col].unique()[0]
            )
            the_future_time_dim = TimeDimension[the_filter_clause]

            # mark the first day of every time bucket
            the_future_time_dim[first_day_col] = the_future_time_dim.groupby(relevant_time_col)[
                time_day_key_col
            ].rank(method="dense", ascending=True)

            # filter only the first days in all buckets
            the_filter_clause = the_future_time_dim[first_day_col] == 1
            the_relevant_time_dim = the_future_time_dim[the_filter_clause]

            # create rank 0 to n based after sorting - to divide by best fit frequency later
            the_relevant_time_dim.sort_values(time_day_key_col, inplace=True)
            the_relevant_time_dim.reset_index(drop=True, inplace=True)

            # copy time dim df and populate version, lob
            the_output = the_relevant_time_dim.copy()
            the_output[version_col] = input_version
            the_output[segmentation_lob_col] = the_lob

            # divide the index by best fit frequency to populate a remainder
            the_output[remainder_col] = the_output.index % the_best_fit_frequency

            # rows where remainder is zero are the buckets where best fit flag is to be set as 1
            the_output[bestfit_calc_flag_col] = np.where(the_output[remainder_col] == 0, 1, 0)

            the_output = the_output[cols_required_in_output]
            logger.debug(f"--- the_output\n{the_output.head().to_csv()}")
            logger.debug(f"--- the_output.shape : {the_output.shape}")
            all_output.append(the_output)

        Output = concat_to_dataframe(all_output)
        if len(Output) == 0:
            Output = pd.DataFrame(columns=cols_required_in_output)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
    return Output

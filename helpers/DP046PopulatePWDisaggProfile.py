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

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def make_valid_time_dim(the_string: str) -> str:
    return f"Time.[{the_string}]"


col_mapping = {"Stat Bucket Weight": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    TimeDimension,
    ForecastConfiguration,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in ForecastConfiguration[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                TimeDimension=TimeDimension,
                ForecastConfiguration=ForecastConfiguration,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    TimeDimension,
    ForecastConfiguration,
    df_keys,
):
    plugin_name = "DP046PopulatePWDisaggProfile"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    day_col = "Time.[Day]"
    partial_week_col = "Time.[Partial Week]"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    fcst_storage_time_bucket_col = "Forecast Storage Time Bucket"
    stat_bucket_weight_col = "Stat Bucket Weight"
    num_days_col = "Number of Days"
    total_num_days_col = "Total Number of Days "
    every_partial_week = "Every Partial Week"
    last_partial_week_of_week = "Last Partial Week of Week"
    last_partial_week_of_month = "Last Partial Week of Month"
    last_partial_week_of_planning_month = "Last Partial Week of Planning Month"

    cols_required_in_output = [
        version_col,
        partial_week_col,
        stat_bucket_weight_col,
    ]
    PWProfile = pd.DataFrame(columns=cols_required_in_output)
    try:
        if ForecastConfiguration.empty:
            logger.warning(
                f"Please retrigger the plugin after setting the measures {fcst_gen_time_bucket_col} and {fcst_storage_time_bucket_col} ..."
            )
            return PWProfile

        if TimeDimension.empty:
            logger.warning("Please retrigger the plugin after populating time master data ...")
            return PWProfile

        # Extract values in configuration
        fcst_gen_time_bucket = ForecastConfiguration[fcst_gen_time_bucket_col].unique()[0]
        fcst_storage_time_bucket = ForecastConfiguration[fcst_storage_time_bucket_col].unique()[0]

        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")
        logger.debug(f"fcst_storage_time_bucket : {fcst_storage_time_bucket}")

        # Formatting
        fcst_gen_time_bucket = make_valid_time_dim(fcst_gen_time_bucket)

        input_version = ForecastConfiguration[version_col].unique()[0]

        if fcst_storage_time_bucket == every_partial_week:
            # collect the required columns
            req_data = TimeDimension[[fcst_gen_time_bucket, partial_week_col, day_col]]

            # Calculate num of days in each partial week
            req_data[num_days_col] = req_data.groupby(partial_week_col)[day_col].transform("count")

            # Calculate num of days by fcst generation bucket
            req_data[total_num_days_col] = req_data.groupby(fcst_gen_time_bucket)[
                day_col
            ].transform("count")

            # Calculate the profile
            req_data[stat_bucket_weight_col] = req_data[num_days_col].divide(
                req_data[total_num_days_col]
            )

            # Collect all unique partial weeks
            PWProfile = req_data[[partial_week_col, stat_bucket_weight_col]].drop_duplicates()
            PWProfile.insert(0, version_col, input_version)
        else:
            # Group Logic based on Storage
            if fcst_storage_time_bucket == last_partial_week_of_week:
                # collect the required columns
                req_data = TimeDimension[[week_col, partial_week_col]].drop_duplicates()
                # select last pw from every member
                PWProfile = req_data.groupby(week_col, sort=False).last().reset_index()
            elif fcst_storage_time_bucket == last_partial_week_of_month:
                # collect the required columns
                req_data = TimeDimension[[month_col, partial_week_col]].drop_duplicates()
                # select last pw from every member
                PWProfile = req_data.groupby(month_col, sort=False).last().reset_index()
            elif fcst_storage_time_bucket == last_partial_week_of_planning_month:
                # collect the required columns
                req_data = TimeDimension[[planning_month_col, partial_week_col]].drop_duplicates()
                # select last pw from every member
                PWProfile = req_data.groupby(planning_month_col, sort=False).last().reset_index()

            # select pw col
            PWProfile = PWProfile[[partial_week_col]]

            # assign 100% weightage
            PWProfile[stat_bucket_weight_col] = 1.0
            PWProfile.insert(0, version_col, input_version)

        logger.debug(f"Generated {stat_bucket_weight_col} succesfully ...")

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        PWProfile = pd.DataFrame(columns=cols_required_in_output)

    return PWProfile

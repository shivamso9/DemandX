import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import filter_relevant_time_mapping
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

col_mapping = {
    "Lifecycle Actual": float,
    "Lifecycle Billing": float,
    "Lifecycle Backorders": float,
    "Lifecycle Orders": float,
    "Lifecycle Shipments": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    input_df,
    TimeDimension,
    TimeLevel,
    Grains,
    ReadFromHive,
    df_keys,
):
    plugin_name = "DP022PopulateLifecycleActual"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here
    version_col = "Version.[Version Name]"
    time_cols = [
        "DayName",
        "DayKey",
    ]
    relevant_time_name = "DayName"
    relevant_time_key = "DayKey"
    frequency = "Daily"
    frequency_initials = "D"  # Monthly:M , Weekly:W , Daily:D

    relevant_time_key_start = "DayKey_start"

    input_measures = [
        "Actual",
        "Billing",
        "Backorders",
        "Orders",
        "Shipments",
    ]

    # output measures
    output_measure_actual = "Lifecycle Actual"
    output_measure_billing = "Lifecycle Billing"
    output_measure_backorders = "Lifecycle Backorders"
    output_measure_orders = "Lifecycle Orders"
    output_measure_shipments = "Lifecycle Shipments"

    output_measures = [
        output_measure_actual,
        output_measure_billing,
        output_measure_backorders,
        output_measure_orders,
        output_measure_shipments,
    ]

    logger.info("Extracting dimension cols ...")

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get granular level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    assert len(dimensions) > 0, "dimensions cannot be empty ..."

    # assert and convert string value to boolean
    assert ReadFromHive in [
        "True",
        "False",
    ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
    ReadFromHive = eval(ReadFromHive)

    if ReadFromHive:
        output_grain = "Lifecycle Time.[Day]"
    else:
        output_grain = "[Lifecycle Time].Day"

    cols_required_in_output_df = [version_col, output_grain] + dimensions + output_measures
    output = pd.DataFrame(columns=cols_required_in_output_df)
    try:
        # check for empty data
        if len(input_df) == 0:
            logger.warning("Input is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")

            return output

        logger.info("time dimension head :")
        logger.info(TimeDimension.head())

        assert len(TimeDimension) > 0, "time mapping df cannot be empty ..."

        # rename time dim columns for existing functions to work
        rename_mapping = {
            "Time.[Day]": "DayName",
            "Time.[DayKey]": "DayKey",
        }
        TimeDimension.rename(columns=rename_mapping, inplace=True)

        # collect datetime key columns
        key_cols = [x for x in time_cols if "Key" in x]
        logger.info("Converting key cols to datetime format ...")

        # convert to datetime
        TimeDimension[key_cols] = TimeDimension[key_cols].apply(
            pd.to_datetime, infer_datetime_format=True
        )

        logger.info("Dropping duplicates from time mapping ...")
        TimeDimension.drop_duplicates(inplace=True)

        assert len(TimeDimension) > 0, "time dimension is empty after dropping duplicates ..."
        logger.info("time_mapping shape : {}".format(TimeDimension.shape))

        # Filter only relevant columns based on frequency
        time_mapping = filter_relevant_time_mapping(frequency, TimeDimension)

        input_df = pd.merge(
            input_df,
            time_mapping,
            how="left",
            left_on=TimeLevel,
            right_on=relevant_time_name,
        )

        # get the starting date combination for each unique combination of Item, ShipTo and Location
        logger.info("Get the start date...")
        input_start = input_df.groupby(dimensions)[relevant_time_key].min().reset_index()

        # add the start date to the parent data frame
        input_df = pd.merge(
            input_df,
            input_start,
            how="left",
            on=dimensions,
            suffixes=("", "_start"),
        )

        logger.info("Get the date difference...")
        # get the date difference
        input_df[output_grain] = (
            round(
                input_df[relevant_time_key].sub(input_df[relevant_time_key_start])
                / np.timedelta64(1, frequency_initials),
                0,
            )
            + 1
        )

        logger.info("Get the day difference...")
        # append D in front of the number
        input_df[output_grain] = frequency_initials + input_df[output_grain].astype(int).astype(
            str
        ).str.zfill(3)

        # relevant columns
        relevant_cols = [version_col, output_grain] + dimensions + input_measures

        # filter relevant columns
        relevant_input_df = input_df[relevant_cols]

        relevant_input_df.columns = cols_required_in_output_df

        output = relevant_input_df.fillna(value=0)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        output = pd.DataFrame(columns=cols_required_in_output_df)

    return output

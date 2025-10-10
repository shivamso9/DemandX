import logging

import pandas as pd
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension

# from o9Reference.common_utils.decorators import map_output_columns_to_dtypes

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


@log_inputs_and_outputs
@timed
# @map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    MeasureNames,
    SellInForecast,
    KeyFigures,
    SellOutForecast,
    df_keys,
):
    plugin_name = "DP079ForecastRaw"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    data_object = "Data Object.[Data Object]"

    include_in_fr = "Include in Forecast Realignment"
    do_planner_input = "Data Object Planner Input"

    fcst_raw_measure_suffix = " FND BB Raw"

    sell_in_dimensions = [x for x in SellInForecast.columns if is_dimension(x)]
    sell_in_measures = [x for x in SellInForecast if x not in sell_in_dimensions]

    sell_out_dimensions = [x for x in SellOutForecast.columns if is_dimension(x)]
    sell_out_measures = [x for x in SellOutForecast if x not in sell_out_dimensions]

    if ~KeyFigures.empty:
        # getting list of measures to be realigned
        measures_to_consider = KeyFigures[KeyFigures[include_in_fr]][data_object].to_list()

        # getting list of override measures
        override_measure_list = KeyFigures[KeyFigures[do_planner_input] == 1][data_object].to_list()
    else:
        logger.warning("No data for input key figures ...")
        logger.warning("No further execution ...")

        sell_in_output_mapping = {x: x + fcst_raw_measure_suffix for x in sell_in_measures}
        sell_out_output_mapping = {x: x + fcst_raw_measure_suffix for x in sell_out_measures}
        SellInForecast.rename(columns=sell_in_output_mapping, inplace=True)
        SellOutForecast.rename(columns=sell_out_output_mapping, inplace=True)

        return SellInForecast, SellOutForecast

    # sell in output
    relevant_sell_in_measures = [
        x for x in sell_in_measures if x in measures_to_consider and x not in override_measure_list
    ]
    sell_in_output_mapping = {x: x + fcst_raw_measure_suffix for x in relevant_sell_in_measures}
    sell_in_output_cols = sell_in_dimensions + list(sell_in_output_mapping.values())
    RawSellInForecast = pd.DataFrame(columns=sell_in_output_cols)

    # sell out output
    relevant_sell_out_measures = [
        x for x in sell_out_measures if x in measures_to_consider and x not in override_measure_list
    ]
    sell_out_output_mapping = {x: x + fcst_raw_measure_suffix for x in relevant_sell_out_measures}
    sell_out_output_cols = sell_out_dimensions + list(sell_out_output_mapping.values())
    RawSellOutForecast = pd.DataFrame(columns=sell_out_output_cols)

    try:
        if SellInForecast.empty:
            logger.warning("Input for sell in is empty ...")
            logger.warning("empty dataframe for sell in output ...")
        else:
            RawSellInForecast = SellInForecast.copy()

            RawSellInForecast.rename(columns=sell_in_output_mapping, inplace=True)
            RawSellInForecast = RawSellInForecast[sell_in_output_cols]

        if SellOutForecast.empty:
            logger.warning("Input for sell out is empty ...")
            logger.warning("empty dataframe for sell out output ...")
        else:
            RawSellOutForecast = SellOutForecast.copy()

            RawSellOutForecast.rename(columns=sell_out_output_mapping, inplace=True)
            RawSellOutForecast = RawSellOutForecast[sell_out_output_cols]
        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

        sell_in_output_mapping = {x: x + fcst_raw_measure_suffix for x in sell_in_measures}
        sell_out_output_mapping = {x: x + fcst_raw_measure_suffix for x in sell_out_measures}
        SellInForecast.rename(columns=sell_in_output_mapping, inplace=True)
        SellOutForecast.rename(columns=sell_out_output_mapping, inplace=True)

        return SellInForecast, SellOutForecast

    return (
        RawSellInForecast,
        RawSellOutForecast,
    )

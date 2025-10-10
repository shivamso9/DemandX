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
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

col_mapping = {
    "Past In Transit Inventory": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    grains: str,
    InTransitInventoryInput: pd.DataFrame,
    CurrentWeek: pd.DataFrame,
    TimeDimension: pd.DataFrame,
    df_keys,
):
    """Calculating Past In Transit Inventory  from In Transit Inventory Input
    Filtering Past week's data and summing it up and storing in Current week's Data for Past In Transit Inventory

    Args:
        grains (str): grains to be used for Input Dataframe
        InTransitInventoryInput (pd.DataFrame): Input Dataframe
        CurrentWeek (pd.DataFrame): Current Week
        TimeDimension (pd.DataFrame): Time master data

    Raises:
        ValueError: If any of the required columns are missing.

    Returns:
        _type_: Dataframe
    """

    plugin_name = "DP044PopulateInTransit"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    WEEK_KEY = "Time.[WeekKey]"
    InTransitInventoryInput_col = "In Transit Inventory Input"
    InTransitInventoryAgg_col = "Past In Transit Inventory"
    grains = get_list_of_grains_from_string(grains)
    cols_required_in_output = (
        [o9Constants.VERSION_NAME] + grains + [o9Constants.WEEK] + [InTransitInventoryAgg_col]
    )

    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        if InTransitInventoryInput[InTransitInventoryInput_col].count() == 0 or TimeDimension.empty:
            raise ValueError("One of the inputs is empty! Check logs/ inputs for error!")
        min_week_key = CurrentWeek[WEEK_KEY].min()
        logger.debug("Joining Input data with Time Dimension ...")
        InTransitInventoryInput = InTransitInventoryInput.merge(
            TimeDimension[[o9Constants.WEEK, WEEK_KEY]],
            on=[o9Constants.WEEK],
            how="inner",
        )
        InTransitInventoryInput = InTransitInventoryInput.drop_duplicates()
        # filtering data for past weeks
        InTransitInventoryAgg = InTransitInventoryInput[
            InTransitInventoryInput[WEEK_KEY] < min_week_key
        ]
        if InTransitInventoryAgg.empty:
            raise ValueError("In Transit Inventory Input is not populated for past weeks")

        # Grouping by dimensions and summing up the data
        logger.debug("Grouping Dataframe by grains ...")
        past_weeks_sum = InTransitInventoryAgg.groupby(grains)[InTransitInventoryInput_col].sum()

        Agg_list = []

        logger.debug("aggregating Past week data and storing in current week ...")
        df_Agg = InTransitInventoryInput.groupby(grains)
        for group, group_df in df_Agg:
            group_df.loc[group_df[WEEK_KEY] == min_week_key, InTransitInventoryAgg_col] = (
                past_weeks_sum.iloc[0]
            )
            Agg_list += [group_df]
        InTransitInventory = concat_to_dataframe(Agg_list)
        InTransitInventory = InTransitInventory[InTransitInventory[WEEK_KEY] == min_week_key]
        Output = InTransitInventory[cols_required_in_output]

        logger.info("Output created... Returning it to the caller fn")
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
    return Output

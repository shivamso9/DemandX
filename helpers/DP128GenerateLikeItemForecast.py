"""This plugin generates like item forecast.

1. Filtering the selected dataframe
2. Calculating the weighted sum of  like item forecast for each new item
3. Calculate the number of like item
4. Perform the respective operation as per the given input (weighted average or weighted sum)
"""

# Library imports
import logging
from typing import Any, Dict, Optional

import pandas as pd
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {"Like Item Fcst L0": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    # Data
    Parameters,
    LikeItemMappings,
    ForecastData,
    SelectedCombinations,
    Grains,
    TimeLevel,
    # Others
    df_keys: Optional[Dict[Any, Any]] = None,
):
    """DP128GenerateLikeItemForecast plugin."""
    plugin_name = "DP128GenerateLikeItemForecast"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"

    from_data_obj = "from.[Data Object].[Data Object]"
    data_obj = "Data Object.[Data Object]"
    from_initiative = "from.[Initiative].[Initiative]"
    initiative = "Initiative.[Initiative]"
    from_item = "from.[Item].[NPI Item]"
    from_pnl = "from.[PnL].[NPI PnL]"
    from_region = "from.[Region].[NPI Region]"
    from_channel = "from.[Channel].[NPI Channel]"
    from_account = "from.[Account].[NPI Account]"
    from_location = "from.[Location].[NPI Location]"
    from_demand_domain = "from.[Demand Domain].[NPI Demand Domain]"
    like_item = "to.[Item].[NPI Item]"
    to_pnl = "to.[PnL].[NPI PnL]"
    to_region = "to.[Region].[NPI Region]"
    to_channel = "to.[Channel].[NPI Channel]"
    to_account = "to.[Account].[NPI Account]"
    to_location = "to.[Location].[NPI Location]"
    to_demand_domain = "to.[Demand Domain].[NPI Demand Domain]"
    NPI_region_col = "Region.[NPI Region]"
    NPI_account_col = "Account.[NPI Account]"
    NPI_channel_col = "Channel.[NPI Channel]"
    NPI_pnl_col = "PnL.[NPI PnL]"
    NPI_item_col = "Item.[NPI Item]"
    NPI_location_col = "Location.[NPI Location]"
    NPI_demand_domain_col = "Demand Domain.[NPI Demand Domain]"
    like_item_fcst_method_col = "Like Item Fcst Method L0"
    like_assortment_weight_col = (
        "630 Initiative Like Assortment Match.[User Override Like Assortment Weight L0]"
    )
    weighted_sum = "Weighted Sum"
    weighted_average = "Weighted Average"
    forecast_data_measure = "Stat Fcst NPI BB L0"
    partial_week_col = "Time.[Partial Week]"

    # Output Measure
    output_measure = "Like Item Fcst L0"

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # Modify this to ensure consistency with SelectedCombinations column names
    dimensions = [f"{x}.{x}" if x not in ["NA", ""] else x for x in all_grains]

    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))
    cols_required_in_output_df_early_exit = (
        [version_col, partial_week_col] + dimensions + [output_measure]
    )

    try:
        assert len(dimensions) > 0, "dimensions cannot be empty ..."

        if len(ForecastData) == 0:
            logger.warning("Input is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        input_version = ForecastData[version_col].iloc[0]

        # shortlist only the selected combinations
        merge_cols = [
            data_obj,
            initiative,
            NPI_item_col,
            NPI_pnl_col,
            NPI_region_col,
            NPI_account_col,
            NPI_channel_col,
            NPI_location_col,
            NPI_demand_domain_col,
        ]
        new_item_customers = pd.merge(
            SelectedCombinations[dimensions],
            LikeItemMappings,
            how="inner",
            left_on=merge_cols,
            right_on=[
                from_data_obj,
                from_initiative,
                from_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
                from_demand_domain,
            ],
        )

        logger.info("finding like items for selected combinations of item and other group...")
        relevant_columns = [
            from_data_obj,
            from_initiative,
            from_item,
            from_pnl,
            from_region,
            from_account,
            from_channel,
            from_location,
            from_demand_domain,
            like_item,
            to_pnl,
            to_region,
            to_account,
            to_channel,
            to_location,
            to_demand_domain,
            like_assortment_weight_col,
        ]

        relevant_like_item_mappings = new_item_customers[relevant_columns].drop_duplicates()

        # Replace NaN values in 'like_assortment_weight_col' with 1
        relevant_like_item_mappings[like_assortment_weight_col] = relevant_like_item_mappings[
            like_assortment_weight_col
        ].fillna(1)

        if len(relevant_like_item_mappings) == 0:
            logger.warning("no relevant like items found for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        logger.info("calculating forecast for new items...")

        merge_cols_fcst = [
            NPI_item_col,
            NPI_pnl_col,
            NPI_region_col,
            NPI_account_col,
            NPI_channel_col,
            NPI_location_col,
            NPI_demand_domain_col,
        ]
        # mapping forecast of like items
        relevant_like_item_forecast = pd.merge(
            relevant_like_item_mappings,
            ForecastData,
            left_on=[
                like_item,
                to_pnl,
                to_region,
                to_account,
                to_channel,
                to_location,
                to_demand_domain,
            ],
            right_on=merge_cols_fcst,
            how="inner",
        )

        logger.info("assigning forecast method for like items...")

        if len(relevant_like_item_forecast) == 0:
            logger.warning("no relevant like item forecast found for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        # weighted sum of  like item forecast for each new item
        relevant_like_item_forecast[weighted_sum] = (
            relevant_like_item_forecast[forecast_data_measure]
            * relevant_like_item_forecast[like_assortment_weight_col]
        )

        relevant_like_item_forecast = (
            relevant_like_item_forecast.groupby(
                (
                    [
                        from_data_obj,
                        from_initiative,
                        from_item,
                        from_pnl,
                        from_region,
                        from_account,
                        from_channel,
                        from_location,
                        from_demand_domain,
                        TimeLevel,
                    ]
                ),
                observed=True,
            )[weighted_sum]
            .sum()
            .reset_index()
        )

        relevant_like_item_weight = (
            relevant_like_item_mappings.groupby(
                [
                    from_data_obj,
                    from_initiative,
                    from_item,
                    from_pnl,
                    from_region,
                    from_account,
                    from_channel,
                    from_location,
                    from_demand_domain,
                ],
                observed=True,
            )[like_assortment_weight_col]
            .sum()
            .reset_index()
        )

        logger.info("Calculating Like Item Forecast...")

        LikeItemForecast = pd.merge(
            relevant_like_item_forecast,
            relevant_like_item_weight,
            on=[
                from_data_obj,
                from_initiative,
                from_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
                from_demand_domain,
            ],
        )

        LikeItemForecast[weighted_average] = (
            LikeItemForecast[weighted_sum] / LikeItemForecast[like_assortment_weight_col]
        )

        # dropping extra columns
        LikeItemForecast.drop(columns=[like_assortment_weight_col], inplace=True)

        column_mapping = {
            from_data_obj: data_obj,
            from_initiative: initiative,
            from_item: NPI_item_col,
            from_pnl: NPI_pnl_col,
            from_region: NPI_region_col,
            from_account: NPI_account_col,
            from_channel: NPI_channel_col,
            from_location: NPI_location_col,
            from_demand_domain: NPI_demand_domain_col,
        }

        # Rename the columns
        LikeItemForecast.rename(columns=column_mapping, inplace=True)

        logger.info("Mapping forecast method for like items...")

        # Extracting forecast method for like items
        Parameters.drop(columns=[version_col], inplace=True)

        # Perform the merge
        LikeItemForecast = pd.merge(
            LikeItemForecast,
            Parameters,
            on=merge_cols,
            how="left",
        )

        # Replace NaN values in LikeItemForecast with Weighted Average
        LikeItemForecast[like_item_fcst_method_col] = LikeItemForecast[
            like_item_fcst_method_col
        ].fillna(weighted_average)

        # update the column values based on like item forecast method
        LikeItemForecast[output_measure] = LikeItemForecast.apply(
            lambda row: (
                row[weighted_average]
                if row[like_item_fcst_method_col] == weighted_average
                else row[weighted_sum]
            ),
            axis=1,
        )

        LikeItemForecast.drop(
            columns=[
                weighted_sum,
                weighted_average,
                like_item_fcst_method_col,
            ],
            inplace=True,
        )

        # Add input version
        LikeItemForecast.insert(loc=0, column=version_col, value=input_version)

        # getting relevant columns
        cols_req = dimensions + [version_col, TimeLevel, output_measure]
        LikeItemForecast = LikeItemForecast[cols_req]

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        LikeItemForecast = pd.DataFrame(columns=cols_required_in_output_df_early_exit)
    return LikeItemForecast

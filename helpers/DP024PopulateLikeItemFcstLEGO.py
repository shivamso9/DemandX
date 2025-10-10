import logging

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
pd.options.mode.chained_assignment = None

col_mapping = {"Like Item Fcst": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    like_item_mappings,
    forecast_data,
    selected_combinations,
    TimeLevel,
    Grains,
    IsAssorted,
    df_keys,
):
    # Configurables - define all column names here
    plugin_name = "DP024PopulateLikeItemFcstLEGO"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version.[Version Name]"
    from_item = "from.[Item].[Stat Item]"
    like_item = "to.[Item].[Stat Item]"
    from_pnl = "from.[PnL].[Stat PnL]"
    from_region = "from.[Region].[Stat Region]"
    from_channel = "from.[Channel].[Stat Channel]"
    from_account = "from.[Account].[Stat Account]"
    from_location = "from.[Location].[Stat Location]"
    pl_region_col = "Region.[Stat Region]"
    pl_account_col = "Account.[Stat Account]"
    pl_channel_col = "Channel.[Stat Channel]"
    pl_pnl_col = "PnL.[Stat PnL]"
    pl_item_col = "Item.[Stat Item]"
    pl_location_col = "Location.[Stat Location]"
    like_item_count_col = "Like Item Count"
    forecast_data_measure = "Stat Fcst NPI BB"
    partial_week_col = "Time.[Partial Week]"
    demand_domain_col = "Demand Domain.[Stat Demand Domain]"

    # output measures
    output_measure = "Like Item Fcst"

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))
    cols_required_in_output_df_early_exit = (
        [version_col, partial_week_col] + dimensions + [output_measure]
    )
    try:

        assert len(dimensions) > 0, "dimensions cannot be empty ..."

        if len(forecast_data) == 0:
            logger.warning("Input is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        input_version = forecast_data[version_col].iloc[0]

        # shortlist only the selected combinations
        merge_cols = [
            pl_item_col,
            pl_pnl_col,
            pl_region_col,
            pl_account_col,
            pl_channel_col,
            pl_location_col,
        ]
        new_item_customers = pd.merge(
            selected_combinations[dimensions],
            like_item_mappings,
            how="inner",
            left_on=merge_cols,
            right_on=[
                from_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
            ],
        )

        logger.info("finding like items for selected combinations of item and other group...")
        relevant_columns = [
            from_item,
            from_pnl,
            from_region,
            from_account,
            from_channel,
            from_location,
            like_item,
        ]

        relevant_like_item_mappings = new_item_customers[relevant_columns].drop_duplicates()

        if len(relevant_like_item_mappings) == 0:
            logger.warning("no relevant like items found for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        logger.info("calculating forecast for new items...")
        # mapping forecast of like items
        relevant_like_item_forecast = pd.merge(
            relevant_like_item_mappings,
            forecast_data,
            left_on=[
                like_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
            ],
            right_on=merge_cols,
            how="inner",
        )

        if len(relevant_like_item_forecast) == 0:
            logger.warning("no relevant like item forecast found for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output_df_early_exit)

        # averaging like item forecast for each new item
        relevant_like_item_forecast = (
            relevant_like_item_forecast.groupby(
                (
                    [
                        from_item,
                        from_pnl,
                        from_region,
                        from_account,
                        from_channel,
                        from_location,
                        demand_domain_col,
                        TimeLevel,
                    ]
                ),
                observed=True,
            )[forecast_data_measure]
            .sum()
            .reset_index()
        )

        # counting no. of like items for each new item
        relevant_like_item_mappings[like_item_count_col] = 1

        relevant_like_item_count = (
            relevant_like_item_mappings.groupby(
                [
                    from_item,
                    from_pnl,
                    from_region,
                    from_account,
                    from_channel,
                    from_location,
                ],
                observed=True,
            )[like_item_count_col]
            .sum()
            .reset_index()
        )

        LikeItemForecast = pd.merge(
            relevant_like_item_forecast,
            relevant_like_item_count,
            on=[
                from_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
            ],
        )

        LikeItemForecast[forecast_data_measure] = (
            LikeItemForecast[forecast_data_measure] / LikeItemForecast[like_item_count_col]
        )

        # dropping extra columns
        LikeItemForecast.drop(columns=[like_item_count_col, demand_domain_col], inplace=True)

        LikeItemForecast = LikeItemForecast.merge(
            IsAssorted[dimensions].drop_duplicates(),
            left_on=[
                from_item,
                from_pnl,
                from_region,
                from_account,
                from_channel,
                from_location,
            ],
            right_on=merge_cols,
            how="inner",
        )

        # renaming output column
        LikeItemForecast.rename(columns={forecast_data_measure: output_measure}, inplace=True)

        # Add input version
        LikeItemForecast.insert(loc=0, column=version_col, value=input_version)

        # getting relevant columns
        cols_req = dimensions + [version_col, TimeLevel, output_measure]
        LikeItemForecast = LikeItemForecast[cols_req]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        LikeItemForecast = pd.DataFrame(columns=cols_required_in_output_df_early_exit)
    return LikeItemForecast

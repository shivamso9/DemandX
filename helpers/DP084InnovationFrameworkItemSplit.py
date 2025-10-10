import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    create_cartesian_product,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.get_moving_avg_forecast import get_moving_avg_forecast

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


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
        logger.exception("Exception for {}, {}".format(e, df[dimensions].iloc[0].values))
    return result


def split_for_items_with_no_actual(
    relevant_actual,
    ForecastBucket,
    ItemMapping,
    cust_grp_item_with_assortment,
    pl_item_col,
    item_col,
    partial_week_key_col,
    intro_date_col,
    disc_date_col,
    actual_col,
    actual_agg_col,
    partial_week_col,
    moving_avg_sku_split,
    pl_level_cols,
):
    pl_item = relevant_actual[pl_item_col].unique()
    the_item_mapping = ItemMapping[ItemMapping[pl_item_col].isin(pl_item)]

    items_with_actual = set(relevant_actual[item_col].unique())
    items_within_pl_item = set(the_item_mapping[item_col].unique())
    items_with_no_actual_list = list(items_within_pl_item - items_with_actual)

    cols_required = pl_level_cols + [
        item_col,
        partial_week_col,
        partial_week_key_col,
        moving_avg_sku_split,
    ]
    output = pd.DataFrame(columns=cols_required)

    if len(items_with_no_actual_list) != 0:
        items_with_no_actual = the_item_mapping[
            the_item_mapping[item_col].isin(items_with_no_actual_list)
        ]

        # consider items with assortment 1
        items_with_no_actual = items_with_no_actual.merge(
            cust_grp_item_with_assortment,
            on=[item_col],
            how="inner",
        )

        if len(items_with_no_actual) == 0:
            return output
        # get partial weeks in ForecastBucket
        items_with_no_actual = create_cartesian_product(
            items_with_no_actual,
            ForecastBucket,
        )

        # filter partial weeks which satisfy corresponding intro & disc dates
        items_with_no_actual = items_with_no_actual[
            (items_with_no_actual[partial_week_key_col] >= items_with_no_actual[intro_date_col])
            & (items_with_no_actual[partial_week_key_col] <= items_with_no_actual[disc_date_col])
        ]

        # getting max of disc date for items with actual
        filtered_df = the_item_mapping[the_item_mapping[item_col].isin(items_with_actual)]
        disc_date = filtered_df[disc_date_col].max()

        # filter intersections for which partial week lies after disc date of items with actual
        items_with_no_actual = items_with_no_actual[
            items_with_no_actual[partial_week_key_col] > disc_date
        ]

        if len(items_with_no_actual) == 0:
            return output

        # populate 1 in Actual col
        items_with_no_actual[actual_col] = 1

        items_with_no_actual = items_with_no_actual.merge(
            relevant_actual[pl_level_cols].drop_duplicates(),
            on=pl_level_cols,
            how="inner",
        ).drop_duplicates()

        # normalizing the moving avg split %
        items_with_no_actual[actual_agg_col] = items_with_no_actual.groupby(
            [partial_week_col] + pl_level_cols,
            observed=True,
        )[actual_col].transform("sum")
        items_with_no_actual[moving_avg_sku_split] = np.where(
            items_with_no_actual[actual_agg_col] != 0,
            items_with_no_actual[actual_col] / items_with_no_actual[actual_agg_col],
            np.nan,
        )

        output = items_with_no_actual[cols_required]

    return output


col_mapping = {
    "Item Split Moving Avg": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ForecastBucket,
    HistoryMeasure,
    Grains,
    TimeLevel,
    Actual,
    CurrentTimePeriod,
    TimeDimension,
    df_keys,
    ItemSplitParameters,
    ItemMapping,
    ItemSplitMethod,
    NPIFcstL1,
    AssortmentFlag,
    multiprocessing_num_cores=4,
) -> pd.DataFrame:
    plugin_name = "DP084InnovationFrameworkItemSplit"
    logger.warning("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    initiative_col = "Initiative.[Initiative]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    pl_location_col = "Location.[Planning Location]"
    item_col = "Item.[Item]"
    pl_item_col = "Item.[Planning Item]"
    item_count = "Item Count"

    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_col = "Time.[Week]"
    week_key_col = "Time.[WeekKey]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"
    pl_month_col = "Time.[Planning Month]"
    pl_month_key_col = "Time.[PlanningMonthKey]"

    actual_col = str(HistoryMeasure)
    actual_agg_col = "Actual Agg"
    item_split_history_period = "Item Split History Period"
    item_split_history_time_bucket = "Item Split History Time Bucket"
    intro_date_col = "Item.[Item Intro Date]"
    disc_date_col = "Item.[Item Disc Date]"
    cumulative_split_col = "Cumulative Split"

    logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    # output measure
    moving_avg_sku_split = "NPI Item Split Moving Avg"

    cols_required_for_moving_avg_split_ratio = (
        [version_col] + dimensions + [initiative_col, TimeLevel, moving_avg_sku_split]
    )

    MovingAvgSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

    SKUSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)
    try:
        common_cols = [x for x in dimensions if x != item_col]
        if Actual is None:
            cols_req = [version_col] + dimensions + [partial_week_col, actual_col]
            Actual = pd.DataFrame(columns=cols_req)

        if NPIFcstL1 is None:
            cols_req = (
                [version_col, initiative_col]
                + common_cols
                + [pl_item_col, partial_week_col, actual_col]
            )
            NPIFcstL1 = pd.DataFrame(columns=cols_req)

        if len(Actual) == 0:
            logger.warning(
                "Actual is empty, check Grain configuration for slice : {} ...".format(df_keys)
            )
            logger.warning("Will not generate output for moving average ...")

        if len(AssortmentFlag) == 0:
            logger.warning(
                "AssortmentFlag cannot be empty, check Grain configuration for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning("Will return empty dataframe for this slice ...")
            return SKUSplit

        input_version = AssortmentFlag[version_col].iloc[0]
        TimeDimension[week_col] = TimeDimension[week_col].astype(str)
        min_date = TimeDimension[partial_week_key_col].min()
        max_date = TimeDimension[partial_week_key_col].max()
        ItemMapping[intro_date_col] = pd.to_datetime(
            ItemMapping[intro_date_col],
            infer_datetime_format=True,
            errors="coerce",
        )
        ItemMapping[disc_date_col] = pd.to_datetime(
            ItemMapping[disc_date_col],
            infer_datetime_format=True,
            errors="coerce",
        )
        ItemMapping[intro_date_col] = ItemMapping[intro_date_col].fillna(min_date)
        ItemMapping[disc_date_col] = ItemMapping[disc_date_col].fillna(max_date)
        # collect datetime key columns
        key_cols = [
            pl_month_key_col,
            month_key_col,
            week_key_col,
            partial_week_key_col,
        ]
        logger.info("Converting key cols to datetime format ...")

        # convert to datetime
        TimeDimension[key_cols] = TimeDimension[key_cols].apply(
            pd.to_datetime, infer_datetime_format=True
        )

        if len(dimensions) == 0:
            logger.warning(
                "Dimensions cannot be empty, check Grain configuration for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning("Will return empty dataframe for this slice ...")
            return SKUSplit

        # getting min and max dates from ForecastBucket
        # to restrict output for forecast buckets
        start_date_key = ForecastBucket[partial_week_key_col].min()
        end_date_key = ForecastBucket[partial_week_key_col].max()

        ForecastBucket.sort_values(partial_week_key_col, inplace=True)

        # getting min and max dates from TimeDimension
        # to get default intro and disc dates
        intro_date_key = TimeDimension[partial_week_key_col].min()
        disc_date_key = TimeDimension[partial_week_key_col].max()

        # setting default intro and disc dates
        ItemMapping[intro_date_col].fillna(intro_date_key, inplace=True)
        ItemMapping[disc_date_col].fillna(disc_date_key, inplace=True)

        if (len(ItemSplitParameters) != 0) and (len(Actual) != 0):
            MovingAvgPeriods = int(ItemSplitParameters[item_split_history_period][0])
            logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))

            if ItemSplitParameters[item_split_history_time_bucket][0] == "Planning Month":
                relevant_time_name = pl_month_col
                relevant_time_key = pl_month_key_col

            elif ItemSplitParameters[item_split_history_time_bucket][0] == "Month":
                relevant_time_name = month_col
                relevant_time_key = month_key_col

            elif ItemSplitParameters[item_split_history_time_bucket][0] == "Week":
                relevant_time_name = week_col
                relevant_time_key = week_key_col

            else:
                logger.warning(
                    "Incorrect history time bucket is provided, check for slice : {} ...".format(
                        df_keys
                    )
                )
                logger.warning("Will return empty dataframe for this slice ...")
                return SKUSplit

            time_mapping = (
                TimeDimension[[relevant_time_name, relevant_time_key]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            time_attribute_dict = {relevant_time_name: relevant_time_key}

            latest_time_name = CurrentTimePeriod[relevant_time_name][0]

            # get last n periods dates
            last_n_periods = get_n_time_periods(
                latest_time_name,
                -MovingAvgPeriods,
                time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            logger.info("last_n_periods : {} ...".format(last_n_periods))

            # get relevant time mapping for copying split values at partial week
            relevant_time_mapping = (
                TimeDimension[
                    [
                        relevant_time_name,
                        partial_week_col,
                        partial_week_key_col,
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            req_cols = [version_col] + dimensions + [partial_week_col, actual_col]
            Actual = Actual[req_cols]

            # merge with TimeDimension to get relevant time
            Actual = Actual.merge(
                relevant_time_mapping,
                on=partial_week_col,
                how="left",
            )

            # Filter out relevant history
            Actual = Actual[Actual[relevant_time_name].isin(last_n_periods)]

            Actual = Actual.merge(
                ItemMapping[[item_col, pl_item_col]],
                on=item_col,
                how="inner",
            )

            # get new items and calculate split ratio
            pl_level_cols = [
                pl_item_col,
                pl_account_col,
                pl_pnl_col,
                pl_location_col,
                pl_channel_col,
                pl_region_col,
                pl_demand_domain_col,
            ]
            cust_grp_pl_item_in_actual = (
                Actual[pl_level_cols].drop_duplicates().reset_index(drop=True)
            )
            cust_grp_item_with_assortment = (
                AssortmentFlag[
                    [
                        item_col,
                        pl_account_col,
                        pl_pnl_col,
                        pl_location_col,
                        pl_channel_col,
                        pl_region_col,
                        pl_demand_domain_col,
                        initiative_col,
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # getting planning item for each item
            cust_grp_pl_item_with_assortment = cust_grp_item_with_assortment.merge(
                ItemMapping,
                on=[item_col],
                how="inner",
            )

            AssortmentFlag.drop(version_col, axis=1, inplace=True)
            Actual = Actual.merge(
                AssortmentFlag,
                on=dimensions,
                how="inner",
            )
            dimensions = dimensions + [initiative_col]

            new_intersections = pd.merge(
                cust_grp_pl_item_with_assortment,
                cust_grp_pl_item_in_actual,
                on=pl_level_cols,
                how="outer",
                indicator=True,
            )

            new_intersections = new_intersections[new_intersections["_merge"] == "left_only"]

            # process only those intersections present in ItemSplitMethod
            actual_df = Actual.merge(
                ItemSplitMethod,
                on=pl_level_cols,
                how="inner",
            )
            pl_level_cols = pl_level_cols + [initiative_col]

            if len(new_intersections) != 0:
                new_intersections[item_count] = new_intersections.groupby(
                    pl_level_cols, observed=True
                )[item_col].transform("count")

                # distributing split ratio for all new items within planning item
                new_intersections[moving_avg_sku_split] = 1 / new_intersections[item_count]

            # to get future periods for relevant time
            relevant_future_periods = ForecastBucket.merge(
                TimeDimension[[partial_week_col, relevant_time_name]].drop_duplicates(),
                on=partial_week_col,
                how="inner",
            )

            # get test period dates
            future_n_periods = list(relevant_future_periods[relevant_time_name].unique())
            logger.info("future_n_periods : {} ...".format(future_n_periods))
            FuturePeriods = int(len(future_n_periods))

            # cap negatives in HistoryMeasure
            Actual[HistoryMeasure] = np.where(Actual[HistoryMeasure] < 0, 0, Actual[HistoryMeasure])

            # getting relevant columns to concat
            cols_required = pl_level_cols + [
                item_col,
                partial_week_col,
                partial_week_key_col,
                moving_avg_sku_split,
            ]
            output_df = pd.DataFrame(columns=cols_required)

            if len(actual_df) != 0:
                # aggregate data to relevant time level
                actual_df = (
                    actual_df.groupby(
                        [
                            pl_account_col,
                            pl_pnl_col,
                            pl_location_col,
                            pl_channel_col,
                            pl_region_col,
                            pl_demand_domain_col,
                            pl_item_col,
                            initiative_col,
                            item_col,
                            relevant_time_name,
                        ]
                    )[actual_col]
                    .sum()
                    .reset_index()
                )

                # Fill missing dates
                relevant_actual_nas_filled = fill_missing_dates(
                    actual=actual_df,
                    forecast_level=dimensions,
                    history_measure=HistoryMeasure,
                    relevant_time_key=relevant_time_key,
                    relevant_time_name=relevant_time_name,
                    relevant_time_periods=last_n_periods,
                    time_mapping=time_mapping,
                    fill_nulls_with_zero=True,
                    filter_from_start_date=False,
                )

                logger.info("Calculating moving average forecast for all intersections ...")
                all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                    delayed(calculate_sliding_window_forecast)(
                        group,
                        MovingAvgPeriods,
                        FuturePeriods,
                        future_n_periods,
                        dimensions,
                        HistoryMeasure,
                        relevant_time_key,
                        relevant_time_name,
                    )
                    for name, group in relevant_actual_nas_filled.groupby(dimensions)
                )

                # Concat all results to one dataframe
                outputData = concat_to_dataframe(all_results)
                # validate if output dataframe contains result for all groups present in input
                validate_output(
                    input_df=relevant_actual_nas_filled,
                    output_df=outputData,
                    forecast_level=dimensions,
                )

                # join with the item mapping to get planning item
                outputData = outputData.merge(
                    ItemMapping,
                    on=item_col,
                    how="inner",
                ).drop_duplicates()

                logger.info("Aggregating actual at planning item level...")
                group_by_cols = pl_level_cols + [relevant_time_name]
                outputData[actual_agg_col] = outputData.groupby(group_by_cols, observed=True)[
                    actual_col
                ].transform("sum")

                logger.info("Calculating split % at planning item level...")
                outputData[moving_avg_sku_split] = np.where(
                    outputData[actual_agg_col] != 0,
                    outputData[actual_col] / outputData[actual_agg_col],
                    0,
                )

                output_df = outputData.merge(
                    relevant_time_mapping,
                    on=relevant_time_name,
                    how="left",
                )

                # get only those partial weeks which satisfy intro & disc dates and are in forecastbucket
                output_df = output_df[
                    (output_df[partial_week_key_col] >= output_df[intro_date_col])
                    & (output_df[partial_week_key_col] <= output_df[disc_date_col])
                ]

                output_df = output_df[
                    (output_df[partial_week_key_col] >= start_date_key)
                    & (output_df[partial_week_key_col] <= end_date_key)
                ]

                output_df = output_df[cols_required]

            # calculating splits for items with no actual
            items_with_no_actual_split = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                delayed(split_for_items_with_no_actual)(
                    relevant_actual=df,
                    ForecastBucket=ForecastBucket,
                    ItemMapping=ItemMapping,
                    cust_grp_item_with_assortment=cust_grp_item_with_assortment,
                    pl_item_col=pl_item_col,
                    item_col=item_col,
                    partial_week_key_col=partial_week_key_col,
                    intro_date_col=intro_date_col,
                    disc_date_col=disc_date_col,
                    actual_col=actual_col,
                    actual_agg_col=actual_agg_col,
                    partial_week_col=partial_week_col,
                    moving_avg_sku_split=moving_avg_sku_split,
                    pl_level_cols=pl_level_cols,
                )
                for name, df in Actual.groupby(pl_level_cols)
            )

            # Concat all results to one dataframe
            items_with_no_actual_split = concat_to_dataframe(items_with_no_actual_split)
            # validate if output dataframe contains result for all groups present in input
            validate_output(
                input_df=Actual,
                output_df=items_with_no_actual_split,
                forecast_level=pl_level_cols,
            )

            # adding check if dataframe is empty
            # creating empty dataframe in case empty dataframe with no columns
            if items_with_no_actual_split.empty:
                items_with_no_actual_split = pd.DataFrame(columns=cols_required)

            items_with_no_actual_split = items_with_no_actual_split[cols_required]

            output_df = pd.concat([output_df, items_with_no_actual_split])

            if len(output_df) != 0:
                # filter intersections present in NPIFcstL1
                output_df = output_df.merge(
                    NPIFcstL1,
                    on=[partial_week_col] + pl_level_cols,
                    how="inner",
                ).drop_duplicates()

                # normalizing the moving avg split %
                output_df[cumulative_split_col] = output_df.groupby(
                    [partial_week_col] + pl_level_cols,
                    observed=True,
                )[moving_avg_sku_split].transform("sum")

                output_df[moving_avg_sku_split] = np.where(
                    output_df[cumulative_split_col] != 0,
                    output_df[moving_avg_sku_split] / output_df[cumulative_split_col],
                    np.nan,
                )
                MovingAvgSplit = output_df[cols_required_for_moving_avg_split_ratio]

            relevant_data = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

            if len(new_intersections) != 0:
                # creating cartesian product to get all partial weeks
                relevant_data = create_cartesian_product(
                    new_intersections,
                    ForecastBucket,
                )

                # getting only those partial weeks which satisfy corresponding intro and disc dates
                relevant_data = relevant_data[
                    (relevant_data[partial_week_key_col] >= relevant_data[intro_date_col])
                    & (relevant_data[partial_week_key_col] <= relevant_data[disc_date_col])
                ]

                relevant_data[cumulative_split_col] = relevant_data.groupby(
                    [partial_week_col] + pl_level_cols,
                    observed=True,
                )[moving_avg_sku_split].transform("sum")

                relevant_data[moving_avg_sku_split] = np.where(
                    relevant_data[cumulative_split_col] != 0,
                    relevant_data[moving_avg_sku_split] / relevant_data[cumulative_split_col],
                    np.nan,
                )
                relevant_data[version_col] = input_version

                relevant_data = relevant_data[cols_required_for_moving_avg_split_ratio]

            MovingAvgSplit = pd.concat([MovingAvgSplit, relevant_data])

        SKUSplit = MovingAvgSplit[cols_required_for_moving_avg_split_ratio]
        logger.warning("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return SKUSplit

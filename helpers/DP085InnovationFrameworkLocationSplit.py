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


def getting_actual_at_higher_level(
    df,
    ItemAttribute,
    TimeDimension,
    search_level,
    item_col,
    location_col,
    actual_col,
    req_cols,
    relevant_time_name,
    last_n_periods,
    df_keys,
):
    cols_required = req_cols + [
        item_col,
        location_col,
        relevant_time_name,
        actual_col,
    ]
    relevant_actual = pd.DataFrame(columns=cols_required)
    try:
        the_item = df[item_col].iloc[0]
        the_item_df = ItemAttribute[ItemAttribute[item_col] == the_item]

        cols_required = req_cols + [location_col, item_col, actual_col]
        the_level_aggregates = pd.DataFrame(columns=cols_required)

        for the_level in search_level:
            the_level_value = the_item_df[the_level].unique()[0]

            logger.info("--------- {} : {}".format(the_level, the_level_value))

            # check if actuals are present at the level
            filter_clause = df[the_level] == the_level_value
            req_data = df[filter_clause]

            if len(req_data) > 0:
                logger.info(
                    "--------- Actual available at {} level, shape : {} ..".format(
                        the_level, req_data.shape
                    )
                )
                fields_to_group = req_cols + [location_col, the_level]
                the_level_aggregates = (
                    req_data.groupby(fields_to_group)[actual_col].sum().reset_index()
                )

                the_level_aggregates[item_col] = the_item
                logger.info(f"the_level_aggregates head\n{the_level_aggregates.head()}")
                break
            else:
                # continue to search the next level in item hierarchy
                continue

        relevant_time_mapping = TimeDimension[
            TimeDimension[relevant_time_name].isin(last_n_periods)
        ]

        if len(the_level_aggregates) != 0:
            relevant_actual = create_cartesian_product(
                the_level_aggregates,
                relevant_time_mapping[[relevant_time_name]].drop_duplicates(),
            )

        else:
            logger.info(f"No actuals were found at search level : {the_level}")

    except Exception as e:
        logger.exception(f"Exception {e}: for slice {df_keys}")

    return relevant_actual


col_mapping = {
    "Location Split Moving Avg": float,
    "Location Split Fixed %": float,
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
    LocationSplitParameters,
    ItemAttribute,
    LocationSplitMethod,
    ItemNPIFcst,
    AssortmentFlag,
    multiprocessing_num_cores=4,
) -> pd.DataFrame:
    plugin_name = "DP085InnovationFrameworkLocationSplit"
    logger.warning("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    initiative_col = "Initiative.[Initiative]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    item_col = "Item.[Item]"
    location_col = "Location.[Location]"
    pl_location_col = "Location.[Planning Location]"

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
    location_split_history_period = "Location Split History Period"
    location_split_history_time_bucket = "Location Split History Time Bucket"
    cumulative_split_col = "Cumulative Split"
    indicator_col = "_merge"

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
    moving_avg_sku_split = "NPI Location Split Moving Avg"

    cols_required_for_moving_avg_split_ratio = (
        [version_col] + dimensions + [initiative_col, TimeLevel, moving_avg_sku_split]
    )
    MovingAvgSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

    SKUSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

    try:
        if AssortmentFlag is None:
            cols_req = (
                [version_col, initiative_col, item_col] + dimensions + ["NPI Assortment Flag"]
            )
            AssortmentFlag = pd.DataFrame(columns=cols_req)

        if Actual is None:
            cols_req = [version_col] + dimensions + [pl_location_col, TimeLevel, HistoryMeasure]
            Actual = pd.DataFrame(columns=cols_req)

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
        AssortmentFlag.drop(columns=version_col, axis=1, inplace=True)
        TimeDimension[week_col] = TimeDimension[week_col].astype(str)

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

        # define item hierarchy
        search_level = [
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
        ]
        logger.info("search_level : {}".format(search_level))

        # Filter relevant columns from Item Attribute
        req_cols = [
            item_col,
            # intro_date_col,
            # disc_date_col,
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
        ]
        ItemAttribute = ItemAttribute[req_cols].drop_duplicates()

        if (len(LocationSplitParameters) != 0) and (len(Actual) != 0):
            Actual = Actual.merge(
                ItemAttribute,
                on=item_col,
                how="inner",
            )

            # identify combinations with consensus fcst but no actual
            cust_grp_item = [
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                pl_location_col,
                item_col,
            ]
            intersections_with_consensus_fcst = ItemNPIFcst[
                cust_grp_item + [initiative_col]
            ].drop_duplicates()

            intersections_with_actual = Actual[cust_grp_item].drop_duplicates()

            # perform a left join, with indicator column
            merged_df = intersections_with_consensus_fcst.merge(
                intersections_with_actual, how="left", indicator=True
            )

            merged_df = merged_df.merge(
                LocationSplitMethod[cust_grp_item].drop_duplicates(),
                on=cust_grp_item,
                how="inner",
            )

            MovingAvgPeriods = int(LocationSplitParameters[location_split_history_period][0])
            logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))

            if LocationSplitParameters[location_split_history_time_bucket][0] == "Planning Month":
                relevant_time_name = pl_month_col
                relevant_time_key = pl_month_key_col

            elif LocationSplitParameters[location_split_history_time_bucket][0] == "Month":
                relevant_time_name = month_col
                relevant_time_key = month_key_col

            elif LocationSplitParameters[location_split_history_time_bucket][0] == "Week":
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

            Actual = Actual.merge(TimeDimension, on=partial_week_col, how="inner")
            Actual = Actual[Actual[relevant_time_name].isin(last_n_periods)]
            cols_req = (
                [version_col]
                + list(set(dimensions).union(set(req_cols)))
                + [partial_week_col, actual_col]
            )
            Actual = Actual[cols_req].drop_duplicates()

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

            intersections_with_consensus_fcst_df = merged_df[
                merged_df[indicator_col] == "left_only"
            ]
            intersections_with_consensus_fcst_and_actual_df = merged_df[
                merged_df[indicator_col] == "both"
            ]

            dimensions = dimensions + [initiative_col]
            req_cols = [
                pl_account_col,
                pl_channel_col,
                pl_region_col,
                pl_pnl_col,
                pl_demand_domain_col,
                pl_location_col,
            ]
            cols_required = dimensions + [relevant_time_name, actual_col]

            outputData = pd.DataFrame(columns=cols_required)
            if len(intersections_with_consensus_fcst_df) != 0:
                intersections_with_consensus_fcst_df = intersections_with_consensus_fcst_df.merge(
                    Actual.drop(columns=[item_col]),
                    on=req_cols,
                    how="inner",
                )

                logger.info("Calculating actual at higher level ...")
                all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                    delayed(getting_actual_at_higher_level)(
                        df=df,
                        ItemAttribute=ItemAttribute,
                        TimeDimension=TimeDimension,
                        search_level=search_level,
                        item_col=item_col,
                        location_col=location_col,
                        actual_col=actual_col,
                        req_cols=req_cols + [initiative_col],
                        relevant_time_name=relevant_time_name,
                        last_n_periods=last_n_periods,
                        df_keys=df_keys,
                    )
                    for name, df in intersections_with_consensus_fcst_df.groupby(
                        cust_grp_item + [initiative_col]
                    )
                )

                # Concat all results to one dataframe
                outputData = concat_to_dataframe(all_results)

            if len(intersections_with_consensus_fcst_and_actual_df) != 0:
                intersections_with_consensus_fcst_and_actual_df = (
                    intersections_with_consensus_fcst_and_actual_df.merge(
                        Actual,
                        on=cust_grp_item,
                        how="inner",
                    )
                )

                intersections_with_consensus_fcst_and_actual_df = (
                    intersections_with_consensus_fcst_and_actual_df.merge(
                        TimeDimension[[relevant_time_name, partial_week_col]].drop_duplicates(),
                        on=partial_week_col,
                        how="inner",
                    )
                )
                intersections_with_consensus_fcst_and_actual_df = (
                    intersections_with_consensus_fcst_and_actual_df[cols_required]
                )

            else:
                intersections_with_consensus_fcst_and_actual_df = pd.DataFrame(
                    columns=cols_required
                )

            relevant_intersections = pd.concat(
                [
                    outputData[cols_required],
                    intersections_with_consensus_fcst_and_actual_df,
                ]
            )

            if relevant_intersections.empty:
                logger.warning(
                    "Returning empty dataframe as there are no relevant intersections to process ..."
                )
                return SKUSplit

            req_cols = req_cols + [initiative_col]
            cust_grp_item = cust_grp_item + [initiative_col]

            relevant_intersections = (
                relevant_intersections.groupby(dimensions + [relevant_time_name])[actual_col]
                .sum()
                .reset_index()
            )

            # Fill missing dates
            relevant_actual_nas_filled = fill_missing_dates(
                actual=relevant_intersections,
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

            logger.info("Aggregating actual at item level...")
            filter_clause = cust_grp_item + [initiative_col, relevant_time_name]
            outputData[actual_agg_col] = outputData.groupby(filter_clause, observed=True)[
                actual_col
            ].transform("sum")

            logger.info("Calculating split % at item level...")
            outputData[moving_avg_sku_split] = np.where(
                outputData[actual_agg_col] != 0,
                outputData[actual_col] / outputData[actual_agg_col],
                0,
            )

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
            output_df = outputData.merge(
                relevant_time_mapping,
                on=relevant_time_name,
                how="inner",
            )

            output_df = output_df.merge(
                ItemAttribute,
                on=item_col,
                how="inner",
            )

            # get only those partial weeks which are in forecastbucket
            output_df = output_df[
                (output_df[partial_week_key_col] >= start_date_key)
                & (output_df[partial_week_key_col] <= end_date_key)
            ]
            output_df[version_col] = input_version

            # filter intersections for which assortment flag is 1
            output_df = output_df.merge(
                AssortmentFlag,
                on=dimensions,
                how="inner",
            )

            # normalizing the moving avg split %
            output_df[cumulative_split_col] = output_df.groupby(
                req_cols + [partial_week_col, item_col],
                observed=True,
            )[moving_avg_sku_split].transform("sum")
            output_df[moving_avg_sku_split] = np.where(
                output_df[cumulative_split_col] != 0,
                output_df[moving_avg_sku_split] / output_df[cumulative_split_col],
                np.nan,
            )

            if len(output_df) != 0:
                MovingAvgSplit = output_df[cols_required_for_moving_avg_split_ratio]

            else:
                MovingAvgSplit = MovingAvgSplit

        SKUSplit = MovingAvgSplit[cols_required_for_moving_avg_split_ratio]
        logger.warning("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return SKUSplit

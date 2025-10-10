import logging
from datetime import timedelta

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


class Constants:
    # Configurables
    version_col = "Version.[Version Name]"
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

    actual_agg_col = "Actual Agg"
    item_split_history_period = "Item Split History Period"
    item_split_history_time_bucket = "Item Split History Time Bucket"
    intro_date_col = "Item.[Item Intro Date]"
    disc_date_col = "Item.[Item Disc Date]"
    from_date_col = "Item Split From Date"
    to_date_col = "Item Split To Date"
    split_col = "Item Split % Input Normalized"
    cumulative_split_col = "Cumulative Split"
    item_split_method_final = "Item Split Method Final"
    consensus_fcst = "Consensus Fcst"

    temp_split = "Fixed Temp Split %"

    # output measure
    moving_avg_sku_split = "Item Split Moving Avg"
    fixed_per_sku_split = "Item Split Fixed %"
    item_split_final = "Item Split Final"
    item_consensus_fcst = "Item Consensus Fcst"


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
    actual_col,
    pl_level_cols,
):
    pl_item = relevant_actual[Constants.pl_item_col].unique()
    the_item_mapping = ItemMapping[ItemMapping[Constants.pl_item_col].isin(pl_item)]

    items_with_actual = set(relevant_actual[Constants.item_col].unique())
    items_within_pl_item = set(the_item_mapping[Constants.item_col].unique())
    items_with_no_actual_list = list(items_within_pl_item - items_with_actual)

    cols_required = pl_level_cols + [
        Constants.item_col,
        Constants.partial_week_col,
        Constants.partial_week_key_col,
        Constants.moving_avg_sku_split,
    ]
    output = pd.DataFrame(columns=cols_required)

    if len(items_with_no_actual_list) != 0:
        items_with_no_actual = the_item_mapping[
            the_item_mapping[Constants.item_col].isin(items_with_no_actual_list)
        ]

        # consider items with assortment 1
        items_with_no_actual = items_with_no_actual.merge(
            cust_grp_item_with_assortment,
            on=[Constants.item_col],
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
            (
                items_with_no_actual[Constants.partial_week_key_col]
                >= items_with_no_actual[Constants.intro_date_col]
            )
            & (
                items_with_no_actual[Constants.partial_week_key_col]
                <= items_with_no_actual[Constants.disc_date_col]
            )
        ]

        # getting max of disc date for items with actual
        filtered_df = the_item_mapping[the_item_mapping[Constants.item_col].isin(items_with_actual)]
        disc_date = filtered_df[Constants.disc_date_col].max()

        # filter intersections for which partial week lies after disc date of items with actual
        items_with_no_actual = items_with_no_actual[
            items_with_no_actual[Constants.partial_week_key_col] > disc_date
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
        items_with_no_actual[Constants.actual_agg_col] = items_with_no_actual.groupby(
            [Constants.partial_week_col] + pl_level_cols,
            observed=True,
        )[actual_col].transform("sum")
        items_with_no_actual[Constants.moving_avg_sku_split] = np.where(
            items_with_no_actual[Constants.actual_agg_col] != 0,
            items_with_no_actual[actual_col] / items_with_no_actual[Constants.actual_agg_col],
            np.nan,
        )

        output = items_with_no_actual[cols_required]

    return output


col_mapping = {
    "Item Split Moving Avg": float,
    "Item Split Fixed %": float,
    "Item Split Final": float,
    "Item Consensus Fcst": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    RunMovingAverage,
    ForecastBucket,
    HistoryMeasure,
    Grains,
    TimeLevel,
    Actual,
    CurrentTimePeriod,
    TimeDimension,
    df_keys,
    ItemSplitParameters,
    ItemSplitPercentInputNormalized,
    ItemMapping,
    ItemSplitMethod,
    ConsensusFcst,
    AssortmentFlag,
    MovingAverageSplitOutput,
    multiprocessing_num_cores=4,
) -> pd.DataFrame:
    plugin_name = "DP052PopulateSKUSplitPercent"
    logger.warning("Executing {} for slice {} ...".format(plugin_name, df_keys))

    actual_col = str(HistoryMeasure)

    logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    cols_required_for_moving_avg_split_ratio = (
        [Constants.version_col] + dimensions + [TimeLevel, Constants.moving_avg_sku_split]
    )
    cols_required_for_fixed_split_ratio = (
        [Constants.version_col] + dimensions + [TimeLevel, Constants.fixed_per_sku_split]
    )
    cols_required_for_output = (
        [Constants.version_col]
        + dimensions
        + [
            TimeLevel,
            Constants.moving_avg_sku_split,
            Constants.fixed_per_sku_split,
            Constants.item_split_final,
            Constants.item_consensus_fcst,
        ]
    )
    MovingAvgSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

    SKUSplit = pd.DataFrame(columns=cols_required_for_output)
    try:
        common_cols = [x for x in dimensions if x != Constants.item_col]
        if Actual is None:
            cols_req = (
                [Constants.version_col] + dimensions + [Constants.partial_week_col, actual_col]
            )
            Actual = pd.DataFrame(columns=cols_req)

        if ConsensusFcst is None:
            cols_req = (
                [Constants.version_col]
                + common_cols
                + [Constants.pl_item_col, Constants.partial_week_col, actual_col]
            )
            ConsensusFcst = pd.DataFrame(columns=cols_req)

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

        if ItemSplitPercentInputNormalized is None:
            cols_req = (
                [Constants.version_col]
                + dimensions
                + [Constants.partial_week_col, Constants.split_col]
            )
            ItemSplitPercentInputNormalized = pd.DataFrame(columns=cols_req)

        input_version = AssortmentFlag[Constants.version_col].iloc[0]
        TimeDimension[Constants.week_col] = TimeDimension[Constants.week_col].astype(str)
        min_date = TimeDimension[Constants.partial_week_key_col].min()
        max_date = TimeDimension[Constants.partial_week_key_col].max()
        ItemMapping[Constants.intro_date_col] = pd.to_datetime(
            ItemMapping[Constants.intro_date_col],
            errors="coerce",
        )
        ItemMapping[Constants.disc_date_col] = pd.to_datetime(
            ItemMapping[Constants.disc_date_col],
            errors="coerce",
        )
        ItemMapping[Constants.intro_date_col] = ItemMapping[Constants.intro_date_col].fillna(
            min_date
        )
        ItemMapping[Constants.disc_date_col] = ItemMapping[Constants.disc_date_col].fillna(max_date)

        # getting correct intro and disco date if falls in between weeks
        ItemMapping = pd.merge_asof(
            ItemMapping.sort_values(by=Constants.intro_date_col),
            TimeDimension[[Constants.week_key_col]].drop_duplicates(),
            left_on=Constants.intro_date_col,
            right_on=Constants.week_key_col,
            direction="backward",
        )
        ItemMapping.drop(columns=Constants.intro_date_col, axis=1, inplace=True)
        ItemMapping.rename(columns={Constants.week_key_col: Constants.intro_date_col}, inplace=True)
        ItemMapping[Constants.intro_date_col] = pd.to_datetime(
            ItemMapping[Constants.intro_date_col], utc=True
        ).dt.tz_localize(None)

        ItemMapping[Constants.disc_date_col] = pd.to_datetime(
            ItemMapping[Constants.disc_date_col], utc=True
        ).dt.tz_localize(None)

        # collect datetime key columns
        key_cols = [
            Constants.pl_month_key_col,
            Constants.month_key_col,
            Constants.week_key_col,
            Constants.partial_week_key_col,
        ]
        logger.info("Converting key cols to datetime format ...")

        # convert to datetime
        TimeDimension[key_cols] = TimeDimension[key_cols].apply(pd.to_datetime)
        for the_col in key_cols:
            if the_col in TimeDimension.columns:
                logger.info(f"Converting {the_col} from datetime64[UTC] to datetime ...")
                TimeDimension[the_col] = pd.to_datetime(
                    TimeDimension[the_col], utc=True
                ).dt.tz_localize(None)

        ForecastBucket[Constants.partial_week_key_col] = pd.to_datetime(
            ForecastBucket[Constants.partial_week_key_col], utc=True
        ).dt.tz_localize(None)

        # getting correct intro and disco date if falls in between weeks
        ItemMapping = pd.merge_asof(
            ItemMapping.sort_values(by=Constants.intro_date_col),
            TimeDimension[[Constants.week_key_col]].drop_duplicates(),
            left_on=Constants.intro_date_col,
            right_on=Constants.week_key_col,
            direction="backward",
        )
        ItemMapping.drop(columns=Constants.intro_date_col, axis=1, inplace=True)
        ItemMapping.rename(columns={Constants.week_key_col: Constants.intro_date_col}, inplace=True)

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
        start_date_key = ForecastBucket[Constants.partial_week_key_col].min()
        end_date_key = ForecastBucket[Constants.partial_week_key_col].max()

        ForecastBucket.sort_values(Constants.partial_week_key_col, inplace=True)
        # end_date = ForecastBucket.tail(1)[Constants.partial_week_col].iloc[0]

        # getting min and max dates from TimeDimension
        # to get default intro and disc dates
        intro_date_key = TimeDimension[Constants.partial_week_key_col].min()
        disc_date_key = TimeDimension[Constants.partial_week_key_col].max()

        ItemSplitPercentInputNormalized.drop(Constants.version_col, axis=1, inplace=True)
        ItemSplitPercentInputNormalized[Constants.split_col].fillna(0, inplace=True)

        # join on time dimension to get partial week key for sorting (Fixed Item Split)
        ItemSplitPercentInputNormalized = ItemSplitPercentInputNormalized.merge(
            TimeDimension[
                [Constants.partial_week_col, Constants.partial_week_key_col]
            ].drop_duplicates(),
            on=Constants.partial_week_col,
            how="inner",
        )

        # sort on intersection and partial week key
        ItemSplitPercentInputNormalized.sort_values(
            dimensions + [Constants.partial_week_key_col], inplace=True
        )
        ItemSplitPercentInputNormalized[Constants.from_date_col] = ItemSplitPercentInputNormalized[
            Constants.partial_week_key_col
        ]

        # drop the partial week key column after sorting
        ItemSplitPercentInputNormalized.drop(
            columns=[Constants.partial_week_key_col, Constants.partial_week_col],
            axis=1,
            inplace=True,
        )

        # # rename the partial week column to start date
        # ItemSplitPercentInputNormalized.rename(
        #     columns={Constants.partial_week_col: Constants.from_date_col}, inplace=True
        # )

        # create the end dates by using start date of next intersection
        ItemSplitPercentInputNormalized[Constants.to_date_col] = pd.to_datetime(
            ItemSplitPercentInputNormalized.groupby(dimensions)[Constants.from_date_col].shift(-1)
        ) - timedelta(days=1)

        # fill the NAs with max date from horizon
        ItemSplitPercentInputNormalized[Constants.to_date_col].fillna(end_date_key, inplace=True)

        # convert to datetime
        ItemSplitPercentInputNormalized[Constants.from_date_col] = pd.to_datetime(
            ItemSplitPercentInputNormalized[Constants.from_date_col]
        )
        ItemSplitPercentInputNormalized[Constants.to_date_col] = pd.to_datetime(
            ItemSplitPercentInputNormalized[Constants.to_date_col]
        )

        # setting default intro and disc dates
        ItemMapping[Constants.intro_date_col].fillna(intro_date_key, inplace=True)
        ItemMapping[Constants.disc_date_col].fillna(disc_date_key, inplace=True)

        # get new items and calculate split ratio
        pl_level_cols = [
            Constants.pl_item_col,
            Constants.pl_account_col,
            Constants.pl_pnl_col,
            Constants.pl_location_col,
            Constants.pl_channel_col,
            Constants.pl_region_col,
            Constants.pl_demand_domain_col,
        ]

        if (len(ItemSplitParameters) != 0) and (len(Actual) != 0) and RunMovingAverage:
            MovingAvgPeriods = int(ItemSplitParameters[Constants.item_split_history_period][0])
            logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))

            if ItemSplitParameters[Constants.item_split_history_time_bucket][0] == "Planning Month":
                relevant_time_name = Constants.pl_month_col
                relevant_time_key = Constants.pl_month_key_col

            elif ItemSplitParameters[Constants.item_split_history_time_bucket][0] == "Month":
                relevant_time_name = Constants.month_col
                relevant_time_key = Constants.month_key_col

            elif ItemSplitParameters[Constants.item_split_history_time_bucket][0] == "Week":
                relevant_time_name = Constants.week_col
                relevant_time_key = Constants.week_key_col

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
                        Constants.partial_week_col,
                        Constants.partial_week_key_col,
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            req_cols = (
                [Constants.version_col] + dimensions + [Constants.partial_week_col, actual_col]
            )
            Actual = Actual[req_cols]

            # merge with TimeDimension to get relevant time
            Actual = Actual.merge(
                relevant_time_mapping,
                on=Constants.partial_week_col,
                how="left",
            )

            # Filter out relevant history
            Actual = Actual[Actual[relevant_time_name].isin(last_n_periods)]

            Actual = Actual.merge(
                ItemMapping[[Constants.item_col, Constants.pl_item_col]],
                on=Constants.item_col,
                how="inner",
            )

            cust_grp_pl_item_in_actual = (
                Actual[pl_level_cols].drop_duplicates().reset_index(drop=True)
            )
            cust_grp_item_with_assortment = (
                AssortmentFlag[
                    [
                        Constants.item_col,
                        Constants.pl_account_col,
                        Constants.pl_pnl_col,
                        Constants.pl_location_col,
                        Constants.pl_channel_col,
                        Constants.pl_region_col,
                        Constants.pl_demand_domain_col,
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # getting planning item for each item
            cust_grp_pl_item_with_assortment = cust_grp_item_with_assortment.merge(
                ItemMapping,
                on=[Constants.item_col],
                how="inner",
            )

            AssortmentFlag.drop(Constants.version_col, axis=1, inplace=True)
            Actual = Actual.merge(
                AssortmentFlag,
                on=dimensions,
                how="inner",
            )

            new_intersections = pd.merge(
                cust_grp_pl_item_with_assortment,
                cust_grp_pl_item_in_actual,
                on=pl_level_cols,
                how="outer",
                indicator=True,
            )
            new_intersections = new_intersections[new_intersections["_merge"] == "left_only"]

            if len(new_intersections) != 0:
                new_intersections[Constants.item_count] = new_intersections.groupby(
                    pl_level_cols, observed=True
                )[Constants.item_col].transform("count")

                # distributing split ratio for all new items within planning item
                new_intersections[Constants.moving_avg_sku_split] = (
                    1 / new_intersections[Constants.item_count]
                )

            # to get future periods for relevant time
            relevant_future_periods = ForecastBucket.merge(
                TimeDimension[[Constants.partial_week_col, relevant_time_name]].drop_duplicates(),
                on=Constants.partial_week_col,
                how="inner",
            )

            # get test period dates
            future_n_periods = list(relevant_future_periods[relevant_time_name].unique())
            logger.info("future_n_periods : {} ...".format(future_n_periods))
            FuturePeriods = int(len(future_n_periods))

            # cap negatives in HistoryMeasure
            Actual[HistoryMeasure] = np.where(Actual[HistoryMeasure] < 0, 0, Actual[HistoryMeasure])

            # process only those intersections present in ItemSplitMethod
            actual_df = Actual.merge(
                ItemSplitMethod,
                on=pl_level_cols,
                how="inner",
            )

            # getting relevant columns to concat
            cols_required = pl_level_cols + [
                Constants.item_col,
                Constants.partial_week_col,
                Constants.partial_week_key_col,
                Constants.moving_avg_sku_split,
            ]
            output_df = pd.DataFrame(columns=cols_required)

            if len(actual_df) != 0:
                # aggregate data to relevant time level
                actual_df = (
                    actual_df.groupby(
                        [
                            Constants.pl_account_col,
                            Constants.pl_pnl_col,
                            Constants.pl_location_col,
                            Constants.pl_channel_col,
                            Constants.pl_region_col,
                            Constants.pl_demand_domain_col,
                            Constants.pl_item_col,
                            Constants.item_col,
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
                    on=Constants.item_col,
                    how="inner",
                ).drop_duplicates()

                logger.info("Aggregating actual at planning item level...")
                group_by_cols = pl_level_cols + [relevant_time_name]
                outputData[Constants.actual_agg_col] = outputData.groupby(
                    group_by_cols, observed=True
                )[actual_col].transform("sum")

                logger.info("Calculating split % at planning item level...")
                outputData[Constants.moving_avg_sku_split] = np.where(
                    outputData[Constants.actual_agg_col] != 0,
                    outputData[actual_col] / outputData[Constants.actual_agg_col],
                    0,
                )

                output_df = outputData.merge(
                    relevant_time_mapping,
                    on=relevant_time_name,
                    how="left",
                )

                # get only those partial weeks which satisfy intro & disc dates and are in forecastbucket
                output_df = output_df[
                    (
                        output_df[Constants.partial_week_key_col]
                        >= output_df[Constants.intro_date_col]
                    )
                    & (
                        output_df[Constants.partial_week_key_col]
                        <= output_df[Constants.disc_date_col]
                    )
                ]

                output_df = output_df[
                    (output_df[Constants.partial_week_key_col] >= start_date_key)
                    & (output_df[Constants.partial_week_key_col] <= end_date_key)
                ]

                output_df = output_df[cols_required]

            # calculating splits for items with no actual
            items_with_no_actual_split = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                delayed(split_for_items_with_no_actual)(
                    relevant_actual=df,
                    ForecastBucket=ForecastBucket,
                    ItemMapping=ItemMapping,
                    cust_grp_item_with_assortment=cust_grp_item_with_assortment,
                    actual_col=actual_col,
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
                # filter intersections present in ConsensusFcst
                output_df = output_df.merge(
                    ConsensusFcst,
                    on=[Constants.partial_week_col] + pl_level_cols,
                    how="inner",
                ).drop_duplicates()

                # normalizing the moving avg split %
                output_df[Constants.cumulative_split_col] = output_df.groupby(
                    [Constants.partial_week_col] + pl_level_cols,
                    observed=True,
                )[Constants.moving_avg_sku_split].transform("sum")

                output_df[Constants.moving_avg_sku_split] = np.where(
                    output_df[Constants.cumulative_split_col] != 0,
                    output_df[Constants.moving_avg_sku_split]
                    / output_df[Constants.cumulative_split_col],
                    np.nan,
                )
                MovingAvgSplit = output_df[
                    cols_required_for_moving_avg_split_ratio + [Constants.pl_item_col]
                ]

            relevant_data = pd.DataFrame(
                columns=cols_required_for_moving_avg_split_ratio + [Constants.pl_item_col]
            )

            if len(new_intersections) != 0:
                # creating cartesian product to get all partial weeks
                relevant_data = create_cartesian_product(
                    new_intersections,
                    ForecastBucket,
                )

                # getting only those partial weeks which satisfy corresponding intro and disc dates
                relevant_data = relevant_data[
                    (
                        relevant_data[Constants.partial_week_key_col]
                        >= relevant_data[Constants.intro_date_col]
                    )
                    & (
                        relevant_data[Constants.partial_week_key_col]
                        <= relevant_data[Constants.disc_date_col]
                    )
                ]

                relevant_data = relevant_data.merge(
                    ConsensusFcst,
                )

                relevant_data[Constants.cumulative_split_col] = relevant_data.groupby(
                    [Constants.partial_week_col] + pl_level_cols,
                    observed=True,
                )[Constants.moving_avg_sku_split].transform("sum")

                relevant_data[Constants.moving_avg_sku_split] = np.where(
                    relevant_data[Constants.cumulative_split_col] != 0,
                    relevant_data[Constants.moving_avg_sku_split]
                    / relevant_data[Constants.cumulative_split_col],
                    np.nan,
                )
                relevant_data[Constants.version_col] = input_version

                relevant_data = relevant_data[
                    cols_required_for_moving_avg_split_ratio + [Constants.pl_item_col]
                ]

            MovingAvgSplit = pd.concat([MovingAvgSplit, relevant_data])

        if MovingAvgSplit.empty:
            MovingAvgSplit = MovingAverageSplitOutput.copy()
            MovingAvgSplit = MovingAvgSplit[
                cols_required_for_moving_avg_split_ratio + [Constants.pl_item_col]
            ]

        if len(ItemSplitPercentInputNormalized) == 0:
            FixedSplit = pd.DataFrame(
                columns=cols_required_for_fixed_split_ratio + [Constants.pl_item_col, "Flag"]
            )

        else:
            ItemSplitPercentInputNormalized = ItemSplitPercentInputNormalized.merge(
                ItemMapping,
                on=Constants.item_col,
                how="left",
            )
            ItemSplitPercentInputNormalized["Flag"] = np.where(
                ItemSplitPercentInputNormalized[Constants.from_date_col] <= start_date_key,
                False,
                True,
            )

            # creating cartesian product to get all partial weeks
            relevant_data_raw = create_cartesian_product(
                ItemSplitPercentInputNormalized,
                ForecastBucket,
            )

            cols_to_convert = [
                Constants.partial_week_key_col,
                Constants.intro_date_col,
                Constants.disc_date_col,
                Constants.from_date_col,
                Constants.to_date_col,
            ]

            for the_col in cols_to_convert:
                if the_col in relevant_data_raw.columns:
                    logger.info(f"Converting {the_col} from datetime64[UTC] to datetime ...")
                    relevant_data_raw[the_col] = pd.to_datetime(
                        relevant_data_raw[the_col], utc=True
                    ).dt.tz_localize(None)

            # get that data which satisfy intro and disc dates
            # also satisfy corresponding from & to dates
            relevant_data_raw = relevant_data_raw[
                (
                    relevant_data_raw[Constants.partial_week_key_col]
                    >= relevant_data_raw[Constants.intro_date_col]
                )
                & (
                    relevant_data_raw[Constants.partial_week_key_col]
                    <= relevant_data_raw[Constants.disc_date_col]
                )
            ]

            # filter intersections present in ItemSplitMethod
            relevant_data_raw = relevant_data_raw[
                (
                    relevant_data_raw[Constants.partial_week_key_col]
                    >= relevant_data_raw[Constants.from_date_col]
                )
                & (
                    relevant_data_raw[Constants.partial_week_key_col]
                    <= relevant_data_raw[Constants.to_date_col]
                )
            ]

            relevant_data_raw.drop_duplicates(inplace=True)
            relevant_data_raw.reset_index(drop=True, inplace=True)

            # getting cumulative sum of split %, to normalize it later
            logger.info("calculating cumulative sum of split %, to normalize it")
            relevant_data_raw[Constants.cumulative_split_col] = relevant_data_raw.groupby(
                [Constants.partial_week_col] + pl_level_cols, observed=True
            )[Constants.split_col].transform("sum")
            relevant_data_raw[Constants.fixed_per_sku_split] = np.where(
                relevant_data_raw[Constants.cumulative_split_col] != 0,
                relevant_data_raw[Constants.split_col]
                / relevant_data_raw[Constants.cumulative_split_col],
                np.nan,
            )
            relevant_data_raw[Constants.version_col] = input_version

            # filter intersections present in ConsensusFcst
            relevant_data_raw = relevant_data_raw.merge(
                ConsensusFcst,
                on=[
                    Constants.version_col,
                    Constants.partial_week_col,
                ]
                + pl_level_cols,
                how="inner",
            )
            FixedSplit = relevant_data_raw[
                cols_required_for_fixed_split_ratio + [Constants.pl_item_col, "Flag"]
            ]

        merge_cols = [
            Constants.version_col,
            Constants.partial_week_col,
            Constants.pl_item_col,
        ] + dimensions
        SKUSplit = pd.merge(
            MovingAvgSplit,
            FixedSplit,
            on=merge_cols,
            how="outer",
        )
        SKUSplit = SKUSplit.merge(
            ItemSplitMethod,
        )
        SKUSplit = SKUSplit.merge(
            ConsensusFcst,
        )
        SKUSplit[Constants.temp_split] = np.where(
            SKUSplit[Constants.fixed_per_sku_split].isna() & SKUSplit["Flag"],
            SKUSplit[Constants.moving_avg_sku_split],
            SKUSplit[Constants.fixed_per_sku_split],
        )
        SKUSplit[Constants.item_split_final] = np.where(
            SKUSplit[Constants.item_split_method_final] == "Moving Avg",
            SKUSplit[Constants.moving_avg_sku_split],
            SKUSplit[Constants.temp_split],
        )
        SKUSplit[Constants.item_consensus_fcst] = (
            SKUSplit[Constants.consensus_fcst] * SKUSplit[Constants.item_split_final]
        )

        SKUSplit = SKUSplit[cols_required_for_output].drop_duplicates()
        logger.warning("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return SKUSplit

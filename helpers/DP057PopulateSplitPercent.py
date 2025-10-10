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


class Constants:
    # Configurables
    version_col = "Version.[Version Name]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    item_col = "Item.[Item]"
    pl_location_col = "Location.[Planning Location]"

    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_col = "Time.[Week]"
    week_key_col = "Time.[WeekKey]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"
    pl_month_col = "Time.[Planning Month]"
    pl_month_key_col = "Time.[PlanningMonthKey]"

    actual_agg_col = "Actual Agg"
    location_split_history_period = "Location Split History Period"
    location_split_history_time_bucket = "Location Split History Time Bucket"
    from_date_col = "Location Split From Date"
    to_date_col = "Location Split To Date"
    split_col = "Location Split % Input Normalized"
    cumulative_split_col = "Cumulative Split"
    method_col = "Location Split Method Final"
    final_fcst = "Final Fcst"
    split_final = "Location Split Final"
    item_consensus_fcst = "Item Consensus Fcst"
    active = "Active"

    temp_fixed_split = "Temp Fixed Split"

    # output measure
    moving_avg_sku_split = "Location Split Moving Avg"
    fixed_per_sku_split = "Location Split Fixed %"


col_mapping = {
    "Location Split Moving Avg": float,
    "Location Split Fixed %": float,
    "Final Fcst": float,
    "Location Split Final": float,
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
    LocationSplitParameters,
    LocationSplitPercentInputNormalized,
    LocationSplitMethod,
    ItemConsensusFcst,
    AssortmentFlag,
    MovingAverageSplitOutput,
    multiprocessing_num_cores=4,
) -> pd.DataFrame:
    plugin_name = "DP057PopulateSplitPercent"
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
            Constants.final_fcst,
            Constants.split_final,
        ]
    )
    MovingAvgSplit = pd.DataFrame(columns=cols_required_for_moving_avg_split_ratio)

    SKUSplit = pd.DataFrame(columns=cols_required_for_output)

    try:
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

        input_version = AssortmentFlag[Constants.version_col].iloc[0]
        TimeDimension[Constants.week_col] = TimeDimension[Constants.week_col].astype(str)

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

        LocationSplitPercentInputNormalized.drop(Constants.version_col, axis=1, inplace=True)
        LocationSplitPercentInputNormalized[Constants.split_col].fillna(0, inplace=True)
        AssortmentFlag.drop(Constants.version_col, axis=1, inplace=True)

        # join on time dimension to get partial week key for sorting
        LocationSplitPercentInputNormalized = LocationSplitPercentInputNormalized.merge(
            TimeDimension[
                [Constants.partial_week_col, Constants.partial_week_key_col]
            ].drop_duplicates(),
            on=Constants.partial_week_col,
            how="inner",
        )

        # sort on intersection and partial week key
        LocationSplitPercentInputNormalized.sort_values(
            dimensions + [Constants.partial_week_key_col],
            inplace=True,
        )
        LocationSplitPercentInputNormalized[Constants.from_date_col] = (
            LocationSplitPercentInputNormalized[Constants.partial_week_key_col]
        )

        # drop the partial week key column after sorting
        LocationSplitPercentInputNormalized.drop(
            columns=[Constants.partial_week_key_col, Constants.partial_week_col],
            axis=1,
            inplace=True,
        )

        # # rename the partial week column to start date
        # LocationSplitPercentInputNormalized.rename(
        #     columns={Constants.partial_week_col: Constants.from_date_col}, inplace=True
        # )

        # create the end dates by using start date of next intersection
        LocationSplitPercentInputNormalized[Constants.to_date_col] = pd.to_datetime(
            LocationSplitPercentInputNormalized.groupby(dimensions)[Constants.from_date_col].shift(
                -1
            )
        ) - timedelta(days=1)

        # fill the NAs with max date from horizon
        LocationSplitPercentInputNormalized[Constants.to_date_col].fillna(
            end_date_key, inplace=True
        )

        # convert to datetime
        LocationSplitPercentInputNormalized[Constants.from_date_col] = pd.to_datetime(
            LocationSplitPercentInputNormalized[Constants.from_date_col]
        )
        LocationSplitPercentInputNormalized[Constants.to_date_col] = pd.to_datetime(
            LocationSplitPercentInputNormalized[Constants.to_date_col]
        )

        # getting only those intersections for which consensus fcst is present
        req_cols = [
            Constants.pl_location_col,
            Constants.pl_channel_col,
            Constants.pl_pnl_col,
            Constants.pl_account_col,
            Constants.pl_region_col,
            Constants.pl_demand_domain_col,
            Constants.item_col,
        ]

        if (len(LocationSplitParameters) != 0) and (len(Actual) != 0) and RunMovingAverage:
            Actual = Actual.merge(
                ItemConsensusFcst[req_cols].drop_duplicates(),
                on=req_cols,
                how="inner",
            )

            MovingAvgPeriods = int(
                LocationSplitParameters[Constants.location_split_history_period][0]
            )
            logger.info("MovingAvgPeriods : {} ...".format(MovingAvgPeriods))

            if (
                LocationSplitParameters[Constants.location_split_history_time_bucket][0]
                == "Planning Month"
            ):
                relevant_time_name = Constants.pl_month_col
                relevant_time_key = Constants.pl_month_key_col

            elif (
                LocationSplitParameters[Constants.location_split_history_time_bucket][0] == "Month"
            ):
                relevant_time_name = Constants.month_col
                relevant_time_key = Constants.month_key_col

            elif LocationSplitParameters[Constants.location_split_history_time_bucket][0] == "Week":
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

            # filter intersections for which assortment flag is 1
            actual_df = Actual.merge(
                AssortmentFlag,
                on=dimensions,
                how="inner",
            )

            # process only those intersections present in LocationSplitMethod
            actual_df = actual_df.merge(
                LocationSplitMethod,
                on=[Constants.version_col] + req_cols,
                how="inner",
            )

            # merge with TimeDimension to get relevant time
            actual_df = actual_df.merge(
                TimeDimension[
                    [
                        relevant_time_name,
                        Constants.partial_week_col,
                        Constants.partial_week_key_col,
                    ]
                ],
                on=Constants.partial_week_col,
                how="left",
            )

            actual_df = actual_df[actual_df[relevant_time_name].isin(last_n_periods)]

            if len(actual_df) != 0:
                # aggregate data to relevant time level
                actual_df = (
                    actual_df.groupby(dimensions + [relevant_time_name])[actual_col]
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

                logger.info("Aggregating actual at item level...")
                group_by_cols = req_cols + [relevant_time_name]
                outputData[Constants.actual_agg_col] = outputData.groupby(
                    group_by_cols, observed=True
                )[actual_col].transform("sum")

                logger.info("Calculating split % at item level...")
                outputData[Constants.moving_avg_sku_split] = np.where(
                    outputData[Constants.actual_agg_col] != 0,
                    outputData[actual_col] / outputData[Constants.actual_agg_col],
                    0,
                )

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
                output_df = outputData.merge(
                    relevant_time_mapping,
                    on=relevant_time_name,
                    how="left",
                )

                output_df = output_df[
                    (output_df[Constants.partial_week_key_col] >= start_date_key)
                    & (output_df[Constants.partial_week_key_col] <= end_date_key)
                ]
                output_df[Constants.version_col] = input_version

                # normalizing the moving avg split %
                output_df[Constants.cumulative_split_col] = output_df.groupby(
                    [Constants.partial_week_col] + req_cols,
                    observed=True,
                )[Constants.moving_avg_sku_split].transform("sum")
                output_df[Constants.moving_avg_sku_split] = np.where(
                    output_df[Constants.cumulative_split_col] != 0,
                    output_df[Constants.moving_avg_sku_split]
                    / output_df[Constants.cumulative_split_col],
                    np.nan,
                )

                MovingAvgSplit = output_df[cols_required_for_moving_avg_split_ratio]

        if MovingAvgSplit.empty:
            MovingAvgSplit = MovingAverageSplitOutput.copy()
            MovingAvgSplit = MovingAvgSplit[cols_required_for_moving_avg_split_ratio]

        if len(LocationSplitPercentInputNormalized) == 0:
            FixedSplit = pd.DataFrame(columns=cols_required_for_fixed_split_ratio + ["Flag"])

        else:
            # creating cartesian product to get all partial weeks
            LocationSplitPercentInputNormalized["Flag"] = np.where(
                LocationSplitPercentInputNormalized[Constants.from_date_col] <= start_date_key,
                False,
                True,
            )
            relevant_data_raw = create_cartesian_product(
                LocationSplitPercentInputNormalized,
                ForecastBucket,
            )

            # get partial weeks which satisfy corresponding from & to dates
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
                [
                    Constants.partial_week_col,
                    Constants.item_col,
                    Constants.pl_channel_col,
                    Constants.pl_pnl_col,
                    Constants.pl_account_col,
                    Constants.pl_region_col,
                    Constants.pl_demand_domain_col,
                    Constants.pl_location_col,
                ],
                observed=True,
            )[Constants.split_col].transform("sum")
            relevant_data_raw[Constants.fixed_per_sku_split] = np.where(
                relevant_data_raw[Constants.cumulative_split_col] != 0,
                relevant_data_raw[Constants.split_col]
                / relevant_data_raw[Constants.cumulative_split_col],
                np.nan,
            )
            relevant_data_raw[Constants.version_col] = input_version
            FixedSplit = relevant_data_raw[cols_required_for_fixed_split_ratio + ["Flag"]]

        # merge MovingAvgSplit and FixedSplit
        merge_cols = [Constants.version_col, Constants.partial_week_col] + dimensions
        SKUSplit = pd.merge(
            MovingAvgSplit,
            FixedSplit,
            on=merge_cols,
            how="outer",
        )

        dimensions_copy = dimensions
        dimensions_copy.remove("Location.[Location]")

        com_cols = [Constants.version_col] + dimensions_copy
        SKUSplit = SKUSplit.merge(
            LocationSplitMethod,
            on=com_cols,
            how="left",
        )
        common_cols = [
            Constants.version_col,
            Constants.partial_week_col,
        ] + dimensions_copy
        SKUSplit = SKUSplit.merge(
            ItemConsensusFcst,
            on=common_cols,
            how="left",
        )

        SKUSplit[Constants.temp_fixed_split] = np.where(
            SKUSplit[Constants.fixed_per_sku_split].isna() & SKUSplit["Flag"],
            SKUSplit[Constants.moving_avg_sku_split],
            SKUSplit[Constants.fixed_per_sku_split],
        )
        SKUSplit[Constants.split_final] = np.where(
            SKUSplit[Constants.method_col] == "Leaf Level Moving Avg",
            SKUSplit[Constants.moving_avg_sku_split],
            SKUSplit[Constants.temp_fixed_split],
        )
        SKUSplit[Constants.final_fcst] = (
            SKUSplit[Constants.split_final] * SKUSplit[Constants.item_consensus_fcst]
        )

        SKUSplit = SKUSplit[cols_required_for_output]
        logger.warning("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return SKUSplit

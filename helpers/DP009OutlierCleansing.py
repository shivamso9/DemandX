import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
    get_seasonal_periods,
)
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data

from helpers.o9Constants import o9Constants
from helpers.outlier_correction import cleanse_data_wrapper
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Outlier Upper Threshold": float,
    "Outlier Lower Threshold": float,
    "Actual Cleansed System": float,
    "Actual Median": float,
    "Abs Actual Cleansed vs Actual Gap": float,
    "Outlier Status": str,
    "Actual Cleansed Adjustment Profile": float,
}


def add_gap_and_status(
    ActualCleansed,
    Actuals,
    grains,
    output_cols,
    cleansed_data_col,
    actual_l1_col,
    abs_threshold_col,
    percent_threshold_col,
    defaults,
    abs_cleansed_vs_actual,
    outlier_status,
):
    cleansed_time_level = ActualCleansed.columns[ActualCleansed.columns.str.contains("Time")][0]
    cols_required_in_output = grains + output_cols + [cleansed_time_level]
    try:
        logger.info(
            "Outlier calculation complete, adding 'Abs Actual Cleansed vs Actual Gap' and 'Outlier Status'"
        )
        ActualCleansed_copy = ActualCleansed.copy()
        if cleansed_time_level not in Actuals.columns:
            raise Exception(f"{cleansed_time_level} not present in Actual L1 columns")

        ActualCleansed_copy = ActualCleansed_copy.merge(
            Actuals, on=grains + [cleansed_time_level], how="left"
        )

        ActualCleansed_copy.fillna(defaults, inplace=True)

        ActualCleansed_copy = ActualCleansed_copy.dropna(subset={cleansed_data_col})

        output_dict = {col: "first" for col in output_cols}
        ActualCleansed_copy = (
            ActualCleansed_copy.groupby(grains + [cleansed_time_level])
            .agg(
                {
                    actual_l1_col: "sum",
                    abs_threshold_col: "first",
                    percent_threshold_col: "first",
                    **output_dict,
                }
            )
            .reset_index()
        )

        # abs actual cleansed vs actual gap calculation
        ActualCleansed_copy.loc[
            ActualCleansed_copy[cleansed_data_col] > 0, abs_cleansed_vs_actual
        ] = round(
            abs(
                ActualCleansed_copy[cleansed_data_col]
                - abs(ActualCleansed_copy[actual_l1_col].fillna(0))
            ),
            0,
        )
        ActualCleansed_copy.loc[
            ActualCleansed_copy[cleansed_data_col] <= 0, abs_cleansed_vs_actual
        ] = round(abs(ActualCleansed_copy[cleansed_data_col]), 0)

        # outlier status calculation
        actual_l1_null_df = ActualCleansed_copy[
            ActualCleansed_copy[actual_l1_col].isna()
        ].reset_index()
        actual_l1_notNull_df = ActualCleansed_copy[
            ~ActualCleansed_copy[actual_l1_col].isna()
        ].reset_index()

        actual_l1_null_df.loc[actual_l1_null_df[cleansed_data_col] == 0, outlier_status] = 0
        actual_l1_null_df.loc[
            (actual_l1_null_df[abs_cleansed_vs_actual] > actual_l1_null_df[abs_threshold_col])
            & (actual_l1_null_df[outlier_status].isna()),
            outlier_status,
        ] = 1
        actual_l1_null_df.loc[
            (actual_l1_null_df[abs_cleansed_vs_actual] <= actual_l1_null_df[abs_threshold_col])
            & (actual_l1_null_df[outlier_status].isna()),
            outlier_status,
        ] = -1

        actual_l1_notNull_df.loc[
            actual_l1_notNull_df[abs_cleansed_vs_actual] == 0, outlier_status
        ] = 0

        actual_l1_notNull_df.loc[
            (actual_l1_notNull_df[abs_cleansed_vs_actual] > actual_l1_notNull_df[abs_threshold_col])
            & (actual_l1_notNull_df[actual_l1_col] == 0)
            & (actual_l1_notNull_df[outlier_status].isna()),
            "safedivide",
        ] = 0

        mask = (
            actual_l1_notNull_df[abs_cleansed_vs_actual] > actual_l1_notNull_df[abs_threshold_col]
        ) & (actual_l1_notNull_df[outlier_status].isna())

        # For rows where actual_l1_col is not zero
        valid_mask = mask & (actual_l1_notNull_df[actual_l1_col] != 0)
        actual_l1_notNull_df.loc[valid_mask, "safedivide"] = (
            actual_l1_notNull_df.loc[valid_mask, abs_cleansed_vs_actual]
            / actual_l1_notNull_df.loc[valid_mask, actual_l1_col]
        )

        # For rows where actual_l1_col is zero
        zero_denom_mask = mask & (actual_l1_notNull_df[actual_l1_col] == 0)
        actual_l1_notNull_df.loc[zero_denom_mask, "safedivide"] = (
            0  # Assign any appropriate value or handle as needed
        )

        actual_l1_notNull_df.loc[
            (actual_l1_notNull_df[abs_cleansed_vs_actual] > actual_l1_notNull_df[abs_threshold_col])
            & (actual_l1_notNull_df["safedivide"] > actual_l1_notNull_df[percent_threshold_col])
            & (actual_l1_notNull_df[outlier_status].isna()),
            outlier_status,
        ] = 1

        actual_l1_notNull_df.loc[
            (actual_l1_notNull_df[abs_cleansed_vs_actual] > actual_l1_notNull_df[abs_threshold_col])
            & (actual_l1_notNull_df["safedivide"] <= actual_l1_notNull_df[percent_threshold_col])
            & (actual_l1_notNull_df[outlier_status].isna()),
            outlier_status,
        ] = -1

        actual_l1_notNull_df.loc[
            (
                actual_l1_notNull_df[abs_cleansed_vs_actual]
                <= actual_l1_notNull_df[abs_threshold_col]
            )
            & (actual_l1_notNull_df[outlier_status].isna()),
            outlier_status,
        ] = -1

        ActualCleansed = concat_to_dataframe([actual_l1_null_df, actual_l1_notNull_df])

        ActualCleansed = ActualCleansed[cols_required_in_output].drop_duplicates()
    except Exception as e:
        logger.warning(f"{e}")
        ActualCleansed[abs_cleansed_vs_actual] = None
        ActualCleansed[outlier_status] = None
    return ActualCleansed


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    ReadFromHive,
    Actual,
    CurrentTimePeriod,
    HistoryPeriod,
    OutlierParameters,
    TimeDimension,
    ForecastGenTimeBucket,
    StatBucketWeight,
    OutlierAbsoluteThreshold,
    OutlierPercentageThreshold,
    SellOutOffset,
    smooth_fraction=0.25,
    multiprocessing_num_cores=4,
    df_keys=None,
):
    try:
        CleansedDataList = list()
        ActualL1List = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_cleansed_data, the_actualL1_output = decorated_func(
                Grains=Grains,
                ReadFromHive=ReadFromHive,
                Actual=Actual,
                CurrentTimePeriod=CurrentTimePeriod,
                HistoryPeriod=HistoryPeriod,
                OutlierParameters=OutlierParameters,
                OutlierAbsoluteThreshold=OutlierAbsoluteThreshold,
                OutlierPercentageThreshold=OutlierPercentageThreshold,
                TimeDimension=TimeDimension,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                SellOutOffset=SellOutOffset,
                smooth_fraction=smooth_fraction,
                multiprocessing_num_cores=multiprocessing_num_cores,
                the_iteration=the_iteration,
                df_keys=df_keys,
            )

            CleansedDataList.append(the_cleansed_data)
            ActualL1List.append(the_actualL1_output)

        CleansedData = concat_to_dataframe(CleansedDataList)
        ActualL1_output = concat_to_dataframe(ActualL1List)
    except Exception as e:
        logger.exception(e)
        CleansedData = None

    return CleansedData, ActualL1_output


def processIteration(
    Grains,
    ReadFromHive,
    Actual,
    CurrentTimePeriod,
    HistoryPeriod,
    OutlierParameters,
    TimeDimension,
    ForecastGenTimeBucket,
    StatBucketWeight,
    the_iteration,
    OutlierAbsoluteThreshold=pd.DataFrame(),
    OutlierPercentageThreshold=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    smooth_fraction=0.25,
    multiprocessing_num_cores=4,
    df_keys=None,
):
    if df_keys is None:
        df_keys = {}
    plugin_name = "DP009OutlierCleansing"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"

    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"

    partial_week_col = "Time.[Partial Week]"
    version_col = "Version.[Version Name]"
    # forecast_iteration_col = "Forecast Iteration.[Forecast Iteration]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    # history_measure_col = "History Measure"
    stat_bucket_weight_col = "Stat Bucket Weight"
    actual_l1_col = "Actual L1"
    abs_threshold_col = "Outlier Absolute Threshold"
    percent_threshold_col = "Outlier Percentage Threshold"
    sell_out_offset_col = "Offset Period"

    # Outlier Parameters
    upper_threshold_col = "Outlier Upper Threshold Limit"
    lower_threshold_col = "Outlier Lower Threshold Limit"
    outlier_correction_col = "Outlier Correction"
    outlier_method_col = "Outlier Method"
    outlier_period_col = "History Period"

    # output measures
    cleansed_data_col = "Actual Cleansed System"
    cleansed_adjustment_data_col = "Actual Cleansed Adjustment Profile"
    upper_bound_col = "Outlier Upper Threshold"
    lower_bound_col = "Outlier Lower Threshold"
    actual_mean_col = "Actual Median"
    abs_cleansed_vs_actual_col = "Abs Actual Cleansed vs Actual Gap"
    outlier_status_col = "Outlier Status"
    stat_actual_col = "Stat Actual"
    slice_association_stat_col = "Slice Association Stat"

    # configure columns to disaggregate
    cols_to_disaggregate = [
        actual_mean_col,
        upper_bound_col,
        lower_bound_col,
        cleansed_data_col,
        abs_cleansed_vs_actual_col,
        outlier_status_col,
    ]

    # default outlier thresholds
    default_abs_threshold = 50
    default_percent_threshold = 0.05

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]

    logger.info("dimensions : {} ...".format(dimensions))

    assert len(dimensions) > 0, "dimensions cannot be empty ..."

    cols_required_in_ActualL1_output = (
        [version_col]
        + dimensions
        + [
            partial_week_col,
            actual_l1_col,
        ]
    )
    cols_required_in_output = (
        [version_col]
        + dimensions
        + [
            partial_week_col,
            upper_bound_col,
            lower_bound_col,
            cleansed_data_col,
            actual_mean_col,
            abs_cleansed_vs_actual_col,
            outlier_status_col,
            cleansed_adjustment_data_col,
        ]
    )

    output_df = pd.DataFrame(columns=cols_required_in_output)
    ActualL1_output = pd.DataFrame(columns=cols_required_in_ActualL1_output)

    try:
        if len(SellOutOffset) == 0:
            logger.warning(
                f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

        history_measure = "Stat Actual"

        # filter out nulls
        Actual = Actual[Actual[history_measure].notna()]
        OutlierParameters.dropna(
            subset=[
                upper_threshold_col,
                lower_threshold_col,
                outlier_correction_col,
                outlier_method_col,
            ],
            how="any",
            inplace=True,
        )

        if Actual is None or len(Actual) == 0:
            logger.warning("Actuals is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return output_df, ActualL1_output

        ActualL1 = Actual.merge(TimeDimension, on=partial_week_col, how="inner")
        ActualL1 = ActualL1.rename(columns={stat_actual_col: actual_l1_col})
        ActualL1 = ActualL1[
            cols_required_in_ActualL1_output
            + [o9Constants.WEEK]
            + [o9Constants.MONTH]
            + [o9Constants.PLANNING_MONTH]
            + [o9Constants.QUARTER]
            + [o9Constants.PLANNING_QUARTER]
            + [slice_association_stat_col]
        ].drop_duplicates()

        ActualL1_output = ActualL1.copy()
        ActualL1_output[actual_l1_col] = np.where(
            ActualL1_output[actual_l1_col] > 0, ActualL1_output[actual_l1_col], 0
        )
        ActualL1_output = ActualL1_output[cols_required_in_ActualL1_output].drop_duplicates()

        input_version = Actual[version_col].iloc[0]

        logger.info("history_measure : {}".format(history_measure))

        cols_required_in_Actual = [partial_week_col] + dimensions + [history_measure]

        for the_col in cols_required_in_Actual:
            assert the_col in list(Actual.columns), "{} missing in Actual dataframe ...".format(
                the_col
            )

        logger.info("filtering {} columns from Actual df ...".format(cols_required_in_Actual))

        Actual = Actual[cols_required_in_Actual]

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return output_df, ActualL1_output

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                planning_quarter_col,
                planning_quarter_key_col,
            ]
            relevant_time_name = planning_quarter_col
            relevant_time_key = planning_quarter_key_col
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                quarter_col,
                quarter_key_col,
            ]
            relevant_time_name = quarter_col
            relevant_time_key = quarter_key_col
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return output_df, ActualL1_output

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        outlier_period = int(HistoryPeriod[outlier_period_col].iloc[0])
        logger.info("outlier_period : {}".format(outlier_period))

        time_series_freq = get_seasonal_periods(frequency)

        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        logger.info("latest_time_name : {} ...".format(latest_time_name))

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        logger.info("Getting last {} period dates for history period ...".format(outlier_period))

        # adjust the latest time according to the forecast iteration's offset before getting n periods for outlier cleansing
        offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
        if offset_periods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        last_n_periods_history = get_n_time_periods(
            latest_time_name,
            -outlier_period,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        filtered_time = pd.DataFrame({relevant_time_name: last_n_periods_history})

        # Join Actuals with time mapping
        Actual = Actual.merge(base_time_mapping, on=partial_week_col, how="inner")

        Actual = Actual.merge(filtered_time, on=relevant_time_name, how="inner")

        if len(Actual) == 0:
            logger.warning("Actuals is empty for last {} period dates".format(outlier_period))
            logger.warning("Returning empty DataFrame")
            return output_df, ActualL1_output
        # filter out leading zeros
        logger.debug(f"Actual.shape : {Actual.shape}")
        cumulative_sum_col = "_".join(["Cumulative", history_measure])
        Actual.sort_values(dimensions + [relevant_time_key], inplace=True)
        Actual[cumulative_sum_col] = Actual.groupby(dimensions)[history_measure].transform(
            pd.Series.cumsum
        )
        Actual = Actual[Actual[cumulative_sum_col] > 0]
        logger.debug(f"Shape after filtering leading zeros : {Actual.shape}")

        # select the relevant columns, groupby and sum history measure
        Actual = (
            Actual.groupby(dimensions + [relevant_time_name]).sum()[[history_measure]].reset_index()
        )

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=Actual,
            forecast_level=dimensions,
            time_mapping=relevant_time_mapping,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods_history,
            fill_nulls_with_zero=True,
        )

        if len(OutlierParameters) == 0:
            logger.warning("OutlierParameters df is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe as result ...")
            return output_df, ActualL1_output

        num_intersections = relevant_history_nas_filled.groupby(dimensions).ngroups
        logger.info(f"Performing outlier correction for {num_intersections} intersections ...")

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(cleanse_data_wrapper)(
                group,
                dimensions,
                relevant_time_key,
                history_measure,
                cleansed_data_col,
                upper_bound_col,
                lower_bound_col,
                actual_mean_col,
                relevant_time_name,
                OutlierParameters,
                upper_threshold_col,
                lower_threshold_col,
                outlier_method_col,
                outlier_correction_col,
                time_series_freq,
                smooth_fraction,
            )
            for name, group in relevant_history_nas_filled.groupby(dimensions)
        )

        # Concatenate all results to one dataframe
        output_df = concat_to_dataframe(all_results)

        if len(output_df) == 0:
            logger.warning(
                "No records after processing for slice : {}, returning empty dataframe".format(
                    df_keys
                )
            )
            return output_df, ActualL1_output

        # validate output, print warning messages if intersections are missing
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=output_df,
            forecast_level=dimensions,
        )

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )
        output_df[abs_cleansed_vs_actual_col] = None
        output_df[outlier_status_col] = None
        if len(ActualL1) != 0:
            if len(OutlierAbsoluteThreshold) == 0:
                OutlierAbsoluteThreshold = pd.DataFrame(
                    [
                        {
                            version_col: input_version,
                            abs_threshold_col: default_abs_threshold,
                        }
                    ]
                )
            if len(OutlierPercentageThreshold) == 0:
                OutlierPercentageThreshold = pd.DataFrame(
                    [
                        {
                            version_col: input_version,
                            percent_threshold_col: default_percent_threshold,
                        }
                    ]
                )

            ActualL1 = ActualL1.merge(OutlierAbsoluteThreshold, on=[version_col], how="inner")
            ActualL1 = ActualL1.merge(OutlierPercentageThreshold, on=[version_col], how="inner")
            # adding abs gap and the outlier status outputs
            output_df = add_gap_and_status(
                ActualCleansed=output_df,
                Actuals=ActualL1,
                grains=dimensions,
                output_cols=cols_to_disaggregate,
                cleansed_data_col=cleansed_data_col,
                actual_l1_col=actual_l1_col,
                abs_threshold_col=abs_threshold_col,
                percent_threshold_col=percent_threshold_col,
                defaults={
                    abs_threshold_col: default_abs_threshold,
                    percent_threshold_col: default_percent_threshold,
                },
                abs_cleansed_vs_actual=abs_cleansed_vs_actual_col,
                outlier_status=outlier_status_col,
            )
        # perform disaggregation
        output_df = disaggregate_data(
            source_df=output_df,
            source_grain=relevant_time_name,
            target_grain=partial_week_col,
            profile_df=StatBucketWeight,
            profile_col=stat_bucket_weight_col,
            cols_to_disaggregate=cols_to_disaggregate,
        )
        output_df["status"] = output_df[outlier_status_col]
        output_df.loc[output_df["status"] == 0, outlier_status_col] = "No Correction"
        output_df.loc[output_df["status"] < 0, outlier_status_col] = "Within Threshold"
        output_df.loc[output_df["status"] > 0, outlier_status_col] = "Exceeds Threshold"

        # Add input version
        output_df.insert(loc=0, column=version_col, value=input_version)
        output_df[cleansed_data_col] = np.where(
            output_df[cleansed_data_col] > 0, output_df[cleansed_data_col], 0
        )
        output_df[cleansed_adjustment_data_col] = output_df[cleansed_data_col]

        output_df = output_df[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        output_df = pd.DataFrame(columns=cols_required_in_output)
        ActualL1_output = pd.DataFrame(columns=cols_required_in_ActualL1_output)

    return output_df, ActualL1_output

import logging

import numpy as np
import pandas as pd
import tsfeatures as tf
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
from pmdarima.arima import ndiffs
from sklearn.linear_model import LinearRegression

from helpers.o9Constants import o9Constants
from helpers.seasonality import ACFDetector
from helpers.utils import filter_for_iteration

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


def get_trend_rate_of_change(X: np.ndarray, y: np.ndarray) -> float:
    # initialize RateChange
    RateChange = 0
    try:
        regressor = LinearRegression()
        # Clipping off 10 % on either side of the time series
        y = np.clip(
            y,
            a_min=np.quantile(y, 0.10),
            a_max=np.quantile(y, 0.90),
        )

        regressor.fit(X, y)
        # Convert the slope of the model to degrees; this would be Trend Degree L1
        slope = regressor.coef_[0]
        intercept = regressor.intercept_
        fitted_line = slope * X + intercept

        if (fitted_line[-1][0] < 0) & (fitted_line[0][0] < 0):
            logger.warning("both first and last values are negative, hence No Trend ...")
            RateChange = 0

        elif (fitted_line[-1][0] + fitted_line[0][0]) == 0:
            logger.warning("fitted line is constant, hence No Trend ...")
            RateChange = 0
        else:
            fitted_line = np.clip(fitted_line, a_min=0, a_max=None)
            RateChange = (fitted_line[-1][0] - fitted_line[0][0]) / (
                (fitted_line[-1][0] + fitted_line[0][0]) / 2
            )

    except Exception as e:
        logger.exception(e)
    return RateChange


def calc_seasonality(
    ActualData: pd.DataFrame,
    history_measure: str,
    frequency: str,
    segmentation_level: list,
    seasonality_l1_col: str,
    seasonal_periods_l1_col: str,
    trend_strength_l1_col: str,
    seasonal_strength_l1_col: str,
    entropy_l1_col: str,
    plc_status_l1_col: str,
    los_l1_col: str,
    skip_lags: int,
    diff: int,
    trend_col: str,
    trend_threshold: float,
    trend_factor_col: str,
    USE_ACF_FOR_SEASONALITY: bool,
    seasonal_diff_col: str,
    seasonality_threshold: float,
    alpha: float == 0.05,
    lower_ci_threshold: float,
    upper_ci_threshold: float,
    RequiredACFLags: list,
) -> pd.DataFrame:
    the_intersection = ActualData[segmentation_level].iloc[0].values
    logger.debug("the_intersection : {}".format(the_intersection))
    try:
        result = ActualData[segmentation_level].drop_duplicates()

        # create fallback dataframe for results
        result[seasonality_l1_col] = "Does not Exist"
        result[seasonal_periods_l1_col] = None
        result[trend_strength_l1_col] = None
        result[seasonal_strength_l1_col] = None
        result[entropy_l1_col] = None
        result[trend_col] = "NO TREND"

        # If NPI or DISCO Product, no need for calculations
        if ActualData[plc_status_l1_col].unique()[0] != "MATURE":
            return result

        # Trend Components
        # check if data has atleast 1 seasonal cycles, or sum of actuals is zero
        if ActualData[los_l1_col].unique()[0] >= get_seasonal_periods(frequency):
            # Updated trend logic
            counter: str = "Counter"
            tmp = ActualData.iloc[:]
            tmp[counter] = np.arange(1, len(tmp) + 1)
            X = tmp[counter].values.reshape(-1, 1)  # Reshape to a 2D array
            y = tmp[history_measure].values
            result[trend_factor_col] = get_trend_rate_of_change(X, y)

            # Assign trend category
            conditions = [
                result[trend_factor_col] > trend_threshold,
                (result[trend_factor_col] <= trend_threshold)
                & (result[trend_factor_col] >= -trend_threshold),
                result[trend_factor_col] < -trend_threshold,
            ]
            choices = ["UPWARD", "NO TREND", "DOWNWARD"]

            # logger.info("Assigning trend categories ...")
            result[trend_col] = np.select(conditions, choices, default=None)

        # Seasonal Components
        # check if data has atleast 2 seasonal cycles, or sum of actuals is zero
        if (
            ActualData[los_l1_col].unique()[0] >= 2 * get_seasonal_periods(frequency)
            and ActualData[history_measure].sum() > 0
        ):
            if USE_ACF_FOR_SEASONALITY:
                acf_seasonal_presence, acf_seasonalities = ACFDetector(
                    ActualData,
                    history_measure,
                    frequency,
                    skip_lags,
                    diff,
                    alpha,
                    lower_ci_threshold,
                    upper_ci_threshold,
                    RequiredACFLags,
                )
                # Storage of results
                result[seasonality_l1_col] = "Exists" if acf_seasonal_presence else "Does not Exist"

                # always calculate seasonal periods irrespective of seasonality exists or not
                result[seasonal_periods_l1_col] = acf_seasonalities
            else:
                result[seasonal_diff_col] = ndiffs(
                    ActualData[history_measure], alpha=0.05, test="kpss"
                )

                # create filter clause
                seasonality_calc_filter = result[seasonal_diff_col] > seasonality_threshold

                # Assign seasonality category
                result[seasonality_l1_col] = np.where(
                    seasonality_calc_filter,
                    "Exists",
                    "Does Not Exist",
                )

            features = tf.stl_features(
                ActualData[history_measure].values,
                get_seasonal_periods(frequency),
            )

            # always calculate entropy for data with 2 cycles
            entropy = tf.entropy(
                ActualData[history_measure].values,
                get_seasonal_periods(frequency),
            )["entropy"]
            result[entropy_l1_col] = entropy
            result[entropy_l1_col] = result[entropy_l1_col].fillna(0)

            # assign trend strength where trend exists
            if result[trend_col].unique()[0] in ["UPWARD", "DOWNWARD"]:
                result[trend_strength_l1_col] = features["trend"]

            # assign seasonal strength only where seasonality exists
            if result[seasonality_l1_col].unique()[0] in ["Exists"]:
                result[seasonal_strength_l1_col] = features["seasonal_strength"]

            # Assign categories, fallback values, fill NAs
            result[seasonality_l1_col] = result[seasonality_l1_col].fillna("Does not Exist")

    except Exception as e:
        logger.exception("intersection : {}, exception : {}".format(the_intersection, e))
        result = pd.DataFrame()

    return result


col_mapping = {
    "Seasonality L1": str,
    "Seasonal Periods L1": str,
    "Trend Strength L1": float,
    "Seasonal Strength L1": float,
    "Entropy L1": float,
    "Trend L1": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    Actual,
    TimeDimension,
    CurrentTimePeriod,
    Parameters,
    ForecastGenTimeBucket,
    SegmentationOutput,
    ReadFromHive,
    multiprocessing_num_cores,
    alpha,
    lower_ci_threshold,
    upper_ci_threshold,
    skip_lags,
    diff,
    RequiredACFLagsInWeeks,
    TrendThreshold: int = 20,
    SellOutOffset=pd.DataFrame(),
    df_keys={},
):
    try:
        SeasonalDetectionDataList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_seasonal_detection_data = decorated_func(
                Grains=Grains,
                Actual=Actual,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                Parameters=Parameters,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                SellOutOffset=SellOutOffset,
                SegmentationOutput=SegmentationOutput,
                ReadFromHive=ReadFromHive,
                multiprocessing_num_cores=multiprocessing_num_cores,
                alpha=alpha,
                df_keys=df_keys,
                lower_ci_threshold=lower_ci_threshold,
                upper_ci_threshold=upper_ci_threshold,
                skip_lags=skip_lags,
                diff=diff,
                RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                the_iteration=the_iteration,
                TrendThreshold=TrendThreshold,
            )

            SeasonalDetectionDataList.append(the_seasonal_detection_data)

        SeasonalDetectionData = concat_to_dataframe(SeasonalDetectionDataList)
    except Exception as e:
        logger.exception(e)
        SeasonalDetectionData = None, None

    return SeasonalDetectionData


def processIteration(
    Grains,
    Actual,
    TimeDimension,
    CurrentTimePeriod,
    Parameters,
    ForecastGenTimeBucket,
    SegmentationOutput,
    ReadFromHive,
    multiprocessing_num_cores,
    alpha,
    df_keys,
    lower_ci_threshold,
    upper_ci_threshold,
    skip_lags,
    diff,
    RequiredACFLagsInWeeks,
    the_iteration,
    SellOutOffset=pd.DataFrame(),
    TrendThreshold: int = 20,
):
    plugin_name = "DP037SeasonalityDetection"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    seasonality_l1_col = "Seasonality L1"
    seasonal_periods_l1_col = "Seasonal Periods L1"
    trend_strength_l1_col = "Trend Strength L1"
    seasonal_strength_l1_col = "Seasonal Strength L1"
    entropy_l1_col = "Entropy L1"
    los_l1_col = "Length of Series L1"
    plc_status_l1_col = "PLC Status L1"
    history_period_col = "History Period"
    trend_col = "Trend L1"
    trend_factor_col = "TrendFactor"
    seasonal_diff_col = "ndiffs"

    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"
    partial_week_key_col = "Time.[PartialWeekKey]"

    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"

    partial_week_col = "Time.[Partial Week]"
    version_col = "Version.[Version Name]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    # history_measure_col = "History Measure"
    sell_out_offset_col = "Offset Period"

    USE_ACF_FOR_SEASONALITY = True
    logger.info(f"USE_ACF_FOR_SEASONALITY : {USE_ACF_FOR_SEASONALITY}")

    logger.info("Extracting dimension cols ...")
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    segmentation_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    logger.info("segmentation_level : {} ...".format(segmentation_level))

    assert len(segmentation_level) > 0, "segmentation_level cannot be empty ..."

    req_cols_in_output = (
        [version_col]
        + segmentation_level
        + [
            seasonality_l1_col,
            seasonal_periods_l1_col,
            trend_strength_l1_col,
            seasonal_strength_l1_col,
            entropy_l1_col,
            trend_col,
        ]
    )
    SeasonalDetectionData = pd.DataFrame(columns=req_cols_in_output)
    try:
        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        # convert script param to float
        alpha = float(alpha)
        logger.debug(f"alpha : {alpha}")

        # getting RequiredACFLags
        RequiredACFLagsInWeeks = RequiredACFLagsInWeeks.split(",")

        # remove leading/trailing spaces if any
        RequiredACFLagsInWeeks = [x.strip() for x in RequiredACFLagsInWeeks]
        RequiredACFLags = [int(x) for x in RequiredACFLagsInWeeks]

        # history_measure = Parameters[history_measure_col].unique()[0]
        # if ReadFromHive:
        #     history_measure = "DP006" + history_measure
        history_measure = "Stat Actual"

        cols_required_in_Actual = [partial_week_col] + segmentation_level + [history_measure]

        Actual = Actual[Actual[history_measure].notna()]

        SegmentationOutput = SegmentationOutput[SegmentationOutput[plc_status_l1_col].notna()]

        for the_col in cols_required_in_Actual:
            assert the_col in list(Actual.columns), "{} missing in Actual dataframe ...".format(
                the_col
            )

        logger.info("filtering {} columns from Actual df ...".format(cols_required_in_Actual))

        Actual = Actual[cols_required_in_Actual]

        # Join Actual with Time mapping
        pw_time_mapping = TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates()
        Actual = Actual.merge(pw_time_mapping, on=partial_week_col, how="inner")

        last_partial_week = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            partial_week_col,
            partial_week_key_col,
        )
        last_partial_week_key = TimeDimension[TimeDimension[partial_week_col] == last_partial_week][
            partial_week_key_col
        ].iloc[0]

        # filter out actuals prior to LastTimePeriod
        filter_clause = Actual[partial_week_key_col] <= last_partial_week_key

        logger.debug(f"Actual: shape : {Actual.shape}")
        logger.debug("Filtering records only prior to LastTimePeriod ...")
        Actual = Actual[filter_clause]
        logger.debug(f"Actual: shape : {Actual.shape}")

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return SeasonalDetectionData

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if Parameters.empty:
            logger.warning(f"Parameters is empty for {df_keys}, cannot process further")
            return SeasonalDetectionData

        history_periods = int(Parameters[history_period_col].unique()[0])

        logger.debug(f"history_periods : {history_periods}")

        logger.info("overwriting the skip_lags and diff value based on generation time bucket ...")

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

        diff = 0  # default value for month and quarter
        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col
            skip_lags = 12
            diff = 1
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col
            skip_lags = 2
            RequiredACFLags = [round(int(i) / 4.34524) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
            skip_lags = 2
            RequiredACFLags = [round(int(i) / 4.34524) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                quarter_col,
                quarter_key_col,
            ]
            relevant_time_name = quarter_col
            relevant_time_key = quarter_key_col
            skip_lags = 0
            RequiredACFLags = [round(int(i) / 13) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                planning_quarter_col,
                planning_quarter_key_col,
            ]
            relevant_time_name = planning_quarter_col
            relevant_time_key = planning_quarter_key_col
            skip_lags = 0
            RequiredACFLags = [round(int(i) / 13) for i in RequiredACFLags]
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return SeasonalDetectionData

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")
        logger.debug(f"diff : {diff}")
        logger.debug(f"skip_lags : {skip_lags}")

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Join Actuals with time mapping
        Actual = Actual.merge(base_time_mapping, on=partial_week_col, how="inner")

        # select the relevant columns, groupby and sum history measure
        Actual = (
            Actual.groupby(segmentation_level + [relevant_time_name])
            .sum()[[history_measure]]
            .reset_index()
        )

        time_attribute_dict = {relevant_time_name: relevant_time_key}
        input_version = Parameters[version_col].unique()[0]

        # note the negative sign to history periods
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # adjust the latest time according to the forecast iteration's offset before getting n periods for considering history
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

        # logger.info("Getting last {} period dates ...".format(history_periods))
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -history_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )
        if len(last_n_periods) == 0:
            logger.warning(
                "No dates found after filtering for {} periods for slice {}".format(
                    history_periods, df_keys
                )
            )
            return SeasonalDetectionData

        # filter relevant history based on dates provided above
        relevant_actuals = Actual[Actual[relevant_time_name].isin(last_n_periods)]

        if len(relevant_actuals) == 0:
            logger.warning(
                "No actuals found after filtering for {} periods for slice {}".format(
                    history_periods, df_keys
                )
            )
            return SeasonalDetectionData

        # Use default thresholds for trend and seasonality since this won't be coming from the tenant base configuration
        # trend_threshold = 0.01
        # logger.info(f"trend_threshold : {trend_threshold}")

        trend_threshold = TrendThreshold / 100
        logger.info(f"trend_threshold : {trend_threshold}")

        # Use default thresholds for seasonality
        seasonality_threshold = 0.64
        if not USE_ACF_FOR_SEASONALITY:
            logger.info(f"seasonality_threshold : {seasonality_threshold}")

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=relevant_actuals,
            forecast_level=segmentation_level,
            time_mapping=relevant_time_mapping,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods,
            fill_nulls_with_zero=True,
        )

        # Join on segmentation attributes
        relevant_history_nas_filled = relevant_history_nas_filled.merge(
            SegmentationOutput, on=segmentation_level, how="inner"
        )

        # sort values for every intersection respecting the time grain
        relevant_history_nas_filled.sort_values(
            segmentation_level + [relevant_time_key], inplace=True
        )

        SeasonalDetectionData_list = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(calc_seasonality)(
                ActualData=group,
                history_measure=history_measure,
                frequency=frequency,
                segmentation_level=segmentation_level,
                seasonality_l1_col=seasonality_l1_col,
                seasonal_periods_l1_col=seasonal_periods_l1_col,
                trend_strength_l1_col=trend_strength_l1_col,
                seasonal_strength_l1_col=seasonal_strength_l1_col,
                entropy_l1_col=entropy_l1_col,
                plc_status_l1_col=plc_status_l1_col,
                los_l1_col=los_l1_col,
                skip_lags=skip_lags,
                diff=diff,
                trend_col=trend_col,
                trend_threshold=trend_threshold,
                trend_factor_col=trend_factor_col,
                USE_ACF_FOR_SEASONALITY=USE_ACF_FOR_SEASONALITY,
                seasonal_diff_col=seasonal_diff_col,
                seasonality_threshold=seasonality_threshold,
                alpha=alpha,
                lower_ci_threshold=lower_ci_threshold,
                upper_ci_threshold=upper_ci_threshold,
                RequiredACFLags=RequiredACFLags,
            )
            for name, group in relevant_history_nas_filled.groupby(
                segmentation_level, observed=True
            )
        )

        logger.info("Collected results from parallel processing ...")

        # concat list of results to df
        SeasonalDetectionData = concat_to_dataframe(list_of_results=SeasonalDetectionData_list)

        if len(SeasonalDetectionData) > 0:

            logger.info("Validating output for all intersections ...")

            # validate if output dataframe contains result for all groups present in input
            validate_output(
                input_df=relevant_history_nas_filled,
                output_df=SeasonalDetectionData,
                forecast_level=segmentation_level,
            )

            logger.info("Assigning values to {} ...".format(seasonality_l1_col))

            # insert version column
            SeasonalDetectionData.insert(0, version_col, input_version)
        else:
            SeasonalDetectionData = pd.DataFrame(columns=req_cols_in_output)

        logger.info(f"Filtering {req_cols_in_output} from output dataframe ...")

        SeasonalDetectionData = SeasonalDetectionData[req_cols_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(e)
        SeasonalDetectionData = pd.DataFrame(columns=req_cols_in_output)

    return SeasonalDetectionData

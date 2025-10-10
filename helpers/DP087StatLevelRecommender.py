import ast
import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
    get_seasonal_periods,
)
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension
from o9Reference.stat_utils.assign_segments import assign_segments
from o9Reference.stat_utils.segmentation_utils import assign_volume_segment_to_group
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration, get_list_of_grains_from_string

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

col_mapping = {}

logger = logging.getLogger("o9_logger")


class constants:
    version_col = o9Constants.VERSION_NAME
    forecast_iteration = o9Constants.FORECAST_ITERATION
    class_col = o9Constants.CLASS
    stat_item_col = o9Constants.STAT_ITEM
    stat_location_col = o9Constants.STAT_LOCATION
    stat_channel_col = o9Constants.STAT_CHANNEL
    stat_region_col = o9Constants.STAT_REGION
    stat_account_col = o9Constants.STAT_ACCOUNT
    stat_pnl_col = o9Constants.STAT_PNL
    stat_dem_dom_col = o9Constants.STAT_DEMAND_DOMAIN

    partial_week_col = o9Constants.PARTIAL_WEEK
    week_col = o9Constants.WEEK
    month_col = o9Constants.MONTH
    planning_month_col = o9Constants.PLANNING_MONTH
    quarter_col = o9Constants.QUARTER
    planning_quarter_col = o9Constants.PLANNING_QUARTER
    day_key_col = o9Constants.DAY_KEY
    partial_week_key_col = o9Constants.PARTIAL_WEEK_KEY
    week_key_col = o9Constants.WEEK_KEY
    month_key_col = o9Constants.MONTH_KEY
    planning_month_key_col = o9Constants.PLANNING_MONTH_KEY
    quarter_key_col = o9Constants.QUARTER_KEY
    planning_quarter_key_col = o9Constants.PLANNING_QUARTER_KEY

    fcst_gen_time_bucket = o9Constants.FORECAST_GEN_TIME_BUCKET
    los_col = "Length of Series"
    num_zeros_col = "Number of Zeros"
    std_dev_col = "Std Dev L1"
    avg_col = "Avg Volume L1"
    cov_segment_col = "COV Segment L1"
    cumulative_vol_col = "Cumulative Volume % L1"
    vol_segment_col = "Volume Segment L1"
    plc_col = "PLC Status L1"
    history_sum_col = "History Sum"
    start_date_col = "Stat Intro Date System L1"
    end_date_col = "Stat Disc Date System L1"
    npi_date_col = "npi_date"
    eol_date_col = "eol_date"
    class_delimiter = ","
    volume_cov_history_period = "Volume-COV History Period"
    history_period = "History Period"
    new_launch_period = "New Launch Period"
    disco_period = "Disco Period"
    offset_period = "Offset Period"

    volume_col = "Volume L1"
    vol_share_col = "Volume % L1"

    # output columns
    weighted_intermittency = "Weighted Intermittency"
    weighted_acf_mass = "Weighted ACF Mass"
    weighted_cov = "Weighted COV"
    weighted_signal_to_noise_ratio = "Weighted Signal to Noise Ratio"
    weighted_spectral_ratio = "Weighted Spectral Ratio"
    total_volume = "Total Volume"
    intersection_count = "Intersection Count"

    intermittency_score = "Intermittency Score"
    cov_score = "COV Score"
    acf_mass_score = "ACF Mass Score"
    signal_to_noise_ratio_score = "Signal to Noise Ratio Score"
    spectral_ratio_score = "Spectral Ratio Score"
    forecastability_score = "Forecastability Score"

    product_customer_l1_segment = "Product Customer L1 Segment"

    cols_required_for_class_output = [
        version_col,
        class_col,
        weighted_cov,
        weighted_intermittency,
        weighted_acf_mass,
        weighted_signal_to_noise_ratio,
        weighted_spectral_ratio,
        total_volume,
        intersection_count,
    ]
    cols_required_for_score_output = [
        version_col,
        intermittency_score,
        cov_score,
        acf_mass_score,
        signal_to_noise_ratio_score,
        spectral_ratio_score,
        forecastability_score,
    ]
    cols_required_product_customer_l1_segment_output = [
        version_col,
        stat_pnl_col,
        stat_account_col,
        stat_region_col,
        stat_location_col,
        stat_channel_col,
        stat_dem_dom_col,
        stat_item_col,
        class_col,
        product_customer_l1_segment,
    ]


def get_metric_score(s, high_good):
    # compute median and iqr
    med = s.median()
    q75, q25 = s.quantile(0.75), s.quantile(0.25)
    iqr = q75 - q25

    # avoid division by zero
    denom = iqr if iqr != 0 else 1.0

    z = (s - med) / denom

    norm = 1 / (1 + np.exp(-z))
    if not high_good:
        norm = 1 - norm
    return norm


def decompose_ma_seasonal(series, seasonal_period):
    if len(series) < seasonal_period:
        seasonal_period = len(series) // 2
    if seasonal_period > 1:
        x = series.values.astype(float)
        n = len(x)
        kernel = np.ones(seasonal_period) / seasonal_period
        trend = np.convolve(x, kernel, mode="same")
        detrended = x - trend
        seasonal_means = np.array(
            [detrended[i::seasonal_period].mean() for i in range(seasonal_period)]
        )
        seasonal = seasonal_means[np.arange(n) % seasonal_period]
        signal = trend + seasonal
        noise = x - signal
        return signal.var() / noise.var()
    else:
        return np.nan


def cal_spectral_peak_ratio(series, fs=1):
    f, P = periodogram(series, fs=fs)
    return P.max() / P.sum() if P.sum() > 0 else np.nan


def acf_mass_sparse(x, K, seasonal_period, fft=False):
    """
    Compute sum(|ACF₁…ACF_K|) + |ACF_seasonal_period|
    without computing all intermediate lags via statsmodels.
    Uses pandas.Series.autocorr for the seasonal lag.
    """
    max_K = max(K)

    if len(x) <= max_K:
        max_K = len(x) // 2

    K = [val for val in K if val <= max_K]
    arr = np.asarray(x)
    s = pd.Series(arr)

    # 1) Compute ACF₁…ACF_K via statsmodels.acf
    acfs_short = acf(arr, nlags=max_K, fft=fft)  # [ACF₀, ACF₁, …, ACF_K]
    mass_1_to_K = np.abs(acfs_short[K]).sum()  # sum |ACF₁…ACF_K|

    if len(x) < seasonal_period:
        seasonal_acf = 0
    else:
        # 2) Compute ACF at lag = seasonal_period via pandas
        seasonal_acf = abs(s.autocorr(lag=seasonal_period))

    return mass_1_to_K + seasonal_acf


col_mapping = {
    constants.weighted_cov: "float",
    constants.weighted_intermittency: "float",
    constants.weighted_acf_mass: "float",
    constants.weighted_signal_to_noise_ratio: "float",
    constants.weighted_spectral_ratio: "float",
    constants.total_volume: "float",
    constants.intersection_count: "float",
    constants.intermittency_score: "float",
    constants.cov_score: "float",
    constants.acf_mass_score: "float",
    constants.signal_to_noise_ratio_score: "float",
    constants.spectral_ratio_score: "float",
    constants.forecastability_score: "float",
    constants.product_customer_l1_segment: "float",
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Rule,
    Grains,
    VolSegLvl,
    COVSegLvl,
    VolumeThresholdMeasure,
    COVThresholdMeasure,
    SegmentationVolumeCalcGrain,
    RequiredACFLagsInWeeks,
    ForecastScoreConfig,
    ForecastParameters,
    ForecastGenTimeBucket,
    TimeDimension,
    StatActual,
    StatLevels,
    CurrentTimePeriod,
    df_keys,
):
    try:
        if constants.forecast_iteration in ForecastGenTimeBucket.columns:
            AllScoreOutputList = list()
            AllClassOutputList = list()
            ProductSegmentOutputList = list()
            for the_iteration in ForecastGenTimeBucket[constants.forecast_iteration].unique():
                logger.warning(f"--- Processing iteration {the_iteration}")

                decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

                the_score_output, the_class_output, the_product_segment_l1 = decorated_func(
                    Rule=Rule,
                    Grains=Grains,
                    VolSegLvl=VolSegLvl,
                    COVSegLvl=COVSegLvl,
                    VolumeThresholdMeasure=VolumeThresholdMeasure,
                    COVThresholdMeasure=COVThresholdMeasure,
                    SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
                    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                    ForecastScoreConfig=ForecastScoreConfig,
                    ForecastParameters=ForecastParameters,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    TimeDimension=TimeDimension,
                    StatActual=StatActual,
                    StatLevels=StatLevels,
                    CurrentTimePeriod=CurrentTimePeriod,
                    df_keys=df_keys,
                )

                AllScoreOutputList.append(the_score_output)
                AllClassOutputList.append(the_class_output)
                ProductSegmentOutputList.append(the_product_segment_l1)

            ScoreOutput = concat_to_dataframe(AllScoreOutputList)
            ClassOutput = concat_to_dataframe(AllClassOutputList)
            ProductSegmentL1 = concat_to_dataframe(ProductSegmentOutputList)

        else:
            ScoreOutput, ClassOutput, ProductSegmentL1 = processIteration(
                Rule=Rule,
                Grains=Grains,
                VolSegLvl=VolSegLvl,
                COVSegLvl=COVSegLvl,
                VolumeThresholdMeasure=VolumeThresholdMeasure,
                COVThresholdMeasure=COVThresholdMeasure,
                SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
                RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                ForecastScoreConfig=ForecastScoreConfig,
                ForecastParameters=ForecastParameters,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                TimeDimension=TimeDimension,
                StatActual=StatActual,
                StatLevels=StatLevels,
                CurrentTimePeriod=CurrentTimePeriod,
                df_keys=df_keys,
            )

    except Exception as e:
        logger.exception(e)
        ScoreOutput, ClassOutput, ProductSegmentL1 = None, None, None
    return ScoreOutput, ClassOutput, ProductSegmentL1


def processIteration(
    Rule,
    Grains,
    VolSegLvl,
    COVSegLvl,
    VolumeThresholdMeasure,
    COVThresholdMeasure,
    SegmentationVolumeCalcGrain,
    RequiredACFLagsInWeeks,
    ForecastScoreConfig,
    ForecastParameters,
    ForecastGenTimeBucket,
    TimeDimension,
    StatActual,
    StatLevels,
    CurrentTimePeriod,
    df_keys,
):
    plugin_name = "DP087StatLevelRecommender"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    dimensions = get_list_of_grains_from_string(Grains)

    score_output = pd.DataFrame(columns=constants.cols_required_for_score_output)
    class_output = pd.DataFrame(columns=constants.cols_required_for_class_output)
    product_segment_l1_output = pd.DataFrame(
        columns=constants.cols_required_product_customer_l1_segment_output
    )
    try:
        if StatLevels is None:
            logger.warning(
                "StatLevels is empty for slice {}, returning empty dataframes as output ...".format(
                    df_keys
                )
            )
            return score_output, class_output, product_segment_l1_output

        if StatLevels.empty:
            logger.warning(
                "StatLevels is empty for slice {}, returning empty dataframes as output ...".format(
                    df_keys
                )
            )
            return score_output, class_output, product_segment_l1_output

        if TimeDimension.empty:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return score_output, class_output, product_segment_l1_output

        if StatActual is None:
            logger.warning(
                "StatActual is empty for slice {}, returning empty dataframes as output ...".format(
                    df_keys
                )
            )
            return score_output, class_output, product_segment_l1_output

        if StatActual.empty:
            logger.warning(
                "StatActual is empty for slice {}, returning empty dataframes as output ...".format(
                    df_keys
                )
            )
            return score_output, class_output, product_segment_l1_output

        if SegmentationVolumeCalcGrain == "None" or not SegmentationVolumeCalcGrain:
            vol_segmentation_level = [constants.version_col]
        else:
            vol_segmentation_level = get_list_of_grains_from_string(SegmentationVolumeCalcGrain)
        logger.info(f"vol_segmentation_level : {vol_segmentation_level}")

        time_key_cols = [
            constants.day_key_col,
            constants.partial_week_key_col,
            constants.week_key_col,
            constants.month_key_col,
            constants.planning_month_key_col,
            constants.quarter_key_col,
            constants.planning_quarter_key_col,
        ]
        for col in time_key_cols:
            if col in TimeDimension.columns:
                TimeDimension[col] = pd.to_datetime(TimeDimension[col], utc=True).dt.tz_localize(
                    None
                )
            if col in CurrentTimePeriod.columns:
                CurrentTimePeriod[col] = pd.to_datetime(
                    CurrentTimePeriod[col], utc=True
                ).dt.tz_localize(None)

        # getting RequiredACFLags
        RequiredACFLagsInWeeks = RequiredACFLagsInWeeks.split(",")

        # remove leading/trailing spaces if any
        RequiredACFLagsInWeeks = [x.strip() for x in RequiredACFLagsInWeeks]
        RequiredACFLags = [int(x) for x in RequiredACFLagsInWeeks]

        ForecastScoreConfig = ast.literal_eval(ForecastScoreConfig)
        ForecastScoreConfig = {
            key: value[1] for key, value in ForecastScoreConfig.items() if value[0]
        }
        weight_sum = sum(ForecastScoreConfig.values())
        if weight_sum != 0:
            ForecastScoreConfig = {
                key: [value / weight_sum] for key, value in ForecastScoreConfig.items()
            }

        Rule = ast.literal_eval(Rule)

        # Filter the required columns from dataframes
        req_cols = [
            constants.version_col,
            constants.new_launch_period,
            constants.disco_period,
            constants.volume_cov_history_period,
            constants.history_period,
            constants.offset_period,
        ]
        MiscThresholds = ForecastParameters[req_cols]

        history_measure = [col for col in StatActual.columns if not is_dimension(col)][0]
        logger.info("history_measure : {}".format(history_measure))

        req_cols = [
            constants.version_col,
            VolumeThresholdMeasure,
            COVThresholdMeasure,
        ]
        SegmentThresholds = ForecastParameters[req_cols]

        common_grains = list(set(dimensions + vol_segmentation_level))
        logger.info("common_grains : {}".format(common_grains))

        logger.info("dimensions : {}".format(dimensions))

        req_cols = [constants.partial_week_col] + common_grains + [history_measure]
        StatActual = StatActual[req_cols]

        # Join StatActual with Time mapping
        pw_time_mapping = TimeDimension[
            [constants.partial_week_col, constants.partial_week_key_col]
        ].drop_duplicates()
        StatActual = StatActual.merge(pw_time_mapping, on=constants.partial_week_col, how="inner")

        last_partial_week = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            constants.partial_week_col,
            constants.partial_week_key_col,
        )
        last_partial_week_key = TimeDimension[
            TimeDimension[constants.partial_week_col] == last_partial_week
        ][constants.partial_week_key_col].iloc[0]

        # filter out actuals prior to last_partial_week_key
        filter_clause = StatActual[constants.partial_week_key_col] <= last_partial_week_key
        logger.debug(f"StatActual: shape : {StatActual.shape}")
        logger.debug("Filtering records only prior to LastTimePeriod ...")
        StatActual = StatActual[filter_clause]
        logger.debug(f"StatActual: shape : {StatActual.shape}")

        # filter out leading zeros
        cumulative_sum_col = "_".join(["Cumulative", history_measure])
        StatActual.sort_values(common_grains + [constants.partial_week_key_col], inplace=True)
        StatActual[cumulative_sum_col] = StatActual.groupby(common_grains)[
            history_measure
        ].transform(pd.Series.cumsum)
        StatActual = StatActual[StatActual[cumulative_sum_col] > 0]
        logger.debug(f"Shape after filtering leading zeros : {StatActual.shape}")

        # StatActual might not be present for a particular slice, check and return empty dataframe
        if StatActual is None or StatActual.empty:
            logger.warning(
                f"StatActual is empty/None for slice {df_keys}, returning empty dataframes as output ..."
            )
            return score_output, class_output, product_segment_l1_output

        input_version = ForecastGenTimeBucket[constants.version_col].iloc[0]

        # split string into lists based on delimiter
        cov_levels = COVSegLvl.split(constants.class_delimiter)
        cov_thresholds = [round(float(x), 4) for x in list(SegmentThresholds[COVThresholdMeasure])]
        # remove duplicates if any
        cov_thresholds = list(set(cov_thresholds))

        # filter cov levels based on thresholds
        cov_levels = [cov_levels[i] for i in range(len(cov_thresholds) + 1)]

        logger.info("cov_levels : {}".format(cov_levels))
        logger.info("cov_thresholds : {}".format(cov_thresholds))

        # split string into lists based on delimiter
        vol_levels = VolSegLvl.split(constants.class_delimiter)
        vol_thresholds = [
            round(float(x), 4) for x in list(SegmentThresholds[VolumeThresholdMeasure])
        ]
        # remove duplicates if any
        vol_thresholds = list(set(vol_thresholds))

        # filter volume levels based on thresholds
        vol_levels = [vol_levels[i] for i in range(len(vol_thresholds) + 1)]

        logger.info("vol_levels : {}".format(vol_levels))
        logger.info("vol_thresholds : {}".format(vol_thresholds))

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[constants.fcst_gen_time_bucket].unique()[0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.week_col,
                constants.week_key_col,
            ]
            relevant_time_name = constants.week_col
            relevant_time_key = constants.week_key_col
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.month_col,
                constants.month_key_col,
            ]
            relevant_time_name = constants.month_col
            relevant_time_key = constants.month_key_col

            RequiredACFLags = [round(int(i) / 4.34524) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.planning_month_col,
                constants.planning_month_key_col,
            ]
            relevant_time_name = constants.planning_month_col
            relevant_time_key = constants.planning_month_key_col

            RequiredACFLags = [round(int(i) / 4.34524) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.quarter_col,
                constants.quarter_key_col,
            ]
            relevant_time_name = constants.quarter_col
            relevant_time_key = constants.quarter_key_col

            RequiredACFLags = [round(int(i) / 13) for i in RequiredACFLags]
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.planning_quarter_col,
                constants.planning_quarter_key_col,
            ]
            relevant_time_name = constants.planning_quarter_col
            relevant_time_key = constants.planning_quarter_key_col

            RequiredACFLags = [round(int(i) / 13) for i in RequiredACFLags]
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return score_output, class_output, product_segment_l1_output

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        seasonal_periods = get_seasonal_periods(frequency)

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Join Actuals with time mapping
        StatActual = StatActual.merge(base_time_mapping, on=constants.partial_week_col, how="inner")

        StatActual = StatActual[common_grains + [relevant_time_name, history_measure]]
        # select the relevant columns, groupby and sum history measure
        StatActual = (
            StatActual.groupby(common_grains + [relevant_time_name])
            .sum()[[history_measure]]
            .reset_index()
        )

        # Dictionary for easier lookups
        relevant_time_mapping_dict = dict(
            zip(
                list(relevant_time_mapping[relevant_time_name]),
                list(relevant_time_mapping[relevant_time_key]),
            )
        )

        segmentation_period = int(MiscThresholds[constants.volume_cov_history_period].iloc[0])
        history_periods = int(MiscThresholds[constants.history_period].iloc[0])
        npi_horizon = int(MiscThresholds[constants.new_launch_period].iloc[0])
        eol_horizon = int(MiscThresholds[constants.disco_period].iloc[0])
        offset_periods = 0  # default value
        if MiscThresholds[constants.offset_period].iloc[0] > 0:
            offset_periods = int(MiscThresholds[constants.offset_period].iloc[0])

        logger.info(
            f"segmentation_period : {segmentation_period}, history_periods : {history_periods}, new_launch_periods : f{npi_horizon}"
        )

        logger.info("---------------------------------------")

        logger.info("filtering rows where {} is not null ...".format(history_measure))
        StatActual = StatActual[StatActual[history_measure].notna()]

        if len(StatActual) == 0:
            logger.warning(
                "Actuals df is empty after filtering non null values for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return score_output, class_output, product_segment_l1_output

        # check if history measure sum is positive before proceeding further
        if StatActual[history_measure].sum() <= 0:
            logger.warning("Sum of actuals is non positive for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return score_output, class_output, product_segment_l1_output

        # cap negative values to zero
        StatActual[history_measure] = np.where(
            StatActual[history_measure] < 0, 0, StatActual[history_measure]
        )

        # Join actuals with time mapping
        StatActual_with_time_key = StatActual.copy().merge(
            relevant_time_mapping,
            on=relevant_time_name,
            how="inner",
        )

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # Gather the latest time name
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        # adjust the latest time according to the forecast iteration's offset before getting n periods for segmentation
        if offset_periods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info("Getting last {} period dates for PLC segmentation ...".format(history_periods))
        # note the negative sign to segmentation period
        last_n_periods_plc = get_n_time_periods(
            latest_time_name,
            -history_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )
        last_n_periods_plc_df = pd.DataFrame({relevant_time_name: last_n_periods_plc})

        # join to get relevant data
        SelectedHistory = StatActual_with_time_key.merge(
            last_n_periods_plc_df,
            on=relevant_time_name,
            how="inner",
        )
        if SelectedHistory.empty:
            logger.warning(
                f"No data found after filtering last {history_periods} periods from time mapping for PLC segmentation, slice : {df_keys} ..."
            )
            logger.warning("Returning empty df for this slice ...")
            return score_output, class_output, product_segment_l1_output

        # fill missing dates
        SelectedHistory_nas_filled = fill_missing_dates(
            actual=SelectedHistory.drop(relevant_time_key, axis=1),
            forecast_level=common_grains,
            history_measure=history_measure,
            relevant_time_key=relevant_time_key,
            relevant_time_name=relevant_time_name,
            relevant_time_periods=last_n_periods_plc,
            time_mapping=relevant_time_mapping,
            fill_nulls_with_zero=True,
        )

        # aggregation to get start and end date
        plc_df = SelectedHistory_nas_filled.groupby(common_grains).agg(
            {
                relevant_time_key: ["min", "max"],
            },
        )
        # set column names to dataframe
        plc_df.columns = [constants.start_date_col, constants.end_date_col]

        # reset index to obtain grain columns
        plc_df.reset_index(inplace=True)

        # get last n periods based on vol-cov segmentation period
        logger.info(
            "Getting last {} period dates for vol-cov segmentation ...".format(segmentation_period)
        )
        # note the negative sign to segmentation period
        last_n_periods_vol_cov = get_n_time_periods(
            latest_time_name,
            -segmentation_period,
            relevant_time_mapping,
            time_attribute_dict,
        )
        # convert to df for join
        last_n_period_vol_cov_df = pd.DataFrame({relevant_time_name: last_n_periods_vol_cov})

        if len(last_n_period_vol_cov_df) == 0:
            logger.warning(
                "No dates found after filtering last {} periods from time mapping for slice {}...".format(
                    segmentation_period, df_keys
                )
            )
            logger.warning("Returning empty dataframes for this slice ...")
            return score_output, class_output, product_segment_l1_output

        # filter relevant history based on dates provided above
        RecentHistory = SelectedHistory_nas_filled.merge(
            last_n_period_vol_cov_df,
            on=relevant_time_name,
            how="inner",
        )
        last_n_periods_npi = get_n_time_periods(
            latest_time_name,
            -npi_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )
        last_n_period_npi = last_n_periods_npi[0]
        npi_cutoff_date = relevant_time_mapping_dict[last_n_period_npi]
        logger.info("npi_cutoff_date : {}".format(npi_cutoff_date))
        plc_df[constants.npi_date_col] = npi_cutoff_date

        last_n_periods_eol = get_n_time_periods(
            latest_time_name,
            -eol_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )

        last_n_period_eol = last_n_periods_eol[0]
        eol_cutoff_date = relevant_time_mapping_dict[last_n_period_eol]
        logger.info("eol_cutoff_date : {}".format(eol_cutoff_date))
        plc_df[constants.eol_date_col] = eol_cutoff_date

        # filter out data for the disc horizon
        data_last_n_period_eol = SelectedHistory_nas_filled.merge(
            pd.DataFrame({relevant_time_name: last_n_periods_eol}),
            on=relevant_time_name,
            how="inner",
        )

        # take sum of history measure
        disc_data_df = data_last_n_period_eol[common_grains + [history_measure]]
        disc_data_df = disc_data_df.groupby(common_grains).sum()[[history_measure]]
        disc_data_df.rename(columns={history_measure: constants.history_sum_col}, inplace=True)

        # assign DISC where history sum is  zero
        disc_data_df[constants.plc_col] = "MATURE"
        disc_data_df.loc[disc_data_df[constants.history_sum_col] == 0, constants.plc_col] = "DISC"

        # reset index and select req cols
        disc_data_df.reset_index(inplace=True)
        disc_data_df = disc_data_df[common_grains + [constants.plc_col]]

        # join with plc df
        plc_df = plc_df.merge(disc_data_df, on=common_grains, how="outer")

        # assign categories NEW LAUNCH
        plc_df[constants.plc_col] = np.where(
            plc_df[constants.start_date_col] > plc_df[constants.npi_date_col],
            "NEW LAUNCH",
            plc_df[constants.plc_col],
        )

        # filter_clause = ~plc_df[constants.plc_col].isin(["NEW LAUNCH", "DISC"])

        # # assign category MATURE
        # plc_df[constants.plc_col] = np.where(filter_clause, "MATURE", plc_df[constants.plc_col])

        plc_df = plc_df[common_grains + [constants.plc_col]]

        recent_history_df = pd.DataFrame()
        recent_history_intersections = pd.DataFrame()
        if not RecentHistory.empty:
            logger.info("Calculating volume segments ...")
            volume_df = RecentHistory[common_grains + [history_measure]]
            volume_df = (
                volume_df.groupby(common_grains)
                .sum()[[history_measure]]
                .rename(columns={history_measure: constants.volume_col})
                .reset_index()
            )
            volume_df = volume_df[volume_df[constants.volume_col] > 0]

            # calculate total volume in respective slice
            volume_df[constants.total_volume] = volume_df.groupby(vol_segmentation_level)[
                [constants.volume_col]
            ].transform("sum")

            # calculate volume share populate zero into volume share where total volume is zero
            volume_df[constants.vol_share_col] = np.where(
                volume_df[constants.total_volume] > 0,
                volume_df[constants.volume_col] / volume_df[constants.total_volume],
                0,
            )
            # find cumulative sum within a group and assign volume segment
            volume_df = (
                volume_df.groupby(vol_segmentation_level)
                .apply(
                    lambda df: assign_volume_segment_to_group(
                        df,
                        constants.vol_share_col,
                        constants.cumulative_vol_col,
                        constants.vol_segment_col,
                        vol_thresholds,
                        vol_levels,
                    )
                )
                .reset_index(drop=True)
            )
            logger.info("Calculating variability segments ...")

            variability_df = pd.DataFrame(
                columns=common_grains + [constants.cov_segment_col, constants.weighted_cov]
            )
            if "CoV" in ForecastScoreConfig.keys():
                # groupby and calculate mean, std
                variability_df = RecentHistory.groupby(common_grains).agg(
                    {history_measure: ["mean", lambda x: np.std(x, ddof=1)]}
                )

                variability_df.columns = [constants.avg_col, constants.std_dev_col]

                # mean cannot be NA, std dev can be NA if there's only one value
                variability_df[constants.std_dev_col].fillna(0, inplace=True)

                # check and calculate cov
                variability_df[constants.weighted_cov] = np.where(
                    variability_df[constants.avg_col] > 0,
                    variability_df[constants.std_dev_col] / variability_df[constants.avg_col],
                    0,
                )

                # reset index to obtain the grain columns
                variability_df.reset_index(inplace=True)

                # assign variability segments
                variability_df[constants.cov_segment_col] = assign_segments(
                    variability_df[constants.weighted_cov].to_numpy(), cov_thresholds, cov_levels
                )

                ForecastScoreConfig["CoV"].append(constants.weighted_cov)
                ForecastScoreConfig["CoV"].append(constants.cov_score)

            logger.info("Merging volume, variability dataframes ...")

            recent_history_df = pd.merge(
                volume_df,
                variability_df[common_grains + [constants.cov_segment_col, constants.weighted_cov]],
                how="outer",
            )
            recent_history_intersections = recent_history_df[common_grains].drop_duplicates()

        if recent_history_intersections.empty:
            logger.warning(
                "No intersections found in recent history for slice {}, returning empty dataframes as output ...".format(
                    df_keys
                )
            )
            return score_output, class_output, product_segment_l1_output

        SelectedHistory = SelectedHistory_nas_filled.merge(recent_history_intersections)

        if "Intermittency" in ForecastScoreConfig.keys():
            SelectedHistory[constants.num_zeros_col] = SelectedHistory.groupby(common_grains)[
                history_measure
            ].transform(lambda x: (x == 0).sum())
            SelectedHistory[constants.los_col] = SelectedHistory.groupby(common_grains)[
                history_measure
            ].transform("count")

            SelectedHistory[constants.weighted_intermittency] = (
                SelectedHistory[constants.num_zeros_col] / SelectedHistory[constants.los_col]
            )
            ForecastScoreConfig["Intermittency"].append(constants.weighted_intermittency)
            ForecastScoreConfig["Intermittency"].append(constants.intermittency_score)

        if "ACF Mass" in ForecastScoreConfig.keys():
            SelectedHistory[constants.weighted_acf_mass] = SelectedHistory.groupby(common_grains)[
                history_measure
            ].transform(
                lambda x: acf_mass_sparse(
                    x, K=RequiredACFLags, seasonal_period=seasonal_periods, fft=False
                )
            )
            ForecastScoreConfig["ACF Mass"].append(constants.weighted_acf_mass)
            ForecastScoreConfig["ACF Mass"].append(constants.acf_mass_score)

        if "Spectral Ratio" in ForecastScoreConfig.keys():
            SelectedHistory[constants.weighted_spectral_ratio] = SelectedHistory.groupby(
                common_grains
            )[history_measure].transform(lambda x: cal_spectral_peak_ratio(series=x))
            ForecastScoreConfig["Spectral Ratio"].append(constants.weighted_spectral_ratio)
            ForecastScoreConfig["Spectral Ratio"].append(constants.spectral_ratio_score)

        if "Signal to Noise Ratio" in ForecastScoreConfig.keys():
            SelectedHistory[constants.weighted_signal_to_noise_ratio] = SelectedHistory.groupby(
                common_grains
            )[history_measure].transform(lambda x: decompose_ma_seasonal(x, seasonal_periods))
            ForecastScoreConfig["Signal to Noise Ratio"].append(
                constants.weighted_signal_to_noise_ratio
            )
            ForecastScoreConfig["Signal to Noise Ratio"].append(
                constants.signal_to_noise_ratio_score
            )

        SelectedHistory = pd.merge(
            SelectedHistory.drop(
                columns=[history_measure, relevant_time_name, relevant_time_key]
            ).drop_duplicates(),
            plc_df,
            how="left",
        )

        SelectedHistory = pd.merge(
            SelectedHistory,
            recent_history_df,
            how="outer",
        )

        groupby_cols = [constants.version_col]
        if "CoV" in ForecastScoreConfig.keys():
            SelectedHistory[constants.class_col] = np.where(
                SelectedHistory[constants.plc_col] == "MATURE",
                SelectedHistory[constants.vol_segment_col]
                + SelectedHistory[constants.cov_segment_col],
                SelectedHistory[constants.plc_col],
            )
            groupby_cols.append(constants.class_col)

        SelectedHistory[constants.version_col] = input_version
        product_segment_l1_output = SelectedHistory.copy()
        cols = [col[1] for col in ForecastScoreConfig.values()]
        SelectedHistory = SelectedHistory[groupby_cols + [constants.volume_col] + cols]

        SelectedHistory[constants.intersection_count] = SelectedHistory.groupby(
            groupby_cols, observed=True
        )[constants.version_col].transform("count")

        volume_sum_across_class = SelectedHistory.groupby(
            constants.class_col, observed=True, as_index=False
        )[[constants.volume_col]].sum(min_count=1)

        SelectedHistory = SelectedHistory.merge(
            volume_sum_across_class.rename(columns={constants.volume_col: constants.total_volume}),
            on=constants.class_col,
        )

        SelectedHistory[cols] = SelectedHistory[cols].multiply(
            SelectedHistory[constants.volume_col] / SelectedHistory[constants.total_volume], axis=0
        )
        SelectedHistory = SelectedHistory.groupby(
            groupby_cols + [constants.intersection_count], observed=True, as_index=False
        )[[constants.volume_col] + cols].sum(min_count=1)

        SelectedHistory.rename(columns={constants.volume_col: constants.total_volume}, inplace=True)

        for col in constants.cols_required_for_class_output:
            if col not in SelectedHistory.columns:
                SelectedHistory[col] = np.nan
        score_output = SelectedHistory.copy()
        class_output = SelectedHistory[constants.cols_required_for_class_output]

        output_cols = []
        for col, value in Rule.items():
            input_col = ForecastScoreConfig[col][1]
            output_col = ForecastScoreConfig[col][2]
            score_output[output_col] = get_metric_score(s=score_output[input_col], high_good=value)
            output_cols.append(output_col)

        score_output = score_output[[constants.version_col, constants.total_volume] + output_cols]
        score_output[output_cols] = score_output[output_cols].multiply(
            score_output[constants.total_volume] / score_output[constants.total_volume].sum(),
            axis=0,
        )
        score_output = score_output.groupby([constants.version_col], observed=True, as_index=False)[
            output_cols
        ].sum(min_count=1)

        ForecastScoreConfig = {value[2]: value[0] for key, value in ForecastScoreConfig.items()}
        score_df = pd.DataFrame([ForecastScoreConfig])
        weights = score_df[output_cols].values.flatten().tolist()

        score_output[constants.forecastability_score] = (score_output[output_cols] * weights).sum(
            axis=1
        )

        product_segment_l1_output[constants.product_customer_l1_segment] = 1.0

        product_segment_l1_output = product_segment_l1_output[
            constants.cols_required_product_customer_l1_segment_output
        ]
        score_output = score_output[constants.cols_required_for_score_output]
        class_output = class_output[constants.cols_required_for_class_output]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
    return (
        score_output,
        class_output,
        product_segment_l1_output,
    )

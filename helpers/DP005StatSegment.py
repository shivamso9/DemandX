import logging
from functools import reduce

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
from o9Reference.stat_utils.assign_segments import assign_segments
from o9Reference.stat_utils.segmentation_utils import assign_volume_segment_to_group
from pmdarima.arima import ndiffs
from sklearn.linear_model import LinearRegression

from helpers.o9Constants import o9Constants
from helpers.seasonality import ACFDetector

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
logger = logging.getLogger("o9_logger")
pd.options.mode.chained_assignment = None

col_mapping = {
    "Length of Series": float,
    "Number of Zeros": float,
    "Product Segment": float,
    "Intermittent": str,
    "PLC Status": str,
    "Trend": str,
    "Seasonality": str,
    "Std Dev": float,
    "Avg Volume": float,
    "COV": float,
    "Volume": float,
    "Volume %": float,
    "Cumulative Volume %": float,
    "Stat Intro Date System": "datetime64[ns]",
    "Stat Disc Date System": "datetime64[ns]",
    "Item Class": str,
}


def add_seg_lob_to_actual(
    SegmentationLOBGroupFilter: pd.DataFrame,
    ItemMasterData: pd.DataFrame,
    Actual: pd.DataFrame,
) -> pd.DataFrame:
    # extract segmentation lob level
    the_level = SegmentationLOBGroupFilter[o9Constants.SEGMENTATION_LOB_GROUP_FILTER].iloc[0]

    # add dimension and square brackets
    the_level = f"Item.[{the_level}]"

    # look up and populate item master
    ItemMasterData[o9Constants.SEGMENTATION_LOB] = ItemMasterData[the_level]

    # join with Actual
    Actual = Actual.merge(
        ItemMasterData[[o9Constants.PLANNING_ITEM, o9Constants.SEGMENTATION_LOB]].drop_duplicates(),
        on=o9Constants.PLANNING_ITEM,
        how="inner",
    )

    return Actual


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

        RateChange = (fitted_line[-1][0] - fitted_line[0][0]) / fitted_line[0][0]

    except Exception as e:
        logger.exception(e)
    return RateChange


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    COVSegLvl,
    Actual,
    Grains,
    DimClass,
    TimeDimension,
    VolSegLvl,
    CurrentTimePeriod,
    ReadFromHive,
    SegmentationVolumeCalcGrain,
    VolumeThreshold,
    COVThreshold,
    IntermittencyThreshold,
    NewLaunchPeriodInWeeks,
    DiscoPeriodInWeeks,
    HistoryMeasure,
    VolumeCOVHistoryPeriodInWeeks,
    HistoryPeriodInWeeks,
    ForecastGenerationTimeBucket,
    df_keys,
    SegmentationLOBGroupFilter,
    ItemMasterData,
    AlphaACF: str = "0.05",
    ACFLowerThreshold: str = "-0.10",
    ACFUpperThreshold: str = "0.90",
    ACFSkipLags: str = "11",
    ACFDiff: str = "0",
    RequiredACFLags: str = "12",
    useACFForSeasonality: str = "False",
    OffsetPeriods: int = 0,
    TrendThreshold: int = 20,
):
    plugin_name = "DP005StatSegment_py"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here
    version_col = "Version.[Version Name]"
    class_delimiter = ","

    # Volume Segmentation
    vol_segment_col = "Volume Segment"

    # variability segmentation
    cov_segment_col = "Variability Segment"

    # PLC Segmentation
    npi_date_col = "npi_date"
    eol_date_col = "eol_date"

    zero_ratio_col = "ZeroRatio"

    seasonal_diff_col = "ndiffs"
    trend_factor_col = "TrendFactor"

    total_vol_col = "Total Volume"
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
    partial_week_key_col = "Time.[PartialWeekKey]"
    history_sum_col = "History Sum"

    # output measures - StatSegmentation
    los_col = "Length of Series"
    num_zeros_col = "Number of Zeros"
    intermittency_col = "Intermittent"
    plc_col = "PLC Status"
    trend_col = "Trend"
    seasonality_col = "Seasonality"
    seasonal_periods_col = "Seasonal Periods"
    std_dev_col = "Std Dev"
    avg_col = "Avg Volume"
    cov_col = "COV"
    volume_col = "Volume"
    vol_share_col = "Volume %"
    cumulative_vol_col = "Cumulative Volume %"
    stat_intro_date_col = "Stat Intro Date System"
    stat_disc_date_col = "Stat Disc Date System"

    # output measures - ProductSegmentation
    prod_segment_col = "Product Segment"

    # output measures - ItemClass
    item_class_col = "Item Class"
    ml_volume_segment_col = "ML Item Volume Segment"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    segmentation_output_grains = [str(x) for x in all_grains if x != "NA" and x != ""]

    stat_segmentation_cols = (
        [version_col]
        + segmentation_output_grains
        + [
            los_col,
            num_zeros_col,
            intermittency_col,
            plc_col,
            trend_col,
            seasonality_col,
            std_dev_col,
            avg_col,
            cov_col,
            volume_col,
            vol_share_col,
            cumulative_vol_col,
            stat_intro_date_col,
            stat_disc_date_col,
        ]
    )
    prod_segmentation_cols = (
        [version_col] + segmentation_output_grains + [DimClass, prod_segment_col]
    )
    item_class_output_cols = (
        [version_col] + segmentation_output_grains + [item_class_col, ml_volume_segment_col]
    )

    # Define output dataframes with valid schema
    StatSegmentation = pd.DataFrame(columns=stat_segmentation_cols)
    ProductSegmentation = pd.DataFrame(columns=prod_segmentation_cols)
    ItemClass = pd.DataFrame(columns=item_class_output_cols)
    try:
        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        history_measure = HistoryMeasure
        if ReadFromHive:
            history_measure = "DP005" + history_measure

        logger.info("history_measure : {}".format(history_measure))

        if SegmentationVolumeCalcGrain == "None" or not SegmentationVolumeCalcGrain:
            vol_segmentation_level = [version_col]
        else:
            vol_segmentation_level = [x.strip() for x in SegmentationVolumeCalcGrain.split(",")]

        logger.info(f"vol_segmentation_level : {vol_segmentation_level}")

        logger.info("Extracting segmentation level ...")

        # Combine grains, drop duplicates
        common_grains = list(set(segmentation_output_grains + vol_segmentation_level))

        logger.info("segmentation_output_grains : {}".format(segmentation_output_grains))

        logger.info("common_grains : {}".format(common_grains))

        if Actual is None or len(Actual) == 0:
            logger.warning("Actuals is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        input_version = Actual[version_col].iloc[0]

        Actual = add_seg_lob_to_actual(
            SegmentationLOBGroupFilter=SegmentationLOBGroupFilter,
            ItemMasterData=ItemMasterData,
            Actual=Actual,
        )
        req_cols = [partial_week_col] + common_grains + [history_measure]

        Actual = Actual[req_cols]

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

        # filter out actuals prior to last_partial_week_key
        filter_clause = Actual[partial_week_key_col] <= last_partial_week_key
        logger.debug(f"Actual: shape : {Actual.shape}")
        logger.debug("Filtering records only prior to LastTimePeriod ...")
        Actual = Actual[filter_clause]
        logger.debug(f"Actual: shape : {Actual.shape}")

        # filter out leading zeros
        cumulative_sum_col = "_".join(["Cumulative", history_measure])
        Actual.sort_values(common_grains + [partial_week_key_col], inplace=True)
        Actual[cumulative_sum_col] = Actual.groupby(common_grains)[history_measure].transform(
            pd.Series.cumsum
        )
        Actual = Actual[Actual[cumulative_sum_col] > 0]
        logger.debug(f"Shape after filtering leading zeros : {Actual.shape}")

        if Actual is None or len(Actual) == 0:
            logger.warning("Actuals is None/Empty for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        # split string into lists based on delimiter
        cov_levels = COVSegLvl.split(class_delimiter)
        cov_thresholds = [round(float(x), 4) for x in [float(x) for x in COVThreshold.split(",")]]
        # remove duplicates if any
        cov_thresholds = list(set(cov_thresholds))

        logger.info("cov_levels : {}".format(cov_levels))
        logger.info("cov_thresholds : {}".format(cov_thresholds))

        # split string into lists based on delimiter
        vol_levels = VolSegLvl.split(class_delimiter)
        vol_thresholds = [
            round(float(x), 4) for x in [float(x) for x in VolumeThreshold.split(",")]
        ]
        # remove duplicates if any
        vol_thresholds = list(set(vol_thresholds))

        logger.info("vol_levels : {}".format(vol_levels))
        logger.info("vol_thresholds : {}".format(vol_thresholds))

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return StatSegmentation, ProductSegmentation, ItemClass

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenerationTimeBucket
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        # getting RequiredACFLags
        RequiredACFLags = RequiredACFLags.split(",")

        # remove leading/trailing spaces if any
        RequiredACFLags = [x.strip() for x in RequiredACFLags]
        RequiredACFLags = [int(x) for x in RequiredACFLags]

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col

            # get the history periods
            history_periods = int(HistoryPeriodInWeeks)
            segmentation_period = int(VolumeCOVHistoryPeriodInWeeks)
            npi_horizon = int(NewLaunchPeriodInWeeks)
            eol_horizon = int(DiscoPeriodInWeeks)

        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col

            # get the history periods - convert to months
            history_periods = round(int(HistoryPeriodInWeeks) / 4.34524)
            segmentation_period = round(int(VolumeCOVHistoryPeriodInWeeks) / 4.34524)
            npi_horizon = round(int(NewLaunchPeriodInWeeks) / 4.34524)
            eol_horizon = round(int(DiscoPeriodInWeeks) / 4.34524)

            RequiredACFLags = [round(int(i) / 4.34524) for i in RequiredACFLags]

        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col

            # get the history periods - convert to months
            history_periods = round(int(HistoryPeriodInWeeks) / 4.34524)
            segmentation_period = round(int(VolumeCOVHistoryPeriodInWeeks) / 4.34524)
            npi_horizon = round(int(NewLaunchPeriodInWeeks) / 4.34524)
            eol_horizon = round(int(DiscoPeriodInWeeks) / 4.34524)

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

            # get the history periods - convert to quarter
            history_periods = round(int(HistoryPeriodInWeeks) / 13)
            segmentation_period = round(int(VolumeCOVHistoryPeriodInWeeks) / 13)
            npi_horizon = round(int(NewLaunchPeriodInWeeks) / 13)
            eol_horizon = round(int(DiscoPeriodInWeeks) / 13)

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

            RequiredACFLags = [round(int(i) / 13) for i in RequiredACFLags]

            # get the history periods - convert to quarter
            history_periods = round(int(HistoryPeriodInWeeks) / 13)
            segmentation_period = round(int(VolumeCOVHistoryPeriodInWeeks) / 13)
            npi_horizon = round(int(NewLaunchPeriodInWeeks) / 13)
            eol_horizon = round(int(DiscoPeriodInWeeks) / 13)

        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return StatSegmentation, ProductSegmentation, ItemClass

        logger.info(f"history_periods : {history_periods}")
        logger.info(f"segmentation_period : {segmentation_period}")
        logger.info(f"npi_horizon : {npi_horizon}")
        logger.info(f"eol_horizon : {eol_horizon}")

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

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
            Actual.groupby(common_grains + [relevant_time_name])
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

        logger.info("filtering rows where {} is not null ...".format(history_measure))
        Actual = Actual[Actual[history_measure].notna()]

        if len(Actual) == 0:
            logger.warning(
                "Actuals df is empty after filtering non null values for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        # check if history measure sum is positive before proceeding further
        if Actual[history_measure].sum() <= 0:
            logger.warning("Sum of actuals is non positive for slice {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        # cap negative values to zero
        Actual[history_measure] = np.where(Actual[history_measure] < 0, 0, Actual[history_measure])

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # Gather the latest time name
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # Adjust latest time according to the forecast iteration's offset
        if OffsetPeriods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -OffsetPeriods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info(f"latest_time_name after offset {OffsetPeriods} : {latest_time_name} ...")

        intermittency_threshold = float(IntermittencyThreshold)

        # Join actuals with time mapping
        Actual_with_time_key = Actual.copy().merge(
            relevant_time_mapping,
            on=relevant_time_name,
            how="inner",
        )

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
        plc_segmentation_input = Actual_with_time_key.merge(
            last_n_periods_plc_df,
            on=relevant_time_name,
            how="inner",
        )

        if len(plc_segmentation_input) == 0:
            logger.warning(
                "No data found after filtering last {} periods from time mapping for PLC segmentation, slice : {} ...".format(
                    history_periods, df_keys
                )
            )
            logger.warning("Returning empty df for this slice ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        # fill missing dates
        plc_segmentation_input_nas_filled = fill_missing_dates(
            actual=plc_segmentation_input.drop(relevant_time_key, axis=1),
            forecast_level=common_grains,
            history_measure=history_measure,
            relevant_time_key=relevant_time_key,
            relevant_time_name=relevant_time_name,
            relevant_time_periods=last_n_periods_plc,
            time_mapping=relevant_time_mapping,
            fill_nulls_with_zero=True,
        )

        logger.info("Calculating start,end dates for PLC Segmentation ...")

        # aggregation to get start and end date - if we use NAs filled df DISC Date won't be accurate
        plc_df = plc_segmentation_input.groupby(common_grains).agg(
            {
                relevant_time_key: [np.min, np.max],
            },
        )

        # set column names to dataframe
        plc_df.columns = [stat_intro_date_col, stat_disc_date_col]

        # reset index to obtain grain columns
        plc_df.reset_index(inplace=True)

        # get NPI, EOL Dates
        last_n_periods_npi = get_n_time_periods(
            latest_time_name,
            -npi_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )
        last_n_period_npi = last_n_periods_npi[0]
        npi_cutoff_date = relevant_time_mapping_dict[last_n_period_npi]
        logger.info("npi_cutoff_date : {}".format(npi_cutoff_date))
        plc_df[npi_date_col] = npi_cutoff_date

        last_n_periods_eol = get_n_time_periods(
            latest_time_name,
            -eol_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )

        last_n_period_eol = last_n_periods_eol[0]
        eol_cutoff_date = relevant_time_mapping_dict[last_n_period_eol]
        logger.info("eol_cutoff_date : {}".format(eol_cutoff_date))
        plc_df[eol_date_col] = eol_cutoff_date

        # filter out data for the disc horizon
        data_last_n_period_eol = plc_segmentation_input_nas_filled.merge(
            pd.DataFrame({relevant_time_name: last_n_periods_eol}),
            on=relevant_time_name,
            how="inner",
        )

        # take sum of history measure
        disc_data_df = data_last_n_period_eol.groupby(common_grains).sum()[[history_measure]]
        disc_data_df.rename(columns={history_measure: history_sum_col}, inplace=True)

        # assign DISC where history sum is  zero
        disc_data_df[plc_col] = np.where(disc_data_df[history_sum_col] == 0, "DISC", np.nan)

        # reset index and select req cols
        disc_data_df.reset_index(inplace=True)
        disc_data_df = disc_data_df[common_grains + [plc_col]]

        # join with plc df
        plc_df = plc_df.merge(disc_data_df, on=common_grains, how="outer")

        # assign categories NEW LAUNCH
        plc_df[plc_col] = np.where(
            plc_df[stat_intro_date_col] > plc_df[npi_date_col],
            "NEW LAUNCH",
            plc_df[plc_col],
        )

        filter_clause = ~plc_df[plc_col].isin(["NEW LAUNCH", "DISC"])

        # assign category MATURE
        plc_df[plc_col] = np.where(filter_clause, "MATURE", plc_df[plc_col])

        # null out stat_disc_date where products are not DISC
        plc_df[stat_disc_date_col] = np.where(
            plc_df[plc_col] == "DISC", plc_df[stat_disc_date_col], pd.NaT
        )

        # convert to datetime - above step converts the column into long (seconds)
        plc_df[stat_disc_date_col] = pd.to_datetime(plc_df[stat_disc_date_col])

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
                "No dates found after filtering last {} periods from time mapping for slice : {}...".format(
                    segmentation_period, df_keys
                )
            )
            logger.warning("Returning empty dataframes for this slice ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        logger.info("Joining actuals with time mapping with last n period dates ... ")
        # filter relevant history based on dates provided above
        vol_cov_segmentation_input = plc_segmentation_input_nas_filled.merge(
            last_n_period_vol_cov_df,
            on=relevant_time_name,
            how="inner",
        )
        # create a copy and use for subsequent joins with other dataframes
        result = plc_df.copy()

        if len(vol_cov_segmentation_input) > 0:
            logger.info("Calculating volume segments ...")
            # groupby and take aggregate volume
            volume_df = (
                vol_cov_segmentation_input.groupby(common_grains)
                .sum()[[history_measure]]
                .rename(columns={history_measure: volume_col})
                .reset_index()
            )

            # calculate total volume in respective slice
            volume_df[total_vol_col] = volume_df.groupby(vol_segmentation_level)[
                [volume_col]
            ].transform(sum)

            # calculate volume share populate zero into volume share where total volume is zero
            volume_df[vol_share_col] = np.where(
                volume_df[total_vol_col] > 0,
                volume_df[volume_col] / volume_df[total_vol_col],
                0,
            )

            logger.info("Calculating cumulative volume and assigning volume segments ...")
            # find cumulative sum within a group and assign volume segment
            volume_df = (
                volume_df.groupby(vol_segmentation_level)
                .apply(
                    lambda df: assign_volume_segment_to_group(
                        df,
                        vol_share_col,
                        cumulative_vol_col,
                        vol_segment_col,
                        vol_thresholds,
                        vol_levels,
                    )
                )
                .reset_index(drop=True)
            )

            logger.info("Calculating variability segments ...")

            # groupby and calculate mean, std
            variability_df = vol_cov_segmentation_input.groupby(common_grains).agg(
                {history_measure: [np.mean, lambda x: np.std(x, ddof=1)]}
            )

            variability_df.columns = [avg_col, std_dev_col]

            # mean cannot be NA, std dev can be NA if there's only one value
            variability_df[std_dev_col].fillna(0, inplace=True)

            # check and calculate cov
            variability_df[cov_col] = np.where(
                variability_df[avg_col] > 0,
                variability_df[std_dev_col] / variability_df[avg_col],
                0,
            )

            # reset index to obtain the grain columns
            variability_df.reset_index(inplace=True)

            # assign variability segments
            variability_df[cov_segment_col] = assign_segments(
                variability_df[cov_col].to_numpy(), cov_thresholds, cov_levels
            )
            logger.info("Merging volume, variability, plc dataframes ...")

            result = reduce(
                lambda x, y: pd.merge(x, y, on=common_grains, how="outer"),
                [volume_df, variability_df, plc_df],
            )

            logger.info("Merge complete, shape  : {}".format(result.shape))

            # Fill NAs with defaults - low volume and high variability segments
            result[vol_segment_col].fillna(vol_levels[-1], inplace=True)
            result[cov_segment_col].fillna(cov_levels[-1], inplace=True)

            logger.info("Assigning final PLC Segment ...")
            result[DimClass] = np.where(
                result[plc_col] == "MATURE",
                result[vol_segment_col] + result[cov_segment_col],
                result[plc_col],
            )

        # Use default thresholds for trend and seasonality since this won't be coming from the tenant base configuration

        trend_threshold = TrendThreshold / 100
        logger.info(f"trend_threshold : {trend_threshold}")

        seasonality_threshold = 0.64
        logger.info(f"seasonality_threshold : {seasonality_threshold}")

        # sort the dataframe by common grains and time so that array is ordered for trend calculation
        plc_segmentation_input_nas_filled.sort_values(
            common_grains + [relevant_time_key], inplace=True
        )

        logger.info("Calculating trend and seasonality...")

        useACFForSeasonality = eval(useACFForSeasonality)

        ts_attributes_df_list = list()
        grouped = plc_segmentation_input_nas_filled.groupby(common_grains)
        total_groups = grouped.ngroups
        logger.info(f"total_groups : {total_groups}")
        progress_interval = total_groups // 10 if total_groups >= 10 else 1

        for the_idx, (the_group, the_df) in enumerate(grouped):
            the_los = len(the_df)
            tmp = the_df.iloc[:]
            counter: str = "Counter"
            tmp[counter] = np.arange(1, len(tmp) + 1)
            X = tmp[counter].values.reshape(-1, 1)  # Reshape to a 2D array
            y = tmp[history_measure].values
            the_trend_factor = get_trend_rate_of_change(X, y)

            if useACFForSeasonality:
                if (
                    the_los >= 2 * get_seasonal_periods(frequency)
                    and the_df[history_measure].sum() > 0
                ):

                    acf_seasonal_presence, acf_seasonalities = ACFDetector(
                        df_sub=the_df,
                        measure_name=history_measure,
                        freq=frequency,
                        skip_lags=int(ACFSkipLags),
                        diff=int(ACFDiff),
                        alpha=float(AlphaACF),
                        lower_ci_threshold=float(ACFLowerThreshold),
                        upper_ci_threshold=float(ACFUpperThreshold),
                        RequiredACFLags=RequiredACFLags,
                    )
                    # use same placeholder as diff, will consolidate after for loop
                    the_seasonal_diff = acf_seasonal_presence
                    the_seasonal_periods = acf_seasonalities
                else:
                    the_seasonal_diff = False
                    the_seasonal_periods = None
            else:
                # backward compatibility
                the_seasonal_diff = ndiffs(y, alpha=0.05, test="kpss")
                the_seasonal_periods = None

            the_num_zeros = the_df[history_measure].value_counts()[0] if 0 in y else 0

            the_summary_dict = {col: val for col, val in zip(common_grains, the_group)}
            the_summary_dict[los_col] = the_los
            the_summary_dict[trend_factor_col] = the_trend_factor
            the_summary_dict[seasonal_diff_col] = the_seasonal_diff
            the_summary_dict[seasonal_periods_col] = the_seasonal_periods
            the_summary_dict[num_zeros_col] = the_num_zeros

            the_attributes = pd.DataFrame(the_summary_dict, index=[0])

            ts_attributes_df_list.append(the_attributes)
            if (the_idx + 1) % progress_interval == 0:
                logger.info(f"Progress: {((the_idx + 1) / total_groups) * 100:.1f}% complete.")

        ts_attributes_df = concat_to_dataframe(list_of_results=ts_attributes_df_list)

        # calculate zero ratio
        ts_attributes_df[zero_ratio_col] = (
            ts_attributes_df[num_zeros_col] / ts_attributes_df[los_col]
        )

        # we might have records with PLC status, but no data to calculate time series attributes
        result = ts_attributes_df.merge(result, on=common_grains, how="right")

        logger.info("df shape after combining plc with ts attributes : {}".format(result.shape))

        # Assign trend category
        conditions = [
            result[trend_factor_col] > trend_threshold,
            (result[trend_factor_col] <= trend_threshold)
            & (result[trend_factor_col] >= -trend_threshold),
            result[trend_factor_col] < -trend_threshold,
        ]
        choices = ["UPWARD", "NO TREND", "DOWNWARD"]

        logger.info("Assigning trend categories ...")
        result[trend_col] = np.select(conditions, choices, default=None)

        logger.info("Assigning seasonality categories ...")

        if useACFForSeasonality:
            # create filter clause
            # seasonality_calc_filter = (result[los_col] > 2 * get_seasonal_periods(frequency)) & (
            #     result[seasonal_diff_col] == True
            # )
            seasonality_calc_filter = (
                result[los_col] > 2 * get_seasonal_periods(frequency)
            ) & result[seasonal_diff_col]
        else:
            # create filter clause
            seasonality_calc_filter = (result[los_col] > 2 * get_seasonal_periods(frequency)) & (
                result[seasonal_diff_col] > seasonality_threshold
            )

        # Assign seasonality category
        result[seasonality_col] = np.where(
            seasonality_calc_filter,
            "Exists",
            "Does Not Exist",
        )

        logger.info("Assigning intermittency categories ...")
        # Assign intermittency category
        result[intermittency_col] = np.where(
            result[zero_ratio_col] >= intermittency_threshold, "YES", "NO"
        )

        # collect version from input data
        result[version_col] = input_version

        # Assign 1.0 value to product segment column
        result[prod_segment_col] = 1.0

        logger.info("Filtering relevant columns to output ...")

        if len(vol_cov_segmentation_input) > 0 and len(plc_segmentation_input) > 0:
            # Filter relevant columns
            StatSegmentation = result[stat_segmentation_cols]

        # Filter relevant columns
        ProductSegmentation = result[prod_segmentation_cols]

        # copy values from DimClass
        result[item_class_col] = result[DimClass]
        result[ml_volume_segment_col] = result[vol_segment_col]
        ItemClass = result[item_class_output_cols]

        logger.info("Successfully executed {} ...".format(plugin_name))
        logger.info("---------------------------------------------")
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        StatSegmentation = pd.DataFrame(columns=stat_segmentation_cols)
        ProductSegmentation = pd.DataFrame(columns=prod_segmentation_cols)
        ItemClass = pd.DataFrame(columns=item_class_output_cols)

    return StatSegmentation, ProductSegmentation, ItemClass

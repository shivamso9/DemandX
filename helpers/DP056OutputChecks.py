import copy
from collections import defaultdict
from logging import getLogger
from math import isclose, isnan

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from pandas import DataFrame, options, set_option, to_datetime
from pandas.api.types import is_numeric_dtype
from scipy.signal import detrend
from sklearn.linear_model import LinearRegression

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

logger = getLogger("o9_logger")


options.display.max_rows = 25
options.display.max_columns = 50
options.display.max_colwidth = 100
set_option("display.width", 1000)
options.display.precision = 3
options.mode.chained_assignment = None


plugin_name = "DP056OutputChecks"


def get_seasonal_periods(frequency: str) -> int:
    """
    Returns num seasonal periods based on frequency provided
    """
    if frequency in ["Daily", "Day"]:
        return 365
    elif frequency in ["Weekly", "Week"]:
        return 52
    elif frequency in ["Monthly", "Month", "Planning Month"]:
        return 12
    elif frequency in ["Quarterly", "Quarter", "Planning Quarter"]:
        return 4
    else:
        raise ValueError(
            "Unknown frequency {}, supported frequencies are Daily, Weekly, Monthly and Quarterly".format(
                frequency
            )
        )


def check_straight_line(_series) -> int:

    _series = _series.values

    differences = np.diff(_series)

    if all(isclose(x, _series[0], rel_tol=0.01) for x in _series):
        return 1
    elif all(isclose(x, differences[0], rel_tol=0.01) for x in differences):
        return 1
    else:
        if max(differences) == 0:
            return 1
        else:
            threshold = 0.01
            diff_range = max(differences) - min(differences)
            if (diff_range / max(differences)) <= threshold:
                return 1
            else:
                return 0


def check_straight_line_with_regression(_series) -> int:
    _series = _series.astype(int)
    x = np.arange(1, len(_series) + 1)

    # Clipping off 10% on either side of the time series
    _series = np.clip(
        _series,
        a_min=np.quantile(_series, 0.10),
        a_max=np.quantile(_series, 0.90),
    )

    model = LinearRegression().fit(x.reshape(-1, 1), _series)

    y_pred = model.coef_[0] * x + model.intercept_
    # Calculate the R-squared value of the model
    r_squared = model.score(y_pred.reshape(-1, 1), _series)
    # Plot the fitted line
    # import matplotlib.pyplot as plt
    # # Plot the data
    # plt.scatter(x, _series)
    # plt.xlabel('Time')
    # plt.ylabel('Series Values')
    # plt.title('Sample Time Series')

    # plt.plot(x, y_pred, color='red')
    # plt.show()
    if r_squared >= 0.95:
        return 1
    else:
        return 0


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

        max_value_15_percent = 0.15 * np.quantile(y, q=1)
        if abs(fitted_line[0][0] - fitted_line[-1][0]) > max_value_15_percent:
            RateChange = (fitted_line[-1][0] - fitted_line[0][0]) / fitted_line[0][0]

    except Exception as e:
        logger.exception(e)
    return RateChange


def reasonability_buckets_check(
    curActual: pd.DataFrame,
    curForecast: pd.DataFrame,
    Actual: pd.DataFrame,
    time_grain: str,
    actual_lastNbuckets_cols: list,
    forecast_cols: list,
    actual_measure: str,
    actual_dates: list,
    fcst_dates: list,
    cycle_period: str,
    version_col: str,
    stat_algo_col: str,
    fcst_next_n_buckets_col: str,
    actual_last_n_buckets_col: str,
    forecast_level: list,
    SYSTEM_STAT_FORECAST: str,
    PLANNER_STAT_FORECAST: str,
    df_keys: dict,
):
    ActualLastNBuckets = pd.DataFrame(columns=actual_lastNbuckets_cols)
    FcstNextNBuckets = pd.DataFrame(columns=forecast_cols)
    try:
        # if there's less than one cycle of data, skip calculations
        if len(Actual) < cycle_period:
            return ActualLastNBuckets, FcstNextNBuckets

        # filter data for relevant time grain
        desired_actuals = curActual[curActual[time_grain].isin(actual_dates)]
        desired_fcsts = curForecast[curForecast[time_grain].isin(fcst_dates)]

        ActualLastNBuckets = (
            desired_actuals.groupby([version_col] + forecast_level).sum().reset_index()
        )
        ActualLastNBuckets.rename(columns={actual_measure: actual_last_n_buckets_col}, inplace=True)
        desired_fcsts.drop(PLANNER_STAT_FORECAST, axis=1, inplace=True)
        # select the stat_fcst cols
        stat_fcst_cols = [x for x in desired_fcsts.columns if "Fcst" in x]

        # drop the time grain and aggregate - min_count=1 is to ensure that values get populated only for the assigned algorithms
        FcstNextNBuckets = (
            desired_fcsts.groupby([version_col] + forecast_level)[stat_fcst_cols]
            .sum(min_count=1)
            .reset_index()
        )

        FcstNextNBuckets = FcstNextNBuckets.melt(
            id_vars=[version_col] + forecast_level,
            value_vars=stat_fcst_cols,
            var_name=stat_algo_col,
            value_name=fcst_next_n_buckets_col,
        )

        system_stat_rows = FcstNextNBuckets[stat_algo_col] == SYSTEM_STAT_FORECAST
        FcstNextNBuckets[stat_algo_col] = np.where(
            system_stat_rows,
            "Bestfit Algorithm",
            FcstNextNBuckets[stat_algo_col].map(lambda x: str(x)[10:]),
        )

        # drop the nas
        FcstNextNBuckets = FcstNextNBuckets[FcstNextNBuckets[fcst_next_n_buckets_col].notna()]

    except Exception as e:
        logger.error(f"Error {e} for slice {df_keys}")
        ActualLastNBuckets = pd.DataFrame(columns=actual_lastNbuckets_cols)
        FcstNextNBuckets = pd.DataFrame(columns=forecast_cols)
    return ActualLastNBuckets, FcstNextNBuckets


def get_previous_cycle_value(value: str, seasonal_periods: int, time_master_list: list) -> str:
    # get the index of current value
    index_current_value = time_master_list.index(value)

    # get the previous cycle index
    index_previous_cycle_value = index_current_value - seasonal_periods

    if 0 <= index_previous_cycle_value < len(time_master_list):
        return time_master_list[index_previous_cycle_value]
    else:
        return None


def calculate_level_violation(
    curActualLevel: tuple,
    level: tuple,
    LevelVariationThreshold: float,
    AbsTolerance: float,
) -> float:
    m50p = abs(curActualLevel[0] - level[0]) / curActualLevel[0] if curActualLevel[0] != 0 else 0
    m25p = abs(curActualLevel[1] - level[1]) / curActualLevel[1] if curActualLevel[1] != 0 else 0
    m75p = abs(curActualLevel[2] - level[2]) / curActualLevel[2] if curActualLevel[2] != 0 else 0
    # if curActualLevel[0] > AbsTolerance and level[0] > AbsTolerance:
    m50v = (
        (1 if m50p >= LevelVariationThreshold else 0) if (curActualLevel[0] > AbsTolerance) else 0
    )
    m25v = (
        (1 if m25p >= LevelVariationThreshold else 0) if (curActualLevel[1] > AbsTolerance) else 0
    )
    m75v = (
        (1 if m75p >= LevelVariationThreshold else 0) if (curActualLevel[2] > AbsTolerance) else 0
    )
    levelViolation = 1 if sum([m50v, m25v, m75v]) >= 2 else 0
    return levelViolation


def calulate_range_violation(
    curActualRange: float,
    rangeF: float,
    RangeVariationThreshold: float,
    AbsTolerance: float,
) -> float:
    rangeViolation = (
        (
            1
            if (abs(curActualRange - rangeF) / curActualRange if curActualRange != 0 else 0)
            > RangeVariationThreshold
            else 0
        )
        if curActualRange > AbsTolerance
        else 0
    )
    return rangeViolation


def get_rule(
    the_algo: str,
    the_system_assigned_algo_list: list,
    the_planner_assigned_algo_list: list,
    the_system_assigned_ensemble_algo_list: list,
    plannerBestFitAlgo: str,
    the_assigned_rule: str,
):
    if the_algo in the_system_assigned_algo_list:
        return the_assigned_rule
    elif the_algo == "Ensemble":
        if set(the_system_assigned_ensemble_algo_list).issubset(set(the_system_assigned_algo_list)):
            return the_assigned_rule
        elif set(the_system_assigned_ensemble_algo_list).issubset(
            set(the_system_assigned_algo_list).union(set(the_planner_assigned_algo_list))
        ):
            return "Planner Override"
        else:
            return "Custom"
    elif the_algo in the_planner_assigned_algo_list or the_algo == plannerBestFitAlgo:
        return "Planner Override"
    else:
        return "Custom"


col_mapping = {
    "Straight Line": float,
    "Trend Violation": float,
    "Level Violation": float,
    "Seasonal Violation": float,
    "Range Violation": float,
    "COCC Violation": float,
    "Run Count": float,
    "Is Bestfit": float,
    "No Alerts": float,
    "Missing Bestfit": float,
    "Actual Last N Buckets": float,
    "Fcst Next N Buckets": float,
    "Bestfit Range Violation": float,
    "Bestfit Seasonal Violation": float,
    "Bestfit Level Violation": float,
    "Bestfit Trend Violation": float,
    "Bestfit Straight Line": float,
    "Bestfit COCC Violation": float,
    "Composite Error": float,
    "Validation Error": float,
    "Validation Method": str,
    "Run Time": float,
    "Algorithm Parameters": str,
}


class COCCViolationCalc:
    def __init__(self, curForecast, cocc_threshold, stat_fcst_l1_lc_col):
        self.curForecast = curForecast
        self.cocc_threshold = cocc_threshold
        self.stat_fcst_l1_lc_col = stat_fcst_l1_lc_col

        # filter out non na values for Stat Fcst L1 LC
        self.relevant_data = self.curForecast[self.curForecast[self.stat_fcst_l1_lc_col].notna()]

    def get_violation(
        self,
        forecast_col: str,
    ):
        # if no values are available in Stat Fcst L1 LC, violation cannot be calculated
        if self.relevant_data.empty:
            return 0

        if self.relevant_data[forecast_col].isnull().all():
            return 0

        # collect np arrays for forecast and lc
        lc_forecast = self.relevant_data[self.stat_fcst_l1_lc_col].to_numpy()
        forecast = self.relevant_data[forecast_col].to_numpy()

        # Find NaN values and replace them with zero
        forecast[np.isnan(forecast)] = 0

        # Calculate the absolute error
        absolute_error = np.abs(lc_forecast - forecast)

        # sum of absolute error
        sum_of_absolute_error = np.sum(absolute_error)

        if np.sum(lc_forecast) == 0 and sum_of_absolute_error > 0:
            cocc = 1.0
        else:
            cocc = sum_of_absolute_error / np.sum(lc_forecast)

        return int(cocc > self.cocc_threshold)


def get_stat_algo_member(col_name: str) -> str:
    if col_name == "Stat Fcst ML L1":
        return "ML L1"
    else:
        return col_name.replace("Stat Fcst ", "")


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actual: DataFrame,
    ForecastData: DataFrame,
    SegmentationOutput: DataFrame,
    TimeDimension: DataFrame,
    CurrentTimePeriod: DataFrame,
    ForecastGenTimeBucket: DataFrame,
    ForecastSetupConfiguration: DataFrame,
    AlgoStats: DataFrame,
    TrendVariationThreshold: float,
    LevelVariationThreshold: float,
    RangeVariationThreshold: float,
    SeasonalVariationThreshold: float,
    SeasonalVariationCountThreshold: float,
    ReasonabilityCycles: float,
    MinimumIndicePercentage: float,
    AbsTolerance: float,
    Grains: str,
    df_keys: dict,
    COCCVariationThreshold: float = 0.10,
    SellOutOffset=pd.DataFrame(),
):
    try:
        OutputAllAlgoList = list()
        OutputBestFitList = list()
        ActualLastNBucketsList = list()
        FcstNextNBucketsList = list()
        AlgoStatsList = list()

        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                the_allalgo,
                the_bestfit,
                the_actuallastnbuckets,
                the_fcstnextnbuckets,
                the_algo_stats_for_bestfit_member,
            ) = decorated_func(
                Actual=Actual,
                ForecastData=ForecastData,
                SegmentationOutput=SegmentationOutput,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                ForecastSetupConfiguration=ForecastSetupConfiguration,
                AlgoStats=AlgoStats,
                TrendVariationThreshold=TrendVariationThreshold,
                LevelVariationThreshold=LevelVariationThreshold,
                RangeVariationThreshold=RangeVariationThreshold,
                SeasonalVariationThreshold=SeasonalVariationThreshold,
                SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
                ReasonabilityCycles=ReasonabilityCycles,
                MinimumIndicePercentage=MinimumIndicePercentage,
                AbsTolerance=AbsTolerance,
                Grains=Grains,
                df_keys=df_keys,
                COCCVariationThreshold=COCCVariationThreshold,
                SellOutOffset=SellOutOffset,
            )

            OutputAllAlgoList.append(the_allalgo)
            OutputBestFitList.append(the_bestfit)
            ActualLastNBucketsList.append(the_actuallastnbuckets)
            FcstNextNBucketsList.append(the_fcstnextnbuckets)
            AlgoStatsList.append(the_algo_stats_for_bestfit_member)

        OutputAllAlgo = concat_to_dataframe(OutputAllAlgoList)
        OutputBestFit = concat_to_dataframe(OutputBestFitList)
        ActualLastNBuckets = concat_to_dataframe(ActualLastNBucketsList)
        FcstNextNBuckets = concat_to_dataframe(FcstNextNBucketsList)
        AlgoStats = concat_to_dataframe(AlgoStatsList)

    except Exception as e:
        logger.exception(e)
        OutputAllAlgo = None
        OutputBestFit = None
        ActualLastNBuckets = None
        FcstNextNBuckets = None
        AlgoStats = None

    return (
        OutputAllAlgo,
        OutputBestFit,
        ActualLastNBuckets,
        FcstNextNBuckets,
        AlgoStats,
    )


def processIteration(
    Actual: DataFrame,
    ForecastData: DataFrame,
    SegmentationOutput: DataFrame,
    TimeDimension: DataFrame,
    CurrentTimePeriod: DataFrame,
    ForecastGenTimeBucket: DataFrame,
    ForecastSetupConfiguration: DataFrame,
    AlgoStats: DataFrame,
    TrendVariationThreshold: float,
    LevelVariationThreshold: float,
    RangeVariationThreshold: float,
    SeasonalVariationThreshold: float,
    SeasonalVariationCountThreshold: float,
    ReasonabilityCycles: float,
    MinimumIndicePercentage: float,
    AbsTolerance: float,
    Grains: str,
    df_keys: dict,
    COCCVariationThreshold: float = 0.10,
    SellOutOffset=pd.DataFrame(),
) -> (DataFrame, DataFrame):
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    VERSION: str = "Version.[Version Name]"
    STAT_ALGO: str = "Stat Algorithm.[Stat Algorithm]"
    PARTIAL_WEEK: str = "Time.[Partial Week]"
    WEEK: str = "Time.[Week]"
    WEEK_KEY: str = "Time.[WeekKey]"
    MONTH: str = "Time.[Month]"
    MONTH_KEY: str = "Time.[MonthKey]"
    PLANNING_MONTH: str = "Time.[Planning Month]"
    PLANNING_MONTH_KEY: str = "Time.[PlanningMonthKey]"
    QUARTER: str = "Time.[Quarter]"
    QUARTER_KEY: str = "Time.[QuarterKey]"
    PLANNING_QUARTER: str = "Time.[Planning Quarter]"
    PLANNING_QUARTER_KEY: str = "Time.[PlanningQuarterKey]"
    ACTUAL_MEASURE: str = "Actual Measure"
    ACTUAL_CLEANSED: str = "Actual Cleansed"
    STAT_ACTUAL: str = "Stat Actual"
    PLC_STATUS: str = "PLC Status L1"
    LEN_SERIES: str = "Length of Series L1"
    INTERMITTENT_L1: str = "Intermittent L1"
    SEASONALITY_L1: str = "Seasonality L1"
    PLANNER_STAT_FORECAST: str = "Planner Stat Fcst L1"
    SYSTEM_STAT_FORECAST: str = "Stat Fcst L1"
    SYSTEM_BESTFIT_ALGORITHM: str = "Bestfit Algorithm"
    PLANNER_BESTFIT_ALGORITHM: str = "Planner Bestfit Algorithm"
    ASSIGNED_RULE: str = "Assigned Rule"
    STAT_RULE: str = "Stat Rule.[Stat Rule]"
    PLANNER_ASSIGNED_ALGORITHM_LIST: str = "Planner Assigned Algorithm List"
    SYSTEM_ASSIGNED_ALGORITHM_LIST: str = "System Assigned Algorithm List"
    System_Ensemble_Algorithm_List: str = "System Ensemble Algorithm List"

    TREND: str = "Trend"
    LEVEL: str = "Level"
    HISTORICAL: str = "Historical"
    FORECAST: str = "Forecast"
    SEASONALITY: str = "Seasonality"
    RANGE: str = "Range"
    HISTORY_PERIOD: str = "History Period"
    FORECAST_PERIOD: str = "Forecast Period"
    forcastGenTimeBucketColName: str = "Forecast Generation Time Bucket"
    planningRank: str = "Planning Rank"
    PARTIAL_WEEK_KEY: str = "Time.[PartialWeekKey]"
    STRAIGHT_LINE: str = "Straight Line"
    TREND_VIOLATION: str = "Trend Violation"
    LEVEL_VIOLATION: str = "Level Violation"
    SEASONAL_VIOLATION: str = "Seasonal Violation"
    RANGE_VIOLATION: str = "Range Violation"
    RUN_COUNT: str = "Run Count"
    IS_BESTFIT: str = "Is Bestfit"
    NO_ALERTS: str = "No Alerts"
    BESTFIT_STRAIGHT_LINE: str = "Bestfit Straight Line"
    BESTFIT_TREND_VIOLATION: str = "Bestfit Trend Violation"
    BESTFIT_LEVEL_VIOLATION: str = "Bestfit Level Violation"
    BESTFIT_SEASONAL_VIOLATION: str = "Bestfit Seasonal Violation"
    BESTFIT_RANGE_VIOLATION: str = "Bestfit Range Violation"
    MISSING_BESTFIT: str = "Missing Bestfit"
    FORECAST_NEXT_N_BUCKETS: str = "Fcst Next N Buckets"
    ACTUAL_LAST_N_BUCKETS: str = "Actual Last N Buckets"
    COCC_VIOLATION: str = "COCC Violation"
    BESTFIT_COCC_VIOLATION: str = "Bestfit COCC Violation"
    STAT_FCST_L1_LC: str = "Stat Fcst L1 LC"
    sell_out_offset_col = "Offset Period"

    allAlgoOutputHash: dict = defaultdict(list)
    bestFitHash: dict = defaultdict(list)

    logger.info("Extracting forecast level ...")

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    cols_required_in_output_allAlgoOutput = (
        [VERSION]
        + forecast_level
        + [
            STAT_ALGO,
            STAT_RULE,
            STRAIGHT_LINE,
            TREND_VIOLATION,
            LEVEL_VIOLATION,
            SEASONAL_VIOLATION,
            RANGE_VIOLATION,
            COCC_VIOLATION,
            RUN_COUNT,
            IS_BESTFIT,
            NO_ALERTS,
        ]
    )

    cols_required_in_output_bestFitOutput = (
        [VERSION]
        + forecast_level
        + [
            BESTFIT_STRAIGHT_LINE,
            BESTFIT_TREND_VIOLATION,
            BESTFIT_LEVEL_VIOLATION,
            BESTFIT_SEASONAL_VIOLATION,
            BESTFIT_RANGE_VIOLATION,
            BESTFIT_COCC_VIOLATION,
            MISSING_BESTFIT,
        ]
    )
    cols_required_in_actual_lastNbuckets = [VERSION] + forecast_level + [ACTUAL_LAST_N_BUCKETS]
    cols_required_in_fcst_nextNbuckets = (
        [VERSION] + forecast_level + [STAT_ALGO, STAT_RULE, FORECAST_NEXT_N_BUCKETS]
    )
    cols_required_in_algo_stats_for_bestfit_member = (
        [VERSION]
        + forecast_level
        + [
            STAT_ALGO,
            STAT_RULE,
            o9Constants.ALGORITHM_PARAMETERS,
            o9Constants.RUN_TIME,
            o9Constants.VALIDATION_METHOD,
            o9Constants.VALIDATION_ERROR,
            o9Constants.COMPOSITE_ERROR,
        ]
    )
    ActualLastNBuckets = pd.DataFrame(columns=cols_required_in_actual_lastNbuckets)
    FcstNextNBuckets = pd.DataFrame(columns=cols_required_in_fcst_nextNbuckets)
    allAlgoOutput = DataFrame(columns=cols_required_in_output_allAlgoOutput)
    bestFitOutput = DataFrame(columns=cols_required_in_output_bestFitOutput)
    algoStatsForBestFitMembers = DataFrame(columns=cols_required_in_algo_stats_for_bestfit_member)
    try:
        # DROP NAs
        Actual = Actual[Actual[[ACTUAL_CLEANSED, STAT_ACTUAL]].notna().any(axis=1)]
        stat_fcst_cols = [x for x in ForecastData.columns if "Stat Fcst" in x]
        ForecastData.dropna(subset=stat_fcst_cols, how="all", inplace=True)

        SegmentationOutput = SegmentationOutput[SegmentationOutput[PLC_STATUS].notna()]

        if len(ForecastData) == 0 or len(Actual) == 0 or len(SegmentationOutput) == 0:
            logger.warning(f"ForecastData/Actual/SegmentationOutput is empty for slice {df_keys}")
            return (
                allAlgoOutput,
                bestFitOutput,
                ActualLastNBuckets,
                FcstNextNBuckets,
                algoStatsForBestFitMembers,
            )

        if len(SellOutOffset) == 0:
            logger.warning(
                "Empty SellOut offset input for the forecast iteration , assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        timeDimLookup = {
            "Week": (WEEK, WEEK_KEY),
            "Month": (MONTH, MONTH_KEY),
            "Planning Month": (PLANNING_MONTH, PLANNING_MONTH_KEY),
            "Quarter": (QUARTER, QUARTER_KEY),
            "Planning Quarter": (PLANNING_QUARTER, PLANNING_QUARTER_KEY),
        }
        forecastGenTimeBucketValue: str = ForecastGenTimeBucket[forcastGenTimeBucketColName].values[
            0
        ]
        FORECAST_GENERATION_TIME_GRAIN: str = timeDimLookup[forecastGenTimeBucketValue][0]
        FORECAST_GENERATION_TIME_KEY: str = timeDimLookup[forecastGenTimeBucketValue][1]
        logger.debug(f"Working with {FORECAST_GENERATION_TIME_GRAIN}")
        forecastPeriod: int = ForecastSetupConfiguration[FORECAST_PERIOD].values[0]
        seasonalPeriod: int = get_seasonal_periods(forecastGenTimeBucketValue)

        # add planner and system measures to dictionary
        allForecast = {
            PLANNER_STAT_FORECAST: {},
            SYSTEM_STAT_FORECAST: {},
        }

        # add all the forecast measures present in input to the dictionary
        for the_col in ForecastData:
            if the_col.startswith("Stat Fcst"):
                logger.debug(f"Adding {the_col} to allForecast dictionary ...")
                allForecast[the_col] = {}

        # Step 1: Run Forecast Insights
        # Merge with Time to get Week & Aggregate data on Week.

        # default for monthly and week
        # for week ranks are decided based on planning month
        modulo_value = 12
        if FORECAST_GENERATION_TIME_GRAIN == PLANNING_QUARTER:
            TimeDimension[planningRank] = TimeDimension[PLANNING_QUARTER_KEY].rank(method="dense")
            groupGrain = (
                [VERSION]
                + forecast_level
                + [
                    FORECAST_GENERATION_TIME_GRAIN,
                    FORECAST_GENERATION_TIME_KEY,
                    planningRank,
                ]
            )
            modulo_value = 4
        elif FORECAST_GENERATION_TIME_GRAIN == QUARTER:
            TimeDimension[planningRank] = TimeDimension[QUARTER_KEY].rank(method="dense")
            groupGrain = (
                [VERSION]
                + forecast_level
                + [
                    FORECAST_GENERATION_TIME_GRAIN,
                    FORECAST_GENERATION_TIME_KEY,
                    planningRank,
                ]
            )
            modulo_value = 4
        elif FORECAST_GENERATION_TIME_GRAIN == PLANNING_MONTH:
            TimeDimension[planningRank] = TimeDimension[PLANNING_MONTH_KEY].rank(method="dense")
            groupGrain = (
                [VERSION]
                + forecast_level
                + [
                    FORECAST_GENERATION_TIME_GRAIN,
                    FORECAST_GENERATION_TIME_KEY,
                    planningRank,
                ]
            )
        elif FORECAST_GENERATION_TIME_GRAIN == MONTH:
            TimeDimension[planningRank] = TimeDimension[MONTH_KEY].rank(method="dense")
            groupGrain = (
                [VERSION]
                + forecast_level
                + [
                    FORECAST_GENERATION_TIME_GRAIN,
                    FORECAST_GENERATION_TIME_KEY,
                    planningRank,
                ]
            )
        elif FORECAST_GENERATION_TIME_GRAIN == WEEK:
            TimeDimension[planningRank] = TimeDimension[PLANNING_MONTH_KEY].rank(method="dense")
            groupGrain = (
                [VERSION]
                + forecast_level
                + [
                    WEEK,
                    WEEK_KEY,
                    planningRank,
                ]
            )
        else:
            groupGrain = []
            logger.warning(f"Unsupported Time Data: {FORECAST_GENERATION_TIME_GRAIN}. Exiting!")
            return (
                allAlgoOutput,
                bestFitOutput,
                ActualLastNBuckets,
                FcstNextNBuckets,
                algoStatsForBestFitMembers,
            )

        TimeDimension[planningRank] = TimeDimension[planningRank] % modulo_value
        logger.debug("Merge Actual and Forecast with Time Dimension.")
        mergeGrain = [
            PARTIAL_WEEK,
            PARTIAL_WEEK_KEY,
            FORECAST_GENERATION_TIME_GRAIN,
            FORECAST_GENERATION_TIME_KEY,
            planningRank,
        ]
        Actual = Actual.merge(
            TimeDimension[mergeGrain].drop_duplicates(),
            on=PARTIAL_WEEK,
            how="left",
        )
        # Filter forecast data.
        Actual = Actual[Actual[PARTIAL_WEEK_KEY] < CurrentTimePeriod[PARTIAL_WEEK_KEY].unique()[0]]

        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            FORECAST_GENERATION_TIME_GRAIN,
            FORECAST_GENERATION_TIME_KEY,
        )

        time_attribute_dict = {FORECAST_GENERATION_TIME_GRAIN: FORECAST_GENERATION_TIME_KEY}

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [FORECAST_GENERATION_TIME_GRAIN, FORECAST_GENERATION_TIME_KEY]
        ].drop_duplicates()

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

        logger.info(f"latest_time_name after offset {offset_periods} : {latest_time_name} ...")

        history_period = int(ForecastSetupConfiguration[HISTORY_PERIOD].iloc[0])

        last_n_periods_history = get_n_time_periods(
            latest_time_name,
            -history_period,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        ForecastData = ForecastData.merge(
            TimeDimension[mergeGrain].drop_duplicates(),
            on=PARTIAL_WEEK,
            how="left",
        )

        # filter forecast data for relevant periods in future
        forecast_periods = get_n_time_periods(
            latest_value=latest_time_name,
            periods=int(forecastPeriod + offset_periods),
            time_mapping=TimeDimension[
                [FORECAST_GENERATION_TIME_GRAIN, FORECAST_GENERATION_TIME_KEY]
            ].drop_duplicates(),
            time_attribute={FORECAST_GENERATION_TIME_GRAIN: FORECAST_GENERATION_TIME_KEY},
            include_latest_value=False,
        )

        forecast_periods = forecast_periods[offset_periods:]

        filter_clause = ForecastData[FORECAST_GENERATION_TIME_GRAIN].isin(forecast_periods)
        ForecastData = ForecastData[filter_clause]
        if ForecastData.empty:
            logger.warning(
                "No records left in ForecastData after filtering data for current and future time horizon ..."
            )
            return (
                allAlgoOutput,
                bestFitOutput,
                ActualLastNBuckets,
                FcstNextNBuckets,
                algoStatsForBestFitMembers,
            )

        # Assigns ACTUAL_MEASURE based on the presence of ACTUAL_CLEANSED
        Actual[ACTUAL_MEASURE] = np.where(
            pd.notnull(Actual[ACTUAL_CLEANSED]),
            Actual[ACTUAL_CLEANSED],
            Actual[STAT_ACTUAL],
        )

        Actual = Actual.groupby(by=groupGrain, as_index=False, observed=True).agg(
            {ACTUAL_MEASURE: "sum"}
        )

        Actual.drop(columns=FORECAST_GENERATION_TIME_KEY, inplace=True)

        # fill missing dates
        Actual = fill_missing_dates(
            actual=Actual,
            forecast_level=forecast_level,
            time_mapping=relevant_time_mapping,
            history_measure=ACTUAL_MEASURE,
            relevant_time_name=FORECAST_GENERATION_TIME_GRAIN,
            relevant_time_key=FORECAST_GENERATION_TIME_KEY,
            relevant_time_periods=last_n_periods_history,
            fill_nulls_with_zero=True,
        )

        ForecastData = ForecastData.groupby(by=groupGrain, as_index=False, observed=True)[
            list(allForecast.keys())
        ].sum(min_count=1)
        # Convert time key to datetime
        timeFormat = "%Y-%m-%d %H:%M:%S"
        Actual[FORECAST_GENERATION_TIME_KEY] = to_datetime(
            Actual[FORECAST_GENERATION_TIME_KEY], format=timeFormat
        )
        ForecastData[FORECAST_GENERATION_TIME_KEY] = to_datetime(
            ForecastData[FORECAST_GENERATION_TIME_KEY], format=timeFormat
        )

        def fillBestFitHash(
            _version: str,
            _grain_value_dict: dict,
            _line: int or None,
            _trend: int or None,
            _level: int or None,
            _season: int or None,
            _range: int or None,
            _missing: int or None,
            _cocc: int or None,
        ):
            nonlocal VERSION
            nonlocal bestFitHash

            bestFitHash[VERSION].append(_version)
            for the_grain, the_value in _grain_value_dict.items():
                bestFitHash[the_grain].append(the_value)
            bestFitHash[BESTFIT_STRAIGHT_LINE].append(_line)
            bestFitHash[BESTFIT_TREND_VIOLATION].append(_trend)
            bestFitHash[BESTFIT_LEVEL_VIOLATION].append(_level)
            bestFitHash[BESTFIT_SEASONAL_VIOLATION].append(_season)
            bestFitHash[BESTFIT_RANGE_VIOLATION].append(_range)
            bestFitHash[MISSING_BESTFIT].append(_missing)
            bestFitHash[BESTFIT_COCC_VIOLATION].append(_cocc)

        def fillAllAlgoHash(
            _version: str,
            _grain_value_dict: dict,
            _algo: str,
            _rule: str,
            _line: int,
            _trend: int,
            _level: int,
            _season: int,
            _range: int,
            _runCount: int,
            _isBestFit: int,
            _noAlert: int,
            _cocc: int,
        ):
            nonlocal allAlgoOutputHash
            nonlocal VERSION

            allAlgoOutputHash[VERSION].append(_version)
            for the_grain, the_value in _grain_value_dict.items():
                allAlgoOutputHash[the_grain].append(the_value)
            allAlgoOutputHash[STAT_ALGO].append(_algo)
            allAlgoOutputHash[STAT_RULE].append(_rule)
            allAlgoOutputHash[STRAIGHT_LINE].append(_line)
            allAlgoOutputHash[TREND_VIOLATION].append(_trend)
            allAlgoOutputHash[LEVEL_VIOLATION].append(_level)
            allAlgoOutputHash[SEASONAL_VIOLATION].append(_season)
            allAlgoOutputHash[RANGE_VIOLATION].append(_range)
            allAlgoOutputHash[RUN_COUNT].append(_runCount)
            allAlgoOutputHash[IS_BESTFIT].append(_isBestFit)
            allAlgoOutputHash[NO_ALERTS].append(_noAlert)
            allAlgoOutputHash[COCC_VIOLATION].append(_cocc)

        currentDate = CurrentTimePeriod[FORECAST_GENERATION_TIME_GRAIN][0]
        logger.info(f"currentDate : {currentDate}")
        fcst_gen_bucket = FORECAST_GENERATION_TIME_GRAIN.replace("Time.[", "").replace("]", "")

        cycle_period = get_seasonal_periods(frequency=fcst_gen_bucket)

        # round off reasonability cycles to 2 decimal places - 0.0083 should become 0.01
        ReasonabilityCycles = round(ReasonabilityCycles, 2)

        # multiply with seasonal periods to get reasonability periods
        ReasonabilityPeriods = ReasonabilityCycles * seasonalPeriod

        if forecastPeriod < ReasonabilityPeriods:
            ReasonabilityPeriods = forecastPeriod

        fcst_dates = get_n_time_periods(
            latest_value=currentDate,
            periods=int(ReasonabilityPeriods),
            include_latest_value=True,
            time_mapping=TimeDimension,
            time_attribute={FORECAST_GENERATION_TIME_GRAIN: FORECAST_GENERATION_TIME_KEY},
        )
        logger.info(f"fcst_dates : {fcst_dates}")

        if ReasonabilityPeriods > seasonalPeriod:
            # get n time periods
            actual_dates = get_n_time_periods(
                latest_value=currentDate,
                periods=-int(ReasonabilityPeriods),
                include_latest_value=False,
                time_mapping=TimeDimension,
                time_attribute={FORECAST_GENERATION_TIME_GRAIN: FORECAST_GENERATION_TIME_KEY},
            )
        else:
            # get the sorted list of time dimenion at relevant grain
            relevant_time_mapping = TimeDimension[
                [FORECAST_GENERATION_TIME_GRAIN, FORECAST_GENERATION_TIME_KEY]
            ].drop_duplicates()

            time_grain_list = list(relevant_time_mapping[FORECAST_GENERATION_TIME_GRAIN])

            # get previous cycle same period dates
            actual_dates = [
                get_previous_cycle_value(
                    value=x,
                    seasonal_periods=seasonalPeriod,
                    time_master_list=time_grain_list,
                )
                for x in fcst_dates
            ]

        logger.info(f"actual_dates : {actual_dates}")

        ActualLastNBuckets_list = []
        FcstNextNBuckets_list = []

        def processSegment(_data: dict, forecast_level: list):
            nonlocal VERSION
            nonlocal PLC_STATUS
            nonlocal LEN_SERIES
            nonlocal FORECAST_GENERATION_TIME_KEY
            nonlocal FORECAST_GENERATION_TIME_GRAIN
            nonlocal INTERMITTENT_L1
            nonlocal ACTUAL_MEASURE
            nonlocal seasonalPeriod
            nonlocal allForecast
            nonlocal TrendVariationThreshold
            nonlocal LevelVariationThreshold
            nonlocal SeasonalVariationCountThreshold
            nonlocal AbsTolerance
            nonlocal MinimumIndicePercentage
            nonlocal RangeVariationThreshold
            nonlocal allAlgoOutputHash
            nonlocal bestFitHash
            nonlocal ActualLastNBuckets_list
            nonlocal FcstNextNBuckets_list

            try:
                # create dictionary of grain and corresponding value

                the_grain_and_value_dict = {
                    the_grain: str(_data[the_grain]) for the_grain in forecast_level
                }

                sIndicesL1 = "Seasonal Indices L1"

                logger.debug(f"Processing intersection {the_grain_and_value_dict.values()}")

                curActual: DataFrame = DataFrame(the_grain_and_value_dict, index=[0]).merge(
                    Actual, on=forecast_level, how="inner"
                )

                if len(curActual) == 0:
                    return
                curActual.sort_values(FORECAST_GENERATION_TIME_KEY, inplace=True)

                curForecast: DataFrame = DataFrame(the_grain_and_value_dict, index=[0]).merge(
                    ForecastData, on=forecast_level, how="inner"
                )

                if len(curForecast) == 0:
                    return

                length_of_series = len(curActual)

                # No need of calculations for less than 1 cycle of data
                if length_of_series < seasonalPeriod and forecastPeriod < seasonalPeriod:
                    return

                # ST - Case
                if length_of_series >= seasonalPeriod and forecastPeriod < seasonalPeriod:
                    # filter data for same period last year
                    curActual = curActual[
                        curActual[FORECAST_GENERATION_TIME_GRAIN].isin(actual_dates)
                    ]
                    nHash = {
                        HISTORICAL: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                        FORECAST: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                    }
                elif length_of_series >= (2 * seasonalPeriod) and forecastPeriod >= (
                    2 * seasonalPeriod
                ):
                    nHash = {
                        HISTORICAL: {
                            TREND: 2,
                            LEVEL: 2,
                            SEASONALITY: 2,
                            RANGE: 2,
                        },
                        FORECAST: {
                            TREND: 2,
                            LEVEL: 2,
                            SEASONALITY: 2,
                            RANGE: 2,
                        },
                    }
                elif (
                    length_of_series >= (2 * seasonalPeriod)
                    and forecastPeriod >= seasonalPeriod
                    and forecastPeriod < 2 * seasonalPeriod
                ):
                    nHash = {
                        HISTORICAL: {
                            TREND: 2,
                            LEVEL: 2,
                            SEASONALITY: 2,
                            RANGE: 2,
                        },
                        FORECAST: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                    }
                elif (
                    length_of_series >= seasonalPeriod
                    and length_of_series < 2 * seasonalPeriod
                    and forecastPeriod >= seasonalPeriod
                    and forecastPeriod < 2 * seasonalPeriod
                ):
                    nHash = {
                        HISTORICAL: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                        FORECAST: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                    }

                else:
                    nHash = {
                        HISTORICAL: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                        FORECAST: {
                            TREND: 1,
                            LEVEL: 1,
                            SEASONALITY: 1,
                            RANGE: 1,
                        },
                    }

                cols_in_fcst_nextnbuckets_except_stat_rule = copy.deepcopy(
                    cols_required_in_fcst_nextNbuckets
                )
                cols_in_fcst_nextnbuckets_except_stat_rule.remove(STAT_RULE)
                (
                    ActualLastNBuckets_rows,
                    FcstNextNBuckets_rows,
                ) = reasonability_buckets_check(
                    curActual=curActual.drop(planningRank, axis=1),
                    curForecast=curForecast.drop(planningRank, axis=1),
                    Actual=Actual,
                    time_grain=FORECAST_GENERATION_TIME_GRAIN,
                    actual_lastNbuckets_cols=cols_required_in_actual_lastNbuckets,
                    forecast_cols=cols_in_fcst_nextnbuckets_except_stat_rule,
                    actual_measure=ACTUAL_MEASURE,
                    actual_dates=actual_dates,
                    fcst_dates=fcst_dates,
                    cycle_period=cycle_period,
                    version_col=VERSION,
                    stat_algo_col=STAT_ALGO,
                    fcst_next_n_buckets_col=FORECAST_NEXT_N_BUCKETS,
                    actual_last_n_buckets_col=ACTUAL_LAST_N_BUCKETS,
                    forecast_level=forecast_level,
                    SYSTEM_STAT_FORECAST=SYSTEM_STAT_FORECAST,
                    PLANNER_STAT_FORECAST=PLANNER_STAT_FORECAST,
                    df_keys=df_keys,
                )
                ActualLastNBuckets_list.append(ActualLastNBuckets_rows)
                FcstNextNBuckets_list.append(FcstNextNBuckets_rows)

                curForecast.sort_values(FORECAST_GENERATION_TIME_KEY, inplace=True)

                cocc_calc = COCCViolationCalc(
                    curForecast=curForecast,
                    cocc_threshold=COCCVariationThreshold,
                    stat_fcst_l1_lc_col=STAT_FCST_L1_LC,
                )
                # fill nulls if present
                # Iterate over columns
                for column in curForecast.columns:
                    if curForecast[column].isnull().all():
                        # Skip column if all values are null
                        continue
                    elif (
                        is_numeric_dtype(curForecast[column].dtype)
                        and curForecast[column].isnull().any()
                    ):
                        # Fill null values with zero for numeric columns
                        curForecast[column].fillna(0, inplace=True)

                # add a default value if the key doesn't exist in dictionary
                bestFitAlgo = _data.get(SYSTEM_BESTFIT_ALGORITHM, float("nan"))
                plannerBestFitAlgo = _data.get(PLANNER_BESTFIT_ALGORITHM, None)

                # get assigned rule
                existing_assigned_rule = _data.get(ASSIGNED_RULE, np.nan)
                if isinstance(existing_assigned_rule, float) and existing_assigned_rule == np.nan:
                    logger.warning(
                        f"Cannot process further, no rule assigned for {the_grain_and_value_dict}"
                    )
                    return

                the_planner_assigned_algo_list = _data.get(PLANNER_ASSIGNED_ALGORITHM_LIST, None)

                if the_planner_assigned_algo_list is None or (
                    isinstance(the_planner_assigned_algo_list, float)
                    and isnan(the_planner_assigned_algo_list)
                ):
                    the_planner_assigned_algo_list = list()
                else:
                    the_planner_assigned_algo_list = [
                        x.strip() for x in the_planner_assigned_algo_list.split(",")
                    ]

                the_system_assigned_algo_list = _data.get(SYSTEM_ASSIGNED_ALGORITHM_LIST, "")
                the_system_assigned_algo_list = [
                    x.strip() for x in the_system_assigned_algo_list.split(",")
                ]
                the_system_assigned_ensemble_algo_list = []
                if o9Constants.SYSTEM_ENSEMBLE_ALGORITHM_LIST in _data.keys():
                    the_system_assigned_ensemble_algo_list = _data.get(
                        System_Ensemble_Algorithm_List, ""
                    )
                    the_system_assigned_ensemble_algo_list = [
                        x.strip() for x in the_system_assigned_ensemble_algo_list.split(",")
                    ]

                newLaunchCond = (
                    _data[PLC_STATUS] == "NEW LAUNCH" or _data[LEN_SERIES] < seasonalPeriod
                )
                intermittentCond = _data[INTERMITTENT_L1] == "YES"
                isBestFitNull = isinstance(bestFitAlgo, float) and bestFitAlgo == np.nan

                if newLaunchCond:
                    isMissingAdded = False
                    for forecast in allForecast.keys():
                        if forecast == "Stat Fcst L1 LC":
                            continue

                        # get algo name
                        the_algo = get_stat_algo_member(col_name=forecast)

                        the_assigned_rule = get_rule(
                            the_algo=the_algo,
                            the_system_assigned_algo_list=the_system_assigned_algo_list,
                            the_planner_assigned_algo_list=the_planner_assigned_algo_list,
                            the_system_assigned_ensemble_algo_list=the_system_assigned_ensemble_algo_list,
                            plannerBestFitAlgo=plannerBestFitAlgo,
                            the_assigned_rule=existing_assigned_rule,
                        )

                        if np.all(curForecast[forecast].isna()):
                            if forecast == SYSTEM_STAT_FORECAST or isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                                isMissingAdded = True
                            continue
                        if forecast == PLANNER_STAT_FORECAST or forecast == SYSTEM_STAT_FORECAST:
                            continue
                        isStraightLine = check_straight_line(
                            curForecast.iloc[-seasonalPeriod:][forecast]
                        )
                        cocc_violation = cocc_calc.get_violation(forecast_col=forecast)
                        isBestFit = 0
                        if not isMissingAdded:
                            if isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                            elif bestFitAlgo in forecast:
                                isBestFit = 1
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=isStraightLine,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=0,
                                    _cocc=cocc_violation,
                                )
                                fillAllAlgoHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _algo="Bestfit Algorithm",
                                    _rule=the_assigned_rule,
                                    _line=isStraightLine,
                                    _trend=0,
                                    _level=0,
                                    _season=0,
                                    _range=0,
                                    _runCount=(0 if curForecast[forecast].isna().all() else 1),
                                    _isBestFit=isBestFit,
                                    _noAlert=(
                                        1 if isStraightLine == 0 and cocc_violation == 0 else 0
                                    ),
                                    _cocc=cocc_violation,
                                )

                        fillAllAlgoHash(
                            _version=_data[VERSION],
                            _grain_value_dict=the_grain_and_value_dict,
                            _algo=the_algo,
                            _rule=the_assigned_rule,
                            _line=isStraightLine,
                            _trend=0,
                            _level=0,
                            _season=0,
                            _range=0,
                            _runCount=(0 if curForecast[forecast].isna().all() else 1),
                            _isBestFit=isBestFit,
                            _noAlert=(1 if isStraightLine == 0 and cocc_violation == 0 else 0),
                            _cocc=cocc_violation,
                        )
                    #  If PLC Status == NPI or Length of Series L1 < 1 Complete Cycle, Skip Intersection.
                    return
                elif intermittentCond:
                    curActualSeasonality = None
                    if (
                        _data[LEN_SERIES] >= nHash[HISTORICAL][SEASONALITY] * seasonalPeriod
                        and sum(curActual[ACTUAL_MEASURE]) > 0
                        and _data[SEASONALITY_L1] == "Exists"
                    ):
                        # Calculate Seasonality
                        curActualSeasonality = (
                            curActual.tail(nHash[HISTORICAL][SEASONALITY] * seasonalPeriod)
                            .groupby(by=[planningRank], as_index=False, sort=False)
                            .mean()[1:-1]
                        )
                        sumDenominator = curActualSeasonality[ACTUAL_MEASURE].sum()
                        curActualSeasonality[sIndicesL1] = (
                            curActualSeasonality[ACTUAL_MEASURE] / sumDenominator
                        )
                    seasonalViolation = 0
                    isMissingAdded = False
                    for forecast in allForecast.keys():
                        if forecast == "Stat Fcst L1 LC":
                            continue

                        # get algo name
                        the_algo = get_stat_algo_member(col_name=forecast)

                        the_assigned_rule = get_rule(
                            the_algo=the_algo,
                            the_system_assigned_algo_list=the_system_assigned_algo_list,
                            the_planner_assigned_algo_list=the_planner_assigned_algo_list,
                            the_system_assigned_ensemble_algo_list=the_system_assigned_ensemble_algo_list,
                            plannerBestFitAlgo=plannerBestFitAlgo,
                            the_assigned_rule=existing_assigned_rule,
                        )
                        if np.all(curForecast[forecast].isna()):
                            if forecast == SYSTEM_STAT_FORECAST or isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                                isMissingAdded = True
                            continue
                        if forecast == PLANNER_STAT_FORECAST or forecast == SYSTEM_STAT_FORECAST:
                            continue
                        isStraightLine = check_straight_line(
                            curForecast.iloc[-seasonalPeriod:][forecast]
                        )
                        cocc_violation = cocc_calc.get_violation(forecast_col=forecast)

                        if (
                            len(curForecast) >= nHash[FORECAST][SEASONALITY] * seasonalPeriod
                            and sum(curActual[ACTUAL_MEASURE]) > 0
                            and _data[SEASONALITY_L1] == "Exists"
                        ):
                            seasonality = curForecast.groupby(
                                by=[planningRank], as_index=False, sort=False
                            ).mean()[1:-1]
                            sumDenominator = seasonality[forecast].sum()
                            seasonality[sIndicesL1] = seasonality[forecast] / sumDenominator
                            # Calculating Seasonal Violation Alert
                            if _data[PLC_STATUS] != "DISC" and isStraightLine:
                                seasonalViolation = 1
                            elif curActualSeasonality is not None:
                                seasonality = seasonality.merge(
                                    curActualSeasonality,
                                    on=[planningRank],
                                    how="inner",
                                )
                                seasonality[f"{sIndicesL1}_x"] = np.where(
                                    seasonality[f"{sIndicesL1}_x"] <= 0.05,
                                    0,
                                    seasonality[f"{sIndicesL1}_x"],
                                )
                                seasonality[f"{sIndicesL1}_y"] = np.where(
                                    seasonality[f"{sIndicesL1}_y"] <= 0.05,
                                    0,
                                    seasonality[f"{sIndicesL1}_y"],
                                )
                                seasonality["x"] = (
                                    abs(
                                        seasonality[f"{sIndicesL1}_x"]
                                        - seasonality[f"{sIndicesL1}_y"]
                                    )
                                    / seasonality[f"{sIndicesL1}_y"]
                                )
                                seasonality["y"] = np.vectorize(
                                    lambda _f, _h, _x: (
                                        0
                                        if (
                                            _f < MinimumIndicePercentage
                                            and _h < MinimumIndicePercentage
                                        )
                                        else (1 if _x >= SeasonalVariationThreshold else 0)
                                    )
                                )(
                                    seasonality[f"{sIndicesL1}_x"],
                                    seasonality[f"{sIndicesL1}_y"],
                                    seasonality["x"],
                                )
                                seasonality["y"] = np.where(
                                    (seasonality[ACTUAL_MEASURE] < AbsTolerance)
                                    & (seasonality[forecast] < AbsTolerance),
                                    0,
                                    seasonality["y"],
                                )

                                seasonalViolation = (
                                    1
                                    if (
                                        len(seasonality.loc[seasonality["y"].values == 1])
                                        / len(seasonality)
                                    )
                                    >= SeasonalVariationCountThreshold
                                    else 0
                                )

                        isBestFit = 0
                        if not isMissingAdded:
                            if isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                            elif bestFitAlgo in forecast:
                                isBestFit = 1
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=isStraightLine,
                                    _trend=0,
                                    _level=0,
                                    _season=seasonalViolation,
                                    _range=0,
                                    _missing=0,
                                    _cocc=cocc_violation,
                                )
                                fillAllAlgoHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _algo="Bestfit Algorithm",
                                    _rule=the_assigned_rule,
                                    _line=isStraightLine,
                                    _trend=0,
                                    _level=0,
                                    _season=seasonalViolation,
                                    _range=0,
                                    _runCount=(0 if curForecast[forecast].isna().all() else 1),
                                    _isBestFit=isBestFit,
                                    _noAlert=(
                                        1
                                        if (isStraightLine + seasonalViolation + cocc_violation)
                                        == 0
                                        else 0
                                    ),
                                    _cocc=cocc_violation,
                                )

                        fillAllAlgoHash(
                            _version=_data[VERSION],
                            _grain_value_dict=the_grain_and_value_dict,
                            _algo=the_algo,
                            _rule=the_assigned_rule,
                            _line=isStraightLine,
                            _trend=0,
                            _level=0,
                            _season=seasonalViolation,
                            _range=0,
                            _runCount=(0 if curForecast[forecast].isna().all() else 1),
                            _isBestFit=isBestFit,
                            _noAlert=(
                                1
                                if (isStraightLine + seasonalViolation + cocc_violation) == 0
                                else 0
                            ),
                            _cocc=cocc_violation,
                        )
                else:
                    # Calculate metric for Trend Degree L1, Level L1, Range L1 values
                    curActualTrend = None
                    curActualLevel = None
                    curActualRange = None
                    curActualSeasonality = None
                    counter: str = "Counter"

                    if not curActual[ACTUAL_MEASURE].isna().any():
                        if len(curActual) >= nHash[HISTORICAL][TREND] * seasonalPeriod:
                            tmp = curActual.iloc[-(nHash[HISTORICAL][TREND] * seasonalPeriod) :]
                            tmp[counter] = np.arange(1, len(tmp) + 1)
                            X = tmp[counter].values.reshape(-1, 1)  # Reshape to a 2D array
                            y = tmp[ACTUAL_MEASURE].values

                            # curActualTrend = degrees(atan(slope))
                            curActualTrend = get_trend_rate_of_change(X, y)
                            # Calculate Median, 25th Percentile, and 75th Percentile of the most recent complete cycle
                            recentActual = curActual.iloc[
                                -(nHash[HISTORICAL][LEVEL] * seasonalPeriod) :
                            ]
                            curActualLevel = (
                                recentActual[ACTUAL_MEASURE].median(),
                                recentActual[ACTUAL_MEASURE].quantile(0.25),
                                recentActual[ACTUAL_MEASURE].quantile(0.75),
                            )

                        curActualRange = np.std(
                            detrend(
                                curActual.iloc[-(nHash[HISTORICAL][RANGE] * seasonalPeriod) :][
                                    ACTUAL_MEASURE
                                ].values
                            )
                        )
                    if (
                        _data[LEN_SERIES] >= nHash[HISTORICAL][SEASONALITY] * seasonalPeriod
                        and sum(curActual[ACTUAL_MEASURE]) > 0
                        and _data[SEASONALITY_L1] == "Exists"
                    ):
                        # Calculate Seasonality
                        curActualSeasonality = (
                            curActual.tail(nHash[HISTORICAL][SEASONALITY] * seasonalPeriod)
                            .groupby(by=[planningRank], as_index=False, sort=False)
                            .mean()[1:-1]
                        )
                        sumDenominator = curActualSeasonality[ACTUAL_MEASURE].sum()
                        curActualSeasonality[sIndicesL1] = (
                            curActualSeasonality[ACTUAL_MEASURE] / sumDenominator
                        )

                    recentLevelForecast = curForecast.iloc[
                        0 : (nHash[FORECAST][LEVEL] * seasonalPeriod)
                    ]
                    nTrend = nHash[FORECAST][TREND]
                    recentTrendForecast = curForecast.iloc[0 : (nTrend * seasonalPeriod)]
                    recentTrendForecast[counter] = np.arange(1, len(recentTrendForecast) + 1)

                    trendViolation = 0
                    levelViolation = 0
                    seasonalViolation = 0
                    rangeViolation = 0
                    cocc_violation = 0
                    actualToBeAddedForTrend = curActual.iloc[-seasonalPeriod:]
                    isMissingAdded = False
                    for forecast in allForecast.keys():
                        if forecast == "Stat Fcst L1 LC":
                            continue

                        # get algo name
                        the_algo = get_stat_algo_member(col_name=forecast)

                        # logger.debug(f"forecast : {forecast}, the_algo : {the_algo}")

                        the_assigned_rule = get_rule(
                            the_algo=the_algo,
                            the_system_assigned_algo_list=the_system_assigned_algo_list,
                            the_planner_assigned_algo_list=the_planner_assigned_algo_list,
                            the_system_assigned_ensemble_algo_list=the_system_assigned_ensemble_algo_list,
                            plannerBestFitAlgo=plannerBestFitAlgo,
                            the_assigned_rule=existing_assigned_rule,
                        )

                        if np.all(curForecast[forecast].isna()):
                            if forecast == SYSTEM_STAT_FORECAST or isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                                isMissingAdded = True
                            continue
                        if forecast == PLANNER_STAT_FORECAST or forecast == SYSTEM_STAT_FORECAST:
                            continue
                        if not curForecast[forecast].isna().any():
                            if len(curForecast) >= nTrend * seasonalPeriod:
                                # Fit a Linear Regression model to the data
                                if nTrend == 1:
                                    # Recent data = last cycle actual + first cycle forecast.
                                    actualToBeAddedForTrend[forecast] = actualToBeAddedForTrend[
                                        ACTUAL_MEASURE
                                    ]
                                    # actualToBeAddedForTrend.rename(columns={ACTUAL_MEASURE: forecast}, inplace=True)
                                    combined = actualToBeAddedForTrend[
                                        [
                                            FORECAST_GENERATION_TIME_KEY,
                                            forecast,
                                        ]
                                    ].append(
                                        recentTrendForecast[
                                            [
                                                FORECAST_GENERATION_TIME_KEY,
                                                forecast,
                                            ]
                                        ]
                                    )
                                    combined[counter] = np.arange(1, len(combined) + 1)
                                    X = combined[counter].values.reshape(-1, 1)
                                    y = combined[forecast].values
                                else:
                                    X = recentTrendForecast[counter].values.reshape(-1, 1)
                                    y = recentTrendForecast[forecast].values

                                trend = get_trend_rate_of_change(X, y)

                                # Calculating Trend Violation Alert
                                if curActualTrend is not None:

                                    minAbsDiff = abs(curActualTrend - trend)
                                    trendViolation = (
                                        1 if abs(minAbsDiff) > TrendVariationThreshold else 0
                                    )
                                curForecastLevel = (
                                    recentLevelForecast[forecast].median(),
                                    recentLevelForecast[forecast].quantile(0.25),
                                    recentLevelForecast[forecast].quantile(0.75),
                                )
                                # Calculating Level Violation Alert
                                if curActualLevel is not None:
                                    levelViolation = calculate_level_violation(
                                        curActualLevel=curActualLevel,
                                        level=curForecastLevel,
                                        LevelVariationThreshold=LevelVariationThreshold,
                                        AbsTolerance=AbsTolerance,
                                    )

                            rangeF = np.std(detrend(recentLevelForecast[forecast].values))
                            # Calculating Range Violation Alert
                            if curActualRange is not None:
                                rangeViolation = calulate_range_violation(
                                    curActualRange=curActualRange,
                                    rangeF=rangeF,
                                    RangeVariationThreshold=RangeVariationThreshold,
                                    AbsTolerance=AbsTolerance,
                                )

                        isStraightLine = check_straight_line(
                            curForecast.iloc[-seasonalPeriod:][forecast]
                        )
                        cocc_violation = cocc_calc.get_violation(forecast_col=forecast)
                        if (
                            len(curForecast) >= nHash[FORECAST][SEASONALITY] * seasonalPeriod
                            and sum(curActual[ACTUAL_MEASURE]) > 0
                            and _data[SEASONALITY_L1] == "Exists"
                        ):
                            seasonality = curForecast.groupby(
                                by=[planningRank], as_index=False, sort=False
                            ).mean()[1:-1]
                            sumDenominator = seasonality[forecast].sum()
                            seasonality[sIndicesL1] = seasonality[forecast] / sumDenominator
                            # Calculating Seasonal Violation Alert
                            if _data[PLC_STATUS] != "DISC" and isStraightLine:
                                seasonalViolation = 1
                            elif curActualSeasonality is not None:
                                seasonality = seasonality.merge(
                                    curActualSeasonality,
                                    on=[planningRank],
                                    how="inner",
                                )
                                seasonality[f"{sIndicesL1}_x"] = np.where(
                                    seasonality[f"{sIndicesL1}_x"] <= 0.05,
                                    0,
                                    seasonality[f"{sIndicesL1}_x"],
                                )
                                seasonality[f"{sIndicesL1}_y"] = np.where(
                                    seasonality[f"{sIndicesL1}_y"] <= 0.05,
                                    0,
                                    seasonality[f"{sIndicesL1}_y"],
                                )
                                seasonality["x"] = (
                                    abs(
                                        seasonality[f"{sIndicesL1}_x"]
                                        - seasonality[f"{sIndicesL1}_y"]
                                    )
                                    / seasonality[f"{sIndicesL1}_y"]
                                )
                                seasonality["y"] = np.vectorize(
                                    lambda _f, _h, _x: (
                                        0
                                        if (
                                            _f < MinimumIndicePercentage
                                            and _h < MinimumIndicePercentage
                                        )
                                        else (1 if _x >= SeasonalVariationThreshold else 0)
                                    )
                                )(
                                    seasonality[f"{sIndicesL1}_x"],
                                    seasonality[f"{sIndicesL1}_y"],
                                    seasonality["x"],
                                )
                                seasonality["y"] = np.where(
                                    (seasonality[ACTUAL_MEASURE] < AbsTolerance)
                                    & (seasonality[forecast] < AbsTolerance),
                                    0,
                                    seasonality["y"],
                                )
                                seasonalViolation = (
                                    1
                                    if (
                                        (
                                            len(seasonality.loc[seasonality["y"].values == 1])
                                            / len(seasonality)
                                        )
                                        >= SeasonalVariationCountThreshold
                                    )
                                    else 0
                                )

                        isBestFit = 0
                        if not isMissingAdded:
                            if isBestFitNull:
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=None,
                                    _trend=None,
                                    _level=None,
                                    _season=None,
                                    _range=None,
                                    _missing=1,
                                    _cocc=None,
                                )
                            elif bestFitAlgo in forecast:
                                isBestFit = 1
                                fillBestFitHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _line=isStraightLine,
                                    _trend=trendViolation,
                                    _level=levelViolation,
                                    _season=seasonalViolation,
                                    _range=(rangeViolation if isStraightLine != 1 else 0),
                                    _missing=0,
                                    _cocc=cocc_violation,
                                )
                                fillAllAlgoHash(
                                    _version=_data[VERSION],
                                    _grain_value_dict=the_grain_and_value_dict,
                                    _algo="Bestfit Algorithm",
                                    _rule=the_assigned_rule,
                                    _line=isStraightLine,
                                    _trend=trendViolation,
                                    _level=levelViolation,
                                    _season=seasonalViolation,
                                    _range=(rangeViolation if isStraightLine != 1 else 0),
                                    _runCount=(0 if curForecast[forecast].isna().all() else 1),
                                    _isBestFit=isBestFit,
                                    _noAlert=(
                                        1
                                        if (
                                            (
                                                isStraightLine
                                                + trendViolation
                                                + levelViolation
                                                + seasonalViolation
                                                + rangeViolation
                                                + cocc_violation
                                            )
                                            == 0
                                        )
                                        else 0
                                    ),
                                    _cocc=cocc_violation,
                                )

                        fillAllAlgoHash(
                            _version=_data[VERSION],
                            _grain_value_dict=the_grain_and_value_dict,
                            _algo=the_algo,
                            _rule=the_assigned_rule,
                            _line=isStraightLine,
                            _trend=trendViolation,
                            _level=levelViolation,
                            _season=seasonalViolation,
                            _range=(rangeViolation if isStraightLine != 1 else 0),
                            _runCount=(0 if curForecast[forecast].isna().all() else 1),
                            _isBestFit=isBestFit,
                            _noAlert=(
                                1
                                if (
                                    (
                                        isStraightLine
                                        + trendViolation
                                        + levelViolation
                                        + seasonalViolation
                                        + rangeViolation
                                        + cocc_violation
                                    )
                                    == 0
                                )
                                else 0
                            ),
                            _cocc=cocc_violation,
                        )
            except Exception as e:
                logger.exception(f"{e} for {_data}")

        logger.debug("Start processing segmentation.")
        SegmentationOutput = SegmentationOutput[~(SegmentationOutput[PLC_STATUS].isna())]

        if len(SegmentationOutput) > 0:
            SegmentationOutput[SEASONALITY_L1] = SegmentationOutput[SEASONALITY_L1].fillna("Exists")
            SegmentationOutput = SegmentationOutput.fillna("")
            SegmentationOutput.apply(
                lambda _x: processSegment(_x.to_dict(), forecast_level), axis=1
            )

            allAlgoOutput = DataFrame.from_dict(allAlgoOutputHash)
            bestFitOutput = DataFrame.from_dict(bestFitHash)

        if allAlgoOutput.empty:
            allAlgoOutput = DataFrame(columns=cols_required_in_output_allAlgoOutput)
        if bestFitOutput.empty:
            bestFitOutput = DataFrame(columns=cols_required_in_output_bestFitOutput)
        allAlgoOutput = allAlgoOutput[cols_required_in_output_allAlgoOutput].drop_duplicates()
        bestFitOutput = bestFitOutput[cols_required_in_output_bestFitOutput].drop_duplicates()

        ActualLastNBuckets = concat_to_dataframe(ActualLastNBuckets_list)
        ActualLastNBuckets = ActualLastNBuckets[
            cols_required_in_actual_lastNbuckets
        ].drop_duplicates()

        FcstNextNBuckets = concat_to_dataframe(FcstNextNBuckets_list)

        bestfitalgo_df = SegmentationOutput[
            forecast_level + [SYSTEM_BESTFIT_ALGORITHM, PLANNER_BESTFIT_ALGORITHM]
        ].drop_duplicates()
        bestfitalgo_df[PLANNER_BESTFIT_ALGORITHM].fillna(
            bestfitalgo_df[SYSTEM_BESTFIT_ALGORITHM], inplace=True
        )
        bestfitalgo_df = bestfitalgo_df[bestfitalgo_df[PLANNER_BESTFIT_ALGORITHM].notna()]

        if bestfitalgo_df.empty:
            pass
        else:
            AlgoStats = AlgoStats.merge(bestfitalgo_df, on=forecast_level, how="inner")

            # filter data for bestfit member
            filter_clause = AlgoStats[STAT_ALGO] == AlgoStats[PLANNER_BESTFIT_ALGORITHM]
            algoStatsForBestFitMembers = AlgoStats[filter_clause]

            # Populate 'Bestfit Algorithm'
            algoStatsForBestFitMembers[STAT_ALGO] = "Bestfit Algorithm"

            algoStatsForBestFitMembers = algoStatsForBestFitMembers[
                cols_required_in_algo_stats_for_bestfit_member
            ].drop_duplicates()

        if not allAlgoOutput.empty and not FcstNextNBuckets.empty:
            # join to add stat rule column
            FcstNextNBuckets = FcstNextNBuckets.merge(
                allAlgoOutput[forecast_level + [STAT_ALGO, STAT_RULE]],
                on=forecast_level + [STAT_ALGO],
                how="inner",
            )
            FcstNextNBuckets = FcstNextNBuckets[
                cols_required_in_fcst_nextNbuckets
            ].drop_duplicates()
        else:
            FcstNextNBuckets = DataFrame(columns=cols_required_in_fcst_nextNbuckets)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        allAlgoOutput = DataFrame(columns=cols_required_in_output_allAlgoOutput)
        bestFitOutput = DataFrame(columns=cols_required_in_output_bestFitOutput)
        ActualLastNBuckets = DataFrame(columns=cols_required_in_actual_lastNbuckets)
        FcstNextNBuckets = DataFrame(columns=cols_required_in_fcst_nextNbuckets)
        algoStatsForBestFitMembers = DataFrame(
            columns=cols_required_in_algo_stats_for_bestfit_member
        )
    return (
        allAlgoOutput,
        bestFitOutput,
        ActualLastNBuckets,
        FcstNextNBuckets,
        algoStatsForBestFitMembers,
    )

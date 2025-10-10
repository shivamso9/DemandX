import logging
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import get_last_time_period
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    create_cartesian_product,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from scipy import signal

from helpers.o9_holidays import get_planning_to_stat_region_mapping
from helpers.o9Constants import o9Constants
from helpers.utils import add_dim_suffix, filter_for_iteration

logger = logging.getLogger("o9_logger")


pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def get_history_periods(HistoryPeriodWeeks, FORECAST_GENERATION_TIME_GRAIN):
    if FORECAST_GENERATION_TIME_GRAIN == "Time.[Week]":
        return int(HistoryPeriodWeeks)
    elif FORECAST_GENERATION_TIME_GRAIN in [
        "Time.[Month]",
        "Time.[Planning Month]",
    ]:
        return round(int(HistoryPeriodWeeks) / 4.34524)
    elif FORECAST_GENERATION_TIME_GRAIN in [
        "Time.[Quarter]",
        "Time.[Planning Quarter]",
    ]:
        return round(int(HistoryPeriodWeeks) / 13)
    else:
        raise ValueError(f"Invalid entry {FORECAST_GENERATION_TIME_GRAIN}")


def generate_time_horizon(
    _actuals,
    HistoryPeriod,
    actualGrains,
    VERSION,
    INPUT_TIME_GRAIN,
    HISTORY_MEASURE,
    GRAINS,
    HOLIDAY,
    PROMO,
    _sortByCols,
    actualGrainsForecastLevel,
    timeDimensionFiltered,
    dimensionGrainsForecastGeneration,
    offset_periods,
):
    logger.info("-- Generating full time horizon")
    zeroIndicator = "isZero"
    _actuals = _actuals[actualGrains]
    _version = _actuals[VERSION].values[0]
    _actuals = pd.merge(
        timeDimensionFiltered,
        _actuals,
        on=INPUT_TIME_GRAIN,
        how="left",
    )
    # check if all values in actuals are null
    if _actuals[HISTORY_MEASURE].isnull().all():
        return

    _actuals[VERSION] = _version
    for _g in GRAINS:
        _atrribute = _actuals[_actuals[_g].notna()][_g].values[0]
        _actuals[_g] = _atrribute
    _actuals.fillna({HISTORY_MEASURE: 0, HOLIDAY: 0, PROMO: 0}, inplace=True)
    logger.info("-- Aggregating table to forecast generation level")
    _actualsForecastLevel = _actuals.groupby(dimensionGrainsForecastGeneration, as_index=False).agg(
        {HISTORY_MEASURE: "sum", HOLIDAY: "max", PROMO: "max"}
    )
    _actualsForecastLevel[HOLIDAY] = _actualsForecastLevel[HOLIDAY].clip(upper=1)
    _actualsForecastLevel[PROMO] = _actualsForecastLevel[PROMO].clip(upper=1)
    _actualsForecastLevel.sort_values(_sortByCols, inplace=True)
    _actualsForecastLevel.reset_index(drop=True, inplace=True)
    try:
        _actualsForecastLevel[zeroIndicator] = np.where(
            _actualsForecastLevel[HISTORY_MEASURE] == 0, True, False
        )
        _firstNonZeroIndex = _actualsForecastLevel[~_actualsForecastLevel[zeroIndicator]]
        if _firstNonZeroIndex.empty:
            return _actualsForecastLevel
        _firstNonZeroIndex = _firstNonZeroIndex.index[0]
        _actualsForecastLevel = _actualsForecastLevel.iloc[_firstNonZeroIndex:]
        _actualsForecastLevel = _actualsForecastLevel[
            -(HistoryPeriod + offset_periods) : None if offset_periods == 0 else -offset_periods
        ]
        _actualsForecastLevel = _actualsForecastLevel[actualGrainsForecastLevel]
        return _actualsForecastLevel
    except IndexError as e:
        logger.exception("Exception {}".format(e))
        _actualAttributes = _actuals[GRAINS].drop_duplicates().values[0]
        logger.info(
            f"{HISTORY_MEASURE} is 0 for intersection {tuple(zip(GRAINS,_actualAttributes))}."
        )
        _actualsForecastLevel = _actualsForecastLevel[
            -(HistoryPeriod + offset_periods) : None if offset_periods == 0 else -offset_periods
        ]
        _actualsForecastLevel = _actualsForecastLevel[actualGrainsForecastLevel]
        return _actualsForecastLevel


def preprocess_inputs(
    _actuals,
    _timeDimension,
    _currentTimePeriod,
    _holidayData,
    _promotionData,
    INPUT_TIME_GRAIN,
    INPUT_TIME_KEY,
    HOLIDAY,
    PROMO,
    actualGrains,
    HISTORY_MEASURE,
):
    logger.info("Preprocessing inputs...")
    lastTimeBucket = get_last_time_period(
        _currentTimePeriod,
        _timeDimension,
        INPUT_TIME_GRAIN,
        INPUT_TIME_KEY,
    )
    lastTimeBucket_key = _timeDimension[_timeDimension[INPUT_TIME_GRAIN] == lastTimeBucket][
        INPUT_TIME_KEY
    ].iloc[0]

    _timeDimension[INPUT_TIME_KEY] = pd.to_datetime(_timeDimension[INPUT_TIME_KEY])
    logger.info("-- Filtering Time Master")
    _timeDimension = _timeDimension[_timeDimension[INPUT_TIME_KEY] <= lastTimeBucket_key]
    logger.info("-- Merging Actuals with Drivers")
    if len(_holidayData) == 0 and (len(_promotionData) != 0):
        _actuals = pd.merge(_actuals, _promotionData, how="outer")
        _actuals[HOLIDAY] = 0
    elif len(_promotionData) == 0 and (len(_holidayData) != 0):
        _actuals = pd.merge(_actuals, _holidayData, how="outer")
        _actuals[PROMO] = 0
    elif (len(_holidayData) == 0) and len(_promotionData) == 0:
        logger.warning("Holiday and Promotion flags are empty!")
        _actuals = _actuals.copy()
        _actuals[HOLIDAY] = 0
        _actuals[PROMO] = 0
    else:
        _actuals = pd.merge(_actuals, _holidayData, how="outer")
        _actuals = pd.merge(_actuals, _promotionData, how="outer")
    _actuals = _actuals[actualGrains]
    _actuals[HISTORY_MEASURE].fillna(0, inplace=True)
    _actuals.drop_duplicates(inplace=True)
    return _actuals, _timeDimension


def check_stockout(
    _df,
    StockoutThreshold,
    HISTORY_MEASURE,
    POTENTIAL_STOCKOUT_PERIOD,
    POTENTIAL_STOCKOUT_FLAG,
    pattern,
):
    logger.info("---- Checking for consecutive stockout")
    _tempStockout = "tempStockout"
    _df = _df.reset_index(drop=True)
    _df[_tempStockout] = np.where(_df[HISTORY_MEASURE] == 0, "0", "_")
    _zeroStr = "".join(_df[_tempStockout])
    matches = [(match.start(), match.end() - 1) for match in re.finditer(pattern, _zeroStr)]
    if len(matches) > 0:
        for _tup in matches:
            start_idx = _tup[0]
            end_idx = _tup[1]
            _df.loc[start_idx:end_idx, POTENTIAL_STOCKOUT_PERIOD] = 1
        _df[POTENTIAL_STOCKOUT_PERIOD].fillna(0, inplace=True)
    else:
        _df.loc[:, POTENTIAL_STOCKOUT_PERIOD] = 0
    _stockoutCount = _df[POTENTIAL_STOCKOUT_PERIOD].sum()
    _stockoutCount = _stockoutCount / len(_df)
    _stockoutflag = _stockoutCount <= float(StockoutThreshold) and _stockoutCount > 0
    if _stockoutflag:
        _df[POTENTIAL_STOCKOUT_FLAG] = 1

    else:
        _df[POTENTIAL_STOCKOUT_FLAG] = 0
        _df.loc[:, POTENTIAL_STOCKOUT_PERIOD] = 0

    _df.drop(_tempStockout, axis=1, inplace=True)
    return _df


def offset_bucket(
    df: pd.DataFrame,
    indiceList: list,
    lead: int,
    lag: int,
    holiday_col: str,
    holiday_offset_col: str,
):
    lagIndicator = "Lag"
    leadIndicator = "Lead"
    for hSpikeIndex in indiceList:
        previousHolidayBuckets = df.loc[hSpikeIndex - lead : hSpikeIndex]
        nextHolidayBuckets = df.loc[hSpikeIndex + 1 : hSpikeIndex + lag]
        lastHolidayIndex = previousHolidayBuckets[previousHolidayBuckets[holiday_col] == 1].index
        nextHolidayIndex = nextHolidayBuckets[nextHolidayBuckets[holiday_col] == 1].index

        if len(nextHolidayIndex) != 0:
            lag_offsets = (nextHolidayIndex - hSpikeIndex).astype(str)
            lag_strings = lagIndicator + " " + lag_offsets + ", "
            lag_combined = "".join(lag_strings)
            df.at[hSpikeIndex, holiday_offset_col] += lag_combined

        if len(lastHolidayIndex) != 0:
            lead_offsets = (hSpikeIndex - lastHolidayIndex).astype(str)
            lead_strings = leadIndicator + " " + lead_offsets + ", "
            lead_combined = "".join(lead_strings)
            df.at[hSpikeIndex, holiday_offset_col] += lead_combined
    return df


def driver_correlation(
    _actuals,
    MINIMUM_PROMINENCE,
    StockoutThreshold,
    HISTORY_MEASURE,
    WINDOW_LENGTH,
    SPIKES,
    DIPS,
    SPIKE_OR_DIP,
    HOLIDAY_SPIKES,
    HOLIDAY_DIPS,
    PROMO_SPIKES,
    PROMO_DIPS,
    HOLIDAY,
    PROMO,
    HOLIDAY_SPIKE_OFFSET,
    HOLIDAY_DIP_OFFSET,
    PROMO_SPIKE_OFFSET,
    PROMO_DIP_OFFSET,
    OTHER_SPIKES,
    HOLIDAY_OUT,
    PROMO_OUT,
    POTENTIAL_STOCKOUT_PERIOD,
    POTENTIAL_STOCKOUT_FLAG,
    pattern,
    hsLead,
    hdLead,
    psLead,
    pdLead,
    hsLag,
    hdLag,
    psLag,
    pdLag,
    OTHER_DIPS,
):
    logger.info("-- Finding Spikes and Dips")
    spikeIndicator = "Spike"
    dipIndicator = "Dip"
    holidaySpikeLeadLag = "HolidaySpikeLeadLag"
    holidayDipLeadLag = "HolidayDipLeadLag"
    promoSpikeLeadLag = "PromoSpikeLeadLag"
    promoDipLeadLag = "PromoDipLeadLag"
    _actuals.reset_index(inplace=True, drop=True)
    _actuals = check_stockout(
        _actuals,
        StockoutThreshold,
        HISTORY_MEASURE,
        POTENTIAL_STOCKOUT_PERIOD,
        POTENTIAL_STOCKOUT_FLAG,
        pattern,
    )
    _peakSignal = _actuals[HISTORY_MEASURE]
    _dipSignal = _peakSignal * -1
    _prominence = max(_peakSignal.median(), int(MINIMUM_PROMINENCE))
    try:
        peak_indices = list(
            signal.find_peaks(_peakSignal, prominence=_prominence, wlen=WINDOW_LENGTH)[0]
        )
        dip_indices = list(
            signal.find_peaks(_dipSignal, prominence=_prominence, wlen=WINDOW_LENGTH)[0]
        )

    except Exception as e:
        logger.exception(e)
        peak_indices = []
        dip_indices = []
    _actuals.loc[peak_indices, SPIKES] = 1
    _actuals.loc[dip_indices, DIPS] = 1
    _actuals[SPIKES] = _actuals[SPIKES].fillna(0)
    _actuals[DIPS] = _actuals[DIPS].fillna(0)
    _actuals.loc[peak_indices, SPIKE_OR_DIP] = spikeIndicator
    _actuals.loc[dip_indices, SPIKE_OR_DIP] = dipIndicator

    _actuals[HOLIDAY_SPIKES] = _actuals[SPIKES]
    _actuals[HOLIDAY_DIPS] = _actuals[DIPS]
    _actuals[PROMO_SPIKES] = _actuals[SPIKES]
    _actuals[PROMO_DIPS] = _actuals[DIPS]

    _actuals[holidaySpikeLeadLag] = _actuals[HOLIDAY]
    _actuals[holidayDipLeadLag] = _actuals[HOLIDAY]
    _actuals[promoSpikeLeadLag] = _actuals[PROMO]
    _actuals[promoDipLeadLag] = _actuals[PROMO]

    holidayIndices = _actuals[_actuals[HOLIDAY] == 1].index
    promoIndices = _actuals[_actuals[PROMO] == 1].index

    for hIndex in holidayIndices:
        _actuals.loc[hIndex - hsLag : hIndex + hsLead, holidaySpikeLeadLag] = 1
        _actuals.loc[hIndex - hdLag : hIndex + hdLead, holidayDipLeadLag] = 1
    for pIndex in promoIndices:
        _actuals.loc[pIndex - psLag : pIndex + psLead, promoSpikeLeadLag] = 1
        _actuals.loc[pIndex - pdLag : pIndex + pdLead, promoDipLeadLag] = 1

    _actuals[HOLIDAY_SPIKE_OFFSET] = ""
    _actuals[HOLIDAY_DIP_OFFSET] = ""
    _actuals[PROMO_SPIKE_OFFSET] = ""
    _actuals[PROMO_DIP_OFFSET] = ""

    logger.info("-- Determining Holiday Spikes, Holiday Dips, Promo Spikes and Promo Dips")
    _actuals[HOLIDAY_SPIKES] = _actuals[HOLIDAY_SPIKES] * _actuals[holidaySpikeLeadLag]
    _actuals[HOLIDAY_DIPS] = _actuals[HOLIDAY_DIPS] * _actuals[holidayDipLeadLag]
    _actuals[PROMO_SPIKES] = _actuals[PROMO_SPIKES] * _actuals[promoSpikeLeadLag]
    _actuals[PROMO_DIPS] = _actuals[PROMO_DIPS] * _actuals[promoDipLeadLag]

    holidaySpikeIndices = _actuals[_actuals[HOLIDAY_SPIKES] == 1].index
    holidayDipIndices = _actuals[_actuals[HOLIDAY_DIPS] == 1].index
    promoSpikeIndices = _actuals[_actuals[PROMO_SPIKES] == 1].index
    promoDipIndices = _actuals[_actuals[PROMO_DIPS] == 1].index

    _actuals = offset_bucket(
        df=_actuals,
        indiceList=list(holidaySpikeIndices),
        lead=hsLead,
        lag=hsLag,
        holiday_col=HOLIDAY,
        holiday_offset_col=HOLIDAY_SPIKE_OFFSET,
    )

    _actuals = offset_bucket(
        df=_actuals,
        indiceList=list(holidayDipIndices),
        lag=hdLag,
        lead=hdLead,
        holiday_col=HOLIDAY,
        holiday_offset_col=HOLIDAY_DIP_OFFSET,
    )

    _actuals = offset_bucket(
        df=_actuals,
        indiceList=list(promoSpikeIndices),
        lag=psLag,
        lead=psLead,
        holiday_col=PROMO,
        holiday_offset_col=PROMO_SPIKE_OFFSET,
    )
    _actuals = offset_bucket(
        df=_actuals,
        indiceList=list(promoDipIndices),
        lag=pdLag,
        lead=pdLead,
        holiday_col=PROMO,
        holiday_offset_col=PROMO_DIP_OFFSET,
    )

    _actuals[HOLIDAY_SPIKE_OFFSET] = _actuals[HOLIDAY_SPIKE_OFFSET].str.rstrip(", ")
    _actuals[HOLIDAY_DIP_OFFSET] = _actuals[HOLIDAY_DIP_OFFSET].str.rstrip(", ")
    _actuals[PROMO_SPIKE_OFFSET] = _actuals[PROMO_SPIKE_OFFSET].str.rstrip(", ")
    _actuals[PROMO_DIP_OFFSET] = _actuals[PROMO_DIP_OFFSET].str.rstrip(", ")
    logger.info("-- Determining other spikes and dips")
    _actuals.loc[
        (_actuals[SPIKES] == 1) & (_actuals[HOLIDAY_SPIKES] != 1) & (_actuals[PROMO_SPIKES] != 1),
        OTHER_SPIKES,
    ] = 1
    _actuals.loc[_actuals[OTHER_SPIKES] == 1, SPIKE_OR_DIP] = None
    _actuals.loc[
        (_actuals[DIPS] == 1) & (_actuals[HOLIDAY_DIPS] != 1) & (_actuals[PROMO_DIPS] != 1),
        OTHER_DIPS,
    ] = 1
    _actuals.loc[_actuals[OTHER_DIPS] == 1, SPIKE_OR_DIP] = None
    _actuals.rename(columns={HOLIDAY: HOLIDAY_OUT, PROMO: PROMO_OUT}, inplace=True)

    return _actuals


def add_stat_region_to_holiday_df(
    RegionLevel: pd.DataFrame,
    RegionMasterData: pd.DataFrame,
    HolidayData: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Adding stat region to holiday data ...")

    pl_to_stat_region_mapping = get_planning_to_stat_region_mapping(
        RegionLevel=RegionLevel, RegionMasterData=RegionMasterData
    )

    # join with holiday data
    HolidayData = HolidayData.merge(
        pl_to_stat_region_mapping, on=o9Constants.PLANNING_REGION, how="inner"
    )
    logger.info("Added stat region to holiday data ...")
    return HolidayData


def process_actuals(
    _actualsMerged,
    HistoryPeriod,
    MINIMUM_PROMINENCE,
    StockoutThreshold,
    multiprocessing_num_cores,
    HISTORY_MEASURE,
    WINDOW_LENGTH,
    SPIKES,
    DIPS,
    SPIKE_OR_DIP,
    HOLIDAY_SPIKES,
    HOLIDAY_DIPS,
    PROMO_SPIKES,
    PROMO_DIPS,
    HOLIDAY,
    PROMO,
    HOLIDAY_SPIKE_OFFSET,
    HOLIDAY_DIP_OFFSET,
    PROMO_SPIKE_OFFSET,
    PROMO_DIP_OFFSET,
    OTHER_SPIKES,
    HOLIDAY_OUT,
    PROMO_OUT,
    POTENTIAL_STOCKOUT_PERIOD,
    POTENTIAL_STOCKOUT_FLAG,
    actualGrains,
    VERSION,
    INPUT_TIME_GRAIN,
    GRAINS,
    _sortByCols,
    actualGrainsForecastLevel,
    TOTAL_SPIKES_AND_DIPS,
    HOLIDAY_SPIKE_RATIO,
    PROMO_SPIKE_RATIO,
    HOLIDAY_DIP_RATIO,
    PROMO_DIP_RATIO,
    OTHER_SPIKE_RATIO,
    OTHER_DIP_RATIO,
    TOTAL_SPIKES,
    TOTAL_DIPS,
    ANALYSIS_OUTPUT_COLS,
    AGGREGATED_OUTPUT_COLS,
    pattern,
    hsLead,
    hdLead,
    psLead,
    pdLead,
    hsLag,
    hdLag,
    psLag,
    pdLag,
    OTHER_DIPS,
    offset_periods,
):
    logger.info("Processing Actuals...")
    # _actualsGrouped = _actualsMerged.groupby(GRAINS, as_index=False)
    # _actualsFullHorizon = _actualsGrouped.apply(
    #     lambda x: generate_time_horizon(x, HistoryPeriod)
    # )
    all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
        delayed(generate_time_horizon)(
            group,
            HistoryPeriod,
            actualGrains,
            VERSION,
            INPUT_TIME_GRAIN,
            HISTORY_MEASURE,
            GRAINS,
            HOLIDAY,
            PROMO,
            _sortByCols,
            actualGrainsForecastLevel,
            timeDimensionFiltered,
            dimensionGrainsForecastGeneration,
            offset_periods,
        )
        for name, group in _actualsMerged.groupby(GRAINS)
    )
    _actualsFullHorizon = concat_to_dataframe(all_results)
    # _actualsGroupedForecastGrain = _actualsFullHorizon.groupby(
    #     GRAINS, as_index=False
    # )
    # logger.info("-- Generating Flags")
    # _actualsDriverAnalysis = _actualsGroupedForecastGrain.apply(
    #     lambda x: driver_correlation(x, MINIMUM_PROMINENCE, StockoutThreshold)
    # )
    full_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
        delayed(driver_correlation)(
            group,
            MINIMUM_PROMINENCE,
            StockoutThreshold,
            HISTORY_MEASURE,
            WINDOW_LENGTH,
            SPIKES,
            DIPS,
            SPIKE_OR_DIP,
            HOLIDAY_SPIKES,
            HOLIDAY_DIPS,
            PROMO_SPIKES,
            PROMO_DIPS,
            HOLIDAY,
            PROMO,
            HOLIDAY_SPIKE_OFFSET,
            HOLIDAY_DIP_OFFSET,
            PROMO_SPIKE_OFFSET,
            PROMO_DIP_OFFSET,
            OTHER_SPIKES,
            HOLIDAY_OUT,
            PROMO_OUT,
            POTENTIAL_STOCKOUT_PERIOD,
            POTENTIAL_STOCKOUT_FLAG,
            pattern,
            hsLead,
            hdLead,
            psLead,
            pdLead,
            hsLag,
            hdLag,
            psLag,
            pdLag,
            OTHER_DIPS,
        )
        for name, group in _actualsFullHorizon.groupby(GRAINS)
    )
    _actualsDriverAnalysis = concat_to_dataframe(full_results)
    logger.info("-- Generating Values")
    _aggregatedDriverAnalysis = _actualsDriverAnalysis.groupby(
        [VERSION] + GRAINS, as_index=False
    ).aggregate(
        {
            HOLIDAY_SPIKES: "sum",
            PROMO_SPIKES: "sum",
            HOLIDAY_DIPS: "sum",
            PROMO_DIPS: "sum",
            OTHER_SPIKES: "sum",
            OTHER_DIPS: "sum",
            SPIKES: "sum",
            DIPS: "sum",
            POTENTIAL_STOCKOUT_FLAG: "first",
        }
    )
    _aggregatedDriverAnalysis[HOLIDAY_SPIKES] = _aggregatedDriverAnalysis[
        HOLIDAY_SPIKES
    ] / _aggregatedDriverAnalysis[SPIKES].clip(lower=1)
    _aggregatedDriverAnalysis[PROMO_SPIKES] = _aggregatedDriverAnalysis[
        PROMO_SPIKES
    ] / _aggregatedDriverAnalysis[SPIKES].clip(lower=1)
    _aggregatedDriverAnalysis[HOLIDAY_DIPS] = _aggregatedDriverAnalysis[
        HOLIDAY_DIPS
    ] / _aggregatedDriverAnalysis[DIPS].clip(lower=1)
    _aggregatedDriverAnalysis[PROMO_DIPS] = _aggregatedDriverAnalysis[
        PROMO_DIPS
    ] / _aggregatedDriverAnalysis[DIPS].clip(lower=1)
    _aggregatedDriverAnalysis[OTHER_SPIKES_AND_DIPS] = (
        _aggregatedDriverAnalysis[OTHER_SPIKES] + _aggregatedDriverAnalysis[OTHER_DIPS]
    )
    _aggregatedDriverAnalysis[OTHER_SPIKES] = _aggregatedDriverAnalysis[
        OTHER_SPIKES
    ] / _aggregatedDriverAnalysis[SPIKES].clip(lower=1)
    _aggregatedDriverAnalysis[OTHER_DIPS] = _aggregatedDriverAnalysis[
        OTHER_DIPS
    ] / _aggregatedDriverAnalysis[DIPS].clip(lower=1)
    _aggregatedDriverAnalysis[TOTAL_SPIKES_AND_DIPS] = (
        _aggregatedDriverAnalysis[SPIKES] + _aggregatedDriverAnalysis[DIPS]
    )
    _aggregatedDriverAnalysis.rename(
        columns={
            HOLIDAY_SPIKES: HOLIDAY_SPIKE_RATIO,
            PROMO_SPIKES: PROMO_SPIKE_RATIO,
            HOLIDAY_DIPS: HOLIDAY_DIP_RATIO,
            PROMO_DIPS: PROMO_DIP_RATIO,
            OTHER_SPIKES: OTHER_SPIKE_RATIO,
            OTHER_DIPS: OTHER_DIP_RATIO,
            SPIKES: TOTAL_SPIKES,
            DIPS: TOTAL_DIPS,
        },
        inplace=True,
    )
    _actualsDriverAnalysis = _actualsDriverAnalysis[ANALYSIS_OUTPUT_COLS]
    _aggregatedDriverAnalysis = _aggregatedDriverAnalysis[AGGREGATED_OUTPUT_COLS]
    return (_actualsDriverAnalysis, _aggregatedDriverAnalysis)


col_mapping = {
    "Holiday Spikes": float,
    "Promo Spikes": float,
    "Promo Dips": float,
    "Holiday Dips": float,
    "Other Spikes": float,
    "Other Dips": float,
    "Potential Stockout Flag": float,
    "Other Spikes and Dips": float,
    "Total Spikes": float,
    "Total Dips": float,
    "Total Spikes and Dips": float,
    "Holiday Spikes Flag": float,
    "Holiday Dips Flag": float,
    "Promo Spikes Flag": float,
    "Promo Dips Flag": float,
    "Other Spikes Flag": float,
    "Other Dips Flag": float,
    "Holiday Spikes Offset": str,
    "Holiday Dips Offset": str,
    "Promo Spikes Offset": str,
    "Promo Dips Offset": str,
    "Potential Stockout Period": float,
    "Holiday Flag": float,
    "Promo Flag": float,
    "Spike or Dip": str,
    "Spikes Flag": float,
    "Dips Flag": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actuals,
    TimeDimension,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    HolidayData,
    PromotionData,
    FlagsOutCols,
    OutputColumnNames,
    HolidaySpikeLag,
    HolidaySpikeLead,
    HolidayDipLag,
    HolidayDipLead,
    PromoSpikeLag,
    PromoSpikeLead,
    PromoDipLag,
    PromoDipLead,
    StockoutPeriod,
    Grains,
    weekWindowLength,
    monthWindowLength,
    quarterWindowLength,
    HistoryPeriodWeeks,
    MINIMUM_PROMINENCE,
    StockoutThreshold,
    df_keys,
    ForecastLevelData=None,
    RegionMasterData=None,
    ItemMasterData=None,
    AccountMasterData=None,
    ChannelMasterData=None,
    PnLMasterData=None,
    DemandDomainMasterData=None,
    LocationMasterData=None,
    multiprocessing_num_cores=4,
    SellOutOffset=pd.DataFrame(),
):
    try:
        FlagsList = list()
        ValuesList = list()

        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_flags, the_values = decorated_func(
                Actuals=Actuals,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                HolidayData=HolidayData,
                PromotionData=PromotionData,
                FlagsOutCols=FlagsOutCols,
                OutputColumnNames=OutputColumnNames,
                HolidaySpikeLag=HolidaySpikeLag,
                HolidaySpikeLead=HolidaySpikeLead,
                HolidayDipLag=HolidayDipLag,
                HolidayDipLead=HolidayDipLead,
                PromoSpikeLag=PromoSpikeLag,
                PromoSpikeLead=PromoSpikeLead,
                PromoDipLag=PromoDipLag,
                PromoDipLead=PromoDipLead,
                StockoutPeriod=StockoutPeriod,
                Grains=Grains,
                weekWindowLength=weekWindowLength,
                monthWindowLength=monthWindowLength,
                quarterWindowLength=quarterWindowLength,
                HistoryPeriodWeeks=HistoryPeriodWeeks,
                MINIMUM_PROMINENCE=MINIMUM_PROMINENCE,
                StockoutThreshold=StockoutThreshold,
                df_keys=df_keys,
                multiprocessing_num_cores=multiprocessing_num_cores,
                ForecastLevelData=ForecastLevelData,
                RegionMasterData=RegionMasterData,
                ItemMasterData=ItemMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
            )

            FlagsList.append(the_flags)
            ValuesList.append(the_values)

        Flags = concat_to_dataframe(FlagsList)
        Values = concat_to_dataframe(ValuesList)

    except Exception as e:
        logger.exception(e)
        Flags, Values = None, None
    return Flags, Values


def processIteration(
    Actuals,
    TimeDimension,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    HolidayData,
    PromotionData,
    FlagsOutCols,
    OutputColumnNames,
    HolidaySpikeLag,
    HolidaySpikeLead,
    HolidayDipLag,
    HolidayDipLead,
    PromoSpikeLag,
    PromoSpikeLead,
    PromoDipLag,
    PromoDipLead,
    StockoutPeriod,
    Grains,
    weekWindowLength,
    monthWindowLength,
    quarterWindowLength,
    HistoryPeriodWeeks,
    MINIMUM_PROMINENCE,
    StockoutThreshold,
    df_keys,
    ForecastLevelData,
    RegionMasterData,
    ItemMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    multiprocessing_num_cores=4,
    SellOutOffset=pd.DataFrame(),
):
    plugin_name = "DP054InputChecks"
    logger.warning("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # Input dimensions
    global GRAINS
    global VERSION
    global INPUT_TIME_GRAIN
    global INPUT_TIME_KEY

    # Input measures
    global HISTORY_MEASURE
    global HOLIDAY
    global PROMO

    # Constants
    global WINDOW_LENGTH

    # Output measures
    global SPIKES
    global DIPS
    global SPIKE_OR_DIP
    global HOLIDAY_SPIKES
    global HOLIDAY_DIPS
    global PROMO_SPIKES
    global PROMO_DIPS
    global HOLIDAY_SPIKE_OFFSET
    global HOLIDAY_DIP_OFFSET
    global PROMO_SPIKE_OFFSET
    global PROMO_DIP_OFFSET
    global OTHER_SPIKES
    global OTHER_DIPS
    global HOLIDAY_OUT
    global PROMO_OUT
    global OTHER_SPIKES_AND_DIPS
    global TOTAL_SPIKES_AND_DIPS
    global POTENTIAL_STOCKOUT_PERIOD
    global POTENTIAL_STOCKOUT_FLAG
    global HOLIDAY_SPIKE_RATIO
    global HOLIDAY_DIP_RATIO
    global PROMO_SPIKE_RATIO
    global PROMO_DIP_RATIO
    global OTHER_SPIKE_RATIO
    global OTHER_DIP_RATIO
    global TOTAL_SPIKES
    global TOTAL_DIPS

    # Output columns
    global ANALYSIS_OUTPUT_COLS
    global AGGREGATED_OUTPUT_COLS

    # Intermediate columns
    global actualGrains
    global actualGrainsForecastLevel
    global dimensionGrainsForecastGeneration
    global _sortByCols

    # Intermediate constants
    global hsLag
    global hsLead
    global hdLag
    global hdLead
    global psLag
    global psLead
    global pdLag
    global pdLead
    global holidaySpikeLeadLag
    global holidayDipLeadLag
    global promoSpikeLeadLag
    global promoDipLeadLag
    global pattern
    global timeDimensionFiltered
    global spikeIndicator
    global dipIndicator
    global leadIndicator
    global lagIndicator
    global zeroIndicator

    PROMO = "Promo Days"
    HOLIDAY = "Is Holiday"
    VERSION = "Version.[Version Name]"
    HISTORY_MEASURE = "Stat Actual"
    INPUT_TIME_GRAIN = "Time.[Partial Week]"
    INPUT_TIME_KEY = "Time.[PartialWeekKey]"
    holidaySpikeLeadLag = "HolidaySpikeLeadLag"
    holidayDipLeadLag = "HolidayDipLeadLag"
    promoSpikeLeadLag = "PromoSpikeLeadLag"
    promoDipLeadLag = "PromoDipLeadLag"
    spikeIndicator = "Spike"
    dipIndicator = "Dip"
    leadIndicator = "Lead"
    lagIndicator = "Lag"
    zeroIndicator = "isZero"
    VERSION = "Version.[Version Name]"
    forcastGenTimeBucketColName = "Forecast Generation Time Bucket"

    # PromotionData has version, planning grains(location not present) and Promo Days measure
    promo_planning_grains = [col for col in PromotionData.columns if "[Planning" in col]
    sell_out_offset_col = "Offset Period"

    cols_required_in_flags_output = list(FlagsOutCols.columns)
    cols_required_in_values_output = list(OutputColumnNames.columns)

    Flags = pd.DataFrame(columns=cols_required_in_flags_output)
    Values = pd.DataFrame(columns=cols_required_in_values_output)

    try:
        GRAINS = Grains.split(",")
        GRAINS = [x.strip() for x in GRAINS]
        GRAINS = [str(x) for x in GRAINS if x != "NA" and x != ""]
        promo_stat_grains = [col for col in GRAINS if "Location" not in col]
        pattern = "0{" + StockoutPeriod + ",}"

        hsLag = int(HolidaySpikeLag)
        hsLead = int(HolidaySpikeLead)
        hdLag = int(HolidayDipLag)
        hdLead = int(HolidayDipLead)
        psLag = int(PromoSpikeLag)
        psLead = int(PromoSpikeLead)
        pdLag = int(PromoDipLag)
        pdLead = int(PromoDipLead)

        timeDimLookup = {
            "Week": ("Time.[Week]", "Time.[WeekKey]"),
            "Partial Week": ("Time.[Partial Week]", "Time.[PartialWeekKey]"),
            "Month": ("Time.[Month]", "Time.[MonthKey]"),
            "Planning Month": (
                "Time.[Planning Month]",
                "Time.[PlanningMonthKey]",
            ),
            "Quarter": ("Time.[Quarter]", "Time.[QuarterKey]"),
            "Planning Quarter": (
                "Time.[Planning Quarter]",
                "Time.[PlanningQuarterKey]",
            ),
        }
        FORECAST_GENERATION_TIME_GRAIN = timeDimLookup[
            ForecastGenTimeBucket[forcastGenTimeBucketColName].values[0]
        ][0]
        FORECAST_GENERATION_TIME_KEY = timeDimLookup[
            ForecastGenTimeBucket[forcastGenTimeBucketColName].values[0]
        ][1]

        # Add sell out offset handling
        if SellOutOffset.empty:
            logger.warning("SellOutOffset table is empty")
            offset_periods = 0
        else:
            try:
                offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing SellOutOffset: {e}")
                offset_periods = 0

        windowLengthLookup = {
            "Time.[Week]": int(weekWindowLength),
            "Time.[Month]": int(monthWindowLength),
            "Time.[Planning Month]": int(monthWindowLength),
            "Time.[Quarter]": int(quarterWindowLength),
            "Time.[Planning Quarter]": int(quarterWindowLength),
        }

        WINDOW_LENGTH = windowLengthLookup[FORECAST_GENERATION_TIME_GRAIN]
        FLAG_DIMENSIONS = [VERSION] + GRAINS + [INPUT_TIME_GRAIN]
        VALUES_DIMENSIONS = [VERSION] + GRAINS
        flagsOutMeasureList = list(FlagsOutCols.columns)
        flagsOutMeasureList = [_col for _col in flagsOutMeasureList if _col not in FLAG_DIMENSIONS]
        valuesOutMeasureList = list(OutputColumnNames.columns)
        valuesOutMeasureList = [
            _col for _col in valuesOutMeasureList if _col not in VALUES_DIMENSIONS
        ]
        [
            HOLIDAY_SPIKES,
            HOLIDAY_DIPS,
            PROMO_SPIKES,
            PROMO_DIPS,
            OTHER_SPIKES,
            OTHER_DIPS,
            HOLIDAY_SPIKE_OFFSET,
            HOLIDAY_DIP_OFFSET,
            PROMO_SPIKE_OFFSET,
            PROMO_DIP_OFFSET,
            POTENTIAL_STOCKOUT_PERIOD,
            HOLIDAY_OUT,
            PROMO_OUT,
            SPIKE_OR_DIP,
            SPIKES,
            DIPS,
        ] = flagsOutMeasureList

        [
            HOLIDAY_SPIKE_RATIO,
            PROMO_SPIKE_RATIO,
            HOLIDAY_DIP_RATIO,
            PROMO_DIP_RATIO,
            OTHER_SPIKE_RATIO,
            OTHER_DIP_RATIO,
            POTENTIAL_STOCKOUT_FLAG,
            OTHER_SPIKES_AND_DIPS,
            TOTAL_SPIKES,
            TOTAL_DIPS,
            TOTAL_SPIKES_AND_DIPS,
        ] = valuesOutMeasureList

        ANALYSIS_OUTPUT_COLS = (
            [VERSION] + GRAINS + [FORECAST_GENERATION_TIME_GRAIN] + flagsOutMeasureList
        )
        AGGREGATED_OUTPUT_COLS = [VERSION] + GRAINS + valuesOutMeasureList

        actualGrains = list(Actuals.columns) + [HOLIDAY, PROMO]
        actualGrainsForecastLevel = (
            [VERSION]
            + GRAINS
            + [FORECAST_GENERATION_TIME_GRAIN]
            + [HISTORY_MEASURE]
            + [HOLIDAY, PROMO]
        )
        dimensionGrainsForecastGeneration = (
            [VERSION] + GRAINS + [FORECAST_GENERATION_TIME_GRAIN, FORECAST_GENERATION_TIME_KEY]
        )

        _sortByCols = [VERSION] + GRAINS + [FORECAST_GENERATION_TIME_KEY]

        inputTableCheck = {
            "Actuals": len(Actuals),
            "TimeDimension": len(TimeDimension),
            "CurrentTimePeriod": len(CurrentTimePeriod),
            "ForecastGenTimeBucket": len(ForecastGenTimeBucket),
        }

        inputTableCheckFlag = 0
        for k, v in inputTableCheck.items():
            if v == 0:
                inputTableCheckFlag = 1
                logger.warning(f"'{k}' table is empty!")

        if inputTableCheckFlag:
            logger.exception("One or more input tables are empty. Exiting plugin...")
            return (Flags, Values)

        logger.info(
            f"HOLIDAY LAGS/LEADS:\nHoliday Spike Lag: {hsLag}\nHoliday Spike Lead: {hsLead}\nHoliday Dip Lag: {hdLag}\nHoliday Dip Lead: {hdLead}"
        )
        logger.info(
            f"PROMO LAGS/LEADS:\nPromo Spike Lag: {psLag}\nPromo Spike Lead: {psLead}\nPromo Dip Lag: {pdLag}\nPromo Dip Lead: {pdLead}"
        )

        # pre ST-MT releases
        if o9Constants.STAT_REGION in HolidayData.columns:
            pass
        else:
            RegionLevel = ForecastLevelData[[o9Constants.VERSION_NAME, o9Constants.REGION_LEVEL]]
            HolidayData = add_stat_region_to_holiday_df(
                RegionLevel=RegionLevel,
                RegionMasterData=RegionMasterData,
                HolidayData=HolidayData,
            )

        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Channel"] = ChannelMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData
        master_data_dict["Region"] = RegionMasterData
        master_data_dict["Account"] = AccountMasterData
        master_data_dict["PnL"] = PnLMasterData
        if o9Constants.PLANNING_REGION in PromotionData:
            level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
            for the_col in level_cols:

                # extract 'Item' from 'Item Level'
                the_dim = the_col.split(" Level")[0]
                logger.debug(f"the_dim : {the_dim}")

                the_child_col = the_dim + ".[Planning " + the_dim + "]"
                logger.debug(f"the_planning_col : {the_child_col}")

                # Item.[Stat Item]
                the_stat_col = the_dim + ".[Stat " + the_dim + "]"
                logger.debug(f"the_stat_col : {the_stat_col}")

                the_dim_data = master_data_dict[the_dim]

                # Eg. the_level = Item.[L2]
                the_level = (
                    the_dim
                    + ".["
                    + add_dim_suffix(input=ForecastLevelData[the_col].iloc[0], dim=the_dim)
                    + "]"
                )
                logger.debug(f"the_level : {the_level}")

                # copy values from L2 to Stat Item
                the_dim_data[the_stat_col] = the_dim_data[the_level]

                # select only relevant columns
                the_dim_data = the_dim_data[[the_child_col, the_stat_col]].drop_duplicates()
                # join with Actual
                PromotionData = PromotionData.merge(the_dim_data, on=the_child_col, how="inner")

                logger.debug("-------------------------")
            cols_req_in_Promo = (
                [o9Constants.VERSION_NAME]
                + promo_planning_grains
                + promo_stat_grains
                + [o9Constants.PARTIAL_WEEK, "Promo Days"]
            )

            PromotionData = PromotionData[cols_req_in_Promo]
        elif o9Constants.STAT_REGION in PromotionData:
            # pre ST-MT releases
            pass
        else:
            logger.warning(
                "Missing relevant grains in PromotionData, please check the input query ..."
            )
            return Flags, Values

        holiday_stat_intersections = Actuals[
            [
                grain
                for grain in actualGrains
                if (("[Stat" in grain) and (grain not in HolidayData.columns))
            ]
        ].drop_duplicates()
        promo_stat_intersections = Actuals[
            [
                grain
                for grain in actualGrains
                if (("[Stat" in grain) and (grain not in PromotionData.columns))
            ]
        ].drop_duplicates()
        if len(HolidayData) > 0:
            HolidayData = HolidayData.merge(
                Actuals[[o9Constants.STAT_REGION]].drop_duplicates(),
                on=o9Constants.STAT_REGION,
                how="inner",
            )
            HolidayData = create_cartesian_product(holiday_stat_intersections, HolidayData)
        if len(PromotionData) > 0:
            PromotionData = PromotionData.merge(
                Actuals[promo_stat_grains].drop_duplicates(), on=promo_stat_grains, how="inner"
            )
            PromotionData = create_cartesian_product(promo_stat_intersections, PromotionData)

        (actualsMerged, timeDimensionFiltered) = preprocess_inputs(
            Actuals,
            TimeDimension,
            CurrentTimePeriod,
            HolidayData,
            PromotionData,
            INPUT_TIME_GRAIN,
            INPUT_TIME_KEY,
            HOLIDAY,
            PROMO,
            actualGrains,
            HISTORY_MEASURE,
        )
        HistoryPeriod = get_history_periods(HistoryPeriodWeeks, FORECAST_GENERATION_TIME_GRAIN)

        (Flags, Values) = process_actuals(
            actualsMerged,
            HistoryPeriod,
            MINIMUM_PROMINENCE,
            StockoutThreshold,
            multiprocessing_num_cores,
            HISTORY_MEASURE,
            WINDOW_LENGTH,
            SPIKES,
            DIPS,
            SPIKE_OR_DIP,
            HOLIDAY_SPIKES,
            HOLIDAY_DIPS,
            PROMO_SPIKES,
            PROMO_DIPS,
            HOLIDAY,
            PROMO,
            HOLIDAY_SPIKE_OFFSET,
            HOLIDAY_DIP_OFFSET,
            PROMO_SPIKE_OFFSET,
            PROMO_DIP_OFFSET,
            OTHER_SPIKES,
            HOLIDAY_OUT,
            PROMO_OUT,
            POTENTIAL_STOCKOUT_PERIOD,
            POTENTIAL_STOCKOUT_FLAG,
            actualGrains,
            VERSION,
            INPUT_TIME_GRAIN,
            GRAINS,
            _sortByCols,
            actualGrainsForecastLevel,
            TOTAL_SPIKES_AND_DIPS,
            HOLIDAY_SPIKE_RATIO,
            PROMO_SPIKE_RATIO,
            HOLIDAY_DIP_RATIO,
            PROMO_DIP_RATIO,
            OTHER_SPIKE_RATIO,
            OTHER_DIP_RATIO,
            TOTAL_SPIKES,
            TOTAL_DIPS,
            ANALYSIS_OUTPUT_COLS,
            AGGREGATED_OUTPUT_COLS,
            pattern,
            hsLead,
            hdLead,
            psLead,
            pdLead,
            hsLag,
            hdLag,
            psLag,
            pdLag,
            OTHER_DIPS,
            offset_periods,
        )

        # filter out intersections where all flags are zero
        row_sums = Flags[flagsOutMeasureList].sum(axis=1)  # Calculate the row-wise sums
        non_zero_rows = row_sums != 0  # Get boolean mask for non-zero sum rows
        Flags = Flags[non_zero_rows]  # Filter out rows with zero sum

        # filter the first child PW against every bucket
        TimeDimension.sort_values(INPUT_TIME_KEY, inplace=True)
        first_child_pw = (
            TimeDimension.groupby(FORECAST_GENERATION_TIME_GRAIN)
            .first()[[INPUT_TIME_GRAIN]]
            .reset_index()
        )

        # spread the flags to first child PW
        Flags = Flags.merge(first_child_pw, on=FORECAST_GENERATION_TIME_GRAIN, how="inner")
        # select relevant columns
        Flags = Flags[FlagsOutCols.columns]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Flags = pd.DataFrame(columns=FlagsOutCols.columns)
        Values = pd.DataFrame(columns=OutputColumnNames.columns)
    return (Flags, Values)

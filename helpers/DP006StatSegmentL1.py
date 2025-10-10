import logging
from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
from o9Reference.stat_utils.assign_segments import assign_segments
from o9Reference.stat_utils.segmentation_utils import assign_volume_segment_to_group

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Product Customer L1 Segment": float,
    "Volume Segment L1": str,
    "COV Segment L1": str,
    "Product Segment L1": str,
    "Length of Series L1": float,
    "Number of Zeros L1": float,
    "Intermittent L1": str,
    "PLC Status L1": str,
    "Std Dev L1": float,
    "Avg Volume L1": float,
    "COV L1": float,
    "Volume L1": float,
    "Volume % L1": float,
    "Cumulative Volume % L1": float,
    "Stat Intro Date System L1": "datetime64[ns]",
    "Stat Disc Date System L1": "datetime64[ns]",
    "Item Class L1": str,
    "Product Customer L1 Segment Planning Cycle": float,
    "Volume Segment L1 Planning Cycle": str,
    "COV Segment L1 Planning Cycle": str,
    "Product Segment L1 Planning Cycle": str,
    "Length of Series L1 Planning Cycle": float,
    "Number of Zeros L1 Planning Cycle": float,
    "Intermittent L1 Planning Cycle": str,
    "PLC Status L1 Planning Cycle": str,
    "Std Dev L1 Planning Cycle": float,
    "Avg Volume L1 Planning Cycle": float,
    "COV L1 Planning Cycle": float,
    "Volume L1 Planning Cycle": float,
    "Volume % L1 Planning Cycle": float,
    "Cumulative Volume % L1 Planning Cycle": float,
    "Stat Intro Date System L1 Planning Cycle": "datetime64[ns]",
    "Stat Disc Date System L1 Planning Cycle": "datetime64[ns]",
    "Item Class L1 Planning Cycle": str,
}


def create_backtest_cycles_from_vp_vf_vs(vp, vf, vs):
    try:
        vp_int = int(vp)
        vf_int = int(vf)
        vs_int = int(vs)
        cycles = [str(x) for x in range(vp_int, vp_int + vf_int * vs_int, vs_int)]
        return cycles
    except Exception as e:
        logger.warning(f"Error creating backtest cycles from VP, VF, VS: {e}")
        return None


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    COVSegLvl,
    ForecastGenTimeBucket,
    COVThresholdMeasure,
    Actual,
    Grains,
    VolumeThresholdMeasure,
    DimClass,
    SegmentThresholds,
    MiscThresholds,
    TimeDimension,
    VolSegLvl,
    CurrentTimePeriod,
    ReadFromHive,
    SegmentationVolumeCalcGrain,
    SellOutOffset,
    PlannerOverrideCycles="None",
    RUN_SEGMENTATION_EVERY_FOLD="False",
    RUN_SEGMENTATION_EVERY_CYCLE="False",
    RUN_SEGMENTATION_FORWARD_CYCLE="True",
    PlanningCycleDates=pd.DataFrame(),
    ValidationParameters=pd.DataFrame(),
    BacktestCycle="None",
    df_keys={},
):
    try:
        StatSegmentation_list = list()
        ProductSegmentation_list = list()
        ItemClass_list = list()
        sell_out_offset_col = "Offset Period"
        planning_cycle_date_key = "Planning Cycle.[PlanningCycleDateKey]"
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")
            validation_cycle = not eval(RUN_SEGMENTATION_FORWARD_CYCLE)
            Backtest_cycles = None
            fcst_gen_time_bucket = ForecastGenTimeBucket[
                ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] == the_iteration
            ][o9Constants.FORECAST_GEN_TIME_BUCKET].unique()[0]
            time_period_key_col = "Time.[{}Key]".format(fcst_gen_time_bucket.replace(" ", ""))
            time_period_col = "Time.[{}]".format(fcst_gen_time_bucket)

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
            offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
            if PlannerOverrideCycles.replace(" ", "").lower() == "none":
                if (ValidationParameters is not None) and (len(ValidationParameters) > 0):
                    ValidationPeriod = (
                        ValidationParameters["Validation Period"].values[0]
                        if "Validation Period" in ValidationParameters.columns
                        else None
                    )
                    ValidationFold = (
                        ValidationParameters["Validation Fold"].values[0]
                        if "Validation Fold" in ValidationParameters.columns
                        else None
                    )
                    ValidationStep = (
                        ValidationParameters["Validation Step Size"].values[0]
                        if "Validation Step Size" in ValidationParameters.columns
                        else None
                    )
                    if ValidationPeriod is None:
                        logger.warning(
                            "Validation Period input is empty, please populate Validation parameters and rerun the plugin ..."
                        )
                        StatSegmentation, ProductSegmentation, ItemClass = None, None, None
                        return StatSegmentation, ProductSegmentation, ItemClass
                    ValidationFold = ValidationFold if ValidationFold is not None else 1
                    ValidationStep = ValidationStep if ValidationStep is not None else 1
                else:
                    logger.warning(
                        "Cannot execute Segmentation for Validation without relevant Validation Parameters, switching to legacy Segmentation run..."
                    )
                    logger.warning(f"CurrentTimePeriod :\n{CurrentTimePeriod}")
                    validation_cycle = False
                if validation_cycle:
                    if (BacktestCycle.strip().lower() == "none") or (BacktestCycle.strip() == ""):
                        if (
                            eval(str(ValidationPeriod)) is not None
                            and eval(str(ValidationFold)) is not None
                            and eval(str(ValidationStep)) is not None
                        ):
                            if eval(RUN_SEGMENTATION_EVERY_FOLD):
                                cycles = create_backtest_cycles_from_vp_vf_vs(
                                    ValidationPeriod, ValidationFold, ValidationStep
                                )
                                if not cycles:
                                    logger.warning(
                                        "No validation cycles available - please review the ValidationPeriod, ValidationFold, ValidationStep parameters..."
                                    )
                                    if eval(ValidationPeriod):
                                        BacktestCycle = ValidationPeriod
                                        Backtest_cycles = np.array([BacktestCycle])
                                    else:
                                        logger.warning(
                                            "ValidationPeriod is not set, returning empty outputs..."
                                        )
                                        StatSegmentation, ProductSegmentation, ItemClass = (
                                            None,
                                            None,
                                            None,
                                        )
                                        return StatSegmentation, ProductSegmentation, ItemClass
                                else:
                                    BacktestCycle = ",".join(cycles)
                                    Backtest_cycles = np.array(list(set(cycles)))
                                    Backtest_cycles = Backtest_cycles.astype(int)
                                    logger.info(f"Backtest cycles : {Backtest_cycles}")
                            else:
                                cycles = create_backtest_cycles_from_vp_vf_vs(
                                    ValidationPeriod, ValidationFold, ValidationStep
                                )
                                if not cycles:
                                    logger.warning(
                                        "No cycles available - failed to create from VP, VF, VS."
                                    )
                                    Backtest_cycles = []
                                else:
                                    BacktestCycle = max(int(c) for c in cycles)
                                    Backtest_cycles = np.array([BacktestCycle])
                                    logger.info(f"Backtest cycles: {Backtest_cycles}")
                        else:
                            logger.warning(
                                "ValidationPeriod, ValidationFold, ValidationStep are not set correctly, returning empty outputs..."
                            )
                            StatSegmentation, ProductSegmentation, ItemClass = None, None, None
                            return StatSegmentation, ProductSegmentation, ItemClass

                    else:
                        Backtest_cycles = np.array(BacktestCycle.replace(" ", "").split(","))
                        logger.info(f"Backtest cycles : {Backtest_cycles}")
                        if eval(RUN_SEGMENTATION_EVERY_CYCLE):
                            logger.info(
                                "RUN_SEGMENTATION_EVERY_CYCLE is True, processing all Backtest Cycles ..."
                            )
                            backtest_cycles_int = Backtest_cycles.astype(int)
                            if (
                                eval(str(ValidationPeriod)) is not None
                                and eval(str(ValidationFold)) is not None
                                and eval(str(ValidationStep)) is not None
                            ):
                                if eval(RUN_SEGMENTATION_EVERY_FOLD):
                                    additional_cycles = [
                                        int(x)
                                        for x in create_backtest_cycles_from_vp_vf_vs(
                                            ValidationPeriod, ValidationFold, ValidationStep
                                        )
                                    ]
                                    if not additional_cycles:
                                        logger.warning(
                                            "No additional segmentation cycles available - please review the ValidationPeriod, ValidationFold, ValidationStep parameters..."
                                        )
                                        if eval(ValidationPeriod):
                                            backtest_cycles_int += [int(ValidationPeriod)]
                                            BacktestCycle = list(set(sorted(backtest_cycles_int)))
                                            Backtest_cycles = np.array([BacktestCycle])
                                            logger.info(f"Backtest cycles: {Backtest_cycles}")
                                    else:
                                        combined_cycles = set()
                                        for backtest_cycle in backtest_cycles_int:
                                            # Add backtest_cycle to each additional cycle
                                            shifted = [
                                                backtest_cycle + ac for ac in additional_cycles
                                            ]
                                            combined_cycles.update(shifted)

                                        # Sort final list
                                        final_cycles = list(set(sorted(combined_cycles)))
                                        logger.info(f"Combined cycles: {final_cycles}")

                                        Backtest_cycles = np.array(final_cycles)
                                else:
                                    additional_cycles = [
                                        int(x)
                                        for x in create_backtest_cycles_from_vp_vf_vs(
                                            ValidationPeriod, ValidationFold, ValidationStep
                                        )
                                    ]
                                    if not additional_cycles:
                                        logger.warning(
                                            "No additional segmentation cycles available - please review the ValidationPeriod, ValidationFold, ValidationStep parameters..."
                                        )
                                        Backtest_cycles = backtest_cycles_int
                                    else:
                                        max_additional_cycle = max(additional_cycles)
                                        # For each backtest cycle, add max_additional_cycle to it
                                        shifted_cycles = [
                                            bc + max_additional_cycle for bc in backtest_cycles_int
                                        ]
                                        Backtest_cycles = np.array(
                                            list(set(sorted(shifted_cycles)))
                                        )
                                        logger.info(f"Backtest cycles: {Backtest_cycles}")
                            else:
                                # ValidationPeriod, ValidationFold, ValidationStep are not set
                                logger.warning(
                                    "ValidationPeriod, ValidationFold, ValidationStep are not set, returning empty outputs..."
                                )
                                StatSegmentation, ProductSegmentation, ItemClass = None, None, None
                                return StatSegmentation, ProductSegmentation, ItemClass
                        else:
                            # RUN_SEGMENTATION_EVERY_CYCLE == False
                            backtest_cycles_int = Backtest_cycles.astype(int)
                            if eval(RUN_SEGMENTATION_EVERY_FOLD):
                                additional_cycles = [
                                    int(x)
                                    for x in create_backtest_cycles_from_vp_vf_vs(
                                        ValidationPeriod, ValidationFold, ValidationStep
                                    )
                                ]
                                if not additional_cycles:
                                    logger.warning(
                                        "No additional segmentation cycles available - please review the ValidationPeriod, ValidationFold, ValidationStep parameters..."
                                    )
                                    Backtest_cycles = backtest_cycles_int
                                else:
                                    max_backtest_cycle = max(backtest_cycles_int)
                                    # For each backtest cycle, add max_additional_cycle to it
                                    shifted_cycles = [
                                        ac + max_backtest_cycle for ac in additional_cycles
                                    ]
                            else:
                                max_additional_cycle = max(
                                    [
                                        int(x)
                                        for x in create_backtest_cycles_from_vp_vf_vs(
                                            ValidationPeriod, ValidationFold, ValidationStep
                                        )
                                    ]
                                )
                                max_backtest_cycle = max(backtest_cycles_int)
                                shifted_cycles = [max_backtest_cycle + max_additional_cycle]
                            Backtest_cycles = np.array(list(set(sorted(shifted_cycles))))
                            logger.info(f"Backtest cycles: {Backtest_cycles}")

                    planning_cycles = []
                    Backtest_cycles += offset_periods
                    run_offset_flag = False
                    for cycle_number in Backtest_cycles:
                        cycle_date = get_n_time_periods(
                            CurrentTimePeriod[time_period_col].values[0],
                            -int(cycle_number),
                            TimeDimension,
                            {time_period_col: time_period_key_col},
                            include_latest_value=False,
                        )
                        planning_cycles.append(cycle_date[0])
                else:
                    planning_cycles = [CurrentTimePeriod[time_period_col].values[0]]
                    run_offset_flag = True

                PlanningCycles = pd.DataFrame({time_period_col: planning_cycles})
                PlanningCycles = PlanningCycles.merge(
                    TimeDimension, on=time_period_col, how="inner"
                ).drop_duplicates(subset=time_period_col)
            else:
                planning_cycles = list(set(PlannerOverrideCycles.replace(" ", "").split(",")))
                missing_cycles = list(
                    set(planning_cycles).difference(
                        set(PlanningCycleDates[o9Constants.PLANNING_CYCLE_DATE].unique())
                    )
                )
                if len(missing_cycles) > 0:
                    logger.warning(
                        f"Input cycles {missing_cycles} not in Planning Cycle Date dimension, please review the values, returning empty outputs..."
                    )
                    StatSegmentation, ProductSegmentation, ItemClass = None, None, None
                    return StatSegmentation, ProductSegmentation, ItemClass
                PlanningCycleDates = PlanningCycleDates.merge(
                    pd.DataFrame({o9Constants.PLANNING_CYCLE_DATE: planning_cycles}),
                    on=o9Constants.PLANNING_CYCLE_DATE,
                    how="inner",
                )
                time_cols = TimeDimension.columns
                if o9Constants.DAY in TimeDimension.columns:
                    PlanningCycles = PlanningCycleDates.merge(
                        TimeDimension,
                        left_on=planning_cycle_date_key,
                        right_on=o9Constants.DAY_KEY,
                        how="inner",
                    )
                else:
                    PlanningCycles = PlanningCycleDates.merge(
                        TimeDimension,
                        left_on=planning_cycle_date_key,
                        right_on=o9Constants.PARTIAL_WEEK_KEY,
                        how="inner",
                    )
                PlanningCycles = PlanningCycles[time_cols]
                if len(PlanningCycles) == 0:
                    logger.warning(
                        "Cannot find correct time records for the planning cycles, please add Time.[Day], Time.[DayKey] to the Time dimension input..."
                    )
                    StatSegmentation, ProductSegmentation, ItemClass = None, None, None
                    return StatSegmentation, ProductSegmentation, ItemClass
            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            def run_for_cycle(planning_cycle):
                # Create a new CurrentTimePeriod DataFrame for this planning cycle
                if not run_offset_flag:
                    current_time_period_for_cycle = PlanningCycles[
                        PlanningCycles[time_period_col] == planning_cycle
                    ].copy()
                else:
                    current_time_period_for_cycle = CurrentTimePeriod.copy()

                stat_seg, prod_seg, item_class = decorated_func(
                    COVSegLvl=COVSegLvl,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    COVThresholdMeasure=COVThresholdMeasure,
                    offset_periods=offset_periods,
                    run_offset_flag=run_offset_flag,
                    Actual=Actual,
                    Grains=Grains,
                    VolumeThresholdMeasure=VolumeThresholdMeasure,
                    DimClass=DimClass,
                    SegmentThresholds=SegmentThresholds,
                    MiscThresholds=MiscThresholds,
                    TimeDimension=TimeDimension,
                    VolSegLvl=VolSegLvl,
                    CurrentTimePeriod=current_time_period_for_cycle,
                    ReadFromHive=ReadFromHive,
                    SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
                    df_keys=df_keys,
                )
                # Add Planning Cycle Date column to each result
                for df in [stat_seg, prod_seg, item_class]:
                    if (
                        (not df.empty)
                        and (Backtest_cycles is not None)
                        and (validation_cycle is True)
                    ):
                        df[o9Constants.PLANNING_CYCLE_DATE] = PlanningCycleDates[
                            PlanningCycleDates[planning_cycle_date_key].dt.tz_localize(None)
                            == current_time_period_for_cycle[time_period_key_col].values[0]
                        ][o9Constants.PLANNING_CYCLE_DATE].values[0]

                return stat_seg, prod_seg, item_class

            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(run_for_cycle)(cycle) for cycle in planning_cycles
            )
            StatSegmentation_list_fi, ProductSegmentation_list_fi, ItemClass_list_fi = zip(*results)
            StatSegmentation_list.extend(StatSegmentation_list_fi)
            ProductSegmentation_list.extend(ProductSegmentation_list_fi)
            ItemClass_list.extend(ItemClass_list_fi)
        StatSegmentation = concat_to_dataframe(StatSegmentation_list)
        ProductSegmentation = concat_to_dataframe(ProductSegmentation_list)
        ItemClass = concat_to_dataframe(ItemClass_list)
        all_output_cols = list(
            set(
                StatSegmentation.columns.union(ProductSegmentation.columns).union(ItemClass.columns)
            )
        )
        measure_cols = [cols for cols in all_output_cols if "]" not in cols]
        if (Backtest_cycles is not None and len(Backtest_cycles) > 0) or (validation_cycle is True):

            def rename_and_order_output_columns(df):
                measure_cols = [cols for cols in df.columns if "]" not in cols]
                dimension_cols = [cols for cols in df.columns if "]" in cols]
                df = df[dimension_cols + measure_cols]
                return df

            measure_cols_mapping = {col: col + " Planning Cycle" for col in measure_cols}
            StatSegmentation.rename(columns=measure_cols_mapping, inplace=True)
            ProductSegmentation.rename(columns=measure_cols_mapping, inplace=True)
            ItemClass.rename(columns=measure_cols_mapping, inplace=True)
            StatSegmentation = rename_and_order_output_columns(StatSegmentation)
            ProductSegmentation = rename_and_order_output_columns(ProductSegmentation)
            ItemClass = rename_and_order_output_columns(ItemClass)
    except Exception as e:
        logger.exception(e)
        StatSegmentation, ProductSegmentation, ItemClass = None, None, None

    return StatSegmentation, ProductSegmentation, ItemClass


def processIteration(
    COVSegLvl,
    ForecastGenTimeBucket,
    COVThresholdMeasure,
    offset_periods,
    run_offset_flag,
    Actual,
    Grains,
    VolumeThresholdMeasure,
    DimClass,
    SegmentThresholds,
    MiscThresholds,
    TimeDimension,
    VolSegLvl,
    CurrentTimePeriod,
    ReadFromHive,
    SegmentationVolumeCalcGrain,
    df_keys,
):
    plugin_name = "DP006StatSegmentL1"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables - define all column names here
    class_delimiter = ","
    # PLC Segmentation
    npi_date_col = "npi_date"
    eol_date_col = "eol_date"
    start_date_col = "Stat Intro Date System L1"
    end_date_col = "Stat Disc Date System L1"
    zero_ratio_col = "ZeroRatio"

    # output measures - StatSegmentation
    vol_segment_col = "Volume Segment L1"
    cov_segment_col = "COV Segment L1"
    prod_segment_l1_col = "Product Segment L1"
    item_class_col = "Item Class L1"
    los_col = "Length of Series L1"
    num_zeros_col = "Number of Zeros L1"
    intermittency_col = "Intermittent L1"
    plc_col = "PLC Status L1"
    std_dev_col = "Std Dev L1"
    avg_col = "Avg Volume L1"
    cov_col = "COV L1"
    volume_col = "Volume L1"
    vol_share_col = "Volume % L1"
    cumulative_vol_col = "Cumulative Volume % L1"
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
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    partial_week_key_col = "Time.[PartialWeekKey]"
    history_sum_col = "History Sum"

    # output measures - ProductS
    prod_segment_col = "Product Customer L1 Segment"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    segmentation_output_grains = [str(x) for x in all_grains if x != "NA" and x != ""]

    stat_segmentation_cols = (
        [o9Constants.VERSION_NAME]
        + segmentation_output_grains
        + [
            vol_segment_col,
            cov_segment_col,
            prod_segment_l1_col,
            los_col,
            num_zeros_col,
            intermittency_col,
            plc_col,
            std_dev_col,
            avg_col,
            cov_col,
            volume_col,
            vol_share_col,
            cumulative_vol_col,
            start_date_col,
            end_date_col,
        ]
    )

    prod_segmentation_cols = (
        [o9Constants.VERSION_NAME] + segmentation_output_grains + [DimClass, prod_segment_col]
    )

    item_class_output_cols = (
        [o9Constants.VERSION_NAME] + segmentation_output_grains + [item_class_col]
    )

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

        # Filter the required columns from dataframes
        req_cols = [
            o9Constants.VERSION_NAME,
            o9Constants.INTERMITTENCY_THRESHOLD,
            o9Constants.NEW_LAUNCH_PERIOD,
            o9Constants.DISCO_PERIOD,
            o9Constants.HISTORY_TIME_BUCKETS,
            o9Constants.HISTORY_MEASURE,
            o9Constants.VOLUME_COV_HISTORY_PERIOD,
            o9Constants.HISTORY_PERIOD,
        ]
        MiscThresholds = MiscThresholds[req_cols]

        # history_measure = str(MiscThresholds[history_measure_col].iloc[0])
        # if ReadFromHive:
        #     history_measure = "DP006" + history_measure
        history_measure = "Stat Actual"

        logger.info("history_measure : {}".format(history_measure))

        req_cols = [
            o9Constants.VERSION_NAME,
            VolumeThresholdMeasure,
            COVThresholdMeasure,
        ]
        SegmentThresholds = SegmentThresholds[req_cols]

        logger.info("Extracting segmentation level ...")

        if SegmentationVolumeCalcGrain == "None" or not SegmentationVolumeCalcGrain:
            vol_segmentation_level = [o9Constants.VERSION_NAME]
        else:
            vol_segmentation_level = [x.strip() for x in SegmentationVolumeCalcGrain.split(",")]

        logger.info(f"vol_segmentation_level : {vol_segmentation_level}")

        # Combine grains, drop duplicates
        common_grains = list(set(segmentation_output_grains + vol_segmentation_level))
        logger.info("common_grains : {}".format(common_grains))

        logger.info("segmentation_output_grains : {}".format(segmentation_output_grains))

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

        # Actuals might not be present for a particular slice, check and return empty dataframe
        if Actual is None or len(Actual) == 0:
            logger.warning("Actuals is None/Empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation, ItemClass

        input_version = ForecastGenTimeBucket[o9Constants.VERSION_NAME].iloc[0]

        # split string into lists based on delimiter
        cov_levels = COVSegLvl.split(class_delimiter)
        cov_thresholds = [round(float(x), 4) for x in list(SegmentThresholds[COVThresholdMeasure])]
        # remove duplicates if any
        cov_thresholds = list(set(cov_thresholds))

        logger.info("cov_levels : {}".format(cov_levels))
        logger.info("cov_thresholds : {}".format(cov_thresholds))

        # split string into lists based on delimiter
        vol_levels = VolSegLvl.split(class_delimiter)
        vol_thresholds = [
            round(float(x), 4) for x in list(SegmentThresholds[VolumeThresholdMeasure])
        ]
        # remove duplicates if any
        vol_thresholds = list(set(vol_thresholds))

        logger.info("vol_levels : {}".format(vol_levels))
        logger.info("vol_thresholds : {}".format(vol_thresholds))

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return StatSegmentation, ProductSegmentation, ItemClass

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
            return StatSegmentation, ProductSegmentation, ItemClass

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
        Actual = Actual[common_grains + [relevant_time_name, history_measure]]
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

        segmentation_period = int(MiscThresholds[o9Constants.VOLUME_COV_HISTORY_PERIOD].iloc[0])
        logger.info("segmentation_period : {}".format(segmentation_period))

        logger.info("---------------------------------------")

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
            logger.warning("Sum of actuals is non positive for slice : {}...".format(df_keys))
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
        if offset_periods > 0 and run_offset_flag:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        # Gather NPI, EOL attributes
        npi_horizon = int(MiscThresholds[o9Constants.NEW_LAUNCH_PERIOD].iloc[0])
        eol_horizon = int(MiscThresholds[o9Constants.DISCO_PERIOD].iloc[0])
        intermittency_threshold = float(MiscThresholds[o9Constants.INTERMITTENCY_THRESHOLD].iloc[0])

        # Join actuals with time mapping
        Actual_with_time_key = Actual.copy().merge(
            relevant_time_mapping,
            on=relevant_time_name,
            how="inner",
        )

        # get the history periods
        history_periods = int(MiscThresholds[o9Constants.HISTORY_PERIOD].iloc[0])

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

        # aggregation to get start and end date
        plc_df = plc_segmentation_input_nas_filled.groupby(common_grains).agg(
            {
                relevant_time_key: [np.min],
            },
        )

        # set column names to dataframe
        plc_df.columns = [start_date_col]

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
        # Step 1: Filter DISC keys
        disc_keys = disc_data_df.loc[disc_data_df[plc_col] == "DISC", common_grains]

        # Step 2: Get non-zero Stat Actuals only for DISC intersections
        disc_nonzero_df = plc_segmentation_input_nas_filled.loc[
            plc_segmentation_input_nas_filled[history_measure] > 0
        ].merge(disc_keys, on=common_grains, how="inner")

        # Step 3: Get last time period with non-zero actual
        last_nonzero_df = (
            disc_nonzero_df.groupby(common_grains, as_index=False)[relevant_time_key]
            .max()
            .rename(columns={relevant_time_key: "LastNonZeroDate"})
        )

        # Step 4: Build sorted time list and next-date lookup
        time_sequence = (
            relevant_time_mapping[relevant_time_key]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        next_date_lookup = {
            current: time_sequence.iloc[i + 1]
            for i, current in enumerate(time_sequence[:-1])  # skip last to avoid IndexError
        }

        # Step 5: Map LastNonZeroDate → next period
        last_nonzero_df[end_date_col] = last_nonzero_df["LastNonZeroDate"].map(next_date_lookup)

        # Step 6: Merge this next-period info into plc_df
        plc_df = plc_df.merge(
            last_nonzero_df[common_grains + [end_date_col]], on=common_grains, how="left"
        )

        # Step 7: Set value only for DISC rows, others get NaT
        plc_df[end_date_col] = np.where(plc_df[plc_col] == "DISC", plc_df[end_date_col], pd.NaT)

        # Step 8: Cleanup
        plc_df[end_date_col] = pd.to_datetime(plc_df[end_date_col])

        # assign categories NEW LAUNCH
        plc_df[plc_col] = np.where(
            plc_df[start_date_col] > plc_df[npi_date_col],
            "NEW LAUNCH",
            plc_df[plc_col],
        )

        filter_clause = ~plc_df[plc_col].isin(["NEW LAUNCH", "DISC"])

        # assign category MATURE
        plc_df[plc_col] = np.where(filter_clause, "MATURE", plc_df[plc_col])

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
        result[prod_segment_l1_col] = result[plc_col]

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
                [volume_df, variability_df, result],
            )

            logger.info("Merge complete, shape  : {}".format(result.shape))

            logger.info("Assigning final PLC Segment ...")
            result[prod_segment_l1_col] = np.where(
                result[prod_segment_l1_col] == "MATURE",
                result[vol_segment_col] + result[cov_segment_col],
                result[prod_segment_l1_col],
            )

        ts_attributes_df = plc_segmentation_input_nas_filled.groupby(common_grains).agg(
            {
                history_measure: [
                    "count",
                    lambda x: x.value_counts()[0] if 0 in x.to_numpy() else 0,
                ],
            }
        )

        # assign colum names
        ts_attributes_df.columns = [
            los_col,
            num_zeros_col,
        ]

        # calculate zero ratio
        ts_attributes_df[zero_ratio_col] = (
            ts_attributes_df[num_zeros_col] / ts_attributes_df[los_col]
        )
        ts_attributes_df.reset_index(inplace=True)

        # we might have records with PLC status, but no data to calculate time series attributes
        result = ts_attributes_df.merge(result, on=common_grains, how="right")

        logger.info("df shape after combining plc with ts attributes : {}".format(result.shape))

        logger.info("Assigning intermittency categories ...")
        # Assign intermittency category
        result[intermittency_col] = np.where(
            result[zero_ratio_col] >= intermittency_threshold, "YES", "NO"
        )

        # collect version from input data
        result[o9Constants.VERSION_NAME] = input_version

        # Assign 1.0 value to product segment column
        result[prod_segment_col] = 1.0

        logger.info("Filtering relevant columns to output ...")

        # Filter relevant columns
        StatSegmentation = result[stat_segmentation_cols]

        result.rename(columns={prod_segment_l1_col: DimClass}, inplace=True)
        # Filter relevant columns
        ProductSegmentation = result[prod_segmentation_cols]
        result[item_class_col] = result[DimClass]

        ItemClass = result[item_class_output_cols]

        logger.info("Successfully executed {} ...".format(plugin_name))
        logger.info("---------------------------------------------")
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        StatSegmentation = pd.DataFrame(columns=stat_segmentation_cols)
        ProductSegmentation = pd.DataFrame(columns=prod_segmentation_cols)
        ItemClass = pd.DataFrame(columns=item_class_output_cols)

    return StatSegmentation, ProductSegmentation, ItemClass

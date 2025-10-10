import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
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

from helpers.o9Constants import o9Constants
from helpers.utils import add_dim_suffix, filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Actual Final Profile PL": float,
    "Stat Fcst PL CML Baseline": float,
    "Stat Fcst PL CML Residual": float,
    "Stat Fcst PL CML Holiday": float,
    "Stat Fcst PL CML Promo": float,
    "Stat Fcst PL CML Marketing": float,
    "Stat Fcst PL CML Price": float,
    "Stat Fcst PL CML Weather": float,
    "Stat Fcst PL CML External Driver": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actual,
    PItemDates,
    StatActual,
    CMLFcstL1,
    TimeDimension,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    ForecastLevelData,
    StatGrains,
    PlanningGrains,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    LocationMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    HistoryWindowinWeeks,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                Actual=Actual,
                PItemDates=PItemDates,
                StatActual=StatActual,
                CMLFcstL1=CMLFcstL1,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                ForecastLevelData=ForecastLevelData,
                StatGrains=StatGrains,
                PlanningGrains=PlanningGrains,
                HistoryWindowinWeeks=HistoryWindowinWeeks,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                LocationMasterData=LocationMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    Actual,
    PItemDates,
    StatActual,
    CMLFcstL1,
    TimeDimension,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    ForecastLevelData,
    StatGrains,
    PlanningGrains,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    LocationMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    HistoryWindowinWeeks,
    df_keys,
):
    plugin_name = "DP082SpreadHistoryDecompFcstL1toPL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # Configurables
    intro_date_col = "Intro Date"
    disc_date_col = "Disco Date"
    stat_history_start_date = "Stat History Start Date"
    actual_final_profile_pl = "Actual Final Profile PL"

    # ML Driver inputs
    ML_FCST_L1_CML_BASELINE: str = "ML Fcst L1 CML Baseline"
    ML_FCST_L1_CML_RESIDUAL: str = "ML Fcst L1 CML Residual"
    ML_FCST_L1_CML_HOLIDAY: str = "ML Fcst L1 CML Holiday"
    ML_FCST_L1_CML_PROMO: str = "ML Fcst L1 CML Promo"
    ML_FCST_L1_CML_MARKETING: str = "ML Fcst L1 CML Marketing"
    ML_FCST_L1_CML_PRICE: str = "ML Fcst L1 CML Price"
    ML_FCST_L1_CML_WEATHER: str = "ML Fcst L1 CML Weather"
    ML_FCST_L1_CML_EXTERNAL_DRIVER: str = "ML Fcst L1 CML External Driver"
    ml_driver_inputs = [
        ML_FCST_L1_CML_BASELINE,
        ML_FCST_L1_CML_RESIDUAL,
        ML_FCST_L1_CML_HOLIDAY,
        ML_FCST_L1_CML_PROMO,
        ML_FCST_L1_CML_MARKETING,
        ML_FCST_L1_CML_PRICE,
        ML_FCST_L1_CML_WEATHER,
        ML_FCST_L1_CML_EXTERNAL_DRIVER,
    ]
    input_prefix = "ML Fcst L1"
    output_prefix = "Stat Fcst PL"
    stat_grains = StatGrains.split(",")
    stat_grains = [x.strip() for x in stat_grains]
    stat_grains = [str(x) for x in stat_grains if x != "NA" and x != ""]
    planning_grains = PlanningGrains.split(",")
    planning_grains = [x.strip() for x in planning_grains]
    planning_grains = [str(x) for x in planning_grains if x != "NA" and x != ""]
    cols_required_in_output = (
        [o9Constants.VERSION_NAME]
        + planning_grains
        + [o9Constants.PARTIAL_WEEK, actual_final_profile_pl]
        + [x.replace(input_prefix, output_prefix) for x in ml_driver_inputs]
    )
    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        if len(Actual) == 0:
            logger.warning("Planning level actuals empty, returning empty output")
            return Output

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return Output
        write_fcst_output = True
        if len(CMLFcstL1) == 0:
            logger.warning(
                "CML Fcst empty for this forecast iteration, returning empty forecast outputs"
            )
            write_fcst_output = False

        if len(StatActual) == 0:
            logger.warning("Stat Actual empty for this forecast iteration, returning empty output")
            return Output

        input_version = StatActual[o9Constants.VERSION_NAME].values[0]
        default_disc_date = (
            pd.to_datetime(CurrentTimePeriod[o9Constants.DAY_KEY]).dt.tz_localize(None).values[0]
        )
        default_intro_date = default_disc_date - pd.DateOffset(weeks=int(HistoryWindowinWeeks))

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[o9Constants.FORECAST_GEN_TIME_BUCKET].unique()[
            0
        ]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [o9Constants.PARTIAL_WEEK, o9Constants.WEEK, o9Constants.WEEK_KEY]
            relevant_time_name = o9Constants.WEEK
            relevant_time_key = o9Constants.WEEK_KEY
            HistoryWindow = int(HistoryWindowinWeeks)
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PLANNING_MONTH,
                o9Constants.PLANNING_MONTH_KEY,
            ]
            relevant_time_name = o9Constants.PLANNING_MONTH
            relevant_time_key = o9Constants.PLANNING_MONTH_KEY
            HistoryWindow = round(int(HistoryWindowinWeeks) / 4.34524)
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.MONTH,
                o9Constants.MONTH_KEY,
            ]
            relevant_time_name = o9Constants.MONTH
            relevant_time_key = o9Constants.MONTH_KEY
            HistoryWindow = round(int(HistoryWindowinWeeks) / 4.34524)
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.QUARTER,
                o9Constants.QUARTER_KEY,
            ]
            relevant_time_name = o9Constants.QUARTER
            relevant_time_key = o9Constants.QUARTER_KEY
            HistoryWindow = round(int(HistoryWindowinWeeks) / 13)
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PLANNING_QUARTER,
                o9Constants.PLANNING_QUARTER_KEY,
            ]
            relevant_time_name = o9Constants.PLANNING_QUARTER
            relevant_time_key = o9Constants.PLANNING_QUARTER_KEY
            HistoryWindow = round(int(HistoryWindowinWeeks) / 13)
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return Output
        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")
        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}

        Actual_copy = Actual.copy()

        # Join Actuals with time mapping
        StatActual = StatActual.merge(base_time_mapping, on=o9Constants.PARTIAL_WEEK, how="inner")

        # select the relevant columns, groupby and sum history measure
        StatActual = (
            StatActual.groupby(stat_grains + [relevant_time_name])
            .agg({o9Constants.STAT_ACTUAL: "sum"})
            .reset_index()
        )
        Actual = Actual.merge(
            TimeDimension[
                [o9Constants.PARTIAL_WEEK, o9Constants.PARTIAL_WEEK_KEY]
            ].drop_duplicates(),
            on=o9Constants.PARTIAL_WEEK,
            how="inner",
        )

        if len(PItemDates) == 0:
            logger.info("Planning item dataframe is empty, creating one using default dates ...")
            # create dataframe with all combinations
            PItemDates = Actual[planning_grains].drop_duplicates()
            PItemDates.reset_index(drop=True, inplace=True)
            PItemDates[intro_date_col] = default_intro_date
            PItemDates[disc_date_col] = default_disc_date
            PItemDates[stat_history_start_date] = np.nan
            PItemDates[o9Constants.VERSION_NAME] = input_version
            PItemDates = PItemDates[
                [o9Constants.VERSION_NAME]
                + planning_grains
                + [intro_date_col, disc_date_col, stat_history_start_date]
            ]
        else:
            if PItemDates[intro_date_col].dtype == "datetime64[ns, UTC]":
                logger.info(f"Converting {intro_date_col} from datetime64[ns, UTC] to datetime ...")
                PItemDates[intro_date_col] = pd.to_datetime(
                    PItemDates[intro_date_col]
                ).dt.tz_localize(None)

            if PItemDates[disc_date_col].dtype == "datetime64[ns, UTC]":
                logger.info(f"Converting {disc_date_col} from datetime64[ns, UTC] to datetime ...")
                PItemDates[disc_date_col] = pd.to_datetime(
                    PItemDates[disc_date_col]
                ).dt.tz_localize(None)
            # create dataframe containing all intersections in Actuals
            all_intersections_df = Actual[planning_grains].drop_duplicates()

            # outer join to get all combinations including the ones with actuals and new items
            PItemDates = PItemDates.merge(
                all_intersections_df,
                how="right",
                on=planning_grains,
            )

            logger.info("Filling missing dates with default values ...")
            # fill missing entries with default values
            PItemDates[o9Constants.VERSION_NAME].fillna(input_version, inplace=True)
            PItemDates[intro_date_col].fillna(default_intro_date, inplace=True)
            PItemDates[disc_date_col].fillna(default_disc_date, inplace=True)

        # Assign forecast start and end dates
        PItemDates[stat_history_start_date].fillna(default_intro_date, inplace=True)
        # PItemDates[stat_history_start_date] = PItemDates[stat_history_start_date].dt.tz_convert('UTC')
        if PItemDates[stat_history_start_date].dtype == "datetime64[ns, UTC]":
            logger.info(
                f"Converting {stat_history_start_date} from datetime64[ns, UTC] to datetime ..."
            )
            PItemDates[stat_history_start_date] = pd.to_datetime(
                PItemDates[stat_history_start_date]
            ).dt.tz_localize(None)
        PItemDates[stat_history_start_date] = pd.to_datetime(
            PItemDates[stat_history_start_date]
        ).dt.normalize()
        # Compare Intro date with Stat History Start Date to find the later start date for profile generation
        PItemDates[intro_date_col] = np.where(
            PItemDates[intro_date_col] < PItemDates[stat_history_start_date],
            PItemDates[stat_history_start_date],
            PItemDates[intro_date_col],
        )
        # Compare Disc date with current date to limit profile generation till current date
        PItemDates[disc_date_col] = np.where(
            PItemDates[disc_date_col] > default_disc_date,
            default_disc_date,
            PItemDates[disc_date_col],
        )
        PItemDates = PItemDates[
            [o9Constants.VERSION_NAME] + planning_grains + [intro_date_col, disc_date_col]
        ].drop_duplicates()

        # piece of code to map the intro date to the start of time bucket, so that transition at the middle of the bucket is respected
        PItemDates[intro_date_col] = pd.to_datetime(
            PItemDates[intro_date_col]
        ).dt.normalize()  # normalizing the timestamps to make it insensitive of time HH:MM:SS values for merging with TimeDim
        PItemDates[disc_date_col] = pd.to_datetime(
            PItemDates[disc_date_col]
        ).dt.normalize()  # normalizing the timestamps to make it insensitive of time HH:MM:SS values for merging with TimeDim
        TimeDimension[o9Constants.DAY_KEY] = (
            TimeDimension[o9Constants.DAY_KEY].dt.tz_localize(None)
        ).dt.normalize()
        TimeDimension[relevant_time_key] = (
            TimeDimension[relevant_time_key].dt.tz_localize(None)
        ).dt.normalize()
        time_cols_required = [o9Constants.DAY_KEY, relevant_time_key]
        PItemDates = PItemDates.merge(
            TimeDimension[time_cols_required].drop_duplicates(),
            left_on=intro_date_col,
            right_on=o9Constants.DAY_KEY,
            how="inner",
        )

        # Update the intro date column to the relevant time key
        PItemDates[intro_date_col] = PItemDates[relevant_time_key]

        # Drop the columns added during the merge to maintain original structure
        PItemDates.drop(columns=[o9Constants.DAY_KEY, relevant_time_key], inplace=True)

        if len(PItemDates) == 0:
            logger.warning(
                "Empty dataframe after joining intersections master data with planning item dates for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe for this slice ...")
            return Output

        logger.info("Assign forecast start and end dates ...")

        # Join Actuals with time mapping
        Actual = Actual.merge(base_time_mapping, on=o9Constants.PARTIAL_WEEK, how="inner")
        Actual = (
            Actual.groupby(
                planning_grains + [relevant_time_name],
                observed=True,
            )
            .agg({o9Constants.ACTUAL: "sum"})
            .reset_index()
        )

        current_time_bucket = CurrentTimePeriod[relevant_time_name][0]
        # get last n periods
        last_n_periods = get_n_time_periods(
            current_time_bucket,
            -HistoryWindow,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        # fill NAs with zero
        relevant_history_nas_filled = fill_missing_dates(
            actual=Actual,
            forecast_level=planning_grains,
            history_measure=o9Constants.ACTUAL,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            time_mapping=relevant_time_mapping,
            relevant_time_periods=last_n_periods,
            fill_nulls_with_zero=True,
            filter_from_start_date=True,
        )
        # cap negatives
        relevant_history_nas_filled[o9Constants.ACTUAL] = np.where(
            relevant_history_nas_filled[o9Constants.ACTUAL] < 0,
            0,
            relevant_history_nas_filled[o9Constants.ACTUAL],
        )
        actual_df = relevant_history_nas_filled.copy()
        actual_df = actual_df.merge(PItemDates, on=planning_grains, how="inner")

        actual_df[relevant_time_key] = (
            actual_df[relevant_time_key].dt.tz_localize(None)
        ).dt.normalize()

        # filter out actuals which are not in the range of the intro/disc dates
        actual_df = actual_df[
            (actual_df[relevant_time_key] >= actual_df[intro_date_col])
            & (actual_df[relevant_time_key] <= actual_df[disc_date_col])
        ]
        # add stat grains to history
        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Channel"] = ChannelMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData
        master_data_dict["Region"] = RegionMasterData
        master_data_dict["Account"] = AccountMasterData
        master_data_dict["PnL"] = PnLMasterData
        master_data_dict["Location"] = LocationMasterData

        all_intersections_in_actual = Actual[planning_grains].drop_duplicates()

        level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
        for the_col in level_cols:

            # extract 'Item' from 'Item Level'
            the_dim = the_col.split(" Level")[0]

            logger.debug(f"the_dim : {the_dim}")

            # all dims exception location will be planning location
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
            all_intersections_in_actual = all_intersections_in_actual.merge(
                the_dim_data, on=the_child_col, how="inner"
            )

            logger.debug("-------------------------")

        History = actual_df.merge(all_intersections_in_actual, on=planning_grains, how="inner")
        History["actual_sum_PL"] = History.groupby(planning_grains)[o9Constants.ACTUAL].transform(
            sum
        )
        History["group_len"] = History.groupby(planning_grains)[o9Constants.ACTUAL].transform(
            "count"
        )
        History["actual_mean"] = History["actual_sum_PL"] / History["group_len"].replace(0, np.nan)
        PLProfile = History[
            [o9Constants.VERSION_NAME] + stat_grains + planning_grains + ["actual_mean"]
        ].drop_duplicates()
        PLProfile["actual_mean_stat"] = PLProfile.groupby([o9Constants.VERSION_NAME] + stat_grains)[
            "actual_mean"
        ].transform(sum)
        PLProfile[actual_final_profile_pl] = PLProfile["actual_mean"] / PLProfile[
            "actual_mean_stat"
        ].replace(0, np.nan)
        PLProfile = PLProfile[
            [o9Constants.VERSION_NAME] + stat_grains + planning_grains + [actual_final_profile_pl]
        ].drop_duplicates()
        l1_to_pl_measure_mapping = {
            l1_measure: l1_measure.replace(input_prefix, output_prefix)
            for l1_measure in ml_driver_inputs
        }

        if write_fcst_output:
            # Use profiles to split Historical Stat measures
            CMLFcstL1 = CMLFcstL1.merge(
                PLProfile, on=[o9Constants.VERSION_NAME] + stat_grains, how="inner"
            )
            CMLFcstL1["stat_fcst_sum"] = 0
            for l1_measure in ml_driver_inputs:
                if l1_measure != ML_FCST_L1_CML_RESIDUAL:
                    CMLFcstL1[l1_to_pl_measure_mapping[l1_measure]] = (
                        CMLFcstL1[l1_measure] * CMLFcstL1[actual_final_profile_pl]
                    )
                    CMLFcstL1["stat_fcst_sum"] += CMLFcstL1[
                        l1_to_pl_measure_mapping[l1_measure]
                    ].fillna(0)
            CMLFcstL1 = CMLFcstL1.merge(
                Actual_copy,
                on=planning_grains + [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK],
                how="left",
            )
            CMLFcstL1[l1_to_pl_measure_mapping[ML_FCST_L1_CML_RESIDUAL]] = (
                CMLFcstL1[o9Constants.ACTUAL].fillna(0) - CMLFcstL1["stat_fcst_sum"]
            )
            CMLFcstL1 = CMLFcstL1.merge(
                TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.PARTIAL_WEEK_KEY]
                ].drop_duplicates(),
                on=o9Constants.PARTIAL_WEEK,
                how="inner",
            )
        else:
            CMLFcstL1 = PLProfile
            for l1_measure in ml_driver_inputs:
                CMLFcstL1[l1_to_pl_measure_mapping[l1_measure]] = np.nan
            TimeDimension = TimeDimension[
                (TimeDimension[relevant_time_key] >= default_intro_date)
                & (TimeDimension[relevant_time_key] <= default_disc_date)
            ]
            CMLFcstL1 = create_cartesian_product(
                TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.PARTIAL_WEEK_KEY]
                ].drop_duplicates(),
                CMLFcstL1,
            )
        CMLFcstL1 = CMLFcstL1[
            CMLFcstL1[o9Constants.PARTIAL_WEEK_KEY].dt.tz_localize(None)
            < CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY].values[0]
        ]
        Output = CMLFcstL1[cols_required_in_output].drop_duplicates()
        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
    return Output

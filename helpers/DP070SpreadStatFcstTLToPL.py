import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from scipy.interpolate import interp1d

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration, get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")


def interpolate_profile_to_df(df, target_bucket_count, method):
    """
    Interpolates a profile to the target number of buckets using the specified method.

    Parameters:
        df: Dataframe of original profile values.
        target_bucket_count (int): Number of output buckets.
        method (str): Interpolation method ('linear', 'cubic', etc.).

    Returns:
        pd.DataFrame: DataFrame with 'Bucket' and 'Value' columns.
    """

    original_values = np.array(df[constants.default_profile])
    original_len = len(original_values)

    # Normalized positions
    original_positions = np.linspace(0, 1, original_len)
    target_positions = np.linspace(0, 1, target_bucket_count)

    # Interpolator
    interpolator = interp1d(original_positions, original_values, kind=method)
    interpolated_values = interpolator(target_positions)

    target_labels = [f"B{i:03d}" for i in range(1, target_bucket_count + 1)]
    result_df = pd.DataFrame(
        {
            constants.lifecycle_bucket_col: target_labels,
            constants.default_profile: np.round(interpolated_values, 2),
        }
    )

    df.drop(columns=[constants.default_profile], inplace=True)
    df = df.merge(result_df)

    return df


def get_eol_profiles(
    df,
    transition_data,
    dimensions,
    DefaultProfiles,
    relevant_time_name,
    relevant_time_key,
    TimeDimension,
):
    df[constants.phase_out_profile].fillna("Default Gradual", inplace=True)
    df = df.merge(
        DefaultProfiles,
        how="left",
        left_on=[constants.version_col, constants.phase_out_profile],
        right_on=[constants.version_col, constants.plc_profile_col],
    )
    adjust_profile = bool(df[constants.adjust_phase_out_profile].iloc[0])
    start_date = df["Start Date"].iloc[0]
    end_date = df["End Date"].iloc[0]
    method = df[constants.phase_out_profile].iloc[0]
    filtered_df = df[dimensions + [constants.planning_item_col]].drop_duplicates()
    transition_data = transition_data[transition_data["count"] == 1]
    filtered_transition = transition_data.merge(filtered_df)

    if filtered_transition.empty:
        logger.warning(f"No transition available for group: {filtered_df}")
        return pd.DataFrame()

    relevant_time_dim = TimeDimension[
        (TimeDimension[relevant_time_key] > start_date)
        & (TimeDimension[relevant_time_key] <= end_date)
    ][[constants.partial_week_col, relevant_time_name, relevant_time_key]].drop_duplicates()

    filtered_transition = filtered_transition.merge(relevant_time_dim)
    relevant_time_dim.drop(columns=[constants.partial_week_col], inplace=True)
    relevant_time_dim = relevant_time_dim.drop_duplicates().reset_index(drop=True)
    if filtered_transition.empty:
        logger.warning(
            f"No transition available for group in specified time periods: {filtered_df}"
        )
        return pd.DataFrame()

    periods = len(relevant_time_dim)

    if method == "Default Gradual":
        step = 1 / (periods + 1)
        profile_values = [1 - step * i for i in range(1, periods + 1)]
        relevant_time_dim[constants.DISAGG_PROPORTION] = profile_values
        filtered_transition = filtered_transition.merge(relevant_time_dim)
        relevant_time_dim.drop(columns=[constants.DISAGG_PROPORTION], inplace=True)

    else:
        if adjust_profile:
            df = interpolate_profile_to_df(
                df=df,
                target_bucket_count=periods,
                method="cubic",
            )
        relevant_time_dim["Time Sequence"] = (
            relevant_time_dim[relevant_time_key].rank().astype("int")
        )
        df["Time Sequence"] = df[constants.lifecycle_bucket_col].apply(lambda x: int(x[1:]))

        relevant_time_dim = relevant_time_dim.merge(
            filtered_transition[
                [constants.planning_item_col, relevant_time_name]
            ].drop_duplicates(),
        )

        relevant_time_dim = relevant_time_dim.merge(
            df[
                [
                    constants.planning_item_col,
                    constants.plc_profile_col,
                    "Time Sequence",
                    constants.default_profile,
                ]
            ].drop_duplicates(),
            how="left",
        )
        relevant_time_dim.rename(
            columns={constants.default_profile: constants.DISAGG_PROPORTION}, inplace=True
        )
        relevant_time_dim.sort_values(
            by=[constants.planning_item_col, relevant_time_key], inplace=True
        )
        relevant_time_dim[constants.DISAGG_PROPORTION] = relevant_time_dim[
            constants.DISAGG_PROPORTION
        ].ffill()
        relevant_time_dim[constants.plc_profile_col] = relevant_time_dim[
            constants.plc_profile_col
        ].ffill()

        filtered_transition = filtered_transition.merge(
            relevant_time_dim,
        )
        filtered_transition.drop(columns=["Time Sequence", constants.plc_profile_col], inplace=True)
        relevant_time_dim.drop(
            columns=[
                constants.planning_item_col,
                constants.plc_profile_col,
                "Time Sequence",
                constants.DISAGG_PROPORTION,
            ],
            inplace=True,
        )

    return filtered_transition


class constants:
    # configurables
    GROUP_SUM: str = "Group Sum"
    DISAGG_PROPORTION: str = "Disagg Proportion"
    CML_ITERATION_DECOMPOSITION: str = "CML Iteration Decomposition"
    STAT_FCST_TL_CML_BASELINE: str = "Stat Fcst TL CML Baseline"
    STAT_FCST_TL_CML_RESIDUAL: str = "Stat Fcst TL CML Residual"
    STAT_FCST_TL_CML_HOLIDAY: str = "Stat Fcst TL CML Holiday"
    STAT_FCST_TL_CML_PROMO: str = "Stat Fcst TL CML Promo"
    STAT_FCST_TL_CML_MARKETING: str = "Stat Fcst TL CML Marketing"
    STAT_FCST_TL_CML_PRICE: str = "Stat Fcst TL CML Price"
    STAT_FCST_TL_CML_WEATHER: str = "Stat Fcst TL CML Weather"
    STAT_FCST_TL_CML_EXTERNAL_DRIVER: str = "Stat Fcst TL CML External Driver"
    STAT_FCST_PL_CML_BASELINE: str = "Stat Fcst PL CML Baseline"
    STAT_FCST_PL_CML_RESIDUAL: str = "Stat Fcst PL CML Residual"
    STAT_FCST_PL_CML_HOLIDAY: str = "Stat Fcst PL CML Holiday"
    STAT_FCST_PL_CML_PROMO: str = "Stat Fcst PL CML Promo"
    STAT_FCST_PL_CML_MARKETING: str = "Stat Fcst PL CML Marketing"
    STAT_FCST_PL_CML_PRICE: str = "Stat Fcst PL CML Price"
    STAT_FCST_PL_CML_WEATHER: str = "Stat Fcst PL CML Weather"
    STAT_FCST_PL_CML_EXTERNAL_DRIVER: str = "Stat Fcst PL CML External Driver"
    STAT_FCST_PL_vs_LC_COCC: str = "Stat Fcst PL vs LC COCC"
    STAT_FCST_PL_LC: str = "Stat Fcst PL LC"
    FCST_NEXT_N_BUCKETS_PL: str = "Fcst Next N Buckets PL"
    PL_VOLUME_LOSS_FLAG: str = "PL Volume Loss Flag"
    CURRENT_PARTIAL_WEEK_KEY: str = "current_partial_weekkey"
    INTRO_DATE: str = "Intro Date"
    DISCO_DATE: str = "Disco Date"
    STAT_FCST_FINAL_PROFILE_PL: str = "Stat Fcst Final Profile PL"

    version_col = o9Constants.VERSION_NAME
    plc_profile_col = "PLC Profile.[PLC Profile]"
    lifecycle_bucket_col = "Lifecycle Time.[Lifecycle Bucket]"
    transition_item_col = o9Constants.TRANSITION_ITEM
    transition_demand_domain = o9Constants.TRANSITION_DEMAND_DOMAIN
    planning_item_col = o9Constants.PLANNING_ITEM
    planning_location_col = o9Constants.PLANNING_LOCATION
    planning_pnl_col = o9Constants.PLANNING_PNL
    planning_channel_col = o9Constants.PLANNING_CHANNEL
    planning_region_col = o9Constants.PLANNING_REGION
    planning_account_col = o9Constants.PLANNING_ACCOUNT
    planning_demand_domain_col = o9Constants.PLANNING_DEMAND_DOMAIN

    forecast_iteration = o9Constants.FORECAST_ITERATION
    STAT_FCST_PL = o9Constants.STAT_FCST_PL
    STAT_FCST_TL = o9Constants.STAT_FCST_TL
    FORECAST_GEN_TIME_BUCKET = o9Constants.FORECAST_GEN_TIME_BUCKET
    STAT_BUCKET_WEIGHT = o9Constants.STAT_BUCKET_WEIGHT

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

    phase_out_profile = "Phase Out Profile"
    phase_out_buckets = "Number of Phase Out Buckets"
    adjust_phase_out_profile = "Adjust Phase Out Profile"
    prod_transition_start_date = "Product Transition Overlap Start Date"
    default_profile = "Default Profile"
    transition_type = "Transition Type"
    transition_period_gap = "Transition Period Gap"


col_mapping = {
    "Stat Fcst PL": float,
    "Stat Fcst PL CML Baseline": float,
    "Stat Fcst PL CML Residual": float,
    "Stat Fcst PL CML Holiday": float,
    "Stat Fcst PL CML Promo": float,
    "Stat Fcst PL CML Marketing": float,
    "Stat Fcst PL CML Price": float,
    "Stat Fcst PL CML Weather": float,
    "Stat Fcst PL CML External Driver": float,
    "Stat Fcst PL vs LC COCC": float,
    "Fcst Next N Buckets PL": float,
    "PL Volume Loss Flag": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    StatFcstFinalProfilePL,
    StatFcstTL,
    CMLFcstTL,
    ItemMasterData,
    DemandDomainMasterData,
    PItemMetaData,
    MLDecompositionFlag,
    StatFcstPLLC,
    CurrentTimePeriod,
    TimeDimension,
    StatBucketWeight,
    ForecastGenTimeBucket,
    NBucketsinMonths,
    DefaultProfiles,
    df_keys,
):
    try:
        OutputList = list()
        CMLOutputList = list()
        Output_volume_loss_flag_list = list()
        StatFcstPLvsLC_COCCList = list()
        for the_iteration in StatFcstTL[constants.forecast_iteration].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            # Filter the row where iteration matches
            match_row = ForecastGenTimeBucket[
                ForecastGenTimeBucket["Forecast Iteration.[Forecast Iteration]"] == the_iteration
            ]

            # Extract the corresponding Forecast Generation Time Bucket value
            if not match_row.empty:
                forecastgenbucketname = match_row[constants.FORECAST_GEN_TIME_BUCKET].values[0]
            else:
                forecastgenbucketname = "Week"

            logger.info(
                f"Forecast Generation Time Bucket for {the_iteration}: {forecastgenbucketname}"
            )

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output, cml_output, cocc_output, the_Output_volume_loss_flag = decorated_func(
                Grains=Grains,
                StatFcstFinalProfilePL=StatFcstFinalProfilePL,
                StatFcstTL=StatFcstTL,
                CMLFcstTL=CMLFcstTL,
                ItemMasterData=ItemMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                PItemMetaData=PItemMetaData,
                the_iteration=the_iteration,
                MLDecompositionFlag=MLDecompositionFlag,
                StatFcstPLLC=StatFcstPLLC,
                CurrentTimePeriod=CurrentTimePeriod,
                TimeDimension=TimeDimension,
                StatBucketWeight=StatBucketWeight,
                forecastgenbucketname=forecastgenbucketname,
                NBucketsinMonths=NBucketsinMonths,
                DefaultProfiles=DefaultProfiles,
                df_keys=df_keys,
            )

            OutputList.append(the_output)
            CMLOutputList.append(cml_output)
            StatFcstPLvsLC_COCCList.append(cocc_output)
            Output_volume_loss_flag_list.append(the_Output_volume_loss_flag)

        Output = concat_to_dataframe(OutputList)
        CMLOutput = concat_to_dataframe(CMLOutputList)
        StatFcstPLvsLC_COCC = concat_to_dataframe(StatFcstPLvsLC_COCCList)
        Output_volume_loss_flag = concat_to_dataframe(Output_volume_loss_flag_list)
    except Exception as e:
        logger.exception(e)
        Output = None
        CMLOutput = None
        StatFcstPLvsLC_COCC = None
        Output_volume_loss_flag = None
    return Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag


def processIteration(
    Grains,
    StatFcstFinalProfilePL,
    StatFcstTL,
    ItemMasterData,
    DemandDomainMasterData,
    PItemMetaData,
    the_iteration,
    forecastgenbucketname,
    CMLFcstTL=pd.DataFrame(),
    MLDecompositionFlag=pd.DataFrame(),
    StatFcstPLLC=pd.DataFrame(),
    CurrentTimePeriod=pd.DataFrame(),
    TimeDimension=pd.DataFrame(),
    StatBucketWeight=pd.DataFrame(),
    NBucketsinMonths="12",
    DefaultProfiles=pd.DataFrame(),
    df_keys={},
):
    plugin_name = "DP070SpreadStatFcstTLToPL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # combine grains to get segmentation level
    dimensions = get_list_of_grains_from_string(input=Grains)

    cols_required_in_output = (
        [constants.version_col, constants.partial_week_col] + dimensions + [constants.STAT_FCST_PL]
    )
    cols_required_in_cocc_output = (
        [constants.version_col, constants.partial_week_col]
        + dimensions
        + [constants.STAT_FCST_PL_vs_LC_COCC]
    )
    cols_required_in_cml_output = (
        [constants.version_col, constants.partial_week_col]
        + dimensions
        + [
            constants.STAT_FCST_PL_CML_BASELINE,
            constants.STAT_FCST_PL_CML_RESIDUAL,
            constants.STAT_FCST_PL_CML_HOLIDAY,
            constants.STAT_FCST_PL_CML_PROMO,
            constants.STAT_FCST_PL_CML_MARKETING,
            constants.STAT_FCST_PL_CML_PRICE,
            constants.STAT_FCST_PL_CML_WEATHER,
            constants.STAT_FCST_PL_CML_EXTERNAL_DRIVER,
        ]
    )

    cols_required_in_volume_loss_flag_output = dimensions + [
        constants.version_col,
        constants.PL_VOLUME_LOSS_FLAG,
    ]

    cols_required_in_fcst_next_n_buckets_pl = dimensions + [
        constants.version_col,
        constants.FCST_NEXT_N_BUCKETS_PL,
    ]
    Output = pd.DataFrame(columns=cols_required_in_output)
    CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
    StatFcstPLvsLC_COCC = pd.DataFrame(columns=cols_required_in_cocc_output)
    Output_volume_loss_flag = pd.DataFrame(
        columns=cols_required_in_volume_loss_flag_output + [constants.FCST_NEXT_N_BUCKETS_PL]
    )
    try:
        # filtering out data where all required columns are null
        StatFcstFinalProfilePL.dropna(
            subset=[constants.STAT_FCST_FINAL_PROFILE_PL], how="all", inplace=True
        )
        StatFcstTL.dropna(subset=[constants.STAT_FCST_TL], how="all", inplace=True)
        CMLFcstTL.dropna(
            subset=[
                constants.STAT_FCST_TL_CML_BASELINE,
                constants.STAT_FCST_TL_CML_EXTERNAL_DRIVER,
                constants.STAT_FCST_TL_CML_HOLIDAY,
                constants.STAT_FCST_TL_CML_MARKETING,
                constants.STAT_FCST_TL_CML_PRICE,
                constants.STAT_FCST_TL_CML_PROMO,
                constants.STAT_FCST_TL_CML_RESIDUAL,
                constants.STAT_FCST_TL_CML_WEATHER,
            ],
            how="all",
            inplace=True,
        )
        StatFcstPLLC.dropna(subset=[constants.STAT_FCST_PL_LC], how="all", inplace=True)

        if StatFcstFinalProfilePL.empty:
            logger.warning(f"StatFcstFinalProfilePL is empty for {df_keys}")
            return Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag

        if StatFcstTL.empty:
            logger.warning(f"StatFcstTL is empty for {df_keys}")
            return Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag

        key_cols = [
            constants.day_key_col,
            constants.partial_week_key_col,
            constants.week_key_col,
            constants.month_key_col,
            constants.planning_month_key_col,
            constants.quarter_key_col,
            constants.planning_quarter_key_col,
            constants.INTRO_DATE,
            constants.DISCO_DATE,
        ]

        for col in key_cols:
            if col in TimeDimension.columns:
                TimeDimension[col] = pd.to_datetime(TimeDimension[col], utc=True).dt.tz_localize(
                    None
                )
            if col in StatFcstTL.columns:
                StatFcstTL[col] = pd.to_datetime(StatFcstTL[col], utc=True).dt.tz_localize(None)
            if col in PItemMetaData.columns:
                PItemMetaData[col] = pd.to_datetime(PItemMetaData[col], utc=True).dt.tz_localize(
                    None
                )
                # Initialize the result
                max_forecast_date = StatFcstTL[constants.partial_week_key_col].max()

        default_intro_date = TimeDimension[constants.partial_week_key_col].min()
        default_disc_date = max_forecast_date

        PItemMetaData[constants.INTRO_DATE].fillna(default_intro_date, inplace=True)
        PItemMetaData[constants.DISCO_DATE].fillna(default_disc_date, inplace=True)

        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData

        # join with ItemMasterData and DemandDomainMasterData with StatFcstFinalProfilePL

        StatFcstFinalProfilePL = StatFcstFinalProfilePL.merge(
            ItemMasterData, on=constants.planning_item_col, how="inner"
        )

        StatFcstFinalProfilePL = StatFcstFinalProfilePL.merge(
            DemandDomainMasterData,
            on=constants.planning_demand_domain_col,
            how="inner",
        )

        logger.info(
            "Merged StatFcstFinalProfilePL with ItemMasterData and DemandDomainMasterData ..."
        )

        StatFcstTL_columns = StatFcstTL.columns.tolist()

        # Remove columns that are not required Stat Fcst TL

        StatFcstTL_columns.remove(constants.STAT_FCST_TL)
        ML_decomposition_flag = MLDecompositionFlag[constants.CML_ITERATION_DECOMPOSITION].values[0]
        if ML_decomposition_flag:
            if len(CMLFcstTL) > 0:
                merge_columns = [
                    col for col in StatFcstTL_columns if col != constants.partial_week_key_col
                ]
                StatFcstTL = StatFcstTL.merge(
                    CMLFcstTL,
                    on=merge_columns,
                    how="left",
                )
            else:
                ML_decomposition_flag = False
                logger.warning(
                    "CML Fcst TL empty for this iteration, returning empty CML outputs ..."
                )

        df = StatFcstFinalProfilePL.copy()
        cols_to_merge = [
            constants.version_col,
            constants.planning_region_col,
            constants.transition_item_col,
            constants.planning_pnl_col,
            constants.planning_location_col,
            constants.transition_demand_domain,
            constants.planning_account_col,
            constants.planning_channel_col,
        ]
        TItemDates = PItemMetaData[
            cols_to_merge
            + [
                constants.planning_item_col,
                constants.planning_demand_domain_col,
                constants.INTRO_DATE,
                constants.DISCO_DATE,
            ]
        ]

        # join with stat fcst l1
        df = df.merge(
            StatFcstTL, on=cols_to_merge + [constants.partial_week_col], how="outer", indicator=True
        )
        df = df[
            (
                df[constants.STAT_FCST_FINAL_PROFILE_PL].isnull()
                | (df[constants.STAT_FCST_FINAL_PROFILE_PL] == 0)
            )
        ]
        df = df.drop(columns=[constants.planning_item_col, constants.planning_demand_domain_col])

        # Join with ItemDates to get Intro and Disco Date populated
        df = df.merge(TItemDates, on=cols_to_merge, how="left")
        df[constants.CURRENT_PARTIAL_WEEK_KEY] = CurrentTimePeriod[
            constants.partial_week_key_col
        ].iloc[0]
        if constants.partial_week_key_col in df.columns:
            df = df.drop(columns=[constants.partial_week_key_col], errors="ignore")
        df = df.merge(
            TimeDimension[
                [constants.partial_week_col, constants.partial_week_key_col]
            ].drop_duplicates(),
            on=constants.partial_week_col,
            how="inner",
        )

        req_time_mapping = {
            "week": constants.week_col,
            "month": constants.month_col,
            "planning month": constants.planning_month_col,
            "planning quarter": constants.planning_quarter_col,
            "quarter": constants.quarter_col,
        }
        if forecastgenbucketname.lower() == "week":
            relevant_time_key = constants.week_key_col
            threshold = pd.Timedelta(weeks=1).days
            days = 7
        elif forecastgenbucketname.lower() == "month":
            relevant_time_key = constants.month_key_col
            threshold = pd.Timedelta(days=30)
            days = 28
        elif forecastgenbucketname.lower() == "planning month":
            relevant_time_key = constants.planning_month_key_col
            threshold = pd.Timedelta(days=30)
            days = 28
        elif forecastgenbucketname.lower() == "quarter":
            relevant_time_key = constants.quarter_key_col
            threshold = pd.Timedelta(days=90)
            days = 89
        elif forecastgenbucketname.lower() == "planning quarter":
            relevant_time_key = constants.planning_quarter_key_col
            threshold = pd.Timedelta(days=90)
            days = 89
        else:
            threshold = pd.Timedelta(0)

        relevant_time_name = req_time_mapping[forecastgenbucketname.lower()]

        def evaluate_flags(group):
            # current_date = group[constants.CURRENT_PARTIAL_WEEK_KEY].iloc[0]

            # Ensure current_date and max_forecast_date are also naive
            current_date = group[constants.CURRENT_PARTIAL_WEEK_KEY].iloc[0].tz_localize(None)
            # 1. Intro Date in the Past with no Actuals
            any_disco_in_range = (
                group[constants.DISCO_DATE].dropna().between(current_date, max_forecast_date).any()
                if group[constants.DISCO_DATE].notna().any()
                else False
            )
            others_intro_before_now = (
                (group[constants.INTRO_DATE].dropna() < current_date).all()
                if group[constants.INTRO_DATE].notna().any()
                else False
            )

            if any_disco_in_range and others_intro_before_now:
                group[constants.PL_VOLUME_LOSS_FLAG] = "Intro Date in the Past with no Actuals"
                return group

            # 2. All discontinued items
            all_disco_in_range = group[constants.DISCO_DATE].between(
                current_date, max_forecast_date
            ).all() and (
                group[constants.INTRO_DATE].isna().all()  # No intro dates at all
                or (
                    group[constants.INTRO_DATE] <= group[constants.DISCO_DATE]
                ).all()  # All intro <= disco
            )

            if all_disco_in_range:
                group[constants.PL_VOLUME_LOSS_FLAG] = "All discontinued Items"
                return group

            # Only keep one row per Transition Item to check intro/disco gap — take max intro and min disco
            planning_summary = (
                group.groupby(constants.planning_item_col)[
                    [constants.INTRO_DATE, constants.DISCO_DATE]
                ]
                .agg({constants.INTRO_DATE: "min", constants.DISCO_DATE: "max"})
                .sort_values(by=constants.INTRO_DATE)
                .reset_index()
            )

            if len(planning_summary) == 1:
                intro_date = pd.to_datetime(group[constants.INTRO_DATE].max())
                disco_date = pd.to_datetime(group[constants.DISCO_DATE].min())
                if pd.notna(intro_date) and pd.notna(disco_date):
                    if intro_date >= disco_date:
                        group[constants.PL_VOLUME_LOSS_FLAG] = constants.transition_period_gap
                        return group

            else:
                # Compare disco of current with intro of next
                for i in range(len(planning_summary) - 1):
                    curr_disco = planning_summary.loc[i, constants.DISCO_DATE]
                    next_intro = planning_summary.loc[i + 1, constants.INTRO_DATE]

                    if pd.isna(curr_disco):
                        curr_disco = pd.Timestamp("2099-12-31")
                    if pd.isna(next_intro):
                        next_intro = pd.Timestamp("2000-01-01")

                    if pd.notna(curr_disco) and pd.notna(next_intro):
                        if isinstance(threshold, pd.Timedelta):
                            if abs(next_intro - curr_disco) > threshold:
                                group[constants.PL_VOLUME_LOSS_FLAG] = (
                                    constants.transition_period_gap
                                )
                                return group
                        else:
                            if abs((next_intro - curr_disco).days) > threshold:
                                group[constants.PL_VOLUME_LOSS_FLAG] = (
                                    constants.transition_period_gap
                                )
                                return group

            # 4. Zero in profile (aggregated)
            if (
                group[constants.STAT_FCST_FINAL_PROFILE_PL].sum(skipna=True) == 0
                and not group[constants.STAT_FCST_FINAL_PROFILE_PL].isna().all()
            ):
                group[constants.PL_VOLUME_LOSS_FLAG] = "Zero in profile"
                return group

            # 5. Unknown (already set by default)
            group[constants.PL_VOLUME_LOSS_FLAG] = "Unknown"
            return group

        # df = df[df[o9Constants.TRANSITION_ITEM]=='A141_12']
        df[constants.INTRO_DATE].fillna(default_intro_date, inplace=True)
        df[constants.DISCO_DATE].fillna(default_disc_date, inplace=True)
        grouped = df.groupby(constants.transition_item_col)

        results = []  # to store processed groups

        for _, group in grouped:
            processed_group = evaluate_flags(group)
            results.append(processed_group)

        # Concatenate all the processed groups
        if results:
            final_df = pd.concat(results, ignore_index=True)
            final_df.reset_index(drop=True, inplace=True)
            Output_volume_loss_flag = final_df[cols_required_in_volume_loss_flag_output]
            Output_volume_loss_flag = Output_volume_loss_flag.drop_duplicates()
            Output_volume_loss_flag = Output_volume_loss_flag.dropna(
                subset=[constants.planning_item_col]
            )

        else:
            logger.warning(f"No volume loss for iteration {the_iteration}")
            Output_volume_loss_flag = pd.DataFrame(columns=cols_required_in_volume_loss_flag_output)

        StatFcstTL_columns.remove(constants.partial_week_key_col)
        logger.info("PL VOLUME LOSS FLAG calculation done ...")

        # join with StatFcstFinalProfilePL with relevant columns from StatFcstTL
        StatFcstFinalProfilePL = StatFcstFinalProfilePL.merge(
            StatFcstTL,
            on=StatFcstTL_columns,
            how="inner",
        )

        logger.info("Merged StatFcstFinalProfilePL with StatFcstTL ...")

        # checking dataframe is empty after merging with StatFcstTL

        if StatFcstFinalProfilePL.empty:
            logger.warning(
                f"StatFcstFinalProfilePL is empty after merging with StatFcstTL for {df_keys}"
            )
            return Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag

        # Output = StatFcstFinalProfilePL[StatFcstTL_columns].drop_duplicates()

        # get planning items count under transition item
        StatFcstFinalProfilePL["count"] = StatFcstFinalProfilePL.groupby(
            cols_to_merge,
            observed=True,
        )[constants.planning_item_col].transform("nunique")

        # create group sum - at TL level
        StatFcstFinalProfilePL[constants.GROUP_SUM] = StatFcstFinalProfilePL.groupby(
            StatFcstTL_columns + [constants.partial_week_col]
        )[constants.STAT_FCST_FINAL_PROFILE_PL].transform("sum")

        eol_profiles = pd.DataFrame(
            columns=list(StatFcstFinalProfilePL.columns) + [constants.DISAGG_PROPORTION]
        )
        if not PItemMetaData.empty:
            PItemMetaData = PItemMetaData[PItemMetaData[constants.transition_type] == "EOL"]
            if not PItemMetaData.empty:
                PItemMetaData[constants.prod_transition_start_date] = PItemMetaData[
                    constants.prod_transition_start_date
                ].dt.tz_localize(None)

                PItemMetaData.sort_values(by=constants.DISCO_DATE, inplace=True)
                PItemMetaData = (
                    pd.merge_asof(
                        PItemMetaData,
                        TimeDimension[[constants.day_key_col, relevant_time_key]],
                        left_on=constants.DISCO_DATE,
                        right_on=constants.day_key_col,
                        # direction="forward",
                    )
                    .drop(columns=[constants.day_key_col])
                    .rename(columns={relevant_time_key: "End Date"})
                )
                PItemMetaData[constants.phase_out_buckets].fillna(0, inplace=True)
                PItemMetaData[constants.day_key_col] = PItemMetaData["End Date"] - pd.to_timedelta(
                    PItemMetaData[constants.phase_out_buckets] * days, unit="D"
                )
                PItemMetaData = (
                    PItemMetaData.merge(
                        TimeDimension[[constants.day_key_col, relevant_time_key]],
                    )
                    .drop(columns=[constants.day_key_col])
                    .rename(columns={relevant_time_key: "Start Date"})
                )

                eol_profiles_list = Parallel(n_jobs=1, verbose=1)(
                    delayed(get_eol_profiles)(
                        df=df,
                        transition_data=StatFcstFinalProfilePL,
                        dimensions=cols_to_merge,
                        DefaultProfiles=DefaultProfiles,
                        relevant_time_name=relevant_time_name,
                        relevant_time_key=relevant_time_key,
                        TimeDimension=TimeDimension,
                    )
                    for _, df in PItemMetaData.groupby(
                        cols_to_merge + [constants.planning_item_col]
                    )
                )
                eol_profiles = concat_to_dataframe(eol_profiles_list)
                if eol_profiles.empty:
                    eol_profiles = pd.DataFrame(
                        columns=list(StatFcstFinalProfilePL.columns) + [constants.DISAGG_PROPORTION]
                    )

        eol_intersections = eol_profiles[dimensions + [constants.partial_week_col]]
        StatFcstFinalProfilePL = StatFcstFinalProfilePL.merge(
            eol_intersections,
            how="left",
            indicator=True,
        )
        StatFcstFinalProfilePL = StatFcstFinalProfilePL[
            StatFcstFinalProfilePL["_merge"] == "left_only"
        ].drop(columns=["_merge"])
        # create proportions
        StatFcstFinalProfilePL[constants.DISAGG_PROPORTION] = (
            StatFcstFinalProfilePL[constants.STAT_FCST_FINAL_PROFILE_PL]
            / StatFcstFinalProfilePL[constants.GROUP_SUM]
        )
        StatFcstFinalProfilePL = pd.concat(
            [StatFcstFinalProfilePL, eol_profiles[StatFcstFinalProfilePL.columns]]
        )

        # multiply value with proportion
        StatFcstFinalProfilePL[constants.STAT_FCST_PL] = (
            StatFcstFinalProfilePL[constants.STAT_FCST_TL]
            * StatFcstFinalProfilePL[constants.DISAGG_PROPORTION]
        )
        # calculating Fcst TL for all CML drivers, Stat Fcst TL = Stat Fcst TL * Disagg Proportion
        if ML_decomposition_flag:
            fields = [
                (constants.STAT_FCST_PL_CML_RESIDUAL, constants.STAT_FCST_TL_CML_RESIDUAL),
                (constants.STAT_FCST_PL_CML_HOLIDAY, constants.STAT_FCST_TL_CML_HOLIDAY),
                (constants.STAT_FCST_PL_CML_PROMO, constants.STAT_FCST_TL_CML_PROMO),
                (constants.STAT_FCST_PL_CML_MARKETING, constants.STAT_FCST_TL_CML_MARKETING),
                (constants.STAT_FCST_PL_CML_PRICE, constants.STAT_FCST_TL_CML_PRICE),
                (constants.STAT_FCST_PL_CML_BASELINE, constants.STAT_FCST_TL_CML_BASELINE),
                (constants.STAT_FCST_PL_CML_WEATHER, constants.STAT_FCST_TL_CML_WEATHER),
                (
                    constants.STAT_FCST_PL_CML_EXTERNAL_DRIVER,
                    constants.STAT_FCST_TL_CML_EXTERNAL_DRIVER,
                ),
            ]

            for pl_field, tl_field in fields:
                StatFcstFinalProfilePL[pl_field] = (
                    StatFcstFinalProfilePL[tl_field]
                    * StatFcstFinalProfilePL[constants.DISAGG_PROPORTION]
                )
            CMLOutput = StatFcstFinalProfilePL[cols_required_in_cml_output]
            CMLOutput = (
                CMLOutput.groupby([constants.version_col, constants.partial_week_col] + dimensions)
                .agg(
                    {
                        constants.STAT_FCST_PL_CML_BASELINE: sum,
                        constants.STAT_FCST_PL_CML_RESIDUAL: sum,
                        constants.STAT_FCST_PL_CML_HOLIDAY: sum,
                        constants.STAT_FCST_PL_CML_PROMO: sum,
                        constants.STAT_FCST_PL_CML_MARKETING: sum,
                        constants.STAT_FCST_PL_CML_PRICE: sum,
                        constants.STAT_FCST_PL_CML_WEATHER: sum,
                        constants.STAT_FCST_PL_CML_EXTERNAL_DRIVER: sum,
                    }
                )
                .reset_index()
            )
            CMLOutput = CMLOutput[cols_required_in_cml_output].drop_duplicates()
        # select relevant columns
        Output = StatFcstFinalProfilePL[cols_required_in_output]

        current_month = CurrentTimePeriod[constants.month_col][0]
        n_bucket_months = get_n_time_periods(
            current_month,
            int(NBucketsinMonths),
            TimeDimension[[constants.month_col, constants.month_key_col]].drop_duplicates(),
            {constants.month_col: constants.month_key_col},
            include_latest_value=True,
        )
        relevant_partial_weeks = TimeDimension[
            TimeDimension[constants.month_col].isin(n_bucket_months)
        ][constants.partial_week_col]

        FcstNextNBuckets = Output[Output[constants.partial_week_col].isin(relevant_partial_weeks)]
        FcstNextNBuckets = FcstNextNBuckets.groupby(
            [constants.version_col] + dimensions, as_index=False
        )[constants.STAT_FCST_PL].sum()
        FcstNextNBuckets.rename(
            columns={constants.STAT_FCST_PL: constants.FCST_NEXT_N_BUCKETS_PL}, inplace=True
        )
        FcstNextNBuckets = FcstNextNBuckets[cols_required_in_fcst_next_n_buckets_pl]

        Output_volume_loss_flag = Output_volume_loss_flag.merge(
            FcstNextNBuckets,
            on=[constants.version_col] + dimensions,
            how="outer",
        )
        Output_volume_loss_flag = Output_volume_loss_flag[
            cols_required_in_volume_loss_flag_output + [constants.FCST_NEXT_N_BUCKETS_PL]
        ]

        if len(StatFcstPLLC) > 0:
            StatFcstPLLC = StatFcstPLLC[
                StatFcstPLLC[constants.partial_week_col].isin(relevant_partial_weeks)
            ]
            StatFcstPLvsLC_COCC = Output.merge(
                StatFcstPLLC,
                on=[constants.version_col, constants.partial_week_col] + dimensions,
                how="inner",
            )
            StatFcstPLvsLC_COCC = StatFcstPLvsLC_COCC.merge(
                TimeDimension[[constants.partial_week_col, relevant_time_name]].drop_duplicates(),
                on=constants.partial_week_col,
                how="inner",
            )
            StatFcstPLvsLC_COCC = StatFcstPLvsLC_COCC.groupby(
                [constants.version_col, relevant_time_name] + dimensions, as_index=False
            ).agg({constants.STAT_FCST_PL: "sum", constants.STAT_FCST_PL_LC: "sum"})
            StatFcstPLvsLC_COCC[constants.STAT_FCST_PL_vs_LC_COCC] = abs(
                StatFcstPLvsLC_COCC[constants.STAT_FCST_PL]
                - StatFcstPLvsLC_COCC[constants.STAT_FCST_PL_LC]
            )

            StatBucketWeight = StatBucketWeight.merge(
                TimeDimension[[constants.partial_week_col, relevant_time_name]].drop_duplicates(),
                on=constants.partial_week_col,
                how="inner",
            )

            StatFcstPLvsLC_COCC = disaggregate_data(
                source_df=StatFcstPLvsLC_COCC,
                source_grain=relevant_time_name,
                target_grain=constants.partial_week_col,
                profile_df=StatBucketWeight.drop(constants.version_col, axis=1),
                profile_col=constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[constants.STAT_FCST_PL_vs_LC_COCC],
            )
            # StatFcstPLvsLC_COCC = StatFcstPLvsLC_COCC.rename(
            #     columns={constants.STAT_FCST_PL_LC: constants.FCST_NEXT_N_BUCKETS_PL}
            # )
            StatFcstPLvsLC_COCC = StatFcstPLvsLC_COCC[cols_required_in_cocc_output]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
        CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
        Output_volume_loss_flag = pd.DataFrame(columns=cols_required_in_volume_loss_flag_output)
    return Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag

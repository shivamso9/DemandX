import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
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
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from scipy.interpolate import interp1d

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

col_mapping = {"Stat Fcst Final Profile PL": float}

logger = logging.getLogger("o9_logger")


class constants:
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
    TRANSITION_FLAG = "510 Product Transition.[Transition Flag]"

    from_pl_item = "from.[Item].[Planning Item]"
    to_pl_item = "to.[Item].[Planning Item]"
    from_pl_pnl = "from.[PnL].[Planning PnL]"
    from_pl_channel = "from.[Channel].[Planning Channel]"
    from_pl_account = "from.[Account].[Planning Account]"
    from_pl_region = "from.[Region].[Planning Region]"
    from_pl_location = "from.[Location].[Planning Location]"
    from_pl_demand_domain = "from.[Demand Domain].[Planning Demand Domain]"

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

    fcst_start_date_col = "Fcst Start Date"
    fcst_end_date_col = "Fcst End Date"
    pitem_intro_date_col = "Intro Date"
    pitem_disc_date_col = "Disco Date"
    key_to_check = "Time Bucket Key of Intro Date"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    stat_bucket_weight_col = "Stat Bucket Weight"
    forecast_period_col = "Forecast Period"
    sell_out_offset_col = "Offset Period"
    actual_last_n_buckets_col = "Actual Last N Buckets PL"
    phase_out_profile = "Phase Out Profile"
    phase_out_buckets = "Number of Phase Out Buckets"
    adjust_phase_out_profile = "Adjust Phase Out Profile"
    phase_in_split_per = "Phase In Split %"
    prod_transition_start_date = "Product Transition Overlap Start Date"
    default_profile = "Default Profile"
    OVERLAP_FLAG = "Overlap Flag"
    AVAILABLE_PROFILE = "Available Profile"
    LIFECYCLE_FLAG = "Lifecycle Flag"
    transition_type = "Transition Type"
    GROUP_SUM = "Transition Group Profile Sum"
    start_date = "Start Date"
    end_date = "End Date"
    time_sequence = "Time Sequence"
    actual_sum = "ActualSum"


def get_eol_profiles(
    df,
    transition_data,
    SalesDomainGrains,
    LocationLevel,
    history_measure,
    DefaultProfiles,
    relevant_time_name,
    relevant_time_key,
    TimeDimension,
):
    df[constants.phase_out_profile].fillna("Default Gradual", inplace=True)
    df = df.merge(
        DefaultProfiles,
        how="left",
        left_on=[o9Constants.VERSION_NAME, constants.phase_out_profile],
        right_on=[o9Constants.VERSION_NAME, constants.plc_profile_col],
    )
    adjust_profile = bool(df[constants.adjust_phase_out_profile].iloc[0])
    start_date = df[constants.start_date].iloc[0]
    end_date = df[constants.end_date].iloc[0]
    method = df[constants.phase_out_profile].iloc[0]
    filtered_df = df[
        SalesDomainGrains
        + [LocationLevel, constants.transition_item_col, constants.planning_item_col]
    ].drop_duplicates()
    filtered_transition = transition_data.merge(filtered_df)

    if filtered_transition.empty:
        logger.warning(f"No transition available for group: {filtered_df}")
        return pd.DataFrame()

    relevant_time_dim = TimeDimension[
        (TimeDimension[relevant_time_key] > start_date)
        & (TimeDimension[relevant_time_key] <= end_date)
    ][[relevant_time_name, relevant_time_key]].drop_duplicates()

    filtered_transition = filtered_transition.merge(relevant_time_dim)
    if filtered_transition.empty:
        logger.warning(
            f"No transition available for group in specified time periods: {filtered_df}"
        )
        return pd.DataFrame()

    periods = len(relevant_time_dim)

    if method == "Default Gradual":
        step = 1 / (periods + 1)
        profile_values = [1 - step * i for i in range(1, periods + 1)]
        relevant_time_dim[constants.AVAILABLE_PROFILE] = profile_values
        filtered_transition = filtered_transition.merge(relevant_time_dim)
        relevant_time_dim.drop(columns=[constants.AVAILABLE_PROFILE], inplace=True)

    else:
        if adjust_profile:
            df = interpolate_profile_to_df(
                df=df,
                target_bucket_count=periods,
                method="cubic",
            )
        relevant_time_dim[constants.time_sequence] = (
            relevant_time_dim[relevant_time_key].rank().astype("int")
        )
        df[constants.time_sequence] = df[constants.lifecycle_bucket_col].apply(lambda x: int(x[1:]))

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
                    constants.time_sequence,
                    constants.default_profile,
                ]
            ].drop_duplicates(),
            how="left",
        )
        relevant_time_dim.rename(
            columns={constants.default_profile: constants.AVAILABLE_PROFILE}, inplace=True
        )
        relevant_time_dim.sort_values(
            by=[constants.planning_item_col, relevant_time_key], inplace=True
        )
        relevant_time_dim[constants.AVAILABLE_PROFILE] = relevant_time_dim[
            constants.AVAILABLE_PROFILE
        ].ffill()
        relevant_time_dim[constants.plc_profile_col] = relevant_time_dim[
            constants.plc_profile_col
        ].ffill()

        filtered_transition = filtered_transition.merge(
            relevant_time_dim,
        )
        filtered_transition.drop(
            columns=[constants.time_sequence, constants.plc_profile_col], inplace=True
        )
        relevant_time_dim.drop(
            columns=[
                constants.planning_item_col,
                constants.plc_profile_col,
                constants.time_sequence,
                constants.AVAILABLE_PROFILE,
            ],
            inplace=True,
        )

    filtered_transition[history_measure] = (
        filtered_transition[history_measure] * filtered_transition[constants.AVAILABLE_PROFILE]
    )
    return filtered_transition


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


def check_all_dimensions(df, grains, default_mapping):
    # checks if all 7 dim are present, if not, adds the dimension with member "All"
    # Renames input stream to Actual as well
    df_copy = df.copy()
    dims = {}
    for x in grains:
        dims[x] = x.strip().split(".")[0]
    try:
        for i in dims:
            if i not in df_copy.columns:
                if dims[i] in default_mapping:
                    df_copy[i] = default_mapping[dims[i]]
                else:
                    logger.warning(
                        f'Column {i} not found in default_mapping dictionary, adding the member "All"'
                    )
                    df_copy[i] = "All"
    except Exception as e:
        logger.exception(f"Error in check_all_dimensions\nError:-{e}")
        return df
    return df_copy


def get_profiles_in_overlap_period(
    df,
    PItemPhaseData,
    transition_data,
    SalesDomainGrains,
    LocationLevel,
    DefaultProfiles,
    relevant_time_name,
    relevant_time_key,
    history_measure,
):
    df_filtered = df[SalesDomainGrains + [LocationLevel]].drop_duplicates()
    # filter relevant intersections
    PItemPhaseData = PItemPhaseData.merge(df_filtered)

    from_items = list(df[constants.from_pl_item].unique())
    to_items = list(df[constants.to_pl_item].unique())

    # overlap periods and active intersections are required
    transition_data = transition_data[transition_data[constants.LIFECYCLE_FLAG] == 1]
    transition_data = transition_data[transition_data[constants.OVERLAP_FLAG] == 1]

    from_item_dataframe = pd.DataFrame(columns=[constants.planning_item_col], data=from_items)
    to_item_dataframe = pd.DataFrame(columns=[constants.planning_item_col], data=to_items)
    items_dataframe = pd.concat([from_item_dataframe, to_item_dataframe])

    from_PItemPhaseData = PItemPhaseData.merge(from_item_dataframe)
    to_PItemPhaseData = PItemPhaseData.merge(to_item_dataframe)

    # check if all from items have part revision as transition type
    from_PItemPhaseData = from_PItemPhaseData[
        from_PItemPhaseData[constants.transition_type] == "Part Revision"
    ]
    if from_PItemPhaseData.empty:
        logger.warning(f"No intersections under part revision for group: {df_filtered}")
        return pd.DataFrame()

    if to_PItemPhaseData[constants.phase_in_split_per].isna().all():
        to_PItemPhaseData[constants.phase_in_split_per] = 1 / len(to_items)
    else:
        to_PItemPhaseData[constants.phase_in_split_per].fillna(0, inplace=True)
        to_PItemPhaseData[constants.phase_in_split_per] = (
            to_PItemPhaseData[constants.phase_in_split_per]
            / to_PItemPhaseData[constants.phase_in_split_per].sum()
        )

    to_PItemPhaseData.rename(
        columns={constants.phase_in_split_per: constants.AVAILABLE_PROFILE}, inplace=True
    )
    from_PItemPhaseData[constants.phase_out_profile] = from_PItemPhaseData[
        constants.phase_out_profile
    ].fillna("Default Gradual")
    from_PItemPhaseData = from_PItemPhaseData.merge(
        DefaultProfiles,
        how="left",
        left_on=[o9Constants.VERSION_NAME, constants.phase_out_profile],
        right_on=[o9Constants.VERSION_NAME, constants.plc_profile_col],
    )

    transition_data = transition_data.merge(items_dataframe)
    transition_data = transition_data.merge(df_filtered)

    from_transition_overlap_data = transition_data.merge(from_item_dataframe)
    to_transition_overlap_data = transition_data.merge(to_item_dataframe)

    check_flag = (
        ~from_transition_overlap_data[history_measure].isna().all()
        & ~to_transition_overlap_data[history_measure].isna().all()
    )
    # find overlap time and periods
    overlap_time_period = transition_data[transition_data[constants.OVERLAP_FLAG] == 1][
        [constants.planning_item_col, relevant_time_name, relevant_time_key]
    ].drop_duplicates()
    if check_flag:
        start_date = from_PItemPhaseData[constants.prod_transition_start_date].iloc[0]
        if pd.isna(start_date):
            start_date = from_PItemPhaseData[constants.pitem_disc_date_col].iloc[
                0
            ] - pd.to_timedelta(180, unit="D")
        overlap_time_period = overlap_time_period[
            overlap_time_period[relevant_time_key] >= start_date
        ]
        from_transition_overlap_data = from_transition_overlap_data.merge(
            overlap_time_period, how="left"
        )

    to_transition_overlap_data = to_transition_overlap_data.merge(overlap_time_period, how="left")
    to_overlap_period = to_transition_overlap_data[
        [relevant_time_name, relevant_time_key]
    ].drop_duplicates()
    overlap_time_period = overlap_time_period.merge(from_item_dataframe)
    overlap_time_period.drop(columns=[constants.planning_item_col], inplace=True)
    overlap_time_period = overlap_time_period.merge(to_overlap_period).drop_duplicates()
    overlap_time_period.sort_values(by=relevant_time_key, inplace=True)
    overlap_periods = len(overlap_time_period)

    from_intersections_initial_actual_sum = (
        from_transition_overlap_data.groupby([relevant_time_name], observed=True)
        .agg(ActualSum=(history_measure, "sum"))
        .reset_index()
    )

    from_transition_overlap_data_list = []
    for name, group in from_PItemPhaseData.groupby(
        [constants.phase_out_profile, constants.adjust_phase_out_profile]
    ):
        method = str(name[0])
        adjust_profile = bool(name[1])
        relevant_cols = SalesDomainGrains + [LocationLevel, constants.planning_item_col]
        relevant_from_transition_overlap_data = from_transition_overlap_data.merge(
            group[relevant_cols].drop_duplicates()
        )
        relevant_from_transition_overlap_data.sort_values(
            by=relevant_cols + [relevant_time_key], inplace=True
        )
        if method == "Default Equal":
            relevant_from_transition_overlap_data = relevant_from_transition_overlap_data.merge(
                overlap_time_period, how="left", indicator=True
            )
            relevant_from_transition_overlap_data[constants.AVAILABLE_PROFILE] = np.where(
                relevant_from_transition_overlap_data["_merge"] == "both",
                0.5,
                np.nan,
            )
            relevant_from_transition_overlap_data.drop(columns="_merge", inplace=True)
        elif method == "Default Gradual":
            step = 1 / (overlap_periods + 1)
            profile_values = [1 - step * i for i in range(1, overlap_periods + 1)]
            overlap_time_period[constants.AVAILABLE_PROFILE] = profile_values
            relevant_from_transition_overlap_data = relevant_from_transition_overlap_data.merge(
                overlap_time_period, how="left"
            )
            overlap_time_period.drop(columns=[constants.AVAILABLE_PROFILE], inplace=True)
        else:
            if DefaultProfiles.empty:
                relevant_from_transition_overlap_data[constants.AVAILABLE_PROFILE] = np.nan
            else:
                if adjust_profile:
                    group = interpolate_profile_to_df(
                        df=group,
                        target_bucket_count=overlap_periods,
                        method="cubic",
                    )
                overlap_time_period[constants.time_sequence] = (
                    overlap_time_period[relevant_time_key].rank().astype("int")
                )
                group[constants.time_sequence] = group[constants.lifecycle_bucket_col].apply(
                    lambda x: int(x[1:])
                )

                overlap_time_period = overlap_time_period.merge(
                    relevant_from_transition_overlap_data[
                        [constants.planning_item_col, relevant_time_name]
                    ].drop_duplicates(),
                )

                overlap_time_period = overlap_time_period.merge(
                    group[
                        [
                            constants.planning_item_col,
                            constants.plc_profile_col,
                            constants.time_sequence,
                            constants.default_profile,
                        ]
                    ].drop_duplicates(),
                    how="left",
                )
                overlap_time_period.rename(
                    columns={constants.default_profile: constants.AVAILABLE_PROFILE}, inplace=True
                )
                overlap_time_period.sort_values(
                    by=[constants.planning_item_col, relevant_time_key], inplace=True
                )
                overlap_time_period[constants.AVAILABLE_PROFILE] = overlap_time_period[
                    constants.AVAILABLE_PROFILE
                ].ffill()
                overlap_time_period[constants.plc_profile_col] = overlap_time_period[
                    constants.plc_profile_col
                ].ffill()

                relevant_from_transition_overlap_data = relevant_from_transition_overlap_data.merge(
                    overlap_time_period,
                    how="left",
                )
                relevant_from_transition_overlap_data.drop(
                    columns=[constants.time_sequence, constants.plc_profile_col], inplace=True
                )
                overlap_time_period.drop(
                    columns=[
                        constants.planning_item_col,
                        constants.plc_profile_col,
                        constants.time_sequence,
                        constants.AVAILABLE_PROFILE,
                    ],
                    inplace=True,
                )

        from_transition_overlap_data_list.append(relevant_from_transition_overlap_data)

    from_transition_overlap_data = concat_to_dataframe(from_transition_overlap_data_list)
    from_transition_overlap_data[constants.AVAILABLE_PROFILE].fillna(1, inplace=True)
    from_transition_overlap_data[history_measure] = (
        from_transition_overlap_data[history_measure]
        * from_transition_overlap_data[constants.AVAILABLE_PROFILE]
    )

    from_intersections_actual_sum = (
        from_transition_overlap_data.groupby([relevant_time_name], observed=True)
        .agg({history_measure: "sum"})
        .reset_index()
    )

    from_intersections_initial_actual_sum = from_intersections_initial_actual_sum.merge(
        from_intersections_actual_sum
    )
    from_intersections_initial_actual_sum[constants.actual_sum] = (
        from_intersections_initial_actual_sum[constants.actual_sum]
        - from_intersections_initial_actual_sum[history_measure]
    )

    to_transition_overlap_data = to_transition_overlap_data.merge(
        to_PItemPhaseData[[constants.planning_item_col, constants.AVAILABLE_PROFILE]]
    )

    to_transition_overlap_data = to_transition_overlap_data.merge(
        from_intersections_initial_actual_sum[[relevant_time_name, constants.actual_sum]],
        how="left",
    )
    to_transition_overlap_data[constants.actual_sum].fillna(
        to_transition_overlap_data[constants.GROUP_SUM], inplace=True
    )
    to_transition_overlap_data[history_measure] = to_transition_overlap_data[
        history_measure
    ].fillna(0) + (
        to_transition_overlap_data[constants.actual_sum]
        * to_transition_overlap_data[constants.AVAILABLE_PROFILE]
    )
    to_transition_overlap_data.drop(columns=[constants.actual_sum], inplace=True)

    from_to_transaction_data = pd.concat([from_transition_overlap_data, to_transition_overlap_data])
    from_to_transaction_data[constants.transition_type] = "Part Revision"

    return from_to_transaction_data  # , items_dataframe


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    LocationLevel,
    ItemLevel,
    Actual,
    TimeDimension,
    PItemDates,
    Parameters,
    SalesDomainLevel,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ReadFromHive,
    OutputMeasure,
    HistoryPeriodsInWeeks,
    MultiprocessingNumCores,
    df_keys,
    TransitionFlag=None,
    DefaultProfiles=None,
    ForecastIterationMasterData=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    NBucketsinMonths="12",
    default_mapping={},
):
    try:
        if TransitionFlag is None:
            TransitionFlag = pd.DataFrame()

        OutputList = list()
        ActualLastNBucketsList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output, the_actuals = decorated_func(
                LocationLevel=LocationLevel,
                ItemLevel=ItemLevel,
                Actual=Actual,
                TimeDimension=TimeDimension,
                PItemDates=PItemDates,
                Parameters=Parameters,
                SalesDomainLevel=SalesDomainLevel,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                ReadFromHive=ReadFromHive,
                OutputMeasure=OutputMeasure,
                HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
                multiprocessing_cores=MultiprocessingNumCores,
                df_keys=df_keys,
                TransitionFlag=TransitionFlag,
                DefaultProfiles=DefaultProfiles,
                ForecastIterationMasterData=ForecastIterationMasterData,
                SellOutOffset=SellOutOffset,
                NBucketsinMonths=NBucketsinMonths,
                default_mapping=default_mapping,
                the_iteration=the_iteration,
            )

            OutputList.append(the_output)
            ActualLastNBucketsList.append(the_actuals)

        PLProfile = concat_to_dataframe(OutputList)
        ActualLastNBucketsPL = concat_to_dataframe(ActualLastNBucketsList)
    except Exception as e:
        logger.exception(e)
        PLProfile = None
        ActualLastNBucketsPL = None
    return PLProfile, ActualLastNBucketsPL


def processIteration(
    LocationLevel,
    ItemLevel,
    Actual,
    TimeDimension,
    PItemDates,
    Parameters,
    SalesDomainLevel,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    StatBucketWeight,
    HistoryPeriodsInWeeks,
    ReadFromHive="False",
    OutputMeasure="Stat Fcst Final Profile PL",
    multiprocessing_cores=1,
    df_keys={},
    TransitionFlag=None,
    DefaultProfiles=None,
    ForecastIterationMasterData=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    NBucketsinMonths="12",
    default_mapping={},
    the_iteration="",
):
    plugin_name = "DP028PlanningLevelProfile"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    SalesDomainGrains = [str(x) for x in SalesDomainLevel.split(",") if x != "NA" and x != ""]

    # combine grains to get segmentation level
    all_grains = [
        LocationLevel,
        ItemLevel,
        constants.transition_item_col,
        constants.transition_demand_domain,
    ] + SalesDomainGrains

    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    cols_required_in_output = (
        [constants.version_col]
        + SalesDomainGrains
        + [ItemLevel, LocationLevel]
        + [constants.partial_week_col, OutputMeasure]
    )
    cols_req_in_actual_lastn_buckets = (
        [constants.version_col]
        + SalesDomainGrains
        + [ItemLevel, LocationLevel]
        + [constants.partial_week_col, constants.actual_last_n_buckets_col]
    )
    PLProfile = pd.DataFrame(columns=cols_required_in_output)
    ActualLastNBucketsPL = pd.DataFrame(columns=cols_req_in_actual_lastn_buckets)
    try:
        input_stream = None
        if not ForecastIterationMasterData.empty:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]

        if input_stream is None:
            logger.warning("Empty input stream, returning empty output")
            return PLProfile, ActualLastNBucketsPL

        history_measure = input_stream
        if isinstance(Actual, dict):
            df = next(
                (df for _, df in Actual.items() if input_stream in df.columns),
                None,
            )
            if df is None:
                logger.warning(
                    f"Input Stream '{input_stream}' not found, returning empty dataframe..."
                )
                return PLProfile, ActualLastNBucketsPL
            Actual_df = df
        else:
            Actual_df = Actual

        history_dims = [col.split(".")[0] for col in Actual_df]
        target_dims = [dim.split(".")[0] for dim in dimensions]
        missing_dims = list(set(target_dims) - set(history_dims))
        missing_grains = list(set(dimensions) - set(Actual_df.columns))
        if Actual_df.empty:
            logger.warning("Actual is empty, returning empty dataframe ...")
            return PLProfile, ActualLastNBucketsPL

        if Parameters.empty:
            logger.warning("Parameters empty, returning empty dataframe ...")
            return PLProfile, ActualLastNBucketsPL

        if TransitionFlag is None:
            TransitionFlag = pd.DataFrame()

        if len(SellOutOffset) == 0:
            logger.warning(
                f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    constants.sell_out_offset_col: [0],
                }
            )

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        agg_history_measure_col = "Agg History Measure"

        if ReadFromHive:
            history_measure = "DP028Actual"
        else:
            history_measure = input_stream

        logger.info("history_measure : {}".format(history_measure))

        forecast_periods = int(Parameters[constants.forecast_period_col].unique()[0])
        logger.info("forecast_periods : {}".format(forecast_periods))

        time_key_cols = [
            constants.day_key_col,
            constants.week_key_col,
            constants.partial_week_key_col,
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

        default_intro_date = TimeDimension[constants.day_key_col].min()
        default_disc_date = TimeDimension[constants.day_key_col].max()

        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return PLProfile, ActualLastNBucketsPL

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[constants.fcst_gen_time_bucket_col].unique()[0]
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

            days = 7
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.planning_month_col,
                constants.planning_month_key_col,
            ]
            relevant_time_name = constants.planning_month_col
            relevant_time_key = constants.planning_month_key_col

            days = 28
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.month_col,
                constants.month_key_col,
            ]
            relevant_time_name = constants.month_col
            relevant_time_key = constants.month_key_col

            days = 28
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.planning_quarter_col,
                constants.planning_quarter_key_col,
            ]
            relevant_time_name = constants.planning_quarter_col
            relevant_time_key = constants.planning_quarter_key_col

            days = 89
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                constants.partial_week_col,
                constants.quarter_col,
                constants.quarter_key_col,
            ]
            relevant_time_name = constants.quarter_col
            relevant_time_key = constants.quarter_key_col

            days = 89
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return PLProfile, ActualLastNBucketsPL

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # if the entire dimension is missing from the input data, we fill the dim with the default values
        missing_dim_grains = []
        if len(missing_dims) > 0:
            for dim in missing_dims:
                missing_dim_grains += [col for col in missing_grains if col.split(".")[0] == dim]
        if len(missing_dim_grains) > 0:
            Actual_df = check_all_dimensions(
                df=Actual_df,
                grains=missing_dim_grains,
                default_mapping=default_mapping,
            )
        missing_grains = list(set(missing_grains) - set(missing_dim_grains))

        if len(missing_grains) > 0:
            logger.warning(
                f"Dimensions {missing_grains} missing in the {input_stream} input query, please add the grains and try again"
            )
            return PLProfile, ActualLastNBucketsPL

        Actuals_PW = Actual_df.copy()
        # Join Actuals with time mapping
        Actual_df = Actual_df.merge(base_time_mapping, on=constants.partial_week_col, how="inner")

        # select the relevant columns, groupby and sum history measure
        Actual_df = (
            Actual_df.groupby(
                dimensions + [relevant_time_name],
                observed=True,
            )
            .sum()[[history_measure]]
            .reset_index()
        )

        name_to_key_mapping = dict(
            zip(
                list(relevant_time_mapping[relevant_time_name]),
                list(relevant_time_mapping[relevant_time_key]),
            )
        )
        time_attribute_dict = {relevant_time_name: relevant_time_key}
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # adjust the latest time according to the forecast iteration's offset before getting n periods for considering history
        offset_periods = int(SellOutOffset[constants.sell_out_offset_col].values[0])
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

        input_version = Parameters[constants.version_col].iloc[0]

        forecast_dates = get_n_time_periods(
            latest_time_name,
            forecast_periods + offset_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        HistoryPeriods = HistoryPeriodsInWeeks.split(",")
        HistoryPeriods = sorted(int(x) for x in HistoryPeriods)

        if len(HistoryPeriods) == 0:
            logger.warning("HistoryPeriods is empty, check HistoryPeriodsInWeeks input ...")
            logger.warning("using default value 13 ...")
            HistoryPeriods = [13]

        if relevant_time_name == constants.week_col:
            history_periods_based_on_gen_bucket = HistoryPeriods
        elif (
            relevant_time_name == constants.quarter_col
            or relevant_time_name == constants.planning_quarter_col
        ):
            history_periods_based_on_gen_bucket = [
                int(round(x * 4 / 52, 0)) for x in HistoryPeriods
            ]
        elif (
            relevant_time_name == constants.month_col
            or relevant_time_name == constants.planning_month_col
        ):
            history_periods_based_on_gen_bucket = [
                int(round(x * 12 / 52, 0)) for x in HistoryPeriods
            ]
        else:
            logger.warning(
                f"Unknown relevant_time_name {relevant_time_name}, using default history periods {HistoryPeriods}"
            )
            history_periods_based_on_gen_bucket = HistoryPeriods

        # fill missing dates using the max value in gen bucket such that no intersections get excluded
        # get last n periods
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -max(history_periods_based_on_gen_bucket),
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        # fill NAs with zero
        relevant_history_nas_filled = fill_missing_dates(
            actual=Actual_df,
            forecast_level=dimensions,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            time_mapping=relevant_time_mapping,
            relevant_time_periods=last_n_periods,
            fill_nulls_with_zero=True,
            filter_from_start_date=True,
        )

        # cap negatives
        relevant_history_nas_filled[history_measure] = np.where(
            relevant_history_nas_filled[history_measure] < 0,
            0,
            relevant_history_nas_filled[history_measure],
        )

        actual_df = relevant_history_nas_filled.copy()
        output_list = []

        transition_group_by_cols = SalesDomainGrains + [
            constants.transition_item_col,
            LocationLevel,
        ]

        for the_history_period in history_periods_based_on_gen_bucket:

            # get last n periods
            last_n_periods = get_n_time_periods(
                latest_time_name,
                -the_history_period,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=True,
            )

            # Filter out relevant history
            relevant_history = actual_df[actual_df[relevant_time_name].isin(last_n_periods)]

            if len(relevant_history) == 0:
                logger.warning(
                    "Empty dataframe after filtering data for {} periods at {} level for slice {}...".format(
                        the_history_period,
                        relevant_time_name,
                        df_keys,
                    )
                )
                continue

            logger.info(
                "Calculating average of last {} periods ...".format(
                    history_periods_based_on_gen_bucket
                )
            )

            # Take average of last n periods
            last_n_period_avg_actuals = relevant_history.groupby(
                dimensions,
                as_index=False,
                observed=True,
            )[[history_measure]].mean()

            # take last n period sum at parent (transition item) level
            last_n_period_avg_actuals[agg_history_measure_col] = last_n_period_avg_actuals.groupby(
                transition_group_by_cols, observed=True
            )[history_measure].transform("sum")

            # filter out intersections with non zero sum at transition level and add to master list
            intersections_with_non_zero_avg = last_n_period_avg_actuals[
                last_n_period_avg_actuals[agg_history_measure_col] != 0
            ]
            output_list.append(intersections_with_non_zero_avg)

            intersections_with_zero_avg = last_n_period_avg_actuals[
                last_n_period_avg_actuals[agg_history_measure_col] == 0
            ]
            relevant_intersection_for_next_iteration = intersections_with_zero_avg[dimensions]

            if (len(relevant_intersection_for_next_iteration) != 0) & (
                the_history_period != history_periods_based_on_gen_bucket[-1]
            ):
                actual_df = actual_df.merge(
                    relevant_intersection_for_next_iteration,
                    on=dimensions,
                    how="inner",
                )
                continue

            else:
                # Search in the space provided, if still zero, retain the value as it is
                output_list.append(intersections_with_zero_avg)
                break

        last_n_period_avg_actuals = concat_to_dataframe(output_list)
        if last_n_period_avg_actuals.empty:
            cols_req = dimensions + [history_measure, agg_history_measure_col]
            last_n_period_avg_actuals = pd.DataFrame(columns=cols_req)
        last_n_period_avg_actuals.drop(agg_history_measure_col, axis=1, inplace=True)

        logger.info("default_intro_date : {}".format(default_intro_date))
        logger.info("default_disc_date : {}".format(default_disc_date))

        logger.info("-------- Printing PItemDates (supplied) -----------")
        logger.info(PItemDates.head())

        # considering transition items that are part of sequence
        cols_req = [
            constants.transition_item_col,
            constants.transition_demand_domain,
            LocationLevel,
        ] + [x for x in SalesDomainGrains if x != o9Constants.PLANNING_DEMAND_DOMAIN]
        transition_items_to_consider = Actual_df[cols_req].drop_duplicates()
        PItemDates = PItemDates.merge(transition_items_to_consider)

        PItemPhaseData = PItemDates.copy()
        if not PItemPhaseData.empty:
            PItemPhaseData[constants.phase_out_buckets] = PItemPhaseData[
                constants.phase_out_buckets
            ].fillna(0)
            PItemPhaseData[constants.adjust_phase_out_profile] = PItemPhaseData[
                constants.adjust_phase_out_profile
            ].fillna(False)

            PItemPhaseData[constants.prod_transition_start_date] = pd.to_datetime(
                PItemPhaseData[constants.prod_transition_start_date], utc=True
            ).dt.tz_localize(None)

            PItemPhaseData[constants.pitem_intro_date_col] = pd.to_datetime(
                PItemPhaseData[constants.pitem_intro_date_col], utc=True
            ).dt.tz_localize(None)

            PItemPhaseData[constants.pitem_disc_date_col] = pd.to_datetime(
                PItemPhaseData[constants.pitem_disc_date_col], utc=True
            ).dt.tz_localize(None)

            PItemPhaseData[constants.pitem_disc_date_col] = PItemPhaseData[
                constants.pitem_disc_date_col
            ].dt.normalize()
            PItemPhaseData[constants.pitem_intro_date_col] = PItemPhaseData[
                constants.pitem_intro_date_col
            ].dt.normalize()
            PItemPhaseData[constants.prod_transition_start_date] = PItemPhaseData[
                constants.prod_transition_start_date
            ].dt.normalize()
            PItemPhaseData[constants.pitem_intro_date_col].fillna(default_intro_date, inplace=True)
            PItemPhaseData[constants.pitem_disc_date_col].fillna(default_disc_date, inplace=True)

            PItemPhaseData.sort_values(by=constants.pitem_disc_date_col, inplace=True)
            PItemPhaseData = (
                pd.merge_asof(
                    PItemPhaseData,
                    TimeDimension[[constants.day_key_col, relevant_time_key]],
                    left_on=constants.pitem_disc_date_col,
                    right_on=constants.day_key_col,
                    # direction="forward",
                )
                .drop(columns=[constants.day_key_col])
                .rename(columns={relevant_time_key: constants.end_date})
            )

            PItemPhaseData[constants.day_key_col] = PItemPhaseData[
                constants.end_date
            ] - pd.to_timedelta(PItemPhaseData[constants.phase_out_buckets] * days, unit="D")
            PItemPhaseData = (
                PItemPhaseData.merge(
                    TimeDimension[[constants.day_key_col, relevant_time_key]],
                )
                .drop(columns=[constants.day_key_col])
                .rename(columns={relevant_time_key: constants.start_date})
            )

        # Filter relevant columns
        PItemDates = PItemDates[
            [constants.version_col]
            + SalesDomainGrains
            + [
                LocationLevel,
                ItemLevel,
                constants.transition_demand_domain,
                constants.transition_item_col,
                constants.pitem_intro_date_col,
                constants.pitem_disc_date_col,
            ]
        ]

        if len(PItemDates) == 0:
            logger.info("Planning item dataframe is empty, creating one using default dates ...")
            # create dataframe with all combinations
            PItemDates = Actual_df[dimensions].drop_duplicates()
            PItemDates.reset_index(drop=True, inplace=True)
            PItemDates[constants.pitem_intro_date_col] = default_intro_date
            PItemDates[constants.pitem_disc_date_col] = default_disc_date
            PItemDates[constants.version_col] = input_version
        else:
            PItemDates[constants.pitem_intro_date_col] = pd.to_datetime(
                PItemDates[constants.pitem_intro_date_col], utc=True
            ).dt.tz_localize(None)

            PItemDates[constants.pitem_disc_date_col] = pd.to_datetime(
                PItemDates[constants.pitem_disc_date_col], utc=True
            ).dt.tz_localize(None)
            # create dataframe containing all intersections in Actuals
            all_intersections_df = Actual_df[dimensions].drop_duplicates()

            # piece of code to map the intro date to the start of time bucket, so that transition at the middle of the bucket is respected
            TimeDimension[o9Constants.DAY_KEY] = TimeDimension[o9Constants.DAY_KEY].dt.normalize()
            time_cols_required = [o9Constants.DAY_KEY, relevant_time_key]
            PItemDates_intro = PItemDates[PItemDates[constants.pitem_intro_date_col].notna()]
            PItemDates_disc = PItemDates[PItemDates[constants.pitem_disc_date_col].notna()]
            PItemDates_intro[constants.pitem_intro_date_col] = PItemDates_intro[
                constants.pitem_intro_date_col
            ].dt.normalize()  # normalizing the timestamps to make it insensitive of time HH:MM:SS values for merging with TimeDim
            PItemDates_intro = PItemDates_intro.merge(
                TimeDimension[time_cols_required].drop_duplicates(),
                left_on=constants.pitem_intro_date_col,
                right_on=o9Constants.DAY_KEY,
                how="inner",
            )
            # Update the intro date column to the relevant time key
            PItemDates_intro[constants.pitem_intro_date_col] = PItemDates_intro[relevant_time_key]

            PItemDates = PItemDates_intro[
                [o9Constants.VERSION_NAME]
                + dimensions
                + [constants.pitem_intro_date_col, o9Constants.DAY_KEY, relevant_time_key]
            ].merge(
                PItemDates_disc[
                    [o9Constants.VERSION_NAME] + dimensions + [constants.pitem_disc_date_col]
                ],
                on=[o9Constants.VERSION_NAME] + dimensions,
                how="outer",
            )

            # outer join to get all combinations including the ones with actuals and new items
            PItemDates = PItemDates.merge(
                all_intersections_df,
                how="outer",
                on=dimensions,
            )

            logger.info("Filling missing dates with default values ...")
            # fill missing entries with default values
            PItemDates[constants.version_col].fillna(input_version, inplace=True)
            # Drop the columns added during the merge to maintain original structure
            PItemDates.drop(columns=[o9Constants.DAY_KEY, relevant_time_key], inplace=True)
            PItemDates[constants.pitem_intro_date_col].fillna(default_intro_date, inplace=True)
            PItemDates[constants.pitem_disc_date_col].fillna(default_disc_date, inplace=True)

        # Convert to datetime
        PItemDates[constants.pitem_intro_date_col] = pd.to_datetime(
            PItemDates[constants.pitem_intro_date_col]
        )
        PItemDates[constants.pitem_disc_date_col] = pd.to_datetime(
            PItemDates[constants.pitem_disc_date_col]
        )

        if len(PItemDates) == 0:
            logger.warning(
                "Empty dataframe after joining intersections master data with planning item dates for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe for this slice ...")
            return PLProfile, ActualLastNBucketsPL

        logger.info("Assign forecast start and end dates ...")
        # Assign forecast start and end dates
        PItemDates[constants.fcst_start_date_col] = pd.to_datetime(
            name_to_key_mapping[forecast_dates[0]]
        )
        PItemDates[constants.fcst_end_date_col] = pd.to_datetime(
            name_to_key_mapping[forecast_dates[-1]]
        )

        PItemDates[constants.fcst_start_date_col] = pd.to_datetime(
            PItemDates[constants.fcst_start_date_col], utc=True
        ).dt.tz_localize(None)

        PItemDates[constants.fcst_end_date_col] = pd.to_datetime(
            PItemDates[constants.fcst_end_date_col], utc=True
        ).dt.tz_localize(None)

        # converting all the times to midnight value to compare the values accurately
        PItemDates[constants.pitem_disc_date_col] = PItemDates[
            constants.pitem_disc_date_col
        ].dt.normalize()
        PItemDates[constants.pitem_intro_date_col] = PItemDates[
            constants.pitem_intro_date_col
        ].dt.normalize()
        PItemDates[constants.fcst_end_date_col] = PItemDates[
            constants.fcst_end_date_col
        ].dt.normalize()
        PItemDates[constants.fcst_start_date_col] = PItemDates[
            constants.fcst_start_date_col
        ].dt.normalize()

        # left join : because we need to populate output for all intersections in PItemDates
        PItemDates = PItemDates.merge(last_n_period_avg_actuals, how="left", on=dimensions)

        # get keys corresponding to forecast date names
        forecast_date_keys = list(
            relevant_time_mapping[relevant_time_mapping[relevant_time_name].isin(forecast_dates)][
                relevant_time_key
            ]
        )

        forecast_dates_df = pd.DataFrame(
            {
                relevant_time_name: forecast_dates,
                relevant_time_key: forecast_date_keys,
            }
        )

        logger.info("Adding forecast dates for all intersections ...")

        # Repeat rows for all the forecast dates present to get planning item properties for the forecast horizon
        data = create_cartesian_product(df1=PItemDates, df2=forecast_dates_df)

        # getting time bucket key corresponding to intro date
        # to address issue of intro date as 1-Apr and time bucket key as 31-Mar
        data.sort_values(by=constants.pitem_intro_date_col, inplace=True)
        data = pd.merge_asof(
            data,
            TimeDimension[[constants.day_key_col, relevant_time_key]],
            left_on=constants.pitem_intro_date_col,
            right_on=constants.day_key_col,
            direction="forward",
            suffixes=("", "_x"),
        )
        data.drop(constants.day_key_col, axis=1, inplace=True)
        data.rename(columns={relevant_time_key + "_x": constants.key_to_check}, inplace=True)

        data[relevant_time_key] = pd.to_datetime(data[relevant_time_key], utc=True).dt.tz_localize(
            None
        )

        # new code here

        # if intro date is in future, profile cannot be present. Null out such profiles
        erratic_data_filter = (
            data[constants.pitem_intro_date_col] >= CurrentTimePeriod[relevant_time_key].iloc[0]
        )
        if len(data[erratic_data_filter]) > 0:
            df_to_log = data[erratic_data_filter][
                transition_group_by_cols + [constants.planning_item_col]
            ].drop_duplicates()
            logger.warning(
                f"Actuals found for planning items introduced in future, nulling such records inside the plugin. Please check data for below intersections\n{df_to_log.to_csv(index=False)} ..."
            )
            data[history_measure] = np.where(erratic_data_filter, np.nan, data[history_measure])

            # ###
            # # populate old/new for items based on intro date
            # data["Type of Item"] = np.where(erratic_data_filter, "New", "Old")
            # ###

        # add a lifecycle flag depending on intro/disc date
        life_cycle_filter = (data[relevant_time_key] >= data[constants.pitem_intro_date_col]) & (
            data[relevant_time_key] <= data[constants.pitem_disc_date_col]
        )
        data[constants.LIFECYCLE_FLAG] = np.where(life_cycle_filter, 1, 0)

        if TransitionFlag.empty:
            combined_data = data.copy()
        else:
            TRANSITIONFLAG_df_copy = TransitionFlag.copy()

            # collect all the intersections involved in transition on from side
            copy_Transition = TransitionFlag.copy()
            copy_Transition[constants.from_pl_item] = copy_Transition[constants.to_pl_item]

            TransitionFlag = pd.concat([TransitionFlag, copy_Transition], ignore_index=True)
            TransitionFlag.drop(
                [constants.version_col, constants.to_pl_item],
                axis=1,
                inplace=True,
            )
            TransitionFlag.drop_duplicates(inplace=True)

            TransitionFlag.rename(
                columns={
                    constants.from_pl_item: constants.planning_item_col,
                    constants.from_pl_pnl: constants.planning_pnl_col,
                    constants.from_pl_demand_domain: constants.planning_demand_domain_col,
                    constants.from_pl_account: constants.planning_account_col,
                    constants.from_pl_region: constants.planning_region_col,
                    constants.from_pl_channel: constants.planning_channel_col,
                    constants.from_pl_location: constants.planning_location_col,
                },
                inplace=True,
            )
            # add transition item column to the Transition Flag to be used in join
            TransitionFlag = TransitionFlag.merge(
                PItemDates[
                    [constants.planning_item_col, constants.transition_item_col]
                ].drop_duplicates(),
                on=constants.planning_item_col,
                how="inner",
            )

            # create a flag to denote transition
            data = data.merge(
                TransitionFlag,
                on=transition_group_by_cols + [constants.planning_item_col],
                how="left",
            )

            filter_transition = data[constants.TRANSITION_FLAG] == 1

            # calculate sum of profile at transition group level so that it can be split equally later
            constants.LIFECYCLE_FLAG_SUM = "Lifecycle Flag Sum"
            data[constants.GROUP_SUM] = data.groupby(
                transition_group_by_cols + [relevant_time_name]
            )[history_measure].transform("sum")

            transition_data = data[filter_transition]
            non_transition_data = data[~filter_transition]

            transition_data[constants.LIFECYCLE_FLAG_SUM] = transition_data.groupby(
                transition_group_by_cols + [relevant_time_name]
            )[constants.LIFECYCLE_FLAG].transform("sum")

            # identify overlap periods
            transition_data[constants.OVERLAP_FLAG] = np.where(
                transition_data[constants.LIFECYCLE_FLAG_SUM] > 1, 1, 0
            )

            # identify available profile from existing planning item belonging to same transition item
            # transition_data[constants.AVAILABLE_PROFILE] = transition_data.groupby(
            #     transition_group_by_cols + [relevant_time_name]
            # )[history_measure].ffill().bfill()

            ###
            overlap_profiles_list = []
            eol_profiles_list = []
            if not PItemPhaseData.empty:
                # new : overlap period disagg logic
                TRANSITIONFLAG_df_copy = TRANSITIONFLAG_df_copy.merge(
                    PItemPhaseData[[ItemLevel, constants.transition_item_col]].drop_duplicates(),
                    how="left",
                    left_on=[constants.from_pl_item],
                    right_on=[ItemLevel],
                )

                TRANSITIONFLAG_df_copy.rename(
                    columns={
                        constants.from_pl_pnl: constants.planning_pnl_col,
                        constants.from_pl_account: constants.planning_account_col,
                        constants.from_pl_region: constants.planning_region_col,
                        constants.from_pl_channel: constants.planning_channel_col,
                        constants.from_pl_location: constants.planning_location_col,
                        constants.from_pl_demand_domain: constants.planning_demand_domain_col,
                    },
                    inplace=True,
                )

                # overlap periods profile calculation for intersection with transition type as part revision
                group_by_cols = SalesDomainGrains + [LocationLevel, constants.transition_item_col]
                overlap_profiles_list = Parallel(n_jobs=1, verbose=1)(
                    delayed(get_profiles_in_overlap_period)(
                        df=df,
                        PItemPhaseData=PItemPhaseData,
                        transition_data=transition_data,
                        SalesDomainGrains=SalesDomainGrains,
                        LocationLevel=LocationLevel,
                        DefaultProfiles=DefaultProfiles,
                        relevant_time_name=relevant_time_name,
                        history_measure=history_measure,
                        relevant_time_key=relevant_time_key,
                    )
                    for name, df in TRANSITIONFLAG_df_copy.groupby(group_by_cols)
                )

                # profile calculations for intersections with transition type as eol
                PItemPhaseData = PItemPhaseData[PItemPhaseData[constants.transition_type] == "EOL"]
                eol_profiles_list = Parallel(n_jobs=1, verbose=1)(
                    delayed(get_eol_profiles)(
                        df=df,
                        transition_data=pd.concat([transition_data, non_transition_data]),
                        SalesDomainGrains=SalesDomainGrains,
                        LocationLevel=LocationLevel,
                        history_measure=history_measure,
                        DefaultProfiles=DefaultProfiles,
                        relevant_time_name=relevant_time_name,
                        relevant_time_key=relevant_time_key,
                        TimeDimension=TimeDimension,
                    )
                    for name, df in PItemPhaseData.groupby(
                        group_by_cols + [constants.planning_item_col]
                    )
                )
            eol_profiles = concat_to_dataframe(eol_profiles_list)
            if eol_profiles.empty:
                eol_profiles = pd.DataFrame(
                    columns=list(transition_data.columns) + [constants.AVAILABLE_PROFILE]
                )

            overlap_profiles = concat_to_dataframe(overlap_profiles_list)
            if overlap_profiles.empty:
                overlap_profiles = pd.DataFrame(
                    columns=list(transition_data.columns) + [constants.AVAILABLE_PROFILE]
                )

            overlap_intersections = overlap_profiles[
                SalesDomainGrains + [LocationLevel, constants.planning_item_col, relevant_time_name]
            ]
            eol_intersections = eol_profiles[
                SalesDomainGrains + [LocationLevel, constants.planning_item_col, relevant_time_name]
            ]
            intersections_considered = pd.concat(
                [overlap_intersections, eol_intersections], ignore_index=True
            )

            transition_data = transition_data.merge(
                intersections_considered,
                how="left",
                indicator=True,
            )
            transition_data = transition_data[transition_data["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
            non_transition_data = non_transition_data.merge(
                eol_intersections,
                how="left",
                indicator=True,
            )
            non_transition_data = non_transition_data[
                non_transition_data["_merge"] == "left_only"
            ].drop(columns=["_merge"])

            # logic changed for overlap period
            # divide profile equally in the overlap period
            transition_data[constants.AVAILABLE_PROFILE] = np.where(
                transition_data[constants.OVERLAP_FLAG] == 1,
                transition_data[constants.GROUP_SUM]
                / transition_data[constants.LIFECYCLE_FLAG_SUM],
                transition_data[constants.GROUP_SUM],
            )

            # copy over to profile
            transition_data[history_measure] = transition_data[constants.AVAILABLE_PROFILE]
            # transition_data[history_measure] = np.where(
            #     transition_data[history_measure].isna() | transition_data[constants.OVERLAP_FLAG]
            #     == 1,
            #     transition_data[constants.AVAILABLE_PROFILE],
            #     transition_data[history_measure],
            # )

            transition_data = pd.concat([transition_data, overlap_profiles, eol_profiles])

            assert not transition_data[constants.AVAILABLE_PROFILE].isnull().values.any()

            # append no longer works so added pd.concat
            # combined_data = non_transition_data.append(
            #     transition_data[non_transition_data.columns], ignore_index=True
            # )

            # append to master
            combined_data = pd.concat(
                [non_transition_data, transition_data[non_transition_data.columns]],
                ignore_index=True,
            )

        # set profile to zero outside lifecycle
        combined_data[OutputMeasure] = np.where(
            combined_data[constants.LIFECYCLE_FLAG] == 0,
            0,
            combined_data[history_measure],
        )

        # new code ends
        cols_required_in_output_at_relevant_time_level = (
            [constants.version_col] + dimensions + [relevant_time_name, OutputMeasure]
        )

        PLProfile = combined_data[cols_required_in_output_at_relevant_time_level]

        relevant_dates = forecast_dates[offset_periods:]
        relevant_dates = pd.DataFrame({relevant_time_name: relevant_dates})
        PLProfile = PLProfile.merge(
            relevant_dates, on=[relevant_time_name], how="inner"
        ).drop_duplicates()

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.merge(
            base_time_mapping, on=constants.partial_week_col, how="inner"
        )

        # perform disaggregation
        PLProfile = disaggregate_data(
            source_df=PLProfile,
            source_grain=relevant_time_name,
            target_grain=constants.partial_week_col,
            profile_df=StatBucketWeight.drop(constants.version_col, axis=1),
            profile_col=constants.stat_bucket_weight_col,
            cols_to_disaggregate=[OutputMeasure],
        )

        # reorder columns
        PLProfile = PLProfile[cols_required_in_output]

        # actuals last n buckets output
        current_month = CurrentTimePeriod[o9Constants.MONTH][0]
        n_bucket_months = get_n_time_periods(
            current_month,
            -int(NBucketsinMonths),
            TimeDimension[[o9Constants.MONTH, o9Constants.MONTH_KEY]].drop_duplicates(),
            {o9Constants.MONTH: o9Constants.MONTH_KEY},
            include_latest_value=False,
        )
        relevant_partial_weeks = TimeDimension[
            TimeDimension[o9Constants.MONTH].isin(n_bucket_months)
        ][o9Constants.PARTIAL_WEEK]
        ActualLastNBucketsPL = Actuals_PW[
            Actuals_PW[o9Constants.PARTIAL_WEEK].isin(relevant_partial_weeks)
        ].rename(columns={history_measure: constants.actual_last_n_buckets_col})
        ActualLastNBucketsPL = ActualLastNBucketsPL[cols_req_in_actual_lastn_buckets]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        PLProfile = pd.DataFrame(columns=cols_required_in_output)
        ActualLastNBucketsPL = pd.DataFrame(columns=cols_req_in_actual_lastn_buckets)
    return PLProfile, ActualLastNBucketsPL

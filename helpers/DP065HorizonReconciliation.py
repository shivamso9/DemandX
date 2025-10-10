import logging
import math
import re

import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")


pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Stat Fcst": float,
    "Stat Fcst CML Baseline": float,
    "Stat Fcst CML Residual": float,
    "Stat Fcst CML Holiday": float,
    "Stat Fcst CML Promo": float,
    "Stat Fcst CML Marketing": float,
    "Stat Fcst CML Price": float,
    "Stat Fcst CML Weather": float,
    "Stat Fcst CML External Driver": float,
    "Reconciliation Ramp Up Weight": float,
}


def realign_and_blend_forecast(
    group,
    custom_order,
    weights_dict,
    transition_df,
    HorizonEnd,
    time_order_dict,
    agg_measure,
    forecast_type_col=o9Constants.FORECAST_ITERATION_TYPE,
):
    try:
        # Map forecast_type to its max horizon (e.g. {"Very Short Term_Actual_3": 52, ...})
        horizon_dict = HorizonEnd.set_index(o9Constants.FORECAST_ITERATION_TYPE)[
            [o9Constants.FORECAST_GEN_TIME_BUCKET, o9Constants.HORIZON_END]
        ].to_dict(orient="index")
        group_temp = group.copy()
        group = group.copy()
        output_frames = []
        initial_end_date = group[o9Constants.PARTIAL_WEEK_KEY].min() - pd.Timedelta(days=1)
        end_date = initial_end_date

        # Filter once outside loop to improve performance and avoid modifying base data
        group = group[group[o9Constants.PARTIAL_WEEK_KEY] > initial_end_date]

        for forecast_type in time_order_dict:
            if (group[forecast_type_col] == forecast_type).any():
                time_col = time_order_dict[forecast_type]
                horizon_info = horizon_dict.get(forecast_type, None)

                if not horizon_info:
                    logger.warning(f"No horizon info found for {forecast_type}")
                    continue

                time_bucket = horizon_info[o9Constants.FORECAST_GEN_TIME_BUCKET]
                horizon_length = horizon_info[o9Constants.HORIZON_END]

                # Normalize time bucket for consistent handling
                normalized_bucket = time_bucket.lower()
                # Always calculate horizon from initial_end_date
                if normalized_bucket in ["week", "partial_week"]:
                    forecast_end_date = initial_end_date + pd.DateOffset(weeks=int(horizon_length))
                elif normalized_bucket in ["month", "planning_month"]:
                    forecast_end_date = initial_end_date + relativedelta(months=int(horizon_length))
                elif normalized_bucket in ["quarter", "planning_quarter"]:
                    forecast_end_date = initial_end_date + relativedelta(
                        months=3 * int(horizon_length)
                    )
                else:
                    logger.warning(f"Unknown time bucket {time_bucket} for {forecast_type}")
                    continue

                group_i = group[group[forecast_type_col] == forecast_type]
                group_i = group_i[group_i[o9Constants.PARTIAL_WEEK_KEY] > end_date]
                group_i = group_i[group_i[o9Constants.PARTIAL_WEEK_KEY] <= forecast_end_date]

                group_i_nas = group_i[group_i[agg_measure].isna()]
                group_i_not_nas = group_i[group_i[agg_measure].notna()]

                forecast_type_order = forecast_type.split("_")[0]
                if (forecast_type_order == custom_order[-1]) or (len(group_i_nas) == 0):
                    output_frames.append(group_i)
                else:
                    aggregated_non_na = group_i_not_nas.groupby(time_col).agg({agg_measure: "sum"})
                    aggregated_input = group_i.groupby(time_col).agg({agg_measure: "sum"})

                    if len(aggregated_non_na) < len(aggregated_input):
                        continue
                    else:
                        output_frames.append(group_i_not_nas)

                end_date = forecast_end_date

            else:
                logger.warning(f"Forecast Type {forecast_type} not found, skipping...")
                continue

        if len(output_frames) > 0:
            realigned_group = pd.concat(output_frames, ignore_index=True)
        else:
            logger.warning(f"{agg_measure}: No data after realignment, returning empty DataFrame")
            return pd.DataFrame(columns=group.columns)

        # Step 2: Smooth transition blending
        blended_frames = []
        transition_df = transition_df[
            transition_df[o9Constants.FORECAST_ITERATION_TYPE].isin(time_order_dict.keys())
        ].reset_index(drop=True)
        for i in range(len(transition_df) - 1):
            ft_early = transition_df.iloc[i]
            ft_late = transition_df.iloc[i + 1]

            early_type = ft_early[forecast_type_col]
            late_type = ft_late[forecast_type_col]
            # transition_period = int(ft_late["Reconciliation Transition Period"])
            transition_period = pd.to_numeric(
                ft_late.get("Reconciliation Transition Period", None), errors="coerce"
            )
            if pd.isna(transition_period):
                logger.warning(f"Transition period is missing or invalid for {ft_late}")
                continue
            transition_period = int(transition_period)

            simplified_early = ft_early[o9Constants.FORECAST_ITERATION_TYPE]
            simplified_late = ft_late[o9Constants.FORECAST_ITERATION_TYPE]
            time_col_early = time_order_dict[simplified_early]
            time_col_late = time_order_dict[simplified_late]
            key_col_early = get_key_version_of_time_col(time_col_early)
            key_col_late = get_key_version_of_time_col(time_col_late)

            group_early = realigned_group[realigned_group[forecast_type_col] == early_type]
            group_late = realigned_group[realigned_group[forecast_type_col] == late_type]

            # Skip if one of the forecast types is missing
            if group_early.empty or group_late.empty:
                logger.warning(
                    f"Skipping smooth transition from {early_type} to {late_type} — missing forecast data for measure: {agg_measure}"
                )
                continue

            group_early = realigned_group[realigned_group[forecast_type_col] == early_type]
            group_late = realigned_group[realigned_group[forecast_type_col] == late_type]

            # transition_keys = group_late[key_col_late].drop_duplicates().sort_values().head(transition_period)
            transition_keys = (
                group_late[key_col_late].drop_duplicates().sort_values().head(transition_period)
            )

            early_trans = group_temp[
                (group_temp[key_col_late].isin(transition_keys))
                & (group_temp[forecast_type_col] == early_type)
            ]
            late_trans = group_temp[
                (group_temp[key_col_late].isin(transition_keys))
                & (group_temp[forecast_type_col] == late_type)
            ]

            # Skip realignment if transition_period is 0 or early_trans is empty
            if transition_period == 0 or early_trans.empty:
                logger.warning(
                    f"Skipping transition from {early_type} to {late_type} — "
                    f"{'transition period is 0' if transition_period == 0 else 'no data available for Iteration Type'} {early_type}"
                )
                continue

            merge_keys = list(set(realigned_group.columns) - {agg_measure, forecast_type_col})
            merged = pd.merge(
                early_trans,
                late_trans,
                on=merge_keys,
                suffixes=("_early", "_late"),
                how="outer",
            )

            if merged.empty:
                logger.warning(
                    f"Overlapping dataframe for transition {early_type} to {late_type} is empty. "
                    f"Check input data."
                )
                continue

            weight_key = f"{early_type.split('_')[0]} to {late_type.split('_')[0]}__weights"
            weight_df = weights_dict.get(weight_key)
            if weight_df is None or weight_df.shape[0] < 2:
                logger.warning(f"Weights for {weight_key} not found or invalid")
                continue

            ramp_down_weights = weight_df.iloc[0].to_dict()
            ramp_up_weights = weight_df.iloc[1].to_dict()

            # sorted_time_keys = sorted(merged[key_col_late].dropna().unique())
            sorted_time_keys = sorted(early_trans[key_col_early].dropna().unique())
            time_bucket_map = {key: f"T{i+1}" for i, key in enumerate(sorted_time_keys)}

            agg_late = merged.groupby(key_col_late)[f"{agg_measure}_late"].sum().reset_index()
            agg_late = agg_late.rename(columns={f"{agg_measure}_late": "agg_fcst_late"})

            merged = merged.merge(agg_late, on=key_col_late, how="left")

            early_sum = merged.groupby(key_col_late)[f"{agg_measure}_early"].transform("sum")
            # updated_fcst = agg_fcst_late * (early_fcst / sum_early_fcst)
            merged[f"{agg_measure}_late"] = merged["agg_fcst_late"] * (
                merged[f"{agg_measure}_early"] / early_sum
            )
            merged = merged.drop(columns=["agg_fcst_late"])

            # Map each row's time_key to its weight bucket
            merged["weight_bucket"] = merged[key_col_early].map(time_bucket_map)

            # Map ramp weights using the weight bucket
            merged["down_weight"] = merged["weight_bucket"].map(ramp_down_weights).fillna(0)
            merged["up_weight"] = merged["weight_bucket"].map(ramp_up_weights).fillna(0)

            # Fill missing values in measure columns with 0
            merged[f"{agg_measure}_early"] = merged[f"{agg_measure}_early"].fillna(0)
            merged[f"{agg_measure}_late"] = merged[f"{agg_measure}_late"].fillna(0)

            # Compute the blended value
            merged[agg_measure] = (
                merged["down_weight"] * merged[f"{agg_measure}_early"]
                + merged["up_weight"] * merged[f"{agg_measure}_late"]
            )

            # Optionally clean up
            merged.drop(columns=["weight_bucket", "down_weight", "up_weight"], inplace=True)

            merged[forecast_type_col] = late_type
            merged = merged[merge_keys + [agg_measure, forecast_type_col]]
            blended_frames.append(merged)

            # Drop early & late data used in blending
            realigned_group_temp = realigned_group[
                ~(
                    (
                        (realigned_group[forecast_type_col] == early_type)
                        & realigned_group[key_col_late].isin(transition_keys)
                    )
                    | (
                        (realigned_group[forecast_type_col] == late_type)
                        & realigned_group[key_col_late].isin(transition_keys)
                    )
                )
            ]

        # Append untouched remaining data
        final_result = pd.concat([realigned_group_temp] + blended_frames, ignore_index=True)
        return final_result.drop_duplicates()
    except Exception as e:
        logger.warning(f"Error in realign_and_blend_forecast: {e}")
        return pd.DataFrame(columns=group.columns)


def normalize_and_interpolate_weights(CustomRampUpWeights, lengthOfSmoothenWindow):
    """
    1) If CustomRampUpWeights is already the right length, return it unchanged.
    2) Otherwise: min–max normalize to [0,1], interpolate to the target length,
       then shrink that [0,1] range into [ε, 1−ε] so you never hit 0 or 1.
    """
    if not CustomRampUpWeights:
        raise ValueError("Need at least one weight in CustomRampUpWeights")

    k = len(CustomRampUpWeights)
    m = lengthOfSmoothenWindow

    # ── 1) EARLY EXIT ──
    if k == m:
        return list(CustomRampUpWeights)

    # ── 2) NORMALIZE to [0,1] ──
    wmin, wmax = min(CustomRampUpWeights), max(CustomRampUpWeights)
    if wmax > wmin:
        norm = [(w - wmin) / (wmax - wmin) for w in CustomRampUpWeights]
    else:
        # flat ramp if all equal
        norm = [0.5] * k

    # ── 3) INTERPOLATE to m points ──
    out = []
    for i in range(m):
        t = i * (k - 1) / (m - 1)
        lo = math.floor(t)
        hi = math.ceil(t)
        if lo == hi:
            out.append(norm[lo])
        else:
            frac = t - lo
            out.append(norm[lo] * (1 - frac) + norm[hi] * frac)

    # ── 4) SHRINK into [ε,1−ε] to avoid exact 0 or 1 ──
    #    pick ε = 1/(m+1) so that the soft endpoints are (1/(m+1)) … (m/(m+1))
    eps = 1.0 / (m + 1)
    scale = 1.0 - 2 * eps
    out = [v * scale + eps for v in out]

    return out


def create_custom_weight_tables(output_dict, CustomRampUpWeights):
    weight_tables = {}

    for key, df in output_dict.items():
        if key.endswith("__transition"):
            window_len = len(df)
            ramp_up = normalize_and_interpolate_weights(CustomRampUpWeights, window_len)
            ramp_down = [round(1 - w, 6) for w in ramp_up]

            # Create labels like "Short Term", "Mid Term", etc.
            label_parts = key.replace("__transition", "").split(" to ")
            down_label = f"{label_parts[0]} Fcst Weight (Ramp Down)"
            up_label = f"{label_parts[1]} Fcst Weight (Ramp Up)"

            df_weights = pd.DataFrame(
                [ramp_down, ramp_up],
                index=[down_label, up_label],
                columns=[f"T{i+1}" for i in range(window_len)],
            )

            weight_tables[key.replace("__transition", "__weights")] = df_weights

    return weight_tables


def create_fixed_weight_tables(output_dict):
    weight_tables = {}

    for key, df in output_dict.items():
        if key.endswith("__transition"):
            window_len = len(df)
            step = 1 / (window_len + 1)

            ramp_up = [round(step * (i + 1), 6) for i in range(window_len)]
            ramp_down = [round(1 - val, 6) for val in ramp_up]

            # Use the names in the key for labeling rows
            label_parts = key.replace("__transition", "").split(" to ")
            down_label = f"{label_parts[0]} Fcst Weight (Ramp Down)"
            up_label = f"{label_parts[1]} Fcst Weight (Ramp Up)"

            df_weights = pd.DataFrame(
                [ramp_down, ramp_up],
                index=[down_label, up_label],
                columns=[f"T{i+1}" for i in range(window_len)],
            )

            weight_tables[key.replace("__transition", "__weights")] = df_weights

    return weight_tables


def get_horizon_and_transition_weeks(
    TimeDimension,
    current_time_col,
    current_time_key_col,
    next_time_col,
    next_time_key_col,
    current_time,
    horizon_periods,
    transition_periods,
):
    # Sort by current time key
    TimeDimension = TimeDimension.sort_values(current_time_key_col).reset_index(drop=True)

    # Start from current time
    start_index = TimeDimension[TimeDimension[current_time_key_col] == current_time].index[0]
    future_time = TimeDimension.iloc[start_index:].copy()

    # --- HORIZON ---
    unique_time_keys = future_time[current_time_key_col].drop_duplicates().reset_index(drop=True)
    selected_horizon_keys = unique_time_keys.iloc[: horizon_periods + 1]
    horizon_df = future_time[
        future_time[current_time_key_col].isin(selected_horizon_keys[:horizon_periods])
    ]

    # --- TRANSITION ---
    last_next_time_key = horizon_df[next_time_key_col].dropna().iloc[-1]
    all_next_time_keys = TimeDimension[next_time_key_col].drop_duplicates().reset_index(drop=True)
    last_idx = all_next_time_keys[all_next_time_keys == last_next_time_key].index[0]

    # Now get the +1 bucket and compare
    # extra_next_time_key = future_time[future_time[current_time_key_col] == selected_horizon_keys.iloc[horizon_periods]][next_time_key_col].dropna().unique()
    extra_next_time_key = (
        future_time[
            future_time[current_time_key_col] == selected_horizon_keys.iloc[horizon_periods]
        ]
        .sort_values(by=next_time_key_col)
        .dropna(subset=[next_time_key_col])[next_time_key_col]
        .iloc[-1:]  # keep the last row only
        .unique()
    )
    if len(extra_next_time_key) > 0 and extra_next_time_key[0] == last_next_time_key:
        # Same as last one → include last_idx in transition
        next_keys = all_next_time_keys.iloc[last_idx : last_idx + transition_periods]
    else:
        # Different → move one step ahead
        next_keys = all_next_time_keys.iloc[last_idx + 1 : last_idx + 1 + transition_periods]
        # next_keys = all_next_time_keys.iloc[last_idx + 1 : last_idx + 1 + transition_periods]
    transition_df = future_time[future_time[next_time_key_col].isin(next_keys)]

    # --- Return only required columns ---
    cols_to_return = [current_time_col, current_time_key_col]
    return (
        horizon_df[cols_to_return].drop_duplicates(),
        transition_df[cols_to_return].drop_duplicates(),
    )


def get_key_version_of_time_col(col):
    return re.sub(r"\[(.*?)\]", lambda m: f'[{m.group(1).replace(" ", "")}Key]', col)


def append_realignment_last_bucket(group, custom_order, time_order_dict, agg_measure):
    required_columns = list(
        set(group.columns).difference(set([o9Constants.FORECAST_ITERATION_TYPE]))
    )
    output = [pd.DataFrame(columns=required_columns)]
    try:
        end_date = group[o9Constants.PARTIAL_WEEK_KEY].min() - pd.Timedelta(days=1)
        # Time order dict : {"Short Term" : "Time.[Week]", "Mid Term" : "Time.[Month]", etc}
        for forecast_type in time_order_dict:
            if (group[o9Constants.FORECAST_ITERATION_TYPE] == forecast_type).any():
                group = group[group[o9Constants.PARTIAL_WEEK_KEY] > end_date]
                group_i = group[group[o9Constants.FORECAST_ITERATION_TYPE] == forecast_type]
                group_i_nas = group_i[group_i[agg_measure].isna()]
                group_i_not_nas = group_i[group_i[agg_measure].notna()]
                forecast_type_order = forecast_type.split("_")[0]
                if (forecast_type_order == custom_order[-1]) or (len(group_i_nas) == 0):
                    output += [group_i]
                else:
                    aggregated_non_na = group_i_not_nas.groupby(time_order_dict[forecast_type]).agg(
                        {agg_measure: "sum"}
                    )
                    aggregated_input = group_i.groupby(time_order_dict[forecast_type]).agg(
                        {agg_measure: "sum"}
                    )
                    if len(aggregated_non_na) < len(aggregated_input):
                        continue
                    else:
                        output += [group_i_not_nas]
                end_date = group_i[o9Constants.PARTIAL_WEEK_KEY].max()
            else:
                logger.warning(
                    f"Forecast Type {forecast_type} not found, skipping to next iteration..."
                )
                continue
        output = concat_to_dataframe(output)
        output = output[required_columns].drop_duplicates()
    except Exception as e:
        logger.warning(e)
        logger.warning(f"Case not covered for combination {group.head(1)}")
        return pd.DataFrame(columns=required_columns)
    return output


def build_ramp_up_df(output_dict, weights_dict, version_name, TimeDimension, measure_name):
    ramp_up_dfs = []

    for key in output_dict:
        if key.endswith("__transition"):
            transition_key = key.replace("__transition", "")
            time_df = output_dict[key]
            weights_row = weights_dict[f"{transition_key}__weights"].iloc[1]

            # Detect the time key column dynamically
            time_key_col = next(col for col in time_df.columns if "Key" in col)

            # Map sorted time keys to T1, T2, ...
            unique_keys = sorted(time_df[time_key_col].dropna().unique())
            key_to_bucket = {key: f"T{i+1}" for i, key in enumerate(unique_keys)}

            # Extract forecast iteration type after "to"
            forecast_iteration_type = transition_key.split(" to ")[-1].strip()

            # Create new columns
            df = time_df.copy()
            df["weight_bucket"] = df[time_key_col].map(key_to_bucket)
            df[measure_name] = df["weight_bucket"].map(weights_row)
            df[o9Constants.VERSION_NAME] = version_name
            df[o9Constants.FORECAST_ITERATION_TYPE] = forecast_iteration_type

            # Join df with time dimension on time_key_col
            df = df.merge(
                TimeDimension[
                    [o9Constants.PARTIAL_WEEK, o9Constants.PARTIAL_WEEK_KEY, time_key_col]
                ],
                on=time_key_col,
                how="left",
            )
            ramp_up_dfs.append(
                df[
                    [
                        o9Constants.VERSION_NAME,
                        o9Constants.FORECAST_ITERATION_TYPE,
                        o9Constants.PARTIAL_WEEK,
                        o9Constants.PARTIAL_WEEK_KEY,
                        measure_name,
                    ]
                ]
            )

    # Combine all ramp-up DataFrames
    y = pd.concat(ramp_up_dfs, ignore_index=True)
    # Sort to ensure earliest Partial Week per weight
    y_sorted = y.sort_values(o9Constants.PARTIAL_WEEK_KEY)

    # Drop duplicates keeping the first Partial Week for each ramp_up_weight
    final_df = y_sorted.drop_duplicates(
        subset=[o9Constants.VERSION_NAME, o9Constants.FORECAST_ITERATION_TYPE, measure_name]
    )
    return final_df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    OutputTables,
    StatFcstPLAgg,
    CMLFcstPLAgg,
    HorizonEnd,
    TimeDimension,
    CurrentTimePeriod,
    ReconciliationMethod,
    ReconciliationTransitionPeriod,
    CustomRampUpWeight,
    ReconcileLastBucket="False",
    ForecastIterationMasterData=pd.DataFrame(),
    SellOut_Output=pd.DataFrame(),
    SellIn_Output=pd.DataFrame(),
    MLDecompositionFlag=pd.DataFrame(),
    IterationTypesOrdered=None,
    df_keys={},
):
    plugin_name = "DP065HorizonReconciliation"

    # Configurables
    STAT_FCST_PL_CML_BASELINE: str = "Stat Fcst PL CML Baseline Agg"
    STAT_FCST_PL_CML_RESIDUAL: str = "Stat Fcst PL CML Residual Agg"
    STAT_FCST_PL_CML_HOLIDAY: str = "Stat Fcst PL CML Holiday Agg"
    STAT_FCST_PL_CML_PROMO: str = "Stat Fcst PL CML Promo Agg"
    STAT_FCST_PL_CML_MARKETING: str = "Stat Fcst PL CML Marketing Agg"
    STAT_FCST_PL_CML_PRICE: str = "Stat Fcst PL CML Price Agg"
    STAT_FCST_PL_CML_WEATHER: str = "Stat Fcst PL CML Weather Agg"
    STAT_FCST_PL_CML_EXTERNAL_DRIVER: str = "Stat Fcst PL CML External Driver Agg"
    RAMP_UP_WEIGHTS_MEASURE: str = "Reconciliation Ramp Up Weight"

    STAT_FCST_CML_BASELINE: str = "Stat Fcst CML Baseline"
    STAT_FCST_CML_RESIDUAL: str = "Stat Fcst CML Residual"
    STAT_FCST_CML_HOLIDAY: str = "Stat Fcst CML Holiday"
    STAT_FCST_CML_PROMO: str = "Stat Fcst CML Promo"
    STAT_FCST_CML_MARKETING: str = "Stat Fcst CML Marketing"
    STAT_FCST_CML_PRICE: str = "Stat Fcst CML Price"
    STAT_FCST_CML_WEATHER: str = "Stat Fcst CML Weather"
    STAT_FCST_CML_EXTERNAL_DRIVER: str = "Stat Fcst CML External Driver"

    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    # get output tables list
    output_tables_list = OutputTables.strip().split(",")

    ReconcileFlag = eval(ReconcileLastBucket)

    cols_required_in_output = (
        [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK]
        + forecast_level
        + [o9Constants.STAT_FCST]
    )
    cols_required_in_cml_output = (
        [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK]
        + forecast_level
        + [
            STAT_FCST_CML_BASELINE,
            STAT_FCST_CML_RESIDUAL,
            STAT_FCST_CML_HOLIDAY,
            STAT_FCST_CML_PROMO,
            STAT_FCST_CML_MARKETING,
            STAT_FCST_CML_PRICE,
            STAT_FCST_CML_WEATHER,
            STAT_FCST_CML_EXTERNAL_DRIVER,
        ]
    )

    cols_required_in_ramp_up_weights_output = [
        o9Constants.VERSION_NAME,
        o9Constants.FORECAST_ITERATION_TYPE,
        o9Constants.PARTIAL_WEEK,
        RAMP_UP_WEIGHTS_MEASURE,
    ]

    Output = pd.DataFrame(columns=cols_required_in_output)
    CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
    RampUpweightsOutput = pd.DataFrame(columns=cols_required_in_ramp_up_weights_output)
    try:

        reconcile_stat, reconcile_cml = True, True
        input_measure = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]
        # output_measure = ForecastIterationMasterData["Iteration Type Output Stream"].values[0]
        output_measure = "Stat Fcst"  # Default output measure
        if (input_measure is None) or (output_measure is None):
            logger.warning("Empty input or output stream values, returning empty output")
            return Output, CMLOutput, RampUpweightsOutput
        ForecastIterationMasterData = ForecastIterationMasterData[
            ForecastIterationMasterData["Iteration Type Input Stream"] == input_measure
        ]
        logger.warning(f"Input Stream Selected : {input_measure}")
        logger.warning(f"Output Stream Selected : {output_measure}")

        # early exit if output tables are not provided
        if len(output_tables_list) == 0:
            logger.warning("OutputTables not provided, returning empty dataframe")
            return Output, CMLOutput, RampUpweightsOutput

        # mapping between output table names and tables
        output_tables = {}
        for table in output_tables_list:
            output_tables[table] = eval(table)

        output_table = None

        # finding the table which has the output stream in it's columns
        output_table = next(
            (df for key, df in output_tables.items() if output_measure in df.columns),
            None,
        )
        if output_table is None:
            logger.warning(
                f"Output Stream '{output_measure}' not found, returning empty dataframe..."
            )
            return Output, CMLOutput, RampUpweightsOutput
        # updating columns required in output
        cols_required_in_output = output_table.columns
        # cols_required_in_output = OutputMeasure.columns
        if output_measure not in cols_required_in_output:
            logger.warning(
                f"{output_measure} measure not found in the {output_table} table, please verify the Input and Output stream settings..."
            )
            logger.warning(f"OutputMeasure Input column settings: {cols_required_in_output}")
            logger.warning("Returning empty output...")
            return Output, CMLOutput, RampUpweightsOutput

        StatFcstPLAgg = StatFcstPLAgg.merge(
            ForecastIterationMasterData[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
            on=o9Constants.FORECAST_ITERATION_TYPE,
            how="inner",
        )
        if StatFcstPLAgg.empty:
            logger.warning(f"StatFcstPLAgg is empty for {df_keys}, returning empty Stat Output")
            reconcile_stat = False

        MLDecompositionFlag = MLDecompositionFlag.merge(
            ForecastIterationMasterData[
                [
                    o9Constants.FORECAST_ITERATION_TYPE,
                    o9Constants.FORECAST_ITERATION,
                ]
            ],
            on=o9Constants.FORECAST_ITERATION,
            how="inner",
        )
        CMLFcstPLAgg = CMLFcstPLAgg.merge(
            MLDecompositionFlag[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
            on=o9Constants.FORECAST_ITERATION_TYPE,
            how="inner",
        )
        CMLFcstPLAgg = CMLFcstPLAgg.merge(
            ForecastIterationMasterData[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
            on=o9Constants.FORECAST_ITERATION_TYPE,
            how="inner",
        )

        if CMLFcstPLAgg.empty:
            logger.warning(f"CMLFcstPLAgg is empty for {df_keys}, returning empty CML Output")
            reconcile_cml = False

        if (not reconcile_stat) and (not reconcile_cml):
            logger.warning("Both StatFcstPLAgg and CMLFcstPLAgg empty, returning empty outputs...")
            return Output, CMLOutput, RampUpweightsOutput

        if reconcile_cml:
            cml_measure_mapping = {
                STAT_FCST_PL_CML_BASELINE: STAT_FCST_CML_BASELINE,
                STAT_FCST_PL_CML_RESIDUAL: STAT_FCST_CML_RESIDUAL,
                STAT_FCST_PL_CML_HOLIDAY: STAT_FCST_CML_HOLIDAY,
                STAT_FCST_PL_CML_PRICE: STAT_FCST_CML_PRICE,
                STAT_FCST_PL_CML_PROMO: STAT_FCST_CML_PROMO,
                STAT_FCST_PL_CML_MARKETING: STAT_FCST_CML_MARKETING,
                STAT_FCST_PL_CML_WEATHER: STAT_FCST_CML_WEATHER,
                STAT_FCST_PL_CML_EXTERNAL_DRIVER: STAT_FCST_CML_EXTERNAL_DRIVER,
            }

        ForecastIterationMasterData = ForecastIterationMasterData[
            ForecastIterationMasterData["Iteration Type Input Stream"] == input_measure
        ]
        HorizonEnd = HorizonEnd.merge(
            ForecastIterationMasterData[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
            on=o9Constants.FORECAST_ITERATION_TYPE,
            how="inner",
        )

        if HorizonEnd.empty:
            logger.warning(f"HorizonEnd is empty for {df_keys}, returning empty outputs")
            return Output, CMLOutput, RampUpweightsOutput

        # Create a temporary column to order the df according to the custom order, without modifying the forecast iteration column
        temp_FI_type = "Temporary Forecast Iteration"

        if len(HorizonEnd[o9Constants.FORECAST_ITERATION_TYPE].str.split("_").values[0]) > 1:
            HorizonEnd[temp_FI_type] = (
                HorizonEnd[o9Constants.FORECAST_ITERATION_TYPE].str.split("_").str[0]
            )
            HorizonEnd = HorizonEnd.loc[
                HorizonEnd[o9Constants.FORECAST_ITERATION_TYPE].str.split("_").str[1]
                == input_measure
            ]
        else:
            HorizonEnd[temp_FI_type] = HorizonEnd[o9Constants.FORECAST_ITERATION_TYPE]

        # Similarly for ReconciliationTransitionPeriod
        if (
            ReconciliationTransitionPeriod[o9Constants.FORECAST_ITERATION_TYPE]
            .str.contains("_")
            .any()
        ):
            ReconciliationTransitionPeriod["Simplified FI Type"] = (
                ReconciliationTransitionPeriod[o9Constants.FORECAST_ITERATION_TYPE]
                .str.split("_")
                .str[0]
            )
            ReconciliationTransitionPeriod = ReconciliationTransitionPeriod[
                ReconciliationTransitionPeriod[o9Constants.FORECAST_ITERATION_TYPE]
                .str.split("_")
                .str[1]
                == input_measure
            ]
        else:
            ReconciliationTransitionPeriod["Simplified FI Type"] = ReconciliationTransitionPeriod[
                o9Constants.FORECAST_ITERATION_TYPE
            ]

        # Define custom order
        if IterationTypesOrdered:
            custom_order = [item.strip() for item in IterationTypesOrdered.split(",")]
        else:
            custom_order = [
                "Very Short Term",
                "Short Term",
                "Mid Term",
                "Long Term",
                "Very Long Term",
            ]

        # Drop all horizons which does not follow the custom order ST-MT-LT
        HorizonEnd = HorizonEnd[HorizonEnd[temp_FI_type].astype(str).isin(custom_order)]

        # Convert 'Category' column to categorical with custom order
        HorizonEnd[temp_FI_type] = pd.Categorical(
            HorizonEnd[temp_FI_type],
            categories=custom_order,
            ordered=True,
        )

        # Sort DataFrame based on the 'Category' column
        HorizonEnd.sort_values(temp_FI_type, inplace=True)

        # Create mapping between fcst gen time bucket and time dim table
        mapping = {
            "Week": o9Constants.WEEK,
            "Month": o9Constants.MONTH,
            "Planning Month": o9Constants.PLANNING_MONTH,
            "Quarter": o9Constants.QUARTER,
            "Planning Quarter": o9Constants.PLANNING_QUARTER,
        }
        key_mapping = {
            "Week": o9Constants.WEEK_KEY,
            "Month": o9Constants.MONTH_KEY,
            "Planning Month": o9Constants.PLANNING_MONTH_KEY,
            "Quarter": o9Constants.QUARTER_KEY,
            "Planning Quarter": o9Constants.PLANNING_QUARTER_KEY,
        }
        # convert to datetime
        TimeDimension[o9Constants.PARTIAL_WEEK_KEY] = pd.to_datetime(
            TimeDimension[o9Constants.PARTIAL_WEEK_KEY]
        )
        CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY] = pd.to_datetime(
            CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY]
        )

        if ReconcileFlag:
            time_order_dict = {}
            if reconcile_stat:
                HorizonEnd = HorizonEnd.merge(
                    StatFcstPLAgg[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
                    on=o9Constants.FORECAST_ITERATION_TYPE,
                    how="inner",
                )
            else:
                HorizonEnd = HorizonEnd.merge(
                    CMLFcstPLAgg[[o9Constants.FORECAST_ITERATION_TYPE]].drop_duplicates(),
                    on=o9Constants.FORECAST_ITERATION_TYPE,
                    how="inner",
                )
            min_date_key = HorizonEnd[o9Constants.FORECAST_GEN_TIME_BUCKET].values[0]
            col_key = key_mapping[min_date_key]
            end_PW_key = CurrentTimePeriod[col_key][0]
            filter_clause = TimeDimension[col_key] >= end_PW_key
            CMLFcstPLAgg = CMLFcstPLAgg.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, col_key]].drop_duplicates(),
                on=o9Constants.PARTIAL_WEEK,
                how="inner",
            )
            CMLFcstPLAgg_history = CMLFcstPLAgg[CMLFcstPLAgg[col_key] < end_PW_key]
            CMLFcstPLAgg_history = CMLFcstPLAgg_history.merge(
                HorizonEnd[[o9Constants.FORECAST_ITERATION_TYPE, temp_FI_type]].drop_duplicates(),
                on=o9Constants.FORECAST_ITERATION_TYPE,
                how="inner",
            )
            CMLFcstPLAgg_history.sort_values(by=[temp_FI_type, col_key], inplace=True)
            CMLFcstPLAgg_history = CMLFcstPLAgg_history.drop(columns=[col_key])
            CMLFcstPLAgg = CMLFcstPLAgg.drop(columns=[col_key])
            CMLFcstPLAgg_history = CMLFcstPLAgg_history.drop_duplicates(
                subset=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + forecast_level
            )
        else:
            # filter time dimension from current partial week
            filter_clause = (
                TimeDimension[o9Constants.PARTIAL_WEEK_KEY]
                >= CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY].iloc[0]
            )
        TimeDimension = TimeDimension[filter_clause]

        TimeDimension.sort_values(o9Constants.PARTIAL_WEEK_KEY, inplace=True)

        if ReconcileFlag:
            time_order_dict = {}
            end_PW_key = CurrentTimePeriod[o9Constants.PARTIAL_WEEK_KEY][0]

        all_relevant_partial_weeks = list()
        all_df_list = list()
        # Collect partial weeks falling into each forecast iteration type
        for _, the_row in HorizonEnd.iterrows():
            # the_fcst_iteration_type = the_row[o9Constants.FORECAST_ITERATION_TYPE]
            the_fcst_iteration_type = the_row[o9Constants.FORECAST_ITERATION_TYPE]
            the_horizon = int(the_row[o9Constants.HORIZON_END])
            the_fcst_gen_time_bucket = the_row[o9Constants.FORECAST_GEN_TIME_BUCKET]

            logger.debug(f"-- the_fcst_iteration_type : {the_fcst_iteration_type}")
            logger.debug(f"-- the_horizon : {the_horizon}")
            logger.debug(f"-- the_fcst_gen_time_bucket : {the_fcst_gen_time_bucket}")
            if ReconcileFlag:
                time_order_dict[the_fcst_iteration_type] = mapping[the_fcst_gen_time_bucket]
            # collect n periods into future
            the_time_periods = list(
                TimeDimension[mapping[the_fcst_gen_time_bucket]].drop_duplicates().head(the_horizon)
            )

            # collect partial weeks corresponding to n periods
            filter_clause = TimeDimension[mapping[the_fcst_gen_time_bucket]].isin(the_time_periods)
            the_time_dim = TimeDimension[filter_clause]
            if ReconcileFlag:
                if end_PW_key < the_time_dim[o9Constants.PARTIAL_WEEK_KEY].max():
                    end_PW_key = the_time_dim[o9Constants.PARTIAL_WEEK_KEY].max()
                    logger.debug(f"End PW key : {end_PW_key}")
            else:
                the_partial_weeks = list(the_time_dim[o9Constants.PARTIAL_WEEK].drop_duplicates())

                # check if they already exist
                unique_entries = [
                    x for x in the_partial_weeks if x not in all_relevant_partial_weeks
                ]

                # add to master list
                all_relevant_partial_weeks.extend(unique_entries)

                # create dataframe
                the_df = pd.DataFrame({o9Constants.PARTIAL_WEEK: unique_entries})
                the_df[o9Constants.FORECAST_ITERATION_TYPE] = the_fcst_iteration_type

                all_df_list.append(the_df)

                # concat to dataframe
                iteration_type_and_partial_week = concat_to_dataframe(all_df_list)

        if ReconcileFlag:
            filter_clause = TimeDimension[o9Constants.PARTIAL_WEEK_KEY] <= end_PW_key
            TimeDimension = TimeDimension[filter_clause]
            if reconcile_stat:
                custom_order_available_stat = [
                    x
                    for x in custom_order
                    if any(
                        x in forecast
                        for forecast in StatFcstPLAgg[o9Constants.FORECAST_ITERATION_TYPE].unique()
                    )
                ]
                StatFcstRealignment_df = StatFcstPLAgg.merge(
                    TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
                )

                # Identify duplicates based on specific columns
                duplicate_mask_stat = StatFcstRealignment_df.duplicated(
                    subset=forecast_level + [o9Constants.PARTIAL_WEEK],
                    keep=False,
                )

                realigned_df_stat = StatFcstRealignment_df[~duplicate_mask_stat]
                realignment_input_stat = StatFcstRealignment_df[duplicate_mask_stat]

                if len(realignment_input_stat) > 0:
                    recon_method = ReconciliationMethod.loc[0, "Reconciliation Method"]
                    if recon_method == "Smooth Transition":
                        iteration_list = list(time_order_dict.items())
                        output_dict = {}
                        for i in range(len(iteration_list) - 1):
                            current_iter, current_time_col = iteration_list[i]
                            next_iter, next_time_col = iteration_list[i + 1]

                            # Simplified versions for keys
                            current_iter_simple = current_iter.split("_")[0]
                            next_iter_simple = next_iter.split("_")[0]
                            key_prefix = f"{current_iter_simple} to {next_iter_simple}"

                            current_time_key_col = get_key_version_of_time_col(current_time_col)

                            next_time_key_col = get_key_version_of_time_col(next_time_col)

                            current_time = CurrentTimePeriod[current_time_key_col][0]

                            # Fetch horizon_weeks from HorizonEnd
                            horizon_val = int(
                                HorizonEnd[
                                    (
                                        HorizonEnd["Temporary Forecast Iteration"]
                                        == current_iter_simple
                                    )
                                ]["Horizon End"].values[0]
                            )

                            # Fetch transition_months from ReconciliationTransitionPeriod
                            transition_val = int(
                                ReconciliationTransitionPeriod[
                                    (
                                        ReconciliationTransitionPeriod["Simplified FI Type"]
                                        == next_iter_simple
                                    )
                                ]["Reconciliation Transition Period"].values[0]
                            )

                            # Call function
                            horizon_df, transition_df = get_horizon_and_transition_weeks(
                                TimeDimension=TimeDimension,
                                current_time_col=current_time_col,
                                current_time_key_col=current_time_key_col,
                                next_time_col=next_time_col,
                                next_time_key_col=next_time_key_col,
                                current_time=current_time,
                                horizon_periods=horizon_val,
                                transition_periods=transition_val,
                            )

                            # Store with simplified key
                            output_dict[f"{key_prefix}__horizon"] = horizon_df
                            output_dict[f"{key_prefix}__transition"] = transition_df

                        if (
                            CustomRampUpWeight is None
                            or len(CustomRampUpWeight) == 0
                            or str(CustomRampUpWeight).strip().lower() in ["none", "null"]
                        ):
                            # Create fixed weight tables
                            weights_dict = create_fixed_weight_tables(output_dict)
                        else:
                            # Use custom ramp up weights
                            # Always convert string weights to float
                            CustomRampUpWeights = sorted(
                                [
                                    float(w.strip())
                                    for w in CustomRampUpWeight.split(",")
                                    if w.strip() != ""
                                ]
                            )

                            logger.info(f"Parsed CustomRampUpWeights: {CustomRampUpWeights}")
                            weights_dict = create_custom_weight_tables(
                                output_dict, CustomRampUpWeights
                            )

                        version_name = ForecastIterationMasterData[o9Constants.VERSION_NAME].values[
                            0
                        ]
                        ramp_up_weights_df = build_ramp_up_df(
                            version_name=version_name,
                            TimeDimension=TimeDimension,
                            weights_dict=weights_dict,
                            output_dict=output_dict,
                            measure_name=RAMP_UP_WEIGHTS_MEASURE,
                        )

                        fi_type_map = dict(
                            zip(
                                ReconciliationTransitionPeriod[o9Constants.FORECAST_ITERATION_TYPE],
                                ReconciliationTransitionPeriod["Simplified FI Type"],
                            )
                        )

                        reverse_fi_type_map = {v: k for k, v in fi_type_map.items()}

                        ramp_up_weights_df[o9Constants.FORECAST_ITERATION_TYPE] = (
                            ramp_up_weights_df[o9Constants.FORECAST_ITERATION_TYPE]
                            .map(reverse_fi_type_map)
                            .fillna(ramp_up_weights_df[o9Constants.FORECAST_ITERATION_TYPE])
                        )
                        ramp_up_weights_df = ramp_up_weights_df.reset_index()
                        RampUpweightsOutput = ramp_up_weights_df[
                            cols_required_in_ramp_up_weights_output
                        ]
                        # Realign and blend forecasts using parallel processing
                        all_results = Parallel(n_jobs=4, verbose=1)(
                            delayed(realign_and_blend_forecast)(
                                group=group,
                                weights_dict=weights_dict,
                                custom_order=custom_order_available_stat,
                                transition_df=ReconciliationTransitionPeriod,
                                HorizonEnd=HorizonEnd,
                                time_order_dict=time_order_dict,
                                agg_measure=o9Constants.STAT_FCST_PL_AGG,
                                forecast_type_col=o9Constants.FORECAST_ITERATION_TYPE,
                            )
                            for name, group in realignment_input_stat.groupby(forecast_level)
                        )

                        # Concatenate final results
                        realigned_df = pd.concat(all_results, ignore_index=True)
                        realigned_df = realigned_df.drop(
                            columns=[o9Constants.FORECAST_ITERATION_TYPE]
                        )
                        StatFcst = concat_to_dataframe([realigned_df_stat, realigned_df])
                        logger.info("StatFcst Realignment completed with Smooth Transition method")
                    else:
                        # Use the existing append_realignment_last_bucket function
                        all_results_stat = Parallel(n_jobs=4, verbose=1)(
                            delayed(append_realignment_last_bucket)(
                                group,
                                custom_order=custom_order_available_stat,
                                time_order_dict=time_order_dict,
                                agg_measure=o9Constants.STAT_FCST_PL_AGG,
                            )
                            for name, group in realignment_input_stat.groupby(forecast_level)
                        )
                        realigned_df2 = concat_to_dataframe(all_results_stat)
                        StatFcst = concat_to_dataframe([realigned_df_stat, realigned_df2])
                else:
                    StatFcst = realigned_df_stat
            else:
                Output = pd.DataFrame(columns=cols_required_in_output)
            if reconcile_cml:
                custom_order_available_cml = [
                    x
                    for x in custom_order
                    if any(
                        x in forecast
                        for forecast in CMLFcstPLAgg[o9Constants.FORECAST_ITERATION_TYPE].unique()
                    )
                ]
                CMLFcstRealignment_df = CMLFcstPLAgg.merge(
                    TimeDimension, on=o9Constants.PARTIAL_WEEK, how="inner"
                )
                duplicate_mask_cml = CMLFcstRealignment_df.duplicated(
                    subset=forecast_level + [o9Constants.PARTIAL_WEEK],
                    keep=False,
                )
                realigned_df_cml = CMLFcstRealignment_df[~duplicate_mask_cml]
                realignment_input_cml = CMLFcstRealignment_df[duplicate_mask_cml]
                cols_required_in_group = (
                    [
                        o9Constants.VERSION_NAME,
                        o9Constants.FORECAST_ITERATION_TYPE,
                    ]
                    + forecast_level
                    + TimeDimension.columns.to_list()
                )
                if len(realignment_input_cml) > 0:
                    all_results_cml_list = []  # Store results for all agg_measures

                    # Group by forecast_level first
                    for name, group in realignment_input_cml.groupby(forecast_level):
                        group_results = []  # Store results for the current group

                        # Process each agg_measure for this group
                        for (
                            agg_measure,
                            cml_output_measure,
                        ) in cml_measure_mapping.items():
                            if recon_method == "Smooth Transition":
                                # Use the realign_and_blend_forecast function for smooth transition
                                result = realign_and_blend_forecast(
                                    group=group[cols_required_in_group + [agg_measure]],
                                    weights_dict=weights_dict,
                                    custom_order=custom_order_available_cml,
                                    transition_df=ReconciliationTransitionPeriod,
                                    HorizonEnd=HorizonEnd,
                                    time_order_dict=time_order_dict,
                                    agg_measure=agg_measure,  # Use the current measure
                                    forecast_type_col=o9Constants.FORECAST_ITERATION_TYPE,
                                )
                                logger.info(
                                    f"Realignment for CML completed for {agg_measure} with Smooth Transition method"
                                )
                            else:
                                result = append_realignment_last_bucket(
                                    group[cols_required_in_group + [agg_measure]],
                                    custom_order=custom_order_available_cml,
                                    time_order_dict=time_order_dict,
                                    agg_measure=agg_measure,  # Use the current measure
                                )
                            result.rename(
                                columns={agg_measure: cml_output_measure},
                                inplace=True,
                            )
                            group_results += [result]  # Collect results for this group
                        all_columns = set(group_results[0].columns)
                        for df in group_results[1:]:
                            all_columns &= set(df.columns)
                        common_columns = list(all_columns)
                        merged_df = group_results[0]  # Start with the first DataFrame
                        for df in group_results[1:]:
                            merged_df = pd.merge(merged_df, df, on=common_columns, how="inner")
                        # print(merged_df)
                        all_results_cml_list += [merged_df]

                    # Concatenate all results at the end
                    realigned_df_cml_result = concat_to_dataframe(all_results_cml_list)
                    realigned_df_cml.rename(columns=cml_measure_mapping, inplace=True)
                    CMLOutput = concat_to_dataframe([realigned_df_cml_result, realigned_df_cml])
                else:
                    CMLOutput = realigned_df_cml
            else:
                CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)

        else:
            if len(iteration_type_and_partial_week) > 0:
                # join on stat fcst reconciled on iteration type and partial week to filter relevant values for relevant partial weeks
                if reconcile_stat:
                    StatFcst = StatFcstPLAgg.merge(
                        iteration_type_and_partial_week,
                        on=[
                            o9Constants.FORECAST_ITERATION_TYPE,
                            o9Constants.PARTIAL_WEEK,
                        ],
                        how="inner",
                    )
                if reconcile_cml:
                    CMLOutput = CMLFcstPLAgg.merge(
                        iteration_type_and_partial_week,
                        on=[
                            o9Constants.FORECAST_ITERATION_TYPE,
                            o9Constants.PARTIAL_WEEK,
                        ],
                        how="inner",
                    )
            else:
                return Output, CMLOutput

        if reconcile_stat:
            # rename output measure
            StatFcst.rename(
                columns={o9Constants.STAT_FCST_PL_AGG: o9Constants.STAT_FCST},
                inplace=True,
            )
            if output_measure != o9Constants.STAT_FCST:
                if o9Constants.PARTIAL_WEEK not in cols_required_in_output:
                    time_level_output = [
                        x for x in cols_required_in_output if x.startswith("Time")
                    ][0]
                    logger.info(f"Output time level : {time_level_output}")
                    if time_level_output not in StatFcst:
                        StatFcst = StatFcst.merge(
                            TimeDimension[[o9Constants.PARTIAL_WEEK, time_level_output]],
                            on=o9Constants.PARTIAL_WEEK,
                            how="inner",
                        )
                        StatFcst.drop(columns=[o9Constants.PARTIAL_WEEK], inplace=True)
                    StatFcst = (
                        StatFcst.groupby(
                            [x for x in cols_required_in_output if x != output_measure]
                        )
                        .agg({o9Constants.STAT_FCST: "sum"})
                        .reset_index()
                    )
                StatFcst.rename(
                    columns={o9Constants.STAT_FCST: output_measure},
                    inplace=True,
                )
                logger.info(f"Output measure selected : {output_measure}")
                logger.info(
                    f"Output measure granularity : {list(set(cols_required_in_output) & set(forecast_level))}"
                )

            Output = StatFcst
        if reconcile_cml:
            CMLFcstPLAgg_history.rename(columns=cml_measure_mapping, inplace=True)
            CMLOutput.rename(columns=cml_measure_mapping, inplace=True)
            CMLOutput = concat_to_dataframe([CMLOutput, CMLFcstPLAgg_history])
        Output = Output[cols_required_in_output]
        CMLOutput = CMLOutput[cols_required_in_cml_output]
        RampUpweightsOutput = RampUpweightsOutput[cols_required_in_ramp_up_weights_output]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
        CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
        RampUpweightsOutput = pd.DataFrame(columns=cols_required_in_ramp_up_weights_output)
    return Output, CMLOutput, RampUpweightsOutput

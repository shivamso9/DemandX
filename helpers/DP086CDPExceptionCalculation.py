import logging

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def convert_cols_to_data_object_lvl(
    dataframe,
    item_master,
    account_master,
    channel_master,
    region_master,
    location_master,
    pnl_master,
    demand_domain_master,
    common_cols,
    pl_lvl_cols,
):
    """Transform columns to the planning level.

    Apply mapping and rename columns as per data object standards.
    Steps:
    1. Iterates through each dimensional level (item, account, etc.).
    2. Checks if the raw-level column (e.g., 'Item') differs from the planning-level column (e.g., 'PL_Item').
    3. Joins the appropriate master to map raw column to planning column.
    4. Filters rows to ensure only valid planning intersections remain.

    """
    item_col, account_col, channel_col, region_col, location_col, pnl_col, demand_domain_col = (
        common_cols
    )
    (
        pl_item_col,
        pl_account_col,
        pl_channel_col,
        pl_region_col,
        pl_location_col,
        pl_pnl_col,
        pl_demand_domain_col,
    ) = pl_lvl_cols

    if pl_item_col != item_col:
        dataframe = dataframe.merge(item_master, on=item_col, how="left")
        dataframe.drop(item_col, axis=1, inplace=True)

    if pl_account_col != account_col:
        dataframe = dataframe.merge(account_master, on=account_col, how="left")
        dataframe.drop(account_col, axis=1, inplace=True)

    if pl_channel_col != channel_col:
        dataframe = dataframe.merge(channel_master, on=channel_col, how="left")
        dataframe.drop(channel_col, axis=1, inplace=True)

    if pl_region_col != region_col:
        dataframe = dataframe.merge(region_master, on=region_col, how="left")
        dataframe.drop(region_col, axis=1, inplace=True)

    if pl_location_col != location_col:
        dataframe = dataframe.merge(location_master, on=location_col, how="left")
        dataframe.drop(location_col, axis=1, inplace=True)

    if pl_pnl_col != pnl_col:
        dataframe = dataframe.merge(pnl_master, on=pnl_col, how="left")
        dataframe.drop(pnl_col, axis=1, inplace=True)

    if pl_demand_domain_col != demand_domain_col:
        dataframe = dataframe.merge(demand_domain_master, on=demand_domain_col, how="left")
        dataframe.drop(demand_domain_col, axis=1, inplace=True)

    return dataframe


def add_default_params(df, update_params):
    """
    Update multiple columns in the dataframe based on the provided parameters.

    :param df: The dataframe to update.
    :param update_params: A list of dictionaries, where each dictionary contains the necessary information
                            to update one column.
    :return: None (The dataframe is modified in place).
    """
    for param in update_params:
        source_col = param["source_col"]
        target_col = param["target_col"]
        split_str = param["split_str"]
        mapping_cols = param["mapping_cols"]

        fill_df = split_str.split(",")
        # Create the mapping dictionary from columns to split values
        mapping_dict = {key: float(fill_df[i]) for i, key in enumerate(mapping_cols)}

        # Update the target column with the mapped values, filling missing values with existing ones
        df.loc[df[target_col].isna(), target_col] = df.loc[df[target_col].isna(), source_col].map(
            mapping_dict
        )


def subtract_months(current_date, months):
    return current_date - relativedelta(months=months)


def add_months(current_date, months):
    return current_date + relativedelta(months=months)


col_mapping = {
    "CDP Exception Planning Calculation Window": float,
    "CDP Exception Planning Tolerance": float,
    "CDP Exception Planning Min Tolerance Freq": float,
    "CDP Exception Planning FVA Tolerance": float,
    "CDP Exception Scope Source": str,
    "Exception Actual": float,
    "Stat Fcst Exception Abs Error": float,
    "Stat Fcst Exception Accuracy": float,
    "Stat Fcst Exception Accuracy Flag": float,
    "Stat Fcst Exception Persistent Accuracy": str,
    "Stat Fcst Exception Accuracy Persistent Count": float,
    "Stat Fcst Exception Persistent Accuracy Flag": float,
    "Stat Fcst Exception Accuracy LC Flag": float,
    "o9 Says - Stat Accuracy": str,
    "Consensus Fcst Exception Abs Error": float,
    "Consensus Fcst Exception Accuracy": float,
    "Consensus Fcst Exception Accuracy Flag": float,
    "Consensus Fcst Exception Persistent Accuracy": str,
    "Consensus Fcst Exception Accuracy Persistent Count": float,
    "Consensus Fcst Exception Persistent Accuracy Flag": float,
    "Consensus Fcst Exception Accuracy LC Flag": float,
    "o9 Says - Consensus Accuracy": str,
    "Exception FVA": float,
    "Exception FVA Flag": float,
    "Exception Persistent FVA": str,
    "Exception FVA Persistent Count": float,
    "Exception Persistent FVA Flag": float,
    "Exception FVA LC Flag": float,
    "o9 Says - FVA": str,
    "Stat Fcst Exception Error": float,
    "Stat Fcst Exception Bias": float,
    "Stat Fcst Exception Bias Flag": float,
    "Stat Fcst Exception Persistent Bias": str,
    "Stat Fcst Exception Bias Persistent Count": float,
    "Stat Fcst Exception Persistent Bias Flag": float,
    "Stat Fcst Exception Bias LC Flag": float,
    "o9 Says - Stat Bias": str,
    "Consensus Fcst Exception Error": float,
    "Consensus Fcst Exception Bias": float,
    "Consensus Fcst Exception Bias Flag": float,
    "Consensus Fcst Exception Persistent Bias": str,
    "Consensus Fcst Exception Bias Persistent Count": float,
    "Consensus Fcst Exception Persistent Bias Flag": float,
    "Consensus Fcst Exception Bias LC Flag": float,
    "o9 Says - Consensus Bias": str,
    "Stat Fcst Exception": float,
    "Stat Fcst Exception LC": float,
    "Stat Fcst Exception vs LC Deviation %": float,
    "o9 Says - Stat Stability": str,
    "Consensus Fcst Exception": float,
    "Consensus Fcst Exception LC": float,
    "Consensus Fcst Exception vs LC Deviation %": float,
    "o9 Says - Consensus Stability": str,
    "Is Already Touchless?": str,
    "o9 Says - Stat Performance": str,
    "o9 Says - Consensus Performance": str,
    "Exception Recommended Action": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Lag,
    DPExceptionCalculationWindow,
    DPExceptionTolerance,
    MinToleranceFreq,
    FVA,
    ProcessOrder,
    ExceptionParams,
    AssortmentFlag,
    SegmentationFlag,
    AccountMaster,
    ChannelMaster,
    DemandDomainMaster,
    ItemMaster,
    LocationMaster,
    PnLMaster,
    RegionMaster,
    ActualL0,
    SystemAndConsensusFcst,
    StatAndConsensusFcst,
    Touchless,
    CurrentTimePeriod,
    TimeMaster,
    df_keys,
):

    plugin_name = "DP086CDPExceptionCalculation"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # input columns
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    pl_account_col = "Account.[Planning Account]"
    pl_channel_col = "Channel.[Planning Channel]"
    pl_region_col = "Region.[Planning Region]"
    pl_pnl_col = "PnL.[Planning PnL]"
    pl_location_col = "Location.[Planning Location]"
    pl_demand_domain_col = "Demand Domain.[Planning Demand Domain]"

    data_object_col = "Data Object.[Data Object]"
    data_validation_col = "Data Validation.[Data Validation]"
    dm_rule_col = "DM Rule.[Rule]"

    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"

    do_account_lvl_col = "Data Object Account Level"
    do_channel_lvl_col = "Data Object Channel Level"
    do_demand_domain_lvl_col = "Data Object Demand Domain Level"
    do_item_lvl_col = "Data Object Item Level"
    do_pnl_lvl_col = "Data Object PnL Level"
    do_region_lvl_col = "Data Object Region Level"
    do_location_lvl_col = "Data Object Location Level"

    cluster_scope_col = "DP Cluster Scope"
    lag_scope_col = "DP Lag Scope"
    account_scope_col = "DP Account Scope UI"
    channel_scope_col = "DP Channel Scope UI"
    demand_domain_scope_col = "DP Demand Domain Scope UI"
    item_scope_col = "DP Item Scope UI"
    location_scope_col = "DP Location Scope UI"
    pnl_scope_col = "DP PnL Scope UI"
    region_scope_col = "DP Region Scope UI"

    exception_type_col = "DP Exception Type"
    calculation_window_col = "DP Exception Calculation Window"
    tolerance_col = "DP Exception Tolerance"
    min_tolerance_freq_col = "DP Exception Min Tolerance Freq"
    fva_tolerance_col = "DP Exception FVA Tolerance"

    assortment_flag_col = "Assortment Collab DP"
    segmentation_flag_col = "CDP Segmentation Flag"
    cluster_col = "Cluster.[Cluster]"
    lag_col = "Lag.[Lag]"
    process_order_col = "Data Object Process Order"

    system_fcst_m_lag_col = "System Fcst Mature M Lag"
    consensus_fcst_m_lag_col = "Consensus Fcst Mature M Lag"
    actual_col = "Actual L0"
    stat_fcst_col = "Stat Fcst L0 Final"
    stat_fcst_lc_col = "Stat Fcst L0 Final LC"
    consensus_fcst_col = "Consensus Fcst"
    consensus_fcst_lc_col = "Consensus Fcst LC"

    # accuracy intermediate columns
    stat_abs_error = "Stat Abs Error"
    stat_mape = "Stat Mape"
    stat_accuracy = "Stat Accuracy"
    stat_accuracy_flag = "Stat Accuracy Flag"
    persistent_stat_accuracy = "Persistent Stat Accuracy"
    persistent_stat_accuracy_flag = "Persistent Stat Accuracy Flag"
    stat_accuracy_persistent_count = "Stat Accuracy Persistent Count"

    consensus_abs_error = "Consensus Abs error"
    consensus_mape = "Consensus MAPE"
    consensus_accuracy = "Consensus Accuracy"
    consensus_accuracy_flag = "Consensus Accuracy Flag"
    persistent_consensus_accuracy = "Persistent Consenus Accuracy"
    persistent_consensus_accuracy_flag = "Persistent Consensus Accuracy Flag"
    consensus_accuracy_persistent_count = "Consensus Accuracy Persistent Count"

    stat_accuracy_flag_interim = "Stat Accuracy Flag_Interim"
    consensus_accuracy_flag_interim = "Consensus Accuracy Flag_Interim"
    lc_flag_interim = "LC Flag Interim"

    stat_accuracy_lc_flag = "Stat Accuracy LC Flag"
    consensus_accuracy_lc_flag = "Consensus Accuracy LC Flag"
    fva = "FVA"
    fva_flag = "FVA Flag"
    fva_flag_interim = "FVA Flag Interim"
    lc_performance_fva = "LC Performance FVA"
    persistent_fva = "Persistent FVA"
    persistent_fva_flag = "Persistent FVA Flag"
    fva_persistent_count = "FVA Persistent Count"
    fva_lc_flag = "FVA LC Flag"

    stat_accuracy_round = "Stat Accuracy Round"
    consensus_accuracy_round = "Consensus Accuracy Round"
    fva_round = "FVA Round"

    # Bias intermediate columns
    stat_error = "Stat Error"
    stat_bias = "Stat Bias"
    stat_bias_flag = "Stat Bias Flag"
    persistent_stat_bias = "Persistent Stat Bias"
    persistent_stat_bias_flag = "Persistent Stat Bias Flag"
    stat_bias_persistent_count = "Stat Bias Persistent Count"

    consensus_error = "Consensus Error"
    consensus_bias = "Consensus Bias"
    consensus_bias_flag = "Consensus Bias Flag"
    persistent_consensus_bias = "Persistent Consenus Bias"
    persistent_consensus_bias_flag = "Persistent Consensus Bias Flag"
    consensus_bias_persistent_count = "Consensus Bias Persisetnt Count"

    stat_bias_flag_iterim = "Stat Bias Flag_Interim"
    consensus_bias_flag_interim = "Consensus Bias Flag_Interim"
    stat_bias_lc_flag = "Stat Bias LC Flag"
    consensus_bias_lc_flag = "Consensus Bias LC Flag"

    stat_bias_round = "Stat Bias Round"
    consensus_bias_round = "Consensus Bias Round"

    # cocc intermediate columns
    stat_deviation = "Stat Deviation %"
    stat_stability_flag = "Stat Stability Flag"

    consensus_deviation = "Consensus Deviation %"
    consensus_stability_flag = "Consensus Stability Flag"

    # o9 says intermediate columns
    stat_performace = "Stat Performance"
    consensus_performance = "Consensus Performance"
    recommendation = "Recommendation"

    # output columns
    calculation_window_at_pl_lvl_col = "CDP Exception Planning Calculation Window"
    tolerance_at_pl_lvl_col = "CDP Exception Planning Tolerance"
    min_tolerance_freq_at_pl_lvl_col = "CDP Exception Planning Min Tolerance Freq"
    fva_tolerance_at_pl_lvl_col = "CDP Exception Planning FVA Tolerance"
    scope_source_col = "CDP Exception Scope Source"

    accuracy_col = "Accuracy Exception"
    cocc_col = "CoCC Exception"
    bias_col = "Bias Exception"

    accuracy_exception_actual = "Accuracy Exception Actual"
    bias_exception_actual = "Bias Exception Actual"

    stat_exception_abs_error = "Stat Fcst Exception Abs Error"
    stat_exception_accuracy = "Stat Fcst Exception Accuracy"
    stat_exception_accuracy_flag = "Stat Fcst Exception Accuracy Flag"
    stat_exception_persistent_accuracy = "Stat Fcst Exception Persistent Accuracy"
    stat_exception_accuracy_persistent_count = "Stat Fcst Exception Accuracy Persistent Count"
    stat_exception_persistent_accuracy_flag = "Stat Fcst Exception Persistent Accuracy Flag"
    stat_exception_accuracy_lc_flag = "Stat Fcst Exception Accuracy LC Flag"

    stat_exception_error = "Stat Fcst Exception Error"
    stat_exception_bias = "Stat Fcst Exception Bias"
    stat_exception_bias_flag = "Stat Fcst Exception Bias Flag"
    stat_exception_persistent_bias = "Stat Fcst Exception Persistent Bias"
    stat_exception_bias_persistent_count = "Stat Fcst Exception Bias Persistent Count"
    stat_exception_persistent_bias_flag = "Stat Fcst Exception Persistent Bias Flag"
    stat_exception_bias_lc_flag = "Stat Fcst Exception Bias LC Flag"

    stat_exception = "Stat Fcst Exception"
    stat_exception_lc = "Stat Fcst Exception LC"
    stat_exception_vs_lc_deviation = "Stat Fcst Exception vs LC Deviation %"
    o9_says_stat_stability = "o9 Says - Stat Stability"

    consensus_exception_abs_error = "Consensus Fcst Exception Abs Error"
    consensus_exception_accuracy = "Consensus Fcst Exception Accuracy"
    consensus_exception_accuracy_flag = "Consensus Fcst Exception Accuracy Flag"
    consensus_exception_persistent_accuracy = "Consensus Fcst Exception Persistent Accuracy"
    consensus_exception_accuracy_persistent_count = (
        "Consensus Fcst Exception Accuracy Persistent Count"
    )
    consensus_exception_persistent_accuracy_flag = (
        "Consensus Fcst Exception Persistent Accuracy Flag"
    )
    consensus_exception_accuracy_lc_flag = "Consensus Fcst Exception Accuracy LC Flag"

    consensus_exception_error = "Consensus Fcst Exception Error"
    consensus_exception_bias = "Consensus Fcst Exception Bias"
    consensus_exception_bias_flag = "Consensus Fcst Exception Bias Flag"
    consensus_exception_persistent_bias = "Consensus Fcst Exception Persistent Bias"
    consensus_exception_bias_persistent_count = "Consensus Fcst Exception Bias Persistent Count"
    consensus_exception_persistent_bias_flag = "Consensus Fcst Exception Persistent Bias Flag"
    consensus_exception_bias_lc_flag = "Consensus Fcst Exception Bias LC Flag"

    consensus_exception = "Consensus Fcst Exception"
    consensus_exception_lc = "Consensus Fcst Exception LC"
    consensus_exception_vs_lc_deviation = "Consensus Fcst Exception vs LC Deviation %"
    o9_says_consensus_stability = "o9 Says - Consensus Stability"

    exception_fva = "Exception FVA"
    exception_fva_flag = "Exception FVA Flag"
    exception_persistent_fva = "Exception Persistent FVA"
    exception_fva_persistent_count = "Exception FVA Persistent Count"
    exception_persistent_fva_flag = "Exception Persistent FVA Flag"
    exception_fva_lc_flag = "Exception FVA LC Flag"

    exception_recommended_action = "Exception Recommended Action"
    is_already_touchless = "Is Already Touchless"

    o9_says_stat_accuracy = "o9 Says - Stat Accuracy"
    o9_says_consensus_accuracy = "o9 Says - Consensus Accuracy"
    o9_says_fva = "o9 Says - FVA"
    o9_says_stat_bias = "o9 Says - Stat Bias"
    o9_says_consensus_bias = "o9 Says - Consensus Bias"
    o9_says_stat_performance = "o9 Says - Stat Performance"
    o9_says_consensus_performance = "o9 Says - Consensus Performance"

    # column lists
    pl_lvl_cols = [
        pl_item_col,
        pl_account_col,
        pl_channel_col,
        pl_region_col,
        pl_location_col,
        pl_pnl_col,
        pl_demand_domain_col,
    ]
    do_lvl_cols = [
        do_item_lvl_col,
        do_account_lvl_col,
        do_channel_lvl_col,
        do_region_lvl_col,
        do_location_lvl_col,
        do_pnl_lvl_col,
        do_demand_domain_lvl_col,
    ]
    scope_lvl_cols = [
        item_scope_col,
        account_scope_col,
        channel_scope_col,
        region_scope_col,
        location_scope_col,
        pnl_scope_col,
        demand_domain_scope_col,
    ]

    cols_required_in_output1 = (
        [version_col]
        + pl_lvl_cols
        + [data_validation_col, cluster_col, lag_col]
        + [
            calculation_window_at_pl_lvl_col,
            tolerance_at_pl_lvl_col,
            min_tolerance_freq_at_pl_lvl_col,
            fva_tolerance_at_pl_lvl_col,
            scope_source_col,
        ]
    )

    cols_required_in_output2 = (
        [version_col]
        + pl_lvl_cols
        + [lag_col, cluster_col]
        + [
            accuracy_exception_actual,
            bias_exception_actual,
            stat_exception_abs_error,
            stat_exception_accuracy,
            stat_exception_accuracy_flag,
            stat_exception_persistent_accuracy,
            stat_exception_accuracy_persistent_count,
            stat_exception_persistent_accuracy_flag,
            stat_exception_accuracy_lc_flag,
            consensus_exception_abs_error,
            consensus_exception_accuracy,
            consensus_exception_accuracy_flag,
            consensus_exception_persistent_accuracy,
            consensus_exception_accuracy_persistent_count,
            consensus_exception_persistent_accuracy_flag,
            consensus_exception_accuracy_lc_flag,
            exception_fva,
            exception_fva_flag,
            exception_persistent_fva,
            exception_fva_persistent_count,
            exception_persistent_fva_flag,
            exception_fva_lc_flag,
            o9_says_stat_accuracy,
            o9_says_consensus_accuracy,
            o9_says_fva,
            stat_exception_error,
            stat_exception_bias,
            stat_exception_bias_flag,
            stat_exception_persistent_bias,
            stat_exception_bias_persistent_count,
            stat_exception_persistent_bias_flag,
            stat_exception_bias_lc_flag,
            o9_says_stat_bias,
            consensus_exception_error,
            consensus_exception_bias,
            consensus_exception_bias_flag,
            consensus_exception_persistent_bias,
            consensus_exception_bias_persistent_count,
            consensus_exception_persistent_bias_flag,
            consensus_exception_bias_lc_flag,
            o9_says_consensus_bias,
            stat_exception,
            stat_exception_lc,
            stat_exception_vs_lc_deviation,
            o9_says_stat_stability,
            consensus_exception,
            consensus_exception_lc,
            consensus_exception_vs_lc_deviation,
            o9_says_consensus_stability,
            o9_says_stat_performance,
            o9_says_consensus_performance,
            exception_recommended_action,
        ]
    )

    ExceptionParamsOP = pd.DataFrame(columns=cols_required_in_output1)
    ExceptionWithLagOP = pd.DataFrame(columns=cols_required_in_output2)

    try:
        dataframes_to_check = {
            "ProcessOrder": ProcessOrder,
            "ExceptionParams": ExceptionParams,
            "AssortmentFlag": AssortmentFlag,
            "SegmentationFlag": SegmentationFlag,
            "AccountMaster": AccountMaster,
            "ChannelMaster": ChannelMaster,
            "DemandDomainMaster": DemandDomainMaster,
            "ItemMaster": ItemMaster,
            "LocationMaster": LocationMaster,
            "PnLMaster": PnLMaster,
            "RegionMaster": RegionMaster,
            "ActualL0": ActualL0,
            "SystemAndConsensusFcst": SystemAndConsensusFcst,
            "StatAndConsensusFcst": StatAndConsensusFcst,
            "CurrentTimePeriod": CurrentTimePeriod,
            "TimeMaster": TimeMaster,
        }

        if any(df.empty for df in dataframes_to_check.values()):
            empty_dfs = [name for name, df in dataframes_to_check.items() if df.empty]
            logger.warning(
                f"One or more required DataFrames are empty: {', '.join(empty_dfs)}. Exiting: {df_keys} ..."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # rename Column
        ExceptionParams = ExceptionParams.rename(columns={exception_type_col: data_validation_col})

        # current month key
        current_month_key = CurrentTimePeriod[month_key_col].iloc[0]

        # populate segmentation at assortment level
        segmented_data = pd.merge(
            AssortmentFlag,
            SegmentationFlag,
            on=[version_col, pl_account_col, pl_item_col],
            how="left",
        )

        # drop unnecessary columns
        segmented_data.drop([assortment_flag_col, segmentation_flag_col], axis=1, inplace=True)

        # for missing records - update segmentation as "New"
        segmented_data[cluster_col].fillna("New", inplace=True)

        # column that needs to be split and exploded
        cols_to_split = [cluster_scope_col, lag_scope_col] + scope_lvl_cols

        # split the values in the column by commas
        ExceptionParams[cols_to_split] = ExceptionParams[cols_to_split].apply(
            lambda col: col.str.split(",")
        )

        # explode all column into separate rows
        for col in cols_to_split:
            ExceptionParams = ExceptionParams.explode(col, ignore_index=True)

        # remove brackets from dimensions (eg:- [Item].[L2] --> Item.[L2])
        ProcessOrder[do_lvl_cols] = ProcessOrder[do_lvl_cols].replace(
            r"^\[([^\]]+)\]", r"\1", regex=True
        )

        # assign process order to rules for exceptions parameters
        exceptions_params_df = pd.merge(
            ExceptionParams, ProcessOrder, on=[version_col, data_object_col], how="left"
        )

        # drop rows with null process order
        exceptions_params_df = exceptions_params_df.dropna(subset=[process_order_col])

        # ------------------------------------------------ Convert Higher Level Grains to Planning Level -------------------------------------------------

        # group by 'Data Object' and create a separate DataFrame for each unique value of 'Data Object'
        dataframes_dict = {
            data_object: exceptions_params_df[exceptions_params_df[data_object_col] == data_object]
            for data_object in exceptions_params_df[data_object_col].unique()
        }

        merged_df = None

        for data_object, dataframe in dataframes_dict.items():
            first_row = dataframe.iloc[0]

            # extract data object lvl columns
            account_col = first_row[do_account_lvl_col].replace("]", "$DisplayName]")
            channel_col = first_row[do_channel_lvl_col].replace("]", "$DisplayName]")
            demand_domain_col = first_row[do_demand_domain_lvl_col].replace("]", "$DisplayName]")
            item_col = first_row[do_item_lvl_col].replace("]", "$DisplayName]")
            location_col = first_row[do_location_lvl_col].replace("]", "$DisplayName]")
            pnl_col = first_row[do_pnl_lvl_col].replace("]", "$DisplayName]")
            region_col = first_row[do_region_lvl_col].replace("]", "$DisplayName]")

            common_cols = [
                item_col,
                account_col,
                channel_col,
                region_col,
                location_col,
                pnl_col,
                demand_domain_col,
            ]

            dataframe.drop(do_lvl_cols, axis=1, inplace=True)

            # rename cdp scope cols to data object level cols
            rename_dict = dict(zip(scope_lvl_cols, common_cols))
            dataframe = dataframe.rename(columns=rename_dict)

            # select relevant data object level cols from ItemMaster, ChannelMaster, LocationMaster
            item_master = ItemMaster[[item_col, pl_item_col]].drop_duplicates()
            account_master = AccountMaster[[account_col, pl_account_col]].drop_duplicates()
            channel_master = ChannelMaster[[channel_col, pl_channel_col]].drop_duplicates()
            region_master = RegionMaster[[region_col, pl_region_col]].drop_duplicates()
            location_master = LocationMaster[[location_col, pl_location_col]].drop_duplicates()
            pnl_master = PnLMaster[[pnl_col, pl_pnl_col]].drop_duplicates()
            demand_domain_master = DemandDomainMaster[
                [demand_domain_col, pl_demand_domain_col]
            ].drop_duplicates()

            # disaggregate data object level to planning level
            dataframe = convert_cols_to_data_object_lvl(
                dataframe,
                item_master,
                account_master,
                channel_master,
                region_master,
                location_master,
                pnl_master,
                demand_domain_master,
                common_cols,
                pl_lvl_cols,
            )

            if merged_df is None:
                merged_df = dataframe
            else:
                merged_df = pd.concat([merged_df, dataframe], ignore_index=True)

        merged_df.drop(data_object_col, axis=1, inplace=True)
        merged_df.rename(
            columns={cluster_scope_col: cluster_col, lag_scope_col: lag_col}, inplace=True
        )

        # -------------------------------------------------- Filter based on process order ------------------------------------------------------------
        # for rows with the same group, select the row with the highest process order.
        idx_max = merged_df.groupby(
            [version_col, data_validation_col, cluster_col, lag_col] + pl_lvl_cols
        )[process_order_col].idxmax()

        # join to keep the rows with max process order
        merged_df = merged_df.loc[idx_max]

        # Drop process order column
        merged_df = merged_df.drop(columns=[process_order_col])

        # -------------------------------------------------- Filter based on rule number --------------------------------------------------------------

        # extract the numeric part of 'dm_rule_col'
        merged_df["dm_rule_num"] = merged_df[dm_rule_col].str.split("Rule_").str[1].astype(int)

        # for rows with the same group, select the row with the highest rule number.
        idx_max = merged_df.groupby(
            [version_col, data_validation_col, cluster_col, lag_col] + pl_lvl_cols
        )["dm_rule_num"].idxmax()

        # join to keep the rows with max process order
        merged_df = merged_df.loc[idx_max]

        # Drop process order column
        merged_df = merged_df.drop(columns=["dm_rule_num"])

        # --------------------------------------------------- Adding Data Source for Params ------------------------------------------------------------
        data_validation_df = pd.DataFrame({data_validation_col: [accuracy_col, bias_col, cocc_col]})

        # create a temporary 'key' column for the cross join
        data_validation_df["key"] = 1
        segmented_data["key"] = 1

        # perform the cross join
        cross_joined_data = pd.merge(data_validation_df, segmented_data, on="key").drop(
            "key", axis=1
        )

        params_data_source_df = pd.merge(
            cross_joined_data,
            merged_df,
            on=[version_col, data_validation_col, cluster_col] + pl_lvl_cols,
            how="left",
        )

        # fill default exception params
        params_data_source_df[dm_rule_col].fillna("Default", inplace=True)
        params_data_source_df[lag_col].fillna(Lag, inplace=True)
        params_data_source_df.loc[
            params_data_source_df[data_validation_col] == accuracy_col, fva_tolerance_col
        ] = params_data_source_df.loc[
            params_data_source_df[data_validation_col] == accuracy_col, fva_tolerance_col
        ].fillna(
            float(FVA)
        )

        update_params = [
            {
                "source_col": data_validation_col,
                "target_col": calculation_window_col,
                "split_str": DPExceptionCalculationWindow,
                "mapping_cols": [cocc_col, accuracy_col, bias_col],
            },
            {
                "source_col": data_validation_col,
                "target_col": tolerance_col,
                "split_str": DPExceptionTolerance,
                "mapping_cols": [cocc_col, accuracy_col, bias_col],
            },
            {
                "source_col": data_validation_col,
                "target_col": min_tolerance_freq_col,
                "split_str": MinToleranceFreq,
                "mapping_cols": [cocc_col, accuracy_col, bias_col],
            },
        ]

        add_default_params(params_data_source_df, update_params)

        # -----------------------------------------------------Calculating Accuracy Exception-------------------------------------------------------------------

        combined_df = pd.merge(
            SystemAndConsensusFcst,
            ActualL0,
            on=[version_col] + pl_lvl_cols + [month_col],
            how="outer",
        ).fillna(0)

        # filter params for accurancy exception
        accuracy_params = params_data_source_df[
            params_data_source_df[data_validation_col] == accuracy_col
        ]

        # add params for accuracy exception
        combined_df = combined_df.merge(
            accuracy_params, on=[version_col] + pl_lvl_cols + [lag_col], how="inner"
        )

        if combined_df.empty:
            logger.warning(
                "No valid intersection for Accuracy Exception in given window length. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # drop unwanted columns
        combined_df = combined_df.drop(columns=[data_validation_col, dm_rule_col])

        # add current month and last month columns
        combined_df["CurrentMonth"] = pd.to_datetime(current_month_key)
        combined_df["CurrentMonth"] = combined_df["CurrentMonth"].dt.date

        # vectorize the subtract_months function
        vectorized_subtract_months = np.vectorize(subtract_months)

        # apply vectorized function to the DataFrame columns
        combined_df["LastMonth"] = vectorized_subtract_months(
            combined_df["CurrentMonth"].values, combined_df[calculation_window_col].values
        )

        # map month column to month key column
        combined_df = combined_df.merge(
            TimeMaster[[month_col, month_key_col]].drop_duplicates(), on=[month_col], how="left"
        )

        combined_df[month_key_col] = combined_df[month_key_col].dt.date

        # filter based on window length
        combined_df = combined_df[
            (combined_df["LastMonth"] <= combined_df[month_key_col])
            & (combined_df[month_key_col] < combined_df["CurrentMonth"])
        ]

        if combined_df.empty:
            logger.warning(
                "No data found for Accuracy Exception. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # add previous month from current month
        combined_df["PreviousMonth"] = pd.to_datetime(combined_df["CurrentMonth"]) - pd.DateOffset(
            months=1
        )
        combined_df["PreviousMonth"] = combined_df["PreviousMonth"].dt.date

        # interim accuracy calculation
        combined_df[stat_abs_error] = abs(
            combined_df[system_fcst_m_lag_col] - combined_df[actual_col]
        )
        combined_df[stat_mape] = np.where(
            combined_df[actual_col] == 0, 1, combined_df[stat_abs_error] / combined_df[actual_col]
        )
        combined_df[stat_accuracy] = np.where(
            combined_df[stat_abs_error] >= combined_df[actual_col], 0, 1 - combined_df[stat_mape]
        )

        combined_df[consensus_abs_error] = abs(
            combined_df[consensus_fcst_m_lag_col] - combined_df[actual_col]
        )
        combined_df[consensus_mape] = np.where(
            combined_df[actual_col] == 0,
            1,
            combined_df[consensus_abs_error] / combined_df[actual_col],
        )
        combined_df[consensus_accuracy] = np.where(
            combined_df[consensus_abs_error] >= combined_df[actual_col],
            0,
            1 - combined_df[consensus_mape],
        )

        # interim accuracy flag calculation
        combined_df[stat_accuracy_flag_interim] = np.where(
            combined_df[stat_accuracy] >= combined_df[tolerance_col], 1, 0
        )
        combined_df[consensus_accuracy_flag_interim] = np.where(
            combined_df[consensus_accuracy] >= combined_df[tolerance_col], 1, 0
        )
        combined_df[lc_flag_interim] = np.where(
            combined_df[month_key_col] == combined_df["PreviousMonth"], 1, 0
        )

        combined_df[stat_accuracy_lc_flag] = (
            combined_df[stat_accuracy_flag_interim] * combined_df[lc_flag_interim]
        )
        combined_df[consensus_accuracy_lc_flag] = (
            combined_df[consensus_accuracy_flag_interim] * combined_df[lc_flag_interim]
        )

        combined_df[fva] = combined_df[consensus_accuracy] - combined_df[stat_accuracy]
        combined_df[fva_flag_interim] = np.where(
            combined_df[fva] < 0,
            np.where(combined_df[fva] <= combined_df[fva_tolerance_col], -1, 0),
            np.where(combined_df[fva] >= combined_df[fva_tolerance_col], 1, 0),
        )
        combined_df[lc_performance_fva] = (
            combined_df[lc_flag_interim] * combined_df[fva_flag_interim]
        )

        # round off upto 2 decimal places calculation
        combined_df[stat_accuracy_round] = (combined_df[stat_accuracy].round(2) * 100).astype(int)
        combined_df[consensus_accuracy_round] = (
            combined_df[consensus_accuracy].round(2) * 100
        ).astype(int)
        combined_df[fva_round] = (
            combined_df[consensus_accuracy_round] - combined_df[stat_accuracy_round]
        ).round(2)

        combined_df[stat_accuracy_round] = combined_df[stat_accuracy_round].astype(str) + "%"
        combined_df[consensus_accuracy_round] = (
            combined_df[consensus_accuracy_round].astype(str) + "%"
        )
        combined_df[fva_round] = combined_df[fva_round].astype(str) + "%"

        # drop month column
        combined_df.drop(
            columns=[
                month_col,
                month_key_col,
                "CurrentMonth",
                "LastMonth",
                "PreviousMonth",
                calculation_window_col,
            ],
            inplace=True,
        )

        # grouping and applying different aggregation methods to different columns
        agg_result = combined_df.groupby([version_col, lag_col, cluster_col] + pl_lvl_cols).agg(
            **{
                accuracy_exception_actual: (actual_col, "sum"),
                stat_abs_error: (stat_abs_error, "sum"),
                tolerance_col: (tolerance_col, "max"),
                persistent_stat_accuracy: (stat_accuracy_round, "; ".join),
                stat_accuracy_persistent_count: (stat_accuracy_flag_interim, "sum"),
                min_tolerance_freq_col: (min_tolerance_freq_col, "max"),
                stat_accuracy_lc_flag: (stat_accuracy_lc_flag, "sum"),
                consensus_abs_error: (consensus_abs_error, "sum"),
                persistent_consensus_accuracy: (consensus_accuracy_round, "; ".join),
                consensus_accuracy_persistent_count: (consensus_accuracy_flag_interim, "sum"),
                consensus_accuracy_lc_flag: (consensus_accuracy_lc_flag, "sum"),
                fva_tolerance_col: (fva_tolerance_col, "max"),
                persistent_fva: (fva_round, "; ".join),
                fva_persistent_count: (fva_flag_interim, "sum"),
                fva_lc_flag: (lc_performance_fva, "sum"),
            }
        )

        # re calculate some columns
        agg_result[stat_accuracy] = np.where(
            1
            - np.where(
                agg_result[accuracy_exception_actual] == 0,
                1,
                agg_result[stat_abs_error] / agg_result[accuracy_exception_actual],
            )
            < 0,
            0,
            1
            - np.where(
                agg_result[accuracy_exception_actual] == 0,
                1,
                agg_result[stat_abs_error] / agg_result[accuracy_exception_actual],
            ),
        )
        agg_result[stat_accuracy_flag] = np.where(
            agg_result[stat_accuracy] >= agg_result[tolerance_col], 1, 0
        )
        agg_result[persistent_stat_accuracy_flag] = np.where(
            agg_result[stat_accuracy_persistent_count] >= agg_result[min_tolerance_freq_col], 1, 0
        )
        agg_result[o9_says_stat_accuracy] = (
            agg_result[stat_accuracy_flag]
            * agg_result[persistent_stat_accuracy_flag]
            * agg_result[stat_accuracy_lc_flag]
        )

        agg_result[consensus_accuracy] = np.where(
            1
            - np.where(
                agg_result[accuracy_exception_actual] == 0,
                1,
                agg_result[consensus_abs_error] / agg_result[accuracy_exception_actual],
            )
            < 0,
            0,
            1
            - np.where(
                agg_result[accuracy_exception_actual] == 0,
                1,
                agg_result[consensus_abs_error] / agg_result[accuracy_exception_actual],
            ),
        )
        agg_result[consensus_accuracy_flag] = np.where(
            agg_result[consensus_accuracy] >= agg_result[tolerance_col], 1, 0
        )
        agg_result[persistent_consensus_accuracy_flag] = np.where(
            agg_result[consensus_accuracy_persistent_count] >= agg_result[min_tolerance_freq_col],
            1,
            0,
        )
        agg_result[o9_says_consensus_accuracy] = (
            agg_result[consensus_accuracy_flag]
            * agg_result[persistent_consensus_accuracy_flag]
            * agg_result[consensus_accuracy_lc_flag]
        )

        agg_result[fva] = agg_result[consensus_accuracy] - agg_result[stat_accuracy]
        agg_result[fva_flag] = np.where(
            agg_result[fva] < 0,
            np.where(agg_result[fva] <= agg_result[fva_tolerance_col] * -1, -1, 0),
            np.where(agg_result[fva] >= agg_result[fva_tolerance_col], 1, 0),
        )
        agg_result[persistent_fva_flag] = np.where(
            abs(agg_result[fva_persistent_count]) >= agg_result[min_tolerance_freq_col],
            np.where(agg_result[fva_persistent_count] < 0, -1, 1),
            0,
        )
        agg_result[o9_says_fva] = np.where(
            (agg_result[fva_flag] == -1)
            & (agg_result[persistent_fva_flag] == -1)
            & (agg_result[fva_lc_flag] == -1),
            -1,
            np.where(
                (agg_result[fva_flag] == 1)
                & (agg_result[persistent_fva_flag] == 1)
                & (agg_result[fva_lc_flag] == 1),
                1,
                0,
            ),
        )

        agg_result.reset_index(inplace=True)

        accuracy_agg_result = agg_result.copy()

        # -----------------------------------------------------Calculating Bias Exception-------------------------------------------------------------------
        combined_df = pd.merge(
            SystemAndConsensusFcst,
            ActualL0,
            on=[version_col] + pl_lvl_cols + [month_col],
            how="outer",
        ).fillna(0)

        # filter params for accurancy exception
        bias_params = params_data_source_df[params_data_source_df[data_validation_col] == bias_col]

        # add params for accuracy exception
        combined_df = combined_df.merge(
            bias_params, on=[version_col] + pl_lvl_cols + [lag_col], how="inner"
        )

        if combined_df.empty:
            logger.warning(
                "No data found for Bias Exception. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # drop unwanted columns
        combined_df = combined_df.drop(columns=[data_validation_col, dm_rule_col])

        # add current month and last month columns
        combined_df["CurrentMonth"] = pd.to_datetime(current_month_key)
        combined_df["CurrentMonth"] = combined_df["CurrentMonth"].dt.date

        # vectorize the subtract_months function
        vectorized_subtract_months = np.vectorize(subtract_months)

        # apply vectorized function to the DataFrame columns
        combined_df["LastMonth"] = vectorized_subtract_months(
            combined_df["CurrentMonth"].values, combined_df[calculation_window_col].values
        )

        # map month column to month key column
        combined_df = combined_df.merge(
            TimeMaster[[month_col, month_key_col]].drop_duplicates(), on=[month_col], how="left"
        )

        combined_df[month_key_col] = combined_df[month_key_col].dt.date

        # filter based on window length
        combined_df = combined_df[
            (combined_df["LastMonth"] <= combined_df[month_key_col])
            & (combined_df[month_key_col] < combined_df["CurrentMonth"])
        ]

        if combined_df.empty:
            logger.warning(
                "No data found for Bias Exception. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # add previous month from current month
        combined_df["PreviousMonth"] = pd.to_datetime(combined_df["CurrentMonth"]) - pd.DateOffset(
            months=1
        )
        combined_df["PreviousMonth"] = combined_df["PreviousMonth"].dt.date

        # iterim bias calculation
        combined_df[stat_error] = combined_df[system_fcst_m_lag_col] - combined_df[actual_col]
        combined_df[stat_bias] = np.where(
            combined_df[actual_col] == 0, 1, combined_df[stat_error] / combined_df[actual_col]
        )
        combined_df[consensus_error] = (
            combined_df[consensus_fcst_m_lag_col] - combined_df[actual_col]
        )
        combined_df[consensus_bias] = np.where(
            combined_df[actual_col] == 0, 1, combined_df[consensus_error] / combined_df[actual_col]
        )

        # interim bias flag calculation
        combined_df[stat_bias_flag_iterim] = np.where(
            combined_df[stat_bias] > 0,
            np.where(combined_df[stat_bias] >= combined_df[tolerance_col], 1, 0),
            np.where(
                combined_df[stat_bias] < 0,
                np.where(abs(combined_df[stat_bias]) >= combined_df[tolerance_col], -1, 0),
                0,
            ),
        )
        combined_df[consensus_bias_flag_interim] = np.where(
            combined_df[consensus_bias] > 0,
            np.where(combined_df[consensus_bias] >= combined_df[tolerance_col], 1, 0),
            np.where(
                combined_df[consensus_bias] < 0,
                np.where(abs(combined_df[consensus_bias]) >= combined_df[tolerance_col], -1, 0),
                0,
            ),
        )
        combined_df[lc_flag_interim] = np.where(
            combined_df[month_key_col] == combined_df["PreviousMonth"], 1, 0
        )

        combined_df[stat_bias_lc_flag] = (
            combined_df[stat_bias_flag_iterim] * combined_df[lc_flag_interim]
        )
        combined_df[consensus_bias_lc_flag] = (
            combined_df[consensus_bias_flag_interim] * combined_df[lc_flag_interim]
        )

        # round off upto 2 decimal places calculation
        combined_df[stat_bias_round] = ((combined_df[stat_bias].round(2) * 100).astype(int)).astype(
            str
        ) + "%"
        combined_df[consensus_bias_round] = (
            (combined_df[consensus_bias].round(2) * 100).astype(int)
        ).astype(str) + "%"

        # grouping and applying different aggregation methods to different columns
        agg_result = combined_df.groupby([version_col, lag_col, cluster_col] + pl_lvl_cols).agg(
            **{
                bias_exception_actual: (actual_col, "sum"),
                stat_error: (stat_error, "sum"),
                tolerance_col: (tolerance_col, "max"),
                persistent_stat_bias: (stat_bias_round, "; ".join),
                stat_bias_persistent_count: (stat_bias_flag_iterim, "sum"),
                min_tolerance_freq_col: (min_tolerance_freq_col, "max"),
                stat_bias_lc_flag: (stat_bias_lc_flag, "sum"),
                consensus_error: (consensus_error, "sum"),
                persistent_consensus_bias: (consensus_bias_round, "; ".join),
                consensus_bias_persistent_count: (consensus_bias_flag_interim, "sum"),
                consensus_bias_lc_flag: (consensus_bias_lc_flag, "sum"),
            }
        )

        # re calculate some columns
        agg_result[stat_bias] = np.where(
            agg_result[bias_exception_actual] == 0,
            1,
            agg_result[stat_error] / agg_result[bias_exception_actual],
        )
        agg_result[stat_bias_flag] = np.where(
            abs(agg_result[stat_bias]) >= agg_result[tolerance_col],
            np.where(agg_result[stat_bias] < 0, -1, 1),
            0,
        )
        agg_result[persistent_stat_bias_flag] = np.where(
            abs(agg_result[stat_bias_persistent_count]) >= agg_result[min_tolerance_freq_col],
            np.where(agg_result[stat_bias_persistent_count] < 0, -1, 1),
            0,
        )

        agg_result[o9_says_stat_bias] = np.where(
            (agg_result[stat_bias_flag] == -1)
            & (agg_result[persistent_stat_bias_flag] == -1)
            & (agg_result[stat_bias_lc_flag] == -1),
            -1,
            np.where(
                (agg_result[stat_bias_flag] == 1)
                & (agg_result[persistent_stat_bias_flag] == 1)
                & (agg_result[stat_bias_lc_flag] == 1),
                1,
                0,
            ),
        )

        agg_result[consensus_bias] = np.where(
            agg_result[bias_exception_actual] == 0,
            1,
            agg_result[consensus_error] / agg_result[bias_exception_actual],
        )
        agg_result[consensus_bias_flag] = np.where(
            abs(agg_result[consensus_bias]) >= agg_result[tolerance_col],
            np.where(agg_result[consensus_bias] < 0, -1, 1),
            0,
        )

        agg_result[persistent_consensus_bias_flag] = np.where(
            abs(agg_result[consensus_bias_persistent_count]) >= agg_result[min_tolerance_freq_col],
            np.where(agg_result[consensus_bias_persistent_count] < 0, -1, 1),
            0,
        )

        agg_result[o9_says_consensus_bias] = np.where(
            (agg_result[consensus_bias_flag] == -1)
            & (agg_result[persistent_consensus_bias_flag] == -1)
            & (agg_result[consensus_bias_lc_flag] == -1),
            -1,
            np.where(
                (agg_result[consensus_bias_flag] == 1)
                & (agg_result[persistent_consensus_bias_flag] == 1)
                & (agg_result[consensus_bias_lc_flag] == 1),
                1,
                0,
            ),
        )

        agg_result.reset_index(inplace=True)

        bias_agg_result = agg_result.copy()

        # -----------------------------------------------------Calculating Cocc Exception-------------------------------------------------------------------
        # map month column to month key column
        StatAndConsensusFcst = StatAndConsensusFcst.merge(
            TimeMaster[[month_col, month_key_col]].drop_duplicates(), on=[month_col], how="left"
        )

        # sort in descending order based on month_col
        StatAndConsensusFcst = StatAndConsensusFcst.sort_values(by=month_key_col, ascending=False)

        forecast_last_month = StatAndConsensusFcst[month_key_col].iloc[0]
        forecast_last_month = pd.to_datetime(forecast_last_month, errors="coerce") - pd.DateOffset(
            months=1
        )

        StatAndConsensusFcst.drop(month_key_col, axis=1, inplace=True)

        StatAndConsensusFcst = StatAndConsensusFcst.fillna(0)

        # filter params for accurancy exception
        cocc_params = params_data_source_df[params_data_source_df[data_validation_col] == cocc_col]

        # add params for accuracy exception
        combined_df = StatAndConsensusFcst.merge(
            cocc_params, on=[version_col] + pl_lvl_cols, how="inner"
        )

        if combined_df.empty:
            logger.warning(
                "No data found for Cocc Exception. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # drop unwanted columns
        combined_df = combined_df.drop(columns=[data_validation_col, dm_rule_col])

        # add current month and last month columns
        combined_df["CurrentMonth"] = pd.to_datetime(current_month_key)
        combined_df["CurrentMonth"] = combined_df["CurrentMonth"].dt.date

        combined_df["Forecast_Last_Month"] = pd.to_datetime(forecast_last_month, errors="coerce")
        combined_df["Forecast_Last_Month"] = combined_df["Forecast_Last_Month"].dt.date

        # vectorize the add_months function
        vectorized_add_months = np.vectorize(add_months)

        # apply vectorized function to the DataFrame columns
        combined_df["NextMonth"] = vectorized_add_months(
            combined_df["CurrentMonth"].values, combined_df[calculation_window_col].values
        )

        # map month column to month key column
        combined_df = combined_df.merge(
            TimeMaster[[month_col, month_key_col]].drop_duplicates(), on=[month_col], how="left"
        )
        combined_df[month_key_col] = combined_df[month_key_col].dt.date

        # filter for calculation window and also take months upto Forecast_Last_Month
        combined_df = combined_df[
            (combined_df["CurrentMonth"] <= combined_df[month_key_col])
            & (combined_df[month_key_col] <= combined_df["Forecast_Last_Month"])
            & (combined_df[month_key_col] < combined_df["NextMonth"])
        ]

        if combined_df.empty:
            logger.warning(
                "No data found for Cocc Exception. Returning empty DataFrame for ExceptionWithLagOP."
            )
            return ExceptionParamsOP, ExceptionWithLagOP

        # grouping and applying different aggregation methods to different columns
        agg_result = combined_df.groupby([version_col, lag_col, cluster_col] + pl_lvl_cols).agg(
            **{
                stat_fcst_col: (stat_fcst_col, "sum"),
                stat_fcst_lc_col: (stat_fcst_lc_col, "sum"),
                tolerance_col: (tolerance_col, "max"),
                consensus_fcst_col: (consensus_fcst_col, "sum"),
                consensus_fcst_lc_col: (consensus_fcst_lc_col, "sum"),
            }
        )

        # calculate cocc deviation and flag
        agg_result[stat_deviation] = np.where(
            agg_result[stat_fcst_lc_col] == 0,
            1,
            (agg_result[stat_fcst_col] - agg_result[stat_fcst_lc_col])
            / agg_result[stat_fcst_lc_col],
        )
        agg_result[stat_stability_flag] = np.where(
            abs(agg_result[stat_deviation]) < agg_result[tolerance_col], 1, 0
        )

        agg_result[consensus_deviation] = np.where(
            agg_result[consensus_fcst_lc_col] == 0,
            1,
            (agg_result[consensus_fcst_col] - agg_result[consensus_fcst_lc_col])
            / agg_result[consensus_fcst_lc_col],
        )
        agg_result[consensus_stability_flag] = np.where(
            abs(agg_result[consensus_deviation]) < agg_result[tolerance_col], 1, 0
        )

        agg_result.reset_index(inplace=True)

        cocc_agg_result = agg_result.copy()

        exception_output = pd.merge(
            accuracy_agg_result,
            bias_agg_result,
            on=[version_col] + pl_lvl_cols + [lag_col, cluster_col],
            how="outer",
        )

        exception_output = pd.merge(
            exception_output,
            cocc_agg_result,
            on=[version_col] + pl_lvl_cols + [lag_col, cluster_col],
            how="outer",
        )

        exception_output = pd.merge(
            exception_output, Touchless, on=[version_col] + pl_lvl_cols, how="left"
        )

        # ---------------------------------------------------------------o9 Says----------------------------------------------------------------------------------

        exception_output[stat_performace] = np.where(
            (exception_output[o9_says_stat_accuracy] == 1)
            & (exception_output[o9_says_stat_bias] == 0)
            & (exception_output[stat_stability_flag] == 1),
            1,
            0,
        )

        exception_output[consensus_performance] = np.where(
            (exception_output[o9_says_consensus_accuracy] == 1)
            & (exception_output[o9_says_consensus_bias] == 0)
            & (exception_output[consensus_stability_flag] == 1),
            1,
            0,
        )

        exception_output[recommendation] = np.where(
            (exception_output[stat_performace] == 1)
            & (exception_output[is_already_touchless] == "Y"),
            "Continue Touchless",
            np.where(
                (exception_output[stat_performace] == 1)
                & (exception_output[is_already_touchless] == "N")
                & (exception_output[o9_says_fva] <= 0),
                "Remove Adjustments",
                np.where(
                    (exception_output[is_already_touchless] == "N")
                    & (exception_output[o9_says_fva] > 0)
                    & (exception_output[consensus_performance] == 1),
                    "Continue Adjustments",
                    np.where(
                        (exception_output[consensus_performance] == 0)
                        & (exception_output[o9_says_consensus_bias].isin([1, -1])),
                        "Fix Bias",
                        "Review Forecast",
                    ),
                ),
            ),
        )

        mapping = {1: "Good", 0: "Bad"}
        exception_output[o9_says_stat_accuracy] = (
            exception_output[o9_says_stat_accuracy].map(mapping).fillna("")
        )
        exception_output[o9_says_consensus_accuracy] = (
            exception_output[o9_says_consensus_accuracy].map(mapping).fillna("")
        )

        fva_mapping = {1: "Good", -1: "Bad", 0: "Within Limit"}
        exception_output[o9_says_fva] = exception_output[o9_says_fva].map(fva_mapping).fillna("")

        bias_mapping = {1: "Over", -1: "Under", 0: "None"}
        exception_output[o9_says_stat_bias] = (
            exception_output[o9_says_stat_bias].map(bias_mapping).fillna("")
        )
        exception_output[o9_says_consensus_bias] = (
            exception_output[o9_says_consensus_bias].map(bias_mapping).fillna("")
        )

        exception_output[stat_performace] = exception_output[stat_performace].map(mapping)
        exception_output[consensus_performance] = exception_output[consensus_performance].map(
            mapping
        )

        exception_output[stat_stability_flag] = (
            exception_output[stat_stability_flag].map(mapping).fillna("")
        )
        exception_output[consensus_stability_flag] = (
            exception_output[consensus_stability_flag].map(mapping).fillna("")
        )

        # ----------------------------------------------------------------Outputs---------------------------------------------------------------------------------

        cols_in_param_data_source = [
            calculation_window_col,
            tolerance_col,
            min_tolerance_freq_col,
            fva_tolerance_col,
            dm_rule_col,
        ]

        cols_in_output1 = [
            calculation_window_at_pl_lvl_col,
            tolerance_at_pl_lvl_col,
            min_tolerance_freq_at_pl_lvl_col,
            fva_tolerance_at_pl_lvl_col,
            scope_source_col,
        ]

        rename_dict = dict(zip(cols_in_param_data_source, cols_in_output1))
        params_data_source_df = params_data_source_df.rename(columns=rename_dict)

        ExceptionParamsOP = params_data_source_df[cols_required_in_output1]

        cols_in_exception_output = [
            stat_abs_error,
            stat_accuracy,
            stat_accuracy_flag,
            persistent_stat_accuracy,
            stat_accuracy_persistent_count,
            persistent_stat_accuracy_flag,
            stat_accuracy_lc_flag,
            o9_says_stat_accuracy,
            consensus_abs_error,
            consensus_accuracy,
            consensus_accuracy_flag,
            persistent_consensus_accuracy,
            consensus_accuracy_persistent_count,
            persistent_consensus_accuracy_flag,
            consensus_accuracy_lc_flag,
            o9_says_consensus_accuracy,
            fva,
            fva_flag,
            persistent_fva,
            fva_persistent_count,
            persistent_fva_flag,
            fva_lc_flag,
            o9_says_fva,
            stat_error,
            stat_bias,
            stat_bias_flag,
            persistent_stat_bias,
            stat_bias_persistent_count,
            persistent_stat_bias_flag,
            stat_bias_lc_flag,
            o9_says_stat_bias,
            consensus_error,
            consensus_bias,
            consensus_bias_flag,
            persistent_consensus_bias,
            consensus_bias_persistent_count,
            persistent_consensus_bias_flag,
            consensus_bias_lc_flag,
            o9_says_consensus_bias,
            stat_fcst_col,
            stat_fcst_lc_col,
            stat_deviation,
            stat_stability_flag,
            consensus_fcst_col,
            consensus_fcst_lc_col,
            consensus_deviation,
            consensus_stability_flag,
            stat_performace,
            consensus_performance,
            recommendation,
        ]

        cols_in_output2 = [
            stat_exception_abs_error,
            stat_exception_accuracy,
            stat_exception_accuracy_flag,
            stat_exception_persistent_accuracy,
            stat_exception_accuracy_persistent_count,
            stat_exception_persistent_accuracy_flag,
            stat_exception_accuracy_lc_flag,
            o9_says_stat_accuracy,
            consensus_exception_abs_error,
            consensus_exception_accuracy,
            consensus_exception_accuracy_flag,
            consensus_exception_persistent_accuracy,
            consensus_exception_accuracy_persistent_count,
            consensus_exception_persistent_accuracy_flag,
            consensus_exception_accuracy_lc_flag,
            o9_says_consensus_accuracy,
            exception_fva,
            exception_fva_flag,
            exception_persistent_fva,
            exception_fva_persistent_count,
            exception_persistent_fva_flag,
            exception_fva_lc_flag,
            o9_says_fva,
            stat_exception_error,
            stat_exception_bias,
            stat_exception_bias_flag,
            stat_exception_persistent_bias,
            stat_exception_bias_persistent_count,
            stat_exception_persistent_bias_flag,
            stat_exception_bias_lc_flag,
            o9_says_stat_bias,
            consensus_exception_error,
            consensus_exception_bias,
            consensus_exception_bias_flag,
            consensus_exception_persistent_bias,
            consensus_exception_bias_persistent_count,
            consensus_exception_persistent_bias_flag,
            consensus_exception_bias_lc_flag,
            o9_says_consensus_bias,
            stat_exception,
            stat_exception_lc,
            stat_exception_vs_lc_deviation,
            o9_says_stat_stability,
            consensus_exception,
            consensus_exception_lc,
            consensus_exception_vs_lc_deviation,
            o9_says_consensus_stability,
            o9_says_stat_performance,
            o9_says_consensus_performance,
            exception_recommended_action,
        ]

        rename_dict = dict(zip(cols_in_exception_output, cols_in_output2))
        exception_output = exception_output.rename(columns=rename_dict)

        string_columns = [
            column for column, dtype in exception_output.dtypes.items() if dtype == object
        ]

        exception_output[string_columns] = exception_output[string_columns].fillna("")

        ExceptionWithLagOP = exception_output[cols_required_in_output2]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        ExceptionParamsOP = pd.DataFrame(columns=cols_required_in_output1)
        ExceptionWithLagOP = pd.DataFrame(columns=cols_required_in_output2)

    return ExceptionParamsOP, ExceptionWithLagOP

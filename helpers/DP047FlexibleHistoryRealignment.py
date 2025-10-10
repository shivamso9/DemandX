import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import is_dimension, remove_first_brackets

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """Constants used in the plugin."""

    # Input Columns
    VERSION = o9Constants.VERSION_NAME
    ACCOUNT = o9Constants.ACCOUNT
    CHANNEL = o9Constants.CHANNEL
    DEMAND_DOMAIN = o9Constants.DEMAND_DOMAIN
    ITEM = o9Constants.ITEM
    LOCATION = o9Constants.LOCATION
    PNL = o9Constants.PNL
    REGION = o9Constants.REGION

    DM_RULE = "DM Rule.[Rule]"
    DATA_OBJECT = "Data Object.[Data Object]"
    DATA_OBJECT_TYPE = "Data Object.[Data Object Type]"
    SEQUENCE = o9Constants.SEQUENCE
    TIME_DAY = o9Constants.DAY
    TIME_DAY_KEY = o9Constants.DAY_KEY
    TIME_WEEK = o9Constants.WEEK

    FROM_ACCOUNT = "DP From Account Scope"
    FROM_CHANNEL = "DP From Channel Scope"
    FROM_DEMAND_DOMAIN = "DP From Demand Domain Scope"
    FROM_ITEM = "DP From Item Scope"
    FROM_LOCATION = "DP From Location Scope"
    FROM_PNL = "DP From PnL Scope"
    FROM_REGION = "DP From Region Scope"

    TO_ACCOUNT = "DP To Account Scope"
    TO_CHANNEL = "DP To Channel Scope"
    TO_DEMAND_DOMAIN = "DP To Demand Domain Scope"
    TO_ITEM = "DP To Item Scope"
    TO_LOCATION = "DP To Location Scope"
    TO_PNL = "DP To PnL Scope"
    TO_REGION = "DP To Region Scope"

    DO_ACCOUNT = "Data Object Account Level"
    DO_CHANNEL = "Data Object Channel Level"
    DO_DEMAND_DOMAIN = "Data Object Demand Domain Level"
    DO_ITEM = "Data Object Item Level"
    DO_LOCATION = "Data Object Location Level"
    DO_PNL = "Data Object PnL Level"
    DO_REGION = "Data Object Region Level"

    CONVERSION_FACTOR = "DP Conversion Factor"
    REALIGNMENT_PERCENTAGE = "DP Realignment Percentage"
    REALIGNMENT_PERCENTAGE_WEIGHTED = "DP Realignment Percentage Weighted"
    REALIGNMENT_STATUS = "DP Full History Realignment Status"
    RULE_SEQUENCE = "DP Realignment Rule Sequence"
    VALIDITY_STATUS = "History Realignment Status"
    TRANSITION_DATE = "Transition Start Date"
    SELF_TRANSITION_FLAG = "Self Transition Flag"
    INCLUDE_IN_HISTORY_REALIGNMENT = "Include in History Realignment"
    HISTORY_REALIGNMENT_ACTIVE_PERIOD = "History Realignment Active Period"

    ACTUAL_RAW_SUFFIX = " Raw"
    ACTUAL_INPUT_SUFFIX = " Input"

    # Intermediate Columns
    BALANCE_PERCENTANGE = "Balance Percentage"
    REALIGNMENT_TYPES = "Realignment Types"
    REALGINMENT_SUM = "Realignment Sum"

    REALIGNMENT_START_DATE = "Realignment Start Date"
    REALIGNMENT_END_DATE = "Realignment End Date"
    BOUNDARY = "boundary"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"

    # Column Lists
    LEAF_DIMENSIONS = [ACCOUNT, CHANNEL, DEMAND_DOMAIN, ITEM, LOCATION, PNL, REGION]

    FROM_DIMENSIONS = [
        FROM_ACCOUNT,
        FROM_CHANNEL,
        FROM_DEMAND_DOMAIN,
        FROM_ITEM,
        FROM_LOCATION,
        FROM_PNL,
        FROM_REGION,
    ]

    TO_DIMENSIONS = [
        TO_ACCOUNT,
        TO_CHANNEL,
        TO_DEMAND_DOMAIN,
        TO_ITEM,
        TO_LOCATION,
        TO_PNL,
        TO_REGION,
    ]

    DO_COLUMNS = [DO_ACCOUNT, DO_CHANNEL, DO_DEMAND_DOMAIN, DO_ITEM, DO_LOCATION, DO_PNL, DO_REGION]


@timed
def calculate_balance_percentage(RealignmentRules):
    """Compute the balance percentage and normalize the realignment percentage based on the sum of realignment percentages."""
    try:
        RealignmentRules[Constants.CONVERSION_FACTOR] = RealignmentRules[
            Constants.CONVERSION_FACTOR
        ].fillna(1)
        RealignmentRules[Constants.RULE_SEQUENCE] = RealignmentRules[
            Constants.RULE_SEQUENCE
        ].fillna(1)

        RealignmentRules = RealignmentRules.fillna(0)

        RealignmentRules = RealignmentRules[RealignmentRules[Constants.VALIDITY_STATUS] == 1]

        # remove intersections where 'FROM_grains' == 'TO_grains'
        condition = (
            RealignmentRules[Constants.FROM_DIMENSIONS[0]]
            == RealignmentRules[Constants.TO_DIMENSIONS[0]]
        )

        for f_col, t_col in zip(Constants.FROM_DIMENSIONS[1:], Constants.TO_DIMENSIONS[1:]):
            condition &= RealignmentRules[f_col] == RealignmentRules[t_col]

        RealignmentRules = RealignmentRules[~condition]

        grouping_cols = [
            Constants.VERSION,
            Constants.DATA_OBJECT,
            Constants.TRANSITION_DATE,
            Constants.RULE_SEQUENCE,
        ] + Constants.FROM_DIMENSIONS

        starts = RealignmentRules[grouping_cols + [Constants.REALIGNMENT_START_DATE]].rename(
            columns={Constants.REALIGNMENT_START_DATE: Constants.BOUNDARY}
        )
        ends = RealignmentRules[grouping_cols + [Constants.REALIGNMENT_END_DATE]].rename(
            columns={Constants.REALIGNMENT_END_DATE: Constants.BOUNDARY}
        )

        all_boundaries = (
            pd.concat([starts, ends], ignore_index=True)
            .drop_duplicates()
            .sort_values(by=grouping_cols + [Constants.BOUNDARY])
        )

        # The start of a period is a boundary, and the end is the next boundary.
        all_boundaries[Constants.PERIOD_START] = all_boundaries[Constants.BOUNDARY]
        all_boundaries[Constants.PERIOD_END] = all_boundaries.groupby(grouping_cols)[
            Constants.BOUNDARY
        ].shift(-1)

        periods = all_boundaries.dropna(subset=[Constants.PERIOD_END])
        periods = periods[periods[Constants.PERIOD_START] < periods[Constants.PERIOD_END]]

        # Merge original rules with their corresponding time periods ---
        merged_df = pd.merge(RealignmentRules, periods, on=grouping_cols)

        # A rule is active if its [start, end) interval overlaps with the period's interval.
        active_rules_mask = (
            merged_df[Constants.REALIGNMENT_START_DATE] <= merged_df[Constants.PERIOD_START]
        ) & (merged_df[Constants.REALIGNMENT_END_DATE] >= merged_df[Constants.PERIOD_END])

        processed_df = merged_df[active_rules_mask].copy()

        # Assign the new period start and end dates.
        processed_df[Constants.REALIGNMENT_START_DATE] = processed_df[Constants.PERIOD_START]
        processed_df[Constants.REALIGNMENT_END_DATE] = processed_df[Constants.PERIOD_END]

        RealignmentRules = processed_df[RealignmentRules.columns].reset_index(drop=True)

        RealignmentRules[Constants.REALGINMENT_SUM] = RealignmentRules.groupby(
            [
                Constants.VERSION,
                Constants.DATA_OBJECT,
                Constants.TRANSITION_DATE,
                Constants.RULE_SEQUENCE,
                Constants.REALIGNMENT_START_DATE,
                Constants.REALIGNMENT_END_DATE,
            ]
            + Constants.FROM_DIMENSIONS
        )[Constants.REALIGNMENT_PERCENTAGE].transform("sum")

        # balance Percentage (if sum < 1)
        RealignmentRules[Constants.BALANCE_PERCENTANGE] = np.where(
            RealignmentRules[Constants.REALGINMENT_SUM] < 1,
            1 - RealignmentRules[Constants.REALGINMENT_SUM],
            0,
        )

        # normalize Realignment Percentage only where sum > 1
        RealignmentRules[Constants.REALIGNMENT_PERCENTAGE] = np.where(
            RealignmentRules[Constants.REALGINMENT_SUM] > 1,
            RealignmentRules[Constants.REALIGNMENT_PERCENTAGE]
            / RealignmentRules[Constants.REALGINMENT_SUM],
            RealignmentRules[Constants.REALIGNMENT_PERCENTAGE],
        )

        # drop intermediate column
        RealignmentRules.drop(columns=Constants.REALGINMENT_SUM, inplace=True)

        return RealignmentRules

    except Exception as e:
        raise RuntimeError(f"Error in calculate_balance_percentage: {e}") from e


@timed
def apply_self_transitions(RealignmentRules):
    """Apply self-transitions to the RealignmentRules DataFrame based on balance_percentage."""
    try:
        # filter and copy relevant columns
        self_transitions = RealignmentRules[RealignmentRules[Constants.BALANCE_PERCENTANGE] >= 0][
            [Constants.VERSION, Constants.DATA_OBJECT, Constants.DM_RULE, Constants.SEQUENCE]
            + Constants.FROM_DIMENSIONS
            + [
                Constants.BALANCE_PERCENTANGE,
                Constants.CONVERSION_FACTOR,
                Constants.REALIGNMENT_STATUS,
                Constants.VALIDITY_STATUS,
                Constants.TRANSITION_DATE,
                Constants.RULE_SEQUENCE,
                Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD,
                Constants.REALIGNMENT_START_DATE,
                Constants.REALIGNMENT_END_DATE,
            ]
        ].copy()

        # set target = source for dimensions
        for i in range(len(Constants.FROM_DIMENSIONS)):
            self_transitions[Constants.TO_DIMENSIONS[i]] = self_transitions[
                Constants.FROM_DIMENSIONS[i]
            ]

        # prepare for row expansion
        all_columns = set(self_transitions.columns)
        key_columns = list(all_columns - {Constants.BALANCE_PERCENTANGE})

        rows = []
        for _, row in self_transitions.iterrows():
            base_data = {col: row[col] for col in key_columns}
            balance_val = row[Constants.BALANCE_PERCENTANGE]

            rows.append(
                {
                    **base_data,
                    Constants.REALIGNMENT_PERCENTAGE: balance_val,
                }
            )

        # create dataFrame from expanded rows
        self_transitions = pd.DataFrame(rows)

        # flag as self transitions
        self_transitions[Constants.SELF_TRANSITION_FLAG] = 1

        # prepare and merge with original
        RealignmentRules.drop([Constants.BALANCE_PERCENTANGE], axis=1, inplace=True)
        RealignmentRules[Constants.SELF_TRANSITION_FLAG] = 0

        RealignmentRules = pd.concat([RealignmentRules, self_transitions], ignore_index=True)
        RealignmentRules[Constants.SELF_TRANSITION_FLAG] = RealignmentRules[
            Constants.SELF_TRANSITION_FLAG
        ].astype(object)

        return RealignmentRules

    except Exception as e:
        raise RuntimeError(f"Error in apply_self_transitions: {e}") from e


@timed
def process_realignments(
    AttributeMapping,
    RealignmentRules,
):
    """Concatenate the realignment types for each data object."""
    try:
        AttributeMapping[Constants.REALIGNMENT_TYPES] = AttributeMapping[
            Constants.DO_COLUMNS
        ].apply(lambda row: ", ".join(row.dropna().astype(str)), axis=1)

        # Drop unnecessary columns from AttributeMapping
        AttributeMapping.drop(columns=Constants.DO_COLUMNS, inplace=True)

        # merge RealignmentRules with AttributeMapping
        RealignmentRules = pd.merge(
            RealignmentRules,
            AttributeMapping,
            on=[Constants.VERSION, Constants.DATA_OBJECT],
            how="inner",
        )

        return RealignmentRules
    except Exception as e:
        raise RuntimeError(f"Error in process_realignments: {e}") from e


@timed
def get_realigned_data(
    Actual,
    realignment_rules,
    StartDate,
    leaf_level_cols,
    from_cols,
    to_cols,
    measures,
):
    """Realign data based on the rules."""
    try:
        if realignment_rules.empty:
            return Actual

        output_dimensions = [dim for dim in Actual.columns if dim not in measures]
        cols_required_in_output = Actual.columns.tolist()

        output_leaf_dimensions = [
            x
            for x in output_dimensions
            if x != Constants.TIME_DAY and x != Constants.TIME_DAY_KEY and x != Constants.VERSION
        ]

        remaining_cols = set(output_leaf_dimensions) - set(leaf_level_cols)
        remaining_dims_list = list(remaining_cols)

        # filters actuals based on history refresh buckets
        if realignment_rules[Constants.REALIGNMENT_STATUS].iloc[0] == 0:
            mask = Actual[Constants.TIME_DAY_KEY] < StartDate
        else:
            mask = Actual[Constants.TIME_DAY_KEY] >= StartDate

        relevant_actuals = Actual[mask]

        # filtering Actuals to only contain relevant_keys combinations that are in Realignment's from and to
        logger.info("filtering out intersections against for and to intersections ...")
        relevant_actuals.set_index(leaf_level_cols, inplace=True)
        realignment_rules.set_index(from_cols, inplace=True)
        filtered_df1 = relevant_actuals[relevant_actuals.index.isin(realignment_rules.index)]
        filtered_df1.reset_index(inplace=True)
        realignment_rules.reset_index(inplace=True)

        realignment_rules.set_index(to_cols, inplace=True)
        filtered_df2 = relevant_actuals[relevant_actuals.index.isin(realignment_rules.index)]
        filtered_df2.reset_index(inplace=True)
        relevant_actuals = pd.concat(
            [filtered_df1, filtered_df2], ignore_index=True
        ).drop_duplicates()
        FilActual_copy = relevant_actuals.copy()
        realignment_rules.reset_index(inplace=True)
        relevant_actuals.reset_index(inplace=True)

        if relevant_actuals.empty:
            logger.warning("no intersections present for realignment ...")

            return Actual

        relevant_actuals.drop_duplicates(inplace=True)

        # merge realignment_rules with relevant_actuals on from columns to get data that needs to be realigned
        logger.info("getting relevant data for realignment ...")
        relevant_actuals = relevant_actuals.merge(
            realignment_rules,
            left_on=[Constants.VERSION] + leaf_level_cols,
            right_on=[Constants.VERSION] + from_cols,
            how="inner",
        )

        # Filter for the history realignment active period
        logger.info("Applying active period filter. Keeping data before ...")
        relevant_actuals = relevant_actuals[
            (
                relevant_actuals[Constants.TIME_DAY_KEY]
                < relevant_actuals[Constants.REALIGNMENT_END_DATE]
            )
            & (
                relevant_actuals[Constants.TIME_DAY_KEY]
                >= relevant_actuals[Constants.REALIGNMENT_START_DATE]
            )
        ]

        if relevant_actuals.empty:
            logger.warning(
                "no intersections present for realignment after filtering for history realignment active period..."
            )

            return Actual

        # getting intersections present in relevant_actuals
        # based on this, later will get the extra intersections which were not considered for realignment
        intersections_realigned = relevant_actuals[
            remaining_dims_list + from_cols + [Constants.TIME_DAY_KEY, Constants.TIME_DAY]
        ].drop_duplicates()
        intersections_realigned.rename(columns=dict(zip(from_cols, leaf_level_cols)), inplace=True)

        logger.info(
            "nulling values for irrelevant intersections between corresponding time frame ..."
        )
        # getting count for each partial week
        group_by_cols = leaf_level_cols + [Constants.TIME_DAY_KEY, Constants.TIME_DAY]

        relevant_actuals["count"] = relevant_actuals.groupby(group_by_cols)[
            Constants.SELF_TRANSITION_FLAG
        ].transform("count")

        # nulling out realignment percentage
        condition = (
            (relevant_actuals[Constants.SELF_TRANSITION_FLAG] == 1)
            & (relevant_actuals[Constants.REALIGNMENT_PERCENTAGE] == 1)
            & (relevant_actuals["count"] > 1)
        )
        relevant_actuals[Constants.REALIGNMENT_PERCENTAGE] = np.where(
            condition,
            0,
            relevant_actuals[Constants.REALIGNMENT_PERCENTAGE],
        )

        for measure in measures:
            relevant_actuals[measure] = (
                relevant_actuals[measure]
                * relevant_actuals[Constants.REALIGNMENT_PERCENTAGE]
                * relevant_actuals[Constants.CONVERSION_FACTOR]
            )

        # replacing leaf level values with corresponding to column values
        relevant_actuals[leaf_level_cols] = relevant_actuals[to_cols].values

        relevant_actuals = relevant_actuals[cols_required_in_output]

        logger.info("getting intersections not considered for realignment ...")
        extra_intersections = FilActual_copy.merge(
            intersections_realigned,
            how="outer",
            indicator=True,
        )
        extra_intersections = extra_intersections[
            extra_intersections["_merge"] == "left_only"
        ].drop(columns="_merge")
        extra_intersections = extra_intersections[cols_required_in_output]

        output_df = pd.concat([relevant_actuals, extra_intersections])

        output_df = output_df.groupby(
            output_dimensions,
            observed=True,
            as_index=False,
        )[measures].sum()

        # update original Actual DataFrame with the realigned data
        Actual.set_index(output_dimensions, inplace=True)
        output_df.set_index(output_dimensions, inplace=True)

        Actual.update(output_df[measures])

        # concatenate rows from output that are not in Actual
        new_rows = output_df[~output_df.index.isin(Actual.index)]

        new_rows = new_rows.reset_index()
        Actual = Actual.reset_index()

        Actual = pd.concat([Actual, new_rows])

        return Actual

    except Exception as e:
        raise RuntimeError(f"Error in get_realigned_data: {e}") from e


@timed
def update_realignment_status(realignment_rules, group):
    """Update the realignment status in `realignment_rules` based on the `group` DataFrame."""
    try:
        # mark the group rows as updated
        group[Constants.REALIGNMENT_STATUS] = 1
        group = group.rename(columns={Constants.REALIGNMENT_STATUS: "updated_flag"})

        # merge updated flag into realignment_rules
        merge_cols = [
            Constants.VERSION,
            Constants.DATA_OBJECT,
            Constants.TRANSITION_DATE,
            Constants.RULE_SEQUENCE,
            Constants.REALIGNMENT_TYPES,
        ]
        req_cols = group[merge_cols + ["updated_flag"]].drop_duplicates()

        realignment_rules = realignment_rules.merge(req_cols, on=merge_cols, how="left")

        # update realignment status where updated_flag is 1
        realignment_rules.loc[
            realignment_rules["updated_flag"] == 1, Constants.REALIGNMENT_STATUS
        ] = 1

        realignment_rules.drop("updated_flag", axis=1, inplace=True)
        group = group.rename(columns={"updated_flag": Constants.REALIGNMENT_STATUS})

        return realignment_rules, group

    except Exception as e:
        raise RuntimeError(f"Error in update_realignment_status: {e}") from e


col_mapping = {
    "DP Realignment Percentage Weighted": float,
    "DP Full History Realignment Status": float,
}


@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    Actual,
    Full_Actual,
    AttributeMapping,
    HistoryRefreshBuckets,
    RealignmentRules,
    ItemMaster,
    RegionMaster,
    AccountMaster,
    ChannelMaster,
    PnlMaster,
    DemandDomainMaster,
    LocationMaster,
    TimeDimension,
    df_keys,
):
    """Implement the DP047 Flexible History Realignment plugin."""
    plugin_name = "DP047FlexibleHistoryRealignment"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    if Actual is None or Full_Actual is None or RealignmentRules is None:
        logger.warning(
            "Input DataFrame(s) Actual or RealignmentRules are None for slice '%s'.", df_keys
        )
        return pd.DataFrame(), pd.DataFrame()

    # drop week column
    Actual.drop(columns=[Constants.TIME_WEEK], axis=1, inplace=True)
    Full_Actual.drop(columns=[Constants.TIME_WEEK], axis=1, inplace=True)

    # define columns required in output DataFrames
    actual_dimensions = [x for x in Full_Actual.columns if is_dimension(x)]
    actual_dimensions = [
        x for x in Full_Actual.columns if is_dimension(x) and x != Constants.TIME_DAY
    ]
    if Constants.TIME_DAY in Full_Actual.columns:
        actual_dimensions.append(Constants.TIME_DAY)

    measures_to_consider = [x for x in Full_Actual.columns if x not in actual_dimensions]

    actual_measures = [x.replace(Constants.ACTUAL_RAW_SUFFIX, "") for x in measures_to_consider]
    actual_raw_measures = [x + Constants.ACTUAL_RAW_SUFFIX for x in actual_measures]
    actual_input_measures = [x + Constants.ACTUAL_INPUT_SUFFIX for x in actual_measures]
    actual_output_cols = actual_dimensions + actual_measures + actual_input_measures

    Actual = Actual[actual_dimensions + actual_raw_measures]
    Full_Actual = Full_Actual[actual_dimensions + actual_raw_measures]

    realignment_dimensions = [x for x in RealignmentRules.columns if is_dimension(x)]
    realignment_output_cols = realignment_dimensions + [
        Constants.REALIGNMENT_PERCENTAGE_WEIGHTED,
        Constants.REALIGNMENT_STATUS,
    ]

    # initialize output DataFrames
    ActualOP = pd.DataFrame(columns=actual_output_cols)
    RealignmentOP = pd.DataFrame(columns=realignment_output_cols)

    try:
        dataframes = {
            "HistoryRefreshBuckets": HistoryRefreshBuckets,
            "AttributeMapping": AttributeMapping,
            "ItemMaster": ItemMaster,
            "RegionMaster": RegionMaster,
            "AccountMaster": AccountMaster,
            "ChannelMaster": ChannelMaster,
            "PnlMaster": PnlMaster,
            "DemandDomainMaster": DemandDomainMaster,
            "LocationMaster": LocationMaster,
        }

        # check if any input dataframes is empty
        empty_dfs = [name for name, df in dataframes.items() if df.empty]

        if empty_dfs:
            logger.warning(
                f"Input DataFrame(s) {', '.join(empty_dfs)} are empty for slice '{df_keys}'."
            )
            logger.warning("No further execution for this slice.")
            return RealignmentOP, ActualOP

        # get the HistoryRefreshPeriod Start Date
        StartDate = pd.to_datetime(HistoryRefreshBuckets[Constants.TIME_DAY]).min()

        # check if RealignmentRules is empty
        if RealignmentRules.empty:
            logger.warning("Realignment Rules DataFrame is empty.")

            RealignmentRules.rename(
                columns={
                    Constants.REALIGNMENT_PERCENTAGE: Constants.REALIGNMENT_PERCENTAGE_WEIGHTED,
                },
                inplace=True,
            )

            Actual = pd.concat([Actual, Full_Actual])

            # set Actual Input = Actual Raw for history refresh buckets
            mask = Actual[Constants.TIME_DAY_KEY] >= StartDate
            for measure in actual_measures:
                raw_col = measure + Constants.ACTUAL_RAW_SUFFIX
                input_col = measure + Constants.ACTUAL_INPUT_SUFFIX

                Actual.loc[mask, input_col] = Actual.loc[mask, raw_col]
                Actual[measure] = Actual[input_col]

            ActualOP = Actual[actual_output_cols]
            RealignmentOP = RealignmentRules[realignment_output_cols]

            logger.warning("No further execution for this slice")
            return RealignmentOP, ActualOP
        if Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD not in RealignmentRules.columns:
            RealignmentRules[Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD] = 90
        else:
            RealignmentRules[Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD].fillna(90, inplace=True)

        RealignmentRules[Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD] = pd.to_numeric(
            RealignmentRules[Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD]
        )

        RealignmentRules[Constants.TRANSITION_DATE] = pd.to_datetime(
            RealignmentRules[Constants.TRANSITION_DATE]
        )
        Actual[Constants.TIME_DAY_KEY] = pd.to_datetime(Actual[Constants.TIME_DAY])
        Full_Actual[Constants.TIME_DAY_KEY] = pd.to_datetime(Full_Actual[Constants.TIME_DAY])

        default_start_date = TimeDimension[Constants.TIME_DAY_KEY].min()

        RealignmentRules[Constants.REALIGNMENT_START_DATE] = default_start_date
        RealignmentRules[Constants.REALIGNMENT_END_DATE] = RealignmentRules[
            Constants.TRANSITION_DATE
        ] + pd.to_timedelta(RealignmentRules[Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD], unit="D")

        columns_mapping = {
            "Account": (
                AccountMaster,
                Constants.ACCOUNT,
                Constants.FROM_ACCOUNT,
                Constants.TO_ACCOUNT,
            ),
            "Channel": (
                ChannelMaster,
                Constants.CHANNEL,
                Constants.FROM_CHANNEL,
                Constants.TO_CHANNEL,
            ),
            "Demand Domain": (
                DemandDomainMaster,
                Constants.DEMAND_DOMAIN,
                Constants.FROM_DEMAND_DOMAIN,
                Constants.TO_DEMAND_DOMAIN,
            ),
            "Item": (ItemMaster, Constants.ITEM, Constants.FROM_ITEM, Constants.TO_ITEM),
            "Location": (
                LocationMaster,
                Constants.LOCATION,
                Constants.FROM_LOCATION,
                Constants.TO_LOCATION,
            ),
            "Pnl": (PnlMaster, Constants.PNL, Constants.FROM_PNL, Constants.TO_PNL),
            "Region": (RegionMaster, Constants.REGION, Constants.FROM_REGION, Constants.TO_REGION),
        }

        # calculate the balance percentage and normalize realignment %
        RealignmentRules = calculate_balance_percentage(RealignmentRules=RealignmentRules)

        # getting self transitions with corresponding balance percentage as realignment percentage
        RealignmentRules = apply_self_transitions(RealignmentRules=RealignmentRules)

        # remove brackets from Attribute Mapping
        for col in Constants.DO_COLUMNS:
            AttributeMapping[col] = AttributeMapping[col].map(remove_first_brackets)

        # assign process order and calculate realignment types
        RealignmentRules = process_realignments(
            AttributeMapping=AttributeMapping, RealignmentRules=RealignmentRules
        )

        # sort based on transition date (ascending order) and rule sequence (ascending order)
        RealignmentRules = RealignmentRules.sort_values(
            by=[Constants.TRANSITION_DATE, Constants.RULE_SEQUENCE], ascending=[True, True]
        ).reset_index(drop=True)

        # Actual Input = Actual Raw for full history periods
        for measure in actual_measures:
            Actual[measure + Constants.ACTUAL_INPUT_SUFFIX] = Actual[
                measure + Constants.ACTUAL_RAW_SUFFIX
            ]
            Full_Actual[measure + Constants.ACTUAL_INPUT_SUFFIX] = Full_Actual[
                measure + Constants.ACTUAL_RAW_SUFFIX
            ]

        Full_Actual.drop(columns=actual_raw_measures, inplace=True)
        Actual.drop(columns=actual_raw_measures, inplace=True)

        # Iterate over each group for realignment
        for (transition_date, rule_sequence, realignment_type), group in RealignmentRules.groupby(
            [Constants.TRANSITION_DATE, Constants.RULE_SEQUENCE, Constants.REALIGNMENT_TYPES],
            sort=True,
        ):
            logger.info(
                f"Starting Realignment for transition date: {transition_date}, rule_sequence: {rule_sequence}, realignment type: {realignment_type}"
            )

            dimensions_key_mapping = {}
            for pair in realignment_type.split(","):
                pair = pair.strip()
                if "." in pair:
                    prefix = pair.split(".", 1)[0]
                    dimensions_key_mapping[prefix.strip()] = pair

            other_level_keys = [
                key
                for key, dim in dimensions_key_mapping.items()
                if dim not in Constants.LEAF_DIMENSIONS
            ]

            # disaggregate higher level to leaf levels
            if other_level_keys:
                for dim in other_level_keys:
                    other_grain = dimensions_key_mapping[dim]
                    leaf_col = columns_mapping[dim][1]
                    from_col = columns_mapping[dim][2]
                    to_col = columns_mapping[dim][3]

                    group = pd.merge(
                        group,
                        columns_mapping[dim][0][[other_grain, leaf_col]].drop_duplicates(),
                        left_on=from_col,
                        right_on=other_grain,
                        how="left",
                    )

                    group[from_col] = group[leaf_col]
                    group[to_col] = group[leaf_col]

                    group.drop(columns=[other_grain, leaf_col], inplace=True)

            leaf_level_cols = [columns_mapping[dim][1] for dim in dimensions_key_mapping.keys()]
            from_cols = [columns_mapping[dim][2] for dim in dimensions_key_mapping.keys()]
            to_cols = [columns_mapping[dim][3] for dim in dimensions_key_mapping.keys()]

            group = group[group[Constants.VALIDITY_STATUS] == 1]

            # realignment before history refresh buckets
            realignment_rules = group[group[Constants.REALIGNMENT_STATUS] == 0]
            Full_Actual = get_realigned_data(
                Actual=Full_Actual,
                realignment_rules=realignment_rules,
                StartDate=StartDate,
                leaf_level_cols=leaf_level_cols,
                from_cols=from_cols,
                to_cols=to_cols,
                measures=actual_input_measures,
            )

            # update full history realignment status = 1
            RealignmentRules, group = update_realignment_status(
                realignment_rules=RealignmentRules, group=group
            )

            # realignment after history refresh buckets
            realignment_rules = group[group[Constants.REALIGNMENT_STATUS] == 1]
            Full_Actual = get_realigned_data(
                Actual=Full_Actual,
                realignment_rules=realignment_rules,
                StartDate=StartDate,
                leaf_level_cols=leaf_level_cols,
                from_cols=from_cols,
                to_cols=to_cols,
                measures=actual_input_measures,
            )

            logger.info(
                f"Completed Realignment for transition date: {transition_date}, rule_sequence: {rule_sequence}, realignment type: {realignment_type}"
            )

        # remove self transitions from RealignmentRules
        RealignmentRules = RealignmentRules[RealignmentRules[Constants.SELF_TRANSITION_FLAG] == 0]

        RealignmentRules[Constants.REALIGNMENT_STATUS] = RealignmentRules[
            Constants.REALIGNMENT_STATUS
        ].replace(0, np.nan)

        RealignmentRules[Constants.REALIGNMENT_PERCENTAGE] = RealignmentRules[
            Constants.REALIGNMENT_PERCENTAGE
        ].replace(0, np.nan)

        RealignmentRules.rename(
            columns={
                Constants.REALIGNMENT_PERCENTAGE: Constants.REALIGNMENT_PERCENTAGE_WEIGHTED,
            },
            inplace=True,
        )

        Actual = pd.concat([Actual, Full_Actual])

        # assign actual measures = actual input measures
        for measure in actual_measures:
            input_col = measure + Constants.ACTUAL_INPUT_SUFFIX
            Actual[input_col] = Actual[input_col].replace(0, np.nan)
            Actual[measure] = Actual[input_col]

        ActualOP = Actual[actual_output_cols]
        RealignmentOP = RealignmentRules[realignment_output_cols].drop_duplicates()

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

        RealignmentRules.rename(
            columns={
                Constants.REALIGNMENT_PERCENTAGE: Constants.REALIGNMENT_PERCENTAGE_WEIGHTED,
            },
            inplace=True,
        )

        for measure in actual_measures:
            raw_col = measure + Constants.ACTUAL_RAW_SUFFIX
            input_col = measure + Constants.ACTUAL_INPUT_SUFFIX
            Actual[input_col] = Actual[raw_col]
            Actual[measure] = Actual[input_col]

        ActualOP = Actual[actual_output_cols]
        RealignmentOP = RealignmentRules[realignment_output_cols]

    return RealignmentOP, ActualOP

import logging

import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import remove_first_brackets

from helpers.o9Constants import o9Constants
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """Constants used in the plugin."""

    # Base Dimension Names
    VERSION = o9Constants.VERSION_NAME
    SEQUENCE = o9Constants.SEQUENCE
    ITEM = o9Constants.ITEM
    LOCATION = o9Constants.LOCATION
    CHANNEL = o9Constants.CHANNEL
    ACCOUNT = o9Constants.ACCOUNT
    PNL = o9Constants.PNL
    DEMAND_DOMAIN = o9Constants.DEMAND_DOMAIN
    REGION = o9Constants.REGION

    # Input Columns
    DATA_OBJECT = "Data Object.[Data Object]"
    DM_RULE = "DM Rule.[Rule]"

    # From Scope Columns
    FROM_ITEM = "DP From Item Scope"
    FROM_LOCATION = "DP From Location Scope"
    FROM_CHANNEL = "DP From Channel Scope"
    FROM_ACCOUNT = "DP From Account Scope"
    FROM_PNL = "DP From PnL Scope"
    FROM_DEMAND_DOMAIN = "DP From Demand Domain Scope"
    FROM_REGION = "DP From Region Scope"

    # To Scope Columns
    TO_ITEM = "DP To Item Scope"
    TO_LOCATION = "DP To Location Scope"
    TO_CHANNEL = "DP To Channel Scope"
    TO_ACCOUNT = "DP To Account Scope"
    TO_PNL = "DP To PnL Scope"
    TO_DEMAND_DOMAIN = "DP To Demand Domain Scope"
    TO_REGION = "DP To Region Scope"

    # Data Object Level Columns
    DO_ITEM = "Data Object Item Level"
    DO_LOCATION = "Data Object Location Level"
    DO_CHANNEL = "Data Object Channel Level"
    DO_ACCOUNT = "Data Object Account Level"
    DO_PNL = "Data Object PnL Level"
    DO_DEMAND_DOMAIN = "Data Object Demand Domain Level"
    DO_REGION = "Data Object Region Level"

    # Rule and Status Columns
    REALIGNMENT_STATUS = "History Realignment Status"
    CLUSTER_ID = "DP Realignment Cluster ID"
    TRANSITION_DATE = "Transition Start Date"
    RULE_SEQUENCE = "DP Realignment Rule Sequence"

    # Output Measure
    CANDIDATE_MEASURE = "Candidate for History Realignment"

    # Intermediate/Helper Columns
    REALIGNMENT_TYPES = "Realignment Types"
    LC_SUFFIX = " LC"

    # Dimension Lists
    DIMENSIONS = [
        "Item",
        "Location",
        "Channel",
        "Region",
        "Account",
        "PnL",
        "Demand Domain",
    ]
    FROM_DIMENSIONS = [
        FROM_ITEM,
        FROM_LOCATION,
        FROM_CHANNEL,
        FROM_REGION,
        FROM_ACCOUNT,
        FROM_PNL,
        FROM_DEMAND_DOMAIN,
    ]
    TO_DIMENSIONS = [
        TO_ITEM,
        TO_LOCATION,
        TO_CHANNEL,
        TO_REGION,
        TO_ACCOUNT,
        TO_PNL,
        TO_DEMAND_DOMAIN,
    ]
    DO_COLUMNS = [
        DO_ITEM,
        DO_LOCATION,
        DO_REGION,
        DO_CHANNEL,
        DO_ACCOUNT,
        DO_PNL,
        DO_DEMAND_DOMAIN,
    ]


def get_output_rows(
    name,
    group,
    dimensions,
    output_grains,
    dataframe_mapping,
    from_dimensions,
    Actual,
):
    """Extract output rows based on the provided group and dimensions."""
    try:
        group = disagg_rules_to_leaf_level_grains(
            name,
            group,
            dimensions,
            output_grains,
            dataframe_mapping,
            from_dimensions,
        )

        group = group.merge(
            Actual[output_grains],
        )

    except Exception as e:
        logger.exception("Exception for group : {}".format(e))

    return group


def disagg_rules_to_leaf_level_grains(
    name,
    group,
    dimensions,
    output_grains,
    dataframe_mapping,
    from_dimensions,
):
    """Disaggregate rules to leaf level grains."""
    try:
        relevant_keys = [dim for dim in dimensions if dim in name]
        realignment_dims = get_list_of_grains_from_string(name)

        leaf_level_grains = [x for x in realignment_dims if x in output_grains]
        leaf_grains_relevant_keys = [
            x for x in relevant_keys if any(x in grain for grain in leaf_level_grains)
        ]
        other_level_grains = [x for x in realignment_dims if x not in leaf_level_grains]
        other_grains_relevant_keys = [
            x for x in relevant_keys if any(x in grain for grain in other_level_grains)
        ]

        if len(leaf_grains_relevant_keys) > 0:
            for key in leaf_grains_relevant_keys:
                group[dataframe_mapping[key][1]] = group[dataframe_mapping[key][2]]

        if len(other_grains_relevant_keys) > 0:
            for key in other_grains_relevant_keys:
                other_grain = [grain for grain in other_level_grains if key in grain][0]
                relevant_df = dataframe_mapping[key][0][[other_grain, dataframe_mapping[key][1]]]

                group = group.merge(
                    relevant_df.drop_duplicates(),
                    left_on=dataframe_mapping[key][2],
                    right_on=other_grain,
                )
                group.drop(columns=other_grain, axis=1, inplace=True)

        group.drop(columns=from_dimensions, axis=1, inplace=True)

    except Exception as e:
        logger.exception("Exception for group : {}".format(e))

    return group


col_mapping = {Constants.CANDIDATE_MEASURE: float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    AttributeMapping,
    RealignmentRules,
    Actual,
    AccountMapping,
    ChannelMapping,
    PnLMapping,
    DemandDomainMapping,
    LocationMapping,
    ItemMapping,
    RegionMapping,
    df_keys,
    multiprocessing_num_cores=4,
):
    plugin_name = "DP082NetChangeHistoryRealignment"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    output_grains = get_list_of_grains_from_string(Grains)

    cols_req_in_output = [Constants.VERSION] + output_grains + [Constants.CANDIDATE_MEASURE]

    CandidateOutput = pd.DataFrame(columns=cols_req_in_output)
    try:
        input_dataframes = {
            "AttributeMapping": AttributeMapping,
            "RealignmentRules": RealignmentRules,
            "Actual": Actual,
            "AccountMapping": AccountMapping,
            "ChannelMapping": ChannelMapping,
            "PnLMapping": PnLMapping,
            "DemandDomainMapping": DemandDomainMapping,
            "LocationMapping": LocationMapping,
            "ItemMapping": ItemMapping,
            "RegionMapping": RegionMapping,
        }

        # check if any input dataframes is empty
        empty_dfs = [name for name, df in input_dataframes.items() if df.empty]

        if empty_dfs:
            logger.warning(
                f"Input DataFrame(s) {', '.join(empty_dfs)} are empty for slice '{df_keys}'."
            )
            return CandidateOutput

        # creating mapping for master data and dimensions
        dataframe_mapping = {
            "Item": (ItemMapping, Constants.ITEM, Constants.FROM_ITEM),
            "Location": (LocationMapping, Constants.LOCATION, Constants.FROM_LOCATION),
            "Region": (RegionMapping, Constants.REGION, Constants.FROM_REGION),
            "Channel": (ChannelMapping, Constants.CHANNEL, Constants.FROM_CHANNEL),
            "PnL": (PnLMapping, Constants.PNL, Constants.FROM_PNL),
            "Account": (AccountMapping, Constants.ACCOUNT, Constants.FROM_ACCOUNT),
            "Demand Domain": (
                DemandDomainMapping,
                Constants.DEMAND_DOMAIN,
                Constants.FROM_DEMAND_DOMAIN,
            ),
        }

        for col in Constants.DO_COLUMNS:
            # remove brackets to get dimension name
            AttributeMapping[col] = AttributeMapping[col].map(remove_first_brackets)

            # Replace NaN with empty strings
            AttributeMapping[col].fillna("", inplace=True)

        # getting all dimensions to realign
        AttributeMapping[Constants.REALIGNMENT_TYPES] = (
            AttributeMapping[Constants.DO_COLUMNS].astype(str).agg(",".join, axis=1)
        )
        AttributeMapping[Constants.REALIGNMENT_TYPES] = AttributeMapping[
            Constants.REALIGNMENT_TYPES
        ].str.strip(",")

        # Drop unnecessary columns
        AttributeMapping.drop(Constants.DO_COLUMNS, axis=1, inplace=True)

        # Find the respective Realignment Rules using merging with Attribute Mapping on Data Object
        RealignmentRules = pd.merge(
            RealignmentRules,
            AttributeMapping,
            on=[Constants.VERSION, Constants.DATA_OBJECT],
            how="inner",
        )

        # dropping nulls and 0
        RealignmentRules = RealignmentRules[RealignmentRules[Constants.REALIGNMENT_STATUS] == 1]

        if RealignmentRules.empty:
            logger.warning(
                "No active Realignment Rules found for populating candidates. Exiting ..."
            )
            return CandidateOutput

        logger.info("Started populating candidates ...")

        common_cols = [Constants.VERSION, Constants.DATA_OBJECT, Constants.REALIGNMENT_TYPES]

        from_dataframe = RealignmentRules[common_cols + Constants.FROM_DIMENSIONS].drop_duplicates()
        to_dataframe = RealignmentRules[common_cols + Constants.TO_DIMENSIONS].drop_duplicates()

        lc_from_dimensions = [x + Constants.LC_SUFFIX for x in Constants.FROM_DIMENSIONS]
        lc_to_dimensions = [x + Constants.LC_SUFFIX for x in Constants.TO_DIMENSIONS]
        lc_from_dataframe = RealignmentRules[common_cols + lc_from_dimensions].drop_duplicates()
        lc_to_dataframe = RealignmentRules[common_cols + lc_to_dimensions].drop_duplicates()

        from_dataframe.dropna(subset=Constants.FROM_DIMENSIONS, how="all", inplace=True)
        to_dataframe.dropna(subset=Constants.TO_DIMENSIONS, how="all", inplace=True)
        lc_from_dataframe.dropna(subset=lc_from_dimensions, how="all", inplace=True)
        lc_to_dataframe.dropna(subset=lc_to_dimensions, how="all", inplace=True)

        lc_from_dataframe.columns = common_cols + Constants.FROM_DIMENSIONS
        lc_to_dataframe.columns = common_cols + Constants.FROM_DIMENSIONS
        to_dataframe.columns = common_cols + Constants.FROM_DIMENSIONS

        from_dataframe = pd.concat(
            [from_dataframe, lc_from_dataframe, to_dataframe, lc_to_dataframe]
        )

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(get_output_rows)(
                name=name,
                group=group,
                dimensions=Constants.DIMENSIONS,
                output_grains=output_grains,
                dataframe_mapping=dataframe_mapping,
                from_dimensions=Constants.FROM_DIMENSIONS,
                Actual=Actual,
            )
            for name, group in from_dataframe.groupby(Constants.REALIGNMENT_TYPES)
        )

        Output = concat_to_dataframe(all_results)
        Output[Constants.CANDIDATE_MEASURE] = 1
        CandidateOutput = Output[cols_req_in_output].drop_duplicates()

        logger.info("Successfully populated candidates ...")

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return CandidateOutput

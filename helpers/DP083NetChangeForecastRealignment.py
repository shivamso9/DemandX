import logging
import re

import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def remove_brackets(val):
    """
    to remove unnecessary brackets from grains.
    [Item].[Item] -> Item.[Item]
    """

    if pd.notnull(val):
        return re.sub(r"^\[([^\]]+)\]", r"\1", val)
    return val


def get_output_rows(
    name,
    group,
    dimensions,
    output_grains,
    dataframe_mapping,
    from_dimensions,
    AssortmentFinal,
):
    relevant_keys = [dim for dim in dimensions if dim in name]
    realignment_dims = get_list_of_grains_from_string(name)

    planning_level_grains = [x for x in realignment_dims if x in output_grains]
    planning_grains_relevant_keys = [
        x for x in relevant_keys if any(x in grain for grain in planning_level_grains)
    ]
    other_level_grains = [x for x in realignment_dims if x not in planning_level_grains]
    other_grains_relevant_keys = [
        x for x in relevant_keys if any(x in grain for grain in other_level_grains)
    ]

    if len(planning_grains_relevant_keys) > 0:
        for key in planning_grains_relevant_keys:
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
    group = group.merge(
        AssortmentFinal[output_grains],
    )
    return group


col_mapping = {
    "Candidate for Forecast Realignment": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    AttributeMapping,
    RealignmentRules,
    AssortmentFinal,
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
    plugin_name = "DP083NetChangeForecastRealignment"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    version_col = "Version.[Version Name]"
    data_object = "Data Object.[Data Object]"

    from_item = "DP From Item Scope"
    from_location = "DP From Location Scope"
    from_channel = "DP From Channel Scope"
    from_account = "DP From Account Scope"
    from_pnl = "DP From PnL Scope"
    from_demand_domain = "DP From Demand Domain Scope"
    from_region = "DP From Region Scope"

    to_item = "DP To Item Scope"
    to_location = "DP To Location Scope"
    to_channel = "DP To Channel Scope"
    to_account = "DP To Account Scope"
    to_pnl = "DP To PnL Scope"
    to_demand_domain = "DP To Demand Domain Scope"
    to_region = "DP To Region Scope"

    do_account = "Data Object Account Level"
    do_channel = "Data Object Channel Level"
    do_item = "Data Object Item Level"
    do_location = "Data Object Location Level"
    do_region = "Data Object Region Level"
    do_pnl = "Data Object PnL Level"
    do_demand_domain = "Data Object Demand Domain Level"

    realignments_col = "Realignment Types"
    realignment_status = "Forecast Realignment Status"
    lc_suffix = " LC"

    output_grains = get_list_of_grains_from_string(Grains)

    # output measure
    candidate_measure = "Candidate for Forecast Realignment"

    cols_req_in_output = [version_col] + output_grains + [candidate_measure]
    CandidateOutput = pd.DataFrame(columns=cols_req_in_output)
    try:
        if len(AttributeMapping) == 0 or len(RealignmentRules) == 0 or len(AssortmentFinal) == 0:
            logger.warning(f"One of the inputs is Null. Exiting : {df_keys} ...")
            return CandidateOutput

        # getting relevant planning item and planning location
        AssortmentFinal = AssortmentFinal.merge(
            ItemMapping[[o9Constants.ITEM, o9Constants.PLANNING_ITEM]].drop_duplicates(),
        )
        AssortmentFinal = AssortmentFinal.merge(
            LocationMapping[
                [o9Constants.LOCATION, o9Constants.PLANNING_LOCATION]
            ].drop_duplicates(),
        )
        AssortmentFinal.drop(columns=[o9Constants.ITEM, o9Constants.LOCATION], axis=1, inplace=True)

        # dropping nulls and 0
        RealignmentRules = RealignmentRules[RealignmentRules[realignment_status] == 1]

        # make sure to have same sequence in dimensions, from_dimensions and to_dimensions
        dimensions = [
            "Item",
            "Location",
            "Channel",
            "Region",
            "Account",
            "PnL",
            "Demand Domain",
        ]
        from_dimensions = [
            from_item,
            from_location,
            from_channel,
            from_region,
            from_account,
            from_pnl,
            from_demand_domain,
        ]
        to_dimensions = [
            to_item,
            to_location,
            to_channel,
            to_region,
            to_account,
            to_pnl,
            to_demand_domain,
        ]

        # creating mapping for master data and dimensions
        dataframe_mapping = {
            "Item": (ItemMapping, o9Constants.PLANNING_ITEM, from_item),
            "Location": (LocationMapping, o9Constants.PLANNING_LOCATION, from_location),
            "Region": (RegionMapping, o9Constants.PLANNING_REGION, from_region),
            "Channel": (ChannelMapping, o9Constants.PLANNING_CHANNEL, from_channel),
            "PnL": (PnLMapping, o9Constants.PLANNING_PNL, from_pnl),
            "Account": (AccountMapping, o9Constants.PLANNING_ACCOUNT, from_account),
            "Demand Domain": (
                DemandDomainMapping,
                o9Constants.PLANNING_DEMAND_DOMAIN,
                from_demand_domain,
            ),
        }

        cols = [
            do_item,
            do_location,
            do_region,
            do_channel,
            do_account,
            do_pnl,
            do_demand_domain,
        ]
        for col in cols:
            # remove brackets to get dimension name
            AttributeMapping[col] = AttributeMapping[col].map(remove_brackets)

            # Replace NaN with empty strings
            AttributeMapping[col].fillna("", inplace=True)

        # getting all dimensions to realign
        AttributeMapping[realignments_col] = (
            AttributeMapping[cols].astype(str).agg(",".join, axis=1)
        )
        AttributeMapping[realignments_col] = AttributeMapping[realignments_col].str.strip(",")

        # Drop unnecessary columns
        AttributeMapping.drop(cols, axis=1, inplace=True)

        # Find the respective Realignment Rules using merging with Attribute Mapping on Data Object
        RealignmentRules = pd.merge(
            RealignmentRules,
            AttributeMapping,
            on=[version_col, data_object],
            how="inner",
        )

        common_cols = [version_col, data_object, realignments_col]

        from_dataframe = RealignmentRules[common_cols + from_dimensions].drop_duplicates()
        to_dataframe = RealignmentRules[common_cols + to_dimensions].drop_duplicates()

        lc_from_dimensions = [x + lc_suffix for x in from_dimensions]
        lc_to_dimensions = [x + lc_suffix for x in to_dimensions]
        lc_from_dataframe = RealignmentRules[common_cols + lc_from_dimensions].drop_duplicates()
        lc_to_dataframe = RealignmentRules[common_cols + lc_to_dimensions].drop_duplicates()

        from_dataframe.dropna(subset=from_dimensions, how="all", inplace=True)
        to_dataframe.dropna(subset=to_dimensions, how="all", inplace=True)
        lc_from_dataframe.dropna(subset=lc_from_dimensions, how="all", inplace=True)
        lc_to_dataframe.dropna(subset=lc_to_dimensions, how="all", inplace=True)

        lc_from_dataframe.columns = common_cols + from_dimensions
        lc_to_dataframe.columns = common_cols + from_dimensions
        to_dataframe.columns = common_cols + from_dimensions

        from_dataframe = pd.concat(
            [from_dataframe, lc_from_dataframe, to_dataframe, lc_to_dataframe]
        )

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(get_output_rows)(
                name=name,
                group=group,
                dimensions=dimensions,
                output_grains=output_grains,
                dataframe_mapping=dataframe_mapping,
                from_dimensions=from_dimensions,
                AssortmentFinal=AssortmentFinal,
            )
            for name, group in from_dataframe.groupby(realignments_col)
        )

        Output = concat_to_dataframe(all_results)
        Output[candidate_measure] = 1
        CandidateOutput = Output[cols_req_in_output].drop_duplicates()

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return CandidateOutput

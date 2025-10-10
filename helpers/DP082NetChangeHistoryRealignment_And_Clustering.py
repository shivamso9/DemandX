import logging

import numpy as np
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
    REALIGNMENT_STATUS_RERUN = "History Realignment Status Rerun"
    CLUSTER_ID = "DP Realignment Cluster ID"
    TRANSITION_DATE = "Transition Start Date"
    RULE_SEQUENCE = "DP Realignment Rule Sequence"
    FULL_HISTORY_REALIGNMENT_STATUS = "DP Full History Realignment Status"
    REALIGNMENT_STATUS_SYSTEM = "History Realignment Status System"
    REALIGNMENT_NET_CHANGE_FLAG = "Realignment Net Change Flag"

    # Output Measure
    CANDIDATE_MEASURE = "Candidate for History Realignment"
    REALIGNMENT_RULE_CANDIDATES_ASSOCIATION = "Realignment Rule Candidates Association"

    # Intermediate/Helper Columns
    REALIGNMENT_TYPES = "Realignment Types"
    UNIQUE_ID = "unique_id"
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
    REQ_COLS = [
        VERSION,
        DATA_OBJECT,
        REALIGNMENT_TYPES,
        UNIQUE_ID,
        CLUSTER_ID,
        REALIGNMENT_STATUS,
        FULL_HISTORY_REALIGNMENT_STATUS,
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


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)
        if rootA != rootB:
            if self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            elif self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1


def assign_clusters(
    dfs,
    output_grains,
    unique_id,
    cluster_id,
    realignment_status,
    full_history_realignment_status,
):
    """
    dfs: list of dataframes, each having columns unique_id, cluster_id (nullable), realignment_status, and output_grains columns
    output_grains: list of columns on which to inner join for linking nodes

    Returns: dataframe with columns unique_id and cluster_id for each node.
    """
    try:
        n = len(dfs)
        if n == 0:
            return pd.DataFrame(columns=[unique_id, cluster_id, realignment_status])

        dsu = DSU(n)

        # Map from index to unique_id, cluster_id, realignment_status
        unique_ids = [df[unique_id].iloc[0] for df in dfs]
        cluster_ids = [
            (
                df[cluster_id].iloc[0]
                if cluster_id in df.columns and pd.notna(df[cluster_id].iloc[0])
                else None
            )
            for df in dfs
        ]
        realignment_flags = [
            df[realignment_status].iloc[0] if realignment_status in df.columns else 0 for df in dfs
        ]
        full_history_flags = [df[full_history_realignment_status].iloc[0] for df in dfs]

        # Precompute sets of output_grains
        grain_sets = [
            set(
                tuple(row)
                for row in df[output_grains].drop_duplicates().itertuples(index=False, name=None)
            )
            for df in dfs
        ]

        # Step 1: Build the graph using inner join checks
        # create mapping from each grain to the set of dataframe indices (i.e., which dataframes contain it)
        grain_to_indices = {}

        for idx, grain_set in enumerate(grain_sets):
            for grain in grain_set:
                if grain not in grain_to_indices:
                    grain_to_indices[grain] = set()
                grain_to_indices[grain].add(idx)

        # union all dataframes that share at least one grain
        for indices in grain_to_indices.values():
            indices = list(indices)
            base = indices[0]
            for other in indices[1:]:
                dsu.union(base, other)

        # Step 2: Find connected components
        components = {}
        for i in range(n):
            root = dsu.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        # Step 3: Assign cluster ids and realignment status to components
        existing_cluster_ids = [cid for cid in cluster_ids if cid is not None]
        max_cluster_id = max(existing_cluster_ids) if existing_cluster_ids else 0

        root_to_cluster = {}
        root_to_realignment = {}
        root_to_full_history = {}
        next_cluster_id = max_cluster_id + 1

        for root, nodes in components.items():
            # Determine cluster id
            comp_cluster_ids = [
                cluster_ids[node] for node in nodes if cluster_ids[node] is not None
            ]
            if comp_cluster_ids:
                assigned_id = min(comp_cluster_ids)
            else:
                assigned_id = next_cluster_id
                next_cluster_id += 1
            root_to_cluster[root] = assigned_id

            # Determine realignment_status
            comp_realignment = any(realignment_flags[node] == 1 for node in nodes)
            root_to_realignment[root] = int(comp_realignment)

            # Determine full history status
            comp_full_history_status = all(full_history_flags[node] != 0 for node in nodes)
            root_to_full_history[root] = int(comp_full_history_status)

        # Step 4: Build output dataframe
        result = []
        for i in range(n):
            root = dsu.find(i)
            result.append(
                {
                    unique_id: unique_ids[i],
                    cluster_id: root_to_cluster[root],
                    realignment_status: root_to_realignment[root],
                    full_history_realignment_status: root_to_full_history[root],
                }
            )

        return pd.DataFrame(result)
    except Exception as e:
        raise RuntimeError(f"Error in assign_clusters: {e}") from e


col_mapping = {
    Constants.CANDIDATE_MEASURE: float,
    Constants.REALIGNMENT_RULE_CANDIDATES_ASSOCIATION: float,
    Constants.CLUSTER_ID: float,
    Constants.REALIGNMENT_STATUS_SYSTEM: float,
    Constants.FULL_HISTORY_REALIGNMENT_STATUS: float,
}


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
    plugin_name = "DP082NetChangeHistoryRealignment_And_Clustering"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    output_grains = get_list_of_grains_from_string(Grains)

    cols_req_in_output = [Constants.VERSION] + output_grains + [Constants.CANDIDATE_MEASURE]
    cols_req_in_realignment_rule_association = (
        [
            Constants.VERSION,
            Constants.DATA_OBJECT,
            Constants.SEQUENCE,
            Constants.DM_RULE,
        ]
        + output_grains
        + [Constants.REALIGNMENT_RULE_CANDIDATES_ASSOCIATION]
    )

    cols_req_in_realignment_output = [
        Constants.VERSION,
        Constants.DATA_OBJECT,
        Constants.SEQUENCE,
        Constants.DM_RULE,
    ] + [
        Constants.CLUSTER_ID,
        Constants.REALIGNMENT_STATUS_SYSTEM,
        Constants.FULL_HISTORY_REALIGNMENT_STATUS,
    ]

    CandidateOutput = pd.DataFrame(columns=cols_req_in_output)
    RealignmentOP = pd.DataFrame(columns=cols_req_in_realignment_output)
    RealignmentRuleAssociationOP = pd.DataFrame(columns=cols_req_in_realignment_rule_association)
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
            return CandidateOutput, RealignmentOP, RealignmentRuleAssociationOP

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

        RealignmentRules.loc[
            RealignmentRules[Constants.REALIGNMENT_STATUS_RERUN] == 1, Constants.REALIGNMENT_STATUS
        ] = 1
        RealignmentRules[Constants.FULL_HISTORY_REALIGNMENT_STATUS].fillna(0, inplace=True)

        # Start of Clustering
        logger.info("Starting clustering of Realignment Rules ...")

        realignment_rules = RealignmentRules.copy()
        relevant_actuals = Actual.copy()

        relevant_actuals = relevant_actuals[[Constants.VERSION] + output_grains]
        relevant_actuals[Constants.CLUSTER_ID] = np.nan

        realignment_rules[Constants.TRANSITION_DATE] = pd.to_datetime(
            realignment_rules[Constants.TRANSITION_DATE]
        )
        realignment_rules[Constants.RULE_SEQUENCE] = realignment_rules[
            Constants.RULE_SEQUENCE
        ].fillna(1)

        realignment_rules = realignment_rules.reset_index(drop=True)
        realignment_rules[Constants.UNIQUE_ID] = realignment_rules.index + 1

        # sort based on transition date (ascending order) and process order (descending order)
        realignment_rules = realignment_rules.sort_values(
            by=[Constants.TRANSITION_DATE, Constants.RULE_SEQUENCE], ascending=[True, True]
        ).reset_index(drop=True)

        grouped_dataframes = []

        for (_, _, _), rule in realignment_rules.groupby(
            [Constants.TRANSITION_DATE, Constants.RULE_SEQUENCE, Constants.UNIQUE_ID]
        ):
            from_dataframe = rule[Constants.REQ_COLS + Constants.FROM_DIMENSIONS].drop_duplicates()
            to_dataframe = rule[Constants.REQ_COLS + Constants.TO_DIMENSIONS].drop_duplicates()

            from_dataframe.dropna(subset=Constants.FROM_DIMENSIONS, how="all", inplace=True)
            to_dataframe.dropna(subset=Constants.TO_DIMENSIONS, how="all", inplace=True)

            if from_dataframe.empty or to_dataframe.empty:
                continue

            to_dataframe.columns = Constants.REQ_COLS + Constants.FROM_DIMENSIONS

            name = rule[Constants.REALIGNMENT_TYPES].iloc[0]

            # disaggregate rules to leaf level grains
            from_rules = disagg_rules_to_leaf_level_grains(
                name=name,
                group=from_dataframe,
                dimensions=Constants.DIMENSIONS,
                output_grains=output_grains,
                dataframe_mapping=dataframe_mapping,
                from_dimensions=Constants.FROM_DIMENSIONS,
            )

            to_rules = disagg_rules_to_leaf_level_grains(
                name=name,
                group=to_dataframe,
                dimensions=Constants.DIMENSIONS,
                output_grains=output_grains,
                dataframe_mapping=dataframe_mapping,
                from_dimensions=Constants.FROM_DIMENSIONS,
            )

            join_cols = [col.strip() for col in name.split(",") if col.strip()]
            other_level_dims = [col for col in join_cols if col not in output_grains]

            if len(other_level_dims) > 0:
                for col in other_level_dims:
                    key_value = col.split(".")[0]
                    leaf_col = dataframe_mapping.get(key_value)[1]

                    # replace col with leaf_col in join_cols
                    join_cols = [leaf_col if item == col else item for item in join_cols]

            df_cols = (
                [Constants.VERSION]
                + join_cols
                + [
                    Constants.UNIQUE_ID,
                    Constants.REALIGNMENT_STATUS,
                    Constants.FULL_HISTORY_REALIGNMENT_STATUS,
                ]
            )

            from_actuals = relevant_actuals.merge(
                from_rules[df_cols], on=[Constants.VERSION] + join_cols, how="inner"
            )
            to_actuals = from_actuals.drop(
                join_cols
                + [
                    Constants.UNIQUE_ID,
                    Constants.REALIGNMENT_STATUS,
                    Constants.FULL_HISTORY_REALIGNMENT_STATUS,
                ],
                axis=1,
            ).copy()

            to_actuals = to_actuals.merge(
                to_rules[df_cols].drop_duplicates(), on=[Constants.VERSION], how="left"
            )

            other_to_actuals = relevant_actuals.merge(
                to_rules[df_cols], on=[Constants.VERSION] + join_cols, how="inner"
            )

            to_actuals = pd.concat(
                [to_actuals, other_to_actuals], ignore_index=True
            ).drop_duplicates()
            rules_actuals = pd.concat(
                [from_actuals, to_actuals], ignore_index=True
            ).drop_duplicates()

            grouped_dataframes.append(rules_actuals)

            key_cols = [Constants.VERSION] + output_grains
            rules_actuals = rules_actuals[key_cols + [Constants.CLUSTER_ID]]

            # remove common rows from relevant_actuals
            relevant_actuals = (
                relevant_actuals.merge(
                    rules_actuals[key_cols].drop_duplicates(),
                    on=key_cols,
                    how="left",
                    indicator=True,
                )
                .query('_merge == "left_only"')
                .drop(columns=["_merge"])
            )

            relevant_actuals = pd.concat(
                [relevant_actuals, rules_actuals],
                ignore_index=True,
            ).drop_duplicates()

        # drop empty dataframes
        grouped_dataframes = [df for df in grouped_dataframes if not df.empty]

        # assign clusters using graph
        result_df = assign_clusters(
            grouped_dataframes,
            output_grains,
            Constants.UNIQUE_ID,
            Constants.CLUSTER_ID,
            Constants.REALIGNMENT_STATUS,
            Constants.FULL_HISTORY_REALIGNMENT_STATUS,
        )

        result_df.rename(
            columns={Constants.REALIGNMENT_STATUS: Constants.REALIGNMENT_STATUS_SYSTEM},
            inplace=True,
        )
        realignment_rules.drop(
            columns=[Constants.CLUSTER_ID, Constants.FULL_HISTORY_REALIGNMENT_STATUS], inplace=True
        )

        # Assign the cluster_id to RealignmentRules
        RealignmentOP = realignment_rules.merge(
            result_df[
                [
                    Constants.UNIQUE_ID,
                    Constants.CLUSTER_ID,
                    Constants.REALIGNMENT_STATUS_SYSTEM,
                    Constants.FULL_HISTORY_REALIGNMENT_STATUS,
                ]
            ],
            on=Constants.UNIQUE_ID,
            how="left",
        )

        RealignmentOP[Constants.REALIGNMENT_STATUS_SYSTEM] = RealignmentOP[
            Constants.REALIGNMENT_STATUS_SYSTEM
        ].replace(0, np.nan)

        RealignmentOP[Constants.FULL_HISTORY_REALIGNMENT_STATUS] = RealignmentOP[
            Constants.FULL_HISTORY_REALIGNMENT_STATUS
        ].replace(0, np.nan)

        RealignmentOP = RealignmentOP[cols_req_in_realignment_output]

        logger.info("Successfully completed clustering of Realignment Rules ...")
        # End of Clustering

        RealignmentRules = RealignmentRules.merge(
            RealignmentOP[
                [
                    Constants.VERSION,
                    Constants.DATA_OBJECT,
                    Constants.SEQUENCE,
                    Constants.DM_RULE,
                    Constants.REALIGNMENT_STATUS_SYSTEM,
                ]
            ],
            on=[Constants.VERSION, Constants.DATA_OBJECT, Constants.SEQUENCE, Constants.DM_RULE],
            how="left",
        )

        # if net change flag is "REMOVED" then update REALIGNMENT_STATUS_SYSTEM = 1
        condition = RealignmentRules[Constants.REALIGNMENT_NET_CHANGE_FLAG] == "REMOVED"
        RealignmentRules.loc[condition, Constants.REALIGNMENT_STATUS_SYSTEM] = 1

        # dropping nulls and 0
        RealignmentRules = RealignmentRules[
            RealignmentRules[Constants.REALIGNMENT_STATUS_SYSTEM] == 1
        ]

        if RealignmentRules.empty:
            logger.warning(
                "No active Realignment Rules found for populating candidates. Exiting ..."
            )
            return CandidateOutput, RealignmentOP, RealignmentRuleAssociationOP

        logger.info("Started populating candidates ...")

        common_cols = [
            Constants.VERSION,
            Constants.DATA_OBJECT,
            Constants.SEQUENCE,
            Constants.DM_RULE,
            Constants.REALIGNMENT_TYPES,
        ]

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
        Output[Constants.REALIGNMENT_RULE_CANDIDATES_ASSOCIATION] = 1
        CandidateOutput = Output[cols_req_in_output].drop_duplicates()
        RealignmentRuleAssociationOP = Output[
            cols_req_in_realignment_rule_association
        ].drop_duplicates()

        logger.info("Successfully populated candidates ...")

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))

    return CandidateOutput, RealignmentOP, RealignmentRuleAssociationOP

import logging

import category_encoders as ce
import numpy as np
import pandas as pd
import scipy
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


def prepare_feat_imp_agg(df: pd.DataFrame, feat_col: str) -> pd.DataFrame:
    """
    Performs necessary transformations on the df provided and returns the same
    """
    if len(df) > 0:
        # transpose to get row values as columns
        df = df.transpose()

        # copy values in the first row as column names
        df.columns = df.loc[feat_col, :]

        # drop the row which contains column names
        df.drop(feat_col, inplace=True)

        # drop and reset index
        df.reset_index(drop=True, inplace=True)

        # convert to numeric
        df = df.apply(pd.to_numeric)

    return df


col_mapping = {
    "620 Like Item Match.[Like Item Rank]": float,
    "620 Like Item Match.[Like Item Distance]": float,
    "620 Like Item Match.[Like Item Similarity]": float,
    "620 Like Item Match.[System Suggested Like Item]": bool,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    sales,
    Item,
    pl_item,
    parameters,
    FeatureWeights,
    AssortmentMapping,
    numerical_cols,
    # ItemColumnNames,
    # PlanningItemColumnNames,
    FeatureLevel,
    ConsensusFcst,
    ReadFromHive,
    generate_match_assortment,
    df_keys,
):
    plugin_name = "DP021SystemLikeItemMatch"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    user_feat_weight_col = "User Feature Weight"
    sys_feat_weight_col = "System Feature Weight"
    feat_col = "Feature"
    item_feat_col = "Item Feature.[Item Feature]"
    feat_imp_col = "Feature_Importance"
    feat_ratio_col = "Feature_Importance Normalization Ratio"
    req_item_hierarchy_cols = [
        "Item.[L4]",
        "Item.[L5]",
        "Item.[L6]",
        "Item.[All Item]",
    ]
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"
    planning_location_col = "Location.[Planning Location]"
    item_col = "Item.[Item]"
    day_col = "Time.[Day]"
    time_col = "Time"
    sales_col = "Sales"
    is_assorted_col = "Is Assorted"
    start_date_col = "Start_Date"
    like_product_distance_col = "Like Product Distance"
    like_product_rank_col = "Like Product Rank"
    like_product_flag_col = "Like Product Flag"
    like_product_col = "Like Product"
    like_product_similarity = "Like Product Similarity"
    new_product_col = "New Product"
    new_product_pl_item_col = "New Product Planning Item"
    like_product_pl_item_col = "Like Product Planning Item"
    numerical_cols_list = []
    generate_system_like_item_match_col = "Generate System Like Item Match Assortment"

    from_planning_item_col = "from.[Item].[Planning Item]"
    from_planning_pnl_col = "from.[PnL].[Planning PnL]"
    from_planning_region_col = "from.[Region].[Planning Region]"
    from_planning_channel_col = "from.[Channel].[Planning Channel]"
    from_planning_account_col = "from.[Account].[Planning Account]"
    from_planning_location_col = "from.[Location].[Planning Location]"
    to_planning_item_col = "to.[Item].[Planning Item]"

    # output measures
    cols_required_in_output = [
        version_col,
        from_planning_item_col,
        from_planning_account_col,
        from_planning_channel_col,
        from_planning_region_col,
        from_planning_pnl_col,
        from_planning_location_col,
        to_planning_item_col,
        "620 Like Item Match.[Like Item Rank]",
        "620 Like Item Match.[Like Item Distance]",
        "620 Like Item Match.[Like Item Similarity]",
        "620 Like Item Match.[System Suggested Like Item]",
    ]
    like_sku_result = pd.DataFrame(columns=cols_required_in_output)
    try:

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        if ReadFromHive:
            like_item_actual_col = "DP021LikeItemActual"
        else:
            like_item_actual_col = "Like Item Actual"

        if numerical_cols.lower() != "none":
            numerical_cols_list = numerical_cols.split(",")

        column_order = [
            version_col,
            from_planning_item_col,
            from_planning_account_col,
            from_planning_channel_col,
            from_planning_region_col,
            from_planning_pnl_col,
            from_planning_location_col,
            to_planning_item_col,
            "Like Product Rank",
            "Like Product Distance",
            "Like Product Similarity",
            "Like Product Flag",
        ]

        search_level = "Item.[" + parameters["Like Item Search Space"].iloc[0] + "]"
        match_num = int(parameters["Num Like Items"].iloc[0])
        assert match_num > 0, "Num Like Items have to be strictly positive ..."

        history_measure = parameters["Like Item Search History Measure"].iloc[0]

        if pd.isna(parameters["Like Item History Period"].iloc[0]):
            history_periods = 365
        else:
            history_periods = int(parameters["Like Item History Period"].iloc[0])

        if len(sales) == 0:
            logger.warning("No data found in sales df for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return like_sku_result

        input_version = sales[version_col].unique()[0]

        logger.info("search_level: {}".format(search_level))
        logger.info("match_num: {}".format(match_num))
        logger.info("history_measure: {}".format(history_measure))
        logger.info("history_periods(in days): {}".format(history_periods))

        logger.info("Processing Item Attributes ...")

        # joining with item attributes
        Item[numerical_cols_list] = Item[numerical_cols_list].replace(
            {np.nan: "0", "NULL": "0", "": "0"}
        )
        Item[numerical_cols_list] = Item[numerical_cols_list].fillna("0")

        Item.replace({np.nan: "dummy"}, inplace=True)
        pl_item.replace({np.nan: "dummy"}, inplace=True)

        Item = Item.merge(pl_item, on=pl_item_col, how="left")

        Item = Item.replace({"": "dummy"})
        Item = Item.fillna("dummy")

        logger.info("Processing AssortmentMapping ...")

        if len(AssortmentMapping) == 0:
            logger.warning(
                "No records found in AssortmentMapping for slice : {}...".format(df_keys)
            )
            logger.warning("Returning empty dataframe ...")
            return like_sku_result

        AssortmentMapping = AssortmentMapping[AssortmentMapping[is_assorted_col] == 1]

        AssortmentMapping[pl_item_col] = AssortmentMapping[pl_item_col].astype(str)
        AssortmentMapping[planning_region_col] = AssortmentMapping[planning_region_col].astype(str)

        AssortmentMapping[planning_account_col] = AssortmentMapping[planning_account_col].astype(
            str
        )

        AssortmentMapping[planning_channel_col] = AssortmentMapping[planning_channel_col].astype(
            str
        )

        AssortmentMapping[planning_pnl_col] = AssortmentMapping[planning_pnl_col].astype(str)

        AssortmentMapping[planning_demand_domain_col] = AssortmentMapping[
            planning_demand_domain_col
        ].astype(str)

        AssortmentMapping[planning_location_col] = AssortmentMapping[planning_location_col].astype(
            str
        )

        # filter using the like item match assortment col
        generate_match_assortment[generate_system_like_item_match_col] = generate_match_assortment[
            generate_system_like_item_match_col
        ].astype(int)
        generate_match_assortment = generate_match_assortment[
            generate_match_assortment[generate_system_like_item_match_col] == 1
        ]
        generate_match_assortment = generate_match_assortment.astype(str)

        AssortmentMapping = AssortmentMapping.merge(
            generate_match_assortment,
            on=[
                version_col,
                planning_demand_domain_col,
                pl_item_col,
                planning_region_col,
                planning_pnl_col,
                planning_account_col,
                planning_location_col,
                planning_channel_col,
            ],
            how="inner",
        )
        AssortmentMapping.drop([generate_system_like_item_match_col], axis=1, inplace=True)

        # collect customer groups to be processed
        groups_to_process_df = AssortmentMapping[
            [
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
            ]
        ].drop_duplicates()

        logger.info("Processing FeatureWeights ...")

        # coalesce values from system feature weight to user feature weight
        FeatureWeights[user_feat_weight_col].fillna(
            FeatureWeights[sys_feat_weight_col], inplace=True
        )

        FeatureWeights.rename(
            columns={
                item_feat_col: feat_col,
                user_feat_weight_col: feat_imp_col,
            },
            inplace=True,
        )
        FeatureWeights[feat_col] = "Item.[" + FeatureWeights[feat_col].astype(str) + "]"
        req_cols = [
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
            "Item.[All Item]",
        ]
        item_hierarchy_df = Item[req_cols]

        # Join feature weights with item attributes to take average later
        FeatureWeights = FeatureWeights.merge(
            item_hierarchy_df[req_item_hierarchy_cols].drop_duplicates(),
            on=FeatureLevel,
            how="left",
        )

        FeatureWeights = FeatureWeights.groupby([search_level, feat_col], as_index=False)[
            feat_imp_col
        ].sum()

        FeatureWeights[feat_ratio_col] = FeatureWeights.groupby([search_level])[
            feat_imp_col
        ].transform("sum")

        FeatureWeights[feat_imp_col] = FeatureWeights[feat_imp_col] / FeatureWeights[feat_ratio_col]

        logger.info("sales data before joining with ConsensusFcst, shape : {}".format(sales.shape))
        ConsensusFcst = ConsensusFcst[
            [
                pl_item_col,
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
            ]
        ].drop_duplicates()

        ConsensusFcst = ConsensusFcst.merge(
            Item[[item_col, pl_item_col]],
            how="left",
            on=pl_item_col,
        )
        ConsensusFcst.drop(pl_item_col, axis=1, inplace=True)

        sales = sales.merge(
            ConsensusFcst,
            on=[
                item_col,
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
            ],
            how="inner",
        )
        logger.info("sales data after joining with ConsensusFcst, shape : {}".format(sales.shape))

        logger.info("groups_to_process_df : {}".format(groups_to_process_df.head()))

        logger.info("Filtering sales data for above groups ...")

        # filter sales for only the relevant customer groups
        sales = sales.merge(
            groups_to_process_df,
            on=[
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
            ],
            how="inner",
        )

        if len(sales) == 0:
            logger.warning(
                f"No sales data found after filtering for the following combinations below for slice : {df_keys}\n{groups_to_process_df.to_csv(index=False)}"
            )
            logger.warning("Returning empty dataframe ...")
            return like_sku_result

        sales = pd.merge(sales, Item, how="left", on=item_col)

        # Converting day column from category to string and then to datetime, otherwise aggregation (min) to get start dates fails
        sales[day_col] = sales[day_col].astype(str)
        sales[day_col] = pd.to_datetime(sales[day_col])
        sales.drop(columns=version_col, axis=1, inplace=True)

        sales_df = sales.rename(columns={day_col: time_col, like_item_actual_col: sales_col})

        sales_df_start_dates = sales_df.groupby([pl_item_col])[time_col].min().reset_index()
        sales_df_start_dates.rename(columns={time_col: start_date_col}, inplace=True)

        # filter combinations in sales df which have data for more than history periods
        curr_day = sales_df[time_col].max()
        filter_date = curr_day - pd.DateOffset(days=history_periods)

        sales_df_start_dates = sales_df_start_dates[
            sales_df_start_dates[start_date_col] <= filter_date
        ]

        sales_df = sales_df.merge(sales_df_start_dates, how="inner")

        logger.info("Aggregating feature importance ...")

        # Defining the list of columns that we want to match
        key_cols = [item_col]
        distance_cols = list(FeatureWeights[feat_col].unique())

        logger.info("Creating item to planning item mapping dictionary ...")
        # create item to pl item mapping
        item_to_pl_item_mapping = dict(
            zip(
                list(Item[item_col]),
                list(Item[pl_item_col]),
            )
        )

        """ # filter new items from item master
        required_cols = [item_col, pl_item_col]
        new_item_master = Item[Item[new_item_flag_col] == 1]

        if len(new_item_master) == 0:
            logger.warning(
                "No new items found in item master for slice : {}, kindly check item master data ...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe for this slice ...")
            return like_sku_result

        new_item_master = new_item_master[required_cols]

        logger.info("FeatureLevel : {}".format(FeatureLevel))

        # join with assortment mapping
        AssortmentMapping = AssortmentMapping.merge(
            new_item_master, on=pl_item_col, how="inner"
        )
        logger.info("------------------------------------------------")

        logger.info(f"sales shape : {sales_df.shape}")
        # exclude the new items from sales dataframe so that these don't come up as recommendation
        # new_item_list = list(new_item_master[item_col].unique())
        # sales_df = sales_df[~sales_df[item_col].isin(new_item_list)]
        # logger.info(
        #     f"after excluding new items, sales shape : {sales_df.shape}"
        # )   """

        pl_item_to_item_mapping = {p_item: item for item, p_item in item_to_pl_item_mapping.items()}
        AssortmentMapping[item_col] = AssortmentMapping[pl_item_col].map(pl_item_to_item_mapping)

        item_group_df = AssortmentMapping[
            [
                item_col,
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
            ]
        ]
        like_sku_result_list = list()
        for (
            the_item,
            the_planning_region,
            the_planning_account,
            the_planning_channel,
            the_planning_pnl,
            the_planning_demand_domain,
            the_planning_location,
        ) in item_group_df.itertuples(index=False):
            logger.info("-------------- the_item : {}".format(the_item))
            logger.info("-------------- the_planning_region : {}".format(the_planning_region))
            logger.info("-------------- the_planning_account : {}".format(the_planning_account))
            logger.info("-------------- the_planning_channel : {}".format(the_planning_channel))
            logger.info("-------------- the_planning_pnl : {}".format(the_planning_pnl))
            logger.info("-------------- the_planning_location : {}".format(the_planning_location))

            try:
                the_new_item_df = Item.loc[Item[item_col] == the_item]

                if the_new_item_df.empty:
                    logger.info("no product attributes found...")
                    continue

                # Filtering featuring importance values at the required level
                the_new_item_feature_level_value = the_new_item_df[search_level].unique()[0]
                logger.info("---- FeatureLevel value : {}".format(the_new_item_feature_level_value))

                feature_imp = FeatureWeights[
                    FeatureWeights[search_level] == the_new_item_feature_level_value
                ][[feat_col, feat_imp_col]]

                if len(feature_imp) == 0:
                    logger.warning(
                        "No feature weights found for {} : {}".format(
                            FeatureLevel, the_new_item_feature_level_value
                        )
                    )
                    continue

                # perform required data transformation steps
                feature_imp = prepare_feat_imp_agg(feature_imp, feat_col)

                # Filter sales data at customer group level and exclude sales for the new item
                the_cust_group_sales_df_raw = sales_df[
                    (sales_df[planning_pnl_col] == the_planning_pnl)
                    & (sales_df[planning_region_col] == the_planning_region)
                    & (sales_df[planning_channel_col] == the_planning_channel)
                    & (sales_df[planning_account_col] == the_planning_account)
                    & (sales[planning_location_col] == the_planning_location)
                    & (sales_df[item_col] != the_item)
                ]

                if the_cust_group_sales_df_raw.empty:
                    logger.info(
                        "no sales for group {}, {}, {}, {}, {} ...".format(
                            the_planning_pnl,
                            the_planning_region,
                            the_planning_channel,
                            the_planning_account,
                            the_planning_location,
                        )
                    )
                    continue

                # Restrict the sales df to contain item data only where the search space is same as new item
                new_item_attribute_value = the_new_item_df[search_level].iloc[0]
                logger.info(
                    "{} value for {} : {}".format(search_level, the_item, new_item_attribute_value)
                )

                logger.info(
                    "Filtering sales data with {} == {}".format(
                        search_level, new_item_attribute_value
                    )
                )
                filter_clause = (
                    the_cust_group_sales_df_raw[search_level] == new_item_attribute_value
                )
                the_cust_group_sales_df_filtered = the_cust_group_sales_df_raw[filter_clause]

                if len(the_cust_group_sales_df_filtered) == 0:
                    logger.warning(
                        "No sales data found where {} value is {}".format(
                            search_level, new_item_attribute_value
                        )
                    )
                    continue

                the_cust_group_sales_df = the_cust_group_sales_df_filtered.groupby(
                    [item_col], as_index=False
                )[sales_col].sum()

                # Join with item master to get item attributes - later category encoded
                # this dataframe contains data for multiple items in that customer group
                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Item,
                    how="left",
                    on=item_col,
                )

                cols_to_encode = [x for x in distance_cols if x not in numerical_cols_list]

                # convert item attribute columns to lower case string for category encoding
                the_cust_group_sales_df[cols_to_encode] = the_cust_group_sales_df[
                    cols_to_encode
                ].astype("object")

                the_cust_group_sales_df[numerical_cols_list] = the_cust_group_sales_df[
                    numerical_cols_list
                ].astype("float")

                the_new_item_df[cols_to_encode] = the_new_item_df[cols_to_encode].astype("object")

                the_new_item_df[numerical_cols_list] = the_new_item_df[numerical_cols_list].astype(
                    "float"
                )

                for the_col in cols_to_encode:
                    the_cust_group_sales_df[the_col] = (
                        the_cust_group_sales_df[the_col].astype(str).str.lower()
                    )
                    the_new_item_df[the_col] = the_new_item_df[the_col].astype(str).str.lower()

                encoder = ce.TargetEncoder(cols=cols_to_encode, drop_invariant=False)
                logger.info("encoding categorical columns ...")
                # category encode item attributes for all items - new item won't be present here since it has no sale
                encoded_df = encoder.fit_transform(
                    the_cust_group_sales_df.drop(sales_col, axis=1),
                    the_cust_group_sales_df[sales_col],
                )

                logger.info("encoding npi categorical columns ...")
                # category encode the relevant planning item using weights from all items
                npi_df_encoded = encoder.transform(the_new_item_df)

                std_scaler = StandardScaler()

                logger.info("scaling distance columns ...")

                # Scale the category encoded numbers - for all items in customer group
                encoded_df[distance_cols] = std_scaler.fit_transform(encoded_df[distance_cols])

                # Scale the category encoded numbers - for the new item
                npi_df_encoded[distance_cols] = std_scaler.transform(npi_df_encoded[distance_cols])

                # for distance_cols not present in feature_imp, provide them weight 0
                non_feat_cols = [
                    col for col in distance_cols if col not in list(feature_imp.columns)
                ]

                if len(non_feat_cols) != 0:
                    for col in non_feat_cols:
                        feature_imp[col] = 0

                # Multiply category encoded numbers with weights - for all items in customer group
                encoded_df_feat_mul = np.multiply(encoded_df[distance_cols], feature_imp)

                # Add key column to resulting matrix
                encoded_df = pd.concat([encoded_df[key_cols], encoded_df_feat_mul], axis=1)

                # Multiple category encoded numbers with feature weights - for the new item
                npi_feature_mul = np.multiply(npi_df_encoded[distance_cols], feature_imp)

                # Add key column to resulting matrix
                npi_df_encoded = pd.concat([npi_df_encoded[key_cols], npi_feature_mul], axis=1)

                logger.info("Calculating euclidean distance ...")

                # Calculating euclidean distance between all other items and the relevant item
                dist_matrix = scipy.spatial.distance.cdist(
                    encoded_df.loc[:, distance_cols],
                    npi_df_encoded.loc[:, distance_cols],
                    metric="euclidean",
                )
                dist_matrix = pd.DataFrame(dist_matrix)

                # Assign column name - will be the relevant item
                dist_matrix.columns = npi_df_encoded.rename(columns={item_col: new_product_col})[
                    new_product_col
                ]

                # Assign index - will be all other items to which we are comparing
                dist_matrix.index = encoded_df.rename(columns={item_col: like_product_col})[
                    like_product_col
                ]

                # Create graph dataframe with tabular form containing distances from new item to each item
                like_item_graph_df = dist_matrix.unstack().reset_index(
                    name=like_product_distance_col
                )

                # Rounding
                like_item_graph_df[like_product_distance_col] = like_item_graph_df[
                    like_product_distance_col
                ].round(6)

                # Add planning item column for the new item and like items identified
                like_item_graph_df[new_product_pl_item_col] = like_item_graph_df[
                    new_product_col
                ].map(item_to_pl_item_mapping)
                like_item_graph_df[like_product_pl_item_col] = like_item_graph_df[
                    like_product_col
                ].map(item_to_pl_item_mapping)

                cols_to_group_by = [
                    new_product_pl_item_col,
                    like_product_pl_item_col,
                ]

                # Identify minimum distance among the from and to planning item combinations
                like_item_graph_df = (
                    like_item_graph_df.groupby(cols_to_group_by)
                    .min()[[like_product_distance_col]]
                    .reset_index()
                )

                # Merging with the start dates of the Like planning item Item to use as tie break
                like_item_graph_df = pd.merge(
                    like_item_graph_df,
                    sales_df_start_dates,
                    how="left",
                    left_on=like_product_pl_item_col,
                    right_on=pl_item_col,
                )

                # Sort based on distance and dates
                like_item_graph_df = like_item_graph_df.sort_values(
                    by=[
                        new_product_pl_item_col,
                        like_product_distance_col,
                        start_date_col,
                    ]
                )

                # Create dense ranks
                like_item_graph_df[like_product_rank_col] = like_item_graph_df.groupby(
                    [new_product_pl_item_col]
                )[like_product_distance_col].rank("dense", ascending=True)

                # Sort on ranking
                like_item_graph_df.sort_values(
                    by=[
                        new_product_pl_item_col,
                        like_product_rank_col,
                        start_date_col,
                    ],
                    inplace=True,
                )

                # Break the tie
                like_item_graph_df[like_product_rank_col] = like_item_graph_df.groupby(
                    [new_product_pl_item_col]
                )[like_product_distance_col].rank("first", ascending=True)
                like_item_graph_df.drop(start_date_col, axis=1, inplace=True)

                # add the group values
                like_item_graph_df[from_planning_pnl_col] = str(the_planning_pnl)
                like_item_graph_df[from_planning_channel_col] = str(the_planning_channel)
                like_item_graph_df[from_planning_region_col] = str(the_planning_region)
                like_item_graph_df[from_planning_account_col] = str(the_planning_account)
                like_item_graph_df[from_planning_location_col] = str(the_planning_location)

                logger.info("Converting distance to similarity ...")
                # Converting distance to Similarity score
                like_item_graph_df[like_product_similarity] = (
                    1 / (1 + like_item_graph_df[like_product_distance_col])
                ).round(6)

                logger.info("Marking system default like item ...")
                # Adding a flag to mark most similar item (rank=1) as system default like item
                like_item_graph_df[like_product_flag_col] = np.where(
                    like_item_graph_df[like_product_rank_col] == 1, 1, 0
                )

                # Filter relevant number of match items
                like_item_graph_df = like_item_graph_df.loc[
                    like_item_graph_df[like_product_rank_col] <= int(match_num),
                    :,
                ]

                logger.info("Formatting output ...")
                # Output formatting
                like_item_graph_df.insert(loc=0, column=version_col, value=input_version)
                # Rename columns
                like_item_graph_df.rename(
                    columns={
                        new_product_pl_item_col: from_planning_item_col,
                        like_product_pl_item_col: to_planning_item_col,
                    },
                    inplace=True,
                )

                like_item_graph_df = like_item_graph_df[column_order]
                like_item_graph_df.columns = cols_required_in_output

                # Append results to master list
                like_sku_result_list.append(like_item_graph_df)

            except Exception as e:
                logger.exception(
                    "Exception {} while processing item : {}, channel : {}, pnl : {}, region : {}, account : {}, location: {}".format(
                        e,
                        the_item,
                        the_planning_channel,
                        the_planning_pnl,
                        the_planning_region,
                        the_planning_account,
                        the_planning_location,
                    )
                )

        logger.info("------- All Iterations complete -----------")
        # convert list of dataframes to dataframe
        like_sku_result = concat_to_dataframe(like_sku_result_list)

        if like_sku_result.empty:
            logger.info("No like-sku matches found, returing empty dataframe ...")
            like_sku_result = pd.DataFrame(columns=cols_required_in_output)

        logger.info("------------ Printing head of output dataframe : head-----------")
        logger.info(like_sku_result.head())
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        like_sku_result = pd.DataFrame(columns=cols_required_in_output)

    return like_sku_result

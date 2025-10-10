"""Generate Like Item Match plugin for Flexible NPI."""

import logging
from typing import Optional

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

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")


def prepare_feat_imp_agg(df: pd.DataFrame, feat_col: str) -> pd.DataFrame:
    """Perform necessary transformations on the df provided and returns the same."""
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


def process_num_cols(df: pd.DataFrame, numerical_cols_list: list):
    """Process numerical columns."""
    cols_with_non_convertible_values = [
        col
        for col in df.columns
        if any(
            not isinstance(val, (int, float))
            and not (isinstance(val, str) and val.replace(".", "", 1).isdigit())
            and val.lower() != "nan"
            for val in df[col]
        )
    ]

    num_cols = list(
        (set(df.columns) & set(numerical_cols_list)) - set(cols_with_non_convertible_values)
    )
    df[num_cols] = df[num_cols].replace({np.nan: "0", "NULL": "0", "": "0"})
    df[num_cols] = df[num_cols].fillna("0")
    return df


def npi_and_tenant_col_level_mapping(df):
    """Map Npi columns with tenant columns."""
    col_mapping = {
        "Item.[NPI Item]": "Item.[Planning Item]",
        "Account.[NPI Account]": "Account.[Planning Account]",
        "Channel.[NPI Channel]": "Channel.[Planning Channel]",
        "Region.[NPI Region]": "Region.[Planning Region]",
        "PnL.[NPI PnL]": "PnL.[Planning PnL]",
        "Demand Domain.[NPI Demand Domain]": "Demand Domain.[Planning Demand Domain]",
        "Location.[NPI Location]": "Location.[Planning Location]",
        "from.[Item].[Planning Item]": "from.[Item].[NPI Item]",
        "from.[Account].[Planning Account]": "from.[Account].[NPI Account]",
        "from.[Channel].[Planning Channel]": "from.[Channel].[NPI Channel]",
        "from.[Region].[Planning Region]": "from.[Region].[NPI Region]",
        "from.[PnL].[Planning PnL]": "from.[PnL].[NPI PnL]",
        "from.[Location].[Planning Location]": "from.[Location].[NPI Location]",
        "from.[Demand Domain].[Planning Demand Domain]": "from.[Demand Domain].[NPI Demand Domain]",
        "to.[Item].[Planning Item]": "to.[Item].[NPI Item]",
        "to.[Account].[Planning Account]": "to.[Account].[NPI Account]",
        "to.[Channel].[Planning Channel]": "to.[Channel].[NPI Channel]",
        "to.[Region].[Planning Region]": "to.[Region].[NPI Region]",
        "to.[PnL].[Planning PnL]": "to.[PnL].[NPI PnL]",
        "to.[Location].[Planning Location]": "to.[Location].[NPI Location]",
        "to.[Demand Domain].[Planning Demand Domain]": "to.[Demand Domain].[NPI Demand Domain]",
    }

    df.rename(columns=col_mapping, inplace=True)

    return df


def load_output_data_from_cloud(
    df_keys, cloud_storage_type: str = "google", local_file_path: Optional[str] = None
):
    """Read full scope feature weights.

    Parameters
    ----------
    df_keys : _type_
        bucket to read the input
    cloud_storage_type : str, optional
        Cloud storage type, by default "google"
    local_file_path : str, optional
        if running the code in the local provide the full scope feature weights path, by default None

    Returns
    -------
    pd.Dataframe
    """
    import os

    if str(os.environ).lower().count("o9") > 80:
        logger.info("Prepare reading full scope feature weights data from the cloud.")
        os.environ["FileStoreType"] = cloud_storage_type
        import glob

        from o9cloudutils import cloud_storage_utils, user_storage_path

        bucket = "bucket_slice {}".format(df_keys)
        local_storage_path = os.path.join(user_storage_path, bucket)
        test_folder_path = os.path.join(local_storage_path, "featureweights")
        os.makedirs(test_folder_path)

        try:
            """
            Code for storage pull
            """
            os.makedirs(test_folder_path, exist_ok=True)
            logger.debug("********List of files before storage_pull*******")
            logger.debug(os.listdir(local_storage_path))
            for filename in glob.iglob(local_storage_path + "**/**/*", recursive=True):
                logger.debug(filename)

            logger.info("Reading files from storage..............................................")
            local_storage_path = os.path.join(user_storage_path, bucket)
            value = cloud_storage_utils.storage_pull(bucket, local_storage_path, overwrite=True)
            if value:
                logger.info(f"successfully pulled data from cloud bucket: {local_storage_path}")
            else:
                logger.exception(
                    f"Error occured during pull the data from the cloud for slice: {df_keys}"
                )

            logger.info("********List of files in bucket after storage pull************")
            for filename in glob.iglob(local_storage_path + "**/**/*", recursive=True):
                logger.debug(filename)

            input_df = pd.read_csv(os.path.join(test_folder_path, "flexinpifeatureweightsfull.csv"))
            logger.info(input_df.head())
        except Exception as e:
            logger.exception(f"Couldn't pull folder. Error: {e}")

        return input_df
    else:
        logger.debug("Reading the full scope feature weights data from local.")
        print("Script is running in the local with user provided input path/current dir.")
        input_df = None
        if local_file_path is None:
            print("Directory is missing. Reading file from the current directory.")
            local_file_path = os.getcwd()
        else:
            print(f"User provided path: \n{local_file_path}")
        if "csv" in local_file_path:
            input_df = pd.read_csv(local_file_path)
        elif "parquet" in local_file_path:
            input_df = pd.read_parquet(local_file_path)
        else:
            raise ValueError("Unrecognized file type. Can't read the full scope feature weights")

        if not isinstance(input_df, pd.DataFrame):
            raise ValueError("Provide the full scope feature weights path")
        else:
            return input_df


def merge_with_attributes(df, attribute, search_space_col, planning_level_col, lower_level_col):
    """Merge with the attributes."""
    df = pd.merge(
        df,
        attribute[[search_space_col, planning_level_col, lower_level_col]]
        .loc[
            :,
            ~attribute[
                [search_space_col, planning_level_col, lower_level_col]
            ].columns.duplicated(),
        ]
        .drop_duplicates(),
        on=planning_level_col,
        how="left",
    )

    return df


col_mapping = {
    "630 Initiative Like Assortment Match.[Like Assortment Rank L0]": float,
    "630 Initiative Like Assortment Match.[Like Assortment Distance L0]": float,
    "630 Initiative Like Assortment Match.[Like Assortment Similarity L0]": float,
    "630 Initiative Like Assortment Match.[System Recommended Like Assortment L0]": bool,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    sales,
    Item,
    Location,
    Account,
    Channel,
    DemandDomain,
    PnL,
    Region,
    pl_item,
    parameters,
    SearchSpace,
    FeatureWeights,
    FullScopeWeights,
    numerical_cols,
    FeatureLevel,
    ConsensusFcst,
    ReadFromHive,
    generate_match_assortment,
    df_keys,
    CurrentDay: pd.DataFrame = None,
    TimeFormat="%d-%b-%y",
):
    """Entry point of the script."""
    plugin_name = "DP126SystemLikeItemMatch"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    item_col = "Item.[Item]"
    account_col = "Account.[Account]"
    loc_col = "Location.[Location]"
    channel_col = "Channel.[Channel]"
    pnl_col = "PnL.[PnL]"
    demand_domain_col = "Demand Domain.[Demand Domain]"
    region_col = "Region.[Region]"
    user_feat_weight_col = "User Feature Weight by Level"
    sys_feat_weight_col = "System Feature Weight by Level"
    feat_col = "Feature"
    item_feat_col = "Item Feature.[Item Feature]"
    feat_imp_col = "Feature_Importance"
    feat_ratio_col = "Feature_Importance Normalization Ratio"
    req_item_hierarchy_cols = [
        "Item.[L1]",
        "Item.[L2]",
        "Item.[L3]",
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
    day_col = "Time.[Day]"
    week_col = "Time.[Weel]"
    partial_week_col = "Time.[Partial Week]"
    month_col = "Time.[Month]"
    pl_month_col = "Time.[Planning Month]"
    daykey_col = "Time.[DayKey]"
    time_col = "Time"
    sales_col = "Sales"
    start_date_col = "Start_Date"
    like_assortment_distance_col = "Like Assortment Distance"
    like_assortment_rank_col = "Like Assortment Rank"
    like_assortment_flag_col = "Like Assortment Flag"
    like_assortment_col = "Like Assortment"
    like_assortment_similarity = "Like Assortment Similarity"
    like_product_col = "Like Product"
    like_account_col = "Like Account"
    like_channel_col = "Like Channel"
    like_location_col = "Like Location"
    like_region_col = "Like Region"
    like_pnl_col = "Like PnL"
    like_dd_col = "Like Demand Domain"
    new_product_col = "New Product"
    new_account_col = "New Account"
    new_channel_col = "New Channel"
    new_location_col = "New Location"
    new_region_col = "New Region"
    new_pnl_col = "New PnL"
    new_dd_col = "New Demand Domain"
    new_product_pl_item_col = "New Product Planning Item"
    new_product_pl_account_col = "New Planning Account"
    new_product_pl_channel_col = "New Planning Channel"
    new_product_pl_location_col = "New Planning Location"
    new_product_pl_region_col = "New Planning Region"
    new_product_pl_pnl_col = "New Planning PnL"
    new_product_pl_dd_col = "New Planning Demand Domain"
    new_assortment_col = "New Assortment"
    like_product_pl_item_col = "Like Product Planning Item"
    like_product_pl_account_col = "Like Planning Account"
    like_product_pl_channel_col = "Like Planning Channel"
    like_product_pl_loc_col = "Like Planning Location"
    like_product_pl_region_col = "Like Planning Region"
    like_product_pl_pnl_col = "Like Planning PnL"
    like_product_pl_dd_col = "Like Planning Demand Domain"
    numerical_cols_list = []
    generate_system_like_item_match_col = "Generate System Like Item Match Assortment L0"

    like_item_search_space = "Like Item Search Space by Level"
    like_account_search_space = "Like Account Search Space by Level"
    like_channel_search_space = "Like Channel Search Space by Level"
    like_demand_domain_search_space = "Like Demand Domain Search Space by Level"
    like_location_search_space = "Like Location Search Space by Level"
    like_pnl_search_space = "Like PnL Search Space by Level"
    like_region_search_space = "Like Region Search Space by Level"

    from_planning_item_col = "from.[Item].[Planning Item]"
    from_planning_pnl_col = "from.[PnL].[Planning PnL]"
    from_planning_region_col = "from.[Region].[Planning Region]"
    from_planning_channel_col = "from.[Channel].[Planning Channel]"
    from_planning_account_col = "from.[Account].[Planning Account]"
    from_planning_location_col = "from.[Location].[Planning Location]"
    from_planning_demand_domain_col = "from.[Demand Domain].[Planning Demand Domain]"
    from_npi_item_col = "from.[Item].[NPI Item]"
    from_npi_account_col = "from.[Account].[NPI Account]"
    from_npi_channel_col = "from.[Channel].[NPI Channel]"
    from_npi_region_col = "from.[Region].[NPI Region]"
    from_npi_pnl_col = "from.[PnL].[NPI PnL]"
    from_npi_location_col = "from.[Location].[NPI Location]"
    from_npi_demand_domain_col = "from.[Demand Domain].[NPI Demand Domain]"
    from_data_object_col = "from.[Data Object].[Data Object]"
    from_initiative_col = "from.[Initiative].[Initiative]"

    to_planning_item_col = "to.[Item].[Planning Item]"
    to_planning_pnl_col = "to.[PnL].[Planning PnL]"
    to_planning_region_col = "to.[Region].[Planning Region]"
    to_planning_channel_col = "to.[Channel].[Planning Channel]"
    to_planning_account_col = "to.[Account].[Planning Account]"
    to_planning_location_col = "to.[Location].[Planning Location]"
    to_planning_demand_domain_col = "to.[Demand Domain].[Planning Demand Domain]"
    to_npi_item_col = "to.[Item].[NPI Item]"
    to_npi_account_col = "to.[Account].[NPI Account]"
    to_npi_channel_col = "to.[Channel].[NPI Channel]"
    to_npi_region_col = "to.[Region].[NPI Region]"
    to_npi_pnl_col = "to.[PnL].[NPI PnL]"
    to_npi_location_col = "to.[Location].[NPI Location]"
    to_npi_demand_domain_col = "to.[Demand Domain].[NPI Demand Domain]"

    NumLikeAssortment_col = "Num Like Assortment by Level"
    LikeAssortmentSearchHistory_col = "Like Assortment Search History Measure by Level"
    LikeAssortmentMinHistoryPeriod_col = "Like Assortment Min History Period by Level"
    LikeAssortmentLaunchPeriod_col = "Like Assortment Launch Period by Level"
    data_object_col = "Data Object.[Data Object]"
    initiative_col = "Initiative.[Initiative]"

    # output measures
    cols_required_in_output = [
        version_col,
        from_initiative_col,
        from_data_object_col,
        from_npi_item_col,
        from_npi_account_col,
        from_npi_channel_col,
        from_npi_region_col,
        from_npi_pnl_col,
        from_npi_location_col,
        from_npi_demand_domain_col,
        to_npi_item_col,
        to_npi_account_col,
        to_npi_channel_col,
        to_npi_region_col,
        to_npi_pnl_col,
        to_npi_location_col,
        to_npi_demand_domain_col,
        "630 Initiative Like Assortment Match.[Like Assortment Rank L0]",
        "630 Initiative Like Assortment Match.[Like Assortment Distance L0]",
        "630 Initiative Like Assortment Match.[Like Assortment Similarity L0]",
        "630 Initiative Like Assortment Match.[System Recommended Like Assortment L0]",
    ]
    like_assortment_result = pd.DataFrame(columns=cols_required_in_output)
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
            from_initiative_col,
            from_data_object_col,
            from_planning_item_col,
            from_planning_account_col,
            from_planning_channel_col,
            from_planning_region_col,
            from_planning_pnl_col,
            from_planning_location_col,
            from_planning_demand_domain_col,
            to_planning_item_col,
            to_planning_account_col,
            to_planning_channel_col,
            to_planning_region_col,
            to_planning_pnl_col,
            to_planning_location_col,
            to_planning_demand_domain_col,
            "Like Assortment Rank",
            "Like Assortment Distance",
            "Like Assortment Similarity",
            "Like Assortment Flag",
        ]

        # As discussed there would be only 1 Data Object value for the plugin run
        item_search_space = "Item.[" + SearchSpace[like_item_search_space].iloc[0] + "]"
        account_search_space = "Account.[" + SearchSpace[like_account_search_space].iloc[0] + "]"
        channel_search_space = "Channel.[" + SearchSpace[like_channel_search_space].iloc[0] + "]"
        demand_domain_search_space = (
            "Demand Domain.[" + SearchSpace[like_demand_domain_search_space].iloc[0] + "]"
        )
        pnl_search_space = "PnL.[" + SearchSpace[like_pnl_search_space].iloc[0] + "]"
        location_search_space = "Location.[" + SearchSpace[like_location_search_space].iloc[0] + "]"
        region_search_space = "Region.[" + SearchSpace[like_region_search_space].iloc[0] + "]"

        match_num = int(parameters[NumLikeAssortment_col].iloc[0])
        assert match_num > 0, "Num Like Assortments have to be strictly positive ..."

        history_measure = parameters[LikeAssortmentSearchHistory_col].iloc[0]

        if pd.isna(parameters[LikeAssortmentMinHistoryPeriod_col].iloc[0]):
            history_periods = 365
        else:
            history_periods = int(parameters[LikeAssortmentMinHistoryPeriod_col].iloc[0])

        like_assortment_launch_period = parameters[LikeAssortmentLaunchPeriod_col].iloc[0]

        if pd.isna(like_assortment_launch_period):
            like_assortment_launch_period = 730
        else:
            like_assortment_launch_period = int(like_assortment_launch_period)

        if len(sales) == 0:
            logger.warning("No data found in sales df for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return like_assortment_result

        input_version = sales[version_col].unique()[0]

        logger.info("item search_level: {}".format(item_search_space))
        logger.info("account search_level: {}".format(account_search_space))
        logger.info("channel search_level: {}".format(channel_search_space))
        logger.info("location search_level: {}".format(location_search_space))
        logger.info("region search_level: {}".format(region_search_space))
        logger.info("pnl search_level: {}".format(pnl_search_space))
        logger.info("demand domain search_level: {}".format(demand_domain_search_space))
        logger.info("match_num: {}".format(match_num))
        logger.info("history_measure: {}".format(history_measure))
        logger.info("history_periods(in days): {}".format(history_periods))

        logger.info("Processing Item Attributes ...")
        Item[numerical_cols_list] = Item[numerical_cols_list].replace(
            {np.nan: "0", "NULL": "0", "": "0"}
        )
        Item[numerical_cols_list] = Item[numerical_cols_list].fillna("0")
        Item.replace({np.nan: "dummy"}, inplace=True)
        pl_item.replace({np.nan: "dummy"}, inplace=True)

        Item = Item.merge(pl_item, on=pl_item_col, how="left")  # joining with item attributes

        Item = Item.replace({"": "dummy"})
        Item = Item.fillna("dummy")

        logger.info("Processing Location Attributes ...")
        Location = process_num_cols(Location, numerical_cols_list)
        Location.replace({np.nan: "dummy"}, inplace=True)
        Location = Location.replace({"": "dummy"})
        Location = Location.fillna("dummy")

        logger.info("Processing Account Attributes ...")
        Account = process_num_cols(Account, numerical_cols_list)
        Account.replace({np.nan: "dummy"}, inplace=True)
        Account = Account.replace({"": "dummy"})
        Account = Account.fillna("dummy")

        logger.info("Processing Channel Attributes ...")
        Channel = process_num_cols(Channel, numerical_cols_list)
        Channel.replace({np.nan: "dummy"}, inplace=True)
        Channel = Channel.replace({"": "dummy"})
        Channel = Channel.fillna("dummy")

        logger.info("Processing Region Attributes ...")
        Region = process_num_cols(Region, numerical_cols_list)
        Region.replace({np.nan: "dummy"}, inplace=True)
        Region = Region.replace({"": "dummy"})
        Region = Region.fillna("dummy")

        logger.info("Processing PnL Attributes ...")
        PnL = process_num_cols(PnL, numerical_cols_list)
        PnL.replace({np.nan: "dummy"}, inplace=True)
        PnL = PnL.replace({"": "dummy"})
        PnL = PnL.fillna("dummy")

        logger.info("Processing Demand Domain Attributes ...")
        DemandDomain = process_num_cols(DemandDomain, numerical_cols_list)
        DemandDomain.replace({np.nan: "dummy"}, inplace=True)
        DemandDomain = DemandDomain.replace({"": "dummy"})
        DemandDomain = DemandDomain.fillna("dummy")

        # filter using the like item match assortment col
        generate_match_assortment = generate_match_assortment[
            pd.notna(generate_match_assortment[generate_system_like_item_match_col])
        ]
        generate_match_assortment[generate_system_like_item_match_col] = generate_match_assortment[
            generate_system_like_item_match_col
        ].astype(int)
        generate_match_assortment = generate_match_assortment[
            generate_match_assortment[generate_system_like_item_match_col] == 1
        ]

        if len(generate_match_assortment) == 0:
            logger.warning(
                "No records found in GenerateSystemLikeItemMatchAssortment for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return like_assortment_result

        generate_match_assortment = generate_match_assortment.astype(str)

        # Column mapping | NPI level to tenant item level
        generate_match_assortment = npi_and_tenant_col_level_mapping(generate_match_assortment)

        generate_match_assortment.drop([generate_system_like_item_match_col], axis=1, inplace=True)

        logger.info("Processing 7D Feature Weights ....")

        if len(FullScopeWeights) == 0:
            logger.warning("No records found in FullScopeWeights for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return like_assortment_result

        FullScopeWeights = FullScopeWeights[FullScopeWeights[version_col] == input_version]
        FullScopeWeights = FullScopeWeights.astype(str)
        FullScopeWeights[sys_feat_weight_col] = FullScopeWeights[sys_feat_weight_col].astype(float)
        FullScopeWeights.rename(
            columns={
                item_feat_col: feat_col,
            },
            inplace=True,
        )

        logger.info("Processing Item Feature Weights ...")
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

        FeatureWeights = FeatureWeights.groupby([item_search_space, feat_col], as_index=False)[
            feat_imp_col
        ].sum()

        # Merge the feature weights to full scope weights
        FeatureWeights[item_search_space] = FeatureWeights[item_search_space].str.lower()
        FullScopeWeights = FullScopeWeights.merge(
            FeatureWeights, on=[item_search_space, feat_col], how="left"
        )
        FullScopeWeights[sys_feat_weight_col] = FullScopeWeights[sys_feat_weight_col].where(
            FullScopeWeights[feat_imp_col].isna(),
            FullScopeWeights[feat_imp_col],
        )
        FullScopeWeights[feat_imp_col] = FullScopeWeights[sys_feat_weight_col]
        FullScopeWeights.drop([sys_feat_weight_col], axis=1, inplace=True)

        FullScopeWeights[feat_ratio_col] = FullScopeWeights.groupby(
            [
                item_search_space,
                location_search_space,
                account_search_space,
                channel_search_space,
                pnl_search_space,
                region_search_space,
                demand_domain_search_space,
            ]
        )[feat_imp_col].transform("sum")

        FullScopeWeights[feat_imp_col] = (
            FullScopeWeights[feat_imp_col] / FullScopeWeights[feat_ratio_col]
        )

        logger.info("sales data before joining with ConsensusFcst, shape : {}".format(sales.shape))
        ConsensusFcst = ConsensusFcst[
            [
                pl_item_col,
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
                planning_demand_domain_col,
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
                planning_demand_domain_col,
            ],
            how="inner",
        )
        logger.info("sales data after joining with ConsensusFcst, shape : {}".format(sales.shape))

        if len(sales) == 0:
            logger.warning(
                f"No sales data found after filtering for the following combinations below for slice : {df_keys}"
            )
            logger.warning("Returning empty dataframe ...")
            return like_assortment_result

        # Sales time level indentifying
        if day_col in sales.columns:
            sales_time_col = day_col
        elif week_col in sales.columns:
            sales_time_col = week_col
        elif partial_week_col in sales.columns:
            sales_time_col = partial_week_col
        elif month_col in sales.columns:
            sales_time_col = month_col
        elif pl_month_col in sales.columns:
            sales_time_col = pl_month_col
        else:
            logger.exception("Could not identified the date column in sales data.")
            logger.warning("Returning empty dataframe ...")
            return like_assortment_result

        sales.drop(columns=version_col, axis=1, inplace=True)
        sales = pd.merge(
            sales, Item[[item_col, pl_item_col]].drop_duplicates(), how="left", on=item_col
        )
        sales_filtered = (
            sales[[item_col, pl_item_col, sales_time_col]].drop_duplicates().copy(deep=True)
        )

        # Converting day column from category to string and then to datetime, otherwise aggregation (min) to get start dates fails
        sales_filtered[sales_time_col] = sales_filtered[sales_time_col].astype("string[pyarrow]")
        sales_filtered[sales_time_col] = pd.to_datetime(
            sales_filtered[sales_time_col], format=TimeFormat, cache=True
        )

        sales_df = sales.rename(columns={sales_time_col: time_col, like_item_actual_col: sales_col})
        sales_filtered = sales_filtered.rename(columns={sales_time_col: time_col})

        sales_df_start_dates = sales_filtered.groupby([pl_item_col])[time_col].min().reset_index()
        sales_df_start_dates.rename(columns={time_col: start_date_col}, inplace=True)

        # get current day
        if isinstance(CurrentDay, pd.DataFrame) and len(CurrentDay) > 0:
            curr_day = pd.to_datetime(CurrentDay[daykey_col].values[0])
        else:
            logger.warning("Current Day is missing. Considering max day of the sales.")
            curr_day = sales_filtered[time_col].max()

        # filter combinations in sales df which have data for more than history periods

        filter_date = curr_day - pd.DateOffset(days=history_periods)

        sales_df_start_dates = sales_df_start_dates[
            sales_df_start_dates[start_date_col] <= filter_date
        ]

        # filter combinations which have launch >= launch window
        filter_date_launch = curr_day - pd.DateOffset(days=like_assortment_launch_period)
        sales_df_start_dates = sales_df_start_dates[
            sales_df_start_dates[start_date_col] >= filter_date_launch
        ]

        # Agg by time
        sales_df = sales_df.groupby(
            [
                item_col,
                pl_item_col,
                planning_region_col,
                planning_account_col,
                planning_channel_col,
                planning_pnl_col,
                planning_location_col,
                planning_demand_domain_col,
            ],
            as_index=False,
        )[sales_col].sum()

        # filterting based on start date
        sales_df = sales_df.merge(
            sales_df_start_dates[[pl_item_col]].drop_duplicates(), how="inner"
        )
        sales_df.drop([pl_item_col], axis=1, inplace=True)

        # Merging sales with dimension attributes
        sales_df = merge_with_attributes(sales_df, Item, item_search_space, item_col, item_col)
        sales_df = merge_with_attributes(
            sales_df, Account, account_search_space, planning_account_col, account_col
        )
        sales_df = merge_with_attributes(
            sales_df, Channel, channel_search_space, planning_channel_col, channel_col
        )
        sales_df = merge_with_attributes(
            sales_df, Region, region_search_space, planning_region_col, region_col
        )
        sales_df = merge_with_attributes(
            sales_df, Location, location_search_space, planning_location_col, loc_col
        )
        sales_df = merge_with_attributes(sales_df, PnL, pnl_search_space, planning_pnl_col, pnl_col)
        sales_df = merge_with_attributes(
            sales_df,
            DemandDomain,
            demand_domain_search_space,
            planning_demand_domain_col,
            demand_domain_col,
        )

        # if search space would have planning level then duplicate cols would be created, removing the dups
        sales_df = sales_df.loc[:, ~sales_df.columns.duplicated()]

        logger.info("Aggregating feature importance ...")

        # Defining the list of columns that we want to match
        key_cols = [
            item_col,
            demand_domain_col,
            channel_col,
            account_col,
            loc_col,
            region_col,
            pnl_col,
        ]
        distance_cols = list(FullScopeWeights[feat_col].unique())

        logger.info("Creating item to planning item mapping dictionary ...")
        # create item to pl item mapping
        item_to_pl_item_mapping = dict(
            zip(
                list(Item[item_col]),
                list(Item[pl_item_col]),
            )
        )

        loc_to_pl_loc_mapping = dict(
            zip(
                list(Location[loc_col]),
                list(Location[planning_location_col]),
            )
        )

        acc_to_pl_acc_mapping = dict(
            zip(
                list(Account[account_col]),
                list(Account[planning_account_col]),
            )
        )

        channel_to_pl_channel_mapping = dict(
            zip(
                list(Channel[channel_col]),
                list(Channel[planning_channel_col]),
            )
        )

        pnl_to_pl_pnl_mapping = dict(
            zip(
                list(PnL[pnl_col]),
                list(PnL[planning_pnl_col]),
            )
        )

        region_to_pl_region_mapping = dict(
            zip(
                list(Region[region_col]),
                list(Region[planning_region_col]),
            )
        )

        demand_domain_to_pl_demand_domain_mapping = dict(
            zip(
                list(DemandDomain[demand_domain_col]),
                list(DemandDomain[planning_demand_domain_col]),
            )
        )

        pl_item_to_item_mapping = {p_item: item for item, p_item in item_to_pl_item_mapping.items()}
        pl_account_to_account_mapping = {
            p_acc: account for account, p_acc in acc_to_pl_acc_mapping.items()
        }
        pl_channel_to_channel_mapping = {
            p_channel: channel for channel, p_channel in channel_to_pl_channel_mapping.items()
        }
        pl_location_to_location_mapping = {
            p_loc: loc for loc, p_loc in loc_to_pl_loc_mapping.items()
        }
        pl_pnl_to_pnl_mapping = {p_pnl: pnl for pnl, p_pnl in pnl_to_pl_pnl_mapping.items()}
        pl_region_to_region_mapping = {
            p_reg: reg for reg, p_reg in region_to_pl_region_mapping.items()
        }
        pl_demand_domain_to_demand_domain_mapping = {
            p_dd: dd for dd, p_dd in demand_domain_to_pl_demand_domain_mapping.items()
        }

        generate_match_assortment[item_col] = generate_match_assortment[pl_item_col].map(
            pl_item_to_item_mapping
        )
        generate_match_assortment[account_col] = generate_match_assortment[
            planning_account_col
        ].map(pl_account_to_account_mapping)
        generate_match_assortment[channel_col] = generate_match_assortment[
            planning_channel_col
        ].map(pl_channel_to_channel_mapping)
        generate_match_assortment[loc_col] = generate_match_assortment[planning_location_col].map(
            pl_location_to_location_mapping
        )
        generate_match_assortment[region_col] = generate_match_assortment[planning_region_col].map(
            pl_region_to_region_mapping
        )
        generate_match_assortment[demand_domain_col] = generate_match_assortment[
            planning_demand_domain_col
        ].map(pl_demand_domain_to_demand_domain_mapping)
        generate_match_assortment[pnl_col] = generate_match_assortment[planning_pnl_col].map(
            pl_pnl_to_pnl_mapping
        )
        generate_match_assortment.drop(
            [
                version_col,
                pl_item_col,
                planning_channel_col,
                planning_account_col,
                planning_location_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
            ],
            axis=1,
            inplace=True,
        )

        item_group_df = generate_match_assortment[
            [
                data_object_col,
                initiative_col,
                item_col,
                region_col,
                account_col,
                channel_col,
                pnl_col,
                demand_domain_col,
                loc_col,
            ]
        ]
        like_assort_result_list = list()
        for (
            the_data_object,
            the_initiative,
            the_item,
            the_region,
            the_account,
            the_channel,
            the_pnl,
            the_demand_domain,
            the_location,
        ) in item_group_df.itertuples(index=False):
            logger.info("-------------- the_item : {}".format(the_item))
            logger.info("-------------- the_region : {}".format(the_region))
            logger.info("-------------- the_account : {}".format(the_account))
            logger.info("-------------- the_channel : {}".format(the_channel))
            logger.info("-------------- the_pnl : {}".format(the_pnl))
            logger.info("-------------- the_location : {}".format(the_location))
            logger.info("-------------- the_demand_domain : {}".format(the_demand_domain))

            try:
                the_new_item_df = Item.loc[Item[item_col] == the_item]
                the_new_loc_df = Location.loc[Location[loc_col] == the_location]
                the_new_acc_df = Account.loc[Account[account_col] == the_account]
                the_new_channel_df = Channel.loc[Channel[channel_col] == the_channel]
                the_new_region_df = Region.loc[Region[region_col] == the_region]
                the_new_pnl_df = PnL.loc[PnL[pnl_col] == the_pnl]
                the_new_dd_df = DemandDomain.loc[
                    DemandDomain[demand_domain_col] == the_demand_domain
                ]

                new_assortment_df = generate_match_assortment[
                    (generate_match_assortment[item_col] == the_item)
                    & (generate_match_assortment[loc_col] == the_location)
                    & (generate_match_assortment[account_col] == the_account)
                    & (generate_match_assortment[channel_col] == the_channel)
                    & (generate_match_assortment[region_col] == the_region)
                    & (generate_match_assortment[pnl_col] == the_pnl)
                    & (generate_match_assortment[demand_domain_col] == the_demand_domain)
                    & (generate_match_assortment[data_object_col] == the_data_object)
                    & (generate_match_assortment[initiative_col] == the_initiative)
                ]

                new_assortment_df.drop([data_object_col, initiative_col], axis=1, inplace=True)

                new_assortment_df = new_assortment_df.merge(
                    the_new_item_df, on=item_col, how="left"
                )
                new_assortment_df = new_assortment_df.merge(the_new_loc_df, on=loc_col, how="left")
                new_assortment_df = new_assortment_df.merge(
                    the_new_acc_df, on=account_col, how="left"
                )
                new_assortment_df = new_assortment_df.merge(
                    the_new_channel_df, on=channel_col, how="left"
                )
                new_assortment_df = new_assortment_df.merge(
                    the_new_region_df, on=region_col, how="left"
                )
                new_assortment_df = new_assortment_df.merge(the_new_pnl_df, on=pnl_col, how="left")
                new_assortment_df = new_assortment_df.merge(
                    the_new_dd_df, on=demand_domain_col, how="left"
                )

                if new_assortment_df.empty:
                    logger.info("no assortment attributes found...")
                    continue

                # Filtering featuring importance values at the required level
                the_new_item_feature_level_value = the_new_item_df[item_search_space].unique()[0]
                the_new_account_feature_level_value = the_new_acc_df[account_search_space].unique()[
                    0
                ]
                the_new_channel_feature_level_value = the_new_channel_df[
                    channel_search_space
                ].unique()[0]
                the_new_region_feature_level_value = the_new_region_df[
                    region_search_space
                ].unique()[0]
                the_new_location_feature_level_value = the_new_loc_df[
                    location_search_space
                ].unique()[0]
                the_new_pnl_feature_level_value = the_new_pnl_df[pnl_search_space].unique()[0]
                the_new_dd_feature_level_value = the_new_dd_df[demand_domain_search_space].unique()[
                    0
                ]

                logger.info(
                    "---- SearchLevel values : {}, {}, {}, {}, {}, {}, {}".format(
                        the_new_item_feature_level_value,
                        the_new_account_feature_level_value,
                        the_new_channel_feature_level_value,
                        the_new_location_feature_level_value,
                        the_new_region_feature_level_value,
                        the_new_pnl_feature_level_value,
                        the_new_dd_feature_level_value,
                    )
                )

                feature_imp = FullScopeWeights[
                    (
                        FullScopeWeights[item_search_space]
                        == the_new_item_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[account_search_space]
                        == the_new_account_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[channel_search_space]
                        == the_new_channel_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[location_search_space]
                        == the_new_location_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[region_search_space]
                        == the_new_region_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[pnl_search_space]
                        == the_new_pnl_feature_level_value.lower()
                    )
                    & (
                        FullScopeWeights[demand_domain_search_space]
                        == the_new_dd_feature_level_value.lower()
                    )
                ][[feat_col, feat_imp_col]].drop_duplicates()

                if len(feature_imp) == 0:
                    logger.warning(
                        "No feature weights found for {} : {}".format(
                            item_search_space, the_new_item_feature_level_value
                        )
                    )
                    continue

                # perform required data transformation steps
                feature_imp = prepare_feat_imp_agg(feature_imp, feat_col)

                # Filter sales data at search space level
                the_cust_group_sales_df_raw = sales_df[
                    (sales_df[pnl_search_space] == the_new_pnl_feature_level_value)
                    & (sales_df[region_search_space] == the_new_region_feature_level_value)
                    & (sales_df[channel_search_space] == the_new_channel_feature_level_value)
                    & (sales_df[account_search_space] == the_new_account_feature_level_value)
                    & (sales_df[location_search_space] == the_new_location_feature_level_value)
                    & (sales_df[item_search_space] == the_new_item_feature_level_value)
                    & (sales_df[demand_domain_search_space] == the_new_dd_feature_level_value)
                ]

                if the_cust_group_sales_df_raw.empty:  # noqa: pydevd
                    logger.info(
                        "no sales for search space {}, {}, {}, {}, {}, {}, {} ...".format(
                            the_new_item_feature_level_value,
                            the_new_account_feature_level_value,
                            the_new_channel_feature_level_value,
                            the_new_location_feature_level_value,
                            the_new_region_feature_level_value,
                            the_new_pnl_feature_level_value,
                            the_new_dd_feature_level_value,
                        )
                    )
                    continue

                the_cust_group_sales_df = the_cust_group_sales_df_raw.groupby(
                    [
                        item_col,
                        account_col,
                        channel_col,
                        region_col,
                        loc_col,
                        pnl_col,
                        demand_domain_col,
                    ],
                    as_index=False,
                )[sales_col].sum()

                # Join with attribute masters to get required attributes - later category encoded
                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Item,
                    how="left",
                    on=item_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Account,
                    how="left",
                    on=account_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Channel,
                    how="left",
                    on=channel_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Location,
                    how="left",
                    on=loc_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    Region,
                    how="left",
                    on=region_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    PnL,
                    how="left",
                    on=pnl_col,
                )

                the_cust_group_sales_df = pd.merge(
                    the_cust_group_sales_df,
                    DemandDomain,
                    how="left",
                    on=demand_domain_col,
                )

                cols_to_encode = [x for x in distance_cols if x not in numerical_cols_list]

                # convert assortment attribute columns to lower case string for category encoding
                the_cust_group_sales_df[cols_to_encode] = (
                    the_cust_group_sales_df[cols_to_encode]
                    .astype("object")
                    .apply(lambda x: x.str.lower())
                )

                the_cust_group_sales_df[numerical_cols_list] = the_cust_group_sales_df[
                    numerical_cols_list
                ].astype("float")

                # convert new assortment attribute columns to lower case string for category encoding
                new_assortment_df[cols_to_encode] = (
                    new_assortment_df[cols_to_encode]
                    .astype("object")
                    .apply(lambda x: x.str.lower())
                )

                new_assortment_df[numerical_cols_list] = new_assortment_df[
                    numerical_cols_list
                ].astype("float")

                encoder = ce.TargetEncoder(cols=cols_to_encode, drop_invariant=False)
                logger.info("encoding categorical columns ...")
                # category encode item attributes for all items - new item won't be present here since it has no sale
                encoded_df = encoder.fit_transform(
                    the_cust_group_sales_df.drop(sales_col, axis=1),
                    the_cust_group_sales_df[sales_col],
                )

                logger.info("encoding new assortment categorical columns ...")
                # category encode the relevant planning item using weights from all items
                new_assortment_df_encoded = encoder.transform(new_assortment_df)

                std_scaler = StandardScaler()

                logger.info("scaling distance columns ...")

                # Scale the category encoded numbers - for all existing assortments
                encoded_df[distance_cols] = std_scaler.fit_transform(encoded_df[distance_cols])

                # Scale the category encoded numbers - for the new assortment
                new_assortment_df_encoded[distance_cols] = std_scaler.transform(
                    new_assortment_df_encoded[distance_cols]
                )

                # for distance_cols not present in feature_imp, provide them weight 0
                non_feat_cols = [
                    col for col in distance_cols if col not in list(feature_imp.columns)
                ]

                if len(non_feat_cols) != 0:
                    for col in non_feat_cols:
                        feature_imp[col] = 0

                # Multiply category encoded numbers with weights - for all existing assortments
                feature_importances = feature_imp.iloc[0]
                encoded_df_feat_mul = encoded_df[distance_cols].multiply(
                    feature_importances, axis=1
                )

                # Add key column to resulting matrix
                encoded_df = pd.concat([encoded_df[key_cols], encoded_df_feat_mul], axis=1)

                # Multiple category encoded numbers with feature weights - for the new assortment
                npi_feature_mul = new_assortment_df_encoded[distance_cols].multiply(
                    feature_importances, axis=1
                )

                # Add key column to resulting matrix
                npi_df_encoded = pd.concat(
                    [new_assortment_df_encoded[key_cols], npi_feature_mul],
                    axis=1,
                )

                logger.info("Calculating euclidean distance ...")

                # Calculating euclidean distance between all other items and the relevant item
                dist_matrix = scipy.spatial.distance.cdist(
                    encoded_df.loc[:, distance_cols],
                    npi_df_encoded.loc[:, distance_cols],
                    metric="euclidean",
                )
                dist_matrix = pd.DataFrame(dist_matrix)

                # Assign column name - will be the relevant item
                npi_df_encoded[new_assortment_col] = range(1, len(npi_df_encoded) + 1)
                npi_df_encoded = npi_df_encoded.rename(
                    columns={
                        item_col: new_product_col,
                        region_col: new_region_col,
                        loc_col: new_location_col,
                        account_col: new_account_col,
                        channel_col: new_channel_col,
                        pnl_col: new_pnl_col,
                        demand_domain_col: new_dd_col,
                    }
                )
                dist_matrix.columns = npi_df_encoded[new_assortment_col]

                # Assign index - will be all other assortments to which we are comparing
                encoded_df[like_assortment_col] = range(1, len(encoded_df) + 1)
                encoded_df = encoded_df.rename(
                    columns={
                        item_col: like_product_col,
                        loc_col: like_location_col,
                        account_col: like_account_col,
                        channel_col: like_channel_col,
                        region_col: like_region_col,
                        pnl_col: like_pnl_col,
                        demand_domain_col: like_dd_col,
                    }
                )
                dist_matrix.index = encoded_df[like_assortment_col]

                # Create graph dataframe with tabular form containing distances from new assortment to each assortment
                like_assortment_graph_df = dist_matrix.unstack().reset_index(
                    name=like_assortment_distance_col
                )

                # Rounding
                like_assortment_graph_df[like_assortment_distance_col] = like_assortment_graph_df[
                    like_assortment_distance_col
                ].round(6)

                # Get like planning assortment details
                like_assortment_graph_df = like_assortment_graph_df.merge(
                    encoded_df, on=[like_assortment_col], how="left"
                )

                like_mapping_cols = [
                    like_product_col,
                    like_account_col,
                    like_channel_col,
                    like_region_col,
                    like_dd_col,
                    like_pnl_col,
                    like_location_col,
                ]

                float_columns = [
                    col
                    for col in like_mapping_cols
                    if col in like_assortment_graph_df.columns
                    and like_assortment_graph_df[col].dtype == "float"
                ]
                like_assortment_graph_df[float_columns] = (
                    like_assortment_graph_df[float_columns].astype(int).astype(str)
                )

                column_mapping = {
                    like_product_col: (
                        like_product_pl_item_col,
                        item_to_pl_item_mapping,
                    ),
                    like_account_col: (
                        like_product_pl_account_col,
                        acc_to_pl_acc_mapping,
                    ),
                    like_channel_col: (
                        like_product_pl_channel_col,
                        channel_to_pl_channel_mapping,
                    ),
                    like_location_col: (
                        like_product_pl_loc_col,
                        loc_to_pl_loc_mapping,
                    ),
                    like_region_col: (
                        like_product_pl_region_col,
                        region_to_pl_region_mapping,
                    ),
                    like_pnl_col: (
                        like_product_pl_pnl_col,
                        pnl_to_pl_pnl_mapping,
                    ),
                    like_dd_col: (
                        like_product_pl_dd_col,
                        demand_domain_to_pl_demand_domain_mapping,
                    ),
                }

                like_assortment_graph_df = like_assortment_graph_df[
                    [
                        like_product_col,
                        like_account_col,
                        like_channel_col,
                        like_location_col,
                        like_region_col,
                        like_pnl_col,
                        like_dd_col,
                        new_assortment_col,
                        like_assortment_col,
                        like_assortment_distance_col,
                    ]
                ].drop_duplicates()

                for old_col, (new_col, mapping) in column_mapping.items():
                    like_assortment_graph_df[new_col] = like_assortment_graph_df[old_col].map(
                        mapping
                    )

                # Get new planning assortment details
                like_assortment_graph_df = like_assortment_graph_df.merge(
                    npi_df_encoded, on=[new_assortment_col], how="left"
                )

                new_mapping_cols = [
                    new_product_col,
                    new_account_col,
                    new_channel_col,
                    new_region_col,
                    new_dd_col,
                    new_pnl_col,
                    new_location_col,
                ]

                float_columns = [
                    col
                    for col in new_mapping_cols
                    if col in like_assortment_graph_df.columns
                    and like_assortment_graph_df[col].dtype == "float"
                ]
                like_assortment_graph_df[float_columns] = (
                    like_assortment_graph_df[float_columns].astype(int).astype(str)
                )

                column_mapping = {
                    new_product_col: (
                        new_product_pl_item_col,
                        item_to_pl_item_mapping,
                    ),
                    new_account_col: (
                        new_product_pl_account_col,
                        acc_to_pl_acc_mapping,
                    ),
                    new_channel_col: (
                        new_product_pl_channel_col,
                        channel_to_pl_channel_mapping,
                    ),
                    new_location_col: (
                        new_product_pl_location_col,
                        loc_to_pl_loc_mapping,
                    ),
                    new_region_col: (
                        new_product_pl_region_col,
                        region_to_pl_region_mapping,
                    ),
                    new_pnl_col: (
                        new_product_pl_pnl_col,
                        pnl_to_pl_pnl_mapping,
                    ),
                    new_dd_col: (
                        new_product_pl_dd_col,
                        demand_domain_to_pl_demand_domain_mapping,
                    ),
                }

                for old_col, (new_col, mapping) in column_mapping.items():
                    like_assortment_graph_df[new_col] = like_assortment_graph_df[old_col].map(
                        mapping
                    )

                cols_to_group_by = [
                    new_assortment_col,
                    new_product_pl_item_col,
                    new_product_pl_account_col,
                    new_product_pl_channel_col,
                    new_product_pl_location_col,
                    new_product_pl_region_col,
                    new_product_pl_pnl_col,
                    new_product_pl_dd_col,
                    like_product_pl_item_col,
                    like_product_pl_account_col,
                    like_product_pl_channel_col,
                    like_product_pl_loc_col,
                    like_product_pl_region_col,
                    like_product_pl_pnl_col,
                    like_product_pl_dd_col,
                ]

                # Remove exactly same assortment as suggestion
                new_cols = [
                    col
                    for col in like_assortment_graph_df.columns
                    if "New" in col and "Planning" in col
                ]
                like_cols = [
                    col
                    for col in like_assortment_graph_df.columns
                    if "Like" in col and "Planning" in col
                ]
                like_assortment_graph_df = like_assortment_graph_df[
                    ~(
                        like_assortment_graph_df[new_cols].values
                        == like_assortment_graph_df[like_cols].values
                    ).all(axis=1)
                ]

                # Identify minimum distance among the from and to planning assortment combinations
                like_assortment_graph_df = (
                    like_assortment_graph_df.groupby(cols_to_group_by)
                    .min()[[like_assortment_distance_col]]
                    .reset_index()
                )

                # Merging with the start dates of the Like planning item Item to use as tie break
                like_assortment_graph_df = pd.merge(
                    like_assortment_graph_df,
                    sales_df_start_dates,
                    how="left",
                    left_on=like_product_pl_item_col,
                    right_on=pl_item_col,
                )

                # Sort based on distance and dates
                like_assortment_graph_df = like_assortment_graph_df.sort_values(
                    by=[
                        new_assortment_col,
                        like_assortment_distance_col,
                        start_date_col,
                    ]
                )

                # Create dense ranks
                like_assortment_graph_df[like_assortment_rank_col] = (
                    like_assortment_graph_df.groupby([new_assortment_col])[
                        like_assortment_distance_col
                    ].rank("dense", ascending=True)
                )

                # Sort on ranking
                like_assortment_graph_df.sort_values(
                    by=[
                        new_assortment_col,
                        like_assortment_rank_col,
                        start_date_col,
                    ],
                    inplace=True,
                )

                # Break the tie
                like_assortment_graph_df[like_assortment_rank_col] = (
                    like_assortment_graph_df.groupby([new_assortment_col])[
                        like_assortment_distance_col
                    ].rank("dense", ascending=True)
                )
                like_assortment_graph_df.drop(start_date_col, axis=1, inplace=True)

                # Rename cols
                like_assortment_graph_df.rename(
                    columns={
                        new_product_pl_item_col: from_planning_item_col,
                        like_product_pl_item_col: to_planning_item_col,
                        new_product_pl_account_col: from_planning_account_col,
                        like_product_pl_account_col: to_planning_account_col,
                        new_product_pl_channel_col: from_planning_channel_col,
                        like_product_pl_channel_col: to_planning_channel_col,
                        new_product_pl_location_col: from_planning_location_col,
                        like_product_pl_loc_col: to_planning_location_col,
                        new_product_pl_region_col: from_planning_region_col,
                        like_product_pl_region_col: to_planning_region_col,
                        new_product_pl_pnl_col: from_planning_pnl_col,
                        like_product_pl_pnl_col: to_planning_pnl_col,
                        new_product_pl_dd_col: from_planning_demand_domain_col,
                        like_product_pl_dd_col: to_planning_demand_domain_col,
                    },
                    inplace=True,
                )

                logger.info("Converting distance to similarity ...")
                # Converting distance to Similarity score
                like_assortment_graph_df[like_assortment_similarity] = (
                    1 / (1 + like_assortment_graph_df[like_assortment_distance_col])
                ).round(6)

                logger.info("Marking system default like assortment ...")
                # Adding a flag to mark most similar item (rank=1) as system default like assortment
                like_assortment_graph_df[like_assortment_flag_col] = np.where(
                    like_assortment_graph_df[like_assortment_rank_col] == 1,
                    1,
                    0,
                )

                # Filter relevant number of match assortments
                like_assortment_graph_df = like_assortment_graph_df.loc[
                    like_assortment_graph_df[like_assortment_rank_col] <= int(match_num),
                    :,
                ]

                logger.info("Formatting output ...")
                # Output formatting
                like_assortment_graph_df.insert(loc=0, column=version_col, value=input_version)

                like_assortment_graph_df = like_assortment_graph_df.drop_duplicates(
                    subset=like_assortment_rank_col, keep="first"
                )

                # adding data object and initiative column
                like_assortment_graph_df[from_data_object_col] = the_data_object
                like_assortment_graph_df[from_initiative_col] = the_initiative

                like_assortment_graph_df = like_assortment_graph_df[column_order]

                like_assortment_graph_df.columns = cols_required_in_output

                # Append results to master list
                like_assort_result_list.append(like_assortment_graph_df)

            except Exception as e:
                logger.exception(
                    "Exception {} while processing item : {}, channel : {}, pnl : {}, region : {}, account : {}, location: {}, demand domain: {}".format(
                        e,
                        the_item,
                        the_channel,
                        the_pnl,
                        the_region,
                        the_account,
                        the_location,
                        the_demand_domain,
                    )
                )

        logger.info("------- All Iterations complete -----------")
        # convert list of dataframes to dataframe
        like_assortment_result = concat_to_dataframe(like_assort_result_list)

        if like_assortment_result.empty:
            logger.info("No like-assortments found, returing empty dataframe ...")
            like_assortment_result = pd.DataFrame(columns=cols_required_in_output)

        logger.info("------------ Printing head of output dataframe : head-----------")
        logger.info(like_assortment_result.head())
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        like_assortment_result = pd.DataFrame(columns=cols_required_in_output)

    return like_assortment_result

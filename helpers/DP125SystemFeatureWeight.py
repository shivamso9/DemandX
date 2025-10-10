"""Feature Weight plugin for Flexible NPI."""

import logging
import os
import re
from functools import reduce
from typing import Union

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")

col_mapping = {"System Feature Weight": float}

MODEL_CONFIGS = {
    "Stable": 100,  # Best for large datasets, highest stability
    "Balanced": 50,  # Good balance between speed & accuracy (default)
    "Fast": 25,  # Faster training while maintaining accuracy
    "VeryFast": 10,  # Very fast but less stable
}


def merge_two(df1, df2_info):
    """Merge two dataframes."""
    df2, col = df2_info
    return pd.merge(df1, df2, how="left", on=col)


def fill_missing_dates(
    actual: pd.DataFrame,
    forecast_level: list,
    history_measure: str,
    relevant_time_name: str,
    relevant_time_key: str,
    time_mapping: pd.DataFrame,
    fill_nulls_with_zero=True,
    filter_from_start_date=True,
    filter_to_end_date=True,
) -> pd.DataFrame:
    """Missing dates fill.

    param actual: DataFrame containing the Actuals
    param forecast_level: dimensions of the DataFrame
    param history_measure: Measure with the historical values
    param relevant_time_name: Name of Time Dimension used
    param relevant_time_key: Key of Time Dimension used
    param time_mapping: DataFrame with mapping of multiple levels of time
    param fill_nulls_with_zero: boolean to determine whether to fill nulls with zero
    param filter_from_start_date: boolean for if to filter relevant records where record date is greater than the intersection start date
    param filter_to_end_date: boolean for if to filter relevant records where record date is less than the intersection end date
    return: DataFrame with missing dates filled
    """
    assert isinstance(actual, pd.DataFrame), "Datatype error : actual"
    assert isinstance(forecast_level, list), "Datatype error : forecast_level,"
    assert isinstance(history_measure, str), "Datatype error : history_measure"
    assert isinstance(relevant_time_name, str), "Datatype error : relevant_time_name"
    assert isinstance(relevant_time_key, str), "Datatype error : relevant_time_key"

    assert isinstance(time_mapping, pd.DataFrame), "Datatype error : time_mapping"

    relevant_actuals_nas_filled = pd.DataFrame()
    try:
        for the_col in forecast_level:
            assert the_col in actual.columns, "{} not present in actual".format(the_col)

        # join actual with time mapping, store the intersection start dates and end dates
        actual_with_time_key = actual.copy().merge(
            time_mapping,
            on=relevant_time_name,
            how="left",
        )
        start_date_col = "start_date"
        intersection_start_dates_df = (
            actual_with_time_key.groupby(forecast_level, observed=True)[relevant_time_key]
            .min()
            .reset_index()
        )
        end_date_col = "end_date"
        intersection_end_dates_df = (
            actual_with_time_key.groupby(forecast_level, observed=True)[relevant_time_key]
            .max()
            .reset_index()
        )
        intersection_end_dates_df.rename(columns={relevant_time_key: end_date_col}, inplace=True)
        intersection_start_dates_df.rename(
            columns={relevant_time_key: start_date_col}, inplace=True
        )

        # create intersections dataframe
        intersections_master = actual_with_time_key[forecast_level].drop_duplicates()

        # join with time master
        relevant_actuals_nas_filled = actual_with_time_key.merge(
            intersections_master,
            how="right",
            on=forecast_level,
        )

        # populate missing entries in the date key column
        date_name_to_key_mapping = dict(
            zip(
                list(time_mapping[relevant_time_name]),
                list(time_mapping[relevant_time_key]),
            )
        )
        relevant_actuals_nas_filled[relevant_time_key] = relevant_actuals_nas_filled[
            relevant_time_name
        ].map(date_name_to_key_mapping)

        # fill NAs
        if fill_nulls_with_zero:
            relevant_actuals_nas_filled[history_measure].fillna(0, inplace=True)

        # join with intersection start dates
        relevant_actuals_nas_filled = relevant_actuals_nas_filled.merge(
            intersection_start_dates_df, on=forecast_level
        )

        # join with intersection end dates
        relevant_actuals_nas_filled = relevant_actuals_nas_filled.merge(
            intersection_end_dates_df, on=forecast_level
        )
        relevant_actuals_nas_filled[relevant_time_key] = pd.to_datetime(
            relevant_actuals_nas_filled[relevant_time_key]
        )
        if filter_from_start_date:
            # filter relevant records where record date is greater than the intersection start date
            filter_clause = (
                relevant_actuals_nas_filled[relevant_time_key]
                >= relevant_actuals_nas_filled[start_date_col]
            )
            relevant_actuals_nas_filled = relevant_actuals_nas_filled[filter_clause]

        if filter_to_end_date:
            # filter relevant records where record date is less than the intersection end date
            filter_clause = (
                relevant_actuals_nas_filled[relevant_time_key]
                <= relevant_actuals_nas_filled[end_date_col]
            )
            relevant_actuals_nas_filled = relevant_actuals_nas_filled[filter_clause]

        # drop the start date col, end date col, sort by intersections and time key
        relevant_actuals_nas_filled.drop(start_date_col, axis=1, inplace=True)
        relevant_actuals_nas_filled.drop(end_date_col, axis=1, inplace=True)
        relevant_actuals_nas_filled.sort_values(forecast_level + [relevant_time_key], inplace=True)
        relevant_actuals_nas_filled.reset_index(drop=True, inplace=True)

        # saw cases where version column is present in dataframe and not populated in output
        version_col = "Version.[Version Name]"
        if version_col in relevant_actuals_nas_filled.columns:
            # collect first non null index
            first_non_null_idx = relevant_actuals_nas_filled[version_col].first_valid_index()

            # collect fill value
            version_fill_value = relevant_actuals_nas_filled[version_col].loc[first_non_null_idx]

            # fill in place
            relevant_actuals_nas_filled[version_col].fillna(version_fill_value, inplace=True)

    except Exception as e:
        logger.exception("Exception {} from fill_missing_dates ...".format(e))

    return relevant_actuals_nas_filled


def calculate_weight(
    df: pd.DataFrame,
    item_search_space,
    location_search_space,
    account_search_space,
    region_search_space,
    channel_search_space,
    demand_domain_search_space,
    pnl_search_space,
    dist_cols,
    time_col,
    like_item_actual_col,
    sys_feature_weight_col,
    item_feature_col,
    round_decimals,
    total_weight_col,
    exclude_level_list,
    is_feature_level=False,
    remove_single_value_col=True,
    n_estimators=50,
):
    """Calculate Weights."""
    agg_cols = []

    # create a column prefix string
    agg_col_prefix = "mean_" + like_item_actual_col + "_"

    # check if the values are varying at search space level or not

    for the_col in dist_cols:
        logger.info("Creating aggregated feature with : {}".format(the_col))
        try:
            # create column name
            the_agg_col = agg_col_prefix + the_col

            # fields to groupby
            if not is_feature_level:
                search_space = (
                    item_search_space
                    + ","
                    + location_search_space
                    + ","
                    + pnl_search_space
                    + ","
                    + region_search_space
                    + ","
                    + account_search_space
                    + ","
                    + demand_domain_search_space
                    + ","
                    + channel_search_space
                )
                search_space = search_space.split(",")
            else:
                search_space = item_search_space
            fields_to_groupby = [time_col, the_col] + (
                search_space if isinstance(search_space, list) else [search_space]
            )

            # groupby and mean
            df[the_agg_col] = df.groupby(fields_to_groupby)[like_item_actual_col].transform(np.mean)

            # add to master list
            agg_cols.append(the_agg_col)
        except Exception as e:
            logger.exception(e)

    # Deleting rows with NA in any columns
    # sales_with_item_attributes.dropna(inplace=True)

    feature_names = np.array(agg_cols)
    logger.info("feature_names for {} : {}".format(search_space, feature_names))
    all_feat_imp_list = []

    for the_search_level, the_data in df.groupby(search_space):
        logger.info(f"--- {search_space} : {the_search_level}")
        try:
            # drop non-varying attributes
            all_cols = dist_cols
            if remove_single_value_col:
                single_value_cols = df.columns[the_data.nunique() == 1]
            else:
                single_value_cols = []
            if len(single_value_cols) != 0:
                logger.info(
                    " dropping columns : {} due to not varying within search space : {}".format(
                        single_value_cols, search_space
                    )
                )
            all_cols = [col for col in all_cols if col not in single_value_cols]
            if is_feature_level and len(all_cols) != 0:
                all_cols = list(set(all_cols + [search_space]))
            new_agg_cols = [agg_col_prefix + col for col in all_cols]
            agg_cols = new_agg_cols

            if len(agg_cols) == 0:
                logger.info(
                    "No Varying attributes found, skipping search space {}".format(search_space)
                )
                continue

            # --- Creating the Training matrix with all feature columns ---
            X = the_data.loc[:, agg_cols]

            # --- Creating response vector
            y = the_data[like_item_actual_col]

            # create regressor object
            regressor = RandomForestRegressor(
                n_estimators=n_estimators, random_state=42, bootstrap=True, n_jobs=-1
            )

            # fit the regressor
            regressor.fit(X, y)

            # create feature importance
            feature_imp_df = pd.DataFrame(
                sorted(
                    zip(
                        map(
                            lambda x: round(x, 4),
                            regressor.feature_importances_,
                        ),
                        agg_cols,
                    ),
                    reverse=True,
                )
            )

            feature_imp_df.columns = [
                sys_feature_weight_col,
                item_feature_col,
            ]
            feature_imp_df[search_space] = the_search_level
            all_feat_imp_list.append(feature_imp_df)

        except Exception as e:
            logger.exception("Exception {} while fitting model for {}".format(e, the_search_level))

    # concat results to a single dataframe
    out_feature_imp_df = concat_to_dataframe(all_feat_imp_list)

    logger.info("Output Preparation and formatting ...")

    # replace prefix
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        agg_col_prefix, ""
    )

    # string manipulations - remove Item.[ and ] from column
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        "Item.[", "", regex=False
    )
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        "]", "", regex=False
    )

    # drop search space from feature weight calculation
    out_feature_imp_df = out_feature_imp_df[
        ~out_feature_imp_df[item_feature_col].isin(exclude_level_list)
    ]

    # calculate total across a featurelevel
    out_feature_imp_df[total_weight_col] = out_feature_imp_df.groupby(search_space)[
        sys_feature_weight_col
    ].transform(sum)

    # Divide by total and get percentages
    filter_clause = out_feature_imp_df[total_weight_col] != 0
    out_feature_imp_df[sys_feature_weight_col] = np.where(
        filter_clause,
        out_feature_imp_df[sys_feature_weight_col] / out_feature_imp_df[total_weight_col],
        0,
    )
    out_feature_imp_df[sys_feature_weight_col] = out_feature_imp_df[sys_feature_weight_col].round(
        round_decimals
    )
    return out_feature_imp_df


def calculate_weight_for_one(
    df,
    the_search_level,
    the_data,
    search_space,
    dist_cols,
    remove_single_value_col,
    is_feature_level,
    agg_col_prefix,
    like_item_actual_col,
    sys_feature_weight_col,
    item_feature_col,
    n_estimators,
):
    """Calculate weight for one intersection."""
    logger.info(f"--- {search_space} : {the_search_level}")
    try:
        # drop non-varying attributes
        all_cols = dist_cols
        if remove_single_value_col:
            single_value_cols = df.columns[the_data.nunique() == 1]
        else:
            single_value_cols = []
        if len(single_value_cols) != 0:
            logger.info(
                " dropping columns : {} due to not varying within search space : {}".format(
                    single_value_cols, search_space
                )
            )
        all_cols = [col for col in all_cols if col not in single_value_cols]
        if is_feature_level and len(all_cols) != 0:
            all_cols = list(set(all_cols + [search_space]))
        new_agg_cols = [agg_col_prefix + col for col in all_cols]
        agg_cols = new_agg_cols

        if len(agg_cols) == 0:
            logger.info(
                "No Varying attributes found, skipping search space {}".format(search_space)
            )

            return None

        # --- Creating the Training matrix with all feature columns ---
        X = the_data.loc[:, agg_cols].to_numpy(dtype=np.float32)

        # --- Creating response vector
        y = the_data[like_item_actual_col].to_numpy(dtype=np.float32)

        # create regressor object
        regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, bootstrap=True, n_jobs=1
        )

        # fit the regressor
        regressor.fit(X, y)

        # create feature importance
        feature_imp_df = pd.DataFrame(
            sorted(
                zip(
                    map(
                        lambda x: round(x, 4),
                        regressor.feature_importances_,
                    ),
                    agg_cols,
                ),
                reverse=True,
            )
        )

        feature_imp_df.columns = [
            sys_feature_weight_col,
            item_feature_col,
        ]
        feature_imp_df[search_space] = the_search_level

        return feature_imp_df

    except Exception as e:
        logger.exception("Exception {} while fitting model for {}".format(e, the_search_level))
        return None


def calculate_weight_using_joblib(
    df,
    search_space,
    dist_cols,
    remove_single_value_col,
    is_feature_level,
    agg_col_prefix,
    like_item_actual_col,
    sys_feature_weight_col,
    item_feature_col,
    n_estimators,
    MultiprocessingNumCores,
):
    """Optimized calculate weight using joblib."""
    results = Parallel(n_jobs=MultiprocessingNumCores, backend="loky")(
        delayed(calculate_weight_for_one)(
            df,
            the_search_level,
            the_data,
            search_space,
            dist_cols,
            remove_single_value_col,
            is_feature_level,
            agg_col_prefix,
            like_item_actual_col,
            sys_feature_weight_col,
            item_feature_col,
            n_estimators,
        )
        for the_search_level, the_data in df.groupby(search_space)
    )

    # Remove None values (failed or skipped groups)
    return [res for res in results if res is not None]


def calculate_weight_parallely(
    df: pd.DataFrame,
    item_search_space,
    location_search_space,
    account_search_space,
    region_search_space,
    channel_search_space,
    demand_domain_search_space,
    pnl_search_space,
    dist_cols,
    time_col,
    like_item_actual_col,
    sys_feature_weight_col,
    item_feature_col,
    round_decimals,
    total_weight_col,
    exclude_level_list,
    is_feature_level=False,
    remove_single_value_col=True,
    n_estimators=50,
    MultiprocessingNumCores=4,
):
    """Calculate weights powered by joblib using parallel threads."""
    agg_cols = []

    # create a column prefix string
    agg_col_prefix = "mean_" + like_item_actual_col + "_"

    # check if the values are varying at search space level or not

    for the_col in dist_cols:
        logger.info("Creating aggregated feature with : {}".format(the_col))
        try:
            # create column name
            the_agg_col = agg_col_prefix + the_col

            # fields to groupby
            if not is_feature_level:
                search_space = (
                    item_search_space
                    + ","
                    + location_search_space
                    + ","
                    + pnl_search_space
                    + ","
                    + region_search_space
                    + ","
                    + account_search_space
                    + ","
                    + demand_domain_search_space
                    + ","
                    + channel_search_space
                )
                search_space = search_space.split(",")
            else:
                search_space = item_search_space
            fields_to_groupby = [time_col, the_col] + (
                search_space if isinstance(search_space, list) else [search_space]
            )

            # groupby and mean
            df[the_agg_col] = df.groupby(fields_to_groupby)[like_item_actual_col].transform(np.mean)

            # add to master list
            agg_cols.append(the_agg_col)
        except Exception as e:
            logger.exception(e)

    # Deleting rows with NA in any columns
    # sales_with_item_attributes.dropna(inplace=True)

    feature_names = np.array(agg_cols)
    logger.info("feature_names for {} : {}".format(search_space, feature_names))
    all_feat_imp_list = []

    # Calculate weights
    all_feat_imp_list = calculate_weight_using_joblib(
        df,
        search_space,
        dist_cols,
        remove_single_value_col,
        is_feature_level,
        agg_col_prefix,
        like_item_actual_col,
        sys_feature_weight_col,
        item_feature_col,
        n_estimators,
        MultiprocessingNumCores,
    )

    # concat results to a single dataframe
    out_feature_imp_df = concat_to_dataframe(all_feat_imp_list)

    logger.info("Output Preparation and formatting ...")

    # replace prefix
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        agg_col_prefix, ""
    )

    # string manipulations - remove Item.[ and ] from column
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        "Item.[", "", regex=False
    )
    out_feature_imp_df[item_feature_col] = out_feature_imp_df[item_feature_col].str.replace(
        "]", "", regex=False
    )

    # drop search space from feature weight calculation
    out_feature_imp_df = out_feature_imp_df[
        ~out_feature_imp_df[item_feature_col].isin(exclude_level_list)
    ]

    # calculate total across a featurelevel
    out_feature_imp_df[total_weight_col] = out_feature_imp_df.groupby(search_space)[
        sys_feature_weight_col
    ].transform(sum)

    # Divide by total and get percentages
    filter_clause = out_feature_imp_df[total_weight_col] != 0
    out_feature_imp_df[sys_feature_weight_col] = np.where(
        filter_clause,
        out_feature_imp_df[sys_feature_weight_col] / out_feature_imp_df[total_weight_col],
        0,
    )
    out_feature_imp_df[sys_feature_weight_col] = out_feature_imp_df[sys_feature_weight_col].round(
        round_decimals
    )
    return out_feature_imp_df


def save_output_data_to_cloud(FeatureWeightsFull, df_keys, cloud_type: str = "google"):
    """Save the full feature weights output data to the cloud.

    Parameters
    ----------
    cloud_type : str, optional
        Type of cloud, by default 'google'
    """
    logger.debug("Preparing to Save data on the cloud  ...")

    if str(os.environ).lower().count("o9") > 80:
        logger.info("Preparing to save the full feature weights to the cloud...")
        os.environ["FileStoreType"] = cloud_type
        import glob

        from o9cloudutils import cloud_storage_utils, user_storage_path

        bucket = "bucket_slice {}".format(df_keys)
        local_storage_path = os.path.join(user_storage_path, bucket)
        test_folder_path = os.path.join(local_storage_path, "featureweights")
        os.makedirs(test_folder_path)
        csv_path = os.path.join(test_folder_path, "flexinpifeatureweightsfull.csv")
        FeatureWeightsFull.to_csv(csv_path, index=False)  # to save files.

        # Pushing to cloud storage
        try:
            logger.debug("********Before storage_push*******")
            logger.debug(os.listdir(local_storage_path))
            value = cloud_storage_utils.storage_push(
                bucket, local_storage_path, overwrite=True
            )  # push bucket to hdfs storage
            if value:
                logger.debug("storage_push successful")
            else:
                logger.debug("storage_push failed")
            logger.debug("********storage_push successful*******")
            logger.debug(os.listdir(local_storage_path))

            logger.info("********List of files in bucket after storage push************")
            for filename in glob.iglob(local_storage_path + "**/**/*", recursive=True):
                logger.debug(filename)  # printing all files/path present in storage.
            logger.info("Successfull saved the full feature wegiths to the cloud.")
        except Exception:
            logger.exception("The feature weights could not be saved to the cloud..")

        return
    else:
        logger.info("Plugin is running in the local. Can not write the output ...")
        return


def remove_single_value_columns(df, col1, col2, dataframe_name):
    """Remove the single value columns."""
    orignal_col = df.columns
    df = df.loc[
        :,
        (df.apply(lambda x: x.nunique(dropna=False) > 1))
        | (df.columns == col1)
        | (df.columns == col2),
    ]
    new_col = df.columns
    removed_col = set(set(orignal_col) - set(new_col))
    logger.info(f"Removed single value columns from {dataframe_name} list: {removed_col}")
    return df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Item,
    Location,
    Account,
    Channel,
    DemandDomain,
    PnL,
    Region,
    sales,
    pl_item,
    Frequency,
    SearchSpace,
    NumericalCols,
    TimeDimension,
    FeatureLevel,
    FeatureNames,
    ReadFromHive,
    df_keys,
    MultiprocessingNumCores: Union[str, int] = "None",
    Model_Config: str = "Balanced",
):
    """Entry point of the script."""
    plugin_name = "DP125SystemFeatureWeight"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    rule_col = "DM Rule.[Rule]"
    data_object_col = "Data Object.[Data Object]"
    pl_item_col = "Item.[Planning Item]"  # lowest item level
    item_delimiter = ","
    item_col = "Item.[Item]"
    account_col = "Account.[Account]"
    loc_col = "Location.[Location]"
    channel_col = "Channel.[Channel]"
    pnl_col = "PnL.[PnL]"
    demand_domain_col = "Demand Domain.[Demand Domain]"
    region_col = "Region.[Region]"
    month_col = "Time.[Month]"
    monthkey_col = "Time.[MonthKey]"
    week_col = "Time.[Week]"
    weekkey_col = "Time.[WeekKey]"
    day_col = "Time.[Day]"
    daykey_col = "Time.[DayKey]"
    time_col = "Time"
    dummy_col = "dummy"
    feature_lower = FeatureLevel + "_lower"
    category_a = "A"
    category_b = "B"
    category_c = "C"
    category_d = "D"
    category_e = "E"

    like_item_search_space = "Like Item Search Space by Level"
    like_account_search_space = "Like Account Search Space by Level"
    like_channel_search_space = "Like Channel Search Space by Level"
    like_demand_domain_search_space = "Like Demand Domain Search Space by Level"
    like_location_search_space = "Like Location Search Space by Level"
    like_pnl_search_space = "Like PnL Search Space by Level"
    like_region_search_space = "Like Region Search Space by Level"
    item_feature_col = "Item Feature.[Item Feature]"
    total_weight_col = "Total Weight"
    total_normal_col = "Total Normal Col"
    ratio_col = "Ratio"
    round_decimals = 10

    # output measures
    sys_feature_weight_col = "System Feature Weight by Level"

    ignore_cols = [
        "Item.[Item]",
        "Account.[Account]",
        "Channel.[Channel]",
        "Location.[Location]",
        "Region.[Region]",
        "PnL.[PnL]",
        "Demand Domain.[Demand Domain]",
    ]

    cols_required_in_output = [
        version_col,
        data_object_col,
        rule_col,
        item_feature_col,
        FeatureLevel,
        sys_feature_weight_col,
    ]
    out_feature_imp_agg = pd.DataFrame(columns=cols_required_in_output)

    try:
        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)
        if NumericalCols == "None":
            NumericalCols = ""
        numerical_cols = NumericalCols.strip().split(",")
        pattern = r"^$|^[^.\[\]]+\.\[[^.\[\]]+\]$"
        numerical_match = [re.match(pattern, x) for x in numerical_cols]

        if not all(numerical_match):
            logger.warning(
                "Script Param NumericalCols must be of the format Dimension.[DimensionAttribute]"
            )
            logger.warning("Returning empty Dataframe")
            return out_feature_imp_agg

        if ReadFromHive:
            like_item_actual_col = "DP020LikeItemActual"
        else:
            like_item_actual_col = "Like Item Actual"

        if str(MultiprocessingNumCores).isdigit():
            MultiprocessingNumCores = int(eval(str(MultiprocessingNumCores)))

            if MultiprocessingNumCores > 15:
                logger.warning(f"MultiprocessingNumCores is too high '{MultiprocessingNumCores}'.")
            logger.info(f"The plugin will run on {MultiprocessingNumCores} threads.")
        else:
            if MultiprocessingNumCores == "None":
                pass
            else:
                logger.warning(
                    f"Not a valid input for MultiprocessingNumCores. MultiprocessingNumCores must be an integer greater than 0 and less than or equal to the available cores. User input is '{MultiprocessingNumCores}' ..."
                )
                logger.warning(
                    "Ignored MultiprocessingNumCores. The plugin will run on a single thread ..."
                )
                MultiprocessingNumCores = "None"

        logger.info("----- FeatureLevel : {}".format(FeatureLevel))
        logger.info("----- FeatureNames : {}".format(FeatureNames))

        # check if user input is empty
        if FeatureLevel == "" or FeatureNames == "":
            logger.warning("FeatureLevel/FeatureNames is empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            return out_feature_imp_agg

        if len(SearchSpace) == 0 or SearchSpace.isna().all().any():
            logger.warning("One or more Search Space(s) is empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty df as a result for this slice ...")
            return out_feature_imp_agg

        if len(sales) == 0:
            logger.warning("Sales df is empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            return out_feature_imp_agg

        # User defined Model n_estimators
        n_estimators = MODEL_CONFIGS.get(Model_Config, "Invalid Model Config")
        if n_estimators == "Invalid Model Config":
            logger.warning(
                f"Model Config is invalid. User input: '{Model_Config}'. Accepted inputs are {list(MODEL_CONFIGS.keys())}"
            )
            logger.warning("Model_Config set to default(Balanced)")
            n_estimators = 50

        input_version = sales[version_col].iloc[0]

        Frequency = Frequency.strip().lower()

        # Excluding Search Space from numerical cols
        item_search_space = SearchSpace[like_item_search_space][0]
        location_search_space = SearchSpace[like_location_search_space][0]
        account_search_space = SearchSpace[like_account_search_space][0]
        region_search_space = SearchSpace[like_region_search_space][0]
        channel_search_space = SearchSpace[like_channel_search_space][0]
        demand_domain_search_space = SearchSpace[like_demand_domain_search_space][0]
        pnl_search_space = SearchSpace[like_pnl_search_space][0]
        exclude_level_list = (
            [item_search_space]
            + [location_search_space]
            + [account_search_space]
            + [region_search_space]
            + [channel_search_space]
            + [demand_domain_search_space]
            + [pnl_search_space]
        )

        item_search_space = "Item.[" + item_search_space + "]"
        location_search_space = "Location.[" + location_search_space + "]"
        account_search_space = "Account.[" + account_search_space + "]"
        region_search_space = "Region.[" + region_search_space + "]"
        channel_search_space = "Channel.[" + channel_search_space + "]"
        demand_domain_search_space = "Demand Domain.[" + demand_domain_search_space + "]"
        pnl_search_space = "PnL.[" + pnl_search_space + "]"
        search_space_list = [
            item_search_space,
            location_search_space,
            account_search_space,
            region_search_space,
            channel_search_space,
            demand_domain_search_space,
            pnl_search_space,
        ]

        # Removing single value column(s)
        Location = remove_single_value_columns(Location, loc_col, location_search_space, "Location")
        Account = remove_single_value_columns(Account, account_col, account_search_space, "Account")
        Channel = remove_single_value_columns(Channel, channel_col, channel_search_space, "Channel")
        DemandDomain = remove_single_value_columns(
            DemandDomain, demand_domain_col, demand_domain_search_space, "Demand Domain"
        )
        PnL = remove_single_value_columns(PnL, pnl_col, pnl_search_space, "PnL")
        Region = remove_single_value_columns(Region, region_col, region_search_space, "Region")

        if Frequency == "daily":
            relevant_time_name = day_col
            relevant_time_key = daykey_col
            base_time_mapping_cols = [day_col, daykey_col]

        elif Frequency == "weekly":
            relevant_time_name = week_col
            relevant_time_key = weekkey_col
            base_time_mapping_cols = [day_col, week_col, weekkey_col]

        elif Frequency == "monthly":
            relevant_time_name = month_col
            relevant_time_key = monthkey_col
            if day_col in sales.columns:
                base_time_mapping_cols = [day_col, month_col, monthkey_col]
            elif week_col in sales.columns:
                base_time_mapping_cols = [week_col, month_col, monthkey_col]
            else:
                raise ValueError(
                    f"Unknown value for time column in sales dataframe, expected values {day_col, week_col}"
                )

        else:
            logger.warning(
                "Unknown value {} for Frequency, assigning value weekly by default".format(
                    Frequency
                )
            )
            relevant_time_name = week_col
            relevant_time_key = weekkey_col
            base_time_mapping_cols = [day_col, week_col, weekkey_col]

        logger.info("Processing Item attributes ...")

        # joining with item attributes
        Item.replace({np.nan: "nan"}, inplace=True)
        pl_item.replace({np.nan: "nan"}, inplace=True)
        Item = Item.merge(pl_item, on=pl_item_col, how="left")
        base_time_mapping = TimeDimension[base_time_mapping_cols].drop_duplicates()
        sales = sales.merge(base_time_mapping)

        logger.info("Processing Account attributes ...")
        Account.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing Channel attributes ...")
        Channel.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing Location attributes ...")
        Location.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing Demand Domain attributes ...")
        DemandDomain.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing PnL attributes ...")
        PnL.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing Region attributes ...")
        Region.replace({np.nan: "nan"}, inplace=True)

        logger.info("Processing Sales data ...")

        # aggregating sales based on Frequency
        sales = (
            sales.groupby(
                [
                    version_col,
                    item_col,
                    account_col,
                    loc_col,
                    demand_domain_col,
                    channel_col,
                    pnl_col,
                    region_col,
                    relevant_time_name,
                ],
                observed=True,
            )
            .sum()[[like_item_actual_col]]
            .reset_index()
        )

        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # fill missing dates
        sales = fill_missing_dates(
            actual=sales,
            forecast_level=[
                version_col,
                item_col,
                account_col,
                channel_col,
                loc_col,
                demand_domain_col,
                pnl_col,
                region_col,
            ],
            time_mapping=relevant_time_mapping,
            history_measure=like_item_actual_col,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            fill_nulls_with_zero=True,
            filter_from_start_date=True,
            filter_to_end_date=True,
        )

        # version column will be added back in the end of execution
        req_cols = [
            item_col,
            account_col,
            loc_col,
            demand_domain_col,
            channel_col,
            pnl_col,
            region_col,
            relevant_time_name,
            like_item_actual_col,
        ]
        sales = sales[req_cols]

        sales.rename(columns={relevant_time_name: time_col}, inplace=True)

        if sum(sales[like_item_actual_col]) == 0:
            logger.warning("Sum of sales col is zero for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe for this slice ...")
            return out_feature_imp_agg

        join_info = [
            (Item, item_col),
            (Location, loc_col),
            (Account, account_col),
            (Channel, channel_col),
            (DemandDomain, demand_domain_col),
            (PnL, pnl_col),
            (Region, region_col),
        ]

        sales_with_attributes = reduce(merge_two, join_info, sales)

        # fill missing values with "dummy"
        object_cols = list(sales_with_attributes.select_dtypes(["object"]).columns)

        sales_with_attributes_copy = sales_with_attributes.copy(deep=True)
        sales_with_attributes_copy = sales_with_attributes_copy.replace(
            {"": "nan", "NaN": "nan", "NULL": "nan"}
        )
        sales_with_attributes_copy.fillna("nan", inplace=True)
        sales_with_attributes_copy = sales_with_attributes_copy.replace({"nan": float("nan")})

        numerical_cols = [col for col in numerical_cols if col not in search_space_list]

        if len(numerical_cols) > 1:
            numerical_cols = [item for item in numerical_cols if item != ""]

        # Final numerical columns | if namerical column present in the sales with attributes data frame
        numerical_cols = list(set(numerical_cols) & set(sales_with_attributes))

        sales_with_attributes = sales_with_attributes_copy.copy(deep=True)

        # convert from string to float
        if len(numerical_cols) >= 1 and numerical_cols[0] != "":
            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].replace(
                {"": "nan"}
            )
            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].replace(
                {"NaN": "nan", "NULL": "nan"}
            )
            columns_with_non_convertible_values = [
                col
                for col in numerical_cols
                if any(
                    not isinstance(val, (int, float))
                    and not (isinstance(val, str) and val.replace(".", "", 1).isdigit())
                    and val.lower() != "nan"
                    for val in sales_with_attributes[col]
                )
            ]
            logger.info(
                f"Attributes not considered as numerical : {columns_with_non_convertible_values}"
            )
            numerical_cols = list(set(numerical_cols) - set(columns_with_non_convertible_values))

            # removing final numerical cols from object cols
            object_cols = [col for col in object_cols if col not in numerical_cols]

            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].fillna(
                "nan",
            )
            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].replace(
                {"nan", float("nan")}
            )
            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].astype(
                float
            )
        sales_with_attributes = sales_with_attributes.replace({"": dummy_col})
        sales_with_attributes = sales_with_attributes.replace({"nan": dummy_col})
        sales_with_attributes[object_cols] = sales_with_attributes[object_cols].fillna(dummy_col)

        # check if Numerical Cols are empty
        if len(numerical_cols) >= 1 and numerical_cols[0] != "":
            logger.debug(f"Selected candidate for binning : {numerical_cols}")
            for quant_col in numerical_cols:
                qt = QuantileTransformer(n_quantiles=4, output_distribution="uniform")
                values = sales_with_attributes[[quant_col]].values
                transformed_values = qt.fit_transform(values)
                sales_with_attributes[quant_col] = transformed_values

                # assign categories A, B, C, D based on the transformed values
                bin_edges = [0.3, 0.5, 0.8, float("inf")]
                bin_indices = np.digitize(np.array(sales_with_attributes[quant_col]), bin_edges)
                bin_labels = {
                    0: category_a,
                    1: category_b,
                    2: category_c,
                    3: category_d,
                    4: category_e,
                }
                sales_with_attributes[quant_col] = bin_indices.ravel()
                sales_with_attributes[quant_col] = sales_with_attributes[quant_col].map(bin_labels)

            sales_with_attributes[numerical_cols] = sales_with_attributes[numerical_cols].astype(
                str
            )
        else:
            logger.debug("None of the candidate are selected for binning")

        # Convert to lower char
        sales_with_attributes[sales_with_attributes.select_dtypes(include=["object"]).columns] = (
            sales_with_attributes.select_dtypes(include=["object"])
            .astype("string")
            .apply(lambda x: x.str.lower())
            .astype("object")
        )

        # Defining the list of columns that we want to match - calc weights and dist
        key_cols = [item_col]

        # Split string on delimiter and obtain list of features
        dist_cols = FeatureNames.split(item_delimiter)

        logger.info("distance columns : {}".format(dist_cols))

        if len(dist_cols) == 0:
            logger.warning("dist_cols is empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty df as result for this slice ...")
            return out_feature_imp_agg

        non_item_feats = [
            col
            for col in sales_with_attributes.columns
            if not col.startswith(("Item", "Time"))
            and col != like_item_actual_col
            and col not in ignore_cols
        ]
        item_feats = dist_cols.copy()
        item_feats = [col for col in item_feats if col not in ignore_cols]
        dist_cols = item_feats + non_item_feats

        # select required attributes
        required_cols = list(set(key_cols + [time_col] + dist_cols + [like_item_actual_col]))

        # get mapping between FeatureLevel and Search Space
        feature_space_mapping = sales_with_attributes[
            [FeatureLevel, item_search_space]
        ].drop_duplicates()

        sales_with_attributes = sales_with_attributes[required_cols]
        sales_with_item_attributes = sales_with_attributes[
            [col for col in sales_with_attributes.columns if col not in non_item_feats]
        ]

        logger.info("Training a supervised ML model to get feature importance ...")

        logger.info("Output Preparation and formatting ...")

        if MultiprocessingNumCores == "None":
            out_feature_imp_df = calculate_weight(
                df=sales_with_item_attributes.copy(),
                item_search_space=FeatureLevel,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=item_feats,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                is_feature_level=True,
                remove_single_value_col=False,
                n_estimators=n_estimators,
            )

            out_feature_imp_df_search_level_full = calculate_weight(
                df=sales_with_attributes.copy(),
                item_search_space=item_search_space,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=dist_cols,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                n_estimators=n_estimators,
            )

            # Outputting all weights at item level - for UI
            out_feature_imp_df_search_level = calculate_weight(
                df=sales_with_item_attributes.copy(),
                item_search_space=item_search_space,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=item_feats,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                is_feature_level=True,
                n_estimators=n_estimators,
            )

        else:
            logger.info("Calculating weight for 1/3 parallely ...")
            out_feature_imp_df = calculate_weight_parallely(
                df=sales_with_item_attributes.copy(),
                item_search_space=FeatureLevel,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=item_feats,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                is_feature_level=True,
                remove_single_value_col=False,
                n_estimators=n_estimators,
                MultiprocessingNumCores=MultiprocessingNumCores,
            )

            logger.info("Calculating weight for 2/3 parallely ...")
            out_feature_imp_df_search_level_full = calculate_weight_parallely(
                df=sales_with_attributes.copy(),
                item_search_space=item_search_space,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=dist_cols,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                n_estimators=n_estimators,
                MultiprocessingNumCores=MultiprocessingNumCores,
            )

            logger.info("Calculating weight for 3/3 parallely ...")
            # Outputting all weights at item level - for UI
            out_feature_imp_df_search_level = calculate_weight_parallely(
                df=sales_with_item_attributes.copy(),
                item_search_space=item_search_space,
                location_search_space=location_search_space,
                region_search_space=region_search_space,
                account_search_space=account_search_space,
                pnl_search_space=pnl_search_space,
                channel_search_space=channel_search_space,
                demand_domain_search_space=demand_domain_search_space,
                dist_cols=item_feats,
                time_col=time_col,
                like_item_actual_col=like_item_actual_col,
                sys_feature_weight_col=sys_feature_weight_col,
                item_feature_col=item_feature_col,
                round_decimals=round_decimals,
                total_weight_col=total_weight_col,
                exclude_level_list=exclude_level_list,
                is_feature_level=True,
                n_estimators=n_estimators,
                MultiprocessingNumCores=MultiprocessingNumCores,
            )
            logger.info("Calculated weights 3/3...")

        final_feature_imp_df = out_feature_imp_df_search_level

        # merge Featurelevel output with search space weights
        if FeatureLevel != item_search_space:

            # Adding missing attributes compared to search space level weights
            out_feature_imp_df = out_feature_imp_df.merge(feature_space_mapping)
            mapping = out_feature_imp_df_search_level[
                [item_search_space, item_feature_col]
            ].drop_duplicates()
            missing_features = pd.merge(
                mapping,
                out_feature_imp_df[[item_search_space, item_feature_col]],
                on=[item_search_space, item_feature_col],
                how="left",
                indicator=True,
            )
            missing_features = missing_features[missing_features["_merge"] == "left_only"]
            new_rows = missing_features[[item_search_space, item_feature_col]].copy()
            new_rows[sys_feature_weight_col] = 1.1
            new_rows = pd.merge(
                new_rows,
                out_feature_imp_df[[item_search_space, FeatureLevel]].drop_duplicates(),
                on=item_search_space,
                how="left",
            )
            out_feature_imp_df = pd.concat([out_feature_imp_df, new_rows], ignore_index=True)

            out_feature_imp_df[total_normal_col] = out_feature_imp_df.groupby(
                [item_search_space, item_feature_col]
            )[sys_feature_weight_col].transform(sum)
            filter_clause = out_feature_imp_df[total_normal_col] != 0
            out_feature_imp_df[ratio_col] = np.where(
                filter_clause,
                out_feature_imp_df[sys_feature_weight_col] / out_feature_imp_df[total_normal_col],
                0,
            )
            out_feature_imp_df = out_feature_imp_df.drop(
                [sys_feature_weight_col, total_weight_col],
                axis=1,
            )
            final_feature_imp_df = out_feature_imp_df_search_level.merge(out_feature_imp_df)
            final_feature_imp_df[sys_feature_weight_col] = (
                final_feature_imp_df[sys_feature_weight_col] * final_feature_imp_df[ratio_col]
            )
            final_feature_imp_df.drop(
                [ratio_col, total_normal_col, item_search_space],
                axis=1,
                inplace=True,
            )

        final_feature_imp_df.insert(0, version_col, input_version)
        final_feature_imp_df.rename(columns={FeatureLevel: feature_lower}, inplace=True)
        upper_lower_map = pd.DataFrame(
            {
                feature_lower: Item[FeatureLevel],
                FeatureLevel: Item[FeatureLevel],
            }
        )
        upper_lower_map[feature_lower] = upper_lower_map[feature_lower].str.lower()
        upper_lower_map.drop_duplicates(inplace=True)
        final_feature_imp_df = final_feature_imp_df.merge(upper_lower_map)

        logger.debug("Processing Full Search Space o/p ..........")
        out_feature_imp_df_search_level_full.insert(0, version_col, input_version)
        mask = ~out_feature_imp_df_search_level_full[item_feature_col].str.contains("\[")
        out_feature_imp_df_search_level_full.loc[mask, item_feature_col] = (
            "Item.[" + out_feature_imp_df_search_level_full.loc[mask, item_feature_col]
        )
        out_feature_imp_df_search_level_full[item_feature_col] = (
            out_feature_imp_df_search_level_full[item_feature_col] + "]"
        )

        # adding rule col
        # select required output columns
        final_feature_imp_df[rule_col] = SearchSpace[rule_col].iloc[0]

        # adding data object col
        final_feature_imp_df[data_object_col] = SearchSpace[data_object_col].iloc[0]

        out_feature_imp_agg = final_feature_imp_df[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        out_feature_imp_agg = pd.DataFrame(columns=cols_required_in_output)

    return out_feature_imp_agg, out_feature_imp_df_search_level_full

import logging
import warnings

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
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tslearn.clustering import TimeSeriesKMeans

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")
warnings.simplefilter("ignore", ConvergenceWarning)


col_mapping = {
    "SCHM Seasonal Index Backtest": float,
    "SCHM Seasonality Level Backtest": str,
    "SCHM Cluster Label Backtest": float,
    "SCHM Validation Cluster Label Backtest": float,
    "SCHM Validation Seasonality Level Backtest": str,
    "SCHM Validation Seasonal Index Backtest": float,
}


# Function to apply smoothing with additional parameters
def smooth_actuals(group, all_weights, sum_weights):
    group["Smoothed_Actuals"] = (
        group["Actual Cleansed"]
        .rolling(window=3, center=True)
        .apply(lambda x: np.sum(all_weights * x) / sum_weights, raw=False)
    )
    group["Smoothed_Actuals"].fillna(group["Actual Cleansed"], inplace=True)
    return group


# Function to de-trend actual values
def detrend_actuals(group):
    group["Index"] = np.arange(1, len(group) + 1)
    X = group["Index"].values.reshape(-1, 1)
    y = group["Actual Cleansed"].values

    model = LinearRegression()
    model.fit(X, y)

    group["Slope"] = model.coef_[0]
    group["Detrended"] = group["Actual Cleansed"] - (group["Slope"] * group["Index"])
    return group


# Function to perform clustering
def perform_clustering(
    distance,
    forecast_level,
    item_level_to_select,
    seasonal_index_col,
    cluster_label_col,
    normalize_clustering_input=False,
    number_of_clusters=None,
    default_num_clusters=12,
    max_iter=100,
    n_init=3,
    init="random",
):
    """
    Perform Time Series K-Means clustering on seasonal data with optimizations.

    Args:
        distance (DataFrame): DataFrame containing the data with seasonal information.
        forecast_level (list): List of columns to use as the forecast level.
        item_level_to_select (str): Column to select for assigning labels to the cluster centers.
        normalize_clustering_input (bool): Whether to normalize the data before clustering.
        number_of_clusters (str or int): Number of clusters. Use 'sqrt' to calculate automatically.
        default_num_clusters (int): Default number of clusters if no sqrt option is used.
        max_iter (int): Maximum iterations for K-Means.
        n_init (int): Number of times K-Means is run with different centroids.
        init (str): Initialization method for centroids.

    Returns:
        DataFrame: DataFrame containing the clustered data and cluster centers.
    """
    # # Step 1: Filter seasonal items and remove duplicates
    # no_dups = distance[distance["Seasonal"] == 1].drop_duplicates(
    #     subset=forecast_level + ["TimePeriod", "Seasonal_Index"]
    # )

    # Step 2: Pivot data to make it usable for K-Means clustering
    pivot_df = distance.pivot(
        index=forecast_level, columns="TimePeriod", values=seasonal_index_col
    ).reset_index()

    # Remove axis labels for clean formatting
    pivot_df = pivot_df.rename_axis(None, axis=1).reset_index(drop=True)

    # Step 5: Determine number of clusters
    if number_of_clusters == "sqrt":
        num_clusters = int(np.sqrt(len(pivot_df[forecast_level].drop_duplicates())).round())
    else:
        num_clusters = default_num_clusters

    # Step 3: Select only numerical columns dynamically
    numerical_columns = pivot_df.columns.difference(forecast_level)
    X_train = pivot_df[numerical_columns]

    # Step 4: Normalize input if specified
    if not eval(normalize_clustering_input):
        X_train_norm = X_train
    else:
        X_train_norm = pd.DataFrame(normalize(X_train), columns=X_train.columns)

    # Step 6: Perform Time Series K-Means clustering
    km = TimeSeriesKMeans(
        n_clusters=num_clusters,
        metric="euclidean",
        max_iter=max_iter,
        n_init=n_init,
        random_state=123,
        init=init,
    )
    y_pred = km.fit_predict(X_train_norm)

    # Step 7: Assign cluster labels
    pivot_df[cluster_label_col] = y_pred

    # Step 8: Get cluster centers
    cluster_center = pd.DataFrame(
        km.cluster_centers_.reshape(num_clusters, -1),
        columns=["Center_" + str(i) for i in X_train_norm.columns],
    )

    cluster_center[cluster_label_col] = cluster_center.index
    # cluster_center[item_level_to_select] = detrended_actuals[item_level_to_select].iloc[0]  # Safeguard for indexing

    # Step 9: Create output DataFrame by merging with cluster centers
    output_df = pivot_df.merge(cluster_center, on=cluster_label_col, how="left")

    return output_df


def shac_model(
    df,
    min_data_points_for_clustering,
    min_ts_for_seasonality_borrowing,
    minimum_points_for_assigning_seasonality,
    normalize_clustering_input,
    number_of_clusters,
    req_data_points_for_hl_seasonality,
    distance_threshold,
    item_level_to_select,
    forecast_level,
    all_weights,
    sum_weights,
    fcst_gen_time_bucket,
    relevant_time_name,
    relevant_time_key,
    item_levels,
    clusterlevelperiods,
    higherlevelperiods,
):

    # configurables

    seasonal_index_col = "SCHM Seasonal Index"
    cluster_label_col = "SCHM Cluster Label"
    seasonal_level_col = "SCHM Seasonality Level"
    borrowed_seasonal_index_col = "Borrowed_Seasonal_Index"
    own_level_col = "Own_Level"
    higherlevel_col = "Higher_Level"

    # fcst_gen_time_bucket number mapping

    var_name_map = {
        "Month": "Month_Number",
        "Planning Month": "Planning_Month_number",
        "Week": "Week_Number",
    }

    var_name = var_name_map.get(fcst_gen_time_bucket)

    # Output dataframes at each level

    req_cols = forecast_level + [
        var_name,
        seasonal_index_col,
        cluster_label_col,
        own_level_col,
    ]
    cluster_seasonal_indices = pd.DataFrame(columns=req_cols)

    req_cols = forecast_level + [
        var_name,
        higherlevel_col,
        borrowed_seasonal_index_col,
    ]
    higherlevel_seasonal_indices = pd.DataFrame(columns=req_cols)

    # Defining Output dataframes and req columns

    req_cols_for_clusterdf = forecast_level + [
        relevant_time_name,
        seasonal_index_col,
        cluster_label_col,
        seasonal_level_col,
    ]

    Clusterdf = pd.DataFrame(columns=req_cols_for_clusterdf)

    the_intersection = df[item_level_to_select].iloc[0]

    # Checking the whether the dataframe is empty or not

    if df.empty:
        logger.warning("Input Dataframe is empty for slice {item_level_to_select}")
        logger.warning(
            "No Seasonality will be done for the slice {item_level_to_select} : {the_intersection}"
        )
        return Clusterdf

    df.reset_index(drop=True, inplace=True)

    try:

        try:

            """########################################## CLUSTERING ##########################################"""

            # creating cluster dataframe using clustering function

            cluster_df = df[df[relevant_time_name].isin(clusterlevelperiods)]

            # Group by forecast_level and apply the smoothing function with additional arguments
            smoothed_actuals = (
                cluster_df.groupby(forecast_level, group_keys=False)
                .apply(smooth_actuals, all_weights, sum_weights)
                .reset_index(drop=True)
            )

            # Replace "Actual Cleansed" with "Smoothed_Actuals"
            smoothed_actuals.drop(columns=[o9Constants.ACTUAL_CLEANSED], inplace=True)
            smoothed_actuals.rename(
                columns={"Smoothed_Actuals": o9Constants.ACTUAL_CLEANSED},
                inplace=True,
            )

            # Add Logger statement that says smoothed_actuals is created
            logger.info("--------smoothed_actuals is created------------")
            logger.info(smoothed_actuals.head(5))

            logger.info("de-trend Actual ...")

            # Group by forecast_level and apply the detrending function
            detrended_actuals = (
                smoothed_actuals.groupby(forecast_level, group_keys=False)
                .apply(detrend_actuals)
                .reset_index(drop=True)
            )

            # Set any negative de-trended values to zero
            detrended_actuals.loc[detrended_actuals["Detrended"] < 0, "Detrended"] = 0

            # Clustering Configurables
            # assign dependent variable during Clustering
            dep_var = "Detrended"

            # Calculate Seasonal Indices based off of De-trended Actual

            if fcst_gen_time_bucket == "Month":
                detrended_actuals["TimePeriod"] = detrended_actuals[relevant_time_key].dt.month
            elif fcst_gen_time_bucket == "Week":
                detrended_actuals["TimePeriod"] = (
                    detrended_actuals[relevant_time_key].dt.isocalendar().week
                )
            elif fcst_gen_time_bucket == "Planning Month":
                detrended_actuals["TimePeriod"] = (
                    detrended_actuals[relevant_time_name].str[1:3].astype(int)
                )

            # Calculate time period mean and time series mean in one step
            timemean = detrended_actuals.groupby(forecast_level + ["TimePeriod"])[dep_var].mean()
            timeseriesmean = detrended_actuals.groupby(forecast_level)[dep_var].mean()

            # Merge both calculated means with detrended_actuals using chained merge approach to save memory
            detrended_actuals = detrended_actuals.merge(
                timemean.rename("TimePeriod_Mean"),
                on=forecast_level + ["TimePeriod"],
                how="left",
            ).merge(
                timeseriesmean.rename("TimeSeries_Mean"),
                on=forecast_level,
                how="left",
            )

            # Compute Seasonal Index directly
            detrended_actuals[seasonal_index_col] = np.round(
                detrended_actuals["TimePeriod_Mean"] / detrended_actuals["TimeSeries_Mean"],
                5,
            )

            # nulls are places where the ts is all zero actuals, so can drop safely
            detrended_actuals.dropna(subset=seasonal_index_col, inplace=True)

            # Checking the whether the dataframe is empty or not

            if detrended_actuals.empty:
                logger.warning(
                    f"Dataframe is empty after de-trending for slice {item_level_to_select} : {the_intersection}"
                )
                logger.warning(
                    f"No Clustering will be done for slice {item_level_to_select} : {the_intersection}"
                )

            # only keep items with atleast 2 years of history
            num_of_datapoints = (
                smoothed_actuals.groupby(forecast_level)[relevant_time_key]
                .nunique()
                .rename("Num Data Points")
                .reset_index()
            )

            detrended_actuals = detrended_actuals.merge(
                num_of_datapoints, how="left", on=forecast_level
            )

            distance = detrended_actuals[
                detrended_actuals["Num Data Points"] >= min_data_points_for_clustering
            ]

            # Checking whether the dataframe is empty or not after filtering
            if distance.empty:
                logger.warning(
                    f"Dataframe is empty after filtering for slice {item_level_to_select} : {the_intersection}"
                )
                logger.warning(
                    f"No Clustering will be done for slice {item_level_to_select} : {the_intersection}"
                )

            # dropping item levels
            distance = distance.drop(columns=item_levels)

            if fcst_gen_time_bucket == "Month" or fcst_gen_time_bucket == "Planning Month":
                distance = distance.assign(
                    Year=np.where(
                        distance["Index"].between(1, 12, inclusive="both"),
                        "Year1",
                        np.where(
                            distance["Index"].between(13, 24, inclusive="both"),
                            "Year2",
                            "Year3",
                        ),
                    )
                )
            elif fcst_gen_time_bucket == "Week":
                distance = distance.assign(
                    Year=np.where(
                        distance["Index"].between(1, 52, inclusive="both"),
                        "Year1",
                        np.where(
                            distance["Index"].between(53, 104, inclusive="both"),
                            "Year2",
                            "Year3",
                        ),
                    )
                )

            # Drop duplicates based on forecast level, TimePeriod, and Year
            distance = distance.drop_duplicates(subset=forecast_level + ["TimePeriod", "Year"])

            # Pivot the DataFrame to align years
            distance_calc = distance.pivot(
                index=forecast_level + ["TimePeriod"],
                columns=["Year"],
                values=dep_var,
            )

            # Check which year columns are present
            years_present = [
                year for year in ["Year1", "Year2", "Year3"] if year in distance_calc.columns
            ]

            # Ensure there's at least one year present
            if not years_present:
                logger.warning(
                    "No valid year columns (Year1, Year2, or Year3) found in the dataset."
                )

            # Calculate Min and Sum across available years only
            distance_calc["Min"] = distance_calc[years_present].min(axis=1)
            distance_calc["Sum"] = distance_calc[years_present].sum(axis=1)

            # Group by forecast level and aggregate min and sum in a single pass using agg
            distance_sums = (
                distance_calc.groupby(forecast_level)
                .agg({"Min": "sum", "Sum": "sum"})
                .reset_index()
            )

            # Calculate the Distance Score
            distance_sums["Distance Score"] = len(years_present) * (
                distance_sums["Min"] / distance_sums["Sum"]
            )

            # assign anything with distance score more than 0.8 as seasonal (but less than / equal to 1)
            distance_sums["Seasonal"] = np.where(
                (distance_sums["Distance Score"] >= distance_threshold)
                & (distance_sums["Distance Score"] <= 1),
                1,
                0,
            )
            distance = distance.merge(
                distance_sums[forecast_level + ["Seasonal"]],
                how="left",
                on=forecast_level,
            )

            logger.info(
                "Number of seasonal items:",
                distance[distance["Seasonal"] == 1][o9Constants.STAT_ITEM].nunique(),
            )
            logger.info(
                "Number of items tested for seasonality:",
                distance[o9Constants.STAT_ITEM].nunique(),
            )

            distance = distance[distance["Seasonal"] == 1].drop_duplicates(
                subset=forecast_level + ["TimePeriod", seasonal_index_col]
            )

            # Checking if distance is empty

            if distance.empty:
                logger.warning(
                    f"No seasonal items found for slice {item_level_to_select} : {the_intersection}"
                )
                logger.warning(
                    f"No Clustering will be done for slice {item_level_to_select} : {the_intersection}"
                )

            # Call the function with custom or default parameters
            cluster_data = perform_clustering(
                distance=distance,
                forecast_level=forecast_level,
                item_level_to_select=item_level_to_select,
                seasonal_index_col=seasonal_index_col,
                cluster_label_col=cluster_label_col,
                normalize_clustering_input=normalize_clustering_input,  # Optional: whether to normalize
                number_of_clusters=number_of_clusters,  # Optional: use square root method to define clusters
                default_num_clusters=12,  # Optional: default number of clusters if not using sqrt
                max_iter=100,  # Optional: max iterations for K-Means
                n_init=3,  # Optional: number of K-Means initializations
                init="random",  # Optional: initialization method for K-Means
            )

            logger.info(
                "Number of clusters:",
                cluster_data[cluster_label_col].nunique(),
            )

            center_columns = cluster_data.columns[
                cluster_data.columns.str.contains("center", case=False, na=False)
            ]
            cluster_indices = cluster_data[
                forecast_level + [cluster_label_col] + center_columns.tolist()
            ]

            # melt columns to rows, get month index number
            cluster_seasonal_indices = cluster_indices.melt(
                id_vars=forecast_level + [cluster_label_col],
                var_name=var_name,
                value_name=seasonal_index_col,
            )

            cluster_seasonal_indices[var_name] = (
                cluster_seasonal_indices[var_name].str.lstrip(to_strip="Center_").astype(int)
            )

            cluster_seasonal_indices["Own_Level"] = "Cluster_Seasonal_Index"

            logger.info("Completed Clustering Seasonal Indices")

        except Exception as e:
            logger.error(f"Error inclustering: {str(e)}")

        """###################################### Higher Level Seasonality ######################################"""
        try:
            # only look back one year
            higherlevel_df = df[df[relevant_time_name].isin(higherlevelperiods)]

            # Group by forecast_level and apply the smoothing function with additional arguments
            smoothed_actuals_higherlevel = (
                higherlevel_df.groupby(forecast_level, group_keys=False)
                .apply(smooth_actuals, all_weights, sum_weights)
                .reset_index(drop=True)
            )

            # Replace "Actual Cleansed" with "Smoothed_Actuals"
            smoothed_actuals_higherlevel.drop(columns=["Actual Cleansed"], inplace=True)
            smoothed_actuals_higherlevel.rename(
                columns={"Smoothed_Actuals": "Actual Cleansed"}, inplace=True
            )

            logger.info(
                "Number of items:",
                smoothed_actuals_higherlevel[o9Constants.STAT_ITEM].nunique(),
            )

            num_of_datapoints = (
                smoothed_actuals_higherlevel.groupby(forecast_level)[relevant_time_key]
                .nunique()
                .rename("Num Data Points")
                .reset_index()
            )

            smoothed_actuals_higherlevel = smoothed_actuals_higherlevel.merge(
                num_of_datapoints, how="left", on=forecast_level
            )
            # can drop null rows (only happens if the series hadn't started)
            smoothed_actuals_higherlevel.dropna(subset="Actual Cleansed", inplace=True)

            # establish threshold for num data points used for calculating hl seasonality
            filtered_actuals_higherlevel = smoothed_actuals_higherlevel[
                smoothed_actuals_higherlevel["Num Data Points"]
                >= req_data_points_for_hl_seasonality - 1
            ]

            if filtered_actuals_higherlevel.empty:
                logger.warning(
                    f"No data points found in last one year for higher level seasonality for slice {item_level_to_select} : {the_intersection}"
                )
                logger.warning(
                    f"No higher level seasonality is calculated for slice {item_level_to_select} : {the_intersection}"
                )

            # Group by forecast_level and apply the detrending function
            detrended_actuals_higherlevel = (
                filtered_actuals_higherlevel.groupby(forecast_level, group_keys=False)
                .apply(detrend_actuals)
                .reset_index(drop=True)
            )

            # Ensure negative detrended values are set to 0
            detrended_actuals_higherlevel.loc[
                detrended_actuals_higherlevel["Detrended"] < 0, "Detrended"
            ] = 0

            # Map the correct TimePeriod based on the fcst_gen_time_bucket
            if fcst_gen_time_bucket == "Month":
                detrended_actuals_higherlevel["TimePeriod"] = detrended_actuals_higherlevel[
                    relevant_time_key
                ].dt.month
            elif fcst_gen_time_bucket == "Week":
                detrended_actuals_higherlevel["TimePeriod"] = (
                    detrended_actuals_higherlevel[relevant_time_key].dt.isocalendar().week
                )
            elif fcst_gen_time_bucket == "Planning Month":
                detrended_actuals_higherlevel["TimePeriod"] = (
                    detrended_actuals[relevant_time_name].str[1:3].astype(int)
                )

            # Group by item levels and calculate unique counts in one go
            for item_level in item_levels:
                detrended_actuals_higherlevel[f"Count_{item_level}"] = (
                    detrended_actuals_higherlevel.groupby(item_level)[
                        o9Constants.STAT_ITEM
                    ].transform("nunique")
                )

            # Seasonal Index Calculation - Optimized Loop

            for item_level in item_levels:
                # Aggregate to item level and sum dependent variable

                # TODO UDAY: Discussed with Josh regarding this and update the code whether to choose rl or rl2 (yet to be tested) for future implementations
                rl = (
                    detrended_actuals_higherlevel.groupby(
                        [relevant_time_key, relevant_time_name, item_level]
                    )[dep_var]
                    .sum()
                    .reset_index()
                )

                fl = forecast_level[:]
                fl.remove(o9Constants.STAT_ITEM)

                # rl2 = (
                #     detrended_actuals_higherlevel.groupby(fl + [relevant_time_key, item_level])[
                #         dep_var
                #     ]
                #     .sum()
                #     .reset_index()
                # )

                # Map TimePeriod
                if fcst_gen_time_bucket == "Month":
                    rl["TimePeriod"] = rl[relevant_time_key].dt.month
                elif fcst_gen_time_bucket == "Week":
                    rl["TimePeriod"] = rl[relevant_time_key].dt.isocalendar().week
                elif fcst_gen_time_bucket == "Planning Month":
                    rl["TimePeriod"] = rl[relevant_time_name].str[1:3].astype(int)

                # Calculate TimePeriod mean and merge
                timemean = (
                    rl.groupby([item_level, "TimePeriod"])[dep_var]
                    .mean()
                    .reset_index()
                    .rename(columns={dep_var: "TimePeriod_Mean"})
                )
                rl = rl.merge(timemean, how="left", on=[item_level, "TimePeriod"])

                # Calculate TimeSeries mean and merge
                timeseriesmean = (
                    rl.groupby([item_level])[dep_var]
                    .mean()
                    .reset_index()
                    .rename(columns={dep_var: "TimeSeries_Mean"})
                )
                rl = rl.merge(timeseriesmean, how="left", on=[item_level])

                # Calculate seasonal index
                rl[f"{item_level}_Seasonal_Index"] = np.round(
                    rl["TimePeriod_Mean"] / rl["TimeSeries_Mean"], 5
                )

                # Merge the results back to the main dataframe if needed
                detrended_actuals_higherlevel = detrended_actuals_higherlevel.merge(
                    rl[
                        [
                            relevant_time_key,
                            item_level,
                            f"{item_level}_Seasonal_Index",
                        ]
                    ],
                    on=[relevant_time_key, item_level],
                    how="left",
                )

            # Dynamically generate level columns based on the item levels
            level_columns = [
                f"Count_{item}" for item in item_levels[:-1]
            ]  # Skipping item_l6 as it's the default

            # Initialize the 'Level' column with the default value as item_l6
            detrended_actuals_higherlevel["Level"] = item_level_to_select

            # Loop through the levels and update based on conditions
            for i, col in enumerate(level_columns):
                mask = (detrended_actuals_higherlevel[col] >= min_ts_for_seasonality_borrowing) & (
                    detrended_actuals_higherlevel["Level"] == item_level_to_select
                )

                detrended_actuals_higherlevel.loc[mask, "Level"] = item_levels[i]

            detrended_actuals_higherlevel["Level"] = (
                detrended_actuals_higherlevel["Level"] + "_Seasonal_Index"
            )
            detrended_actuals_higherlevel.reset_index(inplace=True, drop=True)

            # Get borrowed seasonal index value
            detrended_actuals_higherlevel["Borrowed_Seasonal_Index"] = (
                detrended_actuals_higherlevel.values[
                    detrended_actuals_higherlevel.index,
                    detrended_actuals_higherlevel.columns.get_indexer(
                        detrended_actuals_higherlevel["Level"]
                    ),
                ]
            )

            # Higher Level Seasonal Index

            higherlevel_seasonal_indices = detrended_actuals_higherlevel[
                forecast_level + ["TimePeriod", "Level", "Borrowed_Seasonal_Index"]
            ]

            higherlevel_seasonal_indices = higherlevel_seasonal_indices.assign(
                Level=higherlevel_seasonal_indices["Level"].str.replace(
                    r"_Seasonal_Index$", "", regex=True
                )
            )

            logger.info("Completed higher level seasonality calculation")

            # incorporate higher level seasonalities
            higherlevel_seasonal_indices = higherlevel_seasonal_indices.rename(
                columns={"TimePeriod": var_name, "Level": "Higher_Level"},
            )

        except Exception as e:
            logger.error(f"Error in higher level seasonality calculation: {e}")
            logger.info("No higher level seasonality is calculated")

        # creating month or week or planning month column

        input_dataframe = df.drop(columns=item_levels)

        if fcst_gen_time_bucket == "Month":
            input_dataframe[var_name] = input_dataframe[relevant_time_key].dt.month
        elif fcst_gen_time_bucket == "Week":
            input_dataframe[var_name] = input_dataframe[relevant_time_key].dt.isocalendar().week
        elif fcst_gen_time_bucket == "Planning Month":
            input_dataframe[var_name] = input_dataframe[relevant_time_name].str[1:3].astype(int)

        Clusterdf = input_dataframe.merge(
            cluster_seasonal_indices,
            how="left",
            on=forecast_level + [var_name],
        ).merge(
            higherlevel_seasonal_indices,
            how="left",
            on=forecast_level + [var_name],
        )

        """####################### DETERMINE LEVEL ######################"""

        Clusterdf[seasonal_level_col] = np.where(
            Clusterdf["Own_Level"] == "Cluster_Seasonal_Index",
            Clusterdf["Own_Level"],
            Clusterdf["Higher_Level"],
        )

        # fill "Seasonal Index" with higher level seasonal index column
        Clusterdf[seasonal_index_col].fillna(Clusterdf["Borrowed_Seasonal_Index"], inplace=True)

        # dropping any instances where seasonal index is null - not eligible for this method

        Clusterdf = Clusterdf[req_cols_for_clusterdf].dropna(subset=[seasonal_index_col])
        Clusterdf.drop_duplicates(inplace=True)

    except Exception as e:
        logger.exception(e)
        return Clusterdf

    return Clusterdf


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    min_data_points_for_clustering,
    min_ts_for_seasonality_borrowing,
    minimum_points_for_assigning_seasonality,
    normalize_clustering_input,
    number_of_clusters,
    req_data_points_for_hl_seasonality,
    weights,
    distance_threshold,
    BackTestCyclePeriod,
    item_level_to_select,
    IncludeDiscIntersections,
    Actual,
    ItemMasterData,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    SegmentationOutput,
    StatBucketWeight,
    StatLevel,
    multiprocessing_num_cores,
    df_keys={},
    SellOutOffset=pd.DataFrame(),
):
    try:
        AllClusterList = list()
        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_all_cluster = decorated_func(
                Grains=Grains,
                min_data_points_for_clustering=min_data_points_for_clustering,
                min_ts_for_seasonality_borrowing=min_ts_for_seasonality_borrowing,
                minimum_points_for_assigning_seasonality=minimum_points_for_assigning_seasonality,
                normalize_clustering_input=normalize_clustering_input,
                number_of_clusters=number_of_clusters,
                req_data_points_for_hl_seasonality=req_data_points_for_hl_seasonality,
                weights=weights,
                distance_threshold=distance_threshold,
                BackTestCyclePeriod=BackTestCyclePeriod,
                item_level_to_select=item_level_to_select,
                IncludeDiscIntersections=IncludeDiscIntersections,
                Actual=Actual,
                ItemMasterData=ItemMasterData,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                SegmentationOutput=SegmentationOutput,
                StatBucketWeight=StatBucketWeight,
                StatLevel=StatLevel,
                multiprocessing_num_cores=multiprocessing_num_cores,
                df_keys=df_keys,
                SellOutOffset=SellOutOffset,
            )

            AllClusterList.append(the_all_cluster)

        AllCluster = concat_to_dataframe(AllClusterList)
    except Exception as e:
        logger.exception(e)
        AllCluster = None
    return AllCluster


def processIteration(
    Grains,
    min_data_points_for_clustering,
    min_ts_for_seasonality_borrowing,
    minimum_points_for_assigning_seasonality,
    normalize_clustering_input,
    number_of_clusters,
    req_data_points_for_hl_seasonality,
    weights,
    distance_threshold,
    BackTestCyclePeriod,
    item_level_to_select,
    IncludeDiscIntersections,
    Actual,
    ItemMasterData,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    ForecastGenTimeBucket,
    SegmentationOutput,
    StatBucketWeight,
    StatLevel,
    multiprocessing_num_cores,
    df_keys={},
    SellOutOffset=pd.DataFrame(),
):
    plugin_name = "DP073SCHMClustering"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # configurables

    history_period_col = "History Period"
    forecast_period_col = "Forecast Period"
    validation_period_col = "Validation Period"
    history_time_buckets_col = "History Time Buckets"
    best_fit_method_col = "Bestfit Method"

    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"

    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"

    partial_week_col = "Time.[Partial Week]"
    version_col = "Version.[Version Name]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    stat_bucket_weight_col = "Stat Bucket Weight"

    seasonality_level = "SCHM Seasonality Level"
    cluster_label = "SCHM Cluster Label"
    seasonal_index = "SCHM Seasonal Index"
    # validation_seasonality_level = "SCHM Validation Seasonality Level"
    # validation_cluster_label = "SCHM Validation Cluster Label"
    # validation_seasonal_index = "SCHM Validation Seasonal Index"

    seasonality_level_backtest = "SCHM Seasonality Level Backtest"
    cluster_label_backtest = "SCHM Cluster Label Backtest"
    seasonal_index_backtest = "SCHM Seasonal Index Backtest"
    validation_seasonality_level_backtest = "SCHM Validation Seasonality Level Backtest"
    validation_cluster_label_backtest = "SCHM Validation Cluster Label Backtest"
    validation_seasonal_index_backtest = "SCHM Validation Seasonal Index Backtest"

    intermittent_l1 = "Intermittent L1"
    plc_status_l1 = "PLC Status L1"

    sell_out_offset_col = "Offset Period"

    # splitting on delitmer and removing leading and trailing spaces to obtain forecast level
    all_grains = Grains.split(",")
    all_grains = [x.strip() for x in all_grains]
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    # getting weights from weights

    all_weights = weights.split(",")
    all_weights = [float(x) for x in all_weights]
    sum_weights = np.sum(all_weights)

    # Output dataframe

    All_CLUSTER_COLUMNS = [
        validation_seasonality_level_backtest,
        validation_cluster_label_backtest,
        validation_seasonal_index_backtest,
        seasonality_level_backtest,
        cluster_label_backtest,
        seasonal_index_backtest,
    ]

    AllCluster_cols = [version_col] + forecast_level + [partial_week_col] + All_CLUSTER_COLUMNS

    AllCluster = pd.DataFrame(columns=AllCluster_cols)

    all_item_levels = [
        o9Constants.ITEM_L1,
        o9Constants.ITEM_L2,
        o9Constants.ITEM_L3,
        o9Constants.ITEM_L4,
        o9Constants.ITEM_L5,
        o9Constants.ITEM_L6,
    ]

    logger.info("---------------------------------------")
    logger.info("item_level_to_select : {}".format(item_level_to_select))

    try:
        index = all_item_levels.index(item_level_to_select)
        item_levels = all_item_levels[: index + 1]
    except Exception as e:
        logger.exception(e)
        logger.info("item_level_to_select : {} not found".format(item_level_to_select))
        AllCluster = pd.DataFrame(columns=AllCluster_cols)
        return AllCluster

    logger.info("item_levels : {}".format(item_levels))

    try:

        req_cols = [version_col] + forecast_level + [partial_week_col, o9Constants.ACTUAL_CLEANSED]

        Actual = Actual[req_cols]

        Actual = Actual[Actual[o9Constants.ACTUAL_CLEANSED].notna()]

        req_cols = [
            version_col,
            history_period_col,
            forecast_period_col,
            validation_period_col,
            history_time_buckets_col,
            best_fit_method_col,
        ]

        ForecastParameters = ForecastParameters[req_cols]

        req_cols = [version_col] + forecast_level + [intermittent_l1, plc_status_l1]

        SegmentationOutput = SegmentationOutput[req_cols]

        if Actual is None or len(Actual) == 0:
            logger.warning("Actuals is None/Empty for slice : {}...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return AllCluster

        # Add sell-out offset handling
        if len(SellOutOffset) == 0:
            logger.warning(
                "Empty SellOut offset input for the forecast iteration, assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        # cap negatives in history measure
        filter_clause = Actual[o9Constants.ACTUAL_CLEANSED] < 0
        Actual[o9Constants.ACTUAL_CLEANSED] = np.where(
            filter_clause, 0, Actual[o9Constants.ACTUAL_CLEANSED]
        )

        if Actual[o9Constants.ACTUAL_CLEANSED].sum() == 0:
            logger.warning("Sum of Actuals is zero for slice : {} ...".format(df_keys))
            logger.warning("Returning empty dataframes as result ...")
            return AllCluster

        # Checking whether time dim is empty
        if len(TimeDimension) == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return AllCluster

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]

        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col
            cluster_period = 104
            higher_level_period = 52
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col
            cluster_period = 24
            higher_level_period = 12
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
            cluster_period = 24
            higher_level_period = 12
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return AllCluster

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        history_periods = int(ForecastParameters[history_period_col].iloc[0])
        forecast_periods = int(ForecastParameters[forecast_period_col].iloc[0])
        validation_periods = int(ForecastParameters[validation_period_col].iloc[0])
        validation_method = str(ForecastParameters[best_fit_method_col].iloc[0])
        frequency = ForecastParameters[history_time_buckets_col].iloc[0]
        input_version = Actual[version_col].iloc[0]

        # time mapping with actual cleansed data
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from the mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Join Actuals with time mapping
        Actual = Actual.merge(base_time_mapping, on=partial_week_col, how="inner")

        # select the relevant columns, groupby and sum history measure
        Actual = (
            Actual.groupby([version_col] + forecast_level + [relevant_time_name])
            .sum(numeric_only=True)[[o9Constants.ACTUAL_CLEANSED]]
            .reset_index()
        )

        logger.info("---------------------------------------")

        logger.info("CurrentTimePeriod head : ")
        logger.info(CurrentTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # Get the latest time period
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # Adjust latest time according to the forecast iteration's offset
        offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
        if offset_periods > 0:
            relevant_time_mapping = TimeDimension[
                [relevant_time_name, relevant_time_key]
            ].drop_duplicates()

            time_attribute_dict = {relevant_time_name: relevant_time_key}

            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info(f"latest_time_name after offset {offset_periods}: {latest_time_name}")

        # create intersections dataframe
        intersections_master = Actual[forecast_level].drop_duplicates()

        """################### For Clustering and Higher Level Seasoanality Borrowing ##########################"""

        # get last n time periods

        backtest_periods = get_n_time_periods(
            latest_time_name,
            -(BackTestCyclePeriod + 1),
            relevant_time_mapping,
            time_attribute_dict,
        )

        udpated_latest_time_name = backtest_periods[0]

        last_n_periods = get_n_time_periods(
            udpated_latest_time_name,
            -(history_periods),
            relevant_time_mapping,
            time_attribute_dict,
        )

        if len(last_n_periods) == 0:
            logger.warning(
                "No dates found after filtering for {} periods for slice {}".format(
                    history_periods, df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return AllCluster

        # get clustering n periods

        last_n_clustering_periods = get_n_time_periods(
            udpated_latest_time_name,
            -(cluster_period + validation_periods),
            relevant_time_mapping,
            time_attribute_dict,
        )

        if len(last_n_clustering_periods) == 0:
            logger.warning(
                "No dates found after filtering for {} periods for slice {}".format(
                    cluster_period, df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return AllCluster

        # filtering relevant history based on dates provided above
        relevant_actuals = Actual[Actual[relevant_time_name].isin(last_n_periods)].copy()

        if len(relevant_actuals) == 0 or relevant_actuals[o9Constants.ACTUAL_CLEANSED].sum() == 0:
            logger.warning(
                "Actuals is Empty/Sum of actuals is zero after filtering {} periods of history for slice : {}...".format(
                    history_periods, df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return AllCluster

        # get the forecast dates
        forecast_period_dates = get_n_time_periods(
            udpated_latest_time_name,
            forecast_periods + offset_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        # generate records for forecast dates so that holidays can be joined once for history + forecasts
        forecast_period_date_df = pd.DataFrame({relevant_time_name: forecast_period_dates})

        logger.info("Creating forecast period dates dataframe for all intersections ...")

        # cross join with intersections master
        forecasts_df = create_cartesian_product(
            df1=intersections_master, df2=forecast_period_date_df
        )

        # Add Segmentation part here and filterout the intersections accordingly

        Actual_with_plc = relevant_actuals.merge(
            SegmentationOutput,
            on=([version_col] + forecast_level),
            how="inner",
        )

        # if DISC intersections are to be included, take actual plc as same

        if eval(IncludeDiscIntersections):
            pass
        else:
            non_disc_intersections = Actual_with_plc[plc_status_l1] != "DISC"

            Actual_with_plc_no_disc = Actual_with_plc[non_disc_intersections]

            logger.info("Excluding Disc Intersections .... : ")

            relevant_actuals = Actual_with_plc_no_disc.drop(
                [plc_status_l1, intermittent_l1], axis=1
            )

            # After removing DISC intersections, if no data is left, return empty dataframe

            if (
                len(relevant_actuals) == 0
                or relevant_actuals[o9Constants.ACTUAL_CLEANSED].sum() == 0
            ):
                logger.warning(
                    "Actuals is Empty/Sum of actuals is zero after filtering Disc for slice : {}...".format(
                        df_keys
                    )
                )
                logger.warning("Returning empty dataframes as result ...")
                return AllCluster

        # concat forecast to actuals dataframe
        relevant_actuals = pd.concat([relevant_actuals, forecasts_df], ignore_index=True)

        # creating intersection dataframe along with ItemMasterData

        StatItemLevel = StatLevel["Item Level"][0]
        StatItemLevel = f"Item.[{StatItemLevel}]"

        ItemMasterData[o9Constants.STAT_ITEM] = ItemMasterData[StatItemLevel]

        hierarchy_cols = [
            o9Constants.PLANNING_ITEM,
            o9Constants.TRANSITION_ITEM,
            o9Constants.ITEM_L1,
            o9Constants.ITEM_L2,
            o9Constants.ITEM_L3,
            o9Constants.ITEM_L4,
            o9Constants.ITEM_L4,
            o9Constants.ITEM_L5,
            o9Constants.ITEM_L6,
        ]

        stat_idx = hierarchy_cols.index(StatItemLevel)

        for col in hierarchy_cols[:stat_idx]:
            ItemMasterData[col] = ItemMasterData[StatItemLevel]

        ItemMasterData.drop(
            columns=[o9Constants.PLANNING_ITEM, o9Constants.TRANSITION_ITEM],
            axis=1,
            inplace=True,
        )

        ItemMasterData.drop_duplicates(inplace=True)

        # creating intersection dataframe along with ItemMaster Data

        relevant_actuals = relevant_actuals.merge(
            ItemMasterData, on=o9Constants.STAT_ITEM, how="left"
        )

        # if no data is left after merging with ItemMasterData, return empty dataframe

        if len(relevant_actuals) == 0 or relevant_actuals[o9Constants.ACTUAL_CLEANSED].sum() == 0:
            logger.warning(
                "Actuals is Empty/Sum of actuals is zero after merging with ItemMasterData for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return AllCluster

        # join on time mapping and sort by key
        relevant_actuals = relevant_actuals.merge(
            relevant_time_mapping,
            on=relevant_time_name,
            how="inner",
        )

        # Higher Level Seasonality Configurables

        req_cols = (
            forecast_level
            + [
                relevant_time_name,
                relevant_time_key,
            ]
            + item_levels
            + [o9Constants.ACTUAL_CLEANSED]
        )

        relevant_actuals = relevant_actuals[req_cols]

        # sorting the dataframe
        relevant_actuals.sort_values(forecast_level + [relevant_time_key], inplace=True)

        # reset index

        relevant_actuals.reset_index(drop=True, inplace=True)

        # Getting Clustering periods and Validation periods dates

        validation_n_clustering_periods = last_n_clustering_periods[:cluster_period]

        history_n_clustering_periods = last_n_clustering_periods[validation_periods:]

        # Getting periods for higher level seasonality

        last_n_higherlevel_periods = get_n_time_periods(
            udpated_latest_time_name,
            -(higher_level_period + validation_periods),
            relevant_time_mapping,
            time_attribute_dict,
        )

        if len(last_n_clustering_periods) == 0:
            logger.warning(
                "No dates found after filtering for {} periods for slice {}".format(
                    cluster_period, df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return AllCluster

        validation_n_higherlevel_periods = last_n_higherlevel_periods[:higher_level_period]

        history_n_higherlevel_periods = last_n_higherlevel_periods[validation_periods:]

        # Creating Validation and Forward Cyle Dataframes

        if validation_method == "In Sample":
            logger.debug("Using in sample validation ...")
            # use in sample validation

            validation_n_clustering_periods = history_n_clustering_periods
            validation_n_higherlevel_periods = history_n_higherlevel_periods

        # Parallelizing the process

        """######################## Validation Run ############################################"""

        logger.info("Starting Validation Run ...")

        all_validations_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(shac_model)(
                df=df,
                min_data_points_for_clustering=min_data_points_for_clustering,
                min_ts_for_seasonality_borrowing=min_ts_for_seasonality_borrowing,
                minimum_points_for_assigning_seasonality=minimum_points_for_assigning_seasonality,
                normalize_clustering_input=normalize_clustering_input,
                number_of_clusters=number_of_clusters,
                req_data_points_for_hl_seasonality=req_data_points_for_hl_seasonality,
                distance_threshold=distance_threshold,
                item_level_to_select=item_level_to_select,
                forecast_level=forecast_level,
                all_weights=all_weights,
                sum_weights=sum_weights,
                fcst_gen_time_bucket=fcst_gen_time_bucket,
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                item_levels=item_levels,
                clusterlevelperiods=validation_n_clustering_periods,
                higherlevelperiods=validation_n_higherlevel_periods,
            )
            for name, df in relevant_actuals.groupby(item_level_to_select)
        )

        logger.info("Collected results from parallel processing ...")

        # collect separate lists from the list of tuples returned by multiprocessing function
        all_validations = [x for x in all_validations_results]

        # Concat all validation predictions to one dataframe, format output
        all_validations_df = concat_to_dataframe(all_validations)

        if len(all_validations_df) == 0:
            logger.warning(
                "No records in all_validation predictions for slice : {}".format(df_keys)
            )
            return AllCluster

        # Renaming columns as per validation output format
        all_validations_df = all_validations_df.rename(
            columns={
                seasonal_index: validation_seasonal_index_backtest,
                cluster_label: validation_cluster_label_backtest,
                seasonality_level: validation_seasonality_level_backtest,
            }
        )
        """########################## History Run ############################################"""

        logger.info("Starting Forward Cycle Run ...")

        all_history_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(shac_model)(
                df=df,
                min_data_points_for_clustering=min_data_points_for_clustering,
                min_ts_for_seasonality_borrowing=min_ts_for_seasonality_borrowing,
                minimum_points_for_assigning_seasonality=minimum_points_for_assigning_seasonality,
                normalize_clustering_input=normalize_clustering_input,
                number_of_clusters=number_of_clusters,
                req_data_points_for_hl_seasonality=req_data_points_for_hl_seasonality,
                distance_threshold=distance_threshold,
                item_level_to_select=item_level_to_select,
                forecast_level=forecast_level,
                all_weights=all_weights,
                sum_weights=sum_weights,
                fcst_gen_time_bucket=fcst_gen_time_bucket,
                relevant_time_name=relevant_time_name,
                relevant_time_key=relevant_time_key,
                item_levels=item_levels,
                clusterlevelperiods=history_n_clustering_periods,
                higherlevelperiods=history_n_higherlevel_periods,
            )
            for name, df in relevant_actuals.groupby(item_level_to_select)
        )

        logger.info("Collected results from parallel processing ...")

        # collect separate lists from the list of tuples returned by multiprocessing function

        all_history = [x for x in all_history_results]

        # Concat all validation predictions to one dataframe, format output
        all_history_df = concat_to_dataframe(all_history)

        if len(all_history_df) == 0:
            logger.warning(
                "No records in all_validation predictions for slice : {}".format(df_keys)
            )
            return AllCluster

        # Renaming columns as per validation output format
        all_history_df = all_history_df.rename(
            columns={
                seasonal_index: seasonal_index_backtest,
                cluster_label: cluster_label_backtest,
                seasonality_level: seasonality_level_backtest,
            }
        )
        """########################## Merge History and Validation Runs ###############################"""

        # Merging both all_history_df and all_validations_df to relevant_actuals

        req_cols = forecast_level + [relevant_time_name]

        relevant_actuals = relevant_actuals[req_cols]

        all_clusters_df = relevant_actuals.merge(
            all_validations_df,
            on=forecast_level + [relevant_time_name],
            how="left",
        ).merge(
            all_history_df,
            on=forecast_level + [relevant_time_name],
            how="left",
        )

        all_clusters_df.insert(0, version_col, input_version)

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )

        # perform disaggregation
        AllCluster = disaggregate_data(
            source_df=all_clusters_df,
            source_grain=relevant_time_name,
            target_grain=partial_week_col,
            profile_df=StatBucketWeight.drop(version_col, axis=1),
            profile_col=stat_bucket_weight_col,
            cols_to_disaggregate=[
                validation_seasonal_index_backtest,
                seasonal_index_backtest,
            ],
        )

        # reorder columns
        AllCluster = AllCluster[AllCluster_cols]

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        AllCluster = pd.DataFrame(columns=AllCluster_cols)
        return AllCluster

    return AllCluster

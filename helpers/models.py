"""
Version : 2025.08.00
Maintained by : dpref@o9solutions.com
"""

import logging

import numpy as np
import pandas as pd

from helpers.algo_param_extractor import AlgoParamExtractor
from helpers.o9Forecaster import o9Forecaster
from helpers.utils import get_bound_col_name, get_measure_name, get_model_desc_name

logger = logging.getLogger("o9_logger")


def fit_models(
    algo_list: list,
    train: pd.DataFrame,
    forecast_horizon: int,
    confidence_interval_alpha: float,
    seasonal_periods: int,
    validation_method: str,
    validation_periods: int,
    test_exog_df: pd.DataFrame,
    history_measure: str,
    param_extractor: AlgoParamExtractor,
    holiday_type_col: str,
    test_schm_df: pd.DataFrame,
    seasonal_index_col: str,
    use_holidays: bool,
    is_first_pass: bool,
    trend: str = "NO TREND",
    seasonality: str = "Does Not Exist",
) -> (pd.DataFrame, pd.DataFrame):

    algo_col = "Stat Algorithm.[Stat Algorithm]"
    parameter_col = "Algorithm Parameters"
    validation_method_col = "Validation Method"
    run_time_col = "Run Time"

    # Dataframe to store predictions with intervals
    all_model_pred = pd.DataFrame()

    all_model_description_dict = {
        algo_col: [],
        parameter_col: [],
        validation_method_col: [],
    }

    # Dataframe to store model descriptions
    all_model_descriptions = pd.DataFrame(index=[0])

    in_sample_flag = False
    if is_first_pass and validation_method == "In Sample":
        in_sample_flag = True

    # create object of o9Forecaster
    forecaster = o9Forecaster(
        train=train[history_measure],
        seasonal_periods=seasonal_periods,
        in_sample_flag=in_sample_flag,
        forecast_horizon=forecast_horizon,
        confidence_interval_alpha=confidence_interval_alpha,
        train_exog_df=train[[holiday_type_col]],
        test_exog_df=test_exog_df,
        param_extractor=param_extractor,
        history_measure=history_measure,
        holiday_type_col=holiday_type_col,
        train_schm_df=train[[seasonal_index_col]],
        test_schm_df=test_schm_df,
        seasonal_index_col=seasonal_index_col,
        use_holidays=use_holidays,
        trend=trend,
        seasonality=seasonality,
    )

    try:
        # Iterate through all models
        for the_model_name in algo_list:
            logger.debug("---- Fitting {} ...".format(the_model_name))

            # get measure name, define forecast horizon
            the_measure_name = get_measure_name(the_model_name)

            (
                the_forecast,
                the_forecast_intervals,
                the_fitted_params,
            ) = forecaster.call_function(name=the_model_name)

            assert isinstance(
                the_forecast, pd.Series
            ), "Datatype of the_forecast should be pandas series ..."

            # clip lower fails if there are NAs in the array, convert to array to avoid index related issues while writing to all model_pred
            if not np.isnan(the_forecast).any():
                the_forecast = the_forecast.clip(lower=0).values
            else:
                the_forecast = the_forecast.values

            all_model_pred[the_measure_name] = the_forecast

            # check and populate upper and lower bounds
            all_model_pred = populate_bounds(
                all_model_pred,
                the_measure_name,
                confidence_interval_alpha,
                the_forecast_intervals,
            )

            all_model_description_dict[algo_col].append(the_model_name)
            all_model_description_dict[parameter_col].append(the_fitted_params)
            all_model_description_dict[validation_method_col].append(validation_method)

            # create model description string
            the_model_desc_string = (
                "Algo = {} | Parameters = {} | Validation Method = {} ({})".format(
                    the_model_name,
                    the_fitted_params,
                    validation_method,
                    validation_periods,
                )
            )
            all_model_descriptions[get_model_desc_name(the_model_name)] = the_model_desc_string

    except Exception as ex:
        logger.exception(ex)
    finally:
        # we are assigning index so that descriptions can be fed, df shape becomes (forecast_horizon,0)
        # if there's a case where no models were run, we want to return df with shape (0,0) instead of (forecast_horizon,0)
        if all_model_descriptions.empty:
            all_model_descriptions = pd.DataFrame()

        all_model_descriptions = pd.DataFrame(all_model_description_dict)

        all_model_descriptions[run_time_col] = all_model_descriptions[algo_col].map(
            forecaster.tracker.runtimes
        )

        # drop column if all rows are nan
        all_model_pred.dropna(axis=1, how="all", inplace=True)

        if all_model_pred.empty:
            all_model_pred = pd.DataFrame()

    return all_model_pred, all_model_descriptions


def populate_bounds(
    all_model_pred,
    the_measure_name,
    confidence_interval_alpha,
    the_forecast_intervals,
):
    lower_bound_col = get_bound_col_name(the_measure_name, confidence_interval_alpha, "LB")
    upper_bound_col = get_bound_col_name(the_measure_name, confidence_interval_alpha, "UB")

    # if prediction intervals are available populate them to dataframe
    if not the_forecast_intervals.empty:
        # extract values from dataframe, clip lower to zero
        all_model_pred[lower_bound_col] = np.clip(
            the_forecast_intervals["lower"].values,
            a_min=0,
            a_max=None,
        )
        all_model_pred[upper_bound_col] = np.clip(
            the_forecast_intervals["upper"].values,
            a_min=0,
            a_max=None,
        )
    else:
        # if prediction bounds are not available, leave them as blank
        all_model_pred[lower_bound_col] = np.nan
        all_model_pred[upper_bound_col] = np.nan

    return all_model_pred


def train_models_for_one_intersection(
    df,
    forecast_level,
    relevant_time_name,
    history_measure,
    validation_periods,
    validation_method,
    seasonal_periods,
    forecast_period_dates,
    confidence_interval_alpha,
    assigned_algo_list_col,
    AlgoParameters,
    stat_algo_col,
    stat_parameter_col,
    system_stat_param_value_col,
    holiday_type_col,
    use_holidays,
    validation_seasonal_index_col=None,
    forward_seasonal_index_col=None,
    trend_col: str = "Trend L1",
    seasonality_col: str = "Seasonality L1",
):
    valid_pred_df, forecast_df, forecast_model_desc_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    the_intersection = tuple(df[forecast_level].iloc[0])
    try:
        logger.info("the_intersection  : {}".format(the_intersection))

        # reset index for ease of data manipulation
        df.reset_index(drop=True, inplace=True)

        if trend_col in df.columns:
            trend = df[trend_col].unique()[0]
        else:
            trend = "NO TREND"

        if seasonality_col in df.columns:
            seasonality = df[seasonality_col].unique()[0]
        else:
            seasonality = "Does Not Exist"

        # filter rows where forecast is to be populated
        test_df = df[df[history_measure].isna()]

        if holiday_type_col not in df.columns:
            logger.debug(
                f"{holiday_type_col} not present in dataframe for the intersection {the_intersection}"
            )
            return valid_pred_df, forecast_df, forecast_model_desc_df

        # select relevant columns
        available_actuals = df[
            [
                relevant_time_name,
                history_measure,
                holiday_type_col,
                validation_seasonal_index_col,
                forward_seasonal_index_col,
            ]
        ]

        # filter non null rows
        available_actuals = available_actuals[available_actuals[history_measure].notna()]

        if len(available_actuals) == 0:
            logger.debug(
                "No actuals available configured for the intersection : {}".format(the_intersection)
            )
            return valid_pred_df, forecast_df, forecast_model_desc_df

        # get forecast horizon
        forecast_horizon = len(forecast_period_dates)

        if assigned_algo_list_col not in df.columns:
            logger.debug(
                f"{assigned_algo_list_col} not present in dataframe for the intersection {the_intersection}"
            )
            return valid_pred_df, forecast_df, forecast_model_desc_df

        # get list of algos to be run
        algo_list = df[assigned_algo_list_col].unique()[0].split(",")

        # trim spaces from left/right if present
        algo_list = [x.strip() for x in algo_list]

        logger.debug("------ algo_list : {}".format(algo_list))

        if len(algo_list) == 0:
            logger.debug(
                "No algorithms configured for the intersection : {}".format(the_intersection)
            )
            return valid_pred_df, forecast_df, forecast_model_desc_df

        logger.debug("Extracting all model parameter values ...")

        # Initialize param extractor class
        param_extractor = AlgoParamExtractor(
            forecast_level=forecast_level,
            intersection=the_intersection,
            AlgoParams=AlgoParameters,
            stat_algo_col=stat_algo_col,
            stat_parameter_col=stat_parameter_col,
            system_stat_param_value_col=system_stat_param_value_col,
        )

        if validation_method == "In Sample":
            logger.debug("Using in sample validation ...")
            # use in sample validation
            train, valid = available_actuals, available_actuals
        elif validation_method == "Out Sample":
            if len(available_actuals) == 1:
                validation_method = "In Sample"
                # use in sample validation
                train, valid = available_actuals, available_actuals
            else:
                # we should have atleast one datapoint for training the model
                # reduce validation periods accordingly

                # say train size is 8 and validation periods is 11, override validation periods to 8//2 = 4 so that we have 4 datapoints for training
                if len(available_actuals) <= validation_periods:
                    validation_periods = len(available_actuals) // 2

                valid = available_actuals.tail(int(validation_periods))
                train = available_actuals[~(available_actuals.index.isin(valid.index))]

        else:
            raise ValueError(
                "Unknown validation method {}, In Sample and Out Sample are supported".format(
                    validation_method
                )
            )

        logger.debug(f"--- available_actuals.shape : {available_actuals.shape}")
        logger.debug("--- validation_method : {}".format(validation_method))
        logger.debug(f"--- train.shape : {train.shape}")
        logger.debug(f"--- valid.shape : {valid.shape}")

        logger.debug("---- PASS 1 : Fitting models ...")
        logger.debug("---- train dataset size : {}".format(len(available_actuals)))

        forecast_df, forecast_model_desc_df = fit_models(
            algo_list=algo_list,
            train=available_actuals[
                [history_measure, holiday_type_col, forward_seasonal_index_col]
            ],
            forecast_horizon=forecast_horizon,
            confidence_interval_alpha=confidence_interval_alpha,
            seasonal_periods=seasonal_periods,
            validation_method=validation_method,
            validation_periods=validation_periods,
            test_exog_df=test_df[[holiday_type_col]],
            history_measure=history_measure,
            param_extractor=param_extractor,
            holiday_type_col=holiday_type_col,
            test_schm_df=test_df[[forward_seasonal_index_col]],
            seasonal_index_col=forward_seasonal_index_col,
            use_holidays=use_holidays,
            is_first_pass=False,
            trend=trend,
            seasonality=seasonality,
        )

        if len(forecast_df.columns) > 0:
            forecast_df.insert(0, relevant_time_name, forecast_period_dates)
            # Add dimension columns
            for the_index, the_level in enumerate(forecast_level):
                forecast_df.insert(0, the_level, the_intersection[the_index])

        if len(forecast_model_desc_df.columns) > 0:
            # Add dimension columns
            for the_index, the_level in enumerate(forecast_level):
                forecast_model_desc_df.insert(0, the_level, the_intersection[the_index])

        # If a model cannot produce predictions into future, then null out the valid predictions as well so that best fit doesn't consider this algorithm
        # for the_col in valid_pred_df.columns:
        #     if the_col not in forecast_df.columns or forecast_df[the_col].isnull().all():
        #         valid_pred_df.drop(the_col, axis=1, inplace=True)

        # check and insert dimension columns
        if len(valid_pred_df.columns) > 0:
            valid_pred_df.insert(0, history_measure, valid[history_measure].values)
            valid_pred_df.insert(0, relevant_time_name, valid[relevant_time_name].values)
            # Add dimension columns
            for the_index, the_level in enumerate(forecast_level):
                valid_pred_df.insert(0, the_level, the_intersection[the_index])

            # filter required points from valid predictions
            valid_pred_df = valid_pred_df.tail(validation_periods)
        else:
            # assign a dataframe of shape (0,0) instead of (n,0) where n is validation periods
            valid_pred_df = pd.DataFrame()

        # if forecast is not generated null out the validation method, algorithm parameters and run time
        for the_idx, the_row in forecast_model_desc_df.iterrows():

            the_algo = the_row[stat_algo_col]
            the_forecast_measure = get_measure_name(the_algo)

            if (
                the_forecast_measure not in forecast_df.columns
                or forecast_df[the_forecast_measure].isnull().all()
            ):
                logger.debug(
                    f"forecast not found for {the_algo}, nulling out Algo Params, Run Time and Validation Method ..."
                )
                forecast_model_desc_df.loc[the_idx, "Algorithm Parameters"] = np.nan
                forecast_model_desc_df.loc[the_idx, "Run Time"] = np.nan
                forecast_model_desc_df.loc[the_idx, "Validation Method"] = np.nan

    except Exception as e:
        logger.exception(e)

    return valid_pred_df, forecast_df, forecast_model_desc_df

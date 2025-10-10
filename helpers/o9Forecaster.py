"""
Version : 2025.08.00
Maintained by : dpref@o9solutions.com
"""

import logging
import re
import time

import numpy as np
import pandas as pd
import statsforecast
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.function_timer import timed

# from sktime.forecasting.fbprophet import Prophet
from prophet import Prophet
from pyaf import HierarchicalForecastEngine as hautof
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from statsforecast import StatsForecast
from statsforecast.models import (
    MSTL,
    AutoARIMA,
    AutoETS,
    AutoTBATS,
    CrostonClassic,
    Holt,
    HoltWinters,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    WindowAverage,
)
from statsforecast.utils import ConformalIntervals
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from helpers.algo_param_extractor import AlgoParamExtractor
from helpers.model_params import get_fitted_params
from helpers.utils import get_ts_freq_prophet

logger = logging.getLogger("o9_logger")
logger.info(f"statsforecast version : {statsforecast.__version__}")


class RuntimeTracker:
    def __init__(self):
        self.runtimes = {}

    def track_runtime(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            self.runtimes[func.__doc__] = round(execution_time, 4)
            return result

        return wrapper

    def get_runtime(self, func_name):
        return self.runtimes.get(func_name)


class o9Forecaster:
    tracker = RuntimeTracker()

    def __init__(
        self,
        train: pd.Series,
        seasonal_periods: int,
        in_sample_flag: bool,
        forecast_horizon: int,
        confidence_interval_alpha: float,
        train_exog_df: pd.DataFrame,
        test_exog_df: pd.DataFrame,
        param_extractor: AlgoParamExtractor,
        history_measure: str,
        holiday_type_col: str,
        train_schm_df: pd.DataFrame,
        test_schm_df: pd.DataFrame,
        seasonal_index_col: str,
        use_holidays: bool = False,
        trend: str = "NO TREND",
        seasonality: str = "Does not Exist",
        UseDampedTrend: str = "True",
    ):

        assert isinstance(train, pd.Series), "Datatype error for train..."
        assert isinstance(seasonal_periods, int), "Datatype error for seasonal_periods ..."
        assert isinstance(in_sample_flag, bool), "Datatype error for in_sample_flag ..."
        assert isinstance(forecast_horizon, int), "Datatype error for forecast_horizon..."
        assert isinstance(
            confidence_interval_alpha, float
        ), "Datatype error for confidence_interval_alpha ..."

        self.train = train.astype("float64")
        self.seasonal_periods = seasonal_periods
        self.in_sample_flag = in_sample_flag
        self.forecast_horizon = forecast_horizon
        self.fcst_horizon = np.arange(1, self.forecast_horizon + 1)
        self.confidence_interval_alpha = confidence_interval_alpha
        self.train_exog_df = train_exog_df
        self.test_exog_df = test_exog_df
        self.history_measure = history_measure
        self.holiday_type_col = holiday_type_col
        self.use_holidays = False
        self.param_extractor = param_extractor
        self.use_damped_trend = eval(UseDampedTrend)
        self.train_schm_df = train_schm_df
        self.test_schm_df = test_schm_df
        self.seasonal_index_col = seasonal_index_col

        # random start date to assign for prophet/stlf
        self.random_start_date = "2016-01-01"

        # can be UPWARD/DOWNWARD when run from system stat and 'YES' when run from backtest
        self.trend = True if trend in ["UPWARD", "DOWNWARD", "YES"] else False
        self.seasonality = True if seasonality in ["Exists", "YES"] else False

    def get_func_name(self, algo: str):
        # remove hyphen, underscore, white space
        clean_algo = re.sub("-|_|\s", "", algo)

        # trim leading/trailing spaces if any
        clean_algo = clean_algo.strip()

        # convert to lower case
        clean_algo = clean_algo.lower()

        return "_".join(["get", clean_algo, "forecast"])

    def call_function(self, name: str):
        func_name = self.get_func_name(algo=name)
        logger.debug(f"func_name : {func_name}")

        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            func = getattr(self, func_name)
            try:
                return func(the_model_name=name)
            except Exception as e:
                logger.exception(e)

    def __initialize_results(self):
        the_forecast_length = len(self.train) if self.in_sample_flag else self.forecast_horizon
        the_forecast_array = [np.nan] * the_forecast_length

        # initialize the forecast series
        the_forecast = pd.Series(the_forecast_array)

        the_forecast_intervals = pd.DataFrame()
        the_fitted_params = "No Parameters"
        return the_forecast, the_forecast_intervals, the_fitted_params

    def __pad_series(self):
        """
        If data does not comprise one full cycle, pads the series with zeros
        :return:pd.Series
        """
        # if there are not enough seasonal points present, pad with zeros in points prior to start date
        num_points_to_pad = self.seasonal_periods - len(self.train)
        snaive_train = pd.Series(
            np.pad(
                self.train.values,
                (num_points_to_pad, 0),
                "constant",
            )
        )
        return snaive_train

    def get_seasonalnaiveyoy_forecast(self, the_model_name="Seasonal Naive YoY"):
        return self.get_snaive_forecast()

    @tracker.track_runtime
    def get_snaive_forecast(self):
        """Seasonal Naive YoY"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            the_estimator = SeasonalNaive(
                season_length=self.seasonal_periods,
                prediction_intervals=ConformalIntervals(n_windows=self.seasonal_periods),
            )
            the_snaive_fit = the_estimator.fit(y=np.array(self.train))
            if self.in_sample_flag:
                the_forecast_df = the_snaive_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_snaive_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])

            snaive_dict = the_snaive_fit.__dict__
            wanted_keys = ["season_length", "alias"]
            the_fitted_params_dict = {k: snaive_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_arnnet_forecast(self, the_model_name="AR-NNET"):
        """AR-NNET"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # data check
            if len(self.train) < 2 * self.seasonal_periods:
                logger.debug(
                    f"AR-NNET requires atleast {2 * self.seasonal_periods} datapoints to generate predictions ..."
                )
                return the_forecast, the_forecast_intervals, the_fitted_params

            # AR - NNET is not available in Python, implementing a Linear Regression model which uses past 6 periods
            if self.in_sample_flag:
                # in sample predictions are not yet implemented for sktime make_reduction
                logger.debug("Cannot generate in sample predictions for arnnet model ...")
            else:
                # create linear regression model
                regressor = GradientBoostingRegressor()

                # reduce the data to form
                the_estimator = make_reduction(
                    regressor,
                    window_length=self.seasonal_periods // 2,
                    strategy="recursive",
                )

                # fit model
                the_estimator.fit(self.train)

                # try to get predictions
                the_forecast = the_estimator.predict(fh=self.fcst_horizon)

            params = the_estimator.estimator.get_params()
            the_fitted_params_dict = {
                "alpha": params.get("alpha"),
                "criterion": params.get("criterion"),
                "loss": params.get("loss"),
                "max_depth": params.get("max_depth"),
                "max_features": params.get("max_features"),
                "n_estimators": params.get("n_estimators"),
                "validation_fraction": params.get("validation_fraction"),
                "window_length": getattr(the_estimator, "window_length", None),
            }

            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_movingaverage_forecast(self, the_model_name="Moving Average"):
        return self.get_moving_avg_forecast()

    @tracker.track_runtime
    def get_moving_avg_forecast(self):
        """Moving Average"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # Moving Average
            ma_periods = self.param_extractor.extract_param_value(
                algorithm="Moving Average",
                parameter="Period",
            )
            # convert float to integer
            ma_periods = int(ma_periods)

            logger.debug(f"moving avg periods : {ma_periods}")

            the_estimator = WindowAverage(
                # season_length=self.seasonal_periods,
                window_size=ma_periods
            )

            the_ma_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                # take in sample values
                the_forecast = the_ma_fit.predict_in_sample()
            else:
                the_forecast = pd.Series(the_ma_fit.predict(h=self.forecast_horizon)["mean"])

            ma_dict = the_ma_fit.__dict__
            wanted_keys = ["window_size", "alias", "prediction_intervals"]
            the_fitted_params_dict = {k: ma_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_naiverandomwalk_forecast(self, the_model_name="Naive Random Walk"):
        return self.get_naive_random_walk_forecast()

    @tracker.track_runtime
    def get_naive_random_walk_forecast(self):
        """Naive Random Walk"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # collect latest history datapoint
            latest_history_value = self.train.values[-1]

            if self.in_sample_flag:
                # populate above value as forecast into future n points
                the_forecast = pd.Series([latest_history_value] * len(self.train))
            else:
                # populate above value as forecast into future n points
                the_forecast = pd.Series([latest_history_value] * self.forecast_horizon)
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_stlf_forecast(self, the_model_name="STLF"):
        """STLF"""
        (
            mstl_forecast,
            mstl_forecast_intervals,
            mstl_fitted_params,
        ) = self.__initialize_results()

        try:
            mstl_train_index = pd.date_range(
                start=self.random_start_date,
                periods=len(self.train),
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )
            mstl_train = pd.Series(self.train.values, index=mstl_train_index)

            low_pass_jump = max(1, int(0.15 * (self.seasonal_periods + 1)))
            trend_jump = max(1, int(0.15 * 1.5 * (self.seasonal_periods + 1)))

            # set periods according to train size
            if len(mstl_train) > self.seasonal_periods:
                mstl_periods = self.seasonal_periods
            else:
                mstl_periods = 2

            the_mstl_model = MSTL(
                season_length=mstl_periods,
                alias="MSTL",
                stl_kwargs={
                    "seasonal_jump": 1 if self.seasonal_periods == 12 else 7,
                    "trend_jump": trend_jump,
                    "low_pass_jump": low_pass_jump,
                    "robust": True,
                },
            )
            the_mstl_fit = the_mstl_model.fit(y=mstl_train)

            if self.in_sample_flag:
                mstl_forecast_df = the_mstl_fit.predict_in_sample()
                mstl_forecast = pd.Series(mstl_forecast_df["fitted"])
            else:

                mstl_forecast_df = the_mstl_fit.predict(h=self.forecast_horizon, level=[80])

                mstl_forecast = pd.Series(mstl_forecast_df["mean"])
                mstl_forecast_intervals["lower"] = mstl_forecast_df["lo-80"]
                mstl_forecast_intervals["upper"] = mstl_forecast_df["hi-80"]

            mstl_dict = the_mstl_fit.__dict__
            wanted_keys = [
                "season_length",
                "trend_forecaster",
                "alias",
                "stl_kwargs",
                "prediction_intervals",
            ]
            the_fitted_params_dict = {k: mstl_dict[k] for k in wanted_keys}
            mstl_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception(e)
            logger.warning("end")
        return mstl_forecast, mstl_forecast_intervals, mstl_fitted_params

    @tracker.track_runtime
    def get_autoarima_forecast(self, the_model_name="Auto ARIMA"):
        """Auto ARIMA"""
        return self.sf_auto_arima_updated()

    @tracker.track_runtime
    def get_sarima_forecast(self, the_model_name="sARIMA"):
        """sARIMA"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        # data check
        if len(self.train) < self.seasonal_periods:
            logger.debug(
                f"sARIMA requires atleast {self.seasonal_periods} datapoints to generate predictions ..."
            )
            return the_forecast, the_forecast_intervals, the_fitted_params

        return self.sf_auto_arima_updated()

    @tracker.track_runtime
    def get_prophet_forecast(self, the_model_name="Prophet"):
        """Prophet"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        try:
            # Prophet Parameters
            # prophet_growth_cap = self.param_extractor.extract_param_value(
            #     algorithm="Prophet",
            #     parameter="Growth Cap",
            # )

            # prophet_growth_type = self.param_extractor.extract_param_value(
            #     algorithm="Prophet",
            #     parameter="Growth Type",
            # )

            # prophet_mcmc_samples = self.param_extractor.extract_param_value(
            #     algorithm="Prophet",
            #     parameter="MCMC Samples",
            # )
            prophet_growth_cap = 1.25
            prophet_growth_type = "linear"
            prophet_mcmc_samples = 100

            # generate a time series index based on frequency, assign any random start date and generate equally distanced time series
            # prophet requires equally spaced time series, planning calendar will not work
            prophet_train_index = pd.date_range(
                start=self.random_start_date,
                periods=len(self.train),
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )

            # Prophet dataframe
            prophet_train = pd.DataFrame({"ds": prophet_train_index, "y": self.train.values})

            # Adding cap and floor columns for logistic growth

            prophet_train["cap"] = max(prophet_train["y"]) * prophet_growth_cap
            prophet_train["floor"] = 0

            if prophet_growth_type.lower() == "logistic":
                the_estimator = Prophet(
                    yearly_seasonality="auto",
                    weekly_seasonality="auto",
                    daily_seasonality=False,
                    growth="logistic",
                    uncertainty_samples=100,
                    mcmc_samples=int(prophet_mcmc_samples),
                    seasonality_mode="additive",
                )
            elif prophet_growth_type.lower() == "linear":
                the_estimator = Prophet(
                    yearly_seasonality="auto",
                    weekly_seasonality="auto",
                    daily_seasonality=False,
                    growth="linear",
                    uncertainty_samples=100,
                    mcmc_samples=int(prophet_mcmc_samples),
                    seasonality_mode="additive",
                )
            else:
                raise Exception(f"{prophet_growth_type} is not a valid growth type for Prophet ...")

            the_estimator.fit(prophet_train)

            if self.in_sample_flag:
                fcst_horizon = ForecastingHorizon(values=prophet_train_index, is_relative=False)
                prophet_fcst = pd.DataFrame({"ds": fcst_horizon.to_pandas()})
            else:
                prophet_fcst = the_estimator.make_future_dataframe(
                    periods=len(self.fcst_horizon),
                    freq=get_ts_freq_prophet(self.seasonal_periods),
                    include_history=False,
                )

            # Adding cap and floor to forecast dataframe

            prophet_fcst["cap"] = max(prophet_train["y"]) * 1.25
            prophet_fcst["floor"] = 0

            # Forecast using the fitted model

            fcst_prophet = the_estimator.predict(prophet_fcst)
            the_forecast = pd.Series(
                fcst_prophet[["ds", "yhat"]].reset_index().drop(columns=["index"])["yhat"].values,
                index=fcst_prophet[["ds", "yhat"]]
                .reset_index()
                .drop(columns=["index"])["ds"]
                .values,
            )
            the_forecast_intervals = (
                fcst_prophet[["yhat_lower", "yhat_upper"]]
                .reset_index()
                .drop(columns=["index"])
                .rename(columns={"yhat_lower": "lower", "yhat_upper": "upper"})
            )

            the_fitted_params = get_fitted_params(
                the_model_name="Prophet", the_estimator=the_estimator
            )

            # Add cap and floor to fitted params only if growth is 'logistic'
            if prophet_growth_type.lower() == "logistic":
                cap_values = prophet_fcst.cap[0]
                floor_values = prophet_fcst.floor[0]
                the_fitted_params += ", growth_cap = {}, growth_floor = {}".format(
                    cap_values, floor_values
                )

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_ses_forecast(self, the_model_name="SES"):
        """SES"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            ses_alpha_upper = self.param_extractor.extract_param_value(
                algorithm="SES",
                parameter="Alpha Upper",
            )

            # can switch between SES and SES Optimized

            # the_estimator = SimpleExponentialSmoothingOptimized(
            #     prediction_intervals=ConformalIntervals(n_windows=self.seasonal_periods)
            # )
            the_estimator = SimpleExponentialSmoothing(
                alpha=ses_alpha_upper,
                prediction_intervals=ConformalIntervals(n_windows=self.seasonal_periods),
            )
            the_ses_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                the_forecast_df = the_ses_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_ses_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])
                the_forecast_intervals["lower"] = the_forecast_df["lo-80"]
                the_forecast_intervals["upper"] = the_forecast_df["hi-80"]

            ses_dict = the_ses_fit.__dict__
            wanted_keys = ["alpha", "alias"]
            the_fitted_params_dict = {k: ses_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
        except Exception as e:
            logger.exception(e)
        finally:
            pass
            # the_fitted_params = get_fitted_params(
            #     the_model_name="SES", the_estimator=the_estimator
            # )
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_schm_forecast(self, the_model_name="SCHM", actuals=None):
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        try:
            # Divide the train series by seasonal index
            input_to_algo = self.train / self.train_schm_df[self.seasonal_index_col].reset_index(
                drop=True
            )

            """SCHM"""
            des_alpha_lower = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Alpha Lower",
            )
            des_alpha_lower = self._make_lower_bound_non_zero(bound=des_alpha_lower)
            des_alpha_upper = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Alpha Upper",
            )

            des_beta_lower = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Beta Lower",
            )
            des_beta_lower = self._make_lower_bound_non_zero(bound=des_beta_lower)
            des_beta_upper = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Beta Upper",
            )

            des_phi_lower = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Phi Lower",
            )
            des_phi_lower = self._make_lower_bound_non_zero(bound=des_phi_lower)
            des_phi_upper = self.param_extractor.extract_param_value(
                algorithm="SCHM",
                parameter="Phi Upper",
            )

            bounds_dict = {
                "smoothing_level": (
                    des_alpha_lower,
                    des_alpha_upper,
                )
            }

            if len(input_to_algo) >= 2 and self.trend:
                bounds_dict["smoothing_trend"] = (
                    des_beta_lower,
                    des_beta_upper,
                )
                if self.use_damped_trend:
                    bounds_dict["damping_trend"] = (
                        des_phi_lower,
                        des_phi_upper,
                    )

            the_estimator = ETSModel(
                endog=input_to_algo,
                error="add",
                trend="add" if len(input_to_algo) >= 2 and self.trend else None,
                damped_trend=(
                    True if len(self.train) >= 2 and self.use_damped_trend and self.trend else False
                ),
                initialization_method="estimated",
                bounds=bounds_dict,
            )

            (
                the_forecast,
                the_forecast_intervals,
                the_fitted_params,
            ) = self.__get_exp_smoothing_forecast(the_estimator, the_model_name)

            # Multiply the forecast by seasonal index
            the_forecast = the_forecast * self.test_schm_df[self.seasonal_index_col].reset_index(
                drop=True
            )

            the_forecast_intervals = pd.DataFrame()

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def _make_lower_bound_non_zero(self, bound: float) -> float:
        """
        if the bound specified is zero, make it non zero
        """
        return max(bound, 0.000001)

    @tracker.track_runtime
    def get_des_forecast(self, the_model_name="DES"):
        """DES"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            the_estimator = Holt(season_length=self.seasonal_periods)
            if len(self.train) < 10:
                logger.warning("DES requires 10 datapoints, switching to SES...")
                return self.get_ses_forecast()

            the_des_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                the_forecast_df = the_des_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_des_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])
                the_forecast_intervals["lower"] = the_forecast_df["lo-80"]
                the_forecast_intervals["upper"] = the_forecast_df["hi-80"]

            des_dict = the_des_fit.__dict__
            wanted_keys = [
                "season_length",
                "error_type",
                "alias",
                "model",
                "damped",
                "phi",
                "prediction_intervals",
            ]
            the_fitted_params_dict = {k: des_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_tes_forecast(self, the_model_name="TES"):
        """TES"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            the_estimator = HoltWinters(season_length=self.seasonal_periods)
            if len(self.train) < 10:
                logger.warning("TES requires 10 datapoints, switching to SES...")
                return self.get_ses_forecast()

            the_tes_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                the_forecast_df = the_tes_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_tes_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])
                the_forecast_intervals["lower"] = the_forecast_df["lo-80"]
                the_forecast_intervals["upper"] = the_forecast_df["hi-80"]

            tes_dict = the_tes_fit.__dict__
            wanted_keys = [
                "season_length",
                "error_type",
                "alias",
                "model",
                "damped",
                "phi",
                "prediction_intervals",
            ]
            the_fitted_params_dict = {k: tes_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_theta_forecast(self, the_model_name="Theta"):
        """Theta"""
        from statsmodels.tsa.forecasting.theta import ThetaModel

        the_forecast, the_forecast_intervals, the_fitted_params = self.__initialize_results()
        try:
            # 1. Build a freq‐aware Series
            freq_str = get_ts_freq_prophet(self.seasonal_periods)
            idx = pd.date_range(
                start=self.random_start_date, periods=len(self.train), freq=freq_str
            )
            y = pd.Series(self.train.values, index=idx)

            # 2. Decide whether to deseasonalize & pick period
            deseasonalize = len(self.train) >= 2 * self.seasonal_periods and self.seasonality
            period = self.seasonal_periods if deseasonalize else 1

            # 3. Instantiate & fit the ThetaModel
            tm = ThetaModel(endog=y, period=period, deseasonalize=deseasonalize)
            res = tm.fit()

            # 4. Point forecast
            fc = res.forecast(steps=self.forecast_horizon)
            the_forecast = fc

            theta_dict = res.__dict__
            wanted_keys = ["_b0", "_alpha", "_sigma2", "_nobs", "_use_mle"]
            the_fitted_params_dict = {k: theta_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception("Theta forecasting error:", e)

        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_croston_forecast(self, the_model_name="Croston"):
        """Croston"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # can change model name from CrostonClassic to CrostonOptimized and CrostonSBA
            the_estimator = CrostonClassic(
                prediction_intervals=ConformalIntervals(n_windows=self.seasonal_periods),
            )

            the_croston_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                the_forecast_df = the_croston_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_croston_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])

            the_fitted_params_dict = {"seasonal_periods": self.seasonal_periods}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def __is_constant_time_series(self):
        a = self.train.to_numpy()
        return (a[0] == a).all(0)

    @tracker.track_runtime
    def get_ets_forecast(self, the_model_name="ETS"):
        """ETS"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # if time series is constant, return same value as prediction - Auto ETS fails in this case
            if self.__is_constant_time_series():
                the_forecast = pd.Series([self.train[0]] * self.forecast_horizon)
                the_forecast_intervals["lower"] = the_forecast.values
                the_forecast_intervals["upper"] = the_forecast.values
            else:
                the_estimator = AutoETS(
                    season_length=(
                        self.seasonal_periods if len(self.train) >= 2 * self.seasonal_periods else 1
                    )
                )
                if len(self.train) < 7:
                    logger.info(
                        f"AutoETS requires at least 7 datapoints, length of train series is {len(self.train)}..."
                    )
                    return the_forecast, the_forecast_intervals, the_fitted_params
                # fit model
                the_autoets_fit = the_estimator.fit(y=self.train)

                if self.in_sample_flag:
                    the_forecast_df = the_autoets_fit.predict_in_sample()
                    the_forecast = pd.Series(the_forecast_df["fitted"])
                else:
                    the_forecast_df = the_autoets_fit.predict(h=self.forecast_horizon, level=[80])
                    the_forecast = pd.Series(the_forecast_df["mean"])
                    the_forecast_intervals["lower"] = the_forecast_df["lo-80"]
                    the_forecast_intervals["upper"] = the_forecast_df["hi-80"]

                autoets_dict = the_autoets_fit.__dict__
                wanted_keys = [
                    "season_length",
                    "alias",
                    "model",
                    "damped",
                    "phi",
                    "prediction_intervals",
                ]
                the_fitted_params_dict = {k: autoets_dict[k] for k in wanted_keys}
                the_fitted_params = ", ".join(
                    f"{k}: {v}" for k, v in the_fitted_params_dict.items()
                )

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    @tracker.track_runtime
    def get_tbats_forecast(self, the_model_name="TBATS"):
        """TBATS"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        try:
            the_estimator = AutoTBATS(
                season_length=self.seasonal_periods,
                use_boxcox=None,
                use_trend=None,
                use_damped_trend=None,
                use_arma_errors=True,
            )

            # TBATS Estimators
            # fit model

            the_autotbats_fit = the_estimator.fit(y=np.array(self.train))

            if self.in_sample_flag:
                the_forecast_df = the_autotbats_fit.predict_in_sample()
                the_forecast = pd.Series(the_forecast_df["fitted"])
            else:
                the_forecast_df = the_autotbats_fit.predict(h=self.forecast_horizon, level=[80])
                the_forecast = pd.Series(the_forecast_df["mean"])
                the_forecast_intervals["lower"] = the_forecast_df["lo-80"]
                the_forecast_intervals["upper"] = the_forecast_df["hi-80"]

            autobats_dict = the_autotbats_fit.__dict__
            wanted_keys = [
                "season_length",
                "use_boxcox",
                "bc_lower_bound",
                "bc_upper_bound",
                "use_trend",
                "use_damped_trend",
                "use_arma_errors",
                "alias",
            ]
            the_fitted_params_dict = {k: autobats_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception(e)
        finally:
            pass
            # the_fitted_params = get_fitted_params(
            #     the_model_name="TBATS", the_estimator=the_estimator
            # )
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_linearregression_forecast(self, the_model_name="Linear Regression"):
        return self.get_linear_reg_forecast()

    @tracker.track_runtime
    def get_linear_reg_forecast(self):
        """Linear Regression"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        # data check
        if len(self.train) < self.seasonal_periods:
            logger.debug(
                f"Linear Regression requires atleast {self.seasonal_periods} datapoints to generate predictions ..."
            )
            return the_forecast, the_forecast_intervals, the_fitted_params
        return self.__get_regressor_forecast(LinearRegression())

    def get_knnregression_forecast(self, the_model_name="KNN Regression"):
        return self.get_knn_reg_forecast()

    @tracker.track_runtime
    def get_knn_reg_forecast(self):
        """KNN Regression"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()

        # data check
        if len(self.train) < self.seasonal_periods:
            logger.debug(
                f"KNN Regression requires atleast {self.seasonal_periods} datapoints to generate predictions ..."
            )
            return the_forecast, the_forecast_intervals, the_fitted_params

        return self.__get_regressor_forecast(
            KNeighborsRegressor(n_neighbors=self.seasonal_periods // 3)
        )

    def __get_regressor_forecast(self, the_estimator):
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # AR - NNET is not available in Python, implementing a Linear Regression model which uses past 6 periods
            if self.in_sample_flag:
                # in sample predictions are not yet implemented for sktime make_reduction
                pass
            else:
                # create linear regression model
                regressor = the_estimator

                # reduce the data to form
                the_estimator = make_reduction(
                    regressor,
                    window_length=self.seasonal_periods // 2,
                    strategy="recursive",
                )

                # get combined length to generate trend index
                total_length = len(self.train) + self.forecast_horizon
                trend_idx_col = "trend_index"
                seasonal_idx_col = "seasonal_idx"
                sin_seasonal_idx_col = "sin_" + seasonal_idx_col
                cos_seasonal_idx_col = "cos_" + seasonal_idx_col

                x_train_test = pd.DataFrame(
                    {trend_idx_col: [x for x in range(1, total_length + 1)]}
                )

                # create repeatable seasonal index
                x_train_test[seasonal_idx_col] = x_train_test[trend_idx_col].mod(
                    self.seasonal_periods
                )

                # take sin cos components
                x_train_test[sin_seasonal_idx_col] = np.sin(
                    x_train_test[seasonal_idx_col] / self.seasonal_periods * 2 * np.pi
                )
                x_train_test[cos_seasonal_idx_col] = np.cos(
                    x_train_test[seasonal_idx_col] / self.seasonal_periods * 2 * np.pi
                )
                x_train_test.drop(seasonal_idx_col, axis=1, inplace=True)

                # normalize trend
                x_train_test[trend_idx_col] = (
                    x_train_test[trend_idx_col] / x_train_test[trend_idx_col].max()
                )

                # separate into train and test sets
                x_train = x_train_test.head(len(self.train))
                x_test = x_train_test.tail(self.forecast_horizon)

                # fit model
                the_estimator.fit(self.train, X=x_train)

                # get predictions, confidence interval is not available
                the_forecast = the_estimator.predict(X=x_test, fh=self.fcst_horizon)
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_simplesnaive_forecast(self, the_model_name="Simple Snaive"):
        return self.get_simple_snaive_forecast()

    @tracker.track_runtime
    def get_simple_snaive_forecast(self):
        """Simple Snaive"""
        return self.get_weighted_snaive_forecast(is_weighted=False)

    def get_weightedsnaive_forecast(self, the_model_name="Weighted Snaive"):
        return self.get_weighted_snaive_forecast()

    @tracker.track_runtime
    def get_weighted_snaive_forecast(self, is_weighted=True):
        """Weighted Snaive"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            if is_weighted:
                weights = self.param_extractor.extract_snaive_params(model_name="Weighted Snaive")
            else:
                weights = []

            cycle_col = "cycle"
            forecast_col = "forecast"
            lag_col = "lag"

            if len(self.train) < self.seasonal_periods:
                snaive_train_series = self.__pad_series()
            else:
                snaive_train_series = self.train

            # convert to dataframe for easier manipulation
            snaive_train_df = pd.DataFrame(snaive_train_series)
            snaive_train_df.columns = [self.history_measure]

            # append forecast horizon (one cycle) rows to the train df for ease of data manipulation
            forecast_rows_one_cycle = pd.DataFrame(
                [np.nan] * self.seasonal_periods, columns=[forecast_col]
            )
            snaive_train_df = snaive_train_df.append(forecast_rows_one_cycle, ignore_index=True)

            # create cycle column - this indicates which cycle/year we are in
            snaive_train_df[cycle_col] = (snaive_train_df.index // self.seasonal_periods) + 1

            all_lag_cols = list()

            if weights:
                # iterate as many weights are available
                list_to_iterate = weights
            else:
                # in case of simple snaive weights will be empty, iterate as many cycles in data
                list_to_iterate = list(snaive_train_df[cycle_col].unique())

            # create as many lag columns as weights supplied
            for the_cycle, the_weight in enumerate(list_to_iterate, 1):
                # estimate num of points to shift
                num_points_to_shift = the_cycle * self.seasonal_periods

                # create lag column name
                the_lag_col_name = lag_col + "_" + str(num_points_to_shift)

                # populate lag column with values
                snaive_train_df[the_lag_col_name] = snaive_train_df[self.history_measure].shift(
                    num_points_to_shift
                )

                # append to master list
                all_lag_cols.append(the_lag_col_name)

            if not weights:
                # if weights are not provided, calculate non null average and populate in predictions column
                snaive_train_df[forecast_col] = snaive_train_df[all_lag_cols].mean(
                    axis=1, skipna=True
                )
            else:
                assert (
                    sum(weights) == 1
                ), "Weights provided : {}, Sum of weights provided should add up to 1 ...".format(
                    weights
                )

                # case 1 - all data is available, multiply and add by weights
                # multiply by weights and take summation
                snaive_train_df[forecast_col] = snaive_train_df[all_lag_cols].mul(weights).sum(1)

            if self.in_sample_flag:
                if len(snaive_train_df) == self.seasonal_periods:
                    logger.debug(
                        "Cannot generate in sample predictions with only one cycle of data ..."
                    )
                else:
                    # filter out rows where actuals are available - to obtain in sample data
                    snaive_train_df = snaive_train_df[snaive_train_df[self.history_measure].notna()]

                    # take n points from tail (since padding might have added rows to head)
                    the_forecast = pd.Series(
                        snaive_train_df.tail(len(self.train))[forecast_col].values
                    )
            else:
                # select predictions from first cycle
                forecast_rows = snaive_train_df[snaive_train_df[self.history_measure].isna()].head(
                    self.seasonal_periods
                )

                # calculate num cycles in forecast horizon
                num_repeats = int(np.ceil(self.forecast_horizon / self.seasonal_periods))

                # repeat it
                forecast_rows_repeated = np.tile(
                    forecast_rows[forecast_col].to_numpy(), num_repeats
                )

                # select required number of points based on forecast horizon
                the_forecast = pd.Series(forecast_rows_repeated[: self.forecast_horizon])

                the_fitted_params_dict = {"seasonal_periods": self.seasonal_periods}
                the_fitted_params = ", ".join(
                    f"{k}: {v}" for k, v in the_fitted_params_dict.items()
                )

                if weights:
                    the_fitted_params = (
                        the_fitted_params + f", weights : {[round(x, 2) for x in weights]}"
                    )

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def __get_thief_periods(self, frequency):
        if frequency == "W":
            return ["W", "M", "Q"]
        elif frequency == "M":
            return ["M", "Q"]
        elif frequency == "Q":
            return ["Q"]

    def __get_thief_hierarchy(self, periods):
        lHierarchy = {}
        lHierarchy["Levels"] = None
        lHierarchy["Data"] = None
        lHierarchy["Groups"] = {}
        lHierarchy["Periods"] = periods
        lHierarchy["Type"] = "Temporal"
        return lHierarchy

    def __get_thief_validation_periods(self, frequency):
        if frequency == "W":
            return 4
        elif frequency == "M":
            return 2
        elif frequency == "Q":
            return 1

    @timed
    def get_thief_forecast(self):
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            if self.in_sample_flag:
                logger.debug("Cannot generate in sample predictions with thief model...")
            else:
                # configurables
                date_col = "Date"
                signal_col = "Signal"
                signal_w_col = "Signal_W"
                signal_w_forecast_col = "Signal_W_Forecast"

                # get time series frequency
                frequency = get_ts_freq_prophet(self.seasonal_periods)

                # get periods
                periods = self.__get_thief_periods(frequency)

                # get validation periods for thief
                thief_validation_periods = self.__get_thief_validation_periods(frequency)

                # create date range starting from any date with frequency supplied
                thief_train_dates = pd.date_range(
                    start=self.random_start_date,
                    periods=len(self.train),
                    freq=frequency,
                )

                # create train dataframe with date and values column
                thief_train_df = pd.DataFrame(
                    {
                        date_col: thief_train_dates,
                        signal_col: self.train.values,
                    }
                )

                # get thief hierarchy
                lHierarchy = self.__get_thief_hierarchy(periods)

                # Configure thief model
                np.random.seed(seed=1960)

                # initialize forecaster
                lEngine = hautof.cHierarchicalForecastEngine()
                lEngine.mOptions.mHierarchicalCombinationMethod = [
                    "BU",
                    "TD",
                    "MO",
                    "OC",
                ]

                # fit on train set
                lEngine.train(
                    thief_train_df,
                    date_col,
                    signal_col,
                    thief_validation_periods,
                    lHierarchy,
                    None,
                )

                # produce forecasts for future
                dfapp_in = thief_train_df.copy()
                dfapp_out = lEngine.forecast(dfapp_in, self.forecast_horizon)

                predicted_values_df = dfapp_out[~dfapp_out[signal_w_col].notna()]

                # extract forecast values and store in numpy array
                predicted_values = (
                    predicted_values_df[signal_w_forecast_col].reset_index(drop=True).values
                )

                # convert to series
                the_forecast = pd.Series(predicted_values)

                # delete variables
                del lEngine

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def __get_growth_naive_median_periods(self):
        if self.seasonal_periods in [52, 53]:
            return 13
        elif self.seasonal_periods == 12:
            return 3
        elif self.seasonal_periods == 4:
            return 1

    def get_growthsnaive_forecast(self, the_model_name="Growth Snaive"):
        return self.get_growth_snaive_forecast()

    @tracker.track_runtime
    def get_growth_snaive_forecast(self):
        """Growth Snaive"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            weights = self.param_extractor.extract_snaive_params(model_name="Growth Snaive")

            # check if there are enough number of datapoints to proceed
            median_periods = self.__get_growth_naive_median_periods()
            required_num_points = self.seasonal_periods + median_periods

            if len(self.train) < required_num_points:
                logger.debug(
                    "growth_snaive cannot work with less than {} datapoints".format(
                        required_num_points
                    )
                )
                return (
                    the_forecast,
                    the_forecast_intervals,
                    the_fitted_params,
                )

            assert sum(weights) == 1, "Sum of weights should add up to 1 ..."

            # col names
            cycle_col = "cycle"
            period_col = "period"
            avg_snaive_col = "avg_snaive"
            growth_snaive_col = "growth_snaive"
            lag_col = "lag"
            lag_median_col = "lag_median"
            recent_trend_col = "recent_{}_periods_trend".format(median_periods)
            weighted_trend_col = "weighted_trend"
            trend_ratio_col = "trend_ratio"
            trend_growth_threshold = 1.25
            trend_decline_threshold = 0.75

            if len(self.train) < self.seasonal_periods:
                snaive_train_series = self.__pad_series()
            else:
                snaive_train_series = self.train

            # convert to dataframe for easier manipulation
            snaive_train_df = pd.DataFrame(snaive_train_series)
            snaive_train_df.columns = [self.history_measure]

            # append forecast horizon (one cycle) rows to the train df for ease of data manipulation
            forecast_rows_one_cycle = pd.DataFrame(
                [np.nan] * self.seasonal_periods, columns=[avg_snaive_col]
            )
            snaive_train_df = snaive_train_df.append(forecast_rows_one_cycle, ignore_index=True)

            # create cycle column - this indicates which cycle/year we are in
            snaive_train_df[cycle_col] = (snaive_train_df.index // self.seasonal_periods) + 1

            # create period column - this will run from 1 to seasonal periods cyclically
            snaive_train_df[period_col] = (snaive_train_df.index % self.seasonal_periods) + 1

            # collect median and seasonal periods from last n periods
            last_n_period_df = snaive_train_df[snaive_train_df[self.history_measure].notna()].tail(
                median_periods
            )
            periods_of_interest = list(last_n_period_df[period_col].unique())

            # calculate recent trend using last n values
            snaive_train_df[recent_trend_col] = np.median(
                last_n_period_df[self.history_measure].to_numpy()
            )

            all_lag_cols = list()
            all_lag_median_cols = list()
            null_flag_list = list()

            for the_cycle, the_weight in enumerate(weights, 1):
                # estimate num of points to shift
                num_points_to_shift = the_cycle * self.seasonal_periods

                # create lag column name
                the_lag_col = lag_col + "_" + str(num_points_to_shift)

                # populate lag column with values
                snaive_train_df[the_lag_col] = snaive_train_df[self.history_measure].shift(
                    num_points_to_shift
                )

                # append to master list
                all_lag_cols.append(the_lag_col)

                # create lagged median col name
                the_lag_median_col = lag_median_col + "_" + str(num_points_to_shift)

                # filter relevant data
                filter_clause = (snaive_train_df[period_col].isin(periods_of_interest)) & (
                    snaive_train_df[self.history_measure].notna()
                )
                lagged_values = (
                    snaive_train_df[filter_clause].tail(median_periods)[the_lag_col].to_numpy()
                )

                # calculate median values
                snaive_train_df[the_lag_median_col] = np.median(lagged_values)

                if snaive_train_df[the_lag_median_col].isnull().all():
                    null_flag_list.append(False)
                else:
                    null_flag_list.append(True)

                # append to master list
                all_lag_median_cols.append(the_lag_median_col)

            # calculate average and populate in predictions column
            snaive_train_df[avg_snaive_col] = snaive_train_df[all_lag_cols].mean(
                axis=1, skipna=True
            )

            # Subsetting the first list based on the boolean flags in the second list
            relevant_median_cols = [
                value for value, flag in zip(all_lag_median_cols, null_flag_list) if flag
            ]
            relevant_weights = [value for value, flag in zip(weights, null_flag_list) if flag]

            logger.debug(f"relevant_median_cols : {relevant_median_cols}")
            logger.debug(f"relevant_weights : {relevant_weights}")

            # Calculate the sum of the weights
            total = sum(relevant_weights)

            # Normalize the weights
            normalized_weights = [round(w / total, 2) for w in relevant_weights]
            logger.debug(f"normalized_weights : {normalized_weights}")

            # case 1 - trend value available for all cycles, multiply by weights and take summation
            snaive_train_df[weighted_trend_col] = (
                snaive_train_df[relevant_median_cols].mul(normalized_weights).sum(1)
            )

            # # case 1 - trend value available for all cycles, multiply by weights and take summation
            # snaive_train_df[weighted_trend_col] = (
            #     snaive_train_df[all_lag_median_cols].mul(weights).sum(1)
            # )

            # # case 2 - trend value is available for LY, but not rest
            # filter_clause = pd.Series([True] * len(snaive_train_df))
            # for the_index, the_col in enumerate(all_lag_median_cols, 1):
            #     if the_index == 1:
            #         # clause where LY value is not null
            #         filter_clause = (
            #             filter_clause & snaive_train_df[the_col].notna()
            #         )
            #     else:
            #         # others are null
            #         filter_clause = (
            #             filter_clause & snaive_train_df[the_col].isna()
            #         )

            # # weighted trend will be same as historical trend
            # snaive_train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     snaive_train_df[all_lag_median_cols[0]],
            #     snaive_train_df[weighted_trend_col],
            # )

            # # case 3 - trend LY, LLY is available
            # filter_clause = (
            #     snaive_train_df[all_lag_median_cols[0]].notna()
            #     & snaive_train_df[all_lag_median_cols[1]].notna()
            # )
            # weights = [3 / 4, 1 / 4]
            # snaive_train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     snaive_train_df[all_lag_median_cols[0]] * weights[0]
            #     + snaive_train_df[all_lag_median_cols[1]] * weights[1],
            #     snaive_train_df[weighted_trend_col],
            # )

            # # case 4 - 3 cycles of data is available, last cycle is missing
            # filter_clause = (
            #     snaive_train_df[all_lag_median_cols[0]].notna()
            #     & snaive_train_df[all_lag_median_cols[1]].notna()
            #     & snaive_train_df[all_lag_median_cols[2]].notna()
            # )
            # weights = [5 / 9, 3 / 9, 1 / 9]
            # snaive_train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     snaive_train_df[all_lag_median_cols[0]] * weights[0]
            #     + snaive_train_df[all_lag_median_cols[1]] * weights[1]
            #     + snaive_train_df[all_lag_median_cols[2]] * weights[2],
            #     snaive_train_df[weighted_trend_col],
            # )

            # filter records where value is not zero
            filter_clause = snaive_train_df[weighted_trend_col] != 0

            # calculate trend multiplier
            snaive_train_df[trend_ratio_col] = (
                snaive_train_df[recent_trend_col] / snaive_train_df[weighted_trend_col]
            )

            # capping trend growth based on growth and decline thresholds
            trend_growth_threshold_filter_clause = (
                snaive_train_df[trend_ratio_col] > trend_growth_threshold
            )
            trend_decline_threshold_filter_clause = (
                snaive_train_df[trend_ratio_col] < trend_decline_threshold
            )
            snaive_train_df[trend_ratio_col] = np.where(
                trend_growth_threshold_filter_clause,
                trend_growth_threshold,
                np.where(
                    trend_decline_threshold_filter_clause,
                    trend_decline_threshold,
                    snaive_train_df[trend_ratio_col],
                ),
            )

            # calculate growth snaive value
            snaive_train_df[growth_snaive_col] = np.where(
                filter_clause,
                # happy path
                snaive_train_df[avg_snaive_col] * snaive_train_df[trend_ratio_col],
                # assign avg snaive values as fallback where weighted_trend value is zero
                snaive_train_df[avg_snaive_col],
            )

            if self.in_sample_flag:
                if len(snaive_train_df) == self.seasonal_periods:
                    logger.debug(
                        "Cannot generate in sample predictions with only one cycle of data ..."
                    )
                else:
                    # filter out rows where actuals are available - to obtain in sample data
                    snaive_train_df = snaive_train_df[snaive_train_df[self.history_measure].notna()]

                    # take n points from tail (since padding might have added rows to head)
                    the_forecast = pd.Series(
                        snaive_train_df.tail(len(self.train))[growth_snaive_col].values
                    )
            else:
                # select predictions from first cycle
                forecast_rows = snaive_train_df[snaive_train_df[self.history_measure].isna()].head(
                    self.seasonal_periods
                )

                # calculate num cycles in forecast horizon
                num_repeats = int(np.ceil(self.forecast_horizon / self.seasonal_periods))

                # repeat it
                forecast_rows_repeated = np.tile(
                    forecast_rows[growth_snaive_col].to_numpy(), num_repeats
                )

                # select required number of points based on forecast horizon
                the_forecast = pd.Series(forecast_rows_repeated[: self.forecast_horizon])

                the_fitted_params_dict = {"seasonal_periods": self.seasonal_periods}
                the_fitted_params = ", ".join(
                    f"{k}: {v}" for k, v in the_fitted_params_dict.items()
                )
                if weights:
                    the_fitted_params = (
                        the_fitted_params + f", weights : {[round(x, 2) for x in weights]}"
                    )

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_simpleaoa_forecast(self, the_model_name="Simple AOA"):
        return self.get_simple_aoa_forecast()

    @tracker.track_runtime
    def get_simple_aoa_forecast(self):
        """Simple AOA"""
        return self.get_weighted_aoa_forecast(is_weighted=False)

    def __get_holiday_attention(self, the_idx_col, the_avg_col):

        # combine holiday data with actuals
        combined = pd.DataFrame(self.train).join(self.train_exog_df)

        # means
        holiday_means = combined.groupby(self.holiday_type_col).mean().reset_index()

        count_col = "count"
        new_mean_col = "new_mean"

        # there could be multiple holidays in a week/month/quarter
        # split by delimiter and get count in every week/month/quarter
        holiday_means[count_col] = holiday_means[self.holiday_type_col].apply(
            lambda row: len(row.split(","))
        )

        # divide mean - split the mean equally among all members
        holiday_means[new_mean_col] = holiday_means[self.history_measure].divide(
            holiday_means[count_col]
        )

        # get all combinations
        all_combinations = []

        # iterate over all rows
        for the_idx, the_row in holiday_means.iterrows():

            # data at base level - might have multiple holidays in same time window
            the_df = pd.DataFrame(
                {
                    self.holiday_type_col: the_row[self.holiday_type_col],
                    new_mean_col: the_row[self.history_measure],
                },
                index=[0],
            )
            all_combinations.append(the_df)

            # if there are more than one holiday, split on delimiter and assign new mean to every holiday present
            if "," in the_row[self.holiday_type_col]:
                keys = the_row[self.holiday_type_col].split(",")
                for the_key in keys:
                    the_df = pd.DataFrame(
                        {
                            self.holiday_type_col: the_key,
                            new_mean_col: the_row[new_mean_col],
                        },
                        index=[0],
                    )
                    all_combinations.append(the_df)

        # collect all results
        group_means = concat_to_dataframe(all_combinations)

        # if there are more than one mean for the same holiday, take the max among both
        result = group_means.groupby(self.holiday_type_col).max().reset_index()

        # rename columns for ease of join
        result.rename(
            columns={
                self.holiday_type_col: the_idx_col,
                new_mean_col: the_avg_col,
            },
            inplace=True,
        )

        # whereever holiday type is NA, assign nan
        filter_condition = result[the_idx_col] == "NA"
        result.loc[filter_condition, the_avg_col] = np.nan

        return result

    def __get_attention(self, idx_col, train_df, the_avg_col):
        if idx_col == "holiday_idx":
            attention_df = self.__get_holiday_attention(idx_col, the_avg_col)
        else:
            # remove rows where actual is na
            actuals = train_df[train_df[self.history_measure].notna()]

            # calculate group averages and return dataframe
            attention_df = actuals.groupby(idx_col).mean()[[self.history_measure]].reset_index()

            # rename column
            attention_df.rename(columns={self.history_measure: the_avg_col}, inplace=True)
        return attention_df

    def __add_time_idx_cols(self, train_df, week_idx_col, month_idx_col, quarter_idx_col):
        if self.seasonal_periods in [52, 53]:
            train_df[week_idx_col] = (train_df.index % 52) + 1
            train_df[month_idx_col] = (train_df.index % 12) + 1
        elif self.seasonal_periods == 12:
            train_df[month_idx_col] = (train_df.index % 12) + 1
        elif self.seasonal_periods == 4:
            train_df[quarter_idx_col] = (train_df.index % 4) + 1

        return train_df

    def get_weightedaoa_forecast(self, the_model_name="Weighted AOA"):
        return self.get_weighted_aoa_forecast()

    @tracker.track_runtime
    def get_weighted_aoa_forecast(self, is_weighted=True):
        """Weighted AOA"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            if is_weighted:
                weights = self.param_extractor.extract_attention_weights(
                    model_name="Weighted AOA",
                    seasonal_periods=self.seasonal_periods,
                )
            else:
                weights = []

            week_idx_col = "week_idx"
            month_idx_col = "month_idx"
            quarter_idx_col = "quarter_idx"
            holiday_idx_col = "holiday_idx"
            week_avg_col = "week_avg"
            month_avg_col = "month_avg"
            quarter_avg_col = "quarter_avg"
            holiday_avg_col = "holiday_avg"
            forecast_col = "aoa_forecast"

            if len(self.train) < self.seasonal_periods:
                snaive_train_series = self.__pad_series()
            else:
                snaive_train_series = self.train

            # convert to dataframe for easier manipulation
            train_df = pd.DataFrame(snaive_train_series)
            train_df.columns = [self.history_measure]

            if self.in_sample_flag:
                pass
            else:
                # append forecast horizon rows to the train df for ease of data manipulation
                forecast_rows = pd.DataFrame(
                    [np.nan] * self.forecast_horizon, columns=[forecast_col]
                )
                train_df = train_df.append(forecast_rows, ignore_index=True)

            # add time indices based on frequency
            train_df = self.__add_time_idx_cols(
                train_df, week_idx_col, month_idx_col, quarter_idx_col
            )

            # add holiday idx if needed
            if self.use_holidays:
                if self.in_sample_flag:
                    if len(train_df) != len(self.train_exog_df):
                        # cases where there is less than one cycle of data present, we pad the actuals, need to pad holidays as well
                        # collect number of points to pad
                        points_to_pad = len(train_df) - len(self.train_exog_df)

                        # pad holidays
                        all_holidays = ["NA"] * points_to_pad

                        # collect existing values
                        existing_holidays = list(self.train_exog_df[self.holiday_type_col])

                        # add to the same master list
                        all_holidays.extend(existing_holidays)

                        train_df[holiday_idx_col] = all_holidays
                    else:
                        # happy path
                        train_df[holiday_idx_col] = list(self.train_exog_df[self.holiday_type_col])
                else:
                    # collect holidays from train and test series
                    holiday_list = list(self.train_exog_df[self.holiday_type_col]) + list(
                        self.test_exog_df[self.holiday_type_col]
                    )

                    # assign back to source dataframe
                    train_df[holiday_idx_col] = holiday_list

            # generate averages for every group
            mapping = {
                week_idx_col: week_avg_col,
                month_idx_col: month_avg_col,
                quarter_idx_col: quarter_avg_col,
                holiday_idx_col: holiday_avg_col,
            }

            # get list of available idx cols
            req_idx_cols = [the_col for the_col in train_df.columns if "_idx" in the_col]

            # get list of available avg cols
            req_avg_cols = [mapping[the_idx_col] for the_idx_col in req_idx_cols]

            for the_idx_col in req_idx_cols:
                # get col name corresponding to idx_col
                the_avg_col = mapping[the_idx_col]

                # get attention values
                the_attention = self.__get_attention(the_idx_col, train_df, the_avg_col)

                # join back on train df
                train_df = train_df.merge(the_attention, how="left", on=the_idx_col)

            if not weights:
                # calculate mean across the required columns
                train_df[forecast_col] = train_df[req_avg_cols].mean(axis=1, skipna=True)
            else:
                # if there's only one column available, assign 100% weight
                if len(req_avg_cols) == 1:
                    weights = [1.0]

                # calculate weighted mean across the required columns
                train_df[forecast_col] = (
                    train_df[req_avg_cols].mul(weights[: len(req_avg_cols)]).sum(1)
                )

                if self.use_holidays:
                    # where holidays is na, calculate averages from other available cols
                    filter_clause = train_df[holiday_avg_col].isna()

                    if self.seasonal_periods in [52, 53]:
                        week_weight = 0.75
                        month_weight = 0.25
                        train_df[forecast_col] = np.where(
                            filter_clause,
                            train_df[week_avg_col] * week_weight
                            + train_df[month_avg_col] * month_weight,
                            train_df[forecast_col],
                        )
                    elif self.seasonal_periods == 12:
                        train_df[forecast_col] = np.where(
                            filter_clause,
                            train_df[month_avg_col],
                            train_df[forecast_col],
                        )
                    elif self.seasonal_periods == 4:
                        train_df[forecast_col] = np.where(
                            filter_clause,
                            train_df[quarter_avg_col],
                            train_df[forecast_col],
                        )

            if self.in_sample_flag:
                the_forecast = train_df.head(len(self.train))[forecast_col]
            else:
                # select predictions from first cycle
                forecast_rows = train_df[train_df[self.history_measure].isna()].head(
                    self.seasonal_periods
                )

                # calculate num cycles in forecast horizon
                num_repeats = int(np.ceil(self.forecast_horizon / self.seasonal_periods))

                # repeat it
                forecast_rows_repeated = np.tile(
                    forecast_rows[forecast_col].to_numpy(), num_repeats
                )

                # select required number of points based on forecast horizon
                the_forecast = pd.Series(forecast_rows_repeated[: self.forecast_horizon])

                the_fitted_params_dict = {"seasonal_periods": self.seasonal_periods}
                the_fitted_params = ", ".join(
                    f"{k}: {v}" for k, v in the_fitted_params_dict.items()
                )
                if weights:
                    the_fitted_params = (
                        the_fitted_params + f", weights : {[round(x, 2) for x in weights]}"
                    )

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def get_growthaoa_forecast(self, the_model_name="Growth AOA"):
        return self.get_growth_aoa_forecast()

    @tracker.track_runtime
    def get_growth_aoa_forecast(
        self,
    ):
        """Growth AOA"""
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            aoa_weights = self.param_extractor.extract_attention_weights(
                model_name="Growth AOA", seasonal_periods=self.seasonal_periods
            )

            trend_weights = self.param_extractor.extract_g_woa_params(model_name="Growth AOA")

            # check if there are enough number of datapoints to proceed
            median_periods = self.__get_growth_naive_median_periods()
            required_num_points = self.seasonal_periods + median_periods

            if len(self.train) < required_num_points:
                logger.debug(
                    "growth_aoa cannot work with less than {} datapoints".format(
                        required_num_points
                    )
                )
                return (
                    the_forecast,
                    the_forecast_intervals,
                    the_fitted_params,
                )

            # configurables
            week_idx_col = "week_idx"
            month_idx_col = "month_idx"
            quarter_idx_col = "quarter_idx"
            holiday_idx_col = "holiday_idx"
            week_avg_col = "week_avg"
            month_avg_col = "month_avg"
            quarter_avg_col = "quarter_avg"
            holiday_avg_col = "holiday_avg"
            trend_ratio_col = "trend_ratio"
            trend_growth_threshold = 1.25
            trend_decline_threshold = 0.75

            # col names
            cycle_col = "cycle"
            period_col = "period"
            avg_aoa_col = "avg_aoa"
            growth_aoa_col = "growth_aoa"
            lag_col = "lag"
            lag_median_col = "lag_median"
            median_periods = self.__get_growth_naive_median_periods()
            recent_trend_col = "recent_{}_periods_trend".format(median_periods)
            weighted_trend_col = "weighted_trend"

            if len(self.train) < self.seasonal_periods:
                snaive_train_series = self.__pad_series()
            else:
                snaive_train_series = self.train

            # convert to dataframe for easier manipulation
            train_df = pd.DataFrame(snaive_train_series)
            train_df.columns = [self.history_measure]

            if self.in_sample_flag:
                pass
            else:
                # append forecast horizon rows to the train df for ease of data manipulation
                forecast_rows = pd.DataFrame(
                    [np.nan] * self.forecast_horizon, columns=[avg_aoa_col]
                )
                train_df = train_df.append(forecast_rows, ignore_index=True)

            # add time indices based on frequency
            train_df = self.__add_time_idx_cols(
                train_df, week_idx_col, month_idx_col, quarter_idx_col
            )

            # add holiday idx if needed
            if self.use_holidays:
                if self.in_sample_flag:
                    if len(train_df) != len(self.train_exog_df):
                        # cases where there is less than one cycle of data present, we pad the actuals, need to pad holidays as well
                        # collect number of points to pad
                        points_to_pad = len(train_df) - len(self.train_exog_df)

                        # pad holidays
                        all_holidays = ["NA"] * points_to_pad

                        # collect existing values
                        existing_holidays = list(self.train_exog_df[self.holiday_type_col])

                        # add to the same master list
                        all_holidays.extend(existing_holidays)

                        train_df[holiday_idx_col] = all_holidays
                    else:
                        # happy path
                        train_df[holiday_idx_col] = list(self.train_exog_df[self.holiday_type_col])
                else:
                    # collect holidays from train and test series
                    holiday_list = list(self.train_exog_df[self.holiday_type_col]) + list(
                        self.test_exog_df[self.holiday_type_col]
                    )

                    # assign back to source dataframe
                    train_df[holiday_idx_col] = holiday_list

            # generate averages for every group
            mapping = {
                week_idx_col: week_avg_col,
                month_idx_col: month_avg_col,
                quarter_idx_col: quarter_avg_col,
                holiday_idx_col: holiday_avg_col,
            }

            # get list of available idx cols
            req_idx_cols = [the_col for the_col in train_df.columns if "_idx" in the_col]

            # get list of available avg cols
            req_avg_cols = [mapping[the_idx_col] for the_idx_col in req_idx_cols]

            for the_idx_col in req_idx_cols:
                # get col name corresponding to idx_col
                the_avg_col = mapping[the_idx_col]

                # get attention values
                the_attention = self.__get_attention(the_idx_col, train_df, the_avg_col)

                # join back on train df
                train_df = train_df.merge(the_attention, how="left", on=the_idx_col)

            if not aoa_weights:
                # calculate mean across the required columns
                train_df[avg_aoa_col] = train_df[req_avg_cols].mean(axis=1, skipna=True)
            else:
                # if there's only one column available, assign 100% weight
                if len(req_avg_cols) == 1:
                    aoa_weights = [1.0]

                # calculate weighted mean across the required columns
                train_df[avg_aoa_col] = (
                    train_df[req_avg_cols].mul(aoa_weights[: len(req_avg_cols)]).sum(1)
                )

                if self.use_holidays:
                    # where holidays is na, calculate averages from other available cols
                    filter_clause = train_df[holiday_avg_col].isna()

                    if self.seasonal_periods in [52, 53]:
                        week_weight = 0.75
                        month_weight = 0.25
                        train_df[avg_aoa_col] = np.where(
                            filter_clause,
                            train_df[week_avg_col] * week_weight
                            + train_df[month_avg_col] * month_weight,
                            train_df[avg_aoa_col],
                        )
                    elif self.seasonal_periods == 12:
                        train_df[avg_aoa_col] = np.where(
                            filter_clause,
                            train_df[month_avg_col],
                            train_df[avg_aoa_col],
                        )
                    elif self.seasonal_periods == 4:
                        train_df[avg_aoa_col] = np.where(
                            filter_clause,
                            train_df[quarter_avg_col],
                            train_df[avg_aoa_col],
                        )

            # create cycle column - this indicates which cycle/year we are in
            train_df[cycle_col] = (train_df.index // self.seasonal_periods) + 1

            # create period column - this will run from 1 to seasonal periods cyclically
            train_df[period_col] = (train_df.index % self.seasonal_periods) + 1

            # collect median and seasonal periods from last n periods
            last_n_period_df = train_df[train_df[self.history_measure].notna()].tail(median_periods)
            periods_of_interest = list(last_n_period_df[period_col].unique())

            # calculate recent trend using last n values
            train_df[recent_trend_col] = np.median(
                last_n_period_df[self.history_measure].to_numpy()
            )

            all_lag_cols = list()
            all_lag_median_cols = list()
            null_flag_list = list()
            for the_cycle, the_weight in enumerate(trend_weights, 1):
                # estimate num of points to shift
                num_points_to_shift = the_cycle * self.seasonal_periods

                # create lag column name
                the_lag_col = lag_col + "_" + str(num_points_to_shift)

                # populate lag column with values
                train_df[the_lag_col] = train_df[self.history_measure].shift(num_points_to_shift)

                # append to master list
                all_lag_cols.append(the_lag_col)

                # create lagged median col name
                the_lag_median_col = lag_median_col + "_" + str(num_points_to_shift)

                # filter relevant data
                filter_clause = (train_df[period_col].isin(periods_of_interest)) & (
                    train_df[self.history_measure].notna()
                )
                lagged_values = train_df[filter_clause].tail(median_periods)[the_lag_col].to_numpy()

                # calculate median values
                train_df[the_lag_median_col] = np.median(lagged_values)

                if train_df[the_lag_median_col].isnull().all():
                    null_flag_list.append(False)
                else:
                    null_flag_list.append(True)

                # append to master list
                all_lag_median_cols.append(the_lag_median_col)

            # Subsetting the first list based on the boolean flags in the second list
            relevant_median_cols = [
                value for value, flag in zip(all_lag_median_cols, null_flag_list) if flag
            ]
            relevant_weights = [value for value, flag in zip(trend_weights, null_flag_list) if flag]

            logger.debug(f"relevant_median_cols : {relevant_median_cols}")
            logger.debug(f"relevant_weights : {relevant_weights}")

            # Calculate the sum of the weights
            total = sum(relevant_weights)

            # Normalize the weights
            normalized_weights = [round(w / total, 2) for w in relevant_weights]
            logger.debug(f"normalized_weights : {normalized_weights}")

            # case 1 - trend value available for all cycles, multiply by weights and take summation
            train_df[weighted_trend_col] = (
                train_df[relevant_median_cols].mul(normalized_weights).sum(1)
            )

            # # case 2 - trend value is available for LY, but not rest
            # filter_clause = pd.Series([True] * len(train_df))
            # for the_index, the_col in enumerate(all_lag_median_cols, 1):
            #     if the_index == 1:
            #         # clause where LY value is not null
            #         filter_clause = filter_clause & train_df[the_col].notna()
            #     else:
            #         # others are null
            #         filter_clause = filter_clause & train_df[the_col].isna()

            # # weighted trend will be same as historical trend
            # train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     train_df[all_lag_median_cols[0]],
            #     train_df[weighted_trend_col],
            # )

            # # case 3 - trend LY, LLY is available
            # filter_clause = (
            #     train_df[all_lag_median_cols[0]].notna()
            #     & train_df[all_lag_median_cols[1]].notna()
            # )
            # weights = [3 / 4, 1 / 4]
            # train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     train_df[all_lag_median_cols[0]] * weights[0]
            #     + train_df[all_lag_median_cols[1]] * weights[1],
            #     train_df[weighted_trend_col],
            # )

            # # case 4 - 3 cycles of data is available, last cycle is missing
            # filter_clause = (
            #     train_df[all_lag_median_cols[0]].notna()
            #     & train_df[all_lag_median_cols[1]].notna()
            #     & train_df[all_lag_median_cols[2]].notna()
            # )
            # weights = [5 / 9, 3 / 9, 1 / 9]
            # train_df[weighted_trend_col] = np.where(
            #     filter_clause,
            #     train_df[all_lag_median_cols[0]] * weights[0]
            #     + train_df[all_lag_median_cols[1]] * weights[1]
            #     + train_df[all_lag_median_cols[2]] * weights[2],
            #     train_df[weighted_trend_col],
            # )

            # filter records where value is not zero
            filter_clause = train_df[weighted_trend_col] != 0

            # calculate trend multiplier
            train_df[trend_ratio_col] = train_df[recent_trend_col] / train_df[weighted_trend_col]

            # capping trend growth based on growth and decline threshold
            trend_growth_threshold_filter_clause = (
                train_df[trend_ratio_col] > trend_growth_threshold
            )
            trend_decline_threshold_filter_clause = (
                train_df[trend_ratio_col] < trend_decline_threshold
            )
            train_df[trend_ratio_col] = np.where(
                trend_growth_threshold_filter_clause,
                trend_growth_threshold,
                np.where(
                    trend_decline_threshold_filter_clause,
                    trend_decline_threshold,
                    train_df[trend_ratio_col],
                ),
            )

            # calculate growth snaive value
            train_df[growth_aoa_col] = np.where(
                filter_clause,
                # happy path
                train_df[avg_aoa_col] * train_df[trend_ratio_col],
                # assign avg aoa values as fallback
                train_df[avg_aoa_col],
            )

            if self.in_sample_flag:
                if len(train_df) == self.seasonal_periods:
                    logger.debug(
                        "Cannot generate in sample predictions with only one cycle of data ..."
                    )
                else:
                    # filter out rows where actuals are available - to obtain in sample data
                    snaive_train_df = train_df[train_df[self.history_measure].notna()]

                    # take n points from tail (since padding might have added rows to head)
                    the_forecast = pd.Series(
                        snaive_train_df.tail(len(self.train))[growth_aoa_col].values
                    )
            else:
                # select predictions from first cycle
                forecast_rows = train_df[train_df[self.history_measure].isna()].head(
                    self.seasonal_periods
                )

                # calculate num cycles in forecast horizon
                num_repeats = int(np.ceil(self.forecast_horizon / self.seasonal_periods))

                # repeat it
                forecast_rows_repeated = np.tile(
                    forecast_rows[growth_aoa_col].to_numpy(), num_repeats
                )

                # select required number of points based on forecast horizon
                the_forecast = pd.Series(forecast_rows_repeated[: self.forecast_horizon])

            the_fitted_params_dict = {"seasonal_periods": self.seasonal_periods}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())
            if trend_weights:
                the_fitted_params = (
                    the_fitted_params + f", weights : {[round(x, 2) for x in trend_weights]}"
                )

            if aoa_weights:
                aoa_components = [x.split("_")[0] for x in req_avg_cols]
                the_string = ", ".join(f"{a}:{b}" for a, b in zip(aoa_components, aoa_weights))
                the_fitted_params = the_fitted_params + f", aoa_weights : {the_string}"

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params

    def sf_auto_arima_updated(self):
        (
            the_forecast,
            the_forecast_intervals,
            the_fitted_params,
        ) = self.__initialize_results()
        try:
            # random start date to assign for prophet/mstl
            random_start_date = "2016-01-01"

            # generate a time series index based on frequency, assign any random start date and generate equally distanced time series
            # prophet requires equally spaced time series, planning calendar will not work
            auto_arima_train_index = pd.date_range(
                start=random_start_date,
                periods=len(self.train),
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )

            auto_arima_train = pd.DataFrame(
                data=self.train.values,
                index=auto_arima_train_index,
                columns=["y"],
            )
            auto_arima_train.reset_index(inplace=True)
            auto_arima_train.rename(columns={"index": "ds"}, inplace=True)
            auto_arima_train["unique_id"] = "series_1"

            differencing = self.param_extractor.extract_param_value(
                algorithm="sARIMA",
                parameter="Differencing",
            )

            sf = StatsForecast(
                models=[
                    AutoARIMA(
                        season_length=self.seasonal_periods if self.seasonality else 1,
                        D=int(differencing),
                    )
                ],
                freq=get_ts_freq_prophet(self.seasonal_periods),
                n_jobs=1,
            )

            sf.fit(auto_arima_train)
            if self.in_sample_flag:
                # need to call forecast first and then access fitted values
                _ = sf.forecast(
                    self.forecast_horizon,
                    level=(100 - (self.confidence_interval_alpha * 100),),
                    fitted=True,
                )
                the_forecast = pd.Series(sf.forecast_fitted_values()["AutoARIMA"].values)
            else:
                the_forecast_df = sf.predict(
                    h=self.forecast_horizon,
                    level=[int(100 - (self.confidence_interval_alpha * 100))],
                )

                mean_col = [x for x in the_forecast_df.columns if x.endswith("AutoARIMA")][0]
                lb_col = [x for x in the_forecast_df.columns if x.endswith("lo-80")][0]
                ub_col = [x for x in the_forecast_df.columns if x.endswith("hi-80")][0]

                the_forecast = pd.Series(the_forecast_df[mean_col].values)
                the_forecast_intervals["lower"] = the_forecast_df[lb_col].values
                the_forecast_intervals["upper"] = the_forecast_df[ub_col].values

            AutoARIMA_dict = sf.models[0].__dict__
            wanted_keys = [
                "d",
                "D",
                "start_p",
                "max_p",
                "start_q",
                "max_q",
                "start_P",
                "max_P",
                "start_Q",
                "max_Q",
                "max_order",
                "max_d",
                "max_D",
                "seasonal",
                "test",
                "seasonal_test",
                "season_length",
            ]
            the_fitted_params_dict = {k: AutoARIMA_dict[k] for k in wanted_keys}
            the_fitted_params = ", ".join(f"{k}: {v}" for k, v in the_fitted_params_dict.items())

        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals, the_fitted_params


if __name__ == "__main__":
    seasonal_periods = 52
    forecast_horizon = 120
    np.random.seed(1234)

    # Create train series
    train = pd.Series(np.random.randint(1000, size=169))

    train = pd.Series(
        [
            128,
            60,
            64,
            144,
            160,
            84,
            116,
            148,
            152,
            240,
            172,
            116,
            224,
            136,
            236,
            96,
            156,
            200,
            261,
            75,
            92,
            268,
            75,
            70,
            28,
            0,
            0,
            52,
            168,
            192,
            276,
            263,
            111,
            68,
            0,
            184,
            120,
            242,
            187,
            28,
            300,
            188,
            260,
            158,
            184,
            345.3420211,
            384,
            33,
            96,
            88,
            40,
            68,
            140,
            0,
            96,
            60,
            352,
            108,
            432,
            68,
            172,
            184,
            116,
            184,
            344,
            345.3420211,
            0,
            144,
            20,
            28,
            52,
            156,
            372,
            276,
            252,
            148,
            180,
            228,
            160,
            148,
            172,
            96,
            248,
            196,
            257,
            172,
            183,
            345.3420211,
            345.3420211,
            110,
            236,
            240,
        ]
    )

    # create forecaster object
    forecaster = o9Forecaster(
        train=train,
        seasonal_periods=seasonal_periods,
        in_sample_flag=False,
        forecast_horizon=forecast_horizon,
        confidence_interval_alpha=0.20,
    )
    logger.debug("train : {}".format(train))

    result = forecaster.get_stlf_forecast()

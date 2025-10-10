"""
Version : 0.0.0
Maintained by : dpref@o9solutions.com
"""

import logging

import pandas as pd
import statsmodels as sm
from o9Reference.common_utils.dataframe_utils import create_cartesian_product

logger = logging.getLogger("o9_logger")


# TODO : Deprecate next release
def get_weekly_default_params(
    stat_algo_col: str,
    stat_parameter_col: str,
    stat_param_value_col: str,
):
    """
    Returns the default params at week level
    :param stat_algo_col:
    :param stat_parameter_col:
    :param stat_param_value_col:
    :return:
    """
    # create default params and dummy value column on intersections master df for cross join in pandas
    AlgoParameters_df = pd.DataFrame(
        [
            ["sARIMA", "AR Order", 0],
            ["sARIMA", "Differencing", 0],
            ["sARIMA", "MA Order", 0],
            ["sARIMA", "Seasonal AR Order", 0],
            ["sARIMA", "Seasonal Differencing", 1],
            ["sARIMA", "Seasonal MA Order", 0],
            ["SES", "Alpha Lower", 0],
            ["SES", "Alpha Upper", 0.075],
            ["DES", "Alpha Lower", 0],
            ["DES", "Alpha Upper", 0.075],
            ["DES", "Beta Lower", 0],
            ["DES", "Beta Upper", 0.07],
            ["DES", "Gamma Lower", 0],
            ["DES", "Gamma Upper", 0.3],
            ["DES", "Phi Lower", 0.8],
            ["DES", "Phi Upper", 0.95],
            ["TES", "Alpha Lower", 0],
            ["TES", "Alpha Upper", 0.075],
            ["TES", "Beta Lower", 0],
            ["TES", "Beta Upper", 0.07],
            ["TES", "Gamma Lower", 0],
            ["TES", "Gamma Upper", 0.3],
            ["TES", "Phi Lower", 0.8],
            ["TES", "Phi Upper", 0.95],
            ["Moving Average", "Period", 13],
            ["Naive Random Walk", "No Parameters", 0],
            ["Seasonal Naive YoY", "No Parameters", 0],
            ["ETS", "No Parameters", 0],
            ["Auto ARIMA", "No Parameters", 0],
            ["Croston", "No Parameters", 0],
            ["TBATS", "No Parameters", 0],
            ["STLF", "No Parameters", 0],
            ["AR-NNET", "No Parameters", 0],
            ["Theta", "No Parameters", 0],
            ["Weighted Snaive", "LY Weight", 0.6],
            ["Weighted Snaive", "LLY Weight", 0.25],
            ["Weighted Snaive", "LLLY Weight", 0.15],
            ["Weighted Snaive", "LLLLY Weight", 0],
            ["Growth Snaive", "LY Weight", 0.6],
            ["Growth Snaive", "LLY Weight", 0.25],
            ["Growth Snaive", "LLLY Weight", 0.15],
            ["Growth Snaive", "LLLLY Weight", 0],
            ["Weighted AOA", "Week Attention", 0.6],
            ["Weighted AOA", "Month Attention", 0.25],
            ["Weighted AOA", "Quarter Attention", 0],
            ["Weighted AOA", "Holiday Attention", 0.15],
            ["Growth AOA", "Week Attention", 0.6],
            ["Growth AOA", "Month Attention", 0.25],
            ["Growth AOA", "Quarter Attention", 0],
            ["Growth AOA", "Holiday Attention", 0.15],
            ["Growth AOA", "LY Weight", 0.6],
            ["Growth AOA", "LLY Weight", 0.25],
            ["Growth AOA", "LLLY Weight", 0.15],
            ["Growth AOA", "LLLLY Weight", 0],
        ],
        columns=[stat_algo_col, stat_parameter_col, stat_param_value_col],
    )
    return AlgoParameters_df


# TODO : Deprecate next release
def get_monthly_default_params(
    stat_algo_col: str,
    stat_parameter_col: str,
    stat_param_value_col: str,
):
    """
    Returns the default params at month level
    :param stat_algo_col:
    :param stat_parameter_col:
    :param stat_param_value_col:
    :return:
    """
    # create default params and dummy value column on intersections master df for cross join in pandas
    AlgoParameters_df = pd.DataFrame(
        [
            ["sARIMA", "AR Order", 0],
            ["sARIMA", "Differencing", 0],
            ["sARIMA", "MA Order", 0],
            ["sARIMA", "Seasonal AR Order", 0],
            ["sARIMA", "Seasonal Differencing", 1],
            ["sARIMA", "Seasonal MA Order", 0],
            ["SES", "Alpha Lower", 0],
            ["SES", "Alpha Upper", 0.3],
            ["DES", "Alpha Lower", 0],
            ["DES", "Alpha Upper", 0.3],
            ["DES", "Beta Lower", 0],
            ["DES", "Beta Upper", 0.2],
            ["DES", "Gamma Lower", 0],
            ["DES", "Gamma Upper", 0.3],
            ["DES", "Phi Lower", 0.2],
            ["DES", "Phi Upper", 0.5],
            ["TES", "Alpha Lower", 0],
            ["TES", "Alpha Upper", 0.3],
            ["TES", "Beta Lower", 0],
            ["TES", "Beta Upper", 0.2],
            ["TES", "Gamma Lower", 0],
            ["TES", "Gamma Upper", 0.3],
            ["TES", "Phi Lower", 0.8],
            ["TES", "Phi Upper", 0.95],
            ["Moving Average", "Period", 3],
            ["Naive Random Walk", "No Parameters", 0],
            ["Seasonal Naive YoY", "No Parameters", 0],
            ["ETS", "No Parameters", 0],
            ["Auto ARIMA", "No Parameters", 0],
            ["Croston", "No Parameters", 0],
            ["TBATS", "No Parameters", 0],
            ["STLF", "No Parameters", 0],
            ["AR-NNET", "No Parameters", 0],
            ["Theta", "No Parameters", 0],
            ["Weighted Snaive", "LY Weight", 0.6],
            ["Weighted Snaive", "LLY Weight", 0.25],
            ["Weighted Snaive", "LLLY Weight", 0.15],
            ["Weighted Snaive", "LLLLY Weight", 0],
            ["Growth Snaive", "LY Weight", 0.6],
            ["Growth Snaive", "LLY Weight", 0.25],
            ["Growth Snaive", "LLLY Weight", 0.15],
            ["Growth Snaive", "LLLLY Weight", 0],
            ["Weighted AOA", "Week Attention", 0],
            ["Weighted AOA", "Month Attention", 0.6],
            ["Weighted AOA", "Quarter Attention", 0.25],
            ["Weighted AOA", "Holiday Attention", 0.15],
            ["Growth AOA", "Week Attention", 0],
            ["Growth AOA", "Month Attention", 0.6],
            ["Growth AOA", "Quarter Attention", 0.25],
            ["Growth AOA", "Holiday Attention", 0.15],
            ["Growth AOA", "LY Weight", 0.6],
            ["Growth AOA", "LLY Weight", 0.25],
            ["Growth AOA", "LLLY Weight", 0.15],
            ["Growth AOA", "LLLLY Weight", 0],
        ],
        columns=[stat_algo_col, stat_parameter_col, stat_param_value_col],
    )
    return AlgoParameters_df


# TODO : Deprecate next release
def get_default_params(
    stat_algo_col,
    stat_parameter_col,
    stat_param_value_col,
    frequency,
    intersections_master,
):
    # Default frequency is monthly
    AlgoParameters_df = get_monthly_default_params(
        stat_algo_col=stat_algo_col,
        stat_parameter_col=stat_parameter_col,
        stat_param_value_col=stat_param_value_col,
    )
    if frequency == "Weekly":
        AlgoParameters_df = get_weekly_default_params(
            stat_algo_col=stat_algo_col,
            stat_parameter_col=stat_parameter_col,
            stat_param_value_col=stat_param_value_col,
        )

    # join algo params with intersections master
    AlgoParameters_for_all_intersections = create_cartesian_product(
        df1=AlgoParameters_df, df2=intersections_master
    )

    return AlgoParameters_for_all_intersections


def get_default_value_for_ma_periods(frequency):
    """
    Returns default parameter value for moving average periods based on frequency
    """
    if frequency == "Weekly":
        return 13
    elif frequency == "Monthly":
        return 3
    elif frequency == "Quarterly":
        return 1
    else:
        raise ValueError("Invalid frequency {}".format(frequency))


def round_dict_values(d, k):
    return {key: float(f"{value:.{k}f}") for key, value in d.items()}


def get_fitted_params(the_model_name, the_estimator):
    # initialize fitted params
    fitted_params = "No Parameters"

    try:
        if hasattr(the_estimator, "summary") and callable(getattr(the_estimator, "summary")):
            if isinstance(the_estimator.summary(), sm.iolib.summary.Summary):
                # initialize string
                fitted_params = ""

                # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
                results_as_html = the_estimator.summary().tables[0].as_html()

                # convert statsmodels summary object to df
                summary_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

                if "y" in summary_df.columns:
                    summary_dict = summary_df["y"].to_dict()

                    if "Model:" in summary_dict:
                        fitted_params = fitted_params + f"Model : {summary_dict['Model:']}, "

                if hasattr(the_estimator, "error"):
                    fitted_params = fitted_params + f" Error : {the_estimator.error}, "

                if hasattr(the_estimator, "trend"):
                    fitted_params = fitted_params + f" Trend : {the_estimator.trend}, "

                if hasattr(the_estimator, "seasonal"):
                    fitted_params = fitted_params + f" Seasonality : {the_estimator.seasonal}, "

                # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
                results_as_html = the_estimator.summary().tables[1].as_html()

                # convert statsmodels summary object to df
                summary_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

                if "coef" in summary_df.columns:
                    summary_dict = summary_df["coef"].to_dict()

                    relevant_keys = [
                        "smoothing_level",
                        "smoothing_trend",
                        "smoothing_seasonal",
                        "damping_trend",
                    ]
                    relevant_dict = {
                        key: summary_dict[key]
                        for key in relevant_keys
                        if key in summary_dict.keys()
                    }

                    if relevant_dict:
                        fitted_params = fitted_params + str(round_dict_values(d=relevant_dict, k=2))

        elif the_model_name == "Moving Average":
            fitted_params = "Period = {}".format(int(the_estimator))

        elif the_model_name in ["SES", "DES", "TES", "ETS"]:
            param_to_value_mapping = get_param_to_value_mapping(the_estimator)
            # exponential smoothing family
            # get param, assign default value of 0, strip leading and trailing spaces from string
            alpha = param_to_value_mapping.get("smoothing_level", "NA").strip()
            beta = param_to_value_mapping.get("smoothing_trend", "NA").strip()
            gamma = param_to_value_mapping.get("smoothing_seasonal", "NA").strip()
            phi = param_to_value_mapping.get("damping_trend", "NA").strip()

            fitted_params = "Alpha = {}, Beta = {}, Gamma = {}, Phi = {}".format(
                alpha, beta, gamma, phi
            )
        elif the_model_name == "Prophet":
            daily_seasonality = the_estimator.daily_seasonality
            weekly_seasonality = the_estimator.weekly_seasonality
            yearly_seasonality = the_estimator.yearly_seasonality
            mcmc_samples = the_estimator.mcmc_samples
            uncertainty_samples = the_estimator.uncertainty_samples
            growth = the_estimator.growth

            fitted_params = "daily_seasonality = {}, weekly_seasonality = {}, yearly_seasonality = {}, mcmc_samples = {}, uncertainty_samples = {}, growth = {}".format(
                daily_seasonality,
                weekly_seasonality,
                yearly_seasonality,
                mcmc_samples,
                uncertainty_samples,
                growth,
            )

        # elif the_model_name in ["Auto ARIMA"]:
        #     summary_string = str(the_estimator.summary())
        #     param = re.findall(
        #         "SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)",
        #         summary_string,
        #     )
        #     if len(param) > 0:
        #         p, d, q = (
        #             int(param[0][0]),
        #             int(param[0][1]),
        #             int(param[0][2]),
        #         )
        #         fitted_params = "p = {}, d = {}, q = {}".format(p, d, q)
        #         if len(param) > 1:
        #             P, D, Q = (
        #                 int(param[1][0]),
        #                 int(param[1][1]),
        #                 int(param[1][2]),
        #             )
        #             fitted_params = "p = {}, d = {}, q = {}, P = {}, D = {}, Q = {}".format(
        #                 p, d, q, P, D, Q
        #             )

        elif the_model_name == "Croston":
            fitted_params = f"Smoothing = {the_estimator.smoothing}"
        elif the_model_name == "TBATS":
            fitted_params = (
                f"sp = {the_estimator.sp}, use_arma_errors = {the_estimator.use_arma_errors}, "
                f"use_box_cox = {the_estimator.use_box_cox}, use_damped_trend = {the_estimator.use_damped_trend}, "
                f"use_trend = {the_estimator.use_trend}"
            )
        # elif the_model_name == "sARIMA":
        #     # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
        #     results_as_html = the_estimator.summary().tables[1].as_html()

        #     # convert statsmodels summary object to df
        #     summary_df = pd.read_html(results_as_html, header=0, index_col=0)[
        #         0
        #     ]

        #     if "intercept" in summary_df.index:
        #         # locate intercept value
        #         intercept_value = round(summary_df.loc["intercept", "coef"], 2)
        #     else:
        #         intercept_value = None

        #     # get order - convert to int
        #     (p, d, q) = (int(x) for x in the_estimator.get_params()["order"])

        #     # seasonal order - convert to int
        #     (P, D, Q, m) = (
        #         int(x) for x in the_estimator.get_params()["seasonal_order"]
        #     )

        #     # create fitted params string
        #     fitted_params = f"p = {p}, d = {d}, q = {q}, P = {P}, D = {D}, Q = {Q}, m = {m}, intercept = {intercept_value}"
        else:
            # store string representation of estimator
            fitted_params = str(the_estimator)

        # trim trailing space and comma if present
        if fitted_params != "" and fitted_params[-2:] == ", ":
            fitted_params = fitted_params[: len(fitted_params) - 2]

    except Exception as e:
        logger.exception(
            "Exception {} while trying to fetch fitted model params for {}".format(
                e, the_model_name
            )
        )

    return fitted_params


def get_param_to_value_mapping(the_estimator):
    # re.findall('smoothing_level     \d+.\d+', summary_string)
    coefficients_df = pd.DataFrame(the_estimator.summary().tables[1])
    coefficients_df = coefficients_df[coefficients_df.columns[:2]]
    coefficients_df.columns = ["param", "value"]
    coefficients_df["param"] = coefficients_df["param"].astype("str")
    coefficients_df["value"] = coefficients_df["value"].astype("str")
    # create dictionary for easier lookup and default value assignment
    param_to_value_mapping = dict(
        zip(
            list(coefficients_df["param"]),
            list(coefficients_df["value"]),
        )
    )
    return param_to_value_mapping


def get_default_algo_params(
    stat_algo_col,
    stat_parameter_col,
    system_stat_param_value_col,
    frequency,
    intersections_master,
    DefaultAlgoParameters,
    quarterly_default_col="Stat Parameter.[Stat Parameter Quarterly Default]",
    monthly_default_col="Stat Parameter.[Stat Parameter Monthly Default]",
    weekly_default_col="Stat Parameter.[Stat Parameter Weekly Default]",
) -> pd.DataFrame:
    assert quarterly_default_col in DefaultAlgoParameters.columns
    assert monthly_default_col in DefaultAlgoParameters.columns
    assert weekly_default_col in DefaultAlgoParameters.columns

    # default frequency in monthly
    cols_to_select = [stat_algo_col, stat_parameter_col, monthly_default_col]
    col_to_rename = monthly_default_col

    if frequency == "Weekly":
        cols_to_select = [
            stat_algo_col,
            stat_parameter_col,
            weekly_default_col,
        ]
        col_to_rename = weekly_default_col

    elif frequency == "Quarterly":
        cols_to_select = [
            stat_algo_col,
            stat_parameter_col,
            quarterly_default_col,
        ]
        col_to_rename = quarterly_default_col

    AlgoParameters_df = DefaultAlgoParameters[cols_to_select].drop_duplicates()
    AlgoParameters_df.rename(columns={col_to_rename: system_stat_param_value_col}, inplace=True)

    # join algo params with intersections master
    AlgoParameters_for_all_intersections = create_cartesian_product(
        df1=AlgoParameters_df, df2=intersections_master
    )
    return AlgoParameters_for_all_intersections


if __name__ == "__main__":
    # all_intersections = pd.DataFrame(
    #     {'item': ['item1', 'item1', 'item2', 'item2'], 'store': ['store1', 'store2', 'store1', 'store2']})
    all_intersections = pd.DataFrame({"item": ["item1"], "store": ["store1"]})
    stat_parameter_col = "Stat Parameter.[Stat Parameter]"
    stat_algo_col = "Stat Algorithm.[Stat Algorithm]"
    system_stat_param_value_col = "System Stat Parameter Value"

    temp = get_default_params(
        stat_algo_col=stat_algo_col,
        stat_parameter_col=stat_parameter_col,
        stat_param_value_col=system_stat_param_value_col,
        frequency="Monthly",
        intersections_master=all_intersections,
    )

    print("here")

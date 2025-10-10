"""
Version : 0.0.0
Maintained by : dpref@o9solutions.com
"""

import logging

import pandas as pd

logger = logging.getLogger("o9_logger")


class AlgoParamExtractor:
    def __init__(
        self,
        forecast_level,
        intersection,
        AlgoParams,
        stat_algo_col,
        stat_parameter_col,
        system_stat_param_value_col,
    ):
        self.the_forecast_level = forecast_level
        self.the_intersection = intersection
        self.AlgoParams = AlgoParams
        self.stat_algo_col = stat_algo_col
        self.stat_parameter_col = stat_parameter_col
        self.system_stat_param_value_col = system_stat_param_value_col

        logger.debug("Resetting index for AlgoParams ...")

        # reset and drop index for boolean filtering to work
        self.AlgoParams.reset_index(drop=True, inplace=True)

        # create dummy filter clause with all True
        self.filter_clause = pd.Series([True] * len(self.AlgoParams))

        # Combine elements in tuple into the filter clause to filter for the right intersection
        for the_index, the_level in enumerate(forecast_level):
            self.filter_clause = self.filter_clause & (
                self.AlgoParams[the_level] == self.the_intersection[the_index]
            )

        # filter relevant data for the intersection from AlgoParams
        self.the_intersection_df = self.AlgoParams[self.filter_clause]

    def extract_param_value(
        self,
        algorithm: str,
        parameter: str,
    ) -> object:
        # Add filter clause with algorithm and parameter filter
        filter_clause = (self.the_intersection_df[self.stat_algo_col] == algorithm) & (
            self.the_intersection_df[self.stat_parameter_col] == parameter
        )

        param_value = None
        if len(self.the_intersection_df[filter_clause]) > 0:

            if parameter == "Growth Type":
                param_value = str(
                    self.the_intersection_df[filter_clause][self.system_stat_param_value_col].iloc[
                        0
                    ]
                )
            else:
                param_value = float(
                    self.the_intersection_df[filter_clause][self.system_stat_param_value_col].iloc[
                        0
                    ]
                )
                # round parameter to 2 decimal places since platform adds many decimal places to float
                param_value = round(param_value, 2)
                logger.debug(
                    "the_intersection : {}, algorithm : {}, parameter : {}, param_value : {}".format(
                        self.the_intersection,
                        algorithm,
                        parameter,
                        param_value,
                    )
                )
        else:
            logger.debug(
                "Parameter Value {} not found in AlgoParameters for {}".format(
                    parameter,
                    algorithm,
                )
            )
        return param_value

    def extract_snaive_params(self, model_name: str):
        result = list()
        try:
            weight_1 = self.extract_param_value(model_name, "LY Weight")
            weight_2 = self.extract_param_value(model_name, "LLY Weight")
            weight_3 = self.extract_param_value(model_name, "LLLY Weight")
            weight_4 = self.extract_param_value(model_name, "LLLLY Weight")

            result = [weight_1, weight_2, weight_3, weight_4]

            logger.debug("weights : {}".format(result))
        except Exception as e:
            logger.exception(e)
        return result

    def extract_attention_weights(self, model_name: str, seasonal_periods: int):
        req_attention_weights = list()
        try:
            if seasonal_periods in [52, 53]:
                week_attention = self.extract_param_value(model_name, "Week Attention")
                month_attention = self.extract_param_value(model_name, "Month Attention")
                holiday_attention = self.extract_param_value(model_name, "Holiday Attention")

                req_attention_weights = [
                    week_attention,
                    month_attention,
                    holiday_attention,
                ]
            elif seasonal_periods == 12:
                month_attention = self.extract_param_value(model_name, "Month Attention")
                holiday_attention = self.extract_param_value(model_name, "Holiday Attention")

                req_attention_weights = [month_attention, holiday_attention]
            elif seasonal_periods == 4:
                quarter_attention = self.extract_param_value(model_name, "Quarter Attention")
                holiday_attention = self.extract_param_value(model_name, "Holiday Attention")

                req_attention_weights = [quarter_attention, holiday_attention]
            else:
                raise ValueError("Invalid value {} for seasonal periods".format(seasonal_periods))
            logger.debug("req_attention_weights : {}".format(req_attention_weights))
        except Exception as e:
            logger.exception(e)
        return req_attention_weights

    def extract_g_woa_params(self, model_name):
        trend_weights = list()
        try:
            weight_1 = self.extract_param_value(model_name, "LY Weight")
            weight_2 = self.extract_param_value(model_name, "LLY Weight")
            weight_3 = self.extract_param_value(model_name, "LLLY Weight")
            weight_4 = self.extract_param_value(model_name, "LLLLY Weight")

            trend_weights = [weight_1, weight_2, weight_3, weight_4]
            logger.debug("trend_weights : {}".format(trend_weights))
        except Exception as e:
            logger.exception(e)
        return trend_weights

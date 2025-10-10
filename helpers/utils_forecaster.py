import logging

from statsforecast.models import (  # WindowAverage,
    MSTL,
    TBATS,
    AutoARIMA,
    AutoETS,
    AutoTheta,
    CrostonClassic,
    Holt,
    HoltWinters,
    SeasonalNaive,
    SimpleExponentialSmoothing,
)

from helpers.Custom_Model import (
    GrowthAOA,
    GrowthSNaive,
    LGBMSeasonalLag,
    NaiveRandomWalk,
    RollingWindowAverage,
    SimpleAOA,
    SimpleSNaive,
    WeightedAOA,
    WeightedSNaive,
)

logger = logging.getLogger("o9_logger")


def _get_param(param_dict, model_key, param_key, default_value):
    # Log and fall back to default when missing
    try:
        model_params = param_dict.get(model_key) if isinstance(param_dict, dict) else None
        if not isinstance(model_params, dict):
            logger.warning(
                f"Parameters for model '{model_key}' are missing. Using default for '{param_key}': {default_value}"
            )
            return default_value

        value = model_params.get(param_key)
        if value is None:
            logger.warning(
                f"Parameter '{param_key}' for model '{model_key}' is missing. Using default: {default_value}"
            )
            return default_value

        return value
    except Exception:
        logger.exception(
            f"Error retrieving parameter '{param_key}' for model '{model_key}'. Using default: {default_value}"
        )
        return default_value


def build_model_map(param_dict, seasonal_periods):
    # Returns the model_map for StatsForecast, using param_dict and seasonal_periods
    return {
        "STLF": MSTL(season_length=seasonal_periods, alias="Stat Fcst STLF"),
        "TBATS": TBATS(season_length=seasonal_periods, alias="Stat Fcst TBATS"),
        "sARIMA": AutoARIMA(
            D=int(_get_param(param_dict, "sARIMA", "Differencing", 1.0)),
            alias="Stat Fcst sARIMA",
        ),
        "Auto ARIMA": AutoARIMA(
            D=int(_get_param(param_dict, "sARIMA", "Differencing", 1.0)),
            alias="Stat Fcst Auto ARIMA",
        ),
        "AutoETS": AutoETS(season_length=seasonal_periods, alias="Stat Fcst Auto ETS"),
        # "Moving Average": WindowAverage(
        #     window_size=int(param_dict.get("Moving Average").get("Period")),
        #     alias="Stat Fcst Moving Average",
        # ),
        "Moving Average": RollingWindowAverage(
            window_size=int(
                _get_param(
                    param_dict,
                    "Moving Average",
                    "Period",
                    seasonal_periods,
                )
            ),
            alias="Stat Fcst Moving Average",
        ),
        "DES": Holt(alias="Stat Fcst DES"),
        "TES": HoltWinters(alias="Stat Fcst TES", season_length=seasonal_periods),
        "SES": SimpleExponentialSmoothing(
            alpha=_get_param(param_dict, "SES", "Alpha Upper", 0.075),
            alias="Stat Fcst SES",
        ),
        "Croston": CrostonClassic(
            alias="Stat Fcst Croston",
        ),
        "Seasonal Naive YoY": SeasonalNaive(
            season_length=seasonal_periods, alias="Stat Fcst Seasonal Naive YoY"
        ),
        "Naive Random Walk": NaiveRandomWalk(alias="Stat Fcst Naive Random Walk"),
        "Growth Snaive": GrowthSNaive(
            season_length=seasonal_periods,
            weights=[
                _get_param(param_dict, "Growth Snaive", "LY Weight", 0.6),
                _get_param(param_dict, "Growth Snaive", "LLY Weight", 0.25),
                _get_param(param_dict, "Growth Snaive", "LLLY Weight", 0.15),
                _get_param(param_dict, "Growth Snaive", "LLLLY Weight", 0.0),
            ],
            alias="Stat Fcst Growth Snaive",
        ),
        "Weighted Snaive": WeightedSNaive(
            season_length=seasonal_periods,
            weights=[
                _get_param(param_dict, "Weighted Snaive", "LY Weight", 0.6),
                _get_param(param_dict, "Weighted Snaive", "LLY Weight", 0.25),
                _get_param(param_dict, "Weighted Snaive", "LLLY Weight", 0.15),
                _get_param(param_dict, "Weighted Snaive", "LLLLY Weight", 0.0),
            ],
            weighted=True,
            alias="Stat Fcst Weighted Snaive",
        ),
        "Simple Snaive": SimpleSNaive(
            season_length=seasonal_periods, alias="Stat Fcst Simple Snaive"
        ),
        "Weighted AOA": WeightedAOA(
            season_length=seasonal_periods,
            attention_weights=[
                _get_param(param_dict, "Weighted AOA", "Week Attention", 0.6),
                _get_param(param_dict, "Weighted AOA", "Month Attention", 0.25),
                _get_param(param_dict, "Weighted AOA", "Quarter Attention", 0.0),
                _get_param(param_dict, "Weighted AOA", "Holiday Attention", 0.15),
            ],
            alias="Stat Fcst Weighted AOA",
        ),
        "Simple AOA": SimpleAOA(season_length=seasonal_periods, alias="Stat Fcst Simple AOA"),
        "Growth AOA": GrowthAOA(
            season_length=seasonal_periods,
            attention_weights=[
                _get_param(param_dict, "Growth AOA", "Week Attention", 0.6),
                _get_param(param_dict, "Growth AOA", "Month Attention", 0.25),
                _get_param(param_dict, "Growth AOA", "Quarter Attention", 0.0),
                _get_param(param_dict, "Growth AOA", "Holiday Attention", 0.15),
            ],
            growth_weights=[
                _get_param(param_dict, "Growth AOA", "LY Weight", 0.6),
                _get_param(param_dict, "Growth AOA", "LLY Weight", 0.25),
                _get_param(param_dict, "Growth AOA", "LLLY Weight", 0.15),
                _get_param(param_dict, "Growth AOA", "LLLLY Weight", 0.0),
            ],
            alias="Stat Fcst Growth AOA",
        ),
        "Theta": AutoTheta(
            season_length=seasonal_periods, decomposition_type="additive", alias="Stat Fcst Theta"
        ),
        "AR-NNET": LGBMSeasonalLag(season_length=seasonal_periods, alias="Stat Fcst AR-NNET"),
        "ETS": AutoETS(season_length=seasonal_periods, alias="Stat Fcst ETS"),
    }

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
from scipy.stats import norm
from statsforecast.models import _TS, WindowAverage, _ensure_float


class WeightedSNaive(_TS):
    """
    Weighted or Simple Seasonal Naive model with custom alias support and no pandas/polars dependencies.

    Parameters
    ----------
    seasonal_periods : int
        Number of periods per cycle (e.g., 12 for monthly data).
    weights : list[float]
        Weights for each past season (must sum to 1). Only used if weighted=True.
    weighted : bool, default True
        If False, uses simple SNaive (all cycles weighted equally).
    in_sample : bool, default False
        If True, supports in-sample forecasts when `predict(..., fitted=True)`.
    alias : str, optional
        Custom alias for this model in forecast outputs. Defaults to class name.
    """

    def __init__(
        self,
        season_length: int,
        weights: list[float] = None,
        weighted: bool = True,
        in_sample: bool = False,
        alias: str = None,
    ):
        super().__init__()
        self.season_length = season_length
        self.weights = weights if weights is not None else []
        self.weighted = weighted
        self.in_sample_flag = in_sample
        self.uses_exog = False
        self.alias = alias

    def fit(self, y: np.ndarray, X: np.ndarray = None) -> None:
        self._y = y.copy()
        self._resid = np.diff(y) if len(y) > 1 else np.array([0.0])

    def predict(
        self,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        series = self._y.copy()
        sp = self.season_length
        if len(series) < sp:
            series = np.concatenate([np.full(sp - len(series), np.nan), series])
        n = len(series)
        # Number of available cycles
        cycles = n // sp
        needed = cycles * sp
        if n < needed:
            series = np.concatenate([np.full(needed - n, np.nan), series])
        arr = series[-needed:]
        mat = arr.reshape(cycles, sp)
        mask = ~np.isnan(mat)
        mat_zero = np.where(mask, mat, 0.0)

        # --- Weighted or Simple SNaive logic ---
        if self.weighted and self.weights:
            w = np.array(self.weights[::-1][:cycles])
            w = w / w.sum()  # Ensure sum to 1 (safe)
        else:
            # Simple SNaive: equal weights
            w = np.ones(cycles) / cycles if cycles > 0 else np.ones(1)

        weighted = mat_zero * (w[:, None] * mask)
        denom = (w[:, None] * mask).sum(axis=0)
        denom[denom == 0] = 1
        base_fc = weighted.sum(axis=0) / denom

        # --- Output construction ---
        if fitted and self.in_sample_flag:
            reps = int(np.ceil(len(self._y) / sp))
            vals = np.tile(base_fc, reps)[: len(self._y)]
            out = {"mean": vals}
        else:
            reps = int(np.ceil(h / sp))
            vals = np.tile(base_fc, reps)[:h]
            out = {"mean": vals}

        if level is not None:
            sigma = np.std(self._resid, ddof=1)
            zscores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
            se = sigma * np.sqrt(np.arange(1, h + 1))
            for lev, z in zip(level, zscores):
                out[f"lo-{int(lev)}"] = out["mean"] - z * se
                out[f"hi-{int(lev)}"] = out["mean"] + z * se
        return out

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        self.fit(y, X)
        return self.predict(h, X, X_future, level, fitted)

    def __repr__(self) -> str:
        return self.alias or self.__class__.__name__


class SimpleSNaive(WeightedSNaive):
    """
    Simple Seasonal Naive model (special case of WeightedSNaive with equal weights).
    Uses all past seasons with equal weighting, i.e., a standard SNaive.

    Parameters
    ----------
    season_length : int
        Number of periods per cycle (e.g., 12 for monthly data).
    in_sample : bool, default False
        If True, supports in-sample forecasts when `predict(..., fitted=True)`.
    alias : str, optional
        Custom alias for this model in forecast outputs.
    """

    def __init__(self, season_length: int, in_sample: bool = False, alias: str = None):
        # Call parent with weighted=False
        super().__init__(
            season_length=season_length,
            weights=None,  # No weights needed
            weighted=False,  # Use simple SNaive logic
            in_sample=in_sample,
            alias=alias or "SimpleSNaive",
        )


class GrowthSNaive(_TS):
    def __init__(
        self,
        season_length: int,
        alias: str = "GrowthSNaive",
        weights: list[float] = None,
        max_growth: float = 1.25,
        min_growth: float = 0.75,
        fitted: bool = False,
    ):
        super().__init__()
        self.season_length = season_length
        self.alias = alias
        # Defensive: Use default weights if not provided
        if weights is None:
            weights = [0.6, 0.25, 0.15, 0]
        self.weights = np.array(weights, dtype=float)
        self.max_growth = max_growth
        self.min_growth = min_growth
        self.in_sample_flag = fitted
        self.uses_exog = False

    def _pad_time_series(self, arr, total_length, fill_value=np.nan):
        """Pads the array to the required length."""
        return (
            arr
            if arr.size >= total_length
            else np.concatenate((arr, np.full(total_length - arr.size, fill_value)))
        )

    def _compute_growth_rate(self, input_series, insample_size):
        weights = self.weights
        window_size = self.season_length // 4

        insample_data = input_series[:insample_size]
        if insample_data.size < window_size:
            return 1.0  # Not enough data, no growth

        # Recent median for the most recent window
        recent_window = insample_data[-window_size:]
        recent_median = np.median(recent_window)

        # Compute seasonal medians across the insample data
        seasonal_data = insample_data
        incomplete = len(seasonal_data) % window_size
        if incomplete > 0:
            seasonal_data = self._pad_time_series(
                seasonal_data, len(seasonal_data) + (window_size - incomplete)
            )
        median_matrix = seasonal_data.reshape(-1, window_size)
        seasonal_medians = np.median(median_matrix, axis=1)
        seasonal_medians = seasonal_medians[~np.isnan(seasonal_medians)]

        # Weighted average of medians
        num_medians = min(len(seasonal_medians), len(weights))
        used_medians = seasonal_medians[:num_medians]
        used_weights = weights[:num_medians]
        if used_weights.sum() == 0 or used_medians.size == 0:
            return 1.0
        used_weights = used_weights / used_weights.sum()
        weighted_median = used_medians.dot(used_weights) + 1e-10

        growth_rate = recent_median / weighted_median
        # Clamp to reasonable growth bounds
        return np.clip(growth_rate, self.min_growth, self.max_growth)

    def _forecast(self, input_series, forecast_horizon, insample_size):
        if input_series.size < self.season_length:
            # Pad at the start, so the "recent season" is always available
            fill_value = np.nanmedian(input_series) if input_series.size > 0 else 0.0
            padded = np.full(self.season_length, fill_value)
            padded[-input_series.size :] = input_series
            recent_season = padded
        else:
            recent_season = input_series[insample_size - self.season_length : insample_size]

        growth_rate = self._compute_growth_rate(input_series, insample_size)
        num_repeats = int(np.ceil(forecast_horizon / self.season_length))
        repeated = np.tile(recent_season, num_repeats)[:forecast_horizon]
        return repeated * growth_rate

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        y = np.asarray(y)
        assert y.ndim == 1, "Input series y must be 1-dimensional"
        insample_size = y.size
        # Fill missing values with series median
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        point_forecast = self._forecast(y, h, insample_size)
        output = {"mean": point_forecast}
        if level:
            diffs = np.diff(y)
            residual_std = np.std(diffs, ddof=1) if diffs.size > 0 else 0
            z_scores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
            forecast_stderr = residual_std * np.sqrt(np.arange(1, h + 1))
            for confidence_level, z in zip(level, z_scores):
                output[f"lo-{int(confidence_level)}"] = point_forecast - z * forecast_stderr
                output[f"hi-{int(confidence_level)}"] = point_forecast + z * forecast_stderr
        return output

    def __repr__(self):
        return self.alias


class WeightedAOA(_TS):
    """
    Weighted Average of Averages (AOA) model, NumPy-only, matching pandas plugin logic.
    """

    def __init__(
        self,
        season_length: int,
        attention_weights: list[float],
        alias: str = "WeightedAOA",
        is_weighted: bool = True,
        use_holiday: bool = False,
    ):
        super().__init__()
        self.season_length = season_length
        self.alias = alias
        self.is_weighted = is_weighted
        self.use_holiday = use_holiday

        if attention_weights is None:
            raise ValueError("attention_weights must be specified explicitly.")
        self.attention_weights = np.array(attention_weights, dtype=float)
        s = self.attention_weights.sum()
        if not np.isclose(s, 1.0):
            self.attention_weights = self.attention_weights / s

    def _pad_time_series(self, arr, total_length, fill_value=np.nan):
        arr = np.asarray(arr)
        if arr.size >= total_length:
            return arr
        return np.concatenate([arr, np.full(total_length - arr.size, fill_value)])

    def _make_time_idxs(self, n, h=0):
        """Generate time index arrays with names."""
        n_total = n + h
        idxs = []
        names = []

        if self.season_length >= 52:
            idxs.append((np.arange(n_total) % 52) + 1)
            names.append("week")
        if self.season_length in [12, 52, 53]:
            idxs.append((np.arange(n_total) % 12) + 1)
            names.append("month")
        if self.season_length == 4:
            idxs.append((np.arange(n_total) % 4) + 1)
            names.append("quarter")
        if self.use_holiday:
            idxs.append(((np.arange(n_total) % 7) == 0).astype(int) + 1)
            names.append("holiday")

        return names, idxs

    @staticmethod
    def attention(x, time_idx):
        x = np.asarray(x)
        time_idx = np.asarray(time_idx)
        max_period = np.max(time_idx)
        period_sums = np.zeros(max_period)
        period_counts = np.zeros(max_period)
        for i, period in enumerate(time_idx):
            idx = period - 1
            if not np.isnan(x[i]):
                period_sums[idx] += x[i]
                period_counts[idx] += 1
        attention_values = np.full(max_period, np.nan)
        mask = period_counts > 0
        attention_values[mask] = period_sums[mask] / period_counts[mask]
        result = np.array([attention_values[period - 1] for period in time_idx])
        return result

    @staticmethod
    def compute_attentions(x, time_idxs):
        return np.array([WeightedAOA.attention(x, idx) for idx in time_idxs])

    def _forecast_core(self, x, h, time_info):
        x = np.asarray(x)
        # TODO: Adding this as temp fix for the Validation Forecast
        # Fallback to RollingWindowAverage if not enough data
        if len(x) < self.season_length:
            rwm = RollingWindowAverage(window_size=len(x))
            return rwm.forecast(x, h)["mean"]
        if len(x) < self.season_length:
            x = self._pad_time_series(x, self.season_length)
        x = self._pad_time_series(x, len(x) + h, np.nan)

        names, idxs = time_info

        # compute attentions for all active indices
        attentions_aligned = [self.attention(x, idx) for idx in idxs]
        attentions_aligned = np.array(attentions_aligned)

        if self.is_weighted:
            weights = []
            for name in names:
                if name == "week":
                    weights.append(self.attention_weights[0])
                elif name == "month":
                    weights.append(self.attention_weights[1])
                elif name == "quarter":
                    weights.append(self.attention_weights[2])
                elif name == "holiday":
                    weights.append(self.attention_weights[3])

            weights = np.array(weights, dtype=float)
            if weights.sum() == 0:  # fallback
                forecast = np.nanmean(attentions_aligned, axis=0)
            else:
                weights = weights / weights.sum()
                forecast = np.dot(weights, attentions_aligned)

            # --- Holiday fallback logic ---
            if self.use_holiday and "holiday" in names:
                holiday_idx = names.index("holiday")
                holiday_vals = attentions_aligned[holiday_idx]
                mask = np.isnan(holiday_vals)
                if self.season_length in [52, 53] and "week" in names and "month" in names:
                    week_idx = names.index("week")
                    month_idx = names.index("month")
                    forecast[mask] = (
                        0.75 * attentions_aligned[week_idx][mask]
                        + 0.25 * attentions_aligned[month_idx][mask]
                    )
                elif self.season_length == 12 and "month" in names:
                    month_idx = names.index("month")
                    forecast[mask] = attentions_aligned[month_idx][mask]
                elif self.season_length == 4 and "quarter" in names:
                    quarter_idx = names.index("quarter")
                    forecast[mask] = attentions_aligned[quarter_idx][mask]

        else:
            # simple (equal mean) attention
            forecast = np.nanmean(attentions_aligned, axis=0)

        return forecast[-h:]

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("Input series y must be 1-dimensional")
        n = y.size
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        time_idxs = self._make_time_idxs(n, h)
        point_forecast = self._forecast_core(y, h, time_idxs)
        out = {"mean": point_forecast}
        if level:
            diffs = np.diff(y)
            residual_std = np.std(diffs, ddof=1) if diffs.size > 0 else 0
            z_scores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
            forecast_stderr = residual_std * np.sqrt(np.arange(1, h + 1))
            for confidence_level, z in zip(level, z_scores):
                out[f"lo-{int(confidence_level)}"] = point_forecast - z * forecast_stderr
                out[f"hi-{int(confidence_level)}"] = point_forecast + z * forecast_stderr
        return out

    def __repr__(self):
        return self.alias


class SimpleAOA(WeightedAOA):
    """
    Simple AOA (Average of Averages) model:
    Uses equal weights for all period-based attentions (i.e., arithmetic mean).

    Parameters
    ----------
    season_length : int
        Number of periods per cycle (e.g., 12 for monthly data).
    use_holiday : bool, default False
        If True, adds a holiday index to the time indices.
    alias : str, optional
        Custom alias for this model in forecast outputs.
    """

    def __init__(self, season_length: int, use_holiday: bool = False, alias: str = "SimpleAOA"):
        # Call parent with is_weighted=False and dummy weights
        # Dummy weights: any list of correct length, will be ignored since is_weighted=False
        # Here we provide [1.0] as a placeholder, it's not used
        super().__init__(
            season_length=season_length,
            attention_weights=[1.0],
            alias=alias,
            is_weighted=False,
            use_holiday=use_holiday,
        )


class GrowthAOA(_TS):
    """
    Growth-AOA: Attention Over Attention with growth adjustment.
    Uses attention-weighted seasonal averaging and applies a growth multiplier.
    """

    def __init__(
        self,
        season_length: int,
        attention_weights: list[float] = None,
        growth_weights: list[float] = None,
        max_growth: float = 1.2,
        min_growth: float = 0.8,
        alias: str = "GrowthAOA",
        fitted: bool = False,
    ):
        super().__init__()
        self.season_length = season_length
        self.alias = alias
        # Defensive: Use default attention weights if not provided
        if attention_weights is None:
            # e.g., focus mostly on last 3-4 years/seasons
            attention_weights = [0.5, 0.3, 0.2, 0.0]
        self.attention_weights = np.array(attention_weights, dtype=float)
        self.attention_weights = self.attention_weights / self.attention_weights.sum()
        if growth_weights is None:
            growth_weights = [0.6, 0.25, 0.15, 0.0]
        self.growth_weights = np.array(growth_weights, dtype=float)
        self.growth_weights = self.growth_weights / self.growth_weights.sum()
        self.max_growth = max_growth
        self.min_growth = min_growth
        self.in_sample_flag = fitted
        self.uses_exog = False

    def _pad_time_series(self, arr, total_length, fill_value=np.nan):
        arr = np.asarray(arr)
        if arr.size >= total_length:
            return arr
        return np.concatenate([arr, np.full(total_length - arr.size, fill_value)])

    def _compute_attention_average(self, input_series, insample_size):
        """
        Computes the attention-weighted average for the last available period.
        """
        period = self.season_length
        series = input_series[:insample_size]
        # Number of full seasons available
        n_seasons = series.size // period
        n_weights = len(self.attention_weights)
        use_n = min(n_seasons, n_weights)
        if use_n == 0:
            # fallback: mean of whatever is available
            return np.nanmean(series[-period:])

        attention_weights = self.attention_weights[:use_n]
        attention_weights = attention_weights / attention_weights.sum()
        matrix = series[-use_n * period :].reshape(use_n, period)
        # Each row is a season, last row = most recent season
        # Weighted mean across seasons for each period position
        weighted = np.average(matrix, axis=0, weights=attention_weights)
        return weighted

    def _compute_growth_rate(self, input_series, insample_size):
        """
        Computes a robust growth multiplier using weighted seasonal medians.
        """
        period = self.season_length
        series = input_series[:insample_size]
        window_size = period // 4 or 1

        if series.size < window_size:
            return 1.0

        # Recent window median
        recent_window = series[-window_size:]
        recent_median = np.median(recent_window)

        # Build seasonal medians
        complete = series.size - (series.size % window_size)
        seasonal_data = series[:complete]
        if seasonal_data.size == 0:
            return 1.0

        matrix = seasonal_data.reshape(-1, window_size)
        medians = np.median(matrix, axis=1)
        medians = medians[~np.isnan(medians)]
        n_weights = len(self.growth_weights)
        use_n = min(len(medians), n_weights)
        if use_n == 0 or np.sum(self.growth_weights[:use_n]) == 0:
            return 1.0
        weights = self.growth_weights[:use_n]
        weights = weights / weights.sum()
        weighted_median = np.dot(medians[:use_n], weights) + 1e-10
        growth = recent_median / weighted_median
        return np.clip(growth, self.min_growth, self.max_growth)

    def _forecast(self, input_series, forecast_horizon, insample_size):
        """
        Core logic: repeats attention-weighted season with growth.
        """
        # Fill with median if needed
        y = np.asarray(input_series)
        if y.size < self.season_length:
            fill_value = np.nanmedian(y) if y.size > 0 else 0.0
            padded = np.full(self.season_length, fill_value)
            if y.size > 0:
                padded[-y.size :] = y
            y = padded
            insample_size = y.size

        attention_season = self._compute_attention_average(y, insample_size)
        growth = self._compute_growth_rate(y, insample_size)
        # Repeat for forecast horizon
        repeats = int(np.ceil(forecast_horizon / self.season_length))
        repeated = np.tile(attention_season, repeats)[:forecast_horizon]
        return repeated * growth

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        y = np.asarray(y)
        insample_size = y.size
        # Fill NaNs with series median
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        point_forecast = self._forecast(y, h, insample_size)
        out = {"mean": point_forecast}
        if level:
            diffs = np.diff(y)
            std = np.std(diffs, ddof=1) if diffs.size > 0 else 0
            z_scores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
            forecast_stderr = std * np.sqrt(np.arange(1, h + 1))
            for confidence_level, z in zip(level, z_scores):
                out[f"lo-{int(confidence_level)}"] = point_forecast - z * forecast_stderr
                out[f"hi-{int(confidence_level)}"] = point_forecast + z * forecast_stderr
        return out

    def __repr__(self):
        return self.alias


class NaiveRandomWalk(_TS):
    """
    Naive Random Walk: forecasts the last observed value for all future periods.
    """

    def __init__(
        self,
        alias: str = "NaiveRandomWalk",
        fitted: bool = False,
    ):
        super().__init__()
        self.alias = alias
        self.in_sample_flag = fitted
        self.uses_exog = False

    def _forecast(self, y: np.ndarray, h: int, insample_size: int):
        last_value = y[insample_size - 1] if insample_size > 0 else 0.0
        # In-sample or out-of-sample logic:
        if self.in_sample_flag:
            return np.full(insample_size, last_value)
        else:
            return np.full(h, last_value)

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        y = np.asarray(y)
        insample_size = y.size
        # Fill NaNs with series median for robustness
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmedian(y))
        point_forecast = self._forecast(y, h, insample_size)
        out = {"mean": point_forecast}

        # Add prediction intervals, if requested
        if level:
            diffs = np.diff(y)
            std = np.std(diffs, ddof=1) if diffs.size > 0 else 0
            z_scores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
            forecast_stderr = std * np.sqrt(np.arange(1, len(point_forecast) + 1))
            for confidence_level, z in zip(level, z_scores):
                out[f"lo-{int(confidence_level)}"] = point_forecast - z * forecast_stderr
                out[f"hi-{int(confidence_level)}"] = point_forecast + z * forecast_stderr

        return out

    def __repr__(self):
        return self.alias


class LGBMSeasonalLag(_TS):
    """
    LightGBM-based Univariate Seasonal-Lag Model (NIXTLA-style, NumPy only).

    Parameters
    ----------
    season_length : int
        Number of lag features (seasonal period).
    lgbm_params : dict, optional
        Extra LightGBM parameters.
    alias : str, optional
        Custom name for model reporting.
    fitted : bool, default False
        If True, enables returning in-sample predictions.
    """

    def __init__(
        self, season_length: int, lgbm_params: dict = None, alias: str = None, fitted: bool = False
    ):
        super().__init__()
        self.season_length = season_length
        self.lgbm_params = lgbm_params if lgbm_params else {}
        self.alias = alias
        self.in_sample_flag = fitted
        self.uses_exog = False

    def fit(self, y: np.ndarray, X: np.ndarray = None) -> None:
        try:
            y = np.asarray(y)
            self._y = y.copy()
            n = len(y)
            if n <= self.season_length:
                raise ValueError(f"Series too short for {self.season_length} lags.")
            # Lag features
            X_train = np.lib.stride_tricks.sliding_window_view(y, self.season_length)[:-1]
            y_train = y[self.season_length :]
            self.model_ = lgb.LGBMRegressor(**self.lgbm_params)
            self.model_.fit(X_train, y_train)
            # For recursion
            self.last_values_ = y[-self.season_length :]
            # For intervals
            y_train_pred = self.model_.predict(X_train)
            self._resid = y_train - y_train_pred
        except Exception as e:
            raise ValueError(f"LGBMSeasonalLag failed to fit: {e}")

    def predict(
        self,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        try:
            # Recursive prediction
            preds = []
            last = self.last_values_.copy()
            for _ in range(h):
                x_pred = last[-self.season_length :].reshape(1, -1)
                y_pred = self.model_.predict(x_pred)[0]
                preds.append(y_pred)
                last = np.append(last, y_pred)
            out = {"mean": np.array(preds)}

            # In-sample predictions (optional)
            if fitted and self.in_sample_flag:
                X_train = np.lib.stride_tricks.sliding_window_view(self._y, self.season_length)[:-1]
                fitted_vals = self.model_.predict(X_train)
                pad = np.full(self.season_length, np.nan)
                fitted_full = np.concatenate([pad, fitted_vals])
                out["fitted"] = fitted_full

            # Prediction intervals (parametric, normality assumption)
            if level is not None:
                sigma = np.std(self._resid, ddof=1)
                zscores = norm.ppf(1 - (1 - np.array(level) / 100) / 2)
                se = sigma * np.sqrt(np.arange(1, h + 1))
                for lev, z in zip(level, zscores):
                    out[f"lo-{int(lev)}"] = out["mean"] - z * se
                    out[f"hi-{int(lev)}"] = out["mean"] + z * se

            return out
        except Exception as e:
            raise ValueError(f"LGBMSeasonalLag failed to predict: {e}")

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: np.ndarray = None,
        X_future: np.ndarray = None,
        level: list[float] = None,
        fitted: bool = False,
    ) -> dict[str, np.ndarray]:
        self.fit(y, X)
        return self.predict(h, X, X_future, level, fitted)

    def __repr__(self) -> str:
        return self.alias or self.__class__.__name__


def _nanmean(x: np.ndarray) -> float:
    # robust mean with optional NaNs
    m = x[np.isfinite(x)]
    return float(m.mean()) if m.size else np.nan


class RollingWindowAverage(WindowAverage):
    """
    Rolling window-average forecaster that *grows* its effective window
    each step until it reaches `window_size`, then stays capped.

    - If history is shorter than `window_size`,
        step-1 uses len(y),
        step-2 uses len(y)+1, ... until reaching `window_size`.
    - If history >= `window_size`, all steps use `window_size`.
    - recursive=True (always) because each prediction is appended to the window.
    """

    def __init__(
        self,
        window_size: int,
        skipna: bool = False,
        alias: str = "RollingWindowAverageGrowing",
        prediction_intervals=None,
    ):
        super().__init__(
            window_size=window_size, alias=alias, prediction_intervals=prediction_intervals
        )
        self.skipna = skipna

    def _step_mean(self, window: np.ndarray) -> float:
        return _nanmean(window) if self.skipna else float(window.mean())

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        # store a simple one-step mean for compatibility
        y = _ensure_float(y)
        if y.size == 0:
            self.model_ = {"mean": np.array([np.nan], dtype=float)}
        else:
            eff_ws = min(self.window_size, y.size)
            self.model_ = {"mean": np.array([self._step_mean(y[-eff_ws:])], dtype=float)}
        self._store_cs(y=y, X=X)
        return self

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, np.ndarray]:
        y = _ensure_float(y)
        if y.size == 0:
            res = {"mean": np.full(h, np.nan, dtype=float)}
        else:
            # start with full history; grow until hitting window_size
            window = y.astype(float).copy()
            preds = np.empty(h, dtype=float)
            for i in range(h):
                cur_ws = min(self.window_size, window.size)
                w = window[-cur_ws:]
                preds[i] = self._step_mean(w)
                window = np.append(window, preds[i])  # recursive roll
            res = {"mean": preds}

        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            return self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        raise Exception("You must pass `prediction_intervals` to compute them.")

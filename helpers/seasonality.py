import numpy as np
from o9Reference.common_utils.common_utils import get_seasonal_periods
from statsmodels.tsa.stattools import acf


def get_seasonality_length(d, skip_lags):
    out = []

    if len(d) > 1:
        if all(np.diff(d) == np.diff(d)[0]):
            out.append(max(d))
            return out
    while d:
        k = d.pop(0)
        d = [i for i in d if i % k != 0]
        out.append(k)

    # Reducing the options to avoid consecutive (upto 2) lags
    out.sort(reverse=True)

    cleaned_out = []
    for val in out:
        if len(cleaned_out) < 1:
            cleaned_out.append(val)
        else:
            if cleaned_out[-1] - val <= skip_lags:
                pass
            else:
                cleaned_out.append(val)
    cleaned_out.sort(reverse=True)
    cleaned_out = cleaned_out[:3]  # Top 3 periods only

    return cleaned_out


def ACFDetector(
    df_sub,
    measure_name,
    freq,
    skip_lags,
    diff,
    alpha,
    lower_ci_threshold,
    upper_ci_threshold,
    RequiredACFLags,
):
    ts_diff = df_sub[measure_name].values

    # Parameters
    lags = get_seasonal_periods(freq)  # weekly = 52

    for _ in range(diff):
        ts_diff = np.diff(ts_diff)

    ac, confint, qstat, qval = acf(ts_diff, nlags=lags, qstat=True, alpha=alpha)
    # get seasonality cycle length
    raw_seasonality = []
    for i, _int in enumerate(confint):
        if (
            ((_int[0] >= lower_ci_threshold) or (_int[1] >= upper_ci_threshold))
            and (i > 1)
            and (i in RequiredACFLags)
        ):
            raw_seasonality.append(i)

    seasonality = get_seasonality_length(raw_seasonality, skip_lags)
    seasonality_detected = True if len(seasonality) >= 1 else False
    seasonality = ",".join(map(str, seasonality)) if seasonality_detected else None
    return seasonality_detected, seasonality

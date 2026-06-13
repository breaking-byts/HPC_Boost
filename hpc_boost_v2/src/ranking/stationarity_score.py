from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_score(time_series: np.ndarray) -> float:
    values = np.asarray(time_series, dtype=float)

    if len(values) < 20:
        return 0.0

    centered = values - np.mean(values)

    if np.std(centered) < 1e-10:
        return 0.0

    scores = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            adf_stat, *_ = adfuller(centered, maxlag=min(len(centered) // 4, 20))
            scores.append(-float(adf_stat))
        except Exception:
            pass

        try:
            kpss_stat, *_ = kpss(centered, regression="c", nlags="auto")
            scores.append(-float(kpss_stat))
        except Exception:
            pass

    if not scores:
        return 0.0

    return float(np.mean(scores))

from __future__ import annotations

from typing import List

import numpy as np
import pymannkendall as mk

from src.ranking.spread_score import split_into_segments


def trend_score(time_series: np.ndarray) -> float:
    values = np.asarray(time_series, dtype=float)

    if len(values) < 10:
        return 0.0

    try:
        result = mk.original_test(values)
        return float(result.s)
    except Exception:
        return 0.0


def trend_score_multi_run(runs: List[np.ndarray]) -> float:
    if not runs:
        return 0.0

    s_values = np.asarray([trend_score(run) for run in runs], dtype=float)
    mean_s = float(np.mean(s_values))
    std_s = float(np.std(s_values))

    if std_s == 0:
        return 0.0

    return float(abs(mean_s) / std_s)


def trend_score_single(time_series: np.ndarray) -> float:
    values = np.asarray(time_series, dtype=float)

    if len(values) < 50:
        return abs(trend_score(values))

    return trend_score_multi_run(split_into_segments(values, segments=5))

from __future__ import annotations

from typing import List

import numpy as np


def spread_score(time_series: np.ndarray) -> float:
    values = np.asarray(time_series, dtype=float)

    if len(values) < 4:
        return 0.0

    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        return 0.0

    return float(1.0 / iqr)


def spread_score_multi_run(runs: List[np.ndarray]) -> float:
    if not runs:
        return 0.0

    scores = [spread_score(run) for run in runs]
    finite_scores = [score for score in scores if np.isfinite(score)]

    if not finite_scores:
        return 0.0

    return float(np.median(finite_scores))


def split_into_segments(time_series: np.ndarray, segments: int = 5) -> List[np.ndarray]:
    values = np.asarray(time_series, dtype=float)
    if len(values) < segments:
        return [values]

    split_points = np.array_split(values, segments)
    return [segment for segment in split_points if len(segment) > 0]


def spread_score_single(time_series: np.ndarray) -> float:
    values = np.asarray(time_series, dtype=float)

    if len(values) < 20:
        return spread_score(values)

    return spread_score_multi_run(split_into_segments(values, segments=5))

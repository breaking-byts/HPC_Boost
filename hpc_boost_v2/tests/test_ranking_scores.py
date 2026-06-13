import math

import numpy as np

from src.ranking.correlation_score import correlation_score
from src.ranking.spread_score import spread_score, spread_score_single
from src.ranking.stationarity_score import stationarity_score
from src.ranking.trend_score import trend_score, trend_score_single


def test_spread_constant_series_is_infinite():
    score = spread_score(np.ones(20))
    assert math.isinf(score)


def test_spread_wider_iqr_scores_lower():
    tight = np.array([10, 10, 11, 11, 12, 12, 13, 13], dtype=float)
    wide = np.array([0, 0, 10, 10, 100, 100, 200, 200], dtype=float)

    assert spread_score(tight) > spread_score(wide)


def test_spread_single_uses_segments():
    series = np.tile(np.array([1, 2, 3, 4, 5], dtype=float), 20)
    assert spread_score_single(series) > 0


def test_trend_too_short_series_returns_zero():
    assert trend_score(np.arange(5, dtype=float)) == 0.0


def test_trend_monotonic_series_is_nonzero():
    assert trend_score(np.arange(30, dtype=float)) != 0.0


def test_trend_single_uses_segments():
    series = np.arange(100, dtype=float)
    assert trend_score_single(series) > 0


def test_correlation_duplicate_column_scores_lower_than_independent_column():
    x = np.arange(30, dtype=float)
    duplicate = x.copy()
    alternating = np.array([0, 1] * 15, dtype=float)
    matrix = np.column_stack([x, duplicate, alternating])

    duplicate_score = correlation_score(0, matrix)
    independent_score = correlation_score(2, matrix)

    assert independent_score > duplicate_score


def test_stationarity_too_short_series_returns_zero():
    assert stationarity_score(np.arange(10, dtype=float)) == 0.0


def test_stationarity_constant_series_is_infinite():
    score = stationarity_score(np.ones(50))
    assert math.isinf(score)

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.ranking.correlation_score import correlation_score
from src.ranking.spread_score import spread_score_single
from src.ranking.stationarity_score import stationarity_score
from src.ranking.trend_score import trend_score_single


class HPCBoostRanker:
    """Combines the four HPC-Boost event quality scores into one ranking."""

    def __init__(self, weights: Optional[List[float]] = None) -> None:
        self.weights = weights if weights is not None else [0.25, 0.25, 0.25, 0.25]
        self._validate_weights()

    def _validate_weights(self) -> None:
        if len(self.weights) != 4:
            raise ValueError("weights must contain exactly four values")

        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights must be non-negative")

        total = sum(self.weights)
        if total <= 0:
            raise ValueError("at least one weight must be positive")

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        values = np.asarray(scores, dtype=float)
        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            return [1.0] * len(scores)

        finite_min = float(np.min(finite))
        finite_max = float(np.max(finite))

        if finite_max == finite_min:
            return [0.5 if np.isfinite(score) else 1.0 for score in scores]

        normalized = []
        for score in scores:
            if np.isposinf(score):
                normalized.append(1.0)
            elif np.isneginf(score):
                normalized.append(0.0)
            elif np.isnan(score):
                normalized.append(0.0)
            else:
                normalized.append(float((score - finite_min) / (finite_max - finite_min)))

        return normalized

    def rank_events(self, event_matrix: np.ndarray, event_names: List[str]) -> pd.DataFrame:
        matrix = np.asarray(event_matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("event_matrix must be a 2D array")

        num_events = matrix.shape[1]
        if len(event_names) != num_events:
            raise ValueError("event_names length must match event_matrix columns")

        raw_spread = []
        raw_trend = []
        raw_correlation = []
        raw_stationarity = []

        for event_idx in range(num_events):
            series = matrix[:, event_idx]
            raw_spread.append(spread_score_single(series))
            raw_trend.append(trend_score_single(series))
            raw_correlation.append(correlation_score(event_idx, matrix))
            raw_stationarity.append(stationarity_score(series))

        norm_spread = self._normalize(raw_spread)
        norm_trend = self._normalize(raw_trend)
        norm_correlation = self._normalize(raw_correlation)
        norm_stationarity = self._normalize(raw_stationarity)

        combined = []
        for idx in range(num_events):
            combined.append(
                self.weights[0] * norm_spread[idx]
                + self.weights[1] * norm_trend[idx]
                + self.weights[2] * norm_correlation[idx]
                + self.weights[3] * norm_stationarity[idx]
            )

        results = pd.DataFrame(
            {
                "event_name": event_names,
                "spread_raw": raw_spread,
                "trend_raw": raw_trend,
                "correlation_raw": raw_correlation,
                "stationarity_raw": raw_stationarity,
                "spread_norm": norm_spread,
                "trend_norm": norm_trend,
                "correlation_norm": norm_correlation,
                "stationarity_norm": norm_stationarity,
                "combined_score": combined,
            }
        )

        results = results.sort_values(
            ["combined_score", "event_name"],
            ascending=[False, True],
        ).reset_index(drop=True)
        results["rank"] = range(1, len(results) + 1)

        return results

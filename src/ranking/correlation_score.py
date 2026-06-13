from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def correlation_score(event_idx: int, all_events_matrix: np.ndarray) -> float:
    matrix = np.asarray(all_events_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("all_events_matrix must be a 2D array")

    if event_idx < 0 or event_idx >= matrix.shape[1]:
        raise IndexError("event_idx out of range")

    target = matrix[:, event_idx]
    
    if np.std(target) == 0:
        return 0.0
        
    correlations = []

    for other_idx in range(matrix.shape[1]):
        if other_idx == event_idx:
            continue

        other = matrix[:, other_idx]

        if np.std(target) == 0 or np.std(other) == 0:
            correlations.append(0.0)
            continue

        rho, _ = spearmanr(target, other)
        if np.isnan(rho):
            rho = 0.0

        correlations.append(abs(float(rho)))

    if not correlations:
        return 1.0

    return float(np.mean([1.0 - corr for corr in correlations]))

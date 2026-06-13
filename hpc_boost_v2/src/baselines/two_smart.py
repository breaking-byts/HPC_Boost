from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

class TwoSmartBaseline:
    """2SMaRT (DATE 2019): Global top-K events by Pearson correlation with label."""
    def __init__(self, k: int = 4):
        self.k = k
        self.selected_events: List[str] = []

    def fit(self, X_all: pd.DataFrame, y_labels: np.ndarray, event_columns: List[str]) -> List[str]:
        correlations = {}
        for col in event_columns:
            if X_all[col].std() == 0:
                correlations[col] = 0.0
                continue
            corr = X_all[col].corr(pd.Series(y_labels, index=X_all.index))
            correlations[col] = abs(corr) if pd.notna(corr) else 0.0
        self.selected_events = sorted(correlations, key=correlations.get, reverse=True)[:self.k]
        return self.selected_events

    def get_events(self) -> List[str]:
        return self.selected_events

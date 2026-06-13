from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCABaseline:
    """PCA dimensionality reduction baseline (Kadiyala, TECS 2020)."""
    def __init__(self, k: int = 4):
        self.k = k
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=k)

    def fit(self, X_train: pd.DataFrame) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.pca.fit(X_scaled)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)

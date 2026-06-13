from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class OCSVMDetector:
    """One-Class SVM: trains only on benign, flags anomalies."""
    def __init__(self, kernel='rbf', nu=0.1):
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel=kernel, nu=nu)

    def fit(self, X_benign: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_benign)
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return (preds == -1).astype(int)

class IsolationForestDetector:
    """Isolation Forest: unsupervised anomaly detection."""
    def __init__(self, contamination=0.1, n_estimators=200, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X_benign: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_benign)
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return (preds == -1).astype(int)

class XGBDetector:
    """XGBoost: supervised binary classifier."""
    def __init__(self, n_estimators=200, max_depth=6, random_state=42):
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric='logloss',
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

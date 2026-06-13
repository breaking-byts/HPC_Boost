"""
Experiment 2: HPC-Boost vs Baselines Comparison
================================================
For each event selection strategy, train detectors and measure F1/AUC.
Proves that per-binary event selection (HPC-Boost) outperforms global selection.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.two_smart import TwoSmartBaseline
from src.baselines.pca_selection import PCABaseline
from src.baselines.global_fixed import GlobalFixedBaseline
from src.baselines.random_selection import RandomBaseline
from src.detection.detectors import OCSVMDetector, IsolationForestDetector, XGBDetector
from src.detection.metrics import evaluate_predictions
from src.utils.data_loader import RadarDataLoader

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
SAMPLE_LIMIT = 50
K = 4
N_FOLDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
RANKINGS_DIR = Path("data/processed/rankings/radar/per_sample")
RESULTS_DIR = Path("data/processed/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_sample(trace_events: pd.DataFrame) -> pd.Series:
    """Aggregate a time-series trace into a single feature vector (mean per event)."""
    return trace_events.mean()


def load_dataset(limit: int) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Load samples, aggregate to feature vectors, return (X, y, sample_ids, event_cols)."""
    loader = RadarDataLoader()
    sample_ids = loader.list_sample_ids(limit=limit)
    traces = loader.load_samples(sample_ids)

    rows = []
    labels = []
    ids = []
    for sid in sample_ids:
        tr = traces[sid]
        feat = aggregate_sample(tr.events)
        rows.append(feat)
        labels.append(tr.binary_label)
        ids.append(sid)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    event_cols = list(X.columns)
    return X, y, ids, event_cols


def load_hpcboost_top_events(sample_id: str, k: int) -> List[str]:
    """Load the pre-computed HPC-Boost top-K events for a sample."""
    # Try various filename patterns
    for fname in [f"ranking_{sample_id}.csv", f"ranking_{sample_id}"]:
        path = RANKINGS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            return df.head(k)["event_name"].tolist()
    raise FileNotFoundError(f"No ranking file for {sample_id} in {RANKINGS_DIR}")


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def run_detector_on_events(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    events: List[str], detector_name: str,
) -> Dict[str, float]:
    """Train a detector on selected events, evaluate on test set."""
    Xtr = X_train[events].values
    Xte = X_test[events].values

    if detector_name == "OCSVM":
        det = OCSVMDetector(nu=0.3)
        benign_mask = y_train == 0
        if benign_mask.sum() < 2:
            return {"f1": 0.0, "accuracy": 0.0, "recall_tpr": 0.0, "fpr": 0.0}
        det.fit(Xtr[benign_mask])
        preds = det.predict(Xte)
        return evaluate_predictions(y_test, preds)

    elif detector_name == "IsolationForest":
        det = IsolationForestDetector(contamination=0.3)
        benign_mask = y_train == 0
        if benign_mask.sum() < 2:
            return {"f1": 0.0, "accuracy": 0.0, "recall_tpr": 0.0, "fpr": 0.0}
        det.fit(Xtr[benign_mask])
        preds = det.predict(Xte)
        return evaluate_predictions(y_test, preds)

    elif detector_name == "XGBoost":
        det = XGBDetector(n_estimators=100, max_depth=4)
        det.fit(Xtr, y_train)
        preds = det.predict(Xte)
        proba = det.predict_proba(Xte)
        return evaluate_predictions(y_test, preds, proba)

    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def run_pca_detector(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    k: int, detector_name: str,
) -> Dict[str, float]:
    """PCA baseline: project into k dims, then detect."""
    pca = PCABaseline(k=k)
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)

    if detector_name == "OCSVM":
        det = OCSVMDetector(nu=0.3)
        benign_mask = y_train == 0
        det.fit(Xtr_pca[benign_mask])
        preds = det.predict(Xte_pca)
        return evaluate_predictions(y_test, preds)

    elif detector_name == "IsolationForest":
        det = IsolationForestDetector(contamination=0.3)
        benign_mask = y_train == 0
        det.fit(Xtr_pca[benign_mask])
        preds = det.predict(Xte_pca)
        return evaluate_predictions(y_test, preds)

    elif detector_name == "XGBoost":
        det = XGBDetector(n_estimators=100, max_depth=4)
        det.fit(Xtr_pca, y_train)
        preds = det.predict(Xte_pca)
        proba = det.predict_proba(Xte_pca)
        return evaluate_predictions(y_test, preds, proba)


def run_hpcboost_persample(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    test_ids: List[str], k: int, detector_name: str,
) -> Dict[str, float]:
    """HPC-Boost: each test sample uses its own top-K events."""
    all_preds = []
    all_true = []

    for i, sid in enumerate(test_ids):
        try:
            events = load_hpcboost_top_events(sid, k)
        except FileNotFoundError:
            continue

        # Check all events exist in data
        missing = [e for e in events if e not in X_train.columns]
        if missing:
            continue

        Xtr = X_train[events].values
        Xte_row = X_test[events].iloc[[i]].values

        if detector_name == "OCSVM":
            det = OCSVMDetector(nu=0.3)
            benign_mask = y_train == 0
            if benign_mask.sum() < 2:
                continue
            det.fit(Xtr[benign_mask])
            pred = det.predict(Xte_row)

        elif detector_name == "IsolationForest":
            det = IsolationForestDetector(contamination=0.3)
            benign_mask = y_train == 0
            if benign_mask.sum() < 2:
                continue
            det.fit(Xtr[benign_mask])
            pred = det.predict(Xte_row)

        elif detector_name == "XGBoost":
            det = XGBDetector(n_estimators=100, max_depth=4)
            det.fit(Xtr, y_train)
            pred = det.predict(Xte_row)

        all_preds.append(pred[0])
        all_true.append(y_test[i])

    if len(all_preds) < 2:
        return {"f1": 0.0, "accuracy": 0.0, "recall_tpr": 0.0, "fpr": 0.0}

    return evaluate_predictions(np.array(all_true), np.array(all_preds))


def main():
    print("=" * 70)
    print("EXPERIMENT 2: HPC-Boost vs Baselines Comparison")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"\nLoading {SAMPLE_LIMIT} samples...")
    X, y, sample_ids, event_cols = load_dataset(SAMPLE_LIMIT)
    print(f"  Samples: {len(X)} | Events: {len(event_cols)} | Malware: {y.sum()} | Benign: {(y==0).sum()}")

    # ── Fit 2SMaRT on full dataset (it's a global method) ───────────
    twosmart = TwoSmartBaseline(k=K)
    twosmart_events = twosmart.fit(X, y, event_cols)
    print(f"\n2SMaRT global events: {twosmart_events}")

    global_fixed = GlobalFixedBaseline()
    gf_events = global_fixed.get_events()
    # Verify global fixed events exist
    gf_events = [e for e in gf_events if e in event_cols]
    if len(gf_events) < K:
        # Fill with most variable events
        remaining = [e for e in event_cols if e not in gf_events]
        vars_ = X[remaining].var().sort_values(ascending=False)
        gf_events += vars_.index[:K - len(gf_events)].tolist()
    print(f"Global-Fixed events: {gf_events}")

    detectors = ["OCSVM", "IsolationForest", "XGBoost"]
    strategies = ["HPC-Boost", "2SMaRT", "Global-Fixed", "PCA", "Random"]

    # ── Cross-validation ────────────────────────────────────────────────
    all_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n── Fold {fold_idx + 1}/{N_FOLDS} ──")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_ids = [sample_ids[i] for i in test_idx]

        for det_name in detectors:
            # ── HPC-Boost (per-sample) ──
            metrics = run_hpcboost_persample(
                X_train, y_train, X_test, y_test, test_ids, K, det_name,
            )
            metrics.update({"strategy": "HPC-Boost", "detector": det_name, "fold": fold_idx})
            all_results.append(metrics)
            print(f"  HPC-Boost + {det_name}: F1={metrics['f1']:.3f}")

            # ── 2SMaRT ──
            metrics = run_detector_on_events(
                X_train, y_train, X_test, y_test, twosmart_events, det_name,
            )
            metrics.update({"strategy": "2SMaRT", "detector": det_name, "fold": fold_idx})
            all_results.append(metrics)
            print(f"  2SMaRT    + {det_name}: F1={metrics['f1']:.3f}")

            # ── Global-Fixed ──
            metrics = run_detector_on_events(
                X_train, y_train, X_test, y_test, gf_events, det_name,
            )
            metrics.update({"strategy": "Global-Fixed", "detector": det_name, "fold": fold_idx})
            all_results.append(metrics)
            print(f"  GlobalFix + {det_name}: F1={metrics['f1']:.3f}")

            # ── PCA ──
            metrics = run_pca_detector(
                X_train, y_train, X_test, y_test, K, det_name,
            )
            metrics.update({"strategy": "PCA", "detector": det_name, "fold": fold_idx})
            all_results.append(metrics)
            print(f"  PCA       + {det_name}: F1={metrics['f1']:.3f}")

            # ── Random (average over multiple seeds) ──
            rand_f1s = []
            for seed in RANDOM_SEEDS:
                rb = RandomBaseline(k=K, seed=seed)
                rand_events = rb.select_events(event_cols, seed=seed)
                m = run_detector_on_events(
                    X_train, y_train, X_test, y_test, rand_events, det_name,
                )
                rand_f1s.append(m["f1"])
            avg_rand = {
                "f1": float(np.mean(rand_f1s)),
                "strategy": "Random", "detector": det_name, "fold": fold_idx,
            }
            all_results.append(avg_rand)
            print(f"  Random    + {det_name}: F1={avg_rand['f1']:.3f}")

    # ── Aggregate results ───────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(["strategy", "detector"])["f1"].agg(["mean", "std"]).round(4)
    summary.columns = ["F1_mean", "F1_std"]
    summary = summary.sort_values("F1_mean", ascending=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS: Average F1 across folds")
    print("=" * 70)
    print(summary.to_string())

    # ── Strategy-level summary ──────────────────────────────────────────
    strat_summary = results_df.groupby("strategy")["f1"].agg(["mean", "std"]).round(4)
    strat_summary.columns = ["F1_mean", "F1_std"]
    strat_summary = strat_summary.sort_values("F1_mean", ascending=False)

    print("\n" + "=" * 70)
    print("STRATEGY-LEVEL SUMMARY (averaged across all detectors)")
    print("=" * 70)
    print(strat_summary.to_string())

    # ── Save ────────────────────────────────────────────────────────────
    results_df.to_csv(RESULTS_DIR / "exp2_comparison_results.csv", index=False)
    summary_path = RESULTS_DIR / "exp2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "per_strategy_detector": summary.reset_index().to_dict(orient="records"),
            "per_strategy": strat_summary.reset_index().to_dict(orient="records"),
            "config": {"samples": SAMPLE_LIMIT, "k": K, "folds": N_FOLDS},
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

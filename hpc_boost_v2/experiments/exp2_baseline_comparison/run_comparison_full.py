"""
Experiment 2 v2: HPC-Boost vs Baselines — with richer feature engineering.
Uses mean + std + min + max + skew + kurtosis per event (6 features per event).
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
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.two_smart import TwoSmartBaseline
from src.baselines.pca_selection import PCABaseline
from src.baselines.global_fixed import GlobalFixedBaseline
from src.baselines.random_selection import RandomBaseline
from src.detection.detectors import OCSVMDetector, IsolationForestDetector, XGBDetector
from src.detection.metrics import evaluate_predictions
from src.utils.data_loader import RadarDataLoader

warnings.filterwarnings("ignore")

SAMPLE_LIMIT = None
K = 4
N_FOLDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
RANKINGS_DIR = Path("data/processed/rankings/radar/per_sample")
RESULTS_DIR = Path("data/processed/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_sample_rich(trace_events: pd.DataFrame) -> Dict[str, float]:
    """6 statistical features per event: mean, std, min, max, skew, kurtosis."""
    features = {}
    for col in trace_events.columns:
        vals = trace_events[col].values.astype(float)
        features[f"{col}_mean"] = float(np.mean(vals))
        features[f"{col}_std"] = float(np.std(vals))
        features[f"{col}_min"] = float(np.min(vals))
        features[f"{col}_max"] = float(np.max(vals))
        features[f"{col}_skew"] = float(skew(vals)) if len(vals) > 2 else 0.0
        features[f"{col}_kurt"] = float(kurtosis(vals)) if len(vals) > 2 else 0.0
    return features


def get_event_feature_cols(event_names: List[str]) -> List[str]:
    """Get the 6 feature column names for given events."""
    cols = []
    for e in event_names:
        cols.extend([f"{e}_mean", f"{e}_std", f"{e}_min", f"{e}_max", f"{e}_skew", f"{e}_kurt"])
    return cols


def load_dataset(limit: int):
    """Load samples with rich features."""
    loader = RadarDataLoader()
    sample_ids = loader.list_sample_ids(limit=limit)
    traces = loader.load_samples(sample_ids)

    rows = []
    labels = []
    ids = []
    event_cols = None
    for sid in sample_ids:
        tr = traces[sid]
        if event_cols is None:
            event_cols = list(tr.events.columns)
        feat = aggregate_sample_rich(tr.events)
        rows.append(feat)
        labels.append(tr.binary_label)
        ids.append(sid)

    X = pd.DataFrame(rows).fillna(0)
    y = np.array(labels)
    return X, y, ids, event_cols


def load_hpcboost_top_events(sample_id: str, k: int) -> List[str]:
    for fname in [f"ranking_{sample_id}", f"ranking_{sample_id}.csv"]:
        path = RANKINGS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            return df.head(k)["event_name"].tolist()
    raise FileNotFoundError(f"No ranking for {sample_id}")


def run_detector(Xtr, y_train, Xte, y_test, det_name):
    if det_name == "OCSVM":
        det = OCSVMDetector(nu=0.3)
        benign = y_train == 0
        if benign.sum() < 2:
            return {"f1": 0.0}
        det.fit(Xtr[benign])
        preds = det.predict(Xte)
        return evaluate_predictions(y_test, preds)
    elif det_name == "IsolationForest":
        det = IsolationForestDetector(contamination=0.3)
        benign = y_train == 0
        if benign.sum() < 2:
            return {"f1": 0.0}
        det.fit(Xtr[benign])
        preds = det.predict(Xte)
        return evaluate_predictions(y_test, preds)
    elif det_name == "XGBoost":
        det = XGBDetector(n_estimators=100, max_depth=4)
        det.fit(Xtr, y_train)
        preds = det.predict(Xte)
        proba = det.predict_proba(Xte)
        return evaluate_predictions(y_test, preds, proba)


def main():
    print("=" * 70)
    print("EXPERIMENT 2 v2: Rich Features (6 per event)")
    print("=" * 70)

    print(f"\nLoading {SAMPLE_LIMIT} samples with rich features...")
    X, y, sample_ids, event_cols = load_dataset(SAMPLE_LIMIT)
    all_feature_cols = list(X.columns)
    print(f"  Samples: {len(X)} | Raw events: {len(event_cols)} | Feature cols: {len(all_feature_cols)} | Malware: {y.sum()} | Benign: {(y==0).sum()}")

    # Fit 2SMaRT on mean-only features for event selection
    mean_cols = [f"{e}_mean" for e in event_cols]
    X_means = X[mean_cols].copy()
    X_means.columns = event_cols  # strip _mean suffix for correlation
    twosmart = TwoSmartBaseline(k=K)
    twosmart_events = twosmart.fit(X_means, y, event_cols)
    print(f"\n2SMaRT selected: {twosmart_events}")

    gf = GlobalFixedBaseline()
    gf_events = [e for e in gf.get_events() if e in event_cols]
    if len(gf_events) < K:
        remaining = [e for e in event_cols if e not in gf_events]
        vars_ = X_means[remaining].var().sort_values(ascending=False)
        gf_events += vars_.index[:K - len(gf_events)].tolist()
    print(f"Global-Fixed selected: {gf_events}")

    detectors = ["OCSVM", "IsolationForest", "XGBoost"]
    all_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n── Fold {fold_idx + 1}/{N_FOLDS} ──")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_ids = [sample_ids[i] for i in test_idx]

        for det_name in detectors:
            # ── HPC-Boost per-sample ──
            all_preds, all_true = [], []
            for i, sid in enumerate(test_ids):
                try:
                    events = load_hpcboost_top_events(sid, K)
                except FileNotFoundError:
                    continue
                feat_cols = get_event_feature_cols(events)
                feat_cols = [c for c in feat_cols if c in X.columns]
                if len(feat_cols) < K:
                    continue
                Xtr = X_train[feat_cols].values
                Xte = X_test[feat_cols].iloc[[i]].values
                m = run_detector(Xtr, y_train, Xte, np.array([y_test[i]]), det_name)
                if "f1" in m:
                    pred = 1 if m.get("tp", 0) > 0 or m.get("fp", 0) > 0 else 0
                    # Direct prediction
                    if det_name == "XGBoost":
                        d = XGBDetector(n_estimators=100, max_depth=4)
                        d.fit(Xtr, y_train)
                        pred = d.predict(Xte)[0]
                    elif det_name == "OCSVM":
                        d = OCSVMDetector(nu=0.3)
                        benign = y_train == 0
                        if benign.sum() >= 2:
                            d.fit(Xtr[benign])
                            pred = d.predict(Xte)[0]
                        else:
                            continue
                    elif det_name == "IsolationForest":
                        d = IsolationForestDetector(contamination=0.3)
                        benign = y_train == 0
                        if benign.sum() >= 2:
                            d.fit(Xtr[benign])
                            pred = d.predict(Xte)[0]
                        else:
                            continue
                    all_preds.append(pred)
                    all_true.append(y_test[i])

            if len(all_preds) >= 2:
                hpc_m = evaluate_predictions(np.array(all_true), np.array(all_preds))
            else:
                hpc_m = {"f1": 0.0}
            hpc_m.update({"strategy": "HPC-Boost", "detector": det_name, "fold": fold_idx})
            all_results.append(hpc_m)
            print(f"  HPC-Boost + {det_name}: F1={hpc_m['f1']:.3f}")

            # ── 2SMaRT ──
            feat_cols = get_event_feature_cols(twosmart_events)
            m = run_detector(X_train[feat_cols].values, y_train, X_test[feat_cols].values, y_test, det_name)
            m.update({"strategy": "2SMaRT", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  2SMaRT    + {det_name}: F1={m['f1']:.3f}")

            # ── Global-Fixed ──
            feat_cols = get_event_feature_cols(gf_events)
            m = run_detector(X_train[feat_cols].values, y_train, X_test[feat_cols].values, y_test, det_name)
            m.update({"strategy": "Global-Fixed", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  GlobalFix + {det_name}: F1={m['f1']:.3f}")

            # ── PCA (all events, unfair but included as upper bound) ──
            pca = PCABaseline(k=K*6)  # More components since we have 6x features
            Xtr_pca = pca.fit_transform(X_train)
            Xte_pca = pca.transform(X_test)
            m = run_detector(Xtr_pca, y_train, Xte_pca, y_test, det_name)
            m.update({"strategy": "PCA*", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  PCA*      + {det_name}: F1={m['f1']:.3f}  (uses all 54 events)")

            # ── Random ──
            rand_f1s = []
            for seed in RANDOM_SEEDS:
                rb = RandomBaseline(k=K, seed=seed)
                rand_events = rb.select_events(event_cols, seed=seed)
                feat_cols = get_event_feature_cols(rand_events)
                m = run_detector(X_train[feat_cols].values, y_train, X_test[feat_cols].values, y_test, det_name)
                rand_f1s.append(m["f1"])
            avg_m = {"f1": float(np.mean(rand_f1s)), "strategy": "Random", "detector": det_name, "fold": fold_idx}
            all_results.append(avg_m)
            print(f"  Random    + {det_name}: F1={avg_m['f1']:.3f}")

    # ── Summary ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FINAL RESULTS: Average F1 across folds")
    print("=" * 70)
    summary = results_df.groupby(["strategy", "detector"])["f1"].agg(["mean", "std"]).round(4)
    summary.columns = ["F1_mean", "F1_std"]
    summary = summary.sort_values("F1_mean", ascending=False)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY (avg across all detectors)")
    print("=" * 70)
    strat = results_df.groupby("strategy")["f1"].agg(["mean", "std"]).round(4)
    strat.columns = ["F1_mean", "F1_std"]
    strat = strat.sort_values("F1_mean", ascending=False)
    print(strat.to_string())

    # Note about PCA fairness
    print("\n* PCA* uses all 54 events (not constrained to 4 PMU registers)")
    print("  It is included as an upper bound, not a fair competitor.")

    results_df.to_csv(RESULTS_DIR / "exp2_v2_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/exp2_v2_results.csv")


if __name__ == "__main__":
    main()

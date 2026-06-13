"""
Experiment 2 FAST: Parallelized HPC-Boost vs Baselines (14 cores).
Fixed: env vars before imports, proper feature count check, robust launch.
"""
from __future__ import annotations

# MUST be before numpy/sklearn/scipy imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json, sys, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.two_smart import TwoSmartBaseline
from src.baselines.pca_selection import PCABaseline
from src.baselines.global_fixed import GlobalFixedBaseline
from src.baselines.random_selection import RandomBaseline
from src.detection.detectors import OCSVMDetector, IsolationForestDetector, XGBDetector
from src.detection.metrics import evaluate_predictions
from src.utils.data_loader import RadarDataLoader

warnings.filterwarnings("ignore")

K = 4
FEATS_PER_EVENT = 6
N_FOLDS = 5
N_JOBS = 14
RANDOM_SEEDS = [42, 123, 456]
RANKINGS_DIR = Path("data/processed/rankings/radar/per_sample")
RESULTS_DIR = Path("data/processed/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_sample_rich(trace_events: pd.DataFrame) -> Dict[str, float]:
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
    cols = []
    for e in event_names:
        cols.extend([f"{e}_mean", f"{e}_std", f"{e}_min", f"{e}_max", f"{e}_skew", f"{e}_kurt"])
    return cols


def load_hpcboost_top_events(sample_id: str, k: int) -> List[str]:
    for fname in [f"ranking_{sample_id}", f"ranking_{sample_id}.csv"]:
        path = RANKINGS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            return df.head(k)["event_name"].tolist()
    raise FileNotFoundError(f"No ranking for {sample_id}")


def train_predict_single(sid, i, Xtr_dict, y_train, Xte_dict, y_true_val, det_name, all_cols):
    """Train a detector for one test sample using its HPC-Boost events."""
    try:
        events = load_hpcboost_top_events(sid, K)
    except FileNotFoundError:
        return None
    feat_cols = get_event_feature_cols(events)
    feat_cols = [c for c in feat_cols if c in all_cols]
    if len(feat_cols) < K * FEATS_PER_EVENT:
        return None

    Xtr = Xtr_dict[feat_cols].values
    Xte = Xte_dict[feat_cols].iloc[[i]].values

    try:
        if det_name == "OCSVM":
            d = OCSVMDetector(nu=0.3)
            benign = y_train == 0
            if benign.sum() < 2:
                return None
            d.fit(Xtr[benign])
            pred = d.predict(Xte)[0]
        elif det_name == "IsolationForest":
            d = IsolationForestDetector(contamination=0.3)
            benign = y_train == 0
            if benign.sum() < 2:
                return None
            d.fit(Xtr[benign])
            pred = d.predict(Xte)[0]
        elif det_name == "XGBoost":
            d = XGBDetector(n_estimators=100, max_depth=4)
            d.fit(Xtr, y_train)
            pred = d.predict(Xte)[0]
        return (pred, y_true_val)
    except Exception:
        return None


def run_detector_simple(Xtr, y_train, Xte, y_test, det_name):
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
    print("EXPERIMENT 2 FAST: Full Dataset, 14-core parallel")
    print("=" * 70, flush=True)

    print("\nLoading ALL samples with rich features...", flush=True)
    loader = RadarDataLoader()
    sample_ids = loader.list_sample_ids(limit=None)
    print(f"  Found {len(sample_ids)} sample IDs. Loading traces...", flush=True)
    traces = loader.load_samples(sample_ids)

    rows, labels, ids, event_cols = [], [], [], None
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
    all_cols = set(X.columns)
    print(f"  Samples: {len(X)} | Events: {len(event_cols)} | Features: {len(X.columns)}", flush=True)
    print(f"  Malware: {y.sum()} | Benign: {(y==0).sum()}", flush=True)

    # ── Fit global baselines ────────────────────────────────────────
    mean_cols = [f"{e}_mean" for e in event_cols]
    X_means = X[mean_cols].copy()
    X_means.columns = event_cols
    twosmart = TwoSmartBaseline(k=K)
    twosmart_events = twosmart.fit(X_means, y, event_cols)
    print(f"\n2SMaRT selected: {twosmart_events}", flush=True)

    gf = GlobalFixedBaseline()
    gf_events = [e for e in gf.get_events() if e in event_cols]
    if len(gf_events) < K:
        remaining = [e for e in event_cols if e not in gf_events]
        vars_ = X_means[remaining].var().sort_values(ascending=False)
        gf_events += vars_.index[:K - len(gf_events)].tolist()
    print(f"Global-Fixed: {gf_events}", flush=True)

    detectors = ["OCSVM", "IsolationForest", "XGBoost"]
    all_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*70}", flush=True)
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}  (train={len(train_idx)}, test={len(test_idx)})", flush=True)
        print(f"{'='*70}", flush=True)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        test_ids = [ids[i] for i in test_idx]

        for det_name in detectors:
            # ── HPC-Boost (PARALLEL) ────────────────────────────────
            print(f"  HPC-Boost + {det_name}: {len(test_ids)} samples on {N_JOBS} cores...", end="", flush=True)
            results_list = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(train_predict_single)(
                    sid, i, X_train, y_train, X_test, y_test[i], det_name, all_cols
                )
                for i, sid in enumerate(test_ids)
            )
            valid = [r for r in results_list if r is not None]
            if len(valid) >= 2:
                preds = np.array([r[0] for r in valid])
                trues = np.array([r[1] for r in valid])
                hpc_m = evaluate_predictions(trues, preds)
            else:
                hpc_m = {"f1": 0.0}
            hpc_m.update({"strategy": "HPC-Boost", "detector": det_name, "fold": fold_idx})
            all_results.append(hpc_m)
            print(f" F1={hpc_m['f1']:.4f} ({len(valid)}/{len(test_ids)} ok)", flush=True)

            # ── 2SMaRT ──
            feat_cols = get_event_feature_cols(twosmart_events)
            m = run_detector_simple(X_train[feat_cols].values, y_train, X_test[feat_cols].values, y_test, det_name)
            m.update({"strategy": "2SMaRT", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  2SMaRT    + {det_name}: F1={m['f1']:.4f}", flush=True)

            # ── Global-Fixed ──
            feat_cols = get_event_feature_cols(gf_events)
            m = run_detector_simple(X_train[feat_cols].values, y_train, X_test[feat_cols].values, y_test, det_name)
            m.update({"strategy": "Global-Fixed", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  GlobalFix + {det_name}: F1={m['f1']:.4f}", flush=True)

            # ── PCA* ──
            pca = PCABaseline(k=K * FEATS_PER_EVENT)
            Xtr_pca = pca.fit_transform(X_train)
            Xte_pca = pca.transform(X_test)
            m = run_detector_simple(Xtr_pca, y_train, Xte_pca, y_test, det_name)
            m.update({"strategy": "PCA*", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  PCA*      + {det_name}: F1={m['f1']:.4f}", flush=True)

            # ── Random ──
            rand_f1s = []
            for seed in RANDOM_SEEDS:
                rb = RandomBaseline(k=K, seed=seed)
                rand_events = rb.select_events(event_cols, seed=seed)
                fc = get_event_feature_cols(rand_events)
                m2 = run_detector_simple(X_train[fc].values, y_train, X_test[fc].values, y_test, det_name)
                rand_f1s.append(m2["f1"])
            avg_m = {"f1": float(np.mean(rand_f1s)), "strategy": "Random", "detector": det_name, "fold": fold_idx}
            all_results.append(avg_m)
            print(f"  Random    + {det_name}: F1={avg_m['f1']:.4f}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FINAL RESULTS: Average F1 across folds")
    print("=" * 70)
    summary = results_df.groupby(["strategy", "detector"])["f1"].agg(["mean", "std"]).round(4)
    summary.columns = ["F1_mean", "F1_std"]
    summary = summary.sort_values("F1_mean", ascending=False)
    print(summary.to_string(), flush=True)

    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY (avg across all detectors)")
    print("=" * 70)
    strat = results_df.groupby("strategy")["f1"].agg(["mean", "std"]).round(4)
    strat.columns = ["F1_mean", "F1_std"]
    strat = strat.sort_values("F1_mean", ascending=False)
    print(strat.to_string(), flush=True)

    print("\n* PCA* uses all 54 events (not constrained to 4 PMU registers)", flush=True)

    results_df.to_csv(RESULTS_DIR / "exp2_full_results.csv", index=False)
    with open(RESULTS_DIR / "exp2_full_summary.json", "w") as f:
        json.dump({
            "per_strategy_detector": summary.reset_index().to_dict(orient="records"),
            "per_strategy": strat.reset_index().to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HPC-Boost vs Baselines: Definitive Benchmark
=============================================

Scientific guarantees:
  1. Zero data leakage: 2SMaRT and Global-Fixed fallback fit inside each fold
  2. Fair scoring: ALL strategies scored on identical test sets per fold
  3. HPC-Boost rankings are per-sample (no cross-sample leakage)
  4. Missing rankings → conservative default (predict benign); coverage tracked
  5. PCA* labelled as upper bound (uses all 55 events; infeasible with 4 PMU regs)
  6. Consistent detector hyperparameters across all strategies
  7. Random baseline averaged over 5 seeds for stability

Technical guarantees:
  1. Thread pinning before numpy/sklearn imports (no nested parallelism)
  2. float32 throughout to halve memory
  3. Explicit memory cleanup after data loading
  4. Rankings preloaded once; workers receive numpy arrays + int index lists
  5. XGBoost n_jobs=1 explicit in workers
  6. Exceptions counted and reported (never silently swallowed)
"""
from __future__ import annotations

# ── Thread pinning (MUST be before numpy/scipy/sklearn) ─────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XGB_NTHREAD"] = "1"

import gc
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_loader import RadarDataLoader

warnings.filterwarnings("ignore")

# ── Constants ───────────────────────────────────────────────────────────
K = 4                       # Number of HPC events to select
FEATS_PER_EVENT = 6         # mean, std, min, max, skew, kurtosis
EXPECTED_FEATS = K * FEATS_PER_EVENT  # 24
N_FOLDS = 5
N_JOBS = 14                 # Leave 2 cores free on 16-core machine
RANDOM_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for stable random baseline
RANKINGS_DIR = Path("data/processed/rankings/radar/per_sample")
RESULTS_DIR = Path("data/processed/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Detector hyperparameters (consistent across all strategies)
OCSVM_NU = 0.3
IF_CONTAMINATION = 0.3
IF_N_ESTIMATORS = 200
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4

# Error tracking
_worker_errors: List[str] = []


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def aggregate_sample_rich(trace_events: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate a time-series trace into 6 statistical features per event.
    
    Why these 6: They capture distribution shape (mean, std), range (min, max),
    and higher-order moments (skew, kurtosis). This gives detectors richer
    signal than mean-only, which is critical for HPC-Boost since its ranking
    specifically identifies events with interesting temporal properties.
    """
    features = {}
    for col in trace_events.columns:
        vals = trace_events[col].values.astype(np.float32)
        n = len(vals)
        features[f"{col}_mean"] = float(np.mean(vals))
        features[f"{col}_std"]  = float(np.std(vals))
        features[f"{col}_min"]  = float(np.percentile(vals, 1))
        features[f"{col}_max"]  = float(np.percentile(vals, 99))
        vals_64 = vals.astype(np.float64)
        features[f"{col}_skew"] = float(skew(vals_64)) if n > 2 else 0.0
        features[f"{col}_kurt"] = float(kurtosis(vals_64)) if n > 2 else 0.0
    return features


def load_and_aggregate(limit: Optional[int] = None):
    """
    Load RaDaR traces, aggregate to feature vectors, free raw traces.
    Returns: (X_np float32, y int array, sample_ids list, event_col_names list, col_names list)
    """
    t0 = time.time()
    loader = RadarDataLoader()
    sample_ids = loader.list_sample_ids(limit=limit)
    print(f"  Found {len(sample_ids)} sample IDs. Loading traces...", flush=True)
    traces = loader.load_samples(sample_ids)
    t_load = time.time() - t0
    print(f"  Traces loaded in {t_load:.1f}s. Aggregating features...", flush=True)

    rows, labels, ids = [], [], []
    event_cols = None
    for sid in sample_ids:
        tr = traces[sid]
        if event_cols is None:
            event_cols = list(tr.events.columns)
        rows.append(aggregate_sample_rich(tr.events))
        labels.append(tr.binary_label)
        ids.append(sid)

    # Free raw traces (~1-2 GB)
    del traces
    gc.collect()

    X_df = pd.DataFrame(rows).fillna(0)
    del rows
    gc.collect()

    col_names = list(X_df.columns)
    X_np = X_df.values.astype(np.float32)
    del X_df
    gc.collect()

    y = np.array(labels, dtype=np.int32)
    t_total = time.time() - t0
    print(f"  Aggregation complete in {t_total:.1f}s. Memory freed.", flush=True)

    return X_np, y, ids, event_cols, col_names


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: RANKING PRELOADING
# ═══════════════════════════════════════════════════════════════════════

def get_feat_indices(event_names: List[str], col_to_idx: Dict[str, int]) -> Optional[List[int]]:
    """
    Convert event names to column indices in the feature matrix.
    Returns None if ANY feature column is missing (strict: all K*6 must exist).
    """
    indices = []
    for e in event_names:
        for suffix in ["_mean", "_std", "_min", "_max", "_skew", "_kurt"]:
            key = f"{e}{suffix}"
            if key not in col_to_idx:
                return None
            indices.append(col_to_idx[key])
    return indices


def get_feat_col_names(event_names: List[str]) -> List[str]:
    """Get the 6 feature column names for given events (for pandas operations)."""
    cols = []
    for e in event_names:
        for suffix in ["_mean", "_std", "_min", "_max", "_skew", "_kurt"]:
            cols.append(f"{e}{suffix}")
    return cols


def preload_all_rankings(sample_ids: List[str], k: int) -> Tuple[Dict[str, List[str]], int, int]:
    """
    Load ALL ranking CSVs into memory once.
    Returns: (rankings_dict, found_count, missing_count)
    """
    rankings = {}
    found = 0
    missing = 0
    for sid in sample_ids:
        loaded = False
        # Try both filename patterns (sample IDs may include .csv extension)
        for fname in [f"ranking_{sid}", f"ranking_{sid}.csv"]:
            path = RANKINGS_DIR / fname
            if path.exists():
                df = pd.read_csv(path)
                rankings[sid] = df.head(k)["event_name"].tolist()
                found += 1
                loaded = True
                break
        if not loaded:
            missing += 1
    return rankings, found, missing


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: DETECTOR TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute detection metrics. Self-contained (no external dependency)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr":        float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train_and_predict(Xtr: np.ndarray, y_train: np.ndarray,
                      Xte: np.ndarray, det_name: str) -> np.ndarray:
    """
    Train a detector on Xtr/y_train, return predictions for Xte.
    Self-contained: imports and builds everything internally.
    Consistent hyperparameters regardless of caller.
    """
    scaler = StandardScaler()

    if det_name == "OCSVM":
        from sklearn.svm import OneClassSVM
        benign_mask = y_train == 0
        if benign_mask.sum() < 2:
            return np.zeros(len(Xte), dtype=int)
        Xtr_b = scaler.fit_transform(Xtr[benign_mask])
        model = OneClassSVM(kernel="rbf", nu=OCSVM_NU)
        model.fit(Xtr_b)
        preds = model.predict(scaler.transform(Xte))
        return (preds == -1).astype(int)

    elif det_name == "IsolationForest":
        from sklearn.ensemble import IsolationForest
        benign_mask = y_train == 0
        if benign_mask.sum() < 2:
            return np.zeros(len(Xte), dtype=int)
        Xtr_b = scaler.fit_transform(Xtr[benign_mask])
        model = IsolationForest(
            contamination=IF_CONTAMINATION,
            n_estimators=IF_N_ESTIMATORS,
            random_state=42,
        )
        model.fit(Xtr_b)
        preds = model.predict(scaler.transform(Xte))
        return (preds == -1).astype(int)

    elif det_name == "XGBoost":
        from xgboost import XGBClassifier
        Xtr_s = scaler.fit_transform(Xtr)
        model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,       # Explicit: prevent nested parallelism
            verbosity=0,
        )
        model.fit(Xtr_s, y_train)
        return model.predict(scaler.transform(Xte)).astype(int)

    else:
        raise ValueError(f"Unknown detector: {det_name}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: HPC-BOOST PARALLEL WORKER
# ═══════════════════════════════════════════════════════════════════════

def hpcboost_worker(
    feat_indices: Optional[List[int]],
    Xtr_np: np.ndarray,
    y_train: np.ndarray,
    xte_row: np.ndarray,
    det_name: str,
) -> Tuple[int, bool]:
    """
    Classify ONE test sample using its per-binary HPC-Boost event selection.

    Args:
        feat_indices: Precomputed column indices for this sample's top-K events.
                      None if ranking is missing.
        Xtr_np: Full training feature matrix (float32).
        y_train: Training labels.
        xte_row: Single test sample's feature vector (1D float32).
        det_name: Which detector to use.

    Returns:
        (prediction, had_ranking): prediction is 0 or 1.
        had_ranking is False if we fell back to default.
    """
    if feat_indices is None:
        # No ranking available → conservative default: predict benign
        return (0, False)

    Xtr_sub = Xtr_np[:, feat_indices]
    Xte_sub = xte_row[feat_indices].reshape(1, -1)

    try:
        preds = train_and_predict(Xtr_sub, y_train, Xte_sub, det_name)
        return (int(preds[0]), True)
    except Exception as e:
        # Don't silently swallow — record the error, then fail safe to benign (0).
        # (warnings are globally filtered to "ignore" at module load, so print.)
        print(f"  ⚠ hpcboost_worker fit/predict failed ({det_name}): {e}", flush=True)
        return (0, False)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: BASELINE SELECTION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def fit_twosmart_fold(Xtr_np: np.ndarray, y_train: np.ndarray,
                      event_cols: List[str], col_to_idx: Dict[str, int],
                      k: int) -> List[str]:
    """
    Fit 2SMaRT (Pearson correlation with label) on TRAINING data only.
    Returns k event names with highest |correlation|.
    """
    mean_indices = [col_to_idx[f"{e}_mean"] for e in event_cols]
    Xtr_means = Xtr_np[:, mean_indices]

    correlations = {}
    for j, col in enumerate(event_cols):
        col_vals = Xtr_means[:, j]
        if np.std(col_vals) == 0:
            correlations[col] = 0.0
            continue
        # Pearson correlation with binary label
        corr = np.corrcoef(col_vals, y_train)[0, 1]
        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0

    sorted_events = sorted(correlations, key=correlations.get, reverse=True)
    return sorted_events[:k]


def get_global_fixed_events_fold(event_cols: List[str], Xtr_np: np.ndarray,
                                  col_to_idx: Dict[str, int], k: int) -> List[str]:
    """
    Global-Fixed baseline: hardcoded events, with variance-based fallback
    computed on TRAINING data only if needed.
    """
    hardcoded = ["Instruct", "Core_cyc", "L1D_Miss", "BrMispred"]
    selected = [e for e in hardcoded if e in event_cols]

    if len(selected) < k:
        # Fallback: fill with highest-variance events from TRAINING data
        mean_indices = [col_to_idx[f"{e}_mean"] for e in event_cols]
        Xtr_means = Xtr_np[:, mean_indices]
        variances = np.var(Xtr_means, axis=0)
        remaining = [(event_cols[j], variances[j]) for j in range(len(event_cols))
                     if event_cols[j] not in selected]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for evt, _ in remaining:
            if len(selected) >= k:
                break
            selected.append(evt)

    return selected


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 70)
    print("HPC-BOOST vs BASELINES: DEFINITIVE BENCHMARK")
    print("=" * 70, flush=True)
    print(f"\nConfig: K={K}, folds={N_FOLDS}, jobs={N_JOBS}, random_seeds={len(RANDOM_SEEDS)}")
    print(f"Detector params: OCSVM(nu={OCSVM_NU}), IF(contamination={IF_CONTAMINATION}), "
          f"XGB(n_est={XGB_N_ESTIMATORS}, depth={XGB_MAX_DEPTH})", flush=True)

    # ── Phase 1: Load & aggregate ───────────────────────────────────
    print("\n[Phase 1] Loading and aggregating dataset...", flush=True)
    X_np, y, sample_ids, event_cols, col_names = load_and_aggregate(limit=None)
    col_to_idx = {c: i for i, c in enumerate(col_names)}

    n_samples = len(X_np)
    n_malware = int(y.sum())
    n_benign = n_samples - n_malware
    print(f"  Dataset: {n_samples} samples | {len(event_cols)} events | "
          f"{len(col_names)} features", flush=True)
    print(f"  Classes: {n_malware} malware ({100*n_malware/n_samples:.1f}%) | "
          f"{n_benign} benign ({100*n_benign/n_samples:.1f}%)", flush=True)

    # ── Phase 2: Preload rankings ───────────────────────────────────
    print("\n[Phase 2] Preloading HPC-Boost rankings...", flush=True)
    all_rankings, rank_found, rank_missing = preload_all_rankings(sample_ids, K)

    # Pre-compute feature index arrays (one dict lookup per sample, no work in workers)
    sample_feat_indices: Dict[str, Optional[List[int]]] = {}
    valid_indices = 0
    for sid in sample_ids:
        if sid in all_rankings:
            indices = get_feat_indices(all_rankings[sid], col_to_idx)
            sample_feat_indices[sid] = indices
            if indices is not None:
                valid_indices += 1
        else:
            sample_feat_indices[sid] = None

    coverage_pct = 100 * valid_indices / n_samples
    print(f"  Rankings found: {rank_found}/{n_samples}", flush=True)
    print(f"  Valid feature mappings: {valid_indices}/{n_samples} ({coverage_pct:.1f}%)", flush=True)
    if rank_missing > 0:
        print(f"  ⚠ {rank_missing} samples have no ranking (will default to pred=0)", flush=True)

    # ── Phase 3: Cross-validation ───────────────────────────────────
    print(f"\n[Phase 3] Running {N_FOLDS}-fold stratified CV...", flush=True)

    detectors = ["OCSVM", "IsolationForest", "XGBoost"]
    all_results = []
    fold_errors = 0
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_np, y)):
        t_fold = time.time()
        n_train, n_test = len(train_idx), len(test_idx)

        print(f"\n{'─'*70}", flush=True)
        print(f"FOLD {fold_idx+1}/{N_FOLDS}  "
              f"(train={n_train} [{(y[train_idx]==1).sum()}m/{(y[train_idx]==0).sum()}b], "
              f"test={n_test} [{(y[test_idx]==1).sum()}m/{(y[test_idx]==0).sum()}b])", flush=True)
        print(f"{'─'*70}", flush=True)

        Xtr = X_np[train_idx]
        Xte = X_np[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        test_sids = [sample_ids[i] for i in test_idx]

        # ── Fold-local baseline fitting (NO LEAKAGE) ───────────────
        twosmart_events = fit_twosmart_fold(Xtr, y_train, event_cols, col_to_idx, K)
        gf_events = get_global_fixed_events_fold(event_cols, Xtr, col_to_idx, K)

        if fold_idx == 0:
            print(f"  2SMaRT selected (fold 1): {twosmart_events}", flush=True)
            print(f"  Global-Fixed (fold 1):    {gf_events}", flush=True)

        # Precompute index arrays for baselines (fold-constant)
        ts_indices = get_feat_indices(twosmart_events, col_to_idx)
        gf_indices = get_feat_indices(gf_events, col_to_idx)

        for det_name in detectors:
            # ── HPC-Boost (per-sample, parallel) ───────────────────
            test_feat_idx_list = [sample_feat_indices[sid] for sid in test_sids]
            fold_coverage = sum(1 for fi in test_feat_idx_list if fi is not None)

            print(f"  HPC-Boost + {det_name}: "
                  f"{n_test} samples ({fold_coverage} ranked), "
                  f"{N_JOBS} cores...", end="", flush=True)

            t_hpc = time.time()
            worker_results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(hpcboost_worker)(
                    test_feat_idx_list[i],
                    Xtr, y_train,
                    Xte[i], det_name,
                )
                for i in range(n_test)
            )
            dt_hpc = time.time() - t_hpc

            # Unpack results — ALL samples get a prediction (no dropping)
            hpc_preds = np.array([r[0] for r in worker_results], dtype=int)
            hpc_had_ranking = np.array([r[1] for r in worker_results])
            n_defaulted = int((~hpc_had_ranking).sum())

            hpc_m = evaluate_predictions(y_test, hpc_preds)
            hpc_m.update({
                "strategy": "HPC-Boost", "detector": det_name, "fold": fold_idx,
                "coverage": fold_coverage, "defaulted": n_defaulted,
            })
            all_results.append(hpc_m)
            print(f" F1={hpc_m['f1']:.4f} P={hpc_m['precision']:.3f} "
                  f"R={hpc_m['recall_tpr']:.3f} [{dt_hpc:.1f}s]", flush=True)
            if n_defaulted > 0:
                print(f"    ⚠ {n_defaulted} samples defaulted to pred=0", flush=True)

            # ── 2SMaRT (fold-local, no leakage) ───────────────────
            ts_preds = train_and_predict(Xtr[:, ts_indices], y_train,
                                          Xte[:, ts_indices], det_name)
            m = evaluate_predictions(y_test, ts_preds)
            m.update({"strategy": "2SMaRT", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  2SMaRT    + {det_name}: F1={m['f1']:.4f} "
                  f"P={m['precision']:.3f} R={m['recall_tpr']:.3f}", flush=True)

            # ── Global-Fixed (fold-local fallback) ─────────────────
            gf_preds = train_and_predict(Xtr[:, gf_indices], y_train,
                                          Xte[:, gf_indices], det_name)
            m = evaluate_predictions(y_test, gf_preds)
            m.update({"strategy": "Global-Fixed", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  GlobalFix + {det_name}: F1={m['f1']:.4f} "
                  f"P={m['precision']:.3f} R={m['recall_tpr']:.3f}", flush=True)

            # ── PCA* (upper bound — uses all 55 events) ───────────
            from sklearn.decomposition import PCA
            pca_scaler = StandardScaler()
            pca_model = PCA(n_components=EXPECTED_FEATS)
            Xtr_pca = pca_model.fit_transform(pca_scaler.fit_transform(Xtr))
            Xte_pca = pca_model.transform(pca_scaler.transform(Xte))
            pca_preds = train_and_predict(Xtr_pca, y_train, Xte_pca, det_name)
            m = evaluate_predictions(y_test, pca_preds)
            m.update({"strategy": "PCA*", "detector": det_name, "fold": fold_idx})
            all_results.append(m)
            print(f"  PCA*      + {det_name}: F1={m['f1']:.4f} "
                  f"P={m['precision']:.3f} R={m['recall_tpr']:.3f}", flush=True)

            # ── Random (5 seeds, averaged) ─────────────────────────
            import random as _random
            rand_metrics = {"f1": [], "precision": [], "recall_tpr": [], "accuracy": []}
            for seed in RANDOM_SEEDS:
                rng = _random.Random(seed)
                rand_events = rng.sample(event_cols, K)
                r_indices = get_feat_indices(rand_events, col_to_idx)
                if r_indices is None:
                    continue
                r_preds = train_and_predict(Xtr[:, r_indices], y_train,
                                             Xte[:, r_indices], det_name)
                rm = evaluate_predictions(y_test, r_preds)
                for key in rand_metrics:
                    rand_metrics[key].append(rm[key])

            avg_m = {k: float(np.mean(v)) if v else 0.0 for k, v in rand_metrics.items()}
            avg_m.update({"strategy": "Random", "detector": det_name, "fold": fold_idx})
            all_results.append(avg_m)
            print(f"  Random    + {det_name}: F1={avg_m['f1']:.4f} "
                  f"P={avg_m['precision']:.3f} R={avg_m['recall_tpr']:.3f}", flush=True)

        dt_fold = time.time() - t_fold
        print(f"\n  Fold {fold_idx+1} completed in {dt_fold:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 7: RESULTS
    # ═══════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)
    dt_total = time.time() - t_start

    # ── Per strategy × detector ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (Average ± Std across {N_FOLDS} folds)")
    print(f"{'='*70}")

    for metric_name in ["f1", "precision", "recall_tpr", "accuracy"]:
        if metric_name not in results_df.columns:
            continue
        print(f"\n── {metric_name.upper()} ──")
        tbl = results_df.groupby(["strategy", "detector"])[metric_name].agg(["mean", "std"]).round(4)
        tbl.columns = [f"{metric_name}_mean", f"{metric_name}_std"]
        tbl = tbl.sort_values(f"{metric_name}_mean", ascending=False)
        print(tbl.to_string(), flush=True)

    # ── Strategy summary (primary: F1) ──────────────────────────────
    print(f"\n{'='*70}")
    print("STRATEGY SUMMARY (F1 averaged across all detectors)")
    print(f"{'='*70}")
    strat = results_df.groupby("strategy").agg({
        "f1": ["mean", "std"],
        "precision": ["mean"],
        "recall_tpr": ["mean"],
    }).round(4)
    strat.columns = ["F1_mean", "F1_std", "Prec_mean", "Recall_mean"]
    strat = strat.sort_values("F1_mean", ascending=False)
    print(strat.to_string(), flush=True)

    # ── Coverage & methodology notes ────────────────────────────────
    hpc_rows = results_df[results_df["strategy"] == "HPC-Boost"]
    if "coverage" in hpc_rows.columns:
        avg_cov = hpc_rows["coverage"].mean()
        avg_def = hpc_rows["defaulted"].mean()
        print(f"\nHPC-Boost coverage: {avg_cov:.0f}/{n_test} avg per fold "
              f"({avg_def:.0f} defaulted)", flush=True)

    print(f"\nTotal runtime: {dt_total:.1f}s ({dt_total/60:.1f} min)", flush=True)
    print("\n─── Methodology Notes ───")
    print("• 2SMaRT fitted per-fold on training data only (no leakage)")
    print("• Global-Fixed fallback uses fold-local variance (no leakage)")
    print("• HPC-Boost rankings: per-sample, no cross-sample data used")
    print("• HPC-Boost scored on FULL fold (missing rankings → default pred=0)")
    print("• PCA* uses all 55 events → infeasible with 4 PMU registers (upper bound)")
    print(f"• Random baseline averaged over {len(RANDOM_SEEDS)} seeds", flush=True)

    # ── Save ────────────────────────────────────────────────────────
    results_df.to_csv(RESULTS_DIR / "exp2_full_results.csv", index=False)
    with open(RESULTS_DIR / "exp2_full_summary.json", "w") as f:
        json.dump({
            "per_strategy_detector": (
                results_df.groupby(["strategy", "detector"])["f1"]
                .agg(["mean", "std"]).round(4)
                .reset_index().to_dict(orient="records")
            ),
            "per_strategy": strat.reset_index().to_dict(orient="records"),
            "config": {
                "K": K, "folds": N_FOLDS, "random_seeds": len(RANDOM_SEEDS),
                "samples": n_samples, "malware": n_malware, "benign": n_benign,
                "ranking_coverage": f"{valid_indices}/{n_samples}",
            },
            "methodology": [
                "2SMaRT fitted per-fold (no leakage)",
                "Global-Fixed fallback uses fold-local variance (no leakage)",
                "HPC-Boost rankings computed per-sample (no cross-sample leakage)",
                "HPC-Boost scored on full fold; missing rankings default to pred=0",
                "PCA* uses all 55 events (upper bound, infeasible with 4 PMU regs)",
                f"Random baseline averaged over {len(RANDOM_SEEDS)} seeds",
                f"Detector params: OCSVM(nu={OCSVM_NU}), IF(cont={IF_CONTAMINATION}), "
                f"XGB(n_est={XGB_N_ESTIMATORS}, depth={XGB_MAX_DEPTH})",
            ],
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GroupKFold variant of the detection-aware oracle.

Identical to run_beam_oracle.py except the outer split uses
GroupKFold(groups=family_gene) instead of StratifiedKFold.

Purpose: test whether reported F1 numbers hold under family-disjoint
splitting, which prevents related malware variants from appearing in
both train and test folds.

Usage on SSH server:
    cd ~/hpc_boost_v2
    python experiments/exp3_detection_aware_oracle/run_group_kfold_oracle.py \
        --radar-root ~/hpc_boost_v2/data/radar \
        --output-dir data/processed/results/group_kfold_oracle \
        --jobs 14
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGB_NTHREAD", "1")

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import RadarDataLoader

FEATURE_SUFFIXES = ("mean", "std", "min", "max", "skew", "kurt")
Subset = Tuple[str, ...]


@dataclass(frozen=True)
class SubsetScore:
    events: Subset
    mean_auprc: float
    std_auprc: float


def canonical_subset(events: Iterable[str]) -> Subset:
    return tuple(sorted(set(events)))


def expand_beam(beam: Sequence[Subset], event_names: Sequence[str]) -> List[Subset]:
    expanded = set()
    for subset in beam:
        existing = set(subset)
        for event in event_names:
            if event not in existing:
                expanded.add(canonical_subset((*subset, event)))
    return sorted(expanded)


def prune_scored_subsets(
    scored: Sequence[SubsetScore], limit: Optional[int]
) -> List[SubsetScore]:
    ordered = sorted(scored, key=lambda r: (-r.mean_auprc, r.std_auprc, r.events))
    return ordered if limit is None else ordered[:limit]


def route_candidate_oracle(
    y_true: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_scores: np.ndarray,
    global_candidate_index: int,
    candidate_probabilities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(candidate_predictions, dtype=int)
    scores = np.asarray(candidate_scores, dtype=float)
    if candidate_probabilities is None:
        probabilities = predictions.astype(float)
    else:
        probabilities = np.asarray(candidate_probabilities, dtype=float)

    priority = np.argsort(-scores, kind="stable")
    routed_pred = predictions[global_candidate_index].copy()
    routed_prob = probabilities[global_candidate_index].copy()
    correct_counts = np.sum(predictions == y_true[np.newaxis, :], axis=0)

    for i in range(len(y_true)):
        correct = priority[predictions[priority, i] == y_true[i]]
        if len(correct) == 0:
            continue
        chosen = int(correct[0])
        routed_pred[i] = predictions[chosen, i]
        routed_prob[i] = probabilities[chosen, i]

    coverage = float(np.mean(correct_counts > 0))
    return routed_pred, routed_prob, coverage


def aggregate_trace(events: pd.DataFrame) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for event in events.columns:
        values = events[event].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            finite = np.array([0.0])
        features[f"{event}_mean"] = float(np.mean(finite))
        features[f"{event}_std"] = float(np.std(finite))
        features[f"{event}_min"] = float(np.percentile(finite, 1))
        features[f"{event}_max"] = float(np.percentile(finite, 99))
        features[f"{event}_skew"] = float(skew(finite)) if len(finite) > 2 else 0.0
        features[f"{event}_kurt"] = float(kurtosis(finite)) if len(finite) > 2 else 0.0
    return features


def load_families_and_aggregate(
    radar_root: str, csv_name: str, chunksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """Returns X, y, groups (integer-encoded family_gene), sample_ids, event_names, feature_names.

    NOTE: return order is (X, y, groups, sample_ids, event_names, feature_names) —
    different from run_beam_oracle.load_and_aggregate which returns
    (X, y, sample_ids, event_names, feature_names, metadata).
    """
    loader = RadarDataLoader(root_dir=radar_root, csv_name=csv_name, chunksize=chunksize)
    print("Dataset:", json.dumps(loader.validate(), indent=2), flush=True)
    sample_ids = loader.list_sample_ids(limit=None)
    print(f"Loading {len(sample_ids)} samples...", flush=True)
    traces = loader.load_samples(sample_ids)

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    families: List[str] = []
    event_names = list(loader.hpc_columns)

    for idx, sid in enumerate(sample_ids, start=1):
        trace = traces[sid]
        rows.append(aggregate_trace(trace.events))
        labels.append(trace.binary_label)
        families.append(trace.family)
        if idx % 500 == 0 or idx == len(sample_ids):
            print(f"  aggregated {idx}/{len(sample_ids)}", flush=True)

    del traces
    gc.collect()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(frame.columns)
    X = frame.to_numpy(dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    # Encode family strings to integer group IDs
    unique_families = sorted(set(families))
    family_to_id = {f: i for i, f in enumerate(unique_families)}
    groups = np.asarray([family_to_id[f] for f in families], dtype=np.int32)

    print(f"\nGroup statistics:", flush=True)
    print(f"  Unique families: {len(unique_families)}", flush=True)
    family_counts = pd.Series(families).value_counts()
    print(f"  Largest family:  {family_counts.index[0]} ({family_counts.iloc[0]} samples)", flush=True)
    print(f"  Smallest family: {family_counts.index[-1]} ({family_counts.iloc[-1]} samples)", flush=True)
    print(f"  Top 10 families:\n{family_counts.head(10).to_string()}", flush=True)

    del frame, rows
    gc.collect()
    return X, y, groups, sample_ids, event_names, feature_names


def build_event_indices(
    event_names: Sequence[str], feature_names: Sequence[str]
) -> Dict[str, np.ndarray]:
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    result: Dict[str, np.ndarray] = {}
    for event in event_names:
        columns = [f"{event}_{s}" for s in FEATURE_SUFFIXES]
        result[event] = np.asarray([feature_to_index[c] for c in columns], dtype=np.int32)
    return result


def subset_feature_indices(events: Subset, event_indices: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([event_indices[e] for e in events])


def make_xgb(n_estimators: int, max_depth: int, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist",
        random_state=random_state, n_jobs=1, verbosity=0,
    )


def score_subset_inner_cv(
    events: Subset, X: np.ndarray, y: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int, max_depth: int, random_state: int,
) -> SubsetScore:
    columns = subset_feature_indices(events, event_indices)
    scores = []
    for k, (tr, va) in enumerate(inner_splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr][:, columns])
        X_va = scaler.transform(X[va][:, columns])
        model = make_xgb(n_estimators, max_depth, random_state + k)
        model.fit(X_tr, y[tr])
        scores.append(float(average_precision_score(y[va], model.predict_proba(X_va)[:, 1])))
    return SubsetScore(events=events, mean_auprc=float(np.mean(scores)), std_auprc=float(np.std(scores)))


def score_subsets(
    subsets, X, y, event_indices, inner_splits,
    n_estimators, max_depth, random_state, n_jobs,
) -> List[SubsetScore]:
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(score_subset_inner_cv)(s, X, y, event_indices, inner_splits, n_estimators, max_depth, random_state)
        for s in subsets
    )


def run_beam_search(
    X_train, y_train, event_names, event_indices,
    beam_width, candidate_cap, inner_folds, n_estimators, max_depth, random_state, n_jobs,
) -> List[SubsetScore]:
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    inner_splits = list(splitter.split(X_train, y_train))
    beam: List[Subset] = [(e,) for e in sorted(event_names)]

    for depth in range(1, 5):
        candidates = beam if depth == 1 else expand_beam(beam, event_names)
        print(f"    depth {depth}: {len(candidates)} subsets x {inner_folds} inner folds", flush=True)
        t0 = time.time()
        scored = score_subsets(candidates, X_train, y_train, event_indices, inner_splits,
                               n_estimators, max_depth, random_state + depth * 100, n_jobs)
        limit = candidate_cap if depth == 4 else beam_width
        retained = prune_scored_subsets(scored, limit)
        print(f"      retained {len(retained)}; best={retained[0].mean_auprc:.5f}; {time.time()-t0:.1f}s", flush=True)
        if depth == 4:
            return retained
        beam = [r.events for r in retained]
    raise RuntimeError("Beam search did not reach depth 4")


def fit_candidate_predict(
    events, X_train, y_train, X_test, event_indices, n_estimators, max_depth, random_state,
) -> Tuple[np.ndarray, np.ndarray]:
    columns = subset_feature_indices(events, event_indices)
    scaler = StandardScaler()
    model = make_xgb(n_estimators, max_depth, random_state)
    model.fit(scaler.fit_transform(X_train[:, columns]), y_train)
    probs = model.predict_proba(scaler.transform(X_test[:, columns]))[:, 1]
    return (probs >= 0.5).astype(np.int32), probs.astype(np.float32)


def fit_all_candidates(
    candidates, X_train, y_train, X_test, event_indices,
    n_estimators, max_depth, random_state, n_jobs,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(fit_candidate_predict)(
            c.events, X_train, y_train, X_test, event_indices,
            n_estimators, max_depth, random_state + i,
        )
        for i, c in enumerate(candidates)
    )
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def select_twosmart(X_train, y_train, event_names, event_indices, k=4) -> Subset:
    corrs = []
    for event in event_names:
        col = int(event_indices[event][0])
        vals = X_train[:, col]
        if np.std(vals) == 0:
            corrs.append((0.0, event))
        else:
            c = float(np.corrcoef(vals, y_train)[0, 1])
            corrs.append((abs(c) if np.isfinite(c) else 0.0, event))
    corrs.sort(key=lambda r: (-r[0], r[1]))
    return canonical_subset(e for _, e in corrs[:k])


def compute_metrics(y_true, predictions, probabilities) -> Dict[str, float]:
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    return {
        "f1":        float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall":    float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, predictions)),
        "auprc":     float(average_precision_score(y_true, probabilities)),
        "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GroupKFold oracle for HPC-Boost.")
    parser.add_argument("--radar-root", default="~/hpc_boost_v2/data/radar")
    parser.add_argument("--csv-name", default="combined_hardware_trails.csv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=25)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed/results/group_kfold_oracle")
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DETECTION-AWARE ORACLE — Hybrid GroupKFold (family-disjoint) variant")
    print("Split: GroupKFold on malware families + KFold on benign samples")
    print("=" * 70)
    print(json.dumps(vars(args), indent=2), flush=True)

    X, y, groups, sample_ids, event_names, feature_names = load_families_and_aggregate(
        args.radar_root, args.csv_name, args.chunksize
    )
    event_indices = build_event_indices(event_names, feature_names)

    # Separate malware and benign indices
    malware_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    malware_groups = groups[malware_idx]
    n_mal_families = len(np.unique(malware_groups))

    print(f"\nMatrix: {X.shape}, malware={len(malware_idx)}, benign={len(benign_idx)}, "
          f"malware_families={n_mal_families}", flush=True)
    print("Hybrid split: malware families are GroupKFold-disjoint across folds; "
          "benign samples are randomly split across folds.", flush=True)

    if n_mal_families < args.outer_folds:
        raise ValueError(
            f"Only {n_mal_families} malware families — cannot do {args.outer_folds}-fold "
            f"GroupKFold on malware. Reduce --outer-folds."
        )

    # Build fold indices using hybrid split
    malware_splitter = GroupKFold(n_splits=args.outer_folds)
    benign_splitter = KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)

    malware_folds = list(malware_splitter.split(X[malware_idx], y[malware_idx], groups=malware_groups))
    benign_folds = list(benign_splitter.split(benign_idx))

    all_fold_results = []
    all_preds: List[pd.DataFrame] = []
    total_start = time.time()

    for fold_idx in range(args.outer_folds):
        fold_num = fold_idx + 1

        m_train_local, m_test_local = malware_folds[fold_idx]
        b_train_local, b_test_local = benign_folds[fold_idx]

        train_idx = np.concatenate([malware_idx[m_train_local], benign_idx[b_train_local]])
        test_idx = np.concatenate([malware_idx[m_test_local], benign_idx[b_test_local]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Verify no malware family leaks
        train_mal_fams = set(malware_groups[m_train_local])
        test_mal_fams = set(malware_groups[m_test_local])
        overlap = train_mal_fams & test_mal_fams
        assert len(overlap) == 0, f"Family leak in fold {fold_num}: {overlap}"

        n_malware_test = int(y_test.sum())
        n_benign_test = int((y_test == 0).sum())
        n_malware_train = int(y_train.sum())
        n_benign_train = int((y_train == 0).sum())

        print(f"\n[Fold {fold_num}/{args.outer_folds}] "
              f"train={len(train_idx)} ({n_malware_train}m/{n_benign_train}b) "
              f"test={len(test_idx)} ({n_malware_test}m/{n_benign_test}b) "
              f"test_mal_families={len(test_mal_fams)}",
              flush=True)

        if n_malware_test == 0 or n_benign_test == 0 or n_malware_train == 0 or n_benign_train == 0:
            print(f"  SKIPPING fold {fold_num}: degenerate class distribution.", flush=True)
            continue

        candidates = run_beam_search(
            X_train, y_train, event_names, event_indices,
            args.beam_width, args.candidate_cap, args.inner_folds,
            args.n_estimators, args.max_depth,
            random_state=args.seed + fold_idx * 1000,
            n_jobs=args.jobs,
        )

        cand_preds, cand_probs = fit_all_candidates(
            candidates, X_train, y_train, X_test, event_indices,
            args.n_estimators, args.max_depth,
            random_state=args.seed + fold_idx * 10000,
            n_jobs=args.jobs,
        )
        cand_scores = np.asarray([c.mean_auprc for c in candidates])

        oracle_pred, oracle_prob, coverage = route_candidate_oracle(
            y_test, cand_preds, cand_scores, global_candidate_index=0,
            candidate_probabilities=cand_probs,
        )
        twosmart_events = select_twosmart(X_train, y_train, event_names, event_indices)
        ts_pred, ts_prob = fit_candidate_predict(
            twosmart_events, X_train, y_train, X_test, event_indices,
            args.n_estimators, args.max_depth, args.seed + fold_idx,
        )

        fold_result = {
            "fold": fold_num,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_malware": n_malware_test,
            "test_benign": n_benign_test,
            "train_families": len(train_mal_fams),
            "test_families": len(test_mal_fams),
            "candidate_count": len(candidates),
            "oracle_coverage": coverage,
            "global_beam_events": list(candidates[0].events),
            "twosmart_events": list(twosmart_events),
            "metrics": {
                "2SMaRT":          compute_metrics(y_test, ts_pred, ts_prob),
                "Global-Beam":     compute_metrics(y_test, cand_preds[0], cand_probs[0]),
                "Candidate-Oracle": compute_metrics(y_test, oracle_pred, oracle_prob),
            },
        }
        all_fold_results.append(fold_result)

        m = fold_result["metrics"]
        print(f"  F1: Oracle={m['Candidate-Oracle']['f1']:.4f}  "
              f"Global-Beam={m['Global-Beam']['f1']:.4f}  "
              f"2SMaRT={m['2SMaRT']['f1']:.4f}  coverage={coverage:.4f}", flush=True)

        pred_frame = pd.DataFrame({
            "sample_id": [sample_ids[i] for i in test_idx],
            "family": [str(groups[i]) for i in test_idx],
            "y_true": y_test,
            "twosmart_pred": ts_pred,
            "global_beam_pred": cand_preds[0],
            "oracle_pred": oracle_pred,
        })
        all_preds.append(pred_frame)

        fold_path = out / f"fold_{fold_num}"
        fold_path.mkdir(exist_ok=True)
        with open(fold_path / "summary.json", "w") as fh:
            json.dump(fold_result, fh, indent=2)

        del cand_preds, cand_probs
        gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    rows = []
    for r in all_fold_results:
        for strat, m in r["metrics"].items():
            rows.append({"fold": r["fold"], "strategy": strat, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out / "fold_metrics.csv", index=False)

    agg = metrics_df.groupby("strategy")[["f1", "precision", "recall", "accuracy", "fpr"]].agg(
        ["mean", "std"]
    ).round(6)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "aggregate_metrics.csv", index=False)
    print(agg.to_string(index=False), flush=True)

    # Pooled confusion counts
    print("\nPooled confusion counts:")
    if all_preds:
        pooled = pd.concat(all_preds, ignore_index=True)
        pooled.to_csv(out / "all_predictions.csv", index=False)
        for strat, col in [("2SMaRT", "twosmart_pred"), ("Global-Beam", "global_beam_pred"),
                           ("Candidate-Oracle", "oracle_pred")]:
            y = pooled["y_true"].values
            p = pooled[col].values
            tp = int(np.sum((p==1)&(y==1))); fp = int(np.sum((p==1)&(y==0)))
            tn = int(np.sum((p==0)&(y==0))); fn = int(np.sum((p==0)&(y==1)))
            f1v = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            print(f"  {strat}: TP={tp} FP={fp} TN={tn} FN={fn}  "
                  f"F1={f1v:.4f}  FPR={fpr:.4f}  Errors={fp+fn}")

    # Compare against stratified-sample results
    stratified_f1 = {"2SMaRT": 0.9090, "Global-Beam": 0.9134, "Candidate-Oracle": 0.9781}
    print("\nDelta vs stratified-sample oracle (positive = GroupKFold is better):")
    for strat in ["2SMaRT", "Global-Beam", "Candidate-Oracle"]:
        row = agg[agg["strategy"] == strat]
        if len(row) == 0:
            continue
        gkf_f1 = float(row["f1_mean"].iloc[0])
        delta = gkf_f1 - stratified_f1[strat]
        print(f"  {strat}: GroupKFold F1={gkf_f1:.4f}  Stratified F1={stratified_f1[strat]:.4f}  "
              f"delta={delta:+.4f}")

    summary = {
        "split_method": "GroupKFold",
        "group_column": "family_gene",
        "config": vars(args),
        "folds": all_fold_results,
        "aggregate": agg.to_dict(orient="records"),
        "stratified_comparison": {
            strat: stratified_f1[strat] for strat in stratified_f1
        },
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    elapsed = (time.time() - total_start) / 60
    print(f"\nDone. {len(all_fold_results)} folds in {elapsed:.1f} min.", flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detection-aware candidate-pool oracle for the four-PMU constraint.

Candidate generation is nested inside each outer training fold. Outer-test
labels are used only by the explicitly non-deployable Candidate-Oracle router.
"""
from __future__ import annotations

import os

# joblib owns process-level parallelism. Keep each numerical/model fit single-threaded.
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


@dataclass(frozen=True)
class OracleRouting:
    predictions: np.ndarray
    probabilities: np.ndarray
    candidate_indices: np.ndarray
    correct_candidate_counts: np.ndarray
    coverage: float


def validate_search_parameters(
    beam_width: int,
    candidate_cap: Optional[int],
    inner_folds: int,
) -> None:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if candidate_cap is not None and candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive when provided")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")


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
    scored: Sequence[SubsetScore],
    limit: Optional[int],
) -> List[SubsetScore]:
    ordered = sorted(
        scored,
        key=lambda row: (-row.mean_auprc, row.std_auprc, row.events),
    )
    return ordered if limit is None else ordered[:limit]


def route_candidate_oracle(
    y_true: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_scores: np.ndarray,
    global_candidate_index: int,
    candidate_probabilities: Optional[np.ndarray] = None,
) -> OracleRouting:
    """
    Route each sample to the highest-training-score candidate that is correct.

    This function intentionally uses y_true and is therefore non-deployable.
    Candidate rows may be in any order; candidate_scores determines priority.
    """
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(candidate_predictions, dtype=int)
    scores = np.asarray(candidate_scores, dtype=float)

    if predictions.ndim != 2:
        raise ValueError("candidate_predictions must be candidate-by-sample")
    if predictions.shape[1] != len(y_true):
        raise ValueError("candidate_predictions sample count must match y_true")
    if predictions.shape[0] != len(scores):
        raise ValueError("candidate_scores count must match candidate rows")
    if not 0 <= global_candidate_index < predictions.shape[0]:
        raise ValueError("global_candidate_index is out of range")

    if candidate_probabilities is None:
        probabilities = predictions.astype(float)
    else:
        probabilities = np.asarray(candidate_probabilities, dtype=float)
        if probabilities.shape != predictions.shape:
            raise ValueError("candidate_probabilities shape must match predictions")

    priority = np.argsort(-scores, kind="stable")
    routed_predictions = predictions[global_candidate_index].copy()
    routed_probabilities = probabilities[global_candidate_index].copy()
    routed_indices = np.full(len(y_true), global_candidate_index, dtype=int)
    correct_counts = np.sum(predictions == y_true[np.newaxis, :], axis=0)

    for sample_idx in range(len(y_true)):
        correct = priority[predictions[priority, sample_idx] == y_true[sample_idx]]
        if len(correct) == 0:
            continue
        chosen = int(correct[0])
        routed_indices[sample_idx] = chosen
        routed_predictions[sample_idx] = predictions[chosen, sample_idx]
        routed_probabilities[sample_idx] = probabilities[chosen, sample_idx]

    return OracleRouting(
        predictions=routed_predictions,
        probabilities=routed_probabilities,
        candidate_indices=routed_indices,
        correct_candidate_counts=correct_counts.astype(int),
        coverage=float(np.mean(correct_counts > 0)),
    )


def aggregate_trace(events: pd.DataFrame) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for event in events.columns:
        values = events[event].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            finite = np.array([0.0], dtype=np.float64)
        features[f"{event}_mean"] = float(np.mean(finite))
        features[f"{event}_std"] = float(np.std(finite))
        features[f"{event}_min"] = float(np.percentile(finite, 1))
        features[f"{event}_max"] = float(np.percentile(finite, 99))
        features[f"{event}_skew"] = (
            float(skew(finite.astype(np.float64))) if len(finite) > 2 else 0.0
        )
        features[f"{event}_kurt"] = (
            float(kurtosis(finite.astype(np.float64))) if len(finite) > 2 else 0.0
        )
    return features


def load_and_aggregate(
    radar_root: str,
    csv_name: str,
    sample_limit: Optional[int],
    chunksize: int,
    collect_metadata: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    List[str],
    Dict[str, List[str]],
]:
    loader = RadarDataLoader(
        root_dir=radar_root,
        csv_name=csv_name,
        chunksize=chunksize,
    )
    print("Dataset:", json.dumps(loader.validate(), indent=2), flush=True)
    sample_ids = loader.list_sample_ids(limit=sample_limit)
    print(f"Loading {len(sample_ids)} sample traces in one CSV pass...", flush=True)
    traces = loader.load_samples(sample_ids)

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    metadata: Dict[str, List[str]] = {"category": [], "family": [], "full_label": []}
    event_names = list(loader.hpc_columns)
    for idx, sample_id in enumerate(sample_ids, start=1):
        trace = traces[sample_id]
        rows.append(aggregate_trace(trace.events))
        labels.append(trace.binary_label)
        if collect_metadata:
            metadata["category"].append(trace.category)
            metadata["family"].append(trace.family)
            metadata["full_label"].append(trace.full_label)
        if idx % 250 == 0 or idx == len(sample_ids):
            print(f"  aggregated {idx}/{len(sample_ids)}", flush=True)

    del traces
    gc.collect()

    frame = pd.DataFrame(rows)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(frame.columns)
    X = frame.to_numpy(dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    del frame, rows
    gc.collect()
    return X, y, sample_ids, event_names, feature_names, metadata


def resolve_group_values(
    group_column: Optional[str],
    sample_ids: Sequence[str],
    metadata: Dict[str, List[str]],
) -> Optional[np.ndarray]:
    if group_column is None:
        return None

    # Aliases map user-facing CSV column names to metadata dict keys.
    # The CSV column names ("family_gene", "goal", "Filename") must match
    # RadarDataLoader.family_column, .category_column, and .sample_id_column.
    aliases = {
        "Filename": "sample_id",
        "filename": "sample_id",
        "sample_id": "sample_id",
        "goal": "category",
        "category": "category",
        "family_gene": "family",
        "family": "family",
        "full_label": "full_label",
    }
    key = aliases.get(group_column, group_column)

    if key == "sample_id":
        values = [str(sample_id) for sample_id in sample_ids]
    elif key in metadata:
        raw = metadata[key]
        none_indices = [i for i, v in enumerate(raw) if v is None]
        if none_indices:
            raise ValueError(
                f"--group-column {group_column!r}: {len(none_indices)} sample(s) have a "
                f"None value in '{key}' (first at index {none_indices[0]}). "
                "Ensure every sample has a non-null value for this column."
            )
        values = [str(v) for v in raw]
    else:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown --group-column {group_column!r}. Supported values: {supported}"
        )

    return np.asarray(values, dtype=object)


def _build_hybrid_splits(
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    seed: int,
    group_values: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold on malware families + KFold on benign samples.

    Avoids the degenerate-fold problem that arises when a single-class group
    (e.g. the all-benign 'normal' family) is isolated into one test fold by
    pure GroupKFold.
    """
    malware_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    malware_groups = group_values[malware_idx]
    n_mal_families = len(np.unique(malware_groups))
    if n_mal_families < outer_folds:
        raise ValueError(
            f"Hybrid split requires at least {outer_folds} malware families in the "
            f"group column; got {n_mal_families}. Reduce --outer-folds."
        )
    mal_splitter = GroupKFold(n_splits=outer_folds)
    ben_splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    malware_folds = list(
        mal_splitter.split(X[malware_idx], y[malware_idx], groups=malware_groups)
    )
    benign_folds = list(ben_splitter.split(benign_idx))
    splits = []
    for (m_train_local, m_test_local), (b_train_local, b_test_local) in zip(
        malware_folds, benign_folds
    ):
        train_idx = np.concatenate(
            [malware_idx[m_train_local], benign_idx[b_train_local]]
        )
        test_idx = np.concatenate(
            [malware_idx[m_test_local], benign_idx[b_test_local]]
        )
        splits.append((train_idx, test_idx))
    return splits


def build_outer_splits(
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    seed: int,
    group_values: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if group_values is None:
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(X, y))
        # StratifiedKFold normally guarantees class balance; check defensively for
        # very small datasets where the minority class count < outer_folds.
        for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
            train_classes = set(np.unique(y[train_idx]).tolist())
            test_classes = set(np.unique(y[test_idx]).tolist())
            if train_classes != {0, 1} or test_classes != {0, 1}:
                raise ValueError(
                    f"StratifiedKFold fold {fold_index} is class-degenerate: "
                    f"train_classes={sorted(train_classes)}, "
                    f"test_classes={sorted(test_classes)}. "
                    "Reduce --outer-folds or increase the size of the minority class."
                )
        return splits

    unique_groups = len(np.unique(group_values))
    if unique_groups < outer_folds:
        raise ValueError(
            f"group split requires at least {outer_folds} groups; got {unique_groups}"
        )
    splitter = GroupKFold(n_splits=outer_folds)
    splits = list(splitter.split(X, y, groups=group_values))
    degenerate = any(
        set(np.unique(y[train_idx])) != {0, 1} or set(np.unique(y[test_idx])) != {0, 1}
        for train_idx, test_idx in splits
    )
    if degenerate:
        print(
            "GroupKFold produced class-degenerate folds (a single-class group was "
            "isolated into a test fold). Falling back to hybrid split: "
            "GroupKFold on malware families + KFold on benign samples.",
            flush=True,
        )
        splits = _build_hybrid_splits(X, y, outer_folds, seed, group_values)
    return splits


def build_event_indices(
    event_names: Sequence[str],
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    result: Dict[str, np.ndarray] = {}
    for event in event_names:
        columns = [f"{event}_{suffix}" for suffix in FEATURE_SUFFIXES]
        missing = [column for column in columns if column not in feature_to_index]
        if missing:
            raise KeyError(f"Missing aggregate features for {event}: {missing}")
        result[event] = np.asarray(
            [feature_to_index[column] for column in columns],
            dtype=np.int32,
        )
    return result


def subset_feature_indices(
    events: Subset,
    event_indices: Dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate([event_indices[event] for event in events])


def make_xgb(
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )


def score_subset_inner_cv(
    events: Subset,
    X: np.ndarray,
    y: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> SubsetScore:
    columns = subset_feature_indices(events, event_indices)
    scores = []
    for split_idx, (train_idx, valid_idx) in enumerate(inner_splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx][:, columns])
        X_valid = scaler.transform(X[valid_idx][:, columns])
        model = make_xgb(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + split_idx,
        )
        model.fit(X_train, y[train_idx])
        probabilities = model.predict_proba(X_valid)[:, 1]
        scores.append(float(average_precision_score(y[valid_idx], probabilities)))
    return SubsetScore(
        events=events,
        mean_auprc=float(np.mean(scores)),
        std_auprc=float(np.std(scores)),
    )


def score_subsets(
    subsets: Sequence[Subset],
    X: np.ndarray,
    y: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
) -> List[SubsetScore]:
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(score_subset_inner_cv)(
            events=subset,
            X=X,
            y=y,
            event_indices=event_indices,
            inner_splits=inner_splits,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        for subset in subsets
    )


def run_beam_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_names: Sequence[str],
    event_indices: Dict[str, np.ndarray],
    beam_width: int,
    candidate_cap: Optional[int],
    inner_folds: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[List[SubsetScore], List[Dict[str, object]]]:
    validate_search_parameters(beam_width, candidate_cap, inner_folds)
    splitter = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )
    inner_splits = list(splitter.split(X_train, y_train))
    beam: List[Subset] = [(event,) for event in sorted(event_names)]
    history: List[Dict[str, object]] = []

    for depth in range(1, 5):
        candidates = beam if depth == 1 else expand_beam(beam, event_names)
        t0 = time.time()
        print(
            f"    beam depth {depth}: scoring {len(candidates)} subsets "
            f"x {inner_folds} inner folds",
            flush=True,
        )
        scored = score_subsets(
            subsets=candidates,
            X=X_train,
            y=y_train,
            event_indices=event_indices,
            inner_splits=inner_splits,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + depth * 100,
            n_jobs=n_jobs,
        )
        limit = candidate_cap if depth == 4 else beam_width
        retained = prune_scored_subsets(scored, limit)
        elapsed = time.time() - t0
        history.append(
            {
                "depth": depth,
                "evaluated": len(candidates),
                "retained": len(retained),
                "best_auprc": retained[0].mean_auprc,
                "seconds": elapsed,
            }
        )
        print(
            f"      retained {len(retained)}; best inner AUCPR="
            f"{retained[0].mean_auprc:.5f}; {elapsed:.1f}s",
            flush=True,
        )
        if depth == 4:
            return retained, history
        beam = [row.events for row in retained]

    raise RuntimeError("Beam search did not reach depth four")


def fit_candidate_predict(
    events: Subset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    columns = subset_feature_indices(events, event_indices)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train[:, columns])
    test_scaled = scaler.transform(X_test[:, columns])
    model = make_xgb(n_estimators, max_depth, random_state)
    model.fit(train_scaled, y_train)
    probabilities = model.predict_proba(test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int32)
    return predictions, probabilities.astype(np.float32)


def fit_all_candidates(
    candidates: Sequence[SubsetScore],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(fit_candidate_predict)(
            events=candidate.events,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            event_indices=event_indices,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + idx,
        )
        for idx, candidate in enumerate(candidates)
    )
    predictions = np.stack([row[0] for row in rows])
    probabilities = np.stack([row[1] for row in rows])
    return predictions, probabilities


def select_twosmart(
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_names: Sequence[str],
    event_indices: Dict[str, np.ndarray],
    k: int = 4,
) -> Subset:
    correlations = []
    for event in event_names:
        mean_column = int(event_indices[event][0])
        values = X_train[:, mean_column]
        if np.std(values) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(values, y_train)[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0
        correlations.append((abs(correlation), event))
    correlations.sort(key=lambda row: (-row[0], row[1]))
    return canonical_subset(event for _, event in correlations[:k])


def compute_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "auprc": float(average_precision_score(y_true, probabilities)),
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_outer_fold(
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: Sequence[str],
    event_names: Sequence[str],
    event_indices: Dict[str, np.ndarray],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    fold_number = fold_index + 1
    fold_dir = output_dir / f"fold_{fold_number}"
    summary_path = fold_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(f"[Fold {fold_number}] checkpoint exists; skipping.", flush=True)
        return json.loads(summary_path.read_text())

    fold_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(
        f"[Fold {fold_number}/{args.outer_folds}] train={len(train_idx)} "
        f"test={len(test_idx)}",
        flush=True,
    )
    fold_start = time.time()

    candidates, search_history = run_beam_search(
        X_train=X_train,
        y_train=y_train,
        event_names=event_names,
        event_indices=event_indices,
        beam_width=args.beam_width,
        candidate_cap=args.candidate_cap,
        inner_folds=args.inner_folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + fold_index * 1000,
        n_jobs=args.jobs,
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "candidate_index": idx,
                "events": "|".join(row.events),
                "inner_auprc_mean": row.mean_auprc,
                "inner_auprc_std": row.std_auprc,
            }
            for idx, row in enumerate(candidates)
        ]
    )
    atomic_write_csv(fold_dir / "candidates.csv", candidate_frame)
    atomic_write_json(fold_dir / "search_history.json", search_history)

    print(
        f"    fitting {len(candidates)} retained candidates on full outer train...",
        flush=True,
    )
    fit_start = time.time()
    candidate_predictions, candidate_probabilities = fit_all_candidates(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        event_indices=event_indices,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + fold_index * 10000,
        n_jobs=args.jobs,
    )
    candidate_fit_seconds = time.time() - fit_start

    candidate_scores = np.asarray([row.mean_auprc for row in candidates])
    global_index = 0
    global_metrics = compute_metrics(
        y_test,
        candidate_predictions[global_index],
        candidate_probabilities[global_index],
    )

    routing = route_candidate_oracle(
        y_true=y_test,
        candidate_predictions=candidate_predictions,
        candidate_probabilities=candidate_probabilities,
        candidate_scores=candidate_scores,
        global_candidate_index=global_index,
    )
    oracle_metrics = compute_metrics(
        y_test,
        routing.predictions,
        routing.probabilities,
    )

    twosmart_events = select_twosmart(
        X_train=X_train,
        y_train=y_train,
        event_names=event_names,
        event_indices=event_indices,
    )
    twosmart_predictions, twosmart_probabilities = fit_candidate_predict(
        events=twosmart_events,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        event_indices=event_indices,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + fold_index,
    )
    twosmart_metrics = compute_metrics(
        y_test,
        twosmart_predictions,
        twosmart_probabilities,
    )

    routed_events = [
        "|".join(candidates[idx].events) for idx in routing.candidate_indices
    ]
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [sample_ids[idx] for idx in test_idx],
            "y_true": y_test,
            "global_beam_pred": candidate_predictions[global_index],
            "global_beam_probability": candidate_probabilities[global_index],
            "twosmart_pred": twosmart_predictions,
            "twosmart_probability": twosmart_probabilities,
            "oracle_pred": routing.predictions,
            "oracle_probability": routing.probabilities,
            "oracle_candidate_index": routing.candidate_indices,
            "oracle_events": routed_events,
            "correct_candidate_count": routing.correct_candidate_counts,
        }
    )
    atomic_write_csv(fold_dir / "predictions.csv", prediction_frame)

    summary: Dict[str, object] = {
        "fold": fold_number,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "candidate_count": len(candidates),
        "global_beam_events": list(candidates[global_index].events),
        "global_beam_inner_auprc": candidates[global_index].mean_auprc,
        "twosmart_events": list(twosmart_events),
        "oracle_coverage": routing.coverage,
        "metrics": {
            "Global-Beam": global_metrics,
            "Candidate-Oracle": oracle_metrics,
            "2SMaRT": twosmart_metrics,
        },
        "search_history": search_history,
        "candidate_fit_seconds": candidate_fit_seconds,
        "total_fold_seconds": time.time() - fold_start,
    }
    atomic_write_json(summary_path, summary)
    print(
        f"    F1: Oracle={oracle_metrics['f1']:.4f} "
        f"Global-Beam={global_metrics['f1']:.4f} "
        f"2SMaRT={twosmart_metrics['f1']:.4f} "
        f"coverage={routing.coverage:.4f}",
        flush=True,
    )
    del candidate_predictions, candidate_probabilities
    gc.collect()
    return summary


def aggregate_summaries(
    summaries: Sequence[Dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    rows = []
    for summary in summaries:
        for strategy, metrics in summary["metrics"].items():
            rows.append(
                {
                    "fold": summary["fold"],
                    "strategy": strategy,
                    **metrics,
                    "oracle_coverage": (
                        summary["oracle_coverage"]
                        if strategy == "Candidate-Oracle"
                        else np.nan
                    ),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "fold_metrics.csv", fold_metrics)

    grouped = (
        fold_metrics.groupby("strategy")[["f1", "precision", "recall", "accuracy", "auprc"]]
        .agg(["mean", "std"])
        .round(6)
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    aggregate = grouped.reset_index()
    atomic_write_csv(output_dir / "aggregate_metrics.csv", aggregate)

    payload = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "folds": list(summaries),
        "aggregate": aggregate.to_dict(orient="records"),
        "methodology": {
            "candidate_generation": "outer-training data only",
            "search_objective": "mean inner-CV AUCPR",
            "global_beam": "highest inner-CV candidate",
            "candidate_oracle": (
                "post-hoc label-assisted routing among retained candidates; "
                "explicitly non-deployable"
            ),
        },
    }
    atomic_write_json(output_dir / "summary.json", payload)
    print("\nAggregate metrics:", flush=True)
    print(aggregate.to_string(index=False), flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested beam-search candidate oracle for four HPC events."
    )
    parser.add_argument("--radar-root", default="~/hpc_boost_v2/data/radar")
    parser.add_argument("--csv-name", default="combined_hardware_trails.csv")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=25)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Optional outer-CV grouping column. Supported aliases: "
            "family_gene/family, full_label, goal/category, Filename/sample_id. "
            "When set, outer CV uses GroupKFold and rejects class-degenerate folds."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/results/detection_aware_beam_oracle",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    validate_search_parameters(
        beam_width=args.beam_width,
        candidate_cap=args.candidate_cap,
        inner_folds=args.inner_folds,
    )
    if args.outer_folds < 2:
        parser.error("--outer-folds must be at least 2")
    if args.jobs == 0:
        parser.error("--jobs cannot be zero")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive when provided")
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if args.n_estimators <= 0 or args.max_depth <= 0:
        parser.error("XGBoost parameters must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "config.json", vars(args))

    print("=" * 80)
    print("DETECTION-AWARE CANDIDATE-POOL ORACLE")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2), flush=True)
    print(
        "\nWARNING: Candidate-Oracle uses outer-test labels only for post-hoc "
        "routing. It is a non-deployable ceiling, not a deployable detector.\n",
        flush=True,
    )

    X, y, sample_ids, event_names, feature_names, metadata = load_and_aggregate(
        radar_root=args.radar_root,
        csv_name=args.csv_name,
        sample_limit=args.sample_limit,
        chunksize=args.chunksize,
        collect_metadata=(args.group_column is not None),
    )
    if len(np.unique(y)) != 2:
        raise ValueError("Dataset must contain both benign and malware labels")
    minimum_class = int(np.min(np.bincount(y)))
    required = max(args.outer_folds, args.inner_folds)
    if minimum_class < required:
        raise ValueError(
            f"Smallest class has {minimum_class} samples; need at least {required}"
        )
    event_indices = build_event_indices(event_names, feature_names)
    group_values = resolve_group_values(args.group_column, sample_ids, metadata)

    print(
        f"Aggregated matrix: {X.shape}; malware={int(y.sum())}; "
        f"benign={int((y == 0).sum())}; events={len(event_names)}",
        flush=True,
    )
    depth4_upper = args.beam_width * max(0, len(event_names) - 3)
    retained_estimate = min(args.candidate_cap, depth4_upper)
    print(
        f"Expected final pool: up to {retained_estimate} candidates/fold. "
        f"Parallel jobs: {args.jobs}",
        flush=True,
    )

    outer_splits = build_outer_splits(
        X=X,
        y=y,
        outer_folds=args.outer_folds,
        seed=args.seed,
        group_values=group_values,
    )
    if group_values is None:
        print("Outer split: StratifiedKFold over samples", flush=True)
    else:
        print(
            f"Outer split: GroupKFold over {args.group_column} "
            f"({len(np.unique(group_values))} unique groups)",
            flush=True,
        )

    summaries = []
    start = time.time()
    for fold_index, (train_idx, test_idx) in enumerate(outer_splits):
        summaries.append(
            evaluate_outer_fold(
                fold_index=fold_index,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                sample_ids=sample_ids,
                event_names=event_names,
                event_indices=event_indices,
                args=args,
                output_dir=output_dir,
            )
        )
        aggregate_summaries(summaries, output_dir, args)

    print(
        f"\nCompleted {len(summaries)} folds in {(time.time() - start) / 3600:.2f} hours.",
        flush=True,
    )
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

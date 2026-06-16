#!/usr/bin/env python3
"""
Detection-aware candidate-pool oracle for the four-PMU constraint.

Candidate generation is nested inside each outer training fold. Outer-test
labels are used only by the explicitly non-deployable Candidate-Oracle router.

V3 note — the per-sample / per-group oracle routers intentionally route
different samples (or groups) to different 4-event subsets, which exceeds the
single fixed 4-PMU programming budget that real hardware allows. This is BY
DESIGN, not a defect: these routers are non-deployable CEILINGS whose only job
is to measure headroom. The `global` granularity is the one truly deployable
point (a single fixed 4-PMU subset for everything); the gap above it quantifies
how much accuracy a future static recommender could (per_family/per_full_label)
or could not (per_sample) recover. The experiment exists to justify continued
research by showing that headroom is large, not to ship an implementable
multi-subset detector.
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
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.metrics import tpr_at_fpr
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
    # Per-sample metadata is ALWAYS collected: the per-group oracle decomposition
    # needs category/family/full_label for test samples regardless of --group-column.
    metadata: Dict[str, List[str]] = {"category": [], "family": [], "full_label": []}
    event_names = list(loader.hpc_columns)
    for idx, sample_id in enumerate(sample_ids, start=1):
        trace = traces[sample_id]
        rows.append(aggregate_trace(trace.events))
        labels.append(trace.binary_label)
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
    n_splits: int,
    seed: int,
    group_values: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold on malware families + KFold on benign samples.

    Avoids the degenerate-fold problem that arises when a single-class group
    (e.g. the all-benign 'normal' family) is isolated into one test fold by
    pure GroupKFold. Shared by both the outer split and the grouped inner CV.
    """
    malware_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    malware_groups = group_values[malware_idx]
    n_mal_families = len(np.unique(malware_groups))
    if n_mal_families < n_splits:
        raise ValueError(
            f"Hybrid split requires at least {n_splits} malware families in the "
            f"group column; got {n_mal_families}. Reduce the fold count."
        )
    mal_splitter = GroupKFold(n_splits=n_splits)
    ben_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    malware_folds = list(
        mal_splitter.split(X[malware_idx], y[malware_idx], groups=malware_groups)
    )
    benign_folds = list(ben_splitter.split(benign_idx))
    splits = []
    for (m_train_local, m_test_local), (b_train_local, b_test_local) in zip(
        malware_folds, benign_folds
    ):
        # Family-disjointness guard: a malware family must never appear in both
        # the train and test side of the same fold.
        train_fams = set(malware_groups[m_train_local].tolist())
        test_fams = set(malware_groups[m_test_local].tolist())
        leak = train_fams & test_fams
        if leak:
            raise AssertionError(f"Malware family leak across hybrid fold: {sorted(leak)}")
        train_idx = np.concatenate(
            [malware_idx[m_train_local], benign_idx[b_train_local]]
        )
        test_idx = np.concatenate(
            [malware_idx[m_test_local], benign_idx[b_test_local]]
        )
        splits.append((train_idx, test_idx))
    return splits


def build_grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    group_values: np.ndarray,
    context: str = "split",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group-disjoint CV splits with a hybrid fallback for class degeneracy.

    Uses pure GroupKFold; if any fold becomes single-class (a single-class group
    isolated into a test fold), falls back to the hybrid split (GroupKFold on
    malware families + KFold on benign). Shared by outer and inner CV so the two
    levels stay consistent.
    """
    unique_groups = len(np.unique(group_values))
    if unique_groups < n_splits:
        raise ValueError(
            f"group split requires at least {n_splits} groups; got {unique_groups}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X, y, groups=group_values))
    degenerate = any(
        set(np.unique(y[train_idx])) != {0, 1} or set(np.unique(y[test_idx])) != {0, 1}
        for train_idx, test_idx in splits
    )
    if degenerate:
        print(
            f"GroupKFold ({context}) produced class-degenerate folds (a single-class "
            "group was isolated into a test fold). Falling back to hybrid split: "
            "GroupKFold on malware families + KFold on benign samples.",
            flush=True,
        )
        splits = _build_hybrid_splits(X, y, n_splits, seed, group_values)
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

    return build_grouped_splits(
        X=X,
        y=y,
        n_splits=outer_folds,
        seed=seed,
        group_values=group_values,
        context="outer",
    )


def build_inner_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int,
    seed: int,
    inner_split: str,
    train_group_values: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Inner-CV splits over the OUTER-TRAIN samples.

    When ``inner_split == "group"`` the folds are group-disjoint (with the same
    hybrid fallback as the outer split) so candidate selection / Global-Beam
    never sees a family leak. Otherwise StratifiedKFold over samples.
    """
    if inner_split == "group":
        if train_group_values is None:
            raise ValueError("inner_split='group' requires train_group_values")
        return build_grouped_splits(
            X=X_train,
            y=y_train,
            n_splits=inner_folds,
            seed=seed,
            group_values=train_group_values,
            context="inner",
        )
    splitter = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(X_train, y_train))


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
    inner_split: str = "stratified",
    train_group_values: Optional[np.ndarray] = None,
) -> Tuple[List[SubsetScore], List[Dict[str, object]]]:
    validate_search_parameters(beam_width, candidate_cap, inner_folds)
    inner_splits = build_inner_splits(
        X_train=X_train,
        y_train=y_train,
        inner_folds=inner_folds,
        seed=random_state,
        inner_split=inner_split,
        train_group_values=train_group_values,
    )
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


def threshold_for_target_tpr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_tpr: float,
) -> float:
    """Highest decision threshold whose TPR >= target_tpr on (y_true, prob).

    Thresholds are the unique probability values. Choosing the HIGHEST such
    threshold maximises precision/specificity while still meeting the recall
    floor. Falls back to 0.5 when no positive examples are present.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = probabilities[y_true == 1]
    if positives.size == 0:
        return 0.5
    candidate_thresholds = np.unique(np.concatenate([[0.0], probabilities]))
    n_pos = positives.size
    best: Optional[float] = None
    # Iterate high -> low; first threshold that meets the TPR floor is the highest.
    for threshold in sorted(candidate_thresholds, reverse=True):
        tpr = float(np.sum(positives >= threshold) / n_pos)
        if tpr >= target_tpr:
            best = float(threshold)
            break
    # If even threshold 0.0 cannot reach the target (impossible since all >= 0),
    # fall back to the lowest threshold.
    return best if best is not None else 0.0


def compute_subset_oof_probabilities(
    events: Subset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> np.ndarray:
    """Out-of-fold probabilities over the outer-train rows for one subset.

    Uses the SAME inner-CV folds as candidate scoring so thresholds are tuned on
    train data only and stay consistent with --inner-split.
    """
    columns = subset_feature_indices(events, event_indices)
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    for split_idx, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[inner_train_idx][:, columns])
        X_va = scaler.transform(X_train[inner_valid_idx][:, columns])
        model = make_xgb(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + split_idx,
        )
        model.fit(X_tr, y_train[inner_train_idx])
        oof[inner_valid_idx] = model.predict_proba(X_va)[:, 1]
    return oof


def _tuned_threshold_for_subset(
    events: Subset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    target_tpr: float,
) -> float:
    oof = compute_subset_oof_probabilities(
        events=events,
        X_train=X_train,
        y_train=y_train,
        event_indices=event_indices,
        inner_splits=inner_splits,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    covered = ~np.isnan(oof)
    return threshold_for_target_tpr(y_train[covered], oof[covered], target_tpr)


def tune_candidate_thresholds(
    candidates: Sequence[SubsetScore],
    X_train: np.ndarray,
    y_train: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_folds: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
    target_tpr: float,
    inner_split: str,
    train_group_values: Optional[np.ndarray],
) -> np.ndarray:
    """Train-only tuned decision thresholds, one per candidate.

    NEVER touches outer-test data. Builds OOF probabilities over the outer-train
    via the same split mode as the inner CV (--inner-split).
    """
    inner_splits = build_inner_splits(
        X_train=X_train,
        y_train=y_train,
        inner_folds=inner_folds,
        seed=random_state,
        inner_split=inner_split,
        train_group_values=train_group_values,
    )
    thresholds = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_tuned_threshold_for_subset)(
            events=candidate.events,
            X_train=X_train,
            y_train=y_train,
            event_indices=event_indices,
            inner_splits=inner_splits,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + idx,
            target_tpr=target_tpr,
        )
        for idx, candidate in enumerate(candidates)
    )
    return np.asarray(thresholds, dtype=np.float64)


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
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))

    both_classes = len(np.unique(y_true)) == 2
    # auprc / balanced_accuracy / mcc / roc-based metrics are undefined for a
    # single-class group (which happens for per-group oracle decompositions).
    auprc = float(average_precision_score(y_true, probabilities)) if both_classes else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, predictions)) if both_classes else 0.0
    mcc = float(matthews_corrcoef(y_true, predictions)) if both_classes else 0.0

    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "auprc": auprc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr_at_fpr_0.05": tpr_at_fpr(y_true, probabilities, 0.05),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_true, probabilities, 0.01),
    }


SCALAR_METRIC_KEYS = (
    "f1",
    "balanced_accuracy",
    "fpr",
    "mcc",
    "precision",
    "recall",
    "accuracy",
    "auprc",
)


def predictions_at_threshold(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Vectorised candidate-by-sample predictions at per-candidate thresholds."""
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1, 1)
    return (probabilities >= thresholds).astype(np.int32)


def route_group_oracle(
    y_true: np.ndarray,
    candidate_predictions: np.ndarray,
    group_labels: np.ndarray,
) -> np.ndarray:
    """Per-group oracle: pick ONE candidate per group, maximising group correctness.

    Models a static recommender that may pick at most one subset per group value
    (e.g. one subset per malware family). For each group, the candidate with the
    most correct predictions on that group's samples wins and is applied to every
    sample in the group. Ties break toward the lowest candidate index.
    Returns the routed prediction vector (one prediction per sample).
    """
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(candidate_predictions, dtype=int)
    group_labels = np.asarray(group_labels, dtype=object)

    routed = predictions[0].copy()
    correct = (predictions == y_true[np.newaxis, :]).astype(np.int64)
    for group in np.unique(group_labels):
        mask = group_labels == group
        per_candidate_correct = correct[:, mask].sum(axis=1)
        chosen = int(np.argmax(per_candidate_correct))  # ties -> lowest index
        routed[mask] = predictions[chosen][mask]
    return routed.astype(np.int32)


def route_global_oracle(
    y_true: np.ndarray,
    candidate_predictions: np.ndarray,
) -> np.ndarray:
    """Global (collapsed) oracle: single best candidate over ALL test samples."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(candidate_predictions, dtype=int)
    correct = (predictions == y_true[np.newaxis, :]).sum(axis=1)
    chosen = int(np.argmax(correct))  # ties -> lowest index
    return predictions[chosen].astype(np.int32)


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


GROUP_ORACLE_GRANULARITIES = (
    "global",
    "per_category",
    "per_family",
    "per_full_label",
    "per_sample",
)


def compute_group_oracle_metrics(
    y_test: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_probabilities: np.ndarray,
    candidate_scores: np.ndarray,
    global_index: int,
    test_metadata: Dict[str, List[str]],
    operating_point: str,
) -> List[Dict[str, object]]:
    """Per-group oracle decomposition at one operating point.

    For each granularity the oracle is restricted to picking ONE candidate per
    group value (except per_sample, the existing per-sample router). The gap
    between per_sample and per_family/per_full_label is headroom UNREACHABLE by
    a static family/binary recommender; the gap between per_family and global is
    the REACHABLE headroom.
    """
    group_sources = {
        "per_category": np.asarray(test_metadata["category"], dtype=object),
        "per_family": np.asarray(test_metadata["family"], dtype=object),
        "per_full_label": np.asarray(test_metadata["full_label"], dtype=object),
    }
    rows: List[Dict[str, object]] = []
    for granularity in GROUP_ORACLE_GRANULARITIES:
        if granularity == "global":
            routed = route_global_oracle(y_test, candidate_predictions)
        elif granularity == "per_sample":
            routed = route_candidate_oracle(
                y_true=y_test,
                candidate_predictions=candidate_predictions,
                candidate_probabilities=candidate_probabilities,
                candidate_scores=candidate_scores,
                global_candidate_index=global_index,
            ).predictions
        else:
            routed = route_group_oracle(
                y_test, candidate_predictions, group_sources[granularity]
            )
        # Probabilities are only used for auprc/roc; the global candidate's
        # probabilities are a stable reference across granularities.
        metrics = compute_metrics(
            y_test, routed, candidate_probabilities[global_index]
        )
        rows.append(
            {
                "granularity": granularity,
                "operating_point": operating_point,
                **metrics,
            }
        )
    return rows


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
    group_values: Optional[np.ndarray],
    metadata: Dict[str, List[str]],
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
    inner_split = args.inner_split
    train_group_values = (
        group_values[train_idx] if (group_values is not None and inner_split == "group")
        else None
    )
    test_metadata = {key: [values[i] for i in test_idx] for key, values in metadata.items()}
    print(
        f"[Fold {fold_number}/{args.outer_folds}] train={len(train_idx)} "
        f"test={len(test_idx)} inner_split={inner_split}",
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
        inner_split=inner_split,
        train_group_values=train_group_values,
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

    # Train-only tuned operating point: one threshold per candidate from OOF
    # probabilities over the outer-train. NEVER uses outer-test labels.
    print(
        f"    tuning thresholds (train-only, target TPR={args.target_tpr})...",
        flush=True,
    )
    tuned_thresholds = tune_candidate_thresholds(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        event_indices=event_indices,
        inner_folds=args.inner_folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + fold_index * 1000,
        n_jobs=args.jobs,
        target_tpr=args.target_tpr,
        inner_split=inner_split,
        train_group_values=train_group_values,
    )
    # candidate-by-sample predictions at the tuned per-candidate thresholds.
    candidate_predictions_tuned = predictions_at_threshold(
        candidate_probabilities, tuned_thresholds
    )

    candidate_scores = np.asarray([row.mean_auprc for row in candidates])
    global_index = 0

    # ── Legacy 0.5-threshold operating point (byte-for-byte reproducible) ──
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

    # ── Tuned-threshold operating point ──
    global_metrics_tuned = compute_metrics(
        y_test,
        candidate_predictions_tuned[global_index],
        candidate_probabilities[global_index],
    )
    routing_tuned = route_candidate_oracle(
        y_true=y_test,
        candidate_predictions=candidate_predictions_tuned,
        candidate_probabilities=candidate_probabilities,
        candidate_scores=candidate_scores,
        global_candidate_index=global_index,
    )
    oracle_metrics_tuned = compute_metrics(
        y_test,
        routing_tuned.predictions,
        routing_tuned.probabilities,
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
    # 2SMaRT tuned threshold: same train-only target-TPR rule on its own OOF.
    twosmart_threshold = _tuned_threshold_for_subset(
        events=twosmart_events,
        X_train=X_train,
        y_train=y_train,
        event_indices=event_indices,
        inner_splits=build_inner_splits(
            X_train=X_train,
            y_train=y_train,
            inner_folds=args.inner_folds,
            seed=args.seed + fold_index * 1000,
            inner_split=inner_split,
            train_group_values=train_group_values,
        ),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + fold_index,
        target_tpr=args.target_tpr,
    )
    twosmart_predictions_tuned = (
        twosmart_probabilities >= twosmart_threshold
    ).astype(np.int32)
    twosmart_metrics_tuned = compute_metrics(
        y_test,
        twosmart_predictions_tuned,
        twosmart_probabilities,
    )

    # ── Trivial majority-class baseline (predict training-majority for all) ──
    majority_class = int(np.bincount(y_train).argmax())
    majority_predictions = np.full(len(y_test), majority_class, dtype=np.int32)
    # Constant "probability" equal to the predicted class so auprc is well-defined.
    majority_probabilities = majority_predictions.astype(np.float32)
    majority_metrics = compute_metrics(
        y_test, majority_predictions, majority_probabilities
    )

    # ── Per-group oracle decomposition (the decisive experiment) ──
    group_oracle_rows: List[Dict[str, object]] = []
    group_oracle_rows.extend(
        compute_group_oracle_metrics(
            y_test=y_test,
            candidate_predictions=candidate_predictions,
            candidate_probabilities=candidate_probabilities,
            candidate_scores=candidate_scores,
            global_index=global_index,
            test_metadata=test_metadata,
            operating_point="0.5",
        )
    )
    group_oracle_rows.extend(
        compute_group_oracle_metrics(
            y_test=y_test,
            candidate_predictions=candidate_predictions_tuned,
            candidate_probabilities=candidate_probabilities,
            candidate_scores=candidate_scores,
            global_index=global_index,
            test_metadata=test_metadata,
            operating_point="tuned",
        )
    )
    for row in group_oracle_rows:
        row["fold"] = fold_number
    atomic_write_json(
        fold_dir / "group_oracle_metrics.json",
        {"fold": fold_number, "target_tpr": args.target_tpr, "rows": group_oracle_rows},
    )

    routed_events = [
        "|".join(candidates[idx].events) for idx in routing.candidate_indices
    ]
    routed_events_tuned = [
        "|".join(candidates[idx].events) for idx in routing_tuned.candidate_indices
    ]
    # Legacy columns are kept verbatim; tuned columns are appended.
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
            "global_beam_pred_tuned": candidate_predictions_tuned[global_index],
            "twosmart_pred_tuned": twosmart_predictions_tuned,
            "oracle_pred_tuned": routing_tuned.predictions,
            "oracle_candidate_index_tuned": routing_tuned.candidate_indices,
            "oracle_events_tuned": routed_events_tuned,
            "majority_pred": majority_predictions,
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
        "oracle_coverage_tuned": routing_tuned.coverage,
        "target_tpr": args.target_tpr,
        "global_beam_tuned_threshold": float(tuned_thresholds[global_index]),
        "twosmart_tuned_threshold": float(twosmart_threshold),
        "majority_class": majority_class,
        "metrics": {
            "Global-Beam": global_metrics,
            "Candidate-Oracle": oracle_metrics,
            "2SMaRT": twosmart_metrics,
            "Majority-Baseline": majority_metrics,
            "Global-Beam_tuned": global_metrics_tuned,
            "Candidate-Oracle_tuned": oracle_metrics_tuned,
            "2SMaRT_tuned": twosmart_metrics_tuned,
        },
        "group_oracle_metrics": group_oracle_rows,
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
    print(
        f"    F1[tuned]: Oracle={oracle_metrics_tuned['f1']:.4f} "
        f"Global-Beam={global_metrics_tuned['f1']:.4f} "
        f"2SMaRT={twosmart_metrics_tuned['f1']:.4f} "
        f"Majority={majority_metrics['f1']:.4f}",
        flush=True,
    )
    del candidate_predictions, candidate_probabilities, candidate_predictions_tuned
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
                        else summary.get("oracle_coverage_tuned", np.nan)
                        if strategy == "Candidate-Oracle_tuned"
                        else np.nan
                    ),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "fold_metrics.csv", fold_metrics)

    grouped = (
        fold_metrics.groupby("strategy")[list(SCALAR_METRIC_KEYS)]
        .agg(["mean", "std"])
        .round(6)
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    aggregate = grouped.reset_index()
    atomic_write_csv(output_dir / "aggregate_metrics.csv", aggregate)

    # ── Per-group oracle decomposition aggregate (the decisive experiment) ──
    group_rows: List[Dict[str, object]] = []
    for summary in summaries:
        for row in summary.get("group_oracle_metrics", []):
            group_rows.append(dict(row))
    if group_rows:
        group_frame = pd.DataFrame(group_rows)
        group_grouped = (
            group_frame.groupby(["granularity", "operating_point"])[list(SCALAR_METRIC_KEYS)]
            .agg(["mean", "std"])
            .round(6)
        )
        group_grouped.columns = [
            f"{metric}_{stat}" for metric, stat in group_grouped.columns
        ]
        group_aggregate = group_grouped.reset_index()
        atomic_write_csv(output_dir / "group_oracle_aggregate.csv", group_aggregate)
    else:
        group_aggregate = pd.DataFrame()

    payload = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "folds": list(summaries),
        "aggregate": aggregate.to_dict(orient="records"),
        "group_oracle_aggregate": group_aggregate.to_dict(orient="records"),
        "methodology": {
            "candidate_generation": "outer-training data only",
            "search_objective": "mean inner-CV AUCPR",
            "global_beam": "highest inner-CV candidate",
            "candidate_oracle": (
                "post-hoc label-assisted routing among retained candidates; "
                "explicitly non-deployable"
            ),
            "tuned_operating_point": (
                "per-candidate threshold from train-OOF probabilities at "
                f"target TPR={args.target_tpr}; outer-test labels never used to set it"
            ),
            "group_oracle": (
                "static-recommender ceiling per grouping granularity; "
                "per_sample-vs-per_family gap is unreachable headroom, "
                "per_family-vs-global gap is reachable headroom"
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
        "--inner-split",
        choices=("stratified", "group"),
        default=None,
        help=(
            "Inner-CV split for candidate scoring and threshold tuning. "
            "Defaults to 'group' when --group-column is set (so candidate "
            "selection is also family-disjoint), else 'stratified'."
        ),
    )
    parser.add_argument(
        "--target-tpr",
        type=float,
        default=0.95,
        help=(
            "Target true-positive rate for the train-only tuned operating point. "
            "The highest threshold meeting train-OOF TPR>=target is chosen."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/results/detection_aware_beam_oracle",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    # Default the inner split mode to match the outer split: group-disjoint inner
    # CV whenever the outer CV is grouped, so family leakage cannot re-enter via
    # candidate selection / threshold tuning.
    if args.inner_split is None:
        args.inner_split = "group" if args.group_column is not None else "stratified"
    if args.inner_split == "group" and args.group_column is None:
        parser.error("--inner-split group requires --group-column to be set")

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
    if not 0.0 < args.target_tpr <= 1.0:
        parser.error("--target-tpr must be in (0, 1]")
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
                group_values=group_values,
                metadata=metadata,
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

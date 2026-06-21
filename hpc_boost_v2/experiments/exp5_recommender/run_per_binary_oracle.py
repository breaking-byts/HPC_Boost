#!/usr/bin/env python3
"""exp5 — Per-binary detection-aware oracle gate on our collected traces.

This is the paper's §9.3 go/no-go gate (see exp5/PLAN.md). It reuses the
*validated* exp3 oracle primitives verbatim (scoring, candidate fitting,
threshold tuning, routing, honest metrics) so the comparison stays
methodologically identical, and adds the three things exp3 could not have:

  1. a loader for our `labeled_dataset.csv` (13 events measured in 4 separate-boot
     groups -> per-(sample,rep) *marginal* aggregation, each event taken only from
     the rows of its home group);
  2. the reframed routing ladder
        global -> per_family -> per_binary (DEPLOYABLE target) -> per_run (ceiling),
     where what was exp3's non-deployable per-sample ceiling becomes the
     per-binary *target* because our `sample` is a binary with repeated runs;
  3. a stability analysis across each binary's reps (the data exp3 never had).

Run:
  python -m experiments.exp5_recommender.run_per_binary_oracle \
      --dataset /path/to/labeled_dataset.csv --out experiments/exp5_recommender/results

It is also importable: `load_per_binary` + `run_gate` are used by the test
(`tests/test_per_binary_oracle_gate.py`) against a synthetic fixture so the whole
pipeline is verified before the real data is in hand.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- make `src` importable and load the exp3 oracle primitives by path --------
PKG_ROOT = Path(__file__).resolve().parents[2]  # .../hpc_boost_v2
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Import the validated exp3 oracle primitives by their CANONICAL module name
# (not importlib path-loading): joblib/loky workers re-import the parallelized
# functions by reference, so those functions must live in an importable module.
# `experiments` is a namespace package; exp3 is a regular subpackage. loky
# propagates the parent sys.path (incl. PKG_ROOT) to workers.
from experiments.exp3_detection_aware_oracle import run_beam_oracle as beam  # noqa: E402

# 13 guest-vPMU events, in build_dataset.py order. cycles anchors every group.
EVENTS_13: List[str] = [
    "cycles", "instructions", "branches", "branch-misses",
    "cache-references", "cache-misses", "L1-dcache-loads",
    "L1-dcache-load-misses", "L1-dcache-stores", "dTLB-loads",
    "dTLB-load-misses", "iTLB-loads", "iTLB-load-misses",
]

Subset = Tuple[str, ...]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def derive_event_groups(
    df: pd.DataFrame, events: Sequence[str]
) -> Dict[str, List[str]]:
    """Map each event -> the collection group(s) where it was actually measured.

    Because groups are separate boots and the table is zero-filled, an event is
    only real in rows of its home group. `cycles` is the anchor -> present in all
    groups. Falls back to all groups if an event is all-zero (should not happen).
    """
    all_groups = sorted(df["group"].astype(str).unique().tolist())
    home: Dict[str, List[str]] = {}
    for e in events:
        nz = sorted(df.loc[df[e] != 0, "group"].astype(str).unique().tolist())
        home[e] = nz if nz else all_groups
    return home


def load_per_binary(
    csv_path: str | Path, events: Sequence[str] = EVENTS_13
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """labeled_dataset.csv -> one marginal-feature row per (sample, rep).

    Each event's 6 statistics (exp3 `aggregate_trace`) are computed only over the
    rows of that event's home group, so the zero-fill from other groups never
    dilutes a marginal. Returns (X, y, samples, reps, families, feature_names),
    one row per (sample, rep).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in (["label", "sample", "rep", "group"] + list(events)) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    home = derive_event_groups(df, events)
    rows: List[Dict[str, float]] = []
    samples: List[str] = []
    reps: List[str] = []
    families: List[str] = []
    labels: List[int] = []

    for (sample, rep), grp in df.groupby(["sample", "rep"], sort=True):
        feat: Dict[str, float] = {}
        for e in events:
            vals = grp.loc[grp["group"].astype(str).isin(home[e]), e].to_numpy(
                dtype=np.float64
            )
            # aggregate_trace expects a frame of event columns; one event at a time
            # keeps each marginal tied to its own home-group rows.
            feat.update(beam.aggregate_trace(pd.DataFrame({e: vals})))
        rows.append(feat)
        samples.append(str(sample))
        reps.append(str(rep))
        families.append(str(grp["family"].iloc[0]) if "family" in grp else "?")
        labels.append(1 if str(grp["label"].iloc[0]).lower() == "malware" else 0)

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(frame.columns)
    X = frame.to_numpy(dtype=np.float32)
    return (
        X,
        np.asarray(labels, dtype=np.int32),
        np.asarray(samples, dtype=object),
        np.asarray(reps, dtype=object),
        np.asarray(families, dtype=object),
        feature_names,
    )


# ---------------------------------------------------------------------------
# Candidates + routing helpers
# ---------------------------------------------------------------------------
def build_candidates(
    events: Sequence[str], k: int = 4, max_candidates: Optional[int] = None, seed: int = 42
) -> List[Subset]:
    """All C(len(events), k) subsets (exhaustive; 715 for 13 events, k=4).

    `max_candidates` deterministically subsamples for fast tests only.
    """
    pool = [beam.canonical_subset(c) for c in itertools.combinations(sorted(events), k)]
    if max_candidates is not None and max_candidates < len(pool):
        rng = random.Random(seed)
        pool = sorted(rng.sample(pool, max_candidates))
    return pool


def group_oracle(
    y_true: np.ndarray, preds: np.ndarray, group_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[object, int]]:
    """Per-group oracle (matches exp3 route_group_oracle) returning the routed
    prediction vector AND the chosen candidate index per group (for divergence)."""
    y_true = np.asarray(y_true, dtype=int)
    preds = np.asarray(preds, dtype=int)
    groups = np.asarray(group_labels, dtype=object)
    routed = preds[0].copy()
    correct = (preds == y_true[np.newaxis, :]).astype(np.int64)
    choices: Dict[object, int] = {}
    for grp in np.unique(groups):
        mask = groups == grp
        chosen = int(np.argmax(correct[:, mask].sum(axis=1)))  # ties -> lowest idx
        routed[mask] = preds[chosen][mask]
        choices[grp] = chosen
    return routed.astype(np.int32), choices


# ---------------------------------------------------------------------------
# Fixed-subset selection scoring (used by recommend_fixed_subset.py)
# ---------------------------------------------------------------------------
# Defined here (an importable module, not __main__) so joblib/loky workers can
# re-import this function by reference when the recommender runs as `-m`.
def subset_oof_criteria(
    events: Subset, X: np.ndarray, y: np.ndarray, event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]], n_estimators: int,
    max_depth: int, random_state: int, target_tpr: float,
) -> Tuple[Subset, float, float, float]:
    """One inner-CV out-of-fold pass -> (events, AUCPR, balanced-acc@tuned-thr, thr).

    A *train-only* deployable selection score for a single fixed subset (no test
    labels). AUCPR is threshold-free; balanced accuracy is at the train-tuned
    target-TPR threshold (the honest deployable metric)."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    oof = beam.compute_subset_oof_probabilities(
        events=events, X_train=X, y_train=y, event_indices=event_indices,
        inner_splits=inner_splits, n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state,
    )
    covered = ~np.isnan(oof)
    yc, pc = y[covered], oof[covered]
    if len(np.unique(yc)) < 2:
        return (events, 0.0, 0.0, 0.5)
    thr = beam.threshold_for_target_tpr(yc, pc, target_tpr)
    aucpr = float(average_precision_score(yc, pc))
    balacc = float(balanced_accuracy_score(yc, (pc >= thr).astype(int)))
    return (events, aucpr, balacc, float(thr))


def score_subsets_criteria(
    subsets: Sequence[Subset], X: np.ndarray, y: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]], n_estimators: int,
    max_depth: int, random_state: int, n_jobs: int, target_tpr: float,
) -> List[Tuple[Subset, float, float, float]]:
    """Parallel train-only (AUCPR, balacc, thr) for every candidate subset."""
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(subset_oof_criteria)(
            events=s, X=X, y=y, event_indices=event_indices, inner_splits=inner_splits,
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            target_tpr=target_tpr,
        )
        for s in subsets
    )


# ---------------------------------------------------------------------------
# Per-fold candidate evaluation
# ---------------------------------------------------------------------------
def evaluate_fold(
    candidates: Sequence[Subset],
    X: np.ndarray,
    y: np.ndarray,
    samples: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    event_indices: Dict[str, np.ndarray],
    inner_folds: int,
    n_estimators: int,
    max_depth: int,
    seed: int,
    jobs: int,
    target_tpr: float,
) -> Dict[str, object]:
    """Score (inner-CV AUCPR, train-only), tune thresholds (train-only), fit on
    train, predict on test, at tuned operating point. Returns the candidate-by-
    test prediction matrix + ordering needed for routing."""
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te = X[test_idx]
    samples_tr = samples[train_idx]

    inner_splits = beam.build_inner_splits(
        X_train=X_tr, y_train=y_tr, inner_folds=inner_folds, seed=seed,
        inner_split="group", train_group_values=samples_tr,
    )
    scored = beam.score_subsets(
        subsets=list(candidates), X=X_tr, y=y_tr, event_indices=event_indices,
        inner_splits=inner_splits, n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=jobs,
    )
    scored = beam.prune_scored_subsets(scored, None)  # sort best-AUCPR first
    global_index = 0

    thresholds = beam.tune_candidate_thresholds(
        candidates=scored, X_train=X_tr, y_train=y_tr, event_indices=event_indices,
        inner_folds=inner_folds, n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=jobs, target_tpr=target_tpr, inner_split="group",
        train_group_values=samples_tr,
    )
    _, probs = beam.fit_all_candidates(
        candidates=scored, X_train=X_tr, y_train=y_tr, X_test=X_te,
        event_indices=event_indices, n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=jobs,
    )
    preds = beam.predictions_at_threshold(probs, thresholds)  # candidate x test

    # 2SMaRT baseline (Pearson event-mean vs label), fit on train, predict test.
    twosmart_events = beam.select_twosmart(X_tr, y_tr, list(event_indices.keys()), event_indices)
    ts_pred, _ = beam.fit_candidate_predict(
        events=twosmart_events, X_train=X_tr, y_train=y_tr, X_test=X_te,
        event_indices=event_indices, n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed,
    )
    return {
        "scored": scored,
        "global_index": global_index,
        "scores": np.asarray([s.mean_auprc for s in scored], dtype=float),
        "preds": preds,            # candidate x test, at tuned threshold
        "probs": probs,            # candidate x test
        "twosmart_pred": ts_pred,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Stability & divergence
# ---------------------------------------------------------------------------
def stability_from_fold(
    preds: np.ndarray, y_te: np.ndarray, samples_te: np.ndarray, X_te: np.ndarray,
    event_indices: Dict[str, np.ndarray],
) -> Dict[str, List[float]]:
    """Per-binary rep stability within a fold (a binary's reps are all in the same
    test fold, so candidate order is consistent for that binary)."""
    correct = (preds == y_te[np.newaxis, :]).astype(int)  # candidate x test
    jaccards: List[float] = []
    best_consistency: List[float] = []
    feat_cv: List[float] = []
    # mean-feature column per event (first of the 6 suffixes is "mean").
    mean_cols = [int(idx[0]) for idx in event_indices.values()]
    for s in np.unique(samples_te):
        cols = np.where(samples_te == s)[0]
        if len(cols) < 2:
            continue
        # rep-pair Jaccard of the set of subsets that are correct on each rep.
        correct_sets = [set(np.where(correct[:, c] == 1)[0]) for c in cols]
        pairj = []
        for i in range(len(correct_sets)):
            for j in range(i + 1, len(correct_sets)):
                a, b = correct_sets[i], correct_sets[j]
                union = len(a | b)
                pairj.append(1.0 if union == 0 else len(a & b) / union)
        if pairj:
            jaccards.append(float(np.mean(pairj)))
        # best subset (most reps correct) -> fraction of reps it is correct on.
        frac = correct[:, cols].mean(axis=1)
        best_consistency.append(float(frac.max()))
        # marginal boot-stability: median across events of CV of the event mean.
        sub = X_te[cols][:, mean_cols].astype(np.float64)
        mu = np.abs(sub.mean(axis=0))
        sd = sub.std(axis=0)
        cv = np.divide(sd, mu, out=np.zeros_like(sd), where=mu > 1e-12)
        feat_cv.append(float(np.median(cv)))
    return {"jaccard": jaccards, "best_consistency": best_consistency, "feature_cv": feat_cv}


# ---------------------------------------------------------------------------
# Gate driver
# ---------------------------------------------------------------------------
LADDER = ("global", "per_family", "per_binary", "per_run")


def run_gate(
    X: np.ndarray, y: np.ndarray, samples: np.ndarray, reps: np.ndarray,
    families: np.ndarray, feature_names: List[str], events: Sequence[str] = EVENTS_13,
    outer_folds: int = 5, inner_folds: int = 3, n_estimators: int = 100,
    max_depth: int = 4, seed: int = 42, jobs: int = 8, target_tpr: float = 0.95,
    max_candidates: Optional[int] = None,
) -> Dict[str, object]:
    """Run the per-binary oracle gate; return metrics ladder + stability +
    divergence. Candidate generation/scoring/thresholds are train-only."""
    event_indices = beam.build_event_indices(events, feature_names)
    candidates = build_candidates(events, k=4, max_candidates=max_candidates, seed=seed)

    n_samples = len(np.unique(samples))
    outer_folds = max(2, min(outer_folds, n_samples))
    outer_splits = beam.build_outer_splits(
        X=X, y=y, outer_folds=outer_folds, seed=seed, group_values=samples
    )

    pooled: Dict[str, List[np.ndarray]] = {name: [] for name in LADDER}
    pooled["twosmart"], pooled["majority"], pooled["best_global"] = [], [], []
    pooled_y: List[np.ndarray] = []
    pooled_prob_ref: List[np.ndarray] = []
    distinct_binary_subsets: set = set()
    n_binary_choices = 0
    stab: Dict[str, List[float]] = {"jaccard": [], "best_consistency": [], "feature_cv": []}

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
        fr = evaluate_fold(
            candidates, X, y, samples, train_idx, test_idx, event_indices,
            inner_folds, n_estimators, max_depth, seed + fold_i, jobs, target_tpr,
        )
        preds, probs = fr["preds"], fr["probs"]
        gidx, scores, scored = fr["global_index"], fr["scores"], fr["scored"]
        y_te = y[test_idx]
        samples_te, families_te = samples[test_idx], families[test_idx]

        # routing ladder
        routed = {}
        routed["global"] = beam.route_global_oracle(y_te, preds)
        routed["per_family"], _ = group_oracle(y_te, preds, families_te)
        routed["per_binary"], bin_choices = group_oracle(y_te, preds, samples_te)
        routed["per_run"] = beam.route_candidate_oracle(
            y_true=y_te, candidate_predictions=preds, candidate_probabilities=probs,
            candidate_scores=scores, global_candidate_index=gidx,
        ).predictions
        for name in LADDER:
            pooled[name].append(routed[name])

        # baselines on this fold
        pooled["twosmart"].append(fr["twosmart_pred"])
        pooled["majority"].append(np.ones_like(y_te))
        pooled["best_global"].append(preds[gidx])
        pooled_y.append(y_te)
        pooled_prob_ref.append(probs[gidx])

        # divergence: which subset each binary was routed to
        for _, idx in bin_choices.items():
            distinct_binary_subsets.add(scored[idx].events)
            n_binary_choices += 1

        # stability within this fold
        sfold = stability_from_fold(preds, y_te, samples_te, X[test_idx], event_indices)
        for kk in stab:
            stab[kk].extend(sfold[kk])

    yp = np.concatenate(pooled_y)
    prob_ref = np.concatenate(pooled_prob_ref)
    metrics = {
        name: beam.compute_metrics(yp, np.concatenate(pooled[name]), prob_ref)
        for name in list(LADDER) + ["twosmart", "majority", "best_global"]
    }

    f1 = {k: metrics[k]["f1"] for k in metrics}
    ba = {k: metrics[k]["balanced_accuracy"] for k in metrics}
    summary = {
        "n_rows": int(len(y)), "n_binaries": int(n_samples),
        "n_malware_binaries": int(len(np.unique(samples[y == 1]))),
        "n_benign_binaries": int(len(np.unique(samples[y == 0]))),
        "n_candidates": int(len(candidates)), "outer_folds": int(outer_folds),
        "headroom_per_binary_vs_global_f1": float(f1["per_binary"] - f1["global"]),
        "headroom_per_binary_vs_global_balacc": float(ba["per_binary"] - ba["global"]),
        "headroom_per_binary_vs_best_global_f1": float(f1["per_binary"] - f1["best_global"]),
        "reachable_fraction_balacc": _safe_frac(
            ba["per_binary"] - ba["global"], ba["per_run"] - ba["global"]
        ),
        "divergence_distinct_binary_subsets": int(len(distinct_binary_subsets)),
        "divergence_binary_choices": int(n_binary_choices),
        "stability_median_rep_jaccard": _median(stab["jaccard"]),
        "stability_median_best_consistency": _median(stab["best_consistency"]),
        "stability_median_feature_cv": _median(stab["feature_cv"]),
        "target_tpr": float(target_tpr),
    }
    return {"metrics": metrics, "summary": summary}


def _median(xs: List[float]) -> Optional[float]:
    return float(np.median(xs)) if xs else None


def _safe_frac(num: float, den: float) -> Optional[float]:
    return float(num / den) if abs(den) > 1e-9 else None


# ---------------------------------------------------------------------------
# CLI / reporting
# ---------------------------------------------------------------------------
def format_report(result: Dict[str, object]) -> str:
    s = result["summary"]
    m = result["metrics"]
    lines = ["=== exp5 per-binary oracle gate ==="]
    lines.append(
        f"rows={s['n_rows']} binaries={s['n_binaries']} "
        f"(malware={s['n_malware_binaries']}, benign={s['n_benign_binaries']}) "
        f"candidates={s['n_candidates']} outer_folds={s['outer_folds']}"
    )
    lines.append("")
    lines.append(f"{'strategy':<14}{'F1':>9}{'BalAcc':>9}{'FPR':>9}{'MCC':>9}")
    for name in ["majority", "twosmart", "best_global", *LADDER]:
        r = m[name]
        lines.append(
            f"{name:<14}{r['f1']:>9.4f}{r['balanced_accuracy']:>9.4f}"
            f"{r['fpr']:>9.4f}{r['mcc']:>9.4f}"
        )
    lines.append("")
    lines.append("GATE (paper §9.3):")
    lines.append(
        f"  HEADROOM  per_binary - global   = {s['headroom_per_binary_vs_global_f1']:+.4f} F1, "
        f"{s['headroom_per_binary_vs_global_balacc']:+.4f} balanced-acc  (bar: +0.03..0.05 F1)"
    )
    rf = s["reachable_fraction_balacc"]
    lines.append(
        f"            per_binary - best_global = {s['headroom_per_binary_vs_best_global_f1']:+.4f} F1; "
        f"reachable/ceiling balacc fraction = {('%.2f' % rf) if rf is not None else 'n/a'}"
    )
    lines.append(
        f"  DIVERGENCE distinct subsets routed = {s['divergence_distinct_binary_subsets']} "
        f"over {s['divergence_binary_choices']} binary choices  (>1 => conditional)"
    )
    lines.append(
        f"  STABILITY median rep-jaccard={_fmt(s['stability_median_rep_jaccard'])} "
        f"best-subset-consistency={_fmt(s['stability_median_best_consistency'])} "
        f"feature-CV={_fmt(s['stability_median_feature_cv'])}"
    )
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="labeled_dataset.csv")
    p.add_argument("--out", default=str(Path(__file__).parent / "results"))
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-tpr", type=float, default=0.95)
    p.add_argument("--max-candidates", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    X, y, samples, reps, families, feature_names = load_per_binary(args.dataset)
    print(
        f"Loaded {len(y)} (sample,rep) rows; "
        f"{len(np.unique(samples))} binaries; {int(y.sum())} malware rows.",
        flush=True,
    )
    result = run_gate(
        X, y, samples, reps, families, feature_names, events=EVENTS_13,
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed,
        jobs=args.jobs, target_tpr=args.target_tpr, max_candidates=args.max_candidates,
    )
    report = format_report(result)
    print("\n" + report, flush=True)

    out = Path(args.out)
    beam.atomic_write_json(out / "gate_summary.json", result["summary"])
    beam.atomic_write_json(out / "gate_metrics.json", result["metrics"])
    (out / "gate_report.txt").write_text(report + "\n")
    print(f"\nWrote results to {out}", flush=True)


if __name__ == "__main__":
    main()

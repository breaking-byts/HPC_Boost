#!/usr/bin/env python3
"""exp5 — Deployable FIXED-subset recommender (ship A).

The gate (run_per_binary_oracle.py) showed per-binary *conditioning* buys almost
nothing over a single well-chosen subset, while ~96% of the achievable headroom
is "pick one good fixed 4-event subset". This script builds that deployable
recommender: from training data only it selects ONE 4-event subset (by inner-CV
AUCPR or by inner-CV balanced accuracy), then evaluates it on held-out binaries
(sample-grouped CV) against the 2SMaRT baseline. It reports the recommended
subset, its cross-fold consistency, and a final all-data recommendation.

All selectors use a train-tuned target-TPR threshold (the honest deployable
operating point), so the comparison is apples-to-apples. The oracle single-subset
ceiling (0.977 balacc) is the reference from RESULTS.md.

Run:
  python -m experiments.exp5_recommender.recommend_fixed_subset \
      --dataset experiments/exp5_recommender/data/labeled_dataset_track1.csv \
      --out experiments/exp5_recommender/results
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from experiments.exp5_recommender.run_per_binary_oracle import (  # noqa: E402
    EVENTS_13, beam, build_candidates, load_per_binary, score_subsets_criteria,
    subset_oof_criteria,
)

SELECTORS = ("2smart", "fixed_aucpr", "fixed_balacc")


def _fit_predict_at_threshold(events, X_tr, y_tr, X_te, event_indices, n_estimators,
                              max_depth, seed, thr):
    _, probs = beam.fit_candidate_predict(
        events=events, X_train=X_tr, y_train=y_tr, X_test=X_te,
        event_indices=event_indices, n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed,
    )
    return (probs >= thr).astype(np.int32), probs


def _consistency(subset_list: List[Sequence[str]]) -> Dict[str, object]:
    counter = collections.Counter(tuple(s) for s in subset_list)
    top, n = counter.most_common(1)[0]
    return {"modal_subset": list(top), "modal_folds": int(n),
            "n_folds": len(subset_list), "distinct": len(counter)}


def run_recommender(
    X: np.ndarray, y: np.ndarray, samples: np.ndarray, families: np.ndarray,
    feature_names: List[str], events: Sequence[str] = EVENTS_13, outer_folds: int = 5,
    inner_folds: int = 3, n_estimators: int = 100, max_depth: int = 4, seed: int = 42,
    jobs: int = 8, target_tpr: float = 0.95, max_candidates: Optional[int] = None,
) -> Dict[str, object]:
    event_indices = beam.build_event_indices(events, feature_names)
    candidates = build_candidates(events, k=4, max_candidates=max_candidates, seed=seed)
    n_samples = len(np.unique(samples))
    outer_folds = max(2, min(outer_folds, n_samples))
    outer_splits = beam.build_outer_splits(X, y, outer_folds, seed, group_values=samples)

    pooled: Dict[str, List[np.ndarray]] = {k: [] for k in SELECTORS}
    pooled_y: List[np.ndarray] = []
    ref_probs: List[np.ndarray] = []
    chosen: Dict[str, List[Sequence[str]]] = {"fixed_aucpr": [], "fixed_balacc": []}

    for fi, (tr, te) in enumerate(outer_splits):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        samples_tr = samples[tr]
        inner = beam.build_inner_splits(
            X_train=X_tr, y_train=y_tr, inner_folds=inner_folds, seed=seed + fi,
            inner_split="group", train_group_values=samples_tr,
        )
        crit = score_subsets_criteria(
            candidates, X_tr, y_tr, event_indices, inner, n_estimators, max_depth,
            seed + fi, jobs, target_tpr,
        )
        aucpr_pick = max(crit, key=lambda r: r[1])   # (events, aucpr, balacc, thr)
        balacc_pick = max(crit, key=lambda r: r[2])
        ts_events = beam.select_twosmart(X_tr, y_tr, list(event_indices.keys()), event_indices)
        ts = subset_oof_criteria(ts_events, X_tr, y_tr, event_indices, inner,
                                 n_estimators, max_depth, seed + fi, target_tpr)

        picks = {"2smart": (ts_events, ts[3]),
                 "fixed_aucpr": (aucpr_pick[0], aucpr_pick[3]),
                 "fixed_balacc": (balacc_pick[0], balacc_pick[3])}
        for name, (ev, thr) in picks.items():
            preds, probs = _fit_predict_at_threshold(
                ev, X_tr, y_tr, X_te, event_indices, n_estimators, max_depth, seed + fi, thr)
            pooled[name].append(preds)
            if name == "fixed_balacc":
                ref_probs.append(probs)
        chosen["fixed_aucpr"].append(aucpr_pick[0])
        chosen["fixed_balacc"].append(balacc_pick[0])
        pooled_y.append(y_te)

    yp = np.concatenate(pooled_y)
    refp = np.concatenate(ref_probs)
    metrics = {k: beam.compute_metrics(yp, np.concatenate(pooled[k]), refp) for k in SELECTORS}

    # Final all-data recommendation (the subset to actually deploy).
    inner_all = beam.build_inner_splits(
        X_train=X, y_train=y, inner_folds=inner_folds, seed=seed,
        inner_split="group", train_group_values=samples)
    crit_all = score_subsets_criteria(
        candidates, X, y, event_indices, inner_all, n_estimators, max_depth, seed, jobs, target_tpr)
    final_balacc = max(crit_all, key=lambda r: r[2])
    final_aucpr = max(crit_all, key=lambda r: r[1])

    summary = {
        "n_binaries": int(n_samples), "n_candidates": int(len(candidates)),
        "outer_folds": int(outer_folds), "target_tpr": float(target_tpr),
        "oracle_single_subset_ceiling_balacc": 0.9773,  # from RESULTS.md (reference)
        "consistency_fixed_aucpr": _consistency(chosen["fixed_aucpr"]),
        "consistency_fixed_balacc": _consistency(chosen["fixed_balacc"]),
        "recommended_subset": list(final_balacc[0]),
        "recommended_subset_inner_balacc": float(final_balacc[2]),
        "recommended_subset_aucpr_selection": list(final_aucpr[0]),
        "gain_fixed_balacc_vs_2smart_f1": float(
            metrics["fixed_balacc"]["f1"] - metrics["2smart"]["f1"]),
        "gain_fixed_balacc_vs_2smart_balacc": float(
            metrics["fixed_balacc"]["balanced_accuracy"] - metrics["2smart"]["balanced_accuracy"]),
    }
    return {"metrics": metrics, "summary": summary}


def format_report(result: Dict[str, object]) -> str:
    m, s = result["metrics"], result["summary"]
    lines = ["=== exp5 deployable fixed-subset recommender ==="]
    lines.append(f"binaries={s['n_binaries']} candidates={s['n_candidates']} "
                 f"outer_folds={s['outer_folds']} op=tuned@TPR{s['target_tpr']}")
    lines.append("")
    lines.append(f"{'selector':<14}{'F1':>9}{'BalAcc':>9}{'FPR':>9}{'MCC':>9}")
    for name in SELECTORS:
        r = m[name]
        lines.append(f"{name:<14}{r['f1']:>9.4f}{r['balanced_accuracy']:>9.4f}"
                     f"{r['fpr']:>9.4f}{r['mcc']:>9.4f}")
    lines.append(f"{'(oracle ceil)':<14}{'-':>9}{s['oracle_single_subset_ceiling_balacc']:>9.4f}"
                 f"{'-':>9}{'-':>9}  <- non-deployable reference (RESULTS.md)")
    lines.append("")
    lines.append(f"GAIN fixed_balacc vs 2SMaRT: "
                 f"{s['gain_fixed_balacc_vs_2smart_f1']:+.4f} F1, "
                 f"{s['gain_fixed_balacc_vs_2smart_balacc']:+.4f} balanced-acc")
    lines.append(f"RECOMMENDED SUBSET (deploy): {s['recommended_subset']}  "
                 f"(inner balacc {s['recommended_subset_inner_balacc']:.4f})")
    ca, cb = s["consistency_fixed_aucpr"], s["consistency_fixed_balacc"]
    lines.append(f"cross-fold consistency: balacc-select modal {cb['modal_folds']}/{cb['n_folds']} "
                 f"({cb['distinct']} distinct); aucpr-select modal {ca['modal_folds']}/{ca['n_folds']} "
                 f"({ca['distinct']} distinct)")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
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
    print(f"Loaded {len(y)} rows; {len(np.unique(samples))} binaries.", flush=True)
    result = run_recommender(
        X, y, samples, families, feature_names, events=EVENTS_13,
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed,
        jobs=args.jobs, target_tpr=args.target_tpr, max_candidates=args.max_candidates)
    report = format_report(result)
    print("\n" + report, flush=True)
    out = Path(args.out)
    beam.atomic_write_json(out / "recommender_summary.json", result["summary"])
    beam.atomic_write_json(out / "recommender_metrics.json", result["metrics"])
    (out / "recommender_report.txt").write_text(report + "\n")
    print(f"\nWrote results to {out}", flush=True)


if __name__ == "__main__":
    main()

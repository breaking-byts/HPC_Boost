#!/usr/bin/env python3
"""Compute honest, reviewer-grade metrics from existing oracle result artifacts.

This script is intentionally **standard-library only** (csv, json, glob, math,
statistics, argparse). It does NOT use pandas or numpy so it runs on machines
without the scientific stack.

It reads the read-only result artifacts produced by ``run_beam_oracle.py`` for
two cross-validation regimes and recomputes pooled (not fold-averaged) metrics
for every routing strategy, plus the trivial always-malware baseline, a
fold-mean vs pooled reconciliation, the GroupKFold family structure, and a
capped candidate-pool oracle curve.

Why pooled metrics matter
--------------------------
The class distribution is heavily skewed toward malware (2884 malware vs 586
benign, ~83% positive). On such data the F1 score is *majority-class inflated*:
a degenerate classifier that always predicts "malware" already scores F1 ~0.91.
F1 therefore cannot distinguish a real detector from a trivial baseline. The
**honest** summary metric here is balanced accuracy (mean of recall and
specificity), reported alongside FPR and MCC. We compute the trivial baseline
explicitly so the inflation is visible in every table.

Expected layout under --results-root:
    detection_aware_beam_oracle/   (stratified run)
        aggregate_metrics.csv, fold_metrics.csv, fold_*/predictions.csv
    group_kfold_oracle/            (family-disjoint run)
        aggregate_metrics.csv, fold_metrics.csv, all_predictions.csv,
        fold_*/summary.json

Usage
-----
    python3 analyze_honest_metrics.py \
        --results-root oracle_results/data/processed/results \
        --out-json     outputs/honest_metrics.json \
        --out-md       outputs/honest_metrics_tables.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from typing import Dict, List, Tuple

# Strategy display order used in all tables.
STRATEGIES = ["2SMaRT", "Global-Beam", "Candidate-Oracle"]

# Maps a strategy display name to its prediction column in predictions.csv.
PRED_COLUMN = {
    "2SMaRT": "twosmart_pred",
    "Global-Beam": "global_beam_pred",
    "Candidate-Oracle": "oracle_pred",
}

# Candidate-pool sizes for the capped-oracle curve.
DEFAULT_POOL_SIZES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500]


# --------------------------------------------------------------------------- #
# Core metric helpers
# --------------------------------------------------------------------------- #
def confusion_from_pairs(pairs: List[Tuple[int, int]]) -> Dict[str, int]:
    """Return TP/FP/TN/FN counts from (y_true, y_pred) integer pairs.

    Positive class is malware (label 1).
    """
    tp = fp = tn = fn = 0
    for y_true, y_pred in pairs:
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        else:  # y_true == 1 and y_pred == 0
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_confusion(cm: Dict[str, int]) -> Dict[str, float]:
    """Derive the full metric suite from a TP/FP/TN/FN confusion dict."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # TNR
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    balanced_accuracy = (recall + specificity) / 2

    # Matthews correlation coefficient with a numerically safe denominator.
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def trivial_majority_baseline(n_pos: int, n_neg: int) -> Dict[str, float]:
    """Metrics for the degenerate 'always predict malware (1)' classifier.

    Every malware sample becomes a TP, every benign sample becomes a FP.
    By construction recall=1, FPR=1, balanced accuracy=0.5, MCC=0.
    """
    cm = {"tp": n_pos, "fp": n_neg, "tn": 0, "fn": 0}
    out = metrics_from_confusion(cm)
    out["strategy"] = "Trivial (always malware)"
    return out


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_stratified_rows(strat_dir: str) -> List[Dict[str, str]]:
    """Load and concatenate every fold_*/predictions.csv under the strat dir."""
    rows: List[Dict[str, str]] = []
    paths = sorted(glob.glob(os.path.join(strat_dir, "fold_*", "predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"No fold_*/predictions.csv under {strat_dir}")
    for path in paths:
        fold_name = os.path.basename(os.path.dirname(path))
        for row in read_csv_dicts(path):
            row["fold_dir"] = fold_name
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# (1) Pooled confusion + metrics per strategy
# --------------------------------------------------------------------------- #
def pooled_strategy_metrics(
    rows: List[Dict[str, str]],
    pred_columns: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Pool all samples across folds and compute metrics per strategy."""
    out: Dict[str, Dict[str, float]] = {}
    for strategy, col in pred_columns.items():
        pairs = [(int(r["y_true"]), int(r[col])) for r in rows]
        cm = confusion_from_pairs(pairs)
        m = metrics_from_confusion(cm)
        m["strategy"] = strategy
        out[strategy] = m
    return out


def class_counts(rows: List[Dict[str, str]]) -> Tuple[int, int]:
    """Return (n_malware, n_benign) over the pooled rows."""
    n_pos = sum(1 for r in rows if int(r["y_true"]) == 1)
    n_neg = sum(1 for r in rows if int(r["y_true"]) == 0)
    return n_pos, n_neg


# --------------------------------------------------------------------------- #
# (3) Fold-mean vs pooled reconciliation
# --------------------------------------------------------------------------- #
def fold_mean_f1(fold_metrics_path: str) -> Dict[str, Dict[str, float]]:
    """Read per-fold F1 from fold_metrics.csv and return mean/std per strategy.

    This intentionally re-derives the fold mean from the per-fold CSV (rather
    than trusting aggregate_metrics.csv) so the reconciliation is fully
    reproducible. Both should agree.
    """
    rows = read_csv_dicts(fold_metrics_path)
    by_strategy: Dict[str, List[float]] = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(float(r["f1"]))
    out: Dict[str, Dict[str, float]] = {}
    for strategy, vals in by_strategy.items():
        out[strategy] = {
            "f1_fold_mean": statistics.fmean(vals),
            "f1_fold_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n_folds": len(vals),
        }
    return out


def read_aggregate_f1(aggregate_path: str) -> Dict[str, float]:
    """Read the f1_mean column straight from aggregate_metrics.csv."""
    out: Dict[str, float] = {}
    for r in read_csv_dicts(aggregate_path):
        out[r["strategy"]] = float(r["f1_mean"])
    return out


# --------------------------------------------------------------------------- #
# (4) GroupKFold family structure
# --------------------------------------------------------------------------- #
def family_structure(all_predictions_path: str) -> Dict[str, object]:
    """Summarize family structure from the GroupKFold all_predictions.csv."""
    rows = read_csv_dicts(all_predictions_path)
    fam_labels: Dict[str, set] = {}
    n_pos = n_neg = 0
    for r in rows:
        fam = r["family"]
        y = int(r["y_true"])
        fam_labels.setdefault(fam, set()).add(y)
        if y == 1:
            n_pos += 1
        else:
            n_neg += 1

    benign_fams = sorted(f for f, labs in fam_labels.items() if 0 in labs)
    malware_fams = sorted(f for f, labs in fam_labels.items() if 1 in labs)
    mixed_fams = sorted(f for f, labs in fam_labels.items() if len(labs) > 1)

    return {
        "total_samples": len(rows),
        "n_malware": n_pos,
        "n_benign": n_neg,
        "total_families": len(fam_labels),
        "n_benign_families": len(benign_fams),
        "n_malware_families": len(malware_fams),
        "n_mixed_families": len(mixed_fams),
        "benign_family_ids": benign_fams,
        "malware_family_ids": malware_fams,
        "mixed_family_ids": mixed_fams,
    }


def gkf_per_fold_counts(gkf_dir: str) -> List[Dict[str, object]]:
    """Per-fold malware/benign test counts for GroupKFold.

    Prefers the explicit ``test_malware``/``test_benign`` fields in each
    fold_*/summary.json; falls back to deriving them from the
    Candidate-Oracle confusion (tp+fn malware, fp+tn benign) in
    fold_metrics.csv when summaries are unavailable.
    """
    out: List[Dict[str, object]] = []

    summary_paths = sorted(glob.glob(os.path.join(gkf_dir, "fold_*", "summary.json")))
    if summary_paths:
        for path in summary_paths:
            with open(path) as handle:
                s = json.load(handle)
            out.append(
                {
                    "fold": s.get("fold"),
                    "test_samples": s.get("test_samples"),
                    "test_malware": s.get("test_malware"),
                    "test_benign": s.get("test_benign"),
                    "test_families": s.get("test_families"),
                    "train_families": s.get("train_families"),
                    "source": "summary.json",
                }
            )
        return out

    # Fallback: derive from the Candidate-Oracle confusion in fold_metrics.csv.
    fold_metrics = read_csv_dicts(os.path.join(gkf_dir, "fold_metrics.csv"))
    by_fold: Dict[str, Dict[str, str]] = {}
    for r in fold_metrics:
        if r["strategy"] == "Candidate-Oracle":
            by_fold[r["fold"]] = r
    for fold in sorted(by_fold, key=int):
        r = by_fold[fold]
        tp, fp, tn, fn = (int(r[k]) for k in ("tp", "fp", "tn", "fn"))
        out.append(
            {
                "fold": int(fold),
                "test_samples": tp + fp + tn + fn,
                "test_malware": tp + fn,
                "test_benign": fp + tn,
                "test_families": None,
                "train_families": None,
                "source": "fold_metrics.csv (derived)",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# (5) Capped-oracle curve (stratified run)
# --------------------------------------------------------------------------- #
def capped_oracle_curve(
    rows: List[Dict[str, str]],
    pool_sizes: List[int],
) -> List[Dict[str, float]]:
    """Recompute Candidate-Oracle metrics restricted to candidates index < P.

    Logic (mirrors run_beam_oracle.py semantics, with the in-pool fallback):

      * ``oracle_candidate_index`` is the lowest-index correct candidate; the
        candidate pool is ordered by training score (priority order), so the
        global candidate is index 0 and is always inside any pool.
      * For pool size P == 1, the only available candidate is the global beam
        candidate, so the prediction is ``global_beam_pred``.
      * For P > 1, if ``oracle_candidate_index < P`` the lowest correct
        candidate is inside the pool, so we take ``oracle_pred`` (correct).
      * Otherwise no top-P candidate is correct; we fall back to the global
        candidate (index 0, always in pool), i.e. ``global_beam_pred``.

    This produces a monotone-improving curve from the Global-Beam operating
    point (P=1) up to the full Candidate-Oracle (P >= max retained index).
    """
    curve: List[Dict[str, float]] = []
    for P in pool_sizes:
        pairs: List[Tuple[int, int]] = []
        for r in rows:
            y_true = int(r["y_true"])
            idx = int(r["oracle_candidate_index"])
            if P == 1:
                y_pred = int(r["global_beam_pred"])
            elif idx < P:
                y_pred = int(r["oracle_pred"])
            else:
                y_pred = int(r["global_beam_pred"])
            pairs.append((y_true, y_pred))
        cm = confusion_from_pairs(pairs)
        m = metrics_from_confusion(cm)
        m["pool_size"] = P
        m["errors"] = cm["fp"] + cm["fn"]
        curve.append(m)
    return curve


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def pooled_table_md(
    pooled: Dict[str, Dict[str, float]],
    trivial: Dict[str, float],
    caption: str,
) -> str:
    header = (
        "| Strategy | TP | FP | TN | FN | Precision | Recall | F1 | "
        "FPR | Bal. Acc | MCC |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = ""
    for strat in STRATEGIES:
        m = pooled[strat]
        body += (
            f"| {strat} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | "
            f"{_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | "
            f"{_f(m['fpr'])} | {_f(m['balanced_accuracy'])} | {_f(m['mcc'])} |\n"
        )
    t = trivial
    body += (
        f"| _{t['strategy']}_ | {t['tp']} | {t['fp']} | {t['tn']} | {t['fn']} | "
        f"{_f(t['precision'])} | {_f(t['recall'])} | {_f(t['f1'])} | "
        f"{_f(t['fpr'])} | {_f(t['balanced_accuracy'])} | {_f(t['mcc'])} |\n"
    )
    return f"{caption}\n\n{header}{body}"


def reconciliation_table_md(
    strat_recon: Dict[str, Dict[str, float]],
    gkf_recon: Dict[str, Dict[str, float]],
    caption: str,
) -> str:
    header = (
        "| Run | Strategy | Fold-mean F1 | Pooled F1 | Difference |\n"
        "|---|---|---:|---:|---:|\n"
    )
    body = ""
    for run_name, recon in (
        ("Stratified", strat_recon),
        ("GroupKFold", gkf_recon),
    ):
        for strat in STRATEGIES:
            r = recon[strat]
            diff = r["f1_fold_mean"] - r["f1_pooled"]
            body += (
                f"| {run_name} | {strat} | {_f(r['f1_fold_mean'])} | "
                f"{_f(r['f1_pooled'])} | {diff:+.4f} |\n"
            )
    return f"{caption}\n\n{header}{body}"


def family_table_md(
    fam: Dict[str, object],
    per_fold: List[Dict[str, object]],
    caption: str,
) -> str:
    summary = (
        "| Quantity | Value |\n"
        "|---|---:|\n"
        f"| Total samples | {fam['total_samples']} |\n"
        f"| Malware samples | {fam['n_malware']} |\n"
        f"| Benign samples | {fam['n_benign']} |\n"
        f"| Total distinct families | {fam['total_families']} |\n"
        f"| Distinct benign (y=0) families | {fam['n_benign_families']} |\n"
        f"| Distinct malware (y=1) families | {fam['n_malware_families']} |\n"
        f"| Families with both labels (mixed) | {fam['n_mixed_families']} |\n"
    )
    fold_header = (
        "\n| Fold | Test samples | Test malware | Test benign | "
        "Test families |\n"
        "|---:|---:|---:|---:|---:|\n"
    )
    fold_body = ""
    for r in per_fold:
        fold_body += (
            f"| {r['fold']} | {r['test_samples']} | {r['test_malware']} | "
            f"{r['test_benign']} | {r['test_families']} |\n"
        )
    return f"{caption}\n\n{summary}{fold_header}{fold_body}"


def curve_table_md(curve: List[Dict[str, float]], caption: str) -> str:
    header = (
        "| Pool size P | F1 | Precision | Recall | Accuracy | FPR | "
        "Bal. Acc | Errors | TP | FP | TN | FN |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = ""
    for m in curve:
        body += (
            f"| {m['pool_size']} | {_f(m['f1'])} | {_f(m['precision'])} | "
            f"{_f(m['recall'])} | {_f(m['accuracy'])} | {_f(m['fpr'])} | "
            f"{_f(m['balanced_accuracy'])} | {m['errors']} | {m['tp']} | "
            f"{m['fp']} | {m['tn']} | {m['fn']} |\n"
        )
    return f"{caption}\n\n{header}{body}"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_report(results_root: str, pool_sizes: List[int]) -> Dict[str, object]:
    strat_dir = os.path.join(results_root, "detection_aware_beam_oracle")
    gkf_dir = os.path.join(results_root, "group_kfold_oracle")

    # ---- Stratified run ----
    strat_rows = load_stratified_rows(strat_dir)
    strat_pooled = pooled_strategy_metrics(strat_rows, PRED_COLUMN)
    s_pos, s_neg = class_counts(strat_rows)
    strat_trivial = trivial_majority_baseline(s_pos, s_neg)

    strat_fold = fold_mean_f1(os.path.join(strat_dir, "fold_metrics.csv"))
    strat_agg = read_aggregate_f1(os.path.join(strat_dir, "aggregate_metrics.csv"))
    strat_recon = {
        s: {
            "f1_fold_mean": strat_fold[s]["f1_fold_mean"],
            "f1_fold_mean_from_aggregate_csv": strat_agg.get(s),
            "f1_pooled": strat_pooled[s]["f1"],
            "n_folds": strat_fold[s]["n_folds"],
        }
        for s in STRATEGIES
    }

    curve = capped_oracle_curve(strat_rows, pool_sizes)

    # ---- GroupKFold run ----
    gkf_pred_columns = {
        "2SMaRT": "twosmart_pred",
        "Global-Beam": "global_beam_pred",
        "Candidate-Oracle": "oracle_pred",
    }
    gkf_rows = read_csv_dicts(os.path.join(gkf_dir, "all_predictions.csv"))
    gkf_pooled = pooled_strategy_metrics(gkf_rows, gkf_pred_columns)
    g_pos, g_neg = class_counts(gkf_rows)
    gkf_trivial = trivial_majority_baseline(g_pos, g_neg)

    gkf_fold = fold_mean_f1(os.path.join(gkf_dir, "fold_metrics.csv"))
    gkf_agg = read_aggregate_f1(os.path.join(gkf_dir, "aggregate_metrics.csv"))
    gkf_recon = {
        s: {
            "f1_fold_mean": gkf_fold[s]["f1_fold_mean"],
            "f1_fold_mean_from_aggregate_csv": gkf_agg.get(s),
            "f1_pooled": gkf_pooled[s]["f1"],
            "n_folds": gkf_fold[s]["n_folds"],
        }
        for s in STRATEGIES
    }

    fam = family_structure(os.path.join(gkf_dir, "all_predictions.csv"))
    gkf_per_fold = gkf_per_fold_counts(gkf_dir)

    return {
        "meta": {
            "results_root": os.path.abspath(results_root),
            "note": (
                "F1 is majority-class inflated on this ~83% positive data; "
                "balanced accuracy (mean of recall and specificity) is the "
                "honest summary metric. The trivial always-malware baseline "
                "scores F1 ~0.91 and balanced accuracy exactly 0.50."
            ),
            "positive_class": "malware (label 1)",
        },
        "stratified": {
            "class_counts": {"malware": s_pos, "benign": s_neg},
            "pooled": strat_pooled,
            "trivial_baseline": strat_trivial,
            "fold_vs_pooled": strat_recon,
            "capped_oracle_curve": curve,
        },
        "group_kfold": {
            "class_counts": {"malware": g_pos, "benign": g_neg},
            "pooled": gkf_pooled,
            "trivial_baseline": gkf_trivial,
            "fold_vs_pooled": gkf_recon,
            "family_structure": fam,
            "per_fold_counts": gkf_per_fold,
        },
    }


def write_markdown(report: Dict[str, object], path: str) -> None:
    strat = report["stratified"]
    gkf = report["group_kfold"]

    parts: List[str] = []
    parts.append("# HPC-Boost Honest Metrics\n")
    parts.append(report["meta"]["note"] + "\n")

    parts.append(
        pooled_table_md(
            strat["pooled"],
            strat["trivial_baseline"],
            "## (a) Stratified 5-fold CV: pooled metrics\n\n"
            "_Caption: Pooled (sample-level) confusion and metrics across all "
            "5 folds for the stratified run. F1 is **majority-class inflated** "
            "(the trivial always-malware row scores F1 ~0.91); **balanced "
            "accuracy** and MCC are the honest metrics. Positive class = "
            "malware._",
        )
    )
    parts.append(
        pooled_table_md(
            gkf["pooled"],
            gkf["trivial_baseline"],
            "## (b) GroupKFold (family-disjoint) CV: pooled metrics\n\n"
            "_Caption: Pooled metrics for the family-disjoint GroupKFold run, "
            "where test families never appear in training. Same caveat: F1 is "
            "inflated by the ~83% malware prior; read **balanced accuracy** "
            "and MCC. Positive class = malware._",
        )
    )
    parts.append(
        reconciliation_table_md(
            strat["fold_vs_pooled"],
            gkf["fold_vs_pooled"],
            "## (c) Fold-mean vs pooled F1 reconciliation\n\n"
            "_Caption: Fold-averaged F1 (mean of per-fold F1) vs pooled "
            "(sample-level) F1. Differences arise because folds have unequal "
            "sizes and difficulty; the unweighted fold mean over-weights small "
            "or easy folds. Pooled F1 is the sample-level ground truth._",
        )
    )
    parts.append(
        family_table_md(
            gkf["family_structure"],
            gkf["per_fold_counts"],
            "## (d) GroupKFold family structure\n\n"
            "_Caption: Family composition driving the GroupKFold splits. All "
            "benign samples belong to a single family, so family-disjoint CV "
            "places that entire benign family in one test fold; the other folds "
            "see very few or zero benign test samples, which is why per-fold "
            "specificity (and thus balanced accuracy) is volatile._",
        )
    )
    parts.append(
        curve_table_md(
            strat["capped_oracle_curve"],
            "## (e) Capped candidate-pool oracle curve (stratified run)\n\n"
            "_Caption: Candidate-Oracle metrics when the oracle may route only "
            "among the top-P retained candidates (ordered by training score). "
            "P=1 reproduces the Global-Beam operating point; the curve rises to "
            "the full oracle as P grows. F1 is inflated throughout; track "
            "balanced accuracy and errors._",
        )
    )

    with open(path, "w") as handle:
        handle.write("\n\n".join(parts).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute honest pooled metrics, trivial baseline, fold/pooled "
            "reconciliation, family structure, and capped-oracle curve from "
            "existing oracle result artifacts (stdlib only)."
        )
    )
    parser.add_argument(
        "--results-root",
        default="oracle_results/data/processed/results",
        help=(
            "Directory containing detection_aware_beam_oracle/ and "
            "group_kfold_oracle/ subdirectories."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="outputs/honest_metrics.json",
        help="Path to write the computed JSON.",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/honest_metrics_tables.md",
        help="Path to write the paper-ready Markdown tables.",
    )
    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_POOL_SIZES,
        help="Candidate pool sizes for the capped-oracle curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.results_root, args.pool_sizes)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(report, handle, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    write_markdown(report, args.out_md)

    # Human-readable console summary of the headline honest numbers.
    def _line(tag: str, m: Dict[str, float]) -> str:
        return (
            f"  {tag:<24} F1={m['f1']:.4f}  FPR={m['fpr']:.4f}  "
            f"BalAcc={m['balanced_accuracy']:.4f}  MCC={m['mcc']:.4f}  "
            f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']})"
        )

    print("=" * 78)
    print("STRATIFIED (pooled):")
    for s in STRATEGIES:
        print(_line(s, report["stratified"]["pooled"][s]))
    print(_line("Trivial always-malware", report["stratified"]["trivial_baseline"]))
    print("-" * 78)
    print("GROUP K-FOLD (pooled, family-disjoint):")
    for s in STRATEGIES:
        print(_line(s, report["group_kfold"]["pooled"][s]))
    print(_line("Trivial always-malware", report["group_kfold"]["trivial_baseline"]))
    print("-" * 78)
    fam = report["group_kfold"]["family_structure"]
    print(
        f"FAMILIES: total={fam['total_families']}  "
        f"benign={fam['n_benign_families']}  malware={fam['n_malware_families']}"
    )
    print("=" * 78)
    print(f"Wrote JSON -> {args.out_json}")
    print(f"Wrote MD   -> {args.out_md}")


if __name__ == "__main__":
    main()

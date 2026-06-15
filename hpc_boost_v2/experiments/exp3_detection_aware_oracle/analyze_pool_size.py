#!/usr/bin/env python3
"""Analyze Candidate-Oracle performance as a function of candidate-pool size.

This script uses only fold-level `predictions.csv` files emitted by
run_beam_oracle.py. Candidate indices are sorted by training inner-CV AUCPR in
the original experiment, so restricting to index < P simulates an oracle that
may route only among the top-P retained candidates.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_POOL_SIZES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500]


def load_rows(results_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(results_dir.glob("fold_*/predictions.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["fold_dir"] = path.parent.name
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No fold_*/predictions.csv files under {results_dir}")
    return rows


def compute_metrics(rows: Iterable[Dict[str, str]], pool_size: int) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        y_true = int(row["y_true"])

        if pool_size == 1:
            y_pred = int(row["global_beam_pred"])
        elif (
            int(row["oracle_candidate_index"]) < pool_size
            and int(row["oracle_pred"]) == y_true
        ):
            y_pred = y_true
        else:
            # The full oracle chooses the highest-ranked correct candidate.
            # If that candidate is outside the top-P pool, no top-P candidate
            # was correct. For binary metrics, force an incorrect prediction.
            y_pred = 1 - y_true

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (fp + tn) if (fp + tn) else 0.0
    balanced_accuracy = (recall + tnr) / 2

    return {
        "pool_size": pool_size,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "errors": fp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute capped-pool Candidate-Oracle metrics."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing fold_*/predictions.csv files.",
    )
    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_POOL_SIZES,
        help="Candidate pool sizes to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    rows = load_rows(results_dir)
    metrics = [compute_metrics(rows, pool_size) for pool_size in args.pool_sizes]

    fieldnames = [
        "pool_size",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "fpr",
        "balanced_accuracy",
        "errors",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", newline="")
        close_handle = True
    else:
        import sys

        handle = sys.stdout
        close_handle = False

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row[key] for key in fieldnames})
    finally:
        if close_handle:
            handle.close()


if __name__ == "__main__":
    main()

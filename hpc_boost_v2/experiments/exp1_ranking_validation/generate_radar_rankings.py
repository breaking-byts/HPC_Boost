from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.ranker import HPCBoostRanker
from src.utils.data_loader import RadarDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-sample HPC event rankings for the RaDaR dataset."
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional number of samples to rank. Use 50 for a quick validation run.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="CSV read chunk size.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/rankings/radar",
        help="Output directory for ranking CSV files.",
    )
    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        metavar=("SPREAD", "TREND", "CORRELATION", "STATIONARITY"),
        default=[0.25, 0.25, 0.25, 0.25],
        help="Four ranking weights. Default: equal weights.",
    )
    return parser.parse_args()


def safe_sample_filename(sample_id: str) -> str:
    return sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def build_diversity_report(combined: pd.DataFrame) -> Dict[str, object]:
    top4_by_sample = {}
    category_event_counts = defaultdict(Counter)

    for sample_id, group in combined.groupby("sample_id", sort=False):
        ranked = group.sort_values("rank")
        top4 = tuple(ranked.head(4)["event_name"].tolist())
        top4_by_sample[sample_id] = top4

        category = str(ranked["category"].iloc[0])
        for event_name in top4:
            category_event_counts[category][event_name] += 1

    top4_counts = Counter(top4_by_sample.values())
    total_samples = len(top4_by_sample)
    unique_top4 = len(top4_counts)

    return {
        "total_ranked_samples": total_samples,
        "unique_top4_sets": unique_top4,
        "unique_top4_ratio": unique_top4 / total_samples if total_samples else 0.0,
        "top10_top4_sets": [
            {"events": list(events), "count": count}
            for events, count in top4_counts.most_common(10)
        ],
        "category_top_events": {
            category: [
                {"event_name": event_name, "count": count}
                for event_name, count in counter.most_common(10)
            ]
            for category, counter in sorted(category_event_counts.items())
        },
    }


def main() -> None:
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    per_sample_dir = output_dir / "per_sample"
    results_dir = PROJECT_ROOT / "data/processed/results"

    per_sample_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    loader = RadarDataLoader(chunksize=args.chunksize)
    print("=" * 80)
    print("RADAR RANKING GENERATION")
    print("=" * 80)
    print(loader.validate())
    print("weights:", args.weights)
    print("sample_limit:", args.sample_limit)

    sample_ids = loader.list_sample_ids(limit=args.sample_limit)
    print(f"Ranking {len(sample_ids)} samples")

    ranker = HPCBoostRanker(weights=args.weights)
    ranking_frames: List[pd.DataFrame] = []

    print("Loading selected sample traces in one CSV pass...")
    traces = loader.load_samples(sample_ids)

    for sample_id in tqdm(sample_ids, desc="Ranking samples"):
        trace = traces[sample_id]
        event_matrix = trace.events.to_numpy(dtype=float)

        ranking = ranker.rank_events(event_matrix, list(trace.events.columns))
        ranking["sample_id"] = trace.sample_id
        ranking["binary_label"] = trace.binary_label
        ranking["category"] = trace.category
        ranking["family"] = trace.family
        ranking["full_label"] = trace.full_label

        ranking_frames.append(ranking)

        per_sample_path = per_sample_dir / f"ranking_{safe_sample_filename(sample_id)}"
        ranking.to_csv(per_sample_path, index=False)

    if not ranking_frames:
        raise RuntimeError("No samples were ranked")

    combined = pd.concat(ranking_frames, ignore_index=True)
    combined_path = output_dir / "all_rankings.csv"
    combined.to_csv(combined_path, index=False)

    report = build_diversity_report(combined)
    report_path = results_dir / "radar_ranking_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Combined rankings: {combined_path}")
    print(f"Per-sample rankings: {per_sample_dir}")
    print(f"Validation report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

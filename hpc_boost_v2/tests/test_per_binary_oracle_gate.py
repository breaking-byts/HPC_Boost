"""End-to-end test for the exp5 per-binary oracle gate (exp5/run_per_binary_oracle.py).

Generates a synthetic labeled_dataset.csv matching build_dataset.py's schema
(13 events in 4 separate-boot groups, 5 reps/binary) with a planted, *divergent*
signal — two malware families each separable by a different event group — plus
stable reps. Then it runs the whole gate and checks:

  * the loader produces 78 marginal features (13 events x 6 stats), one row per
    (sample, rep);
  * the routing ladder is accuracy-monotonic (global <= per_family <= per_binary
    <= per_run), which is mathematically guaranteed because each finer level has
    strictly more routing freedom over the same candidate predictions — a strong
    correctness check on the routing wiring;
  * stability / divergence / metrics outputs are well-formed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

MODULE_PATH = PKG_ROOT / "experiments" / "exp5_recommender" / "run_per_binary_oracle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_per_binary_oracle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()

# Event -> home group (cycles anchors every group); mirrors guest_groups.txt layout.
GROUPS = {
    "g000": ["cycles", "instructions", "branches", "branch-misses"],
    "g001": ["cycles", "cache-references", "cache-misses", "L1-dcache-loads"],
    "g002": ["cycles", "L1-dcache-load-misses", "L1-dcache-stores", "dTLB-loads"],
    "g003": ["cycles", "dTLB-load-misses", "iTLB-loads", "iTLB-load-misses"],
}


def _profile(label: str, family: str) -> dict:
    """Per-event base rate. Benign = high IPC; malwareA = cache-bound (g001);
    malwareB = TLB-bound (g003). Different families separable by different groups
    -> exercises divergence."""
    base = {e: 1.0 for e in mod.EVENTS_13}
    base["cycles"] = 1_000_000.0
    if label == "benign":
        base.update({"instructions": 2_600_000, "branches": 500_000, "branch-misses": 5_000,
                     "cache-references": 200_000, "cache-misses": 8_000, "L1-dcache-loads": 700_000,
                     "L1-dcache-load-misses": 20_000, "L1-dcache-stores": 300_000, "dTLB-loads": 600_000,
                     "dTLB-load-misses": 3_000, "iTLB-loads": 90_000, "iTLB-load-misses": 600})
    elif family == "Tsunami":  # cache-bound
        base.update({"instructions": 900_000, "branches": 220_000, "branch-misses": 30_000,
                     "cache-references": 900_000, "cache-misses": 260_000, "L1-dcache-loads": 800_000,
                     "L1-dcache-load-misses": 240_000, "L1-dcache-stores": 260_000, "dTLB-loads": 500_000,
                     "dTLB-load-misses": 4_000, "iTLB-loads": 95_000, "iTLB-load-misses": 700})
    else:  # Kaiji: TLB-bound
        base.update({"instructions": 950_000, "branches": 240_000, "branch-misses": 9_000,
                     "cache-references": 230_000, "cache-misses": 12_000, "L1-dcache-loads": 760_000,
                     "L1-dcache-load-misses": 26_000, "L1-dcache-stores": 280_000, "dTLB-loads": 520_000,
                     "dTLB-load-misses": 180_000, "iTLB-loads": 400_000, "iTLB-load-misses": 120_000})
    return base


def make_synthetic_dataset(path: Path, seed: int = 7, ticks: int = 16) -> Path:
    rng = np.random.default_rng(seed)
    binaries = (
        [(f"benign_{i}", "benign", "sortbench" if i % 2 else "idle") for i in range(6)]
        + [(f"mwA_{i}", "malware", "Tsunami") for i in range(4)]
        + [(f"mwB_{i}", "malware", "Kaiji") for i in range(4)]
    )
    rows = []
    for sample, label, family in binaries:
        prof = _profile(label, family)
        for rep in range(5):
            rep_jit = {e: 1.0 + rng.normal(0, 0.05) for e in mod.EVENTS_13}  # stable across reps
            for group, evlist in GROUPS.items():
                for t in range(ticks):
                    row = {"label": label, "family": family, "sample": sample,
                           "rep": rep, "group": group, "ts_ms": t * 10}
                    for e in mod.EVENTS_13:
                        if e in evlist:
                            val = prof[e] * rep_jit[e] * (1.0 + rng.normal(0, 0.08))
                            row[e] = max(0.0, val)
                        else:
                            row[e] = 0.0
                    rows.append(row)
    header = ["label", "family", "sample", "rep", "group", "ts_ms"] + mod.EVENTS_13
    pd.DataFrame(rows)[header].to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("exp5") / "labeled_dataset.csv"
    return make_synthetic_dataset(path)


def test_loader_marginal_features(dataset):
    X, y, samples, reps, families, feature_names = mod.load_per_binary(dataset)
    assert X.shape[1] == len(mod.EVENTS_13) * 6 == 78
    assert X.shape[0] == 14 * 5  # 14 binaries x 5 reps
    assert len(np.unique(samples)) == 14
    assert set(np.unique(y).tolist()) == {0, 1}
    # every event contributes its 6 suffixes
    for e in mod.EVENTS_13:
        assert f"{e}_mean" in feature_names and f"{e}_kurt" in feature_names


def test_gate_ladder_monotonic_and_well_formed(dataset):
    X, y, samples, reps, families, feature_names = mod.load_per_binary(dataset)
    result = mod.run_gate(
        X, y, samples, reps, families, feature_names, events=mod.EVENTS_13,
        outer_folds=5, inner_folds=3, n_estimators=25, max_depth=3, seed=1,
        jobs=2, target_tpr=0.9, max_candidates=30,
    )
    m, s = result["metrics"], result["summary"]

    # routing freedom strictly increases -> accuracy is non-decreasing along ladder
    acc = [m[name]["accuracy"] for name in mod.LADDER]
    assert acc == pytest.approx(sorted(acc), abs=1e-9), f"ladder not monotonic: {acc}"

    # well-formed metrics
    for name in list(mod.LADDER) + ["twosmart", "majority", "best_global"]:
        for key in ("f1", "balanced_accuracy", "fpr", "accuracy"):
            assert 0.0 <= m[name][key] <= 1.0

    # summary fields present + sane
    assert s["n_binaries"] == 14 and s["n_candidates"] == 30
    assert s["divergence_distinct_binary_subsets"] >= 1
    for k in ("stability_median_rep_jaccard", "stability_median_best_consistency",
              "stability_median_feature_cv"):
        assert s[k] is None or s[k] >= 0.0
    # the planted signal is real -> the oracle ceiling should clear the majority floor
    assert m["per_run"]["balanced_accuracy"] >= m["majority"]["balanced_accuracy"]


def load_recommender():
    path = PKG_ROOT / "experiments" / "exp5_recommender" / "recommend_fixed_subset.py"
    spec = importlib.util.spec_from_file_location("recommend_fixed_subset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_subset_recommender(dataset):
    rec = load_recommender()
    X, y, samples, reps, families, fn = mod.load_per_binary(dataset)
    result = rec.run_recommender(
        X, y, samples, families, fn, events=mod.EVENTS_13, outer_folds=5,
        inner_folds=3, n_estimators=25, max_depth=3, seed=1, jobs=2,
        target_tpr=0.9, max_candidates=30,
    )
    m, s = result["metrics"], result["summary"]
    for name in rec.SELECTORS:
        for key in ("f1", "balanced_accuracy", "fpr"):
            assert 0.0 <= m[name][key] <= 1.0
    assert len(s["recommended_subset"]) == 4
    assert s["consistency_fixed_balacc"]["n_folds"] == 5
    assert s["consistency_fixed_balacc"]["distinct"] >= 1

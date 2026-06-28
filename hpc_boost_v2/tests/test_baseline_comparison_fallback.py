"""Regression tests for the exp2 baseline-comparison fail-safe (V6).

The HPC-Boost per-sample worker must fail *safe* to benign (pred=0) whenever it
cannot produce a real prediction — i.e. when no event ranking is available, or
when the detector fit/predict raises. The original code returned pred=1
(malware), which silently inflates recall/F1 on the ~83% malware RaDaR set and
contradicts the experiment's own "missing rankings -> default pred=0" reporting.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# run_comparison_final.py imports `from src.utils...` at module load, so the
# package root must be importable before we exec the module.
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

MODULE_PATH = (
    PKG_ROOT
    / "experiments"
    / "exp2_baseline_comparison"
    / "run_comparison_final.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_comparison_final", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_ranking_defaults_to_benign():
    """feat_indices is None -> conservative benign default, flagged as default."""
    mod = load_module()
    pred, had_ranking = mod.hpcboost_worker(
        None,
        np.zeros((4, 3), dtype=np.float32),
        np.array([0, 1, 0, 1]),
        np.zeros(3, dtype=np.float32),
        "XGBoost",
    )
    assert pred == 0, "missing ranking must fail safe to benign (0), not malware (1)"
    assert had_ranking is False


def test_fit_failure_defaults_to_benign():
    """An exception inside the detector path must fail safe to benign (0)."""
    mod = load_module()
    # An unknown detector makes train_and_predict raise, exercising the except branch.
    pred, had_ranking = mod.hpcboost_worker(
        [0, 1],
        np.zeros((4, 3), dtype=np.float32),
        np.array([0, 1, 0, 1]),
        np.zeros(3, dtype=np.float32),
        "NoSuchDetector",
    )
    assert pred == 0, "fit/predict failure must fail safe to benign (0), not malware (1)"
    assert had_ranking is False

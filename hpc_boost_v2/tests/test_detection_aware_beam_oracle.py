from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "exp3_detection_aware_oracle"
    / "run_beam_oracle.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_beam_oracle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expand_beam_returns_unique_canonical_subsets():
    module = load_module()

    expanded = module.expand_beam(
        beam=[("event_b",), ("event_a",)],
        event_names=["event_c", "event_b", "event_a"],
    )

    assert expanded == [
        ("event_a", "event_b"),
        ("event_a", "event_c"),
        ("event_b", "event_c"),
    ]


def test_prune_scored_subsets_is_deterministic_on_ties():
    module = load_module()
    scored = [
        module.SubsetScore(("b",), 0.8, 0.02),
        module.SubsetScore(("a",), 0.8, 0.02),
        module.SubsetScore(("c",), 0.7, 0.01),
    ]

    assert [row.events for row in module.prune_scored_subsets(scored, 2)] == [
        ("a",),
        ("b",),
    ]


def test_route_candidate_oracle_uses_best_ranked_correct_candidate():
    module = load_module()
    y_true = np.array([1, 0, 1, 0])
    candidate_predictions = np.array(
        [
            [0, 0, 1, 1],  # best global candidate
            [1, 1, 0, 0],
            [1, 0, 0, 1],
        ]
    )
    scores = np.array([0.90, 0.80, 0.70])

    routed = module.route_candidate_oracle(
        y_true=y_true,
        candidate_predictions=candidate_predictions,
        candidate_scores=scores,
        global_candidate_index=0,
    )

    assert routed.predictions.tolist() == [1, 0, 1, 0]
    assert routed.candidate_indices.tolist() == [1, 0, 0, 1]
    assert routed.correct_candidate_counts.tolist() == [2, 2, 1, 1]
    assert routed.coverage == 1.0


def test_route_candidate_oracle_falls_back_when_no_candidate_is_correct():
    module = load_module()
    y_true = np.array([1, 0])
    candidate_predictions = np.array([[0, 1], [0, 1]])

    routed = module.route_candidate_oracle(
        y_true=y_true,
        candidate_predictions=candidate_predictions,
        candidate_scores=np.array([0.9, 0.8]),
        global_candidate_index=0,
    )

    assert routed.predictions.tolist() == [0, 1]
    assert routed.candidate_indices.tolist() == [0, 0]
    assert routed.correct_candidate_counts.tolist() == [0, 0]
    assert routed.coverage == 0.0


@pytest.mark.parametrize(
    ("beam_width", "candidate_cap"),
    [(0, None), (-1, None), (5, 0), (5, -3)],
)
def test_validate_search_parameters_rejects_non_positive_values(
    beam_width, candidate_cap
):
    module = load_module()

    with pytest.raises(ValueError):
        module.validate_search_parameters(
            beam_width=beam_width,
            candidate_cap=candidate_cap,
            inner_folds=3,
        )

from __future__ import annotations

import importlib.util
import unittest.mock as mock
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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


def test_resolve_group_values_supports_radar_aliases():
    module = load_module()
    sample_ids = ["s1", "s2"]
    metadata = {
        "category": ["normal", "ransomware"],
        "family": ["benign_app", "lockbit"],
        "full_label": ["normal.benign_app", "ransomware.lockbit"],
    }

    assert module.resolve_group_values("family_gene", sample_ids, metadata).tolist() == [
        "benign_app",
        "lockbit",
    ]
    assert module.resolve_group_values("Filename", sample_ids, metadata).tolist() == [
        "s1",
        "s2",
    ]


def test_resolve_group_values_rejects_unknown_column():
    module = load_module()

    with pytest.raises(ValueError):
        module.resolve_group_values(
            "missing",
            ["s1"],
            {"category": ["normal"], "family": ["benign"], "full_label": ["normal"]},
        )


def test_build_outer_splits_degenerate_group_fold_falls_back_to_hybrid():
    module = load_module()
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    # All benign in one group, all malware in another — pure GroupKFold puts each
    # class in a separate test fold (degenerate). The malware group also has only 1
    # unique group, so the hybrid fallback raises because it needs >= 2 mal families.
    groups = np.array(["benign", "benign", "benign", "mal", "mal", "mal"])

    with pytest.raises(ValueError):
        module.build_outer_splits(
            X=X,
            y=y,
            outer_folds=2,
            seed=42,
            group_values=groups,
        )


def test_build_outer_splits_hybrid_fallback_produces_balanced_folds():
    module = load_module()
    # 4 benign samples in one "normal" group, 4 malware samples in 2 families.
    # Pure GroupKFold with 2 folds isolates "normal" into one test fold → degenerate.
    # Hybrid fallback (GroupKFold on malware + KFold on benign) should produce balanced folds.
    X = np.zeros((8, 2))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(
        ["normal", "normal", "normal", "normal", "fam_a", "fam_a", "fam_b", "fam_b"]
    )

    splits = module.build_outer_splits(X=X, y=y, outer_folds=2, seed=42, group_values=groups)

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert set(y[train_idx].tolist()) == {0, 1}, "train fold must have both classes"
        assert set(y[test_idx].tolist()) == {0, 1}, "test fold must have both classes"


def test_load_and_aggregate_returns_six_tuple_with_metadata():
    module = load_module()

    class FakeTrace:
        def __init__(self, binary_label, category, family, full_label):
            self.binary_label = binary_label
            self.category = category
            self.family = family
            self.full_label = full_label
            self.events = pd.DataFrame({"evt_a": [1.0, 2.0], "evt_b": [3.0, 4.0]})

    fake_traces = {
        "s1": FakeTrace(1, "malware", "fam_a", "malware.fam_a"),
        "s2": FakeTrace(0, "normal", "normal", "normal.normal"),
    }

    mock_instance = mock.MagicMock()
    mock_instance.validate.return_value = {}
    mock_instance.list_sample_ids.return_value = ["s1", "s2"]
    mock_instance.load_samples.return_value = fake_traces
    mock_instance.hpc_columns = ["evt_a", "evt_b"]

    with mock.patch.object(module, "RadarDataLoader", return_value=mock_instance):
        result = module.load_and_aggregate(
            radar_root="/fake",
            csv_name="fake.csv",
            sample_limit=None,
            chunksize=100,
        )

    X, y, sample_ids, event_names, feature_names, metadata = result
    assert len(result) == 6, "load_and_aggregate must return exactly 6 values"
    assert y.tolist() == [1, 0]
    assert sample_ids == ["s1", "s2"]
    assert event_names == ["evt_a", "evt_b"]
    assert metadata["category"] == ["malware", "normal"]
    assert metadata["family"] == ["fam_a", "normal"]
    assert metadata["full_label"] == ["malware.fam_a", "normal.normal"]


def test_load_and_aggregate_always_collects_metadata():
    """New contract: metadata is collected unconditionally (no gating flag)."""
    module = load_module()

    class FakeTrace:
        def __init__(self, binary_label, category, family, full_label):
            self.binary_label = binary_label
            self.category = category
            self.family = family
            self.full_label = full_label
            self.events = pd.DataFrame({"evt_a": [1.0], "evt_b": [2.0]})

    fake_traces = {
        "s1": FakeTrace(1, "malware", "fam_a", "malware.fam_a"),
        "s2": FakeTrace(0, "normal", "normal", "normal.normal"),
    }

    mock_instance = mock.MagicMock()
    mock_instance.validate.return_value = {}
    mock_instance.list_sample_ids.return_value = ["s1", "s2"]
    mock_instance.load_samples.return_value = fake_traces
    mock_instance.hpc_columns = ["evt_a", "evt_b"]

    with mock.patch.object(module, "RadarDataLoader", return_value=mock_instance):
        _, _, _, _, _, metadata = module.load_and_aggregate(
            radar_root="/fake",
            csv_name="fake.csv",
            sample_limit=None,
            chunksize=100,
        )

    assert metadata["category"] == ["malware", "normal"]
    assert metadata["family"] == ["fam_a", "normal"]
    assert metadata["full_label"] == ["malware.fam_a", "normal.normal"]


def test_build_inner_splits_group_mode_is_family_disjoint():
    module = load_module()

    # 8 samples, 4 malware families + benign; group inner CV must keep each
    # family entirely on one side of every fold.
    X = np.zeros((8, 2))
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(
        ["fa", "fa", "fb", "fb", "fc", "fc", "fd", "fd"], dtype=object
    )

    splits = module.build_inner_splits(
        X_train=X,
        y_train=y,
        inner_folds=2,
        seed=0,
        inner_split="group",
        train_group_values=groups,
    )

    assert len(splits) == 2
    for train_idx, valid_idx in splits:
        train_fams = set(groups[train_idx].tolist())
        valid_fams = set(groups[valid_idx].tolist())
        assert not (train_fams & valid_fams), "family leaked across inner fold"


def test_build_inner_splits_group_mode_falls_back_to_hybrid():
    module = load_module()

    # All benign share one group -> pure GroupKFold would isolate benign into a
    # single-class fold. The hybrid fallback must produce two-class folds.
    X = np.zeros((8, 2))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(
        ["normal", "normal", "normal", "normal", "fa", "fa", "fb", "fb"],
        dtype=object,
    )

    splits = module.build_inner_splits(
        X_train=X,
        y_train=y,
        inner_folds=2,
        seed=1,
        inner_split="group",
        train_group_values=groups,
    )

    assert len(splits) == 2
    for train_idx, valid_idx in splits:
        assert set(y[train_idx].tolist()) == {0, 1}
        assert set(y[valid_idx].tolist()) == {0, 1}


def test_build_inner_splits_stratified_mode_ignores_groups():
    module = load_module()

    X = np.zeros((10, 2))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    splits = module.build_inner_splits(
        X_train=X,
        y_train=y,
        inner_folds=2,
        seed=3,
        inner_split="stratified",
        train_group_values=None,
    )
    assert len(splits) == 2


def test_threshold_for_target_tpr_is_train_only():
    module = load_module()

    # Train OOF probabilities/labels only. There is no test argument, so the
    # tuned threshold cannot leak test labels.
    y_train = np.array([0, 0, 1, 1, 1, 1])
    train_oof = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    thr = module.threshold_for_target_tpr(y_train, train_oof, target_tpr=0.95)
    # 4 positives at .4,.6,.8,.9 -> all must pass for TPR>=0.95; highest
    # qualifying threshold is 0.4.
    assert thr == pytest.approx(0.4)
    # Deterministic / idempotent for identical train inputs.
    assert module.threshold_for_target_tpr(y_train, train_oof, 0.95) == thr


def test_threshold_for_target_tpr_picks_higher_threshold_for_lower_target():
    module = load_module()

    y_train = np.array([0, 0, 1, 1, 1, 1])
    train_oof = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    # target 0.5 -> need 2 of 4 positives >= thr; highest qualifying is 0.8.
    thr = module.threshold_for_target_tpr(y_train, train_oof, target_tpr=0.5)
    assert thr == pytest.approx(0.8)


def test_route_group_oracle_per_family_beats_global_loses_to_per_sample():
    module = load_module()

    # Families A (idx 0,1) and B (idx 2,3); two candidates.
    #   cand0 perfect on A, wrong on B; cand1 perfect on B, wrong on A.
    # No single candidate is correct everywhere (global caps at 50%).
    # per_family picks cand0 for A, cand1 for B -> all correct.
    y_true = np.array([1, 0, 1, 0])
    cand_preds = np.array(
        [
            [1, 0, 0, 1],  # cand0 correct on A (idx0,1)
            [0, 1, 1, 0],  # cand1 correct on B (idx2,3)
        ]
    )
    families = np.array(["A", "A", "B", "B"], dtype=object)

    global_pred = module.route_global_oracle(y_true, cand_preds)
    global_correct = int(np.sum(global_pred == y_true))
    assert global_correct == 2

    fam_pred = module.route_group_oracle(y_true, cand_preds, families)
    fam_correct = int(np.sum(fam_pred == y_true))
    assert fam_correct == 4
    assert fam_correct > global_correct

    scores = np.array([0.9, 0.8])
    sample_routed = module.route_candidate_oracle(
        y_true=y_true,
        candidate_predictions=cand_preds,
        candidate_scores=scores,
        global_candidate_index=0,
    )
    sample_correct = int(np.sum(sample_routed.predictions == y_true))
    assert sample_correct == 4
    assert sample_correct >= fam_correct


def test_route_global_oracle_breaks_ties_toward_lowest_index():
    module = load_module()

    y_true = np.array([1, 0])
    # both candidates get 1/2 correct -> tie, lowest index (0) wins.
    cand_preds = np.array([[1, 1], [0, 0]])
    routed = module.route_global_oracle(y_true, cand_preds)
    assert routed.tolist() == [1, 1]


def test_predictions_at_threshold_applies_per_candidate_thresholds():
    module = load_module()

    probabilities = np.array([[0.2, 0.6, 0.9], [0.2, 0.6, 0.9]])
    thresholds = np.array([0.5, 0.7])
    preds = module.predictions_at_threshold(probabilities, thresholds)
    assert preds.tolist() == [[0, 1, 1], [0, 0, 1]]


def test_compute_metrics_exposes_honest_metric_keys():
    module = load_module()

    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.7, 0.9])

    metrics = module.compute_metrics(y_true, predictions, probabilities)
    for key in [
        "f1", "precision", "recall", "accuracy", "auprc",
        "fpr", "balanced_accuracy", "mcc",
        "tp", "fp", "tn", "fn",
        "tpr_at_fpr_0.05", "tpr_at_fpr_0.01",
    ]:
        assert key in metrics, f"missing metric key: {key}"
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 0
    assert metrics["fpr"] == pytest.approx(0.5)


def test_compute_metrics_single_class_group_is_safe():
    module = load_module()

    # A per-group oracle may evaluate an all-malware group; auprc/mcc/bal-acc
    # are undefined and must default to 0.0 without raising.
    y_true = np.array([1, 1, 1])
    predictions = np.array([1, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.2])

    metrics = module.compute_metrics(y_true, predictions, probabilities)
    assert metrics["auprc"] == 0.0
    assert metrics["mcc"] == 0.0
    assert metrics["balanced_accuracy"] == 0.0
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_parse_args_defaults_inner_split_to_group_with_group_column():
    module = load_module()

    args = module.parse_args(["--group-column", "family_gene"])
    assert args.inner_split == "group"
    assert args.target_tpr == pytest.approx(0.95)

    args2 = module.parse_args([])
    assert args2.inner_split == "stratified"


def test_parse_args_rejects_group_inner_without_group_column():
    module = load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--inner-split", "group"])


def test_parse_args_rejects_out_of_range_target_tpr():
    module = load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--target-tpr", "1.5"])

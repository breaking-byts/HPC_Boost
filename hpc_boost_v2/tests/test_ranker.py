import numpy as np
import pytest

from src.ranking.ranker import HPCBoostRanker


def make_event_matrix(rows=100, cols=55):
    base = np.arange(rows, dtype=float)
    columns = []
    for i in range(cols):
        if i == 0:
            columns.append(np.ones(rows))
        else:
            columns.append(base + i)
    return np.column_stack(columns)


def test_ranker_outputs_one_row_per_event_and_ranks_are_contiguous():
    matrix = make_event_matrix()
    event_names = [f"event_{i}" for i in range(55)]

    ranking = HPCBoostRanker().rank_events(matrix, event_names)

    assert len(ranking) == 55
    assert list(ranking["rank"]) == list(range(1, 56))
    assert set(ranking["event_name"]) == set(event_names)


def test_ranker_sorts_highest_combined_score_first():
    matrix = make_event_matrix()
    event_names = [f"event_{i}" for i in range(55)]

    ranking = HPCBoostRanker().rank_events(matrix, event_names)

    assert ranking.iloc[0]["rank"] == 1
    assert ranking.iloc[0]["combined_score"] >= ranking.iloc[-1]["combined_score"]
    assert ranking["combined_score"].is_monotonic_decreasing


def test_ranker_accepts_custom_weights():
    ranker = HPCBoostRanker(weights=[1.0, 0.0, 0.0, 0.0])

    assert ranker.weights == [1.0, 0.0, 0.0, 0.0]


def test_ranker_rejects_wrong_weight_count():
    with pytest.raises(ValueError):
        HPCBoostRanker(weights=[0.5, 0.5])


def test_ranker_rejects_mismatched_event_names():
    matrix = make_event_matrix(cols=3)

    with pytest.raises(ValueError):
        HPCBoostRanker().rank_events(matrix, ["a", "b"])

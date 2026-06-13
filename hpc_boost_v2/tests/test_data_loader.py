import os

import pandas as pd
import pytest

from src.utils.data_loader import RadarDataLoader, SampleTrace


def make_radar_csv(tmp_path):
    metadata_prefix = ["Unnamed: 0", "Filename"]
    hpc_cols = [f"event_{i}" for i in range(55)]
    metadata_suffix = [
        "full_label",
        "label",
        "binarylabel",
        "method",
        "goal",
        "family_gene",
        "keylog",
        "bkdoor",
        "infosteal",
        "rootkits",
        "method_encoded",
        "goal_encoded",
        "family_encoded",
        "infosteal_encoded",
    ]
    columns = metadata_prefix + hpc_cols + metadata_suffix

    rows = []
    for sample_id, binarylabel, goal, family, full_label in [
        ("sample_a.csv", 0, "normal", "normal", "clean"),
        ("sample_b.csv", 1, "ransomware", "Sigma", "Win32.Filecoder.Sigma"),
    ]:
        for t in range(3):
            row = [t, sample_id]
            row.extend([i + t for i in range(55)])
            row.extend([
                full_label,
                0,
                binarylabel,
                "normal" if binarylabel == 0 else "trojan",
                goal,
                family,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
            ])
            rows.append(row)

    csv_path = tmp_path / "combined_hardware_trails.csv"
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    return tmp_path


def test_validate_reads_header_and_extracts_hpc_columns(tmp_path):
    root = make_radar_csv(tmp_path)
    loader = RadarDataLoader(root_dir=str(root), chunksize=2)

    info = loader.validate()

    assert info["num_columns"] == 71
    assert info["num_hpc_events"] == 55
    assert loader.hpc_columns[0] == "event_0"
    assert loader.hpc_columns[-1] == "event_54"


def test_load_sample_returns_trace_with_metadata_and_events(tmp_path):
    root = make_radar_csv(tmp_path)
    loader = RadarDataLoader(root_dir=str(root), chunksize=2)

    trace = loader.load_sample("sample_b.csv")

    assert isinstance(trace, SampleTrace)
    assert trace.sample_id == "sample_b.csv"
    assert trace.binary_label == 1
    assert trace.category == "ransomware"
    assert trace.family == "Sigma"
    assert trace.full_label == "Win32.Filecoder.Sigma"
    assert trace.num_timesteps == 3
    assert trace.events.shape == (3, 55)


def test_missing_csv_raises_file_not_found(tmp_path):
    loader = RadarDataLoader(root_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        loader.validate()


def test_load_samples_returns_multiple_traces_in_one_call(tmp_path):
    root = make_radar_csv(tmp_path)
    loader = RadarDataLoader(root_dir=str(root), chunksize=2)

    traces = loader.load_samples(["sample_a.csv", "sample_b.csv"])

    assert list(traces.keys()) == ["sample_a.csv", "sample_b.csv"]
    assert traces["sample_a.csv"].binary_label == 0
    assert traces["sample_b.csv"].binary_label == 1
    assert traces["sample_a.csv"].events.shape == (3, 55)
    assert traces["sample_b.csv"].events.shape == (3, 55)

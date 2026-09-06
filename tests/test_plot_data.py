"""Checkpoint safety for the standalone time-sweep script, without rendering plots."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from surface_code.simulation.aer import MAX_SEED


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_memory_by_time.py"
    spec = spec_from_file_location("plot_memory_by_time", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint():
    return {
        "metadata": {
            "profile": "baseline",
            "prep": "encoder",
            "times_us": [10],
            "rounds": [0, 1],
            "round_duration_us": 1,
            "shots_per_point": 10,
            "source_digest": "original-source",
            "created_utc": "2026-09-06T00:00:00+00:00",
        },
        "points": [
            {
                "total_time_us": 10,
                "rounds": 0,
                "basis": "Z",
                "shots": 10,
                "logical_failures": 1,
                "logical_failure_rate": 0.1,
                "confidence_low": 0.02,
                "confidence_high": 0.4,
                "idle_per_round_us": 10,
            }
        ],
    }


def test_partial_checkpoint_can_resume_but_cannot_be_plotted(script, checkpoint):
    metadata, rows = script.resume_data(checkpoint, script._settings(checkpoint["metadata"]))
    assert rows[0]["source_digest"] == "original-source"
    with pytest.raises(ValueError, match="cover the grid"):
        script.validate_data(metadata, rows)


def test_resume_rejects_duplicate_points_before_overwriting_data(script, checkpoint):
    checkpoint["points"].append(dict(checkpoint["points"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        script.resume_data(checkpoint, script._settings(checkpoint["metadata"]))


def test_resume_rejects_corrupt_measurement_counts(script, checkpoint):
    checkpoint["points"][0]["logical_failures"] = 4
    with pytest.raises(ValueError, match="inconsistent saved point"):
        script.resume_data(checkpoint, script._settings(checkpoint["metadata"]))


def test_resume_rejects_changed_shot_count(script, checkpoint):
    checkpoint["metadata"]["shots_per_point"] = 5000
    checkpoint["points"][0].update(shots=5000, logical_failures=500)
    requested = {**script._settings(checkpoint["metadata"]), "shots_per_point": 500}
    with pytest.raises(ValueError, match="different settings or source"):
        script.resume_data(checkpoint, requested)


def test_source_override_preserves_per_point_provenance(script, checkpoint):
    requested = {**script._settings(checkpoint["metadata"]), "source_digest": "new-source"}
    with pytest.raises(ValueError, match="different settings or source"):
        script.resume_data(checkpoint, requested)
    metadata, rows = script.resume_data(checkpoint, requested, ignore_source_change=True)
    assert metadata["source_digest"] == "new-source"
    assert rows[0]["source_digest"] == "original-source"
    # A subsequent resume must not relabel the older measurements.
    _, resumed_rows = script.resume_data({"metadata": metadata, "points": rows}, requested)
    assert resumed_rows == rows


def test_seed_offsets_remain_in_the_backend_range(script):
    seeds = {
        script._point_seed(MAX_SEED, time, n) for time in script.TIMES_US for n in script.ROUNDS
    }
    assert len(seeds) == len(script.TIMES_US) * len(script.ROUNDS)
    assert all(0 <= seed <= MAX_SEED for seed in seeds)


def test_checkpoint_save_is_readable_and_replaces_the_temporary_file(script, checkpoint, tmp_path):
    target = tmp_path / "data.json"
    script.save_data(target, checkpoint["metadata"], checkpoint["points"])
    assert json.loads(target.read_text()) == checkpoint
    assert not (tmp_path / "data.json.tmp").exists()

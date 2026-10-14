"""The paper's effect-size interval counts a day, not a pair-day, as its block."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def analysis(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs" / "analysis"))
    import f_treatments

    return f_treatments


def test_interval_uses_daily_log_means(analysis):
    days = np.repeat(np.arange(5), 2)
    placements = np.tile([1, 2], 5)
    y = np.array([0.1, 0.3, 0.2, 0.4, 0.15, 0.45, 0.25, 0.35, 0.3, 0.5])
    result = analysis.summary(y, 10, days, placements)
    expected = analysis.t_interval(y.reshape(5, 2).mean(axis=1))
    assert result["n"] == 5
    assert result["of"] == 10
    assert result["ratio"] == pytest.approx(np.exp(y.mean()))
    assert result["ci95"] == pytest.approx(expected["ci95"])
    assert result["pair_day_interval"]["n"] == 10
    assert result["placement_intervals"]["1"]["n"] == 5
    reverse = analysis.summary(y[::-1], 10, days[::-1], placements[::-1])
    assert reverse == result


def test_within_day_offsets_do_not_create_between_day_scatter(analysis):
    days = np.repeat(np.arange(5), 2)
    placements = np.tile([1, 2], 5)
    result = analysis.summary(np.tile([-0.2, 0.2], 5), 5, days, placements)
    assert result["ci95"] == pytest.approx([1.0, 1.0])
    assert result["pair_day_interval"]["ci95"][0] < 1.0


def test_committed_intervals_reproduce_from_stored_fits(analysis):
    fits = analysis.result("fits.json")
    per_rep, predictions = fits["per_rep"], fits["predictions"]
    keys = sorted(per_rep, key=lambda k: (per_rep[k]["day"], per_rep[k]["rep"]))
    days = np.array([analysis.day_number(per_rep[k]["day"]) for k in keys])
    placements = np.array([per_rep[k]["placement"] for k in keys])
    saved = analysis.result("f_treatments.json")["table"]
    for row, source in analysis.STORED.items():
        for quantity in analysis.QUANTITIES:
            observed = np.array([analysis.observed(per_rep[k], quantity)[0] for k in keys])
            predicted = np.array(
                [analysis.predicted(predictions, per_rep[k], source, quantity) for k in keys]
            )
            expected = analysis.summary(
                np.log(observed / predicted), int((observed > predicted).sum()), days, placements
            )
            assert saved[row][quantity] == expected
    weighted = analysis.result("f_weighted_mean.json")["binomial"]["O_over_M"]["weighted-mean f"]
    for quantity in analysis.QUANTITIES:
        ratios = np.array(weighted[quantity]["per_repetition"])
        expected = analysis.summary(np.log(ratios), int((ratios > 1).sum()), days, placements)
        assert saved[analysis.WEIGHTED][quantity] == expected


@pytest.mark.parametrize(
    "y, days, placements",
    [
        ([0.1, 0.2], [1], [1, 2]),
        ([0.1, 0.2], [1, 1], [1, 2]),
        ([0.1, np.nan], [1, 2], [1, 2]),
        ([0.1, 0.2], [1, 2], [1, 2]),
    ],
)
def test_invalid_blocks_rejected(analysis, y, days, placements):
    with pytest.raises(ValueError):
        analysis.summary(y, 1, days, placements)

"""Omitting a day changes both the sample size and the t critical value."""

import numpy as np
import pytest
from scipy.stats import t


@pytest.fixture
def sensitivity(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs" / "analysis"))
    import day_sensitivity

    return day_sensitivity


def test_omission_uses_four_days_and_three_degrees_of_freedom(sensitivity):
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    saved = sensitivity.leave_one_out(y)
    assert list(saved) == ["1", "2", "3", "4", "5"]
    subset = y[1:]
    half = t.ppf(0.995, 3) * subset.std(ddof=1) / 2
    assert saved["1"]["ratio"] == pytest.approx(np.exp(subset.mean()))
    assert saved["1"]["ci95_family_of_five"] == pytest.approx(
        np.exp([subset.mean() - half, subset.mean() + half])
    )


@pytest.mark.parametrize("values", [[0.1] * 4, [0.1] * 6, [0.1, 0.2, 0.3, 0.4, np.nan]])
def test_rejects_incomplete_or_nonfinite_days(sensitivity, values):
    with pytest.raises(ValueError):
        sensitivity.leave_one_out(values)


def test_committed_results_follow_the_primary_daily_means(sensitivity):
    source = sensitivity.result("stored_priors.json")
    saved = sensitivity.result("day_sensitivity.json")["by_quantity"]
    for quantity, row in source["primary"].items():
        assert saved[quantity] == sensitivity.leave_one_out(row["day_log_means"])


def test_extrapolation_bound_is_inactive_in_the_five_reported_windows(sensitivity):
    saved = sensitivity.result("day_sensitivity.json")["window_boundaries"]
    for source in ("device", "model"):
        for name, row in saved[source].items():
            if name != "3-8":
                assert row["active_at_mle"] == []
                assert row["minimum_fitted_P1"] > 0.04
    late = saved["device"]["3-8"]["active_at_mle"]
    assert len(late) == 3
    assert all(basis == "Z" and patch == "5" for _, basis, patch in late)

"""Light checks of the saved-data idle diagnostic, not a hardware test."""

import numpy as np
import pytest


@pytest.fixture
def checks(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs/analysis"))
    import idle_checks

    return idle_checks


def test_vectorized_rate_matches_archived_estimator(checks):
    t = np.array([1, 2, 4, 8, 16.0])
    g = np.array([0.98, 0.96, 0.94, 0.92, 0.90])
    value = 0.92 * g * np.exp(-0.07 * t)
    variance = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    archived = checks.a.log_decay(t, value, variance, g)
    rate, se, kept = checks.rate_batch(t, value[None, :], variance[None, :], g)
    assert rate[0] == pytest.approx(archived["rate"][0])
    assert se[0] == pytest.approx(archived["rate"][1])
    assert kept[0] == 5


def test_joint_residuals_recover_saved_fit_statistics(checks):
    # Raw contrasts and exposures are also retained in the frozen Z-idle archive.
    saved = checks.read(checks.REPO / "docs/analysis/results/idle_checks.json")
    archive = checks.read(checks.REPO / "runs/ibm_kingston-followup/analysis/life.json")
    joint = archive["analysis_life"]["in_situ"]["qubits"]
    for row in joint:
        found = saved["joint_residuals"][str(row["qubit"])]
        assert found["chi2_red"] == pytest.approx(row["chi2_red"], rel=1e-10)
        assert found["nominal_dof"] == sum(row["used"]) - 4


def test_saved_planted_checks_are_bounded_and_do_not_claim_total_coverage(checks):
    saved = checks.read(checks.REPO / "docs/analysis/results/idle_checks.json")
    assert saved["seed"] == 1313
    for rows in saved["planted_checks"].values():
        assert len(rows) == 25
        for row in rows.values():
            assert row["repetitions"] == row["valid"] == 2000
            assert row["observed_points_retained"] == 5
            assert row["midpoint_adjustments"] == 0
            assert 0 <= row["nominal95_coverage"] <= 1
            assert 0 <= row["point_discard_fraction"] <= 1
    assert "drift" in saved["caveat"] and "Fez" in saved["caveat"]

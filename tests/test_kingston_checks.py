"""Depth-level checks do not turn an effective fitted rate into a stationary process."""

import hashlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def checks(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs" / "analysis"))
    import kingston_checks

    return kingston_checks


def test_summary_recovers_an_exponential_curve(checks):
    n = np.arange(1, 5)
    p = (1 - 0.97 * (1 - 2 * 0.06) ** n) / 2
    row = checks.summarize_curve({"failures": np.round(1e6 * p).astype(int)}, 1_000_000)
    assert row["eps"] == pytest.approx(0.06, abs=5e-6)
    assert row["A"] == pytest.approx(0.97, abs=5e-6)
    assert row["max_abs_residual_P"] < 2e-6


def test_fixed_variants_keep_device_counts_and_decoder(checks, repo_root):
    saved = checks.read(repo_root / "docs/analysis/results/kingston_checks.json")
    driver = Path("followup/reproduce.py")
    expected = hashlib.sha256((repo_root / driver).read_bytes()).hexdigest()
    baseline = saved["variants"]["measured_tq"]["memory"]["observed"]
    for name, row in saved["variants"].items():
        if name != "measured_tq":
            assert row["hardware_priors"] == "measured"
            assert row["simulation_priors"] == "hardware"
            assert row["input_sha256"][driver.as_posix()] == expected
        for basis in "XZ":
            for d in "35":
                observed = row["memory"]["observed"][basis]["eps"][d]
                assert observed["failures"] == baseline[basis]["eps"][d]["failures"]
                predicted = row["memory"]["predicted"][basis]["eps"][d]
                ratio = row["memory"]["observed_over_model"][basis]["binomial"][f"eps{d}"][0]
                assert ratio == pytest.approx(observed["binomial"][0] / predicted["binomial"][0])


def test_residuals_and_fixed_depth_differences_are_direct_arithmetic(checks):
    saved = checks.read(checks.REPO / "docs/analysis/results/kingston_checks.json")
    for row in saved["curves"].values():
        observed = np.array(row["observed"]["P"])
        for variant, curve in row.items():
            assert curve["P"] == pytest.approx(np.array(curve["failures"]) / curve["shots"])
            assert curve["residual_P"] == pytest.approx(np.array(curve["P"]) - curve["fit_P"])
            if variant != "observed":
                assert curve["observed_minus_model_P"] == pytest.approx(observed - curve["P"])
            if variant in ("ibm", "measured_tq"):
                assert curve["max_abs_residual_P"] < 0.0055  # plotted residual limits


def test_stored_and_job_gaps_use_their_own_device_counts(checks):
    saved = checks.read(checks.REPO / "docs/analysis/results/kingston_checks.json")
    groups = [saved["stored_curves"]["curves"], *saved["by_job_curves"]["jobs"].values()]
    for curves in groups:
        for curve in curves.values():
            observed = curve["observed"]
            assert curve["observed_minus_model_P"] == pytest.approx(
                np.array(observed["failures"]) / observed["shots"] - curve["model_P"]
            )
    first, second = saved["by_job_curves"]["jobs"].values()
    for key, pooled in saved["curves"].items():
        np.testing.assert_array_equal(
            np.array(first[key]["observed"]["failures"]) + second[key]["observed"]["failures"],
            pooled["observed"]["failures"],
        )

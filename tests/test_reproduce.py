"""The stored analyses come back from the stored shots: scripts/reproduce.py, as a test.

Pooling the stored analyses runs by default (seconds). Re-analysing the fifteen jobs
takes about 15 minutes with 4 workers and runs with --reproduce; --strict-floats then
asks for byte-identical files, which holds on the environment that analysed the runs
(requirements-lock.txt, Linux x86-64). The tiers are in scripts/reproduce.py.
Nothing under runs/ is written: every job is copied to a scratch folder first.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce.py"
_spec = importlib.util.spec_from_file_location("reproduce", SCRIPT)
assert _spec and _spec.loader
reproduce = importlib.util.module_from_spec(_spec)
sys.modules["reproduce"] = reproduce  # its dataclass looks itself up there
_spec.loader.exec_module(reproduce)

pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("qiskit")

JOBS = reproduce.jobs()
WORKERS = int(os.environ.get("HEAVYHEX_WORKERS", min(4, os.cpu_count() or 1)))
SIMULATED_SHOTS = reproduce.heavyhex_run().SIMULATED_SHOTS
SIDE_SHOTS = reproduce.heavyhex_run().SIDE_SHOTS


def stored(name: str) -> Path:
    return reproduce.RUNS / name / reproduce.ANALYSIS


def test_combine_reproduces_combined_json(run_folders, tmp_path, strict):
    """Pooling the stored analyses gives the stored combined.json, scipy's quantiles to an ulp."""
    reproduce.copy_stored(tmp_path)
    report = reproduce.compare_combined(
        reproduce.combine_in(tmp_path),
        reproduce.RUNS / reproduce.COMBINED,
        strict=strict,
        rule=lambda path: reproduce.ULP,
    )
    assert report.passed, report.failures


# --- the comparison itself, on one stored job --------------------------------------------


@pytest.fixture
def one_job(run_folders, tmp_path):
    """A stored analysis.json to perturb, and where to write the perturbed copy."""
    return json.loads(stored(JOBS[0]).read_text()), tmp_path / reproduce.ANALYSIS


def compare(analysis: dict, path: Path, *, strict: bool = False):
    path.write_text(json.dumps(analysis, indent=1) + "\n")
    return reproduce.compare_analysis(path, stored(JOBS[0]), strict=strict)


def test_an_unchanged_analysis_is_identical(one_job):
    analysis, path = one_job
    analysis["code"]["analyzed"] = {"commit": "elsewhere"}  # who re-analysed it doesn't count
    report = compare(analysis, path, strict=True)
    assert report.identical and report.passed


def test_strict_asks_for_the_bytes_analyze_writes(one_job):
    analysis, path = one_job
    path.write_text(json.dumps(analysis, indent=2))  # the same values, laid out otherwise
    assert reproduce.compare_analysis(path, stored(JOBS[0])).passed
    report = reproduce.compare_analysis(path, stored(JOBS[0]), strict=True)
    assert not report.identical and not report.passed


def test_a_changed_decoded_result_fails(one_job):
    analysis, path = one_job
    analysis["observed"]["settings"][0]["logical_error"][0] += 1e-12
    assert not compare(analysis, path).passed


def test_float_noise_in_a_fit_passes_except_when_strict(one_job):
    analysis, path = one_job
    analysis["idle_test"]["median"] *= 1 + 1e-13
    assert compare(analysis, path).passed
    assert not compare(analysis, path, strict=True).passed


def test_a_prediction_passes_within_the_simulators_noise(one_job):
    analysis, path = one_job
    row = analysis["predicted_at_measured_dephasing"]["settings"][0]["logical_error"]
    p, shots = row[0], SIMULATED_SHOTS
    row[0] = p + 2 * (2 * p * (1 - p) / shots) ** 0.5
    assert compare(analysis, path).passed
    row[0] = p + 8 * (2 * p * (1 - p) / shots) ** 0.5
    assert not compare(analysis, path).passed
    row[0] = -1e-6  # not a probability
    assert not compare(analysis, path).passed


def test_predictions_each_close_but_all_off_together_fail(one_job):
    """Three standard errors apart everywhere: no value fails alone, their mean square does."""
    analysis, path = one_job
    for block, shots in (
        ("predicted_at_measured_dephasing", SIMULATED_SHOTS),
        ("predicted_from_calibration_after", SIMULATED_SHOTS),
        ("predicted_with_each_qubits_dephasing", SIDE_SHOTS),
    ):
        for setting in analysis[block]["settings"]:
            for key in ("logical_error", "detector_rates"):
                values = setting.get(key, [])
                for k, p in enumerate(values):
                    values[k] = p + 3 * (2 * max(p * (1 - p), 1 / shots) / shots) ** 0.5
    report = compare(analysis, path)
    assert not report.failures and report.mean_square > reproduce.MEAN_SQUARE
    assert not report.passed


def test_what_depends_on_the_samples_is_not_compared(one_job):
    analysis, path = one_job
    analysis["worst_detectors"] = analysis["worst_detectors"][:1]
    analysis["detectors_beyond_chance"] += 3
    for row in analysis["agreement"]["tests"].values():
        row["agrees"] = not row["agrees"]
    assert compare(analysis, path).passed


def test_the_decisions_are_compared(run_folders, tmp_path):
    """combined.json's chi2 within 25%, and its verdicts equal away from the bound."""
    combined = json.loads((reproduce.RUNS / reproduce.COMBINED).read_text())

    def passes(change) -> bool:
        document = json.loads(json.dumps(combined))
        change(document)
        path = tmp_path / reproduce.COMBINED
        path.write_text(json.dumps(document, indent=1) + "\n")
        return reproduce.compare_combined(path, reproduce.RUNS / reproduce.COMBINED).passed

    def scale(name, factor):
        return lambda d: d["model"][name].update(chi2=d["model"][name]["chi2"] * factor)

    def flip(*keys):
        def change(d):
            *head, last = keys
            target = d
            for key in head:
                target = target[key]
            target[last] = not target[last]

        return change

    assert passes(lambda d: None)
    assert passes(scale("X eps3", 1.10))
    assert not passes(scale("X eps3", 1.50))
    assert not passes(flip("model", "X eps3", "agrees"))  # chi2 52, far from the bound
    assert passes(flip("model", "X eps5", "agrees"))  # chi2 19.8, within 25% of 18.31
    assert not passes(flip("matches", "Z"))


def test_a_missing_or_extra_key_fails(one_job):
    analysis, path = one_job
    analysis["leakage"]["extra"] = {}
    assert not compare(analysis, path).passed


def test_a_job_not_re_analysed_fails_the_comparison(run_folders, tmp_path):
    copy = tmp_path / "runs" / JOBS[0]
    copy.mkdir(parents=True)
    shutil.copy2(stored(JOBS[0]), copy / reproduce.ANALYSIS)
    assert reproduce.main(["compare", "--out", str(tmp_path), "--job", JOBS[0]]) == 0
    assert reproduce.main(["compare", "--out", str(tmp_path)]) == 1  # the other jobs are missing


def test_the_scratch_folder_is_never_runs():
    for out in (REPO, reproduce.RUNS, reproduce.RUNS / JOBS[0] if JOBS else reproduce.RUNS):
        with pytest.raises(ValueError, match="scratch folder"):
            reproduce.scratch(out)
        assert reproduce.main(["combine", "--out", str(out)]) == 2
    assert reproduce.scratch(REPO / "build" / "reproduce") == REPO / "build" / "reproduce"


# --- re-analysis of every job (--reproduce) -----------------------------------------------


@pytest.fixture(scope="session")
def reanalysed(tmp_path_factory) -> Path:
    """Every job re-analysed once, in parallel, and pooled, in a scratch folder."""
    out = tmp_path_factory.mktemp("reproduce")
    command = [sys.executable, str(SCRIPT), "analyze", "--out", str(out), "--jobs", str(WORKERS)]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return out


@pytest.mark.reproduce
@pytest.mark.parametrize("name", JOBS)
def test_reanalysis_matches_the_stored_analysis(name, reanalysed, strict):
    report = reproduce.compare_analysis(
        reanalysed / "runs" / name / reproduce.ANALYSIS, stored(name), strict=strict
    )
    assert report.passed, report.failures[:20]


@pytest.mark.reproduce
def test_reanalysis_pools_to_the_stored_combined_json(reanalysed, strict):
    report = reproduce.compare_combined(
        reanalysed / "runs" / reproduce.COMBINED,
        reproduce.RUNS / reproduce.COMBINED,
        strict=strict,
    )
    assert report.passed, report.failures[:20]

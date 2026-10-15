"""Primary depth differences are averaged within days before uncertainty is estimated."""

import numpy as np
import pytest
from scipy.stats import t


@pytest.fixture
def early(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs" / "analysis"))
    import early_depths

    return early_depths


def test_interval_is_additive_and_uses_sixteen_cell_family(early):
    daily = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    row = early.interval(daily)
    half = t.ppf(1 - 0.05 / 32, 4) * daily.std(ddof=1) / np.sqrt(5)
    assert row["mean"] == pytest.approx(0.03)
    assert row["ci95_family16"] == pytest.approx([0.03 - half, 0.03 + half])


@pytest.mark.parametrize("values", [[0.1] * 4, [0.1] * 6, [0.1] * 4 + [np.nan]])
def test_interval_rejects_invalid_daily_panel(early, values):
    with pytest.raises(ValueError):
        early.interval(values)


def test_primary_omissions_use_four_days_and_three_degrees_of_freedom(early):
    daily = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    rows = early.interval(daily)["leave_one_day_out"]
    for day in range(5):
        subset = np.delete(daily, day)
        half = t.ppf(1 - 0.05 / 32, 3) * subset.std(ddof=1) / 2
        assert rows[str(day + 1)]["mean"] == pytest.approx(subset.mean())
        assert rows[str(day + 1)]["ci95_family16"] == pytest.approx(
            [subset.mean() - half, subset.mean() + half]
        )


def test_saved_depth_cells_reconstruct_from_forecast_and_device_counts(early):
    saved = early.result("early_depths.json")["cells"]
    device = early.result("stored_priors.json")["curves"]
    forecast = early.result("fits_counts.json")["jobs"]
    reps = early.result("fits.json")["per_rep"]
    positive = 0
    fragile = 0
    for key, depths in saved.items():
        basis, d = key
        for depth, row in depths.items():
            daily = {}
            for rep in reps.values():
                job = rep["jobs"][d]
                observed = device[job][basis][d]["P"][int(depth) - 1]
                model = forecast[job]["predicted"]["frozen_f1.0"]["P"][basis][d][int(depth) - 1]
                daily.setdefault(rep["day"], []).append(observed - model)
            expected = early.interval(np.array([np.mean(daily[d]) for d in sorted(daily)]))
            assert row == expected
            positive += row["ci95_family16"][0] > 0
            fragile += row["ci95_family16"][0] > 0 and any(
                r["ci95_family16"][0] <= 0 for r in row["leave_one_day_out"].values()
            )
    assert positive == 15
    assert saved["Z3"]["1"]["ci95_family16"][0] < 0
    assert fragile == 7


def test_exact_ml_summary_uses_probabilities_not_legacy_rate_estimator(early):
    rows = early.result("decoder_ml.json")["patches"]
    change, gap = [], []
    for patch in rows.values():
        for row in patch["by_rounds"]:
            change.append(row["device_ml"] - row["device_matching"])
            gap.append(row["device_ml"] - row["model_ml"])
    saved = early.result("early_depths.json")["exact_ml_depth_diagnostic"]
    assert saved["device_probability_change_range"] == [min(change), max(change)]
    assert saved["device_minus_model_after_ml_range"] == [min(gap), max(gap)]


def test_reset_ground_bookkeeping_is_an_explicit_approximation(repo_root):
    from qiskit import QuantumCircuit

    from heavyhex.patches.placement import Calibration
    from heavyhex.simulation.scheduled import noisy_scheduled

    calibration = Calibration.load(repo_root / "runs/ibm_fez-2026-10-04-r1/calibration.json")
    circuit = QuantumCircuit(156, 1)
    circuit.reset(0)
    circuit.delay(100, 0, unit="dt")
    circuit.measure(0, 0)
    model, _ = noisy_scheduled(circuit, calibration, 2.22e-10)
    names = [instruction.name for instruction in model]
    assert "X_ERROR" in names
    assert "PAULI_CHANNEL_1" not in names
    # An actual reset-error population would partly relax during that skipped wait.
    e, ratio = 0.02, 0.1
    assert e - e * np.exp(-ratio) == pytest.approx(e * (1 - np.exp(-ratio)))

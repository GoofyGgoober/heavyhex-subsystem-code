"""The follow-up's analysis: the fits, T_q from the idle test, the records read back as sent,
and run 1's placements. Nothing here contacts IBM."""

from math import exp, pi
from pathlib import Path

import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("samplomatic")
pytest.importorskip("scipy")
pytest.importorskip("stim")

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKingston
from qiskit_ibm_runtime.results.quantum_program import (
    ItemMetadata,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SchedulerTiming,
)

from followup import analysis, model, submit
from followup import leakage_control as exp3
from followup.circuits import ROUNDS, Setting
from followup.placement import place
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import fit_per_round

OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
DT = 4e-9


def counts(eps: float, a: float, shots: int, rounds=ROUNDS) -> np.ndarray:
    return np.array([shots * (1 - a * (1 - 2 * eps) ** n) / 2 for n in rounds])


def test_binomial_returns_eps_and_amplitude_of_exact_counts():
    fit = analysis.binomial(ROUNDS, counts(0.06, 0.9, 8192), 8192)
    assert fit.eps == pytest.approx(0.06, abs=1e-6)
    assert fit.amplitude == pytest.approx(0.9, abs=1e-5)
    assert fit.dispersion == 1.0
    assert fit.low < 0.06 < fit.high


def test_binomial_and_log_fits_agree_on_sampled_counts():
    rng = np.random.default_rng(1)
    expected = counts(0.08, 0.95, 8192) / 8192
    pulls = []
    for _ in range(25):
        k = rng.binomial(8192, expected)
        b = analysis.binomial(ROUNDS, k, 8192)
        log = fit_per_round(ROUNDS, list(k / 8192), 8192)
        assert abs(b.eps - log.per_round) < log.uncertainty
        pulls.append((b.eps - 0.08) / b.sigma)
    assert abs(np.mean(pulls)) < 0.5
    assert 0.6 < np.std(pulls) < 1.4


def test_fits_give_lambda_by_both_fits():
    failures = {}
    for basis in "XZ":
        for d, eps in ((3, 0.04), (5, 0.08)):
            for n, k in zip(ROUNDS, counts(eps, 1.0, 8192), strict=True):
                failures[basis, d, n] = round(k)
    found = analysis.fits(failures, 8192)
    for basis in "XZ":
        for fit in analysis.FITS:
            value, sigma = found[basis]["lambda"][fit]
            assert value == pytest.approx(0.5, abs=0.01)
            assert 0 < sigma < 0.05


def test_ratio_propagates_relative_errors():
    value, sigma = analysis.ratio([2.0, 0.2], [4.0, 0.3])
    assert value == 0.5
    assert sigma == pytest.approx(0.5 * np.hypot(0.1, 0.075))


def toy_idle(waits_dt: tuple[int, ...], before_dt: int = 0) -> QuantumCircuit:
    """One data qubit: a wait before its first pulse, then H and a wait for each of waits_dt,
    H, read into d1[0]. Its state is on the equator in the first, third, ... wait."""
    qc = QuantumCircuit(1)
    qc.add_register(ClassicalRegister(1, "d1"))
    if before_dt:
        qc.delay(before_dt, 0, unit="dt")
    for wait in (*waits_dt, None):
        qc.rz(pi / 2, 0)
        qc.sx(0)
        qc.rz(pi / 2, 0)
        if wait is not None:
            qc.delay(wait, 0, unit="dt")
    qc.measure(0, 0)
    qc.delay(500, 0, unit="dt")  # after the readout: nothing to the readout
    return qc


TOY = Calibration(
    backend="toy",
    calibrated="",
    pulled="",
    cz={},
    readout={0: 0.02},
    t1_us={0: 100.0},
    t2_us={0: 70.0},
    single_qubit={0: 3e-4},
)


def test_equator_times_count_delays_with_the_state_off_the_z_axis():
    times = analysis.equator_times(toy_idle((2500, 700, 300), before_dt=1000), [0], DT)
    assert times[0] == pytest.approx((2500 + 300) * DT * 1e6)


def test_pulses_leave_the_gates_readout_and_t1_and_take_out_the_equator():
    """G = (1 - 2e)^m (1 - 2p) e^(-t_z/T1): each SX's depolarizing at 3/2 its error e, the
    readout, and T1 along Z; the equator's e^(-t/T2) is taken out."""
    for waits in ((0,), (2500,), (25000,), (2500, 700, 300)):
        t_z = sum(waits[1::2]) * DT * 1e6
        expected = (1 - 2 * 3e-4) ** (len(waits) + 1) * (1 - 2 * 0.02) * exp(-t_z / 100.0)
        assert analysis.pulses(toy_idle(waits), TOY, DT)[0] == pytest.approx(expected, rel=1e-9)


def test_equator_times_refuse_two_qubit_gates():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    with pytest.raises(ValueError, match="one-qubit gates only"):
        analysis.equator_times(qc, [0], DT)


def test_decay_time_returns_t_q_on_the_delay_clock():
    t = np.array([5.0, 11.0, 23.0, 47.0, 95.0])
    g = np.array([0.97, 0.95, 0.9, 0.82, 0.7])
    x = 0.98 * g * np.exp(-t / 60.0)
    n = np.full(len(t), 1e9)
    fit = analysis.decay_time(t, x, np.zeros_like(x), n, n, g)
    assert fit["t_q_us"] == pytest.approx(60.0, rel=1e-4)
    assert fit["offset"] == pytest.approx(0.98, rel=1e-4)
    assert fit["used"] == 5


def test_decay_time_turns_a_rotation_into_r_and_drops_lost_points():
    t = np.array([5.0, 11.0, 23.0, 47.0, 95.0])
    r = np.exp(-t / 20.0)
    angle = 0.1 * t
    n = np.full(len(t), 1e9)
    fit = analysis.decay_time(t, r * np.cos(angle), r * np.sin(angle), n, n, np.ones(len(t)))
    assert fit["t_q_us"] == pytest.approx(20.0, rel=1e-4)
    assert fit["used"] == 4  # e^(-95/20) is below MIN_R


def test_refiring_is_zero_for_independent_firings():
    rng = np.random.default_rng(0)
    assert abs(analysis.refiring(rng.random((20000, 4)) < 0.1)) < 0.01


def test_refiring_of_pairs():
    """Half the shots fire in rounds 1 and 2: after a firing 1/2 fire, after quiet 1/4."""
    pairs = np.zeros((1000, 4), dtype=bool)
    pairs[:500, 1:3] = True
    assert analysis.refiring(pairs) == pytest.approx(0.25)


@pytest.fixture(scope="module")
def kingston():
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns or {})


def write_job(folder: Path, job: submit.Job, target, arrays) -> Path:
    """A job's record as submit.submit writes it, with these readouts."""
    folder.mkdir(parents=True)
    submit._write(folder / "run.json", submit.describe(job, place(OCTOBER_13), target, OCTOBER_13))
    OCTOBER_13.save(folder / "calibration.json")
    submit.save_circuits(folder, job)
    results = [
        QuantumProgramItemResult(
            data,
            ItemMetadata(scheduler_timing=SchedulerTiming("", round(item.seconds / target.dt))),
        )
        for item, data in zip(job.items, arrays, strict=True)
    ]
    submit.save_results(folder, QuantumProgramResult(results))
    return folder


def test_memory_circuit_read_back_as_sent_decodes_to_nothing_without_noise(tmp_path, kingston):
    setting = Setting("memory", "Z", 3, 2)
    item = submit.twirled(setting, place(OCTOBER_13)[3].qubits, kingston)
    item = submit.Item(item.label, item.circuit, item.seconds, 4, item.samplex, 2, setting)
    drawn = submit.instances(item, 2, seed=3)
    blocks: dict[str, list[np.ndarray]] = {}
    for bound in drawn:
        noisy, records = model.noisy(bound, OCTOBER_13, DT)
        measured = noisy.without_noise().compile_sampler(seed=1).sample(4)
        for name in ("m3", "d3"):
            size = 1 + max(i for reg, i in records if reg == name)
            bits = np.zeros((4, size), dtype=bool)
            for (reg, i), k in records.items():
                if reg == name:
                    bits[:, i] = measured[:, k]
            blocks.setdefault(name, []).append(bits)
    data = {name: np.stack(parts) for name, parts in blocks.items()}  # randomizations x shots
    folder = write_job(tmp_path / "exp1-1", submit.Job("exp1-1", (item,)), kingston, [data])

    found = analysis.sent(folder)
    assert list(found) == [item.label]
    assert found[item.label][0] == setting
    assert found[item.label][1] == submit.instances(item, 1)[0]
    shots = analysis.readouts([folder, folder])
    assert shots[item.label]["m3"].shape[0] == 16
    row = analysis.memory(
        setting,
        found[item.label][1],
        shots[item.label],
        OCTOBER_13,
        DT,
        decay_us=None,
        caps=(0.1, None),
    )
    assert row["shots"] == 16
    assert row["failures"] == {"0.1": 0, "None": 0}
    assert not any(row["detection"].values())

    # The check before the next job passes these readouts, and not with the bits reversed.
    checks = submit.inspect(folder)
    assert checks["passed"] and checks["detection"][item.label]["chip"] == 0
    assert 0.02 < checks["detection"][item.label]["model"] < 0.2
    reversed_bits = {name: bits[..., ::-1] for name, bits in data.items()}
    other = write_job(
        tmp_path / "reversed", submit.Job("exp1-1", (item,)), kingston, [reversed_bits]
    )
    assert any("detectors fire" in why for why in submit.inspect(other)["failed"])


def test_exp3_jobs_join_the_pieces_and_read_the_period(tmp_path, kingston):
    selection = exp3.select(place(OCTOBER_13)[3].layout)
    items = submit.exp3_items("A", selection, kingston)
    data = exp3.simulate_job("A", [exp3.Leak()] * len(selection.qubits), seed=2)
    pieces = {name: list(np.split(v, len(v) // exp3.SHOTS["R0"])) for name, v in data.items()}
    arrays = [{"m_iq": pieces[item.label].pop(0)} for item in items]
    folder = write_job(
        tmp_path / "exp3-1", submit.Job("exp3-1", items, "kerneled"), kingston, arrays
    )

    jobs, period_us, kerneled = analysis.exp3_jobs([folder])
    assert kerneled
    assert period_us == pytest.approx(exp3.timing(selection, kingston).period_us)
    for name in ("R0", "R1", "A"):
        assert np.array_equal(jobs[0][name], data[name])


def test_run1_placements_pool_the_days_and_compare():
    def row(eps: float) -> dict:
        return {"log": [eps, 0.01], "binomial": [eps, 0.01]}

    jobs = [
        {
            "placement": 1,
            "day": day,
            "fits": {"None": {b: {3: row(0.10), 5: row(0.20)} for b in "XZ"}},
        }
        for day in ("a", "b", "c")
    ]
    found = analysis.placements(jobs, "None")
    eps3, sigma3 = found[1]["X"]["eps"][3]["log"]
    assert eps3 == pytest.approx(0.10)
    assert sigma3 == pytest.approx(0.01 / np.sqrt(3))
    assert found[1]["Z"]["lambda"]["binomial"][0] == pytest.approx(0.5)
    observed = {
        b: {"eps": {3: row(0.05), 5: row(0.10)}, "lambda": {f: [0.5, 0.05] for f in analysis.FITS}}
        for b in "XZ"
    }
    compared = analysis.against_run1(observed, found)
    assert compared[1]["X"]["log"]["eps3"][0] == pytest.approx(0.5)
    assert compared[1]["X"]["log"]["lambda_change"][0] == pytest.approx(0.0)


def test_against_model_divides_each_eps_and_lambda():
    fits = analysis.fits(
        {
            (b, d, n): round(k)
            for b in "XZ"
            for d, eps in ((3, 0.04), (5, 0.08))
            for n, k in zip(ROUNDS, counts(eps, 1.0, 8192), strict=True)
        },
        8192,
    )
    for basis in analysis.against(fits, fits).values():
        for values in basis.values():
            assert all(v[0] == pytest.approx(1.0) for v in values.values())


def test_decay_time_is_nan_with_one_point_left():
    t = np.array([5.0, 50.0])
    x = np.array([0.5, 0.01])
    n = np.full(2, 1e6)
    fit = analysis.decay_time(t, x, 0 * x, n, n, np.ones(2))
    assert np.isnan(fit["t_q_us"])
    assert fit["used"] == 1

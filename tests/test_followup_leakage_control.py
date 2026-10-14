"""Experiment 3's leakage control: the ten qubits, the arms' circuits and their timing, and
the analysis on synthetic IQ data with a planted leak."""

from collections import Counter
from dataclasses import replace
from math import isclose, pi
from pathlib import Path

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("stim")

import numpy as np
import stim

from followup.leakage_control import (
    ARMS,
    IQ,
    JOBS,
    PERIODS,
    Clouds,
    Leak,
    Selection,
    analyze,
    changes,
    circuit,
    classify,
    fit_hmm,
    qpu_seconds,
    select,
    simulate,
    simulate_job,
    third_cluster,
    timing,
)
from followup.placement import chip_map
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.layout import build_layout
from heavyhex.patches.placement import Calibration

# IBM's calibration of ibm_kingston on 13 October 2026, from which the plan was made.
OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
# The plan's d=3 patch, nested in the d=5 footprint.
PATCH = build_layout(3, *chip_map(), origin=(11, 9), direction=(1, -1))
SELECTION = select(PATCH)


@pytest.fixture(scope="module")
def target():
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    assert OCTOBER_13.durations_ns
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns)


def test_the_qubits_are_the_flags_x_ancillas_and_relays():
    assert dict(zip(SELECTION.qubits, SELECTION.roles, strict=True)) == {
        51: "flag",
        53: "X ancilla",
        55: "relay",
        69: "relay",
        71: "flag",
        73: "flag",
        75: "relay",
        89: "relay",
        91: "X ancilla",
        93: "flag",
    }
    czs = dict(zip(SELECTION.qubits, SELECTION.czs, strict=True))
    assert {q for q, n in czs.items() if n == 6} == {51, 71, 73, 93}
    assert {q for q, n in czs.items() if n == 4} == {53, 55, 69, 75, 89, 91}
    assert dict(zip(SELECTION.qubits, SELECTION.partners, strict=True))[51] == (50, 52, 58)


def test_coupled_qubits_are_refused():
    with pytest.raises(ValueError, match="coupled"):
        select(PATCH, [*chip_map()[1], [51, 53]])


def test_each_pair_takes_two_slots_and_no_qubit_two_cz_at_once():
    assert len(SELECTION.slots) == 6
    pairs = Counter(pair for slot in SELECTION.slots for pair in slot)
    expected = {
        (q, p) for q, ps in zip(SELECTION.qubits, SELECTION.partners, strict=True) for p in ps
    }
    assert set(pairs) == expected and set(pairs.values()) == {2}
    couplers = {frozenset((a, b)) for _, a, b in PATCH.couplings}
    assert all(frozenset(pair) in couplers for pair in pairs)
    for slot in SELECTION.slots:
        qubits = [q for pair in slot for q in pair]
        assert len(qubits) == len(set(qubits))


def test_the_period_is_4864_ns_and_a_job_about_4_s(target):
    t = timing(SELECTION, target)
    assert t.period == 1216 and isclose(t.period_us, 4.864)
    assert all(3.7 < qpu_seconds(job, t) < 3.9 for job in JOBS)


def cycle_lengths(qc, target):
    """Each cycle's length on every qubit, in dt, from a clock per qubit; every CZ's two
    qubits must reach it at the same time."""
    clock: Counter[int] = Counter()
    last: Counter[int] = Counter()
    cycles = []
    for item in qc.data:
        name = item.operation.name
        qubits = [qc.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            cycles.append({q: clock[q] - last[q] for q in qubits})
            last = clock.copy()
            continue
        if name == "delay":
            length = item.operation.duration
        elif name == "rz":
            length = 0
        else:
            gate = target[name].get(tuple(qubits)) or target[name][tuple(qubits[::-1])]
            length = round(gate.duration / target.dt)
        assert len({clock[q] for q in qubits}) == 1, (name, qubits)
        for q in qubits:
            clock[q] += length
    return cycles


@pytest.mark.parametrize("arm", ARMS)
def test_every_qubit_fills_every_cycle(arm, target):
    qc = circuit(arm, SELECTION, target, cycles=3)
    period = PERIODS[arm] * timing(SELECTION, target).period
    cycles = cycle_lengths(qc, target)
    assert len(cycles) == 3
    assert all(set(c.values()) == {period} for c in cycles)
    names = Counter(item.operation.name for item in qc.data)
    assert names["cz"] == (3 * sum(SELECTION.czs) if arm == "B" else 0)
    assert names["reset"] == (0 if arm == "C" else 3 * 10)
    assert names["x"] == 3 * len(SELECTION.partner_qubits)  # one per partner per cycle
    bits = [
        qc.find_bit(item.clbits[0]).index for item in qc.data if item.operation.name == "measure"
    ]
    assert sorted(bits) == list(range(30))


def test_the_references_read_each_qubit_once(target):
    for arm, gates in (("R0", {"measure": 10}), ("R1", {"x": 10, "measure": 10})):
        qc = circuit(arm, SELECTION, target)
        assert Counter(item.operation.name for item in qc.data) == gates
        assert not sample(qc).any() if arm == "R0" else sample(qc).all()


def sample(qc, shots=32):
    """The circuit's readouts in stim, by classical bit. Every rz is +-pi/2 (an S)."""
    out, order = stim.Circuit(), []
    for item in qc.data:
        name = item.operation.name
        qubits = [qc.find_bit(q).index for q in item.qubits]
        if name == "rz":
            out.append("S" if isclose(item.operation.params[0], pi / 2) else "S_DAG", qubits)
        elif name in ("sx", "x", "cz", "reset"):
            out.append({"sx": "SQRT_X", "x": "X", "cz": "CZ", "reset": "R"}[name], qubits)
        elif name == "measure":
            out.append("M", qubits)
            order.append(qc.find_bit(item.clbits[0]).index)
    bits = np.zeros((shots, qc.num_clbits), dtype=np.uint8)
    bits[:, order] = out.compile_sampler(seed=1).sample(shots)
    return bits


@pytest.mark.parametrize("arm", ARMS)
def test_every_readout_is_ideally_0(arm, target):
    assert not sample(circuit(arm, SELECTION, target, cycles=4)).any()


def test_one_cz_with_each_partner_reads_at_random(target):
    """The partners sit in |+>: half the CZ block would flip the qubit under test."""
    half = replace(SELECTION, slots=(*SELECTION.slots[:3], (), (), ()))
    bits = sample(circuit("B", half, target, cycles=4), shots=256)
    assert 0.35 < bits.mean() < 0.65


def test_arm_c_reads_changes_and_a_blip_once():
    readouts = np.array([[0, 1, 1, 0, 2, 0, 1, 2, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0]])
    assert changes(readouts).tolist() == [[0, 1, 0, 1, 2, 0, 1, 2, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0]]


def test_misreads_in_arm_c_are_no_persistent_state():
    job = simulate_job("C", [Leak(), Leak(0.001, 0.15)], seed=2)
    clouds = Clouds.fit(job["R0"][:, 0], job["R1"][:, 0])
    readouts = job["C"].reshape(len(job["C"]), -1, 2)
    assert not fit_hmm(changes(classify(readouts[:, :, 0], clouds)))["present"]
    fit = fit_hmm(changes(classify(readouts[:, :, 1], clouds)))
    assert fit["present"] and isclose(fit["a"], 0.001, rel_tol=0.3)


def test_the_reference_clouds_ignore_the_decay_tail():
    rng = np.random.default_rng(3)
    iq = IQ(decay=0.05)
    clouds = Clouds.fit(
        simulate("R0", 4000, 1, Leak(), iq, rng), simulate("R1", 4000, 1, Leak(), iq, rng)
    )
    assert np.allclose(clouds.zero.mean, [0, 0], atol=0.02)
    assert np.allclose(clouds.one.mean, [1, 0], atol=0.02)
    for cloud in (clouds.zero, clouds.one):
        assert np.allclose(cloud.cov, iq.sigma**2 * np.eye(2), atol=0.1 * iq.sigma**2)
    points = np.array([0, 1, iq.leaked, 0.3, 0.7j, 1j])
    assert classify(points, clouds).tolist() == [0, 1, 2, 0, 0, 2]


@pytest.fixture(scope="module")
def leaky():
    """Arm A of three qubits: a leak, none, and a small one."""
    job = simulate_job("A", [Leak(0.01, 0.15), Leak(), Leak(0.002, 0.15)], seed=1)
    clouds = Clouds.fit(job["R0"][:, 0], job["R1"][:, 0])
    readouts = job["A"].reshape(len(job["A"]), -1, 3)
    return clouds, [readouts[:, :, i] for i in range(3)]


def test_the_model_finds_a_planted_leak_and_its_cloud(leaky):
    clouds, readouts = leaky
    for z, a in ((readouts[0], 0.01), (readouts[2], 0.002)):
        fit = fit_hmm(classify(z, clouds))
        assert fit["present"] and not fit["capped"]
        assert isclose(fit["a"], a, rel_tol=0.2) and isclose(fit["b"], 0.15, abs_tol=0.025)
        assert fit["e_P"][2] > 0.85  # the persistent state reads outside
        third = third_cluster(z, clouds)
        assert third["accepted"]
        assert isclose(third["along"], 1.3, abs_tol=0.05)
        assert isclose(third["across"], 1.0, abs_tol=0.05)


def test_without_a_leak_there_is_no_persistent_state_or_third_cloud(leaky):
    clouds, readouts = leaky
    assert not fit_hmm(classify(readouts[1], clouds))["present"]
    assert not third_cluster(readouts[1], clouds)["accepted"]


def test_the_fallback_finds_the_leak_in_0_1_readouts():
    job = simulate_job("A", [Leak(0.01, 0.15), Leak()], kerneled=False, seed=2)
    readouts = job["A"].reshape(len(job["A"]), -1, 2)
    fit = fit_hmm(readouts[:, :, 0], symbols=2)
    assert fit["present"] and isclose(fit["a"], 0.01, rel_tol=0.25)
    assert isclose(fit["b"], 0.15, abs_tol=0.03)
    assert not fit_hmm(readouts[:, :, 1], symbols=2)["present"]


def test_arm_c_bounds_a_state_that_reads_a_steady_1():
    """In 0/1 readouts the changes can't see it; as read, it shows together with the |1>
    that the missing reset leaves, so its onset is an upper bound."""
    read = simulate_job("C", [Leak(0.002, 0.15)], kerneled=False, seed=3)["C"]
    assert not fit_hmm(changes(read), symbols=2)["present"]
    bound = fit_hmm(read, symbols=2)
    assert bound["present"] and bound["a"] > 0.002


def test_a_state_read_inside_the_1_cloud_leaves_reset_and_readout_unsplit():
    """Arm C can't see a state that reads a steady 1, so a_A - a_C isn't the reset's part:
    only the bounds are given."""
    relay = Selection(qubits=(55,), roles=("relay",), partners=((54, 56),), slots=())
    iq = IQ(leaked=1 + 0j)
    shots = {"R0": 1000, "R1": 1000, "A": 1500, "C": 1500}
    jobs = [
        simulate_job(arm, [Leak(0.008, 0.2)], iq, shots=shots, seed=i) for i, arm in enumerate("AC")
    ]
    found = analyze(jobs, relay, 4.864, reps=3)["qubits"][0]
    assert found["steady"]
    assert found["arms"]["A"]["hmm"]["present"] and not found["arms"]["A"]["third"]["accepted"]
    assert not found["arms"]["C"]["hmm"]["present"]
    assert "reset" not in found["causes"] and "readout_idle" not in found["causes"]
    assert found["causes"]["readout_idle_at_most"][0] >= 0.008


def test_the_causes_come_apart():
    """A flag whose onset is 0.1% per CZ, 0.2% per reset and 0.1% from the rest."""
    flag = Selection(qubits=(51,), roles=("flag",), partners=((50, 52, 58),), slots=())
    onsets = {"A": 0.003, "B": 0.009, "C": 0.001}
    jobs = [simulate_job(arm, [Leak(a, 0.15)], seed=i) for i, (arm, a) in enumerate(onsets.items())]
    found = analyze(jobs, flag, 4.864, reps=20)["qubits"][0]
    assert found["arms"]["B"]["hmm"]["present"] and not found["steady"]
    for cause, value in (("cz_each", 0.001), ("reset", 0.002), ("readout_idle", 0.001)):
        fitted, low, high = found["causes"][cause]
        assert low <= fitted <= high
        assert isclose(fitted, value, abs_tol=0.0006), (cause, fitted)
    assert found["causes"]["readout_idle_at_most"][0] >= found["causes"]["readout_idle"][0]
    assert isclose(found["arms"]["B"]["hmm"]["lifetime_us"], -4.864 / np.log(0.85), rel_tol=0.2)
    assert found["references"][0]["p1_given_0"] < 0.02

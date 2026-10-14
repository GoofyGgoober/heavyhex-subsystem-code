"""The follow-up's calibration model: its channels, carried qubits, misreads and predictions."""

from dataclasses import replace
from math import exp, inf
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("samplomatic")
pytest.importorskip("stim")

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKingston

from followup import decoder, model, native
from followup.circuits import Setting, noiseless_check
from followup.placement import place
from heavyhex.circuits.flagged import qubit_roles
from heavyhex.experiment import circuits as run1
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import single_qubit_error
from heavyhex.simulation.scheduled import noisy_scheduled

OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
DT = 4e-9
# Nothing but the readouts' errors, which the tests set qubit by qubit.
QUIET = replace(
    OCTOBER_13,
    cz=dict.fromkeys(OCTOBER_13.cz, 0.0),
    readout=dict.fromkeys(OCTOBER_13.readout, 0.0),
    t1_us=dict.fromkeys(OCTOBER_13.t1_us, inf),
    t2_us=dict.fromkeys(OCTOBER_13.t2_us, inf),
    single_qubit=dict.fromkeys(OCTOBER_13.readout, 0.0),
)


@pytest.fixture(scope="module")
def target():
    assert OCTOBER_13.durations_ns is not None
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns)


@pytest.fixture(scope="module")
def placements():
    return place(OCTOBER_13)


def channels(circuit):
    """(name, qubits, arguments) of every instruction."""
    return [
        (item.name, [t.value for t in item.targets_copy()], item.gate_args_copy())
        for item in circuit.flattened()
    ]


def idle(t_ns, q, t2_us=None):
    t1 = OCTOBER_13.t1_us[q] * 1e3
    t2 = min((t2_us or OCTOBER_13.t2_us[q]) * 1e3, 2 * t1)
    px = (1 - exp(-t_ns / t1)) / 4
    return pytest.approx([px, px, (1 - exp(-t_ns / t2)) / 2 - px])


def test_a_qubit_is_quiet_until_its_first_gate_and_after_a_reset():
    c = QuantumCircuit(1, 1)
    c.delay(100, 0, unit="dt")
    c.sx(0)
    c.delay(100, 0, unit="dt")
    c.reset(0)
    c.delay(100, 0, unit="dt")
    c.x(0)
    out, _ = model.noisy(c, OCTOBER_13, DT)
    names = [name for name, _, _ in channels(out)]
    assert names == ["SQRT_X", "DEPOLARIZE1", "PAULI_CHANNEL_1", "R", "X_ERROR", "X", "DEPOLARIZE1"]
    assert channels(out)[2][2] == idle(400, 0)
    assert channels(out)[1][2] == [pytest.approx(1.5 * single_qubit_error(OCTOBER_13, 0))]
    assert channels(out)[4][2] == [OCTOBER_13.readout[0]]


def test_a_read_qubit_decays_until_it_is_used_again():
    c = QuantumCircuit(1, 1)
    c.sx(0)
    c.measure(0, 0)
    c.delay(500, 0, unit="dt")
    out, records = model.noisy(c, OCTOBER_13, DT)
    (name, _, (px, py, _)) = channels(out)[-1]
    assert name == "PAULI_CHANNEL_1"
    # A flip with probability (1 - e^(-t/T1))/2: |1> decays, averaged over the readout.
    assert px + py == pytest.approx((1 - exp(-2000 / (OCTOBER_13.t1_us[0] * 1e3))) / 2)
    assert channels(out)[2] == ("M", [0], [OCTOBER_13.readout[0]])
    assert records == {("c", 0): 0}


def test_each_data_qubits_own_decay_time_capped_at_2_t1():
    c = QuantumCircuit(2)
    c.sx([0, 1])
    c.delay(1000, [0, 1], unit="dt")
    out, _ = model.noisy(c, OCTOBER_13, DT, decay_us={0: 20.0, 1: 1e9})
    by_qubit = {q[0]: args for name, q, args in channels(out) if name == "PAULI_CHANNEL_1"}
    assert by_qubit[0] == idle(4000, 0, t2_us=20.0)
    assert by_qubit[1] == idle(4000, 1, t2_us=2 * OCTOBER_13.t1_us[1])


def test_a_data_qubit_takes_its_in_situ_t1():
    c = QuantumCircuit(2)
    c.sx([0, 1])
    c.delay(1000, [0, 1], unit="dt")
    out, _ = model.noisy(c, model.with_t1(OCTOBER_13, {0: 30.0}), DT)
    by_qubit = {q[0]: args for name, q, args in channels(out) if name == "PAULI_CHANNEL_1"}
    px = (1 - exp(-4000 / 30e3)) / 4
    t2 = min(OCTOBER_13.t2_us[0], 60.0) * 1e3
    assert by_qubit[0] == pytest.approx([px, px, (1 - exp(-4000 / t2)) / 2 - px])
    assert by_qubit[1] == idle(4000, 1)
    assert OCTOBER_13.t1_us[0] != 30.0


def test_twirled_cz_and_virtual_rz():
    c = QuantumCircuit(2)
    c.rz(3.141592653589793 / 2, 0)
    c.rz(-3.141592653589793, 1)
    c.cz(0, 1)
    out, _ = model.noisy(c, OCTOBER_13, DT)
    error = OCTOBER_13.cz[0, 1]
    assert error is not None
    assert channels(out) == [
        ("S", [0], []),
        ("Z", [1], []),
        ("CZ", [0, 1], []),
        ("DEPOLARIZE2", [0, 1], [pytest.approx(1.25 * error)]),
    ]
    c.rz(0.3, 0)
    with pytest.raises(ValueError, match="not a Clifford"):
        model.noisy(c, OCTOBER_13, DT)


def test_run1s_circuits_get_run1s_model(target, placements):
    """On run 1's design the model is run 1's own, with IBM's T2 throughout (f = 1)."""
    circuit, _ = run1.logical_circuit(
        run1.Setting("memory", "Z", 2), {"3": placements[3].qubits}, target.num_qubits
    )
    sent = run1.for_fez(circuit, target)
    mine, records = model.noisy(sent, OCTOBER_13, target.dt)
    theirs, their_records = noisy_scheduled(sent, OCTOBER_13, target.dt, decoupled_dephasing=1.0)
    assert mine == theirs and records == their_records


@pytest.mark.parametrize("basis", "XZ")
def test_the_compiled_circuits_are_deterministic_without_noise(target, placements, basis):
    for d in (3, 5):
        t = native.template(Setting("memory", basis, d, 2), placements[d].qubits, target)
        noisy, labels = model.memory(native.instance(t), t.schedule, OCTOBER_13, target.dt)
        noiseless_check(noisy)
        assert len(labels) == noisy.num_detectors


@pytest.mark.parametrize("kind", ["memory", "idle"])
def test_the_model_of_the_circuits_as_sent(target, placements, kind):
    """submit.py's circuits, as sent, and native.py's give the model the same errors."""
    submit = pytest.importorskip("followup.submit")
    setting = Setting(kind, "Z" if kind == "memory" else "X", 3 if kind == "memory" else 5, 2)
    chip = placements[setting.distance].qubits
    sent = submit.instances(submit.twirled(setting, chip, target), 1, seed=5)[0]
    mine = native.instance(native.template(setting, chip, target), seed=6)
    if kind == "memory":
        _, schedule = native.logical_circuit(setting, chip, target.num_qubits)
        models = [model.memory(c, schedule, OCTOBER_13, target.dt)[0] for c in (sent, mine)]
        errors = [m.detector_error_model(decompose_errors=True) for m in models]
        assert errors[0].approx_equals(errors[1], atol=1e-12)
    else:  # the same channels on the same qubits, whatever the Paulis drawn
        noise = [
            sorted((name, str(q), str(args)) for name, q, args in channels(noisy) if args)
            for noisy in (model.noisy(c, OCTOBER_13, target.dt)[0] for c in (sent, mine))
        ]
        assert noise[0] == noise[1]


def misread_rounds(target, placements, *, resets):
    """The rounds of the detectors that each misread of one Z4 ancilla fires, Z memory."""
    patch, placement = build_operators(3), placements[3]
    gauge = next(iter(qubit_roles(patch).z4_arms))
    ancilla = placement.layout.z_ancillas[gauge]
    calibration = replace(QUIET, readout={**QUIET.readout, ancilla: 0.01})
    setting = Setting("memory", "Z", 3, 4)
    t = native.template(setting, placement.qubits, target, twirl=False, xy4=False, resets=resets)
    noisy, labels = model.memory(t.circuit, t.schedule, calibration, target.dt)
    fired, _ = decoder.faults(noisy.detector_error_model())
    return {tuple(labels[d][0] for d in np.flatnonzero(row)) for row in fired}


def test_a_misread_fires_detectors_two_rounds_apart(target, placements):
    # In round t the misread spoils z_t and the carry of z_(t+1): D_t and D_(t+2), the last
    # round's D_3 and the data readout's D_4.
    assert misread_rounds(target, placements, resets=False) == {(0, 2), (1, 3), (2, 4), (3, 4)}
    # With run 1's resets: D_t and D_(t+1).
    assert misread_rounds(target, placements, resets=True) == {(0, 1), (1, 2), (2, 3), (3, 4)}


def test_predictions(target, placements):
    chips = {d: p.qubits for d, p in placements.items()}
    found = model.predict(native.run1_circuits(chips, target), OCTOBER_13, target.dt, shots=4000)
    assert found["rule"] is None  # at d=3, 4.5-5.2% data readouts make two-edge detours likelier
    assert found["fits"] == found["decoders"][str(decoder.PRIMARY)]["fits"]  # cap 0.1 leads
    assert all(0 < p < 0.5 for p in found["logical_error"].values())
    for basis in "XZ":
        eps, lam = found["fits"][basis]["eps"], found["fits"][basis]["lambda"]
        assert 0 < eps[3][0] < eps[5][0] < 0.25 and lam[0] < 1
        for d in (3, 5):
            assert 0.001 < found["precision"][basis]["eps"][d][1] < 0.01  # at 8192 shots
    assert set(found["detection"]["Z", 3, 2]) == {"stabilizer", "final", "z_flag", "relay"}

"""The stim version of the flagged circuit: gates, detectors, noise and the fit."""

from dataclasses import replace

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("qiskit")

import heavyhex.simulation.noisy as noisy  # noqa: E402
from heavyhex.circuits.flagged import memory_circuit_flagged  # noqa: E402
from heavyhex.decoders.shots import Memory, read_shot  # noqa: E402
from heavyhex.patches.operators import build_operators  # noqa: E402
from heavyhex.patches.placement import LAST_CALIBRATION, Calibration  # noqa: E402

SAVED = Calibration.load(LAST_CALIBRATION)
TYPICAL = noisy.typical(SAVED)
CASES = [(d, basis) for d in (3, 5) for basis in ("Z", "X")]


def measurement_order(circuit, schedule):
    """(register, bit) of each measurement, in circuit order."""
    gauge = len(schedule.measurements)
    order = []
    for item in circuit.data:
        if item.operation.name == "measure":
            i = circuit.clbits.index(item.clbits[0])
            order.append(("m", i) if i < gauge else ("d", i - gauge))
    return order


@pytest.mark.parametrize("distance, basis", CASES)
def test_gates_are_the_qiskit_circuit(distance, basis):
    circuit, _ = memory_circuit_flagged(build_operators(distance), rounds=2, basis=basis)
    names = {"h": "H", "cx": "CX", "reset": "R", "measure": "M"}
    expected = [
        (names[i.operation.name], tuple(circuit.find_bit(q).index for q in i.qubits))
        for i in circuit.data
        if i.operation.name != "barrier"
    ]
    got = []
    for i in noisy.noisy_circuit(distance, basis, 2, TYPICAL).flattened():
        if i.name in ("H", "CX", "R", "M"):
            targets = [t.value for t in i.targets_copy()]
            step = 2 if i.name == "CX" else 1
            got += [(i.name, tuple(targets[k : k + step])) for k in range(0, len(targets), step)]
    assert got == expected


@pytest.mark.parametrize("distance, basis", CASES)
def test_noiseless_detectors_and_logical_never_fire(distance, basis):
    circuit = noisy.noisy_circuit(distance, basis, 3, TYPICAL).without_noise()
    detectors, logical = circuit.compile_detector_sampler(seed=1).sample(
        200, separate_observables=True
    )
    assert not detectors.any() and not logical.any()


@pytest.mark.parametrize("distance, basis", CASES)
@pytest.mark.parametrize("rounds", [1, 3])
def test_detectors_and_logical_match_read_shot(distance, basis, rounds):
    patch = build_operators(distance)
    circuit = noisy.noisy_circuit(distance, basis, rounds, TYPICAL, scale=3)
    qiskit_circuit, schedule = memory_circuit_flagged(patch, rounds=rounds, basis=basis)
    order = measurement_order(qiskit_circuit, schedule)
    measurements = circuit.compile_sampler(seed=2).sample(300)
    detectors, logical = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True
    )
    used = Memory(patch, basis, rounds).basis_indices
    # In Z memory every flag and relay should read 0, so each is a detector too.
    kinds = ("z_flag", "relay") if basis == "Z" else ()
    flags_and_relays = [m.bit for m in schedule.measurements if m.kind in kinds]
    assert circuit.num_detectors == len(used) + len(flags_and_relays)
    for shot, detector_row, logical_bit in zip(measurements, detectors, logical[:, 0]):
        gauge_bits = [0] * len(schedule.measurements)
        data_bits = [0] * patch.code.n
        for (register, i), bit in zip(order, shot):
            (gauge_bits if register == "m" else data_bits)[i] = int(bit)
        expected = read_shot(schedule, tuple(gauge_bits), tuple(data_bits))
        assert [int(b) for b in detector_row] == [
            *(expected.detectors[i] for i in used),
            *(gauge_bits[b] for b in flags_and_relays),
        ]
        assert logical_bit == expected.logical_bit


@pytest.mark.parametrize(
    "distance, basis, fewest_faults",
    [(3, "Z", 3), (5, "Z", 5), (3, "X", 3), (5, "X", 5)],
)
def test_circuit_distance(distance, basis, fewest_faults):
    circuit = noisy.noisy_circuit(distance, basis, 2, TYPICAL)
    assert len(circuit.shortest_graphlike_error()) == fewest_faults


def test_no_errors_without_noise():
    circuit = noisy.noisy_circuit(3, "X", 2, TYPICAL, scale=0)
    assert noisy.logical_errors(circuit, 1000, seed=1) == 0


def test_scale_multiplies_gate_errors():
    def cz_errors(scale):
        circuit = noisy.noisy_circuit(3, "Z", 1, TYPICAL, scale=scale)
        return [i.gate_args_copy()[0] for i in circuit.flattened() if i.name == "DEPOLARIZE2"]

    assert cz_errors(2) == pytest.approx([2 * p for p in cz_errors(1)])
    assert cz_errors(1)[0] == pytest.approx(1.25 * TYPICAL.cz[next(iter(TYPICAL.cz))])


@pytest.mark.parametrize("scale", [1, 50])
def test_idle_channels_are_valid(scale):
    circuit = noisy.noisy_circuit(3, "X", 2, TYPICAL, scale=scale)
    idles = [i.gate_args_copy() for i in circuit.flattened() if i.name == "PAULI_CHANNEL_1"]
    assert idles and all(px == py and pz >= 0 and px + py + pz <= 1 for px, py, pz in idles)


def test_decoupling_leaves_almost_no_dephasing():
    circuit = noisy.noisy_circuit(3, "X", 2, TYPICAL, decoupling=True)
    for i in circuit.flattened():
        if i.name == "PAULI_CHANNEL_1":
            px, _, pz = i.gate_args_copy()
            assert pz < 2 * px * px + 1e-12  # second order in t / T1


def test_no_idle_noise_between_a_reset_and_the_first_cx():
    just_reset = set()
    for i in noisy.noisy_circuit(3, "Z", 2, TYPICAL).flattened():
        qubits = [t.value for t in i.targets_copy() if t.is_qubit_target]
        if i.name == "R":
            just_reset.update(qubits)
        elif i.name == "CX":
            just_reset.difference_update(qubits)
        elif i.name == "PAULI_CHANNEL_1":
            assert not just_reset & set(qubits)


def test_typical_puts_everything_at_the_median():
    assert len(set(TYPICAL.cz.values())) == len(set(TYPICAL.readout.values())) == 1
    assert len(set(TYPICAL.t1_us.values())) == len(set(TYPICAL.t2_us.values())) == 1
    assert TYPICAL.broken_qubits == ()
    assert 0.001 < TYPICAL.cz[0, 1] < 0.01 and 0.005 < TYPICAL.readout[0] < 0.02


def test_repaired_fixes_only_broken_parts():
    broken = replace(
        SAVED,
        cz={**SAVED.cz, (0, 1): 1.0},
        readout={**SAVED.readout, 5: 1.0},
        t1_us={**SAVED.t1_us, 6: 20.0},
        t2_us={**SAVED.t2_us, 6: 0.0},
    )
    fixed, median = noisy.repaired(broken), noisy.typical(broken)
    assert fixed.cz[0, 1] == median.cz[0, 1]
    assert fixed.readout[5] == median.readout[5]
    assert fixed.t2_us[6] <= 2 * fixed.t1_us[6]  # a broken qubit is fixed whole
    assert fixed.cz[1, 2] == SAVED.cz[1, 2] and fixed.readout[0] == SAVED.readout[0]
    assert fixed.broken_qubits == ()


@pytest.mark.parametrize("intercept", [1.0, 0.8])
def test_decay_recovers_a_known_error_per_round(monkeypatch, intercept):
    def exact(n, shots, seed=None):
        return round(shots * (1 - intercept * (1 - 2 * 0.03) ** n) / 2)

    monkeypatch.setattr(noisy, "noisy_circuit", lambda distance, basis, n, *a, **k: n)
    monkeypatch.setattr(noisy, "logical_errors", exact)
    fit = noisy.decay(3, "X", TYPICAL, shots=10**7)
    assert fit.per_round == pytest.approx(0.03, abs=1e-4)


def test_decay_gives_up_when_every_point_is_random(monkeypatch):
    monkeypatch.setattr(noisy, "noisy_circuit", lambda distance, basis, n, *a, **k: n)
    monkeypatch.setattr(noisy, "logical_errors", lambda n, shots, seed=None: shots // 2)
    assert noisy.decay(3, "X", TYPICAL, shots=1000).per_round == 0.5


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_d5_beats_d3_well_below_threshold(basis):
    kwargs = dict(rounds=(2, 4, 6), shots=100_000, seed=1, scale=0.05, decoupling=True)
    d3 = noisy.decay(3, basis, TYPICAL, **kwargs).per_round
    d5 = noisy.decay(5, basis, TYPICAL, **kwargs).per_round
    assert d5 < d3 / 2

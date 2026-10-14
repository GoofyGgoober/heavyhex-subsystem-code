"""The follow-up's circuits: run 1's without resets, the carries, and the detectors."""

import random
from itertools import groupby

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("stim")

import stim

from followup.circuits import (
    SETTINGS,
    Setting,
    add_detectors,
    logical_circuit,
    memory_circuit,
    noiseless_check,
    stim_circuit,
    to_stim,
)
from heavyhex.circuits.flagged import memory_circuit_flagged, qubit_roles
from heavyhex.circuits.shots import Memory, read_shot
from heavyhex.core import Pauli
from heavyhex.patches.operators import D3, D5
from heavyhex.simulation.aer import run_flagged_circuit

CASES = [(patch, basis) for patch in (D3, D5) for basis in "XZ"]
IDS = [f"d{patch.distance}-{basis}" for patch, basis in CASES]
MEMORY = [s for s in SETTINGS if s.decoded]


def gates(circuit):
    """(name, qubits, clbits) of every instruction."""
    return [
        (
            item.operation.name,
            tuple(circuit.find_bit(q).index for q in item.qubits),
            tuple(circuit.find_bit(c).index for c in item.clbits),
        )
        for item in circuit.data
    ]


def flat(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    return [i for i in circuit.flattened() if isinstance(i, stim.CircuitInstruction)]


def with_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """p after every gate, before every readout, and on every readout's record."""
    out = stim.Circuit()
    for item in flat(circuit):
        targets = item.targets_copy()
        if item.name == "M":
            out.append("X_ERROR", targets, p)
            out.append("M", targets, p)
            continue
        out.append(item)
        if item.name in ("H", "S", "S_DAG"):
            out.append("DEPOLARIZE1", targets, p)
        elif item.name == "CX":
            out.append("DEPOLARIZE2", targets, p)
    return out


def patch_records(records):
    return {(name[0], i): k for (name, i), k in records.items()}


def test_settings_follow_the_plan():
    idle = [s for s in SETTINGS if not s.decoded]
    assert len(MEMORY) == 16
    assert {(s.distance, s.basis, s.rounds) for s in MEMORY} == {
        (d, basis, n) for d in (3, 5) for basis in "XZ" for n in (1, 2, 3, 4)
    }
    assert all(s.distance == 5 and s.basis == "X" for s in idle)
    assert sorted((s.readout, s.rounds) for s in idle) == [
        (readout, n) for readout in "XY" for n in (1, 2, 4, 8, 16)
    ]


@pytest.mark.parametrize("patch, basis", CASES, ids=IDS)
def test_gates_are_run_1s_without_resets(patch, basis):
    old, old_schedule = memory_circuit_flagged(patch, rounds=3, basis=basis)
    new, schedule = memory_circuit(patch, rounds=3, basis=basis)
    assert gates(new) == [g for g in gates(old) if g[0] != "reset"]
    # Every readout has its own bit.
    assert sorted(c for name, _, c in gates(new) if name == "measure") == [
        (k,) for k in range(new.num_clbits)
    ]
    assert schedule.measurements == old_schedule.measurements
    old_gates, new_gates = gates(old), gates(new)
    for before, after in zip(old_schedule.gadgets, schedule.gadgets, strict=True):
        kept = [old_gates[i] for i in before.gates if old_gates[i][0] != "reset"]
        assert [new_gates[i] for i in after.gates] == kept


@pytest.mark.parametrize("patch, basis", CASES, ids=IDS)
def test_carries_are_the_hand_derived_ones(patch, basis):
    """X gauge: the ancilla's last readout. Z4: the ancilla's. Z2: the ancilla's and both
    relays'. Flag or relay: its own."""
    _, schedule = memory_circuit(patch, rounds=3, basis=basis)
    roles = qubit_roles(patch)
    last: dict[int, int] = {}
    expected = []
    for _, half in groupby(schedule.measurements, key=lambda m: (m.round, m.half)):
        before, relays_read = dict(last), {}
        for m in half:
            if m.kind in ("x_gauge", "z_flag"):
                qubit = roles.x_ancillas[m.gauge]
            elif m.kind == "z_syn":
                qubit = roles.z_ancillas[m.gauge]
            else:
                k = relays_read[m.gauge] = relays_read.get(m.gauge, -1) + 1
                qubit = roles.z2_arms[m.gauge][k][1]
            sources = [qubit]
            if m.kind == "z_syn" and m.gauge in roles.z2_arms:
                sources += [relay for _, relay in roles.z2_arms[m.gauge]]
            expected.append(tuple(sorted(before[q] for q in sources if q in before)))
            last[qubit] = m.bit
    assert schedule.carries == tuple(expected)


@pytest.mark.parametrize("setting", MEMORY, ids=str)
def test_noiseless_detectors_and_logical_are_zero(setting):
    noiseless_check(stim_circuit(setting))
    noiseless_check(stim_circuit(setting, flags=True))


def twirled(circuit: stim.Circuit, seed: int) -> stim.Circuit:
    """Each CX between a random Pauli and the same Pauli pushed through the CX."""
    rng = random.Random(seed)
    out = stim.Circuit()
    for item in flat(circuit):
        if item.name != "CX":
            out.append(item)
            continue
        targets = [t.value for t in item.targets_copy()]
        for pair in zip(targets[::2], targets[1::2], strict=True):
            before = stim.PauliString(rng.choice("IXYZ") + rng.choice("IXYZ"))
            for pauli in (before, None, before.after(stim.Circuit("CX 0 1"))):
                if pauli is None:
                    out.append("CX", pair)
                    continue
                for k, qubit in enumerate(pair):
                    if pauli[k]:
                        out.append("_XYZ"[pauli[k]], [qubit])
    return out


@pytest.mark.parametrize("setting", MEMORY, ids=str)
def test_twirled_detectors_and_logical_are_zero(setting):
    for seed in range(3):
        noiseless_check(twirled(stim_circuit(setting, flags=True), seed))


@pytest.mark.parametrize("patch, basis", CASES, ids=IDS)
def test_aer_shots_give_quiet_detectors(patch, basis):
    # Run 1's own shot reader, on Aer's samples: independent of stim and of add_detectors.
    circuit, schedule = memory_circuit(patch, rounds=2, basis=basis)
    flags = [m.bit for m in schedule.measurements if m.kind in ("z_flag", "relay")]
    records = run_flagged_circuit(circuit, schedule, shots=64, seed=5)
    for record in records:
        assert not any(record["detectors"]) and record["logical_bit"] == 0
        assert not any(schedule.values(record["gauge_bits"])[b] for b in flags)
    # The readouts themselves are not quiet: the flags repeat the random X gauges.
    assert any(record["gauge_bits"][b] for record in records for b in flags)


def single_qubit_paulis(qubit: int) -> tuple[Pauli, ...]:
    x, z = Pauli.x_on((qubit,)), Pauli.z_on((qubit,))
    return (x, z, x * z)


@pytest.mark.parametrize("basis", "XZ")
@pytest.mark.parametrize("qubit", D3.data_qubits)
def test_a_data_error_fires_once(basis, qubit):
    # Injected after round 0, it shows in round 1's detectors and in none later.
    last_half = "z" if basis == "X" else "x"
    for error in single_qubit_paulis(qubit):
        circuit, schedule = memory_circuit(
            D3, rounds=3, basis=basis, error=error, inject_at=f"after_{last_half}0"
        )
        size = D3.code.syndrome_size
        for record in run_flagged_circuit(circuit, schedule, shots=8, seed=3):
            detectors = record["detectors"]
            assert not any(detectors[:size]), error
            assert detectors[size : 2 * size] == D3.code.syndrome(error), error
            assert not any(detectors[2 * size :]), error


@pytest.mark.parametrize("patch, basis", CASES, ids=IDS)
def test_stim_detectors_match_run_1s_shot_reader(patch, basis):
    setting = Setting("memory", basis, patch.distance, 3)
    circuit, schedule = logical_circuit(setting)
    out, records = to_stim(circuit)
    out = with_noise(out, 0.02)
    records = patch_records(records)
    add_detectors(out, schedule, records, flags=True)
    measured = out.compile_sampler(seed=2).sample(200)
    fired, flips = out.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=measured, separate_observables=True
    )
    used = Memory(patch, basis, 3).basis_indices
    flags = [m.bit for m in schedule.measurements if m.kind in ("z_flag", "relay") and m.round >= 0]
    assert fired.any() and flips.any()
    for shot, row, flip in zip(measured, fired, flips[:, 0], strict=True):
        gauge_bits = tuple(int(shot[records["m", k]]) for k in range(len(schedule.measurements)))
        data_bits = tuple(int(shot[records["d", q]]) for q in range(patch.code.n))
        expected = read_shot(schedule, gauge_bits, data_bits)
        values = schedule.values(gauge_bits)
        assert row.astype(int).tolist() == [
            *(expected.detectors[i] for i in used),
            *(values[b] for b in flags),
        ]
        assert flip == expected.logical_bit


def detectors_reading(setting: Setting, kind: str, gauge: str, round: int) -> list:
    """The labels of the detectors that one readout goes into, as a misread flips them."""
    circuit, schedule = logical_circuit(setting)
    out, records = to_stim(circuit)
    labels: list = []
    add_detectors(out, schedule, patch_records(records), labels=labels)
    bit = next(  # a relay: the first of its gauge's two
        m.bit for m in schedule.measurements if (m.kind, m.gauge, m.round) == (kind, gauge, round)
    )
    k = patch_records(records)["m", bit] - out.num_measurements
    detectors = [item for item in flat(out) if item.name == "DETECTOR"]
    return sorted(
        label
        for label, item in zip(labels, detectors, strict=True)
        if any(t.value == k for t in item.targets_copy())
    )


def stabilizer_of(gauge: str) -> str:
    (name,) = (s for s, gauges in D3.stabilizer_gauge_factors.items() if gauge in gauges)
    return name


def test_a_misread_fires_its_round_and_the_one_after_next():
    z, x = Setting("memory", "Z", 3, 4), Setting("memory", "X", 3, 4)
    stab_z = stabilizer_of("Z2Z3Z5Z6")
    assert detectors_reading(z, "z_syn", "Z2Z3Z5Z6", 1) == [(1, stab_z), (3, stab_z)]
    # A relay's readout is its own and its next detector's, and enters the Z2 gauge after.
    stab_z = stabilizer_of("Z1Z2")
    assert detectors_reading(z, "relay", "Z1Z2", 1) == sorted(
        [(1, "relay Z1Z2"), (2, "relay Z1Z2"), (2, stab_z), (3, stab_z)]
    )
    stab_x = stabilizer_of("X1X4")
    assert stab_x == stabilizer_of("X2X5")
    assert detectors_reading(x, "x_gauge", "X1X4", 1) == [(1, stab_x), (3, stab_x)]
    # An X ancilla that flags a Z4 gauge reads as a flag next, so its misread fires the
    # next round; the flag's readout is carried into the round after.
    assert detectors_reading(x, "x_gauge", "X2X5", 1) == [(1, stab_x), (2, stab_x)]
    assert detectors_reading(x, "z_flag", "X2X5", 1) == [(2, stab_x), (3, stab_x)]


@pytest.mark.parametrize("patch, basis", CASES, ids=IDS)
def test_circuit_distance(patch, basis):
    out = with_noise(stim_circuit(Setting("memory", basis, patch.distance, 2)), 0.001)
    out.detector_error_model(decompose_errors=True)
    assert len(out.shortest_graphlike_error()) == patch.distance


@pytest.mark.parametrize("readout", "XY")
def test_idle_test_waits_out_every_cx(readout):
    memory, _ = logical_circuit(Setting("memory", "X", 5, 2))
    idle, _ = logical_circuit(Setting("idle", "X", 5, 2, readout))
    names = [name for name, _, _ in gates(idle)]
    assert "cx" not in names and "reset" not in names
    assert names.count("wait") == [name for name, _, _ in gates(memory)].count("cx")
    assert names.count("rz") == (25 if readout == "Y" else 0)
    out, records = to_stim(idle)
    data = [records["d5", q] for q in range(25)]
    bits = out.compile_sampler(seed=1).sample(64)[:, data]
    # Without noise the data stay in |+>: X reads 0, Y is random.
    assert bits.any() == (readout == "Y")


def test_on_chip_qubits_and_registers():
    setting = Setting("memory", "Z", 3, 2)
    chip = [100 + 2 * i for i in range(23)]
    here, _ = logical_circuit(setting)
    there, _ = logical_circuit(setting, chip, 156)
    assert there.num_qubits == 156
    assert [r.name for r in there.cregs] == ["m3", "d3"]
    assert [(n, tuple(chip[q] for q in qs), c) for n, qs, c in gates(here)] == gates(there)


def test_values_need_every_bit():
    _, schedule = memory_circuit(D3, rounds=1, basis="Z")
    with pytest.raises(ValueError, match="binary bits"):
        schedule.values((0,) * (len(schedule.measurements) - 1))

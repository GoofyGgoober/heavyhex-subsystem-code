"""Flagged d=3 circuits: structure, deflagging, and single-fault tolerance."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from heavyhex.circuits.flagged import (
    FALCON,
    FLAG_OF_X,
    NUM_QUBITS,
    Z2_ARMS,
    memory_circuit_flagged,
    propagate_fault,
    single_faults,
    with_fault,
)
from heavyhex.core import Pauli
from heavyhex.patches.operators import D3, build_operators

pytest.importorskip("qiskit_aer")

from heavyhex.simulation.aer import (  # noqa: E402
    run_flagged_circuit,
    run_memory_flagged,
)

CODE = D3.code
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
FLAG_QUBITS = frozenset(FLAG_OF_X.values())
RELAY_QUBITS = frozenset(relay for arms in Z2_ARMS.values() for _, relay in arms)


def single_qubit_paulis(qubit: int) -> tuple[Pauli, ...]:
    x, z = Pauli.x_on((qubit,)), Pauli.z_on((qubit,))
    return (x, z, x * z)


WEIGHT_ONE_OR_LESS = (Pauli(),) + tuple(
    pauli for qubit in CODE.data_qubits for pauli in single_qubit_paulis(qubit)
)


def in_gauge_times_weight_one(pauli: Pauli) -> bool:
    return any(CODE.in_gauge_group(pauli * other) for other in WEIGHT_ONE_OR_LESS)


def test_circuit_uses_23_qubits_and_18_measurements_per_round():
    circuit, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    assert circuit.num_qubits == NUM_QUBITS == 23
    # Prep X half (6) + Z half (12) + X half (6).
    assert len(schedule.measurements) == 24
    kinds = [m.kind for m in schedule.measurements if m.round == 0]
    assert kinds.count("z_syn") == 4
    assert kinds.count("z_flag") == 4
    assert kinds.count("relay") == 4
    assert kinds.count("x_gauge") == 6
    assert len(schedule.fragments) == 6 + 10


def test_falcon_roles_cover_the_documented_23_qubit_patch():
    assert len(set(FALCON.values())) == NUM_QUBITS
    device_map = json.loads((FIGURES / "fez_map.json").read_text())
    assert max(FALCON.values()) < len(device_map["coords"])


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_flagged_memory_survives(basis):
    records = run_memory_flagged(rounds=1, basis=basis, shots=200, seed=7)
    assert all(record["success"] for record in records)


def test_relays_return_to_zero_in_the_absence_of_faults():
    _, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    relay_bits = [m.bit for m in schedule.measurements if m.kind == "relay"]
    assert len(relay_bits) == 4
    records = run_memory_flagged(rounds=1, basis="Z", shots=100, seed=9)
    for record in records:
        assert all(record["gauge_bits"][bit] == 0 for bit in relay_bits)


@pytest.mark.parametrize("qubit", D3.code.data_qubits)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_every_single_qubit_data_error_is_corrected_flagged(basis, qubit):
    for error in single_qubit_paulis(qubit):
        records = run_memory_flagged(basis=basis, error=error, shots=16, seed=11)
        assert all(record["syndrome"] == CODE.syndrome(error) for record in records)
        assert all(record["success"] for record in records), (basis, error)


def test_lone_flags_are_deflagged_to_success():
    _, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    flag_bits = {(m.round, m.gauge): m.bit for m in schedule.measurements if m.kind == "z_flag"}
    records = run_memory_flagged(rounds=1, basis="Z", shots=400, seed=13)
    lone = [
        record
        for record in records
        if (record["gauge_bits"][flag_bits[0, "X2X5"]] == 1)
        != (record["gauge_bits"][flag_bits[0, "X3X6"]] == 1)
    ]
    assert len(lone) > 50  # lone-flag branches are common, not corner cases
    assert all(record["success"] for record in lone)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_single_faults_never_grow_or_go_silent(basis):
    """Every gadget-local single fault stays correctable, by propagation + Aer.

    Unflagged faults leave at most weight 1 modulo the gauge group and, when
    fully visible, decode to success; flagged faults leave at most weight 2 and
    never differ from a same-signature sibling by a bare logical.
    """
    circuit, schedule = memory_circuit_flagged(rounds=1, basis=basis)
    faults = [
        (fragment, index, fault)
        for fragment in schedule.fragments
        for index, fault in single_faults(circuit, fragment)
    ]
    assert len(faults) > 500
    flagged_groups: dict[tuple, list[Pauli]] = {}
    visible = nontrivial = 0
    for fragment, index, fault in faults:
        context = (basis, fragment.name, index, fault)
        outgoing, flipped = propagate_fault(circuit, fragment, index, fault)
        flag_flips = flipped & FLAG_QUBITS if fragment.half == "Z" else frozenset()
        if flag_flips:
            assert outgoing.weight() <= 2, context
            key = (fragment.name, flag_flips, flipped - FLAG_QUBITS - RELAY_QUBITS)
            flagged_groups.setdefault(key, []).append(outgoing)
        else:
            assert in_gauge_times_weight_one(outgoing), context

        faulty = with_fault(circuit, index, fault)
        records = run_flagged_circuit(faulty, schedule, shots=8, seed=17)
        syndromes = {record["syndrome"] for record in records}
        assert len(syndromes) == 1, context  # detectors are deterministic given a fault
        syndrome = syndromes.pop()
        # Other faults are measurement-like or land after the last check: they
        # need multi-round matching, so they are not graded here.
        if not flag_flips and syndrome == CODE.syndrome(outgoing):
            visible += 1
            nontrivial += any(syndrome)
            assert all(record["success"] for record in records), context
    assert visible > 100
    assert nontrivial > 50
    assert len(flagged_groups) > 10
    for key, errors in flagged_groups.items():
        for first, second in combinations(errors, 2):
            assert not CODE.is_harmful_undetectable(first * second), (basis, key)


def test_flagged_memory_rejects_bad_options():
    with pytest.raises(ValueError, match="positive integer"):
        run_memory_flagged(shots=0)
    with pytest.raises(ValueError, match="d=3 and d=5 only"):
        memory_circuit_flagged(build_operators(7))

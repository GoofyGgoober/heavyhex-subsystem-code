"""Flagged d=3 circuit: its shape, the flags, and single faults."""

from itertools import combinations

import pytest

from heavyhex.circuits.flagged import (
    memory_circuit_flagged,
    propagate_fault,
    qubit_roles,
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
ROLES = qubit_roles(D3)
FLAG_QUBITS = frozenset(ROLES.x_ancillas.values())
RELAY_QUBITS = frozenset(relay for arms in ROLES.z2_arms.values() for _, relay in arms)


def single_qubit_paulis(qubit: int) -> tuple[Pauli, ...]:
    x, z = Pauli.x_on((qubit,)), Pauli.z_on((qubit,))
    return (x, z, x * z)


WEIGHT_ONE_OR_LESS = (Pauli(),) + tuple(
    pauli for qubit in CODE.data_qubits for pauli in single_qubit_paulis(qubit)
)


def in_gauge_times_weight_one(pauli: Pauli) -> bool:
    return any(CODE.in_gauge_group(pauli * other) for other in WEIGHT_ONE_OR_LESS)


def test_d3_roles_are_unchanged():
    assert ROLES.x_ancillas == {
        "X1X4": 9, "X2X5": 10, "X3X6": 11, "X4X7": 12, "X5X8": 13, "X6X9": 14
    }  # fmt: skip
    assert ROLES.z_ancillas == {"Z1Z2": 15, "Z2Z3Z5Z6": 16, "Z4Z5Z7Z8": 17, "Z8Z9": 18}
    assert ROLES.z2_arms == {"Z1Z2": ((0, 19), (1, 20)), "Z8Z9": ((7, 21), (8, 22))}
    assert ROLES.z4_arms == {
        "Z2Z3Z5Z6": (((1, 4), "X2X5"), ((2, 5), "X3X6")),
        "Z4Z5Z7Z8": (((3, 6), "X4X7"), ((4, 7), "X5X8")),
    }


def test_circuit_uses_23_qubits_and_18_measurements_per_round():
    circuit, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    assert circuit.num_qubits == ROLES.num_qubits == 23
    # Prep X half (6) + Z half (12) + X half (6).
    assert len(schedule.measurements) == 24
    kinds = [m.kind for m in schedule.measurements if m.round == 0]
    assert kinds.count("z_syn") == 4
    assert kinds.count("z_flag") == 4
    assert kinds.count("relay") == 4
    assert kinds.count("x_gauge") == 6
    assert len(schedule.gadgets) == 6 + 10


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


def test_shots_with_one_flag_fired_still_succeed():
    _, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    flag_bits = {(m.round, m.gauge): m.bit for m in schedule.measurements if m.kind == "z_flag"}
    records = run_memory_flagged(rounds=1, basis="Z", shots=400, seed=13)
    lone = [
        record
        for record in records
        if (record["gauge_bits"][flag_bits[0, "X2X5"]] == 1)
        != (record["gauge_bits"][flag_bits[0, "X3X6"]] == 1)
    ]
    assert len(lone) > 50  # flags fire at random, so this is common
    assert all(record["success"] for record in lone)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_single_faults_never_grow_or_go_silent(basis):
    """Push every single fault through its gadget, then run it on Aer.

    Without a flag, a fault leaves at most weight 1 up to a gauge, and decodes
    fine when the syndrome shows it. With a flag, it leaves at most weight 2, and
    faults with the same flags and syndrome never differ by a logical.
    """
    circuit, schedule = memory_circuit_flagged(rounds=1, basis=basis)
    faults = [
        (gadget, index, fault)
        for gadget in schedule.gadgets
        for index, fault in single_faults(circuit, gadget)
    ]
    assert len(faults) > 500
    flagged_groups: dict[tuple, list[Pauli]] = {}
    visible = nontrivial = 0
    for gadget, index, fault in faults:
        context = (basis, gadget, index, fault)
        outgoing, flipped = propagate_fault(circuit, gadget, index, fault)
        flag_flips = flipped & FLAG_QUBITS if gadget.half == "Z" else frozenset()
        if flag_flips:
            assert outgoing.weight() <= 2, context
            key = (gadget, flag_flips, flipped - FLAG_QUBITS - RELAY_QUBITS)
            flagged_groups.setdefault(key, []).append(outgoing)
        else:
            assert in_gauge_times_weight_one(outgoing), context

        faulty = with_fault(circuit, index, fault)
        records = run_flagged_circuit(faulty, schedule, shots=8, seed=17)
        syndromes = {record["syndrome"] for record in records}
        assert len(syndromes) == 1, context  # a fault gives the same syndrome every shot
        syndrome = syndromes.pop()
        # The rest look like measurement errors or come after the last check.
        # Those need several rounds to decode, so skip them here.
        if not flag_flips and syndrome == CODE.syndrome(outgoing):
            visible += 1
            nontrivial += any(syndrome)
            assert all(record["success"] for record in records), context
    assert visible > 100
    assert nontrivial > 50
    assert len(flagged_groups) > 10
    for key, errors in flagged_groups.items():
        for first, second in combinations(errors, 2):
            assert not CODE.is_logical(first * second), (basis, key)


def test_flagged_memory_rejects_bad_options():
    with pytest.raises(ValueError, match="positive integer"):
        run_memory_flagged(shots=0)
    with pytest.raises(ValueError, match="d=3 and d=5 only"):
        memory_circuit_flagged(build_operators(7))

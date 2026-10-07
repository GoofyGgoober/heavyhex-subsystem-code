"""Flagged d=3 circuit: its shape, the flags, and single faults."""

import pytest

from heavyhex.circuits.flagged import (
    memory_circuit_flagged,
    propagate_fault,
    qubit_roles,
    single_faults,
)
from heavyhex.core import Pauli
from heavyhex.patches.operators import D3, D5, build_operators

pytest.importorskip("qiskit_aer")

from heavyhex.simulation.aer import run_memory_flagged  # noqa: E402

CODE = D3.code
ROLES = qubit_roles(D3)


def single_qubit_paulis(qubit: int) -> tuple[Pauli, ...]:
    x, z = Pauli.x_on((qubit,)), Pauli.z_on((qubit,))
    return (x, z, x * z)


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
    assert all(not any(record["detectors"]) and record["logical_bit"] == 0 for record in records)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_flags_and_relays_read_zero_without_faults(basis):
    _, schedule = memory_circuit_flagged(rounds=2, basis=basis)
    bits = [m.bit for m in schedule.measurements if m.kind in ("z_flag", "relay")]
    assert len(bits) == 8 * 3 if basis == "X" else 8 * 2
    for record in run_memory_flagged(rounds=2, basis=basis, shots=100, seed=9):
        assert all(record["gauge_bits"][bit] == 0 for bit in bits)


@pytest.mark.parametrize("qubit", D3.code.data_qubits)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_every_single_qubit_data_error_shows_its_syndrome(basis, qubit):
    for error in single_qubit_paulis(qubit):
        records = run_memory_flagged(basis=basis, error=error, shots=16, seed=11)
        assert all(record["syndrome"] == CODE.syndrome(error) for record in records), error


def allowed_leftovers(patch) -> list[Pauli]:
    """What one fault may leave on the data: X, Y or Z on one qubit, or Z on two
    neighbours in the same column (at right angles to the logical Z, so harmless)."""
    d = patch.distance
    single = [p for q in patch.data_qubits for p in single_qubit_paulis(q)]
    column_pairs = [Pauli.z_on((q, q + 1)) for q in patch.data_qubits if q % d != d - 1]
    return [Pauli(), *single, *column_pairs]


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_single_faults_leave_one_qubit_or_a_column_pair(patch, basis):
    circuit, schedule = memory_circuit_flagged(patch, basis=basis)
    allowed = allowed_leftovers(patch)
    count = 0
    for gadget in schedule.gadgets:
        for index, fault in single_faults(circuit, gadget):
            count += 1
            outgoing, _ = propagate_fault(circuit, gadget, index, fault)
            assert any(patch.code.in_gauge_group(outgoing * p) for p in allowed), (
                basis,
                gadget,
                index,
                fault,
            )
    assert count > 30 * len(schedule.gadgets)  # every gadget has dozens of fault spots


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_d5_rounds_are_no_longer_than_d3_rounds(basis):
    # Gadgets take turns instead of queueing on shared data qubits, so a half-round
    # doesn't get longer as the patch grows.
    depths = [memory_circuit_flagged(p, rounds=3, basis=basis)[0].depth() for p in (D3, D5)]
    assert depths[0] == depths[1]


def test_flagged_memory_rejects_bad_options():
    with pytest.raises(ValueError, match="positive integer"):
        run_memory_flagged(shots=0)
    with pytest.raises(ValueError, match="d=3 and d=5 only"):
        memory_circuit_flagged(build_operators(7))

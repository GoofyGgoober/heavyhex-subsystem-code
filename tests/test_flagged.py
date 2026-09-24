"""Flagged d=3 circuit: its shape, the flags, and single faults."""

import pytest

from heavyhex.circuits.flagged import (
    memory_circuit_flagged,
    propagate_fault,
    qubit_roles,
    single_faults,
    with_fault,
)
from heavyhex.core import Pauli
from heavyhex.patches.operators import D3, D5, build_operators

pytest.importorskip("qiskit_aer")

from heavyhex.simulation.aer import (  # noqa: E402
    run_flagged_circuit,
    run_memory_flagged,
)

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
    assert all(record["success"] for record in records)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_flags_and_relays_read_zero_without_faults(basis):
    _, schedule = memory_circuit_flagged(rounds=2, basis=basis)
    bits = [m.bit for m in schedule.measurements if m.kind in ("z_flag", "relay")]
    assert len(bits) == 8 * 3 if basis == "X" else 8 * 2
    for record in run_memory_flagged(rounds=2, basis=basis, shots=100, seed=9):
        assert all(record["gauge_bits"][bit] == 0 for bit in bits)


@pytest.mark.parametrize("qubit", D3.code.data_qubits)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_every_single_qubit_data_error_is_corrected_flagged(basis, qubit):
    for error in single_qubit_paulis(qubit):
        records = run_memory_flagged(basis=basis, error=error, shots=16, seed=11)
        assert all(record["syndrome"] == CODE.syndrome(error) for record in records)
        assert all(record["success"] for record in records), (basis, error)


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
def test_single_faults_the_syndrome_shows_are_corrected(basis):
    """Run every single gadget fault on Aer and decode round 0 with the lookup table."""
    circuit, schedule = memory_circuit_flagged(rounds=1, basis=basis)
    visible = nontrivial = 0
    for gadget in schedule.gadgets:
        for index, fault in single_faults(circuit, gadget):
            context = (basis, gadget, index, fault)
            outgoing, _ = propagate_fault(circuit, gadget, index, fault)
            records = run_flagged_circuit(
                with_fault(circuit, index, fault), schedule, shots=8, seed=17
            )
            syndromes = {record["syndrome"] for record in records}
            assert len(syndromes) == 1, context  # a fault gives the same syndrome every shot
            syndrome = syndromes.pop()
            # The rest look like measurement errors or come after the last check.
            # Those need several rounds to decode, so skip them here.
            if syndrome == CODE.syndrome(outgoing):
                visible += 1
                nontrivial += any(syndrome)
                assert all(record["success"] for record in records), context
    assert visible > 100
    assert nontrivial > 50


def test_flagged_memory_rejects_bad_options():
    with pytest.raises(ValueError, match="positive integer"):
        run_memory_flagged(shots=0)
    with pytest.raises(ValueError, match="d=3 and d=5 only"):
        memory_circuit_flagged(build_operators(7))

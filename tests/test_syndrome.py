from surface_code import (
    ANCILLAS,
    DATA_QUBITS,
    LOGICAL_X,
    LOGICAL_Z,
    STABILIZERS,
    Pauli,
    abstract_syndrome,
    extract_syndrome,
)
from surface_code.syndrome import CX, H, MeasureZ, SYNDROME_CIRCUIT


def test_one_ancilla_per_stabilizer():
    assert len(ANCILLAS) == len(STABILIZERS) == 8
    assert set(ANCILLAS).isdisjoint(DATA_QUBITS)


def test_no_error_is_trivial_syndrome():
    zeros = (0,) * 8
    assert extract_syndrome(Pauli()) == zeros
    assert abstract_syndrome(Pauli()) == zeros


def test_logicals_are_invisible_to_the_checks():
    assert extract_syndrome(LOGICAL_X) == (0,) * 8
    assert extract_syndrome(LOGICAL_Z) == (0,) * 8


def test_center_x_flips_the_two_z_plaquettes():
    # X on 4 anticommutes with the two weight-4 Z tiles; X tiles do not see it.
    assert extract_syndrome(Pauli.x_on((4,))) == (0, 0, 0, 0, 1, 1, 0, 0)


def test_circuit_matches_commutation_for_every_single_qubit_error():
    for qubit in DATA_QUBITS:
        for error in (
            Pauli.x_on((qubit,)),
            Pauli.z_on((qubit,)),
            Pauli.x_on((qubit,)) * Pauli.z_on((qubit,)),
        ):
            assert extract_syndrome(error) == abstract_syndrome(error)


def test_circuit_is_cnots_hadamards_and_eight_z_measures():
    assert all(isinstance(op, (H, CX, MeasureZ)) for op in SYNDROME_CIRCUIT)
    measures = [op for op in SYNDROME_CIRCUIT if isinstance(op, MeasureZ)]
    assert tuple(op.qubit for op in measures) == ANCILLAS

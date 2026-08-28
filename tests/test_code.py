import pytest

from surface_code import PauliError, RotatedSurfaceCode


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_code_dimensions_and_commutation(distance):
    code = RotatedSurfaceCode(distance)
    assert code.n_data_qubits == distance**2
    assert len(code.stabilizers) == distance**2 - 1
    assert len(code.x_stabilizers) == len(code.z_stabilizers)
    assert all(len(x.qubits & z.qubits) % 2 == 0 for x in code.x_stabilizers for z in code.z_stabilizers)


def test_single_qubit_x_error_flips_neighboring_z_checks():
    code = RotatedSurfaceCode(3)
    syndrome = code.syndrome(PauliError.from_ops([(4, "X")]))
    assert sum(syndrome) == 2


def test_y_error_has_both_components():
    error = PauliError.from_ops([(0, "Y")])
    assert error.x == error.z == frozenset((0,))


def test_repeated_pauli_cancels():
    assert PauliError.from_ops([(2, "X"), (2, "X")]) == PauliError()


def test_invalid_distance_is_rejected():
    with pytest.raises(ValueError):
        RotatedSurfaceCode(4)

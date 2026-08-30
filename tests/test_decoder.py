from surface_code import DATA_QUBITS, LOGICAL_X, LOGICAL_Z, Pauli, decode, extract_syndrome, z_basis_success
from surface_code.patches.parameters import in_stabilizer_group


def _single_qubit_errors() -> list[Pauli]:
    errors = []
    for qubit in DATA_QUBITS:
        x = Pauli.x_on((qubit,))
        z = Pauli.z_on((qubit,))
        errors.extend((x, z, x * z))
    return errors


def test_trivial_syndrome_is_identity():
    assert decode((0, 0, 0, 0, 0, 0, 0, 0)) == Pauli()


def test_weight1_errors_are_corrected_up_to_a_stabilizer():
    for error in _single_qubit_errors():
        correction = decode(extract_syndrome(error))
        assert correction is not None
        assert correction.weight() <= 1
        assert in_stabilizer_group(correction * error)


def test_unknown_syndrome_is_none():
    known = {extract_syndrome(error) for error in _single_qubit_errors()}
    known.add((0, 0, 0, 0, 0, 0, 0, 0))
    missing = next(
        bits
        for n in range(256)
        if (bits := tuple((n >> i) & 1 for i in range(8))) not in known
    )
    assert decode(missing) is None


def test_z_basis_success():
    assert z_basis_success(Pauli()) == 1
    assert all(z_basis_success(Pauli.x_on((q,))) == 1 for q in DATA_QUBITS)
    assert extract_syndrome(LOGICAL_X) == (0,) * 8
    assert z_basis_success(LOGICAL_X) == 0
    assert z_basis_success(LOGICAL_Z) == 1

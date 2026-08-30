from surface_code import DATA_QUBITS, Pauli, decode, extract_syndrome
from surface_code.decoder import LOOKUP
from surface_code.parameters import in_stabilizer_group


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
        assert in_stabilizer_group(correction * error)


def test_table_only_has_identity_and_weight1_rows():
    assert LOOKUP[(0, 0, 0, 0, 0, 0, 0, 0)] == Pauli()
    assert all(pauli.weight() <= 1 for pauli in LOOKUP.values())
    assert len(LOOKUP) < 256


def test_unknown_syndrome_is_none():
    missing = next(
        bits
        for n in range(256)
        if (bits := tuple((n >> i) & 1 for i in range(8))) not in LOOKUP
    )
    assert decode(missing) is None

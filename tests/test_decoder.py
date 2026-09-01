from surface_code import DATA_QUBITS, LOGICAL_X, LOGICAL_Z, Pauli, decode, extract_syndrome, z_basis_success
from surface_code.patches import PATCH
from surface_code.patches.parameters import in_stabilizer_group


def _single_qubit_errors() -> list[Pauli]:
    errors = []
    for qubit in DATA_QUBITS:
        x = Pauli.x_on((qubit,))
        z = Pauli.z_on((qubit,))
        errors.extend((x, z, x * z))
    return errors


def _all_syndromes() -> list[tuple[int, ...]]:
    width = PATCH.code.syndrome_size
    return [tuple((n >> i) & 1 for i in range(width)) for n in range(1 << width)]


def test_trivial_syndrome_is_identity():
    assert decode((0, 0, 0, 0, 0, 0, 0, 0)) == Pauli()


def test_weight1_errors_are_corrected_up_to_a_stabilizer():
    for error in _single_qubit_errors():
        correction = decode(extract_syndrome(error))
        assert correction.weight() <= 1
        assert in_stabilizer_group(correction * error)


def test_every_syndrome_has_a_min_weight_correction():
    lightest: dict[tuple[int, ...], int] = {}
    for error in _single_qubit_errors():
        lightest.setdefault(extract_syndrome(error), 1)
    lightest.setdefault((0,) * PATCH.code.syndrome_size, 0)

    for syndrome in _all_syndromes():
        correction = decode(syndrome)
        assert extract_syndrome(correction) == syndrome
        if syndrome in lightest:
            assert correction.weight() == lightest[syndrome]
        else:
            assert correction.weight() >= 2


def test_two_errors_on_logical_x_decode_the_other_end():
    error = Pauli.x_on((0, 3))
    assert decode(extract_syndrome(error)) == Pauli.x_on((6,))
    assert z_basis_success(error) == 0


def test_z_basis_success():
    assert z_basis_success(Pauli()) == 1
    assert all(z_basis_success(Pauli.x_on((q,))) == 1 for q in DATA_QUBITS)
    assert extract_syndrome(LOGICAL_X) == (0,) * 8
    assert z_basis_success(LOGICAL_X) == 0
    assert z_basis_success(LOGICAL_Z) == 1

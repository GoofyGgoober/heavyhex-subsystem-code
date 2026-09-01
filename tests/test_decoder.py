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


def _syndrome_xor(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a ^ b for a, b in zip(left, right))


def _minimum_weights_by_syndrome() -> dict[tuple[int, ...], int]:
    """Independent shortest-path oracle using weight-1 syndrome transitions."""
    zero = (0,) * PATCH.code.syndrome_size
    steps = {extract_syndrome(error) for error in _single_qubit_errors()}
    weights = {zero: 0}
    frontier = {zero}
    while frontier:
        next_frontier: set[tuple[int, ...]] = set()
        for syndrome in frontier:
            for step in steps:
                candidate = _syndrome_xor(syndrome, step)
                if candidate not in weights:
                    weights[candidate] = weights[syndrome] + 1
                    next_frontier.add(candidate)
        frontier = next_frontier
    return weights


def test_trivial_syndrome_is_identity():
    assert decode((0, 0, 0, 0, 0, 0, 0, 0)) == Pauli()


def test_weight1_errors_are_corrected_up_to_a_stabilizer():
    for error in _single_qubit_errors():
        correction = decode(extract_syndrome(error))
        assert correction.weight() <= 1
        assert in_stabilizer_group(correction * error)


def test_every_syndrome_has_a_min_weight_correction():
    minimum_weights = _minimum_weights_by_syndrome()
    assert set(minimum_weights) == set(_all_syndromes())

    for syndrome in _all_syndromes():
        correction = decode(syndrome)
        assert extract_syndrome(correction) == syndrome
        assert correction.weight() == minimum_weights[syndrome]


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

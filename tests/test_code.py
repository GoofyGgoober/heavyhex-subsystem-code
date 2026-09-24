"""The heavy-hex code itself: stabilizers, ranks, distance."""

import pytest

from heavyhex.core import Pauli
from heavyhex.core.subsystem import SubsystemCode
from heavyhex.patches.operators import D3, D5, build_operators

PATCHES = [pytest.param(D3, id="d3"), pytest.param(D5, id="d5")]

# d=3 gauge qubits A = (X1X4, Z1Z2) and B = (X5X8, Z8Z9), 0-based ids.
GAUGE_QUBITS_D3 = (
    (Pauli.x_on((0, 3)), Pauli.z_on((0, 1))),
    (Pauli.x_on((4, 7)), Pauli.z_on((7, 8))),
)


def test_d3_operator_names_match_the_paper_convention():
    assert set(D3.x_gauges) == {"X1X4", "X2X5", "X3X6", "X4X7", "X5X8", "X6X9"}
    assert set(D3.z_gauges) == {"Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9"}
    assert set(D3.x_stabilizers) == {"X1X2X4X5", "X4X7", "X3X6", "X5X6X8X9"}
    assert set(D3.z_stabilizers) == {"Z1Z2Z4Z5Z7Z8", "Z2Z3Z5Z6Z8Z9"}
    assert D3.logical_x == (1, 2, 3)
    assert D3.logical_z == (1, 4, 7)


@pytest.mark.parametrize("patch", PATCHES)
def test_counts_follow_the_chamberland_formulas(patch):
    d = patch.distance
    assert len(patch.x_gauges) == d * (d - 1)
    assert len(patch.z_gauges) == (d * d - 1) // 2
    assert len(patch.z_stabilizers) == d - 1
    assert patch.code.gauge_qubit_count() == (d - 1) ** 2 // 2
    assert patch.code.logical_count() == 1


@pytest.mark.parametrize("patch", PATCHES)
def test_stabilizers_are_independent_products_of_same_basis_gauges(patch):
    # Constructing SubsystemCode already asserts centrality and gauge membership.
    stabilizers = {**patch.x_stabilizers, **patch.z_stabilizers}
    gauges = {**patch.x_gauges, **patch.z_gauges}
    assert patch.code.stabilizer_rank() == len(stabilizers)
    factors = patch.stabilizer_gauge_factors
    assert set(factors) == set(stabilizers)
    for name, support in stabilizers.items():
        product: set[int] = set()
        for gauge in factors[name]:
            product ^= set(gauges[gauge])
        assert product == set(support)


@pytest.mark.parametrize("patch", PATCHES)
def test_stabilizer_names_track_the_code_stabilizer_order(patch):
    # Syndromes pair these names with code.stabilizers.
    supports = {**patch.x_stabilizers, **patch.z_stabilizers}
    assert patch.stabilizer_names == tuple(supports)
    for name, pauli in zip(patch.stabilizer_names, patch.code.stabilizers, strict=True):
        assert pauli.x | pauli.z == {q - 1 for q in supports[name]}


def test_d3_stabilizer_gauge_factors():
    factors = D3.stabilizer_gauge_factors
    assert set(factors["Z1Z2Z4Z5Z7Z8"]) == {"Z1Z2", "Z4Z5Z7Z8"}
    assert set(factors["X1X2X4X5"]) == {"X1X4", "X2X5"}
    assert factors["X4X7"] == ("X4X7",)


def test_d3_gauge_qubits_factor_the_codespace():
    code = D3.code
    logicals = (
        (code.logical_x, code.logical_z),
        *GAUGE_QUBITS_D3,
    )
    for xbar, zbar in logicals:
        assert not xbar.commutes(zbar)
    for i, (x_a, z_a) in enumerate(logicals):
        for x_b, z_b in logicals[i + 1 :]:
            assert x_a.commutes(x_b) and x_a.commutes(z_b)
            assert z_a.commutes(x_b) and z_a.commutes(z_b)
    # Every measured gauge acts as identity on the logical factor.
    for gauge in code.gauge_x + code.gauge_z:
        assert gauge.commutes(code.logical_x) and gauge.commutes(code.logical_z)


@pytest.mark.parametrize("patch", PATCHES)
def test_no_logical_below_weight_three(patch):
    assert patch.code.distance(max_weight=2) is None


def test_d3_distance_is_three():
    assert D3.code.distance() == 3


def test_gauge_group_rejects_non_central_stabilizers():
    code = D3.code
    bad = code.stabilizers[0] * next(iter(GAUGE_QUBITS_D3[0]))
    with pytest.raises(ValueError, match="not central"):
        SubsystemCode(
            data_qubits=code.data_qubits,
            gauge_x=code.gauge_x,
            gauge_z=code.gauge_z,
            stabilizers=(bad,) + code.stabilizers[1:],
            logical_x=code.logical_x,
            logical_z=code.logical_z,
        )


@pytest.mark.parametrize("distance", [True, 2, 4, 5.0])
def test_invalid_distance_is_rejected(distance):
    with pytest.raises(ValueError, match="odd integer"):
        build_operators(distance)

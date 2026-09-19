"""Algebra of the heavy-hex subsystem code: center, ranks, distance."""

import pytest

from surface_code.core.subsystem import SubsystemCode
from surface_code.patches.heavyhex import (
    D3,
    D5,
    GAUGE_QUBITS_D3,
    build_operators,
)


def test_d3_operator_names_match_the_paper_convention():
    assert set(D3.x_gauges) == {"X1X4", "X2X5", "X3X6", "X4X7", "X5X8", "X6X9"}
    assert set(D3.z_gauges) == {"Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9"}
    assert set(D3.x_stabilizers) == {"X1X2X4X5", "X4X7", "X3X6", "X5X6X8X9"}
    assert set(D3.z_stabilizers) == {"Z1Z2Z4Z5Z7Z8", "Z2Z3Z5Z6Z8Z9"}
    assert D3.logical_x == (1, 2, 3)
    assert D3.logical_z == (1, 4, 7)


def test_counts_follow_the_chamberland_formulas():
    for patch in (D3, D5):
        d = patch.distance
        assert len(patch.x_gauges) == d * (d - 1)
        assert len(patch.z_gauges) == (d * d - 1) // 2
        assert len(patch.z_stabilizers) == d - 1
        code = patch.code
        assert code.gauge_qubit_count() == (d - 1) ** 2 // 2
        assert code.logical_count() == 1


def test_stabilizers_are_central_gauge_products():
    # Constructing SubsystemCode already asserts centrality and gauge membership.
    for patch in (D3, D5):
        code = patch.code
        assert code.stabilizer_rank() == len(patch.x_stabilizers) + len(patch.z_stabilizers)
        factors = patch.stabilizer_gauge_factors()
        assert set(factors) == set(patch.x_stabilizers) | set(patch.z_stabilizers)


def test_d3_stabilizer_gauge_factors():
    factors = D3.stabilizer_gauge_factors()
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


def test_d3_distance_is_three_with_weight_two_holes_in_the_gauge_group():
    code = D3.code
    assert code.min_harmful_weight(max_weight=2) is None
    assert code.distance() == 3


def test_d5_has_no_harmful_error_below_weight_three():
    assert D5.code.min_harmful_weight(max_weight=2) is None


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

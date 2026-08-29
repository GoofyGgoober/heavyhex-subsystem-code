from itertools import combinations

from surface_code import (
    LOGICAL_X,
    LOGICAL_Z,
    STABILIZERS,
    X_STABILIZERS,
    Z_STABILIZERS,
    Pauli,
    is_logical,
    parameters,
)


def test_stabilizers_are_the_eight_css_generators():
    assert len(X_STABILIZERS) == len(Z_STABILIZERS) == 4
    assert STABILIZERS == X_STABILIZERS + Z_STABILIZERS
    assert all(s.z == frozenset() for s in X_STABILIZERS)
    assert all(s.x == frozenset() for s in Z_STABILIZERS)


def test_every_pair_of_stabilizers_commutes():
    for a, b in combinations(STABILIZERS, 2):
        assert a.commutes(b)


def test_logicals_commute_with_stabilizers_and_anticommute():
    assert all(LOGICAL_X.commutes(s) for s in STABILIZERS)
    assert all(LOGICAL_Z.commutes(s) for s in STABILIZERS)
    assert not LOGICAL_X.commutes(LOGICAL_Z)
    assert LOGICAL_X.weight() == LOGICAL_Z.weight() == 3


def test_logicals_are_nontrivial():
    assert is_logical(LOGICAL_X)
    assert is_logical(LOGICAL_Z)
    assert is_logical(LOGICAL_X * LOGICAL_Z)
    assert not is_logical(Pauli())
    assert not is_logical(STABILIZERS[0])


def test_parameters_are_nine_one_three():
    assert parameters() == (9, 1, 3)

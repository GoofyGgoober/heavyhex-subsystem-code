"""Lookup decoder: every correction is minimum weight and subsystem-correct."""

import pytest

from heavyhex.core import Pauli
from heavyhex.decoders.lookup import decode, residual_in_gauge_group
from heavyhex.patches.operators import D3, D5

CODE = D3.code


def test_every_single_qubit_error_corrects_into_the_gauge_group():
    assert residual_in_gauge_group(Pauli())
    for qubit in CODE.data_qubits:
        for error in (
            Pauli.x_on((qubit,)),
            Pauli.z_on((qubit,)),
            Pauli.x_on((qubit,)) * Pauli.z_on((qubit,)),
        ):
            assert residual_in_gauge_group(error)


def test_table_entries_are_minimum_weight():
    best: dict[tuple[int, ...], int] = {}
    for weight in range(CODE.n + 1):
        for error in CODE.paulis_of_weight(weight):
            best.setdefault(CODE.syndrome(error), weight)
    assert len(best) == 1 << CODE.syndrome_size
    for syndrome, weight in best.items():
        assert decode(syndrome).weight() == weight


def test_logical_operators_are_invisible_but_fatal():
    assert CODE.syndrome(CODE.logical_x) == (0,) * CODE.syndrome_size
    assert decode((0,) * CODE.syndrome_size) == Pauli()
    assert not residual_in_gauge_group(CODE.logical_x)


@pytest.mark.parametrize("syndrome", [(0,) * 5, (0,) * 7, (2,) * 6])
def test_decode_rejects_malformed_syndromes(syndrome):
    with pytest.raises(ValueError, match="6 bits"):
        decode(syndrome)


def test_lookup_refuses_distances_beyond_reach():
    with pytest.raises(ValueError, match="infeasible"):
        decode((0,) * D5.code.syndrome_size, D5.code)

"""Code-capacity depolarizing sampling."""

import random

import pytest

from heavyhex import Pauli
from heavyhex.patches import D3
from heavyhex.simulation.noise import depolarizing_error


def test_p_zero_is_identity():
    assert depolarizing_error(0.0, D3.data_qubits, random.Random(0)) == Pauli()


def test_p_one_hits_every_qubit():
    error = depolarizing_error(1.0, D3.data_qubits, random.Random(0))
    assert error.weight() == len(D3.data_qubits)


def test_seeded_draw_is_deterministic():
    qubits = D3.data_qubits
    assert depolarizing_error(0.5, qubits, random.Random(1)) == depolarizing_error(
        0.5, qubits, random.Random(1)
    )


def test_support_stays_inside_the_given_qubits():
    error = depolarizing_error(0.5, (0, 1, 2), random.Random(3))
    assert error.x | error.z <= {0, 1, 2}


@pytest.mark.parametrize("p", [-0.1, 1.5, "half", True])
def test_rejects_non_probabilities(p):
    with pytest.raises(ValueError, match="probability"):
        depolarizing_error(p, D3.data_qubits, random.Random(0))

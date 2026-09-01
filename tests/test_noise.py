import random

from surface_code import Pauli
from surface_code.patches import PATCH
from surface_code.simulation.noise import depolarizing_error


def test_p_zero_is_identity():
    rng = random.Random(0)
    assert depolarizing_error(0.0, PATCH.data_qubits, rng) == Pauli()


def test_p_one_hits_every_qubit():
    rng = random.Random(0)
    error = depolarizing_error(1.0, PATCH.data_qubits, rng)
    assert error.weight() == len(PATCH.data_qubits)


def test_seeded_draw_is_deterministic():
    qubits = PATCH.data_qubits
    assert depolarizing_error(0.5, qubits, random.Random(1)) == depolarizing_error(
        0.5, qubits, random.Random(1)
    )

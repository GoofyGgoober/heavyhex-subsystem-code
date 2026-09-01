import pytest

from surface_code import Pauli, extract_syndrome
from surface_code.patches import PATCH

qiskit_aer = pytest.importorskip("qiskit_aer")

from surface_code.simulation import run_aer, shot_z_success, to_qiskit


def test_qiskit_circuit_has_17_qubits_and_both_registers():
    circuit = to_qiskit()
    assert circuit.num_qubits == 17
    names = {creg.name for creg in circuit.cregs}
    assert names == {"syn", "data"}


def test_noiseless_logical_zero_has_trivial_syndrome():
    shots = run_aer(Pauli(), shots=64)
    assert all(syndrome == (0,) * 8 for syndrome, _ in shots)
    assert all(shot_z_success(syndrome, data) == 1 for (syndrome, data) in shots)


def test_aer_syndrome_matches_pauli_frame_for_single_qubit_x():
    for qubit in PATCH.data_qubits:
        error = Pauli.x_on((qubit,))
        shots = run_aer(error, shots=32)
        want = extract_syndrome(error)
        assert all(syndrome == want for syndrome, _ in shots)
        assert all(shot_z_success(syndrome, data) == 1 for syndrome, data in shots)


def test_logical_x_is_a_z_failure_on_aer():
    shots = run_aer(PATCH.logical_x, shots=16)
    (syndrome, data_bits), _ = next(iter(shots.items()))
    assert syndrome == (0,) * 8
    assert shot_z_success(syndrome, data_bits) == 0


def test_seeded_aer_runs_are_reproducible():
    assert run_aer(Pauli(), shots=32, seed=7) == run_aer(Pauli(), shots=32, seed=7)


@pytest.mark.parametrize("shots", [0, -1, 1.5, True])
def test_aer_rejects_invalid_shot_counts(shots):
    with pytest.raises(ValueError, match="positive integer"):
        run_aer(Pauli(), shots=shots)


def test_aer_rejects_seed_above_signed_64_bit_range():
    with pytest.raises(ValueError, match="seed must be an integer"):
        run_aer(Pauli(), shots=1, seed=1 << 63)

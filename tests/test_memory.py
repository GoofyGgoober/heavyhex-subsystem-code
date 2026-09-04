import pytest

from surface_code import Pauli
from surface_code.patches import PATCH

pytest.importorskip("qiskit_aer")

from surface_code.simulation.memory import (
    BASELINE_NOISE,
    CircuitNoise,
    MemoryShot,
    run_memory,
    summarize_memory,
    to_memory_circuit,
)


def test_memory_circuit_reuses_measured_and_reset_ancillas():
    circuit = to_memory_circuit(3)
    assert circuit.num_qubits == 17
    assert [register.name for register in circuit.cregs] == [
        "syn_0",
        "syn_1",
        "syn_2",
        "data",
    ]
    operations = circuit.count_ops()
    assert operations["reset"] == 3 * len(PATCH.ancillas)
    assert operations["measure"] == 3 * len(PATCH.ancillas) + len(PATCH.data_qubits)


def test_noiseless_repeated_rounds_preserve_logical_zero():
    tallies = run_memory(3, shots=32, seed=7, noise=CircuitNoise.ideal())
    assert all(shot.syndromes == ((0,) * 8,) * 3 for shot in tallies)
    summary = summarize_memory(tallies)
    assert summary.syndrome_trigger_rate == (0.0, 0.0, 0.0)
    assert summary.detection_event_rate == (0.0, 0.0, 0.0)
    assert summary.raw_z_success_rate == 1.0
    assert summary.last_round_z_success_rate == 1.0


def test_logical_x_is_visible_in_final_memory_readout():
    summary = summarize_memory(
        run_memory(
            2,
            shots=16,
            seed=3,
            error=PATCH.logical_x,
            noise=CircuitNoise.ideal(),
        )
    )
    assert summary.syndrome_trigger_rate == (0.0, 0.0)
    assert summary.raw_z_success_rate == 0.0
    assert summary.last_round_z_success_rate == 0.0


def test_readout_noise_creates_time_resolved_detection_events():
    tallies = run_memory(
        2,
        shots=8,
        seed=5,
        noise=CircuitNoise(readout=1.0),
    )
    assert all(shot.syndromes == ((1,) * 8, (1,) * 8) for shot in tallies)
    summary = summarize_memory(tallies)
    assert summary.syndrome_trigger_rate == (1.0, 1.0)
    assert summary.detection_event_rate == (1.0, 0.0)
    assert summary.raw_z_success_rate == 0.0


def test_seeded_memory_runs_are_reproducible():
    noise = CircuitNoise(single_qubit=0.01, two_qubit=0.02, readout=0.03, reset=0.01)
    assert run_memory(2, shots=16, seed=11, noise=noise) == run_memory(
        2, shots=16, seed=11, noise=noise
    )


def test_default_memory_noise_is_the_project_baseline():
    assert BASELINE_NOISE == CircuitNoise(
        single_qubit=0.001,
        two_qubit=0.01,
        readout=0.02,
        reset=0.01,
    )
    assert not BASELINE_NOISE.is_ideal


def test_memory_shot_computes_detection_events():
    first = (1, 0, 0, 0, 0, 0, 0, 0)
    second = (1, 1, 0, 0, 0, 0, 0, 0)
    shot = MemoryShot((first, second), (0,) * 9)
    assert shot.detection_events() == (
        first,
        (0, 1, 0, 0, 0, 0, 0, 0),
    )


@pytest.mark.parametrize("rounds", [-1, 1.5, True])
def test_memory_rejects_invalid_round_counts(rounds):
    with pytest.raises(ValueError, match="non-negative integer"):
        to_memory_circuit(rounds)


def test_circuit_noise_validates_probabilities():
    with pytest.raises(ValueError, match="two_qubit"):
        CircuitNoise(two_qubit=1.1)
    with pytest.raises(ValueError, match="readout"):
        CircuitNoise(readout=-0.1)


def test_zero_round_baseline_has_no_decoded_metric():
    summary = summarize_memory(
        run_memory(0, shots=8, seed=2, error=Pauli(), noise=CircuitNoise.ideal())
    )
    assert summary.rounds == 0
    assert summary.raw_z_success_rate == 1.0
    assert summary.last_round_z_success_rate is None

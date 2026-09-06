from math import exp

import pytest

from surface_code import Pauli
from surface_code.decoders.infer_logical import logical_bit_from_data_readout, select_check_results
from surface_code.patches import PATCH

pytest.importorskip("qiskit_aer")

from surface_code.simulation.record_shots import (
    BASELINE_IDLE_NOISE,
    BASELINE_NOISE,
    CircuitNoise,
    IdleNoise,
    MemoryShot,
    idle_duration_per_round,
    run_memory,
    run_timed_memory,
    summarize_memory,
    to_memory_circuit,
    to_timed_memory_circuit,
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


def test_timed_circuit_divides_idle_time_between_rounds():
    circuit = to_timed_memory_circuit(10, 2, round_duration_us=1)
    assert circuit.count_ops()["delay"] == 2 * len(PATCH.data_qubits)
    durations = {
        instruction.operation.duration
        for instruction in circuit.data
        if instruction.operation.name == "delay"
    }
    assert durations == {4.0}


def test_zero_round_timed_circuit_idles_for_the_whole_window():
    circuit = to_timed_memory_circuit(10, 0, round_duration_us=1)
    assert circuit.count_ops()["delay"] == len(PATCH.data_qubits)
    durations = {
        instruction.operation.duration
        for instruction in circuit.data
        if instruction.operation.name == "delay"
    }
    assert durations == {10.0}


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
    assert not BASELINE_IDLE_NOISE.is_ideal


def test_idle_noise_is_a_normalized_continuous_time_channel():
    noise = IdleNoise(x_rate=0.01, y_rate=0.02, z_rate=0.03)
    probabilities = noise.pauli_probabilities(5)
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability >= 0 for probability in probabilities)
    assert noise.pauli_probabilities(0) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_idle_probabilities_preserve_very_rare_errors():
    identity, x, y, z = IdleNoise(x_rate=0.01).pauli_probabilities(1e-18)
    assert x == pytest.approx(1e-20, rel=1e-12, abs=0)
    assert (identity, y, z) == (1.0, 0.0, 0.0)


def test_zero_duration_has_no_noise_even_at_large_finite_rates():
    noise = IdleNoise(x_rate=1e308, y_rate=1e308)
    assert noise.pauli_probabilities(0) == (1, 0, 0, 0)
    assert noise.pauli_probabilities(1) == (0.25, 0.25, 0.25, 0.25)


def test_idle_probabilities_match_the_pauli_walk_solution():
    noise = IdleNoise(x_rate=0.01, y_rate=0.02, z_rate=0.03)
    duration = 5
    a = exp(-2 * (noise.x_rate + noise.y_rate) * duration)
    b = exp(-2 * (noise.z_rate + noise.y_rate) * duration)
    c = exp(-2 * (noise.x_rate + noise.z_rate) * duration)
    assert noise.pauli_probabilities(duration) == pytest.approx(
        ((1 + a + b + c) / 4, (1 - a + b - c) / 4, (1 - a - b + c) / 4, (1 + a - b - c) / 4)
    )


def test_memory_shot_rejects_float_bits_before_xor():
    with pytest.raises(ValueError, match="binary bits"):
        MemoryShot(((0.0,) * 8,), (0,) * 9)


def test_timing_rejects_more_syndrome_time_than_total_time():
    assert idle_duration_per_round(10, 2, 1) == 4
    with pytest.raises(ValueError, match="do not fit"):
        idle_duration_per_round(2, 3, 1)


def test_ideal_timed_memory_preserves_both_logical_bases():
    for basis in ("X", "Z"):
        tallies = run_timed_memory(
            4,
            2,
            round_duration_us=1,
            shots=16,
            seed=5,
            noise=CircuitNoise.ideal(),
            idle_noise=IdleNoise.ideal(),
            basis=basis,
        )
        assert sum(tallies.values()) == 16
        assert all(
            all(syndrome == (0,) * 8 for syndrome in shot.syndromes)
            and logical_bit_from_data_readout(shot.data_bits, basis) == 0
            for shot in tallies
        )


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


def test_product_preparation_skips_the_encoder():
    encoder = to_memory_circuit(1).count_ops()
    product = to_memory_circuit(1, preparation="product").count_ops()
    assert encoder["swap"] == 6 and encoder["cx"] > 24
    assert "swap" not in product and product["cx"] == 24


def test_x_basis_product_preparation_starts_in_plus():
    circuit = to_memory_circuit(0, basis="X", preparation="product")
    # Nine Hadamards to prepare |+> and nine more before the X-basis readout.
    assert circuit.count_ops()["h"] == 2 * len(PATCH.data_qubits)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_product_preparation_fixes_the_measured_checks_and_logical(basis):
    tallies = run_timed_memory(
        4,
        2,
        round_duration_us=1,
        shots=64,
        seed=3,
        basis=basis,
        preparation="product",
        noise=CircuitNoise.ideal(),
        idle_noise=IdleNoise.ideal(),
    )
    other = "X" if basis == "Z" else "Z"
    first_round_frames = set()
    for shot in tallies:
        for syndrome in shot.syndromes:
            assert select_check_results(syndrome, check_type=basis) == (0, 0, 0, 0)
        frames = {select_check_results(syndrome, check_type=other) for syndrome in shot.syndromes}
        assert len(frames) == 1, "the other check type is random once, then fixed"
        first_round_frames |= frames
        assert logical_bit_from_data_readout(shot.data_bits, basis) == 0
    assert len(first_round_frames) > 1, "the unmeasured checks start in a random frame"


def test_invalid_preparation_is_rejected():
    with pytest.raises(ValueError, match="preparation must be one of"):
        to_memory_circuit(1, preparation="teleport")


def test_summarize_memory_refuses_the_x_basis():
    tallies = run_memory(1, shots=4, seed=1, noise=CircuitNoise.ideal())
    with pytest.raises(ValueError, match="Z basis"):
        summarize_memory(tallies, basis="X")


def test_encoder_only_uses_gates_the_noise_model_covers():
    from surface_code.simulation.aer import _logical_state_circuit

    covered = {"h", "s", "sdg", "x", "y", "z", "cx", "swap"}
    for basis in ("Z", "X"):
        assert set(_logical_state_circuit(basis).count_ops()) <= covered

from math import nextafter

import pytest

pytest.importorskip("qiskit_aer")

from surface_code.simulation.record_shots import CircuitNoise, IdleNoise
from surface_code.simulation.sweep_n import (
    _odd_parity_probability,
    _wilson_interval,
    run_cadence_experiment,
    run_full_cadence_experiment,
)


def test_ideal_cadence_sweep_has_no_logical_failures():
    experiment = run_cadence_experiment(
        4,
        (0, 1, 2),
        round_duration_us=1,
        basis="Z",
        shots=16,
        seed=7,
        circuit_noise=CircuitNoise.ideal(),
        idle_noise=IdleNoise.ideal(),
    )
    assert [point.rounds for point in experiment.points] == [0, 1, 2]
    assert all(point.logical_failures == 0 for point in experiment.points)
    assert experiment.optimum_rounds == 0


def test_full_experiment_checks_bit_and_phase_memory():
    experiments = run_full_cadence_experiment(
        2,
        (0, 1),
        round_duration_us=1,
        shots=8,
        seed=3,
        circuit_noise=CircuitNoise.ideal(),
        idle_noise=IdleNoise.ideal(),
    )
    assert tuple(experiment.basis for experiment in experiments) == ("Z", "X")
    assert all(
        point.logical_failure_rate == 0 for experiment in experiments for point in experiment.points
    )


def test_cadence_rejects_rounds_that_do_not_fit_total_time():
    with pytest.raises(ValueError, match="do not fit"):
        run_cadence_experiment(
            2,
            (3,),
            round_duration_us=1,
            basis="Z",
            shots=1,
            circuit_noise=CircuitNoise.ideal(),
            idle_noise=IdleNoise.ideal(),
        )


def test_seeded_cadence_sweeps_are_reproducible():
    kwargs = {
        "round_duration_us": 0.5,
        "basis": "Z",
        "shots": 16,
        "seed": 11,
        "circuit_noise": CircuitNoise(readout=0.01),
        "idle_noise": IdleNoise(x_rate=0.01),
    }
    assert run_cadence_experiment(2, (0, 1, 2), **kwargs) == run_cadence_experiment(
        2, (0, 1, 2), **kwargs
    )


def test_preparation_is_threaded_through_both_bases():
    z_experiment, x_experiment = run_full_cadence_experiment(
        3,
        (0, 1),
        round_duration_us=1,
        shots=16,
        seed=5,
        circuit_noise=CircuitNoise.ideal(),
        idle_noise=IdleNoise.ideal(),
        preparation="product",
    )
    for experiment in (z_experiment, x_experiment):
        assert experiment.preparation == "product"
        assert all(point.logical_failures == 0 for point in experiment.points)
        # Only the decoder's own check type is counted, so a random frame adds nothing.
        assert all(point.mean_detection_events == 0 for point in experiment.points)


def test_detection_events_count_only_the_decoded_check_type():
    experiment = run_cadence_experiment(
        3,
        (2,),
        round_duration_us=1,
        basis="Z",
        shots=16,
        seed=9,
        circuit_noise=CircuitNoise.ideal(),
        idle_noise=IdleNoise(x_rate=0.2),
    )
    (point,) = experiment.points
    assert point.mean_detection_events > 0


@pytest.mark.parametrize("seed", [-1, True, 1 << 63, 1.5])
def test_cadence_validates_seed_before_offsetting_it(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        run_cadence_experiment(2, (0,), round_duration_us=1, basis="Z", seed=seed)


@pytest.mark.parametrize("rounds", [(1, True), (1, 1.0), (0, "2"), ()])
def test_cadence_validates_rounds_before_sorting_or_deduplicating(rounds):
    with pytest.raises(ValueError, match="rounds must contain"):
        run_cadence_experiment(2, rounds, round_duration_us=1, basis="Z")


def test_cadence_validates_the_entire_grid_before_running(monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("simulation started before validating every grid point")

    monkeypatch.setattr("surface_code.simulation.sweep_n.run_timed_memory", unexpected_run)
    with pytest.raises(ValueError, match="do not fit"):
        run_cadence_experiment(2, (0, 1, 3), round_duration_us=1, basis="Z")


@pytest.mark.parametrize("confidence", [0.95, nextafter(1.0, 0.0)])
@pytest.mark.parametrize("failures", [0, 1, 2500, 5000])
def test_wilson_intervals_are_finite_probabilities(failures, confidence):
    low, high = _wilson_interval(failures, 5000, confidence)
    assert 0 <= low <= failures / 5000 <= high <= 1


def test_combining_flip_probabilities_preserves_rare_faults():
    assert _odd_parity_probability((1e-20, 1e-20)) == pytest.approx(2e-20, abs=0)
    assert _odd_parity_probability((1, 1)) == 0
    assert _odd_parity_probability(()) == 0

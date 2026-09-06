import pytest

pytest.importorskip("qiskit_aer")

from surface_code.simulation.sweep_n import (
    run_cadence_experiment,
    run_full_cadence_experiment,
)
from surface_code.simulation.record_shots import CircuitNoise, IdleNoise


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
        point.logical_failure_rate == 0
        for experiment in experiments
        for point in experiment.points
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
        3, (2,), round_duration_us=1, basis="Z", shots=16, seed=9,
        circuit_noise=CircuitNoise.ideal(), idle_noise=IdleNoise(x_rate=0.2),
    )
    (point,) = experiment.points
    assert point.mean_detection_events > 0

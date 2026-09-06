import math

import pytest

from surface_code.simulation import (
    BASELINE_IDLE_NOISE,
    BASELINE_NOISE,
    BASELINE_PROFILE,
    PROFILES,
    NoiseProfile,
    get_profile,
    idle_noise_from_t1_t2,
)


def _t1(idle):
    return 1 / (2 * (idle.x_rate + idle.y_rate))


def _t2(idle):
    return 1 / (2 * (idle.y_rate + idle.z_rate))


def test_t1_t2_conversion_reproduces_both_decay_times():
    idle = idle_noise_from_t1_t2(200, 150)
    assert math.isclose(_t1(idle), 200)
    assert math.isclose(_t2(idle), 150)
    assert idle.x_rate == idle.y_rate == pytest.approx(1 / 800)


def test_t2_at_the_relaxation_limit_has_no_pure_dephasing():
    idle = idle_noise_from_t1_t2(100, 200)
    assert idle.z_rate == 0
    assert math.isclose(_t2(idle), 200)


@pytest.mark.parametrize("t1, t2", [(100, 201), (0, 50), (50, 0), (-1, 10)])
def test_conversion_rejects_unphysical_times(t1, t2):
    with pytest.raises(ValueError):
        idle_noise_from_t1_t2(t1, t2)


def test_profiles_are_keyed_by_their_own_names():
    assert all(name == profile.name for name, profile in PROFILES.items())
    assert all(profile.round_duration_us > 0 for profile in PROFILES.values())
    assert BASELINE_PROFILE.circuit_noise == BASELINE_NOISE
    assert BASELINE_PROFILE.idle_noise == BASELINE_IDLE_NOISE


def test_device_like_profiles_have_lower_gate_noise_than_the_baseline():
    for name in ("ibm-heron", "google-willow"):
        assert get_profile(name).circuit_noise.two_qubit < BASELINE_NOISE.two_qubit
        assert get_profile(name).circuit_noise.readout < BASELINE_NOISE.readout


def test_unknown_profile_lists_the_choices():
    with pytest.raises(ValueError, match="baseline"):
        get_profile("nope")


def test_profile_rejects_negative_round_duration():
    with pytest.raises(ValueError, match="round_duration_us"):
        NoiseProfile("bad", "", "", BASELINE_NOISE, BASELINE_IDLE_NOISE, -1)

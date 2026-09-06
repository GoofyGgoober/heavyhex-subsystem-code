"""Named noise profiles for the memory experiments.

The project baseline is a fixed reference for reproducible comparisons. The
device-like profiles convert published typical figures for public
superconducting processors into this package's parameters. They are not live
calibration data: replace them with measured backend properties before making
any claim about a specific machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._validation import validate_nonnegative_number
from .record_shots import BASELINE_IDLE_NOISE, BASELINE_NOISE, CircuitNoise, IdleNoise


def idle_noise_from_t1_t2(t1_us: float, t2_us: float) -> IdleNoise:
    """Pauli-twirled relaxation and dephasing as per-microsecond event rates.

    Population decays as ``exp(-t / T1)``, which the independent-event model
    reproduces with X and Y rates of ``1 / (4 T1)`` each. Coherence decays as
    ``exp(-t / T2)``; the extra loss beyond relaxation is pure dephasing with
    ``1 / Tphi = 1 / T2 - 1 / (2 T1)``, reproduced by a Z rate of ``1 / (2 Tphi)``.
    """
    validate_nonnegative_number("t1_us", t1_us)
    validate_nonnegative_number("t2_us", t2_us)
    if t1_us == 0 or t2_us == 0:
        raise ValueError("T1 and T2 must be positive")
    inverse_tphi = 1 / t2_us - 1 / (2 * t1_us)
    if inverse_tphi < -1e-12:
        raise ValueError(f"T2 cannot exceed 2 T1, got T1={t1_us:g} us and T2={t2_us:g} us")
    per_pauli = 1 / (4 * t1_us)
    return IdleNoise(x_rate=per_pauli, y_rate=per_pauli, z_rate=max(inverse_tphi, 0.0) / 2)


@dataclass(frozen=True)
class NoiseProfile:
    """A complete parameter set for a fixed-duration memory experiment."""

    name: str
    summary: str
    source: str
    circuit_noise: CircuitNoise
    idle_noise: IdleNoise
    round_duration_us: float

    def __post_init__(self) -> None:
        validate_nonnegative_number("round_duration_us", self.round_duration_us)


BASELINE_PROFILE = NoiseProfile(
    name="baseline",
    summary=(
        "Circuit noise 1q 0.1%, 2q 1%, readout 2%, reset 1%; "
        "idle X/Y/Z events 0.001 per µs each; 1 µs rounds."
    ),
    source="Project baseline rates, not a device.",
    circuit_noise=BASELINE_NOISE,
    idle_noise=BASELINE_IDLE_NOISE,
    round_duration_us=1.0,
)

IBM_HERON_PROFILE = NoiseProfile(
    name="ibm-heron",
    summary=(
        "Circuit noise 1q 0.03%, 2q 0.4%, readout 1%, reset 0.5%; "
        "T1 200 µs, T2 150 µs; 2 µs rounds."
    ),
    source="Published typical medians for IBM Heron processors, not live calibration.",
    circuit_noise=CircuitNoise(single_qubit=0.0003, two_qubit=0.004, readout=0.01, reset=0.005),
    idle_noise=idle_noise_from_t1_t2(200.0, 150.0),
    round_duration_us=2.0,
)

GOOGLE_WILLOW_PROFILE = NoiseProfile(
    name="google-willow",
    summary=(
        "Circuit noise 1q 0.035%, 2q 0.33%, readout 0.8%, reset 0.3%; "
        "T1 68 µs, T2 89 µs; 1 µs rounds."
    ),
    source="Published device means for Google Willow (2024), not live calibration.",
    circuit_noise=CircuitNoise(single_qubit=0.00035, two_qubit=0.0033, readout=0.008, reset=0.003),
    idle_noise=idle_noise_from_t1_t2(68.0, 89.0),
    round_duration_us=1.0,
)

PROFILES: dict[str, NoiseProfile] = {
    profile.name: profile
    for profile in (BASELINE_PROFILE, IBM_HERON_PROFILE, GOOGLE_WILLOW_PROFILE)
}


def get_profile(name: str) -> NoiseProfile:
    """Look up a profile by name, listing the valid names on failure."""
    try:
        return PROFILES[name]
    except KeyError:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown noise profile {name!r}; choose from {choices}") from None

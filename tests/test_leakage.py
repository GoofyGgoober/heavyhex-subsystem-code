"""The leakage simulators of docs/analysis on the smallest case of a left-out CZ.

q0 is leaked in |1> and its CZ with q1, prepared in |+>, must not act: q1 is then measured
in the X basis and reads 0, so the observable flips in every shot. The exact simulators
leave the CZ out; frame_sim.py's default ("reference") applies it with q0's noiseless Z
value, which here is a definite 1, and so never flips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "analysis"))

import exact_sim
import frame_sim
import tableau_ref

EXAMPLE = stim.Circuit(
    """
    R 0 1 2
    X 0
    CZ 0 2
    S 1
    SQRT_X 1
    S 1
    CZ 0 1
    S 1
    SQRT_X 1
    S 1
    M 1
    OBSERVABLE_INCLUDE(0) rec[-1]
    """
)
LEAKED = {"leak": {0: 1.0}, "partner_phase": 0.0, "transfer": 0.0}
SHOTS = 400


def flip_rate(observables: np.ndarray) -> float:
    return float(observables.mean())


def test_the_exact_simulators_leave_the_cz_out():
    leakage = exact_sim.Leakage(**LEAKED)
    assert flip_rate(tableau_ref.simulate(EXAMPLE, SHOTS, leakage, "identity", seed=1)[1]) == 1.0
    assert flip_rate(exact_sim.simulate(EXAMPLE, SHOTS, leakage, "identity", seed=1)[1]) == 1.0


def test_the_frame_simulator_applies_the_cz_with_the_noiseless_value():
    leakage = frame_sim.Leakage(**LEAKED)
    assert leakage.skipped_cz == "reference"
    assert flip_rate(frame_sim.simulate(EXAMPLE, SHOTS, leakage, seed=1)[1]) == 0.0


def test_a_random_phase_on_the_partner_flips_half_the_shots():
    leakage = frame_sim.Leakage(**LEAKED, skipped_cz="dephase")
    assert abs(flip_rate(frame_sim.simulate(EXAMPLE, 4000, leakage, seed=1)[1]) - 0.5) < 0.05


def test_without_leakage_the_two_handlings_are_the_same_simulation():
    noisy = stim.Circuit(
        """
        R 0 1 2 3
        SQRT_X 0 2
        CZ 0 1 2 3
        DEPOLARIZE2(0.01) 0 1 2 3
        SQRT_X 1 3
        M(0.01) 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
        """
    )
    a = frame_sim.simulate(noisy, SHOTS, frame_sim.Leakage(skipped_cz="reference"), seed=5)
    b = frame_sim.simulate(noisy, SHOTS, frame_sim.Leakage(skipped_cz="dephase"), seed=5)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))

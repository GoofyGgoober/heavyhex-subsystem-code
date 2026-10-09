"""A Pauli-frame simulator for the noisy stim circuits, with leakage.

Stim cannot keep a qubit out of {|0>, |1>}. This replays the same stim circuit the
analysis builds (every gate, wait and error channel of the calibration model, in order)
on Pauli frames, shot by shot in numpy, and adds a leaked state per qubit:

- Each CZ leaves each of its two qubits leaked with probability leak[q] (a qubit already
  leaked stays so). While a qubit is leaked, its CZs act on the partner instead of the
  gate: with probability partner_strength a random Pauli (partner_kind "depolarize"), an
  X ("bitflip", as an exchange of excitation would give) or a Z ("phase"), and with
  probability transfer the leakage spreads to the partner.
- A leaked qubit reads 1 with probability read_leaked, whatever its frame. A reset, which
  on IBM hardware measures and flips, leaves it leaked.
- At every TICK (each half round) a leaked qubit returns with probability
  1 - exp(-half / lifetime), to a random state in {|0>, |1>}.

With no leakage and no overrides it reproduces the model exactly (checked against stim's
own sampler in validate()). Optional overrides: cz_scale multiplies every CZ's
depolarizing error; reset_error replaces the reset's X error (the model uses the readout
error); readout gives per-qubit (P(1|0), P(0|1)) in place of the symmetric readout error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

PAULI2 = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]  # 0 I, 1 X, 2 Y, 3 Z
XBIT = np.array([0, 1, 1, 0], dtype=bool)
ZBIT = np.array([0, 0, 1, 1], dtype=bool)


@dataclass
class Leakage:
    leak: dict = field(default_factory=dict)  # chip qubit -> probability per CZ it takes part in
    lifetime_us: float = 25.0
    half_round_us: float = 3.95
    read_leaked: float = 0.9
    partner_phase: float = 0.5  # partner_strength, kept under its first name
    partner_kind: str = "phase"
    transfer: float = 0.0
    cz_scale: float = 1.0
    reset_error: dict | None = None  # chip qubit -> X error after a reset
    readout: dict | None = None  # chip qubit -> (P(1|0), P(0|1))


def simulate(model: stim.Circuit, shots: int, leakage: Leakage | None = None, seed: int = 0):
    """Detector and observable flips (shots x detectors, shots x observables)."""
    leakage = leakage or Leakage()
    rng = np.random.default_rng(seed)
    n = model.num_qubits
    x = np.zeros((n, shots), bool)
    z = rng.random((n, shots)) < 0.5
    leaked = np.zeros((n, shots), bool)
    reference = model.reference_sample()
    flips: list[np.ndarray] = []
    detectors: list[np.ndarray] = []
    observables: dict[int, np.ndarray] = {}
    decay = 1 - np.exp(-leakage.half_round_us / leakage.lifetime_us)
    last_reset: set[int] = set()

    def bern(p, size=shots):
        return rng.random(size) < p

    for ins in model.flattened():
        name = ins.name
        targets = [t.value for t in ins.targets_copy()]
        args = ins.gate_args_copy()
        just_reset, last_reset = last_reset, set()
        if name == "TICK":
            if leakage.leak:
                back = leaked & (rng.random(leaked.shape) < decay)
                leaked &= ~back
                x ^= back & (rng.random(leaked.shape) < 0.5)
                z ^= back & (rng.random(leaked.shape) < 0.5)
        elif name == "R":
            for q in targets:
                x[q] = False
                z[q] = bern(0.5)
            last_reset = set(targets)
        elif name == "X_ERROR":
            for q in targets:
                p = args[0]
                if q in just_reset and leakage.reset_error is not None:
                    p = leakage.reset_error.get(q, p)
                x[q] ^= bern(p)
        elif name == "M":
            p = args[0] if args else 0.0
            for q in targets:
                k = len(flips)
                if leakage.readout is not None and q in leakage.readout:
                    p10, p01 = leakage.readout[q]
                    true = reference[k] ^ x[q]
                    flip = x[q] ^ np.where(true, bern(p01), bern(p10))
                else:
                    flip = x[q] ^ bern(p)
                if leakage.leak:
                    flip = np.where(leaked[q], bern(leakage.read_leaked) ^ reference[k], flip)
                flips.append(flip)
                z[q] = bern(0.5)
        elif name in ("S", "S_DAG"):
            for q in targets:
                z[q] ^= x[q]
        elif name == "SQRT_X":
            for q in targets:
                x[q] ^= z[q]
        elif name in ("X", "Y", "Z", "I"):
            pass
        elif name == "CZ":
            for a, b in zip(targets[::2], targets[1::2]):
                if leakage.leak:
                    la, lb = leaked[a].copy(), leaked[b].copy()
                    za, zb = z[a] ^ x[b], z[b] ^ x[a]
                    z[a] = np.where(lb, z[a], za)
                    z[b] = np.where(la, z[b], zb)
                    for q, other in ((a, lb), (b, la)):  # q's partner is leaked
                        hit = other & bern(leakage.partner_phase)
                        if leakage.partner_kind == "bitflip":  # an exchange of excitation: an X
                            x[q] ^= hit
                        elif leakage.partner_kind == "depolarize":
                            kind = rng.integers(1, 4, shots)
                            x[q] ^= hit & XBIT[kind]
                            z[q] ^= hit & ZBIT[kind]
                        else:
                            z[q] ^= hit
                        if leakage.transfer:
                            moved = other & ~leaked[q] & bern(leakage.transfer)
                            leaked[q] |= moved
                    for q in (a, b):
                        p = leakage.leak.get(q, 0.0)
                        if p:
                            leaked[q] |= bern(p)
                else:
                    z[a] ^= x[b]
                    z[b] ^= x[a]
        elif name == "DEPOLARIZE1":
            p = args[0]
            for q in targets:
                hit = bern(p)
                kind = rng.integers(1, 4, shots)
                x[q] ^= hit & XBIT[kind]
                z[q] ^= hit & ZBIT[kind]
        elif name == "DEPOLARIZE2":
            p = min(args[0] * leakage.cz_scale, 15 / 16)
            for a, b in zip(targets[::2], targets[1::2]):
                hit = bern(p)
                kind = rng.integers(0, 15, shots)
                pa = np.array([PAULI2[i][0] for i in range(15)])[kind]
                pb = np.array([PAULI2[i][1] for i in range(15)])[kind]
                x[a] ^= hit & XBIT[pa]; z[a] ^= hit & ZBIT[pa]
                x[b] ^= hit & XBIT[pb]; z[b] ^= hit & ZBIT[pb]
        elif name == "PAULI_CHANNEL_1":
            px, py, pz = args
            for q in targets:
                u = rng.random(shots)
                isx, isy = u < px, (u >= px) & (u < px + py)
                isz = (u >= px + py) & (u < px + py + pz)
                x[q] ^= isx | isy
                z[q] ^= isy | isz
        elif name == "DETECTOR":
            recs = [t.value for t in ins.targets_copy() if t.is_measurement_record_target]
            d = np.zeros(shots, bool)
            for r in recs:
                d ^= flips[len(flips) + r]
            detectors.append(d)
        elif name == "OBSERVABLE_INCLUDE":
            k = int(args[0])
            o = observables.setdefault(k, np.zeros(shots, bool))
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    o ^= flips[len(flips) + t.value]
        elif name in ("QUBIT_COORDS", "SHIFT_COORDS"):
            pass
        else:
            raise ValueError(f"unhandled instruction {name}")
    det = np.array(detectors).T if detectors else np.zeros((shots, 0), bool)
    obs = np.array([observables[k] for k in sorted(observables)]).T if observables else np.zeros((shots, 0), bool)
    return det, obs


def validate(model: stim.Circuit, shots: int = 20000) -> float:
    """Largest gap between this simulator's detector rates (no leakage) and stim's sampler's."""
    mine, _ = simulate(model, shots, seed=1)
    theirs = model.compile_detector_sampler(seed=1).sample(shots)
    return float(np.abs(mine.mean(0) - theirs.mean(0)).max())

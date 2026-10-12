"""An exact, shot-by-shot stabilizer simulation of the same leakage model, for checking leaky.py.

Unlike the Pauli-frame simulator, this keeps the actual quantum state of every shot in a
stim.TableauSimulator, so a CZ with a leaked qubit can really be left out. It runs the
same flattened noisy circuit and the same leakage rules as leaky.Leakage:

  skipped "identity"  the CZ does not act at all (the paper's literal wording)
  skipped "dephase"   the CZ does not act and the unleaked partner gets a uniformly random
                      phase (Z with probability 1/2), the model leaky_fix/leaky.py implements

then, as in leaky.py, the partner_kind disturbance with probability partner_phase and the
transfer. A leaked qubit's own state is never used again: its CZs are skipped, its
readouts are replaced by Bernoulli(read_leaked), and when it returns it is reset to a
random computational state. Measuring or resetting it meanwhile only acts on a qubit
that is traced out, so it changes nothing for the others.

Slow (Python loop per shot); for validation on small circuits and spot checks only.
"""

from __future__ import annotations

import numpy as np
import stim

PLAIN = {"S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "H", "X", "Y", "Z", "I",
         "DEPOLARIZE1", "PAULI_CHANNEL_1", "X_ERROR", "Z_ERROR", "Y_ERROR"}


def _program(model: stim.Circuit, leakage):
    """Flatten the circuit into steps; runs of plain instructions become one stim.Circuit."""
    steps, block = [], stim.Circuit()
    last_reset: set[int] = set()

    def flush():
        nonlocal block
        if len(block):
            steps.append(("plain", block))
            block = stim.Circuit()

    for ins in model.flattened():
        name = ins.name
        targets = [t.value for t in ins.targets_copy()]
        args = ins.gate_args_copy()
        just_reset, last_reset = last_reset, set()
        if name in ("QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name == "X_ERROR" and leakage is not None and leakage.reset_error is not None:
            for q in targets:
                p = args[0]
                if q in just_reset:
                    p = leakage.reset_error.get(q, p)
                block.append("X_ERROR", [q], [p])
            continue
        if name == "DEPOLARIZE2":
            scale = leakage.cz_scale if leakage is not None else 1.0
            block.append("DEPOLARIZE2", targets, [min(args[0] * scale, 15 / 16)])
            continue
        if name in PLAIN:
            block.append(name, targets, args)
            continue
        flush()
        if name == "R":
            steps.append(("R", targets))
            last_reset = set(targets)
        elif name == "M":
            steps.append(("M", targets, args[0] if args else 0.0))
        elif name == "CZ":
            steps.append(("CZ", list(zip(targets[::2], targets[1::2]))))
        elif name == "TICK":
            steps.append(("TICK",))
        elif name == "DETECTOR":
            steps.append(("DET", [t.value for t in ins.targets_copy() if t.is_measurement_record_target]))
        elif name == "OBSERVABLE_INCLUDE":
            steps.append(("OBS", int(args[0]), [t.value for t in ins.targets_copy() if t.is_measurement_record_target]))
        else:
            raise ValueError(f"unhandled instruction {name}")
    flush()
    return steps


def simulate(model: stim.Circuit, shots: int, leakage=None, skipped: str = "identity", seed: int = 0):
    """Detector and observable flips (shots x detectors, shots x observables), exactly."""
    rng = np.random.default_rng(seed)
    steps = _program(model, leakage)
    reference = model.reference_sample()
    n = model.num_qubits
    on = leakage is not None and bool(leakage.leak)
    decay = 1 - np.exp(-leakage.half_round_us / leakage.lifetime_us) if on else 0.0
    dets, obss = [], []
    for shot in range(shots):
        sim = stim.TableauSimulator(seed=int(rng.integers(2**63)))
        sim.set_num_qubits(n)
        leaked = np.zeros(n, bool)
        rec: list[bool] = []
        det: list[bool] = []
        obs: dict[int, bool] = {}
        for st in steps:
            kind = st[0]
            if kind == "plain":
                sim.do_circuit(st[1])
            elif kind == "R":
                sim.reset(*st[1])  # on a leaked qubit this acts on a traced-out qubit
            elif kind == "M":
                _, targets, p = st
                vals = sim.measure_many(*targets)
                for q, v in zip(targets, vals):
                    _k = len(rec)
                    if leakage is not None and leakage.readout is not None and q in leakage.readout:
                        p10, p01 = leakage.readout[q]
                        v = v ^ (rng.random() < (p01 if v else p10))
                    else:
                        v = v ^ (rng.random() < p)
                    if on and leaked[q]:
                        v = rng.random() < leakage.read_leaked
                    rec.append(bool(v))
            elif kind == "CZ":
                pairs = st[1]
                if not on:
                    for a, b in pairs:
                        sim.cz(a, b)
                    continue
                for a, b in pairs:
                    la, lb = leaked[a], leaked[b]
                    if not la and not lb:
                        sim.cz(a, b)
                    for q, other in ((a, lb), (b, la)):
                        if not other:
                            continue
                        if skipped == "dephase" and rng.random() < 0.5:
                            sim.z(q)
                        if rng.random() < leakage.partner_phase:
                            if leakage.partner_kind == "bitflip":
                                sim.x(q)
                            elif leakage.partner_kind == "depolarize":
                                (sim.x, sim.y, sim.z)[rng.integers(3)](q)
                            else:
                                sim.z(q)
                        if leakage.transfer and not leaked[q] and rng.random() < leakage.transfer:
                            leaked[q] = True
                    for q in (a, b):
                        p = leakage.leak.get(q, 0.0)
                        if p and rng.random() < p:
                            leaked[q] = True
            elif kind == "TICK":
                if on:
                    for q in np.flatnonzero(leaked):
                        if rng.random() < decay:
                            leaked[q] = False
                            sim.reset(int(q))
                            if rng.random() < 0.5:
                                sim.x(int(q))
            elif kind == "DET":
                v = False
                for r in st[1]:
                    v ^= rec[len(rec) + r] ^ bool(reference[len(rec) + r])
                det.append(v)
            elif kind == "OBS":
                v = obs.get(st[1], False)
                for r in st[2]:
                    v ^= rec[len(rec) + r] ^ bool(reference[len(rec) + r])
                obs[st[1]] = v
        dets.append(det)
        obss.append([obs[k] for k in sorted(obs)])
    return np.array(dets, bool).reshape(shots, -1), np.array(obss, bool).reshape(shots, -1)

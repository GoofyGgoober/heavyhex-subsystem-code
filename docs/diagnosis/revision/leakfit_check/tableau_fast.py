"""leaky_fix/tableau_ref.simulate, made faster without changing a single output bit.

The exact leakage simulation (one stim.TableauSimulator per shot) spends most of its time
in Python loops over CZ pairs and readouts. This version runs the same flattened program
(tableau_ref._program, imported) and consumes the random numbers in the same order, so for
the same seed it returns identical detector and observable bits (checked by selftest()):

  - a CZ layer none of whose operands is leaked at its start is applied as one stim call,
    and its per-qubit leak draws are one rng.random(n) call (n scalar draws in the same
    order, as numpy's Generator produces them); otherwise the original pair-by-pair loop runs;
  - a readout layer none of whose qubits is leaked draws its readout flips in one call;
    otherwise the original qubit-by-qubit loop runs;
  - detectors and observables are computed after the shot, as parities of the record,
    instead of step by step (no random numbers are involved).

    from tableau_fast import simulate   # same signature as tableau_ref.simulate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import stim

FIX = Path(__file__).resolve().parents[1] / "leaky_fix"
if str(FIX) not in sys.path:
    sys.path.insert(0, str(FIX))
import tableau_ref  # noqa: E402


def _compile(model: stim.Circuit, leakage):
    steps = tableau_ref._program(model, leakage)
    out, n_rec = [], 0
    det_rows, obs_rows = [], {}
    for st in steps:
        kind = st[0]
        if kind == "M":
            _, targets, p = st
            out.append(("M", list(targets), p, np.array(targets)))
            n_rec += len(targets)
        elif kind == "CZ":
            pairs = st[1]
            flat = [q for ab in pairs for q in ab]
            disjoint = len(set(flat)) == len(flat)
            out.append(("CZ", pairs, flat, np.array(flat, dtype=np.int64), disjoint))
        elif kind == "DET":
            det_rows.append([n_rec + r for r in st[1]])
        elif kind == "OBS":
            obs_rows.setdefault(st[1], []).extend(n_rec + r for r in st[2])
        else:
            out.append(st)
    return out, n_rec, det_rows, obs_rows


def simulate(model: stim.Circuit, shots: int, leakage=None, skipped: str = "identity", seed: int = 0):
    """Detector and observable flips (shots x detectors, shots x observables), exactly; = tableau_ref.simulate."""
    rng = np.random.default_rng(seed)
    steps, n_rec, det_rows, obs_rows = _compile(model, leakage)
    reference = np.array(model.reference_sample(), dtype=bool)
    n = model.num_qubits
    on = leakage is not None and bool(leakage.leak)
    decay = 1 - np.exp(-leakage.half_round_us / leakage.lifetime_us) if on else 0.0
    # per-step thresholds for the vectorised paths
    leak_p = None
    if on:
        leak_p = np.array([leakage.leak.get(q, 0.0) for q in range(n)])
    ro = leakage.readout if leakage is not None and leakage.readout is not None else {}
    Dm = np.zeros((len(det_rows), n_rec), np.uint8)
    for i, r in enumerate(det_rows):
        for k in r:
            Dm[i, k] ^= 1
    okeys = sorted(obs_rows)
    Om = np.zeros((len(okeys), n_rec), np.uint8)
    for i, k in enumerate(okeys):
        for j in obs_rows[k]:
            Om[i, j] ^= 1
    m_thr = {}
    for st in steps:
        if st[0] == "M":
            _, targets, p, tarr = st
            p10 = np.array([ro[q][0] if q in ro else p for q in targets])
            p01 = np.array([ro[q][1] if q in ro else p for q in targets])
            m_thr[id(st)] = (p10, p01)
    recs = np.empty((shots, n_rec), bool)
    for shot in range(shots):
        sim = stim.TableauSimulator(seed=int(rng.integers(2**63)))
        sim.set_num_qubits(n)
        leaked = np.zeros(n, bool)
        k = 0
        rec = recs[shot]
        for st in steps:
            kind = st[0]
            if kind == "plain":
                sim.do_circuit(st[1])
            elif kind == "R":
                sim.reset(*st[1])
            elif kind == "M":
                _, targets, p, tarr = st
                vals = np.array(sim.measure_many(*targets), dtype=bool)
                if not (on and leaked[tarr].any()):
                    p10, p01 = m_thr[id(st)]
                    u = rng.random(len(targets))
                    vals ^= u < np.where(vals, p01, p10)
                    rec[k:k + len(targets)] = vals
                    k += len(targets)
                else:
                    for q, v in zip(targets, vals):
                        if q in ro:
                            p10, p01 = ro[q]
                            v = v ^ (rng.random() < (p01 if v else p10))
                        else:
                            v = v ^ (rng.random() < p)
                        if on and leaked[q]:
                            v = rng.random() < leakage.read_leaked
                        rec[k] = v
                        k += 1
            elif kind == "CZ":
                _, pairs, flat, farr, disjoint = st
                if not on:
                    sim.cz(*flat)
                    continue
                pf = leak_p[farr]
                if disjoint and not leaked[farr].any() and (pf > 0).all():
                    sim.cz(*flat)
                    hit = rng.random(len(flat)) < pf
                    if hit.any():
                        leaked[farr[hit]] = True
                    continue
                for a, b in pairs:  # tableau_ref's loop
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
    flips = (recs ^ reference[None, :n_rec]).astype(np.uint8)
    det = (flips @ Dm.T.astype(np.int64)) % 2 if len(det_rows) else np.zeros((shots, 0), np.int64)
    obs = (flips @ Om.T.astype(np.int64)) % 2 if len(okeys) else np.zeros((shots, 0), np.int64)
    return det.astype(bool), obs.astype(bool)


def selftest(model, leakage, shots=40, seed=3, skipped="identity") -> bool:
    d1, o1 = tableau_ref.simulate(model, shots, leakage, skipped, seed=seed)
    d2, o2 = simulate(model, shots, leakage, skipped, seed=seed)
    return bool(np.array_equal(d1, d2) and np.array_equal(o1, o2))

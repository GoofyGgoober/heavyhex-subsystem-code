"""Exact per-shot stabilizer simulation of the leakage model, fast enough for grid fits.

The model is the one of tableau_ref.py (and of the paper), simulated exactly:
every shot keeps its own stabilizer state, so a CZ with a leaked operand can really be
left out. Two channels for that CZ (`skipped`):

  "identity"  the CZ does not act (the paper's stated channel: a leaked qubit's CZ is absent)
  "dephase"   the CZ does not act and the unleaked partner gets Z with probability 1/2,
              i.e. an independently random conditional phase at every such CZ (a surrogate
              inspired by the leakage-conditional phases of Varbanov et al.,
              arXiv:2002.07119, which are random per pair but fixed within a run)
  "pi"        the CZ does not act and the unleaked partner gets Z: a conditional phase of
              pi, as if the leaked qubit were |1> (a supplementary check)

then, as before, the partner gets the partner_kind disturbance with probability
partner_phase ("bitflip": X; "depolarize": X, Y or Z; "phase": Z) and the leakage passes to
it with probability transfer. Each CZ leaves each operand leaked with probability leak[q].
A leaked qubit's own quantum state is never used again: its CZs are skipped, its readouts
are Bernoulli(read_leaked), and at a TICK it returns with probability
1 - exp(-half_round_us / lifetime_us), reset to a random computational state.

Why this is faster than tableau_ref.py while computing the same thing. The leakage process
(which qubit is leaked when, partner hits, transfers, returns, leaked readouts) is classical
and does not depend on the quantum state, so it is sampled for all shots at once (numpy, in
circuit order, with the same rules as tableau_ref.py). Each shot's quantum state is then
evolved by stim's TableauSimulator in C++ over precompiled circuit segments, broken only at
the CZs and TICKs that this shot's leakage touches. Readout errors and leaked readouts act
on the recorded bit only (as in tableau_ref.py), so they are applied afterwards, vectorized.
Checked against tableau_ref.py in leakage_checks.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

PLAIN = {
    "S",
    "S_DAG",
    "SQRT_X",
    "SQRT_X_DAG",
    "H",
    "X",
    "Y",
    "Z",
    "I",
    "DEPOLARIZE1",
    "PAULI_CHANNEL_1",
    "X_ERROR",
    "Z_ERROR",
    "Y_ERROR",
}
BLOCK = 16  # events per precompiled block (a block with no event touched runs in one call)


@dataclass
class Leakage:
    """Same fields and meaning as frame_sim.Leakage (so tableau_ref.py accepts it)."""

    leak: dict = field(default_factory=dict)  # chip qubit -> leak probability per CZ (.get(q))
    lifetime_us: float = 25.0
    half_round_us: float = 3.95
    read_leaked: float = 0.9
    partner_phase: float = 0.5  # the partner disturbance's probability ("strength")
    partner_kind: str = "phase"
    transfer: float = 0.0
    cz_scale: float = 1.0
    reset_error: dict | None = None  # chip qubit -> X error after a reset
    readout: dict | None = None  # chip qubit -> (P(1|0), P(0|1))


class Uniform(dict):
    """Every qubit leaks with the same probability per CZ (rerun._AllQubits with one value)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def get(self, q, default=None):
        return self.p

    def __bool__(self):
        return True


def instructions(model: stim.Circuit) -> list:
    """The flattened circuit as plain tuples, at full precision (stim's text and pickle round
    the gate arguments to 6 digits), for passing to worker processes."""
    out = []
    for ins in model.flattened():
        tg = [
            ("rec", t.value) if t.is_measurement_record_target else ("q", t.value)
            for t in ins.targets_copy()
        ]
        out.append((ins.name, tg, ins.gate_args_copy()))
    return out


def rebuild(data: list) -> stim.Circuit:
    c = stim.Circuit()
    for name, tg, args in data:
        c.append(name, [stim.target_rec(v) if k == "rec" else v for k, v in tg], args)
    return c


class Program:
    """The noisy circuit cut at its CZs and TICKs, with the per-circuit overrides baked in."""

    def __init__(self, model: stim.Circuit, leakage: Leakage | None):
        self.n = model.num_qubits
        self.reference = np.array(model.reference_sample(), bool)
        segs: list[stim.Circuit] = []
        seg = stim.Circuit()
        self.events: list[tuple] = []  # ("CZ", [(a, b), ...]) or ("TICK",)
        self.meas_q: list[int] = []
        self.meas_p: list[float] = []
        dets: list[list[int]] = []
        obs: dict[int, list[int]] = {}
        last_reset: set[int] = set()
        for ins in model.flattened():
            name = ins.name
            targets = [t.value for t in ins.targets_copy()]
            args = ins.gate_args_copy()
            just_reset, last_reset = last_reset, set()
            if name in ("QUBIT_COORDS", "SHIFT_COORDS"):
                continue
            if name == "X_ERROR" and leakage is not None and leakage.reset_error is not None:
                for q in targets:
                    p = leakage.reset_error.get(q, args[0]) if q in just_reset else args[0]
                    seg.append("X_ERROR", [q], [p])
            elif name == "DEPOLARIZE2":
                scale = leakage.cz_scale if leakage is not None else 1.0
                seg.append("DEPOLARIZE2", targets, [min(args[0] * scale, 15 / 16)])
            elif name in PLAIN:
                seg.append(name, targets, args)
            elif name == "R":
                seg.append("R", targets)
                last_reset = set(targets)
            elif name == "M":  # raw measurement; the readout error is applied to the record later
                seg.append("M", targets)
                self.meas_q += targets
                self.meas_p += [args[0] if args else 0.0] * len(targets)
            elif name in ("CZ", "TICK"):
                segs.append(seg)
                seg = stim.Circuit()
                if name == "CZ":
                    self.events.append(("CZ", list(zip(targets[::2], targets[1::2]))))
                else:
                    self.events.append(("TICK",))
            elif name == "DETECTOR":
                m = len(self.meas_q)
                dets.append(
                    [m + t.value for t in ins.targets_copy() if t.is_measurement_record_target]
                )
            elif name == "OBSERVABLE_INCLUDE":
                m = len(self.meas_q)
                obs.setdefault(int(args[0]), []).extend(
                    m + t.value for t in ins.targets_copy() if t.is_measurement_record_target
                )
            else:
                raise ValueError(f"unhandled instruction {name}")
        segs.append(seg)
        self.segments = segs
        # each segment followed by its event acting normally (a TICK does nothing quantum)
        self.plain_step = []
        for i, ev in enumerate(self.events):
            c = segs[i].copy()
            if ev[0] == "CZ":
                c.append("CZ", [q for pr in ev[1] for q in pr])
            self.plain_step.append(c)
        nb = (len(self.events) + BLOCK - 1) // BLOCK
        self.blocks = []
        for b in range(nb):
            c = stim.Circuit()
            for c2 in self.plain_step[b * BLOCK : (b + 1) * BLOCK]:
                c += c2
            self.blocks.append(c)
        nm = len(self.meas_q)
        self.det = np.zeros((len(dets), nm), np.float32)
        for i, d in enumerate(dets):
            for k in d:
                self.det[i, k] += 1
        self.obs = np.zeros((len(obs), nm), np.float32)
        for i, k in enumerate(sorted(obs)):
            for m in obs[k]:
                self.obs[i, m] += 1


def simulate(
    model: stim.Circuit,
    shots: int,
    leakage: Leakage | None = None,
    skipped: str = "identity",
    seed: int = 0,
    program: Program | None = None,
    return_raw: bool = False,
):
    """Detector and observable flips (shots x detectors, shots x observables), exactly.

    Same signature and output as tableau_ref.simulate (plus an optional precompiled program).
    """
    if skipped not in ("identity", "dephase", "pi"):
        raise ValueError(f"unknown skipped channel {skipped}")
    prog = program or Program(model, leakage)
    rng = np.random.default_rng(seed)
    n, ev = prog.n, prog.events
    on = leakage is not None and bool(leakage.leak)
    nm = len(prog.meas_q)

    # ---- 1. the classical leakage process, all shots at once, in circuit order
    leaked = np.zeros((n, shots), bool)
    aff = np.zeros((len(ev), shots), bool)
    pair_index = []  # event -> slice into the pair arrays
    npairs = sum(len(e[1]) for e in ev if e[0] == "CZ")
    skip = np.zeros((npairs, shots), bool)
    pa = np.zeros((npairs, 2, shots), np.uint8)  # Pauli on (a, b) after the CZ: bit0 X, bit1 Z
    ret: dict[int, tuple] = {}  # TICK event -> (qubits, shots, x flips)
    leaked_at_m = np.zeros((nm, shots), bool)
    decay = 1 - np.exp(-leakage.half_round_us / leakage.lifetime_us) if on else 0.0
    # measurement instructions sit inside segments: record which events precede each record
    seg_meas = _meas_per_segment(prog)
    k = 0
    p0 = 0

    def bern(p):
        return rng.random(shots) < p

    for i in range(len(ev) + 1):
        for _ in range(seg_meas[i]):  # leaked state at each measurement of segment i
            if on:
                leaked_at_m[k] = leaked[prog.meas_q[k]]
            k += 1
        if i == len(ev):
            break
        e = ev[i]
        if e[0] == "TICK":
            pair_index.append(None)
            if on:
                back = leaked & (rng.random(leaked.shape) < decay)
                if back.any():
                    qs, ss = np.nonzero(back)
                    ret[i] = (qs, ss, rng.random(len(qs)) < 0.5)
                    aff[i, ss] = True
                    leaked &= ~back
            continue
        pairs = e[1]
        pair_index.append(slice(p0, p0 + len(pairs)))
        for j, (a, b) in enumerate(pairs):
            if not on:
                continue
            la, lb = leaked[a].copy(), leaked[b].copy()
            skip[p0 + j] = la | lb
            for side, q, other in ((0, a, lb), (1, b, la)):
                if not other.any():
                    continue
                x = np.zeros(shots, bool)
                if skipped == "dephase":
                    z = other & bern(0.5)
                elif skipped == "pi":
                    z = other.copy()
                else:
                    z = np.zeros(shots, bool)
                hit = other & bern(leakage.partner_phase)
                if leakage.partner_kind == "bitflip":
                    x ^= hit
                elif leakage.partner_kind == "depolarize":
                    kind = rng.integers(1, 4, shots)  # 1 X, 2 Y, 3 Z
                    x ^= hit & ((kind == 1) | (kind == 2))
                    z ^= hit & ((kind == 2) | (kind == 3))
                else:
                    z ^= hit
                pa[p0 + j, side] = x.astype(np.uint8) | (z.astype(np.uint8) << 1)
                if leakage.transfer:
                    leaked[q] |= other & ~leaked[q] & bern(leakage.transfer)
            for q in (a, b):
                p = leakage.leak.get(q, 0.0)
                if p:
                    leaked[q] |= bern(p)
        if on:
            aff[i] = skip[p0 : p0 + len(pairs)].any(0)
        p0 += len(pairs)
    assert k == nm

    # ---- 2. the quantum state of each shot, exactly (stim tableau, C++)
    stim_seeds = rng.integers(2**63, size=shots)
    nblocks = len(prog.blocks)
    pad = nblocks * BLOCK - len(ev)
    block_aff = (
        np.concatenate([aff, np.zeros((pad, shots), bool)]).reshape(nblocks, BLOCK, shots).any(1)
    )
    ret_by_shot: dict[tuple[int, int], list] = {}
    for i, (qs, ss, xs) in ret.items():
        for q, s, x in zip(qs.tolist(), ss.tolist(), xs.tolist()):
            ret_by_shot.setdefault((i, s), []).append((q, x))
    raw = np.zeros((shots, nm), bool)
    for s in range(shots):
        sim = stim.TableauSimulator(seed=int(stim_seeds[s]))
        sim.set_num_qubits(n)
        baff = block_aff[:, s].tolist()
        for b in range(nblocks):
            if not baff[b]:
                sim.do_circuit(prog.blocks[b])
                continue
            lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(ev))
            col = aff[lo:hi, s].tolist()
            for i in range(lo, hi):
                if not col[i - lo]:
                    sim.do_circuit(prog.plain_step[i])
                    continue
                sim.do_circuit(prog.segments[i])
                e = ev[i]
                if e[0] == "TICK":
                    for q, x in ret_by_shot[(i, s)]:
                        sim.reset(q)
                        if x:
                            sim.x(q)
                    continue
                sl = pair_index[i]
                for j, (a, bq) in enumerate(e[1]):
                    if not skip[sl.start + j, s]:
                        sim.cz(a, bq)
                    for side, q in ((0, a), (1, bq)):
                        c = int(pa[sl.start + j, side, s])
                        if c == 1:
                            sim.x(q)
                        elif c == 2:
                            sim.z(q)
                        elif c == 3:
                            sim.y(q)
        sim.do_circuit(prog.segments[-1])
        raw[s] = sim.current_measurement_record()

    # ---- 3. readout errors and leaked readouts on the record, then detectors
    rec = raw.copy()
    readout = leakage.readout if leakage is not None else None
    for k in range(nm):
        q = prog.meas_q[k]
        if readout is not None and q in readout:
            p10, p01 = readout[q]
            flip = rng.random(shots) < np.where(raw[:, k], p01, p10)
        else:
            flip = rng.random(shots) < prog.meas_p[k]
        rec[:, k] ^= flip
        if on:
            lk = leaked_at_m[k]
            if lk.any():
                rec[:, k] = np.where(lk, rng.random(shots) < leakage.read_leaked, rec[:, k])
    flips = (rec ^ prog.reference[None, :]).astype(np.float32)
    det = (flips @ prog.det.T).astype(np.int64) & 1
    obs = (flips @ prog.obs.T).astype(np.int64) & 1
    if return_raw:
        return det.astype(bool), obs.astype(bool), rec
    return det.astype(bool), obs.astype(bool)


def _meas_per_segment(prog: Program) -> list[int]:
    """Number of measurement records in each segment (segments[i] precedes events[i])."""
    return [sum(len(ins.targets_copy()) for ins in seg if ins.name == "M") for seg in prog.segments]

"""Experiment 3: is the persistent flag/relay state leakage, and what drives it?

The flags, X ancillas and relays of the d=3 patch, no two of them coupled, each run N
cycles of one arm, every readout to its own bit:

    R0, R1  references: |0> then measure, and X then measure
    A       H, wait, H, measure, reset
    B       H, two CZs with each usual partner, H, measure, reset
    C       as A, with the reset replaced by a wait as long
    D       as A, with a wait of one more period at the end of each cycle

The CZ block fills six slots in every arm (waits outside B), so A, B and C share one period,

    t_c = 2 t_sx + 6 t_cz + t_measure + t_reset    (4.864 us on Kingston),

and D takes 2 t_c. CZ CZ = I, so every readout is ideally 0. The partners start in |+> and
take one X per cycle in every arm, which holds their |1> share at 1/2 against decay.

Readouts are kerneled, one IQ point each. Each qubit's references give two 2-D Gaussian
clouds, and a readout is 0, 1, or outside (Mahalanobis distance > 4 from both). A third
cloud is accepted when a mixture with the two reference clouds fixed gains Delta BIC > 10
from it and it sits beyond both. A two-state hidden Markov model per qubit and arm,

    P(persistent at cycle 1) = pi,  onset a,  exit b,  emissions over {0, 1, outside},

gives the lifetime tau = -t_c / ln(1 - b). In C the 0/1 symbol is the change from the
last 0/1 readout, a one-readout blip counted once. The causes are a_B - a_A (the CZs, also
per CZ), a_A - a_C (the reset) and a_C (readout and idle), with a = 0 in an arm where the
persistent state isn't present, and intervals from a bootstrap over shots; C fitted to its
readouts as read bounds the last two. A state that reads a steady 1 in A (other than 1 less
than MIN_CONTRAST of the time) makes no changes in C: then only those bounds split the reset
from readout and idle. Without IQ data the same model runs on IBM's 0/1 readouts, where a
state read outside the clouds most likely reads 1.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import exp, log, pi
from typing import TYPE_CHECKING

import numpy as np

from .placement import by_role, chip_map

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

    from heavyhex.patches.layout import HeavyHexLayout

CYCLES = 30
SLOTS = 6  # the CZ block: two CZs with each of up to three partners
ARMS = ("A", "B", "C", "D")
REFERENCES = ("R0", "R1")
SHOTS = {"R0": 1000, "R1": 1000, "A": 3000, "B": 3000, "C": 3000, "D": 3000}
JOBS = (("R0", "R1", "A"), ("R0", "R1", "B"), ("R0", "R1", "C"))  # ("R0", "R1", "D") optional
PERIODS = {"A": 1, "B": 1, "C": 1, "D": 2}  # the arm's cycle, in t_c
MEAS_TYPE = "kerneled"
REP_DELAY_US = 250
# IBM's billing rule, as run 1 was billed: a load per job, then the circuit and rep delay
# per shot, plus 3 us.
LOAD_S = 2.1
SHOT_OVERHEAD_US = 3
OUTSIDE = 4.0  # Mahalanobis distance from both clouds
MIN_DELTA_BIC = 10.0
TRIM = 3.0  # a reference cloud is fitted to its points within this distance
MAX_EXIT = 0.5  # the persistent state's exit per cycle, at most
MIN_CONTRAST = 0.25  # how often the persistent state reads other than 0, at least


@dataclass(frozen=True)
class Selection:
    """The qubits under test, their usual CZ partners, and the CZ block's slots."""

    qubits: tuple[int, ...]  # in bit order
    roles: tuple[str, ...]  # "flag", "X ancilla" or "relay"
    partners: tuple[tuple[int, ...], ...]
    slots: tuple[tuple[tuple[int, int], ...], ...]  # (qubit, partner) pairs in each slot

    @property
    def partner_qubits(self) -> tuple[int, ...]:
        """Every partner, once."""
        return tuple(sorted({p for ps in self.partners for p in ps}))

    @property
    def czs(self) -> tuple[int, ...]:
        """CZs per cycle in arm B, for each qubit."""
        return tuple(2 * len(ps) for ps in self.partners)


def select(layout: HeavyHexLayout, edges: Sequence[Sequence[int]] | None = None) -> Selection:
    """The patch's flags, X ancillas and relays, with the partners its circuit couples them to."""
    edges = chip_map()[1] if edges is None else edges
    roles = by_role(layout)
    flags = set(roles["flags"])
    qubits = tuple(sorted(roles["x_ancillas"] + roles["relays"]))
    bonds = {frozenset(edge) for edge in edges}
    coupled = [pair for pair in combinations(qubits, 2) if frozenset(pair) in bonds]
    if coupled:
        raise ValueError(f"qubits under test are coupled: {coupled}")
    partners = tuple(
        tuple(sorted({b if a == q else a for _, a, b in layout.couplings if q in (a, b)}))
        for q in qubits
    )
    pairs = [(q, p) for q, ps in zip(qubits, partners, strict=True) for p in ps]
    colours = _colour(pairs, SLOTS // 2)
    half = tuple(tuple(pair for pair in pairs if colours[pair] == c) for c in range(SLOTS // 2))
    return Selection(
        qubits=qubits,
        roles=tuple(
            "flag" if q in flags else "relay" if q in layout.relays else "X ancilla" for q in qubits
        ),
        partners=partners,
        slots=half + half,
    )


def _colour(pairs: list[tuple[int, int]], colours: int) -> dict[tuple[int, int], int]:
    """A slot for each pair so that no qubit is in two pairs of a slot (backtracking)."""
    found: dict[tuple[int, int], int] = {}

    def fill(k: int) -> bool:
        if k == len(pairs):
            return True
        used = {found[pair] for pair in pairs[:k] if set(pair) & set(pairs[k])}
        for c in range(colours):
            if c not in used:
                found[pairs[k]] = c
                if fill(k + 1):
                    return True
        found.pop(pairs[k], None)
        return False

    if not fill(0):
        raise ValueError(f"the CZ pairs don't fit in {2 * colours} slots")
    return found


@dataclass(frozen=True)
class Timing:
    """The layers of a cycle, in dt: the longest instance of each over the qubits used."""

    dt: float  # s
    pulse: int  # SX or X
    slot: int  # CZ
    measure: int
    reset: int

    @property
    def period(self) -> int:
        return 2 * self.pulse + SLOTS * self.slot + self.measure + self.reset

    @property
    def period_us(self) -> float:
        return self.period * self.dt * 1e6


def timing(selection: Selection, target: Target) -> Timing:
    """The cycle's layers from the target's gate times."""
    if target.granularity != 1 or target.pulse_alignment != 1:
        raise ValueError("delays would need aligning to the device's time grid")
    test, partners = selection.qubits, selection.partner_qubits
    return Timing(
        dt=target.dt,
        pulse=max(
            *(_length(target, "sx", q) for q in (*test, *partners)),
            *(_length(target, "x", q) for q in partners),
        ),
        slot=max(_length(target, "cz", *pair) for pairs in selection.slots for pair in pairs),
        measure=max(_length(target, "measure", q) for q in test),
        reset=max(_length(target, "reset", q) for q in test),
    )


def _length(target: Target, name: str, *qubits: int) -> int:
    properties = target[name].get(qubits) or target[name].get(qubits[::-1])
    if properties is None or properties.duration is None:
        raise ValueError(f"the target has no {name} on {qubits}")
    return round(properties.duration / target.dt)


def circuit(arm: str, selection: Selection, target: Target, cycles: int = CYCLES) -> QuantumCircuit:
    """One arm in the device's native gates on the whole chip, every wait a delay in dt.

    Register m holds every readout, cycle by cycle: bit cycle * n + i is qubit i's. A
    barrier closes each cycle, and every qubit's delays fill it to the same length.
    """
    from qiskit import ClassicalRegister, QuantumCircuit

    if arm not in REFERENCES + ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    test, partners = selection.qubits, selection.partner_qubits
    n = len(test)
    bits = ClassicalRegister(n if arm in REFERENCES else cycles * n, "m")
    qc = QuantumCircuit(target.num_qubits, name=f"leakage {arm}")
    qc.add_register(bits)
    if arm in REFERENCES:
        for i, q in enumerate(test):
            if arm == "R1":
                qc.x(q)
            qc.measure(q, bits[i])
        return qc

    t = timing(selection, target)
    everyone = (*test, *partners)
    idle: dict[int, int] = defaultdict(int)

    def wait(qubits: Sequence[int], length: int) -> None:
        for q in qubits:
            idle[q] += length

    def flush(*qubits: int) -> None:
        for q in qubits:
            if idle[q]:
                qc.delay(idle[q], q, unit="dt")
                idle[q] = 0

    def pulse(name: str, q: int) -> None:
        flush(q)
        if name == "h":
            qc.rz(pi / 2, q)
            qc.sx(q)
            qc.rz(pi / 2, q)
        else:
            qc.x(q)
        wait([q], t.pulse - _length(target, "x" if name == "x" else "sx", q))

    for cycle in range(cycles):
        for q in test:
            pulse("h", q)
        for q in partners:
            if cycle == 0:
                pulse("h", q)  # into |+>
            else:
                wait([q], t.pulse)
        for pairs in selection.slots:
            busy = set()
            if arm == "B":
                for q, p in pairs:
                    flush(q, p)
                    qc.cz(q, p)
                    wait([q, p], t.slot - _length(target, "cz", q, p))
                    busy |= {q, p}
            wait([q for q in everyone if q not in busy], t.slot)
        for q in test:
            pulse("h", q)
        for q in partners:
            pulse("x", q)
        for i, q in enumerate(test):
            flush(q)
            qc.measure(q, bits[cycle * n + i])
            wait([q], t.measure - _length(target, "measure", q))
            if arm == "C":
                wait([q], t.reset)
            else:
                flush(q)
                qc.reset(q)
                wait([q], t.reset - _length(target, "reset", q))
        wait(partners, t.measure + t.reset)
        if arm == "D":
            wait(everyone, t.period)
        flush(*everyone)
        qc.barrier(*everyone)
    return qc


def qpu_seconds(job: Sequence[str], t: Timing, cycles: int = CYCLES) -> float:
    """A job's QPU time by IBM's billing rule, with SHOTS shots of each circuit."""
    seconds = LOAD_S
    for arm in job:
        if arm in REFERENCES:
            length_us = (t.measure + t.pulse) * t.dt * 1e6
        else:
            length_us = cycles * PERIODS[arm] * t.period_us
        seconds += SHOTS[arm] * (length_us + REP_DELAY_US + SHOT_OVERHEAD_US) * 1e-6
    return seconds


# Readout clouds


def _xy(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(float)


@dataclass(frozen=True)
class Cloud:
    """A 2-D Gaussian in the IQ plane."""

    mean: np.ndarray  # (2,)
    cov: np.ndarray  # (2, 2)

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of points (..., 2)."""
        d = x - self.mean
        return np.sqrt(np.einsum("...i,ij,...j->...", d, np.linalg.inv(self.cov), d))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * self.distance(x) ** 2 - log(2 * pi) - 0.5 * log(np.linalg.det(self.cov))


def fit_cloud(z: np.ndarray, trim: float = TRIM) -> Cloud:
    """A Gaussian fitted to a reference's core.

    Points beyond `trim` (preparation errors, decay during readout) are left out, and the
    covariance is corrected for the cut: within distance r a 2-D Gaussian keeps
    c(r) = [2 - (r^2 + 2) e^(-r^2/2)] / [2 (1 - e^(-r^2/2))] of its variance.
    """
    x = _xy(np.ravel(z))
    s = trim**2
    kept = (2 - (s + 2) * exp(-s / 2)) / (2 * (1 - exp(-s / 2)))
    mean = np.median(x, axis=0)
    spread = 1.4826 * np.median(np.abs(x - mean), axis=0)
    cloud = Cloud(mean, np.diag(spread**2))
    for _ in range(50):
        core = x[cloud.distance(x) < trim]
        new = Cloud(core.mean(axis=0), np.cov(core.T) / kept)
        if np.allclose(new.mean, cloud.mean) and np.allclose(new.cov, cloud.cov):
            break
        cloud = new
    return cloud


@dataclass(frozen=True)
class Clouds:
    """A qubit's |0> and |1> readout clouds, from one job's references."""

    zero: Cloud
    one: Cloud

    @classmethod
    def fit(cls, z0: np.ndarray, z1: np.ndarray) -> Clouds:
        return cls(fit_cloud(z0), fit_cloud(z1))


def classify(z: np.ndarray, clouds: Clouds) -> np.ndarray:
    """0 or 1 by the likelier cloud, and 2 (outside) beyond OUTSIDE from both."""
    x = _xy(z)
    symbols = (clouds.one.logpdf(x) > clouds.zero.logpdf(x)).astype(np.int8)
    symbols[(clouds.zero.distance(x) > OUTSIDE) & (clouds.one.distance(x) > OUTSIDE)] = 2
    return symbols


def third_cluster(z: np.ndarray, clouds: Clouds, iters: int = 200) -> dict:
    """Is there a third cloud? A mixture with the reference clouds fixed, against one without.

    Delta BIC = 2 (ll_3 - ll_2) - 6 ln n: the third cloud adds a weight, a mean and a
    covariance (floored at a quarter of the references' narrowest spread). It is accepted
    with Delta BIC > MIN_DELTA_BIC and a mean beyond OUTSIDE from both clouds. Its place
    is given along the |0>-|1> axis (0 at |0>, 1 at |1>) and across it, in the same unit.
    """
    x = _xy(np.ravel(z))
    n = len(x)
    fixed = np.stack([clouds.zero.logpdf(x), clouds.one.logpdf(x)])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        r = _responsibilities(fixed, w)
        if np.allclose(r.mean(1), w, rtol=0, atol=1e-10):
            break
        w = r.mean(1)
    ll2 = float(_logsumexp(fixed + np.log(w)[:, None]).sum())
    outside = classify(z, clouds).ravel() == 2
    result = {"points": n, "outside": int(outside.sum()), "delta_bic": None, "accepted": False}
    if outside.sum() < 10:
        return result
    floor = 0.25 * min(
        np.linalg.eigvalsh(clouds.zero.cov)[0], np.linalg.eigvalsh(clouds.one.cov)[0]
    )
    third = Cloud(x[outside].mean(0), _floored(np.cov(x[outside].T), floor))
    w = np.array([*(1 - outside.mean()) * w, outside.mean()])
    ll3 = last = -np.inf
    for _ in range(iters):
        logs = np.vstack([fixed, third.logpdf(x)[None]])
        ll3 = float(_logsumexp(logs + np.log(w)[:, None]).sum())
        r = _responsibilities(logs, w)
        w = r.mean(1)
        mean = r[2] @ x / r[2].sum()
        d = x - mean
        third = Cloud(mean, _floored((r[2][:, None] * d).T @ d / r[2].sum(), floor))
        if ll3 - last < 1e-3:
            break
        last = ll3
    axis = clouds.one.mean - clouds.zero.mean
    offset = third.mean - clouds.zero.mean
    beyond = (
        clouds.zero.distance(third.mean) > OUTSIDE and clouds.one.distance(third.mean) > OUTSIDE
    )
    delta_bic = 2 * (ll3 - ll2) - 6 * log(n)
    return result | {
        "delta_bic": float(delta_bic),
        "accepted": bool(delta_bic > MIN_DELTA_BIC and beyond),
        "beyond": bool(beyond),
        "weight": float(w[2]),
        "mean": third.mean.tolist(),
        "along": float(offset @ axis / (axis @ axis)),
        "across": float((axis[0] * offset[1] - axis[1] * offset[0]) / (axis @ axis)),
    }


def _logsumexp(logs: np.ndarray) -> np.ndarray:
    top = logs.max(0)
    return top + np.log(np.exp(logs - top).sum(0))


def _responsibilities(logs: np.ndarray, w: np.ndarray) -> np.ndarray:
    weighted = logs + np.log(w)[:, None]
    return np.exp(weighted - _logsumexp(weighted))


def _floored(cov: np.ndarray, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(cov)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def changes(symbols: np.ndarray) -> np.ndarray:
    """Arm C's symbols: a 0/1 readout becomes 1 where it differs from the last 0/1 readout
    (0 before the first); outside stays 2.

    A one-readout blip, a misread most often, counts once: the change straight back after
    a counted change is 0. Counted twice, a misread would make two changes in a row, which
    the model takes for a short stay in the persistent state.
    """
    out = np.array(symbols, dtype=np.int8)
    last = np.zeros(len(out), dtype=np.int8)
    counted = np.zeros(len(out), dtype=bool)  # the last 0/1 readout was a counted change
    for t in range(out.shape[1]):
        inside = out[:, t] < 2
        value = out[inside, t].copy()
        change = (value != last[inside]) & ~counted[inside]
        out[inside, t] = change
        counted[inside], last[inside] = change, value
    return out


# The hidden Markov model


def fit_hmm(
    sequences: np.ndarray,
    symbols: int = 3,
    start: dict | None = None,
    iters: int = 2000,
    tol: float = 1e-7,
) -> dict:
    """Baum-Welch for the normal and persistent states on shots x cycles of symbols.

    The persistent state starts as the one less likely to read 0, and it more often stays
    than leaves: b is held at MAX_EXIT at most, so that a memoryless excursion (a + b = 1,
    which the data can't tell from a second emission of the normal state) isn't taken for
    one. Even so, fitted to shots without one, it fits their noise with a small a. It is
    present when it beats one state reading every cycle at the same rates by
    Delta BIC > MIN_DELTA_BIC (symbols + 2 more parameters) and reads other than 0 at
    least MIN_CONTRAST of the time; a long stay that reads like the normal state is a pair
    of changes some cycles apart (arm C), not a state the readouts see. llr is the gain
    over one state with its own rates in each cycle, as in run 1.

    Shots with the same sequence enter once, weighted by their number. Stops when the
    log-likelihood rises by less than tol, or after iters updates ("capped").
    """
    patterns, counts = np.unique(np.asarray(sequences, dtype=np.int8), axis=0, return_counts=True)
    return _baum_welch(patterns, counts.astype(float), symbols, start, iters, tol)


def _baum_welch(
    o: np.ndarray, w: np.ndarray, symbols: int, start: dict | None, iters: int, tol: float
) -> dict:
    keep = w > 0
    o, w = o[keep], w[keep]
    S, T = o.shape
    onehot = np.eye(symbols)[o].reshape(S * T, symbols)
    if start is None:
        p, a, b = 0.02, 0.01, 0.2
        rates = w @ onehot.reshape(S, T * symbols)
        rates = rates.reshape(T, symbols).sum(0) / (w.sum() * T)
        e = np.array([0.9 * rates + 0.1 / symbols, [0.2] + [0.8 / (symbols - 1)] * (symbols - 1)])
    else:
        p, a, b = start["pi"], start["a"], start["b"]
        e = np.array([start["e_N"], start["e_P"]])
    ll, last, capped, iterations = -np.inf, -np.inf, True, 0
    for _ in range(iters):
        iterations += 1
        A = np.array([[1 - a, a], [b, 1 - b]])
        B = e[:, o].transpose(1, 2, 0)  # S x T x 2
        alpha = np.empty((S, T, 2))
        c = np.empty((S, T))
        x = np.array([1 - p, p]) * B[:, 0]
        c[:, 0] = x.sum(1)
        alpha[:, 0] = x / c[:, :1]
        for t in range(1, T):
            x = (alpha[:, t - 1] @ A) * B[:, t]
            c[:, t] = x.sum(1)
            alpha[:, t] = x / c[:, t : t + 1]
        beta = np.ones((S, T, 2))
        for t in range(T - 2, -1, -1):
            beta[:, t] = ((B[:, t + 1] * beta[:, t + 1]) @ A.T) / c[:, t + 1 : t + 2]
        gamma = alpha * beta
        gamma /= gamma.sum(2, keepdims=True)
        ahead = B[:, 1:] * beta[:, 1:] / c[:, 1:, None]  # S x T-1 x 2
        ll = float(w @ np.log(c).sum(1))
        p = float(w @ gamma[:, 0, 1] / w.sum())
        stay = w @ gamma[:, :-1].sum(1)  # expected cycles in each state, but the last
        a = a * float(w @ (alpha[:, :-1, 0] * ahead[:, :, 1]).sum(1)) / stay[0]
        b = b * float(w @ (alpha[:, :-1, 1] * ahead[:, :, 0]).sum(1)) / stay[1]
        e = (gamma * w[:, None, None]).reshape(S * T, 2).T @ onehot
        e = np.clip(e / e.sum(1, keepdims=True), 1e-12, 1)
        e /= e.sum(1, keepdims=True)
        p, a = (float(np.clip(v, 1e-9, 1 - 1e-9)) for v in (p, a))
        b = float(np.clip(b, 1e-9, MAX_EXIT))
        if ll - last < tol:
            capped = False
            break
        last = ll
    counts = (w @ onehot.reshape(S, T * symbols)).reshape(T, symbols)
    llr = ll - _log_likelihood(counts)  # against independent rates per cycle
    llr_one = ll - _log_likelihood(counts.sum(0))  # against one rate for every cycle
    delta_bic = 2 * llr_one - (symbols + 2) * log(w.sum() * T)
    return {
        "pi": p,
        "a": a,
        "b": b,
        "e_N": e[0].tolist(),
        "e_P": e[1].tolist(),
        "llr": llr,
        "delta_bic": delta_bic,
        "present": bool(delta_bic > MIN_DELTA_BIC and e[1, 0] <= 1 - MIN_CONTRAST),
        "shots": int(w.sum()),
        "iterations": iterations,
        "capped": capped,
    }


def _log_likelihood(counts: np.ndarray) -> float:
    """Of counts under their own rates (each row's, for a 2-D array)."""
    rates = counts / counts.sum(-1, keepdims=True)
    return float((counts * np.log(np.where(rates > 0, rates, 1))).sum())


def bootstrap(
    sequences: np.ndarray, fit: dict, reps: int, seed: int = 0, symbols: int = 3
) -> dict[str, np.ndarray]:
    """The model refitted on shots drawn with replacement, from the fit's own parameters."""
    patterns, counts = np.unique(np.asarray(sequences, dtype=np.int8), axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    n = int(counts.sum())
    draws = defaultdict(list)
    for _ in range(reps):
        again = _baum_welch(
            patterns, rng.multinomial(n, counts / n).astype(float), symbols, fit, 300, 1e-4
        )
        for key in ("pi", "a", "b", "e_P", "present"):
            draws[key].append(again[key])
    return {key: np.array(values) for key, values in draws.items()}


def interval(values: np.ndarray) -> list[float]:
    """The central 95% of bootstrap draws."""
    return np.percentile(values, [2.5, 97.5]).tolist()


def lifetime_us(b: np.ndarray | float, period_us: float) -> np.ndarray:
    """tau = -t / ln(1 - b), for exit b per cycle of length t."""
    return -period_us / np.log1p(-np.asarray(b))


# The analysis


def analyze(
    jobs: Sequence[Mapping[str, np.ndarray]],
    selection: Selection,
    period_us: float,
    *,
    kerneled: bool = True,
    reps: int = 200,
    seed: int = 0,
) -> dict:
    """Each qubit's references, its third-cluster test and model in each arm, and the causes.

    Each job maps R0, R1 and its arm to what the chip returned for them: one column per
    bit, IQ points if kerneled, else 0/1. Each job's references classify its own arm.
    """
    n = len(selection.qubits)
    rows = []
    for i, (q, role, czs) in enumerate(
        zip(selection.qubits, selection.roles, selection.czs, strict=True)
    ):
        references, arms, onsets = [], {}, {}
        for job in jobs:
            z0, z1 = (np.asarray(job[name])[:, i] for name in REFERENCES)
            ran = [arm for arm in ARMS if arm in job]
            clouds = Clouds.fit(z0, z1) if kerneled else None
            references.append({"arms": ran} | _references(z0, z1, clouds))
            for arm in ran:
                readouts = np.asarray(job[arm])
                readouts = readouts.reshape(len(readouts), -1, n)[:, :, i]
                arms[arm], found = _arm(arm, readouts, clouds, period_us, reps, seed)
                onsets |= found
        rows.append(
            {
                "qubit": q,
                "role": role,
                "czs": czs,
                "references": references,
                "arms": arms,
                "steady": _steady(arms),
                "causes": causes(onsets, czs, steady=_steady(arms)),
            }
        )
    return {"period_us": period_us, "kerneled": kerneled, "reps": reps, "qubits": rows}


def _references(z0: np.ndarray, z1: np.ndarray, clouds: Clouds | None) -> dict:
    """How the references read: misreads, readouts outside, and the third-cluster test on
    them, which shows what readout alone puts beyond the clouds."""
    if clouds is None:
        return {"p1_given_0": float(np.mean(z0)), "p0_given_1": float(1 - np.mean(z1))}
    s0, s1 = classify(z0, clouds), classify(z1, clouds)
    return {
        "zero": {"mean": clouds.zero.mean.tolist(), "cov": clouds.zero.cov.tolist()},
        "one": {"mean": clouds.one.mean.tolist(), "cov": clouds.one.cov.tolist()},
        "p1_given_0": float((s0 == 1).mean()),
        "p0_given_1": float((s1 == 0).mean()),
        "outside_0": float((s0 == 2).mean()),
        "outside_1": float((s1 == 2).mean()),
        "third": third_cluster(np.concatenate([z0, z1]), clouds),
    }


def _arm(
    arm: str,
    readouts: np.ndarray,
    clouds: Clouds | None,
    period_us: float,
    reps: int,
    seed: int,
) -> tuple[dict, dict[str, tuple[float, np.ndarray]]]:
    """One qubit in one arm: its symbols, model and intervals; and its onsets with their draws.

    Arm C is also fitted to its readouts as read ("as_read"). A state that reads a steady 1
    is entered with one change and left with another, so the changes can't see it; as
    read, it is seen, together with every |1> the missing reset leaves, so that onset is
    an upper bound.
    """
    read = readouts.astype(np.int8) if clouds is None else classify(readouts, clouds)
    t = PERIODS[arm] * period_us
    out, onset = _model(changes(read) if arm == "C" else read, t, reps, seed, clouds is not None)
    onsets = {arm: onset}
    if clouds is not None:
        out["third"] = third_cluster(readouts, clouds)
    if arm == "C":
        out["as_read"], onsets["C as read"] = _model(read, t, reps, seed, clouds is not None)
    return out, onsets


def _model(
    sequences: np.ndarray, period_us: float, reps: int, seed: int, kerneled: bool
) -> tuple[dict, tuple[float, np.ndarray]]:
    """The model on one series, with intervals; and its onset and draws, 0 where absent."""
    symbols = 3 if kerneled else 2
    fit = fit_hmm(sequences, symbols)
    boot = bootstrap(sequences, fit, reps, seed, symbols)
    out = {
        "symbols": np.bincount(sequences.ravel(), minlength=symbols).tolist(),
        "first_cycle": float((sequences[:, 0] > 0).mean()),  # carry-over from the shot before
        "later_cycles": float((sequences[:, 1:] > 0).mean()),
        "hmm": fit
        | {
            "lifetime_us": float(lifetime_us(fit["b"], period_us)),
            "occupancy": fit["a"] / (fit["a"] + fit["b"]),
            "a_95": interval(boot["a"]),
            "b_95": interval(boot["b"]),
            "pi_95": interval(boot["pi"]),
            "lifetime_us_95": interval(lifetime_us(boot["b"], period_us)),
        },
    }
    if kerneled:
        out["hmm"]["outside_95"] = interval(boot["e_P"][:, 2])
    return out, (fit["a"] * fit["present"], boot["a"] * boot["present"])


def causes(
    onsets: Mapping[str, tuple[float, np.ndarray]], czs: int, *, steady: bool = False
) -> dict:
    """The onset split by cause, each as [value, 2.5%, 97.5%]: the CZs (a_B - a_A, and per
    CZ), the reset (a_A - a_C), readout and idle (a_C), and time (a_D - a_A). From arm C as
    read come bounds: the reset's part at least a_A - a_C', readout's and idle's at most a_C'.
    steady: arm C's changes can't see the state (_steady), so the bounds alone are given.

    onsets maps each arm to its onset and bootstrap draws, each 0 where the persistent
    state isn't present. The arms ran in separate jobs, so their draws pair up independently.
    """

    def part(arm: str, less: str | None = None) -> list[float]:
        value, draws = onsets[arm]
        if less is not None:
            value, draws = value - onsets[less][0], draws - onsets[less][1]
        return [float(value), *interval(draws)]

    found = {}
    if {"A", "B"} <= set(onsets):
        found["cz"] = part("B", "A")
        found["cz_each"] = [v / czs for v in found["cz"]]
    if {"A", "C"} <= set(onsets):
        if not steady:
            found["reset"] = part("A", "C")
        found["reset_at_least"] = part("A", "C as read")
    if "C" in onsets:
        if not steady:
            found["readout_idle"] = part("C")
        found["readout_idle_at_most"] = part("C as read")
    if {"A", "D"} <= set(onsets):
        found["time"] = part("D", "A")
    return found


def _steady(arms: Mapping[str, dict]) -> bool:
    """Arm A's persistent state reads 1 all but less than MIN_CONTRAST of the time, so arm C,
    which counts changes, can't see it: a |2> read inside the |1> cloud, or any state in 0/1
    readouts that reads 1."""
    hmm = arms["A"]["hmm"] if "A" in arms else None
    return bool(hmm and hmm["present"] and hmm["e_P"][1] > 1 - MIN_CONTRAST)


# A dry run on synthetic IQ data


@dataclass(frozen=True)
class Leak:
    """A planted persistent state: onset a and exit b per cycle, occupied at first with pi."""

    a: float = 0.0
    b: float = 0.15
    pi: float = 0.0


@dataclass(frozen=True)
class IQ:
    """Synthetic readout of one qubit: the clouds' centres and width, |1>'s decay tail, and
    the chance per readout of being in the other computational state."""

    zero: complex = 0j
    one: complex = 1 + 0j
    leaked: complex = 1.3 + 1.0j
    sigma: float = 0.2
    decay: float = 0.02  # |1> readouts that decay part-way to |0>
    flip: float = 0.005


def simulate(
    arm: str, shots: int, cycles: int, leak: Leak, iq: IQ, rng: np.random.Generator
) -> np.ndarray:
    """One qubit's IQ points, shots x cycles (shots for a reference).

    In A, B and D each readout is |1> with chance flip; in C a flip stays until the next one.
    The persistent state reads from the third cloud.
    """
    if arm in REFERENCES:
        state = (rng.random(shots) < iq.flip) ^ (arm == "R1")
        return _points(state.astype(np.int8), iq, rng)
    flips = rng.random((shots, cycles)) < iq.flip
    state = (np.cumsum(flips, axis=1) % 2 if arm == "C" else flips).astype(np.int8)
    leaked = np.empty((shots, cycles), dtype=bool)
    leaked[:, 0] = rng.random(shots) < leak.pi
    u = rng.random((shots, cycles))
    for t in range(1, cycles):
        leaked[:, t] = np.where(leaked[:, t - 1], u[:, t] >= leak.b, u[:, t] < leak.a)
    state[leaked] = 2
    return _points(state, iq, rng)


def _points(state: np.ndarray, iq: IQ, rng: np.random.Generator) -> np.ndarray:
    centre = np.array([iq.zero, iq.one, iq.leaked])[state]
    decayed = (state == 1) & (rng.random(state.shape) < iq.decay)
    centre[decayed] = iq.zero + rng.random(decayed.sum()) * (iq.one - iq.zero)
    noise = rng.normal(size=state.shape) + 1j * rng.normal(size=state.shape)
    return centre + iq.sigma * noise


def simulate_job(
    arm: str,
    leaks: Sequence[Leak],
    iq: IQ | None = None,
    *,
    cycles: int = CYCLES,
    shots: Mapping[str, int] = SHOTS,
    kerneled: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """A job as the chip returns it: R0, R1 and the arm, one column per bit, bit cycle * n + i
    for qubit i. Without kerneled, IBM's 0/1: the nearer of the two centres."""
    iq = iq or IQ()
    rng = np.random.default_rng(seed)
    job = {
        name: np.stack([simulate(name, shots[name], cycles, leak, iq, rng) for leak in leaks], -1)
        for name in (*REFERENCES, arm)
    }
    job[arm] = job[arm].reshape(shots[arm], -1)
    if kerneled:
        return job
    axis = iq.one - iq.zero
    return {
        name: (((z - iq.zero) * np.conj(axis)).real / abs(axis) ** 2 > 0.5).astype(np.int8)
        for name, z in job.items()
    }

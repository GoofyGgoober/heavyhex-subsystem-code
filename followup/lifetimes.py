"""Lifetimes on the d=5 footprint, against the ones IBM reports: job life-1, after exp1-1.

    Z idle   Exp 1's idle test, each CZ a wait as long, with the same pulses, XY4 and twirl,
             its data starting in |0> or |1> instead of |+> and read in Z; rounds 1, 2, 4, 8,
             16. Each data qubit's flips per round, as the data of a memory circuit see them
    T1       every qubit of the footprint at once: X, a wait t, readout
    T2       likewise, with Hahn's echo, as IBM measures the T2 it reports:
             SX, t/2, X, t/2, SX^dag, readout. It ends in |1>, where qiskit-experiments' ends
             in |0> (SX last): P(1) = (1 + e^(-t/T2))/2, the same T2

T1 and T2 wait six times, from 0.1 to 2 times the footprint's median on 13 October (223 and
121 us), with no decoupling but the echo's X. Every circuit runs SHOTS shots.

The fits, each with its standard error:

    T1, T2   P(1) = A e^(-t/T) + B, weighted by the binomial variance at the fit, the error
             widened by √(χ²/dof) above 1. Compared in the rate 1/T, whose error holds where
             T's doesn't: a T beyond the longest wait comes out too low for its error too
             often. T1's B gives the qubit's excited share at rest, π = (B - r)/(1 - 2r), r
             its readout error (IBM's)
    flips    P(1) after n rounds from |s>, M + D_s λ^n for both starts together: the flips per
             round, p↓ + p↑ = 1 - λ. Only their sum: the twirl's Paulis and XY4 keep a qubit
             in each state about half of each wait, so the memory's frame settles at
             (M - r)/(1 - 2r) ≈ 1/2 whatever the qubit's own split p↑/(p↓ + p↑), which is π

A readout that errs δ more often from |1> than from |0> reads either share about δ/2 low.

The in-situ T1 and T_q, which the model takes, come from this test and Exp 1's idle test
fitted together on the model's clock (analysis.in_situ).
"""

from __future__ import annotations

from math import inf, isfinite, pi, sqrt
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from .circuits import IDLE_DISTANCE, IDLE_ROUNDS, Setting, read_into

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

WAITS_US = {"T1": (20, 40, 75, 135, 245, 450), "T2": (12, 22, 40, 72, 132, 240)}
Z_IDLE = tuple(
    Setting("idle", "X", IDLE_DISTANCE, n, "Z", start) for start in "01" for n in IDLE_ROUNDS
)
SHOTS = 1024  # each circuit
MAX_T_US = 1e5  # the fits' bounds on T
SHOWN_US = 1e4  # a T beyond this, 20 times the longest wait, shows as ∞


def label(kind: str, wait_us: float) -> str:
    return f"{kind}, {wait_us:g} us"


def kind_of(name: str) -> str | None:
    """T1 or T2, for those circuits' labels; None for any other."""
    kind = name.split(",")[0]
    return kind if kind in WAITS_US else None


def circuit(kind: str, qubits: Sequence[int], wait_us: float, target: Target) -> QuantumCircuit:
    """T1's or T2's circuit at one wait, on every qubit at once, in the device's gates: qubit
    i read into bit i of m, every wait a delay in dt. SX^dag is rz(pi) SX rz(pi)."""
    from qiskit import ClassicalRegister, QuantumCircuit

    if target.granularity != 1:
        raise ValueError("delays would need aligning to the device's time grid")
    half = round(wait_us * 1e-6 / target.dt / 2)
    bits = ClassicalRegister(len(qubits), "m")
    qc = QuantumCircuit(target.num_qubits, name=label(kind, wait_us))
    qc.add_register(bits)
    if kind == "T1":
        qc.x(qubits)
        qc.delay(2 * half, qubits, unit="dt")
    elif kind == "T2":
        qc.sx(qubits)
        qc.delay(half, qubits, unit="dt")
        qc.x(qubits)
        qc.delay(half, qubits, unit="dt")
        qc.rz(pi, qubits)
        qc.sx(qubits)
        qc.rz(pi, qubits)
    else:
        raise ValueError(f"unknown lifetime {kind!r}")
    qc.measure(qubits, bits)
    return qc


def circuits(qubits: Sequence[int], target: Target) -> list[tuple[str, QuantumCircuit]]:
    """T1's circuits, then T2's, each with its label."""
    return [
        (label(kind, t), circuit(kind, qubits, t, target))
        for kind, waits in WAITS_US.items()
        for t in waits
    ]


def wait_us(circuit: QuantumCircuit, dt: float) -> float:
    """A T1 or T2 circuit's wait as sent: the delays of the qubit read into bit 0."""
    from .model import _ns

    qubit = read_into(circuit, "m")[0]
    return sum(
        _ns(item.operation, dt) / 1e3
        for item in circuit.data
        if item.operation.name == "delay" and circuit.find_bit(item.qubits[0]).index == qubit
    )


def prepared(items: Sequence[Mapping], data: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    """How often each circuit that should read as prepared does: T1 and T2 at their shortest
    waits read 1, the Z idle test's data their start after one round. Well above 1/2 unless
    a pulse or a bit went missing."""
    expected: dict[str, tuple[str, bool]] = {}
    for row in items:
        s = row["setting"]
        if row["label"] in (label(kind, waits[0]) for kind, waits in WAITS_US.items()):
            expected[row["label"]] = ("m", True)
        elif s and s["kind"] == "idle" and s["readout"] == "Z" and s["rounds"] == 1:
            expected[row["label"]] = (f"d{s['distance']}", s["start"] == "1")
    return {
        name: float((data[name][register] == value).mean())
        for name, (register, value) in expected.items()
        if name in data
    }


# Fits


def decay(
    t_us: Sequence[float] | np.ndarray, ones: np.ndarray, shots: float, guess_us: float
) -> dict:
    """T from P(1) = A e^(-t/T) + B, the readouts at each wait t reading 1 `ones` times of
    `shots`: least squares weighted by the binomial variance at the fit, twice, and the
    standard errors of T and B widened by √(χ²/dof) above 1. NaN if the fit fails."""
    from scipy.optimize import curve_fit

    t, p = np.asarray(t_us, float), np.asarray(ones, float) / shots

    def f(t: np.ndarray, a: float, tau: float, b: float) -> np.ndarray:
        return a * np.exp(-t / tau) + b

    guess = [p[0] - p[-1], min(max(guess_us, t[0]), MAX_T_US / 2), p[-1]]
    sigma = _binomial_sigma(p, shots)
    try:
        for _ in range(2):
            params, cov = curve_fit(
                f, t, p, p0=guess, sigma=sigma, absolute_sigma=True,
                bounds=([-2, 0.1, -1], [2, MAX_T_US, 2]), maxfev=20_000,
            )  # fmt: skip
            guess, sigma = params, _binomial_sigma(f(t, *params), shots)
    except (RuntimeError, ValueError):
        nan = float("nan")
        return {"t_us": nan, "sigma_us": nan, "amplitude": nan, "offset": nan,
                "sigma_offset": nan, "chi2_red": nan}  # fmt: skip
    chi2 = float((((p - f(t, *params)) / sigma) ** 2).sum())
    dof = len(t) - 3
    widen = sqrt(max(chi2 / dof, 1.0)) if dof > 0 else 1.0
    return {
        "t_us": float(params[1]),
        "sigma_us": float(sqrt(cov[1, 1]) * widen),
        "amplitude": float(params[0]),
        "offset": float(params[2]),
        "sigma_offset": float(sqrt(cov[2, 2]) * widen),
        "chi2_red": chi2 / dof if dof > 0 else float("nan"),
    }


def rate(t_us: float, sigma_us: float) -> list[float]:
    """1/T and its error in 1/us, from T ± σ: the rate's error holds where T's doesn't."""
    return [1 / t_us, sigma_us / t_us**2]


def excited(offset: float, sigma: float, readout: float) -> list[float]:
    """A qubit's excited share at rest from T1's B, π = (B - r)/(1 - 2r), r its readout
    error, and its error."""
    return [(offset - readout) / (1 - 2 * readout), sigma / (1 - 2 * readout)]


def product(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """a b and its error, for independent [value, σ]."""
    (x, u), (y, v) = a, b
    return [x * y, sqrt((x * v) ** 2 + (y * u) ** 2)]


def flips(
    rounds: Sequence[int],
    p0: np.ndarray,
    p1: np.ndarray,
    shots0: np.ndarray,
    shots1: np.ndarray,
    readout: float,
) -> dict:
    """One data qubit's flips per round from the Z idle test: P(1) after n rounds from |0>
    (p0) and from |1> (p1), fitted together to M + D_s λ^n, weighted by the binomial variance
    at the fit; then p↓ + p↑ = 1 - λ, and the share the memory's frame settles at,
    (M - r)/(1 - 2r), r the readout error: about 1/2 under the twirl and XY4, a check rather
    than the qubit's own. Each [value, σ], the σ widened by √(χ²/dof) above 1."""
    from scipy.optimize import curve_fit

    n = np.asarray(rounds, float)
    x = np.concatenate([n, n])
    one = np.repeat([0.0, 1.0], len(n))
    p = np.concatenate([p0, p1]).astype(float)
    shots = np.concatenate([shots0, shots1]).astype(float)

    def f(_: np.ndarray, lam: float, m: float, d0: float, d1: float) -> np.ndarray:
        return m + (d0 + (d1 - d0) * one) * lam**x

    contrast = (p1 - p0).astype(float)
    ratio = contrast[-1] / contrast[0] if contrast[0] > 0 and contrast[-1] > 0 else 0.5
    lam = min(max(ratio ** (1 / (n[-1] - n[0])), 0.5), 0.999)
    middle = float((p0[-1] + p1[-1]) / 2)
    guess = [lam, middle, p0[0] - middle, p1[0] - middle]
    sigma = _binomial_sigma(p, shots)
    try:
        for _ in range(2):
            params, cov = curve_fit(
                f, x, p, p0=guess, sigma=sigma, absolute_sigma=True,
                bounds=([0, -1, -2, -2], [1, 2, 2, 2]), maxfev=20_000,
            )  # fmt: skip
            guess, sigma = params, _binomial_sigma(f(x, *params), shots)
    except (RuntimeError, ValueError):
        nan = [float("nan")] * 2
        return {"lambda": nan, "flips": nan, "settled": nan, "chi2_red": nan[0]}
    chi2 = float((((p - f(x, *params)) / sigma) ** 2).sum())
    dof = len(p) - 4
    widen = sqrt(max(chi2 / dof, 1.0)) if dof > 0 else 1.0
    (lam, m), (s_lam, s_m) = params[:2], np.sqrt(np.diag(cov)[:2]) * widen
    return {
        "lambda": [float(lam), float(s_lam)],
        "flips": [float(1 - lam), float(s_lam)],
        "settled": [float((m - readout) / (1 - 2 * readout)), float(s_m / (1 - 2 * readout))],
        "chi2_red": chi2 / dof if dof > 0 else float("nan"),
    }


def _binomial_sigma(p: np.ndarray, shots: int | float | np.ndarray) -> np.ndarray:
    """The standard error of a share p of shots, kept off 0 at p = 0 or 1."""
    q = np.clip(p, 0.5 / shots, 1 - 0.5 / shots)
    return np.sqrt(q * (1 - q) / shots)


# The comparison


def table(rows: Sequence[Mapping]) -> str:
    """Each qubit's T1 and T2 in us against IBM's, then each data qubit in situ, then the
    medians of the ratios.

    T shows with its 1σ range from the rate, 1/(Γ ± σ), and Δ/σ is IBM's rate less ours in
    our σ: above 0 where ours is longer. π is the excited share at rest. In situ: the flips a
    round read in Z, those of them that go |0> -> |1> (π of them), the decay a round each
    memory's data see, and T1 and T_q apart, which the two idle tests barely tell apart."""
    lines = [
        f"{'qubit':>5} {'role':10} {'T1 IBM':>6} {'ours':^16} {'Δ/σ':>5}  "
        f"{'T2 IBM':>6} {'ours':^16} {'Δ/σ':>5}  {'π %':^14}"
    ]
    for row in rows:
        ibm, ours = row["ibm"], row["ours"]
        cells = [
            f"{ibm[f'{kind}_us']:6.0f} {_span(ours[f'rate_{kind}'])} "
            f"{_pull(ours[f'rate_{kind}'], ibm[f'{kind}_us'])}"
            for kind in ("t1", "t2")
        ]
        lines.append(
            f"{row['qubit']:>5} {row['role']:10} {'  '.join(cells)}  {_percent(ours['excited'])}"
        )
    situ = [row for row in rows if row["in_situ"]]
    if situ:
        lines += [
            "",
            f"{'qubit':>5} {'flips %':^14} {'↑ %':^14} {'decay Z %':^14} {'decay X %':^14} "
            f"{'T1 in situ':^16} {'T_q':^16}",
        ]
        for row in situ:
            s = row["in_situ"]
            cells = [_percent(s.get(key)) for key in ("flips", "up", "decay_z", "decay_x")]
            cells += [_span(s.get(key)) for key in ("rate_t1", "rate_t_q")]
            lines.append(f"{row['qubit']:>5} {' '.join(cells)}")
    lines.append("")
    for name, values in ratios(rows).items():
        if values:
            lines.append(f"{name}: median {median(values):.2f} over {len(values)} qubits")
    return "\n".join(lines)


def ratios(rows: Sequence[Mapping]) -> dict[str, list[float]]:
    """Each qubit's ours/IBM and in situ/IBM, for T1 and T2 (inf where no decay is seen), and
    the decay a round each memory's data see in situ against IBM's T1 and T2."""
    found: dict[str, list[float]] = {
        "T1 ours/IBM": [], "T2 ours/IBM": [], "T1 in situ/IBM": [], "T_q/T2 IBM": [],
        "decay Z in situ/IBM": [], "decay X in situ/IBM": [],
    }  # fmt: skip
    for row in rows:
        ibm, ours, situ = row["ibm"], row["ours"], row["in_situ"] or {}
        times = (
            ("T1 ours/IBM", ours["rate_t1"], ibm["t1_us"]),
            ("T2 ours/IBM", ours["rate_t2"], ibm["t2_us"]),
            ("T1 in situ/IBM", situ.get("rate_t1"), ibm["t1_us"]),
            ("T_q/T2 IBM", situ.get("rate_t_q"), ibm["t2_us"]),
        )
        for name, value, reference in times:
            if value is not None and value[0] == value[0] and reference > 0:  # not NaN
                found[name].append(1 / (value[0] * reference) if value[0] > 0 else inf)
        for memory in "ZX":
            value = situ.get(f"decay_{memory.lower()}")
            if value is not None and value[0] == value[0]:
                reference = situ[f"decay_{memory.lower()}_ibm"]
                found[f"decay {memory} in situ/IBM"].append(value[0] / reference)
    return found


def _shown(t_us: float) -> str:
    return f"{t_us:.0f}" if t_us < SHOWN_US else "∞"


def _span(rate: Sequence[float] | None) -> str:
    """T = 1/Γ and its 1σ range 1/(Γ ± σ), from Γ ± σ in 1/us, 16 wide."""
    if rate is None or rate[0] != rate[0]:
        return " " * 16
    g, s = rate
    t = _shown(1 / g if g > 0 else inf)
    if s != s:
        return f"{t:>6}{'':10}"
    low, high = (_shown(1 / x if x > 0 else inf) for x in (g + s, g - s))
    return f"{t:>6} {low:>4}-{high:<4}"


def _pull(rate: Sequence[float], t_us: float) -> str:
    """(1/T_IBM - Γ)/σ, 5 wide."""
    g, s = rate
    if not (s > 0 and isfinite(g)):
        return " " * 5
    return f"{max(min((1 / t_us - g) / s, 99.9), -99.9):+5.1f}"


def _percent(value: Sequence[float] | None) -> str:
    """[x, σ] in %, 14 wide."""
    if value is None or value[0] != value[0]:
        return " " * 14
    x, s = 100 * value[0], 100 * value[1]
    error = " " * 6 if s != s else f"±{s:5.2f}" if s < 100 else f"{'±∞':>6}"
    return f"{x:7.2f} {error}"

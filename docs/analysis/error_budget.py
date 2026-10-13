"""The model's error budget: how much of eps each error channel carries, and where the device's
excess could sit.

Each job's model (common.Job.model, at the job's measured f) has five channels:

  cz         the depolarizing error after each CZ
  sq         the depolarizing error after each SX and X, the decoupling pulses included
  readout    the flip of every measurement and every reset, both at the qubit's readout error
             (split into measure and reset where named)
  t1         the T1 part of every idle
  dephasing  the rest of every idle: pure dephasing, f times IBM's where decoupled

An idle's PAULI_CHANNEL_1(px, px, pz) has eigenvalues lz = 1 - 4 px = exp(-t/T1) and
lx = ly = 1 - 2 (px + pz) = exp(-t/T2). It is exactly the T1 channel (lz, lx = sqrt lz)
followed by a phase flip with 1 - 2 q = lx / sqrt lz, so the two parts switch off and scale
apart: a factor k on a part raises its eigenvalues to the power k, an idle k times as long.

Simulated shots are decoded with the full model's matching graph, as the device's shots were;
the settings with a channel off and the shortened idles are also decoded with their own graph
("own"). eps is fitted per curve both ways (fitting.py), from SHOTS shots per circuit.

  budget    eps with every subset S of the five channels on (32 settings, eps(none) = 0): each
            channel alone, the other four without it, and its Shapley share

                phi_c = sum over S not holding c of |S|! (4 - |S|)! / 5! [eps(S + c) - eps(S)],

            which adds up to eps (sum_c phi_c = eps(all)); the model's Lambda with each channel off.
  excess    each channel alone multiplied by k in GRID: the k that gives the device's eps (as
            error_scale.py does with one k on every channel: about 1.2 in X memory and 1.75 in
            Z) and the k that gives its mean bulk stabilizer detection rate. A quadratic in ln k
            through ln eps (error_scale.curve) up to the first eps above EPS_CAP; no k beyond the
            grid is extrapolated. One k for both distances: least squares in ln eps over the
            twenty curves of a basis that reach it within the grid (error_scale.joint's rule).
  idles     every idle shortened to the zero-duration-reset round (4.528 of 7.896 us; against
            no_reset.py's circuits translated again) and to Harper et al.'s 3.2 us round. Gates,
            readout and resets as run: this shortens the waits, it is not a parallel schedule.
  changes   what the model says would change eps and Lambda: these together with the pre-run
            rows of docs/figures/figure3.json, no_reset.json, uniform_scale.json and
            error_scale.json.

A parallel X/Z schedule is not simulated: the X ancillas are also the Z4 gauges' flags
(heavyhex.circuits.flagged), so measuring both kinds of gauge at once needs a new gadget.

    python docs/analysis/error_budget.py               # the summary from results/error_budget_runs.json
    python docs/analysis/error_budget.py --recompute   # simulations first, 25 min on 2 workers
                                                       # [--jobs 2026-10-04-r1 ...] (5 min a day)
                                                       # resumes from cache/
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from math import exp, factorial, log, sqrt
from multiprocessing import Pool

import numpy as np
import pymatching
import stim
from scipy.optimize import brentq, minimize_scalar

from heavyhex.experiment.circuits import ROUNDS

import fitting as F
from common import (
    CACHE,
    REPO,
    RESULTS,
    WORKERS,
    day_number,
    jobs,
    marginals,
    memory_index,
    read,
    result,
    write_json,
)
from effect_sizes import t_interval
from error_scale import EPS_CAP, bulk, curve, device_eps

CHANNELS = ("cz", "sq", "readout", "t1", "dephasing")
CHANNEL_OF = {  # the parts a setting scales, and their channels
    "cz": "cz",
    "sq": "sq",
    "measure": "readout",
    "reset": "readout",
    "t1": "t1",
    "dephasing": "dephasing",
}
FULL = "+".join(CHANNELS)  # every channel on: the model
NAMES = {
    "cz": "CZ",
    "sq": "single-qubit gates",
    "readout": "readout and reset",
    "t1": "T1 decay while idle",
    "dephasing": "dephasing while idle (at f)",
}
GRID = (1.25, 1.5, 2.0, 3.0, 5.0, 8.0)
LN_K = (log(0.5), log(max(GRID)))  # k is not extrapolated beyond the grid
HARPER_ROUND_US = 3.2  # Harper et al.'s reset-free round on a Heron r2
SHOTS = 20_000
LONGEST = max(ROUNDS)
FITS = ("binomial", "wls")
PLACEMENTS = (1, 2)
RUNS = RESULTS / "error_budget_runs.json"
CACHED = CACHE / "error_budget"  # one file per job and basis, so a stopped run resumes

# The pre-run rows of docs/figures/figure3.json (day 1, compiled circuits, f = 1) used below:
# (scenario, the scenario it is compared with).
FIGURE3 = (
    ("weak parts of both patches at the chip median", "as placed"),
    ("d = 3 on qubits of the d = 5 patch", "as placed"),
    ("every part at the chip median", "as placed"),
    ("CZ error halved", "every part at the chip median"),
    ("readout error halved", "every part at the chip median"),
    ("T1 and T2 doubled", "every part at the chip median"),
)
QUANTITIES = ("eps3", "eps5", "lambda")


# ----------------------------------------------------------------------------- settings
def label(on) -> str:
    on = set(on)
    return "+".join(c for c in CHANNELS if c in on) or "none"


def factors(on=CHANNELS, **scale: float) -> dict:
    """Each part's factor: 1 if its channel is on, 0 if off, unless scale names it or its channel."""
    return {p: float(scale.get(p, scale.get(c, c in on))) for p, c in CHANNEL_OF.items()}


def round_lengths() -> dict:
    nr = result("no_reset.json")["placement"]
    return {"as_run": nr["as_run"]["round_us"], "no_reset_time": nr["no_reset_time"]["round_us"]}


def settings(k_all: float) -> dict[str, dict]:
    """Every setting simulated, by label. k_all: error_scale.py's joint k for the basis."""
    out = {label(s): factors(s) for r in range(1, 6) for s in combinations(CHANNELS, r)}
    out["measure off"] = factors(measure=0.0)
    out["reset off"] = factors(reset=0.0)
    for c in CHANNELS:
        for k in GRID:
            out[f"{c} x{k:g}"] = factors(**{c: k})
    out["all x k_all"] = dict.fromkeys(CHANNEL_OF, k_all)
    lengths = round_lengths()
    for name, us in (
        ("zero-duration resets", lengths["no_reset_time"]),
        ("3.2 us", HARPER_ROUND_US),
    ):
        r = us / lengths["as_run"]
        out[f"idles x{r:.3f} ({name})"] = factors(t1=r, dephasing=r)
    return out


def own_decoded(labels) -> list[str]:
    """The settings also decoded with their own matching graph."""
    return [x for x in labels if x.startswith("idles")] + [
        label(set(CHANNELS) - {off}) for off in CHANNELS
    ]


# ----------------------------------------------------------------------------- simulation
def parsed(model: stim.Circuit) -> list[tuple[str, str, list[float]]]:
    """The model's instructions as (name, targets as text, arguments)."""
    out = []
    for ins in model:
        text, args = str(ins), ins.gate_args_copy()
        head = text.index(")") + 1 if args else len(ins.name)
        out.append((ins.name, text[head:].strip(), args))
    return out


def with_factors(items, f: dict) -> stim.Circuit:
    """The model with each part's errors times its factor (0 switches it off)."""
    lines = []
    for name, targets, args in items:
        if name == "DEPOLARIZE2":
            args = [min(f["cz"] * args[0], 15 / 16)]
        elif name == "DEPOLARIZE1":
            args = [min(f["sq"] * args[0], 0.75)]
        elif name == "M" and args:
            args = [min(f["measure"] * args[0], 0.5)]
        elif name == "X_ERROR":  # only after a reset in these models
            args = [min(f["reset"] * args[0], 0.5)]
        elif name == "PAULI_CHANNEL_1":
            px, py, pz = args
            assert px == py
            lz, lx = 1 - 4 * px, 1 - 2 * (px + pz)
            phase = lx / sqrt(lz)  # the pure-dephasing eigenvalue
            lz_k = lz ** f["t1"]
            lx_k = sqrt(lz_k) * phase ** f["dephasing"]
            px = (1 - lz_k) / 4
            args = [px, px, max((1 - lx_k) / 2 - px, 0.0)]
        elif args and name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            raise ValueError(f"unexpected channel {name}")
        head = f"{name}({','.join(map(repr, args))})" if args else name
        lines.append(f"{head} {targets}")
    return stim.Circuit("\n".join(lines))


def matching(circuit: stim.Circuit) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )


def simulate_part(task: tuple) -> dict:
    """Failures per round count and setting, and the bulk detection rates, for one job and basis."""
    name, basis, shots, k_all = task
    job = next(j for j in jobs() if j.name == name)
    plan = settings(k_all)
    own = set(own_decoded(plan))
    out: dict = {"job": name, "basis": basis, "failures": {}, "failures_own": {}, "rate": {}}
    for rounds in ROUNDS:
        idx = memory_index(basis, rounds)
        model, _, labels = job.model(idx)
        items = parsed(model)
        fixed = matching(model)
        patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
        if rounds == LONGEST:
            seen = job.analysis["observed"]["settings"][idx]
            rates = np.array(seen["detector_rates"])
            out["device_rate"] = {p: float(rates[bulk(labels, p, rounds)].mean()) for p in patches}
        for i, (key, f) in enumerate(plan.items()):
            noisy = with_factors(items, f)
            det, obs = noisy.compile_detector_sampler(seed=1000 * rounds + i).sample(
                shots, separate_observables=True
            )
            decoders = [("failures", fixed)] + (
                [("failures_own", matching(noisy))] if key in own else []
            )
            for where, m in decoders:
                failed = m.decode_batch(det).astype(bool) != obs
                for j, p in enumerate(patches):
                    out[where].setdefault(key, {}).setdefault(p, []).append(int(failed[:, j].sum()))
            if rounds == LONGEST:
                exact = marginals(noisy)
                out["rate"][key] = {
                    p: float(exact[bulk(labels, p, rounds)].mean()) for p in patches
                }
    return out


def simulate(names: list[str], shots: int, workers: int) -> None:
    CACHED.mkdir(parents=True, exist_ok=True)
    k_all = {
        b: result("error_scale.json")["eps"]["binomial"][b]["both distances"]["joint"]["k"]
        for b in "XZ"
    }
    tasks = [
        (n, b, shots, k_all[b])
        for n in names
        for b in "XZ"
        if not (CACHED / f"{n}_{b}.json").exists()
    ]
    t0 = time.time()
    with Pool(workers) as pool:
        for part in pool.imap_unordered(simulate_part, tasks):
            path = CACHED / f"{part['job']}_{part['basis']}.json"
            path.write_text(json.dumps({"shots": shots, **part}) + "\n")
            print(f"{part['job']} {part['basis']} ({time.time() - t0:.0f} s)", flush=True)
    found = [json.loads(p.read_text()) for p in sorted(CACHED.glob("*.json"))]
    runs: dict = {}
    for p in found:
        if p["shots"] == shots:
            runs.setdefault(p["job"], {})[p["basis"]] = {
                k: p[k] for k in ("failures", "failures_own", "rate", "device_rate")
            }
    RUNS.write_text(
        json.dumps(
            {
                "shots": shots,
                "rounds": list(ROUNDS),
                "k_all": k_all,
                "settings": {b: settings(k_all[b]) for b in "XZ"},
                "jobs": runs,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print("wrote", RUNS, f"({len(runs)} jobs)")


# ----------------------------------------------------------------------------- summary
def fitted(counts: dict, shots: int) -> dict:
    """eps of every setting's curve, both fits, and the WL shot-noise error."""
    keys = list(counts)
    k = np.array([counts[x] for x in keys], float)
    binomial = F.mle_batch(k, shots, np.array(ROUNDS, float))[0]
    out = {}
    for x, kx, b in zip(keys, k, binomial):
        if not kx.any():  # no logical error in any shot: Z memory with dephasing alone
            out[x] = {"binomial": 0.0, "wls": 0.0, "se": 0.0}
            continue
        w = F.wls(ROUNDS, kx / shots, shots)
        # under two round counts with 0 < P < 1/2 and none at 1/2: too few failures for WL
        few = w.used < 2 and kx.max() < shots / 2
        out[x] = {"binomial": float(b), "wls": float(b) if few else w.eps, "se": w.se_raw}
    return out


def shapley(eps: dict) -> dict:
    """Each channel's Shapley share of eps(all) from eps of every subset (eps(none) = 0)."""
    n = len(CHANNELS)
    value = lambda s: eps[label(s)] if s else 0.0  # noqa: E731
    out = {}
    for c in CHANNELS:
        rest = [x for x in CHANNELS if x != c]
        out[c] = sum(
            factorial(r)
            * factorial(n - r - 1)
            / factorial(n)
            * (value(set(s) | {c}) - value(set(s)))
            for r in range(n)
            for s in combinations(rest, r)
        )
    return out


def solve(coef: np.ndarray, target: float) -> float | None:
    """ln k where the quadratic reaches ln target on its rising branch, within LN_K."""
    f = lambda x: np.polyval(coef, x) - log(target)  # noqa: E731
    grid = np.linspace(*LN_K, 2001)
    rising = np.polyval(np.polyder(coef), grid) > 0
    for a, b, ra, rb in zip(grid[:-1], grid[1:], rising[:-1], rising[1:]):
        if ra and rb and np.sign(f(a)) != np.sign(f(b)):
            return brentq(f, a, b)
    return None


def scaled_curve(scales: np.ndarray, values: np.ndarray, se: np.ndarray):
    """curve() through the grid up to the first eps above EPS_CAP; None if under three points."""
    keep = np.cumprod(values < EPS_CAP).astype(bool)
    if keep.sum() < 3:
        return None, keep
    return curve(scales[keep], values[keep], se[keep]), keep


def per_curve(res: dict, per_rep: dict) -> list[dict]:
    shots = res["shots"]
    scales = np.array((1.0, *GRID))
    rows = []
    for name, by_basis in res["jobs"].items():
        job = f"ibm_fez-{name}"
        for basis, r in by_basis.items():
            for d in r["failures"][FULL]:
                dev = device_eps(per_rep, job, basis, d)
                eps = fitted({x: v[d] for x, v in r["failures"].items()}, shots)
                own = fitted({x: v[d] for x, v in r["failures_own"].items()}, shots)
                row = {
                    "job": name,
                    "rep": dev["key"],
                    "day": day_number(dev["key"][:10]),
                    "placement": int(dev["key"][-1]),
                    "basis": basis,
                    "d": int(d),
                    "device_eps": {fit: [dev[fit], dev[f"{fit}_se"]] for fit in FITS},
                    "eps": {fit: {x: e[fit] for x, e in eps.items()} for fit in FITS},
                    "eps_own_decoder": {fit: {x: e[fit] for x, e in own.items()} for fit in FITS},
                    "rate": {x: v[d] for x, v in r["rate"].items()},
                    "device_rate": r["device_rate"][d],
                }
                for fit in FITS:
                    e = row["eps"][fit]
                    phi = shapley(e)
                    row.setdefault("budget", {})[fit] = {
                        c: {
                            "only": e[c] / e[FULL],
                            "off": 1 - e[label(set(CHANNELS) - {c})] / e[FULL],
                            "shapley": phi[c] / e[FULL],
                        }
                        for c in CHANNELS
                    } | {
                        "measure (off)": 1 - e["measure off"] / e[FULL],
                        "reset (off)": 1 - e["reset off"] / e[FULL],
                        "sum of only": sum(e[c] for c in CHANNELS) / e[FULL],
                        "sum of off": sum(
                            1 - e[label(set(CHANNELS) - {c})] / e[FULL] for c in CHANNELS
                        ),
                    }
                    # one channel times k: the k that gives the device's eps
                    target, target_se = row["device_eps"][fit]
                    for c in CHANNELS:
                        keys = [FULL] + [f"{c} x{k:g}" for k in GRID]
                        values = np.array([e[x] for x in keys])
                        se = np.array([eps[x]["se"] for x in keys])
                        coef, keep = scaled_curve(scales, values, se)
                        out: dict = {"grid_points_used": int(keep.sum())}
                        if coef is not None:
                            out["ln_eps_quadratic_in_ln_k"] = coef.tolist()
                            x = solve(coef, target)
                            if x is not None:
                                elasticity = float(np.polyval(np.polyder(coef), x))
                                rel_sim = float(np.interp(x, np.log(scales), se / values))
                                sigma = sqrt((target_se / target) ** 2 + rel_sim**2)
                                out |= {
                                    "k": exp(x),
                                    "sigma_ln_k": sigma / elasticity,
                                    "sigma_ln_eps": sigma,
                                }
                        row.setdefault("excess", {}).setdefault(fit, {})[c] = out
                # one channel times k: the k that gives the device's bulk detection rate
                for c in CHANNELS:
                    rate = np.array(
                        [row["rate"][x] for x in [FULL] + [f"{c} x{k:g}" for k in GRID]]
                    )
                    x = solve(curve(scales, rate), row["device_rate"])
                    row.setdefault("excess_detection_rate", {})[c] = None if x is None else exp(x)
                rows.append(row)
    rows.sort(key=lambda r: (r["basis"], r["d"], r["placement"], r["day"]))
    return rows


def joint(rows: list[dict], fit: str, c: str) -> dict:
    """One k on channel c for the curves given that reach the device's eps within the grid:
    least squares in ln eps, its error from Delta chi2 = 1 (error_scale.joint's rule). The
    curves that do not reach it are counted; leaving them out pulls k down."""
    solved = [r for r in rows if "k" in r["excess"][fit][c]]
    out: dict = {
        "curves": len(rows),
        "beyond_the_grid": len(rows) - len(solved),
        "k_median_of_solved": float(np.median([r["excess"][fit][c]["k"] for r in solved]))
        if solved
        else None,
    }
    terms = [
        (
            np.array(r["excess"][fit][c]["ln_eps_quadratic_in_ln_k"]),
            log(r["device_eps"][fit][0]),
            r["excess"][fit][c]["sigma_ln_eps"],
        )
        for r in solved
    ]
    if len(terms) < 2:
        return out

    def chi2(x: float) -> float:
        return sum(((np.polyval(q, x) - t) / s) ** 2 for q, t, s in terms)

    best = minimize_scalar(chi2, bounds=LN_K, method="bounded", options={"xatol": 1e-8})
    x, c0, nu = best.x, best.fun, len(terms) - 1
    h = 1e-4
    curvature = (chi2(x + h) - 2 * c0 + chi2(x - h)) / h**2
    sigma = sqrt(2 / curvature) * sqrt(max(c0 / nu, 1.0))
    return out | {"k": exp(x), "ci68": [exp(x - sigma), exp(x + sigma)], "chi2": c0, "dof": nu}


def spread(values: list[float]) -> dict:
    v = np.array(values)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def budget(rows: list[dict], fit: str) -> dict:
    """Shares of eps per basis and distance: placement 1, placement 2 and all ten (days)."""
    out: dict = {}
    for basis in "XZ":
        for d in (3, 5):
            sel = [r for r in rows if r["basis"] == basis and r["d"] == d]
            groups = {f"placement {p}": [r for r in sel if r["placement"] == p] for p in PLACEMENTS}
            groups["all ten"] = sel
            blk = {}
            for g, mine in groups.items():
                b = [r["budget"][fit] for r in mine]
                blk[g] = {
                    "model_eps": spread([r["eps"][fit][FULL] for r in mine]),
                    "device_eps": spread([r["device_eps"][fit][0] for r in mine]),
                    **{
                        c: {k: spread([x[c][k] for x in b]) for k in ("only", "off", "shapley")}
                        for c in CHANNELS
                    },
                    **{
                        k: spread([x[k] for x in b])
                        for k in ("measure (off)", "reset (off)", "sum of only", "sum of off")
                    },
                }
            out[f"{basis} d{d}"] = blk
    return out


def lambdas(rows: list[dict], fit: str, keys: list[str], own: bool = False) -> dict:
    """The model's Lambda per placement for each setting: geometric mean over the five days."""
    where = "eps_own_decoder" if own else "eps"
    out: dict = {}
    for basis in "XZ":
        for p in PLACEMENTS:
            by_day: dict = {}
            for r in rows:
                if r["basis"] == basis and r["placement"] == p:
                    by_day.setdefault(r["day"], {})[r["d"]] = r[where][fit]
            blk = {}
            for k in keys:
                if k not in by_day[1][3]:
                    continue
                e3, e5 = (np.array([v[d][k] for v in by_day.values()]) for d in (3, 5))
                defined = (e3 > 0).all() and (e5 > 0).all()  # eps is 0 with dephasing alone in Z
                blk[k] = {
                    "lambda": float(np.exp(np.log(e3 / e5).mean())) if defined else None,
                    "eps3": float(e3.mean()),
                    "eps5": float(e5.mean()),
                    "lambda_by_day": (e3 / e5).tolist() if defined else None,
                }
            out[f"{basis} placement {p}"] = blk
    return out


def excess(rows: list[dict], fit: str) -> dict:
    """Per basis and channel: the one-channel k that gives the device's eps."""
    es = result("error_scale.json")["eps"][fit]
    out: dict = {}
    for basis in "XZ":
        mine = [r for r in rows if r["basis"] == basis]
        every = es[basis]["both distances"]["joint"]
        blk: dict = {"every channel (error_scale.py)": {k: every[k] for k in ("k", "ci68")}}
        for c in CHANNELS:
            blk[c] = {
                "both distances": joint(mine, fit, c),
                **{f"d{d}": joint([r for r in mine if r["d"] == d], fit, c) for d in (3, 5)},
            }
        out[basis] = blk
    return out


def excess_rate(rows: list[dict]) -> dict:
    """The one-channel k that gives the device's bulk detection rate (exact, no shot noise)."""
    es = result("error_scale.json")["detection_rate"]
    out: dict = {}
    for basis in "XZ":
        for d in (3, 5):
            mine = [r for r in rows if r["basis"] == basis and r["d"] == d]
            blk = {"every channel (error_scale.py)": es[basis][f"d{d}"]["all ten"]}
            for c in CHANNELS:
                ks = [r["excess_detection_rate"][c] for r in mine]
                solved = [k for k in ks if k is not None]
                blk[c] = (t_interval(np.log(solved)) if len(solved) > 1 else {}) | {
                    "beyond_the_grid": len(ks) - len(solved)
                }
            out[f"{basis} d{d}"] = blk
    return out


def check_k_all(rows: list[dict], fit: str) -> dict:
    """eps with every channel times k_all against error_scale.py's curve at that k."""
    es = {(r["job"], r["basis"], r["d"]): r for r in result("error_scale.json")["per_curve"]}
    k_all = json.loads(RUNS.read_text())["k_all"]
    ratios = []
    for r in rows:
        other = es[(r["job"], r["basis"], r["d"])]
        coef = np.array(other["ln_eps_quadratic_in_ln_k"][fit])
        ratios.append(r["eps"][fit]["all x k_all"] / exp(np.polyval(coef, log(k_all[r["basis"]]))))
    return {
        "k_all": k_all,
        "ratio_median": float(np.median(ratios)),
        "ratio_range": [min(ratios), max(ratios)],
    }


def check_stored(rows: list[dict], fit: str) -> dict:
    """The full model here against the predictions the analysis stored (200,000 shots)."""
    pred = result("fits.json")["predictions"]
    ratios = [
        r["eps"][fit][FULL]
        / pred[f"ibm_fez-{r['job']}"]["predicted_at_measured_dephasing"][r["basis"]][str(r["d"])][
            fit
        ]["eps"]
        for r in rows
    ]
    return {"ratio_median": float(np.median(ratios)), "ratio_range": [min(ratios), max(ratios)]}


def idles_against_no_reset(rows: list[dict], fit: str) -> dict:
    """Day 1, placement 1: idles shortened to the zero-reset round, against no_reset.py."""
    nr = result("no_reset.json")["placement"]
    key = next(k for k in rows[0]["eps"][fit] if "zero-duration resets" in k)
    out = {}
    for basis in "XZ":
        mine = {
            r["d"]: r for r in rows if r["basis"] == basis and r["day"] == 1 and r["placement"] == 1
        }
        out[basis] = {
            f"eps{d}": {
                "idles shortened / as run (here)": mine[d]["eps"][fit][key]
                / mine[d]["eps"][fit][FULL],
                "zero-duration resets / as run (no_reset.py)": nr["no_reset_time"][basis][f"eps{d}"]
                / nr["as_run"][basis][f"eps{d}"],
            }
            for d in (3, 5)
        }
    return out


def changes(lam: dict, lam_own: dict) -> list[dict]:
    """What the model says would change eps and Lambda, each against its own baseline.

    figure3.json's nested row has eps3 as the mean over the distance-3 sub-patches of the
    distance-5 patch.
    """
    fig3 = {r["scenario"]: r for r in read(REPO / "docs/figures/figure3.json")["rows"]}
    nr, us = result("no_reset.json"), result("uniform_scale.json")
    es = result("error_scale.json")["lambda_at_joint_k"]["binomial"]

    def at_f1(scenario: str, basis: str) -> dict:
        v = fig3[scenario][basis]["1.0"]
        return {
            q: v["eps3in_mean" if q == "eps3" and "eps3in_mean" in v else q][0] for q in QUANTITIES
        }

    def figure3(scenario: str, base: str) -> dict:
        return {
            "source": "docs/figures/figure3.json (pre-run, day 1, compiled circuits, f = 1)",
            "baseline": base,
            **{b: at_f1(scenario, b) | {"baseline": at_f1(base, b)} for b in "XZ"},
        }

    def no_reset(setting: str, source: str, base: str) -> dict:
        blk = nr[setting]
        return {
            "source": f"no_reset.json ({source})",
            "baseline": base,
            **{
                b: {q: blk["no_reset_time"][b][q] for q in QUANTITIES}
                | {"baseline": {q: blk["as_run"][b][q] for q in QUANTITIES}}
                for b in "XZ"
            },
        }

    def budget_row(key: str) -> dict:
        def one(source: dict) -> dict:
            return {
                f"placement {p}": {
                    b: {q: source[f"{b} placement {p}"][key][q] for q in QUANTITIES}
                    | {"baseline": {q: lam[f"{b} placement {p}"][FULL][q] for q in QUANTITIES}}
                    for b in "XZ"
                }
                for p in PLACEMENTS
            }

        return {
            "source": "error_budget.py (each job's model at its f; eps mean, Lambda geometric "
            "mean over the five days)",
            "baseline": "the job's model",
            **one(lam),
            "own decoder": one(lam_own),
        }

    rows = [{"change": s, **figure3(s, b)} for s, b in FIGURE3]
    rows += [
        {
            "change": "zero-duration resets (a stand-in for tracking measured ancillas in software)",
            **no_reset("placement", "circuits translated again, f = 1.06", "pair 1, day 1, as run"),
        },
        {
            "change": "zero-duration resets",
            **no_reset("uniform", "logical circuits, f = 1", "uniform chip at day 5's medians"),
        },
        {
            "change": "every error rate times s, the s at which Lambda = 1",
            "source": "uniform_scale.json (logical circuits, f = 1)",
            "baseline": "uniform chip at day 5's medians",
            **{b: {"s": us[b]["crosses_at"]} for b in "XZ"},
        },
        {
            "change": "every error times k, the k that gives the device's eps",
            "source": "error_scale.json (each job's model; Lambda geometric mean over the days)",
            "baseline": "the job's model",
            **{
                b: {"k": es[b]["k"]}
                | {
                    f"placement {p}": {
                        "lambda": es[b][f"placement {p}"]["model_at_k"],
                        "baseline": {"lambda": es[b][f"placement {p}"]["model_at_1"]},
                    }
                    for p in PLACEMENTS
                }
                for b in "XZ"
            },
        },
    ]
    for c in CHANNELS:
        rows.append({"change": f"{NAMES[c]} off", **budget_row(label(set(CHANNELS) - {c}))})
    for key in lam["X placement 1"]:
        if key.startswith("idles"):
            rows.append({"change": key.replace("idles", "every idle"), **budget_row(key)})
    return rows


def summarize() -> dict:
    res = json.loads(RUNS.read_text())
    fits = result("fits.json")
    rows = per_curve(res, fits["per_rep"])
    keys = [FULL] + [label(set(CHANNELS) - {c}) for c in CHANNELS] + list(CHANNELS)
    keys += ["measure off", "reset off"] + [
        k for k in rows[0]["eps"]["wls"] if k.startswith("idles")
    ]
    own_keys = [k for k in keys if k in rows[0]["eps_own_decoder"]["wls"]]
    out: dict = {
        "what": "eps of each job's model with error channels switched off, on alone or scaled; "
        "shares are of the full model's eps (only: eps with that channel alone; off: the drop "
        "with it off; shapley: its Shapley share, which adds to 1). Fixed decoder (the full "
        "model's graph) unless named.",
        "channels": NAMES,
        "grid": GRID,
        "shots": res["shots"],
        "jobs": len(res["jobs"]),
        "parallel_schedule": "not simulated: the X ancillas are also the Z4 gauges' flags, so "
        "measuring X and Z gauges at once needs a new gadget design. The shortened idles shorten "
        "the waits only.",
        "checks": {
            "full model against the stored predictions": {f: check_stored(rows, f) for f in FITS},
            "every channel times k_all against error_scale.py": {
                f: check_k_all(rows, f) for f in FITS
            },
            "idles shortened to the zero-reset round against no_reset.py (day 1, pair 1)": {
                f: idles_against_no_reset(rows, f) for f in FITS
            },
        },
        "budget": {f: budget(rows, f) for f in FITS},
        "lambda": {f: lambdas(rows, f, keys) for f in FITS},
        "lambda_own_decoder": {f: lambdas(rows, f, own_keys, own=True) for f in FITS},
        "excess": {f: excess(rows, f) for f in FITS},
        "excess_detection_rate": excess_rate(rows),
    }
    out["changes (binomial)"] = changes(
        out["lambda"]["binomial"], out["lambda_own_decoder"]["binomial"]
    )
    out["per_curve"] = [
        {k: v for k, v in r.items() if k not in ("eps", "eps_own_decoder", "rate")}
        | {"eps": {f: {k: r["eps"][f][k] for k in keys} for f in FITS}}
        for r in rows
    ]
    write_json("error_budget.json", out)
    return out


def report(out: dict) -> None:
    fit = "binomial"
    for name, check in out["checks"].items():
        print(f"{name}: {json.dumps(check[fit])}")
    print(f"\n% of the model's eps ({fit} fit, mean over the ten pair-days): alone / off / Shapley")
    for bd, blk in out["budget"][fit].items():
        b = blk["all ten"]
        print(
            f"  {bd} (model {100 * b['model_eps']['mean']:.1f}%): "
            + "; ".join(
                f"{c} {100 * b[c]['only']['mean']:.0f}/{100 * b[c]['off']['mean']:.0f}/"
                f"{100 * b[c]['shapley']['mean']:.0f}"
                for c in CHANNELS
            )
        )
    print("\nLambda (geometric mean over the days), eps3 and eps5 (mean):")
    for bp, blk in out["lambda"][fit].items():
        print(f"  {bp}")
        for k, v in blk.items():
            if k == FULL or k not in CHANNELS:
                print(
                    f"    {'model' if k == FULL else k:40s} {v['lambda']:.3f}  "
                    f"{100 * v['eps3']:5.2f}%  {100 * v['eps5']:5.2f}%"
                )
    print("\none channel times k for the device's eps (both distances; d3; d5), and for its")
    print("bulk detection rate (d3; d5):")
    for basis, blk in out["excess"][fit].items():
        print(f"  {basis}: every channel {blk['every channel (error_scale.py)']['k']:.2f}")
        for c in CHANNELS:
            ks = [blk[c][x] for x in ("both distances", "d3", "d5")]
            rate = [out["excess_detection_rate"][f"{basis} {d}"][c] for d in ("d3", "d5")]
            print(
                f"    {c:10s}"
                + "; ".join(
                    (f"{j['k']:.2f}" if "k" in j else "-")
                    + (f" ({j['beyond_the_grid']} beyond)" if j["beyond_the_grid"] else "")
                    for j in ks
                )
                + "  |  "
                + "; ".join(
                    (f"{r['ratio']:.2f}" if "ratio" in r else "-")
                    + (f" ({r['beyond_the_grid']} beyond)" if r["beyond_the_grid"] else "")
                    for r in rate
                )
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="simulate first (heavy)")
    ap.add_argument("--jobs", nargs="*", help="job names (default all fifteen), e.g. 2026-10-04-r1")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.recompute:
        simulate(a.jobs or [j.name for j in jobs()], a.shots, a.workers)
        if a.jobs:
            return
    report(summarize())


if __name__ == "__main__":
    main()

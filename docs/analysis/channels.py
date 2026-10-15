"""Each error channel's factor with widened intervals, one factor for both bases, and two
channels fitted together (Sec. where, Table budget, App. budget).

error_budget.py gives, per basis, distance and channel, the factor k on that channel alone that
gives the device's eps (one least-squares k over the ten curves) and its mean bulk stabilizer
detection rate (one k per curve). Here:

  factors     each k with a 95% interval widened for the scatter of the ten curves beyond their
              errors. For eps: sigma from Delta chi2 = 1, times sqrt(max(1, chi2/nu)) (Birge),
              and the Student t quantile with nu = n - 1. For the detection rate, which the model
              gives exactly, the t interval of the curves' ln k. Where more than half the curves
              need a k beyond the simulated grid (8), only that bound is given.
  one factor  whether one k on a channel fits several targets (basis, distance, eps or detection
              rate) at once: the inverse-variance mean of the targets' ln k and chi2 about it
              (T - 1 dof), and z of ln k_Z - ln k_X. The eps and detection-rate targets of one
              basis and distance come from the same shots; placements also share days/qubits.
              No joint covariance is estimated, so these p values are nominal diagnostic
              scores, not calibrated channel-selection tests in the current paper.
  two         two channels fitted together to both bases at both distances (four eps targets,
              two factors): dephasing while idle with CZ, and with T1 decay while idle.
              Dephasing cannot flip Z memory's logical value (switching it off moves Z eps by
              0.7% at the median, within shot noise), so Z memory fixes the first factor, from
              error_budget's one-channel grid, and
              X memory then fixes the dephasing factor b, from X-memory simulations on a grid of
              both factors (--recompute; b from 0, no pure dephasing, to 1, the job's f), fitted
              per curve by a polynomial quadratic in ln a and cubic in b and not extrapolated.
              Per target the ten curves enter with their own errors, Birge-widened by that
              target's scatter; chi2 has 4 - 2 dof. Also with the eight targets that add the
              detection rates.
  spread      what "plausible" can be measured against: how far IBM's own numbers move (the
              largest day-to-day ratio of the patches' geometric mean of each reported quantity,
              and the chip's 90th percentile over its median, per day), and the idle test's
              error on f.
  plausible   a factor at most the chip's 90th percentile over its median of the matching
              reported quantity (every qubit or coupler of the patches as bad as the chip's worst
              tenth): within, beyond (even the interval's lower end is above), or undecided.

    python docs/analysis/channels.py               # the summary from results/channels_runs.json
    python docs/analysis/channels.py --recompute   # the X-memory grid first: 7 min on 2 workers,
                                                   # 0.3 GB each; resumes from cache/
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from math import exp, log, sqrt
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from heavyhex.experiment.circuits import ROUNDS

import fitting as F
from common import CACHE, RESULTS, WORKERS, jobs, marginals, memory_index, read, result, write_json
from error_budget import (
    CHANNELS,
    FULL,
    GRID,
    LN_K,
    factors,
    matching,
    parsed,
    solve,
    with_factors,
)
from error_scale import EPS_CAP, bulk, curve

FIT = "binomial"
SHOTS = 20_000
LONGEST = max(ROUNDS)
PAIRS = {"cz": (3.0, 4.0, 5.0), "t1": (2.0, 3.0, 4.0)}  # the first channel's grid, with dephasing
DEPHASING = (0.0, 0.25, 0.5, 0.75, 1.0)  # 0: no pure dephasing at all
RUNS = RESULTS / "channels_runs.json"
CACHED = CACHE / "channels"
TOP = max(GRID)


# ----------------------------------------------------------------------------- simulation
def grid_settings() -> dict[str, dict]:
    return {
        f"{c} x{a:g}, dephasing x{b:g}": factors(**{c: a, "dephasing": b})
        for c, grid in PAIRS.items()
        for a in grid
        for b in DEPHASING
    }


def simulate_part(task: tuple) -> dict:
    """X memory of one job at the settings named: failures per round count, bulk rates."""
    name, shots, keys = task
    job = next(j for j in jobs() if j.name == name)
    plan = {k: v for k, v in grid_settings().items() if k in keys}
    out: dict = {"job": name, "failures": {}, "rate": {}}
    for rounds in ROUNDS:
        idx = memory_index("X", rounds)
        model, _, labels = job.model(idx)
        items, fixed = parsed(model), matching(model)
        patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
        for key, f in plan.items():
            noisy = with_factors(items, f)
            seed = zlib.crc32(f"{name} {key} {rounds}".encode())
            det, obs = noisy.compile_detector_sampler(seed=seed).sample(
                shots, separate_observables=True
            )
            failed = fixed.decode_batch(det).astype(bool) != obs
            for j, p in enumerate(patches):
                out["failures"].setdefault(key, {}).setdefault(p, []).append(
                    int(failed[:, j].sum())
                )
            if rounds == LONGEST:
                exact = marginals(noisy)
                out["rate"][key] = {
                    p: float(exact[bulk(labels, p, rounds)].mean()) for p in patches
                }
    return out


def simulate(names: list[str], shots: int, workers: int) -> None:
    """Every job at every grid setting not yet in cache/channels (one file per job)."""
    CACHED.mkdir(parents=True, exist_ok=True)
    plan = list(grid_settings())

    def cached(n: str) -> dict:
        path = CACHED / f"{n}.json"
        found = json.loads(path.read_text()) if path.exists() else {}
        if found.get("shots") != shots:
            return {"job": n, "shots": shots, "failures": {}, "rate": {}}
        return found

    tasks = [(n, shots, missing) for n in names if (missing := set(plan) - set(cached(n)["rate"]))]
    t0 = time.time()
    with Pool(workers) as pool:
        for part in pool.imap_unordered(simulate_part, tasks):
            old = cached(part["job"])
            for k in ("failures", "rate"):
                old[k] |= part[k]
            (CACHED / f"{part['job']}.json").write_text(json.dumps(old) + "\n")
            print(f"{part['job']} ({time.time() - t0:.0f} s)", flush=True)
    found = [cached(j.name) for j in jobs()]
    runs = {
        p["job"]: {k: {s: p[k][s] for s in plan} for k in ("failures", "rate")}
        for p in found
        if set(plan) <= set(p["rate"])
    }
    RUNS.write_text(
        json.dumps(
            {
                "shots": shots,
                "rounds": list(ROUNDS),
                "basis": "X",
                "pairs": PAIRS,
                "dephasing": DEPHASING,
                "decoder": "the full model's matching graph, as the device's shots",
                "jobs": runs,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print("wrote", RUNS, f"({len(runs)} jobs)")


# ----------------------------------------------------------------------------- one channel
def t95(nu: int) -> float:
    return float(stats.t.ppf(0.975, nu))


def eps_target(rows: list[dict], c: str) -> dict:
    """One k on channel c for the device's eps over the curves given (error_budget.joint's least
    squares), with its Birge-widened error and 95% t interval; a bound if most curves need more."""
    solved = [r for r in rows if "k" in r["excess"][FIT][c]]
    out: dict = {"curves": len(rows), "beyond_the_grid": len(rows) - len(solved)}
    if 2 * out["beyond_the_grid"] > len(rows) or len(solved) < 2:
        return out | {"bound": f"> {TOP:g}"}
    terms = [
        (
            np.array(r["excess"][FIT][c]["ln_eps_quadratic_in_ln_k"]),
            log(r["device_eps"][FIT][0]),
            r["excess"][FIT][c]["sigma_ln_eps"],
        )
        for r in solved
    ]

    def chi2(x: float) -> float:
        return sum(((np.polyval(q, x) - t) / s) ** 2 for q, t, s in terms)

    best = minimize_scalar(chi2, bounds=LN_K, method="bounded", options={"xatol": 1e-9})
    x, c0, nu = best.x, best.fun, len(terms) - 1
    h = 1e-4
    curvature = (chi2(x + h) - 2 * c0 + chi2(x - h)) / h**2
    birge = max(c0 / nu, 1.0)
    sigma = sqrt(2 / curvature) * sqrt(birge)
    return out | {
        "k": exp(x),
        "ci95": [exp(x - t95(nu) * sigma), exp(x + t95(nu) * sigma)],
        "sigma_ln_k": sigma,
        "chi2": c0,
        "dof": nu,
        "p_scatter": float(stats.chi2.sf(c0, nu)),
        "birge_factor": sqrt(birge),
    }


def rate_curves(runs: dict, c: str, basis: str) -> list[dict]:
    """Per curve: ln rate as a quadratic in ln k for channel c alone (error_budget's grid)."""
    scales = np.array((1.0, *GRID))
    keys = [FULL] + [f"{c} x{k:g}" for k in GRID]
    out = []
    for name, by_basis in runs["jobs"].items():
        r = by_basis[basis]
        for p in r["device_rate"]:
            rate = np.array([r["rate"][k][p] for k in keys])
            out.append(
                {"job": name, "d": int(p), "q": curve(scales, rate), "device": r["device_rate"][p]}
            )
    return out


def rate_target(curves: list[dict]) -> dict:
    """The t interval of the curves' ln k for the device's bulk detection rate."""
    ks = [solve(r["q"], r["device"]) for r in curves]
    solved = np.array([x for x in ks if x is not None])
    out: dict = {"curves": len(ks), "beyond_the_grid": len(ks) - len(solved)}
    if 2 * out["beyond_the_grid"] > len(ks) or len(solved) < 2:
        return out | {"bound": f"> {TOP:g}"}
    n = len(solved)
    m, se = float(solved.mean()), float(solved.std(ddof=1) / sqrt(n))
    return out | {
        "k": exp(m),
        "ci95": [exp(m - t95(n - 1) * se), exp(m + t95(n - 1) * se)],
        "sigma_ln_k": se,
        "dof": n - 1,
    }


def targets(eb: dict, ebr: dict) -> dict:
    """Every channel's factor, by basis, distance and quantity, and error_scale's for all."""
    rows = eb["per_curve"]
    es = result("error_scale.json")
    out: dict = {}
    for basis in "XZ":
        for d in (3, 5):
            sel = [r for r in rows if r["basis"] == basis and r["d"] == d]
            blk: dict = {}
            for c in CHANNELS:
                rc = [r for r in rate_curves(ebr, c, basis) if r["d"] == d]
                blk[c] = {"eps": eps_target(sel, c), "detection": rate_target(rc)}
            blk["all channels (error_scale.py)"] = {  # the curves' k, random-effects / t interval
                q: {"k": v["ratio"], "ci95": v["ci95"]}
                for q, v in (
                    ("eps", es["eps"][FIT][basis][f"d{d}"]["all ten"]),
                    ("detection", es["detection_rate"][basis][f"d{d}"]["all ten"]),
                )
            }
            out[f"{basis} d{d}"] = blk
    return out


def common_k(entries: dict) -> dict:
    """Diagonal-weight diagnostic; returned p is nominal, not covariance-calibrated."""
    bounds = [t for t, e in entries.items() if "k" not in e]
    if bounds:
        fitted = {t: e for t, e in entries.items() if "k" in e}
        top = max((e["ci95"][1] for e in fitted.values()), default=None)
        return {
            "beyond the grid": bounds,
            "verdict": (
                f"incompatible: {', '.join(bounds)} need more than {TOP:g}, the others' 95% "
                f"intervals end at {top:.2f}"
                if top is not None and top < TOP
                else "undecided"
            ),
        }
    x = np.array([log(e["k"]) for e in entries.values()])
    w = np.array([1 / e["sigma_ln_k"] ** 2 for e in entries.values()])
    mean = float((w * x).sum() / w.sum())
    chi2 = float((w * (x - mean) ** 2).sum())
    nu = len(x) - 1
    se = 1 / sqrt(w.sum())
    return {
        "k": exp(mean),
        "ci95": [exp(mean - 1.96 * se), exp(mean + 1.96 * se)],
        "chi2": chi2,
        "dof": nu,
        "p": float(stats.chi2.sf(chi2, nu)),
    }


def x_against_z(x: dict, z: dict) -> dict:
    if "k" not in z:
        return {"Z": z.get("bound"), "X ci95": x.get("ci95")}
    if "k" not in x:
        return {"X": x.get("bound")}
    diff = log(z["k"]) - log(x["k"])
    s = sqrt(x["sigma_ln_k"] ** 2 + z["sigma_ln_k"] ** 2)
    return {"Z / X": exp(diff), "z": diff / s, "p": float(2 * stats.norm.sf(abs(diff / s)))}


def one_factor(t: dict) -> dict:
    """Does one factor on a channel fit several targets at once?"""
    out: dict = {}
    for c in CHANNELS:

        def pick(*keys, c=c):
            return {f"{b} d{d} {q}": t[f"{b} d{d}"][c][q] for b, d, q in keys}

        both = [(b, d, q) for b in "XZ" for d in (3, 5) for q in ("eps", "detection")]
        out[c] = {
            "X: eps and detection, d3 and d5": common_k(pick(*[k for k in both if k[0] == "X"])),
            "Z: eps and detection, d3 and d5": common_k(pick(*[k for k in both if k[0] == "Z"])),
            "X and Z eps, d3 and d5": common_k(pick(*[k for k in both if k[2] == "eps"])),
            "all eight": common_k(pick(*both)),
            **{
                f"Z against X, {q}, d{d}": x_against_z(t[f"X d{d}"][c][q], t[f"Z d{d}"][c][q])
                for q in ("eps", "detection")
                for d in (3, 5)
            },
        }
    return out


# ----------------------------------------------------------------------------- two channels
POWERS = [(i, j) for i in range(3) for j in range(4)]  # (ln a)^i b^j: quadratic times cubic


def design(u, b) -> np.ndarray:
    u, b = np.atleast_1d(u), np.atleast_1d(b)
    return np.column_stack([u**i * b**j for i, j in POWERS])


def surface(u: np.ndarray, b: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """y as a polynomial in (ln a, b), quadratic in ln a and cubic in b, weighted when w is
    given, through the grid's points; and its largest residual, also in units of 1/w."""
    assert len(y) > len(POWERS), "too few grid points for the surface"
    A = design(u, b)
    W = np.ones_like(y) if w is None else w
    coef = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)[0]
    resid = A @ coef - y
    return coef, [float(np.abs(resid).max()), float(np.abs(resid * W).max())]


def on_surface(coef: np.ndarray, u: float, b: float) -> float:
    return float(design(u, b)[0] @ coef)


def x_curves(runs: dict, rows: list[dict], c: str) -> list[dict]:
    """Every X curve's eps and bulk rate on the two-channel grid, as surfaces in (ln a, b)."""
    shots = runs["shots"]
    dev = {(r["job"], r["d"]): r for r in rows if r["basis"] == "X"}
    keys = [(a, b) for a in PAIRS[c] for b in DEPHASING]
    out = []
    for name, r in runs["jobs"].items():
        for p in r["failures"][f"{c} x{keys[0][0]:g}, dephasing x{keys[0][1]:g}"]:
            k = np.array(
                [r["failures"][f"{c} x{a:g}, dephasing x{b:g}"][p] for a, b in keys], float
            )
            eps = F.mle_batch(k, shots, np.array(ROUNDS, float))[0]
            se = np.array([F.wls(ROUNDS, kk / shots, shots).se_raw for kk in k])
            rate = np.array([r["rate"][f"{c} x{a:g}, dephasing x{b:g}"][p] for a, b in keys])
            u = np.log([a for a, _ in keys])
            bb = np.array([b for _, b in keys], float)
            keep = eps < EPS_CAP
            ce, res_e = surface(u[keep], bb[keep], np.log(eps[keep]), (eps / se)[keep])
            cr, res_r = surface(u, bb, np.log(rate))
            row = dev[(name, int(p))]
            target, target_se = row["device_eps"][FIT]
            out.append(
                {
                    "job": name,
                    "d": int(p),
                    "eps": ce,
                    "rate": cr,
                    "y_eps": log(target),
                    "sigma": sqrt((target_se / target) ** 2 + float(np.median(se / eps)) ** 2),
                    "y_rate": log(row["device_rate"]),
                    "points_used": int(keep.sum()),
                    "largest_residual_ln": [res_e, res_r],
                }
            )
    return out


def z_curves(rows: list[dict], ebr: dict, c: str) -> list[dict]:
    """Every Z curve's eps and bulk rate against channel c alone (error_budget's grid)."""
    rates = {(r["job"], r["d"]): r for r in rate_curves(ebr, c, "Z")}
    out = []
    for r in rows:
        if r["basis"] != "Z":
            continue
        e = r["excess"][FIT][c]
        rc = rates[(r["job"], r["d"])]
        out.append(
            {
                "job": r["job"],
                "d": r["d"],
                "eps": np.array(e["ln_eps_quadratic_in_ln_k"]) if "k" in e else None,
                "y_eps": log(r["device_eps"][FIT][0]),
                "sigma": e.get("sigma_ln_eps"),
                "rate": rc["q"],
                "y_rate": log(rc["device"]),
            }
        )
    return out


def target_chi2(resid: np.ndarray, sigma: np.ndarray | None) -> float:
    """sum (r/sigma)^2 with the target's scatter about its mean folded in: Birge-widened by
    max(1, S/(n-1)) where sigma is given, scaled by S/(n-1) for the exact detection rates."""
    n = len(resid)
    w = np.ones(n) if sigma is None else 1 / sigma**2
    mean = (w * resid).sum() / w.sum()
    scatter = (w * (resid - mean) ** 2).sum() / (n - 1)
    scale = scatter if sigma is None else max(scatter, 1.0)
    return float(w.sum() * mean**2 / scale)


def two_channels(c: str, xs: list[dict], zs: list[dict], with_rates: bool, ds=(3, 5)) -> dict:
    """a on channel c and b on dephasing, fitted to Z and X at the distances given."""
    zs = [z for z in zs if z["eps"] is not None]

    def chi2(theta) -> float:
        u, b = theta
        total = 0.0
        for d in ds:
            zd = [z for z in zs if z["d"] == d]
            xd = [x for x in xs if x["d"] == d]
            total += target_chi2(
                np.array([np.polyval(z["eps"], u) - z["y_eps"] for z in zd]),
                np.array([z["sigma"] for z in zd]),
            )
            total += target_chi2(
                np.array([on_surface(x["eps"], u, b) - x["y_eps"] for x in xd]),
                np.array([x["sigma"] for x in xd]),
            )
            if with_rates:
                total += target_chi2(
                    np.array([np.polyval(z["rate"], u) - z["y_rate"] for z in zd]), None
                )
                total += target_chi2(
                    np.array([on_surface(x["rate"], u, b) - x["y_rate"] for x in xd]), None
                )
        return total

    # only inside the simulated box: the X surfaces are not extrapolated
    box_u = (log(min(PAIRS[c])), log(max(PAIRS[c])))
    box_b = (min(DEPHASING), max(DEPHASING))
    starts = [(u0, b0) for u0 in np.linspace(*box_u, 4) for b0 in np.linspace(*box_b, 5)]
    best = min(
        (minimize(chi2, s, method="L-BFGS-B", bounds=[box_u, box_b]) for s in starts),
        key=lambda m: m.fun,
    )
    (u, b), c0 = best.x, float(best.fun)
    nu = (4 if with_rates else 2) * len(ds) - 2
    at_edge = [
        name
        for name, x, (lo, hi) in (("first", u, box_u), ("dephasing", b, box_b))
        if min(x - lo, hi - x) < 1e-6
    ]
    out = {
        "targets": (4 if with_rates else 2) * len(ds),
        "first channel factor": exp(u),
        "dephasing factor": float(b),
        "chi2": c0,
        "dof": nu,
        "p": float(stats.chi2.sf(c0, nu)) if nu > 0 else None,
        "at the edge of the simulated box": at_edge,
    }
    if nu > 0:  # 95% profile intervals inside the box, widened by chi2/nu when it exceeds one
        level = stats.chi2.ppf(0.95, 1) * max(1.0, c0 / nu)

        def prof_u(x: float) -> float:
            return minimize_scalar(lambda bb: chi2((x, bb)), bounds=box_b, method="bounded").fun

        def prof_b(x: float) -> float:
            return minimize_scalar(lambda uu: chi2((uu, x)), bounds=box_u, method="bounded").fun

        def edges(prof, lo, hi):
            inside = [x for x in np.linspace(lo, hi, 201) if prof(x) - c0 <= level]
            return [min(inside), max(inside)] if inside else None

        iu = edges(prof_u, *box_u)
        out["first channel factor ci95 (within the box)"] = [exp(iu[0]), exp(iu[1])] if iu else None
        out["dephasing factor ci95 (within the box)"] = edges(prof_b, *box_b)
    return out


def x_at_z(c: str, xs: list[dict], zs: list[dict]) -> dict:
    """X memory's eps, model over device (geometric mean over the ten curves), with channel c at
    the factor Z memory's eps needs at that distance and dephasing at 0 or 1."""
    out = {}
    for d in (3, 5):
        zd = [z for z in zs if z["d"] == d and z["eps"] is not None]

        def chi2(x: float, zd=zd) -> float:
            return sum(((np.polyval(z["eps"], x) - z["y_eps"]) / z["sigma"]) ** 2 for z in zd)

        u = minimize_scalar(chi2, bounds=LN_K, method="bounded").x
        xd = [x for x in xs if x["d"] == d]
        inside = log(min(PAIRS[c])) <= u <= log(max(PAIRS[c]))
        out[f"d{d}"] = {
            "Z's factor": exp(u),
            **{
                f"X model / device at dephasing x{b:g}": (
                    float(np.exp(np.mean([on_surface(x["eps"], u, b) - x["y_eps"] for x in xd])))
                    if inside
                    else None
                )
                for b in (0.0, 1.0)
            },
        }
    return out


def pair_fits(eb: dict, ebr: dict, runs: dict) -> dict:
    rows = eb["per_curve"]
    out: dict = {}
    for c in PAIRS:
        xs, zs = x_curves(runs, rows, c), z_curves(rows, ebr, c)
        out[f"{c} + dephasing"] = {
            "eps, d3 and d5": two_channels(c, xs, zs, False),
            "eps, d3 only": two_channels(c, xs, zs, False, (3,)),
            "eps, d5 only": two_channels(c, xs, zs, False, (5,)),
            "eps and detection, d3 and d5": two_channels(c, xs, zs, True),
            "X eps with the factor Z eps needs": x_at_z(c, xs, zs),
            "surfaces": {
                "largest residual in ln eps": max(x["largest_residual_ln"][0][0] for x in xs),
                "largest residual in ln eps over its shot-noise error": max(
                    x["largest_residual_ln"][0][1] for x in xs
                ),
                "largest residual in ln rate": max(x["largest_residual_ln"][1][0] for x in xs),
                "fewest grid points under the eps cap": min(x["points_used"] for x in xs),
            },
        }
    return out


# ----------------------------------------------------------------------------- spread
def ibm_spread() -> dict:
    """How far IBM's own numbers move: per reported quantity (errors, and rates 1/T1, 1/T2)."""
    cals: dict = {}
    used: set[int] = set()
    for j in jobs():
        cals.setdefault(j.day, read(j.folder / "calibration.json"))
        used |= {q for qs in j.info["chips"].values() for q in qs}
    first = next(iter(cals.values()))
    couplers = [k for k in first["cz"] if all(int(q) in used for q in k.split("-"))]

    def table(cal: dict, kind: str) -> dict:
        if kind == "cz":
            return cal["cz"]
        if kind == "sq":
            return cal.get("single_qubit") or {}
        if kind == "readout":
            return cal["readout"]
        return {q: 1 / v for q, v in cal[f"{kind}_us"].items() if v}

    out = {}
    for kind, name in (
        ("cz", "CZ error"),
        ("sq", "single-qubit error"),
        ("readout", "readout error"),
        ("t1", "1/T1"),
        ("t2", "1/T2"),
    ):
        keys = couplers if kind == "cz" else [str(q) for q in sorted(used)]
        rows = []
        for k in keys:
            v = [table(c, kind).get(k) for c in cals.values()]
            if all(x is not None and x > 0 and (kind in ("t1", "t2") or x < 0.5) for x in v):
                rows.append(np.log(v))
        m = np.array(rows)
        gm = np.exp(m.mean(axis=0))
        chip = []
        for c in cals.values():
            xs = np.array(
                [x for x in table(c, kind).values() if x and (kind in ("t1", "t2") or x < 0.5)]
            )
            chip.append(float(np.percentile(xs, 90) / np.median(xs)))
        out[name] = {
            "components": len(rows),
            "patches' geometric mean, largest day-to-day ratio": float(gm.max() / gm.min()),
            "one component, 95th percentile of its day / its five-day mean": float(
                np.exp(np.percentile(np.abs(m - m.mean(axis=1, keepdims=True)), 95))
            ),
            "chip 90th percentile / median, range over the days": [min(chip), max(chip)],
        }
    return out


def idle_test() -> dict:
    """The idle test's f per job and its statistical error (f_models.json)."""
    per = result("f_models.json")["sigma_f"]["per_job"]
    f = np.array([v["f"] for v in per.values()])
    s = np.array([v["idle_noise_only (bootstrap, ddof 1)"] for v in per.values()])
    rel = s / np.abs(f)
    return {
        "f median over the jobs": float(np.median(f)),
        "f range": [float(f.min()), float(f.max())],
        "relative error of a job's f (bootstrap), range": [float(rel.min()), float(rel.max())],
        "relative error of a factor common to the 15 jobs": float(1 / sqrt((1 / rel**2).sum())),
    }


# ----------------------------------------------------------------------------- summary
def summarize() -> dict:
    eb = result("error_budget.json")
    ebr = read(RESULTS / "error_budget_runs.json")
    runs = read(RUNS)
    rows = eb["per_curve"]
    without = {
        b: [
            r["eps"][FIT]["+".join(x for x in CHANNELS if x != "dephasing")] / r["eps"][FIT][FULL]
            for r in rows
            if r["basis"] == b
        ]
        for b in "XZ"
    }
    t = targets(eb, ebr)
    out = {
        "what": "error_budget.py's one-channel factors with Birge-widened 95% intervals, whether "
        "one factor on a channel fits several targets, two channels fitted together, and what "
        "'plausible' can be measured against. Binomial fit; fixed decoder (the full model's "
        "graph), as the device's shots.",
        "intervals": "eps: sigma from Delta chi2 = 1 times sqrt(max(1, chi2/nu)), Student t with "
        "nu = n - 1; detection rate (exact in the model): t interval of the curves' ln k. "
        f"'> {TOP:g}': more than half the curves need a factor beyond the simulated grid.",
        "checks": {
            "Z eps without dephasing / with (median, range)": [
                float(np.median(without["Z"])),
                min(without["Z"]),
                max(without["Z"]),
            ],
            "X eps without dephasing / with (median, range)": [
                float(np.median(without["X"])),
                min(without["X"]),
                max(without["X"]),
            ],
        },
        "factors": t,
        "one factor": one_factor(t),
        "two channels": pair_fits(eb, ebr, runs),
    }
    spread = ibm_spread()
    out["spread"] = {"IBM's reported values": spread, "idle test f": idle_test()}
    out["plausible"] = plausible(t, spread)
    out["grid"] = {"shots": runs["shots"], "pairs": PAIRS, "dephasing": DEPHASING}
    write_json("channels.json", out)
    return out


SPREAD_OF = {  # the reported quantity whose spread bounds each channel's factor
    "cz": "CZ error",
    "sq": "single-qubit error",
    "readout": "readout error",
    "t1": "1/T1",
    "dephasing": "1/T2",
}


def plausible(t: dict, spread: dict) -> dict:
    """A factor is plausible if it makes the patches' errors of that kind no worse than the
    chip's own spread allows: at most the chip's 90th percentile over its median of the reported
    quantity, on the day where that ratio is largest. Beyond: even the 95% interval's lower end
    is above it (or the factor is past the grid's 8); within: the factor itself is at most it;
    otherwise undecided."""
    out: dict = {
        "definition": "factor <= chip 90th percentile / median of the reported quantity (largest "
        "over the five days): every qubit or coupler of the patches as bad as the chip's worst "
        "tenth",
        "bound": {
            c: spread[q]["chip 90th percentile / median, range over the days"][1]
            for c, q in SPREAD_OF.items()
        },
    }
    for bd, blk in t.items():
        for c in CHANNELS:
            bound = out["bound"][c]
            for q in ("eps", "detection"):
                e = blk[c][q]
                if "k" not in e or e["ci95"][0] > bound:
                    verdict = "beyond"
                elif e["k"] <= bound:
                    verdict = "within"
                else:
                    verdict = "undecided"
                out.setdefault(bd, {}).setdefault(c, {})[q] = verdict
    return out


def fmt(e: dict) -> str:
    if "k" not in e:
        return f"{e.get('bound', '-')} ({e['beyond_the_grid']}/{e['curves']} beyond)"
    lo, hi = e["ci95"]
    more = f", {e['beyond_the_grid']} beyond" if e.get("beyond_the_grid") else ""
    return f"{e['k']:.2f} ({lo:.2f}-{hi:.2f}{more})"


def report(out: dict) -> None:
    print("checks:", json.dumps(out["checks"]))
    print("\nfactor on one channel (95%, widened): eps | detection rate")
    for bd, blk in out["factors"].items():
        print(f"  {bd}")
        for c in CHANNELS:
            print(f"    {c:10s} {fmt(blk[c]['eps']):28s} | {fmt(blk[c]['detection'])}")
    print("\none factor for several targets:")
    for c, blk in out["one factor"].items():
        print(f"  {c}")
        for k, v in blk.items():
            if "chi2" in v:
                print(
                    f"    {k:36s} k {v['k']:.2f}, chi2 {v['chi2']:.1f}/{v['dof']}, p {v['p']:.2g}"
                )
            elif "z" in v:
                print(f"    {k:36s} Z/X {v['Z / X']:.2f}, z {v['z']:.1f}, p {v['p']:.2g}")
            else:
                print(f"    {k:36s} {json.dumps(v)}")
    print("\ntwo channels:")
    for pair, blk in out["two channels"].items():
        print(f"  {pair}: {json.dumps(blk['surfaces'])}")
        print(
            f"    X eps with the factor Z eps needs: {json.dumps(blk['X eps with the factor Z eps needs'])}"
        )
        for k, v in blk.items():
            if k in ("surfaces", "X eps with the factor Z eps needs"):
                continue
            ci_a = v.get("first channel factor ci95 (within the box)")
            ci_b = v.get("dephasing factor ci95 (within the box)")
            edge = v["at the edge of the simulated box"]
            print(
                f"    {k:30s} a {v['first channel factor']:.2f}"
                + (f" ({ci_a[0]:.2f}-{ci_a[1]:.2f})" if ci_a else "")
                + f", b {v['dephasing factor']:.2f}"
                + (f" ({ci_b[0]:.2f}-{ci_b[1]:.2f})" if ci_b else "")
                + f", chi2 {v['chi2']:.1f}/{v['dof']}"
                + (f", p {v['p']:.2g}" if v["p"] is not None else "")
                + (f" (at the edge: {', '.join(edge)})" if edge else "")
            )
    print("\nspread:", json.dumps(out["spread"], indent=1))
    p = out["plausible"]
    print(
        "\nplausible (factor <= bound):",
        json.dumps({c: round(b, 2) for c, b in p["bound"].items()}),
    )
    for bd in out["factors"]:
        print(
            f"  {bd}: "
            + "; ".join(f"{c} {p[bd][c]['eps']}/{p[bd][c]['detection']}" for c in CHANNELS)
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="simulate the X-memory grid first")
    ap.add_argument("--jobs", nargs="*", help="job names (default all fifteen)")
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

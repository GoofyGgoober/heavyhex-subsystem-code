"""One factor on every error of the model: the k that gives the device's logical error per round,
and the k that gives its stabilizer detection rate (Sec. agreement, App. stats).

Each job's model (common.Job.model: the scheduled circuit at the job's measured f) with every
error channel multiplied by k: CZ and single-qubit depolarizing, readout and reset flips, and
idling, whose decay rates 1/T1 and 1/T2 (at the job's f) are multiplied by k, so an idle of t
decays as one of k t. k = 1 is the model itself. Simulated shots are decoded with the k = 1
model's matching graph, as the device's shots were.

  eps        each curve's eps(k) at every k on the grid, from SHOTS shots per circuit, fitted as
             the device's (prespecified WL and binomial); a quadratic in ln k through ln eps(k)
             solves eps(k) = eps observed, over the grid up to the first eps above EPS_CAP, where
             a curve saturates within a round or two. Its error is sigma_O through the slope, with
             the simulation noise added. Beyond the points used it extrapolates (flagged).
  detectors  the mean detection rate of the patch's stabilizer detectors in rounds 1-7 of the
             eight-round circuit (flags and relays left out, as in cz_scaled.py), exact from the
             scaled model's error mechanisms (common.marginals), against the device's.
  summary    per basis, distance and placement: the geometric mean of k over the five days with
             a Student t 95% interval; over the ten repetitions a DerSimonian-Laird mean with the
             Hartung-Knapp interval (effect_sizes.random_effects). One k for both distances:
             least squares in ln eps over the curves, its error from Delta chi2 = 1, widened by
             sqrt(chi2/nu) when above one. The model's Lambda at that k against the device's.

    python docs/analysis/error_scale.py               # the summary from results/error_scale_runs.json, seconds
    python docs/analysis/error_scale.py --recompute   # the simulations first, 7 min on 2 workers
                                                      # [--jobs 2026-10-04-r1 ...] resumes from cache/
"""

from __future__ import annotations

import argparse
import json
import time
from math import exp, log, sqrt
from multiprocessing import Pool

import numpy as np
import pymatching
import stim
from scipy.optimize import brentq, minimize_scalar

from heavyhex.experiment.circuits import ROUNDS

import fitting as F
from common import (
    CACHE,
    RESULTS,
    WORKERS,
    day_number,
    jobs,
    marginals,
    memory_index,
    result,
    write_json,
)
from effect_sizes import random_effects, t_interval

SCALES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
SHOTS = 20_000
LONGEST = max(ROUNDS)
RUNS = RESULTS / "error_scale_runs.json"
PARTS = CACHE / "error_scale"  # one file per job and basis, so a stopped run resumes
FITS = ("binomial", "wls")
PLACEMENTS = (1, 2)
LN_K = (log(0.5), log(6.0))  # where k is searched; beyond the grid it is extrapolated
EPS_CAP = 0.4  # the interpolation stops at the first simulated eps above this


# ----------------------------------------------------------------------------- simulation
def scaled(model: stim.Circuit, k: float) -> stim.Circuit:
    """The model with every error channel times k (idles: every decay rate times k)."""
    out = stim.Circuit()
    for ins in model:
        name, args = ins.name, ins.gate_args_copy()
        if name == "PAULI_CHANNEL_1":
            # noisy.idle_error: 1 - 4 px = e^(-t/T1) and 1 - 2 (px + pz) = e^(-t/T2)
            px, py, pz = args
            assert px == py
            px_k = (1 - (1 - 4 * px) ** k) / 4
            args = [px_k, px_k, (1 - (1 - 2 * (px + pz)) ** k) / 2 - px_k]
        elif name == "DEPOLARIZE1":
            args = [min(k * args[0], 0.75)]
        elif name == "DEPOLARIZE2":
            args = [min(k * args[0], 15 / 16)]
        elif name in ("X_ERROR", "M") and args:
            args = [min(k * args[0], 0.5)]
        elif args and name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            raise ValueError(f"unexpected channel {name}")
        out.append(stim.CircuitInstruction(name, ins.targets_copy(), args))
    return out


def bulk(labels, patch: str, rounds: int) -> list[int]:
    """The patch's stabilizer detectors in rounds 1 .. rounds - 1."""
    return [
        c
        for c, (p, r, n) in enumerate(labels)
        if str(p) == patch and " " not in n and 1 <= r <= rounds - 1
    ]


def simulate_part(task: tuple) -> dict:
    """Failures per round count and k, and the detection rates at each k, for one job and basis."""
    name, basis, shots, scales = task
    job = next(j for j in jobs() if j.name == name)
    out: dict = {"job": name, "basis": basis, "failures": {}, "rate": {}, "device_rate": {}}
    for rounds in ROUNDS:
        idx = memory_index(basis, rounds)
        model, _, labels = job.model(idx)
        matching = pymatching.Matching.from_detector_error_model(
            model.detector_error_model(decompose_errors=True)
        )
        patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
        if rounds == LONGEST:
            seen = job.analysis["observed"]["settings"][idx]
            assert [list(map(str, x)) for x in seen["detectors"]] == [
                list(map(str, x)) for x in labels
            ]
            rates = np.array(seen["detector_rates"])
            out["device_rate"] = {p: float(rates[bulk(labels, p, rounds)].mean()) for p in patches}
        for i, k in enumerate(scales):
            noisy = scaled(model, k)
            det, obs = noisy.compile_detector_sampler(seed=10 * rounds + i).sample(
                shots, separate_observables=True
            )
            failed = matching.decode_batch(det).astype(bool) != obs
            for j, p in enumerate(patches):
                out["failures"].setdefault(p, {}).setdefault(f"{k:g}", []).append(
                    int(failed[:, j].sum())
                )
            if rounds == LONGEST:
                exact = marginals(noisy)
                for p in patches:
                    out["rate"].setdefault(p, {})[f"{k:g}"] = float(
                        exact[bulk(labels, p, rounds)].mean()
                    )
    return out


def simulate(names: list[str], shots: int, scales: tuple, workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    tasks = [
        (n, b, shots, scales) for n in names for b in "XZ" if not (PARTS / f"{n}_{b}.json").exists()
    ]
    t0 = time.time()
    with Pool(workers) as pool:
        for part in pool.imap_unordered(simulate_part, tasks):
            path = PARTS / f"{part['job']}_{part['basis']}.json"
            path.write_text(json.dumps({"shots": shots, "scales": scales, **part}) + "\n")
            print(f"{part['job']} {part['basis']} ({time.time() - t0:.0f} s)", flush=True)
    found = [json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))]
    found = [p for p in found if p["shots"] == shots and tuple(p["scales"]) == tuple(scales)]
    runs: dict = {}
    for p in found:
        runs.setdefault(p["job"], {})[p["basis"]] = {
            k: p[k] for k in ("failures", "rate", "device_rate")
        }
    RUNS.write_text(
        json.dumps(
            {"shots": shots, "scales": list(scales), "rounds": list(ROUNDS), "jobs": runs},
            indent=1,
        )
        + "\n"
    )
    print("wrote", RUNS, f"({len(runs)} jobs)")


# ----------------------------------------------------------------------------- summary
def model_eps(counts: list[int], shots: int) -> dict:
    """The simulated curve fitted both ways; the WL shot-noise error stands in for both."""
    w = F.wls(ROUNDS, np.array(counts) / shots, shots)
    return {
        "wls": w.eps,
        "binomial": F.binomial_point(ROUNDS, counts, shots),
        "se": w.se_raw,
    }


def curve(scales, values, ses=None) -> np.ndarray:
    """Quadratic in ln k through ln values (weighted by the relative errors when given)."""
    x, y = np.log(scales), np.log(values)
    w = None if ses is None else np.asarray(values) / np.asarray(ses)
    return np.polyfit(x, y, 2, w=w)


def eps_curve(scales, eps, se) -> tuple[np.ndarray, np.ndarray]:
    """curve() through the grid from k = 1 up to the first point above EPS_CAP: beyond it a curve
    saturates within a round or two and its fitted eps scatters. Returns the coefficients and
    the points kept."""
    keep = np.cumprod(np.asarray(eps) < EPS_CAP).astype(bool)
    assert keep.sum() >= 3, "fewer than three usable grid points"
    return curve(np.asarray(scales)[keep], np.asarray(eps)[keep], np.asarray(se)[keep]), keep


def solve(coef: np.ndarray, target: float) -> float | None:
    """ln k where the quadratic reaches ln target, on the rising branch within LN_K."""
    f = lambda x: np.polyval(coef, x) - log(target)  # noqa: E731
    grid = np.linspace(*LN_K, 2001)
    rising = np.polyval(np.polyder(coef), grid) > 0
    for a, b, ra, rb in zip(grid[:-1], grid[1:], rising[:-1], rising[1:]):
        if ra and rb and np.sign(f(a)) != np.sign(f(b)):
            return brentq(f, a, b)
    return None


def device_eps(per_rep: dict, job: str, basis: str, d: str) -> dict:
    """The device's eps (both fits) and errors for the job's patch, from fits.json."""
    for key, r in per_rep.items():
        if r["jobs"].get(d) == job:
            e = r[basis][f"eps{d}"]
            return {
                "key": key,
                "wls": e["wls"][0],
                "wls_se": e["wls"][1],
                "binomial": e["binomial"],
                "binomial_se": (e["binomial_hi68_quasi"] - e["binomial_lo68_quasi"]) / 2,
            }
    raise KeyError(job)


def per_curve(res: dict, per_rep: dict) -> list[dict]:
    scales, shots = np.array(res["scales"]), res["shots"]
    rows = []
    for name, by_basis in res["jobs"].items():
        job = f"ibm_fez-{name}"
        for basis, r in by_basis.items():
            for d, by_k in r["failures"].items():
                dev = device_eps(per_rep, job, basis, d)
                fitted = [model_eps(by_k[f"{k:g}"], shots) for k in scales]
                se = np.array([m["se"] for m in fitted])
                row = {
                    "job": name,
                    "rep": dev["key"],
                    "day": day_number(dev["key"][:10]),
                    "placement": int(dev["key"][-1]),
                    "basis": basis,
                    "d": int(d),
                    "model_eps": {fit: [m[fit] for m in fitted] for fit in FITS},
                    "device_eps": {fit: [dev[fit], dev[f"{fit}_se"]] for fit in FITS},
                }
                for fit in FITS:
                    eps = np.array(row["model_eps"][fit])
                    coef, keep = eps_curve(scales, eps, se)
                    row.setdefault("ln_eps_quadratic_in_ln_k", {})[fit] = coef.tolist()
                    resid = np.log(eps[keep]) - np.polyval(coef, np.log(scales[keep]))
                    target, target_se = row["device_eps"][fit]
                    x = solve(coef, target)
                    if x is None:
                        row[f"k_{fit}"] = None
                        continue
                    elasticity = float(np.polyval(np.polyder(coef), x))  # d ln eps / d ln k
                    rel_sim = float(np.interp(x, np.log(scales), se / eps))
                    sigma_ln_eps = sqrt((target_se / target) ** 2 + rel_sim**2)
                    row[f"k_{fit}"] = {
                        "k": exp(x),
                        "sigma_ln_k": sigma_ln_eps / elasticity,
                        "sigma_ln_eps": sigma_ln_eps,
                        "extrapolated": not (scales[keep][0] <= exp(x) <= scales[keep][-1]),
                        "grid_points_used": int(keep.sum()),
                        "elasticity_at_k": elasticity,
                        "elasticity_at_1": float(np.polyval(np.polyder(coef), 0.0)),
                        "max_abs_residual_ln_eps": float(np.abs(resid).max()),
                    }
                rate = np.array([r["rate"][d][f"{k:g}"] for k in scales])
                coef = curve(scales, rate)
                x = solve(coef, r["device_rate"][d])
                row["detection_rate"] = {
                    "device": r["device_rate"][d],
                    "model_k1": float(rate[0]),
                    "model": rate.tolist(),
                    "k": None if x is None else exp(x),
                    "extrapolated": x is None or not (scales[0] <= exp(x) <= scales[-1]),
                    "max_abs_residual_ln_rate": float(
                        np.abs(np.log(rate) - np.polyval(coef, np.log(scales))).max()
                    ),
                }
                rows.append(row)
    rows.sort(key=lambda r: (r["basis"], r["d"], r["placement"], r["day"]))
    return rows


def joint(rows: list[dict], fit: str) -> dict:
    """One k for every curve given: least squares in ln eps, errors from Delta chi2 = 1."""
    terms = []
    for r in rows:
        k = r[f"k_{fit}"]
        if k is None:
            continue
        target = r["device_eps"][fit][0]
        terms.append((np.array(r["ln_eps_quadratic_in_ln_k"][fit]), log(target), k["sigma_ln_eps"]))

    def chi2(x: float) -> float:
        return sum(((np.polyval(c, x) - t) / s) ** 2 for c, t, s in terms)

    best = minimize_scalar(chi2, bounds=LN_K, method="bounded", options={"xatol": 1e-8})
    x, c0, nu = best.x, best.fun, len(terms) - 1
    h = 1e-4
    curvature = (chi2(x + h) - 2 * c0 + chi2(x - h)) / h**2
    sigma = sqrt(2 / curvature) * sqrt(max(c0 / nu, 1.0)) if nu > 0 else sqrt(2 / curvature)
    return {
        "k": exp(x),
        "ci68": [exp(x - sigma), exp(x + sigma)],
        "chi2": c0,
        "dof": nu,
        "chi2_per_dof": c0 / nu if nu else None,
        "curves": len(terms),
    }


def summarize_k(values: list[float], sigmas: list[float], placement: list[int]) -> dict:
    y, v, pl = np.log(values), np.array(sigmas) ** 2, np.array(placement)
    return {
        **{f"placement {p}": t_interval(y[pl == p]) for p in PLACEMENTS if (pl == p).sum() > 1},
        "all ten": random_effects(y, v),
        "per_repetition": [float(x) for x in np.exp(y)],
    }


def lambda_at(rows: list[dict], fit: str, k: float) -> dict:
    """The model's Lambda at k (each curve's quadratic) against the device's, per placement."""
    out = {}
    for p in PLACEMENTS:
        mine = [r for r in rows if r["placement"] == p]
        by_day: dict = {}
        for r in mine:
            coef = np.array(r["ln_eps_quadratic_in_ln_k"][fit])
            by_day.setdefault(r["day"], {})[r["d"]] = (
                exp(np.polyval(coef, log(k))),
                exp(np.polyval(coef, 0.0)),
                r["device_eps"][fit][0],
            )
        lam = {
            day: [v[3][i] / v[5][i] for i in range(3)]
            for day, v in sorted(by_day.items())
            if 3 in v and 5 in v
        }
        model_k, model_1, device = (np.array([v[i] for v in lam.values()]) for i in range(3))
        out[f"placement {p}"] = {
            "model_at_k": float(np.exp(np.log(model_k).mean())),
            "model_at_1": float(np.exp(np.log(model_1).mean())),
            "device": float(np.exp(np.log(device).mean())),
            "per_day": {str(d): [float(x) for x in v] for d, v in lam.items()},
        }
    return out


def summarize() -> dict:
    res = json.loads(RUNS.read_text())
    fits = result("fits.json")
    rows = per_curve(res, fits["per_rep"])
    out: dict = {
        "what": "k multiplies every error channel of the model (idles: every decay rate); "
        "k_eps matches the device's eps per round, k_det the mean bulk stabilizer detection rate "
        "of the eight-round circuit. Geometric means; placement intervals Student t over the five "
        "days, 'all ten' DerSimonian-Laird with Hartung-Knapp 95%.",
        "shots": res["shots"],
        "scales": res["scales"],
        "jobs": len(res["jobs"]),
        "eps": {},
        "detection_rate": {},
        "lambda_at_joint_k": {},
    }
    # the k = 1 simulation against the predictions the analysis stored (200,000 shots)
    pred = fits["predictions"]
    check = [
        r["model_eps"]["binomial"][0]
        / pred[f"ibm_fez-{r['job']}"]["predicted_at_measured_dephasing"][r["basis"]][str(r["d"])][
            "binomial"
        ]["eps"]
        for r in rows
    ]
    out["k1_against_stored_prediction"] = {
        "ratio_range": [min(check), max(check)],
        "ratio_median": float(np.median(check)),
    }
    for fit in FITS:
        out["eps"][fit] = {}
        for basis in "XZ":
            blk: dict = {}
            mine = [r for r in rows if r["basis"] == basis]
            for d in (3, 5):
                sel = [r for r in mine if r["d"] == d and r[f"k_{fit}"]]
                blk[f"d{d}"] = {
                    **summarize_k(
                        [r[f"k_{fit}"]["k"] for r in sel],
                        [r[f"k_{fit}"]["sigma_ln_k"] for r in sel],
                        [r["placement"] for r in sel],
                    ),
                    "extrapolated": sum(r[f"k_{fit}"]["extrapolated"] for r in sel),
                    "unsolved": sum(1 for r in mine if r["d"] == d and not r[f"k_{fit}"]),
                    "elasticity_at_1_median": float(
                        np.median([r[f"k_{fit}"]["elasticity_at_1"] for r in sel])
                    ),
                }
                for p in PLACEMENTS:
                    blk[f"d{d}"][f"joint placement {p}"] = joint(
                        [r for r in sel if r["placement"] == p], fit
                    )
            blk["both distances"] = {
                "joint": joint(mine, fit),
                **{
                    f"joint placement {p}": joint([r for r in mine if r["placement"] == p], fit)
                    for p in PLACEMENTS
                },
            }
            out["eps"][fit][basis] = blk
            out["lambda_at_joint_k"].setdefault(fit, {})[basis] = {
                "k": blk["both distances"]["joint"]["k"],
                **lambda_at(mine, fit, blk["both distances"]["joint"]["k"]),
            }
    for basis in "XZ":
        blk = {}
        for d in (3, 5):
            sel = [
                r for r in rows if r["basis"] == basis and r["d"] == d and r["detection_rate"]["k"]
            ]
            k = np.array([r["detection_rate"]["k"] for r in sel])
            pl = np.array([r["placement"] for r in sel])
            blk[f"d{d}"] = {
                **{
                    f"placement {p}": t_interval(np.log(k[pl == p]))
                    for p in PLACEMENTS
                    if (pl == p).sum() > 1
                },
                "all ten": t_interval(np.log(k)),
                "per_repetition": k.tolist(),
                "extrapolated": sum(r["detection_rate"]["extrapolated"] for r in sel),
                "device_rate_median": float(
                    np.median([r["detection_rate"]["device"] for r in sel])
                ),
                "model_rate_median": float(
                    np.median([r["detection_rate"]["model_k1"] for r in sel])
                ),
            }
        out["detection_rate"][basis] = blk
    out["per_curve"] = rows
    write_json("error_scale.json", out)
    return out


def report(out: dict) -> None:
    def ci(b: dict) -> str:
        return f"{b['ratio']:.2f} ({b['ci95'][0]:.2f}-{b['ci95'][1]:.2f})"

    c = out["k1_against_stored_prediction"]
    print(
        f"k = 1 against the stored predictions: ratio {c['ratio_median']:.3f} "
        f"({c['ratio_range'][0]:.3f}-{c['ratio_range'][1]:.3f})"
    )
    for fit in FITS:
        print(f"\nk matching eps ({fit} fit):")
        for basis in "XZ":
            b = out["eps"][fit][basis]
            for d in ("d3", "d5"):
                x = b[d]
                print(
                    f"  {basis} {d}: all ten {ci(x['all ten'])}  placement 1 {ci(x['placement 1'])}"
                    f"  placement 2 {ci(x['placement 2'])}  (extrapolated {x['extrapolated']},"
                    f" unsolved {x['unsolved']}, d ln eps/d ln k at 1: {x['elasticity_at_1_median']:.2f})"
                )
            j = b["both distances"]
            for key in ("joint", "joint placement 1", "joint placement 2"):
                v = j[key]
                print(
                    f"  {basis} both distances, {key}: k = {v['k']:.2f} "
                    f"({v['ci68'][0]:.2f}-{v['ci68'][1]:.2f}), chi2/nu = {v['chi2']:.1f}/{v['dof']}"
                )
            lam = out["lambda_at_joint_k"][fit][basis]
            print(
                f"  {basis} Lambda at k = {lam['k']:.2f}: "
                + "  ".join(
                    f"placement {p}: model {lam[f'placement {p}']['model_at_1']:.3f} -> "
                    f"{lam[f'placement {p}']['model_at_k']:.3f}, device {lam[f'placement {p}']['device']:.3f}"
                    for p in PLACEMENTS
                )
            )
    print("\nk matching the bulk stabilizer detection rate:")
    for basis in "XZ":
        for d in ("d3", "d5"):
            x = out["detection_rate"][basis][d]
            print(
                f"  {basis} {d}: all ten {ci(x['all ten'])}  placement 1 {ci(x['placement 1'])}"
                f"  placement 2 {ci(x['placement 2'])}  (device {100 * x['device_rate_median']:.1f}%,"
                f" model {100 * x['model_rate_median']:.1f}%, extrapolated {x['extrapolated']})"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="simulate first (heavy)")
    ap.add_argument("--jobs", nargs="*", help="job names (default all fifteen), e.g. 2026-10-06-r1")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.recompute:
        simulate(a.jobs or [j.name for j in jobs()], a.shots, SCALES, a.workers)
    report(summarize())


if __name__ == "__main__":
    main()

"""The model at the inverse-variance mean of the per-qubit f instead of their median (App. f).

The prespecified model gives every qubit the median of the idle test's per-qubit f. The
inverse-variance mean of the same values, which the prespecified uniformity test computes
(analysis.json idle_test.uniformity.weighted_mean), is far lower, because the qubits with
short T2 pin their f best. Here each job's model is simulated at that mean (SHOTS shots per
memory circuit, decoded with its own priors, as run.predict does), and the device's shots are
decoded again with those priors. O/M per placement (geometric mean over the five days, Student
t 95%) and over the ten repetitions (DerSimonian-Laird, Hartung-Knapp), for both fits, beside
the same at the median: the analysis's stored predictions (200,000 shots). sigma_M is the
simulation noise alone in both, so "all ten" at the median differs slightly from
effect_sizes.py, whose sigma_M carries the f term; the placement means do not depend on it.

Also the sign test of f read off the memory runs' detection rates against the idle test's
median (analysis.json dephasing_from_memory; the one from eps is idle_f_t2.py's).

    python docs/analysis/f_weighted_mean.py               # the summary from results/f_weighted_mean_runs.json
    python docs/analysis/f_weighted_mean.py --recompute   # the simulations first, 2 min on 2 workers
"""

from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from multiprocessing import Pool

import numpy as np
import pymatching
from scipy.stats import binomtest

from heavyhex.experiment.circuits import ROUNDS

import fitting as F
from common import RESULTS, WORKERS, jobs, memory_index, result, write_json
from effect_sizes import random_effects, t_interval

SHOTS = 20_000
RUNS = RESULTS / "f_weighted_mean_runs.json"
TESTS = ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5", "Z lambda")
FITS = ("binomial", "wls")
PLACEMENTS = (1, 2)


# ----------------------------------------------------------------------------- simulation
def simulate_job(task: tuple) -> dict:
    name, shots = task
    job = next(j for j in jobs() if j.name == name)
    f = job.analysis["idle_test"]["uniformity"]["weighted_mean"]
    out: dict = {"f_median": job.f, "f_weighted_mean": f, "model": {}, "device": {}}
    for basis in "XZ":
        for rounds in ROUNDS:
            idx = memory_index(basis, rounds)
            model, records, labels = job.model(idx, f=max(f, 0.0))
            matching = pymatching.Matching.from_detector_error_model(
                model.detector_error_model(decompose_errors=True)
            )
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            det, obs = model.compile_detector_sampler(seed=rounds).sample(
                shots, separate_observables=True
            )
            sim = matching.decode_batch(det).astype(bool) != obs
            chip_det, chip_obs = job.chip_detectors(idx, model, records)
            chip = matching.decode_batch(chip_det).astype(bool) != chip_obs
            for k, p in enumerate(patches):
                out["model"].setdefault(basis, {}).setdefault(p, []).append(int(sim[:, k].sum()))
                out["device"].setdefault(basis, {}).setdefault(p, []).append(int(chip[:, k].sum()))
    out["device_shots"] = int(chip.shape[0])
    return {"job": name, **out}


def simulate(shots: int, workers: int) -> None:
    t0 = time.time()
    found = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(simulate_job, [(j.name, shots) for j in jobs()]):
            found[r.pop("job")] = r
            print(f"{len(found)} jobs ({time.time() - t0:.0f} s)", flush=True)
    RUNS.write_text(
        json.dumps(
            {"shots": shots, "rounds": list(ROUNDS), "jobs": dict(sorted(found.items()))},
            indent=1,
        )
        + "\n"
    )


# ----------------------------------------------------------------------------- summary
def fitted(counts: list[int], shots: int) -> dict:
    """(eps, sigma) both ways: WL with its widened error, binomial with its quasi interval."""
    w = F.wls(ROUNDS, np.array(counts) / shots, shots)
    b = F.binomial(ROUNDS, counts, shots)
    return {"wls": (w.eps, w.se), "binomial": (b.eps, (b.hi_q - b.lo_q) / 2)}


def quantities(e3: tuple, e5: tuple) -> dict:
    lam = e3[0] / e5[0]
    return {
        "eps3": e3,
        "eps5": e5,
        "lambda": (lam, lam * sqrt((e3[1] / e3[0]) ** 2 + (e5[1] / e5[0]) ** 2)),
    }


def ratios(O: dict, M: dict, placement: np.ndarray) -> dict:
    """O/M per test: per placement and over the ten repetitions."""
    out = {}
    for q in TESTS:
        o, so = np.array(O[q]).T
        m, sm = np.array(M[q]).T
        y, v = np.log(o / m), (so / o) ** 2 + (sm / m) ** 2
        out[q] = {
            **{f"placement {p}": t_interval(y[placement == p]) for p in PLACEMENTS},
            "all ten": random_effects(y, v),
            "per_repetition": np.exp(y).tolist(),
        }
    return out


def summarize() -> dict:
    res = json.loads(RUNS.read_text())
    shots, sims = res["shots"], res["jobs"]
    fits = result("fits.json")
    per_rep, pred = fits["per_rep"], fits["predictions"]
    keys = sorted(per_rep, key=lambda k: (per_rep[k]["day"], per_rep[k]["rep"]))
    placement = np.array([per_rep[k]["placement"] for k in keys])
    variants = ("median f", "weighted-mean f, device decoded at the median", "weighted-mean f")
    out: dict = {
        "what": "observed over predicted with every qubit at the job's median f (prespecified) "
        "and at the inverse-variance mean of the per-qubit f",
        "f_by_job": {
            n: {"median": s["f_median"], "weighted_mean": s["f_weighted_mean"]}
            for n, s in sims.items()
        },
        "shots": shots,
    }
    for fit in FITS:
        O = {v: {q: [] for q in TESTS} for v in variants}
        M = {v: {q: [] for q in TESTS} for v in variants}
        for key in keys:
            r = per_rep[key]
            for basis in "XZ":
                obs, mod_med, mod_w, dev_w = {}, {}, {}, {}
                for d in ("3", "5"):
                    job = r["jobs"][d]
                    e = r[basis][f"eps{d}"]
                    obs[d] = (
                        (e["wls"][0], e["wls"][1])
                        if fit == "wls"
                        else (
                            e["binomial"],
                            (e["binomial_hi68_quasi"] - e["binomial_lo68_quasi"]) / 2,
                        )
                    )
                    p = pred[job]["predicted_at_measured_dephasing"][basis][d][fit]
                    mod_med[d] = (p["eps"], p["se"] if fit == "wls" else p["se_quasi"])
                    s = sims[job.removeprefix("ibm_fez-")]
                    mod_w[d] = fitted(s["model"][basis][d], shots)[fit]
                    dev_w[d] = fitted(s["device"][basis][d], s["device_shots"])[fit]
                rows = {
                    variants[0]: (
                        quantities(obs["3"], obs["5"]),
                        quantities(mod_med["3"], mod_med["5"]),
                    ),
                    variants[1]: (
                        quantities(obs["3"], obs["5"]),
                        quantities(mod_w["3"], mod_w["5"]),
                    ),
                    variants[2]: (
                        quantities(dev_w["3"], dev_w["5"]),
                        quantities(mod_w["3"], mod_w["5"]),
                    ),
                }
                for v, (o, m) in rows.items():
                    for name in ("eps3", "eps5", "lambda"):
                        O[v][f"{basis} {name}"].append(o[name])
                        M[v][f"{basis} {name}"].append(m[name])
        model = {v: {q: np.array(M[v][q])[:, 0] for q in TESTS} for v in variants}
        out[fit] = {
            "O_over_M": {v: ratios(O[v], M[v], placement) for v in variants},
            "model_at_weighted_mean_over_at_median (geometric mean)": {
                q: float(np.exp(np.log(model[variants[2]][q] / model[variants[0]][q]).mean()))
                for q in TESTS
            },
        }

    # f from the memory runs' detection rates against the idle test's median
    dm = [j.analysis["dephasing_from_memory"] for j in jobs()]
    above = sum(m["from_detector_rates"] > m["idle_test"] for m in dm)
    out["f_from_detection_rates_above_idle"] = {
        "above": above,
        "of": len(dm),
        "sign_test_p_two_sided": binomtest(above, len(dm)).pvalue,
        "median_difference": float(
            np.median([m["from_detector_rates"] - m["idle_test"] for m in dm])
        ),
    }
    write_json("f_weighted_mean.json", out)
    return out


def report(out: dict) -> None:
    def ci(b: dict) -> str:
        return f"{b['ratio']:.2f} ({b['ci95'][0]:.2f}-{b['ci95'][1]:.2f})"

    fw = np.array([v["weighted_mean"] for v in out["f_by_job"].values()])
    fm = np.array([v["median"] for v in out["f_by_job"].values()])
    print(f"f: median {fm.min():.2f}-{fm.max():.2f}, weighted mean {fw.min():.2f}-{fw.max():.2f}")
    for fit in FITS:
        print(f"\nO/M, {fit} fit:")
        for v, blk in out[fit]["O_over_M"].items():
            print(f"  {v}:")
            for q in TESTS:
                x = blk[q]
                print(
                    f"    {q}: placement 1 {ci(x['placement 1'])}  placement 2 {ci(x['placement 2'])}"
                    f"  all ten {ci(x['all ten'])}"
                )
        print(
            "  model at the weighted mean over at the median: "
            + "  ".join(
                f"{q} {v:.3f}"
                for q, v in out[fit][
                    "model_at_weighted_mean_over_at_median (geometric mean)"
                ].items()
            )
        )
    s = out["f_from_detection_rates_above_idle"]
    print(
        f"\nf from the detection rates above the idle test's median in {s['above']} of {s['of']} jobs "
        f"(sign test p = {s['sign_test_p_two_sided']:.2g}, median difference {s['median_difference']:.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="simulate first (heavy)")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.recompute:
        simulate(a.shots, a.workers)
    report(summarize())


if __name__ == "__main__":
    main()

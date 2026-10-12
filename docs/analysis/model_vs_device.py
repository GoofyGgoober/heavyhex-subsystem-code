"""The prediction against the device, beyond the prespecified tests (Sec. agreement, Table days).

  days         Table days: when each day's jobs ran, the calibration they were prepared from,
               its chip medians over working parts (heavyhex.simulation.noisy.typical), and the
               range of the measured f over the day's three jobs
  scatter      an exploratory second agreement test: the day-to-day scatter of placement 1's
               five predictions, s_Q, added to each prediction's variance,
               sigma_M'^2 = sigma_M^2 + (s_Q M)^2 (the prespecified test allows only shot
               noise, f and the before/after calibration gap within a job)
  detectors    detectors firing above prediction by more than sqrt(2 ln N) standard errors,
               per job (analysis.json), and the largest observed/predicted ratio
  slope        the excess rise of the detection rate over the eight rounds (analysis.json
               "leakage"): cases above three standard errors, by basis and distance
  f            the idle test's f per job (the prespecified median beside the inverse-variance
               mean and the chi2/nu of the qubits about it), and f read off the memory runs
               (analysis.json "dephasing_from_memory"): jobs where the two agree
  drift        the prespecified half-against-half drift check, over the 40 comparisons
  model_curves the model's own curves are not exactly exponential: the fit of the day-1
               repetition-1 predictions (fits.json)

Reads runs/ and results/fits.json only; a few seconds.

    python docs/analysis/model_vs_device.py      # writes results/model_vs_device.json
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import chi2

from heavyhex.simulation.noisy import typical

from common import RUNS, day_number, jobs, read, result, write_json

TESTS = ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5")


def utc(stamp: str) -> str:
    return datetime.fromisoformat(stamp).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def days() -> dict:
    out: dict = {}
    for job in jobs():
        sent = read(job.folder / "job.json")
        row = out.setdefault(job.day, {"day": day_number(job.day), "jobs": [], "f": []})
        row["jobs"].append(job.name)
        row["f"].append(job.f)
        row.setdefault("submitted_utc", []).append(sent["submitted_utc"])
        row.setdefault("finished_utc", []).append(sent["finished_utc"])
        if "calibration_utc" not in row:
            cal = job.calibration
            t = typical(cal)
            row["calibration_utc"] = utc(read(job.folder / "calibration.json")["calibrated"])
            row["median_cz_error"] = next(iter(t.cz.values()))
            row["median_readout_error"] = next(iter(t.readout.values()))
            row["median_t1_us"] = next(iter(t.t1_us.values()))
            row["median_t2_us"] = next(iter(t.t2_us.values()))
    for row in out.values():
        row["jobs_utc"] = [utc(min(row.pop("submitted_utc"))), utc(max(row.pop("finished_utc")))]
        row["f_range"] = [min(row["f"]), max(row["f"])]
    return out


def scatter(combined: dict) -> dict:
    reps = combined["reps"]
    rep1 = [i for i, r in enumerate(reps) if r["rep"] == 1]  # one placement, five calibrations
    bar = float(chi2.ppf(0.95, len(reps)))
    out = {}
    for q in TESTS:
        o = np.array([r["tests"][q]["observed"] for r in reps])
        p = np.array([r["tests"][q]["predicted"] for r in reps])
        half = np.array([r["tests"][q]["bound"] / 2 for r in reps])
        s = float(
            np.std(p[rep1], ddof=1) / np.mean(p[rep1])
        )  # relative day-to-day spread of one placement's prediction
        before = float((((o - p) / half) ** 2).sum())
        after = float((((o - p) ** 2) / (half**2 + (s * p) ** 2)).sum())
        out[q] = {
            "chi2": before,
            "chi2_with_scatter": after,
            "bar": bar,
            "relative_scatter": s,
            "placement_1_predicted_range": [float(p[rep1].min()), float(p[rep1].max())],
            "placement_1_observed_range": [float(o[rep1].min()), float(o[rep1].max())],
        }
        print(
            f"  {q:9s} chi2 {before:7.1f} -> {after:6.1f}  (bar {bar:.1f}; prediction scatter {100 * s:.0f}%) "
            f"{'passes' if after < bar else 'fails'}"
        )
    return out


def detectors() -> dict:
    per_job, ratio = {}, []
    for job in jobs():
        a = job.analysis
        n = round(math.exp(a["chance_excess"] ** 2 / 2))  # the bar is sqrt(2 ln N)
        per_job[job.name] = {
            "detectors": n,
            "bar": a["chance_excess"],
            "beyond": a["detectors_beyond_chance"],
            "share": a["detectors_beyond_chance"] / n,
            "patches": len(a["patches"]),
        }
        ratio += [(d["ratio"], d["setting"], d["detector"]) for d in a["worst_detectors"]]
    two = [v for v in per_job.values() if v["patches"] == 2]
    one = [v for v in per_job.values() if v["patches"] == 1]
    top = max(ratio)
    return {
        "per_job": per_job,
        "two_patch_beyond_range": [min(v["beyond"] for v in two), max(v["beyond"] for v in two)],
        "two_patch_detectors": sorted({v["detectors"] for v in two}),
        "two_patch_share_range": [min(v["share"] for v in two), max(v["share"] for v in two)],
        "one_patch_share_range": [min(v["share"] for v in one), max(v["share"] for v in one)],
        "largest_ratio": {"ratio": top[0], "setting": top[1], "detector": top[2]},
    }


def slope() -> dict:
    count: dict = defaultdict(lambda: [0, 0])
    jobs_with = 0
    for job in jobs():
        any_rise = False
        for case, v in job.analysis["leakage"].items():
            basis, _d = case.split()
            for key in (case, basis):
                count[key][0] += bool(v["rises"])
                count[key][1] += 1
            count["all"][0] += bool(v["rises"])
            count["all"][1] += 1
            any_rise |= bool(v["rises"])
        jobs_with += any_rise
    return {
        "above_3_se [cases, of]": dict(count),
        "jobs_with_a_rise": jobs_with,
        "jobs": len(jobs()),
    }


def f() -> dict:
    per_job = {}
    for job in jobs():
        it = job.analysis["idle_test"]
        m = job.analysis["dephasing_from_memory"]
        per_job[job.name] = {
            "median": it["median"],
            "median_uncertainty": it["median_uncertainty"],
            "weighted_mean": it["uniformity"]["weighted_mean"],
            "chi2_per_dof": it["uniformity"]["chi2_per_dof"],
            "quartiles": it["uniformity"]["quartiles"],
            "from_memory_eps": m["from_eps"],
            "from_memory": m["from_eps3"],  # named eps3 for either
            "from_memory_agrees": m["eps3_agrees"],
        }
    meds = [v["median"] for v in per_job.values()]
    return {
        "per_job": per_job,
        "median_range": [min(meds), max(meds)],
        "median_of_medians": float(np.median(meds)),
        "chi2_per_dof_range": [
            min(v["chi2_per_dof"] for v in per_job.values()),
            max(v["chi2_per_dof"] for v in per_job.values()),
        ],
        "weighted_mean_range": [
            min(v["weighted_mean"] for v in per_job.values()),
            max(v["weighted_mean"] for v in per_job.values()),
        ],
        "from_memory_agrees_in": sum(bool(v["from_memory_agrees"]) for v in per_job.values()),
        "from_memory_higher_in": sum(
            v["from_memory"] > v["median"] and not v["from_memory_agrees"] for v in per_job.values()
        ),
    }


def drift() -> dict:
    rel_se, changes, later_worse = [], [], 0
    for path in sorted(RUNS.glob("*/analysis.json")):
        for d in read(path)["drift"].values():
            early, late, change = d["early"], d["late"], d["change"]
            se = (late - early) / change
            rel_se.append(se / ((early + late) / 2))
            changes.append(change)
            later_worse += late > early
    rel_se = np.array(rel_se)
    return {
        "comparisons": len(rel_se),
        "se_over_mean_of_halves_min": float(rel_se.min()),
        "se_over_mean_of_halves_max": float(rel_se.max()),
        "se_over_mean_of_halves_median": float(np.median(rel_se)),
        "threshold_3.5_times_median": float(3.5 * np.median(rel_se)),
        "largest_abs_change_in_se": float(np.max(np.abs(changes))),
        "prespecified_threshold_se": 3.5,
        "second_half_worse": int(later_worse),
    }


def model_curves() -> dict:
    """The fits of the day-1, repetition-1 predictions (2026-10-04-r1), both prediction sources."""
    pred = result("fits.json")["predictions"]["ibm_fez-2026-10-04-r1"]
    return {
        src: {
            f"{b} d{p}": {"chi2_red": c["wls"]["chi2_red"], "A": c["wls"]["A"]}
            for b, by_p in v.items()
            for p, c in by_p.items()
        }
        for src, v in pred.items()
        if src in ("frozen_f0.0", "predicted_at_measured_dephasing")
    }


def main() -> None:
    combined = read(RUNS / "combined.json")
    print(
        "agreement over the reps, chi2 against the 95% bar, as prespecified -> with the calibration's day-to-day scatter:"
    )
    out = {
        "days": days(),
        "scatter": scatter(combined),
        "detectors": detectors(),
        "slope": slope(),
        "f": f(),
        "drift": drift(),
        "model_curves": model_curves(),
    }
    d = out["detectors"]
    print(
        f"detectors beyond chance, two-patch jobs: {d['two_patch_beyond_range']} of {d['two_patch_detectors']}; "
        f"largest ratio {d['largest_ratio']['ratio']:.1f}"
    )
    print(f"excess slope above 3 se: {out['slope']['above_3_se [cases, of]']}")
    print(
        f"f from memory agrees in {out['f']['from_memory_agrees_in']} of {len(out['f']['per_job'])} jobs"
    )
    write_json("model_vs_device.json", out)


if __name__ == "__main__":
    main()

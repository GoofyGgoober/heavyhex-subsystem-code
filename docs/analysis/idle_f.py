"""The idle test's f (App. f), and its variance of ln r for separately sampled <X> and <Y>.

The prespecified code (src/heavyhex/experiment/run.py:1059) weights each ln r point of the
idle test by the variance of a single binomial mean,

  old    Var(ln r) = (1 - r^2) / (N r^2),   N = (N_X + N_Y) / 2,

but <X> and <Y> are sampled separately. The delta method gives

  new    r2 = (1 + 1/N_X) x^2 + (1 + 1/N_Y) y^2 - 1/N_X - 1/N_Y   (as run.py:947)
         Var(r2) = 4 [(1+1/N_X)^2 x^2 (1-x^2)/N_X + (1+1/N_Y)^2 y^2 (1-y^2)/N_Y]
         Var(ln r) = Var(r2) / (4 r2^2).

src/ stays as it ran. This refits every data qubit's f both ways (the old way reproduces
analysis.json exactly) and reports what changes: per-qubit f and its error, the exclusions,
the job medians, and, from idle_f_reanalyze.py's reruns of the whole analysis, every
decision downstream.

Stage 1 caches, per job and data qubit, <X>, <Y>, the shot counts and the exact model
references ln x0, ln x1 (run._idle_dephasing) in results/idle_f_inputs.json (a few minutes
the first time). Then a few seconds.

    python docs/analysis/idle_f.py      # writes results/idle_f.json
"""

from __future__ import annotations

import json
from math import sqrt

import numpy as np
from scipy.stats import spearmanr

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import SETTINGS
from heavyhex.patches.placement import Calibration

from common import RESULTS, RUNS, day_number, job_folders, read, result, write_json
from placements import pool, span

CACHE = RESULTS / "idle_f_inputs.json"


def stage1() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    found = {}
    for folder in job_folders():
        info = json.loads((folder / "run.json").read_text())
        calibration = Calibration.load(folder / "calibration.json")
        circuits = R.load_circuits(folder)
        shots = R.load_shots(folder)
        idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "X"]
        in_y = {
            s.rounds: i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "Y"
        }
        rounds = [SETTINGS[i].rounds for i in idle]
        xs = [R.x_values(shots[i]) for i in idle]
        ys = [R.x_values(shots[in_y[n]]) for n in rounds]
        ends = [[R._exact_x(circuits[i], calibration, info["dt"], f) for i in idle] for f in R.ENDS]
        job = {
            "rounds": rounds,
            "day": info["rep"]["day"],
            "rep": info["rep"]["rep"],
            "chips": info["chips"],
            "qubits": {},
        }
        for d in xs[0]:
            for q in range(int(d) ** 2):
                job["qubits"][f"{d}/{q}"] = {
                    "x": [v[d][q] for v in xs],
                    "y": [v[d][q] for v in ys],
                    "nx": [len(shots[i][f"d{d}"]) for i in idle],
                    "ny": [len(shots[in_y[n]][f"d{d}"]) for n in rounds],
                    "log_x0": [float(np.log(e[d][q])) for e in ends[0]],
                    "log_x1": [float(np.log(e[d][q])) for e in ends[1]],
                }
        found[folder.name] = job
        print("cached", folder.name, flush=True)
    CACHE.write_text(json.dumps(found))
    return found


def fit(seen, log_x0, log_x1, var_log):
    """run.py _fit_share with the weight 1/var_log passed in (identical otherwise)."""
    keep = seen > 0.05
    extra = (log_x0 - np.log(np.where(keep, seen, 1)))[keep]
    full = (log_x0 - log_x1)[keep]
    weight = (1 / var_log)[keep]
    nan = (float("nan"), float("nan"))
    if not keep.any():
        return nan
    if keep.sum() == 1:
        if full[0] <= 0:
            return nan
        share, sigma = extra[0] / full[0], 1 / (full[0] * sqrt(weight[0]))
    else:
        w, wx, wy = weight.sum(), (weight * full).sum(), (weight * extra).sum()
        spread = w * (weight * full**2).sum() - wx**2
        if spread <= 0:
            return nan
        share = (w * (weight * full * extra).sum() - wx * wy) / spread
        sigma = sqrt(w / spread)
    return (float(share), float(sigma)) if sigma <= R.LOOSEST_F else nan


def variances(q):
    x, y = np.array(q["x"]), np.array(q["y"])
    nx, ny = np.array(q["nx"], float), np.array(q["ny"], float)
    r2 = x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny
    r = np.sqrt(np.maximum(r2, 0))
    shots = (nx + ny) / 2
    old = (1 - np.minimum(r, 0.999) ** 2) / (shots * r**2)  # exactly 1/weight of run.py:1059
    var_r2 = 4 * (
        (1 + 1 / nx) ** 2 * x**2 * (1 - x**2) / nx + (1 + 1 / ny) ** 2 * y**2 * (1 - y**2) / ny
    )
    new = var_r2 / (4 * np.maximum(r2, 1e-300) ** 2)
    return r, old, new


def summary(shares, sigmas):
    s, u = np.array(shares), np.array(sigmas)
    ok = np.isfinite(s) & np.isfinite(u)
    s, u = s[ok], u[ok]
    uni = R._uniformity(s, u)
    return {
        "n": int(ok.sum()),
        "median": float(np.median(s)),
        "median_uncertainty": float(1.25 * np.std(s) / sqrt(len(s))),
        "chi2_per_dof": uni["chi2_per_dof"],
        "weighted_mean": uni["weighted_mean"],
    }


def per_qubit(data: dict) -> tuple[dict, list, list]:
    """Every data qubit fitted both ways; the job summaries checked against analysis.json."""
    jobs, per, ratios = {}, [], []
    for name, job in data.items():
        a = read(RUNS / name / "analysis.json")
        so, uo, sn, un = [], [], [], []
        for key, q in job["qubits"].items():
            d, k = key.split("/")
            with np.errstate(divide="ignore", invalid="ignore"):  # r = 0 points are cut below
                r, old, new = variances(q)
            keep = r > 0.05
            ratios += list(np.sqrt(new[keep] / old[keep]))
            l0, l1 = np.array(q["log_x0"]), np.array(q["log_x1"])
            f_o, s_o = fit(r, l0, l1, old)
            f_n, s_n = fit(r, l0, l1, new)
            # the old variance reproduces the prespecified numbers exactly
            ref = a["idle_test"]["per_qubit"][d][int(k)]
            assert (np.isnan(f_o) and (ref is None or not np.isfinite(ref))) or abs(
                f_o - ref
            ) < 1e-9, (name, key, f_o, ref)
            so.append(f_o)
            uo.append(s_o)
            sn.append(f_n)
            un.append(s_n)
            per.append(
                {
                    "job": name,
                    "patch": d,
                    "index": int(k),
                    "physical_qubit": job["chips"][d][int(k)],
                    "f_old": f_o,
                    "sigma_old": s_o,
                    "f_new": f_n,
                    "sigma_new": s_n,
                    "points_kept": int(keep.sum()),
                    "smallest_r_kept": float(r[keep].min()) if keep.any() else None,
                    "_args": (r, l0, l1, new),
                }
            )
        o, n = summary(so, uo), summary(sn, un)
        assert abs(o["median"] - a["idle_test"]["median"]) < 1e-12
        assert abs(o["median_uncertainty"] - a["idle_test"]["median_uncertainty"]) < 1e-12
        jobs[name] = {"old": o, "new": n, "day": job["day"], "rep": job["rep"]}
    return jobs, per, ratios


def downstream(combined: dict) -> dict:
    """The corrected rerun of the whole analysis against an unpatched rerun in the same environment."""
    cmp = result("idle_f_compare.json")
    fixed_all = result("idle_f_reanalysis.json")
    ctrl_all = result("idle_f_reanalysis_control.json")
    comb_fix = result("idle_f_combined_fixed.json")
    comb_ctl = result("idle_f_combined_control.json")
    for comb, ra in ((comb_fix, fixed_all), (comb_ctl, ctrl_all)):
        assert abs(comb["pooled_x"]["lambda"] - ra["pooled_x"]["lambda"][1]) < 1e-12
        assert all(abs(comb["model"][t]["chi2"] - ra["model"][t][1]) < 1e-9 for t in comb["model"])
    flips = []
    for a, b in zip(ctrl_all["reps"], fixed_all["reps"]):
        for t in a["tests"]:
            if a["tests"][t]["agrees"][1] != b["tests"][t]["agrees"][1]:
                flips.append(
                    {
                        "day": a["day"],
                        "rep": a["rep"],
                        "test": t,
                        "control": a["tests"][t]["agrees"][1],
                        "fixed": b["tests"][t]["agrees"][1],
                    }
                )
    pq_flips = []
    for rep, zc in cmp["control"]["per_qubit_f"]["z"].items():
        for t, z in zc.items():
            zf = cmp["fixed"]["per_qubit_f"]["z"][rep][t]
            if (abs(z) < 2) != (abs(zf) < 2):
                pq_flips.append({"rep": rep, "test": t, "z_control": z, "z_fixed": zf})

    def placements(comb):
        res = {}
        for t in ("X lambda", "X eps3", "X eps5", "Z lambda"):
            for pl in (1, 2):
                rows = [r["tests"][t] for r in comb["reps"] if r["rep"] == pl]
                p = pool([r["observed"] for r in rows], [r["observed_uncertainty"] for r in rows])
                pm = float(np.mean([r["predicted"] for r in rows]))
                res[f"{t} placement {pl}"] = {
                    "observed": p["value"],
                    "se": p["se"],
                    "chi2_per_dof": p["chi2_per_dof"],
                    "predicted_mean": pm,
                    "observed_over_predicted": p["value"] / pm,
                }
        return res

    committed_obs = [r["tests"]["X lambda"]["observed"] for r in combined["reps"]]
    control_obs = [r["tests"]["X lambda"]["observed"] for r in comb_ctl["reps"]]
    largest_obs = max(
        zip(combined["reps"], comb_fix["reps"]),
        key=lambda ab: abs(
            ab[1]["tests"]["X lambda"]["observed"] - ab[0]["tests"]["X lambda"]["observed"]
        ),
    )
    out = {
        "source": "results/idle_f_{compare,reanalysis,reanalysis_control,combined_fixed,combined_control}.json; "
        "control = unpatched rerun in the same environment",
        "control_observed_equals_committed_max_diff": float(
            np.max(np.abs(np.array(committed_obs) - control_obs))
        ),
        "memory_f_agreements": {
            v: sum(bool(j["memory_f_agrees"]) for j in cmp[v]["jobs"].values())
            for v in ("control", "fixed")
        },
        "median_f": {
            v: {
                k: cmp[v]["median_f"][k]
                for k in ("chi2", "reps_agreeing", "all_three_X", "pred_lambda_mean")
            }
            for v in ("control", "fixed")
        },
        "per_qubit_f": {
            v: {
                k: cmp[v]["per_qubit_f"][k]
                for k in ("chi2", "reps_agreeing", "all_three_X", "pred_lambda_mean")
            }
            for v in ("control", "fixed")
        },
        "pooled_x": {
            "committed": combined["pooled_x"],
            "control": comb_ctl["pooled_x"],
            "fixed": comb_fix["pooled_x"],
        },
        "per_rep_verdicts_fixed": {
            v: [r["verdict"] for r in comb_fix["reps"]].count(v)
            for v in sorted({r["verdict"] for r in comb_fix["reps"]})
        },
        "median_f_per_rep_agreement_flips": flips,
        "per_qubit_f_per_rep_agreement_flips_X": pq_flips,
        "per_placement_control": placements(comb_ctl),
        "per_placement_fixed": placements(comb_fix),
        "largest_observed_Lambda_X_change": {
            "day": largest_obs[0]["day"],
            "rep": largest_obs[0]["rep"],
            "committed": largest_obs[0]["tests"]["X lambda"]["observed"],
            "fixed": largest_obs[1]["tests"]["X lambda"]["observed"],
            "note": "observed values move because the decoder weights depend on f",
        },
        "largest_predicted_eps3_change": max(
            (
                {
                    "day": a["day"],
                    "rep": a["rep"],
                    "control": a["tests"]["X eps3"]["predicted"],
                    "fixed": b["tests"]["X eps3"]["predicted"],
                }
                for a, b in zip(comb_ctl["reps"], comb_fix["reps"])
            ),
            key=lambda r: abs(r["fixed"] - r["control"]),
        ),
    }
    model_fail_both = all(
        not comb_ctl["model"][t]["agrees"] and not comb_fix["model"][t]["agrees"]
        for t in comb_fix["model"]
    )
    out["verdict_changes"] = {
        "pooled_verdict_changed": comb_ctl["pooled_x"]["verdict"]
        != comb_fix["pooled_x"]["verdict"],
        "any_per_rep_Lambda_verdict_changed": [r["verdict"] for r in comb_ctl["reps"]]
        != [r["verdict"] for r in comb_fix["reps"]],
        "any_model_agreement_decision_changed": not model_fail_both,
        "median_f_reps_agreeing_changed": cmp["control"]["median_f"]["reps_agreeing"]
        != cmp["fixed"]["median_f"]["reps_agreeing"],
        "per_qubit_f_reps_agreeing_changed": cmp["control"]["per_qubit_f"]["reps_agreeing"]
        != cmp["fixed"]["per_qubit_f"]["reps_agreeing"],
    }
    return out


def correction(jobs: dict, per: list, ratios: list, combined: dict) -> dict:
    fo = np.array([p["f_old"] for p in per])
    fn = np.array([p["f_new"] for p in per])
    so = np.array([p["sigma_old"] for p in per])
    sn = np.array([p["sigma_new"] for p in per])
    both = np.isfinite(fo) & np.isfinite(fn)
    newly_excluded = []
    for p in per:
        if np.isfinite(p["f_old"]) and not np.isfinite(p["f_new"]):
            loosest = R.LOOSEST_F
            R.LOOSEST_F = float("inf")
            try:
                f_unc, s_unc = fit(*p["_args"])
            finally:
                R.LOOSEST_F = loosest
            newly_excluded.append(
                {k2: v for k2, v in p.items() if k2 != "_args"}
                | {"f_new_uncapped": f_unc, "sigma_new_uncapped": s_unc, "cut_sigma": loosest}
            )
    newly_included = [
        {k2: v for k2, v in p.items() if k2 != "_args"}
        for p in per
        if not np.isfinite(p["f_old"]) and np.isfinite(p["f_new"])
    ]
    ratios = np.array(ratios)
    df = np.abs(fn[both] - fo[both])

    def dist(f, s):
        m = np.isfinite(f)
        q1, q3 = np.percentile(f[m], [25, 75])
        return {
            "measured": int(m.sum()),
            "excluded": int((~m).sum()),
            "negatives": int((f[m] < 0).sum()),
            "iqr": [float(q1), float(q3)],
            "spearman_f_sigma": float(spearmanr(f[m], s[m]).statistic),
        }

    job_rows = {}
    for name, v in jobs.items():
        o, n = v["old"], v["new"]
        job_rows[name] = {
            "day": v["day"],
            "day_number": day_number(v["day"]),
            "rep": v["rep"],
            "median_old": o["median"],
            "median_new": n["median"],
            "change": n["median"] - o["median"],
            "change_over_se_old": (n["median"] - o["median"]) / o["median_uncertainty"],
            "se_old": o["median_uncertainty"],
            "se_new": n["median_uncertainty"],
            "chi2_per_dof_old": o["chi2_per_dof"],
            "chi2_per_dof_new": n["chi2_per_dof"],
            "weighted_mean_old": o["weighted_mean"],
            "weighted_mean_new": n["weighted_mean"],
            "qubits_old": o["n"],
            "qubits_new": n["n"],
        }
    largest = max(job_rows, key=lambda j: abs(job_rows[j]["change"]))
    others = [abs(r["change"]) for j, r in job_rows.items() if j != largest]
    others_z = [abs(r["change_over_se_old"]) for j, r in job_rows.items() if j != largest]

    def ranges(ver):
        meds = [r[f"median_{ver}"] for r in job_rows.values()]
        by_day: dict = {}
        for r in job_rows.values():
            by_day.setdefault(r["day_number"], []).append(r[f"median_{ver}"])
        return {
            "job_median_range": span(meds),
            "median_of_job_medians": float(np.median(meds)),
            "chi2_per_dof_range": span([r[f"chi2_per_dof_{ver}"] for r in job_rows.values()]),
            "weighted_mean_range": span([r[f"weighted_mean_{ver}"] for r in job_rows.values()]),
            "f_range_by_day (Table days)": {d: span(v) for d, v in sorted(by_day.items())},
        }

    per_qubit_out = {
        "points_kept_r_above_0.05": len(ratios),
        "se_ln_r_new_over_old": {
            "median": float(np.median(ratios)),
            "p95": float(np.percentile(ratios, 95)),
            "max": float(ratios.max()),
            "above_1.10": int((ratios > 1.10).sum()),
        },
        "data_qubits": len(per),
        "old": dist(fo, so),
        "new": dist(fn, sn),
        "abs_change_f": {
            "median": float(np.median(df)),
            "p95": float(np.percentile(df, 95)),
            "max": float(df.max()),
        },
        "abs_change_f_over_sigma_old_max": float((df / so[both]).max()),
        "sigma_f_new_over_old": {
            "median": float(np.median(sn[both] / so[both])),
            "p95": float(np.percentile(sn[both] / so[both], 95)),
            "max": float((sn[both] / so[both]).max()),
        },
        "newly_excluded": newly_excluded,
        "newly_included": newly_included,
    }
    return {
        "what": "old: Var(ln r) = (1 - r^2)/(N r^2) as prespecified (run.py:1059); new: delta method for "
        "separately sampled <X> and <Y>, Var(ln r) = [x^2(1-x^2)/N_X + y^2(1-y^2)/N_Y]/r^4 "
        "(with the (1+1/N) shot-noise factors).",
        "per_qubit": per_qubit_out,
        "jobs": job_rows,
        "largest_job_change": {"job": largest, **job_rows[largest]},
        "other_jobs_max_abs_change": max(others),
        "other_jobs_max_abs_change_over_se": max(others_z),
        "old": ranges("old"),
        "new": ranges("new"),
        "downstream": downstream(combined),
    }


def main() -> None:
    jobs, per, ratios = per_qubit(stage1())
    out = {
        "per_job": {name: {"old": v["old"], "new": v["new"]} for name, v in jobs.items()},
        "correction": correction(jobs, per, ratios, read(RUNS / "combined.json")),
    }
    c = out["correction"]
    q = c["per_qubit"]
    print(
        f"SE(ln r) new/old: median {q['se_ln_r_new_over_old']['median']:.4f}, max {q['se_ln_r_new_over_old']['max']:.4f}"
    )
    print(
        f"excluded {q['old']['excluded']} -> {q['new']['excluded']} of {q['data_qubits']}; "
        f"largest job change {c['largest_job_change']['job']}: {c['largest_job_change']['median_old']:.3f} -> "
        f"{c['largest_job_change']['median_new']:.3f}"
    )
    print("verdict changes:", c["downstream"]["verdict_changes"])
    write_json("idle_f.json", out)


if __name__ == "__main__":
    main()

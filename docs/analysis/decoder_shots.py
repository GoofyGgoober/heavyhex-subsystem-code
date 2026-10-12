"""Re-decode every hardware shot of the 15 jobs with uniform-weight matching.

For each job and memory setting: the stim model is rebuilt exactly as analyze()
built it (job calibration, dt, the job's measured f), the chip's shots are turned
into detectors with that model's m2d converter, and decoded twice:
  calibrated: run._decode (the paper's decoder) -- must reproduce analysis.json;
  uniform:    the same matching graph with all edge weights equal.
Fits use the paper's procedure unchanged (run._fits -> fit_per_round on rounds
1,2,3,4,6,8, shot-noise weights, points with P >= 0.5 dropped, uncertainty widened
by sqrt(chi2/dof) when > 1); Lambda per rep as run.combine (same-job ratio for rep 1,
the two jobs' eps for rep 2, both with the independent-error formula), pooled as
run.combine (inverse variance, widened by sqrt(chi2/nu) when > 1, Student t, nu = R-1).

The two decoders see the same shots, so their difference is judged paired:
shot-level discordant counts (McNemar) per setting, and a paired bootstrap over
shots (same resampled shots for both decoders, every setting of every job, B
replicates) for the differences in eps and Lambda, per rep and pooled.

Optionally (--simulate N) the calibration model itself is sampled (N shots per
setting, seeds as predict()) and decoded both ways: what the model expects the
change of weights to do.

Heavy (the bootstrap, and the simulation); decoder.py reads the committed outputs:

    python docs/analysis/decoder_shots.py                   # results/decoder_shots.json
    python docs/analysis/decoder_shots.py --simulate 20000  # results/decoder_shots_sim20000.json
"""

from __future__ import annotations

import argparse
import json
from math import sqrt

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student_t

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.simulation.noisy import fit_per_round

from common import RESULTS, jobs
from matching import calibrated_matching, uniform_matching

MEMORY = [i for i, s in enumerate(SETTINGS) if s.kind == "memory"]
DECODERS = ("calibrated", "uniform")


def fits_from_errors(errors: dict[int, list[float]], shots: int, patches: list[str]) -> dict:
    """run._fits on {setting index: logical_error per observable}."""
    settings = [{"logical_error": errors[i]} if i in errors else {} for i in range(len(SETTINGS))]
    return R._fits(settings, shots, patches)["fits"]


def eps_only(errors, shots, patches):
    """eps per basis and patch from logical errors, as fit_per_round, without the rest."""
    out = {}
    for basis in "XZ":
        rows = [errors[i] for i in MEMORY if SETTINGS[i].basis == basis]
        for k, d in enumerate(patches):
            f = fit_per_round(ROUNDS, [r[k] for r in rows], shots)
            out[f"{basis}{d}"] = f.per_round
    return out


def lam(e3, u3, e5, u5):
    value = e3 / e5
    return value, value * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)


def pool(values, sigmas):
    """run.combine's pooled Lambda: inverse-variance mean, sigma widened by sqrt(chi2/nu) if > 1, Student t."""
    w = 1 / np.asarray(sigmas) ** 2
    v = np.asarray(values)
    m = float((w * v).sum() / w.sum())
    spread = float((w * (v - m) ** 2).sum() / (len(v) - 1))
    s = sqrt(max(spread, 1.0) / w.sum())
    crit = float(student_t.ppf(0.95, len(v) - 1))
    verdict = (
        "d = 5 better" if m - crit * s > 1 else "d = 5 worse" if m + crit * s < 1 else "undecided"
    )
    return {
        "lambda": m,
        "lambda_uncertainty": s,
        "spread": spread,
        "critical": crit,
        "verdict": verdict,
    }


def decode_job(job, simulate: int = 0) -> dict:
    patches = job.analysis["patches"]
    shots = R._memory_shots(job.shots)
    fails = {d: {} for d in DECODERS}
    sim = {d: {} for d in DECODERS}
    mcnemar = {}
    for i in MEMORY:
        model, records, _labels = job.model(i)
        dets, flips = job.chip_detectors(i, model, records)
        cal = calibrated_matching(model)
        uni = uniform_matching(model, cal)
        f_cal = R._decode(model, dets) != flips
        f_uni = uni.decode_batch(dets).astype(bool) != flips
        assert np.array_equal(cal.decode_batch(dets).astype(bool) != flips, f_cal)
        saved = job.analysis["observed"]["settings"][i]["logical_error"]
        assert np.array_equal(f_cal.mean(0), np.array(saved)), (job.name, i, f_cal.mean(0), saved)
        fails["calibrated"][i], fails["uniform"][i] = f_cal, f_uni
        mcnemar[str(SETTINGS[i])] = {
            patches[k]: {
                "cal_only_fails": int((f_cal[:, k] & ~f_uni[:, k]).sum()),
                "uni_only_fails": int((f_uni[:, k] & ~f_cal[:, k]).sum()),
            }
            for k in range(len(patches))
        }
        if simulate:
            s = SETTINGS[i]
            d_sim, o_sim = model.compile_detector_sampler(seed=s.rounds).sample(
                simulate, separate_observables=True
            )
            sim["calibrated"][i] = (cal.decode_batch(d_sim).astype(bool) != o_sim).mean(0).tolist()
            sim["uniform"][i] = (uni.decode_batch(d_sim).astype(bool) != o_sim).mean(0).tolist()
    out = {"patches": patches, "shots": shots, "f": job.f, "mcnemar": mcnemar, "fails": fails}
    out["fits"] = {
        d: fits_from_errors({i: fails[d][i].mean(0).tolist() for i in MEMORY}, shots, patches)
        for d in DECODERS
    }
    for basis in (
        "XZ"
    ):  # the paper's eps, to rounding (the stored fits may come from another numpy/LAPACK build)
        for d, (e, u) in out["fits"]["calibrated"][basis]["per_round"].items():
            e0, u0 = job.analysis["observed"]["fits"][basis]["per_round"][d]
            assert abs(e - e0) < 1e-9 * e0 and abs(u - u0) < 1e-6 * u0, (
                job.name,
                basis,
                d,
                e,
                e0,
                u,
                u0,
            )
    if simulate:
        out["simulated_fits"] = {d: fits_from_errors(sim[d], simulate, patches) for d in DECODERS}
        out["simulated_errors"] = {
            d: {str(SETTINGS[i]): sim[d][i] for i in MEMORY} for d in DECODERS
        }
    return out


def reps_from(per_job: dict, key="fits") -> list[dict]:
    """Per rep (day, rep): eps3, eps5 and Lambda in X and Z, as run.combine pairs the jobs."""
    reps: dict = {}
    for name, info in per_job.items():
        job = info["job"]
        r = reps.setdefault(
            (job.day, job.rep), {"day": job.day, "rep": job.rep, "folders": [], "eps": {}}
        )
        r["folders"].append(name)
        for dec in DECODERS:
            for basis in "XZ":
                for d, (e, u) in info[key][dec][basis]["per_round"].items():
                    r["eps"].setdefault(dec, {})[f"{basis} eps{d}"] = (e, u)
    out = []
    for (day, rep), r in sorted(reps.items()):
        row = {"day": day, "rep": rep, "folders": r["folders"]}
        for dec in DECODERS:
            e = r["eps"][dec]
            row[dec] = {k: list(v) for k, v in e.items()}
            for basis in "XZ":
                row[dec][f"{basis} lambda"] = list(lam(*e[f"{basis} eps3"], *e[f"{basis} eps5"]))
        out.append(row)
    return out


def bootstrap(per_job: dict, B: int, seed: int = 7) -> dict:
    """Paired bootstrap over shots: per job and setting, the same resampled shot indices for both
    decoders (each setting resampled independently, as its shots are independent)."""
    rng = np.random.default_rng(seed)
    names = list(per_job)
    rows = []
    for _ in range(B):
        eps = {}
        for name in names:
            info = per_job[name]
            n = info["shots"]
            errs = {dec: {} for dec in DECODERS}
            for i in MEMORY:
                idx = rng.integers(0, n, n)
                for dec in DECODERS:
                    errs[dec][i] = info["fails"][dec][i][idx].mean(0).tolist()
            for dec in DECODERS:
                for key, v in eps_only(errs[dec], n, info["patches"]).items():
                    eps[(name, dec, key)] = v
        rows.append(eps)
    jobs_by_rep: dict = {}
    for name in names:
        j = per_job[name]["job"]
        jobs_by_rep.setdefault((j.day, j.rep), []).append(name)
    out = {"B": B, "reps": {}}
    lam_samples = {dec: {basis: [] for basis in "XZ"} for dec in DECODERS}
    quantities = ("X eps3", "X eps5", "Z eps3", "Z eps5", "X lambda", "Z lambda")
    for (day, rep), names_r in sorted(jobs_by_rep.items()):
        samples = {dec: {q: [] for q in quantities} for dec in DECODERS}
        for eps in rows:
            for dec in DECODERS:
                got = {}
                for name in names_r:
                    for basis in "XZ":
                        for d in "35":
                            if (name, dec, f"{basis}{d}") in eps:
                                got[f"{basis} eps{d}"] = eps[(name, dec, f"{basis}{d}")]
                for q, v in got.items():
                    samples[dec][q].append(v)
                for basis in "XZ":
                    samples[dec][f"{basis} lambda"].append(
                        got[f"{basis} eps3"] / got[f"{basis} eps5"]
                    )
        res = {}
        for q in quantities:
            c, u = np.array(samples["calibrated"][q]), np.array(samples["uniform"][q])
            ok = np.isfinite(c) & np.isfinite(u)
            res[q] = {
                "sd_calibrated": float(np.std(c[ok], ddof=1)),
                "sd_uniform": float(np.std(u[ok], ddof=1)),
                "sd_difference_uniform_minus_calibrated": float(np.std(u[ok] - c[ok], ddof=1)),
            }
        out["reps"][f"{day} r{rep}"] = res
        for dec in DECODERS:
            for basis in "XZ":
                lam_samples[dec][basis].append(samples[dec][f"{basis} lambda"])
    out["lambda_samples"] = {
        dec: {b: np.array(v).T.tolist() for b, v in s.items()} for dec, s in lam_samples.items()
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--simulate", type=int, default=0)
    args = ap.parse_args()
    per_job = {}
    for job in jobs():
        info = decode_job(job, args.simulate)
        info["job"] = job
        per_job[job.name] = info
        fx = info["fits"]
        print(
            job.name,
            " ".join(
                f"{dec[:3]} X "
                + " ".join(f"e{d}={fx[dec]['X']['per_round'][d][0]:.4f}" for d in info["patches"])
                + " Z "
                + " ".join(f"e{d}={fx[dec]['Z']['per_round'][d][0]:.4f}" for d in info["patches"])
                for dec in DECODERS
            ),
            flush=True,
        )
    reps = reps_from(per_job)
    boot = bootstrap(per_job, args.boot) if args.boot else None
    pooled = {}
    for dec in DECODERS:
        for basis in "XZ":
            v = [r[dec][f"{basis} lambda"][0] for r in reps]
            s = [r[dec][f"{basis} lambda"][1] for r in reps]
            pooled[f"{dec} {basis}"] = pool(v, s)
            pooled[f"{dec} {basis}"]["unweighted_mean"] = float(np.mean(v))
            pooled[f"{dec} {basis}"]["unweighted_se"] = float(np.std(v, ddof=1) / sqrt(len(v)))
    if boot:
        # shot-noise sd of the pooled difference: the pooled Lambda recomputed in each replicate,
        # with each decoder's inverse-variance weights held at their point-estimate values
        for basis in "XZ":
            diffs = []
            for b in range(boot["B"]):
                vals = {}
                for dec in DECODERS:
                    lv = np.array(boot["lambda_samples"][dec][basis][b])
                    w = 1 / np.array([r[dec][f"{basis} lambda"][1] for r in reps]) ** 2
                    vals[dec] = float((w * lv).sum() / w.sum())
                diffs.append(vals["uniform"] - vals["calibrated"])
            d = np.array(diffs)
            pooled[f"difference {basis}"] = {
                "uniform_minus_calibrated": pooled[f"uniform {basis}"]["lambda"]
                - pooled[f"calibrated {basis}"]["lambda"],
                "paired_bootstrap_sd_shot_noise": float(np.nanstd(d, ddof=1)),
                "bootstrap_B": boot["B"],
            }
    mc = {}
    for _name, info in per_job.items():
        for setting, by_patch in info["mcnemar"].items():
            basis = setting[0]
            for p, c in by_patch.items():
                row = mc.setdefault(f"{basis} d{p}", {"cal_only_fails": 0, "uni_only_fails": 0})
                row["cal_only_fails"] += c["cal_only_fails"]
                row["uni_only_fails"] += c["uni_only_fails"]
    for row in mc.values():
        n = row["cal_only_fails"] + row["uni_only_fails"]
        row["p_two_sided"] = float(binomtest(row["cal_only_fails"], n, 0.5).pvalue) if n else 1.0
    raw = {
        name: {
            dec: {str(SETTINGS[i]): info["fails"][dec][i].mean(0).tolist() for i in MEMORY}
            for dec in DECODERS
        }
        for name, info in per_job.items()
    }
    result = {
        "method": __doc__,
        "per_job": {
            name: {
                "patches": i["patches"],
                "shots_per_setting": i["shots"],
                "f": i["f"],
                "fits": i["fits"],
                "logical_error": raw[name],
                "mcnemar": i["mcnemar"],
                **(
                    {
                        "simulated_fits": i["simulated_fits"],
                        "simulated_errors": i["simulated_errors"],
                    }
                    if "simulated_fits" in i
                    else {}
                ),
            }
            for name, i in per_job.items()
        },
        "reps": reps,
        "pooled": pooled,
        "mcnemar_pooled": mc,
        "bootstrap_per_rep": boot["reps"] if boot else None,
    }
    if args.simulate:
        sreps = reps_from(per_job, key="simulated_fits")
        result["simulated_reps"] = sreps
        result["simulated_pooled"] = {}
        for dec in DECODERS:
            for basis in "XZ":
                v = [r[dec][f"{basis} lambda"][0] for r in sreps]
                s = [r[dec][f"{basis} lambda"][1] for r in sreps]
                result["simulated_pooled"][f"{dec} {basis}"] = pool(v, s)
    tag = f"_sim{args.simulate}" if args.simulate else ""
    (RESULTS / f"decoder_shots{tag}.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8"
    )

    print(
        "\nper rep:  X eps3, X eps5, Lambda_X | Z eps3, Z eps5, Lambda_Z   (calibrated -> uniform, paired-bootstrap sd of the difference)"
    )
    for r in reps:
        c, u = r["calibrated"], r["uniform"]
        bx = boot["reps"][f"{r['day']} r{r['rep']}"] if boot else {}

        def fmt(q, c=c, u=u, bx=bx):
            s = f"{c[q][0]:.4f}+-{c[q][1]:.4f} -> {u[q][0]:.4f}+-{u[q][1]:.4f}"
            if bx:
                s += f" (diff sd {bx[q]['sd_difference_uniform_minus_calibrated']:.4f})"
            return s

        print(
            f"{r['day']} r{r['rep']}: "
            + " | ".join(
                fmt(q) for q in ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5", "Z lambda")
            )
        )
    print("\npooled:")
    for k, v in pooled.items():
        print(" ", k, json.dumps(v))
    print("McNemar:", json.dumps(mc))
    if args.simulate:
        print("simulated pooled:", json.dumps(result["simulated_pooled"], indent=0))


if __name__ == "__main__":
    main()

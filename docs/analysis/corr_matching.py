"""The device's and the model's shots decoded with correlated matching (Sec. code, App. decoder).

Stim splits each fault that fires more than two detectors into edges for matching; at distance
5 in Z memory, pieces of flag and relay faults form logical paths of only three edges. PyMatching
2.4's correlated matching (enable_correlations) uses the split: once matching uses one piece of
a fault, the fault's other pieces become cheaper. Every memory circuit of every job is decoded
with it on the same detector error model as the paper's decoder (decomposed, at the job's
measured f), and refitted with both fits (refit.py): the device's shots, and 10,000 shots
sampled from the model (refit.sim_seed, the same shots as pij_decode.py's control).

A few faults are split with a piece that holds only the logical observable, which correlated
matching refuses; those pieces are dropped (correlatable). Without correlations the graph is
then unchanged: it decodes every hardware shot exactly as the paper's decoder (checked).

Near P = 1/2 the prespecified fit drops round counts, and a change of decoder can add or remove
such a point; per repetition, Lambda_Z then moves through the fit as much as through the
decoder. The per-repetition values carry the number of points at or above 1/2 under each
decoder.

About 3 minutes on 2 workers, 0.5 GB each.

    python docs/analysis/corr_matching.py     # writes results/corr_matching.json
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pymatching
import stim

import refit
from common import WORKERS, jobs, write_json
from matching import calibrated_matching

DECODERS = ("calibrated", "correlated")


def correlatable(dem: stim.DetectorErrorModel) -> tuple[stim.DetectorErrorModel, int]:
    """The decomposed DEM without split pieces that fire no detector, and how many it dropped."""
    out = stim.DetectorErrorModel()
    dropped = 0
    for ins in dem.flattened():
        if ins.type != "error":
            out.append(ins)
            continue
        parts: list[list] = [[]]
        for t in ins.targets_copy():
            if t.is_separator():
                parts.append([])
            else:
                parts[-1].append(t)
        kept = [p for p in parts if any(t.is_relative_detector_id() for t in p)]
        dropped += len(parts) - len(kept)
        targets: list = []
        for j, p in enumerate(kept):
            targets += ([stim.target_separator()] if j else []) + p
        if targets:
            out.append("error", ins.args_copy(), targets)
    return out, dropped


def job_decode(k: int) -> dict:
    t0 = time.time()
    job = jobs()[k]
    patches = job.analysis["patches"]
    fails: dict = {d: {} for d in DECODERS}
    sim_fails: dict = {d: {} for d in DECODERS}
    dropped = 0
    for i in refit.MEMORY:
        model, records, _labels = job.model(i)
        dets, flips = job.chip_detectors(i, model, records)
        clean, n = correlatable(model.detector_error_model(decompose_errors=True))
        dropped += n
        cal = calibrated_matching(model)
        f_cal = cal.decode_batch(dets).astype(bool) != flips
        saved = job.analysis["observed"]["settings"][i]["logical_error"]
        assert np.array_equal(f_cal.mean(0), np.array(saved)), (job.name, i)
        plain = pymatching.Matching.from_detector_error_model(clean)
        assert np.array_equal(plain.decode_batch(dets).astype(bool) != flips, f_cal), (job.name, i)
        corr = pymatching.Matching.from_detector_error_model(clean, enable_correlations=True)
        fails["calibrated"][i] = f_cal
        fails["correlated"][i] = (
            corr.decode_batch(dets, enable_correlations=True).astype(bool) != flips
        )
        sd, so = model.compile_detector_sampler(seed=refit.sim_seed(i)).sample(
            refit.SIM_SHOTS, separate_observables=True
        )
        sim_fails["calibrated"][i] = cal.decode_batch(sd).astype(bool) != so
        sim_fails["correlated"][i] = (
            corr.decode_batch(sd, enable_correlations=True).astype(bool) != so
        )
    out = {
        "name": job.name,
        "day": job.day,
        "rep": job.rep,
        "patches": patches,
        "pieces_dropped": dropped,
        "curves": refit.job_curves(fails, patches),
        "boot": refit.job_boot(fails, patches, seed=100 * k),
        "sim_curves": refit.job_curves(sim_fails, patches, profile=False),
        "sim_boot": refit.job_boot(sim_fails, patches, seed=100 * k + 50),
        "seconds": time.time() - t0,
    }
    print(f"{job.name}: {out['seconds']:.0f} s", flush=True)
    return out


def fragility(reps: list[dict]) -> list[dict]:
    """Per repetition: Lambda_Z and eps5 under both decoders and fits, and the points at or
    above 1/2 (eps3, eps5) under each."""
    out = []
    for r in reps:
        row = {"day": r["day"], "placement": r["placement"]}
        for basis in "XZ":
            for d in DECODERS:
                x = r["decoders"][d][basis]
                row[f"{basis} {d}"] = {
                    "lambda_wls": x["lambda"]["wls"],
                    "lambda_binomial": x["lambda"]["binomial"],
                    "eps5_wls": x["eps5"]["wls"],
                    "points_at_or_above_half": x["points_at_or_above_half"],
                }
            c, k = row[f"{basis} calibrated"], row[f"{basis} correlated"]
            row[f"{basis} change"] = {
                "lambda_wls": k["lambda_wls"][0] - c["lambda_wls"][0],
                "lambda_binomial": k["lambda_binomial"][0] - c["lambda_binomial"][0],
                "points_at_or_above_half_changed": k["points_at_or_above_half"]
                != c["points_at_or_above_half"],
            }
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(args.workers) as pool:
        done = pool.map(job_decode, range(len(jobs())))
    per_job = {j["name"]: j for j in done}
    sim_jobs = {k: {**j, "curves": j["sim_curves"]} for k, j in per_job.items()}
    reps = refit.reps(per_job)
    sim_reps = refit.reps(sim_jobs)
    boots = {n: j["boot"] for n, j in per_job.items()}
    sim_boots = {n: j["sim_boot"] for n, j in per_job.items()}
    frag = fragility(reps)
    sim_frag = fragility(sim_reps)
    changes = {}
    for basis in "XZ":
        for fit in ("wls", "binomial"):
            dev = [f[f"{basis} change"][f"lambda_{fit}"] for f in frag]
            sim = [f[f"{basis} change"][f"lambda_{fit}"] for f in sim_frag]
            changes[f"{basis} {fit}"] = {
                "device_per_rep_range": [min(dev), max(dev)],
                "model_per_rep_range": [min(sim), max(sim)],
                "reps_whose_points_at_half_changed": sum(
                    f[f"{basis} change"]["points_at_or_above_half_changed"] for f in frag
                ),
            }
    out = {
        "method": __doc__,
        "pieces_dropped": sum(j["pieces_dropped"] for j in done),
        "device_correlated_minus_calibrated": refit.paired_differences(
            per_job, boots, reps, "correlated", "calibrated"
        ),
        "model_correlated_minus_calibrated": refit.paired_differences(
            sim_jobs, sim_boots, sim_reps, "correlated", "calibrated"
        ),
        "lambda_change_per_rep": changes,
        "placements": refit.placements(reps),
        "model_placements": refit.placements(sim_reps),
        "per_rep": frag,
        "model_per_rep": sim_frag,
        "seconds": time.time() - t0,
    }
    write_json("corr_matching.json", out)
    for name in ("device_correlated_minus_calibrated", "model_correlated_minus_calibrated"):
        print(name)
        for k, v in out[name].items():
            print(
                f"  {k}: eps3 {100 * v['eps3'][0]:+.2f}+-{100 * v['eps3'][1]:.2f} pp,"
                f" eps5 {100 * v['eps5'][0]:+.2f}+-{100 * v['eps5'][1]:.2f} pp,"
                f" lambda {v['lambda'][0]:+.3f}+-{v['lambda'][1]:.3f}"
            )
    print(changes)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

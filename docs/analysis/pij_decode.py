"""The device's shots decoded with priors learned from its own detection events (Sec. agreement, App. decoder).

How much of the underprediction is the decoder's? The decoder's priors come from the
calibration model, which the device does not follow; simulated shots are decoded with priors
that match them exactly. Here every hardware shot, at every round count, is decoded on the
same matching graph with learned edge probabilities, and refitted with both fits (refit.py).
The change of eps, learned - calibrated, is set against the gap, observed (calibrated priors)
- predicted (fits.json). The learned values are pij.json's: p_ij of each edge type, averaged
over the rounds of the job's 6- and 8-round circuits (detectors.priors). Three ways to use them:

  learned             each edge takes its type's learned p (the usual p_ij priors)
  learned, scaled     each edge's calibrated p times its type's learned / model ratio, both as
                      the estimator sees them; this keeps the model's split of faults that
                      fire three or more detectors, which the pair estimator cannot resolve
  learned pair edges  as "learned", but every boundary edge keeps its calibrated p: a check
                      of how much the boundary estimates, the remainder of p_i, matter

Control: the estimator's own effect. For each circuit, 10,000 shots sampled from the model
(refit.sim_seed) are decoded with the calibrated priors and with priors learned the same way
from the model's own shots: from 2500 shots of each learning circuit (as many as the device
has), in each of the three ways, and directly from all 10,000. On the model the calibrated
priors are exact, so any change there comes from the estimator: its noise, the round averaging
and, for the direct priors, the pair projection of faults that fire three or more detectors.
Each device change is also given net of its control.

The changes' uncertainties are paired bootstraps over shots with the priors held fixed
(refit.py); the priors' own noise is what the control measures.

About 5 minutes on 2 workers, 0.5 GB each.

    python docs/analysis/pij.py           # first: the learned priors
    python docs/analysis/pij_decode.py    # writes results/pij_decode.json
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from heavyhex.experiment.circuits import SETTINGS

import detectors as DT
import refit
from common import WORKERS, jobs, result, write_json
from matching import calibrated_matching

USED = (6, 8)
LEARN_SHOTS = 2500
MODEL_MIN = 1e-4  # a type whose model p (as the estimator sees it) is below this is not scaled
VARIANTS = {  # name: (scaled, boundary edges learned too, its control)
    "learned": (False, True, "learned (2500 shots)"),
    "learned, scaled": (True, True, "learned, scaled (2500 shots)"),
    "learned pair edges": (False, False, "learned pair edges (2500 shots)"),
}
SIM = {  # name: (shots learned from, scaled, boundary edges learned too)
    "learned (2500 shots)": (LEARN_SHOTS, False, True),
    "learned, scaled (2500 shots)": (LEARN_SHOTS, True, True),
    "learned pair edges (2500 shots)": (LEARN_SHOTS, False, False),
    "learned (10,000 shots)": (refit.SIM_SHOTS, False, True),
}


def ratios(learned_p: dict, model_p: dict) -> dict:
    return {
        t: v / model_p[t] for t, v in learned_p.items() if t in model_p and model_p[t] > MODEL_MIN
    }


def job_decode(k: int) -> dict:
    t0 = time.time()
    job = jobs()[k]
    patches = job.analysis["patches"]
    types = result("pij.json")["per_job"][job.name]["types"]
    device = {t: row[5] for t, row in types.items() if row[5] is not None}
    device_ratio = ratios(device, {t: row[4] for t, row in types.items()})
    built = {}
    for i in refit.MEMORY:
        s = SETTINGS[i]
        model, records, labels = job.model(i)
        labels = DT.numbered(labels)
        dets, flips = job.chip_detectors(i, model, records)
        cal = calibrated_matching(model)
        g = DT.graph(cal)
        ids = [DT.type_id(s.basis, key) for key in DT.types(g, labels, s.rounds)[0]]
        sd, so = model.compile_detector_sampler(seed=refit.sim_seed(i)).sample(
            refit.SIM_SHOTS, separate_observables=True
        )
        me = DT.model_estimate(DT.model_pairs(model), g) if s.rounds in USED else None
        built[i] = (model, labels, dets, flips, cal, g, ids, sd, so, me)
    # the control's priors, learned from the model's own shots of the 6- and 8-round circuits
    collected: dict = {LEARN_SHOTS: {}, refit.SIM_SHOTS: {}, "model": {}}
    for i, (_m, labels, _d, _f, _c, g, ids, sd, _so, me) in built.items():
        if SETTINGS[i].rounds in USED:
            for shots in (LEARN_SHOTS, refit.SIM_SHOTS):
                DT.by_type(ids, DT.learned(sd[:shots], g, labels, me), collected[shots])
            DT.by_type(ids, me, collected["model"])
    model_p = DT.type_means(collected["model"])
    sim_values = {}
    for name, (shots, scaled, _b) in SIM.items():
        means = DT.type_means(collected[shots])
        sim_values[name] = ratios(means, model_p) if scaled else means
    fails: dict = {d: {} for d in ("calibrated", *VARIANTS)}
    sim_fails: dict = {d: {} for d in ("calibrated", *SIM)}
    kept = {"edges": 0, "kept_calibrated": 0, "p_sum": 0.0, "p_sum_kept": 0.0}
    for i, (model, _labels, dets, flips, cal, g, ids, sd, so, _me) in built.items():
        nd, no = model.num_detectors, model.num_observables
        f_cal = cal.decode_batch(dets).astype(bool) != flips
        saved = job.analysis["observed"]["settings"][i]["logical_error"]
        assert np.array_equal(f_cal.mean(0), np.array(saved)), (job.name, i)
        fails["calibrated"][i] = f_cal
        for name, (scaled, bound, _c) in VARIANTS.items():
            p, keep = DT.priors(g, ids, device_ratio if scaled else device, scaled, bound)
            m = DT.with_priors(g, p, nd, no)
            fails[name][i] = m.decode_batch(dets).astype(bool) != flips
            if name == "learned":
                kept["edges"] += len(p)
                kept["kept_calibrated"] += int(keep.sum())
                kept["p_sum"] += float(g["p"].sum())
                kept["p_sum_kept"] += float(g["p"][keep].sum())
        sim_fails["calibrated"][i] = cal.decode_batch(sd).astype(bool) != so
        for name, (_shots, scaled, bound) in SIM.items():
            p = DT.priors(g, ids, sim_values[name], scaled, bound)[0]
            sim_fails[name][i] = DT.with_priors(g, p, nd, no).decode_batch(sd).astype(bool) != so
    out = {
        "name": job.name,
        "day": job.day,
        "rep": job.rep,
        "patches": patches,
        "edges": kept,
        "curves": refit.job_curves(fails, patches),
        "boot": refit.job_boot(fails, patches, seed=100 * k),
        "sim_curves": refit.job_curves(sim_fails, patches, profile=False),
        "sim_boot": refit.job_boot(sim_fails, patches, seed=100 * k + 50),
    }
    out["seconds"] = time.time() - t0
    print(f"{job.name}: {out['seconds']:.0f} s", flush=True)
    return out


def gaps() -> dict:
    """Observed (calibrated priors) - predicted eps per placement, basis and fit (plain means
    over the days), from fits.json."""
    per_rep = result("fits.json")["per_rep"]
    out = {}
    for pl in (1, 2):
        rows = [r for r in per_rep.values() if r["placement"] == pl]
        for basis in "XZ":
            for fit in ("wls", "binomial"):
                for e in ("eps3", "eps5"):
                    obs = [
                        r[basis][e]["wls"][0] if fit == "wls" else r[basis][e]["binomial"]
                        for r in rows
                    ]
                    pred = [r[basis]["predicted"][e][fit] for r in rows]
                    out[f"placement {pl} {basis} {fit} {e}"] = {
                        "observed": float(np.mean(obs)),
                        "predicted": float(np.mean(pred)),
                        "gap": float(np.mean(obs) - np.mean(pred)),
                    }
    return out


def compare(diff: dict, gap: dict, control: dict | None = None) -> dict:
    """Each change of eps against its gap: the change and its SE, -change / gap, and with a
    control the same net of the control's change."""
    out = {}
    for key, d in diff.items():
        _, pl, basis, fit = key.split()
        for e in ("eps3", "eps5"):
            g = gap[f"placement {pl} {basis} {fit} {e}"]["gap"]
            row = {"change": d[e][0], "se": d[e][1], "gap": g, "share_of_gap_closed": -d[e][0] / g}
            if control is not None:
                c = control[key][e]
                row["control_change"], row["control_se"] = c
                row["share_closed_net_of_control"] = -(d[e][0] - c[0]) / g
            out[f"{key} {e}"] = row
        out[f"{key} lambda"] = {"change": d["lambda"][0], "se": d["lambda"][1]}
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
    fits = result("fits.json")["per_rep"]  # the calibrated refit reproduces fits.json
    for r in reps:
        for basis in "XZ":
            for e in ("eps3", "eps5"):
                got = r["decoders"]["calibrated"][basis][e]["wls"][0]
                want = fits[f"{r['day']} r{r['placement']}"][basis][e]["wls"][0]
                assert abs(got - want) < 1e-9, (r["day"], basis, e, got, want)
    boots = {n: j["boot"] for n, j in per_job.items()}
    sim_boots = {n: j["sim_boot"] for n, j in per_job.items()}
    gap = gaps()
    control = {
        name: refit.paired_differences(sim_jobs, sim_boots, sim_reps, name, "calibrated")
        for name in SIM
    }
    device = {
        name: compare(
            refit.paired_differences(per_job, boots, reps, name, "calibrated"),
            gap,
            control[ctl],
        )
        for name, (_s, _b, ctl) in VARIANTS.items()
    }
    edges = {
        "edges": sum(j["edges"]["edges"] for j in done),
        "kept_calibrated": sum(j["edges"]["kept_calibrated"] for j in done),
        "share_of_calibrated_p_on_kept_edges": sum(j["edges"]["p_sum_kept"] for j in done)
        / sum(j["edges"]["p_sum"] for j in done),
    }
    out = {
        "method": __doc__,
        "edges_keeping_the_calibrated_p": edges,
        "gaps": gap,
        "device_minus_calibrated [variant][placement basis fit quantity]": device,
        "control_minus_calibrated [variant][placement basis fit quantity]": {
            name: compare(c, gap) for name, c in control.items()
        },
        "placements": refit.placements(reps),
        "control_placements": refit.placements(sim_reps),
        "reps": reps,
        "seconds": time.time() - t0,
    }
    write_json("pij_decode.json", out)
    print(f"edges keeping the calibrated p: {edges}")
    for name, rows in device.items():
        print(name)
        for k, v in rows.items():
            if "lambda" in k:
                print(f"  {k}: {v['change']:+.3f} +- {v['se']:.3f}")
            else:
                print(
                    f"  {k}: {100 * v['change']:+.2f} +- {100 * v['se']:.2f} pp (gap {100 * v['gap']:.2f}),"
                    f" closes {100 * v['share_of_gap_closed']:.0f}%; control {100 * v['control_change']:+.2f}"
                    f" +- {100 * v['control_se']:.2f}, net {100 * v['share_closed_net_of_control']:.0f}%"
                )
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

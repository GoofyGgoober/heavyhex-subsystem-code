"""The calibration model with Harper et al.'s mid-circuit reset and readout errors, on day 1 (Sec. leakage, resets and readout).

The model gives each reset and each readout the qubit's reported readout error. Ref. harper2026
fitted a noise model to Heron r2 data and found a reset error of 7.5% and a classical readout
error of 4.2%. Here every reset of day 1's three jobs gets 7.5% and every mid-circuit readout
4.2% (the final readout of the data qubits keeps its reported error); the rest of the model is
unchanged, at each job's measured f. 10,000 shots per memory circuit (refit.sim_seed) are
decoded with the calibration model's priors, as the device's shots are, and fitted with both
fits. Against the device: eps, the detection-event fraction per detector class (8 rounds), and
the time-like edges (a stabilizer's consecutive rounds, its mid-circuit readout and reset),
learned from 2500 of the simulated shots exactly as pij.py learns them from the device's.

Under a minute on 2 workers.

    python docs/analysis/reset_harper.py      # writes results/reset_harper.json
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import stim

from heavyhex.experiment.circuits import SETTINGS

import detectors as DT
import refit
from common import WORKERS, jobs, result, write_json
from matching import calibrated_matching

DAY = "2026-10-04"
PRED = "predicted_at_measured_dephasing"
RESET, READOUT = 0.075, 0.042  # Harper et al., Heron r2
USED = (6, 8)


def harper(model: stim.Circuit, records: dict) -> stim.Circuit:
    """The model with every reset at RESET and every readout but the data's final one at READOUT."""
    data = {k for (reg, _i), k in records.items() if reg.startswith("d")}
    out = stim.Circuit()
    m = 0
    reset: set[int] = set()
    for ins in model.flattened():
        name, targets, args = ins.name, ins.targets_copy(), ins.gate_args_copy()
        if name == "X_ERROR" and reset and {t.value for t in targets} <= reset:
            out.append("X_ERROR", targets, [RESET])
        elif name == "M":
            for t in targets:
                out.append("M", [t], [args[0] if m in data else READOUT])
                m += 1
        else:
            out.append(ins)
        reset = {t.value for t in targets} if name == "R" else set()
    assert m == model.num_measurements
    return out


def job_run(k: int) -> dict:
    t0 = time.time()
    job = jobs()[k]
    patches = job.analysis["patches"]
    fails: dict = {"harper": {}}
    frac: dict = {}
    timelike: dict = {}
    for i in refit.MEMORY:
        s = SETTINGS[i]
        model, records, labels = job.model(i)
        labels = DT.numbered(labels)
        h = harper(model, records)
        sd, so = h.compile_detector_sampler(seed=refit.sim_seed(i)).sample(
            refit.SIM_SHOTS, separate_observables=True
        )
        cal = calibrated_matching(model)
        fails["harper"][i] = cal.decode_batch(sd).astype(bool) != so
        if s.rounds == 8:  # this model, the device and the calibration model, same detectors
            rates = zip(
                sd.mean(0),
                job.analysis["observed"]["settings"][i]["detector_rates"],
                job.analysis[PRED]["settings"][i]["detector_rates"],
            )
            for lab, r in zip(labels, rates):
                frac.setdefault(f"{s.basis} d{lab[0]} {DT.detector_class(lab[2])}", []).append(r)
        if s.rounds in USED:
            g = DT.graph(cal)
            keys, cls, _ = DT.types(g, labels, s.rounds)
            me = DT.model_estimate(DT.model_pairs(model), g)
            est = DT.learned(sd[:2500], g, labels, me)
            for e, key in enumerate(keys):
                if cls[e] == "time-like":
                    timelike.setdefault(DT.type_id(s.basis, key), []).append(est[e])
    out = {
        "name": job.name,
        "day": job.day,
        "rep": job.rep,
        "patches": patches,
        "curves": refit.job_curves(fails, patches, profile=False),
        "fractions": {
            key: dict(zip(("harper", "device", "model"), np.mean(v, axis=0).tolist()))
            for key, v in frac.items()
        },
        "timelike": DT.type_means(timelike),
        "seconds": time.time() - t0,
    }
    print(f"{job.name}: {out['seconds']:.0f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    t0 = time.time()
    picks = [k for k, j in enumerate(jobs()) if j.day == DAY]
    with Pool(args.workers) as pool:
        done = pool.map(job_run, picks)
    per_job = {j["name"]: j for j in done}
    reps = refit.reps(per_job)
    fits = result("fits.json")["per_rep"]
    eps = {}
    for r in reps:
        f = fits[f"{r['day']} r{r['placement']}"]
        for basis in "XZ":
            for e in ("eps3", "eps5"):
                eps[f"placement {r['placement']} {basis} {e}"] = {
                    "device": f[basis][e]["wls"][0],
                    "model": f[basis]["predicted"][e]["wls"],
                    "harper": r["decoders"]["harper"][basis][e]["wls"][0],
                    "harper_binomial": r["decoders"]["harper"][basis][e]["binomial"][0],
                    "device_binomial": f[basis][e]["binomial"],
                    "model_binomial": f[basis]["predicted"][e]["binomial"],
                }
            eps[f"placement {r['placement']} {basis} lambda"] = {
                "device": f[basis]["lambda"]["wls"][0],
                "model": f[basis]["predicted"]["lambda"]["wls"],
                "harper": r["decoders"]["harper"][basis]["lambda"]["wls"][0],
            }
    # detection-event fractions per class (8 rounds): this model, the device, the calibration model
    fractions = {
        f"{key} placement {j['rep']}": v
        for j in per_job.values()
        for key, v in j["fractions"].items()
    }
    # time-like edges: device, model and this model, per type of day 1's jobs
    pij = result("pij.json")["per_job"]
    tl = []
    for name, j in per_job.items():
        for t, v in j["timelike"].items():
            row = pij[name]["types"].get(t)
            if row is not None and row[5] is not None:
                tl.append((row[4], row[5], v))
    a = np.array(tl)
    out = {
        "method": __doc__,
        "reset_error": RESET,
        "mid_circuit_readout_error": READOUT,
        "eps_day1 (wls unless named)": eps,
        "detection_event_fraction_8_rounds": fractions,
        "time_like_edges_day1": {
            "types": len(a),
            "median_model_p": float(np.median(a[:, 0])),
            "median_device_p": float(np.median(a[:, 1])),
            "median_harper_p": float(np.median(a[:, 2])),
            "median_device_over_harper": float(np.median(a[:, 1] / a[:, 2])),
            "share_device_below_harper": float((a[:, 1] < a[:, 2]).mean()),
        },
        "seconds": time.time() - t0,
    }
    write_json("reset_harper.json", out)
    for k, v in eps.items():
        print(f"  {k}: " + ", ".join(f"{n} {x:.3f}" for n, x in v.items()))
    print(out["time_like_edges_day1"])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

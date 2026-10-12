"""Independent simulation checks (own code; uses only the repo's model builder, decoder and
per-round fit as primitives). Local only, no QPU.

A. Redecode the chip's X/Z memory shots of selected jobs with median-f and per-qubit-f
   weights; median-f must reproduce analysis.json 'observed'.
B. Rerun the stored X-memory predictions (median f at 200k, per-qubit f at 50k, seed =
   rounds, as predict() does) for one job to test bit-reproducibility.
C. Mini Monte Carlo of the per-qubit-f X prediction spread under the idle test's own
   per-qubit uncertainties, with an independent RNG, for a few jobs.
"""
import json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
_cache = {}


def ctx(name):
    if name not in _cache:
        from heavyhex.experiment import run as R
        from heavyhex.patches.placement import Calibration
        p = REPO / "runs" / name
        _cache[name] = dict(
            run=json.loads((p / "run.json").read_text()),
            a=json.loads((p / "analysis.json").read_text()),
            cal=Calibration.load(p / "calibration.json"),
            circ=R.load_circuits(p),
        )
    return _cache[name]


def own_f(a):
    return {r["qubit"]: r["dephasing"] for r in a["dephasing_by_qubit"]}


def fit(errors_by_patch, shots):
    from heavyhex.experiment.circuits import ROUNDS
    from heavyhex.simulation.noisy import fit_per_round
    out = {}
    for p, errs in errors_by_patch.items():
        ft = fit_per_round(ROUNDS, errs, shots)
        out[p] = (ft.per_round, ft.uncertainty)
    return out


def redecode(name):
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS
    c = ctx(name)
    a = c["a"]
    shots = R.load_shots(REPO / "runs" / name)
    n = R._memory_shots(shots)
    f = a["decoupled_dephasing"]
    res = {}
    for label, byq in (("median", None), ("pq", own_f(a))):
        errs = {b: {p: [] for p in a["patches"]} for b in "XZ"}
        for i, s in enumerate(SETTINGS):
            if s.kind != "memory":
                continue
            model, records, _ = R.simulated(s, c["circ"][i], c["cal"], c["run"]["dt"], f, byq)
            meas = R.in_record_order(shots[i], records)
            det, obs = model.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
            failed = R._decode(model, det) != obs
            for j, p in enumerate(a["patches"]):
                errs[s.basis][p].append(float(failed[:, j].mean()))
        res[label] = {b: fit(errs[b], n) for b in "XZ"}
    return name, res


def simulate_x(name, base, byq, shots):
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS
    c = ctx(name)
    errs = {p: [] for p in c["a"]["patches"]}
    raw = {}
    for i, s in enumerate(SETTINGS):
        if not (s.kind == "memory" and s.basis == "X"):
            continue
        model, _, _ = R.simulated(s, c["circ"][i], c["cal"], c["run"]["dt"], base, byq)
        det, obs = model.compile_detector_sampler(seed=s.rounds).sample(shots, separate_observables=True)
        failed = R._decode(model, det) != obs
        raw[i] = failed.mean(axis=0).tolist()
        for j, p in enumerate(c["a"]["patches"]):
            errs[p].append(float(failed[:, j].mean()))
    return fit(errs, shots), raw


def repro(task):
    name, which = task
    c = ctx(name)
    a = c["a"]
    f = a["decoupled_dephasing"]
    if which == "median":
        fits, raw = simulate_x(name, f, None, 200_000)
    else:
        fits, raw = simulate_x(name, f, own_f(a), 50_000)
    return name, which, fits, raw


def mc_draw(task):
    name, k = task
    c = ctx(name)
    a = c["a"]
    idle = a["idle_test"]
    rng = np.random.default_rng([777, k, sum(map(ord, name))])
    base = max(idle["median"] + idle["median_uncertainty"] * rng.standard_normal(), 0.0)
    byq = {}
    for r in a["dephasing_by_qubit"]:
        if r["measured"]:
            q = int(r["data"][1:]) - 1
            share = idle["per_qubit"][str(r["patch"])][q]
            byq[r["qubit"]] = max(share + r["uncertainty"] * rng.standard_normal(), 0.0)
        else:
            byq[r["qubit"]] = base
    fits, _ = simulate_x(name, base, byq, 50_000)
    return name, k, fits


if __name__ == "__main__":
    t0 = time.time()
    rd_jobs = ["ibm_fez-2026-10-04-r1", "ibm_fez-2026-10-06-r2-d3", "ibm_fez-2026-10-07-r2-d5", "ibm_fez-2026-10-05-r1"]
    mc_jobs = {"ibm_fez-2026-10-05-r1": 30, "ibm_fez-2026-10-04-r1": 30, "ibm_fez-2026-10-04-r2-d5": 20, "ibm_fez-2026-10-06-r2-d3": 20}
    out = {"redecode": {}, "repro": {}, "mc": {}}
    with Pool(14) as pool:
        rd_async = pool.map_async(redecode, rd_jobs)
        rp_async = pool.map_async(repro, [("ibm_fez-2026-10-04-r2-d3", "median"), ("ibm_fez-2026-10-04-r2-d3", "pq"),
                                          ("ibm_fez-2026-10-08-r2-d5", "pq")])
        mc_tasks = [(n, k) for n, m in mc_jobs.items() for k in range(m)]
        mc_res = pool.map(mc_draw, mc_tasks)
        for name, res in rd_async.get():
            out["redecode"][name] = res
        for name, which, fits, raw in rp_async.get():
            out["repro"][f"{name}|{which}"] = {"fits": fits, "raw": raw}
    for name, k, fits in mc_res:
        out["mc"].setdefault(name, []).append(fits)
    out["seconds"] = time.time() - t0
    (OUT / "verify_sim.json").write_text(json.dumps(out, indent=1))
    print("done", out["seconds"])

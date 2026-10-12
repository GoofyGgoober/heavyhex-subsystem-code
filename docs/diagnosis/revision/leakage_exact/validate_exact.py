"""Validation of exact_sim.py, and of the simulators against each other with leakage on.

  A. The reviewer's counterexample (leaked q0 is |1>, its CZ with |+> partner q1 must be
     absent): flip rate of the observable, 1.0 = the omitted-CZ answer. Original leaky.py,
     corrected frame simulator (leaky_fix, random phase), tableau_ref.py and exact_sim.py.
  B. Leakage off: exact_sim.py's detector rates against stim's own sampler (two circuits).
  C. Random 4-qubit circuits with leakage (leaky_fix/check_fix.py's generator and settings,
     24 circuits; and check_det1.py's circuits whose CZ operands include a deterministic |1>):
     total-variation distance of the joint outcome distribution, exact_sim.py vs
     tableau_ref.py for each channel, against the seed-to-seed distance; and the original
     simulator against the exact "CZ left out".
  D. Real memory circuits at the published leakage parameters (0.2%, 25%, 5%): exact_sim.py
     (20,000 shots) against tableau_ref.py (1,500-4,000 shots) for both channels: detector
     rates (chi2 per detector), decoded logical error per patch, the fit's statistics.

    python validate_exact.py [A B C D]      (default: all)

Writes validate_exact.json next to this file.
"""

from __future__ import annotations

import os

# one BLAS thread per process: OpenBLAS otherwise commits ~0.5 GB per process on Windows,
# which the multiprocessing pools below cannot afford
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
FIX = DIAG / "revision/leaky_fix"
sys.path[:0] = [str(HERE), str(DIAG)]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


orig = _load("leaky_orig", DIAG / "leaky.py")
fixed = _load("leaky_fixed", FIX / "leaky.py")
tableau_ref = _load("tableau_ref", FIX / "tableau_ref.py")
import exact_sim as E  # noqa: E402

CE = """
R 0 1 2
X 0
CZ 0 2
S 1
SQRT_X 1
S 1
CZ 0 1
S 1
SQRT_X 1
S 1
M 1
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def joint(obs):
    k = obs.shape[1]
    idx = obs.astype(np.int64) @ (1 << np.arange(k))
    return np.bincount(idx, minlength=1 << k) / len(obs)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def part_a(shots=4000):
    c = stim.Circuit(CE)
    kw = dict(leak={0: 1.0}, partner_phase=0.0, transfer=0.0)
    out = {
        "original leaky.py": orig.simulate(c, shots, orig.Leakage(**kw), seed=1)[1].mean(),
        "corrected frame (random phase)": fixed.simulate(c, shots, fixed.Leakage(**kw), seed=1)[1].mean(),
        "tableau_ref, CZ left out": tableau_ref.simulate(c, shots, E.Leakage(**kw), "identity", seed=1)[1].mean(),
        "tableau_ref, random phase": tableau_ref.simulate(c, shots, E.Leakage(**kw), "dephase", seed=1)[1].mean(),
        "exact_sim, CZ left out": E.simulate(c, shots, E.Leakage(**kw), "identity", seed=1)[1].mean(),
        "exact_sim, random phase": E.simulate(c, shots, E.Leakage(**kw), "dephase", seed=1)[1].mean(),
        "exact_sim, phase pi": E.simulate(c, shots, E.Leakage(**kw), "pi", seed=1)[1].mean(),
    }
    print("A. counterexample, observable flip rate (1 = the omitted-CZ answer; 0.5 = random phase; 0 = phase pi)")
    for k, v in out.items():
        print(f"   {k:32s} {v:.3f}")
    return {k: float(v) for k, v in out.items()}


def part_b(shots=20000):
    from common import jobs, memory_index
    out = {}
    for name, basis in (("2026-10-08-r2-d3", "Z"), ("2026-10-08-r1", "Z")):
        j = next(j for j in jobs() if j.name == name)
        m, _, _ = j.model(memory_index(basis, 8))
        d, o = E.simulate(m, shots, None, seed=3)
        d2, o2 = m.compile_detector_sampler(seed=4).sample(10 * shots, separate_observables=True)
        p1, p2 = d.mean(0), d2.mean(0)
        se = np.sqrt(p2 * (1 - p2) * (1 / shots + 1 / (10 * shots)))
        z = (p1 - p2) / np.where(se > 0, se, 1)
        so = np.sqrt(o2.mean(0) * (1 - o2.mean(0)) * (1 / shots + 1 / (10 * shots)))
        out[f"{name} {basis}8"] = {"n_detectors": int(len(z)), "mean_z2": float((z ** 2).mean()),
                                   "max_abs_z": float(np.abs(z).max()),
                                   "observable_z": ((o.mean(0) - o2.mean(0)) / so).tolist()}
        print(f"B. leakage off, {name} {basis}8: {len(z)} detectors, mean z^2 {(z ** 2).mean():.2f}, "
              f"max |z| {np.abs(z).max():.2f}; observable z {out[f'{name} {basis}8']['observable_z']}")
    return out


def _c_one(args):
    kind_of_set, i, c_text, kw, shots = args
    c = stim.Circuit(c_text)
    r = {"set": kind_of_set, "circuit": i, "kind": kw.get("partner_kind", "phase"), "strength": kw["partner_phase"],
         "transfer": kw["transfer"]}
    lk = E.Leakage(**kw)
    channels = ("identity", "dephase") if kind_of_set == "random" else ("identity",)
    for ch in channels:
        a = joint(E.simulate(c, shots, lk, ch, seed=i)[1])
        a2 = joint(E.simulate(c, shots, lk, ch, seed=5000 + i)[1])
        t = joint(tableau_ref.simulate(c, shots, lk, ch, seed=9000 + i)[1])
        r[f"TV exact_sim vs tableau_ref ({ch})"] = tv(a, t)
        r[f"TV exact_sim seed noise ({ch})"] = tv(a, a2)
        if ch == "identity":
            ok = orig.Leakage(**kw)
            r["TV original vs exact (identity)"] = tv(joint(orig.simulate(c, shots, ok, seed=i)[1]), a)
        else:
            fk = fixed.Leakage(**kw, skipped_cz="dephase")
            r["TV corrected frame vs exact (dephase)"] = tv(joint(fixed.simulate(c, shots, fk, seed=i)[1]), a)
    # phase pi = CZ left out + Z on the partner = tableau_ref "identity" with a certain Z disturbance
    if kind_of_set == "random" and kw["partner_phase"] == 0.0:
        lk_pi = E.Leakage(**{**kw, "partner_kind": "phase", "partner_phase": 0.0})
        lk_t = E.Leakage(**{**kw, "partner_kind": "phase", "partner_phase": 1.0})
        a = joint(E.simulate(c, shots, lk_pi, "pi", seed=i)[1])
        t = joint(tableau_ref.simulate(c, shots, lk_t, "identity", seed=9000 + i)[1])
        r["TV exact_sim pi vs tableau_ref identity+Z"] = tv(a, t)
    return r


def part_c(shots=6000, procs=3):
    check_fix = _load("check_fix", FIX / "check_fix.py")
    rng = np.random.default_rng(7)
    kinds = ["bitflip", "depolarize", "phase"]
    tasks = []
    for i in range(24):  # exactly check_fix.random_checks' circuits and settings
        c = check_fix.random_circuit(rng)
        kw = dict(leak={q: (0.3 if q == 0 else 0.02) for q in range(4)}, partner_phase=[0.0, 0.25, 0.5, 1.0][i % 4],
                  partner_kind=kinds[i % 3], transfer=0.1 if i % 2 else 0.0, lifetime_us=8.0, half_round_us=1.0,
                  read_leaked=0.5)
        tasks.append(("random", i, str(c), kw, shots))
    # check_det1.py's circuits (rng 11), keeping those with a deterministic |1> CZ operand
    det1 = _load("check_det1_defs", HERE / "_det1_defs.py")
    rng = np.random.default_rng(11)
    n_with = 0
    while n_with < 10:
        c = det1.circ(rng, allow_x=True)
        if det1.det1(c) > 0:
            kw = dict(leak={q: (0.3 if q == 0 else 0.02) for q in range(4)}, partner_phase=0.0, transfer=0.0,
                      lifetime_us=8.0, half_round_us=1.0, read_leaked=0.5)
            tasks.append(("det1", 100 + n_with, str(c), kw, shots))
            n_with += 1
    with mp.Pool(procs) as pool:
        rows = pool.map(_c_one, tasks, chunksize=1)
    summary = {}
    for s in ("random", "det1"):
        rs = [r for r in rows if r["set"] == s]
        keys = sorted({k for r in rs for k in r if k.startswith("TV")})
        summary[s] = {k: {"mean": float(np.mean([r[k] for r in rs if k in r])),
                          "max": float(np.max([r[k] for r in rs if k in r]))} for k in keys}
        print(f"C. {s} circuits ({len(rs)}):")
        for k in keys:
            print(f"   {k:46s} mean {summary[s][k]['mean']:.3f}  max {summary[s][k]['max']:.3f}")
    return {"rows": rows, "summary": summary}


def _d_one(args):
    name, basis, channel, sim, shots, seed = args
    import rerun
    from common import ROUND_US, jobs, memory_index
    from fit_leak2 import stats
    from heavyhex.experiment import run as R
    job = next(j for j in jobs() if j.name == name)
    _, _, life, read = rerun.leakage_params()
    q = rerun.PROPS[job.day]["qubits"]
    lk = E.Leakage(leak=rerun._AllQubits(defaultdict(lambda: 0.002)), lifetime_us=life, half_round_us=ROUND_US / 2,
                   read_leaked=read, partner_phase=0.25, partner_kind="bitflip", transfer=0.05,
                   reset_error={int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None},
                   readout={int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                            if v.get("prob_meas1_prep0") is not None})
    model, records, labels = job.model(memory_index(basis, 8))
    t = time.time()
    if sim == "exact_sim":
        det, obs = E.simulate(model, shots, lk, channel, seed=seed)
    else:
        det, obs = tableau_ref.simulate(model, shots, lk, channel, seed=seed)
    failed = R._decode(model, det) != obs
    return (name, basis, channel, sim), {"shots": shots, "rates": det.mean(0), "logical": failed.mean(0),
                                         "stats": stats(det, labels), "seconds": time.time() - t}


def part_d(procs=3):
    circuits = [("2026-10-08-r2-d3", "Z", 4000), ("2026-10-08-r2-d3", "X", 4000), ("2026-10-08-r2-d5", "Z", 2000),
                ("2026-10-08-r1", "Z", 1500)]
    tasks = []
    for name, basis, n_ref in circuits:
        for ch in ("identity", "dephase"):
            tasks.append((name, basis, ch, "tableau_ref", n_ref, 77))
            tasks.append((name, basis, ch, "exact_sim", 20000, 78))
    tasks.sort(key=lambda t: -t[4] * (3 if "r1" in t[0] else 1))
    with mp.Pool(procs) as pool:
        res = dict(pool.map(_d_one, tasks, chunksize=1))
    out = {}
    for name, basis, _ in circuits:
        for ch in ("identity", "dephase"):
            a, t = res[(name, basis, ch, "exact_sim")], res[(name, basis, ch, "tableau_ref")]
            pa, pt = a["rates"], t["rates"]
            pbar = (pa * a["shots"] + pt * t["shots"]) / (a["shots"] + t["shots"])
            se = np.sqrt(pbar * (1 - pbar) * (1 / a["shots"] + 1 / t["shots"]))
            z = (pa - pt) / np.where(se > 0, se, 1)
            la, lt = a["logical"], t["logical"]
            lbar = (la * a["shots"] + lt * t["shots"]) / (a["shots"] + t["shots"])
            lz = (la - lt) / np.sqrt(lbar * (1 - lbar) * (1 / a["shots"] + 1 / t["shots"]))
            key = f"{name} {basis}8 {ch}"
            out[key] = {"detector_mean_z2": float((z ** 2).mean()), "detector_max_abs_z": float(np.abs(z).max()),
                        "n_detectors": int(len(z)), "logical_exact_sim": la.tolist(), "logical_tableau_ref": lt.tolist(),
                        "logical_z": lz.tolist(), "stats_exact_sim": a["stats"], "stats_tableau_ref": t["stats"],
                        "shots": [a["shots"], t["shots"]],
                        "ms_per_shot": [1000 * a["seconds"] / a["shots"], 1000 * t["seconds"] / t["shots"]]}
            print(f"D. {key}: detectors mean z^2 {(z ** 2).mean():.2f} (max |z| {np.abs(z).max():.1f}); logical "
                  + ", ".join(f"{x:.4f} vs {y:.4f} (z {w:+.1f})" for x, y, w in zip(la, lt, lz))
                  + "; " + "  ".join(f"{k} {a['stats'][k]:.4g}/{t['stats'][k]:.4g}" for k in a["stats"])
                  + f"; ms/shot {out[key]['ms_per_shot'][0]:.1f} vs {out[key]['ms_per_shot'][1]:.1f}", flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parts = sys.argv[1:] or ["A", "B", "C", "D"]
    path = HERE / "validate_exact.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for p in parts:
        out[p] = {"A": part_a, "B": part_b, "C": part_c, "D": part_d}[p]()
        path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

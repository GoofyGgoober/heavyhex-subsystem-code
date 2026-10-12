"""Checks of the leakage simulators, and of how frame_sim.py handles a CZ with a leaked operand.

  A  The three-qubit example: q0 leaked in |1>, its CZ with a |+> partner q1 left out, so
     the measured q1 must flip with probability 1. Flip rate under frame_sim.py
     ("reference" and "dephase"), tableau_ref.py and exact_sim.py (each channel).
  B  Leakage off: exact_sim.py's detector rates against stim's own sampler (two circuits).
  C  Random 4-qubit circuits with leakage (24 circuits over every partner kind, transfer and
     lifetime; and 10 whose CZ operands include a definite |1>): total-variation distance of
     the joint outcome distribution, exact_sim.py against tableau_ref.py for each channel,
     beside the seed-to-seed distance; and frame_sim.py ("reference") against the exact
     "CZ left out".
  D  Real memory circuits at the fitted parameters (0.2%, 25%, 5%): exact_sim.py (20,000
     shots) against tableau_ref.py (1,500-4,000 shots), both channels: detector rates,
     decoded logical error per patch, the fit's statistics.
  E  Where "reference" can differ from the CZ left out: through the leaked qubit's Z value in
     the noiseless reference only. The CZ operands of the fit jobs' 1- and 8-round memory
     circuits whose Z value is definitely 0, definitely 1, or random (TableauSimulator.peek_z).
  F  frame_sim.py's "dephase" against tableau_ref.py's random phase on C's 24 circuits (the
     same up to shot noise), and "reference" against both; with no leakage the two handlings
     give identical output.
  X  exact_sim.py against tableau_ref.py on the 8-round circuits of 2026-10-08-r1 at (0.1%,
     25%, 5%), 8000 shots each, independent random numbers: detector rates and the fit's
     statistics with shot-bootstrap errors (the simulations are cached).

A and E take seconds, B and F a minute or two; C, D and X are heavy (tableau_ref.py is slow).
tableau_ref.py draws its measurement outcomes from stim, whose seeded sampling differs between
stim builds, so C, D, F and X repeat on another machine only within their sampling noise.

    python docs/analysis/leakage_checks.py [A B C D E F X] [--workers 2]    (default: all)

Writes (updates) results/leakage_checks.json.
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits about 0.5 GB per process)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import stim

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS

import exact_sim as E
import frame_sim
import leakage
import tableau_ref
from common import RESULTS, ROUND_US, WORKERS, jobs, memory_index
from leakage import FIT_JOBS, AllQubits, hmm_inputs, ibm_errors, stats

PATH = RESULTS / "leakage_checks.json"
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


def reference(**kw) -> frame_sim.Leakage:
    return frame_sim.Leakage(**kw, skipped_cz="reference")


def dephase(**kw) -> frame_sim.Leakage:
    return frame_sim.Leakage(**kw, skipped_cz="dephase")


def random_circuit(rng, n=4, layers=8):
    """A random small Clifford circuit with CZ layers, mid-circuit readouts and resets."""
    c = stim.Circuit()
    c.append("R", range(n))
    for q in range(n):
        if rng.random() < 0.5:
            c.append("S", [q])
            c.append("SQRT_X", [q])
            c.append("S", [q])  # H
    nm = 0
    for _ in range(layers):
        for q in range(n):
            g = rng.choice(["I", "S", "SQRT_X", "X", "H"])
            if g == "H":
                c.append("S", [q])
                c.append("SQRT_X", [q])
                c.append("S", [q])
            elif g != "I":
                c.append(g, [q])
        perm = rng.permutation(n)
        pairs = [int(x) for x in perm[: 2 * (n // 2)]]
        c.append("CZ", pairs)
        c.append("DEPOLARIZE2", pairs, [0.01])
        if rng.random() < 0.3:
            q = int(rng.integers(n))
            c.append("M", [q], [0.01])
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm])
            nm += 1
            c.append("R", [q])
            c.append("X_ERROR", [q], [0.01])
        c.append("TICK")
    for q in range(n):
        if rng.random() < 0.5:
            c.append("S", [q])
            c.append("SQRT_X", [q])
            c.append("S", [q])
        c.append("M", [q])
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm])
        nm += 1
    return c


def definite_one_circuit(rng, allow_x):
    """A random 4-qubit circuit for C's second set (kept when a CZ operand is a definite |1>)."""
    c = stim.Circuit()
    n = 4
    c.append("R", range(n))
    nm = 0
    for _ in range(8):
        for q in range(n):
            g = rng.choice(["I", "S", "SQRT_X", "H"] + (["X"] if allow_x else []))
            if g == "H":
                c.append("S", [q])
                c.append("SQRT_X", [q])
                c.append("S", [q])
            elif g != "I":
                c.append(g, [q])
        pairs = [int(x) for x in rng.permutation(n)]
        c.append("CZ", pairs)
        c.append("DEPOLARIZE2", pairs, [0.01])
        if rng.random() < 0.3:
            q = int(rng.integers(n))
            c.append("M", [q], [0.01])
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm])
            nm += 1
            c.append("R", [q])
        c.append("TICK")
    for q in range(n):
        c.append("M", [q])
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm])
        nm += 1
    return c


def definite_ones(c) -> int:
    """CZ operands whose Z value is a definite 1 in the noiseless circuit."""
    sim = stim.TableauSimulator()
    k = 0
    for ins in c.without_noise().flattened():
        if ins.name == "CZ":
            k += sum(sim.peek_z(t.value) == -1 for t in ins.targets_copy())
        if ins.name not in ("OBSERVABLE_INCLUDE", "TICK"):
            sim.do(ins)
    return k


def random_settings(i: int) -> dict:
    kinds = ["bitflip", "depolarize", "phase"]
    return {
        "leak": {q: (0.3 if q == 0 else 0.02) for q in range(4)},
        "partner_phase": [0.0, 0.25, 0.5, 1.0][i % 4],
        "partner_kind": kinds[i % 3],
        "transfer": 0.1 if i % 2 else 0.0,
        "lifetime_us": 8.0,
        "half_round_us": 1.0,
        "read_leaked": 0.5,
    }


# ----------------------------------------------------------------------------- A
def part_a(workers, shots=4000):
    c = stim.Circuit(CE)
    kw = {"leak": {0: 1.0}, "partner_phase": 0.0, "transfer": 0.0}
    out = {
        "frame_sim, reference": frame_sim.simulate(c, shots, reference(**kw), seed=1)[1].mean(),
        "frame_sim, random phase": frame_sim.simulate(c, shots, dephase(**kw), seed=1)[1].mean(),
        "tableau_ref, CZ left out": tableau_ref.simulate(
            c, shots, E.Leakage(**kw), "identity", seed=1
        )[1].mean(),
        "tableau_ref, random phase": tableau_ref.simulate(
            c, shots, E.Leakage(**kw), "dephase", seed=1
        )[1].mean(),
        "exact_sim, CZ left out": E.simulate(c, shots, E.Leakage(**kw), "identity", seed=1)[
            1
        ].mean(),
        "exact_sim, random phase": E.simulate(c, shots, E.Leakage(**kw), "dephase", seed=1)[
            1
        ].mean(),
        "exact_sim, phase pi": E.simulate(c, shots, E.Leakage(**kw), "pi", seed=1)[1].mean(),
    }
    print(
        "A. the three-qubit example, observable flip rate (1 = CZ left out; 0.5 = random phase; 0 = phase pi)"
    )
    for k, v in out.items():
        print(f"   {k:32s} {v:.3f}")
    return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- B
def part_b(workers, shots=20000):
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
        out[f"{name} {basis}8"] = {
            "n_detectors": len(z),
            "mean_z2": float((z**2).mean()),
            "max_abs_z": float(np.abs(z).max()),
            "observable_z": ((o.mean(0) - o2.mean(0)) / so).tolist(),
        }
        print(
            f"B. leakage off, {name} {basis}8: {len(z)} detectors, mean z^2 {(z**2).mean():.2f}, "
            f"max |z| {np.abs(z).max():.2f}; observable z {out[f'{name} {basis}8']['observable_z']}"
        )
    return out


# ----------------------------------------------------------------------------- C
def _c_one(args):
    kind_of_set, i, c_text, kw, shots = args
    c = stim.Circuit(c_text)
    r = {
        "set": kind_of_set,
        "circuit": i,
        "kind": kw.get("partner_kind", "phase"),
        "strength": kw["partner_phase"],
        "transfer": kw["transfer"],
    }
    lk = E.Leakage(**kw)
    channels = ("identity", "dephase") if kind_of_set == "random" else ("identity",)
    for ch in channels:
        a = joint(E.simulate(c, shots, lk, ch, seed=i)[1])
        a2 = joint(E.simulate(c, shots, lk, ch, seed=5000 + i)[1])
        t = joint(tableau_ref.simulate(c, shots, lk, ch, seed=9000 + i)[1])
        r[f"TV exact_sim vs tableau_ref ({ch})"] = tv(a, t)
        r[f"TV exact_sim seed noise ({ch})"] = tv(a, a2)
        if ch == "identity":
            r["TV frame reference vs exact (identity)"] = tv(
                joint(frame_sim.simulate(c, shots, reference(**kw), seed=i)[1]), a
            )
        else:
            r["TV frame random phase vs exact (dephase)"] = tv(
                joint(frame_sim.simulate(c, shots, dephase(**kw), seed=i)[1]), a
            )
    # phase pi = CZ left out + Z on the partner = tableau_ref "identity" with a certain Z disturbance
    if kind_of_set == "random" and kw["partner_phase"] == 0.0:
        lk_pi = E.Leakage(**{**kw, "partner_kind": "phase", "partner_phase": 0.0})
        lk_t = E.Leakage(**{**kw, "partner_kind": "phase", "partner_phase": 1.0})
        a = joint(E.simulate(c, shots, lk_pi, "pi", seed=i)[1])
        t = joint(tableau_ref.simulate(c, shots, lk_t, "identity", seed=9000 + i)[1])
        r["TV exact_sim pi vs tableau_ref identity+Z"] = tv(a, t)
    return r


def part_c(workers, shots=6000):
    rng = np.random.default_rng(7)
    tasks = [("random", i, str(random_circuit(rng)), random_settings(i), shots) for i in range(24)]
    # circuits with a definite |1> CZ operand (rng 11)
    rng = np.random.default_rng(11)
    n_with = 0
    while n_with < 10:
        c = definite_one_circuit(rng, allow_x=True)
        if definite_ones(c) > 0:
            kw = {
                "leak": {q: (0.3 if q == 0 else 0.02) for q in range(4)},
                "partner_phase": 0.0,
                "transfer": 0.0,
                "lifetime_us": 8.0,
                "half_round_us": 1.0,
                "read_leaked": 0.5,
            }
            tasks.append(("det1", 100 + n_with, str(c), kw, shots))
            n_with += 1
    with mp.Pool(workers) as pool:
        rows = pool.map(_c_one, tasks, chunksize=1)
    summary = {}
    for s in ("random", "det1"):
        rs = [r for r in rows if r["set"] == s]
        keys = sorted({k for r in rs for k in r if k.startswith("TV")})
        summary[s] = {
            k: {
                "mean": float(np.mean([r[k] for r in rs if k in r])),
                "max": float(np.max([r[k] for r in rs if k in r])),
            }
            for k in keys
        }
        print(f"C. {s} circuits ({len(rs)}):")
        for k in keys:
            print(f"   {k:46s} mean {summary[s][k]['mean']:.3f}  max {summary[s][k]['max']:.3f}")
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------- D
def _d_one(args):
    name, basis, channel, sim, shots, seed = args
    job = next(j for j in jobs() if j.name == name)
    life, read = hmm_inputs()
    reset, readout = ibm_errors(job.day)
    lk = E.Leakage(
        leak=AllQubits(defaultdict(lambda: 0.002)),
        lifetime_us=life,
        half_round_us=ROUND_US / 2,
        read_leaked=read,
        partner_phase=0.25,
        partner_kind="bitflip",
        transfer=0.05,
        reset_error=reset,
        readout=readout,
    )
    model, _records, labels = job.model(memory_index(basis, 8))
    t = time.time()
    if sim == "exact_sim":
        det, obs = E.simulate(model, shots, lk, channel, seed=seed)
    else:
        det, obs = tableau_ref.simulate(model, shots, lk, channel, seed=seed)
    failed = R._decode(model, det) != obs
    return (name, basis, channel, sim), {
        "shots": shots,
        "rates": det.mean(0),
        "logical": failed.mean(0),
        "stats": stats(det, labels),
        "seconds": time.time() - t,
    }


def part_d(workers):
    circuits = [
        ("2026-10-08-r2-d3", "Z", 4000),
        ("2026-10-08-r2-d3", "X", 4000),
        ("2026-10-08-r2-d5", "Z", 2000),
        ("2026-10-08-r1", "Z", 1500),
    ]
    tasks = []
    for name, basis, n_ref in circuits:
        for ch in ("identity", "dephase"):
            tasks.append((name, basis, ch, "tableau_ref", n_ref, 77))
            tasks.append((name, basis, ch, "exact_sim", 20000, 78))
    tasks.sort(key=lambda t: -t[4] * (3 if "r1" in t[0] else 1))
    with mp.Pool(workers) as pool:
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
            out[key] = {
                "detector_mean_z2": float((z**2).mean()),
                "detector_max_abs_z": float(np.abs(z).max()),
                "n_detectors": len(z),
                "logical_exact_sim": la.tolist(),
                "logical_tableau_ref": lt.tolist(),
                "logical_z": lz.tolist(),
                "stats_exact_sim": a["stats"],
                "stats_tableau_ref": t["stats"],
                "shots": [a["shots"], t["shots"]],
                "ms_per_shot": [1000 * a["seconds"] / a["shots"], 1000 * t["seconds"] / t["shots"]],
            }
            print(
                f"D. {key}: detectors mean z^2 {(z**2).mean():.2f} (max |z| {np.abs(z).max():.1f}); logical "
                + ", ".join(f"{x:.4f} vs {y:.4f} (z {w:+.1f})" for x, y, w in zip(la, lt, lz))
                + "; "
                + "  ".join(f"{k} {a['stats'][k]:.4g}/{t['stats'][k]:.4g}" for k in a["stats"])
                + f"; ms/shot {out[key]['ms_per_shot'][0]:.1f} vs {out[key]['ms_per_shot'][1]:.1f}",
                flush=True,
            )
    return out


# ----------------------------------------------------------------------------- E
def part_e(workers):
    circuits, total = {}, Counter()
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        for basis in "XZ":
            for rounds in (ROUNDS[0], ROUNDS[-1]):
                model, _, _ = j.model(memory_index(basis, rounds))
                sim = stim.TableauSimulator()
                c = Counter()
                for ins in model.without_noise().flattened():
                    if ins.name == "CZ":
                        for q in (x.value for x in ins.targets_copy()):
                            c[{1: "definite 0", -1: "definite 1", 0: "random"}[sim.peek_z(q)]] += 1
                    if ins.name in (
                        "DETECTOR",
                        "OBSERVABLE_INCLUDE",
                        "QUBIT_COORDS",
                        "SHIFT_COORDS",
                        "TICK",
                    ):
                        continue
                    sim.do(ins)
                total.update(c)
                circuits[f"{j.name} {basis}{rounds}"] = dict(c)
                print(f"E. {j.name} {basis}{rounds}: {dict(c)}", flush=True)
    print(f"E. CZ operands, all: {dict(total)}")
    return {"circuits": circuits, "total": dict(total)}


# ----------------------------------------------------------------------------- F
def part_f(workers, ncirc=24, shots=6000, example_shots=4000):
    c = stim.Circuit(CE)
    kw = {"leak": {0: 1.0}, "partner_phase": 0.0, "transfer": 0.0}
    n = example_shots
    out = {
        "counterexample": {
            "frame_sim, reference": float(
                frame_sim.simulate(c, n, reference(**kw), seed=1)[1].mean()
            ),
            "frame_sim, random phase": float(
                frame_sim.simulate(c, n, dephase(**kw), seed=1)[1].mean()
            ),
            "exact, CZ left out": float(
                tableau_ref.simulate(c, n, dephase(**kw), "identity", seed=1)[1].mean()
            ),
            "exact, CZ -> random phase": float(
                tableau_ref.simulate(c, n, dephase(**kw), "dephase", seed=1)[1].mean()
            ),
        }
    }
    c = random_circuit(np.random.default_rng(3))
    d1 = frame_sim.simulate(c, 2000, reference(), seed=5)
    d2 = frame_sim.simulate(c, 2000, dephase(), seed=5)
    out["no_leak_identical"] = bool((d1[1] == d2[1]).all())
    rng = np.random.default_rng(7)
    rows = []
    for i in range(ncirc):
        c = random_circuit(rng)
        kw = random_settings(i)
        a = joint(frame_sim.simulate(c, shots, reference(**kw), seed=i)[1])
        b = joint(frame_sim.simulate(c, shots, dephase(**kw), seed=i)[1])
        b2 = joint(frame_sim.simulate(c, shots, dephase(**kw), seed=1000 + i)[1])
        ident = joint(tableau_ref.simulate(c, shots, dephase(**kw), "identity", seed=i)[1])
        deph = joint(tableau_ref.simulate(c, shots, dephase(**kw), "dephase", seed=i)[1])
        rows.append(
            {
                "circuit": i,
                "kind": kw["partner_kind"],
                "strength": kw["partner_phase"],
                "transfer": kw["transfer"],
                "TV frame random phase vs exact-dephase": tv(b, deph),
                "TV frame random phase seed noise": tv(b, b2),
                "TV frame reference vs exact-dephase": tv(a, deph),
                "TV frame reference vs exact-identity": tv(a, ident),
                "TV exact-identity vs exact-dephase": tv(ident, deph),
            }
        )
        print(
            f"F. circuit {i:2d}: "
            + "  ".join(f"{k} {v:.3f}" for k, v in rows[-1].items() if k.startswith("TV")),
            flush=True,
        )
    out["random"] = rows
    keys = [k for k in rows[0] if k.startswith("TV")]
    out["random_mean"] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["random_max"] = {k: float(np.max([r[k] for r in rows])) for k in keys}
    return out


# ----------------------------------------------------------------------------- X
def part_x(workers, job="2026-10-08-r1"):
    life, read = hmm_inputs()
    pt = leakage.Point(0.001, 0.25, 0.05, life, read)
    for sim in ("x", "t"):  # exact_sim.py, then tableau_ref.py
        os.environ["LEAKFIT_SIM"] = leakage.SIM = sim
        leakage.ensure(
            [("det", job, b, 8, pt, leakage.GRID_SHOTS, leakage.GRID_SEED) for b in "XZ"], workers
        )
    os.environ["LEAKFIT_SIM"] = leakage.SIM = "x"
    labels = {c: lab for c, (_, lab) in leakage.device_cases().items()}
    out = {}
    for b in "XZ":
        x = leakage.load(job, b, 8, pt, sim="x")
        t = leakage.load(job, b, 8, pt, sim="t")
        px, pt_ = x.mean(0), t.mean(0)
        pool = (px + pt_) / 2
        se = np.sqrt(pool * (1 - pool) * (1 / len(x) + 1 / len(t)))
        z = (px - pt_) / np.where(se > 0, se, np.inf)
        sx, st = leakage.stats_full(x, labels[(job, b)]), leakage.stats_full(t, labels[(job, b)])
        keys = [k for k in sx if k != "profile"]
        rng = np.random.default_rng(0)  # shot-bootstrap standard errors of each statistic
        se = {}
        for name, arr in (("x", x), ("t", t)):
            pr = leakage.Prepared(arr, labels[(job, b)])
            reps = [
                pr.stats(np.bincount(rng.integers(0, len(arr), len(arr)), minlength=len(arr)))
                for _ in range(200)
            ]
            se[name] = {k: float(np.std([r[k] for r in reps], ddof=1)) for k in keys}
        zstat = {
            k: (sx[k] - st[k]) / np.hypot(se["x"][k], se["t"][k])
            for k in keys
            if np.hypot(se["x"][k], se["t"][k]) > 0
        }
        out[b] = {
            "detectors": len(z),
            "chi2_per_detector": float(np.mean(z**2)),
            "max_abs_z": float(np.abs(z).max()),
            "mean_rate_exact_sim": float(px.mean()),
            "mean_rate_tableau": float(pt_.mean()),
            "stats_exact_sim": {k: sx[k] for k in keys},
            "stats_tableau": {k: st[k] for k in keys},
            "stats_z": zstat,
        }
        print(
            f"X. {job} {b}8: {len(z)} detectors, chi2/detector {np.mean(z**2):.2f} (1 expected), "
            f"max |z| {np.abs(z).max():.1f}; mean rate {px.mean():.4f} vs {pt_.mean():.4f}"
        )
    return out


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f, "X": part_x}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("parts", nargs="*", help="any of " + " ".join(PARTS) + " (default: all)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if unknown := set(a.parts) - set(PARTS):
        ap.error(f"unknown parts {sorted(unknown)}")
    out = json.loads(PATH.read_text()) if PATH.exists() else {}
    for p in a.parts or PARTS:
        out[p] = PARTS[p](a.workers)
        PATH.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

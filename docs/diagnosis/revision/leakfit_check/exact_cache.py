"""Exact leakage simulations for the leak-fit checks, run in parallel and cached on disk.

Simulator: revision/leakage_exact/exact_sim.py (SIM = "x", the default), the exact per-shot
stabilizer simulation of the paper's leakage model written for the exact-model grid fit
(validated there against leaky_fix/tableau_ref.py), in "identity" mode: a CZ with a leaked
operand does not act. It is reused here unchanged, so the synthetic data and reruns of
these checks come from the same simulator as leakage_exact's grid. SIM = "t" uses
tableau_fast.py instead (bit-identical to leaky_fix/tableau_ref.py, about 10x slower);
its outputs are kept apart (cache/<kind>/ vs cache/<kind>_x/) and serve as a cross-check.

The model is built exactly as leaky_fix/fit_leak2_fix.py and leakage_exact/grid_exact.py
build it: every qubit leaks with the same probability per CZ, the partner of a leaked CZ
gets an X error with probability `strength` and leaks with probability `spread`, the
leaked qubit returns after an exponential lifetime and reads 1 with probability `read`,
and each reset and readout has IBM's reported error for the job's day. This module only
schedules runs and keeps their outputs.

A run is one circuit (job, basis, rounds) at one parameter point, shot count and seed.
Detector bits go to cache/ (git-ignored, regenerated on demand); a held-out rerun task
instead decodes the shots and keeps only the logical failure counts.

    from exact_cache import Point, ensure, load
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits ~0.5 GB per process on Windows)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
DIAG = REV.parent
FIX = REV / "leaky_fix"
# leaky_fix first, so that rerun.py's `import leaky` gets the fixed simulator, as in fit_leak2_fix.py
for p in (str(DIAG), str(FIX), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
sys.path[:0] = [str(HERE), str(FIX), str(DIAG)]

CACHE = HERE / "cache"
LEAK_EXACT = REV / "leakage_exact"
SIM = os.environ.get("LEAKFIT_SIM", "x")  # "x": leakage_exact/exact_sim.py; "t": tableau_fast.py
FIT_JOBS = ("2026-10-08-r1", "2026-10-08-r2-d5", "2026-10-08-r2-d3", "2026-10-06-r1")  # fit_leak2.py
GRID_SHOTS, GRID_SEED = 8000, 21  # fit_leak2.py / fit_leak2_fix.py


@dataclass(frozen=True)
class Point:
    leak: float  # per CZ, every qubit
    strength: float  # P(partner of a leaked CZ gets an X error)
    spread: float  # P(partner of a leaked CZ leaks)
    life: float  # leaked lifetime, us
    read: float  # P(leaked qubit reads 1)

    def tag(self) -> str:
        return (f"L{self.leak:g}_S{self.strength:g}_T{self.spread:g}"
                f"_life{self.life:.4f}_read{self.read:.4f}")


def hmm_inputs(jobs_subset=None) -> tuple[float, float]:
    """(lifetime us, leaked readout) as rerun.leakage_params computes them, optionally over a subset of jobs."""
    import json
    from common import REPO, ROUND_US
    hmm = json.loads((REPO / "docs/diagnosis/results/leakage_hmm.json").read_text())
    if jobs_subset is not None:
        hmm = [r for r in hmm if r["job"] in jobs_subset]
    life = float(np.median([1 / max(r["chip"]["b"], 1e-6) for r in hmm])) * ROUND_US
    read = float(np.median([r["chip"]["e_L"] for r in hmm]))
    return life, read


# ---------------------------------------------------------------- worker side
_JOBS: dict = {}
_MODELS: dict = {}


def _job(name):
    from common import jobs
    if name not in _JOBS:
        _JOBS[name] = next(j for j in jobs() if j.name == name)
    return _JOBS[name]


def _model(name, basis, rounds):
    from common import memory_index
    key = (name, basis, rounds)
    if key not in _MODELS:
        _MODELS[key] = _job(name).model(memory_index(basis, rounds))
    return _MODELS[key]


def _leakage(name, pt: Point):
    import leaky
    import rerun
    from common import ROUND_US
    assert Path(leaky.__file__).resolve().parent == FIX, leaky.__file__
    job = _job(name)
    q = rerun.PROPS[job.day]["qubits"]
    return leaky.Leakage(
        leak=rerun._AllQubits(defaultdict(lambda: pt.leak)), lifetime_us=pt.life,
        half_round_us=ROUND_US / 2, read_leaked=pt.read, partner_phase=pt.strength,
        partner_kind="bitflip", transfer=pt.spread,
        reset_error={int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None},
        readout={int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                 if v.get("prob_meas1_prep0") is not None},
        skipped_cz="dephase")  # unused by tableau_ref; set as fit_leak2_fix.py sets it


def path_of(kind, name, basis, rounds, pt: Point, shots, seed, sim: str | None = None) -> Path:
    sim = SIM if sim is None else sim
    return CACHE / (kind if sim == "t" else f"{kind}_{sim}") / f"{name}_{basis}{rounds}_{pt.tag()}_n{shots}_s{seed}.npz"


_PROGS: dict = {}


def _simulate(name, basis, rounds, pt: Point, shots, seed):
    model, records, labels = _model(name, basis, rounds)
    lk = _leakage(name, pt)
    if SIM == "t":
        import tableau_fast  # bit-identical to leaky_fix/tableau_ref.simulate (tableau_fast.selftest)
        return tableau_fast.simulate(model, shots, lk, "identity", seed=seed)
    if str(LEAK_EXACT) not in sys.path:
        sys.path.append(str(LEAK_EXACT))
    import exact_sim
    xl = exact_sim.Leakage(**{k: getattr(lk, k) for k in exact_sim.Leakage.__dataclass_fields__})
    key = (name, basis, rounds)  # the program depends on the leakage only through the job's reset errors
    if key not in _PROGS:
        _PROGS[key] = exact_sim.Program(model, xl)
    return exact_sim.simulate(model, shots, xl, "identity", seed=seed, program=_PROGS[key])


def _run(task):
    kind, name, basis, rounds, pt, shots, seed = task
    out = path_of(kind, name, basis, rounds, pt, shots, seed)
    if out.exists():
        return str(out), 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:  # another run (another script) may be simulating the same task
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        if time.time() - lock.stat().st_mtime < 3 * 3600:
            return str(out), -1.0
    t0 = time.time()
    model, records, labels = _model(name, basis, rounds)
    det, obs = _simulate(name, basis, rounds, pt, shots, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with open(part, "wb") as fh:
        if kind == "det":
            np.savez_compressed(fh, det=np.packbits(det, axis=1), n=det.shape[1])
        elif kind == "fail":  # decoded as rerun.py / rerun_fix.py decode
            from heavyhex.experiment import run as R
            failed = R._decode(model, det) != obs
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            np.savez_compressed(fh, failures=failed.sum(0), patches=np.array(patches), shots=shots)
        else:
            raise ValueError(kind)
    os.replace(part, out)
    lock.unlink(missing_ok=True)
    return str(out), time.time() - t0


def ensure(tasks, workers: int = 8) -> None:
    """Run every (kind, job, basis, rounds, Point, shots, seed) task whose output is missing."""
    todo = [t for t in dict.fromkeys(tasks) if not path_of(*t).exists()]
    if not todo:
        return
    # longest first: 8-round, two-patch jobs
    todo.sort(key=lambda t: -(t[3] * t[5] * (2 if t[1].endswith("r1") else 1)))
    print(f"exact simulations: {len(todo)} runs on {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers, maxtasksperchild=6) as pool:  # recycled workers keep memory bounded
        for i, (p, dt) in enumerate(pool.imap_unordered(_run, todo, chunksize=1), 1):
            if i % 8 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done ({time.time()-t0:.0f}s; last {dt:.0f}s {Path(p).name})", flush=True)
    # tasks another run had locked: wait for them
    while missing := [t for t in todo if not path_of(*t).exists()]:
        print(f"  waiting for {len(missing)} runs locked by another process", flush=True)
        time.sleep(60)


def load(name, basis, rounds, pt: Point, shots=GRID_SHOTS, seed=GRID_SEED, sim=None) -> np.ndarray:
    z = np.load(path_of("det", name, basis, rounds, pt, shots, seed, sim))
    return np.unpackbits(z["det"], axis=1, count=int(z["n"])).astype(bool)


def load_fail(name, basis, rounds, pt: Point, shots, seed) -> dict:
    z = np.load(path_of("fail", name, basis, rounds, pt, shots, seed))
    return {str(p): int(k) for p, k in zip(z["patches"], z["failures"])}

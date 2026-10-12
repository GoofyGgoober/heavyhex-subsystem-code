"""fit_leak2.py's grid fit, rerun with the fixed simulator (leaky_fix/leaky.py).

Same four jobs, same six detection statistics (fit_leak2.stats), same 8000 shots and
seed, same loss. Only the simulator's skipped-CZ channel differs:

    python fit_leak2_fix.py KIND MODE GRID
      KIND  bitflip | depolarize          (partner disturbance, as in fit_leak2.py)
      MODE  dephase | reference           (reference reproduces the original numbers)
            | exact-identity | exact-dephase  (tableau_ref.py: CZ left out / random phase)
      GRID  paper | wide                  (paper: the published grid; wide: extended)

Writes fit_leak2_<KIND>_<MODE>_<GRID>.json next to this file.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(DIAG)]

GRIDS = {
    "paper": {"bitflip": ((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05)),
              "depolarize": ((0.001, 0.002, 0.003, 0.0045), (0.25, 0.5, 1.0), (0.0, 0.05, 0.15))},
    "wide": {k: ((0.0005, 0.001, 0.0015, 0.002, 0.003), (0.0, 0.1, 0.25, 0.5, 1.0), (0.0, 0.05, 0.15))
             for k in ("bitflip", "depolarize")},
}
SHOTS = 8000
_CASES = None


def _init():
    global _CASES
    import leaky
    assert Path(leaky.__file__).resolve().parent == HERE, leaky.__file__
    import rerun
    from common import ROUND_US, jobs, memory_index
    from fit_leak2 import FIT_JOBS
    hmm = json.loads((DIAG / "results/leakage_hmm.json").read_text())
    life = float(np.median([1 / max(r["chip"]["b"], 1e-6) for r in hmm])) * ROUND_US
    read = float(np.median([r["chip"]["e_L"] for r in hmm]))
    _CASES = []
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        ro = rerun.variant(j, "leak+ro", *rerun.leakage_params(), 1.0)
        for basis in "XZ":
            model, records, labels = j.model(memory_index(basis, 8))
            _CASES.append((j.name, basis, model, labels, ro.reset_error, ro.readout, life, read, ROUND_US))
    assert len(_CASES) == NCASES, len(_CASES)


def _case(args):
    kind, mode, leak, strength, spread, ci = args
    import leaky
    import rerun
    from fit_leak2 import stats
    name, basis, model, labels, reset_error, readout, life, read, round_us = _CASES[ci]
    lk = leaky.Leakage(leak=rerun._AllQubits(defaultdict(lambda: leak)), lifetime_us=life,
                       half_round_us=round_us / 2, read_leaked=read, partner_phase=strength,
                       partner_kind=kind, transfer=spread, reset_error=reset_error, readout=readout,
                       skipped_cz="dephase" if mode.startswith("exact-") else mode)
    if mode.startswith("exact-"):  # exact stabilizer simulation, tableau_ref.py
        import tableau_ref
        det, _ = tableau_ref.simulate(model, SHOTS, lk, mode.removeprefix("exact-"), seed=21)
    else:
        det, _ = leaky.simulate(model, SHOTS, lk, seed=21)
    return {f"{basis} {k}": v for k, v in stats(det, labels).items()}


# The exact simulation is slow (d=5 circuits ~50 ms per shot); with an exact MODE the
# "paper" grid is cut to the published optimum's neighbourhood (strength 0.25, plus 0.5 at
# leak 0.2%); the strength-1.0 points lost by a factor of 10 or more in every other run.
EXACT_SUBSET = [(0.001, 0.25, 0.0), (0.001, 0.25, 0.05), (0.002, 0.25, 0.0), (0.002, 0.25, 0.05),
                (0.003, 0.25, 0.0), (0.003, 0.25, 0.05), (0.002, 0.5, 0.0), (0.002, 0.5, 0.05)]
NCASES = 8


def main() -> None:
    kind, mode, grid = sys.argv[1], sys.argv[2], sys.argv[3]
    leaks, strengths, spreads = GRIDS[grid][kind]
    chip = json.loads((DIAG / f"results/fit_leak2_{kind}.json").read_text())["chip"]
    pts = list(itertools.product(leaks, strengths, spreads))
    if mode.startswith("exact-") and grid == "paper":
        pts = EXACT_SUBSET
        print("exact mode: paper grid cut to", pts, flush=True)
    tasks = [(kind, mode, *p, ci) for p in pts for ci in range(NCASES)]
    with mp.Pool(12, initializer=_init) as pool:
        per_case = pool.map(_case, tasks, chunksize=1)
    sims = []
    for i in range(len(pts)):
        tot = {}
        for st in per_case[i * NCASES:(i + 1) * NCASES]:
            for k, v in st.items():
                tot.setdefault(k, []).append(v)
        sims.append({k: float(np.nanmean(v)) for k, v in tot.items()})
    results = []
    for (leak, strength, spread), sim in zip(pts, sims):
        rel = {k: (sim[k] - chip[k]) / abs(chip[k]) for k in chip if np.isfinite(chip[k])}
        loss = float(np.sqrt(np.mean([v ** 2 for v in rel.values()])))
        results.append({"leak": leak, "strength": strength, "spread": spread, "loss": loss, "sim": sim, "rel": rel})
        print(f"leak {leak:<6g} strength {strength:<4g} spread {spread:<4g} loss {loss:.3f}  " +
              "  ".join(f"{k} {100*v:+.0f}%" for k, v in sorted(rel.items())), flush=True)
    best = min(results, key=lambda r: r["loss"])
    print(f"\nbest ({kind}, {mode}, {grid}): leak {best['leak']} strength {best['strength']} spread {best['spread']} loss {best['loss']:.3f}")
    print("  " + "  ".join(f"{k} {100*v:+.0f}%" for k, v in sorted(best["rel"].items())))
    (HERE / f"fit_leak2_{kind}_{mode}_{grid}.json").write_text(
        json.dumps({"kind": kind, "mode": mode, "grid": grid, "chip": chip, "points": results, "best": best}, indent=1) + "\n")


if __name__ == "__main__":
    main()

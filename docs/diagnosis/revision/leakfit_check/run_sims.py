"""Run every exact simulation the leak-fit checks need, in one pool (outputs go to cache/).

recovery.py and heldout_procedure.py each run their own missing simulations too; this
only schedules them together so that the pool stays full (roundcond_fit.py simulates
nothing: it reads revision/leakage_exact's exact grid):

  recovery        3 truths x NREP seeds, the same circuits at 2500 shots (+ Z at 4 and 6 rounds for truth A)
  heldout         the 8 exact-subset grid points with the fit-jobs-only hidden Markov inputs, and the rerun
                  of the eleven held-out jobs at the published optimum with those inputs
                  (all memory circuits, 10,000 shots, seed = rounds); heldout_procedure.py adds
                  the rerun at another point if its grid fit picks one

    python docs/diagnosis/revision/leakfit_check/run_sims.py [--workers 11] [--nrep 4]
"""

from __future__ import annotations

import argparse

import exact_cache as E
import heldout_procedure as HP
import recovery as RV
import roundcond_fit as RC
from exact_cache import FIT_JOBS, Point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=["heldout", "recovery", "roundcond-mc", "xcheck"])
    a = ap.parse_args()
    life, read = E.hmm_inputs()
    tasks = []
    if "recovery" in a.only:
        for t, p in RV.TRUTHS.items():
            pt = Point(*p, life, read)
            for s in [RV.SEED0 + i for i in range(a.nrep)]:
                tasks += [("det", j, b, 8, pt, RV.SHOTS, s) for j, b in RC.CASES]
                if t == "A":
                    tasks += [("det", j, "Z", r, pt, RV.SHOTS, s) for j in FIT_JOBS for r in (4, 6)]
    if "heldout" in a.only:
        life_f, read_f = E.hmm_inputs(FIT_JOBS)
        tasks += [("det", j, b, 8, pt, E.GRID_SHOTS, E.GRID_SEED) for pt in RC.points(life_f, read_f, with_profile=False).values() for j, b in RC.CASES]
        tasks += HP.eval_tasks(Point(*RC.BEST_PUBLISHED, life_f, read_f), HP.heldout_jobs())
    if "roundcond-mc" in a.only:  # roundcond_fit.py's Monte Carlo replicates of its leading points
        lead = [(0.002, 0.25, 0.05), (0.001, 0.5, 0.05), (0.003, 0.25, 0.0), (0.003, 0.25, 0.05)]
        tasks += [("det", j, b, 8, Point(*p, life, read), E.GRID_SHOTS, s) for p in lead for s in RC.MC_SEEDS for j, b in RC.CASES]
    if "xcheck" in a.only:  # exact_sim against tableau_fast (= tableau_ref) on the same circuits and point
        pt = Point(0.001, 0.25, 0.05, life, read)
        tasks += [("det", "2026-10-08-r1", b, 8, pt, E.GRID_SHOTS, E.GRID_SEED) for b in "XZ"]
    E.ensure(tasks, a.workers)


if __name__ == "__main__":
    main()

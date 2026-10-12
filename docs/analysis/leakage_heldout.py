"""A held-out test of the whole leakage-fit procedure, not only of its grid search.

In leakage_fit.py the seven "held-out" repetitions are held out from the grid fit of
(leak, partner X error, transfer) only: the lifetime and the leaked readout are medians of
hidden Markov fits over all fifteen jobs. Here every fitted input comes from the four fit
jobs only (leakage.FIT_JOBS: 2026-10-06 r1 and 2026-10-08 r1 and r2, three repetitions):

  1. lifetime and leaked readout: the same medians (7.9 us x median 1/b, median e_P), over
     the 96 flag/relay series of the fit jobs only (each series' fit uses only its own job);
  2. the grid fit, redone under the exact simulator (leakage.ensure) with those inputs, on
     the eight published-grid points of leakage.EXACT_SUBSET (all strength-25% points and
     strength 50% at 0.2%; the others lost by a factor of 4 or more in every run), with the
     pooled and the round-conditioned co-streak;
  3. the fitted model rerun on the eleven jobs of the seven other repetitions, decoded as
     the reruns decode (10,000 shots, seed = rounds), eps fitted with the prespecified
     weighted-log fit and with the binomial likelihood, and compared with the same rerun at
     the all-job inputs (results/leakage_rerun_identity_paperpoint.json: same simulator,
     seeds and shots, so the comparison is paired).

Closure = (mean eps_leak - mean eps_model) / (mean eps_device - mean eps_model) over the
held-out repetitions, as leakage_fit.py. What stays shared with the held-out data: the model
family, the six statistics and the grid, chosen during the exploratory diagnosis of all
jobs; IBM's calibration inputs of each day.

Heavy the first time (the simulations are cached).

    python docs/analysis/leakage_heldout.py [--workers 2]

Writes results/leakage_heldout.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

import fitting
import leakage as E
import leakage_fit as Hd
from common import RESULTS, WORKERS, jobs
from leakage import (
    BEST_PUBLISHED,
    CASES,
    FIT_JOBS,
    Point,
    argmin,
    case_stats,
    device_cases,
    fit_table,
    merged,
    points,
)

SHOTS = 10000  # as the reruns
CELLS = [("X", "3"), ("X", "5"), ("Z", "3"), ("Z", "5")]


def heldout_jobs() -> list[str]:
    return [j.name for j in jobs() if j.name not in FIT_JOBS]


def eval_tasks(pt: Point, names) -> list:
    return [("fail", n, b, r, pt, SHOTS, r) for n in names for b in "XZ" for r in ROUNDS]


def eps_of(pt: Point, names) -> dict:
    """{job: {basis: {patch: {'wls': (eps, se), 'binomial': eps, 'k': [...]}}}}"""
    out = {}
    for n in names:
        out[n] = {}
        for b in "XZ":
            ks = {}
            for r in ROUNDS:
                for patch, k in E.load_fail(n, b, r, pt, SHOTS, r).items():
                    ks.setdefault(patch, []).append(k)
            out[n][b] = {}
            for patch, k in ks.items():
                f = fit_per_round(ROUNDS, [x / SHOTS for x in k], SHOTS)
                out[n][b][patch] = {
                    "wls": (float(f.per_round), float(f.uncertainty)),
                    "binomial": fitting.binomial_point(ROUNDS, k, SHOTS),
                    "k": k,
                }
    return out


def closures(leak: dict, est: str) -> dict:
    """Held-out-group closures with leakage_fit.py's grouping, device and model as it has them."""
    base = Hd.load_wls(Hd.FRAME_RERUN, "fit-bitflip") if est == "wls" else Hd.load_binomial()
    data = {}
    for job, by_b in leak.items():
        rep = Hd.rep_of(job)
        d = data.setdefault(
            rep, {"device": base[rep]["device"], "model": base[rep]["model"], "leak": {}}
        )
        for b, by_p in by_b.items():
            for p, v in by_p.items():
                d["leak"][(b, p)] = v["wls"] if est == "wls" else (v["binomial"], np.nan)
    reps = sorted(data)
    return Hd.group(data, reps), reps


def leakexact_reference(names) -> dict | None:
    """The exact rerun at the published optimum with the all-job inputs
    (leakage_rerun_identity_paperpoint.json: the same simulator, seeds and shots as the
    evaluation here, so the two differ only by the fit-jobs-only lifetime and leaked readout)."""
    p = RESULTS / "leakage_rerun_identity_paperpoint.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not all(n in d["jobs"] for n in names):
        return None
    leak = {}
    for n in names:
        j = d["jobs"][n]
        leak[n] = {
            b: {
                pp: {
                    "wls": tuple(j["leak"][b][pp]),
                    "binomial": j["counts"][b][pp]["binomial"],
                    "k": j["counts"][b][pp]["k"],
                }
                for pp in j["leak"][b]
            }
            for b in "XZ"
        }
    return leak


def published_exact_heldout() -> dict | None:
    """tableau_ref.py's exact rerun at the published inputs, held-out group (weighted-log eps)."""
    p = RESULTS / "leakage_frame_rerun_exactidentity_paperparams.json"
    if not p.exists():
        return None
    data = Hd.load_wls(p, "leak")
    fit = sorted({Hd.rep_of(j) for j in FIT_JOBS})
    return Hd.group(data, [r for r in sorted(data) if r not in fit])


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--sim-only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    life_all, read_all = E.hmm_inputs()
    life, read = E.hmm_inputs(FIT_JOBS)
    print(
        f"1. hidden Markov inputs from the fit jobs only: lifetime {life:.2f} us, leaked readout {read:.4f} "
        f"(all 15 jobs: {life_all:.2f} us, {read_all:.4f})",
        flush=True,
    )
    grid = points(life, read)  # the exact subset of the published grid
    E.ensure(
        [("det", j, b, 8, pt, E.GRID_SHOTS, E.GRID_SEED) for pt in grid.values() for j, b in CASES],
        a.workers,
    )

    dev_cases = device_cases()
    labels = {c: dev_cases[c][1] for c in CASES}
    dev = merged(case_stats({c: dev_cases[c][0] for c in CASES}, labels))
    sims = {
        p: merged(case_stats({c: E.load(*c, 8, pt) for c in CASES}, labels))
        for p, pt in grid.items()
    }
    tb = fit_table(sims, dev)
    best = {}
    print("2. grid fit on the fit jobs with these inputs (loss pooled / slot):")
    for p in sorted(grid, key=lambda q: tb[q]["pooled"][0]):
        print(f"   {p!s:24s} {tb[p]['pooled'][0]:.3f} / {tb[p]['slot'][0]:.3f}")
    for v in ("pooled", "slot"):
        best[v] = argmin(tb, v)
        print(f"   best ({v}): {best[v]}")
    chosen = sorted(set(best.values()))
    names = heldout_jobs()
    tasks = []
    for p in chosen:
        tasks += eval_tasks(grid[p], names)
    E.ensure(tasks, a.workers)
    if a.sim_only:
        return

    ref = published_exact_heldout()
    out = {
        "inputs_fit_jobs": {"lifetime_us": life, "read_leaked": read},
        "inputs_all_jobs": {"lifetime_us": life_all, "read_leaked": read_all},
        "grid": [
            {"point": list(p), "loss_pooled": tb[p]["pooled"][0], "loss_slot": tb[p]["slot"][0]}
            for p in grid
        ],
        "best": {k: list(v) for k, v in best.items()},
        "heldout_jobs": names,
        "eval": {},
    }
    print(
        f"\n3. the fitted model on the {len(names)} jobs of the other repetitions (closure; leak - device in points)"
    )
    for p in chosen:
        leak = eps_of(grid[p], names)
        res = {}
        for est in ("wls", "binomial"):
            g, _reps = closures(leak, est)
            res[est] = g
            line = f"   {p} {est:8s}: " + "  ".join(
                f"{b} d{d} {100 * g[f'{b} d{d}']['closure']:.0f}% ({100 * g[f'{b} d{d}']['leak_minus_device']:+.1f})"
                for b, d in CELLS
            )
            line += f"  Lambda X {g['X lambda']['leak']:.3f} Z {g['Z lambda']['leak']:.3f}"
            print(line)
        out["eval"][str(p)] = {"closures": res, "eps": leak}
    if ref:
        print(
            "   for comparison, tableau_ref.py's exact rerun at the all-job inputs, wls: "
            + "  ".join(
                f"{b} d{d} {100 * ref[f'{b} d{d}']['closure']:.0f}% ({100 * ref[f'{b} d{d}']['leak_minus_device']:+.1f})"
                for b, d in CELLS
            )
        )
        out["reference_tableau_ref_all_job_inputs_heldout_wls"] = ref
    le = leakexact_reference(names)
    if le:
        out["reference_exact_sim_all_job_inputs"] = {}
        p0 = tuple(BEST_PUBLISHED)
        mine = out["eval"].get(str(p0))
        for est in ("wls", "binomial"):
            g, _ = closures(le, est)
            out["reference_exact_sim_all_job_inputs"][est] = g
            line = (
                f"   exact_sim.py's rerun at the all-job inputs ({est}, same seeds): "
                + "  ".join(
                    f"{b} d{d} {100 * g[f'{b} d{d}']['closure']:.0f}% ({100 * g[f'{b} d{d}']['leak_minus_device']:+.1f})"
                    for b, d in CELLS
                )
            )
            if mine:
                line += "  | paired change from holding out the inputs: " + "  ".join(
                    f"{b} d{d} {100 * (mine['closures'][est][f'{b} d{d}']['closure'] - g[f'{b} d{d}']['closure']):+.1f} points"
                    for b, d in CELLS
                )
            print(line)
    (RESULTS / "leakage_heldout.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", RESULTS / "leakage_heldout.json")


if __name__ == "__main__":
    main()

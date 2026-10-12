"""Gap closure of the leakage model on the repetitions its grid fit used and on the others.

The leakage model's three per-CZ probabilities (leak, partner error, transfer) were fitted
in fit_leak2.py to detection statistics of four jobs, FIT_JOBS there: 2026-10-06 r1 and
2026-10-08 r1 and r2 (r2 is two jobs, d3 and d5), i.e. three of the ten repetitions. The
other seven repetitions were not used by the grid fit. They are not fully held out: the
lifetime and the leaked readout come from the hidden Markov fits of all fifteen jobs
(leakage_hmm.py). IBM's reset and readout errors are each day's reported values.

For each group of repetitions and each (basis, d) this computes, from saved results only,

    closure = (mean eps_leak - mean eps_model) / (mean eps_device - mean eps_model)

with means over the group's repetitions, as rerun_summary.py does over all ten. In X memory
the leakage model predicts more error than the device, so closure > 1 there; the overshoot
is also given as eps_leak - eps_device. Inputs:

    docs/diagnosis/results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json  (the paper's leakage
        model, partner X error; prespecified weighted-log fit of eps)
    docs/diagnosis/results/rerun_model-fit_cz1.5_d1_p0.5.json  (partner random Pauli)
    docs/diagnosis/revision/refits/rerun_refit.json, refits.json  (binomial-likelihood eps,
        as a sensitivity check, fit-bitflip only)

It also gives each repetition's own closure, a shot-noise standard error (all inputs treated
as independent), and where the actual fit/held-out split falls among all C(10,3) = 120 ways
to pick three repetitions. Exploratory; no simulation is rerun.

    python docs/diagnosis/revision/heldout/heldout.py
"""

from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESULTS = REPO / "docs/diagnosis/results"
CELLS = [("X", "3"), ("X", "5"), ("Z", "3"), ("Z", "5")]


def fit_jobs() -> tuple[str, ...]:
    """FIT_JOBS as written in fit_leak2.py (read, not imported, so nothing runs)."""
    tree = ast.parse((REPO / "docs/diagnosis/fit_leak2.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FIT_JOBS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError("FIT_JOBS not found in fit_leak2.py")


def rep_of(job: str) -> str:
    """'2026-10-08-r2-d5' -> '2026-10-08 r2'."""
    return f"{job[:10]} {job[11:13]}"


def load_wls(path: Path, variant: str) -> dict:
    """{rep: {src: {(b, d): (eps, se)}}} for src in device, model, leak."""
    res = json.loads(path.read_text())
    out: dict = {}
    for job, r in res["jobs"].items():
        d = out.setdefault(rep_of(job), {"device": {}, "model": {}, "leak": {}})
        for src, key in (("device", "chip"), ("model", "model"), ("leak", variant)):
            for b in "XZ":
                for p, v in r[key][b].items():
                    d[src][(b, p)] = (float(v[0]), float(v[1]))
    return out


def load_binomial() -> dict:
    rr = json.loads((REPO / "docs/diagnosis/revision/refits/rerun_refit.json").read_text())
    ref = json.loads((REPO / "docs/diagnosis/revision/refits/refits.json").read_text(encoding="utf-8"))
    out: dict = {}
    for rep, x in ref["per_rep"].items():
        d = out.setdefault(rep, {"device": {}, "model": {}, "leak": {}})
        for b, p in CELLS:
            d["device"][(b, p)] = (float(x[b][f"eps{p}"]["binomial"]), np.nan)
            d["model"][(b, p)] = (float(rr["per_curve"][f"{rep} model {b} {p}"]["binomial"]), np.nan)
            d["leak"][(b, p)] = (float(rr["per_curve"][f"{rep} fit-bitflip {b} {p}"]["binomial"]), np.nan)
    return out


def group(data: dict, reps: list[str]) -> dict:
    out = {}
    n = len(reps)
    for b, p in CELLS:
        m = {s: np.array([data[r][s][(b, p)][0] for r in reps]) for s in ("device", "model", "leak")}
        se = {s: np.sqrt(sum(data[r][s][(b, p)][1] ** 2 for r in reps)) / n for s in ("device", "model", "leak")}
        C, M, L = (float(m[s].mean()) for s in ("device", "model", "leak"))
        gap = C - M
        clo = (L - M) / gap
        # d clo / d(L, C, M), all inputs independent
        grad = (1 / gap, -(L - M) / gap ** 2, (L - C) / gap ** 2)
        se_clo = float(np.sqrt((grad[0] * se["leak"]) ** 2 + (grad[1] * se["device"]) ** 2 + (grad[2] * se["model"]) ** 2))
        se_over = float(np.sqrt(se["leak"] ** 2 + se["device"] ** 2))
        per_rep = (m["leak"] - m["model"]) / (m["device"] - m["model"])
        out[f"{b} d{p}"] = {
            "n_reps": n, "device": C, "model": M, "leak": L, "gap": gap, "closure": clo,
            "closure_shot_se": se_clo if np.isfinite(se_clo) else None,
            "leak_minus_device": L - C,
            "leak_minus_device_shot_se": se_over if np.isfinite(se_over) else None,
            "per_rep_closure": {r: float(c) for r, c in zip(reps, per_rep)},
        }
    for b in "XZ":
        lam = {s: float(np.mean([data[r][s][(b, "3")][0] / data[r][s][(b, "5")][0] for r in reps]))
               for s in ("device", "model", "leak")}
        out[f"{b} lambda"] = lam
    return out


def closure_of(data: dict, reps, b: str, p: str) -> float:
    C, M, L = (np.mean([data[r][s][(b, p)][0] for r in reps]) for s in ("device", "model", "leak"))
    return float((L - M) / (C - M))


def splits(data: dict, fit: list[str]) -> dict:
    """All 3-of-10 splits: where the actual split's held-out minus fit closure falls."""
    reps = sorted(data)
    out = {}
    for b, p in CELLS:
        diffs, ins = [], []
        for s in itertools.combinations(reps, len(fit)):
            rest = [r for r in reps if r not in s]
            ci, co = closure_of(data, s, b, p), closure_of(data, rest, b, p)
            ins.append(ci); diffs.append(co - ci)
        actual = closure_of(data, [r for r in reps if r not in fit], b, p) - closure_of(data, fit, b, p)
        diffs, ins = np.array(diffs), np.array(ins)
        out[f"{b} d{p}"] = {
            "actual_heldout_minus_fit": actual,
            "share_of_splits_with_abs_diff_at_least_actual": float(np.mean(np.abs(diffs) >= abs(actual) - 1e-12)),
            "three_rep_closure_min": float(ins.min()), "three_rep_closure_max": float(ins.max()),
            "three_rep_closure_p5_p95": [float(np.percentile(ins, 5)), float(np.percentile(ins, 95))],
            "n_splits": len(diffs),
        }
    return out


def report(name: str, data: dict, fit: list[str], held: list[str]) -> dict:
    out = {g: group(data, reps) for g, reps in (("all", sorted(data)), ("fit", fit), ("heldout", held))}
    print(f"\n== {name} ==")
    print(f"{'':6s} {'group':8s} {'n':>2s} {'device':>7s} {'model':>7s} {'leak':>7s} {'closure':>8s} {'shot se':>7s} {'leak-dev':>8s} {'shot se':>7s}")
    pct = lambda v: f"{100*v:6.1f}%" if v is not None else f"{'-':>7s}"
    for b, p in CELLS:
        for g in ("all", "fit", "heldout"):
            x = out[g][f"{b} d{p}"]
            print(f"{b} d{p}  {g:8s} {x['n_reps']:2d} {100*x['device']:7.2f} {100*x['model']:7.2f} {100*x['leak']:7.2f} "
                  f"{100*x['closure']:7.1f}% {pct(x['closure_shot_se'])} {100*x['leak_minus_device']:+7.2f}pp "
                  f"{pct(x['leak_minus_device_shot_se'])}")
    for b in "XZ":
        for g in ("all", "fit", "heldout"):
            lam = out[g][f"{b} lambda"]
            print(f"{b} Lambda {g:8s} device {lam['device']:.3f} model {lam['model']:.3f} leak {lam['leak']:.3f}")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fj = fit_jobs()
    fit = sorted({rep_of(j) for j in fj})
    variants = {
        "fit-bitflip (paper; weighted-log eps)": load_wls(RESULTS / "rerun_model-fit-bitflip_cz1.5_d1_p0.5.json", "fit-bitflip"),
        "fit, random-Pauli partner (weighted-log eps)": load_wls(RESULTS / "rerun_model-fit_cz1.5_d1_p0.5.json", "fit"),
        "fit-bitflip (binomial-likelihood eps)": load_binomial(),
    }
    held = sorted(r for r in next(iter(variants.values())) if r not in fit)
    print("grid-fit jobs (fit_leak2.py FIT_JOBS):", ", ".join(fj))
    print("fit repetitions:", ", ".join(fit))
    print("held-out repetitions:", ", ".join(held))
    out = {
        "fit_jobs": list(fj), "fit_reps": fit, "heldout_reps": held,
        "note": ("Held out from the grid fit of leak, partner error and transfer only (fit_leak2.py). "
                 "Lifetime (35 us) and leaked readout (0.51) are medians of hidden Markov fits over all 15 jobs."),
        "definition": "closure = (mean eps_leak - mean eps_model) / (mean eps_device - mean eps_model), means over the group's repetitions",
        "variants": {},
    }
    for name, data in variants.items():
        assert sorted(data) == sorted(fit + held), name
        res = report(name, data, fit, held)
        res["splits"] = splits(data, fit)
        for cell, s in res["splits"].items():
            print(f"{cell} split check: held-out minus fit {100*s['actual_heldout_minus_fit']:+.1f} points; "
                  f"{100*s['share_of_splits_with_abs_diff_at_least_actual']:.0f}% of the 120 splits differ at least as much; "
                  f"3-rep closure ranges {100*s['three_rep_closure_min']:.0f}% to {100*s['three_rep_closure_max']:.0f}%")
        z = {g: {k: v for k, v in res[g].items()} for g in ("fit", "heldout")}
        for g in ("fit", "heldout"):
            print(f"Z per-rep closure, {g}: " + ", ".join(
                f"{r} d3 {100*z[g]['Z d3']['per_rep_closure'][r]:.0f}% d5 {100*z[g]['Z d5']['per_rep_closure'][r]:.0f}%"
                for r in z[g]["Z d3"]["per_rep_closure"]))
        out["variants"][name] = res
    (HERE / "heldout.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote", HERE / "heldout.json")


if __name__ == "__main__":
    main()

"""Numbers for the paper's leakage section under the exact simulation (summary_exact.json).

(a) the grid fits (grid_<channel>.json): every point's loss, the best point, which of its
    parameters sit on an edge of the points run, the loss profile (minimum over the other two
    parameters) and the slices through the best point, the Monte Carlo spread of the loss at
    the best point (seed replicates), the round-conditioned (per-slot) loss, and the
    comparison with the published grid (original simulator) and with leaky_fix's
    tableau_ref.py grid;
(b) Table IV (device, calibration model, leakage model; means over the ten repetitions);
(c) the Z-memory gap closure and the X-memory overshoot, over all ten repetitions, the three
    of the grid fit and the seven others, with the prespecified weighted-log fit of eps and with
    the binomial fit (heldout/heldout.py's definitions), and the 120-split check;
(d) the same for the other channels (random phase; phase pi);
(e) curve-by-curve agreement of the reruns: exact (this directory) vs the published rerun
    (original simulator), and vs leaky_fix's tableau_ref.py rerun at the same parameters.

    python summarize_exact.py RERUN_TAG[=label] ...    (e.g. identity_best=identity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
REV = DIAG / "revision"
sys.path[:0] = [str(REV / "heldout")]
import heldout as H  # noqa: E402

CELLS = [("X", "3"), ("X", "5"), ("Z", "3"), ("Z", "5")]
PUBLISHED_RERUN = DIAG / "results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json"


# ----------------------------------------------------------------------------- (a) grids
def grid_summary(path: Path, seed: int = 21) -> dict:
    g = json.loads(path.read_text())
    pts = [p for p in g["points"] if p["seed"] == seed and p["shots"] == 8000]
    key = lambda p: (p["leak"], p["strength"], p["spread"])  # noqa: E731
    best = min(pts, key=lambda p: p["loss"])
    axes = {"leak": sorted({p["leak"] for p in pts}), "strength": sorted({p["strength"] for p in pts}),
            "spread": sorted({p["spread"] for p in pts})}
    edges = {}
    for ax, vals in axes.items():
        # neighbours of the best point along this axis, the other two held at the best
        line = sorted((p[ax], p["loss"]) for p in pts if all(p[o] == best[o] for o in axes if o != ax))
        xs = [x for x, _ in line]
        i = xs.index(best[ax])
        edges[ax] = {"value": best[ax], "on_edge_of_run": i in (0, len(xs) - 1),
                     "slice": [[x, round(v, 4)] for x, v in line]}
    profile = {ax: [[v, round(min(p["loss"] for p in pts if p[ax] == v), 4)] for v in vals] for ax, vals in axes.items()}
    reps = [p for p in g["points"] if key(p) == key(best) and p["shots"] == 8000]
    mc = [p["loss"] for p in reps]
    ranked = sorted(pts, key=lambda p: p["loss"])
    # points simulated with several seeds: mean and spread of the loss (Monte Carlo noise)
    multi = {}
    for p in g["points"]:
        if p["shots"] == 8000:
            multi.setdefault(key(p), []).append(p)
    replicated = sorted(([*k, len(v), float(np.mean([q["loss"] for q in v])), float(np.std([q["loss"] for q in v], ddof=1)),
                          float(np.mean([q["loss_slot"] for q in v]))]
                         for k, v in multi.items() if len(v) > 1), key=lambda r: r[4])
    out = {
        "file": path.name, "sim": g.get("sim", "exact"), "n_points": len(pts), "axes": axes,
        "best": {k: best[k] for k in ("leak", "strength", "spread", "loss", "loss_slot", "rel", "rel_slot")},
        "best_slot_statistic": {k: v for k, v in min(pts, key=lambda p: p["loss_slot"]).items()
                                if k in ("leak", "strength", "spread", "loss_slot")},
        "edges": edges, "profile_min_over_others": profile,
        "best_seed_replicates": {"seeds": [p["seed"] for p in reps], "loss": mc,
                                 "sd": float(np.std(mc, ddof=1)) if len(mc) > 1 else None},
        "top5": [{k: p[k] for k in ("leak", "strength", "spread", "loss", "loss_slot")} for p in ranked[:5]],
        "seed_replicated": [dict(zip(("leak", "strength", "spread", "n_seeds", "mean_loss", "sd_loss", "mean_loss_slot"), r))
                            for r in replicated],
        "points": [{k: p[k] for k in ("leak", "strength", "spread", "loss", "loss_slot")} for p in
                   sorted(pts, key=key)],
    }
    return out


def print_grid(name: str, s: dict) -> None:
    b = s["best"]
    print(f"\n== grid {name} ({s['file']}, {s['n_points']} points, sim {s['sim']}) ==")
    print(f"best: leak {b['leak']} strength {b['strength']} spread {b['spread']}  loss {b['loss']:.3f} (per-slot {b['loss_slot']:.3f})")
    print("   " + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in b["rel"].items()))
    for ax, e in s["edges"].items():
        print(f"   {ax:8s} {e['value']:<7g} {'ON EDGE' if e['on_edge_of_run'] else 'interior'}; slice: "
              + ", ".join(f"{x:g}: {v:.3f}" for x, v in e["slice"]))
    for ax, pr in s["profile_min_over_others"].items():
        print(f"   profile {ax:8s} " + ", ".join(f"{x:g}: {v:.3f}" for x, v in pr))
    r = s["best_seed_replicates"]
    if r["sd"] is not None:
        print(f"   seed replicates at best: {', '.join(f'{v:.3f}' for v in r['loss'])} (sd {r['sd']:.3f})")
    print("   top 5: " + "; ".join(f"({p['leak']:g}, {p['strength']:g}, {p['spread']:g}) {p['loss']:.3f}" for p in s["top5"]))
    print(f"   best under the per-slot statistic: {s['best_slot_statistic']}")
    for r in s["seed_replicated"]:
        print(f"   seeds: ({r['leak']:g}, {r['strength']:g}, {r['spread']:g}) n={r['n_seeds']} mean loss {r['mean_loss']:.3f} "
              f"sd {r['sd_loss']:.3f} (per-slot {r['mean_loss_slot']:.3f})")


def compare_published_grid(exact_path: Path) -> dict:
    """Published (original simulator) and tableau_ref grid losses against the exact ones."""
    ex = {(p["leak"], p["strength"], p["spread"]): p["loss"] for p in json.loads(exact_path.read_text())["points"]
          if p["seed"] == 21 and p["shots"] == 8000}
    pub = json.loads((DIAG / "results/fit_leak2_bitflip.json").read_text())
    chip = pub["chip"]
    orig = {}
    for p in pub["grid"]:
        rel = [(p["sim"][k] - chip[k]) / abs(chip[k]) for k in chip if np.isfinite(chip[k])]
        orig[(p["leak"], p["strength"], p["spread"])] = float(np.sqrt(np.mean(np.square(rel))))
    tab = {(p["leak"], p["strength"], p["spread"]): p["loss"] for p in
           json.loads((REV / "leaky_fix/fit_leak2_bitflip_exact-identity_paper.json").read_text())["points"]}
    rows = []
    for k in sorted(orig):
        rows.append({"point": k, "original_sim": orig[k], "exact": ex.get(k), "tableau_ref": tab.get(k)})
    print("\n== published grid points: loss with the original simulator, tableau_ref.py (leaky_fix) and exact_sim.py ==")
    for r in rows:
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "  # noqa: E731
        print(f"   {r['point']}: original {f(r['original_sim'])}  tableau_ref {f(r['tableau_ref'])}  exact_sim {f(r['exact'])}")
    return {"rows": [{**r, "point": list(r["point"])} for r in rows]}


# ----------------------------------------------------------------------------- (b, c) reruns
def load_rerun(path: Path) -> tuple[dict, dict]:
    """(weighted-log data, binomial data) in heldout.py's format {rep: {src: {(b, d): (eps, se)}}}."""
    wls = H.load_wls(path, "leak")
    refits = json.loads((REV / "refits/refits.json").read_text(encoding="utf-8"))
    rr = json.loads((REV / "refits/rerun_refit.json").read_text())
    res = json.loads(path.read_text())
    binom: dict = {}
    for job, r in res["jobs"].items():
        rep = H.rep_of(job)
        d = binom.setdefault(rep, {"device": {}, "model": {}, "leak": {}})
        for b in "XZ":
            for p, c in r["counts"][b].items():
                d["leak"][(b, p)] = (float(c["binomial"]), np.nan)
                d["device"][(b, p)] = (float(refits["per_rep"][rep][b][f"eps{p}"]["binomial"]), np.nan)
                d["model"][(b, p)] = (float(rr["per_curve"][f"{rep} model {b} {p}"]["binomial"]), np.nan)
    return wls, binom


def table_and_closures(data: dict) -> dict:
    fit = sorted({H.rep_of(j) for j in H.fit_jobs()})
    reps = sorted(data)
    held = [r for r in reps if r not in fit]
    out = {g: H.group(data, rs) for g, rs in (("all", reps), ("fit", fit), ("heldout", held))}
    out["splits"] = H.splits(data, fit)
    return out


def print_rerun(name: str, est: str, r: dict) -> None:
    a = r["all"]
    print(f"\n-- {name}, {est} eps (means over 10 repetitions) --")
    print("   Table IV: " + "; ".join(
        f"{b} eps{p} dev {100 * a[f'{b} d{p}']['device']:.1f} model {100 * a[f'{b} d{p}']['model']:.1f} "
        f"leak {100 * a[f'{b} d{p}']['leak']:.1f}" for b, p in CELLS))
    print("   Lambda: " + "; ".join(f"{b} dev {a[f'{b} lambda']['device']:.2f} model {a[f'{b} lambda']['model']:.2f} "
                                   f"leak {a[f'{b} lambda']['leak']:.2f}" for b in "XZ"))
    for g in ("all", "fit", "heldout"):
        x = r[g]
        print(f"   [{g:7s} n={x['Z d3']['n_reps']:2d}] Z closure d3 {100 * x['Z d3']['closure']:.0f}% d5 {100 * x['Z d5']['closure']:.0f}%"
              f" | X overshoot d3 {100 * x['X d3']['leak_minus_device']:+.1f} pp d5 {100 * x['X d5']['leak_minus_device']:+.1f} pp"
              f" | Z leak-dev d3 {100 * x['Z d3']['leak_minus_device']:+.1f} d5 {100 * x['Z d5']['leak_minus_device']:+.1f} pp")
    s = r["splits"]
    print("   split check (held-out minus fit closure; share of the 120 splits at least as large): " + "; ".join(
        f"{c} {100 * s[c]['actual_heldout_minus_fit']:+.0f} pts, {100 * s[c]['share_of_splits_with_abs_diff_at_least_actual']:.0f}%"
        for c in ("Z d3", "Z d5")))


def curve_compare(a_path: Path, a_key: str, b_path: Path, b_key: str) -> dict:
    """Per-curve z of eps between two reruns (weighted-log eps and its stored uncertainty)."""
    A, B = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    z = []
    for job in A["jobs"]:
        for b in "XZ":
            for p, (e1, s1) in A["jobs"][job][a_key][b].items():
                e2, s2 = B["jobs"][job][b_key][b][p]
                z.append((e1 - e2) / np.hypot(s1, s2))
    z = np.array(z)
    return {"n_curves": int(len(z)), "mean_z": float(z.mean()), "chi2_per_curve": float((z ** 2).mean()),
            "max_abs_z": float(np.abs(z).max())}


def twice_cz(path: Path) -> dict:
    """The paper's comparison with a no-leakage model with twice the CZ error (cz_scaled.py),
    averaged over the four fit jobs (each patch once): eps(cz x 2) - eps(leakage model)."""
    cz = json.loads((REV / "cz_scaled/cz_scaled.json").read_text())["jobs"]
    lk = json.loads(path.read_text())["jobs"]
    out = {}
    for b, p in CELLS:
        d = [cz[j]["eps"]["cz2"][b][p][0] - lk[j]["leak"][b][p][0] for j in cz if p in lk[j]["leak"][b]]
        out[f"{b} d{p}"] = {"n_jobs": len(d), "mean_cz2_minus_leak": float(np.mean(d))}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out: dict = {"grids": {}, "reruns": {}, "curve_checks": {}}
    for name in ("identity", "dephase_frame", "dephase", "pi"):
        p = HERE / f"grid_{name}.json"
        if p.exists():
            out["grids"][name] = grid_summary(p)
            print_grid(name, out["grids"][name])
    if (HERE / "grid_identity.json").exists():
        out["published_grid_comparison"] = compare_published_grid(HERE / "grid_identity.json")
    runs = {"published (original simulator), 0.2%/25%/5%": (PUBLISHED_RERUN, None)}
    for arg in sys.argv[1:]:
        tag, _, label = arg.partition("=")
        runs[label or tag] = (HERE / f"rerun_{tag}.json", tag)
    for label, (path, tag) in runs.items():
        if tag is None:
            wls = H.load_wls(path, "fit-bitflip")
            binom = H.load_binomial()
        else:
            wls, binom = load_rerun(path)
            out["reruns"].setdefault(label, {})["params"] = json.loads(path.read_text())["params"]
        res = {"wls": table_and_closures(wls), "binomial": table_and_closures(binom)}
        out["reruns"].setdefault(label, {}).update(res)
        for est in ("wls", "binomial"):
            print_rerun(label, est, res[est])
        if tag is not None:
            out["reruns"][label]["twice_cz_minus_leak_fit_jobs"] = t = twice_cz(path)
            print("   cz x 2 minus leakage model, mean over the four fit jobs (pp): " +
                  ", ".join(f"{k} {100 * v['mean_cz2_minus_leak']:+.1f}" for k, v in t.items()))
            out["curve_checks"][f"{label} vs published (original simulator)"] = c = curve_compare(path, "leak", PUBLISHED_RERUN, "fit-bitflip")
            print(f"   curves vs published rerun: {c}")
            prm = json.loads(path.read_text())["params"]
            tab = REV / "leaky_fix/rerun_exactidentity_paperparams.json"
            if prm["channel"] == "identity" and (prm["leak"], prm["strength"], prm["spread"]) == (0.002, 0.25, 0.05):
                out["curve_checks"][f"{label} vs tableau_ref rerun (leaky_fix)"] = c = curve_compare(path, "leak", tab, "leak")
                print(f"   curves vs tableau_ref rerun at the same parameters: {c}")
    if (HERE / "validate_exact.json").exists():
        out["validation"] = json.loads((HERE / "validate_exact.json").read_text())
    (HERE / "summary_exact.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("\nwrote", (HERE / "summary_exact.json").relative_to(DIAG.parent.parent))


if __name__ == "__main__":
    main()

"""The leakage-aware simulation's numbers (Sec. leakage, Table rerun), from the committed runs.

exact (the leakage model simulated exactly, exact_sim.py):
  grids        every point's loss (leakage_grid_<channel>.json), the best point, which of its
               parameters sit on an edge of the points run, the loss profile (minimum over the
               other two parameters) and the slices through the best point, the Monte Carlo
               spread of the loss (seed replicates), the round-conditioned (per-slot) loss
  reruns       Table rerun (device, calibration model, leakage model; means over the ten
               repetitions) and the Z-memory gap closure and X-memory overshoot over all ten
               repetitions, the three of the grid fit and the seven others, with the
               prespecified weighted-log fit of eps and with the binomial fit, and where the
               actual three/seven split falls among all 120; for the identity channel at
               three points and for the other channels (random phase; phase pi)
  curve_checks curve-by-curve agreement of the reruns with the frame simulator's rerun
               (leakage_frame_rerun_bitflip.json) and with tableau_ref.py's rerun at the same
               point (leakage_frame_rerun_exactidentity_paperparams.json)
  published_grid_comparison  the published grid's points: loss with the frame simulator,
               tableau_ref.py (leakage_frame_grid_bitflip_exact-identity_paper.json) and exact_sim.py

frame (the frame simulator, as first run):
  frame_binomial  Table rerun under both estimators, from leakage_frame_counts.json
  frame_heldout   the closures on the grid fit's three repetitions and the seven others, and
                  the 120 three/seven splits

closure = (mean eps_leak - mean eps_model) / (mean eps_device - mean eps_model), means over a
group's repetitions. The seven "held-out" repetitions were held out from the grid fit of
leak, partner error and transfer only: the lifetime and the leaked readout are medians of
hidden Markov fits over all fifteen jobs (leakage_heldout.py fits those on the fit jobs only).

Under a minute.

    python docs/analysis/leakage_fit.py     # writes results/leakage_fit.json
"""

from __future__ import annotations

import itertools
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np

import fitting as F
from common import RESULTS, result, write_json
from leakage import FIT_JOBS

CELLS = [("X", "3"), ("X", "5"), ("Z", "3"), ("Z", "5")]
FRAME_RERUN = RESULTS / "leakage_frame_rerun_bitflip.json"
# leakage_rerun.py's runs, by tag, and their labels
RERUNS = {
    "identity_paperpoint": "identity_published_point",
    "identity_extbest": "identity_ext_optimum",
    "identity_alt": "identity_alt_valley_point",
    "dephase_refit": "random_phase_refit",
    "dephase_alt": "random_phase_alt",
    "pi_refit": "phase_pi_refit",
}
PUBLISHED = "published (original simulator), 0.2%/25%/5%"


# ----------------------------------------------------------------------------- grouping
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


@cache
def frame_binomial() -> dict:
    """Table rerun (device, calibration model, frame-simulated leakage model fit-bitflip) under
    the weighted-log and the binomial fit, from the regenerated counts."""
    rr = result("leakage_frame_counts.json")
    ref = result("fits.json")
    rounds, N = tuple(rr["rounds"]), rr["shots"]
    eps: dict = {}  # (rep key, variant, basis, d) -> {wls, binomial}
    for _name, j in rr["jobs"].items():
        key = f"{j['day']} r{j['rep']}"
        for v, by_b in j["failures"].items():
            for b, by_p in by_b.items():
                for p, k in by_p.items():
                    w = F.wls(rounds, np.array(k) / N, N)
                    m = F.binomial_point(rounds, k, N)
                    eps[(key, v, b, p)] = {"wls": w.eps, "binomial": m, "k": k}
    keys = sorted(ref["per_rep"])
    table: dict = {}
    for src in ("device", *rr["variants"]):
        for est in ("wls", "binomial"):
            col = {}
            for b in "XZ":
                e3, e5 = [], []
                for k in keys:
                    if src == "device":
                        x = ref["per_rep"][k][b]
                        e3.append(x["eps3"]["wls"][0] if est == "wls" else x["eps3"]["binomial"])
                        e5.append(x["eps5"]["wls"][0] if est == "wls" else x["eps5"]["binomial"])
                    else:
                        e3.append(eps[(k, src, b, "3")][est])
                        e5.append(eps[(k, src, b, "5")][est])
                e3, e5 = np.array(e3), np.array(e5)
                col[f"{b} eps3"] = float(e3.mean())
                col[f"{b} eps5"] = float(e5.mean())
                col[f"{b} lambda"] = float((e3 / e5).mean())
            table[f"{src} {est}"] = col
    return {
        "shots_per_circuit_simulated": N,
        "means_over_ten_reps": table,
        "per_curve": {" ".join(map(str, k)): v for k, v in eps.items()},
        "regenerated_matches_stored_eps_to": rr["max_abs_difference_from_stored_eps"],
    }


def load_binomial() -> dict:
    """Binomial-likelihood eps of the device, the calibration model and the frame-simulated
    leakage model, in load_wls's layout (no standard errors)."""
    rr = frame_binomial()
    ref = result("fits.json")
    out: dict = {}
    for rep, x in ref["per_rep"].items():
        d = out.setdefault(rep, {"device": {}, "model": {}, "leak": {}})
        for b, p in CELLS:
            d["device"][(b, p)] = (float(x[b][f"eps{p}"]["binomial"]), np.nan)
            d["model"][(b, p)] = (
                float(rr["per_curve"][f"{rep} model {b} {p}"]["binomial"]),
                np.nan,
            )
            d["leak"][(b, p)] = (
                float(rr["per_curve"][f"{rep} fit-bitflip {b} {p}"]["binomial"]),
                np.nan,
            )
    return out


def group(data: dict, reps: list[str]) -> dict:
    out = {}
    n = len(reps)
    for b, p in CELLS:
        m = {
            s: np.array([data[r][s][(b, p)][0] for r in reps]) for s in ("device", "model", "leak")
        }
        se = {
            s: np.sqrt(sum(data[r][s][(b, p)][1] ** 2 for r in reps)) / n
            for s in ("device", "model", "leak")
        }
        C, M, L = (float(m[s].mean()) for s in ("device", "model", "leak"))
        gap = C - M
        clo = (L - M) / gap
        # d clo / d(L, C, M), all inputs independent
        grad = (1 / gap, -(L - M) / gap**2, (L - C) / gap**2)
        se_clo = float(
            np.sqrt(
                (grad[0] * se["leak"]) ** 2
                + (grad[1] * se["device"]) ** 2
                + (grad[2] * se["model"]) ** 2
            )
        )
        se_over = float(np.sqrt(se["leak"] ** 2 + se["device"] ** 2))
        per_rep = (m["leak"] - m["model"]) / (m["device"] - m["model"])
        out[f"{b} d{p}"] = {
            "n_reps": n,
            "device": C,
            "model": M,
            "leak": L,
            "gap": gap,
            "closure": clo,
            "closure_shot_se": se_clo if np.isfinite(se_clo) else None,
            "leak_minus_device": L - C,
            "leak_minus_device_shot_se": se_over if np.isfinite(se_over) else None,
            "per_rep_closure": {r: float(c) for r, c in zip(reps, per_rep)},
        }
    for b in "XZ":
        lam = {
            s: float(np.mean([data[r][s][(b, "3")][0] / data[r][s][(b, "5")][0] for r in reps]))
            for s in ("device", "model", "leak")
        }
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
            ins.append(ci)
            diffs.append(co - ci)
        actual = closure_of(data, [r for r in reps if r not in fit], b, p) - closure_of(
            data, fit, b, p
        )
        diffs, ins = np.array(diffs), np.array(ins)
        out[f"{b} d{p}"] = {
            "actual_heldout_minus_fit": actual,
            "share_of_splits_with_abs_diff_at_least_actual": float(
                np.mean(np.abs(diffs) >= abs(actual) - 1e-12)
            ),
            "three_rep_closure_min": float(ins.min()),
            "three_rep_closure_max": float(ins.max()),
            "three_rep_closure_p5_p95": [
                float(np.percentile(ins, 5)),
                float(np.percentile(ins, 95)),
            ],
            "n_splits": len(diffs),
        }
    return out


def table_and_closures(data: dict) -> dict:
    fit = sorted({rep_of(j) for j in FIT_JOBS})
    reps = sorted(data)
    held = [r for r in reps if r not in fit]
    out = {g: group(data, rs) for g, rs in (("all", reps), ("fit", fit), ("heldout", held))}
    out["splits"] = splits(data, fit)
    return out


# ----------------------------------------------------------------------------- exact grids
def grid_summary(path: Path, seed: int = 21) -> dict:
    g = json.loads(path.read_text())
    pts = [p for p in g["points"] if p["seed"] == seed and p["shots"] == 8000]
    key = lambda p: (p["leak"], p["strength"], p["spread"])  # noqa: E731
    best = min(pts, key=lambda p: p["loss"])
    axes = {
        "leak": sorted({p["leak"] for p in pts}),
        "strength": sorted({p["strength"] for p in pts}),
        "spread": sorted({p["spread"] for p in pts}),
    }
    edges = {}
    for ax in axes:
        # neighbours of the best point along this axis, the other two held at the best
        line = sorted(
            (p[ax], p["loss"]) for p in pts if all(p[o] == best[o] for o in axes if o != ax)
        )
        xs = [x for x, _ in line]
        i = xs.index(best[ax])
        edges[ax] = {
            "value": best[ax],
            "on_edge_of_run": i in (0, len(xs) - 1),
            "slice": [[x, round(v, 4)] for x, v in line],
        }
    profile = {
        ax: [[v, round(min(p["loss"] for p in pts if p[ax] == v), 4)] for v in vals]
        for ax, vals in axes.items()
    }
    reps = [p for p in g["points"] if key(p) == key(best) and p["shots"] == 8000]
    mc = [p["loss"] for p in reps]
    ranked = sorted(pts, key=lambda p: p["loss"])
    # points simulated with several seeds: mean and spread of the loss (Monte Carlo noise)
    multi: dict = {}
    for p in g["points"]:
        if p["shots"] == 8000:
            multi.setdefault(key(p), []).append(p)
    replicated = sorted(
        (
            [
                *k,
                len(v),
                float(np.mean([q["loss"] for q in v])),
                float(np.std([q["loss"] for q in v], ddof=1)),
                float(np.mean([q["loss_slot"] for q in v])),
            ]
            for k, v in multi.items()
            if len(v) > 1
        ),
        key=lambda r: r[4],
    )
    return {
        "file": path.name,
        "sim": g.get("sim", "exact"),
        "n_points": len(pts),
        "axes": axes,
        "best": {
            k: best[k]
            for k in ("leak", "strength", "spread", "loss", "loss_slot", "rel", "rel_slot")
        },
        "best_slot_statistic": {
            k: v
            for k, v in min(pts, key=lambda p: p["loss_slot"]).items()
            if k in ("leak", "strength", "spread", "loss_slot")
        },
        "edges": edges,
        "profile_min_over_others": profile,
        "best_seed_replicates": {
            "seeds": [p["seed"] for p in reps],
            "loss": mc,
            "sd": float(np.std(mc, ddof=1)) if len(mc) > 1 else None,
        },
        "top5": [
            {k: p[k] for k in ("leak", "strength", "spread", "loss", "loss_slot")}
            for p in ranked[:5]
        ],
        "seed_replicated": [
            dict(
                zip(
                    (
                        "leak",
                        "strength",
                        "spread",
                        "n_seeds",
                        "mean_loss",
                        "sd_loss",
                        "mean_loss_slot",
                    ),
                    r,
                )
            )
            for r in replicated
        ],
        "points": [
            {k: p[k] for k in ("leak", "strength", "spread", "loss", "loss_slot")}
            for p in sorted(pts, key=key)
        ],
    }


def print_grid(name: str, s: dict) -> None:
    b = s["best"]
    print(f"\n== grid {name} ({s['file']}, {s['n_points']} points, sim {s['sim']}) ==")
    print(
        f"best: leak {b['leak']} strength {b['strength']} spread {b['spread']}  loss {b['loss']:.3f} (per-slot {b['loss_slot']:.3f})"
    )
    print("   " + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in b["rel"].items()))
    for ax, e in s["edges"].items():
        print(
            f"   {ax:8s} {e['value']:<7g} {'ON EDGE' if e['on_edge_of_run'] else 'interior'}; slice: "
            + ", ".join(f"{x:g}: {v:.3f}" for x, v in e["slice"])
        )
    for ax, pr in s["profile_min_over_others"].items():
        print(f"   profile {ax:8s} " + ", ".join(f"{x:g}: {v:.3f}" for x, v in pr))
    r = s["best_seed_replicates"]
    if r["sd"] is not None:
        print(
            f"   seed replicates at best: {', '.join(f'{v:.3f}' for v in r['loss'])} (sd {r['sd']:.3f})"
        )
    print(
        "   top 5: "
        + "; ".join(
            f"({p['leak']:g}, {p['strength']:g}, {p['spread']:g}) {p['loss']:.3f}"
            for p in s["top5"]
        )
    )
    print(f"   best under the per-slot statistic: {s['best_slot_statistic']}")
    for r in s["seed_replicated"]:
        print(
            f"   seeds: ({r['leak']:g}, {r['strength']:g}, {r['spread']:g}) n={r['n_seeds']} mean loss {r['mean_loss']:.3f} "
            f"sd {r['sd_loss']:.3f} (per-slot {r['mean_loss_slot']:.3f})"
        )


def compare_published_grid(exact_path: Path) -> dict:
    """The published grid's points: loss with the frame simulator, tableau_ref.py and exact_sim.py."""
    ex = {
        (p["leak"], p["strength"], p["spread"]): p["loss"]
        for p in json.loads(exact_path.read_text())["points"]
        if p["seed"] == 21 and p["shots"] == 8000
    }
    pub = result("leakage_frame_grid_bitflip.json")
    chip = pub["chip"]
    orig = {}
    for p in pub["grid"]:
        rel = [(p["sim"][k] - chip[k]) / abs(chip[k]) for k in chip if np.isfinite(chip[k])]
        orig[(p["leak"], p["strength"], p["spread"])] = float(np.sqrt(np.mean(np.square(rel))))
    tab = {
        (p["leak"], p["strength"], p["spread"]): p["loss"]
        for p in result("leakage_frame_grid_bitflip_exact-identity_paper.json")["points"]
    }
    rows = [
        {"point": k, "original_sim": orig[k], "exact": ex.get(k), "tableau_ref": tab.get(k)}
        for k in sorted(orig)
    ]
    print(
        "\n== published grid points: loss with the frame simulator, tableau_ref.py and exact_sim.py =="
    )
    for r in rows:
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "  # noqa: E731
        print(
            f"   {r['point']}: frame {f(r['original_sim'])}  tableau_ref {f(r['tableau_ref'])}  exact_sim {f(r['exact'])}"
        )
    return {"rows": [{**r, "point": list(r["point"])} for r in rows]}


# ----------------------------------------------------------------------------- exact reruns
def load_rerun(path: Path) -> tuple[dict, dict]:
    """(weighted-log data, binomial data) of an exact rerun, in load_wls's layout."""
    wls = load_wls(path, "leak")
    refits = result("fits.json")
    rr = frame_binomial()
    res = json.loads(path.read_text())
    binom: dict = {}
    for job, r in res["jobs"].items():
        rep = rep_of(job)
        d = binom.setdefault(rep, {"device": {}, "model": {}, "leak": {}})
        for b in "XZ":
            for p, c in r["counts"][b].items():
                d["leak"][(b, p)] = (float(c["binomial"]), np.nan)
                d["device"][(b, p)] = (
                    float(refits["per_rep"][rep][b][f"eps{p}"]["binomial"]),
                    np.nan,
                )
                d["model"][(b, p)] = (
                    float(rr["per_curve"][f"{rep} model {b} {p}"]["binomial"]),
                    np.nan,
                )
    return wls, binom


def print_rerun(name: str, est: str, r: dict) -> None:
    a = r["all"]
    print(f"\n-- {name}, {est} eps (means over 10 repetitions) --")
    print(
        "   Table rerun: "
        + "; ".join(
            f"{b} eps{p} dev {100 * a[f'{b} d{p}']['device']:.1f} model {100 * a[f'{b} d{p}']['model']:.1f} "
            f"leak {100 * a[f'{b} d{p}']['leak']:.1f}"
            for b, p in CELLS
        )
    )
    print(
        "   Lambda: "
        + "; ".join(
            f"{b} dev {a[f'{b} lambda']['device']:.2f} model {a[f'{b} lambda']['model']:.2f} "
            f"leak {a[f'{b} lambda']['leak']:.2f}"
            for b in "XZ"
        )
    )
    for g in ("all", "fit", "heldout"):
        x = r[g]
        print(
            f"   [{g:7s} n={x['Z d3']['n_reps']:2d}] Z closure d3 {100 * x['Z d3']['closure']:.0f}% d5 {100 * x['Z d5']['closure']:.0f}%"
            f" | X overshoot d3 {100 * x['X d3']['leak_minus_device']:+.1f} pp d5 {100 * x['X d5']['leak_minus_device']:+.1f} pp"
            f" | Z leak-dev d3 {100 * x['Z d3']['leak_minus_device']:+.1f} d5 {100 * x['Z d5']['leak_minus_device']:+.1f} pp"
        )
    s = r["splits"]
    print(
        "   split check (held-out minus fit closure; share of the 120 splits at least as large): "
        + "; ".join(
            f"{c} {100 * s[c]['actual_heldout_minus_fit']:+.0f} pts, {100 * s[c]['share_of_splits_with_abs_diff_at_least_actual']:.0f}%"
            for c in ("Z d3", "Z d5")
        )
    )


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
    return {
        "n_curves": len(z),
        "mean_z": float(z.mean()),
        "chi2_per_curve": float((z**2).mean()),
        "max_abs_z": float(np.abs(z).max()),
    }


def twice_cz(path: Path) -> dict:
    """A no-leakage model with twice the CZ error (cz_scaled.py) against the leakage model,
    averaged over the four fit jobs (each patch once): eps(cz x 2) - eps(leakage model)."""
    cz = result("cz_scaled_runs.json")["jobs"]
    lk = json.loads(path.read_text())["jobs"]
    out = {}
    for b, p in CELLS:
        d = [
            cz[j]["eps"]["cz2"][b][p][0] - lk[j]["leak"][b][p][0]
            for j in cz
            if p in lk[j]["leak"][b]
        ]
        out[f"{b} d{p}"] = {"n_jobs": len(d), "mean_cz2_minus_leak": float(np.mean(d))}
    return out


def exact() -> dict:
    out: dict = {"grids": {}, "reruns": {}, "curve_checks": {}}
    for name in ("identity", "dephase_frame", "dephase", "pi"):
        p = RESULTS / f"leakage_grid_{name}.json"
        if p.exists():
            out["grids"][name] = grid_summary(p)
            print_grid(name, out["grids"][name])
    out["published_grid_comparison"] = compare_published_grid(
        RESULTS / "leakage_grid_identity.json"
    )
    runs = {PUBLISHED: (FRAME_RERUN, None)} | {
        label: (RESULTS / f"leakage_rerun_{tag}.json", tag) for tag, label in RERUNS.items()
    }
    for label, (path, tag) in runs.items():
        if tag is None:
            wls, binom = load_wls(path, "fit-bitflip"), load_binomial()
        else:
            wls, binom = load_rerun(path)
            out["reruns"].setdefault(label, {})["params"] = json.loads(path.read_text())["params"]
        res = {"wls": table_and_closures(wls), "binomial": table_and_closures(binom)}
        out["reruns"].setdefault(label, {}).update(res)
        for est in ("wls", "binomial"):
            print_rerun(label, est, res[est])
        if tag is None:
            continue
        out["reruns"][label]["twice_cz_minus_leak_fit_jobs"] = t = twice_cz(path)
        print(
            "   cz x 2 minus leakage model, mean over the four fit jobs (pp): "
            + ", ".join(f"{k} {100 * v['mean_cz2_minus_leak']:+.1f}" for k, v in t.items())
        )
        out["curve_checks"][f"{label} vs published (original simulator)"] = c = curve_compare(
            path, "leak", FRAME_RERUN, "fit-bitflip"
        )
        print(f"   curves vs the frame simulator's rerun: {c}")
        prm = json.loads(path.read_text())["params"]
        if prm["channel"] == "identity" and (prm["leak"], prm["strength"], prm["spread"]) == (
            0.002,
            0.25,
            0.05,
        ):
            tab = RESULTS / "leakage_frame_rerun_exactidentity_paperparams.json"
            out["curve_checks"][f"{label} vs tableau_ref rerun"] = c = curve_compare(
                path, "leak", tab, "leak"
            )
            print(f"   curves vs tableau_ref.py's rerun at the same parameters: {c}")
    return out


def frame_heldout() -> dict:
    fit = sorted({rep_of(j) for j in FIT_JOBS})
    variants = {
        "fit-bitflip (paper; weighted-log eps)": load_wls(FRAME_RERUN, "fit-bitflip"),
        "fit, random-Pauli partner (weighted-log eps)": load_wls(
            RESULTS / "leakage_frame_rerun_depolarize.json", "fit"
        ),
        "fit-bitflip (binomial-likelihood eps)": load_binomial(),
    }
    held = sorted(r for r in next(iter(variants.values())) if r not in fit)
    out = {
        "fit_jobs": list(FIT_JOBS),
        "fit_reps": fit,
        "heldout_reps": held,
        "note": (
            "Held out from the grid fit of leak, partner error and transfer only (leakage_frame_grid.py). "
            "Lifetime (35 us) and leaked readout (0.51) are medians of hidden Markov fits over all 15 jobs."
        ),
        "definition": "closure = (mean eps_leak - mean eps_model) / (mean eps_device - mean eps_model), means over the group's repetitions",
        "variants": {},
    }
    for name, data in variants.items():
        assert sorted(data) == sorted(fit + held), name
        res = {
            g: group(data, reps)
            for g, reps in (("all", sorted(data)), ("fit", fit), ("heldout", held))
        }
        res["splits"] = splits(data, fit)
        for cell, s in res["splits"].items():
            print(
                f"frame, {name}, {cell}: held-out minus fit {100 * s['actual_heldout_minus_fit']:+.1f} points; "
                f"{100 * s['share_of_splits_with_abs_diff_at_least_actual']:.0f}% of the 120 splits differ at least as much"
            )
        out["variants"][name] = res
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = exact()
    out["frame_binomial"] = frame_binomial()
    out["frame_heldout"] = frame_heldout()
    print("\nwrote", write_json("leakage_fit.json", out))


if __name__ == "__main__":
    main()

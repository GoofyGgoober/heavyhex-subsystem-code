"""The calibration model without leakage but with every CZ error scaled, against the leakage
model and the chip: does more CZ error alone give the chip's logical errors?

Jobs: the four the leakage fit used (leakage.FIT_JOBS: days 3 and 5). Variants, each
simulated with frame_sim.py and decoded exactly as the chip's shots were (pymatching on the
calibration model's own error model at the job's measured f), as leakage_frame_rerun.py does:

  model   the calibration model
  fit     the fitted leakage model, random-Pauli partner (leakage_frame_rerun.py "fit")
  czK     the calibration model without leakage, every CZ's depolarizing error times K
          (frame_sim.Leakage(cz_scale=K) with no leak; only DEPOLARIZE2 follows CZ in the model)

The error per round is fitted as the prespecified analysis fits it (fit_per_round); the chip's
comes from each job's analysis.json. Seeds follow leakage_frame_rerun.py (seed = number of
rounds), so "model" and "fit" reproduce leakage_frame_rerun_depolarize.json. Also records
the detection rates per patch of the 8-round circuits (stabilizer detectors of rounds 1-7,
and flags and relays), for the device and every variant.

The summary (results/cz_scaled.json): the error per round per job and averaged over the four
jobs; Lambda per repetition; and whether the logical error follows the detection rate. For
each job, basis and patch the no-leakage family (K = 1, 1.5, 2, 3) traces eps against the
mean bulk stabilizer detection rate of the 8-round circuit. Interpolating along that curve
gives the family's eps at the leakage model's and at the chip's detection rate, to set
against their actual eps (beyond the largest K the curve is extrapolated from the last two
K, and flagged).

    python docs/analysis/cz_scaled.py               # the summary from results/cz_scaled_runs.json, seconds
    python docs/analysis/cz_scaled.py --recompute   # the simulations first (heavy) [--shots 10000] [--scales 1.5 2 3]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

import frame_sim
import leakage_frame_rerun as rerun
from common import RESULTS, jobs, memory_index
from leakage import FIT_JOBS

LONGEST = max(ROUNDS)
RUNS = RESULTS / "cz_scaled_runs.json"


# ----------------------------------------------------------------------------- simulation
def make_variant(job, name, params):
    if name == "model":
        return None
    if name == "fit":
        return rerun.variant(job, "fit", *params, 1.0)
    if name.startswith("cz"):
        return frame_sim.Leakage(cz_scale=float(name[2:]))  # no leak: leak dict empty
    raise ValueError(name)


def detection_rates(det: np.ndarray, labels, rounds: int) -> dict:
    """Per patch: mean rate of bulk stabilizer detectors and of flag/relay detectors."""
    out = {}
    for patch in dict.fromkeys(str(p) for p, _, _ in labels):
        stab = [
            c
            for c, (p, r, n) in enumerate(labels)
            if str(p) == patch and " " not in n and 1 <= r <= rounds - 1
        ]
        flag = [c for c, (p, r, n) in enumerate(labels) if str(p) == patch and " " in n]
        out[patch] = {
            "stab": float(det[:, stab].mean()) if stab else None,
            "flag": float(det[:, flag].mean()) if flag else None,
        }
    return out


def run_job(job, variants, shots, params):
    errors = {v: {b: defaultdict(list) for b in "XZ"} for v in variants}
    rates = {v: {} for v in ["chip", *variants]}
    for basis in "XZ":
        for rounds in ROUNDS:
            idx = memory_index(basis, rounds)
            model, records, labels = job.model(idx)
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            if rounds == LONGEST:
                chip_det, _ = job.chip_detectors(idx, model, records)
                rates["chip"][basis] = detection_rates(chip_det, labels, rounds)
            for v in variants:
                det, obs = frame_sim.simulate(
                    model, shots, make_variant(job, v, params), seed=rounds
                )
                failed = R._decode(model, det) != obs
                for k, patch in enumerate(patches):
                    errors[v][basis][patch].append(float(failed[:, k].mean()))
                if rounds == LONGEST:
                    rates[v][basis] = detection_rates(det, labels, rounds)
    out = {"eps": {}, "raw": errors, "rates": rates}
    for v in variants:
        out["eps"][v] = {
            b: {
                p: [
                    fit_per_round(ROUNDS, e, shots).per_round,
                    fit_per_round(ROUNDS, e, shots).uncertainty,
                ]
                for p, e in errors[v][b].items()
            }
            for b in "XZ"
        }
    obs = job.analysis["observed"]["fits"]
    out["eps"]["chip"] = {b: {p: obs[b]["per_round"][p] for p in obs[b]["per_round"]} for b in "XZ"}
    return out


def simulate(shots: int, scales: list[float], names: list[str]) -> None:
    variants = ["model", "fit", *[f"cz{s:g}" for s in scales]]
    params = rerun.leakage_params()
    results = {}
    for job in jobs():
        if job.name not in names:
            continue
        t0 = time.time()
        results[job.name] = run_job(job, variants, shots, params)
        e = results[job.name]["eps"]
        print(job.name, f"({time.time() - t0:.0f} s)", flush=True)
        for b in "XZ":
            for p in e["chip"][b]:
                print(
                    f"  {b} d{p}: "
                    + "  ".join(f"{v} {100 * e[v][b][p][0]:.2f}" for v in ["chip", *variants]),
                    flush=True,
                )
    RUNS.write_text(
        json.dumps(
            {
                "shots": shots,
                "scales": scales,
                "variants": variants,
                "rounds": list(ROUNDS),
                "fit_params": rerun.FIT,
                "leak_params": params[1:],
                "jobs": results,
            },
            indent=1,
        )
        + "\n"
    )
    print("wrote", RUNS)


# ----------------------------------------------------------------------------- summary
def interp(x, xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x <= xs[-1]:
        return float(np.interp(x, xs, ys)), x < xs[0]
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + slope * (x - xs[-1])), True


def summarize() -> None:
    res = json.loads(RUNS.read_text())
    runs = res["jobs"]
    variants = res["variants"]
    cz = [v for v in variants if v.startswith("cz")]
    family = ["model", *cz]  # K = 1 then the scaled ones
    srcs = ["chip", *variants]
    out: dict = {"per_job": {}, "mean_over_jobs": {}, "lambda_per_rep": {}, "follow": []}

    print(
        f"Error per round in % (10,000 simulated shots per circuit; chip from analysis.json). Variants: {', '.join(srcs)}"
    )
    for name, r in runs.items():
        out["per_job"][name] = {}
        for b in "XZ":
            for d in sorted(r["eps"]["chip"][b]):
                row = {s: r["eps"][s][b][d] for s in srcs}
                out["per_job"][name][f"{b} d{d}"] = row
                print(
                    f"  {name:18s} {b} d{d}: "
                    + "  ".join(f"{s} {100 * v[0]:.1f}+-{100 * v[1]:.1f}" for s, v in row.items())
                )

    print("\nArithmetic mean over the four jobs (each patch once):")
    for b in "XZ":
        for d in ("3", "5"):
            vals = {
                s: [r["eps"][s][b][d][0] for r in runs.values() if d in r["eps"]["chip"][b]]
                for s in srcs
            }
            m = {s: float(np.mean(v)) for s, v in vals.items()}
            out["mean_over_jobs"][f"{b} d{d}"] = m
            gap = m["chip"] - m["model"]
            print(
                f"  {b} d{d}: "
                + "  ".join(f"{s} {100 * v:.1f}" for s, v in m.items())
                + "   | share of chip-model gap closed: "
                + "  ".join(f"{s} {100 * (m[s] - m['model']) / gap:.0f}%" for s in variants[1:])
            )

    print("\nLambda = eps3/eps5 per repetition:")
    reps: dict = {}
    for name, r in runs.items():
        key = name[:13]
        for s in srcs:
            for b in "XZ":
                for d, v in r["eps"][s][b].items():
                    reps.setdefault(key, {}).setdefault(s, {})[(b, d)] = v[0]
    for key, e in reps.items():
        row = {}
        for b in "XZ":
            row[b] = {s: e[s][(b, "3")] / e[s][(b, "5")] for s in srcs}
        out["lambda_per_rep"][key] = row
        print(
            f"  {key}: "
            + "   ".join(
                f"Lambda{b}: " + " ".join(f"{s} {v:.2f}" for s, v in row[b].items()) for b in "XZ"
            )
        )

    print("\nDetection rate (bulk stabilizer detectors of the 8-round circuit, %) and eps (%):")
    print(
        "  family = calibration model with CZ x K, no leakage; 'family eps at rate' = eps the family gives at that detection rate"
    )
    for name, r in runs.items():
        for b in "XZ":
            for d in sorted(r["eps"]["chip"][b]):
                rate = {s: r["rates"][s][b][d]["stab"] for s in srcs}
                flag = {s: r["rates"][s][b][d]["flag"] for s in srcs}
                eps = {s: r["eps"][s][b][d][0] for s in srcs}
                xs = [rate[s] for s in family]
                ys = [eps[s] for s in family]
                rec = {"job": name, "basis": b, "d": d, "rate": rate, "flag_rate": flag, "eps": eps}
                print(
                    f"  {name:18s} {b} d{d}: "
                    + "  ".join(f"{s} {100 * rate[s]:.2f}/{100 * eps[s]:.1f}" for s in srcs)
                )
                if any(v is not None for v in flag.values()):
                    print(
                        "      flag/relay rate: "
                        + "  ".join(f"{s} {100 * flag[s]:.2f}" for s in srcs if flag[s] is not None)
                    )
                ks = [1.0, *[float(v[2:]) for v in cz]]
                for s in ("fit", "chip"):
                    pred, out_of_range = interp(rate[s], xs, ys)
                    k_at, _ = interp(rate[s], xs, ks)
                    rec[f"family_eps_at_{s}_rate"] = pred
                    rec[f"{s}_extrapolated"] = bool(out_of_range)
                    rec[f"cz_factor_at_{s}_rate"] = k_at
                    print(
                        f"      at {s}'s rate {100 * rate[s]:.2f}% (CZ x {k_at:.1f}): family eps {100 * pred:.1f}%"
                        f"{' (extrapolated)' if out_of_range else ''} vs {s} eps {100 * eps[s]:.1f}%  (ratio {eps[s] / pred:.2f})"
                    )
                out["follow"].append(rec)

    print(
        "\nRatio (actual eps / family eps at the same bulk stabilizer detection rate), per case; mean over"
        " all cases and over interpolated cases only:"
    )
    for s in ("fit", "chip"):
        for b in "XZ":
            cases = [f for f in out["follow"] if f["basis"] == b]
            ratios = [f["eps"][s] / f[f"family_eps_at_{s}_rate"] for f in cases]
            inner = [r for r, f in zip(ratios, cases) if not f[f"{s}_extrapolated"]]
            print(
                f"  {s} {b} memory: "
                + " ".join(
                    f"{x:.2f}{'*' if f[f'{s}_extrapolated'] else ''}" for x, f in zip(ratios, cases)
                )
                + f"   mean {np.mean(ratios):.2f}; interpolated only: n={len(inner)}, mean {np.mean(inner):.2f},"
                f" range {min(inner):.2f}-{max(inner):.2f}"
            )
            out.setdefault("follow_ratio_mean", {})[f"{s} {b}"] = {
                "all": float(np.mean(ratios)),
                "interpolated": float(np.mean(inner)),
                "n_interpolated": len(inner),
                "min_interpolated": float(min(inner)),
                "max_interpolated": float(max(inner)),
            }
    print("  (* = detection rate outside the family's range, eps extrapolated)")

    print(
        "\nScaled no-leakage model minus leakage model ('fit'), percentage points of eps per round:"
    )
    for key in ["2026-10-08-r1", "mean"]:
        for v in cz:
            if key == "mean":
                diffs = {
                    k: out["mean_over_jobs"][k][v] - out["mean_over_jobs"][k]["fit"]
                    for k in out["mean_over_jobs"]
                }
            else:
                diffs = {k: row[v][0] - row["fit"][0] for k, row in out["per_job"][key].items()}
            out.setdefault("minus_fit", {}).setdefault(key, {})[v] = diffs
            print(
                f"  {key:14s} {v:6s}: "
                + "  ".join(f"{k} {100 * x:+.1f}" for k, x in diffs.items())
                + f"   mean |diff| {100 * np.mean(np.abs(list(diffs.values()))):.1f}"
            )

    (RESULTS / "cz_scaled.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--recompute", action="store_true", help="run the simulations (heavy) before the summary"
    )
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--scales", nargs="*", type=float, default=[1.5, 2.0, 3.0])
    ap.add_argument("--jobs", nargs="*", default=list(FIT_JOBS))
    a = ap.parse_args()
    if a.recompute:
        simulate(a.shots, a.scales, a.jobs)
    summarize()


if __name__ == "__main__":
    main()

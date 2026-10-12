"""Summarize cz_scaled.json: logical error per round and detection rates, per job and averaged.

1. Error per round (X and Z memory, d = 3 and 5) for the chip, the calibration model, the
   leakage model ("fit") and the no-leakage model with CZ error x K, per job and as the
   arithmetic mean over the four fit jobs; Lambda = eps3/eps5 per repetition.
2. Does the logical error follow the detection rate? For each job, basis and patch the
   no-leakage family (K = 1, 1.5, 2, ...) traces eps against the mean bulk stabilizer
   detection rate of the 8-round circuit. We read off, by linear interpolation along that
   curve, the eps the family gives at the leakage model's and at the chip's detection rate,
   and compare with their actual eps. Points beyond the largest K are extrapolated linearly
   from the last two K and flagged.

    python docs/diagnosis/revision/cz_scaled/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def interp(x, xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x <= xs[-1]:
        return float(np.interp(x, xs, ys)), x < xs[0]
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + slope * (x - xs[-1])), True


def main() -> None:
    res = json.loads((HERE / "cz_scaled.json").read_text())
    jobs = res["jobs"]
    variants = res["variants"]
    cz = [v for v in variants if v.startswith("cz")]
    family = ["model", *cz]  # K = 1 then the scaled ones
    srcs = ["chip", *variants]
    lines = []
    out = {"per_job": {}, "mean_over_jobs": {}, "lambda_per_rep": {}, "follow": []}

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"Error per round in % (10,000 simulated shots per circuit; chip from analysis.json). Variants: {', '.join(srcs)}")
    for name, r in jobs.items():
        out["per_job"][name] = {}
        for b in "XZ":
            for d in sorted(r["eps"]["chip"][b]):
                row = {s: r["eps"][s][b][d] for s in srcs}
                out["per_job"][name][f"{b} d{d}"] = row
                p(f"  {name:18s} {b} d{d}: " + "  ".join(f"{s} {100 * v[0]:.1f}+-{100 * v[1]:.1f}" for s, v in row.items()))

    p("\nArithmetic mean over the four jobs (each patch once):")
    for b in "XZ":
        for d in ("3", "5"):
            vals = {s: [r["eps"][s][b][d][0] for r in jobs.values() if d in r["eps"]["chip"][b]] for s in srcs}
            m = {s: float(np.mean(v)) for s, v in vals.items()}
            out["mean_over_jobs"][f"{b} d{d}"] = m
            gap = m["chip"] - m["model"]
            p(f"  {b} d{d}: " + "  ".join(f"{s} {100 * v:.1f}" for s, v in m.items())
              + "   | share of chip-model gap closed: " + "  ".join(f"{s} {100 * (m[s] - m['model']) / gap:.0f}%" for s in variants[1:]))

    p("\nLambda = eps3/eps5 per repetition:")
    reps = {}
    for name, r in jobs.items():
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
        p(f"  {key}: " + "   ".join(f"Lambda{b}: " + " ".join(f"{s} {v:.2f}" for s, v in row[b].items()) for b in "XZ"))

    p("\nDetection rate (bulk stabilizer detectors of the 8-round circuit, %) and eps (%):")
    p("  family = calibration model with CZ x K, no leakage; 'family eps at rate' = eps the family gives at that detection rate")
    for name, r in jobs.items():
        for b in "XZ":
            for d in sorted(r["eps"]["chip"][b]):
                rate = {s: r["rates"][s][b][d]["stab"] for s in srcs}
                flag = {s: r["rates"][s][b][d]["flag"] for s in srcs}
                eps = {s: r["eps"][s][b][d][0] for s in srcs}
                xs = [rate[s] for s in family]
                ys = [eps[s] for s in family]
                rec = {"job": name, "basis": b, "d": d, "rate": rate, "flag_rate": flag, "eps": eps}
                p(f"  {name:18s} {b} d{d}: " + "  ".join(f"{s} {100 * rate[s]:.2f}/{100 * eps[s]:.1f}" for s in srcs))
                if any(v is not None for v in flag.values()):
                    p("      flag/relay rate: " + "  ".join(f"{s} {100 * flag[s]:.2f}" for s in srcs if flag[s] is not None))
                ks = [1.0, *[float(v[2:]) for v in cz]]
                for s in ("fit", "chip"):
                    pred, out_of_range = interp(rate[s], xs, ys)
                    k_at, _ = interp(rate[s], xs, ks)
                    rec[f"family_eps_at_{s}_rate"] = pred
                    rec[f"{s}_extrapolated"] = bool(out_of_range)
                    rec[f"cz_factor_at_{s}_rate"] = k_at
                    p(f"      at {s}'s rate {100 * rate[s]:.2f}% (CZ x {k_at:.1f}): family eps {100 * pred:.1f}%"
                      f"{' (extrapolated)' if out_of_range else ''} vs {s} eps {100 * eps[s]:.1f}%  (ratio {eps[s] / pred:.2f})")
                out["follow"].append(rec)

    p("\nRatio (actual eps / family eps at the same bulk stabilizer detection rate), per case; mean over"
      " all cases and over interpolated cases only:")
    for s in ("fit", "chip"):
        for b in "XZ":
            cases = [f for f in out["follow"] if f["basis"] == b]
            ratios = [f["eps"][s] / f[f"family_eps_at_{s}_rate"] for f in cases]
            inner = [r for r, f in zip(ratios, cases) if not f[f"{s}_extrapolated"]]
            p(f"  {s} {b} memory: " + " ".join(f"{x:.2f}{'*' if f[f'{s}_extrapolated'] else ''}" for x, f in zip(ratios, cases))
              + f"   mean {np.mean(ratios):.2f}; interpolated only: n={len(inner)}, mean {np.mean(inner):.2f},"
              f" range {min(inner):.2f}-{max(inner):.2f}")
            out.setdefault("follow_ratio_mean", {})[f"{s} {b}"] = {
                "all": float(np.mean(ratios)), "interpolated": float(np.mean(inner)), "n_interpolated": len(inner),
                "min_interpolated": float(min(inner)), "max_interpolated": float(max(inner))}
    p("  (* = detection rate outside the family's range, eps extrapolated)")

    p("\nScaled no-leakage model minus leakage model ('fit'), percentage points of eps per round:")
    for key in ["2026-10-08-r1", "mean"]:
        for v in cz:
            if key == "mean":
                diffs = {k: out["mean_over_jobs"][k][v] - out["mean_over_jobs"][k]["fit"] for k in out["mean_over_jobs"]}
            else:
                diffs = {k: row[v][0] - row["fit"][0] for k, row in out["per_job"][key].items()}
            out.setdefault("minus_fit", {}).setdefault(key, {})[v] = diffs
            p(f"  {key:14s} {v:6s}: " + "  ".join(f"{k} {100 * x:+.1f}" for k, x in diffs.items())
              + f"   mean |diff| {100 * np.mean(np.abs(list(diffs.values()))):.1f}")

    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

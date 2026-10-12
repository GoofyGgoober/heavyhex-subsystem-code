"""Leakage per readout qubit: a two-state hidden Markov model fitted to each flag's and relay's firings.

In memory every flag and relay should read 0 each round, in either basis, and the calibration
model makes each round independent. A qubit that leaks out of {|0>, |1>} reads as 1 for as long
as it stays out, because a measure-and-flip reset cannot bring it back. Each readout
qubit's 0/1 record over the rounds is fitted with two hidden states, normal (N) and
leaked (L):

    P(L at round 0) = pi,  P(N -> L) per round = a,  P(L -> N) per round = b,
    P(fires | N) = e_N,    P(fires | L) = e_L.

The same fit on shots sampled from the calibration model is the control: there it
should find no persistent state. Rounds 4, 6 and 8 of one basis are pooled per job.
Z memory is the paper's, and hmm_bootstrap.py gives its medians' uncertainty; X memory
(common.readout_model) repeats the fit on the same qubits.

    python docs/analysis/hmm.py               # the medians from results/hmm.json: results/hmm_summary.json
    python docs/analysis/hmm.py --recompute   # fit every series first (2-3 min): results/hmm.json
    python docs/analysis/hmm.py --basis X --recompute   # X memory: hmm_x.json, hmm_x_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import (
    RESULTS,
    ROUND_US,
    Job,
    job_folders,
    memory_index,
    readout_model,
    readout_series,
    write_json,
)

LENGTHS = (4, 6, 8)
NAMES = {"Z": ("hmm.json", "hmm_summary.json"), "X": ("hmm_x.json", "hmm_x_summary.json")}


def em(groups: list[np.ndarray], iters: int = 300, tol: float = 1e-7) -> dict:
    """Baum-Welch on 0/1 sequences (one array of shots x rounds per length).

    Shots with the same 0/1 pattern enter once, weighted by their number: the same fit as shot
    by shot, to rounding, in a fraction of the time. Stops when the log-likelihood rises by less
    than tol, or after iters updates ("capped").
    """
    pi, a, b, e0, e1 = 0.02, 0.01, 0.3, max(np.mean([g.mean() for g in groups]), 1e-3), 0.7
    data = [np.unique(o, axis=0, return_counts=True) for o in groups]
    last = -np.inf
    capped, iterations = True, 0
    for _ in range(iters):
        iterations += 1
        n = defaultdict(float)
        ll = 0.0
        for o, w in data:
            S, T = o.shape
            w = w.astype(float)
            A = np.array([[1 - a, a], [b, 1 - b]])
            B = np.stack([np.where(o, e0, 1 - e0), np.where(o, e1, 1 - e1)], axis=2)  # S x T x 2
            alpha = np.empty((S, T, 2))
            c = np.empty((S, T))
            x = np.stack([np.full(S, 1 - pi), np.full(S, pi)], axis=1) * B[:, 0]
            c[:, 0] = x.sum(1)
            alpha[:, 0] = x / c[:, :1]
            for t in range(1, T):
                x = (alpha[:, t - 1] @ A) * B[:, t]
                c[:, t] = x.sum(1)
                alpha[:, t] = x / c[:, t : t + 1]
            beta = np.ones((S, T, 2))
            for t in range(T - 2, -1, -1):
                beta[:, t] = ((B[:, t + 1] * beta[:, t + 1]) @ A.T) / c[:, t + 1 : t + 2]
            gamma = alpha * beta
            gamma /= gamma.sum(2, keepdims=True)
            xi = (
                alpha[:, :-1, :, None]
                * A[None, None]
                * (B[:, 1:] * beta[:, 1:])[:, :, None, :]
                / c[:, 1:, None, None]
            )
            ll += w @ np.log(c).sum(1)
            n["pi"] += w @ gamma[:, 0, 1]
            n["S"] += w.sum()
            n["01"] += w @ xi[:, :, 0, 1].sum(1)
            n["0"] += w @ gamma[:, :-1, 0].sum(1)
            n["10"] += w @ xi[:, :, 1, 0].sum(1)
            n["1"] += w @ gamma[:, :-1, 1].sum(1)
            n["f0"] += w @ (gamma[:, :, 0] * o).sum(1)
            n["g0"] += w @ gamma[:, :, 0].sum(1)
            n["f1"] += w @ (gamma[:, :, 1] * o).sum(1)
            n["g1"] += w @ gamma[:, :, 1].sum(1)
        pi = n["pi"] / n["S"]
        a = n["01"] / max(n["0"], 1e-12)
        b = n["10"] / max(n["1"], 1e-12)
        e0 = n["f0"] / max(n["g0"], 1e-12)
        e1 = n["f1"] / max(n["g1"], 1e-12)
        a, b, pi = (float(np.clip(v, 1e-6, 1 - 1e-6)) for v in (a, b, pi))
        e0, e1 = float(np.clip(e0, 1e-6, 1 - 1e-6)), float(np.clip(e1, 1e-6, 1 - 1e-6))
        if ll - last < tol:
            capped = False
            break
        last = ll
    # One state, an independent rate per round: the calibration model's shape.
    ll1 = sum(
        (
            o.sum(0) * np.log(np.clip(o.mean(0), 1e-9, 1))
            + (~o).sum(0) * np.log(np.clip(1 - o.mean(0), 1e-9, 1))
        ).sum()
        for o in groups
    )
    return {
        "pi": pi,
        "a": a,
        "b": b,
        "e_N": e0,
        "e_L": e1,
        "llr": float(ll - ll1),
        "shots": int(sum(len(o) for o in groups)),
        "iterations": iterations,
        "capped": capped,
    }


def fit_all(basis: str = "Z") -> list[dict]:
    rows = []
    for job in map(Job, job_folders()):  # one job in memory at a time
        chip_groups, sim_groups = defaultdict(list), defaultdict(list)
        for rounds in LENGTHS:
            idx = memory_index(basis, rounds)
            model, records, labels = readout_model(job, idx)
            chip, _ = job.chip_detectors(idx, model, records)
            sim = np.asarray(model.compile_detector_sampler(seed=rounds).sample(len(chip)))
            for key, cols in readout_series(labels).items():
                chip_groups[key].append(chip[:, cols])
                sim_groups[key].append(sim[:, cols])
        for key in chip_groups:
            patch, name, k = key
            row = {
                "job": job.name,
                "day": job.day,
                "rep": job.rep,
                "patch": patch,
                "readout": name,
                "qubit": job.readout_qubit(patch, name, k),
            }
            row["chip"] = em(chip_groups[key])
            row["model"] = em(sim_groups[key])
            rows.append(row)
        print(job.name, "done", flush=True)
    (RESULTS / NAMES[basis][0]).write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Medians over the series, on the device and on the model's shots."""
    out: dict = {
        "series": len(rows),
        "flags": sum(r["readout"].startswith("z_flag") for r in rows),
        "relays": sum(r["readout"].startswith("relay") for r in rows),
        "qubits": len({r["qubit"] for r in rows}),
        "jobs": len({r["job"] for r in rows}),
    }
    for src in ("chip", "model"):
        v = {k: np.array([r[src][k] for r in rows]) for k in ("a", "b", "e_N", "e_L", "pi", "llr")}
        life = -ROUND_US / np.log(1 - v["b"])
        out[src] = {
            "a_median": float(np.median(v["a"])),
            "a_iqr": np.percentile(v["a"], [25, 75]).tolist(),
            "b_median": float(np.median(v["b"])),
            "lifetime_us_median": float(np.median(life)),
            "lifetime_us_iqr": np.percentile(life, [25, 75]).tolist(),
            "round_us_times_median_1_over_b": float(np.median(1 / np.maximum(v["b"], 1e-6)))
            * ROUND_US,
            "e_L_median": float(np.median(v["e_L"])),
            "e_N_median": float(np.median(v["e_N"])),
            "pi_median": float(np.median(v["pi"])),
            "llr_median": float(np.median(v["llr"])),
            "llr_max": float(v["llr"].max()),
            "llr_above_10": int((v["llr"] > 10).sum()),
        }
        if all("capped" in r[src] for r in rows):  # recorded since the X-memory fits
            out[src]["capped_at_300"] = sum(r[src]["capped"] for r in rows)
        s = out[src]
        print(
            f"{src:5s}: onset a {s['a_median']:.4f}; lifetime {s['lifetime_us_median']:.0f} us "
            f"(7.9 us x median 1/b = {s['round_us_times_median_1_over_b']:.1f} us); e_L {s['e_L_median']:.2f}; "
            f"e_N {s['e_N_median']:.3f}; log-likelihood gain median {s['llr_median']:.1f}, > 10 in {s['llr_above_10']}"
        )
    print(
        f"{out['series']} series ({out['flags']} flags, {out['relays']} relays) from {out['qubits']} qubits"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="fit every series again (2-3 min)")
    ap.add_argument("--basis", choices=("Z", "X"), default="Z", help="memory basis (default Z)")
    a = ap.parse_args()
    rows = (
        fit_all(a.basis) if a.recompute else json.loads((RESULTS / NAMES[a.basis][0]).read_text())
    )
    write_json(NAMES[a.basis][1], summarize(rows))


if __name__ == "__main__":
    main()

"""Why do some qubits leak and err more than the calibration says? Per readout qubit.

Joins each flag's and relay's fitted leakage (hmm.json) with IBM's properties for that day
(properties.json) and with what the qubit does in the circuit as sent, then ranks the
candidate causes by Spearman correlation:

- leak per CZ: the fitted leak rate per round over the CZs the qubit takes part in per round;
- fresh excess: e_N, the firing rate outside leakage, minus the model's own e_N;
- candidates: the CZ errors of the qubit's couplers, its readout and reset errors, T1, how
  many CZs it does per round, and crosstalk exposure: for each of its CZs, how many other
  CZs run at the same time on couplers that touch a neighbour of the pair (from the
  schedule the job sent, with each gate's calibrated duration).

Also the onset rate by role (flag or relay), the qubits that served in both roles, and the
line through onset rate against CZs per round. Flags take 8.25 CZs per round and relays 3,
so that line only joins the two roles' means. Under a minute.

    python docs/analysis/hmm_causes.py   # writes results/hmm_causes.json (per series) and hmm_causes_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from common import REPO, RESULTS, jobs, memory_index, write_json

FEZ = json.loads((REPO / "docs/figures/fez_map.json").read_text())
FEATURES = [
    "cz_per_round",
    "crowding",
    "cz_error_mean",
    "cz_error_max",
    "p10",
    "p01",
    "init_error",
    "t1",
]
TARGETS = ["leak_per_round", "leak_per_cz", "fresh_excess", "lifetime_rounds", "read_leaked"]


def neighbours() -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for a, b in FEZ["edges"]:
        out[a].add(b)
        out[b].add(a)
    return out


def schedule(job, idx: int):
    """Each CZ as (start, end, a, b) in ns, timed as the job's circuit runs."""
    circuit = job.circuits[idx]
    ns = job.calibration.durations_ns or {"x": 24, "cz": 68, "measure": 1660, "reset": 1684}
    length = {
        "x": ns["x"],
        "sx": ns["x"],
        "cz": ns["cz"],
        "measure": ns["measure"],
        "reset": ns["reset"],
        "rz": 0,
    }
    dt = job.info["dt"] * 1e9
    free: dict[int, float] = defaultdict(float)
    czs = []
    for item in circuit.data:
        name = item.operation.name
        qs = [circuit.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            if item.operation.label == "h frame":
                continue
            t = max(free[q] for q in qs)
            for q in qs:
                free[q] = t
            continue
        start = max(free[q] for q in qs)
        d = item.operation.duration * dt if name == "delay" else length[name]
        if name == "cz":
            czs.append((start, start + d, qs[0], qs[1]))
        for q in qs:
            free[q] = start + d
    return czs


def exposure(czs, nb) -> tuple[dict, dict]:
    """Per qubit: CZs per run, and the mean number of simultaneous CZs next to each of its CZs."""
    count, crowd = defaultdict(int), defaultdict(list)
    for i, (s, e, a, b) in enumerate(czs):
        near = (nb[a] | nb[b]) - {a, b}
        others = sum(
            1
            for j, (s2, e2, c, d) in enumerate(czs)
            if j != i and s2 < e and s < e2 and (c in near or d in near)
        )
        for q in (a, b):
            count[q] += 1
            crowd[q].append(others)
    return count, {q: float(np.mean(v)) for q, v in crowd.items()}


def main() -> None:
    hmm = json.loads((RESULTS / "hmm.json").read_text())
    props = json.loads((RESULTS / "properties.json").read_text())
    nb = neighbours()
    feats_by_job = {}
    for j in jobs():
        czs = schedule(j, memory_index("Z", 8))
        count, crowd = exposure(czs, nb)
        feats_by_job[j.name] = ({q: c / 8 for q, c in count.items()}, crowd)
    rows = []
    for r in hmm:
        q, day = r["qubit"], r["day"]
        P, C = props[day]["qubits"], props[day]["cz"]
        pq = P[str(q)]
        per_round, crowd = feats_by_job[r["job"]]
        n_cz = per_round.get(q, np.nan)
        couplers = [C.get("-".join(map(str, sorted((q, n))))) for n in nb[q]]
        couplers = [c for c in couplers if c is not None and c < 0.5]
        rows.append(
            {
                "job": r["job"],
                "day": day,
                "qubit": q,
                "role": r["readout"].split()[0],
                "leak_per_round": r["chip"]["a"],
                "leak_per_cz": r["chip"]["a"] / n_cz if n_cz else np.nan,
                "lifetime_rounds": 1 / max(r["chip"]["b"], 1e-6),
                "read_leaked": r["chip"]["e_L"],
                "fresh_excess": r["chip"]["e_N"] - r["model"]["e_N"],
                "cz_per_round": n_cz,
                "crowding": crowd.get(q, np.nan),
                "cz_error_mean": float(np.mean(couplers)) if couplers else np.nan,
                "cz_error_max": float(np.max(couplers)) if couplers else np.nan,
                "p10": pq["prob_meas1_prep0"],
                "p01": pq["prob_meas0_prep1"],
                "readout": pq["readout_error"],
                "init_error": pq.get("init_error"),
                "t1": pq["T1"],
                "llr": r["chip"]["llr"],
            }
        )
    (RESULTS / "hmm_causes.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    summary: dict = {"spearman": {}}
    for target in TARGETS:
        y = np.array([r[target] for r in rows], float)
        print(
            f"\n{target}: median {np.nanmedian(y):.4g}, IQR {np.nanpercentile(y, 25):.4g}-{np.nanpercentile(y, 75):.4g} over {np.isfinite(y).sum()} qubit-jobs"
        )
        summary["spearman"][target] = {}
        for fe in FEATURES:
            x = np.array([np.nan if r[fe] is None else r[fe] for r in rows], float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[ok], y[ok])
            summary["spearman"][target][fe] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
            print(f"   vs {fe:14s} rho {rho:+.2f}  p {p:.1e}{'  <--' if p < 0.01 else ''}")
    summary["by_role"] = {}
    for role in ("z_flag", "relay"):
        sel = [r for r in rows if r["role"] == role]
        summary["by_role"][role] = {
            "series": len(sel),
            "onset_per_round_median": float(np.median([r["leak_per_round"] for r in sel])),
            "onset_per_cz_median": float(np.nanmedian([r["leak_per_cz"] for r in sel])),
            "cz_per_round": float(np.mean([r["cz_per_round"] for r in sel])),
        }
        print(
            f"\n{role}: n={len(sel)}, leak per round {np.median([r['leak_per_round'] for r in sel]):.4f}, "
            f"per CZ {np.nanmedian([r['leak_per_cz'] for r in sel]):.4f}, CZs per round {np.nanmedian([r['cz_per_round'] for r in sel]):.1f}, "
            f"crowding {np.nanmedian([r['crowding'] for r in sel]):.1f}"
        )
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qubit"]].append(r["leak_per_cz"])
    multi = {q: v for q, v in by_q.items() if len(v) >= 3}
    within = np.mean([np.std(v) for v in multi.values()])
    between = np.std([np.mean(v) for v in multi.values()])
    print(
        f"\nleak per CZ, qubits seen 3+ times: spread within a qubit {within:.4f}, between qubits {between:.4f} ({len(multi)} qubits)"
    )
    top = sorted(multi.items(), key=lambda kv: -np.mean(kv[1]))[:6]
    print("leakiest:", "; ".join(f"q{q} {np.mean(v):.4f}" for q, v in top))

    # qubits that served as a flag in some jobs and as a relay in others
    roles = defaultdict(lambda: defaultdict(list))
    for r in rows:
        roles[r["qubit"]][r["role"]].append(r["leak_per_round"])
    both = {q: d for q, d in roles.items() if len(d) == 2}
    ratio = {q: float(np.mean(d["z_flag"]) / np.mean(d["relay"])) for q, d in both.items()}
    summary["both_roles"] = {
        "qubits": len(both),
        "flag_over_relay_onset": ratio,
        "median_ratio": float(np.median(list(ratio.values()))),
    }
    print(
        f"\n{len(both)} qubits in both roles: onset as a flag / as a relay, median {np.median(list(ratio.values())):.2f}"
    )

    # onset per round = per_cz * CZs per round + per_round, least squares over the series
    x = np.array([r["cz_per_round"] for r in rows], float)
    y = np.array([r["leak_per_round"] for r in rows], float)
    per_cz, per_round = np.polyfit(x, y, 1)
    summary["onset_line"] = {
        "per_cz": float(per_cz),
        "per_round": float(per_round),
        "cz_per_round_values": sorted({float(v) for v in x}),
        "note": "two CZ counts only (relays 3, flags 8.25): the line joins the two roles' means",
    }
    print(
        f"onset per round = {100 * per_cz:.2f}% per CZ x CZs per round + {100 * per_round:.2f}% (two CZ counts only)"
    )
    write_json("hmm_causes_summary.json", summary)


if __name__ == "__main__":
    main()

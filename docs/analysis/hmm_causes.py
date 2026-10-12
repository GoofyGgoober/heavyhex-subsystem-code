"""Why do some qubits leak and err more than the calibration says? Per readout qubit.

Joins each flag's and relay's fitted leakage (hmm.json, Z memory) with IBM's properties for
that day (properties.json) and with what the qubit does in the circuit as sent, then ranks the
candidate causes by Spearman correlation:

- leak per CZ: the fitted leak rate per round over the CZs the qubit takes part in per round;
- leak per event: the same over its readouts plus resets per round;
- fresh excess: e_N, the firing rate outside leakage, minus the model's own e_N;
- candidates: the CZ errors of the qubit's couplers, its readout and reset errors, T1, how
  many CZs it does per round, and crowding: for each of its CZs, how many other CZs run at
  the same time on couplers that touch a neighbour of the pair (from the schedule the job
  sent, with each gate's calibrated duration).

Operations per round are the steady state, (count in 8 rounds - count in 4 rounds) / 4, which
leaves out the preparation half: a flag takes part in 8 CZs, 2 readouts and 2 resets per
round, a relay in 3, 1 and 1. The correlations are given over the 320 series and, since the
series come from 38 physical qubits, over qubits (each qubit's mean over its series), with a
95% interval from a bootstrap over qubits. Also the onset rate by role under each
normalization, the qubits that served in both roles, and the line through onset rate
against CZs per round, which with two CZ counts only joins the two roles' means.
Paper: Sec. VI (onset rate by role, the normalizations, the correlations) and Fig. leak (b).
Under a minute.

    python docs/analysis/hmm_causes.py   # writes results/hmm_causes.json (per series) and hmm_causes_summary.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr

from common import REPO, RESULTS, Job, job_folders, memory_index, write_json

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
QUBIT_TARGETS = ["leak_per_round", "leak_per_round_role_adjusted"]
BOOT, SEED = 4000, 1


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


def operations(job, idx: int) -> tuple[Counter, Counter]:
    """Per chip qubit: its CZs, and its readouts plus resets, in the job's circuit idx."""
    circuit = job.circuits[idx]
    cz, events = Counter(), Counter()
    for item in circuit.data:
        name = item.operation.name
        if name in ("cz", "measure", "reset"):
            for q in item.qubits:
                (cz if name == "cz" else events)[circuit.find_bit(q).index] += 1
    return cz, events


def steady_state(job) -> tuple[dict, dict]:
    """CZs and readouts plus resets per round, from the 8- and 4-round Z-memory circuits."""
    (cz8, ev8), (cz4, ev4) = (operations(job, memory_index("Z", n)) for n in (8, 4))
    return (
        {q: (cz8[q] - cz4[q]) / 4 for q in cz8},
        {q: (ev8[q] - ev4[q]) / 4 for q in ev8},
    )


def exposure(czs, nb) -> dict:
    """Per qubit: the mean number of simultaneous CZs next to each of its CZs."""
    crowd = defaultdict(list)
    for i, (s, e, a, b) in enumerate(czs):
        near = (nb[a] | nb[b]) - {a, b}
        others = sum(
            1
            for j, (s2, e2, c, d) in enumerate(czs)
            if j != i and s2 < e and s < e2 and (c in near or d in near)
        )
        for q in (a, b):
            crowd[q].append(others)
    return {q: float(np.mean(v)) for q, v in crowd.items()}


def spearman_ci(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """Spearman rho over units (qubits) with a 95% percentile interval from resampling them."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y)[0]
    boot = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boot.append(spearmanr(x[i], y[i])[0])
    return {
        "rho": float(rho),
        "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "n": int(ok.sum()),
    }


def main() -> None:
    hmm = json.loads((RESULTS / "hmm.json").read_text())
    props = json.loads((RESULTS / "properties.json").read_text())
    nb = neighbours()
    feats_by_job = {}
    for j in map(Job, job_folders()):  # one job in memory at a time
        cz, events = steady_state(j)
        feats_by_job[j.name] = (cz, events, exposure(schedule(j, memory_index("Z", 8)), nb))
    rows = []
    for r in hmm:
        q, day = r["qubit"], r["day"]
        P, C = props[day]["qubits"], props[day]["cz"]
        pq = P[str(q)]
        per_round, events, crowd = feats_by_job[r["job"]]
        n_cz = per_round.get(q, np.nan)
        n_ev = events.get(q, np.nan)
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
                "leak_per_event": r["chip"]["a"] / n_ev if n_ev else np.nan,
                "lifetime_rounds": 1 / max(r["chip"]["b"], 1e-6),
                "read_leaked": r["chip"]["e_L"],
                "fresh_excess": r["chip"]["e_N"] - r["model"]["e_N"],
                "cz_per_round": n_cz,
                "events_per_round": n_ev,
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
    role_median = {}
    for role in ("z_flag", "relay"):
        sel = [r for r in rows if r["role"] == role]
        role_median[role] = float(np.median([r["leak_per_round"] for r in sel]))
        summary["by_role"][role] = {
            "series": len(sel),
            "onset_per_round_median": role_median[role],
            "onset_per_cz_median": float(np.nanmedian([r["leak_per_cz"] for r in sel])),
            "onset_per_event_median": float(np.nanmedian([r["leak_per_event"] for r in sel])),
            "cz_per_round": sorted({r["cz_per_round"] for r in sel}),
            "events_per_round": sorted({r["events_per_round"] for r in sel}),
        }
        s = summary["by_role"][role]
        print(
            f"\n{role}: n={len(sel)}, onset per round {100 * s['onset_per_round_median']:.2f}%, "
            f"per CZ {100 * s['onset_per_cz_median']:.2f}% ({s['cz_per_round']} CZs per round), per readout or "
            f"reset {100 * s['onset_per_event_median']:.2f}% ({s['events_per_round']} per round), "
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

    # Qubits that served as a flag in some jobs and as a relay in others. Each normalization
    # predicts a flag/relay ratio: 1 per round, 4/2 per readout or reset, 8/3 per CZ.
    rng = np.random.default_rng(SEED)
    roles = defaultdict(lambda: defaultdict(list))
    for r in rows:
        roles[r["qubit"]][r["role"]].append(r["leak_per_round"])
    both = {q: d for q, d in roles.items() if len(d) == 2}
    ratio = {q: float(np.mean(d["z_flag"]) / np.mean(d["relay"])) for q, d in both.items()}
    logs = np.log(list(ratio.values()))
    geo = [np.exp(logs[rng.integers(0, len(logs), len(logs))].mean()) for _ in range(BOOT)]
    flag, relay = summary["by_role"]["z_flag"], summary["by_role"]["relay"]
    summary["both_roles"] = {
        "qubits": len(both),
        "flag_over_relay_onset": ratio,
        "series_per_role": {q: {k: len(v) for k, v in d.items()} for q, d in both.items()},
        "median_ratio": float(np.median(list(ratio.values()))),
        "geometric_mean_ratio": float(np.exp(logs.mean())),
        "geometric_mean_ratio_ci95_over_qubits": np.percentile(geo, [2.5, 97.5]).tolist(),
        "flag_above_relay": int((logs > 0).sum()),
        "sign_test_p_two_sided": float(binomtest(int((logs > 0).sum()), len(logs)).pvalue),
        "ratio_predicted": {
            "per_round": 1.0,
            "per_readout_or_reset": flag["events_per_round"][0] / relay["events_per_round"][0],
            "per_cz": flag["cz_per_round"][0] / relay["cz_per_round"][0],
        },
    }
    b = summary["both_roles"]
    print(
        f"\n{len(both)} qubits in both roles: onset as a flag / as a relay, median {b['median_ratio']:.2f}, "
        f"geometric mean {b['geometric_mean_ratio']:.2f} ({b['geometric_mean_ratio_ci95_over_qubits'][0]:.2f}-"
        f"{b['geometric_mean_ratio_ci95_over_qubits'][1]:.2f}), above 1 in {b['flag_above_relay']} "
        f"(sign test p {b['sign_test_p_two_sided']:.3f}); predicted per readout or reset "
        f"{b['ratio_predicted']['per_readout_or_reset']:.2f}, per CZ {b['ratio_predicted']['per_cz']:.2f}"
    )

    # Over qubits: each physical qubit's mean over its series, so the 320 series count as 38
    # units. The role-adjusted onset divides each series' a by its role's median first.
    for r in rows:
        r["leak_per_round_role_adjusted"] = r["leak_per_round"] / role_median[r["role"]]
    units = defaultdict(list)
    for r in rows:
        units[r["qubit"]].append(r)
    qubits = sorted(units)

    def per_qubit(key: str) -> np.ndarray:
        out = []
        for q in qubits:
            v = np.array([np.nan if r[key] is None else r[key] for r in units[q]], float)
            v = v[np.isfinite(v)]
            out.append(v.mean() if len(v) else np.nan)
        return np.array(out)

    summary["spearman_over_qubits"] = {}
    print(f"\nover {len(qubits)} qubits, 95% interval from resampling qubits:")
    for target in QUBIT_TARGETS:
        y = per_qubit(target)
        summary["spearman_over_qubits"][target] = {}
        for fe in FEATURES[1:]:
            s = spearman_ci(per_qubit(fe), y, rng)
            summary["spearman_over_qubits"][target][fe] = s
            print(
                f"   {target:28s} vs {fe:14s} rho {s['rho']:+.2f} ({s['ci95'][0]:+.2f} to {s['ci95'][1]:+.2f}), n {s['n']}"
            )

    # onset per round = per_cz * CZs per round + per_round, least squares over the series
    x = np.array([r["cz_per_round"] for r in rows], float)
    y = np.array([r["leak_per_round"] for r in rows], float)
    per_cz, per_round = np.polyfit(x, y, 1)
    summary["onset_line"] = {
        "per_cz": float(per_cz),
        "per_round": float(per_round),
        "cz_per_round_values": sorted({float(v) for v in x}),
        "note": "two CZ counts only (relays 3, flags 8): the line joins the two roles' means",
    }
    print(
        f"\nonset per round = {100 * per_cz:.2f}% per CZ x CZs per round + {100 * per_round:.2f}% (two CZ counts only)"
    )
    write_json("hmm_causes_summary.json", summary)


if __name__ == "__main__":
    main()

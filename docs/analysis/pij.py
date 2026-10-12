"""Edge probabilities learned from the detection events, against the model's (Sec. agreement, App. decoder).

For every job and both bases, the 6- and 8-round circuits: each edge of the decoder's matching
graph (the calibration model at the job's measured f) gets p_ij from the device's detection
events, and each boundary edge the remainder of p_i (detectors.py). Edges of one type (same
detectors, same round difference, same position in time) are averaged over rounds and over
the two circuits. The model enters as the estimator sees it: the same formulas on the model's
exact moments (detectors.model_pairs), so pieces of mechanisms that fire three or more
detectors are counted the same way on both sides.

  types      per job, patch, basis and edge type: class, instances, the graph's p, the model's
             p as the estimator sees it, the device's p with its bootstrap SE over shots
             (200 replicates), and the same estimator on 2500 shots sampled from the model
             (the first 2500 of pij_decode.py's 10,000: the control); pij_decode.py decodes
             with the device's values
  classes    medians of device/model per class (space-like, time-like, diagonal, boundary,
             flag/relay) and basis, per distance, pooled over jobs and per placement, with the
             bootstrap SE of the median, the share of types above one, and the control's
             median; time-like and flag/relay boundary edges also binned by the model's p,
             mostly the readout and reset errors of the qubits read (each the reported
             readout error)
  unmodelled pairs the device's p_ij for pairs of detectors that no mechanism of the model
             flips together, in bulk rounds, by round difference and whether it is the same
             detector, against the control (mean over a patch's pairs in a job, then mean and SE
             over the patches of the jobs; pairs with an estimator denominator below 0.1 left
             out; "both" pools the two distances)
  high_p     every edge of the 8-round circuits' graphs with p > 0.1, per type and job, with
             the circuit faults behind it (stim's explain_detector_error_model_errors): kind
             (idle, CZ, readout, reset, one-qubit gate), chip qubits, role, reported T1, T2
             and readout error; and which qubits lead them over the fifteen jobs. The eight
             rounds hold every edge type of the shorter circuits except, in the one-round
             circuit, those from round 1 to the final layer.
  matrix     the device's and the model's p_ij for every pair of detectors of placement 1's
             distance-3 patch, eight rounds, both bases, mean over the five days (Fig. pij)

Edges that touch the distance-5 Z-memory stabilizers are not estimated (detectors.saturated).

About a minute on 2 workers, 0.4 GB each.

    python docs/analysis/pij.py           # writes results/pij.json
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import stim

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.patches.operators import build_operators

import detectors as DT
from common import WORKERS, jobs, memory_index, write_json
from matching import calibrated_matching
from refit import SIM_SHOTS, sim_seed

USED = (6, 8)  # round counts the edge probabilities are learned from
B = 200
LEARN_SHOTS = 2500  # the control learns from as many shots as the device has
HIGH_P = 0.1


def sig(x) -> float | None:
    """Five significant digits, NaN as None (keeps pij.json small)."""
    x = float(x)
    return float(f"{x:.5g}") if np.isfinite(x) else None


# ------------------------------------------------------------------ physical origin of an edge
KIND = {
    "PAULI_CHANNEL_1": "idle",
    "DEPOLARIZE2": "CZ",
    "DEPOLARIZE1": "one-qubit gate",
    "X_ERROR": "reset",
    "M": "readout",
}


def location_p(loc) -> float:
    it = loc.instruction_targets
    a = it.args
    if it.gate == "PAULI_CHANNEL_1":
        t = loc.flipped_pauli_product[0].gate_target
        return a[0] if t.is_x_target else a[1] if t.is_y_target else a[2]
    if it.gate == "DEPOLARIZE1":
        return a[0] / 3
    if it.gate == "DEPOLARIZE2":
        return a[0] / 15
    return a[0]


def role(job, chip_qubit: int) -> str:
    for patch, chip in job.info["chips"].items():
        if chip_qubit not in chip:
            continue
        idx = chip.index(chip_qubit)
        r = qubit_roles(build_operators(int(patch)))
        if idx < r.num_data:
            return f"d{patch} data Q{idx + 1}"
        for name, q in r.x_ancillas.items():
            if q == idx:
                return f"d{patch} X ancilla {name}"
        for name, q in r.z_ancillas.items():
            if q == idx:
                return f"d{patch} Z ancilla {name}"
        for name, arms in r.z2_arms.items():
            for k, (_data, relay) in enumerate(arms):
                if relay == idx:
                    return f"d{patch} relay {name}#{k}"
    return "?"


def canon(dets: set[int], obs: set[int]) -> str:
    return " ".join([f"D{d}" for d in sorted(dets)] + [f"L{o}" for o in sorted(obs)])


def origins(job, model: stim.Circuit, g: dict, edges: list[int]) -> dict[int, dict]:
    """For each edge index, the circuit faults behind it, summed by (kind, chip qubits), the
    three largest with their share of the explained probability.

    The faults are those whose whole symptom is the edge (with or without the logical); an
    edge made only of pieces of larger faults is explained by its two largest such faults
    ("pieces"). Stim matches faults by symptom, not by how the DEM split them, so a share is
    of the faults with that symptom, not of the edge's p exactly."""
    by_comp = defaultdict(list)
    for ins in model.detector_error_model(decompose_errors=True).flattened():
        if ins.type != "error":
            continue
        parts: list[set] = [set()]
        dets: set[int] = set()
        obs: set[int] = set()
        for t in ins.targets_copy():
            if t.is_separator():
                parts.append(set())
            elif t.is_relative_detector_id():
                parts[-1].add(t.val)
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        for part in parts:
            by_comp[frozenset(part)].append((ins.args_copy()[0], canon(dets, obs), dets))
    wanted, symptoms = {}, set()
    for e in edges:
        key = frozenset([int(g["u"][e])] + ([int(g["v"][e])] if g["v"][e] >= 0 else []))
        whole = sorted({(p, s) for p, s, d in by_comp[key] if d == key}, reverse=True)
        top = whole or sorted({(p, s) for p, s, _ in by_comp[key]}, reverse=True)[:2]
        wanted[e] = (top, not whole)
        symptoms |= {s for _, s in top}
    filt = stim.DetectorErrorModel("\n".join(f"error(0.1) {s}" for s in sorted(symptoms)))
    found: dict = defaultdict(lambda: defaultdict(float))
    for ex in model.explain_detector_error_model_errors(
        dem_filter=filt, reduce_to_one_representative_error=False
    ):
        terms = [t.dem_target for t in ex.dem_error_terms]
        s = canon(
            {t.val for t in terms if t.is_relative_detector_id()},
            {t.val for t in terms if t.is_logical_observable_id()},
        )
        for loc in ex.circuit_error_locations:
            it = loc.instruction_targets
            if it.gate in ("M", "X_ERROR"):
                qubits = tuple(t.gate_target.qubit_value for t in it.targets_in_range)
            else:
                qubits = tuple(
                    sorted({t.gate_target.qubit_value for t in loc.flipped_pauli_product})
                )
            found[s][(KIND.get(it.gate, it.gate), qubits)] += location_p(loc)
    cal = job.calibration
    out = {}
    for e, (top, pieces) in wanted.items():
        total = defaultdict(float)
        for s in {s for _, s in top}:
            for k, v in found.get(s, {}).items():
                total[k] += v
        norm = sum(total.values()) or 1.0
        rows = []
        for (kind, qubits), v in sorted(total.items(), key=lambda kv: -kv[1])[:3]:
            rows.append(
                [
                    kind,
                    list(qubits),
                    [role(job, q) for q in qubits],
                    round(v / norm, 3),
                    [round(cal.t1_us[q], 1) for q in qubits],
                    [round(cal.t2_us[q], 1) for q in qubits],
                    [round(cal.readout[q], 4) for q in qubits],
                ]
            )
        out[e] = (pieces, rows)
    return out


ORIGIN = ("kind", "chip qubits", "roles", "share", "T1 (us)", "T2 (us)", "readout error")
HIGH = ("class", "p (min, max over rounds)", "rounds", "device p", "device SE", "pieces", "faults")


# ------------------------------------------------------------------------------- per job
def job_pij(k: int) -> dict:
    t0 = time.time()
    job = jobs()[k]
    out: dict = {"day": job.day, "rep": job.rep, "patches": job.analysis["patches"], "types": {}}
    nonedge = defaultdict(lambda: [0.0, 0.0, 0])  # (patch, basis, same, lag) -> sums
    high: dict = {}
    matrix: dict = {}
    for basis in "XZ":
        inst = defaultdict(list)  # (patch, key) -> rows
        for n in USED:
            i = memory_index(basis, n)
            model, records, labels = job.model(i)
            labels = DT.numbered(labels)
            dets, _ = job.chip_detectors(i, model, records)
            sim = model.compile_detector_sampler(seed=sim_seed(i)).sample(SIM_SHOTS)[:LEARN_SHOTS]
            g = DT.graph(calibrated_matching(model))
            keys, cls, sub = DT.types(g, labels, n)
            pairs = DT.model_pairs(model)
            me = DT.model_estimate(pairs, g)
            w = DT.multinomial_weights(dets.shape[0], B, seed=1000 * k + i)
            dev = DT.learned(dets, g, labels, me)
            boot = DT.learned(dets, g, labels, me, w)
            ctl = DT.learned(sim, g, labels, me)
            for e, key in enumerate(keys):
                inst[(key[0], DT.key_str(key[1:]))].append(
                    (cls[e], sub[e], g["p"][e], me[e], dev[e], boot[:, e], ctl[e])
                )
            # pairs the model never flips together, in bulk rounds, where the estimator's
            # denominator is above 0.1 on both the device's and the model's shots (never with
            # a saturated detector): device and control p_ij
            (p_dev, den_dev), (p_ctl, den_ctl) = DT.all_pairs(dets), DT.all_pairs(sim)
            sat = np.array([DT.saturated(lab) for lab in labels])
            patch = np.array([lab[0] for lab in labels])
            rnd = np.array([lab[1] for lab in labels])
            name = np.array([lab[2] for lab in labels])
            iu, ju = np.triu_indices(len(labels), 1)
            keep = (
                (patch[iu] == patch[ju])
                & (pairs[iu, ju] == 0)
                & ~sat[iu]
                & ~sat[ju]
                & (den_dev[iu, ju] > 0.1)
                & (den_ctl[iu, ju] > 0.1)
                & ~np.isin(rnd[iu], (0, n))
                & ~np.isin(rnd[ju], (0, n))
            )
            for a, b in zip(iu[keep], ju[keep]):
                s = nonedge[(patch[a], basis, name[a] == name[b], min(abs(rnd[a] - rnd[b]), 3))]
                s[0] += p_dev[a, b]
                s[1] += p_ctl[a, b]
                s[2] += 1
            if n == 8:
                big = [e for e in range(len(keys)) if g["p"][e] > HIGH_P]
                seen = {}
                for e in big:
                    tk = (keys[e][0], DT.key_str(keys[e][1:]))
                    if tk not in seen or keys[e][-1] == "bulk":
                        seen[tk] = e
                why = origins(job, model, g, sorted(set(seen.values())))
                for tk, e in seen.items():
                    ps = [g["p"][f] for f in big if (keys[f][0], DT.key_str(keys[f][1:])) == tk]
                    high[f"{basis} {tk[0]} {tk[1]}"] = [
                        cls[e],
                        [sig(min(ps)), sig(max(ps))],
                        len(ps),
                        None,  # device p and SE, filled in below from the type table
                        None,
                        *why[e],
                    ]
                if job.rep == 1:  # placement 1's distance-3 patch: the same circuit every day
                    ix = [a for a, lab in enumerate(labels) if lab[0] == "3"]
                    matrix[basis] = {
                        "labels": [list(labels[a]) for a in ix],
                        "device": p_dev[np.ix_(ix, ix)],
                        "model": pairs[np.ix_(ix, ix)] * (1 - np.eye(len(ix))),
                    }
        for (patch, key), rows in inst.items():
            dev = np.array([r[4] for r in rows])
            bt = np.array([r[5] for r in rows])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bmean = np.nanmean(bt, axis=0) if np.isfinite(bt).any() else np.full(B, np.nan)
            ok = np.isfinite(dev).all() and np.isfinite(bmean).mean() > 0.9
            out["types"][f"{basis} {patch} {key}"] = [
                rows[0][0],
                rows[0][1],
                len(rows),
                sig(np.mean([r[2] for r in rows])),
                sig(np.mean([r[3] for r in rows])),
                sig(dev.mean()) if ok else None,
                sig(np.nanstd(bmean, ddof=1)) if ok else None,
                sig(np.mean([r[6] for r in rows])) if ok else None,
            ]
            out.setdefault("boot", {})[f"{basis} {patch} {key}"] = bmean if ok else None
    for name, h in high.items():
        row = out["types"].get(name)
        if row is not None:
            h[3], h[4] = row[5], row[6]
    out["high_p"] = high
    out["nonedge"] = {
        f"{p} {b} {'same' if s else 'other'} {lag}": v for (p, b, s, lag), v in nonedge.items()
    }
    out["matrix"] = matrix
    out["seconds"] = time.time() - t0
    print(f"{job.name}: {out['seconds']:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------------- summary
FIELDS = (
    "class",
    "sub-class",
    "instances",
    "graph_p",
    "model_p",
    "device_p",
    "device_se",
    "control_p",
)


BINS = (0.05, 0.1)  # time-like and flag/relay boundary edges, by the model's p


def p_bin(p: float) -> str:
    if p < BINS[0]:
        return f"model p < {BINS[0]}"
    return f"model p {BINS[0]}-{BINS[1]}" if p < BINS[1] else f"model p >= {BINS[1]}"


def summarize(per_job: dict, select) -> dict:
    """Median device/model by class over the (job, type) rows that select() keeps."""
    rows = defaultdict(list)
    for _name, j in per_job.items():
        for tk, t in j["types"].items():
            basis, patch, _key = tk.split(" ", 2)
            if not select(j, basis, patch):
                continue
            names = {t[0], t[1]}
            if t[1] in ("time-like", "flag/relay boundary"):
                names.add(f"{t[1]}, {p_bin(t[4])}")
            for c in names:
                rows[c].append((t, j["boot"].get(tk)))
    out = {}
    for c, rs in rows.items():
        est = [(t, b) for t, b in rs if t[5] is not None]
        r = np.array([t[5] / t[4] for t, _ in est]) if est else np.array([])
        ctl = np.array([t[7] / t[4] for t, _ in est if t[7] is not None]) if est else np.array([])
        reps = (
            np.median(np.array([b / t[4] for t, b in est]), axis=0) if est else np.array([np.nan])
        )
        out[c] = {
            "types": len(rs),
            "estimated": len(est),
            "median_model_p": float(np.median([t[4] for t, _ in rs])),
            "median_device_p": float(np.median([t[5] for t, _ in est])) if est else None,
            "median_ratio": float(np.median(r)) if est else None,
            "median_ratio_se": float(np.nanstd(reps, ddof=1)) if est else None,
            "ratio_quartiles": [float(np.percentile(r, 25)), float(np.percentile(r, 75))]
            if est
            else None,
            "share_above_1": float((r > 1).mean()) if est else None,
            "control_median_ratio": float(np.nanmedian(ctl)) if est else None,
        }
    return out


def matrix(per_job: dict) -> dict:
    """The p_ij matrices of placement 1's distance-3 patch, averaged over the five days."""
    out = {}
    for basis in "XZ":
        ms = [j["matrix"][basis] for j in per_job.values() if basis in j["matrix"]]
        assert all(m["labels"] == ms[0]["labels"] for m in ms)
        out[basis] = {
            "days": len(ms),
            "labels": ms[0]["labels"],
            **{
                src: np.round(np.nanmean([m[src] for m in ms], axis=0), 5).tolist()
                for src in ("device", "model")
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    names = [j.name for j in jobs()]
    with Pool(args.workers) as pool:
        per_job = dict(zip(names, pool.map(job_pij, range(len(names)))))
    classes = {}
    for basis in "XZ":
        for d in ("3", "5", "both"):
            classes[f"{basis} d{d}"] = summarize(
                per_job, lambda j, b, p, basis=basis, d=d: b == basis and (d == "both" or p == d)
            )
            for pl in (1, 2):
                classes[f"{basis} d{d} placement {pl}"] = summarize(
                    per_job,
                    lambda j, b, p, basis=basis, d=d, pl=pl: (
                        b == basis and (d == "both" or p == d) and j["rep"] == pl
                    ),
                )
    per_key: dict = defaultdict(list)  # each job's (and patch's) mean over its pairs
    for j in per_job.values():
        for k, (dev, ctl, count) in j["nonedge"].items():
            if count:
                per_key[k].append((dev / count, ctl / count, count))
                per_key[f"both {k.split(' ', 1)[1]}"].append((dev / count, ctl / count, count))
    unmodelled = {}
    for k, v in sorted(per_key.items()):
        a = np.array(v)
        se = a[:, :2].std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else [np.nan, np.nan]
        unmodelled[k] = {
            "patches_of_jobs": len(a),
            "pairs": int(a[:, 2].sum()),
            "device_mean_p": [float(a[:, 0].mean()), float(se[0])],
            "control_mean_p": [float(a[:, 1].mean()), float(se[1])],
        }
    high = {n: j["high_p"] for n, j in per_job.items()}
    by_kind: dict = defaultdict(int)
    by_qubit: dict = {}
    for n, h in high.items():
        for v in h.values():
            if not v[6]:
                continue
            kind, qubits, roles, _share, _t1, t2, ro = v[6][0]
            by_kind[f"{v[0]}: {kind}"] += 1
            q = by_qubit.setdefault(
                f"{kind} q{'-q'.join(map(str, qubits))}",
                {
                    "jobs": set(),
                    "types": 0,
                    "p_max": 0.0,
                    "roles": set(),
                    "T2_us": [],
                    "readout": [],
                },
            )
            q["jobs"].add(n)
            q["types"] += 1
            q["p_max"] = max(q["p_max"], v[1][1])
            q["roles"].update(roles)
            q["T2_us"] += t2
            q["readout"] += ro
    leading = {
        k: {
            "jobs": len(q["jobs"]),
            "edge_types": q["types"],
            "p_max": q["p_max"],
            "roles": sorted(q["roles"]),
            "T2_us": [min(q["T2_us"]), max(q["T2_us"])],
            "readout": [min(q["readout"]), max(q["readout"])],
        }
        for k, q in sorted(by_qubit.items(), key=lambda kv: -kv[1]["types"])
    }
    write_json(
        "pij.json",
        {
            "method": __doc__,
            "fields": FIELDS,
            "classes": classes,
            "unmodelled_pairs [patch basis same-detector lag(3 = 3+)], mean and SE over jobs": unmodelled,
            "high_p_fields": {"edge": HIGH, "fault": ORIGIN},
            "high_p_edges [job][basis patch type]": high,
            "high_p_types": sum(len(h) for h in high.values()),
            "high_p_types_by_class_and_leading_fault": dict(by_kind),
            "high_p_leading_faults_by_qubit": leading,
            "matrix [basis], placement 1, d = 3, 8 rounds, mean over the five days": matrix(
                per_job
            ),
            "per_job": {
                n: {k: j[k] for k in ("day", "rep", "patches", "types", "seconds")}
                for n, j in per_job.items()
            },
        },
    )
    for k, v in classes.items():
        if "placement" in k:
            continue
        print(k)
        for c, s in v.items():
            if s["median_ratio"] is not None:
                print(
                    f"  {c:22s} {s['estimated']:4d}/{s['types']:4d}  model {s['median_model_p']:.4f}"
                    f"  device {s['median_device_p']:.4f}  ratio {s['median_ratio']:.2f}"
                    f" +- {s['median_ratio_se']:.2f}  >1 {s['share_above_1']:.2f}"
                    f"  control {s['control_median_ratio']:.2f}"
                )
    for k, v in unmodelled.items():
        print(f"  unmodelled {k}: {v}")


if __name__ == "__main__":
    main()

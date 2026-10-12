"""Every single fault, and every pair at d = 5, through the decoder; the circuits' distances.

For every job (15), basis (X, Z) and memory length (1, 2, 3, 4, 6, 8 rounds):

* Each error mechanism of the decomposed detector error model that the decoder uses
  (one line of stim's DEM = one class of single circuit faults with the same
  symptom) is applied alone: its detectors are decoded and the prediction compared
  with the observable it actually flips. Done for the d=3 and d=5 patches, with the
  calibrated weights (run._decode's graph) and with uniform weights (same graph,
  all edges equal).
* For the d=5 patch every unordered pair of distinct mechanisms is applied too.
  Pairs that are themselves undetectable logical errors (empty syndrome, logical
  flipped) would be a property of the circuit, not of the decoder: counted apart.
* Exact low-weight logical search: every set of at most two mechanisms is hashed by
  its syndrome; two sets with the same syndrome and different logical parity give a
  logical of weight <= 4, and the smallest such sum is the exact minimum weight of an
  undetectable logical error if it is <= 4 (">4" otherwise). Counts hyperedge
  mechanisms (e.g. a fault hitting a flag, a relay and two stabilizers) as one fault.
* stim's Circuit.shortest_graphlike_error (as the repo's test does; hyperedges
  ignored) and Circuit.search_for_undetectable_logical_errors (heuristic, limits
  below) per patch.

Heavy (14.3 million pairs and stim's searches); decoder.py reads the committed
results/decoder_faults.json.

    python docs/analysis/decoder_faults.py [job-substring ...]
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np

from heavyhex.experiment.circuits import ROUNDS

from common import RESULTS, jobs, memory_index
from matching import (
    calibrated_matching,
    dense,
    mechanisms,
    patch_of_observable,
    single_observable_circuit,
    uniform_matching,
)

SEARCH = {  # stim's heuristic search limits
    "dont_explore_detection_event_sets_with_size_above": 6,
    "dont_explore_edges_with_degree_above": 6,
    "dont_explore_edges_increasing_symptom_degree": False,
}
PAIR_CHUNK = 200_000
EXAMPLES = 8


def decode_cols(matching, syn, col):
    return matching.decode_batch(syn)[:, col].astype(np.uint8)


def exact_min_logical(S: np.ndarray, O: np.ndarray) -> int | None:
    """Smallest number of mechanisms forming an undetectable logical, if <= 4."""
    n = len(S)
    packed = np.packbits(S, axis=1)
    best: dict[bytes, list] = defaultdict(
        lambda: [99, 99]
    )  # key -> [min size with obs 0, with obs 1]
    best[np.packbits(np.zeros(S.shape[1], np.uint8)).tobytes()][0] = 0
    for i in range(n):
        k = packed[i].tobytes()
        b = best[k]
        b[O[i]] = min(b[O[i]], 1)
    for a in range(n - 1):
        x = packed[a + 1 :] ^ packed[a]
        o = O[a + 1 :] ^ O[a]
        for row, ob in zip(x, o):
            b = best[row.tobytes()]
            if b[ob] > 2:
                b[ob] = 2
    m = min((b[0] + b[1] for b in best.values() if b[0] < 99 and b[1] < 99), default=None)
    return m


def scan(job, basis: str, rounds: int, pairs: bool, search: bool) -> dict:
    model, _, labels = job.model(memory_index(basis, rounds))
    cal = calibrated_matching(model)
    uni = uniform_matching(model, cal)
    probs, dets, obs, comps = mechanisms(model)
    det_patch = np.array([int(str(p)) for p, _, _ in labels])
    obs_patch = {k: int(p) for k, p in patch_of_observable(job, labels).items()}
    patch_obs = {p: k for k, p in obs_patch.items()}
    mech_patch = []
    for ds, o in zip(dets, obs):
        ps = {int(det_patch[d]) for d in ds} | {
            obs_patch[k] for k in range(len(obs_patch)) if o >> k & 1
        }
        assert len(ps) <= 1, (ds, o)
        mech_patch.append(ps.pop() if ps else 0)
    mech_patch = np.array(mech_patch)
    S_all = dense(dets, model.num_detectors)
    out: dict = {
        "num_detectors": model.num_detectors,
        "num_mechanisms": len(dets),
        "num_graph_edges": cal.num_edges,
        "patches": {},
    }
    for patch, k in sorted(patch_obs.items()):
        sel = np.flatnonzero(mech_patch == patch)
        S, O = S_all[sel], ((obs[sel] >> k) & 1).astype(np.uint8)
        hyper = np.array([c.count("^") > 0 for c in np.array(comps, dtype=object)[sel]])
        row: dict = {
            "mechanisms": len(sel),
            "hyperedge_mechanisms": int(hyper.sum()),
            "undetectable_single": int(((S.sum(1) == 0) & (O == 1)).sum()),
        }
        for name, m in (("calibrated", cal), ("uniform", uni)):
            pred = decode_cols(m, S, k)
            bad = np.flatnonzero(pred != O)
            row[f"single_miscorrections_{name}"] = len(bad)
            row[f"single_miscorrections_{name}_hyperedge"] = int(hyper[bad].sum())
            row[f"single_examples_{name}"] = [
                {"dem": comps[sel[i]], "p": float(probs[sel[i]]), "flips": int(O[i])}
                for i in bad[:EXAMPLES]
            ]
        if pairs and patch == 5:
            n = len(sel)
            a_idx, b_idx = np.triu_indices(n, 1)
            tot = len(a_idx)
            counts = {
                "pairs": int(tot),
                "undetectable_logical_pairs": 0,
                "miscorrections_calibrated": 0,
                "miscorrections_uniform": 0,
                "miscorrections_calibrated_with_hyperedge": 0,
                "miscorrections_uniform_with_hyperedge": 0,
                "examples_calibrated": [],
                "examples_uniform": [],
            }
            for s in range(0, tot, PAIR_CHUNK):
                a, b = a_idx[s : s + PAIR_CHUNK], b_idx[s : s + PAIR_CHUNK]
                syn = S[a] ^ S[b]
                act = O[a] ^ O[b]
                empty = syn.sum(1) == 0
                counts["undetectable_logical_pairs"] += int((empty & (act == 1)).sum())
                for name, m in (("calibrated", cal), ("uniform", uni)):
                    pred = decode_cols(m, syn, k)
                    bad = np.flatnonzero((pred != act) & ~empty)
                    counts[f"miscorrections_{name}"] += len(bad)
                    counts[f"miscorrections_{name}_with_hyperedge"] += int(
                        (hyper[a[bad]] | hyper[b[bad]]).sum()
                    )
                    ex = counts[f"examples_{name}"]
                    for i in bad[: max(0, EXAMPLES - len(ex))]:
                        ex.append(
                            {
                                "a": comps[sel[a[i]]],
                                "pa": float(probs[sel[a[i]]]),
                                "b": comps[sel[b[i]]],
                                "pb": float(probs[sel[b[i]]]),
                                "flips": int(act[i]),
                            }
                        )
            row["pairs"] = counts
        row["exact_min_logical_weight_le4"] = exact_min_logical(S, O)
        if search:
            alone = single_observable_circuit(model, k)
            row["stim_shortest_graphlike_error"] = len(
                alone.shortest_graphlike_error(canonicalize_circuit_errors=True)
            )
            t = time.time()
            found = alone.search_for_undetectable_logical_errors(
                **SEARCH, canonicalize_circuit_errors=True
            )
            row["stim_search_undetectable"] = len(found)
            row["stim_search_seconds"] = round(time.time() - t, 2)
        out["patches"][str(patch)] = row
    return out


def main() -> None:
    want = sys.argv[1:]
    results = {"search_limits": SEARCH, "jobs": {}}
    for job in jobs():
        if want and not any(w in job.name for w in want):
            continue
        results["jobs"][job.name] = {}
        for basis in "XZ":
            for rounds in ROUNDS:
                t = time.time()
                r = scan(job, basis, rounds, pairs=True, search=True)
                results["jobs"][job.name][f"{basis}{rounds}"] = r
                msg = []
                for p, row in r["patches"].items():
                    s = (
                        f"d{p}: {row['mechanisms']} mech, single miscorr cal {row['single_miscorrections_calibrated']}"
                        f" uni {row['single_miscorrections_uniform']}"
                    )
                    if "pairs" in row:
                        c = row["pairs"]
                        s += (
                            f"; pairs {c['pairs']}: cal {c['miscorrections_calibrated']} uni {c['miscorrections_uniform']}"
                            f" undetectable {c['undetectable_logical_pairs']}"
                        )
                    s += (
                        f"; exact min logical {row['exact_min_logical_weight_le4']}, graphlike "
                        f"{row['stim_shortest_graphlike_error']}, search {row['stim_search_undetectable']}"
                    )
                    msg.append(s)
                print(
                    f"{job.name} {basis}{rounds} ({time.time() - t:.0f}s) | " + " | ".join(msg),
                    flush=True,
                )
        name = "decoder_faults.json" if not want else f"decoder_faults_{'_'.join(want)}.json"
        (RESULTS / name).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

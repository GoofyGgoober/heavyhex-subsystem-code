"""How many of the uniform-weight miscorrections are ties (App. decoder).

The fault patterns of decoder_faults.py, decoded on the same matching graph with every
weight 1 and with weights 1 + 1e-3 U(-1, 1) (twelve jitter seeds). A pattern that some
jittered decoder corrects is an exact tie under uniform weights: the wrong correction weighs
as much as the right one, and the tie-break picks it. Patterns: Z-memory single faults at
d = 3 (all 60 circuits), Z-memory fault pairs at d = 5 in the one- and two-round circuits,
and X-memory single faults at d = 3. A few minutes.

    python docs/analysis/decoder_ties.py     # writes results/decoder_ties.json
"""

from __future__ import annotations

import json

import numpy as np
import pymatching

from heavyhex.experiment.circuits import ROUNDS

from common import RESULTS, jobs, memory_index
from matching import calibrated_matching, mechanisms

NSEED = 12
CASES = (  # basis, patch, rounds, pairs
    ("Z", "3", tuple(ROUNDS), False),
    ("Z", "5", (1, 2), True),
    ("X", "3", tuple(ROUNDS), False),
)


def reweighted(cal: pymatching.Matching, weight) -> pymatching.Matching:
    """The calibrated graph (nodes, edges, boundary edges, observables) with new weights."""
    m = pymatching.Matching()
    for u, v, a in cal.edges():
        if v is None:
            m.add_boundary_edge(u, fault_ids=set(a["fault_ids"]), weight=weight())
        else:
            m.add_edge(u, v, fault_ids=set(a["fault_ids"]), weight=weight())
    assert m.num_edges == cal.num_edges
    return m


def jittered(cal: pymatching.Matching, seed: int, eps: float = 1e-3) -> pymatching.Matching:
    rng = np.random.default_rng(seed)
    return reweighted(cal, lambda: 1.0 + eps * rng.uniform(-1, 1))


def main() -> None:
    out = {}
    for basis, patch, rounds_list, pairs in CASES:
        tot = {
            "patterns": 0,
            "mis_uniform": 0,
            "mis_any_jitter": 0,
            "mis_all_jitter": 0,
            "verdict_changes": 0,
            "mis_jitter_seed": [0] * NSEED,
        }
        for job in jobs():
            patches = job.analysis["patches"]
            if patch not in patches:
                continue
            k = patches.index(patch)
            for n in rounds_list:
                model, _, labels = job.model(memory_index(basis, n))
                dp = np.array([str(p) for p, _, _ in labels])
                cal = calibrated_matching(model)
                mats = [reweighted(cal, lambda: 1.0)] + [
                    jittered(cal, s) for s in range(1, NSEED + 1)
                ]
                _, dets, obs, _ = mechanisms(model)
                mech = [i for i, ds in enumerate(dets) if ds and all(dp[y] == patch for y in ds)]
                M = np.zeros((len(mech), model.num_detectors), np.uint8)
                for row, i in enumerate(mech):
                    M[row, dets[i]] = 1
                O = np.array([(obs[i] >> k) & 1 for i in mech], np.uint8)
                if pairs:
                    S = np.concatenate([M[i] ^ M[i + 1 :] for i in range(len(mech) - 1)])
                    Ob = np.concatenate([O[i] ^ O[i + 1 :] for i in range(len(mech) - 1)])
                else:
                    S, Ob = M, O
                res = [mt.decode_batch(S)[:, k] != Ob for mt in mats]
                J = np.stack(res[1:])
                tot["patterns"] += len(Ob)
                tot["mis_uniform"] += int(res[0].sum())
                tot["mis_any_jitter"] += int(J.any(0).sum())
                tot["mis_all_jitter"] += int(J.all(0).sum())
                tot["verdict_changes"] += int(
                    (J.any(0) != J.all(0)).sum() + ((J.all(0) == J.any(0)) & (J[0] != res[0])).sum()
                )
                for s in range(NSEED):
                    tot["mis_jitter_seed"][s] += int(J[s].sum())
        key = f"{basis} d{patch} {'pairs' if pairs else 'singles'} rounds {rounds_list}"
        out[key] = tot
        print(key, tot, flush=True)
    (RESULTS / "decoder_ties.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

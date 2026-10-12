"""How much of the uniform-weight result is tie-breaking? Decode the same fault patterns with
weights 1 and with weights 1 + 1e-3 * U(-1, 1) (NSEED jitter seeds), same graph. A pattern whose
verdict changes with the jitter is an exact tie under uniform weights.
Covers: Z d3 singles (all 60 circuits), Z d5 / X d5 pairs (2-round circuits, 10 jobs)."""
import json

import numpy as np
import pymatching

from vcommon import HERE, MEM, cal_matching, folders, meta, model, uni_matching
from v_faults import lines


def jittered(cal, seed, eps=1e-3):
    rng = np.random.default_rng(seed)
    m = pymatching.Matching()
    for u, v, a in cal.edges():
        w = 1.0 + eps * rng.uniform(-1, 1)
        if v is None:
            m.add_boundary_edge(u, fault_ids=set(a["fault_ids"]), weight=w)
        else:
            m.add_edge(u, v, fault_ids=set(a["fault_ids"]), weight=w)
    return m


NSEED = 12
out = {}
for basis, patch, rounds_list, pairs in (("Z", "3", (1, 2, 3, 4, 6, 8), False), ("Z", "5", (1, 2), True), ("X", "3", (1, 2, 3, 4, 6, 8), False)):
    tot = {"patterns": 0, "mis_uniform": 0, "mis_any_jitter": 0, "mis_all_jitter": 0, "verdict_changes": 0,
           "mis_jitter_seed": [0] * NSEED}
    for f in folders():
        me = meta(f)
        if patch not in me["patches"]:
            continue
        k = me["patches"].index(patch)
        for n in rounds_list:
            idx = MEM[(basis, n)]
            m = model(f, idx)
            labels = me["settings"][str(idx)]["labels"]
            dp = np.array([str(p) for p, _, _ in labels])
            cal = cal_matching(m)
            mats = [uni_matching(cal)] + [jittered(cal, s) for s in range(1, NSEED + 1)]
            mech = [x for x in lines(m.detector_error_model(decompose_errors=True))
                    if x[1] and all(dp[y] == patch for y in x[1])]
            M = np.zeros((len(mech), m.num_detectors), np.uint8)
            for i, x in enumerate(mech):
                M[i, list(x[1])] = 1
            O = np.array([(x[2] >> k) & 1 for x in mech], np.uint8)
            if pairs:
                S = np.concatenate([M[i] ^ M[i + 1:] for i in range(len(mech) - 1)])
                Ob = np.concatenate([O[i] ^ O[i + 1:] for i in range(len(mech) - 1)])
            else:
                S, Ob = M, O
            res = [mt.decode_batch(S)[:, k] != Ob for mt in mats]
            J = np.stack(res[1:])
            tot["patterns"] += len(Ob)
            tot["mis_uniform"] += int(res[0].sum())
            tot["mis_any_jitter"] += int(J.any(0).sum())
            tot["mis_all_jitter"] += int(J.all(0).sum())
            tot["verdict_changes"] += int((J.any(0) != J.all(0)).sum() + ((J.all(0) == J.any(0)) & (J[0] != res[0])).sum())
            for s in range(NSEED):
                tot["mis_jitter_seed"][s] += int(J[s].sum())
    key = f"{basis} d{patch} {'pairs' if pairs else 'singles'} rounds {rounds_list}"
    out[key] = tot
    print(key, tot, flush=True)
(HERE / "v_ties12.json").write_text(json.dumps(out, indent=1))

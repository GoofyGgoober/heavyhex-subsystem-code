"""Why calibration weights miscorrect low-weight faults: the weighted distance of the matching graph.

For each job, basis, rounds and patch, on run._decode's matching graph:
  W     = the smallest total weight (sum of ln((1-p)/p)) of a logical error in the graph:
          a path from the boundary back to the boundary, or a cycle, with odd parity
          of the patch's observable (Dijkstra on (node, parity));
  hops  = the same with every edge weight 1 (the graphlike distance in edges);
  heavy = edges whose weight exceeds W/2. A single graph edge e can be miscorrected
          only if some opposite-parity path between its ends is lighter than e, and e
          plus that path is a logical, so w(e) > W/2 is necessary. With calibrated
          weights, many single-fault edges on this chip are that heavy.
Also the heaviest and lightest edges, and the boundary edges' probabilities.

    python docs/analysis/decoder_weights.py      # writes results/decoder_weights.json
"""

from __future__ import annotations

import heapq
import json

import numpy as np

from heavyhex.experiment.circuits import ROUNDS

from common import RESULTS, jobs, memory_index
from matching import calibrated_matching, patch_of_observable


def lightest_logical(edges, n, k, unit=False):
    """Dijkstra over (node, parity); boundary is node n."""
    adj = [[] for _ in range(n + 1)]
    for u, v, a in edges:
        v = n if v is None else v
        w = 1.0 if unit else a["weight"]
        par = 1 if k in a["fault_ids"] else 0
        adj[u].append((v, w, par))
        adj[v].append((u, w, par))
    best = np.inf
    # a logical is an odd-parity closed walk; start from every node (cheap: graphs are small)
    for s in range(n + 1):
        dist = {(s, 0): 0.0}
        pq = [(0.0, s, 0)]
        while pq:
            d, x, p = heapq.heappop(pq)
            if d > dist.get((x, p), np.inf) or d >= best:
                continue
            if x == s and p == 1:
                best = min(best, d)
                break
            for y, w, q in adj[x]:
                nd, np_ = d + w, p ^ q
                if nd < dist.get((y, np_), np.inf):
                    dist[(y, np_)] = nd
                    heapq.heappush(pq, (nd, y, np_))
    return best


def main() -> None:
    out = {}
    for job in jobs():
        out[job.name] = {}
        for basis in "XZ":
            for rounds in ROUNDS:
                model, _, labels = job.model(memory_index(basis, rounds))
                m = calibrated_matching(model)
                edges = m.edges()
                det_patch = np.array([int(str(p)) for p, _, _ in labels])
                row = {}
                for k, patch in patch_of_observable(job, labels).items():
                    mine = [e for e in edges if det_patch[e[0]] == int(patch)]
                    W = lightest_logical(mine, model.num_detectors, k)
                    hops = lightest_logical(mine, model.num_detectors, k, unit=True)
                    w = np.array([e[2]["weight"] for e in mine])
                    p = np.array([e[2]["error_probability"] for e in mine])
                    bnd = np.array([e[2]["error_probability"] for e in mine if e[1] is None])
                    row[patch] = {
                        "weighted_distance_W": W,
                        "graphlike_hops": hops,
                        "edges": len(mine),
                        "edges_heavier_than_half_W": int((w > W / 2).sum()),
                        "max_edge_weight": float(w.max()),
                        "min_edge_weight": float(w.min()),
                        "median_edge_p": float(np.median(p)),
                        "max_edge_p": float(p.max()),
                        "median_boundary_p": float(np.median(bnd)),
                        "max_boundary_p": float(bnd.max()),
                    }
                out[job.name][f"{basis}{rounds}"] = row
                print(
                    job.name,
                    f"{basis}{rounds}",
                    " | ".join(
                        f"d{p}: W={r['weighted_distance_W']:.2f} hops={r['graphlike_hops']:.0f} "
                        f"heavy(>W/2)={r['edges_heavier_than_half_W']}/{r['edges']} max w={r['max_edge_weight']:.2f} "
                        f"median p={r['median_edge_p']:.4f} max p={r['max_edge_p']:.3f} median bdy p={r['median_boundary_p']:.3f}"
                        for p, r in row.items()
                    ),
                    flush=True,
                )
    (RESULTS / "decoder_weights.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

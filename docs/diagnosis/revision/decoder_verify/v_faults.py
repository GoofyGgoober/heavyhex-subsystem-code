"""(2) Independent exhaustive fault test, distances and decoder-graph properties.

Per job folder (15) and memory setting (12), from the cached repo model:
  * mechanisms = lines of the decomposed DEM the decoder is built from (symptom = XOR
    of the components, observable = XOR); also the undecomposed DEM for exact distance.
  * every single mechanism decoded with the calibrated graph and the uniform graph;
    for the d=5 patch, every unordered pair of its mechanisms.
  * exact undetectable-logical search up to 4 mechanisms (undecomposed DEM, meet in the
    middle on pair syndromes), stim shortest_graphlike_error (circuit, and decoder DEM),
    stim search_for_undetectable_logical_errors (2-round circuits by default, --all for all).
  * decoder-graph lightest logical (weighted and in hops) by Dijkstra on the
    parity-doubled graph; edge probability ranges.
Run per folder in a process pool; results to v_faults.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import stim

from vcommon import HERE, MEM, ROUNDS, cal_matching, folders, meta, model, uni_matching


def lines(dem: stim.DetectorErrorModel):
    out = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        d, o, ncomp, hyper = set(), 0, 1, False
        comp = set()
        for t in ins.targets_copy():
            if t.is_separator():
                ncomp += 1
                hyper |= len(comp) > 2
                comp = set()
            elif t.is_relative_detector_id():
                d ^= {t.val}
                comp ^= {t.val}
            elif t.is_logical_observable_id():
                o ^= 1 << t.val
        hyper |= len(comp) > 2
        out.append((ins.args_copy()[0], tuple(sorted(d)), o, ncomp))
    return out


def strip_observables(m: stim.Circuit, keep: int) -> stim.Circuit:
    out = stim.Circuit()
    for ins in m.flattened():
        if ins.name == "OBSERVABLE_INCLUDE" and int(ins.gate_args_copy()[0]) != keep:
            continue
        out.append(ins)
    return out


def dem_keep_observable(dem: stim.DetectorErrorModel, keep: int) -> stim.DetectorErrorModel:
    s = []
    for ins in dem.flattened():
        if ins.type != "error":
            s.append(str(ins))
            continue
        toks = []
        for t in ins.targets_copy():
            if t.is_separator():
                toks.append("^")
            elif t.is_relative_detector_id():
                toks.append(f"D{t.val}")
            elif t.is_logical_observable_id():
                if t.val == keep:
                    toks.append(f"L{t.val}")
        # clean separators left dangling by removed tokens
        while toks and toks[-1] == "^":
            toks.pop()
        while toks and toks[0] == "^":
            toks.pop(0)
        toks2 = []
        for t in toks:
            if t == "^" and toks2 and toks2[-1] == "^":
                continue
            toks2.append(t)
        if not toks2:
            continue
        s.append(f"error({ins.args_copy()[0]}) " + " ".join(toks2))
    return stim.DetectorErrorModel("\n".join(s))


def exact_upto4(mechs, k_obs):
    """Smallest undetectable logical (flip of bit k_obs) among <=4 mechanisms; None if none."""
    syn = [frozenset(d) for _, d, _ in mechs]
    obs = [(o >> k_obs) & 1 for _, _, o in mechs]
    n = len(mechs)
    best = None
    single = defaultdict(set)
    for i in range(n):
        if not syn[i] and obs[i]:
            return 1
        single[syn[i]].add(obs[i])
    for v in single.values():
        if len(v) > 1:
            best = 2
    if best:
        return best
    pair = defaultdict(set)
    for i in range(n):
        si = syn[i]
        for j in range(i + 1, n):
            pair[si ^ syn[j]].add(obs[i] ^ obs[j])
    # size 3: single c with syn(c) == syn(a)^syn(b), parity differs
    for s, v in single.items():
        if s in pair:
            for oc in v:
                if (1 - oc) in pair[s]:
                    best = 3  # may also be overlapping (a==c) -> then size 1 is found; excluded above
    if best:
        return best
    for v in pair.values():
        if len(v) > 1:
            return 4
    return None


def logical_paths(mt, det_mask, k_obs):
    """Min weight and min hops of an odd-parity closed walk (via boundary or not) on the patch's subgraph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    nodes = sorted(set(np.flatnonzero(det_mask).tolist()))
    index = {v: i for i, v in enumerate(nodes)}
    B = len(nodes)
    N = B + 1
    rows, cols, wts = [], [], []
    ps, bps, wl = [], [], []
    for u, v, a in mt.edges():
        if u not in index or (v is not None and v not in index):
            continue
        a_, b_ = index[u], (B if v is None else index[v])
        par = 1 if k_obs in a["fault_ids"] else 0
        w = a["weight"]
        ps.append(a["error_probability"])
        wl.append(w)
        if v is None:
            bps.append(a["error_probability"])
        for x, y in ((a_, b_), (b_, a_)):
            for s in (0, 1):
                rows.append(x + s * N)
                cols.append(y + (s ^ par) * N)
                wts.append(max(w, 1e-12))
    out = {}
    for name, ww in (("W", np.array(wts)), ("hops", np.ones(len(wts)))):
        g = csr_matrix((ww, (rows, cols)), shape=(2 * N, 2 * N))
        dist = dijkstra(g, directed=True, indices=list(range(N)))
        out[name] = float(min(dist[i, i + N] for i in range(N)))
    ws = np.array(wl)
    out["frac_heavier_half_W"] = float(np.mean(ws > out["W"] / 2))
    out["n_edges"] = len(ps)
    out["p_min"] = float(min([x for x in ps if x >= 0], default=-1)); out["n_p_negative"] = sum(1 for x in ps if x < 0); out["w_min"] = float(min(wl))
    out["p_max"] = float(max(ps, default=-1))
    out["median_boundary_p"] = float(np.median(bps)) if bps else None
    out["w_max"] = float(ws.max())
    return out


def scan_folder(args):
    folder, do_search_all = args
    from scipy.stats import binomtest  # noqa: F401  (import check)

    me = meta(folder)
    patches = me["patches"]
    res = []
    for basis in "XZ":
        for n in ROUNDS:
            idx = MEM[(basis, n)]
            t0 = time.time()
            m = model(folder, idx)
            labels = me["settings"][str(idx)]["labels"]
            dpatch = np.array([str(p) for p, _, _ in labels])
            dem = m.detector_error_model(decompose_errors=True)
            cal = cal_matching(m)
            uni = uni_matching(cal)
            mechs_all = lines(dem)
            dem_u = m.detector_error_model(decompose_errors=False)
            mechs_u_all = lines(dem_u)
            nd = m.num_detectors
            for k, patch in enumerate(patches):
                row = {"folder": folder, "rep": me["rep"]["rep"], "day": me["rep"]["day"], "basis": basis,
                       "rounds": n, "patch": patch}

                def in_patch(d):
                    return len(d) > 0 and all(dpatch[x] == patch for x in d)

                mixed = [x for x in mechs_all if x[1] and len({dpatch[y] for y in x[1]}) > 1]
                row["mixed_patch_mechanisms"] = len(mixed)
                mech = [x for x in mechs_all if in_patch(x[1])]
                undet = [x for x in mechs_all if not x[1] and (x[2] >> k) & 1]
                row["undetectable_single"] = len(undet)
                row["mechanisms"] = len(mech)
                row["hyper_mechanisms"] = sum(1 for x in mech if x[3] > 1)
                row["sum_p"] = float(sum(x[0] for x in mech))
                M = np.zeros((len(mech), nd), dtype=np.uint8)
                for i, (_, d, _, _) in enumerate(mech):
                    M[i, list(d)] = 1
                O = np.array([(x[2] >> k) & 1 for x in mech], dtype=np.uint8)
                P = np.array([x[0] for x in mech])
                H = np.array([x[3] > 1 for x in mech])
                pc = cal.decode_batch(M)[:, k]
                pu = uni.decode_batch(M)[:, k]
                mc, mu = pc != O, pu != O
                row["single_mis_cal"] = int(mc.sum())
                row["single_mis_uni"] = int(mu.sum())
                row["single_mis_cal_hyper"] = int((mc & H).sum())
                row["single_mis_uni_hyper"] = int((mu & H).sum())
                row["single_mis_cal_p"] = float(P[mc].sum())
                row["single_mis_uni_p"] = float(P[mu].sum())
                row["single_mis_cal_examples"] = [[mech[i][0], list(mech[i][1]), int(mech[i][3])] for i in np.flatnonzero(mc)[:3]]
                if patch == "5":
                    npairs = mis_c = mis_u = und = 0
                    mis_c_p = mis_u_p = tot_p = 0.0
                    for i in range(len(mech) - 1):
                        S = M[i] ^ M[i + 1:]
                        Ob = O[i] ^ O[i + 1:]
                        empty = ~S.any(axis=1)
                        und += int((empty & (Ob == 1)).sum())
                        a = cal.decode_batch(S)[:, k] != Ob
                        b = uni.decode_batch(S)[:, k] != Ob
                        pp = P[i] * P[i + 1:]
                        npairs += len(Ob)
                        mis_c += int(a.sum())
                        mis_u += int(b.sum())
                        mis_c_p += float(pp[a].sum())
                        mis_u_p += float(pp[b].sum())
                        tot_p += float(pp.sum())
                    row.update(pairs=npairs, pair_mis_cal=mis_c, pair_mis_uni=mis_u, pair_undetectable=und,
                               pair_mis_cal_p=mis_c_p, pair_mis_uni_p=mis_u_p, pair_total_p=tot_p)
                # exact distance (undecomposed DEM lines of this patch)
                mech_u = [(p, d, o) for p, d, o, _ in mechs_u_all if in_patch(d) or (not d and (o >> k) & 1)]
                row["exact_le4"] = exact_upto4(mech_u, k)
                # stim graphlike distance on the circuit (real graphlike faults only)
                mk = strip_observables(m, k)
                row["stim_graphlike_circuit"] = len(mk.shortest_graphlike_error())
                # graphlike distance of the decoder's decomposed DEM (includes decomposition pieces)
                row["stim_graphlike_decoder_dem"] = len(dem_keep_observable(dem, k).shortest_graphlike_error())
                if do_search_all or n == 2:
                    t1 = time.time()
                    try:
                        e = mk.search_for_undetectable_logical_errors(
                            dont_explore_detection_event_sets_with_size_above=6,
                            dont_explore_edges_with_degree_above=6,
                            dont_explore_edges_increasing_symptom_degree=False,
                            canonicalize_circuit_errors=True,
                        )
                        row["stim_search"] = len(e)
                    except Exception as ex:  # noqa: BLE001
                        row["stim_search"] = f"error: {ex}"
                    row["stim_search_s"] = time.time() - t1
                det_mask = dpatch == patch
                row["graph"] = logical_paths(cal, det_mask, k)
                row["graph_uniform_hops"] = logical_paths(uni, det_mask, k)["hops"]
                row["seconds"] = time.time() - t0
                res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="stim search on every round count")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    fs = folders() if not a.only else [a.only]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(scan_folder, [(f, a.all) for f in fs]):
            rows += r
            print(r[0]["folder"], "done", flush=True)
    out = HERE / ("v_faults.json" if not a.only else f"v_faults_{a.only}.json")
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

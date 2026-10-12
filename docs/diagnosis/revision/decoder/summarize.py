"""Aggregate the decoder audit's outputs into summary.json (and print it).

Reads fault_scan.json, weights.json, single_mass.json, redecode.json (and
redecode_sim*.json if present) from this folder.

    python docs/diagnosis/revision/decoder/summarize.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out: dict = {}
    fs = load("fault_scan.json")
    if fs:
        agg = defaultdict(lambda: defaultdict(int))
        dist = defaultdict(set)
        for job, rows in fs["jobs"].items():
            for setting, r in rows.items():
                basis, rounds = setting[0], int(setting[1:])
                for patch, row in r["patches"].items():
                    for key in (f"{basis} d{patch}", f"{basis} d{patch} rounds={rounds}"):
                        a = agg[key]
                        a["circuits"] += 1
                        a["mechanisms"] += row["mechanisms"]
                        a["hyperedge_mechanisms"] += row["hyperedge_mechanisms"]
                        a["undetectable_single"] += row["undetectable_single"]
                        a["circuits_with_single_miscorrection_calibrated"] += row["single_miscorrections_calibrated"] > 0
                        a["circuits_with_single_miscorrection_uniform"] += row["single_miscorrections_uniform"] > 0
                        for dec in ("calibrated", "uniform"):
                            a[f"single_miscorrections_{dec}"] += row[f"single_miscorrections_{dec}"]
                            a[f"single_miscorrections_{dec}_hyperedge"] += row[f"single_miscorrections_{dec}_hyperedge"]
                        if "pairs" in row:
                            c = row["pairs"]
                            a["pairs"] += c["pairs"]
                            a["undetectable_logical_pairs"] += c["undetectable_logical_pairs"]
                            for dec in ("calibrated", "uniform"):
                                a[f"pair_miscorrections_{dec}"] += c[f"miscorrections_{dec}"]
                                a[f"pair_miscorrections_{dec}_with_hyperedge"] += c[f"miscorrections_{dec}_with_hyperedge"]
                                a[f"circuits_with_pair_miscorrection_{dec}"] += c[f"miscorrections_{dec}"] > 0
                    dist[f"{basis} d{patch}"].add((row["exact_min_logical_weight_le4"], row["stim_shortest_graphlike_error"],
                                                   row["stim_search_undetectable"]))
        out["fault_scan"] = {k: dict(v) for k, v in sorted(agg.items())}
        out["distances_seen (exact<=4 [None = >4], stim graphlike, stim search)"] = {k: sorted(map(str, v)) for k, v in dist.items()}
    w = load("weights.json")
    if w:
        g = defaultdict(list)
        for job, rows in w.items():
            for setting, r in rows.items():
                for patch, row in r.items():
                    g[f"{setting[0]} d{patch}"].append(row)
        out["matching_graph"] = {
            k: {"decoder_graph_hops_seen": sorted({r["graphlike_hops"] for r in v}),
                "weighted_distance_W_range": [min(r["weighted_distance_W"] for r in v), max(r["weighted_distance_W"] for r in v)],
                "max_edge_weight_range": [min(r["max_edge_weight"] for r in v), max(r["max_edge_weight"] for r in v)],
                "max_edge_p_range": [min(r["max_edge_p"] for r in v), max(r["max_edge_p"] for r in v)],
                "median_boundary_p_range": [min(r["median_boundary_p"] for r in v), max(r["median_boundary_p"] for r in v)],
                "fraction_edges_heavier_than_half_W": float(np.mean([r["edges_heavier_than_half_W"] / r["edges"] for r in v]))}
            for k, v in sorted(g.items())}
    sm = load("single_mass.json")
    if sm:
        g = defaultdict(lambda: defaultdict(list))
        for job, rows in sm.items():
            for setting, r in rows.items():
                for patch, row in r.items():
                    for key, val in row.items():
                        g[f"{setting[0]} d{patch} rounds={setting[1:]}"][key].append(val)
        out["miscorrected_probability_mass (mean over jobs)"] = {
            k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in sorted(g.items(), key=lambda x: (x[0][:4], int(x[0].split("=")[1])))}
    rd = load("redecode.json")
    if rd:
        reps = rd["reps"]
        mean = {}
        for dec in ("calibrated", "uniform"):
            for q in ("X eps3", "X eps5", "Z eps3", "Z eps5"):
                v = np.array([r[dec][q][0] for r in reps])
                mean[f"{dec} {q}"] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
        out["hardware"] = {"pooled": rd["pooled"], "mcnemar": rd["mcnemar_pooled"], "mean_eps_over_reps (mean, se over reps)": mean}
    for p in sorted(HERE.glob("redecode_sim*.json")):
        s = json.loads(p.read_text())
        mean = {}
        for dec in ("calibrated", "uniform"):
            for q in ("X eps3", "X eps5", "Z eps3", "Z eps5", "X lambda", "Z lambda"):
                v = np.array([r[dec][q][0] for r in s["simulated_reps"]])
                mean[f"{dec} {q}"] = float(v.mean())
        out[f"model_simulation {p.stem}"] = {"pooled": s["simulated_pooled"], "mean_over_reps": mean}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

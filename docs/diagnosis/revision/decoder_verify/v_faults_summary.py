"""Aggregate v_faults.json per (basis, patch) and compare with the audited decoder/summary.json."""
import json

import numpy as np

from vcommon import HERE, REPO

rows = json.loads((HERE / "v_faults.json").read_text())
main = json.loads((REPO / "docs/diagnosis/revision/decoder/summary.json").read_text())["fault_scan"]  # ../decoder
out = {}
for basis in "XZ":
    for patch in "35":
        R = [r for r in rows if r["basis"] == basis and r["patch"] == patch]
        key = f"{basis} d{patch}"
        s = {
            "circuits": len(R),
            "mechanisms": sum(r["mechanisms"] for r in R),
            "hyper": sum(r["hyper_mechanisms"] for r in R),
            "mixed_patch": sum(r["mixed_patch_mechanisms"] for r in R),
            "undetectable_single": sum(r["undetectable_single"] for r in R),
            "single_mis_cal": sum(r["single_mis_cal"] for r in R),
            "single_mis_uni": sum(r["single_mis_uni"] for r in R),
            "single_mis_cal_hyper": sum(r["single_mis_cal_hyper"] for r in R),
            "single_mis_uni_hyper": sum(r["single_mis_uni_hyper"] for r in R),
            "circuits_single_mis_cal": sum(r["single_mis_cal"] > 0 for r in R),
            "circuits_single_mis_uni": sum(r["single_mis_uni"] > 0 for r in R),
            "jobs_single_mis_cal": sorted({(r["day"], r["rep"], r["rounds"]) for r in R if r["single_mis_cal"]}),
            "exact_le4": sorted({str(r["exact_le4"]) for r in R}),
            "stim_graphlike_circuit": sorted({r["stim_graphlike_circuit"] for r in R}),
            "stim_graphlike_decoder_dem(stim ignores decomposed lines)": sorted({r["stim_graphlike_decoder_dem"] for r in R}),
            "stim_search": sorted({str(r.get("stim_search")) for r in R}),
            "stim_search_circuits": sum("stim_search" in r for r in R),
            "decoder_graph_hops": sorted({r["graph"]["hops"] for r in R}),
            "uniform_graph_hops": sorted({r["graph_uniform_hops"] for r in R}),
            "W_range": [min(r["graph"]["W"] for r in R), max(r["graph"]["W"] for r in R)],
            "edge_p_min": min(r["graph"]["p_min"] for r in R),
            "edge_p_max_range": [min(r["graph"]["p_max"] for r in R), max(r["graph"]["p_max"] for r in R)],
            "w_max_range": [min(r["graph"]["w_max"] for r in R), max(r["graph"]["w_max"] for r in R)],
            "median_boundary_p_range": [min(r["graph"]["median_boundary_p"] for r in R), max(r["graph"]["median_boundary_p"] for r in R)],
            "frac_heavier_half_W_edge_weighted": float(sum(r["graph"]["frac_heavier_half_W"] * r["graph"]["n_edges"] for r in R) / sum(r["graph"]["n_edges"] for r in R)),
            "sum_p_two_round_mean": float(np.mean([r["sum_p"] for r in R if r["rounds"] == 2])),
        }
        if patch == "5":
            s.update(
                pairs=sum(r["pairs"] for r in R),
                pair_mis_cal=sum(r["pair_mis_cal"] for r in R),
                pair_mis_uni=sum(r["pair_mis_uni"] for r in R),
                pair_undetectable=sum(r["pair_undetectable"] for r in R),
                circuits_pair_mis_cal=sum(r["pair_mis_cal"] > 0 for r in R),
                circuits_pair_mis_uni=sum(r["pair_mis_uni"] > 0 for r in R),
            )
        a = main[key.replace(" d", " d")] if key in main else main[f"{basis} d{patch}"]
        s["main"] = {k: a.get(k) for k in ("circuits", "mechanisms", "single_miscorrections_calibrated",
                                             "single_miscorrections_uniform", "circuits_with_single_miscorrection_calibrated",
                                             "pairs", "pair_miscorrections_calibrated", "pair_miscorrections_uniform",
                                             "undetectable_single", "undetectable_logical_pairs")}
        out[key] = s
(HERE / "v_faults_summary.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in out.items():
    print(k)
    for kk, vv in v.items():
        if kk == "jobs_single_mis_cal":
            vv = f"{len(vv)} circuits: " + ", ".join(sorted({f'{d} r{r}' for d, r, _ in vv}))
        print("   ", kk, vv)

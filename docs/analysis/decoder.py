"""The decoder checks, gathered for the paper (Sec. code and circuits, App. decoder).

Reads the outputs of decoder_faults.py, decoder_weights.py, decoder_mass.py,
decoder_examples.py, decoder_shots.py (with and without --simulate) and decoder_ties.py,
adds three light checks, and writes results/decoder.json:

  pieces    Z memory at d = 5 (day 5, rep 1, two rounds): stim's decomposition of flag and
            relay faults leaves pieces that occur only inside decomposed errors, and how many
            circuit fault locations feed the two boundary edges of the X-memory example
  edge_p    the range of the matching graph's edge probabilities, p = 1 / (1 + e^w)
  dropped   hardware points with P(n) >= 1/2, which the prespecified fit leaves out

and, in "paper", Lambda per placement under both weightings and the per-round probability
mass of the single faults that calibrated matching miscorrects. Under a minute.

    python docs/analysis/decoder.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import exp

import numpy as np

from common import RESULTS, jobs, memory_index, result, write_json

CELLS = ("X d3", "X d5", "Z d3", "Z d5")


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def verdict(lam: float, sigma: float, critical: float = 1.645) -> str:
    """run.verdict (src/heavyhex/experiment/run.py): one-sided 5% each way."""
    if lam - critical * sigma > 1:
        return "d = 5 better"
    if lam + critical * sigma < 1:
        return "d = 5 worse"
    return "undecided"


def collect() -> dict:
    """The aggregates of the step scripts' outputs."""
    out: dict = {}
    fs = load("decoder_faults.json")
    if fs:
        agg = defaultdict(lambda: defaultdict(int))
        dist = defaultdict(set)
        for _job, rows in fs["jobs"].items():
            for setting, r in rows.items():
                basis, rounds = setting[0], int(setting[1:])
                for patch, row in r["patches"].items():
                    for key in (f"{basis} d{patch}", f"{basis} d{patch} rounds={rounds}"):
                        a = agg[key]
                        a["circuits"] += 1
                        a["mechanisms"] += row["mechanisms"]
                        a["hyperedge_mechanisms"] += row["hyperedge_mechanisms"]
                        a["undetectable_single"] += row["undetectable_single"]
                        a["circuits_with_single_miscorrection_calibrated"] += (
                            row["single_miscorrections_calibrated"] > 0
                        )
                        a["circuits_with_single_miscorrection_uniform"] += (
                            row["single_miscorrections_uniform"] > 0
                        )
                        for dec in ("calibrated", "uniform"):
                            a[f"single_miscorrections_{dec}"] += row[f"single_miscorrections_{dec}"]
                            a[f"single_miscorrections_{dec}_hyperedge"] += row[
                                f"single_miscorrections_{dec}_hyperedge"
                            ]
                        if "pairs" in row:
                            c = row["pairs"]
                            a["pairs"] += c["pairs"]
                            a["undetectable_logical_pairs"] += c["undetectable_logical_pairs"]
                            for dec in ("calibrated", "uniform"):
                                a[f"pair_miscorrections_{dec}"] += c[f"miscorrections_{dec}"]
                                a[f"pair_miscorrections_{dec}_with_hyperedge"] += c[
                                    f"miscorrections_{dec}_with_hyperedge"
                                ]
                                a[f"circuits_with_pair_miscorrection_{dec}"] += (
                                    c[f"miscorrections_{dec}"] > 0
                                )
                    dist[f"{basis} d{patch}"].add(
                        (
                            row["exact_min_logical_weight_le4"],
                            row["stim_shortest_graphlike_error"],
                            row["stim_search_undetectable"],
                        )
                    )
        out["fault_scan"] = {k: dict(v) for k, v in sorted(agg.items())}
        out["distances_seen (exact<=4 [None = >4], stim graphlike, stim search)"] = {
            k: sorted(map(str, v)) for k, v in dist.items()
        }
    w = load("decoder_weights.json")
    if w:
        g = defaultdict(list)
        for _job, rows in w.items():
            for setting, r in rows.items():
                for patch, row in r.items():
                    g[f"{setting[0]} d{patch}"].append(row)
        out["matching_graph"] = {
            k: {
                "decoder_graph_hops_seen": sorted({r["graphlike_hops"] for r in v}),
                "weighted_distance_W_range": [
                    min(r["weighted_distance_W"] for r in v),
                    max(r["weighted_distance_W"] for r in v),
                ],
                "max_edge_weight_range": [
                    min(r["max_edge_weight"] for r in v),
                    max(r["max_edge_weight"] for r in v),
                ],
                "max_edge_p_range": [
                    min(r["max_edge_p"] for r in v),
                    max(r["max_edge_p"] for r in v),
                ],
                "median_boundary_p_range": [
                    min(r["median_boundary_p"] for r in v),
                    max(r["median_boundary_p"] for r in v),
                ],
                "fraction_edges_heavier_than_half_W": float(
                    np.mean([r["edges_heavier_than_half_W"] / r["edges"] for r in v])
                ),
            }
            for k, v in sorted(g.items())
        }
    sm = load("decoder_mass.json")
    if sm:
        g = defaultdict(lambda: defaultdict(list))
        for _job, rows in sm.items():
            for setting, r in rows.items():
                for patch, row in r.items():
                    for key, val in row.items():
                        g[f"{setting[0]} d{patch} rounds={setting[1:]}"][key].append(val)
        out["miscorrected_probability_mass (mean over jobs)"] = {
            k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
            for k, v in sorted(g.items(), key=lambda x: (x[0][:4], int(x[0].split("=")[1])))
        }
    rd = load("decoder_shots.json")
    if rd:
        reps = rd["reps"]
        mean = {}
        for dec in ("calibrated", "uniform"):
            for q in ("X eps3", "X eps5", "Z eps3", "Z eps5"):
                v = np.array([r[dec][q][0] for r in reps])
                mean[f"{dec} {q}"] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
        out["hardware"] = {
            "pooled": rd["pooled"],
            "mcnemar": rd["mcnemar_pooled"],
            "mean_eps_over_reps (mean, se over reps)": mean,
        }
    s = load("decoder_shots_sim20000.json")
    if s:
        mean = {}
        for dec in ("calibrated", "uniform"):
            for q in ("X eps3", "X eps5", "Z eps3", "Z eps5", "X lambda", "Z lambda"):
                v = np.array([r[dec][q][0] for r in s["simulated_reps"]])
                mean[f"{dec} {q}"] = float(v.mean())
        out["model_simulation_20000_shots"] = {
            "pooled": s["simulated_pooled"],
            "mean_over_reps": mean,
        }
    return out


def pieces() -> dict:
    """Decomposition pieces that occur only inside decomposed errors, and the fault locations
    behind the X-memory example's two boundary edges (day 5, rep 1, two rounds)."""
    job = next(j for j in jobs() if j.name == "2026-10-08-r1")
    out: dict = {}
    for basis in "XZ":
        model, _, _ = job.model(memory_index(basis, 2))
        dem = model.detector_error_model(decompose_errors=True)
        whole, parts_seen = set(), set()
        for ins in dem.flattened():
            if ins.type != "error":
                continue
            parts: list[list[int]] = [[]]
            for t in ins.targets_copy():
                if t.is_separator():
                    parts.append([])
                elif t.is_relative_detector_id():
                    parts[-1].append(t.val)
            if len(parts) == 1:
                whole.add(tuple(sorted(parts[0])))
            else:
                parts_seen.update(tuple(sorted(p)) for p in parts)
        only = sorted(parts_seen - whole)
        try:
            model.detector_error_model(
                decompose_errors=True, block_decomposition_from_introducing_remnant_edges=True
            )
            blocked = "ok"
        except Exception as ex:  # stim says why it cannot
            blocked = "fails: " + str(ex).splitlines()[0][:200]
        out[basis] = {
            "pieces_not_standalone_detector_sets": [list(x) for x in only],
            "n": len(only),
            "block_remnants": blocked,
        }
    model, _, _ = job.model(memory_index("X", 2))
    dem = model.detector_error_model(decompose_errors=True)
    comp_of: dict = {}
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        parts, full, fo = [[]], set(), 0
        for t in ins.targets_copy():
            if t.is_separator():
                parts.append([])
                continue
            if t.is_relative_detector_id():
                parts[-1].append(t.val)
                full ^= {t.val}
            else:
                fo ^= 1 << t.val
        comp_of.setdefault((tuple(sorted(full)), fo), []).append([tuple(sorted(p)) for p in parts])
    locations, lines = {0: 0, 5: 0}, {0: 0, 5: 0}
    for e in model.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
        d = tuple(
            sorted(
                t.dem_target.val
                for t in e.dem_error_terms
                if t.dem_target.is_relative_detector_id()
            )
        )
        o = sum(
            1 << t.dem_target.val
            for t in e.dem_error_terms
            if t.dem_target.is_logical_observable_id()
        )
        for decomposition in comp_of.get((d, o), []):
            for b in (0, 5):
                if (b,) in decomposition:
                    locations[b] += len(e.circuit_error_locations)
                    lines[b] += 1
            break
    out["boundary_feed_X_example"] = {"locations": locations, "dem_lines": lines}
    return out


def edge_p(weights: dict) -> dict:
    """Smallest and largest edge probability of the matching graph, per basis and distance."""
    lo, hi = defaultdict(list), defaultdict(list)
    for rows in weights.values():
        for setting, r in rows.items():
            for patch, row in r.items():
                key = f"{setting[0]} d{patch}"
                lo[key].append(1 / (1 + exp(row["max_edge_weight"])))
                hi[key].append(row["max_edge_p"])
    out = {k: {"min": min(lo[k]), "max": max(hi[k])} for k in sorted(lo)}
    out["all"] = [min(v["min"] for v in out.values()), max(v["max"] for v in out.values())]
    return out


def dropped(shots: dict) -> dict:
    """Hardware points with P(n) >= 1/2, per basis, distance and decoder."""
    count: dict = defaultdict(int)
    for info in shots["per_job"].values():
        for dec, by_setting in info["logical_error"].items():
            for setting, errors in by_setting.items():
                for k, patch in enumerate(info["patches"]):
                    if errors[k] >= 0.5:
                        count[f"{setting[0]} d{patch} {dec}"] += 1
    return dict(sorted(count.items()))


def per_placement_lambda(shots: dict) -> dict:
    """Lambda per placement under both weightings: inverse-variance mean over the five days,
    its error widened by sqrt(chi2/nu) when chi2/nu > 1, as placements.py does it."""
    out: dict = {}
    for dec in ("calibrated", "uniform"):
        for q in ("X lambda", "Z lambda"):
            for placement in (1, 2):
                rows = [r[dec][q] for r in shots["reps"] if r["rep"] == placement]
                v = np.array([x[0] for x in rows])
                w = 1 / np.array([x[1] for x in rows]) ** 2
                m = float((w * v).sum() / w.sum())
                red = float((w * (v - m) ** 2).sum() / (len(v) - 1))
                out[f"{dec} {q} placement {placement}"] = {
                    "value": m,
                    "se": float(np.sqrt(max(red, 1.0) / w.sum())),
                    "chi2_per_dof": red,
                }
    return out


def miscorrected_per_round(mass: dict) -> dict:
    """Probability mass of the single faults that calibrated matching miscorrects, per round:
    (8-round mass - 1-round mass) / 7, per job, patch and basis. To first order this is what
    those faults add to eps."""
    out: dict = {}
    for job, rows in mass.items():
        for basis in "XZ":
            for patch in ("3", "5"):
                if patch not in rows.get(f"{basis}1", {}):
                    continue
                m1 = rows[f"{basis}1"][patch]["miscorrected_single_p_calibrated"]
                m8 = rows[f"{basis}8"][patch]["miscorrected_single_p_calibrated"]
                out[f"{job} {basis} d{patch}"] = (m8 - m1) / 7
    return out


def paper(s: dict, extra: dict) -> dict:
    """The numbers the paper quotes."""
    rd = result("decoder_shots.json")
    ex = result("decoder_examples.json")
    ties = result("decoder_ties.json")
    fs, hw = s["fault_scan"], s["hardware"]
    sim = s["model_simulation_20000_shots"]["pooled"]
    mean_eps = hw["mean_eps_over_reps (mean, se over reps)"]
    shifts = {
        k: 100 * (mean_eps[f"uniform {k}"][0] - mean_eps[f"calibrated {k}"][0])
        for k in ("X eps3", "X eps5", "Z eps3", "Z eps5")
    }
    uni_x = [(r["day"], r["rep"], *r["uniform"]["X lambda"]) for r in rd["reps"]]
    worst = max(uni_x, key=lambda r: r[2])
    keys = (
        "circuits",
        "mechanisms",
        "undetectable_single",
        "single_miscorrections_calibrated",
        "circuits_with_single_miscorrection_calibrated",
        "single_miscorrections_uniform",
    )
    pkeys = (
        "pairs",
        "undetectable_logical_pairs",
        "pair_miscorrections_calibrated",
        "pair_miscorrections_uniform",
    )
    x_d5 = [
        [
            name[:10],
            int(name[12]),
            int(setting[1:]),
            row["patches"]["5"]["single_miscorrections_calibrated"],
        ]
        for name, rows in result("decoder_faults.json")["jobs"].items()
        for setting, row in rows.items()
        if setting[0] == "X"
        and "5" in row["patches"]
        and row["patches"]["5"]["single_miscorrections_calibrated"]
    ]
    z3 = ties["Z d3 singles rounds (1, 2, 3, 4, 6, 8)"]
    z5 = ties["Z d5 pairs rounds (1, 2)"]
    example = ex["examples"][0]
    return {
        "circuit_distance (exact<=4 [None = >4], stim graphlike, stim search)": s[
            "distances_seen (exact<=4 [None = >4], stim graphlike, stim search)"
        ],
        "fault_scan": {
            b: {k: fs[b][k] for k in keys + (pkeys if b.endswith("d5") else ())} for b in CELLS
        },
        "X d5 single miscorrections [day, rep, rounds, faults]": x_d5,
        "uniform_ties_12_seeds": {
            "Z d3 singles": {
                "wrong_with_pymatching_tiebreak": z3["mis_uniform"],
                "wrong_under_every_perturbed_tiebreak": z3["mis_all_jitter"],
            },
            "Z d5 pairs, rounds 1-2": {
                "wrong_with_pymatching_tiebreak": z5["mis_uniform"],
                "wrong_under_every_perturbed_tiebreak (upper bound)": z5["mis_all_jitter"],
            },
        },
        "matching_graph": {
            "shortest_logical_edges": {
                b: s["matching_graph"][b]["decoder_graph_hops_seen"] for b in s["matching_graph"]
            },
            "lightest_logical_weight_range": {
                b: s["matching_graph"][b]["weighted_distance_W_range"] for b in s["matching_graph"]
            },
            "edge_probability_range": extra["edge_p"]["all"],
            "fraction_edges_heavier_than_half_lightest_logical": {
                b: s["matching_graph"][b]["fraction_edges_heavier_than_half_W"]
                for b in s["matching_graph"]
            },
            "z_d5_flag_relay_pieces": extra["pieces"]["Z"]["n"],
        },
        "example_X_d3": {
            "job": ex["job"],
            "fault_dem_p": [m["p"] for m in example["actual_fault_dem_mechanisms"]],
            "calibrated_edges_p": [e["p"] for e in example["calibrated_matching_edges"]],
        },
        "hardware": {
            "pooled_prespecified_rule": hw["pooled"],
            "only_one_decoder_fails": hw["mcnemar"],
            "mean_eps_over_reps": mean_eps,
            "mean_eps_shift_uniform_minus_calibrated_pp": shifts,
            "mean_eps_shift_range_pp": [min(shifts.values()), max(shifts.values())],
            "per_rep_Lambda_X_uniform_all_below_1": all(r[2] < 1 for r in uni_x),
            "per_rep_Lambda_X_uniform_verdicts": [verdict(r[2], r[3]) for r in uni_x],
            "per_rep_Lambda_X_uniform_largest": {
                "day": worst[0],
                "rep": worst[1],
                "value": worst[2],
                "se": worst[3],
            },
            "points_dropped_P_ge_half": extra["dropped"],
            "per_placement_lambda": per_placement_lambda(rd),
        },
        "miscorrected_single_mass_per_round_calibrated": miscorrected_per_round(
            result("decoder_mass.json")
        ),
        "model_simulation_20000_shots": {
            "pooled_lambda": {k: sim[k]["lambda"] for k in sim},
            "Lambda_X_drop_uniform": sim["calibrated X"]["lambda"] - sim["uniform X"]["lambda"],
        },
    }


def main() -> None:
    s = collect()
    extra = {
        "pieces": pieces(),
        "edge_p": edge_p(result("decoder_weights.json")),
        "dropped": dropped(result("decoder_shots.json")),
    }
    s["checks"] = extra
    s["paper"] = paper(s, extra)
    write_json("decoder.json", s)
    print(json.dumps(s["paper"], indent=1))


if __name__ == "__main__":
    main()

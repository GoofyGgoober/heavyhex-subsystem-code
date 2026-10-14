"""Every number the paper quotes, collected from the results files, each section with its source.

No computation: this only selects from results/*.json (and runs/combined.json, the stored
predictions in runs/*/prediction.json, docs/figures/figure3.json) and writes results/paper_numbers.json and a readable digest,
results/paper_numbers.txt. Run it after the scripts whose results it reads; it takes a second.

    python docs/analysis/ledger.py
"""

from __future__ import annotations

import math
import re
import sys

from common import REPO, RESULTS, jobs, memory_index, read, result, write_json

FITS = ("wls", "binomial")
BASES = ("X", "Z")
QUANTITIES = ("eps3", "eps5", "lambda")
PLACEMENTS = (1, 2)
INPUTS: set[str] = set()


def weight(edges: list[float]) -> float:
    """Total matching weight of a set of edges, sum of ln[(1-p)/p]."""
    return sum(math.log((1 - p) / p) for p in edges)


def take(name: str) -> dict:
    INPUTS.add(f"docs/analysis/results/{name}")
    return result(name)


def model_vs_device() -> dict:
    m = take("model_vs_device.json")
    d, f = m["detectors"], m["f"]
    return {
        "days (Table days)": m["days"],
        "calibration_scatter_check_exploratory": {
            "label": "exploratory; the added spread comes from the predictions only (calibrations and idle-test f)",
            **m["scatter"],
        },
        "detectors_beyond_chance": {k: v for k, v in d.items() if k != "per_job"},
        "excess_slope": m["slope"],
        "f": {k: v for k, v in f.items() if k != "per_job"},
        "selection_check (reported errors on the chosen day / other days)": m["selection"],
        "prespecified_drift_check": m["drift"],
        "model_curves_day1_rep1": m["model_curves"],
    }


def effect_sizes() -> dict:
    """The model-versus-device effect sizes (effect_sizes.py), without the per-repetition and
    per-curve detail that results/effect_sizes.json keeps."""
    e = take("effect_sizes.json")
    shown = ("placement 1", "placement 2", "all ten")
    chi2_keys = ("chi2", "p", "agree", "z_positive", "mean_z", "bias", "scatter")
    return {
        "headline": e["headline"],
        "O_over_M": {
            fit: {q: {k: v[k] for k in shown} for q, v in by_q.items()}
            for fit, by_q in e["ratios"].items()
        },
        "chi2": {
            fit: {
                q: {
                    **{k: v[k] for k in chi2_keys},
                    "p_holm_over_the_three_X_tests": v.get("p_holm_over_the_three_X_tests"),
                    "per_placement": v["per_placement"],
                    "without_f_term": {
                        k: v["without_f_term"][k] for k in ("chi2", "p", "agree", "per_placement")
                    },
                }
                for q, v in by_q.items()
            }
            for fit, by_q in e["chi2"].items()
        },
        "sigma_M_budget (WL)": {
            q: {p: v[p] for p in ("placement 1", "placement 2")}
            for q, v in e["budget"]["wls"].items()
        },
        "sigma_f_by_job_kind": e["budget"]["sigma_f_by_job_kind"],
        "power": {k: e["power"][k] for k in ("wls", "binomial", "idealized_power_json")},
        "tracking": e["tracking"],
        "prejob": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_repetition"} if k in FITS else v
            for k, v in e["prejob"].items()
        },
        "table_one_convention": e["table"],
        "errors": {k: v for k, v in e["errors"].items() if k != "per_curve"},
        "window": e["window"],
        "selection": e["selection"],
        "prespecified_pool": e["pool"],
    }


def sensitivity() -> dict:
    """The agreement test under other sigma_f, its Monte Carlo spread, Lambda_X's dependence on
    the window and the fit, intervals, power and day-to-day tracking (sensitivity.py)."""
    s = take("sensitivity.json")
    sf, mc, w = s["sigma_f"], s["monte_carlo"], s["windows"]
    keys = (
        "chi2",
        "p",
        "agree",
        "per_placement",
        "offset",
        "scatter",
        "offset_t_test",
        "constant_multiple (O = k M, 9 dof)",
    )
    return {
        "sigma_f_range": sf["sigma_f_range"],
        "sigma_f_idle_noise_over_prespecified": sf["ratio_range"],
        "f_term_share (WL)": sf["f_term_share_of_the_z_variance (WL, prespecified) [min, max]"],
        "x_tests by sigma_f": {
            fit: {
                label: {q: {k: v[k] for k in keys} for q, v in blk.items()}
                for label, blk in sf[fit].items()
            }
            for fit in FITS
        },
        "monte_carlo (WL)": {label: mc[label] for label in ("prespecified", "idle_noise_only")},
        "monte_carlo_slopes, parabola jobs": {
            j: v for j, v in mc["slopes_by_job"].items() if v["rule"] == "parabola"
        },
        "line_slope_above_f_1.05 (against the secant from f = 1 and the parabola)": {
            j: v for j, v in mc["slopes_by_job"].items() if v["f"] > 1.05
        },
        "control_rerun": next(v for k, v in mc.items() if k.startswith("control_rerun")),
        "seeds_check": mc["seeds_check"],
        "Lambda_X windows and fits": {
            "systematic": {
                label: {x: b[x] for x in ("device", "model", "device_minus_model")}
                for label, b in w["shifts"].items()
            },
            "leaving_out_round_1": w["leaving_out_round_1 (2-8 minus 1-8)"],
        },
        "intervals": s["intervals"],
        "power": s["power"],
        "tracking placement 1 Lambda_X": {
            fit: {k: s["tracking"][fit][k] for k in ("all days", "without day 1", "without day 2")}
            for fit in FITS
        },
        "X5X6X8X9 detection fraction by day (placement 1, d = 3, bulk rounds)": {
            day: v["stabilizers"]["X5X6X8X9"] if day.startswith("day") else v
            for day, v in s["tracking"]["stabilizers (placement 1, distance 3)"].items()
        },
        "calibration_scatter_test_power": s["scatter_power"],
    }


def dependence() -> dict:
    """The X tests with the repetitions' dependence allowed for, the raw counts against the model
    and Lambda_X against a practical margin (dependence.py), without the per-curve detail."""
    d = take("dependence.json")
    return {
        "headline": d["headline"],
        "structure": d["structure"],
        "blocks": d["blocks"],
        "counts": {k: d["counts"][k] for k in ("what", "tally", "groups")},
        "equivalence": d["equivalence"],
    }


def prespecified_tests() -> dict:
    """The tests fixed before the run under both sigma_f, the pooled verdict, the pairs' table
    with predictions widened like observations, the fit-window systematic and the power
    (prespecified_tests.py); the ten z values of each test are kept once, per pair-day."""
    t = take("prespecified_tests.json")

    def no_z(blk: dict) -> dict:
        return {q: {k: v for k, v in x.items() if k != "z"} for q, x in blk.items()}

    od = t["sigma_f"]["per_qubit_fits_overdispersed"]
    return {
        "headline": t["headline"],
        "per_pair_day": [
            {
                "day": r["day"],
                "pair": r["pair"],
                "verdict (WL)": r["verdict (WL)"],
                **{
                    fit: {
                        q: {k: v for k, v in c.items() if k.startswith(("z", "sigma_M"))}
                        for q, c in r[fit].items()
                    }
                    for fit in FITS
                },
            }
            for r in t["per_pair_day"]
        ],
        "chi2": {
            "bounds": t["chi2"]["bounds"],
            **{fit: {label: no_z(b) for label, b in t["chi2"][fit].items()} for fit in FITS},
        },
        "pool": t["pool"],
        "sigma_f": {
            **{k: v for k, v in t["sigma_f"].items() if k != "per_qubit_fits_overdispersed"},
            "per_qubit_fits_overdispersed": {k: v for k, v in od.items() if k != "per_job"},
        },
        "windows": t["windows"],
        "table_pairs": t["table_pairs"],
        "power": t["power"],
    }


def per_qubit_f_side() -> dict:
    r = take("perqubit_f.json")
    model = r["versions"]["per_qubit_f"]["model"]
    reps = r["versions"]["per_qubit_f"]["reps"]
    pl = r["placements"]
    return {
        "chi2": {q: model[q]["chi2"] for q in model},
        "bar": model["X lambda"]["bar"],
        "agrees": {q: model[q]["agrees"] for q in model},
        "reps_agreeing": {q: model[q]["reps_agreeing"] for q in model},
        "reps_agreeing_on_all_three_X": sum(
            all(e["tests"][f"X {q}"]["agrees"] for q in QUANTITIES) for e in reps
        ),
        "predicted_Lambda_X_mean": {
            f"placement {p}": {
                "median_f": pl["median_f"][str(p)]["X lambda"]["predicted_mean"],
                "per_qubit_f": pl["per_qubit_f"][str(p)]["X lambda"]["predicted_mean"],
            }
            for p in PLACEMENTS
        },
    }


F_MODELS = (
    "stored median",
    "stored per_qubit",
    "median",
    "per_qubit",
    "q150",
    "shrunk",
    "shrunk_scaled",
)


def f_models() -> dict:
    """The q150 artefact and the f model behind the placement result (f_models.py)."""
    r = take("f_models.json")
    slope = r["slope"]
    return {
        "q150": r["q150"],
        "pair_1_predicted (eps3, eps5, Lambda_X)": {
            m: {
                q: {
                    k: r["per_pair"]["pair 1"][m][q][k]
                    for k in ("by_day", "mean", "sim_se_of_mean")
                }
                for q in QUANTITIES
            }
            for m in F_MODELS
        },
        "pair_2_predicted_Lambda_X_mean": {
            m: r["per_pair"]["pair 2"][m]["lambda"]["mean"] for m in F_MODELS
        },
        "placement_difference": r["placement"],
        "share_of_the_per_qubit_Lambda_X_shift_from_q150": r[
            "share_of_the_per_qubit_shift_from_q150 (pair 1 Lambda_X, reruns)"
        ],
        "X_tests (chi2, agreeing pair-days)": {
            label: {
                m: {q: [v["chi2"], v["agree"]] for q, v in by.items()} for m, by in by_model.items()
            }
            for label, by_model in r["tests"].items()
        },
        "sigma_f": {k: v for k, v in r["sigma_f"].items() if k != "per_job"},
        "slope": {
            "by_job": slope["by_job"],
            "X_tests (chi2, agreeing pair-days)": {
                where: {
                    label: {
                        k: {q: [v["chi2"], v["agree"]] for q, v in b.items()}
                        for k, b in by_slope.items()
                    }
                    for label, by_slope in by_label.items()
                }
                for where, by_label in slope["tests"].items()
            },
        },
        "reruns_check": r["reruns_check"],
    }


def f_models_digest(f: dict) -> list[str]:
    pl = f["placement_difference"]
    o = pl["observed (binomial, inverse-variance over the days; placements.py)"]
    p1 = f["pair_1_predicted (eps3, eps5, Lambda_X)"]
    q = list(f["q150"].values())
    steep = f["slope"]["by_job"]["ibm_fez-2026-10-06-r2-d3"]["eps3"]
    eps3 = f["slope"]["X_tests (chi2, agreeing pair-days)"]["f = 1.52 only"]
    sf = f["sigma_f"]
    return [
        "   f models (f_models.py): pair 1 - pair 2 predicted Lambda_X against observed "
        f"{fmt_pm(o['value'], o['se'])}: "
        + ", ".join(
            f"{m} {pl[m]['predicted_difference']:.3f} (z {pl[m]['z (with sim se)']:+.1f})"
            for m in F_MODELS
        ),
        "   pair 1 predicted eps5 / Lambda_X (mean): "
        + ", ".join(
            f"{m} {100 * p1[m]['eps5']['mean']:.2f}% / {p1[m]['lambda']['mean']:.3f}"
            for m in F_MODELS
        )
        + f"; q150 alone gives {f['share_of_the_per_qubit_Lambda_X_shift_from_q150']:.2f} "
        "of the per-qubit Lambda_X shift",
        "   q150 Z-or-Y per round, median f -> own f, days 1-5: "
        + ", ".join(
            f"{100 * d['z_or_y_per_round']['median']:.1f}->{100 * d['z_or_y_per_round']['per_qubit']:.1f}%"
            for d in q
        ),
        f"   sigma_f prespecified (ddof 0) {sf['range prespecified (ddof 0)'][0]:.3f}-"
        f"{sf['range prespecified (ddof 0)'][1]:.3f}, ddof 1 {sf['range same with ddof 1'][0]:.3f}-"
        f"{sf['range same with ddof 1'][1]:.3f}; noise only (SD of bootstrap medians, ddof 1) "
        f"{sf['range idle_noise_only (bootstrap, ddof 1)'][0]:.3f}-"
        f"{sf['range idle_noise_only (bootstrap, ddof 1)'][1]:.3f}",
        f"   dM/df at f = 1.52 (eps3): line {steep['slope']:.4f}, secant "
        f"{fmt_pm(steep['slope_secant_from_1'], steep['slope_secant_from_1 mc_sd'], 4)} "
        f"(line/secant {steep['line_over_secant']:.2f}), parabola "
        f"{fmt_pm(steep['slope_parabola'], steep['slope_parabola mc_sd'], 4)} "
        f"(line/parabola {steep['line_over_parabola']:.2f}); X eps3 chi2 line/secant/parabola "
        + " | ".join(
            f"{label} " + "/".join(f"{v['X eps3'][0]:.1f}" for v in by.values())
            for label, by in eps3.items()
        ),
    ]


def heating() -> dict:
    h = take("heating.json")

    def metric(key):
        a = h["time_H1"][key]["across_jobs"]
        return {
            "random_effects_DL": {
                "change": a["random"]["slope"],
                "se": a["random"]["se"],
                "ci95": a["random"]["ci95"],
                "tau": a["random"]["tau"],
            },
            "hartung_knapp_ci95": a["hartung_knapp"]["ci95"],
            "fixed_effect": {"change": a["fixed"]["slope"], "se": a["fixed"]["se"]},
            "plain_t_over_jobs_ci95": a["plain_mean_t"]["ci95"],
            "jobs": a["jobs"],
            "positive_slopes": a["positive_slopes"],
            "z_min": a["z_min"],
            "z_max": a["z_max"],
            "stouffer_z": a["stouffer_z"],
            "jobs_rising_bonferroni": a["jobs_rising_bonferroni"],
            "jobs_falling_bonferroni": a["jobs_falling_bonferroni"],
        }

    det = metric("detection_events")
    eps = h["time_H1"]["logical_eps_per_round"]["across_jobs"]
    anc = h["time_H1"]["idle_ancilla_per_qubit"]
    step = h["ordering"]["step"]
    clock = h["job_wall_clock"]
    shot_seconds = list(h["seconds_per_job_of_shots"].values())
    return {
        "what": h["what"],
        "bins_per_job": h["bins"],
        "bonferroni_z": h["bonferroni_z"],
        "qpu_seconds_billed_per_job": sorted({c[2] for c in clock.values()}),
        "shot_seconds_per_job_ibm_estimate": [min(shot_seconds), max(shot_seconds)],
        "detection_events": det,
        "detection_events_rise_excluded_above (95% upper ends)": {
            "DL": det["random_effects_DL"]["ci95"][1],
            "HKSJ": det["hartung_knapp_ci95"][1],
            "plain_t": det["plain_t_over_jobs_ci95"][1],
        },
        "detection_events_d3": metric("detection_events_d3"),
        "detection_events_d5": metric("detection_events_d5"),
        "flags_relays_Z_memory": metric("flags_relays_Z_memory"),
        "logical_failure": metric("logical_failure"),
        "logical_eps_per_round_per_bin": {
            "random_effects_DL": eps["random"],
            "positive_jobs": eps["positive_slopes"],
        },
        "idle_ancilla_ones": metric("idle_ancilla_ones"),
        "idle_data_ones_Y_readout": metric("idle_data_ones_Y_readout"),
        "idle_data_ones_X_readout": metric("idle_data_ones_X_readout"),
        "idle_contrast_loss": {
            "random_effects_DL": h["time_H1"]["idle_contrast_loss"]["across_jobs"]["random"]
        },
        "idle_phase_drift": h["time_H1"]["idle_phase_drift"],
        "idle_ancillas_per_qubit": {
            k: anc[k]
            for k in (
                "qubit_jobs",
                "positive",
                "z_above_3",
                "z_below_minus_3",
                "expected_beyond_3_each_side_if_no_drift",
                "median_slope",
            )
        },
        "step": {
            k: step[k]
            for k in (
                "job",
                "qubit",
                "switch_shot_index",
                "rate_before",
                "rate_after",
                "pieces",
                "pieces_lower_after",
                "sign_test_p",
            )
        },
        "adjacent_shot_correlation_lag1": {
            "interleaved": h["ordering"]["adjacent_shot_correlation"]["H1 lag 1"],
            "piece_by_piece": h["ordering"]["adjacent_shot_correlation"]["H2 lag 1"],
        },
        "pieces_covary": {
            k: h["ordering"]["common_mode"][k]
            for k in (
                "mean_pair_corr_same_job",
                "null_mean",
                "null_sd",
                "p_one_sided",
                "mean_pair_corr_cross_job",
            )
        },
        "alternative_piece_order_detection_events": h["piece_order_H2"]["detection_events"][
            "pooled"
        ],
        "prespecified_drift_reproduced_max_abs_difference": h["reproduce_prespecified_drift"][
            "max_abs_difference"
        ],
    }


def leakage() -> dict:
    ref = take("refiring.json")
    hmm = take("hmm_summary.json")
    boot = take("hmm_bootstrap.json")
    causes = take("hmm_causes_summary.json")
    co = take("costreak.json")
    fit = take("leakage_fit.json")
    checks = take("leakage_checks.json")
    cz = take("cz_scaled.json")
    life = take("leakage_lifetime.json")
    recovery = take("leakage_recovery.json")
    exact = fit["reruns"]
    return {
        "refiring": {
            k: ref[k]
            for k in (
                "lags",
                "series",
                "chip_above_model_at_lag_1",
                "runs_of_5",
                "base_odds",
                "conditional_by_lag",
            )
        },
        "hmm": hmm,
        "hmm_bootstrap": {
            k: boot[k]
            for k in (
                "published",
                "shot_bootstrap_ci95",
                "cluster",
                "lengths",
                "capped_at_300",
                "converged_paper_start",
                "starts",
                "best_of_starts",
            )
        },
        "hmm_on_simulated_leakage (lifetime in us -> fitted tau_P, e_P, Delta(1))": {
            k: {"tau_P_us": v["life_us_cont"], "e_P": v["e_P"], "refire_lag_1": v["refire"][0]}
            for k, v in life["lifetimes"].items()
        },
        "simulated_lifetime_matching_the_device (us, and Delta(1) there)": life["matching_device"],
        "parameter_recovery_rank_of_truth": {
            t: {"truth": v["truth"], "pooled": [r["pooled rank_of_truth"] for r in v["rows"]]}
            for t, v in recovery["grid_fit"].items()
        },
        "causes": causes,
        "costreak": {
            k: {
                kk: v[kk]
                for kk in (
                    "rows",
                    "physical_pairs",
                    "chip",
                    "chip_se_pair_clustered",
                    "model_exact",
                    "per_slot_definition",
                )
                if kk in v
            }
            for k, v in co["classes"].items()
        }
        | {"streak_probability_by_slot": co["streak_probability_by_slot"]},
        "exact_grid_identity": {
            k: fit["grids"]["identity"][k] for k in ("best", "top5", "seed_replicated", "edges")
        },
        "exact_grid_other_channels_best": {
            k: fit["grids"][k]["best"] for k in fit["grids"] if k != "identity"
        },
        "published_grid_comparison": fit["published_grid_comparison"],
        "table_rerun_and_closures": {
            label: {
                est: {
                    g: {
                        cell: {
                            kk: r[est][g][cell][kk]
                            for kk in ("device", "model", "leak", "closure", "leak_minus_device")
                            if kk in r[est][g][cell]
                        }
                        for cell in ("X d3", "X d5", "Z d3", "Z d5")
                    }
                    for g in ("all", "fit", "heldout")
                }
                for est in FITS
            }
            for label, r in exact.items()
        },
        "splits": {
            label: {est: {c: r[est]["splits"][c] for c in ("Z d3", "Z d5")} for est in FITS}
            for label, r in exact.items()
        },
        "closure_against_ibm_reset_and_readout (wls, all ten)": {
            label: r["wls"]["closure_against_ibm_reset_and_readout"] for label, r in exact.items()
        },
        "frame_binomial_means": fit["frame_binomial"]["means_over_ten_reps"],
        "checks_three_qubit_example": checks["A"],
        "checks_cz_operands_with_definite_z": checks["E"]["total"],
        "cz_scaled": {
            "follow_ratio_mean": cz["follow_ratio_mean"],
            "twice_cz_minus_leakage_model_fit_jobs (the paper's CZ x 2 sentence)": exact[
                "identity_ext_optimum"
            ]["twice_cz_minus_leak_fit_jobs"],
            "minus_fit_old_frame_model_at_0.2_25_5 (not quoted)": cz["minus_fit"],
        },
    }


def streak_effects() -> dict:
    """What a flag or relay streak predicts (streak_effects.py, exploratory): the near- and
    far-stabilizer detection differences, streak against quiet and against a single firing, by
    lag, and the logical failure of shots with a streak, device against model."""
    s = take("streak_effects.json")
    keep = (
        "units",
        "physical_qubits",
        "chip_quiet",
        "chip_single",
        "chip",
        "chip_ci95_qubits",
        "model",
        "chip_minus_model_ci95_qubits",
        "qubits_with_chip_effect_above_zero",
        "chip_flip_equivalent",
        "chip_flip_equivalent_ci95_qubits",
    )
    groups = [f"{b} d{d} both" for b in "XZ" for d in ("3", "5", "both")]
    groups += [f"{b} dboth {r}" for b in "XZ" for r in ("flag", "relay")]
    det = s["detection_events_after_a_streak"]
    failure = s["logical_failure"]
    cells = [f"{b} d{d}" for b in "XZ" for d in ("3", "5", "both")]
    return {
        "about": s["about"],
        "detection_events": {
            g: {
                lag: {
                    o: {
                        c: {k: v[c][k] for k in keep if k in v[c]}
                        for c in v
                        if c != "single - quiet"
                    }
                    for o, v in by_o.items()
                }
                for lag, by_o in det[g].items()
            }
            for g in groups
        },
        "logical_failure": {
            cell: {
                exp: {
                    src: {
                        k: failure[cell][exp][src][k]
                        for k in (
                            "failure_reference",
                            "failure_exposed",
                            "rd",
                            "rd_ci95",
                            "rr",
                            "rr_ci95",
                        )
                    }
                    for src in ("chip", "model")
                }
                | {"chip_minus_model_rd": failure[cell][exp]["chip_minus_model_rd"]}
                for exp in ("any", "count matched", "flag only", "relay only")
            }
            | {
                "failure_per_firing": failure[cell]["failure_per_firing"],
                "shots_with_a_streak_8_rounds": failure[cell]["shots_with_a_streak_8_rounds"],
            }
            for cell in cells
        },
        "per_repetition_summary": {
            k: v for k, v in failure["per_repetition_any_streak_chip"].items() if "summary" in k
        },
    }


def placements_and_inputs() -> dict:
    """Placement counts and ranks, relay q51, and the calibration medians the paper quotes."""
    pc = take("placement_counts.json")
    cal = take("calibration_inputs.json")
    run_py = (REPO / "src/heavyhex/experiment/run.py").read_text(encoding="utf-8")
    rehearsal = re.search(r"rehearsal reached (\d+\.\d+)", run_py)
    d3, d5 = pc["d3"], pc["d5"]
    return {
        "d3": {
            k: d3[k] for k in ("searched_sets", "embeddings", "all_sets", "searched_subset_of_all")
        }
        | {
            "usable (searched, all) by day": {
                day: [v["usable_searched"], v["usable_all"]] for day, v in d3["per_day"].items()
            },
            "rank of repetition 2's d3 patch among all usable, by day": {
                day: v["rank_d3_rep2"] for day, v in d3["per_day"].items()
            },
        },
        "d5": {k: d5[k] for k in ("searched_sets", "all_sets")}
        | {
            "usable and stricter bars by day": {
                day: {k: v for k, v in x.items() if k != "usable_searched"}
                for day, x in d5["per_day"].items()
            }
        },
        "relay_q51": {
            day: {
                "reported_readout_error": v["reported_readout_error"],
                "read_1_range": [
                    min(v["read_1_fraction_by_round"]),
                    max(v["read_1_fraction_by_round"]),
                ],
                "stabilizer_range_after_round_1": [
                    min(v["stabilizer_detection_fraction_by_round"][1:]),
                    max(v["stabilizer_detection_fraction_by_round"][1:]),
                ],
            }
            for day, v in pc["relay_q51"].items()
        },
        "weak_parts_by_day (placement.weak)": {
            day: {name: len(parts) for name, parts in v["weak"].items()}
            for day, v in pc["weak"].items()
        },
        "recurring_weak_parts_of_placement_1_by_day": {
            day: v["recurring"] for day, v in pc["weak"].items()
        },
        "pair_2_d3_location_by_day": pc["pair2_d3"],
        "round_us": pc["round_us"],
        "single_qubit_error_median_by_day": cal["single_qubit"],
        "calibration_age_by_day": cal["age"],
        "readout_gap_P01_minus_P10_flags_relays": cal["readout_gap"],
        "t1_of_the_flag_and_relay_qubits": cal["t1_flags_relays"],
        "drift_rehearsal_largest_change_se (comment above DRIFT_BAR, src/heavyhex/experiment/run.py)": (
            float(rehearsal.group(1)) if rehearsal else None
        ),
    }


def diagnostics() -> dict:
    """The detector, decoder-prior, dephasing and flag-record analyses added after the run."""
    de = take("detection_events.json")
    pij = take("pij.json")
    pd = take("pij_decode.json")
    cm = take("corr_matching.json")
    rh = take("reset_harper.json")
    dom = take("decoder_dominant.json")
    idle = take("idle_f_t2.json")
    flags = take("flags_summary.json")
    hmm_x = take("hmm_x_summary.json")
    f = idle["f_vs_t2"]
    return {
        "detection_event_fraction": {
            "detectors": de["detectors"],
            "per_detector (device / model)": {
                k: de["per_detector"][k]
                for k in ("all", "X weight 4", "X weight 2", "Z", "flag", "relay")
            },
            "bulk rounds": de["bulk"],
            "rise from t = 2 to t = 8": de["rise"],
        },
        "pij": {
            "classes (median device / model)": {
                c: {
                    k: v[k]
                    for k in (
                        "types",
                        "median_model_p",
                        "median_device_p",
                        "median_ratio",
                        "median_ratio_se",
                        "share_above_1",
                        "control_median_ratio",
                    )
                }
                for c, v in pij["classes"]["X dboth"].items()
            },
            "classes Z d3 (median device / model)": {
                c: {k: v[k] for k in ("types", "median_ratio", "median_ratio_se")}
                for c, v in pij["classes"]["Z d3"].items()
            },
            "pairs with no model edge": pij[
                "unmodelled_pairs [patch basis same-detector lag(3 = 3+)], mean and SE over jobs"
            ],
            "edges above p = 0.1": pij["high_p_types"],
            "by class and leading fault": pij["high_p_types_by_class_and_leading_fault"],
        },
        "learned_priors": {
            "gaps": pd["gaps"],
            "device_minus_calibrated": pd[
                "device_minus_calibrated [variant][placement basis fit quantity]"
            ],
            "edges_keeping_the_calibrated_p": pd["edges_keeping_the_calibrated_p"],
        },
        "correlated_matching": {
            "device": cm["device_correlated_minus_calibrated"],
            "model": cm["model_correlated_minus_calibrated"],
            "pieces_dropped": cm["pieces_dropped"],
        },
        "harper_reset_readout_day1": {
            "time_like_edges": rh["time_like_edges_day1"],
            "eps": rh["eps_day1 (wls unless named)"],
        },
        "miscorrected_single_faults_X_d3": {
            "share_of_predicted_eps3_range": dom["share_range"],
            "largest_by_patch": {k: v["largest"] for k, v in dom["patches"].items()},
        },
        "idle_f_against_T2": {
            k: f[k]
            for k in (
                "qubit_jobs_used",
                "physical_qubits",
                "median_f_over_qubit_jobs",
                "spearman_f_t2",
                "beta_gamma_dd_on_gamma_phi",
                "leverage (the short-T2 qubits left out)",
                "t2_bins",
            )
        }
        | {
            "ratio_artefact": {
                k: v for k, v in f["ratio_artefact"].items() if "by_qubit" not in str(v)
            }
        },
        "f_memory_against_idle": {k: v for k, v in idle["memory"].items() if k != "jobs"},
        "per_qubit_f_decoding": {k: v for k, v in idle["decode"].items() if k != "rows"},
        "flag_records_both_bases": {
            "refiring": {
                b: {k: v for k, v in r.items() if k not in ("lags", "lag_us")}
                for b, r in flags["refiring"].items()
                if b in ("Z", "X")
            }
            | {"X_minus_Z": flags["refiring"]["X_minus_Z"]},
            "hmm": flags["hmm"],
            "alternatives": flags["alternatives"],
            "roles_x_memory": flags["roles_x_memory"],
            "streak_covariance": flags["costreak"],
        },
        "hmm_x_memory": hmm_x,
    }


def scale_and_f() -> dict:
    """One factor on every error, the dephasing floor, the model at the weighted-mean f and the
    resolution of the readout inputs."""
    es = take("error_scale.json")
    fw = take("f_weighted_mean.json")

    def no_reps(blk: dict) -> dict:
        return {
            q: {k: v for k, v in x.items() if k != "per_repetition"} if isinstance(x, dict) else x
            for q, x in blk.items()
        }

    return {
        "error_scale": {
            "what": es["what"],
            "k1_against_stored_prediction": es["k1_against_stored_prediction"],
            "eps": {
                fit: {b: {d: no_reps(x) for d, x in blk.items()} for b, blk in by_b.items()}
                for fit, by_b in es["eps"].items()
            },
            "detection_rate": {
                b: {
                    d: {k: v for k, v in x.items() if k != "per_repetition"} for d, x in blk.items()
                }
                for b, blk in es["detection_rate"].items()
            },
            "lambda_at_joint_k": es["lambda_at_joint_k"],
        },
        "dephasing_floor": take("dephasing_floor.json"),
        "f_weighted_mean": {
            "f_by_job": fw["f_by_job"],
            **{
                fit: {
                    "O_over_M": {v: no_reps(blk) for v, blk in fw[fit]["O_over_M"].items()},
                    **{k: v for k, v in fw[fit].items() if k != "O_over_M"},
                }
                for fit in FITS
            },
            "f_from_detection_rates_above_idle": fw["f_from_detection_rates_above_idle"],
        },
        "input_resolution": take("input_resolution.json"),
    }


def error_budget() -> dict:
    """The model's error budget, one channel at a time, and what the model says would change
    eps and Lambda (error_budget.py), without the per-curve detail."""
    return {k: v for k, v in take("error_budget.json").items() if k != "per_curve"}


def error_budget_digest(b: dict) -> list[str]:
    fit = "binomial"
    channels = list(b["channels"])
    full = "+".join(channels)
    off = {c: "+".join(x for x in channels if x != c) for c in channels}
    L = [
        "22. Error budget (error_budget.py; binomial; % of the model's eps, mean over the ten "
        "pair-days: alone / drop when off / Shapley share)"
    ]
    for bd, blk in b["budget"][fit].items():
        a = blk["all ten"]
        idle = a["t1"]["shapley"]["mean"] + a["dephasing"]["shapley"]["mean"]
        L.append(
            f"   {bd}, model {100 * a['model_eps']['mean']:.1f}%: "
            + "; ".join(
                f"{c} {100 * a[c]['only']['mean']:.0f}/{100 * a[c]['off']['mean']:.0f}/"
                f"{100 * a[c]['shapley']['mean']:.0f}"
                for c in channels
            )
            + f"; idle Shapley {100 * idle:.0f}; measure off {100 * a['measure (off)']['mean']:.0f},"
            f" reset off {100 * a['reset (off)']['mean']:.0f}"
        )
    for bp, blk in b["lambda"][fit].items():
        model = blk[full]
        idles = [(k, v) for k, v in blk.items() if k.startswith("idles")]
        L.append(
            f"   {bp}: Lambda {model['lambda']:.3f}; with one channel off "
            + ", ".join(f"{c} {blk[off[c]]['lambda']:.3f}" for c in channels)
            + "; "
            + ", ".join(
                f"{k.split()[1]} idles: Lambda {v['lambda']:.3f}, eps3 x{v['eps3'] / model['eps3']:.2f},"
                f" eps5 x{v['eps5'] / model['eps5']:.2f}"
                for k, v in idles
            )
        )
    top = max(b["grid"])
    for basis, blk in b["excess"][fit].items():

        def k_of(j: dict) -> str:
            beyond = j["beyond_the_grid"]
            if "k" not in j or 2 * beyond > j["curves"]:
                return f"beyond {top:g} on {beyond}/{j['curves']}"
            more = f", beyond {top:g} on {beyond}/{j['curves']}" if beyond else ""
            return f"{j['k']:.2f} ({j['ci68'][0]:.2f}-{j['ci68'][1]:.2f}{more})"

        L.append(
            f"   one channel times k for the device's eps, {basis} (every channel "
            f"{blk['every channel (error_scale.py)']['k']:.2f}): "
            + ", ".join(f"{c} {k_of(blk[c]['both distances'])}" for c in channels)
        )
    for bd, blk in b["excess_detection_rate"].items():
        L.append(
            f"   one channel times k for the device's bulk detection rate, {bd} (every channel "
            f"{blk['every channel (error_scale.py)']['ratio']:.2f}): "
            + ", ".join(
                f"{c} {blk[c]['ratio']:.2f}" if "ratio" in blk[c] else f"{c} beyond {top:g}"
                for c in channels
            )
            + "".join(
                f" ({c}: {blk[c]['beyond_the_grid']} beyond)"
                for c in channels
                if "ratio" in blk[c] and blk[c]["beyond_the_grid"]
            )
        )
    L.append(f"   {b['parallel_schedule']}")
    return L


def stat_checks() -> dict:
    """The statistics with the binomial fit first, the size of the decision rule, the fit-window
    systematic with rounds 3-8, Lambda_X without day 2 of pair 1, the detector check's bar and
    the f read-offs (stat_checks.py), without the detector check's per-job detail."""
    s = take("stat_checks.json")
    return {**s, "detector_bar": {k: v for k, v in s["detector_bar"].items() if k != "per_job"}}


def stat_checks_digest(s: dict) -> list[str]:
    b = s["binomial_primary"]
    L = ["23. The statistics with the binomial fit first (stat_checks.py)"]
    for p in PLACEMENTS:
        blk = b[f"pair {p}"]
        L.append(
            f"   pair {p}, observed (predicted) binomial / log-linear: "
            + "; ".join(
                f"{q} "
                + " / ".join(
                    f"{scale * blk[q][fit]['observed']:.{d}f} ({scale * blk[q][fit]['predicted']:.{d}f})"
                    for fit in ("binomial", "wls")
                )
                for q, scale, d in (("eps3", 100, 1), ("eps5", 100, 1), ("lambda", 1, 3))
            )
            + f"; the binomial fit removes "
            f"{100 * blk['lambda']['share of the log-linear gap removed by the binomial fit']:.0f}% "
            "of the log-linear Lambda_X gap"
        )
    t = b["X tests [chi2, p, |z| < 2, Holm p] by fit and sigma_f"]["binomial"]["prespecified"]
    L.append(
        "   binomial X tests (prespecified sigma_f): "
        + ", ".join(f"{q} {v['chi2']:.1f} (p {v['p']:.2g})" for q, v in t.items())
    )
    r = s["rule_size"]
    own = r["each pair-day's own correlations (WL, prespecified sigma_f)"]
    bound = own["bound for a familywise 5% (95th percentile of the largest chi2)"]
    L.append(
        f"   size of 'any chi2 > 18.31': independent {r['independent tests (exact)']:.3f}, "
        f"z_Lambda = (z3 - z5)/sqrt 2 "
        f"{r['z_Lambda = (z3 - z5)/sqrt(2), z3 and z5 independent']['any of the three']:.3f}, "
        f"own correlations {own['any of the three']:.3f}; familywise-5% bound {bound:.2f} "
        f"(Bonferroni {r['bonferroni bound (chi2_10 at 1 - 0.05/3)']:.2f}), passed by "
        + ", ".join(q for q, v in r["rejected at the familywise-5% bound"].items() if v)
        + "; Holm p "
        + ", ".join(f"{q} {p:.2g}" for q, p in r["holm p (prespecified)"].items())
    )
    w = s["fit_window"]
    L.append(
        "   fit-window systematic on Lambda_X, device without / with rounds 3-8: "
        + "; ".join(
            f"{p} +-{w[p]['device']['rounds 3-8 left out']:.3f} / +-"
            f"{w[p]['device']['with rounds 3-8']:.3f}"
            for p in ("pair 1", "pair 2")
        )
    )
    o = s["lambda_offset"]
    for fit in ("binomial", "wls"):
        L.append(
            f"   Lambda_X O/M ({fit}, floored interval): "
            + "; ".join(
                f"{k} {v['ratio']:.2f} ({v['ci95 (floored, as the paper)'][0]:.2f}-"
                f"{v['ci95 (floored, as the paper)'][1]:.2f})"
                for k, v in o[fit].items()
                if k != "t test on z (prespecified sigma_f)"
            )
            + f"; t test on z without day 2 of pair 1: p "
            f"{o[fit]['t test on z (prespecified sigma_f)']['without day 2 of pair 1']['p']:.3f}"
        )
    for fit, blk in s["tracking"].items():
        if fit == "what":
            continue
        a, x = blk["all days"], blk["without day 2"]
        L.append(
            f"   pair 1, Lambda_X = c M ({fit}): all days c {a['c']:.3f}, chi2 {a['chi2 at c']:.1f}; "
            f"without day 2 c {x['c']:.3f}, chi2 {x['chi2 at c']:.1f} (p {x['p at c']:.2f}), at c = 1 "
            f"{x['chi2 at c = 1']:.1f} (p {x['p at c = 1']:.3f}), c = 1 against c p "
            f"{x['c = 1 against c: p (1 dof)']:.3f}"
        )
    d = s["detector_bar"]
    L.append(
        f"   detector bar sqrt(2 ln N) {d['bar range'][0]:.2f}-{d['bar range'][1]:.2f}, expected "
        f"largest {d['expected largest range'][0]:.2f}-{d['expected largest range'][1]:.2f}, 95th "
        f"percentile {d['95th percentile range'][0]:.2f}-{d['95th percentile range'][1]:.2f}; "
        "share beyond: "
        + ", ".join(
            f"{k.removeprefix('share beyond ')} {100 * v[0]:.0f}-{100 * v[1]:.0f}%"
            for k, v in d.items()
            if k.startswith("share beyond")
        )
    )
    f = s["f_read_off"]
    L.append(
        f"   f read off the memory runs agrees with the idle test: from eps in "
        f"{f['from eps agrees']}/{f['jobs']}, from the detection rates in "
        f"{f['from detection rates agrees']}/{f['jobs']}"
    )
    return L


def input_noise() -> dict:
    """What the errors of IBM's inputs add to sigma_M (input_noise.py), without the per-curve
    check."""
    n = take("input_noise.json")
    return {**n, "check": {k: v for k, v in n["check"].items() if k != "per_curve"}}


def input_noise_digest(n: dict) -> list[str]:
    c, i = n["check"], n["inputs"]
    hour = i["hour: day 1's two readout calibrations 56 minutes apart (the jobs' qubits)"]
    days = i["days: sd of ln x over the five days, by kind (the patches' qubits and couplers)"]
    shot = i["shot: readout errors' binomial sd of ln, median and range over the qubit-days"]
    L = [
        "24. The errors of IBM's inputs in sigma_M (input_noise.py)",
        f"   sum of elasticities {c['median here / error_scale'][0]:.2f} against error_scale "
        f"{c['median here / error_scale'][1]:.2f} (median |diff| "
        f"{c['median |here - error_scale|']:.2f}); inputs' sd of ln: readout shot noise median "
        f"{100 * shot[0]:.0f}%, readout an hour apart {100 * hour['sd_per_calibration (rms)']:.0f}% "
        f"(robust {100 * hour['sd_per_calibration (robust, 1.4826 MAD)']:.0f}%), day to day "
        + ", ".join(f"{k} {100 * v['median']:.0f}%" for k, v in days.items()),
    ]
    for case, blk in n["headline"].items():
        for key, tests in blk.items():
            L.append(
                f"   {case}, {key}: "
                + "; ".join(
                    f"{q} chi2 {v['chi2 [as tested, diagonal, covariance]'][0]:.1f} -> "
                    f"{v['chi2 [as tested, diagonal, covariance]'][1]:.1f} / "
                    f"{v['chi2 [as tested, diagonal, covariance]'][2]:.1f} (p "
                    f"{v['p [diagonal, covariance]'][1]:.2g}), sigma_in/M "
                    f"{100 * v['sigma_in / M range'][0]:.1f}-{100 * v['sigma_in / M range'][1]:.1f}%"
                    for q, v in tests.items()
                )
            )
    return L


def f_treatments() -> dict:
    """Observed over predicted under each treatment of f, each pair's means, Lambda_X without
    day 2 of pair 1, the range over the windows of rounds and every curve (f_treatments.py)."""
    return take("f_treatments.json")


def f_treatments_digest(f: dict) -> list[str]:
    def cell(v: dict) -> str:
        return (
            f"{v['ratio']:.2f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f}) above {v['above']}/{v['of']}"
        )

    L = [
        "26. O/M under each treatment of f (f_treatments.py; binomial; geometric mean, "
        "day-block t 95%, 4 dof)"
    ]
    for row, blk in f["table"].items():
        L.append(f"   {row}: " + "; ".join(f"{q} {cell(v)}" for q, v in blk.items()))
    L.append(
        "   f range: " + "; ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in f["f_range"].items())
    )
    for pair, blk in f["pairs"].items():
        L.append(
            f"   {pair} mean, device / at f = 1 / at the job's f: "
            + "; ".join(
                f"{q} "
                + " / ".join(
                    f"{v[k]:.3f}" if "lambda" in q else f"{100 * v[k]:.2f}"
                    for k in ("observed", "at f = 1", "at the job's f")
                )
                for q, v in blk.items()
            )
        )
    L.append(
        "   Lambda_X O/M without day 2 of pair 1: "
        + "; ".join(
            f"{k} {v['ratio']:.2f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"
            for k, v in f["lambda_without_day2_pair1"].items()
        )
    )
    w = f["windows"]
    L.append(
        f"   O/M range over windows {', '.join(w['windows'])} (at the job's f): "
        + "; ".join(f"{q} {r[0]:.2f}-{r[1]:.2f}" for q, r in w.items() if q != "windows")
    )
    return L


def channels() -> dict:
    """Each channel's factor with Birge-widened 95% intervals, one factor across targets, two
    channels fitted together, and the spreads 'plausible' is measured against (channels.py)."""
    return take("channels.json")


def channels_digest(c: dict) -> list[str]:
    def k_of(e: dict) -> str:
        if "k" not in e:
            return f"{e['bound']} ({e['beyond_the_grid']}/{e['curves']})"
        return f"{e['k']:.2f} ({e['ci95'][0]:.2f}-{e['ci95'][1]:.2f})"

    L = ["25. One channel, one factor; two channels (channels.py; binomial; 95%, Birge-widened)"]
    for bd, blk in c["factors"].items():
        L.append(
            f"   {bd}, eps / detection: "
            + "; ".join(
                f"{ch} {k_of(v['eps'])} / {k_of(v['detection'])}"
                for ch, v in blk.items()
                if not ch.startswith("all")
            )
        )
    for ch, blk in c["one factor"].items():
        parts = []
        for name, v in blk.items():
            if "chi2" in v:
                parts.append(
                    f"{name}: k {v['k']:.2f}, chi2 {v['chi2']:.1f}/{v['dof']}, p {v['p']:.2g}"
                )
            elif "z" in v:
                parts.append(f"{name}: Z/X {v['Z / X']:.2f}, z {v['z']:.1f}")
            elif "verdict" in v:
                parts.append(f"{name}: {v['verdict'].split(':')[0]}")
        L.append(f"   one factor on {ch}: " + "; ".join(parts))
    zk, x0, x1 = (
        "Z's factor",
        "X model / device at dephasing x0",
        "X model / device at dephasing x1",
    )
    for pair, blk in c["two channels"].items():
        at = blk["X eps with the factor Z eps needs"]
        L.append(
            f"   {pair}: X eps model/device at Z's factor, no pure dephasing / at f: "
            + "; ".join(f"{d} (x{v[zk]:.2f}) {v[x0]:.2f} / {v[x1]:.2f}" for d, v in at.items())
        )
        for name in ("eps, d3 and d5", "eps and detection, d3 and d5"):
            v = blk[name]
            ca = v["first channel factor ci95 (within the box)"]
            cb = v["dephasing factor ci95 (within the box)"]
            edge = v["at the edge of the simulated box"]
            L.append(
                f"     {name}: {v['first channel factor']:.2f} ({ca[0]:.2f}-{ca[1]:.2f}), "
                f"dephasing {v['dephasing factor']:.2f} ({cb[0]:.2f}-{cb[1]:.2f}), "
                f"chi2 {v['chi2']:.1f}/{v['dof']}, p {v['p']:.2g}"
                + (f", at the edge: {', '.join(edge)}" if edge else "")
            )
    day, chip = (
        "patches' geometric mean, largest day-to-day ratio",
        "chip 90th percentile / median, range over the days",
    )
    L.append(
        "   IBM's spread (patches' mean, largest day-to-day ratio; chip p90/median): "
        + "; ".join(
            f"{q} {v[day]:.2f}; {v[chip][0]:.1f}-{v[chip][1]:.1f}"
            for q, v in c["spread"]["IBM's reported values"].items()
        )
    )
    f = c["spread"]["idle test f"]
    rel = f["relative error of a job's f (bootstrap), range"]
    L.append(
        f"   idle test f: median {f['f median over the jobs']:.2f}, a job's relative error "
        f"{100 * rel[0]:.0f}-{100 * rel[1]:.0f}%, of a common factor "
        f"{100 * f['relative error of a factor common to the 15 jobs']:.1f}%"
    )
    p = c["plausible"]
    L.append(
        "   plausible: factor <= chip p90/median ("
        + ", ".join(f"{ch} {b:.2f}" for ch, b in p["bound"].items())
        + "); eps/detection: "
        + "; ".join(
            f"{bd} " + ", ".join(f"{ch} {v['eps']}/{v['detection']}" for ch, v in p[bd].items())
            for bd in c["factors"]
        )
    )
    return L


def stored_pl8() -> dict:
    """P_L(8) of each patch as stored before its job at f = 0 and f = 1 (Sec. discussion)."""
    out = {}
    for job in jobs():
        INPUTS.add(f"runs/{job.folder.name}/prediction.json")
        stored = read(job.folder / "prediction.json")
        out[job.name] = {
            f"f = {f}": {
                b: dict(
                    zip(
                        (f"d{d}" for d in job.info["chips"]),
                        stored[f]["settings"][memory_index(b, 8)]["logical_error"],
                    )
                )
                for b in BASES
            }
            for f in stored
        }
    return out


def timing() -> dict:
    """One round as compiled (Fig. timing), without the drawn qubits' segment lists."""
    t = take("timing.json")
    return {k: v for k, v in t.items() if k != "patches"} | {
        "patches": {
            p: {k: v for k, v in body.items() if k != "drawn"}
            | {
                "drawn_us": {
                    role: {"qubit": d["qubit"], **d["total_us"]}
                    for role, d in body["drawn"].items()
                }
            }
            for p, body in t["patches"].items()
        }
    }


def decoder_graph() -> dict:
    """Stim's observable-only pieces, single faults on a uniform chip and the shortest logical
    in edges (decoder_graph.py)."""
    g = take("decoder_graph.json")
    s = g["summary"]
    return {
        "pymatching_behaviour": g["pymatching_behaviour"],
        "pieces": {k: v for k, v in s["pieces"].items() if k != "per_round_8_rounds"},
        "uniform_chip": {
            day: {k: v for k, v in row.items() if "examples" not in k}
            for day, row in s["uniform_chip"].items()
        },
        "path": {k: v for k, v in s["path"].items() if "example" not in k},
    }


def other() -> dict:
    out = {
        "readout": take("readout.json"),
        "uniform_scale": take("uniform_scale.json"),
        "no_reset": take("no_reset.json"),
        "timing": timing(),
        "day_swing (placement 1, reported values set to the other days' median)": take(
            "day_swing.json"
        ),
        "repredict (a stored prediction derived again)": take("repredict.json"),
        "stored P_L(8) by job, f, basis and patch": stored_pl8(),
    }
    INPUTS.add("docs/figures/figure3.json")
    out["prerun_figure3_rows"] = read(REPO / "docs/figures/figure3.json")["rows"]
    if (RESULTS / "power.json").exists():
        out["power"] = take("power.json")
    return out


# ---------------------------------------------------------------------------- digest
def fmt_pm(v, s, d=3):
    return f"{v:.{d}f} +- {s:.{d}f}"


def digest(res: dict) -> str:
    L = []
    pp = res["per_placement"]
    L.append(
        "1. Per placement over 5 days (eps: plain mean +- se over days, then the inverse-variance mean;"
        " Lambda: inverse-variance mean; chi2/nu with nu = 4; ratio = plain observed / plain mean predicted)"
    )
    for b in BASES:
        for pl in PLACEMENTS:
            for fit in FITS:
                blk = pp[b][f"placement {pl}"][fit]
                parts = []
                for q in QUANTITIES:
                    o = blk[q]["observed"]
                    scale, d = (100, 1) if q != "lambda" else (1, 3)
                    plain = ""
                    if q != "lambda":
                        op = blk[q]["observed_plain"]
                        plain = (
                            f"{fmt_pm(scale * op['value'], scale * op['se_over_days'], d)}; ivw "
                        )
                    parts.append(
                        f"{q} {plain}{fmt_pm(scale * o['value'], scale * o['se'], d)} (chi2/nu {o['chi2_per_dof']:.2f}; "
                        f"pred {scale * blk[q]['predicted_mean']:.{d}f}; ratio {blk[q]['observed_over_predicted']:.3f}; "
                        f"agrees {blk[q]['days_model_agrees']}/5)"
                    )
                L.append(f"   {b} placement {pl} {fit:8s}: " + "; ".join(parts))
    for k, v in pp["observed_over_predicted_range_eps3_eps5_both_placements"].items():
        L.append(f"   observed/predicted eps3, eps5 range, {k}: {v[0]:.2f}-{v[1]:.2f}")
    cells = res["effect_sizes"]["table_one_convention"]["cells"]
    for b in BASES:
        L.append(
            f"   Table placement, predicted mean +- sigma_M (WL), {b}: "
            + "; ".join(
                f"P{pl} "
                + ", ".join(
                    f"{q} {(100 if q != 'lambda' else 1) * c['mean']:.{1 if q != 'lambda' else 3}f} +- "
                    f"{(100 if q != 'lambda' else 1) * c['sigma_M']:.{1 if q != 'lambda' else 3}f}"
                    for q in QUANTITIES
                    for c in [cells[f"{b} placement {pl} wls {q}"]["predicted"]]
                )
                for pl in PLACEMENTS
            )
        )
    pc = res["placements_compared"]
    L.append("2. Placements compared")
    for b in BASES:
        for fit in FITS:
            v = pc[f"Lambda_{b} {fit}"]
            L.append(
                f"   Lambda_{b} {fit}: P1 - P2 = {fmt_pm(v['placement_1_minus_placement_2'], v['se'])}"
            )
    ov = pc["overlap"]
    L.append(
        f"   d5 patches share {ov['d5_shared']} of {ov['d5_qubits_each']} qubits; placement-2 d3 inside "
        f"placement-1 d5: {ov['placement_2_d3_inside_placement_1_d5']}"
    )
    pr = res["per_repetition"]
    L.append("3. Per repetition")
    L.append(
        f"   Lambda_X range wls {pr['Lambda_X_range']['wls'][0]:.3f}-{pr['Lambda_X_range']['wls'][1]:.3f}, "
        f"binomial {pr['Lambda_X_range']['binomial'][0]:.3f}-{pr['Lambda_X_range']['binomial'][1]:.3f}; "
        f"verdicts {pr['verdicts']}"
    )
    L.append(
        f"   binomial lowers Lambda_X in {pr['binomial_lowers_Lambda_X_in']} of 10 (max rise {pr['binomial_raises_Lambda_X_max']:.4f}); "
        f"eps5 shift (pp) {pr['eps5_binomial_minus_wls_pp']}"
    )
    rec = res["prespecified_pooled_record"]["prespecified"]
    L.append("4. Prespecified summary value: Lambda_X pooled over the 10 repetitions")
    L.append(
        f"   Lambda_X {fmt_pm(rec['Lambda_X'], rec['se'])}, chi2/nu {rec['chi2_per_dof']:.2f} (nu 9): {rec['verdict']}; "
        f"predicted {rec['predicted_inverse_variance_weighted']:.3f}"
    )
    L.append("5. Agreement tests (chi2 vs 18.31; reps agreeing)")
    for fit in FITS:
        a = res["agreement_tests"][fit]
        L.append(
            f"   {fit}: "
            + ", ".join(
                f"{q} {a[q]['chi2']:.1f} ({a[q]['reps_agreeing']})"
                for q in ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5")
            )
            + f"; all three X: {a['reps_agreeing_on_all_three_X']}"
        )
    fb = res["fit_bias"]
    L.append(
        "6. Weighted-log bias (bootstrap, pp): "
        + ", ".join(
            f"{k} {fb[k]['wls_bias_pp_mean']:+.2f} (shift {fb[k]['observed_shift_binomial_minus_wls_pp_mean']:+.2f})"
            for k in ("X eps3", "X eps5", "Z eps3", "Z eps5")
        )
    )
    c = res["cross_pairing"]
    L.append(
        f"7. Cross pairing (mean): wls {c['wls']['mean_rep2_eps3_over_rep1_eps5']:.3f} vs own "
        f"{c['wls']['mean_rep2_own']:.3f}; binomial {c['binomial']['mean_rep2_eps3_over_rep1_eps5']:.3f} vs own "
        f"{c['binomial']['mean_rep2_own']:.3f}"
    )
    m = res["model_vs_device"]
    d = m["prespecified_drift_check"]
    L.append(
        f"8. Prespecified drift check: {d['comparisons']} comparisons, max |change| {d['largest_abs_change_in_se']:.2f} SE, "
        f"second half worse {d['second_half_worse']}/{d['comparisons']}"
    )
    cs = m["calibration_scatter_check_exploratory"]
    L.append(
        "   calibration-scatter check (exploratory): "
        + ", ".join(
            f"{q} chi2 {cs[q]['chi2']:.1f}->{cs[q]['chi2_with_scatter']:.1f} (scatter {100 * cs[q]['relative_scatter']:.0f}%)"
            for q in ("X lambda", "X eps3", "X eps5")
        )
    )
    de = m["detectors_beyond_chance"]
    L.append(
        f"   detectors beyond chance (two-patch jobs) {de['two_patch_beyond_range'][0]}-{de['two_patch_beyond_range'][1]} "
        f"of {de['two_patch_detectors'][0]}; largest ratio {de['largest_ratio']['ratio']:.1f}; excess slope "
        f"{m['excess_slope']['above_3_se [cases, of]']['all']}; f from memory agrees in {m['f']['from_memory_agrees_in']}/15"
    )
    xy = res["xy_variance_correction"]
    q = xy["per_qubit"]
    L.append("9. XY-variance correction")
    L.append(
        f"   excluded {q['old']['excluded']} -> {q['new']['excluded']} of {q['data_qubits']}; sigma_f new/old median "
        f"{q['sigma_f_new_over_old']['median']:.3f}; IQR {q['new']['iqr'][0]:.2f}-{q['new']['iqr'][1]:.2f}"
    )
    lj = xy["largest_job_change"]
    L.append(
        f"   largest job change {lj['job']}: median f {lj['median_old']:.3f} -> {lj['median_new']:.3f}; "
        f"verdict changes {xy['downstream']['verdict_changes']}"
    )
    pq = res["per_qubit_f_side_prediction"]
    L.append(
        "   per-qubit f (side prediction): chi2 "
        + ", ".join(f"{q} {pq['chi2'][q]:.1f}" for q in ("X lambda", "X eps3", "X eps5"))
        + "; predicted Lambda_X "
        + ", ".join(
            f"{k} {v['median_f']:.3f}->{v['per_qubit_f']:.3f}"
            for k, v in pq["predicted_Lambda_X_mean"].items()
        )
    )
    L += f_models_digest(res["f_models"])
    h = res["heating_exploratory"]
    det = h["detection_events"]
    L.append("10. Heating (exploratory; fractional change start->end of a job)")
    L.append(
        f"   detection events DL {100 * det['random_effects_DL']['change']:+.2f}% "
        f"[{100 * det['random_effects_DL']['ci95'][0]:+.2f}, {100 * det['random_effects_DL']['ci95'][1]:+.2f}]; "
        f"rise excluded above (DL/HKSJ/t) "
        + "/".join(
            f"{100 * v:.2f}%"
            for v in h["detection_events_rise_excluded_above (95% upper ends)"].values()
        )
    )
    L.append(
        f"   step q{h['step']['qubit']} at shot {h['step']['switch_shot_index']} in {h['step']['pieces_lower_after']}/{h['step']['pieces']} pieces"
    )
    dc = res["decoder_audit"]
    hw = dc["hardware"]["pooled_prespecified_rule"]
    L.append("11. Decoder")
    L.append(
        f"   Lambda_X calibrated {fmt_pm(hw['calibrated X']['lambda'], hw['calibrated X']['lambda_uncertainty'])}, uniform "
        f"{fmt_pm(hw['uniform X']['lambda'], hw['uniform X']['lambda_uncertainty'])}; edge p "
        f"{dc['matching_graph']['edge_probability_range'][0]:.4f}-{dc['matching_graph']['edge_probability_range'][1]:.2f}"
    )
    lk = res["leakage"]
    hm = lk["hmm"]["chip"]
    L.append("12. Leakage")
    L.append(
        f"   fires in {100 * lk['refiring']['base_odds']['chip']['fires_per_round']:.1f}% of rounds (model "
        f"{100 * lk['refiring']['base_odds']['model']['fires_per_round']:.1f}%); HMM onset {100 * hm['a_median']:.1f}%, "
        f"lifetime {hm['lifetime_us_median']:.0f} us, e_L {hm['e_L_median']:.2f}"
    )
    on = lk["causes"]["onset_line"]
    L.append(
        f"   onset per round = {100 * on['per_cz']:.2f}% per CZ x CZs + {100 * on['per_round']:.2f}% (two CZ counts only)"
    )
    se = res["streak_effects_exploratory"]
    for b in "XZ":
        near = {
            lag: se["detection_events"][f"{b} dboth both"][f"lag {lag}"]["near"]["streak - quiet"]
            for lag in (-2, 2)
        }
        lf = se["logical_failure"][f"{b} dboth"]
        L.append(
            f"   {b} memory, near stabilizers before/after a streak (lag -2 / 2): "
            + ", ".join(
                f"{100 * v['chip']:+.1f} pts (model {100 * v['model']:+.1f}; q {v['chip_flip_equivalent']:.2f})"
                for v in near.values()
            )
            + f"; failure RD {100 * lf['any']['chip']['rd']:+.1f} pts (model {100 * lf['any']['model']['rd']:+.1f}),"
            f" at matched firings {100 * lf['count matched']['chip']['rd']:+.1f}; per firing "
            f"{100 * lf['failure_per_firing']['chip']['slope']:.2f} (model {100 * lf['failure_per_firing']['model']['slope']:.2f})"
        )
    rs = res["readout"]["resets"]
    cm = rs["calibration_model_with_ibm_errors"]
    L.append("13. IBM's reset and readout errors")
    L.append(
        "   init_error / the model's reset error, medians by day: "
        + ", ".join(
            f"{v['ratio_of_medians']:.2f}"
            for v in rs["init_error_against_model_reset_error"].values()
        )
    )
    L.append(
        "   in the calibration model, change in Z eps (pp, mean over patches): "
        + "; ".join(
            f"d{d} {fmt_pm(100 * cm[f'Z d{d}']['mean_change'], 100 * cm[f'Z d{d}']['se_over_patches'], 2)}"
            for d in "35"
        )
    )
    for b in BASES:
        j = cm[f"{b} detectors, 8 rounds, jobs"]
        L.append(
            f"   {b} memory, jobs of {j['all']}: model's mean detector rate below the device's "
            f"{j['model_mean_below_device']}, lowered further {j['ibm_mean_below_model']}, "
            f"per-detector deviation larger {j['ibm_farther_from_device']}"
        )
    if "power" in res:
        rows = [r for f in res["power"]["f"].values() for k, r in f.items() if k.endswith("reps")]

        def span(xs):
            return f"{100 * min(xs):.1f}-{100 * max(xs):.1f}%"

        L.append("14. Operating characteristics (power.py; 5 and 10 reps, f = 0 and 1)")
        L.append(
            "   Lambda = 1 called, scatter 1x/2x/4x: "
            + ", ".join(
                span([r["called_at_lambda_1"][f"scatter {s}"] for r in rows]) for s in (1, 2, 4)
            )
            + f"; X tests fail when right {span([r['x_tests_fail_when_right'] for r in rows])}; "
            f"gap 0.06 caught {span([r['gap_caught_80pct']['caught'] for r in rows])}"
        )
    h = res["effect_sizes"]["headline"]
    L.append(
        "15. Effect sizes (effect_sizes.py; binomial O/M, 95%: placements t4, all ten DL-HKSJ)"
    )
    for q, v in h["O_over_M (binomial) [ratio, 95% low, high]"].items():
        L.append(
            f"   {q:9s} "
            + "; ".join(
                f"{k} {v[k][0]:.3f} ({v[k][1]:.3f}-{v[k][2]:.3f})"
                for k in ("placement 1", "placement 2", "all ten")
            )
            + f"; tau {v['tau']:.3f}"
        )
    c = h["prespecified chi2 (WL) [chi2, p, Holm p over the three X tests]"]
    bs = h["bias and scatter (WL) [bias chi2, p, scatter chi2, p]"]
    L.append(
        "   prespecified chi2 (p, Holm; bias, scatter): "
        + ", ".join(
            f"{q} {c[q][0]:.1f} ({c[q][1]:.2g}, {c[q][2]:.2g}; {bs[q][0]:.1f}, {bs[q][2]:.1f})"
            for q in ("X eps3", "X eps5", "X lambda")
        )
    )
    t = h["tracking, placement 1 Lambda_X"]
    pw = h[
        "power (WL bounds) [Lambda_X at 0.06, Lambda_X found at 80%, eps3 and eps5 relative at 80%]"
    ]
    L.append(
        "   placement 1 Lambda_X, constant vs follows the prediction (4 dof): "
        + ", ".join(
            f"{fit} {v['constant'][0]:.1f} vs {v['follows_prediction'][0]:.1f} (r {v['pearson_r']:+.2f}; "
            f"without day 2 {v['without day 2 (3 dof)'][0]:.1f})"
            for fit, v in t.items()
        )
        + f"; power at 0.06 {pw[0]:.2f}, 80% at {pw[1]:.3f} (eps {100 * pw[2]:.0f}%, {100 * pw[3]:.0f}%)"
    )
    pi = res["placements_and_calibration_inputs"]
    d3 = pi["d3"]
    L.append("16. Placements (searched / all embeddings) and calibration inputs")
    L.append(
        f"   d3 sets {d3['searched_sets']} / {d3['all_sets']} ({d3['embeddings']} embeddings); usable by day "
        + ", ".join(f"{a}/{b}" for a, b in d3["usable (searched, all) by day"].values())
        + "; rep-2 d3 rank among all usable "
        + ", ".join(
            str(v["rank_among_all_usable"])
            for v in d3["rank of repetition 2's d3 patch among all usable, by day"].values()
        )
    )
    L.append(
        "   d5 stricter bars passing (day 1): "
        + str(next(iter(pi["d5"]["usable and stricter bars by day"].values())))
    )
    L.append(
        "   q51 readout by day "
        + ", ".join(f"{100 * v['reported_readout_error']:.1f}%" for v in pi["relay_q51"].values())
        + f"; readout gap P(0|1)-P(1|0) median {100 * pi['readout_gap_P01_minus_P10_flags_relays']['median']:.2f} points"
    )
    sq, ages = pi["single_qubit_error_median_by_day"], pi["calibration_age_by_day"]
    L.append(
        "   SX pair noise / CZ noise by day "
        + ", ".join(f"{v['sx_pair_noise_over_cz_noise']:.3f}" for v in sq.values())
        + "; snapshot in force / previous, minutes old at the first job "
        + ", ".join(
            f"{a['in_force_minutes_old']:.1f}/{a['previous_minutes_old']:.1f}"
            for a in ages.values()
        )
        + "; equal to previous (T1, T2, CZ, readout) "
        + ", ".join(
            "/".join(
                f"{e[0]}of{e[1]}"
                for e in (
                    a["equal_to_previous [equal, compared]"][k]
                    for k in ("t1", "t2", "cz", "readout")
                )
            )
            for a in ages.values()
        )
    )
    rec = pi["recurring_weak_parts_of_placement_1_by_day"]
    counts = pi["weak_parts_by_day (placement.weak)"]
    r = pi["round_us"]["range"]

    def span(kind: str, part, scale: float = 100.0) -> str:
        values = [scale * day[kind][str(part)] for day in rec.values()]
        return f"{min(values):.1f}-{max(values):.1f}"

    L.append(
        "   weak parts by day (placement.weak), placement 1 / placement 2: "
        + ", ".join(f"{c['placement 1 (d5)']}/{c['placement 2 (d5)']}" for c in counts.values())
        + "; placement 1 every day: CZ 63-64 "
        + span("cz", "63-64")
        + "%, 102-103 "
        + span("cz", "102-103")
        + "%; readout q83 "
        + span("readout", 83)
        + "%, q102 "
        + span("readout", 102)
        + "%; T2 q149 "
        + span("t2_us", 149, 1.0)
        + " us, q150 "
        + span("t2_us", 150, 1.0)
        + f" us; round {r[0]:.3f}-{r[1]:.3f} us"
    )
    loc = pi["pair_2_d3_location_by_day"]
    L.append(
        "   pair 2's d3 patch by day: inside placement 1 "
        + ", ".join(str(v["inside placement 1"]) for v in loc.values())
        + "; shares with placement 2 "
        + ", ".join(f"{v['shared with placement 2']}/{v['qubits']}" for v in loc.values())
        + "; day 1's qubits "
        + ", ".join(str(v["same qubits as day 1"]) for v in loc.values())
    )
    tm = res["timing"]
    roles = tm["patches"]["3"]["by_role"]
    L.append(
        f"   Fig. timing (day 1, pair 1, d3): round {tm['round_us']:.3f} us, readout and reset "
        f"{tm['readout_and_reset_us']:.3f} us ({100 * tm['readout_and_reset_share_of_round']:.0f}%)"
        f"; path "
        + " ".join(f"{k} {v:.3f}" for k, v in tm["critical_path_us"])
        + f"; zero-duration resets {tm['round_us_zero_duration_resets']:.3f} us"
        + "; data idle (any, decoupled) "
        + "-".join(f"{100 * roles['data']['share_idle_any'][m]:.1f}" for m in ("min", "max"))
        + "%, "
        + "-".join(
            f"{100 * roles['data']['share_of_round']['idle_decoupled'][m]:.1f}"
            for m in ("min", "max")
        )
        + "%; flags in readout or reset "
        + f"{100 * roles['flag']['share_readout_and_reset']['median']:.1f}%"
    )
    dg = res["diagnostics"]
    pdet = dg["detection_event_fraction"]["per_detector (device / model)"]
    L.append("17. Diagnostics added after the run")
    L.append(
        "   detection-event fraction device/model, median (share above 1): "
        + ", ".join(
            f"{k} {v['median_ratio']:.2f} ({100 * v['share_above_1']:.0f}%)"
            for k, v in pdet.items()
        )
    )
    t = dg["idle_f_against_T2"]["spearman_f_t2"]
    m = dg["f_memory_against_idle"]
    L.append(
        f"   f vs T2: rho {t['rho']:.2f} [{t['ci95_bootstrap_over_qubits'][0]:.2f}, {t['ci95_bootstrap_over_qubits'][1]:.2f}]; "
        f"f_mem > f_idle in {m['above']}/{m['of']} (medians {m['median_f_mem']:.2f}, {m['median_f_idle']:.2f}); "
        f"beyond the simulated f in {m['beyond_simulated_f']}/{m['of']}, at the search edge 2.0 in {m['at_search_edge_2.0']}"
    )
    ms = dg["miscorrected_single_faults_X_d3"]["share_of_predicted_eps3_range"]
    q44 = dg["miscorrected_single_faults_X_d3"]["largest_by_patch"]["2026-10-04-r1"]
    L.append(
        f"   miscorrected single faults: {100 * ms[0]:.0f}-{100 * ms[1]:.0f}% of predicted eps3; "
        f"day 1 pair 1: {q44['fault']} {100 * q44['p_per_round']:.1f}% per round over "
        f"{q44['mechanisms']} mechanisms, largest single {q44['largest_single_mechanism_p']:.4f}, "
        f"{100 * q44['share_of_miscorrected']:.0f}% of the miscorrected; its own edge "
        + ", ".join(f"{p:.4f}" for p in q44["largest_single_mechanism_edges_p"])
        + f" (weight {weight(q44['largest_single_mechanism_edges_p']):.2f}); matching chose edges "
        + ", ".join(f"{p:.4f}" for p in q44["decoder_chose_edges_p"])
        + f" (weight {weight(q44['decoder_chose_edges_p']):.2f})"
    )
    ml = res["decoder_ml"]
    drop, shift, n4 = (
        ml["model_eps3_drop_with_ml_range"],
        ml["device_eps3_change_pp_range"],
        ml["n4_miscorrected_single_fault_p"],
    )
    L.append(
        f"   exact maximum likelihood (d = 3 X memory, n = 1-4): model eps3 {100 * drop[0]:.1f}-"
        f"{100 * drop[1]:.1f}% lower than with matching; device eps3 {shift[0]:+.2f} to "
        f"{shift[1]:+.2f} pp (mean {ml['device_eps3_change_pp_mean']:+.2f}); of the single-fault "
        f"probability matching miscorrects at n = 4 ({n4[0]:.3f}), ML also misassigns {n4[1]:.3f} "
        f"({100 * n4[2]:.0f}%)"
    )
    gr = res["decoder_graph"]
    pc = gr["pieces"]
    sh, alone = pc["shots_decoded_differently"], pc["faults_decoded_alone"]
    scan = pc["z_d3_single_scan (decoder_faults.py's, repeated)"]
    lo, hi = pc["fault_probability_range"]
    L.append(
        f"   observable-only pieces: {pc['count_by_basis_and_patch']} (one per round, p "
        f"{lo:.4f}-{hi:.4f}); graph unchanged without them "
        f"{pc['graph_unchanged_without_pieces_every_circuit']} and folded into any piece "
        f"{pc['graph_unchanged_folded_into_any_piece_every_circuit']}; shots decoded differently "
        f"{sh['device matching_folded_vs_paper'] + sh['device correlated_folded_vs_dropped']} of "
        f"{sh['device shots']} device, {sh['model matching_folded_vs_paper'] + sh['model correlated_folded_vs_dropped']}"
        f" of {sh['model shots']} model; alone, calibrated wrong on {alone['calibrated_wrong']} "
        f"({scan['calibrated_wrong_with_observable_only_piece']} of the {scan['calibrated_wrong']} Z d3 "
        f"single miscorrections), uniform on {alone['uniform_wrong']}; up to "
        f"{100 * pc['share_of_predicted_Z_eps3_range'][1]:.1f}% of predicted Z eps3"
    )
    for day, u in gr["uniform_chip"].items():
        x3 = u["X d3 per-job model"]
        L.append(
            f"   uniform chip {day} (f = 1): X d3 single faults miscorrected {x3['n=2'][0]} of "
            f"{x3['n=2'][1]} at n = 2 ({x3['n=2 by symptom'][0]} of {x3['n=2 by symptom'][1]} "
            f"symptoms), {x3['n=8'][0]} of {x3['n=8'][1]} at n = 8 (f = 0: "
            f"{u['X d3 pre-run simulator f=0']['n=8'][0]}, no idle decay: "
            f"{u['X d3 pre-run simulator f=1 no idle']['n=8'][0]}), "
            f"{100 * x3['share_of_eps3']:.0f}% of eps3; boundary p up to {x3['boundary_p'][1]:.3f}; "
            f"d = 5: {u['X d5 per-job model']['n=8'][0]}"
        )
    L.append(
        "   shortest logical in edges: "
        + ", ".join(f"{k} {v}" for k, v in gr["path"]["shortest_logical_edges_seen"].items())
    )
    scope = res["idle_test_scope"]
    ones, left = scope["ancilla_reads_1"], scope["f_left_out_of_the_median"]
    L.append(
        f"   idle test: ancillas read 1 in {100 * ones['idle_range'][0]:.1f}-"
        f"{100 * ones['idle_range'][1]:.1f}% (memory {100 * ones['memory_range'][0]:.1f}-"
        f"{100 * ones['memory_range'][1]:.1f}%); {left['qubit_jobs']} qubit-jobs left out of the "
        f"median f, median T2 {left['median_t2_us']:.0f} us (used: {left['used_median_t2_us']:.0f} us)"
    )
    for kind, ways in res["fit_bias_weights"]["bias_pp"].items():
        L.append(
            f"   log-linear bias of eps, {kind}: planned weights {ways['planned']['mean']:+.2f} pp, "
            f"weights at the true P {ways['true_weights']['mean']:+.2f} pp"
        )
    L += sensitivity_digest(res["sensitivity"])
    L += dependence_digest(res["dependence"])
    L += scale_and_f_digest(res["error_scale_and_f"])
    L += prespecified_tests_digest(res["prespecified_tests"])
    L += error_budget_digest(res["error_budget"])
    L += stat_checks_digest(res["stat_checks"])
    L += input_noise_digest(res["input_noise"])
    L += channels_digest(res["channels"])
    L += f_treatments_digest(res["f_treatments"])
    return "\n".join(L) + "\n"


def prespecified_tests_digest(t: dict) -> list[str]:
    labels = ("prespecified", "idle_noise_only", "no_f_term")
    L = [
        "21. The tests fixed before the run, both sigma_f (prespecified_tests.py; WL unless named)"
    ]
    L.append("   X tests, chi2 p (|z| < 2): sigma_f prespecified / idle noise only / no f term")
    for fit in FITS:
        c = t["chi2"][fit]
        for q in ("X lambda", "X eps3", "X eps5"):
            L.append(
                f"     {fit:8s} {q}: "
                + " / ".join(
                    f"{c[x][q]['chi2']:.1f} {c[x][q]['p']:.2g} ({c[x][q]['abs_z_below_2']})"
                    for x in labels
                )
                + "; Holm p "
                + " / ".join(f"{c[x][q]['p_holm_over_the_three_X_tests']:.2g}" for x in labels[:2])
            )
        L.append(
            f"     {fit:8s} per pair (5 dof), chi2 p, prespecified -> idle noise: "
            + "; ".join(
                f"{q} P{p} "
                + " -> ".join(
                    f"{c[x][q]['per_pair'][f'pair {p}']['chi2']:.1f} "
                    f"{c[x][q]['per_pair'][f'pair {p}']['p']:.2g}"
                    for x in labels[:2]
                )
                for q in ("X lambda", "X eps3", "X eps5")
                for p in PLACEMENTS
            )
        )
    z = t["chi2"]["wls"]["Z memory (computed, not decided; prespecified sigma_f)"]
    L.append(
        "   Z memory (computed, not decided): "
        + ", ".join(f"{q} {v['chi2']:.1f} (p {v['p']:.2g})" for q, v in z.items())
    )
    pl = t["pool"]
    pre = pl["prespecified"]
    within = [pl["after the run"][f"pair {p} (WL, same rule)"] for p in PLACEMENTS]
    L.append(
        f"   pool: {pre['statement']} (t_9,0.95; chi2_0/nu {pre['chi2_0_per_dof']:.2f}, "
        f"unwidened se {pre['se_unwidened']:.4f}); predicted {pre['predicted (same weights)']:.3f}, "
        f"O/M {pre['observed_over_predicted']:.3f}; per pair-day verdicts "
        f"{pl['per_pair_day_verdicts (Lambda_X + 1.645 sigma_O < 1)']} (largest Lambda + 1.645 "
        f"sigma {pl['largest Lambda_X + 1.645 sigma_O']:.3f}); within pairs chi2_0/nu "
        + ", ".join(f"{w['chi2_0_per_dof']:.2f}" for w in within)
    )
    sf = t["sigma_f"]
    od = sf["per_qubit_fits_overdispersed"]
    L.append(
        f"   sigma_f prespecified {sf['prespecified_range'][0]:.3f}-{sf['prespecified_range'][1]:.3f}, "
        f"idle noise {sf['idle_noise_range'][0]:.3f}-{sf['idle_noise_range'][1]:.3f}; per-qubit idle "
        f"fits: median reduced chi2 {od['median_reduced_chi2']:.2f} over {od['fits_with_dof']}, "
        f"p < 0.05 in {100 * od['share_p_below_0.05']:.0f}%, p < 0.001 in "
        f"{100 * od['share_p_below_0.001']:.0f}%; uniformity chi2/nu "
        f"{od['uniformity_chi2_per_dof_range'][0]:.1f}-{od['uniformity_chi2_per_dof_range'][1]:.1f}, "
        f"with scaled errors {od['with_scaled_errors_range'][0]:.1f}-{od['with_scaled_errors_range'][1]:.1f}"
    )
    w = t["windows"]
    sys_key = "systematic (half range, rounds 3-8 left out)"
    drop = w["leaving out round 1 (rounds 2-8 minus 1-8)"]["wls"]["all ten"]
    L.append(
        "   fit-window systematic on Lambda_X, device / model (device with rounds 3-8): "
        + "; ".join(
            f"P{p} +-{w[f'pair {p}']['device'][sys_key]:.3f} / +-{w[f'pair {p}']['model'][sys_key]:.3f} "
            f"(+-{w[f'pair {p}']['device']['half range with rounds 3-8']:.3f})"
            for p in PLACEMENTS
        )
        + f"; leaving out round 1: device {drop['device']:+.3f} +- "
        f"{drop['device_se_over_repetitions']:.3f}, model {drop['model']:+.3f}"
    )
    L.append(
        "   pairs' table, errors widened alike (observed / predicted [chi2_0/4 of the predictions])"
    )
    for k, c in t["table_pairs"]["cells"].items():
        L.append(
            f"     {k}: {c['observed']['shown']} / {c['predicted']['shown']} "
            f"[{c['predicted']['chi2_per_dof']:.2f}], range "
            + "-".join(
                f"{(1 if k.endswith('lambda') else 100) * x:.{3 if k.endswith('lambda') else 1}f}"
                for x in c["predicted"]["range"]
            )
        )
    pw = t["power"]
    L.append(
        "   power at a Lambda_X miss of 0.06: one pair-day P1 / P2, a pair's chi2 P1 / P2, all ten; "
        "80% at"
    )
    for x in labels:
        lx = pw[x]["Lambda_X"]
        one = [lx[f"pair {p}"]["power_one_pair_day_range"] for p in PLACEMENTS]
        L.append(
            f"     {x}: {one[0][0]:.2f}-{one[0][1]:.2f} / {one[1][0]:.2f}-{one[1][1]:.2f}, "
            + " / ".join(f"{lx[f'pair {p}']['power_chi2_5dof']:.2f}" for p in PLACEMENTS)
            + f", {lx['power_chi2_10dof']:.2f}; 80% at {lx['miss_found_at_80pct_chi2_10dof']:.3f} "
            f"(median sigma {lx['median_sigma']:.3f}); eps3, eps5 relative at 80% "
            + ", ".join(
                f"{pw[x][q]['relative_miss_found_at_80pct_chi2_10dof']:.3f}"
                for q in ("X eps3", "X eps5")
            )
        )
    ds = next(v for k, v in pw.items() if k.startswith("design study"))
    L.append(
        "   design study (f known): "
        + "; ".join(
            f"{f}, 10 pair-days: sigma {v['10 reps']['sigma_one_pair_day']:.3f}, caught "
            f"{v['10 reps']['caught']:.3f}"
            for f, v in ds.items()
        )
    )
    L.append(
        "   per pair-day z (WL; X: prespecified [idle noise]): Lambda_X eps3 eps5 | Lambda_Z eps3 eps5"
    )
    for r in t["per_pair_day"]:
        c = r["wls"]
        L.append(
            f"     day {r['day']} pair {r['pair']}: "
            + " ".join(
                f"{c[q]['z']:+.2f} [{c[q]['z_idle_noise_only']:+.2f}]"
                for q in ("X lambda", "X eps3", "X eps5")
            )
            + " | "
            + " ".join(f"{c[q]['z']:+.2f}" for q in ("Z lambda", "Z eps3", "Z eps5"))
            + f"; Lambda_X + 1.645 sigma {r['verdict (WL)']['Lambda_X + 1.645 sigma_O']:.3f}"
        )
    return L


def scale_and_f_digest(s: dict) -> list[str]:
    def ci(b: dict) -> str:
        return f"{b['ratio']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]"

    es = s["error_scale"]
    L = ["20. One factor on every error; the dephasing floor; f's summary; readout resolution"]
    for b in BASES:
        eps, det = es["eps"]["binomial"][b], es["detection_rate"][b]
        both = eps["both distances"]
        L.append(
            f"   k_eps {b} (binomial; all ten, P1, P2): "
            + "; ".join(
                f"{d} {ci(eps[d]['all ten'])}, {ci(eps[d]['placement 1'])}, {ci(eps[d]['placement 2'])}"
                for d in ("d3", "d5")
            )
            + f"; both distances k {both['joint']['k']:.2f}, chi2/nu {both['joint']['chi2']:.1f}/{both['joint']['dof']}"
        )
        L.append(
            f"   k_det {b} (all ten, P1, P2): "
            + "; ".join(
                f"{d} {ci(det[d]['all ten'])}, {ci(det[d]['placement 1'])}, {ci(det[d]['placement 2'])}"
                for d in ("d3", "d5")
            )
        )
    fl = s["dephasing_floor"]
    for label in ("per_physical_qubit (medians, least squares)", "per_physical_qubit, T2 >= 20 us"):
        fit = fl[label]
        a, ci = fit["additive"], fit["additive"]["ci95_bootstrap_over_qubits"]
        L.append(
            f"   Gamma_dd = k Gamma_phi + Gamma_0, {label}: k {a['params'][0]:.2f}, Gamma_0 "
            f"{a['params'][1]:.4f}/us (95% {ci['gamma_0_per_us'][0]:.4f}-{ci['gamma_0_per_us'][1]:.4f}); "
            f"F {fit['F_test_floor_over_multiple']['F']:.1f}, p {fit['F_test_floor_over_multiple']['p']:.2g}; "
            f"floor phase flip per round {100 * a['floor_phase_flip_per_round']:.2f}% over "
            f"{fl['decoupled_idle_per_round_us']['median']:.2f} us"
        )
    idle = fl["model_idle_error_per_round"]["range"]
    L.append(
        "   model idle error per data qubit and round, patch medians at the job's f: every Pauli "
        f"{100 * idle['all_paulis'][0]:.1f}-{100 * idle['all_paulis'][1]:.1f}%, Z or Y "
        f"{100 * idle['z_or_y'][0]:.1f}-{100 * idle['z_or_y'][1]:.1f}%"
    )
    fw = s["f_weighted_mean"]["binomial"]["O_over_M"]
    for v in ("median f", "weighted-mean f"):
        L.append(
            f"   O/M (binomial) at the {v}: "
            + "; ".join(
                f"{q} P1 {fw[v][q]['placement 1']['ratio']:.2f}, P2 {fw[v][q]['placement 2']['ratio']:.2f}"
                for q in ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5")
            )
        )
    ir = s["input_resolution"]
    L.append(
        f"   readout error sigma/e (4096 shots per state): median {100 * ir['relative_sigma_median']:.1f}%, "
        f"{100 * ir['relative_sigma_at_1_percent']:.1f}% at e = 1%"
    )
    return L


def sensitivity_digest(s: dict) -> list[str]:
    L = ["18. Sensitivity of the agreement test (sensitivity.py; WL unless named)"]
    r = s["sigma_f_range"]
    L.append(
        f"   sigma_f prespecified {r['prespecified'][0]:.2f}-{r['prespecified'][1]:.2f}, "
        f"idle-test noise only {r['idle_noise_only'][0]:.3f}-{r['idle_noise_only'][1]:.3f}"
    )
    for label, blk in s["x_tests by sigma_f"]["wls"].items():
        L.append(
            f"   {label}: "
            + ", ".join(
                f"{q} {v['chi2']:.1f} ({v['agree']}; P2 {v['per_placement']['placement 2']['chi2']:.1f}, "
                f"k {v['constant_multiple (O = k M, 9 dof)']['chi2']:.1f})"
                for q, v in blk.items()
            )
        )
    mc = s["monte_carlo (WL)"]["prespecified"]
    L.append(
        "   Monte Carlo sd of chi2: "
        + ", ".join(
            f"{q} {v['prespecified_slope']['chi2_redrawn']['sd']:.2f}" for q, v in mc.items()
        )
        + f"; eps5 fails in {100 * mc['X eps5']['prespecified_slope']['fails (chi2 > 18.31)']:.0f}%"
        f", with the line slope {mc['X eps5']['line_slope']['chi2_stored']:.2f}"
    )
    above = s["line_slope_above_f_1.05 (against the secant from f = 1 and the parabola)"]
    name, job = max(above.items(), key=lambda kv: kv[1]["f"])
    q = next(k for k in job if k.startswith("eps"))
    v = job[q]
    L.append(
        f"   line slope at the largest f ({name}, f {job['f']:.2f}, {q}): {v['slope_line']:.4f}, "
        f"{100 * (v['slope_line'] / v['slope_secant_from_1'] - 1):.0f}% above the secant from f = 1 "
        f"({v['slope_secant_from_1']:.4f}), {100 * (v['slope_line'] / v['slope_parabola'] - 1):.0f}% "
        f"above the parabola ({v['slope_parabola']:.4f})"
    )
    sy = s["Lambda_X windows and fits"]["systematic"]
    L.append(
        "   Lambda_X systematic (half range, windows and fits): "
        + ", ".join(
            f"{p} device {v['device']['systematic (half range without 3-8)']:.3f}, model "
            f"{v['model']['systematic (half range without 3-8)']:.3f}"
            for p, v in sy.items()
        )
    )
    iv = s["intervals"]["binomial"]["prespecified"]
    L.append(
        "   binomial O/M, Hartung-Knapp floored: "
        + ", ".join(
            f"{q} {v['ratio']:.2f} ({v['hartung_knapp_floored_ci95'][0]:.2f}-"
            f"{v['hartung_knapp_floored_ci95'][1]:.2f})"
            for q, v in iv.items()
        )
    )
    off = s["intervals"]["offset"]["wls"]["X lambda"]
    pw = s["power"]
    made = pw["provenance"]["docs/analysis/results/power.json first committed"] or {}
    L.append(
        f"   Lambda_X offset t test p {off['t_test (9 dof)']['p']:.2f}; power at 0.06: "
        + ", ".join(
            f"{k} {pw[k]['X lambda']['absolute']['0.06']:.2f}"
            for k in ("prespecified", "idle_noise_only", "no_f_term")
        )
        + f"; power.json first committed {made.get('utc')} UTC ({made.get('commit')})"
    )
    t = s["tracking placement 1 Lambda_X"]["wls"]
    st = s["X5X6X8X9 detection fraction by day (placement 1, d = 3, bulk rounds)"]
    L.append(
        f"   placement 1 follows its prediction: chi2 {t['all days']['follows_prediction']['chi2']:.1f}, "
        f"without day 2 {t['without day 2']['follows_prediction']['chi2']:.1f} "
        f"(p {t['without day 2']['follows_prediction']['p']:.2f}); X5X6X8X9 device/model by day "
        + ", ".join(
            f"{v['device']:.3f}/{v['model']:.3f}" for k, v in st.items() if k.startswith("day")
        )
    )
    sc = s["calibration_scatter_test_power"]["X lambda"]
    L.append(
        f"   calibration-scatter test: 80% power at a Lambda_X miss of {sc['absolute_found_at_80pct']:.2f}"
    )
    return L


def dependence_digest(d: dict) -> list[str]:
    h = d["headline"]
    L = ["19. Dependent repetitions, raw counts, Lambda_X margin (dependence.py)"]
    L.append(
        "   X tests (WL) p: prespecified / placements correlated / P1 alone / P2 alone / "
        "offset day-block t / sign flip / hierarchical"
    )
    for q, v in h["X tests (WL), p"].items():
        L.append(f"     {q}: " + " / ".join(f"{p:.2g}" for p in v.values()))
    for q, v in h["O/M (binomial) [ratio, 95% low, high]"].items():
        L.append(
            f"   O/M {q}: "
            + ", ".join(
                f"{k} {x[0]:.2f} ({x[1]:.2f}-{x[2]:.2f})"
                for k, x in v.items()
                if not k.startswith("tau")
            )
        )
    for c, v in h["raw counts against the model (prespecified sigma_f)"].items():
        x = v["chi2/dof [all, round 1, rounds 2-8, P_model < 0.45]"]
        r = v["eps after round 1, device / model [ratio, 95% low, high]"]
        a0 = v["fidelity ratio at n = 0 [ratio, 95% low, high]"]
        L.append(
            f"   counts {c}: chi2 {x[0][0]:.0f}/{x[0][1]} (round 1 {x[1][0]:.0f}/{x[1][1]}), excess pp "
            + " ".join(f"{e:+.1f}" for e in v["excess_pp by n"].values())
            + f"; after round 1 eps x{r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f}), "
            f"{v['excess eps per round after round 1 (pp)']:.1f} pp/round; F(0) x{a0[0]:.2f}; "
            f"round 1 off trend {v['round 1 above the trend (pp)']:+.1f} pp"
        )
    for k, v in h["Lambda_X O - M"].items():
        extra = (
            f", certified {v['certified_margin']:.3f}, chance to certify 0.05 "
            f"{v['chance to certify 0.05 if exact [observed sd, stated errors]'][0]:.2f}/"
            f"{v['chance to certify 0.05 if exact [observed sd, stated errors]'][1]:.2f}"
            if "certified_margin" in v
            else ""
        )
        L.append(
            f"   Lambda_X O - M {k}: {v['mean']:+.3f} (90% {v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}){extra}"
        )
    return L


def main() -> None:
    pl = take("placements.json")
    res = {
        "about": {
            "script": "docs/analysis/ledger.py",
            "fits": {
                "wls": "prespecified weighted-log fit of ln[1 - 2P(n)], primary for the X-memory decision",
                "binomial": "binomial maximum-likelihood refit of all round counts (App. stats)",
            },
            "chip_level_inference": "none: results are per placement",
        },
        **{
            k: pl[k]
            for k in (
                "per_placement",
                "placements_compared",
                "per_repetition",
                "prespecified_pooled_record",
                "agreement_tests",
                "fit_bias",
                "cross_pairing",
            )
        },
        "model_vs_device": model_vs_device(),
        "effect_sizes": effect_sizes(),
        "sensitivity": sensitivity(),
        "dependence": dependence(),
        "prespecified_tests": prespecified_tests(),
        "xy_variance_correction": take("idle_f.json")["correction"],
        "per_qubit_f_side_prediction": per_qubit_f_side(),
        "f_models": f_models(),
        "heating_exploratory": heating(),
        "decoder_audit": take("decoder.json")["paper"],
        "decoder_ml": take("decoder_ml.json")["summary"],
        "decoder_graph": decoder_graph(),
        "idle_test_scope": {
            "ancilla_reads_1": {
                k: v
                for k, v in take("idle_test_scope.json")["ancilla_reads_1"].items()
                if k != "per_job"
            },
            "f_left_out_of_the_median": take("idle_test_scope.json")["f_left_out_of_the_median"],
        },
        "fit_bias_weights": take("fit_bias.json"),
        "leakage": leakage(),
        "streak_effects_exploratory": streak_effects(),
        "placements_and_calibration_inputs": placements_and_inputs(),
        "diagnostics": diagnostics(),
        "error_scale_and_f": scale_and_f(),
        "error_budget": error_budget(),
        "stat_checks": stat_checks(),
        "input_noise": input_noise(),
        "channels": channels(),
        "f_treatments": f_treatments(),
        **other(),
    }
    res["about"]["inputs"] = sorted(INPUTS | {"docs/analysis/results/placements.json"})
    write_json("paper_numbers.json", res)
    text = digest(res)
    (RESULTS / "paper_numbers.txt").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

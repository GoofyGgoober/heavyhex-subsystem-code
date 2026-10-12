"""Every number the paper quotes, collected from the results files, each section with its source.

No computation: this only selects from results/*.json (and runs/combined.json, docs/figures/
figure3.json) and writes results/paper_numbers.json and a readable digest,
results/paper_numbers.txt. Run it after the scripts whose results it reads; it takes a second.

    python docs/analysis/ledger.py
"""

from __future__ import annotations

import sys

from common import REPO, RESULTS, read, result, write_json

FITS = ("wls", "binomial")
BASES = ("X", "Z")
QUANTITIES = ("eps3", "eps5", "lambda")
PLACEMENTS = (1, 2)
INPUTS: set[str] = set()


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
        "prespecified_drift_check": m["drift"],
        "model_curves_day1_rep1": m["model_curves"],
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
            for k in ("published", "shot_bootstrap_ci95", "cluster", "lengths", "capped_at_300")
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
        "frame_binomial_means": fit["frame_binomial"]["means_over_ten_reps"],
        "checks_three_qubit_example": checks["A"],
        "checks_cz_operands_with_definite_z": checks["E"]["total"],
        "cz_scaled": {"follow_ratio_mean": cz["follow_ratio_mean"], "minus_fit": cz["minus_fit"]},
    }


def other() -> dict:
    out = {
        "readout": take("readout.json"),
        "uniform_scale": take("uniform_scale.json"),
        "no_reset": take("no_reset.json"),
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
        "1. Per placement (inverse-variance mean over 5 days; chi2/nu with nu = 4; ratio = observed / mean predicted)"
    )
    for b in BASES:
        for pl in PLACEMENTS:
            for fit in FITS:
                blk = pp[b][f"placement {pl}"][fit]
                parts = []
                for q in QUANTITIES:
                    o = blk[q]["observed"]
                    scale, d = (100, 1) if q != "lambda" else (1, 3)
                    parts.append(
                        f"{q} {fmt_pm(scale * o['value'], scale * o['se'], d)} (chi2/nu {o['chi2_per_dof']:.2f}; "
                        f"pred {scale * blk[q]['predicted_mean']:.{d}f}; ratio {blk[q]['observed_over_predicted']:.3f}; "
                        f"agrees {blk[q]['days_model_agrees']}/5)"
                    )
                L.append(f"   {b} placement {pl} {fit:8s}: " + "; ".join(parts))
    for k, v in pp["observed_over_predicted_range_eps3_eps5_both_placements"].items():
        L.append(f"   observed/predicted eps3, eps5 range, {k}: {v[0]:.2f}-{v[1]:.2f}")
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
    L.append("4. Prespecified 10-repetition rule (record only)")
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
    return "\n".join(L) + "\n"


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
        "xy_variance_correction": take("idle_f.json")["correction"],
        "per_qubit_f_side_prediction": per_qubit_f_side(),
        "heating_exploratory": heating(),
        "decoder_audit": take("decoder.json")["paper"],
        "leakage": leakage(),
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

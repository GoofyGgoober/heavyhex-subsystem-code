"""Lambda and eps per placement, per repetition and under both fits (Sec. results, Table
placement, Table reps, App. stats).

The two repetitions of each day ran on the same two distance-5 placements every day, and
repetition 2's d = 5 patch shares 51 of its 65 qubits with repetition 1's, so the results are
reported per placement, over its five days, with no estimate or test across placements.

  per_placement               eps3, eps5 and Lambda for X and Z memory on each placement under
                              the prespecified weighted-log fit ("wls") and the binomial refit
                              ("binomial"): inverse-variance mean over the five days, its error
                              widened by sqrt(chi2/nu) when chi2/nu > 1 (nu = 4); plain mean
                              over the days with its standard error over the days; the model's
                              plain mean prediction under the same fit; and observed / predicted,
                              both as plain means (the paper's eps columns and factors)
  placements_compared         placement 1 minus placement 2, and the qubits they share
  per_repetition              the ten repetitions under both fits, and their verdicts
  prespecified_pooled_record  the prespecified rule over all ten repetitions as if
                              independent (runs/combined.json; a record for the appendix)
  agreement_tests             the prespecified model-agreement test under both fits
  fit_bias                    the bootstrap bias of the weighted-log fit, and the deviance of
                              the binomial fits
  cross_pairing               placement 2's d = 3 patch paired with placement 1's d = 5 patch
  check                       each repetition's Lambda recomputed from the jobs'
                              analysis.json against runs/combined.json

Reads results/fits.json (fits.py) and runs/; a few seconds.

    python docs/analysis/placements.py      # writes results/placements.json
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats

from common import RUNS, read, result, write_json

FITS = ("wls", "binomial")
BASES = ("X", "Z")
QUANTITIES = ("eps3", "eps5", "lambda")
PLACEMENTS = (1, 2)
SOURCE = "docs/analysis/results/fits.json"


def rep_value(entry: dict, q: str, fit: str) -> tuple[float, float]:
    """(value, sigma) of one repetition, as fits.py pools it."""
    x = entry[q]
    if fit == "wls":
        return float(x["wls"][0]), float(x["wls"][1])
    if q == "lambda":
        return float(x["binomial"]), float(x["binomial_se_quasi"])
    return float(x["binomial"]), float((x["binomial_hi68_quasi"] - x["binomial_lo68_quasi"]) / 2)


def pool(values, sigmas) -> dict:
    """Inverse-variance mean; error widened by sqrt(chi2/nu) when > 1 (the prespecified rule)."""
    v, w = np.asarray(values, float), 1 / np.asarray(sigmas, float) ** 2
    m = float((w * v).sum() / w.sum())
    chi2 = float((w * (v - m) ** 2).sum())
    nu = len(v) - 1
    red = chi2 / nu if nu else 1.0
    se0 = float(1 / sqrt(w.sum()))
    return {
        "value": m,
        "se": se0 * sqrt(max(red, 1.0)),
        "se_unwidened": se0,
        "chi2": chi2,
        "nu": nu,
        "chi2_per_dof": red,
        "widened": red > 1,
    }


def lambda_t_test(m: float, se: float, nu: int) -> dict:
    """Lambda < 1 within one placement, Student t with nu degrees of freedom."""
    t = (m - 1) / se
    return {"nu": nu, "one_sided_p_lambda_below_1": float(stats.t.cdf(t, df=nu))}


def verdict(lam: float, sigma: float, critical: float = 1.645) -> str:
    """run.verdict (src/heavyhex/experiment/run.py): one-sided 5% each way."""
    if lam - critical * sigma > 1:
        return "d = 5 better"
    if lam + critical * sigma < 1:
        return "d = 5 worse"
    return "undecided"


def span(values) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


class Data:
    def __init__(self) -> None:
        self.refits = result("fits.json")
        self.combined = read(RUNS / "combined.json")
        self.per_rep = self.refits["per_rep"]
        self.keys = sorted(self.per_rep)

    def day_number(self, day: str) -> int:
        days = sorted({v["day"] for v in self.per_rep.values()})
        return days.index(day) + 1


def per_placement(D: Data) -> dict:
    out: dict = {
        "source": SOURCE,
        "method": "observed: inverse-variance mean over the five days, se widened by "
        "sqrt(chi2/nu) when chi2/nu > 1 (nu = 4); binomial sigmas are half-widths "
        "of the quasi-likelihood 68% profile intervals. observed_plain: plain mean over "
        "the five days, se = sd / sqrt(5). predicted: plain mean over the five days of "
        "the model's prediction (each job's measured f), fitted with the same estimator. "
        "ratio = observed_plain / predicted_mean, so both sides are averaged alike; its "
        "se carries the observed error only. Lambda is pooled from the daily ratios "
        "(inverse-variance). The t test is within one placement (nu = 4); there is no "
        "test across placements.",
    }
    checks = []
    for b in BASES:
        out[b] = {}
        for pl in PLACEMENTS:
            keys = [k for k in D.keys if D.per_rep[k]["placement"] == pl]
            out[b][f"placement {pl}"] = {}
            for fit in FITS:
                block = {}
                for q in QUANTITIES:
                    vals = [rep_value(D.per_rep[k][b], q, fit) for k in keys]
                    p = pool([v for v, _ in vals], [s for _, s in vals])
                    ref = D.refits["summary"]["placement_pooled"][f"{b} {q} {fit} placement {pl}"]
                    checks.append(
                        max(
                            abs(p["value"] - ref["pooled"]),
                            abs(p["se"] - ref["se_widened"]),
                            abs(p["chi2_per_dof"] - ref["chi2_per_dof"]),
                        )
                    )
                    pred = [float(D.per_rep[k][b]["predicted"][q][fit]) for k in keys]
                    pm = float(np.mean(pred))
                    ratios = [v / pr for (v, _), pr in zip(vals, pred)]
                    agree = [
                        bool(D.refits["agreement_tests"][fit][k][f"{b} {q}"]["agrees"])
                        for k in keys
                    ]
                    plain = [v for v, _ in vals]
                    om = float(np.mean(plain))
                    ose = float(np.std(plain, ddof=1) / sqrt(len(plain)))
                    block[q] = {
                        "observed": p,
                        "observed_plain": {"value": om, "se_over_days": ose},
                        "predicted_mean": pm,
                        "predicted_range": span(pred),
                        "observed_over_predicted": om / pm,
                        "observed_over_predicted_se_observed_only": ose / pm,
                        "inverse_variance_observed_over_predicted": p["value"] / pm,
                        "per_day_observed_over_predicted_range": span(ratios),
                        "days_model_agrees": int(sum(agree)),
                        "per_day": [
                            {
                                "day": D.per_rep[k]["day"],
                                "value": v,
                                "se": s,
                                "predicted": pr,
                                "ratio": v / pr,
                                "model_agrees": a,
                            }
                            for k, (v, s), pr, a in zip(keys, vals, pred, agree)
                        ],
                    }
                block["lambda"]["within_placement_test"] = lambda_t_test(
                    block["lambda"]["observed"]["value"], block["lambda"]["observed"]["se"], 4
                )
                out[b][f"placement {pl}"][fit] = block
    assert max(checks) < 1e-12, max(checks)
    out["replicates_fits_placement_pooled_to"] = max(checks)
    ranges = {}
    for fit in FITS:
        for b in BASES:
            r = [
                out[b][f"placement {pl}"][fit][q]["observed_over_predicted"]
                for pl in PLACEMENTS
                for q in ("eps3", "eps5")
            ]
            ranges[f"{b} {fit}"] = span(r)
    out["observed_over_predicted_range_eps3_eps5_both_placements"] = ranges
    return out


def placements_compared(D: Data, pp: dict) -> dict:
    """Difference between the two placements and their overlap; no estimate across them."""
    from heavyhex.patches.placement import FEZ_MAP, build_layout

    out: dict = {
        "note": "No estimate, p-value or bound across placements is given: on every day these were "
        "the only usable distance-5 placements, and they overlap."
    }
    for b in BASES:
        for fit in FITS:
            l1 = pp[b]["placement 1"][fit]["lambda"]["observed"]
            l2 = pp[b]["placement 2"][fit]["lambda"]["observed"]
            diff = l1["value"] - l2["value"]
            se = sqrt(l1["se"] ** 2 + l2["se"] ** 2)
            out[f"Lambda_{b} {fit}"] = {
                "placement_1_minus_placement_2": diff,
                "se": se,
                "chi2_between_placements_1dof": (diff / se) ** 2,
                "plain_mean_of_the_two_descriptive_only": (l1["value"] + l2["value"]) / 2,
            }
    dev = read(FEZ_MAP)

    def qubits(d, spot):
        origin, direction = spot
        lay = build_layout(
            d, dev["coords"], dev["edges"], origin=tuple(origin), direction=tuple(direction)
        )
        return set(lay.physical_qubits)

    spots = {(r["rep"], d): set() for r in D.combined["reps"] for d in ("3", "5")}
    for r in D.combined["reps"]:
        for d in ("3", "5"):
            spots[(r["rep"], d)].add(tuple(map(tuple, r["spots"][d])))
    assert len(spots[(1, "5")]) == len(spots[(2, "5")]) == len(spots[(1, "3")]) == 1
    p1 = qubits(5, next(iter(spots[(1, "5")])))
    p2 = qubits(5, next(iter(spots[(2, "5")])))
    d3_p2 = [qubits(3, s) for s in sorted(spots[(2, "3")])]
    d3_p1 = qubits(3, next(iter(spots[(1, "3")])))
    out["overlap"] = {
        "source": "runs/combined.json spots + src/heavyhex/patches/placement.py (read only)",
        "d5_qubits_each": len(p1),
        "d5_shared": len(p1 & p2),
        "placement_2_d3_spots": [list(map(list, s)) for s in sorted(spots[(2, "3")])],
        "placement_2_d3_inside_placement_1_d5": all(q <= p1 for q in d3_p2),
        "placement_1_d3_qubits_in_placement_2_d5": len(d3_p1 & p2),
    }
    return out


def per_repetition(D: Data) -> dict:
    rows = []
    for k in D.keys:
        e = D.per_rep[k]
        row = {
            "day": e["day"],
            "day_number": D.day_number(e["day"]),
            "rep": e["rep"],
            "placement": e["placement"],
            "pairing": e["pairing"],
        }
        for b in BASES:
            for fit in FITS:
                row[f"{b} {fit}"] = {q: list(rep_value(e[b], q, fit)) for q in QUANTITIES}
                lam, s = rep_value(e[b], "lambda", fit)
                row[f"{b} {fit}"]["verdict"] = verdict(lam, s)
            row[f"{b} predicted wls"] = {q: float(e[b]["predicted"][q]["wls"]) for q in QUANTITIES}
        rows.append(row)
    lx = {fit: [r[f"X {fit}"]["lambda"][0] for r in rows] for fit in FITS}
    pre = [r["verdict"] for r in D.combined["reps"]]
    diff = np.array(lx["binomial"]) - np.array(lx["wls"])
    return {
        "source": SOURCE + " (per_rep); prespecified verdicts from runs/combined.json",
        "rows": rows,
        "Lambda_X_range": {fit: span(lx[fit]) for fit in FITS},
        "prespecified_verdicts": {v: pre.count(v) for v in sorted(set(pre))},
        "verdicts": {
            f"{b} {fit}": {
                v: [r[f"{b} {fit}"]["verdict"] for r in rows].count(v)
                for v in sorted({r[f"{b} {fit}"]["verdict"] for r in rows})
            }
            for b in BASES
            for fit in FITS
        },
        "binomial_lowers_Lambda_X_in": int((diff < 0).sum()),
        "binomial_raises_Lambda_X_max": float(diff.max()),
        "eps5_binomial_minus_wls_pp": {
            b: {
                "mean": float(
                    100
                    * np.mean(
                        [r[f"{b} binomial"]["eps5"][0] - r[f"{b} wls"]["eps5"][0] for r in rows]
                    )
                ),
                "max": float(
                    100
                    * np.max(
                        [r[f"{b} binomial"]["eps5"][0] - r[f"{b} wls"]["eps5"][0] for r in rows]
                    )
                ),
            }
            for b in BASES
        },
        "per_rep_observed_over_predicted_range": {
            f"{b} {q} {fit}": span(
                [
                    rep_value(D.per_rep[k][b], q, fit)[0] / D.per_rep[k][b]["predicted"][q][fit]
                    for k in D.keys
                ]
            )
            for b in BASES
            for q in QUANTITIES
            for fit in FITS
        },
    }


def prespecified_record(D: Data) -> dict:
    px = D.combined["pooled_x"]
    nu = len(D.combined["reps"]) - 1
    record = {
        "source": "runs/combined.json (pooled_x), the prespecified code at v0.2.0",
        "Lambda_X": px["lambda"],
        "se": px["lambda_uncertainty"],
        "chi2_per_dof": px["spread"],
        "nu": nu,
        "chi2": px["spread"] * nu,
        "birge_ratio": sqrt(px["spread"]),
        "t9_critical": px["critical"],
        "verdict": px["verdict"],
        "predicted_inverse_variance_weighted": px["predicted"],
        "note": "Record only (appendix): the rule treats the ten repetitions as independent.",
    }
    rp = D.refits["summary"]["pooled_lambda"]
    assert abs(rp["X wls"]["lambda"] - px["lambda"]) < 1e-12
    other = {}
    for b in BASES:
        for fit in FITS:
            v = rp[f"{b} {fit}"]
            other[f"{b} {fit}"] = {
                "Lambda": v["lambda"],
                "se": v["se_widened"],
                "sqrt_chi2_per_dof": v["sqrt_chi2_per_dof"],
                "verdict": v["verdict"],
                "predicted_inverse_variance_weighted": v["predicted_weighted"],
                "prespecified": b == "X" and fit == "wls",
            }
    return {
        "prespecified": record,
        "same_rule_other_fits_and_Z": other,
        "same_rule_other_fits_source": SOURCE + " (summary.pooled_lambda)",
    }


def agreement_tests(D: Data) -> dict:
    out = {
        "source": SOURCE + " (agreement, agreement_tests); wls replicates runs/combined.json model"
    }
    for fit in FITS:
        model = D.refits["agreement"][fit]
        tests = D.refits["agreement_tests"][fit]
        out[fit] = {
            q: {k: model[q][k] for k in ("chi2", "bar", "agrees", "reps_agreeing")} for q in model
        }
        out[fit]["reps_agreeing_on_all_three_X"] = int(
            sum(all(tests[k][f"X {q}"]["agrees"] for q in QUANTITIES) for k in D.keys)
        )
    for q, v in D.combined["model"].items():
        assert abs(v["chi2"] - out["wls"][q]["chi2"]) < 1e-9
    return out


def fit_bias(D: Data) -> dict:
    curves = D.refits["curves"]
    out = {"source": SOURCE + " (curves[*].bootstrap: parametric, 2000 replicates; per_rep)"}
    for b in BASES:
        for d in ("3", "5"):
            cs = [c for name, c in curves.items() if name.endswith(f"{b} d{d}")]
            shift = [
                D.per_rep[k][b][f"eps{d}"]["binomial"] - D.per_rep[k][b][f"eps{d}"]["wls"][0]
                for k in D.keys
            ]
            out[f"{b} eps{d}"] = {
                "wls_bias_pp_mean": 100 * float(np.mean([c["bootstrap"]["wls_bias"] for c in cs])),
                "wls_bias_pp_range": [
                    100 * v for v in span([c["bootstrap"]["wls_bias"] for c in cs])
                ],
                "binomial_bias_pp_mean": 100
                * float(np.mean([c["bootstrap"]["binomial_bias"] for c in cs])),
                "observed_shift_binomial_minus_wls_pp_mean": 100 * float(np.mean(shift)),
            }
    gof = D.refits["goodness_of_fit"]
    out["binomial_goodness_of_fit"] = {
        "deviance": gof["deviance_sum"],
        "dof": gof["dof_sum"],
        "p": gof["p_total"],
        "curves_with_A_above_1": len(gof["A_above_1"]),
        "points_with_P_at_or_above_half": sum(gof["curves_with_points_at_or_above_half"].values()),
    }
    return out


def cross_pairing(D: Data) -> dict:
    days = sorted({v["day"] for v in D.per_rep.values()})
    out: dict = {"source": SOURCE}
    for fit in FITS:
        rows = []
        for day in days:
            r1, r2 = D.per_rep[f"{day} r1"], D.per_rep[f"{day} r2"]
            e3_r2, e5_r1 = rep_value(r2["X"], "eps3", fit)[0], rep_value(r1["X"], "eps5", fit)[0]
            e5_r2, e3_r1 = rep_value(r2["X"], "eps5", fit)[0], rep_value(r1["X"], "eps3", fit)[0]
            rows.append(
                {
                    "day": day,
                    "rep2_eps3_over_rep1_eps5": e3_r2 / e5_r1,
                    "rep2_own": e3_r2 / e5_r2,
                    "rep1_own": e3_r1 / e5_r1,
                }
            )
        out[fit] = {
            "per_day": rows,
            **{
                f"mean_{k}": float(np.mean([r[k] for r in rows]))
                for k in ("rep2_eps3_over_rep1_eps5", "rep2_own", "rep1_own")
            },
        }
    return out


def check(D: Data) -> dict:
    """Each repetition's Lambda and its error recomputed from the jobs' eps fits
    (Var(Lambda)/Lambda^2 = Var(eps3)/eps3^2 + Var(eps5)/eps5^2) against runs/combined.json."""
    dev = {"X": [0.0, 0.0], "Z": [0.0, 0.0]}
    for r in D.combined["reps"]:
        tests = {}
        for folder in r["folders"]:
            tests.update(read(RUNS / folder / "analysis.json")["agreement"]["tests"])
        for b in "XZ":
            e3, e5 = tests[f"{b} eps3"], tests[f"{b} eps5"]
            lam = e3["observed"] / e5["observed"]
            se = lam * sqrt(
                (e3["observed_uncertainty"] / e3["observed"]) ** 2
                + (e5["observed_uncertainty"] / e5["observed"]) ** 2
            )
            t = r["tests"][f"{b} lambda"]
            dev[b][0] = max(dev[b][0], abs(lam - t["observed"]))
            dev[b][1] = max(dev[b][1], abs(se - t["observed_uncertainty"]))
    return {b: {"max_abs_dev_lambda": v[0], "max_abs_dev_se": v[1]} for b, v in dev.items()}


def main() -> None:
    D = Data()
    pp = per_placement(D)
    out = {
        "per_placement": pp,
        "placements_compared": placements_compared(D, pp),
        "per_repetition": per_repetition(D),
        "prespecified_pooled_record": prespecified_record(D),
        "agreement_tests": agreement_tests(D),
        "fit_bias": fit_bias(D),
        "cross_pairing": cross_pairing(D),
        "check": check(D),
    }
    for b in BASES:
        for pl in PLACEMENTS:
            o = pp[b][f"placement {pl}"]["wls"]["lambda"]["observed"]
            print(
                f"Lambda_{b} placement {pl}: {o['value']:.3f} +- {o['se']:.3f} (chi2/nu {o['chi2_per_dof']:.2f})"
            )
    print("recomputed from analysis.json:", out["check"])
    write_json("placements.json", out)


if __name__ == "__main__":
    main()

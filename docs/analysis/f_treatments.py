"""Observed over predicted under each treatment of f, by pair, and the value of every curve
(Sec. VI A, Table III, Fig. 3, Table A2).

The model needs one number the calibration does not give, the dephasing multiplier f. The
predictions exist under five treatments of it: f = 1 and f = 0, stored before each job; the
job's median f (the planned tests); each data qubit at its own f; and the inverse-variance mean
of the per-qubit f, whose priors also decode the device's shots again (f_weighted_mean.py).
For each treatment and each quantity, O/M is the geometric mean over the ten pair-days with a
Student t 95% interval (9 dof) from the spread of ln(O/M), the paper's one estimator for a
ratio over pair-days; the binomial fit throughout. Also how many of the ten curves lie above
the prediction.

  table      O/M of X eps3, eps5, Lambda_X, Z eps3, eps5 under each treatment, and the range of f
  pairs      each pair's mean over the five days: the device, the prediction at f = 1 and at the
             job's f (Lambda: the mean of the daily ratios)
  lambda_without_day2_pair1
             Lambda_X's O/M over the other nine pair-days, at f = 1 and at the job's f
  windows    the range of O/M (at the job's f, binomial, both sides fitted alike) over the
             windows of rounds in results/effect_sizes.json
  curves     every curve: the device's eps (binomial, its 68% quasi-profile half-width), the
             predictions at f = 1 and at the job's f, and the job's f; Lambda_X likewise

Reads results/fits.json, results/f_weighted_mean.json and results/effect_sizes.json, and each
job's f from runs/; no simulation; a second.

    python docs/analysis/f_treatments.py      # writes results/f_treatments.json
"""

from __future__ import annotations

import numpy as np

from common import day_number, jobs, result, write_json
from effect_sizes import t_interval

QUANTITIES = ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5")
STORED = {
    "f = 1, stored before the job": "frozen_f1.0",
    "f = 0, stored before the job": "frozen_f0.0",
    "the job's median f (planned)": "predicted_at_measured_dephasing",
    "each data qubit at its own f": "predicted_with_each_qubits_dephasing",
}
WEIGHTED = "inverse-variance mean of the per-qubit f"
F1, MEDIAN = "frozen_f1.0", "predicted_at_measured_dephasing"


def observed(rep: dict, q: str) -> tuple[float, float]:
    """The device's value (binomial) and its 68% half-width."""
    basis, name = q.split()
    e = rep[basis][name]
    return e["binomial"], (e["binomial_hi68_quasi"] - e["binomial_lo68_quasi"]) / 2


def predicted(pred: dict, rep: dict, source: str, q: str) -> float:
    basis, name = q.split()
    eps = {d: pred[rep["jobs"][d]][source][basis][d]["binomial"]["eps"] for d in "35"}
    return eps["3"] / eps["5"] if name == "lambda" else eps[name[-1]]


def summary(y: np.ndarray, above: int) -> dict:
    return {**t_interval(y), "above": above, "of": len(y)}


def main() -> None:
    fits, weighted = result("fits.json"), result("f_weighted_mean.json")
    per_rep, pred = fits["per_rep"], fits["predictions"]
    keys = sorted(per_rep, key=lambda k: (per_rep[k]["day"], per_rep[k]["rep"]))
    table: dict = {}
    for row, source in STORED.items():
        table[row] = {}
        for q in QUANTITIES:
            o = np.array([observed(per_rep[k], q)[0] for k in keys])
            m = np.array([predicted(pred, per_rep[k], source, q) for k in keys])
            table[row][q] = summary(np.log(o / m), int((o > m).sum()))
    w = weighted["binomial"]["O_over_M"]["weighted-mean f"]
    table[WEIGHTED] = {
        q: summary(
            np.log(w[q]["per_repetition"]), int((np.array(w[q]["per_repetition"]) > 1).sum())
        )
        for q in QUANTITIES
    }

    found = jobs()
    f_of = {j.folder.name: j.f for j in found}
    median = [j.f for j in found]
    mean = [j.analysis["idle_test"]["uniformity"]["weighted_mean"] for j in found]
    f_range = {
        "f = 1, stored before the job": [1.0, 1.0],
        "f = 0, stored before the job": [0.0, 0.0],
        "the job's median f (planned)": [min(median), max(median)],
        WEIGHTED: [min(mean), max(mean)],
    }

    pairs: dict = {}
    for p in (1, 2):
        mine = [k for k in keys if per_rep[k]["placement"] == p]
        pairs[f"pair {p}"] = {
            q: {
                "observed": float(np.mean([observed(per_rep[k], q)[0] for k in mine])),
                "at f = 1": float(np.mean([predicted(pred, per_rep[k], F1, q) for k in mine])),
                "at the job's f": float(
                    np.mean([predicted(pred, per_rep[k], MEDIAN, q) for k in mine])
                ),
            }
            for q in QUANTITIES
        }

    rest = [
        k for k in keys if not (day_number(per_rep[k]["day"]) == 2 and per_rep[k]["placement"] == 1)
    ]
    without_day2 = {
        name: t_interval(
            np.log(
                [
                    observed(per_rep[k], "X lambda")[0]
                    / predicted(pred, per_rep[k], src, "X lambda")
                    for k in rest
                ]
            )
        )
        for name, src in (("at f = 1", F1), ("at the job's f", MEDIAN))
    }

    by_window = result("effect_sizes.json")["window"]["by_window"]
    windows = {
        "windows": sorted(by_window),
        **{
            q: [
                min(w[q]["binomial_O_over_M"]["ratio"] for w in by_window.values()),
                max(w[q]["binomial_O_over_M"]["ratio"] for w in by_window.values()),
            ]
            for q in ("X eps3", "X eps5", "Z eps3", "Z eps5")
        },
    }

    curves = []
    for k in keys:
        r = per_rep[k]
        row: dict = {
            "day": day_number(r["day"]),
            "pair": r["placement"],
            "f": {d: f_of[r["jobs"][d]] for d in "35"},
        }
        for q in QUANTITIES:
            row[q] = {
                "observed": observed(r, q),
                "at f = 1": predicted(pred, r, F1, q),
                "at the job's f": predicted(pred, r, MEDIAN, q),
            }
        curves.append(row)

    out = {
        "what": "observed over predicted (binomial fit) under each treatment of f: geometric mean "
        "over the ten pair-days, Student t 95% interval (9 dof); each pair's means; every curve",
        "table": table,
        "f_range": f_range,
        "pairs": pairs,
        "lambda_without_day2_pair1": without_day2,
        "windows": windows,
        "curves": curves,
    }
    write_json("f_treatments.json", out)
    for row, blk in table.items():
        cells = "  ".join(
            f"{q} {v['ratio']:.2f} ({v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}) {v['above']}/{v['of']}"
            for q, v in blk.items()
        )
        print(f"{row}: {cells}")
    print("f range:", {k: [round(v, 2) for v in r] for k, r in f_range.items()})


if __name__ == "__main__":
    main()

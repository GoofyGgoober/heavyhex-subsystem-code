"""Flags and relays against IBM's quoted readout error, and IBM's own reset and readout errors.

1. Each flag's and relay's firing rate, averaged over the Z-memory runs, set against the
   model's prediction and binned by the qubit's quoted readout error, first the day's own
   and then the other days' (which the day's choice of qubits cannot have selected on); how
   steady the quotes are from day to day; which quote predicts the flags better.
2. The model's resets take the calibration's readout error; IBM also reports a reset error
   (init_error), several times smaller: the ratio of the two by day. Then what IBM's reset
   error and separate P(1|0), P(0|1) do to the calibration model (readout_ibm.py): the change
   in eps, averaged over the patches of each size, and whether the predicted detector rates
   move towards the device's or away. Both on days 1-3 only, where IBM's figures are recorded
   for the calibration in force (properties.py).

Under a minute, after readout_ibm.py.

    python docs/analysis/readout.py      # writes results/readout.json
"""

from __future__ import annotations

import collections
import json
import statistics as st

import numpy as np

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.patches.operators import build_operators

from common import RESULTS, job_folders, result, write_json
from properties import days_in_force, init_error, readout_in_force

BINS = ((0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1))


def flags_against_quotes() -> dict:
    folders = job_folders()
    cal_by_day = {}
    for f in folders:
        day = f.name.split("fez-")[1][:10]
        cal_by_day.setdefault(day, json.loads((f / "calibration.json").read_text()))
    days = sorted(cal_by_day)
    rows = []  # (day, kind, qubit, obs, pred, ro_same, ro_other)
    for f in folders:
        day = f.name.split("fez-")[1][:10]
        run = json.loads((f / "run.json").read_text())
        a = json.loads((f / "analysis.json").read_text())
        o = a["observed"]["settings"]
        p = a["predicted_at_measured_dephasing"]["settings"]
        acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for i, s in enumerate(run["settings"]):
            if s["kind"] != "memory" or s["basis"] != "Z":
                continue
            seen = collections.Counter()
            for (patch, rnd, name), ro, rp in zip(
                o[i]["detectors"], o[i]["detector_rates"], p[i]["detector_rates"]
            ):
                if not (name.startswith("z_flag") or name.startswith("relay")):
                    continue
                roles = qubit_roles(build_operators(int(patch)))
                if name.startswith("z_flag"):
                    ci = roles.x_ancillas[name.split()[1]]
                else:
                    k = seen[(patch, rnd, name)]
                    seen[(patch, rnd, name)] += 1
                    ci = roles.z2_arms[name.split()[1]][k][1]
                q = run["chips"][patch][ci]
                x = acc[(name.split()[0], q)]
                x[0] += ro
                x[1] += rp
                x[2] += 1
        for (kind, q), (ob, pr, n) in acc.items():
            same = cal_by_day[day]["readout"][str(q)]
            other = st.mean(cal_by_day[d]["readout"][str(q)] for d in days if d != day)
            rows.append((day, kind, q, ob / n, pr / n, same, other))
    out: dict = {
        "readouts": len(rows),
        "flags": sum(r[1] == "z_flag" for r in rows),
        "relays": sum(r[1] == "relay" for r in rows),
        "days": days,
    }
    print(
        f"{len(rows)} (day, qubit) readouts: {out['flags']} flags, {out['relays']} relays, days {days}"
    )

    # how stable are IBM's quoted readout errors for these qubits, day to day?
    qs = sorted({r[2] for r in rows})
    M = np.array([[cal_by_day[d]["readout"][str(q)] for d in days] for q in qs])
    L = np.log(M)
    cors = [
        np.corrcoef(L[:, i], L[:, j])[0, 1]
        for i in range(len(days))
        for j in range(i + 1, len(days))
    ]
    out["quote_day_to_day"] = {
        "log_correlation_range": [float(min(cors)), float(max(cors))],
        "median_relative_change": float(np.median(np.abs(np.diff(M, axis=1)) / M[:, :-1])),
    }
    print(
        f"quoted readout, same qubits across days: day-to-day correlation of log values {min(cors):.2f}-{max(cors):.2f}; "
        f"median relative day-to-day change {out['quote_day_to_day']['median_relative_change']:.0%}"
    )

    # excess binned by the same day's quote and by the other days' quotes
    for idx, label in ((5, "same_day_quote"), (6, "other_days_quote")):
        out[f"binned_by_{label}"] = []
        print(f"binned by {label.replace('_', ' ')}:")
        for lo, hi in BINS:
            sel = [r for r in rows if lo <= r[idx] < hi]
            if sel:
                b = {
                    "bin": [lo, hi],
                    "n": len(sel),
                    "observed": st.mean(r[3] for r in sel),
                    "predicted": st.mean(r[4] for r in sel),
                    "excess": st.mean(r[3] - r[4] for r in sel),
                }
                out[f"binned_by_{label}"].append(b)
                print(
                    f"   {lo:.0%}-{hi:.0%}: n={len(sel):3d}  observed {b['observed']:.3f}  predicted {b['predicted']:.3f}  "
                    f"excess {b['excess']:+.3f}"
                )
    # consistently good qubits: quoted below 1.5% on every day
    good = [r for r in rows if max(cal_by_day[d]["readout"][str(r[2])] for d in days) < 0.015]
    out["quoted_below_1.5pct_every_day"] = {
        "n": len(good),
        "excess": st.mean(r[3] - r[4] for r in good),
    }
    print(
        f"qubits quoted under 1.5% on all {len(days)} days: n={len(good)}, excess {out['quoted_below_1.5pct_every_day']['excess']:+.3f}"
    )
    # which quote predicts the observed rate better?
    obs = np.array([r[3] for r in rows])
    pred = np.array([r[4] for r in rows])
    out["correlation_with_observed"] = {}
    for idx, label in ((5, "same_day_log_quote"), (6, "other_days_log_quote")):
        c = float(np.corrcoef(np.log([r[idx] for r in rows]), obs)[0, 1])
        out["correlation_with_observed"][label] = c
        print(f"correlation of observed rate with {label.replace('_', ' ')}: {c:.2f}")
    out["correlation_with_observed"]["model_prediction"] = float(np.corrcoef(obs, pred)[0, 1])
    print(
        f"correlation of observed with the model's prediction: {out['correlation_with_observed']['model_prediction']:.2f}"
    )
    # implied effective readout+reset error: excess/2 on top of the quote (a flag passes one reset and one readout)
    eff = [(r[5], r[5] + (r[3] - r[4]) / 2) for r in rows if r[5] < 0.02]
    out["quoted_below_2pct"] = {
        "quoted_median": float(np.median([e[0] for e in eff])),
        "implied_effective_median": float(np.median([e[1] for e in eff])),
    }
    print(
        f"for qubits quoted under 2%: quoted median {out['quoted_below_2pct']['quoted_median']:.2%}, "
        f"implied effective per reset/readout {out['quoted_below_2pct']['implied_effective_median']:.2%}"
    )
    return out


def reset_errors() -> dict:
    """IBM's init_error against the model's reset error (the calibration's readout error), per
    day, over the qubits with an init_error; and IBM's errors in the calibration model. Only the
    days whose IBM figures are recorded for the calibration in force (properties.py)."""
    days = days_in_force()
    ratio = {}
    for day in days:
        q = init_error(day)
        readout = readout_in_force(day)
        init = float(np.median(list(q.values())))
        model = float(np.median([readout[str(k)] for k in q]))
        ratio[day] = {
            "qubits": len(q),
            "init_error_median": init,
            "model_reset_error_median": model,
            "ratio_of_medians": init / model,
        }
    print(
        "IBM's reset error / the model's, medians by day: "
        + ", ".join(f"{d} {v['ratio_of_medians']:.2f}" for d, v in ratio.items())
    )
    return {
        "days": days,
        "init_error_against_model_reset_error": ratio,
        "calibration_model_with_ibm_errors": in_calibration_model(days),
    }


def in_calibration_model(days: list[str]) -> dict:
    """readout_ibm.json: IBM's reset and readout errors in the calibration model, in the jobs of
    the given days. The change in eps against the model, mean over the patches of each size,
    with both ("ibm") and with the resets alone ("ibm_reset"); and whether the predicted
    detector rates move towards the device's or away."""
    r = result("readout_ibm.json")
    runs = {name: job for name, job in r["jobs"].items() if name[:10] in days}
    out: dict = {"shots": r["shots"], "jobs": len(runs)}
    for basis in "XZ":
        for d in ("3", "5"):
            change = {
                variant: [
                    e[variant][0] - e["model"][0]
                    for job in runs.values()
                    for patch, e in job[basis]["eps"].items()
                    if patch == d
                ]
                for variant in ("ibm", "ibm_reset")
            }
            out[f"{basis} d{d}"] = cell = {
                "patches": len(change["ibm"]),
                "mean_change": float(np.mean(change["ibm"])),
                "se_over_patches": float(
                    np.std(change["ibm"], ddof=1) / np.sqrt(len(change["ibm"]))
                ),
                "changes": change["ibm"],
                "mean_change_resets_only": float(np.mean(change["ibm_reset"])),
            }
            print(
                f"{basis} d{d}: eps with IBM's reset and readout errors minus the model's, mean over "
                f"{cell['patches']} patches {100 * cell['mean_change']:+.2f} pp "
                f"(se {100 * cell['se_over_patches']:.2f}); resets alone "
                f"{100 * cell['mean_change_resets_only']:+.2f} pp"
            )
        rates = [job[basis]["detectors_8_rounds"] for job in runs.values()]
        jobs = out[f"{basis} detectors, 8 rounds, jobs"] = {
            "all": len(rates),
            "model_mean_below_device": sum(x["model"]["mean"] < x["device_mean"] for x in rates),
            "ibm_mean_below_model": sum(x["ibm"]["mean"] < x["model"]["mean"] for x in rates),
            "ibm_farther_from_device": sum(
                x["ibm"]["rms_relative_to_device"] > x["model"]["rms_relative_to_device"]
                for x in rates
            ),
        }
        print(f"{basis} detector rates, 8 rounds, jobs:", jobs)
    return out


def main() -> None:
    write_json(
        "readout.json", {"flags_against_quotes": flags_against_quotes(), "resets": reset_errors()}
    )
    print("wrote", RESULTS / "readout.json")


if __name__ == "__main__":
    main()

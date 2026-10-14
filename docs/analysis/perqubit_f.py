"""The prediction with each data qubit's own measured f, beside the prespecified median-f one.

For every job, analysis.json holds the prespecified prediction at the
job's median f ("predicted_at_measured_dephasing") and a side prediction in which each
data qubit gets its own idle-test f ("predicted_with_each_qubits_dephasing"; ancillas,
flags and data qubits the idle test could not pin get the median; negative estimates are
set to zero; 50,000 simulated shots per circuit instead of 200,000).

For both, this script rebuilds the per-repetition agreement tests and the experiment-wide
chi^2 test exactly as src/heavyhex/experiment/run.py does (_agreement, _ratio_test,
combine):

    z = (O - M) / (bound / 2),  bound / 2 = sqrt(sigma_O^2 + sigma_M^2),
    sigma_M^2 = simulator_noise^2 + from_f^2 + calibration_gap^2 / 2,
    chi^2 = sum of z^2 over the 10 repetitions, against chi2.ppf(0.95, 10) = 18.31.

simulator_noise is the fit's standard error of the prediction itself (its own shot
count); from_f = (dQ/df) x sigma_f of the median f; calibration_gap = |M(before) - M(after)|
of IBM's calibrations around the job. The repo computes from_f and calibration_gap only
for the median-f prediction. Versions tested:

  median_f            the prespecified test, recomputed with the repo's functions and
                      checked against analysis.json and runs/combined.json.
  per_qubit_f         M and simulator_noise from the per-qubit prediction; from_f and
                      calibration_gap carried over unchanged from the prespecified test.
  per_qubit_f_mc      as per_qubit_f, but in X memory from_f is the Monte Carlo spread of the
                      per-qubit prediction under each qubit's own f uncertainty
                      (perqubit_f_mc.py; used if results/perqubit_f_mc.json exists).
  per_qubit_f_redecoded  as per_qubit_f, with the chip's shots decoded with the per-qubit
                      model's weights (perqubit_f_decode.py; used if its results exist).

Under a minute.

    python docs/analysis/perqubit_f.py      # writes results/perqubit_f.json and prints the tables
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from heavyhex.experiment import run as R

from common import RESULTS, RUNS, day_number, job_folders, read

TESTS = ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5", "Z lambda")
PLACEMENT = {1: "placement 1 (rep 1)", 2: "placement 2 (rep 2)"}


def row(value, sigma, guess, noise, from_f, gap) -> dict:
    """One test row, as _agreement builds it."""
    bound = 2 * sqrt(sigma**2 + noise**2 + from_f**2 + gap**2 / 2)
    return {
        "observed": value,
        "predicted": guess,
        "observed_uncertainty": sigma,
        "simulator_noise": noise,
        "from_f": from_f,
        "calibration_gap": gap,
        "bound": bound,
        "agrees": abs(value - guess) < bound,
    }


def job_tests(folder: Path, mc: dict | None, redecoded: dict | None) -> dict:
    a = read(folder / "analysis.json")
    frozen = read(folder / "prediction.json")
    f = a["decoupled_dephasing"]
    seen = a["observed"]["fits"]
    expected = a["predicted_at_measured_dephasing"]
    after = a["predicted_from_calibration_after"]
    sigma_f = a["idle_test"]["median_uncertainty"]

    # The prespecified test, rebuilt with the repo's own functions, must equal the stored one.
    median = R._agreement(
        seen, expected["fits"], after and after["fits"], R._slopes(frozen, expected, f), sigma_f
    )["tests"]
    stored = a["agreement"]["tests"]
    assert median.keys() == stored.keys(), folder
    for key in median:
        for field in ("observed", "predicted", "bound"):
            assert abs(median[key][field] - stored[key][field]) < 1e-12, (folder, key, field)

    own = a["predicted_with_each_qubits_dephasing"]["fits"]
    versions: dict[str, dict] = {"median_f": median, "per_qubit_f": {}}
    if mc:
        versions["per_qubit_f_mc"] = {}
    if redecoded:
        versions["per_qubit_f_redecoded"] = {}
    for basis in "XZ":
        observed = R._quantities(seen, basis)
        predicted = R._quantities(own, basis)
        observed_rd = R._quantities(redecoded["per_qubit_f"], basis) if redecoded else {}
        for name, (value, sigma) in observed.items():
            key = f"{basis} {name}"
            guess, noise = predicted[name]
            from_f, gap = median[key]["from_f"], median[key]["calibration_gap"]
            versions["per_qubit_f"][key] = row(value, sigma, guess, noise, from_f, gap)
            if mc:
                mc_f = mc["sd"][name] if basis == "X" else from_f
                versions["per_qubit_f_mc"][key] = row(value, sigma, guess, noise, mc_f, gap)
            if redecoded:
                v2, s2 = observed_rd[name]
                versions["per_qubit_f_redecoded"][key] = row(v2, s2, guess, noise, from_f, gap)
    return versions


def combine(jobs: dict[str, dict], infos: dict[str, dict]) -> tuple[list[dict], dict]:
    """Reps and the chi^2 test over them, as run.combine() does it."""
    reps: dict = {}
    for name, tests in jobs.items():
        info = infos[name]["rep"]
        entry = reps.setdefault(
            (info["day"], info["rep"]),
            {
                "day": info["day"],
                "rep": info["rep"],
                "pairing": info["pairing"],
                "folders": [],
                "tests": {},
            },
        )
        entry["folders"].append(name)
        entry["tests"].update({k: dict(v) for k, v in tests.items()})
    complete = []
    for entry in reps.values():
        tests = entry["tests"]
        for basis in "XZ":
            if f"{basis} lambda" not in tests:
                tests[f"{basis} lambda"] = R._ratio_test(
                    tests[f"{basis} eps3"], tests[f"{basis} eps5"]
                )
        for row_ in tests.values():
            row_["z"] = (row_["observed"] - row_["predicted"]) / (row_["bound"] / 2)
        complete.append(entry)
    complete.sort(key=lambda e: (e["day"], e["rep"]))
    model = {}
    for test in TESTS:
        z = [e["tests"][test]["z"] for e in complete]
        stat, bar = float(sum(v * v for v in z)), float(chi2.ppf(0.95, len(z)))
        model[test] = {
            "chi2": stat,
            "reps": len(z),
            "bar": bar,
            "p_value": float(chi2.sf(stat, len(z))),
            "agrees": stat < bar,
            "reps_agreeing": int(sum(e["tests"][test]["agrees"] for e in complete)),
        }
        for placement in (1, 2):
            zp = [e["tests"][test]["z"] for e in complete if e["rep"] == placement]
            sp = float(sum(v * v for v in zp))
            model[test][f"placement_{placement}"] = {
                "chi2": sp,
                "reps": len(zp),
                "bar": float(chi2.ppf(0.95, len(zp))),
                "agrees": sp < float(chi2.ppf(0.95, len(zp))),
            }
    return complete, model


def pooled(values, sigmas) -> tuple[float, float, float]:
    """Inverse-variance mean, its SE widened by sqrt(chi2/dof) when above 1."""
    values, w = np.asarray(values), 1 / np.asarray(sigmas) ** 2
    m = float((w * values).sum() / w.sum())
    spread = float((w * (values - m) ** 2).sum() / (len(values) - 1))
    return m, float(sqrt(max(spread, 1.0) / w.sum())), spread


def f_summary(folder: Path) -> dict:
    a = read(folder / "analysis.json")
    idle, table = a["idle_test"], a["dephasing_by_qubit"]
    raw = {
        d: [s if s is not None and np.isfinite(s) else np.nan for s in shares]
        for d, shares in idle["per_qubit"].items()
    }
    excluded = sum(1 for r in table if not r["measured"])
    zeroed = sum(1 for d in raw for s in raw[d] if np.isfinite(s) and s < 0)
    assert zeroed == sum(1 for r in table if r["measured"] and r["dephasing"] == 0.0)
    used = {d: [r["dephasing"] for r in table if str(r["patch"]) == d] for d in raw}
    return {
        "data_qubits": len(table),
        "excluded_nan_to_median": excluded,
        "negative_set_to_zero": zeroed,
        "excluded_by_patch": {d: sum(1 for s in raw[d] if not np.isfinite(s)) for d in raw},
        "zeroed_by_patch": {d: sum(1 for s in raw[d] if np.isfinite(s) and s < 0) for d in raw},
        "median_f": idle["median"],
        "median_f_uncertainty": idle["median_uncertainty"],
        "used_f_by_patch": {
            d: {
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for d, v in used.items()
        },
        "used_f_all": {"mean": float(np.mean([r["dephasing"] for r in table]))},
    }


def main() -> None:
    folders = job_folders()
    infos = {f.name: read(f / "run.json") for f in folders}
    mc_all = (
        read(RESULTS / "perqubit_f_mc.json")["jobs"]
        if (RESULTS / "perqubit_f_mc.json").exists()
        else {}
    )
    rd_all = (
        read(RESULTS / "perqubit_f_decode.json")
        if (RESULTS / "perqubit_f_decode.json").exists()
        else {}
    )
    per_job = {f.name: job_tests(f, mc_all.get(f.name), rd_all.get(f.name)) for f in folders}
    versions = list(next(iter(per_job.values())))

    results: dict = {"versions": {}, "f_by_job": {f.name: f_summary(f) for f in folders}}
    for version in versions:
        reps, model = combine({n: v[version] for n, v in per_job.items()}, infos)
        results["versions"][version] = {"reps": reps, "model": model}

    # The median-f version must reproduce the prespecified combined.json.
    prespecified = read(RUNS / "combined.json")
    for test in TESTS:
        mine = results["versions"]["median_f"]["model"][test]["chi2"]
        assert abs(mine - prespecified["model"][test]["chi2"]) < 1e-9, (test, mine)

    # Per-placement summaries (mean over the five days).
    summary: dict = {}
    for version in versions:
        reps = results["versions"][version]["reps"]
        for placement in (1, 2):
            sel = [e for e in reps if e["rep"] == placement]
            for test in TESTS:
                rows = [e["tests"][test] for e in sel]
                o = np.array([r["observed"] for r in rows])
                so = np.array([r["observed_uncertainty"] for r in rows])
                m = np.array([r["predicted"] for r in rows])
                sm = np.array(
                    [sqrt((r["bound"] / 2) ** 2 - r["observed_uncertainty"] ** 2) for r in rows]
                )
                po, pse, pspread = pooled(o, so)
                summary.setdefault(version, {}).setdefault(placement, {})[test] = {
                    "observed_mean": float(o.mean()),
                    "observed_mean_se_days": float(o.std(ddof=1) / sqrt(len(o))),
                    "observed_mean_se_propagated": float(sqrt((so**2).sum()) / len(o)),
                    "observed_pooled": po,
                    "observed_pooled_se_widened": pse,
                    "observed_chi2_per_dof_days": pspread,
                    "predicted_by_day": m.tolist(),
                    "predicted_mean": float(m.mean()),
                    "predicted_mean_se_days": float(m.std(ddof=1) / sqrt(len(m))),
                    "predicted_sd_days": float(m.std(ddof=1)),
                    "predicted_mean_sigma_M_propagated": float(sqrt((sm**2).sum()) / len(m)),
                    "predicted_min": float(m.min()),
                    "predicted_max": float(m.max()),
                    "sigma_M_by_day": sm.tolist(),
                }
    results["placements"] = summary

    # Shift of the prediction, per-qubit minus median, with the simulators' shot noise only
    # (both use the same calibration and the same idle-test data).
    shift = {}
    med, pq = results["versions"]["median_f"]["reps"], results["versions"]["per_qubit_f"]["reps"]
    for e1, e2 in zip(med, pq):
        key = f"day {day_number(e1['day'])}, rep {e1['rep']}"
        shift[key] = {
            test: {
                "delta": e2["tests"][test]["predicted"] - e1["tests"][test]["predicted"],
                "shot_noise": sqrt(
                    e1["tests"][test]["simulator_noise"] ** 2
                    + e2["tests"][test]["simulator_noise"] ** 2
                ),
            }
            for test in TESTS
        }
    results["shift_per_qubit_minus_median"] = shift

    if rd_all:
        results["redecode_reproduces_prespecified"] = all(
            v["median_f_reproduces_prespecified"] for v in rd_all.values()
        )
    if mc_all:
        results["mc_check"] = {
            name: {
                "central": entry["central"],
                "stored": {
                    f"eps{p}": read(RUNS / name / "analysis.json")[
                        "predicted_with_each_qubits_dephasing"
                    ]["fits"]["X"]["per_round"][p]
                    for p in read(RUNS / name / "analysis.json")["patches"]
                },
                "sd": entry["sd"],
                "mean": entry["mean"],
                "draws": len(entry["draws"]),
            }
            for name, entry in mc_all.items()
        }

    (RESULTS / "perqubit_f.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    report(results, versions)


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.2f}"


def report(results: dict, versions: list[str]) -> None:
    print(
        "Per-qubit f vs median f: predicted eps3, eps5 (percent per round) and Lambda per repetition"
    )
    print("O = observed (prespecified decoding) +- sigma_O; M = prediction +- sigma_M, where")
    print(
        "sigma_M = sqrt(sim_noise^2 + from_f^2 + gap^2/2) (from_f, gap carried over from the median-f test);"
    )
    print(
        "z = (O - M)/sqrt(sigma_O^2 + sigma_M^2); * = |z| < 2 (agrees under the prespecified bound)\n"
    )
    med = results["versions"]["median_f"]["reps"]
    pq = results["versions"]["per_qubit_f"]["reps"]
    for basis in "XZ":
        print(f"{basis} memory")
        print(
            f"{'day,rep':8s} {'qty':6s} {'O':>16s} {'M median-f':>17s} {'z':>6s} {'M per-qubit-f':>17s} {'z':>6s}"
        )
        for e1, e2 in zip(med, pq):
            for q in ("eps3", "eps5", "lambda"):
                t1, t2 = e1["tests"][f"{basis} {q}"], e2["tests"][f"{basis} {q}"]
                s1 = sqrt((t1["bound"] / 2) ** 2 - t1["observed_uncertainty"] ** 2)
                s2 = sqrt((t2["bound"] / 2) ** 2 - t2["observed_uncertainty"] ** 2)
                scale, _unit = (100, "%") if q != "lambda" else (1, "")
                o = f"{scale * t1['observed']:.3f}+-{scale * t1['observed_uncertainty']:.3f}"
                m1 = f"{scale * t1['predicted']:.3f}+-{scale * s1:.3f}"
                m2 = f"{scale * t2['predicted']:.3f}+-{scale * s2:.3f}"
                tag = f"{day_number(e1['day'])},{e1['rep']}"
                print(
                    f"{tag:8s} {q:6s} {o:>16s} {m1:>17s} {t1['z']:5.2f}{'*' if t1['agrees'] else ' '}"
                    f" {m2:>17s} {t2['z']:5.2f}{'*' if t2['agrees'] else ' '}"
                )
        print()

    print("Per placement, mean over the five days (eps in percent):")
    print(
        "  O mean +- SD/sqrt5 [paper-style pooled +- widened SE]; M mean +- SD/sqrt5 over days (min-max) [propagated sigma_M of the mean]"
    )
    for basis in "XZ":
        for placement in (1, 2):
            print(f"  {basis} memory, {PLACEMENT[placement]}")
            for q in ("eps3", "eps5", "lambda"):
                test = f"{basis} {q}"
                s = results["placements"]["median_f"][placement][test]
                scale = 100 if q != "lambda" else 1
                line = (
                    f"    {q:6s} O {scale * s['observed_mean']:.3f}+-{scale * s['observed_mean_se_days']:.3f}"
                    f" [{scale * s['observed_pooled']:.3f}+-{scale * s['observed_pooled_se_widened']:.3f}]"
                )
                for version, label in (("median_f", "M med"), ("per_qubit_f", "M pq")):
                    p = results["placements"][version][placement][test]
                    line += (
                        f" | {label} {scale * p['predicted_mean']:.3f}+-{scale * p['predicted_mean_se_days']:.3f}"
                        f" ({scale * p['predicted_min']:.3f}-{scale * p['predicted_max']:.3f})"
                        f" [{scale * p['predicted_mean_sigma_M_propagated']:.3f}]"
                    )
                print(line)
    print()

    print(
        "Agreement test, chi^2 over the 10 repetitions (bar chi2_0.95(10) = 18.31); reps agreeing in brackets"
    )
    header = f"{'test':9s}" + "".join(f"{v:>32s}" for v in versions)
    print(header)
    for test in TESTS:
        line = f"{test:9s}"
        for v in versions:
            m = results["versions"][v]["model"][test]
            cell = f"{m['chi2']:.1f} {'pass' if m['agrees'] else 'FAIL'} p={m['p_value']:.1e} [{m['reps_agreeing']}]"
            line += f"{cell:>32s}"
        print(line)
    print("\nPer placement (5 reps, bar chi2_0.95(5) = 11.07):")
    for placement in (1, 2):
        for test in TESTS:
            line = f"  P{placement} {test:9s}"
            for v in versions:
                m = results["versions"][v]["model"][test][f"placement_{placement}"]
                line += f"   {v}: {m['chi2']:7.1f} {'pass' if m['agrees'] else 'FAIL'}"
            print(line)
    print()

    print(
        "Shift of the prediction, per-qubit f minus median f, by day; mean +- SD/sqrt5 over days;"
    )
    print("chi^2 of the five shifts against the two simulators' shot noise alone (5 dof):")
    for test in TESTS:
        for placement in (1, 2):
            d = [
                (v[test]["delta"], v[test]["shot_noise"])
                for k, v in results["shift_per_qubit_minus_median"].items()
                if k.endswith(f"rep {placement}")
            ]
            delta, noise = np.array([x for x, _ in d]), np.array([y for _, y in d])
            s = 100 if "eps" in test else 1
            print(
                f"  {test:9s} P{placement} "
                + " ".join(f"{s * x:+.3f}" for x in delta)
                + f"  mean {s * delta.mean():+.3f} +- {s * delta.std(ddof=1) / sqrt(len(delta)):.3f};"
                f" chi2 vs shot noise {((delta / noise) ** 2).sum():.1f}"
            )
    print()

    print("Placement-1 predicted Lambda_X by day (median f | per-qubit f):")
    for version in ("median_f", "per_qubit_f"):
        p = results["placements"][version][1]["X lambda"]
        print(
            f"  {version:12s} "
            + " ".join(f"{v:.4f}" for v in p["predicted_by_day"])
            + f"  mean {p['predicted_mean']:.4f} +- {p['predicted_mean_se_days']:.4f} (SD/sqrt5 over days)"
        )
    print()

    print(
        "Idle-test f per job (Appendix C): data qubits, excluded (f not pinned to +-0.5 -> median used),"
    )
    print(
        "negative estimates set to zero; median f +- SE; mean of the per-qubit f actually used, by patch"
    )
    for name, s in results["f_by_job"].items():
        by = " ".join(
            f"d{d}: median {u['median']:.2f} mean {u['mean']:.2f} range {u['min']:.2f}-{u['max']:.2f}"
            for d, u in s["used_f_by_patch"].items()
        )
        print(
            f"  {name:24s} n={s['data_qubits']:2d} excluded={s['excluded_nan_to_median']} zeroed={s['negative_set_to_zero']}"
            f"  median {s['median_f']:.3f}+-{s['median_f_uncertainty']:.3f} | {by}"
        )
    jobs = results["f_by_job"].values()
    print(
        f"  total over {len(results['f_by_job'])} jobs: {sum(s['data_qubits'] for s in jobs)} data-qubit measurements, "
        f"{sum(s['excluded_nan_to_median'] for s in jobs)} excluded, {sum(s['negative_set_to_zero'] for s in jobs)} set to zero"
    )
    if "mc_check" in results:
        print(
            "\nMonte Carlo f-propagation (X memory), SD over draws, and the central re-run vs the stored prediction:"
        )
        for name, c in results["mc_check"].items():
            cen = c["central"]
            parts = []
            for p, (val, err) in c["stored"].items():
                parts.append(
                    f"{p}: stored {100 * val:.2f}+-{100 * err:.2f}% rerun {100 * cen[p]:.2f}%"
                )
            sd = " ".join(f"{k} {v:.4f}" for k, v in c["sd"].items())
            print(f"  {name:24s} SD: {sd} | {'; '.join(parts)}")
    if "redecode_reproduces_prespecified" in results:
        print(
            f"\nRe-decoding check: median-f decode reproduces prespecified observed fits: {results['redecode_reproduces_prespecified']}"
        )


if __name__ == "__main__":
    main()

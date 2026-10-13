"""The q150 artefact: pair 1's prediction under five ways of giving the data qubits their f, and
what each does to the placement result (Sec. results, Sec. discussion, App. f, App. robust).

Pair 1's distance-5 patch held q150 (Q25), a corner data qubit with T2 = 5.1-5.9 us on all five
days. The prespecified model gives it the job's median f; the idle test measured its own f as
0.00, 0.12, 0.00, 0.87 and 0.19. Variants, in X memory (Z memory barely depends on f):

  median     every qubit at the job's median f: the prespecified model
  per_qubit  each data qubit at its own f, clipped at 0; ancillas, flags, relays and the data
             qubits the idle test could not pin at the median: the stored side prediction
  q150       the median, but q150 at its own f (pair 1's jobs; elsewhere the median)
  shrunk     each measured data qubit at f'_q = mu + w_q (f_q - mu), clipped at 0, with mu the
             job's median, w_q = tau^2 / (tau^2 + u_q^2), u_q the qubit's idle-fit standard
             error and tau^2 = max(0, s^2 - mean u_q^2) over the job's measured data qubits
             (s^2 with ddof 1); the rest as per_qubit
  shrunk_scaled  the same with each u_q times sqrt(max(1, chi2_red)) of the qubit's own idle
             fit, since the per-qubit fits scatter beyond their shot noise

Every variant is simulated again at 50,000 shots per circuit with the software's seeds (seed =
rounds, as run.predict), so the variants share their random numbers and their differences carry
little shot noise. The reruns of median and per_qubit are checked against the stored predictions
(200,000 and 50,000 shots, made on another machine).

  q150         its Z-or-Y error per round in the model under each variant, exact from the stim
               model's idle channels (the 3-round circuit over the 2-round one)
  dephasing_by_patch  the same for every data qubit: each patch's mean under each variant, and
               the qubits the median f over-dephases most
  pair_days    eps3, eps5 and Lambda_X per pair-day under each variant; per pair, plain means
  placement    pair 1 minus pair 2 predicted Lambda_X under each variant, against the observed
               difference (placements.py: WL, inverse-variance over the days); also the observed
               plain-mean difference with the device decoded with per-qubit priors
               (perqubit_f_decode.json)
  tests        the X agreement tests over the ten pair-days under each variant, with the f term
               and the calibration gap carried over from the prespecified test (as perqubit_f.py),
               with sigma_f as prespecified and from the idle test's noise alone
  sigma_f      per job, the prespecified sigma_f = 1.25 s_f / sqrt(n) with s_f numpy's default
               standard deviation (ddof 0, as run._idle_dephasing), the same with ddof 1, and the
               noise-only sigma_f, the standard deviation (ddof 1) of 2000 parametric-bootstrap
               medians (sensitivity.py)
  slope        dM/df above f = 1.05, where the prespecified rule takes the line through f = 0 and
               1: against the secant through f = 1 and the measured f and the parabola through
               all three, with Monte Carlo errors (sensitivity.job_predictions); the X tests with
               the f term from the secant or the parabola at f = 1.52 (day 3, pair 2, distance 3)
               only, or in every job above f = 1.05 (at 1.05-1.06 the secant is mostly noise)

    python docs/analysis/f_models.py               # the summary from results/f_models_runs.json, seconds
    python docs/analysis/f_models.py --recompute   # the simulations first, 6 min on 2 workers, 0.4 GB
                                                   # each [--jobs 2026-10-04-r1 ...]; resumes from cache/
"""

from __future__ import annotations

import argparse
import json
import time
from functools import cache
from math import sqrt
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.simulation.noisy import fit_per_round

import fits as FT
import perqubit_f as PQ
import sensitivity as S
from common import (
    CACHE,
    RESULTS,
    RUNS,
    WORKERS,
    day_number,
    job_folders,
    memory_index,
    read,
    result,
    write_json,
)
from prespecified_tests import idle_r, residual_chi2

VARIANTS = ("median", "per_qubit", "q150", "shrunk", "shrunk_scaled")
QUBIT = 150  # Q25 of pair 1's distance-5 patch
SHOTS = R.SIDE_SHOTS
X_MEMORY = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == "X"]
X_TESTS = ("X eps3", "X eps5", "X lambda")
SIGMA_F = ("prespecified", "idle_noise_only")
RUNS_FILE = RESULTS / "f_models_runs.json"
PARTS = CACHE / "f_models"  # one file per job and variant, so a stopped run resumes
SEED = 1  # the Monte Carlo redraws of the stored curves (slope errors)


# ------------------------------------------------------------------------------ the f values
@cache
def idle_inputs() -> dict:
    return result("idle_f_inputs.json")


def chi2_scales(name: str, chips: dict) -> dict[int, float]:
    """sqrt(max(1, reduced chi2)) of each data qubit's idle fit, by chip qubit."""
    out = {}
    for key, v in idle_inputs()[name]["qubits"].items():
        patch, q = key.split("/")
        x, y, nx, ny, l0, l1 = (
            np.array(v[k], float) for k in ("x", "y", "nx", "ny", "log_x0", "log_x1")
        )
        r = idle_r(x, y, nx, ny)
        share, _ = R._fit_share(r, l0, l1, (nx + ny) / 2)
        if not np.isfinite(share):
            continue
        c, dof = residual_chi2(r, l0, l1, (nx + ny) / 2, share)
        out[chips[patch][int(q)]] = sqrt(max(1.0, c / dof)) if dof else 1.0
    return out


def shrinkage(analysis: dict, scale: dict[int, float] | None = None) -> dict:
    """Each measured data qubit's f pulled toward the job's median by its idle-test noise."""
    idle, mu = analysis["idle_test"], analysis["decoupled_dephasing"]
    raw = {}
    for row in analysis["dephasing_by_qubit"]:
        if row["measured"]:
            k, q = str(row["patch"]), int(row["data"][1:]) - 1
            u = idle["uncertainty"][k][q] * (scale or {}).get(row["qubit"], 1.0)
            raw[row["qubit"]] = (idle["per_qubit"][k][q], u)
    s, u = (np.array(v) for v in zip(*raw.values(), strict=True))
    tau2 = max(0.0, float(s.var(ddof=1) - (u**2).mean()))
    w = tau2 / (tau2 + u**2)
    shrunk = np.maximum(mu + w * (s - mu), 0.0)
    return {
        "mu": mu,
        "tau": sqrt(tau2),
        "s_ddof1": float(s.std(ddof=1)),
        "rms_noise": float(sqrt((u**2).mean())),
        "weight_median": float(np.median(w)),
        "weight_range": [float(w.min()), float(w.max())],
        "f": {q: float(v) for q, v in zip(raw, shrunk, strict=True)},
        "raw": {q: [float(a), float(b)] for q, (a, b) in raw.items()},
    }


def assignments(folder: Path) -> dict:
    """(base f, {chip qubit: f} or None) of every variant, and the shrinkage behind 'shrunk*'."""
    a, info = read(folder / "analysis.json"), read(folder / "run.json")
    base, table = max(a["decoupled_dephasing"], 0.0), a["dephasing_by_qubit"]
    own = {row["qubit"]: row["dephasing"] for row in table}
    eb = shrinkage(a)
    scaled = shrinkage(a, chi2_scales(folder.name, info["chips"]))
    return {
        "median": (base, None),
        "per_qubit": (base, own),
        "q150": (base, {QUBIT: own[QUBIT]} if QUBIT in own else None),
        "shrunk": (base, {**own, **eb["f"]}),
        "shrunk_scaled": (base, {**own, **scaled["f"]}),
        "shrinkage": {"shrunk": eb, "shrunk_scaled": scaled},
    }


# ------------------------------------------------------------------------------- simulation
_loaded: dict = {}


def _job(folder: str):
    if folder not in _loaded:
        from heavyhex.patches.placement import Calibration

        _loaded.clear()  # one job in memory per worker
        path = Path(folder)
        _loaded[folder] = (
            read(path / "run.json"),
            Calibration.load(path / "calibration.json"),
            R.load_circuits(path),
        )
    return _loaded[folder]


def simulate(task) -> tuple[str, str, dict]:
    """Failures per X-memory circuit and patch, as run.predict draws them, at SHOTS shots."""
    folder, variant, base, own, patches = task
    info, calibration, circuits = _job(folder)
    failures: dict = {p: [] for p in patches}
    for i in X_MEMORY:
        setting = SETTINGS[i]
        model, _, _ = R.simulated(setting, circuits[i], calibration, info["dt"], base, own)
        sampler = model.compile_detector_sampler(seed=setting.rounds)
        detectors, flips = sampler.sample(SHOTS, separate_observables=True)
        failed = R._decode(model, detectors) != flips
        for k, p in enumerate(patches):
            failures[p].append(int(failed[:, k].sum()))
    return Path(folder).name, variant, failures


def idle_flip(folder: Path, base: float, own: dict | None) -> dict[int, float]:
    """Each data qubit's Z-or-Y error per round in the model: the 3-round circuit's idle channels
    over the 2-round one's, each channel flipping X with probability py + pz."""
    info, calibration, circuits = _job(str(folder))
    data = [r["qubit"] for r in read(folder / "analysis.json")["dephasing_by_qubit"]]
    keep = {}
    for n in (2, 3):
        i = memory_index("X", n)
        model, _, _ = R.simulated(SETTINGS[i], circuits[i], calibration, info["dt"], base, own)
        keep[n] = dict.fromkeys(data, 1.0)
        for ins in model.flattened():
            if ins.name == "PAULI_CHANNEL_1":
                _, py, pz = ins.gate_args_copy()
                for t in ins.targets_copy():
                    if t.value in keep[n]:
                        keep[n][t.value] *= 1 - 2 * (py + pz)
    return {q: (1 - keep[3][q] / keep[2][q]) / 2 for q in data}


def recompute(only: list[str] | None, workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    folders = [f for f in job_folders() if not only or f.name.removeprefix("ibm_fez-") in only]
    tasks = []
    for folder in folders:
        f = assignments(folder)
        patches = read(folder / "analysis.json")["patches"]
        for v in VARIANTS:
            if v == "q150" and f[v][1] is None:
                continue  # no q150 in this job: the median
            if not (PARTS / f"{folder.name}_{v}.json").exists():
                tasks.append((str(folder), v, *f[v], patches))
    start = time.time()
    with Pool(workers) as pool:
        for n, (name, v, failures) in enumerate(pool.imap(simulate, tasks), 1):
            (PARTS / f"{name}_{v}.json").write_text(json.dumps(failures) + "\n")
            print(f"{n}/{len(tasks)} {name} {v}: {time.time() - start:.0f} s", flush=True)


def assemble() -> bool:
    """Collect the parts into results/f_models_runs.json once every job has them."""
    out: dict = {"shots": SHOTS, "seeds": "rounds, as run.predict", "jobs": {}}
    for folder in job_folders():
        f = assignments(folder)
        entry: dict = {"f": {}, "failures": {}}
        for v in VARIANTS:
            part = PARTS / f"{folder.name}_{v}.json"
            if v == "q150" and f[v][1] is None:
                part = PARTS / f"{folder.name}_median.json"
            if not part.exists():
                print(f"missing {part.name}; run --recompute")
                return False
            entry["failures"][v] = json.loads(part.read_text())
            base, own = f[v]
            entry["f"][v] = {"base": base, "own": own}
        entry["shrinkage"] = f["shrinkage"]
        entry["idle_flip_per_round"] = {v: idle_flip(folder, *f[v]) for v in VARIANTS}
        out["jobs"][folder.name] = entry
    write_json(RUNS_FILE.name, out)
    return True


# ----------------------------------------------------------------------------------- summary
def fits_of(failures: dict) -> dict:
    """The X fits as run._fits makes them: eps per patch (value, se) and Lambda."""
    per_round = {}
    for p, k in failures.items():
        fit = fit_per_round(ROUNDS, [x / SHOTS for x in k], SHOTS)
        per_round[p] = [fit.per_round, fit.uncertainty]
    return {"X": {"per_round": per_round, **R.ratio(per_round)}}


def predictions(runs: dict) -> dict:
    """Every job's X fits: the stored median and per-qubit ones, and the reruns of every variant."""
    out = {}
    for folder in job_folders():
        a = read(folder / "analysis.json")
        out[folder.name] = {
            "stored median": a["predicted_at_measured_dephasing"]["fits"],
            "stored per_qubit": a["predicted_with_each_qubits_dephasing"]["fits"],
            **{v: fits_of(runs["jobs"][folder.name]["failures"][v]) for v in VARIANTS},
        }
    return out


def pair_days(pred: dict) -> list[dict]:
    """eps3, eps5 and Lambda_X (value, simulator se) per pair-day and model."""
    rows = []
    for r in read(RUNS / "combined.json")["reps"]:
        f3 = next(f for f in r["folders"] if read(RUNS / f / "analysis.json")["patches"][0] == "3")
        f5 = next(f for f in r["folders"] if "5" in read(RUNS / f / "analysis.json")["patches"])
        row = {"day": day_number(r["day"]), "pair": r["rep"], "jobs": sorted(set(r["folders"]))}
        for model in pred[f3]:
            (e3, s3), (e5, s5) = (
                pred[f3][model]["X"]["per_round"]["3"],
                pred[f5][model]["X"]["per_round"]["5"],
            )
            lam = e3 / e5
            row[model] = {
                "eps3": [e3, s3],
                "eps5": [e5, s5],
                "lambda": [lam, lam * sqrt((s3 / e3) ** 2 + (s5 / e5) ** 2)],
            }
            if f3 == f5:  # one job: run.ratio's Lambda, the same numbers
                assert abs(pred[f3][model]["X"]["lambda"] - lam) < 1e-12
        rows.append(row)
    return rows


def per_pair(rows: list[dict], models: list[str]) -> dict:
    out: dict = {}
    for pair in (1, 2):
        sel = [r for r in rows if r["pair"] == pair]
        out[f"pair {pair}"] = {}
        for m in models:
            block = {}
            for q in ("eps3", "eps5", "lambda"):
                v = np.array([r[m][q][0] for r in sel])
                s = np.array([r[m][q][1] for r in sel])
                block[q] = {
                    "by_day": v.tolist(),
                    "mean": float(v.mean()),
                    "sim_se_of_mean": float(sqrt((s**2).sum()) / len(s)),
                    "sd_over_days": float(v.std(ddof=1)),
                }
            out[f"pair {pair}"][m] = block
    return out


def placement(pairs: dict, models: list[str]) -> dict:
    """Predicted pair 1 minus pair 2 Lambda_X against the observed difference."""
    pl = result("placements.json")
    obs = pl["placements_compared"]["Lambda_X wls"]
    o, so = obs["placement_1_minus_placement_2"], obs["se"]
    plain = {
        p: pl["per_placement"]["X"][f"placement {p}"]["wls"]["lambda"]["observed_plain"]["value"]
        for p in (1, 2)
    }
    out: dict = {
        "observed (WL, inverse-variance over the days; placements.py)": {"value": o, "se": so},
        "observed plain means": {
            "pair 1": plain[1],
            "pair 2": plain[2],
            "difference": plain[1] - plain[2],
        },
    }
    for m in models:
        a, b = (pairs[f"pair {p}"][m]["lambda"] for p in (1, 2))
        diff = a["mean"] - b["mean"]
        sim = sqrt(a["sim_se_of_mean"] ** 2 + b["sim_se_of_mean"] ** 2)
        out[m] = {
            "pair 1": a["mean"],
            "pair 2": b["mean"],
            "predicted_difference": diff,
            "sim_se": sim,
            "observed_minus_predicted": o - diff,
            "z (observed se only)": (o - diff) / so,
            "z (with sim se)": (o - diff) / sqrt(so**2 + sim**2),
            "pair 1 observed plain mean minus predicted": plain[1] - a["mean"],
        }
    # The device decoded with per-qubit priors instead of the median's (perqubit_f_decode.py).
    dec = result("perqubit_f_decode.json")
    lam: dict = {1: [], 2: []}
    for r in read(RUNS / "combined.json")["reps"]:
        eps = {
            p: v[0]
            for f in r["folders"]
            for p, v in dec[f]["per_qubit_f"]["X"]["per_round"].items()
        }
        lam[r["rep"]].append(eps["3"] / eps["5"])
    means = {p: float(np.mean(v)) for p, v in lam.items()}
    out["observed plain means, device decoded with per-qubit priors"] = {
        "pair 1": means[1],
        "pair 2": means[2],
        "difference": means[1] - means[2],
    }
    return out


def tests(pred: dict, models: list[str]) -> dict:
    """The X agreement tests under each model, f term and calibration gap carried over."""
    boot = result("sensitivity.json")["sigma_f"]["per_job"]
    folders = job_folders()
    infos = {f.name: read(f / "run.json") for f in folders}
    out: dict = {}
    for label in SIGMA_F:
        out[label] = {}
        for m in models:
            jobs = {}
            for f in folders:
                a = read(f / "analysis.json")
                stored = a["agreement"]["tests"]
                scale = 1.0 if label == "prespecified" else boot[f.name]["ratio"]
                seen = R._quantities(a["observed"]["fits"], "X")
                want = R._quantities(pred[f.name][m], "X")
                rows = {k: dict(v) for k, v in stored.items() if k.startswith("Z ")}
                for name, (value, sigma) in seen.items():
                    key = f"X {name}"
                    guess, noise = want[name]
                    rows[key] = PQ.row(
                        value,
                        sigma,
                        guess,
                        noise,
                        stored[key]["from_f"] * scale,
                        stored[key]["calibration_gap"],
                    )
                jobs[f.name] = rows
            _, model = PQ.combine(jobs, infos)
            out[label][m] = {
                q: {
                    "chi2": model[q]["chi2"],
                    "p": model[q]["p_value"],
                    "agree": model[q]["reps_agreeing"],
                    "pair 1 chi2": model[q]["placement_1"]["chi2"],
                    "pair 2 chi2": model[q]["placement_2"]["chi2"],
                }
                for q in X_TESTS
            }
    pre = read(RUNS / "combined.json")["model"]
    for q in X_TESTS:  # the stored median rebuilds the prespecified test
        assert abs(out["prespecified"]["stored median"][q]["chi2"] - pre[q]["chi2"]) < 1e-9
    return out


def sigma_f() -> dict:
    """Both sigma_f per job, with the ddof each uses."""
    boot = result("sensitivity.json")["sigma_f"]["per_job"]
    out: dict = {
        "prespecified": "1.25 s_f / sqrt(n), s_f = np.std (ddof 0, the population SD) of the job's "
        "n finite per-qubit idle-test f, negative values kept (run._idle_dephasing)",
        "idle_noise_only": "the SD (ddof 1) of 2000 parametric-bootstrap medians: every data "
        "qubit's <X> and <Y> redrawn from its shot counts and refitted (sensitivity.py)",
        "per_job": {},
    }
    for folder in job_folders():
        idle = read(folder / "analysis.json")["idle_test"]
        s = np.array(
            [
                v
                for d in idle["per_qubit"]
                for v, u in zip(idle["per_qubit"][d], idle["uncertainty"][d], strict=True)
                if v is not None and u is not None and np.isfinite(v) and np.isfinite(u)
            ]
        )
        n = len(s)
        ddof0 = 1.25 * s.std() / sqrt(n)
        assert abs(ddof0 - idle["median_uncertainty"]) < 1e-12
        out["per_job"][folder.name] = {
            "n": n,
            "f": idle["median"],
            "prespecified (ddof 0)": ddof0,
            "same with ddof 1": 1.25 * s.std(ddof=1) / sqrt(n),
            "idle_noise_only (bootstrap, ddof 1)": boot[folder.name]["sigma_f_idle_noise"],
            "noise_only_over_prespecified": boot[folder.name]["ratio"],
        }
    rows = out["per_job"].values()
    for k in ("prespecified (ddof 0)", "same with ddof 1", "idle_noise_only (bootstrap, ddof 1)"):
        out[f"range {k}"] = [min(r[k] for r in rows), max(r[k] for r in rows)]
    out["ddof1_over_ddof0 = sqrt(n / (n - 1))"] = [
        min(sqrt(r["n"] / (r["n"] - 1)) for r in rows),
        max(sqrt(r["n"] / (r["n"] - 1)) for r in rows),
    ]
    return out


def slopes() -> dict:
    """dM/df above f = 1.05: the rule's line against the secant from f = 1 and the parabola."""
    D = S.Data()
    stored = S.job_predictions(D, None)
    redrawn = S.job_predictions(D, np.random.default_rng(SEED))
    boot = result("sensitivity.json")["sigma_f"]["per_job"]
    out: dict = {
        "what": "slope = the prespecified rule's (line through f = 0 and 1 above f = 0.95); "
        "secant through the simulated f = 1 and the measured f; parabola through f = 0, 1 and the "
        f"measured f; mc_sd from {S.B} binomial redraws of the stored 200,000-shot curves",
        "by_job": {},
    }
    for name, j in D.counts["jobs"].items():
        if j["f"] <= 1.05:
            continue
        out["by_job"][name] = {"f": j["f"]}
        for q, v in stored[name].items():
            cell = {}
            for k in ("slope", "slope_secant_from_1", "slope_parabola"):
                cell[k] = float(v[k][0])
                cell[f"{k} mc_sd"] = float(redrawn[name][q][k].std(ddof=1))
            cell["line_over_secant"] = cell["slope"] / cell["slope_secant_from_1"]
            cell["line_over_parabola"] = cell["slope"] / cell["slope_parabola"]
            out["by_job"][name][q] = cell

    # The X tests with the f term from each slope at f = 1.52 only, or in every job above 1.05.
    steep = max(out["by_job"], key=lambda n: out["by_job"][n]["f"])
    target = next(i for i in range(len(D.reps)) if D.job(i, "3") == steep)
    out["steepest_job"] = {"job": steep, "pair_day": D.keys[target]}
    sigma_f = {n: D.analysis[n]["idle_test"]["median_uncertainty"] for n in out["by_job"]}
    out["tests"] = {}
    for where, jobs in (
        ("f = 1.52 only", {steep}),
        ("every job above f = 1.05", set(out["by_job"])),
    ):
        for label in SIGMA_F:
            scale = {n: 1.0 if label == "prespecified" else v["ratio"] for n, v in boot.items()}
            for k in ("slope", "slope_secant_from_1", "slope_parabola"):
                rows = S.scaled(D, "wls", scale)
                for i, row in enumerate(rows):
                    j3, j5 = D.job(i, "3"), D.job(i, "5")
                    for d, j in (("3", j3), ("5", j5)):
                        if j in jobs:
                            slope = stored[j][f"eps{d}"][k][0]
                            row[f"X eps{d}"] = {
                                **row[f"X eps{d}"],
                                "from_f": slope * sigma_f[j] * scale[j],
                            }
                    if j3 == j5 and j3 in jobs:
                        lam = stored[j3]["lambda"][k][0] * sigma_f[j3] * scale[j3]
                        row["X lambda"] = {**row["X lambda"], "from_f": lam}
                    elif j3 != j5:
                        lam = FT.ratio_test(row["X eps3"], row["X eps5"])["from_f"]
                        row["X lambda"] = {**row["X lambda"], "from_f": lam}
                block = {}
                for q in X_TESTS:
                    O, M, sO, sM = S.terms(rows, q)
                    z = (O - M) / np.hypot(sO, sM)
                    block[q] = {
                        "chi2": float((z**2).sum()),
                        "p": float(stats.chi2.sf((z**2).sum(), len(z))),
                        "agree": int((np.abs(z) < 2).sum()),
                        "pair 2 chi2": float((z[D.mask(2)] ** 2).sum()),
                        "pair-day at f = 1.52": {
                            "sigma_M": float(sM[target]),
                            "z": float(z[target]),
                        },
                        "sigma_M_by_pair_day": sM.tolist(),
                    }
                out["tests"].setdefault(where, {}).setdefault(label, {})[k] = block
    pre = out["tests"]["f = 1.52 only"]["prespecified"]["slope"]
    local = out["tests"]["every job above f = 1.05"]["idle_noise_only"]["slope_secant_from_1"]
    ratio = {}
    for q in X_TESTS:
        r = np.array(pre[q]["sigma_M_by_pair_day"]) / np.array(local[q]["sigma_M_by_pair_day"])
        ratio[q] = {
            sel: [float(r[m].min()), float(np.median(r[m])), float(r[m].max())]
            for sel, m in (("all ten", slice(None)), ("pair 2", D.mask(2)))
        }
    out[
        "sigma_M prespecified over noise-only sigma_f with the secant above f = 1.05 [min, median, max]"
    ] = ratio
    return out


def q150_summary(runs: dict) -> dict:
    out = {}
    for name, entry in runs["jobs"].items():
        if str(QUBIT) not in entry["idle_flip_per_round"]["median"]:
            continue
        a = read(RUNS / name / "analysis.json")
        row = next(r for r in a["dephasing_by_qubit"] if r["qubit"] == QUBIT)
        out[name] = {
            "day": day_number(read(RUNS / name / "run.json")["rep"]["day"]),
            "t2_us": row["t2_us"],
            "own_f": [row["dephasing"], row["uncertainty"]],
            "f": {
                v: (entry["f"][v]["own"] or {}).get(str(QUBIT), entry["f"][v]["base"])
                for v in VARIANTS
            },
            "z_or_y_per_round": {v: x[str(QUBIT)] for v, x in entry["idle_flip_per_round"].items()},
        }
    return out


def dephasing_by_patch(runs: dict) -> dict:
    """Mean Z-or-Y error per round over each patch's data qubits under each variant, and which
    qubits the median f over-dephases most (its error at the median minus at its own f)."""
    out = {}
    for name, entry in runs["jobs"].items():
        a = read(RUNS / name / "analysis.json")
        flips = entry["idle_flip_per_round"]
        out[name] = {}
        for patch in a["patches"]:
            rows = [r for r in a["dephasing_by_qubit"] if str(r["patch"]) == patch]
            qs = [str(r["qubit"]) for r in rows]
            drop = {q: flips["median"][q] - flips["per_qubit"][q] for q in qs}
            top = sorted(rows, key=lambda r: -drop[str(r["qubit"])])[:3]
            out[name][f"d{patch}"] = {
                "mean": {v: float(np.mean([flips[v][q] for q in qs])) for v in VARIANTS},
                "largest drops, median to own f": [
                    {
                        "qubit": r["qubit"],
                        "t2_us": r["t2_us"],
                        "median_f": flips["median"][str(r["qubit"])],
                        "own_f": flips["per_qubit"][str(r["qubit"])],
                    }
                    for r in top
                ],
                "qubits whose error drops by more than 0.05": sum(d > 0.05 for d in drop.values()),
                "qubits whose error rises by more than 0.05": sum(d < -0.05 for d in drop.values()),
            }
    return out


def shrinkage_summary(runs: dict) -> dict:
    out = {}
    for name, entry in runs["jobs"].items():
        out[name] = {}
        for v, eb in entry["shrinkage"].items():
            out[name][v] = {
                k: eb[k]
                for k in ("mu", "tau", "s_ddof1", "rms_noise", "weight_median", "weight_range")
            }
            out[name][v]["max_abs_shrunk_minus_own"] = max(
                abs(eb["f"][q] - max(eb["raw"][q][0], 0.0)) for q in eb["f"]
            )
    return out


def reruns_check(pred: dict) -> dict:
    """The reruns of the median and per-qubit models against the stored predictions."""
    out = {}
    for stored, rerun in (("stored median", "median"), ("stored per_qubit", "per_qubit")):
        z = []
        for p in pred.values():
            for d, (v, s) in p[stored]["X"]["per_round"].items():
                v2, s2 = p[rerun]["X"]["per_round"][d]
                z.append((v2 - v) / sqrt(s**2 + s2**2))
        out[f"{rerun}: z of rerun minus stored, every eps"] = {
            "n": len(z),
            "rms": float(sqrt(np.mean(np.square(z)))),
            "max_abs": float(np.max(np.abs(z))),
        }
    return out


def summarize() -> dict:
    runs = result(RUNS_FILE.name)
    pred = predictions(runs)
    models = ["stored median", "stored per_qubit", *VARIANTS]
    rows = pair_days(pred)
    pairs = per_pair(rows, models)
    out = {
        "what": (__doc__ or "").split("\n\n")[0],
        "q150": q150_summary(runs),
        "dephasing_by_patch": dephasing_by_patch(runs),
        "shrinkage": shrinkage_summary(runs),
        "reruns_check": reruns_check(pred),
        "pair_days": rows,
        "per_pair": pairs,
        "placement": placement(pairs, models),
        "share_of_the_per_qubit_shift_from_q150 (pair 1 Lambda_X, reruns)": (
            pairs["pair 1"]["q150"]["lambda"]["mean"] - pairs["pair 1"]["median"]["lambda"]["mean"]
        )
        / (
            pairs["pair 1"]["per_qubit"]["lambda"]["mean"]
            - pairs["pair 1"]["median"]["lambda"]["mean"]
        ),
        "tests": tests(pred, models),
        "sigma_f": sigma_f(),
        "slope": slopes(),
    }
    return out


def report(s: dict) -> None:
    print("q150 by day: T2; its f and its Z-or-Y error per round under each variant")
    for q in s["q150"].values():
        print(
            f"  day {q['day']} T2 {q['t2_us']:.2f} us, own f {q['own_f'][0]:.2f}+-{q['own_f'][1]:.2f}: "
            + ", ".join(
                f"{v} {q['f'][v]:.2f} {100 * q['z_or_y_per_round'][v]:.1f}%" for v in VARIANTS
            )
        )
    print("\nMean Z-or-Y error per round over pair 1's data qubits (%), d3 / d5:")
    for name, by in s["dephasing_by_patch"].items():
        if name.endswith("-r1"):
            print(
                f"  {name}: "
                + ", ".join(
                    f"{v} {100 * by['d3']['mean'][v]:.1f} / {100 * by['d5']['mean'][v]:.1f}"
                    for v in VARIANTS
                )
            )
    print("\nPair 1 predicted eps3, eps5 (%) and Lambda_X by day | mean:")
    for m, v in s["per_pair"]["pair 1"].items():
        e3, e5, lam = v["eps3"], v["eps5"], v["lambda"]
        print(
            f"  {m:17s} eps3 "
            + " ".join(f"{100 * x:5.2f}" for x in e3["by_day"])
            + f" | {100 * e3['mean']:5.2f}  eps5 "
            + " ".join(f"{100 * x:5.2f}" for x in e5["by_day"])
            + f" | {100 * e5['mean']:5.2f}  Lambda "
            + " ".join(f"{x:.3f}" for x in lam["by_day"])
            + f" | {lam['mean']:.3f}+-{lam['sim_se_of_mean']:.3f}"
        )
    print("\nPair 2 predicted Lambda_X mean:")
    for m, v in s["per_pair"]["pair 2"].items():
        print(f"  {m:17s} {v['lambda']['mean']:.3f}+-{v['lambda']['sim_se_of_mean']:.3f}")
    p = s["placement"]
    o = p["observed (WL, inverse-variance over the days; placements.py)"]
    print(f"\nPlacement difference, observed {o['value']:.3f} +- {o['se']:.3f}; predicted:")
    for m in s["per_pair"]["pair 1"]:
        v = p[m]
        print(
            f"  {m:17s} {v['pair 1']:.3f} - {v['pair 2']:.3f} = {v['predicted_difference']:.3f}"
            f" +- {v['sim_se']:.3f}   z {v['z (with sim se)']:+.1f}"
        )
    dec = p["observed plain means, device decoded with per-qubit priors"]
    print(
        f"  observed plain means {p['observed plain means']['difference']:.3f}; "
        f"decoded with per-qubit priors {dec['difference']:.3f}"
    )
    print(
        f"\nShare of pair 1's per-qubit Lambda_X shift that q150 alone gives: "
        f"{s['share_of_the_per_qubit_shift_from_q150 (pair 1 Lambda_X, reruns)']:.2f}"
    )
    print("\nX tests, chi2 (agreeing of 10), sigma_f prespecified | idle noise only:")
    for m in s["tests"]["prespecified"]:
        print(
            f"  {m:17s} "
            + "  ".join(
                f"{q} {s['tests']['prespecified'][m][q]['chi2']:5.1f} ({s['tests']['prespecified'][m][q]['agree']})"
                f" | {s['tests']['idle_noise_only'][m][q]['chi2']:5.1f} ({s['tests']['idle_noise_only'][m][q]['agree']})"
                for q in X_TESTS
            )
        )
    sf = s["sigma_f"]
    print(
        f"\nsigma_f prespecified (ddof 0) {sf['range prespecified (ddof 0)'][0]:.3f}-{sf['range prespecified (ddof 0)'][1]:.3f}; "
        f"ddof 1 {sf['range same with ddof 1'][0]:.3f}-{sf['range same with ddof 1'][1]:.3f}; "
        f"noise only (ddof 1) {sf['range idle_noise_only (bootstrap, ddof 1)'][0]:.3f}-"
        f"{sf['range idle_noise_only (bootstrap, ddof 1)'][1]:.3f}"
    )
    print("\nSlopes above f = 1.05 (line / secant / parabola, +- MC sd):")
    for name, j in s["slope"]["by_job"].items():
        for q, c in j.items():
            if q == "f":
                continue
            print(
                f"  {name} f={j['f']:.3f} {q:6s} {c['slope']:.4f} / {c['slope_secant_from_1']:.4f}"
                f"+-{c['slope_secant_from_1 mc_sd']:.4f} / {c['slope_parabola']:.4f}+-{c['slope_parabola mc_sd']:.4f}"
                f"   line/secant {c['line_over_secant']:.2f}, line/parabola {c['line_over_parabola']:.2f}"
            )
    print("X tests with the f term from each slope (line / secant / parabola), chi2 (agreeing):")
    for where, by_label in s["slope"]["tests"].items():
        for label, by in by_label.items():
            print(
                f"  {where:25s} {label:16s} "
                + "  ".join(
                    f"{q} "
                    + " / ".join(f"{by[k][q]['chi2']:.1f} ({by[k][q]['agree']})" for k in by)
                    for q in X_TESTS
                )
            )
    key = "sigma_M prespecified over noise-only sigma_f with the secant above f = 1.05 [min, median, max]"
    print(f"  {key}: {s['slope'][key]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--jobs", nargs="*", help="e.g. 2026-10-04-r1; default all")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.recompute:
        recompute(a.jobs, a.workers)
        if not assemble():
            return
    out = summarize()
    write_json("f_models.json", out)
    report(out)


if __name__ == "__main__":
    main()

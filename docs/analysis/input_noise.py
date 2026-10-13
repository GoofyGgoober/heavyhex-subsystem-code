"""What the errors of IBM's reported inputs add to sigma_M of the X tests (Sec. tests, App. robust).

The prespecified sigma_M (Eq. z) takes the reported calibration as exact. If the model is right
with the device's true inputs, the reported ones are estimates, and their errors move each
prediction by, to first order,

    delta M = M sum_i e_i delta ln x_i,    e_i = d ln M / d ln x_i,

over every input of the patch: each qubit's readout error (its readouts and resets), T1, T2 and
single-qubit error, and each coupler's CZ error. One calibration serves a day's three jobs, so
these errors are shared by the day's two pair-days wherever their patches share qubits (pair 2's
distance-3 patch lies inside placement 1, pair 1's distance-5 patch, every day).

  gradients  d P_L(n) / d ln x_i for every X-memory circuit (n = 1..8 rounds) of every job, at
             the job's measured f. The circuit's detector error model is sampled with its faults
             (stim, return_errors) and decoded as the analysis decodes (run._decode), and
             d P_L / d p_j = Cov(F, e_j) / (p_j (1 - p_j)) for each error mechanism j (F a
             failure, e_j the mechanism firing). d p_j / d ln x_i follows from stim's
             explanation of each mechanism into circuit faults, each fault's probability
             recomputed from the inputs as heavyhex.simulation.scheduled sets it (checked: the
             mechanisms rebuilt from their faults match stim's to 1e-9). Only the mechanisms of
             the patch whose failure is counted enter. Two independent halves of the shots, A
             and B, so that a sum of squares is estimated as sum e^A e^B, without its noise.
             e_i then follows through the fit of Eq. (fit) linearised at the predicted curve
             (weights of Eq. wlvar; the binomial fit has the same first order).
  check      sum_i e_i with every rate times k (T1 and T2 divided by k) against error_scale.py's
             d ln eps / d ln k at k = 1, curve by curve
  inputs     sigma_i = sd of ln x_i, three cases: (shot) the readout errors' binomial error from
             4096 shots per prepared state (input_resolution.py), every other input exact;
             (hour) every readout error at the scatter between the two readout calibrations
             IBM made 56 minutes apart on day 1 (16:09 and 17:05 UTC, in the git history as
             prepared that day; the 17:05 values were still in force at the run), every other
             input exact; (days) every input at the typical day-to-day scatter of its kind, the
             median over the patches' qubits or couplers of the sd of ln x over the five days,
             which also holds real changes and so is an upper estimate
  tests      sigma_in^2 = M^2 sum_i e_i^2 sigma_i^2 for each X test and pair-day, and the
             covariance of two pair-days of one day; each X test's chi2 with sigma_in added to
             sigma_M, diagonal and with the covariance (r' C^-1 r), against chi2_{10,0.95},
             both fits, sigma_f prespecified and from the idle test's noise

    python docs/analysis/input_noise.py               # the summary from results/input_noise_runs.json, seconds
    python docs/analysis/input_noise.py --recompute   # the gradients first, 3 min on 2 workers, 0.5 GB each
                                                      # [--jobs 2026-10-04-r1 ...] resumes from cache/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from math import log1p, sqrt
from multiprocessing import Pool

import numpy as np
import pymatching
import stim
from scipy import sparse, stats

from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import decoupled_t2, idle_error, single_qubit_error

import prespecified_tests as PT
from common import CACHE, REPO, RESULTS, WORKERS, jobs, memory_index, read, result, write_json
from properties import readout_split

SHOTS = 50_000  # per half, per circuit
STEP = 1e-5  # relative step of the numerical derivatives in an input
RUNS_FILE = RESULTS / "input_noise_runs.json"
PARTS = CACHE / "input_noise"  # one file per job, so a stopped run resumes
KINDS = ("readout", "t1", "t2", "cz", "sx")
SHOTS_PER_STATE = 4096  # IBM's readout calibration (input_resolution.py)
HOUR_APART = (  # day 1's two readout calibrations, as pulled while the day was prepared
    "86b4478:runs/ibm_fez-2026-10-03/calibration.json",  # IBM's 16:09 UTC calibration
    "12826a5:docs/figures/fez_calibration.json",  # 17:05 UTC, still in force at the run
)
X_TESTS = PT.X_TESTS
FITS = ("binomial", "wls")


def floats(x) -> list:
    return [float(v) for v in x]


# ------------------------------------------------------------------------------ the model
def depolarize_fault(p: float, paulis: int) -> float:
    """stim's independent probability per Pauli of DEPOLARIZE1 (3) or DEPOLARIZE2 (15)."""
    half = 2 if paulis == 3 else 8  # how many of the Paulis anticommute with a given one
    return (1 - (1 - (paulis + 1) * p / paulis) ** (1 / half)) / 2


def pauli_fault(px: float, py: float, pz: float, which: str) -> float:
    """stim's independent X, Y or Z probability equivalent to PAULI_CHANNEL_1(px, py, pz)."""
    cx, cy, cz = 1 - 2 * (py + pz), 1 - 2 * (px + pz), 1 - 2 * (px + py)
    r = {"X": cy * cz / cx, "Y": cx * cz / cy, "Z": cx * cy / cz}[which]
    return (1 - sqrt(r)) / 2


def fault_probability(kind: str, args: list[float], pauli: str | None) -> float:
    if kind == "DEPOLARIZE2":
        return depolarize_fault(args[0], 15)
    if kind == "DEPOLARIZE1":
        return depolarize_fault(args[0], 3)
    if kind == "PAULI_CHANNEL_1":
        return pauli_fault(*args, pauli)
    return args[0]  # X_ERROR after a reset, M(p)


def noise_sources(model: stim.Circuit, calibration, f: float) -> dict:
    """(instruction, first target) -> (input keys, channel arguments as a function of the inputs'
    factors, channel), for every noise channel of the model, as scheduled.noisy_scheduled sets it.

    An idle's length is read back from its X probability, and whether it was decoupled from its
    Z probability; every channel is checked against the model's own arguments.
    """
    out = {}
    for o, ins in enumerate(model):
        name, args = ins.name, ins.gate_args_copy()
        targets = [t.value for t in ins.targets_copy()]
        if name == "DEPOLARIZE2":
            for s in range(0, len(targets), 2):
                pair = tuple(sorted(targets[s : s + 2]))
                c = calibration.cz[pair]
                out[o, s] = ([("cz", pair)], lambda x, c=c: [min(1.25 * c * x[0], 15 / 16)], name)
        elif name == "DEPOLARIZE1":
            for s, q in enumerate(targets):
                e = single_qubit_error(calibration, q)
                key = ("sx", q) if (calibration.single_qubit or {}).get(q) is not None else None
                out[o, s] = ([key], lambda x, e=e: [min(1.5 * e * x[0], 0.75)], name)
        elif name in ("X_ERROR", "M") and args:
            for s, q in enumerate(targets):
                r = calibration.readout[q]
                out[o, s] = ([("readout", q)], lambda x, r=r: [r * x[0]], name)
        elif name == "PAULI_CHANNEL_1":
            for s, q in enumerate(targets):
                t1, t2 = calibration.t1_us[q] * 1e3, calibration.t2_us[q] * 1e3
                t = -t1 * log1p(-4 * args[0])

                def channel(x, t=t, t1=t1, t2=t2, dd=False):
                    a = t1 * x[0]
                    b = min(t2 * x[1], 2 * a)
                    px, pz = idle_error(t, a, decoupled_t2(a, b, f) if dd else b)
                    return [px, px, pz]

                fits = [
                    fun
                    for fun in (channel, lambda x, c=channel: c(x, dd=True))
                    if np.allclose(fun([1.0, 1.0]), args, rtol=1e-9, atol=1e-15)
                ]
                assert fits, (o, q, args)
                out[o, s] = ([("t1", q), ("t2", q)], fits[-1], name)
            continue
        else:
            continue
        for s in range(0, len(targets), 2 if name == "DEPOLARIZE2" else 1):
            assert np.isclose(out[o, s][1]([1.0])[0], args[0], rtol=1e-12)
    return out


def symptom(targets) -> tuple:
    dets: set[int] = set()
    obs = 0
    for t in targets:
        t = getattr(t, "dem_target", t)
        if t.is_relative_detector_id():
            dets ^= {t.val}
        elif t.is_logical_observable_id():
            obs ^= 1 << t.val
    return tuple(sorted(dets)), obs


def mechanism_inputs(model: stim.Circuit, dem: stim.DetectorErrorModel, calibration, f: float):
    """Each mechanism's probability, symptom and d p_j / d ln x_i (sparse, mechanisms x inputs).

    p_j is the XOR of its faults, p_j = (1 - prod_k (1 - 2 q_k)) / 2, so
    d p_j / d q_k = (1 - 2 p_j) / (1 - 2 q_k).
    """
    sources = noise_sources(model, calibration, f)
    index, probs, symptoms = {}, [], []
    for ins in dem.flattened():
        if ins.type == "error":
            key = symptom(ins.targets_copy())
            assert key not in index
            index[key] = len(probs)
            probs.append(ins.args_copy()[0])
            symptoms.append(key)
    probs = np.array(probs)
    keys: list = []
    column: dict = {}
    rows, cols, vals = [], [], []
    xor = np.ones(len(probs))
    explained = model.explain_detector_error_model_errors(reduce_to_one_representative_error=False)
    assert len(explained) == len(probs)
    for ex in explained:
        j = index[symptom(ex.dem_error_terms)]
        for loc in ex.circuit_error_locations:
            (frame,) = loc.stack_frames  # no REPEAT blocks
            inputs, channel, kind = sources[
                frame.instruction_offset, loc.instruction_targets.target_range_start
            ]
            pauli = None
            if kind == "PAULI_CHANNEL_1":
                (p,) = loc.flipped_pauli_product
                g = p.gate_target
                pauli = "X" if g.is_x_target else "Y" if g.is_y_target else "Z"
            ones = [1.0] * len(inputs)
            q = fault_probability(kind, channel(ones), pauli)
            xor[j] *= 1 - 2 * q
            for a, key in enumerate(inputs):
                if key is None:  # no reported value: the model's default
                    continue
                up, down = list(ones), list(ones)
                up[a], down[a] = np.exp(STEP), np.exp(-STEP)
                dq = (
                    fault_probability(kind, channel(up), pauli)
                    - fault_probability(kind, channel(down), pauli)
                ) / (2 * STEP)
                if key not in column:
                    column[key] = len(keys)
                    keys.append(key)
                rows.append(j)
                cols.append(column[key])
                vals.append(dq / (1 - 2 * q))
    rebuilt = (1 - xor) / 2
    mismatch = float(np.max(np.abs(rebuilt - probs) / probs))
    assert mismatch < 1e-9, mismatch
    vals = np.array(vals) * (1 - 2 * probs[np.array(rows, dtype=int)])
    jac = sparse.csr_matrix((vals, (rows, cols)), shape=(len(probs), len(keys)))
    return probs, symptoms, jac, keys, mismatch


def failure_gradients(dem, matching, probs, owner, shots: int, seed: int) -> list[tuple]:
    """Per half: d P_L / d p_j for each observable (observables x mechanisms), zero outside the
    observable's patch, and the failure rate."""
    out = []
    for half in range(2):
        sampler = dem.compile_sampler(seed=seed + half)
        det, obs, err = sampler.sample(shots, return_errors=True)
        failed = (matching.decode_batch(det).astype(bool) != obs).astype(np.float32)
        centred = failed - failed.mean(0)
        g = np.zeros((failed.shape[1], len(probs)))
        for start in range(0, shots, 10_000):
            e = err[start : start + 10_000].astype(np.float32) - probs.astype(np.float32)
            g += centred[start : start + 10_000].T @ e
        g /= shots * probs * (1 - probs)
        g[owner[None, :] != np.arange(failed.shape[1])[:, None]] = 0.0
        out.append((g, failed.mean(0)))
    return out


def fit_jacobian(P: np.ndarray) -> tuple[float, np.ndarray]:
    """eps of a curve P(n) over ROUNDS and d eps / d P(n), Eq. (fit) linearised with the weights
    of Eq. (wlvar)."""
    n = np.array(ROUNDS, float)
    w = (1 - 2 * P) ** 2 / (P * (1 - P))
    nbar = (w * n).sum() / w.sum()
    c = w * (n - nbar) / (w * (n - nbar) ** 2).sum()
    b = float((c * np.log(1 - 2 * P)).sum())
    eps = (1 - np.exp(b)) / 2
    return eps, (1 - 2 * eps) * c / (1 - 2 * P)


def job_part(name: str) -> dict:
    """Every input's elasticity of each X-memory eps of the job's patches, halves A and B."""
    job = next(j for j in jobs() if j.name == name)
    pred = job.analysis["predicted_at_measured_dephasing"]
    patches = list(pred["fits"]["X"]["per_round"])
    P = np.array([pred["settings"][memory_index("X", n)]["logical_error"] for n in ROUNDS])
    dP: list = [[{} for _ in patches] for _ in range(2)]  # half, patch -> key -> per n
    sampled = np.zeros((2, len(ROUNDS), len(patches)))
    worst = 0.0
    for r, n in enumerate(ROUNDS):
        model, _, labels = job.model(memory_index("X", n))
        dem = model.detector_error_model(decompose_errors=False)
        matching = pymatching.Matching.from_detector_error_model(
            model.detector_error_model(decompose_errors=True)  # run._decode
        )
        probs, symptoms, jac, keys, mismatch = mechanism_inputs(model, dem, job.calibration, job.f)
        worst = max(worst, mismatch)
        assert list(dict.fromkeys(str(p) for p, _, _ in labels)) == patches  # observable order
        patch_of = [patches.index(str(p)) for p, _, _ in labels]
        owner = np.array(
            [
                patch_of[d[0]] if d else int(np.log2(o)) if o and not o & (o - 1) else -1
                for d, o in symptoms
            ]
        )
        for d, o in symptoms:  # no mechanism touches two patches
            assert len({patch_of[x] for x in d} | ({int(np.log2(o))} if o else set())) <= 1
        halves = failure_gradients(dem, matching, probs, owner, SHOTS, seed=1000 * r + 7)
        for h, (g, rate) in enumerate(halves):
            sampled[h, r] = rate
            d = (jac.T @ g.T).T  # observables x inputs
            for k in range(len(patches)):
                for a in np.flatnonzero(d[k]):
                    dP[h][k].setdefault(keys[a], np.zeros(len(ROUNDS)))[r] = d[k, a]
    out: dict = {"job": name, "f": job.f, "fault_rebuild_mismatch": worst, "patches": {}}
    for k, patch in enumerate(patches):
        eps, jfit = fit_jacobian(P[:, k])
        inputs = sorted(set(dP[0][k]) | set(dP[1][k]))
        elastic = [
            [float(jfit @ dP[h][k].get(key, np.zeros(len(ROUNDS))) / eps) for key in inputs]
            for h in range(2)
        ]
        se = np.sqrt(P[:, k] * (1 - P[:, k]) / SHOTS)
        out["patches"][patch] = {
            "eps_from_curve": float(eps),
            "eps_stored": pred["fits"]["X"]["per_round"][patch][0],
            "sampled_minus_stored_P_in_se": ((sampled[:, :, k] - P[:, k]) / se).tolist(),
            "inputs": [[kind, list(i) if isinstance(i, tuple) else i] for kind, i in inputs],
            "elasticity": elastic,
        }
    return out


def recompute(names: list[str], workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (PARTS / f"{n}.json").exists()]
    t0 = time.time()
    with Pool(workers) as pool:
        for part in pool.imap_unordered(job_part, todo):
            (PARTS / f"{part['job']}.json").write_text(json.dumps({"shots": SHOTS, **part}) + "\n")
            print(f"{part['job']} ({time.time() - t0:.0f} s)", flush=True)
    found = [json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))]
    runs = {p["job"]: p for p in found if p["shots"] == SHOTS}
    RUNS_FILE.write_text(
        json.dumps({"shots_per_half": SHOTS, "rounds": list(ROUNDS), "jobs": runs}) + "\n"
    )
    print("wrote", RUNS_FILE, f"({len(runs)} jobs)")


# ----------------------------------------------------------------------------- the inputs
def as_key(kind: str, i) -> tuple:
    return (kind, tuple(i) if isinstance(i, list) else i)


def calibrations() -> dict[str, dict]:
    """The calibration in force each day, checked to be the one all of the day's jobs used."""
    out: dict = {}
    for job in jobs():
        cal = read(job.folder / "calibration.json")
        assert out.setdefault(job.day, cal) == cal
    return out


def value(cal: dict, kind: str, i) -> float | None:
    if kind == "readout":
        return cal["readout"][str(i)]
    if kind in ("t1", "t2"):
        return cal[f"{kind}_us"][str(i)]
    if kind == "cz":
        return cal["cz"][f"{i[0]}-{i[1]}"]
    return (cal.get("single_qubit") or {}).get(str(i))


def shot_noise(day: str, cal: dict, q: int) -> float:
    """sd of ln(readout error) from IBM's 4096 shots per prepared state (input_resolution.py)."""
    e = cal["readout"][str(q)]
    split = readout_split(day)
    if q in split and (split[q][0] + split[q][1]) / 2 == e:
        p10, p01 = split[q]
        sigma = sqrt(p10 * (1 - p10) + p01 * (1 - p01)) / (2 * sqrt(SHOTS_PER_STATE))
    else:
        sigma = sqrt(e * (1 - e) / (2 * SHOTS_PER_STATE))
    return sigma / e


def git_json(spec: str) -> dict:
    shown = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, check=True)
    return json.loads(shown.stdout)


def hour_scatter(used: set[int]) -> dict:
    """sd of ln(readout error) per calibration from two calibrations 56 minutes apart."""
    a, b = (git_json(s) for s in HOUR_APART)
    assert a["calibrated"].startswith("2026-10-03 19:09") and b["calibrated"].startswith(
        "2026-10-03 20:05"
    )
    qubits = sorted(used)
    ra = np.array([a["readout"][str(q)] for q in qubits])
    rb = np.array([b["readout"][str(q)] for q in qubits])
    d = np.log(ra) - np.log(rb)
    shot = np.array([(1 - r) / (2 * SHOTS_PER_STATE * r) for r in (ra + rb) / 2])
    return {
        "qubits": len(qubits),
        "changed": int((ra != rb).sum()),
        "sd_per_calibration (rms)": float(np.sqrt((d**2).mean() / 2)),
        "sd_per_calibration (robust, 1.4826 MAD)": float(
            1.4826 * np.median(np.abs(d - np.median(d))) / sqrt(2)
        ),
        "shot_noise_sd (rms of the bound)": float(np.sqrt(shot.mean())),
    }


def day_scatter(cals: dict, used: dict[str, set]) -> dict:
    """Per kind: the sd of ln x over the five days for each qubit or coupler the patches used."""
    out = {}
    for kind, keys in used.items():
        sds = []
        for i in sorted(keys, key=str):
            v = [value(c, kind, i) for c in cals.values()]
            if all(x is not None and x > 0 for x in v):
                sds.append(float(np.std(np.log(v), ddof=1)))
        sds_arr = np.array(sds)
        out[kind] = {
            "components": len(sds),
            "median": float(np.median(sds_arr)),
            "rms": float(np.sqrt((sds_arr**2).mean())),
            "quartiles": np.percentile(sds_arr, [25, 75]).tolist(),
        }
    return out


# ------------------------------------------------------------------------------- summary
class Gradients:
    """Every pair-day's elasticities of eps3 and eps5 (halves A and B) keyed by (day, input)."""

    def __init__(self, D: PT.Data, runs: dict) -> None:
        self.n = len(D.reps)
        self.eps: dict = {}
        for i in range(self.n):
            day = D.reps[i]["day"]
            for d in "35":
                part = runs["jobs"][D.job(i, d).removeprefix("ibm_fez-")]["patches"][d]
                keys = [(day, as_key(*k)) for k in part["inputs"]]
                self.eps[i, d] = [dict(zip(keys, part["elasticity"][h])) for h in range(2)]

    def vector(self, i: int, q: str, h: int) -> dict:
        if q == "X eps3":
            return self.eps[i, "3"][h]
        if q == "X eps5":
            return self.eps[i, "5"][h]
        out = dict(self.eps[i, "3"][h])
        for k, v in self.eps[i, "5"][h].items():
            out[k] = out.get(k, 0.0) - v
        return out


def covariance(G: Gradients, q: str, M: np.ndarray, sigma: dict) -> np.ndarray:
    """M_a M_b sum_i sigma_i^2 (e_a^A e_b^B + e_a^B e_b^A) / 2 over the ten pair-days."""
    C = np.zeros((G.n, G.n))
    vec = [[G.vector(i, q, h) for h in range(2)] for i in range(G.n)]
    for a in range(G.n):
        for b in range(a, G.n):
            s = 0.0
            for h in range(2):
                va, vb = vec[a][h], vec[b][1 - h]
                s += sum(v * vb[k] * sigma.get(k, 0.0) ** 2 for k, v in va.items() if k in vb)
            C[a, b] = C[b, a] = M[a] * M[b] * s / 2
    return C


def by_kind(G: Gradients, i: int, q: str) -> dict:
    """sum over each kind of inputs of e^A e^B: sigma_in^2 / M^2 per unit relative error."""
    a, b = G.vector(i, q, 0), G.vector(i, q, 1)
    return {
        kind: float(sum(v * b.get(k, 0.0) for k, v in a.items() if k[1][0] == kind))
        for kind in KINDS
    }


def chi2_with(r, base, C) -> dict:
    total = np.diag(base**2) + C
    z = r / np.sqrt(np.diag(total))
    full = float(r @ np.linalg.solve(total, r))
    return {
        "chi2_diagonal": float((z**2).sum()),
        "p_diagonal": float(stats.chi2.sf((z**2).sum(), len(r))),
        "abs_z_below_2": int((np.abs(z) < 2).sum()),
        "chi2_with_covariance": full,
        "p_with_covariance": float(stats.chi2.sf(full, len(r))),
    }


def check(runs: dict) -> dict:
    """sum_i e_i at k (rates times k) against error_scale.py's d ln eps / d ln k at k = 1."""
    rows = []
    curves = result("error_scale.json")["per_curve"]
    stored = {(r["job"], r["d"]): r for r in curves if r["basis"] == "X"}
    for name, part in runs["jobs"].items():
        for d, p in part["patches"].items():
            sign = np.array([-1.0 if k in ("t1", "t2") else 1.0 for k, _ in p["inputs"]])
            mine = [float((sign * np.array(e)).sum()) for e in p["elasticity"]]
            ref = stored[name, int(d)]
            rows.append(
                {
                    "job": name,
                    "d": int(d),
                    "here (halves A, B)": mine,
                    "error_scale (binomial, wls)": [
                        ref["k_binomial"]["elasticity_at_1"],
                        ref["k_wls"]["elasticity_at_1"],
                    ],
                    "eps_from_curve_minus_stored": p["eps_from_curve"] - p["eps_stored"],
                    "sampled P against stored, largest |z|": float(
                        np.abs(p["sampled_minus_stored_P_in_se"]).max()
                    ),
                }
            )
    here = np.array([np.mean(r["here (halves A, B)"]) for r in rows])
    there = np.array([np.mean(r["error_scale (binomial, wls)"]) for r in rows])
    halves = np.array([r["here (halves A, B)"] for r in rows])
    return {
        "median here / error_scale": [float(np.median(here)), float(np.median(there))],
        "median |here - error_scale|": float(np.median(np.abs(here - there))),
        "rms half A - half B": float(np.sqrt(((halves[:, 0] - halves[:, 1]) ** 2).mean())),
        "largest fault-rebuild mismatch": max(
            p["fault_rebuild_mismatch"] for p in runs["jobs"].values()
        ),
        "largest |eps from curve - stored|": float(
            max(abs(r["eps_from_curve_minus_stored"]) for r in rows)
        ),
        "largest |z| of sampled P against stored": max(
            r["sampled P against stored, largest |z|"] for r in rows
        ),
        "per_curve": rows,
    }


def x_tests(D: PT.Data, G: Gradients, sigma: dict, fit: str, label: str) -> dict:
    """The three X tests with sigma_in added to sigma_M, for one fit and one sigma_f."""
    rows = PT.x_rows(D, fit, label)
    stored = result("prespecified_tests.json")["chi2"][fit][label]
    out = {}
    for q in X_TESTS:
        O, M = (np.array([r[q][k] for r in rows]) for k in ("observed", "predicted"))
        sM = np.array([PT.sigma_M(r[q]) for r in rows])
        base = np.hypot([r[q]["observed_uncertainty"] for r in rows], sM)
        C = covariance(G, q, M, sigma)
        assert np.linalg.eigvalsh(np.diag(base**2) + C).min() > 0
        before = float((((O - M) / base) ** 2).sum())
        assert abs(before - stored[q]["chi2"]) < 1e-9
        s_in = np.sqrt(np.diag(C))
        total = np.sqrt(base**2 + np.diag(C))
        out[q] = {
            "chi2_as_tested": before,
            **chi2_with(O - M, base, C),
            "sigma_in_over_M": floats(s_in / M),
            "sigma_in_over_sigma_M": floats(s_in / sM),
            "within-day correlation of the two pair-days' z": floats(
                C[i, i + 1] / (total[i] * total[i + 1]) for i in range(0, len(rows), 2)
            ),
        }
    return out


def summarize() -> dict:
    runs = read(RUNS_FILE)
    D = PT.Data()
    G = Gradients(D, runs)
    cals = calibrations()
    used: dict[str, set] = {k: set() for k in KINDS}
    for _, (kind, i) in {k for v in G.eps.values() for h in v for k in h}:
        used[kind].add(i)
    day1 = {q for j in jobs() if j.day == "2026-10-04" for c in j.info["chips"].values() for q in c}
    hour = hour_scatter(day1)
    days = day_scatter(cals, used)
    sigmas = {
        "shot": {
            (day, ("readout", q)): shot_noise(day, cal, q)
            for day, cal in cals.items()
            for q in used["readout"]
        },
        "hour": {
            (day, ("readout", q)): hour["sd_per_calibration (rms)"]
            for day in cals
            for q in used["readout"]
        },
        "days": {
            (day, (kind, i)): days[kind]["median"]
            for day in cals
            for kind, keys in used.items()
            for i in keys
        },
    }
    shot = np.array(list(sigmas["shot"].values()))
    out: dict = {
        "what": "sigma_in, the prediction's error from the errors of IBM's inputs, to first order "
        "in each input, added to sigma_M with the covariance between a day's two pair-days. In "
        "the case 'days' the T2 term overlaps the f term of sigma_M, which also covers how the "
        "qubits' dephasing departs from the reported T2.",
        "shots_per_half": runs["shots_per_half"],
        "check": check(runs),
        "inputs": {
            "shot: readout errors' binomial sd of ln, median and range over the qubit-days": [
                float(np.median(shot)),
                [float(shot.min()), float(shot.max())],
            ],
            "hour: day 1's two readout calibrations 56 minutes apart (the jobs' qubits)": hour,
            "days: sd of ln x over the five days, by kind (the patches' qubits and couplers)": days,
        },
        "cases": {
            case: {
                f"{fit}, sigma_f {label}": x_tests(D, G, sigma, fit, label)
                for fit in FITS
                for label in ("prespecified", "idle_noise_only")
            }
            for case, sigma in sigmas.items()
        },
    }
    out["per unit relative error, by kind (sum e^A e^B), WL"] = [
        {"day": int(D.day[i]), "pair": int(D.pair[i]), **{q: by_kind(G, i, q) for q in X_TESTS}}
        for i in range(G.n)
    ]
    out["largest elasticities (mean of the halves), day 1"] = {
        f"pair {D.pair[i]} {q}": largest(G, i, q)
        for i in range(G.n)
        if D.day[i] == 1
        for q in X_TESTS
    }
    return {"headline": headline(out), **out}


def headline(res: dict) -> dict:
    """chi2 as tested -> with sigma_in (diagonal, with the covariance), and sigma_in / M."""
    out: dict = {}
    for case, blk in res["cases"].items():
        out[case] = {}
        for key in ("wls, sigma_f prespecified", "binomial, sigma_f prespecified"):
            out[case][key] = {
                q: {
                    "chi2 [as tested, diagonal, covariance]": [
                        v["chi2_as_tested"],
                        v["chi2_diagonal"],
                        v["chi2_with_covariance"],
                    ],
                    "p [diagonal, covariance]": [v["p_diagonal"], v["p_with_covariance"]],
                    "sigma_in / M range": [min(v["sigma_in_over_M"]), max(v["sigma_in_over_M"])],
                }
                for q, v in blk[key].items()
            }
    return out


def largest(G: Gradients, i: int, q: str, many: int = 5) -> list:
    a, b = G.vector(i, q, 0), G.vector(i, q, 1)
    mean = {k[1]: (v + b.get(k, 0.0)) / 2 for k, v in a.items()}
    top = sorted(mean.items(), key=lambda kv: -abs(kv[1]))[:many]
    return [[kind, list(x) if isinstance(x, tuple) else x, e] for (kind, x), e in top]


def report(res: dict) -> None:
    c = res["check"]
    print(
        f"check: sum of elasticities {c['median here / error_scale'][0]:.2f} against error_scale "
        f"{c['median here / error_scale'][1]:.2f} "
        f"(median |diff| {c['median |here - error_scale|']:.2f})"
    )
    for case, blk in res["cases"].items():
        t = blk["wls, sigma_f prespecified"]
        print(
            f"{case:5s} "
            + "  ".join(
                f"{q} {v['chi2_as_tested']:.1f} -> {v['chi2_diagonal']:.1f} / "
                f"{v['chi2_with_covariance']:.1f}"
                for q, v in t.items()
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="the gradients first (heavy)")
    ap.add_argument("--jobs", nargs="*", help="job names (default all fifteen), e.g. 2026-10-06-r1")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.recompute:
        recompute(a.jobs or [j.name for j in jobs()], a.workers)
    res = summarize()
    write_json("input_noise.json", res)
    report(res)


if __name__ == "__main__":
    main()

"""Do flag and relay streaks predict later detection events and logical failures?

The flags and relays carry a persistent state (refiring.py, hmm.py). If the state touches the
code, a streak should be followed by more detection events on the stabilizers next to the qubit,
and shots with a streak should fail more often than the calibration model says for the same
readouts. Decided after the run; no new decoder. Every memory circuit with two or more rounds
(X and Z memory, 15 jobs, 2500 shots each), decoded as the analysis decoded them (run._decode).

Notation as in the paper: x_{i,t} is the readout of flag or relay i in round t (t = 1..n; the
code's label round is t - 1), and i is in a streak in round t if x_{i,t-1} = x_{i,t} = 1.

A. Detection events after a streak. A unit is one series i (a physical qubit in one role, job
   and circuit) and one round t >= 2. Its shots fall into three groups by (x_{i,t-1}, x_{i,t}):
   streak (1, 1), single (0, 1) and quiet (0, 0). The outcome y is the fraction of the patch's
   stabilizer detectors in round t + lag that fire, over the "near" ones (support holds a data
   qubit that i meets through a CX, common.data_read) or the "far" ones (the rest of the patch).
   lag 1 is the first round wholly after the streak; lag 0 is its last round (in X memory the
   round's X gauges are read before its flags), lag -1 its first, and lag -2 the round before
   it, where the model gives no difference. Per unit the difference of
   group means d_u = ybar_1 - ybar_0 is pooled with Mantel-Haenszel weights,

       D = sum_u w_u d_u / sum_u w_u,    w_u = n_1 n_0 / (n_1 + n_0),

   so a streak is only ever compared with the same qubit, role, job, circuit and round. Units
   with fewer than two shots in either group are left out.
   The model's D uses the same w_u and the calibration model's exact conditional probabilities
   (costreak.ExactModel, no sampling), so chip - model is what the independent-fault model does
   not explain. Errors: (shots) the linearization over shots, a shot entering many units,
   conditional on the groups; (qubits) clustered by physical qubit, CR1, every unit of a qubit
   in any job, role or circuit in one cluster. The stabilizers fire in 20-49% of rounds, so a
   difference D is also given as the flip probability it implies, D / (1 - 2 ybar_0): an extra,
   independent flip with probability q moves a detector's rate from P to P + q (1 - 2P).
B. Logical failures. Strata are (job, circuit, patch). A shot is exposed if any flag or relay of
   the patch has a streak in any round; the logical failure is the prespecified decoder's.
   Mantel-Haenszel risk difference RD (Sato's variance) and risk ratio RR (Greenland-Robins
   variance of ln RR), exposed against shots without a streak. Variants: strata also by the
   number of flag and relay firings K in the patch (streak shots against shots whose flags and
   relays fired as often, never twice running); a streak on flags only or on relays only; the
   longest streak 2 or 3 and more rounds. And the failure per firing: the least-squares slope
   of the failure on K within strata, b = sum_s S_Ky / sum_s S_KK, with a sandwich error.
   The model: the same on 20,000 shots per circuit sampled from it (stim, seed SEED + setting
   index) and decoded the same way.

Chip numbers repeat exactly; the model's in B repeat only with the same stim build. 3 min on
2 workers, under 0.5 GB each. Writes results/streak_effects.json.

    python docs/analysis/streak_effects.py
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pymatching
from scipy.stats import chi2

from heavyhex.experiment.circuits import SETTINGS

from common import (
    WORKERS,
    Job,
    data_read,
    day_number,
    job_folders,
    readout_model,
    readout_series,
    write_json,
)
from costreak import ExactModel

SEED = 7000
MODEL_SHOTS = 20_000
LAGS = (-2, -1, 0, 1, 2, 3)
OUTCOMES = ("near", "far")
CONTRASTS = (("streak", "quiet"), ("streak", "single"), ("single", "quiet"))
EXPOSURES = ("any", "count matched", "flag only", "relay only", "longest 2", "longest 3+")
Z95 = 1.959964
CIRCUITS = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.rounds >= 2]


def support(name: str) -> set[int]:
    return {int(q) for q in re.findall(r"\d+", name)}


def role_of(name: str) -> str:
    return "flag" if name.startswith("z_flag") else "relay"


class Joint:
    """Pr(all of these detectors fire) in the model, exactly, with a product over groups of
    detectors that share no error mechanism."""

    def __init__(self, model) -> None:
        self.exact = ExactModel(model)
        self.mechs = defaultdict(frozenset, {d: frozenset(m) for d, m in self.exact.by_det.items()})
        self.cache: dict = {}

    def __call__(self, *dets: int) -> float:
        key = tuple(sorted(dets))
        if key not in self.cache:
            groups: list[tuple[list[int], set]] = []
            for d in key:
                hit = [g for g in groups if g[1] & self.mechs[d]]
                merged = ([d], set(self.mechs[d]))
                for g in hit:
                    merged[0].extend(g[0])
                    merged[1].update(g[1])
                    groups.remove(g)
                groups.append(merged)
            self.cache[key] = float(np.prod([self.exact.all_fire(sorted(g[0])) for g in groups]))
        return self.cache[key]

    def conditional(self, y: int, a: int, b: int) -> dict[str, float]:
        """Pr(y fires | streak, single, quiet) for the readouts a (round t - 1) and b (round t)."""
        pa, pb, pab = self(a), self(b), self(a, b)
        py, pya, pyb, pyab = self(y), self(y, a), self(y, b), self(y, a, b)
        return {
            "streak": pyab / pab,
            "single": (pyb - pyab) / (pb - pab),
            "quiet": (py - pya - pyb + pyab) / (1 - pa - pb + pab),
        }


def longest_run(x: np.ndarray) -> np.ndarray:
    """Longest run of 1s along the last axis."""
    best = np.zeros(x.shape[:-1], int)
    cur = np.zeros(x.shape[:-1], int)
    for t in range(x.shape[-1]):
        cur = np.where(x[..., t], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def aggregate_keys(basis: str, patch: str, role: str) -> list[tuple[str, str, str]]:
    return [(basis, d, r) for d in (patch, "both") for r in (role, "both")]


def after_streak(job: Job, basis: str, labels, chip: np.ndarray, joint: Joint):
    """Part A for one circuit: unit rows and the per-shot influence of every aggregate."""
    near_far = defaultdict(lambda: ([], []))
    stab = defaultdict(list)
    for c, (p, r, name) in enumerate(labels):
        if " " not in name:
            stab[str(p), int(r)].append((c, support(name)))
    rows, psi = [], defaultdict(lambda: np.zeros(len(chip)))
    for (patch, name, k), cols in readout_series(labels).items():
        role, qubit, data = (
            role_of(name),
            job.readout_qubit(patch, name, k),
            data_read(patch, name, k),
        )
        x = chip[:, cols]
        n = len(cols)
        for t in range(1, n):  # code rounds: the streak ends at t (paper round t + 1)
            prev, now = x[:, t - 1], x[:, t]
            groups = {"streak": prev & now, "single": ~prev & now, "quiet": ~prev & ~now}
            sizes = {g: int(m.sum()) for g, m in groups.items()}
            for lag in LAGS:
                if not 0 <= t + lag <= n:
                    continue
                key = (patch, t + lag, name, k)
                if key not in near_far:
                    for c, s in stab[patch, t + lag]:
                        near_far[key][0 if s & data else 1].append(c)
                for outcome, dcols in zip(OUTCOMES, near_far[key]):
                    if not dcols:
                        continue
                    y = chip[:, dcols].mean(1)
                    model = [joint.conditional(c, cols[t - 1], cols[t]) for c in dcols]
                    for g1, g0 in CONTRASTS:
                        n1, n0 = sizes[g1], sizes[g0]
                        if n1 < 2 or n0 < 2:
                            continue
                        m1, m0 = groups[g1], groups[g0]
                        y1, y0 = y[m1].mean(), y[m0].mean()
                        w = n1 * n0 / (n1 + n0)
                        e = np.zeros(len(chip))
                        e[m1] = w * (y[m1] - y1) / n1
                        e[m0] = -w * (y[m0] - y0) / n0
                        contrast = f"{g1} - {g0}"
                        for agg in aggregate_keys(basis, patch, role):
                            psi[(*agg, lag, outcome, contrast)] += e
                        expected = [np.mean([m[g] for m in model]) for g in (g1, g0)]
                        unit = (basis, patch, role, qubit, lag, outcome, contrast)
                        rows.append((*unit, n1, n0, w, y1, y0, *expected))
    return rows, {k: float((v**2).sum()) for k, v in psi.items()}


def exposures(x: np.ndarray, roles: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Part B exposures of one patch: {name: (exposed, reference)}; x is shots x series x rounds."""
    streak = (x[:, :, 1:] & x[:, :, :-1]).any(2)
    flag = streak[:, [r == "flag" for r in roles]].any(1)
    relay = streak[:, [r == "relay" for r in roles]].any(1)
    anyone = flag | relay
    longest = longest_run(x).max(1)
    none = ~anyone
    return {
        "any": (anyone, none),
        "flag only": (flag & ~relay, none),
        "relay only": (relay & ~flag, none),
        "longest 2": (longest == 2, none),
        "longest 3+": (longest >= 3, none),
    }


def failure_strata(job, basis, idx, patches, labels, det, fail, src) -> tuple[list, list]:
    """Part B for one circuit and source: (stratum, exposure, K, a, n1, c, n0) rows, and per
    stratum the moments of the failure's slope on K, the patch's flag and relay firings:
    (stratum, sum k^2, sum k e, sum k^2 e^2, sum k^3 e, sum k^4) with k = K - mean, e = y - mean."""
    series = readout_series(labels)
    out, moments = [], []
    for p_obs, patch in enumerate(patches):
        keys = [k for k in series if k[0] == patch]
        x = np.stack([det[:, series[k]] for k in keys], axis=1)
        y = fail[:, p_obs]
        base = (src, job.name, day_number(job.day), job.rep, basis, SETTINGS[idx].rounds, patch)
        exp = exposures(x, [role_of(k[1]) for k in keys])
        for name, (e, ref) in exp.items():
            counts = (y[e].sum(), e.sum(), y[ref].sum(), ref.sum())
            out.append((*base, name, None, *map(int, counts)))
        fired = x.sum((1, 2))
        e, ref = exp["any"]
        for K in np.unique(fired[e]):
            m = fired == K
            counts = (y[e & m].sum(), (e & m).sum(), y[ref & m].sum(), (ref & m).sum())
            out.append((*base, "count matched", int(K), *map(int, counts)))
        k, r = fired - fired.mean(), y - y.mean()
        moments.append(
            (*base, *(float(v.sum()) for v in (k**2, k * r, k**2 * r**2, k**3 * r, k**4)))
        )
    return out, moments


def per_firing(moments: list[tuple]) -> dict:
    """Pooled within-stratum least-squares slope of the failure on K, b = sum Sky / sum Skk,
    with the sandwich error sum_i k_i^2 (e_i - b k_i)^2 / (sum Skk)^2."""
    if not moments:
        return {}
    skk, sky, a, b3, c4 = np.array(moments, float).T
    b = sky.sum() / skk.sum()
    se = float(np.sqrt(a.sum() - 2 * b * b3.sum() + b**2 * c4.sum()) / skk.sum())
    return {"slope": float(b), "se": se, "ci95": ci(b, se)}


def job_effects(i: int, model_shots: int = MODEL_SHOTS) -> dict:
    start = time.time()
    job = Job(job_folders()[i])
    units, s2, strata, moments, check = [], defaultdict(float), [], [], []
    for idx in CIRCUITS:
        basis = SETTINGS[idx].basis
        plain, _, plain_labels = job.model(idx)
        model, records, labels = readout_model(job, idx)
        nd = plain.num_detectors
        assert [tuple(map(str, l)) for l in labels[:nd]] == [
            tuple(map(str, l)) for l in plain_labels
        ]
        patches = list(dict.fromkeys(str(p) for p, _, _ in plain_labels))
        matching = pymatching.Matching.from_detector_error_model(
            plain.detector_error_model(decompose_errors=True)
        )  # run._decode
        chip, chip_obs = job.chip_detectors(idx, model, records)
        chip_fail = matching.decode_batch(chip[:, :nd]).astype(bool) != chip_obs
        stored = job.analysis["observed"]["settings"][idx]["logical_error"]
        check.append(float(np.abs(chip_fail.mean(0) - np.array(stored)).max()))
        sim, sim_obs = model.compile_detector_sampler(seed=SEED + idx).sample(
            model_shots, separate_observables=True
        )
        sim_fail = matching.decode_batch(sim[:, :nd]).astype(bool) != sim_obs
        rows, sq = after_streak(job, basis, labels, chip, Joint(model))
        units += rows
        for k, v in sq.items():
            s2[k] += v
        for src, det, fail in (("chip", chip, chip_fail), ("model", sim, sim_fail)):
            rows, mom = failure_strata(job, basis, idx, patches, labels, det, fail, src)
            strata += rows
            moments += mom
    print(f"{job.name} done in {time.time() - start:.0f} s", flush=True)
    return {
        "units": units,
        "s2": dict(s2),
        "strata": strata,
        "moments": moments,
        "check": max(check),
    }


# ---------------------------------------------------------------------------- summaries
def ci(v: float, se: float) -> list[float]:
    return [v - Z95 * se, v + Z95 * se]


def clustered(q: np.ndarray, contrib: np.ndarray, W: float) -> float:
    """CR1 standard error of a weighted mean, clustered by q; contrib = w_u (d_u - D)."""
    e = np.bincount(q, weights=contrib) / W
    G = len(np.unique(q))
    return float(np.sqrt(G / (G - 1) * (e**2).sum()))


def detection_summary(units: list[tuple], s2: dict) -> dict:
    names = ("basis", "patch", "role", "qubit", "lag", "outcome", "contrast")
    cols = {k: np.array([u[j] for u in units]) for j, k in enumerate(names)}
    n1, _, w, y1, y0, m1, m0 = np.array([u[7:] for u in units], float).T
    out: dict = {}
    for (basis, dist, role, lag, outcome, contrast), sq in sorted(s2.items(), key=str):
        sel = (
            (cols["basis"] == basis)
            & ((cols["patch"] == dist) | (dist == "both"))
            & ((cols["role"] == role) | (role == "both"))
            & (cols["lag"] == lag)
            & (cols["outcome"] == outcome)
            & (cols["contrast"] == contrast)
        )
        W = w[sel].sum()
        d, m = (y1 - y0)[sel], (m1 - m0)[sel]
        D, M = (w[sel] * d).sum() / W, (w[sel] * m).sum() / W
        q = cols["qubit"][sel].astype(int)
        G = len(set(q))
        by_q = np.bincount(q, weights=w[sel] * d)[np.unique(q)]
        se_shots = float(np.sqrt(sq) / W)
        se_q, se_q_diff = (
            clustered(q, w[sel] * (d - D), W),
            clustered(q, w[sel] * (d - m - D + M), W),
        )
        g1, g0 = contrast.split(" - ")
        base = 1 - 2 * (w[sel] * y0[sel]).sum() / W
        out.setdefault(f"{basis} d{dist} {role}", {}).setdefault(f"lag {lag}", {}).setdefault(
            outcome, {}
        )[contrast] = {
            "units": int(sel.sum()),
            "physical_qubits": G,
            f"shots_{g1}": int(n1[sel].sum()),
            f"chip_{g1}": float((w[sel] * y1[sel]).sum() / W),
            f"chip_{g0}": float((w[sel] * y0[sel]).sum() / W),
            f"model_{g1}": float((w[sel] * m1[sel]).sum() / W),
            f"model_{g0}": float((w[sel] * m0[sel]).sum() / W),
            "chip": float(D),
            "chip_ci95_shots": ci(D, se_shots),
            "chip_ci95_qubits": ci(D, se_q),
            "model": float(M),
            "chip_minus_model": float(D - M),
            "chip_minus_model_ci95_shots": ci(D - M, se_shots),
            "chip_minus_model_ci95_qubits": ci(D - M, se_q_diff),
            "qubits_with_chip_effect_above_zero": float((by_q > 0).mean()),
            "chip_flip_equivalent": float(D / base),
            "chip_flip_equivalent_ci95_qubits": [v / base for v in ci(D, se_q)],
        }
    return out


def mantel_haenszel(rows: list[tuple]) -> dict:
    """RD (Sato's variance) and RR (Greenland-Robins variance of ln RR) over strata (a, n1, c, n0)."""
    s = np.array([r for r in rows if r[1] > 0 and r[3] > 0], float)
    if not len(s):
        return {"strata": 0}
    a, n1, c, n0 = s.T
    N = n1 + n0
    W = (n1 * n0 / N).sum()
    rd = ((a * n0 - c * n1) / N).sum() / W
    P = (n1**2 * c - n0**2 * a + n1 * n0 * (n0 - n1) / 2) / N**2
    Q = (a * (n0 - c) + c * (n1 - a)) / (2 * N)
    se_rd = float(np.sqrt(max(rd * P.sum() + Q.sum(), 0.0)) / W)
    R, S = (a * n0 / N).sum(), (c * n1 / N).sum()
    out = {
        "strata": len(s),
        "exposed_shots": int(n1.sum()),
        "reference_shots": int(n0.sum()),
        "failure_exposed": float(a.sum() / n1.sum()),
        "failure_reference": float(c.sum() / n0.sum()),
        "rd": float(rd),
        "rd_se": se_rd,
        "rd_ci95": ci(rd, se_rd),
    }
    if R > 0 and S > 0:
        se_log = float(np.sqrt(((n1 * n0 * (a + c) - a * c * N) / N**2).sum() / (R * S)))
        out |= {
            "rr": float(R / S),
            "ln_rr_se": se_log,
            "rr_ci95": [float(R / S * np.exp(-Z95 * se_log)), float(R / S * np.exp(Z95 * se_log))],
        }
    return out


def failure_summary(strata: list[tuple], moments: list[tuple]) -> dict:
    stratum = ("src", "job", "day", "placement", "basis", "rounds", "patch")
    rows = [dict(zip((*stratum, "exposure", "K", "a", "n1", "c", "n0"), r)) for r in strata]
    mom = [dict(zip(stratum, m), values=m[len(stratum) :]) for m in moments]

    def match(r: dict, kw: dict) -> bool:
        return all(r[k] == v or (k == "patch" and v == "both") for k, v in kw.items())

    def pick(**kw) -> list[tuple]:
        return [(r["a"], r["n1"], r["c"], r["n0"]) for r in rows if match(r, kw)]

    out: dict = {}
    for basis in "XZ":
        for patch in ("3", "5", "both"):
            group = out.setdefault(f"{basis} d{patch}", {})
            for exp in EXPOSURES:
                c, m = (
                    mantel_haenszel(pick(src=src, basis=basis, patch=patch, exposure=exp))
                    for src in ("chip", "model")
                )
                res = {"chip": c, "model": m}
                if "rd" in c and "rd" in m:
                    se = np.hypot(c["rd_se"], m["rd_se"])
                    res["chip_minus_model_rd"] = [c["rd"] - m["rd"], *ci(c["rd"] - m["rd"], se)]
                if "rr" in c and "rr" in m:
                    lr, se = np.log(c["rr"] / m["rr"]), np.hypot(c["ln_rr_se"], m["ln_rr_se"])
                    res["chip_over_model_rr"] = [float(np.exp(v)) for v in (lr, *ci(lr, se))]
                group[exp] = res
            kw = {"basis": basis, "patch": patch}
            group["failure_per_firing"] = {
                src: per_firing([r["values"] for r in mom if match(r, {"src": src, **kw})])
                for src in ("chip", "model")
            }
            group["shots_with_a_streak_8_rounds"] = {}
            for src in ("chip", "model"):
                sel = pick(src=src, exposure="any", rounds=8, **kw)
                n1, n0 = sum(r[1] for r in sel), sum(r[3] for r in sel)
                group["shots_with_a_streak_8_rounds"][src] = n1 / (n1 + n0)
    reps: dict = {}
    for basis in "XZ":
        for patch in ("3", "5"):
            vals = []
            for day in range(1, 6):
                for pl in (1, 2):
                    kw = {"basis": basis, "patch": patch, "day": day, "placement": pl}
                    r = mantel_haenszel(pick(src="chip", exposure="any", **kw))
                    reps[f"{basis} d{patch} day {day} placement {pl}"] = {
                        k: r[k] for k in ("rd", "rd_se", "rr")
                    }
                    vals.append((r["rd"], r["rd_se"]))
            v = np.array(vals)
            wts = 1 / v[:, 1] ** 2
            q = float((wts * (v[:, 0] - (wts * v[:, 0]).sum() / wts.sum()) ** 2).sum())
            reps[f"{basis} d{patch} summary"] = {
                "repetitions_with_rd_above_zero": int((v[:, 0] > 0).sum()),
                "rd_range": [float(v[:, 0].min()), float(v[:, 0].max())],
                "cochran_q": q,
                "dof": len(v) - 1,
                "p": float(chi2.sf(q, len(v) - 1)),
            }
    out["per_repetition_any_streak_chip"] = reps
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--model-shots", type=int, default=MODEL_SHOTS)
    args = ap.parse_args()
    n = len(job_folders())
    with Pool(args.workers) as pool:
        parts = pool.starmap(job_effects, [(i, args.model_shots) for i in range(n)])
    units = [u for p in parts for u in p["units"]]
    s2: dict = defaultdict(float)
    for p in parts:
        for k, v in p["s2"].items():
            s2[k] += v
    strata = [s for p in parts for s in p["strata"]]
    moments = [m for p in parts for m in p["moments"]]
    check = max(p["check"] for p in parts)
    assert check < 1e-12, f"decoded failures differ from analysis.json by {check}"
    out = {
        "script": "docs/analysis/streak_effects.py",
        "about": {
            "streak": "flag or relay i fired in rounds t-1 and t (x_{i,t-1} = x_{i,t} = 1)",
            "A": (
                "unit = (job, circuit, series, round t); groups streak (1,1), single (0,1), quiet (0,0); "
                "outcome = fraction of the patch's near (share a data qubit) or far stabilizer detectors "
                "firing in round t + lag; D = sum w (ybar1 - ybar0) / sum w, w = n1 n0 / (n1 + n0); model = "
                "same w, exact model probabilities; CIs over shots (linearized) and clustered by physical qubit"
            ),
            "B": (
                "strata (job, circuit, patch); exposed = a streak on any flag or relay of the patch; "
                "Mantel-Haenszel RD (Sato) and RR (Greenland-Robins); count matched = strata also by the "
                "number K of flag and relay firings in the patch; failure_per_firing = within-stratum "
                "least-squares slope of the failure on K; model = the same on its sampled, decoded shots"
            ),
        },
        "circuits": f"memory, rounds >= 2, X and Z, {n} jobs; model shots {args.model_shots} per circuit, seed {SEED} + setting index",
        "decoded_failures_match_analysis_json": True,
        "detection_events_after_a_streak": detection_summary(units, s2),
        "logical_failure": failure_summary(strata, moments),
    }
    write_json("streak_effects.json", out)
    report(out)


def report(out: dict) -> None:
    det = out["detection_events_after_a_streak"]
    print(
        "\nA. Near (far) stabilizers, detection fraction difference in points, chip [95% CI over qubits] model"
    )
    for key in [f"{b} d{d} both" for b in "XZ" for d in "35"] + [
        f"{b} dboth {r}" for b in "XZ" for r in ("flag", "relay")
    ]:
        for g1, g0 in CONTRASTS:
            parts = []
            for lag in LAGS:
                r = det[key].get(f"lag {lag}", {})
                near, far = (r.get(o, {}).get(f"{g1} - {g0}") for o in OUTCOMES)
                if near:
                    lo, hi = near["chip_ci95_qubits"]
                    f = f" ({100 * far['chip']:+.1f})" if far else ""
                    parts.append(
                        f"{lag:+d}: {100 * near['chip']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] {100 * near['model']:+.1f}{f}"
                    )
            print(f"  {key:15s} {g1 + ' - ' + g0:16s} " + "; ".join(parts))
    print(
        "\nB. Logical failure: reference -> exposed; chip RD [95% CI] RR [95% CI] | model RD [CI] RR"
    )
    for key, v in out["logical_failure"].items():
        if key.startswith("per_"):
            continue
        for exp in EXPOSURES:
            c, m = v[exp]["chip"], v[exp]["model"]
            if "rr" not in c:
                continue
            print(
                f"  {key:7s} {exp:13s}: {100 * c['failure_reference']:5.1f}% -> {100 * c['failure_exposed']:5.1f}% "
                f"RD {100 * c['rd']:+.1f} [{100 * c['rd_ci95'][0]:+.1f}, {100 * c['rd_ci95'][1]:+.1f}] "
                f"RR {c['rr']:.3f} [{c['rr_ci95'][0]:.3f}, {c['rr_ci95'][1]:.3f}] | model RD {100 * m['rd']:+.1f} "
                f"[{100 * m['rd_ci95'][0]:+.1f}, {100 * m['rd_ci95'][1]:+.1f}] RR {m.get('rr', np.nan):.3f}"
            )
        c, m = (v["failure_per_firing"][src] for src in ("chip", "model"))
        print(
            f"  {key:7s} per firing   : chip {100 * c['slope']:+.2f} [{100 * c['ci95'][0]:+.2f}, "
            f"{100 * c['ci95'][1]:+.2f}] points, model {100 * m['slope']:+.2f} [{100 * m['ci95'][0]:+.2f}, "
            f"{100 * m['ci95'][1]:+.2f}]"
        )


if __name__ == "__main__":
    main()

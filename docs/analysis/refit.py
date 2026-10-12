"""eps and Lambda of re-decoded shots, as in Table placement, for pij_decode.py and corr_matching.py.

Each curve (job, decoder, basis, patch) is fitted with the prespecified weighted-log fit and the
binomial refit (fitting.py, without points dropped; the model's shots get the point estimate
only); Lambda per repetition is eps3 / eps5 with the error propagated from the two curves
(rep_lambda). Per placement, eps is the plain mean over the five days and Lambda the
inverse-variance mean of the daily ratios, widened by sqrt(chi2/nu) when above one (fits.pooled).

Decoders are compared on the same shots: a paired bootstrap over shots (one set of multinomial
weights per circuit, used for every decoder) gives the shot-noise error of each difference,
with the decoders' priors held fixed. The model's own shots for the controls are sampled with
seed SIM_SEED + setting index, which neither the stored predictions (seed = rounds) nor the
other post hoc steps use.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from heavyhex.experiment.circuits import ROUNDS, SETTINGS

import fitting as F
from fits import pooled

MEMORY = [i for i, s in enumerate(SETTINGS) if s.kind == "memory"]
SIM_SHOTS = 10_000
SIM_SEED = 9000
BOOT = 200
N_ROUNDS = np.array(ROUNDS, float)
COARSE = np.concatenate([np.linspace(1e-4, 0.45, 240), np.linspace(0.45, F.EPS_MAX, 40)[1:]])


def sim_seed(i: int) -> int:
    return SIM_SEED + i


def curve(k, N: int, profile: bool = True) -> dict:
    """Both fits of one curve: failures k at the six round counts out of N shots."""
    k = np.asarray(k, float)
    w = F.wls(ROUNDS, k / N, N)
    out = {
        "P": (k / N).tolist(),
        "points_at_or_above_half": int((k / N >= 0.5).sum()),
        "wls": [w.eps, w.se],
    }
    if profile:
        b = F.binomial(ROUNDS, k, N)
        out["binomial"] = [b.eps, (b.hi_q - b.lo_q) / 2]
    else:
        out["binomial"] = [F.binomial_point(N_ROUNDS, k, N), float("nan")]
    return out


def rep_lambda(c3: dict, c5: dict) -> dict:
    """Lambda = eps3 / eps5 with the error propagated from the two curves' errors (for the
    binomial fit, the quasi-likelihood half-widths: fits.py's binomial_se_propagated, which
    weights the days almost as its joint-profile error does and is much faster)."""
    out = {}
    for fit in ("wls", "binomial"):
        (e3, u3), (e5, u5) = c3[fit], c5[fit]
        out[fit] = [e3 / e5, e3 / e5 * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)]
    return out


def boot_eps(P: np.ndarray, N: int) -> dict:
    """Both fits of B replicate curves (B x 6 failure rates)."""
    wls = F.wls_batch(N_ROUNDS, P, N)[0]
    mle = np.concatenate(
        [
            F.mle_batch(np.round(P[s : s + 100] * N), N, N_ROUNDS, grid=COARSE)[0]
            for s in range(0, len(P), 100)
        ]
    )
    return {"wls": wls, "binomial": mle}


def weights(n: int, seed: int, B: int = BOOT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)


def job_curves(fails: dict, patches: list[str], profile: bool = True) -> dict:
    """fails: {decoder: {setting index: shots x patches bool}} -> curves and counts per
    decoder, basis and patch."""
    out = {}
    for dec, by_i in fails.items():
        out[dec] = {}
        for basis in "XZ":
            idx = [i for i in MEMORY if SETTINGS[i].basis == basis]
            for k, p in enumerate(patches):
                counts = [int(by_i[i][:, k].sum()) for i in idx]
                N = by_i[idx[0]].shape[0]
                out[dec][f"{basis}{p}"] = {"k": counts, "N": N, **curve(counts, N, profile)}
    return out


def job_boot(fails: dict, patches: list[str], seed: int) -> dict:
    """Paired bootstrap replicates of eps: {decoder: {basis+patch: {fit: B array}}}."""
    out: dict = {dec: {} for dec in fails}
    first = next(iter(fails.values()))
    for basis in "XZ":
        idx = [i for i in MEMORY if SETTINGS[i].basis == basis]
        W = {i: weights(first[i].shape[0], seed + i) for i in idx}
        for k, p in enumerate(patches):
            for dec, by_i in fails.items():
                N = by_i[idx[0]].shape[0]
                P = np.stack([W[i] @ by_i[i][:, k].astype(np.float32) / N for i in idx], axis=1)
                out[dec][f"{basis}{p}"] = boot_eps(P.astype(float), N)
    return out


def reps(per_job: dict) -> list[dict]:
    """Per repetition (day, placement): eps3, eps5 and Lambda per decoder and basis.

    per_job: {name: {"day", "rep", "patches", "curves": job_curves(...)}}."""
    by: dict = {}
    for name, j in per_job.items():
        r = by.setdefault(
            (j["day"], j["rep"]), {"day": j["day"], "placement": j["rep"], "jobs": {}}
        )
        for p in j["patches"]:
            r["jobs"][p] = name
    out = []
    for (_day, _rep), r in sorted(by.items()):
        row = {**r, "decoders": {}}
        decs = per_job[r["jobs"]["3"]]["curves"]
        for dec in decs:
            row["decoders"][dec] = {}
            for basis in "XZ":
                c3 = per_job[r["jobs"]["3"]]["curves"][dec][f"{basis}3"]
                c5 = per_job[r["jobs"]["5"]]["curves"][dec][f"{basis}5"]
                row["decoders"][dec][basis] = {
                    "eps3": {"wls": c3["wls"], "binomial": c3["binomial"]},
                    "eps5": {"wls": c5["wls"], "binomial": c5["binomial"]},
                    "lambda": rep_lambda(c3, c5),
                    "points_at_or_above_half": [
                        c3["points_at_or_above_half"],
                        c5["points_at_or_above_half"],
                    ],
                }
        out.append(row)
    return out


def placements(rows: list[dict]) -> dict:
    """Per placement, decoder, basis and fit: plain-mean eps3 and eps5 (with the SE over days)
    and the inverse-variance Lambda (widened by sqrt(chi2/nu) when above one)."""
    out: dict = {}
    for pl in (1, 2):
        sel = [r for r in rows if r["placement"] == pl]
        for dec in sel[0]["decoders"]:
            for basis in "XZ":
                for fit in ("wls", "binomial"):
                    q = {}
                    for e in ("eps3", "eps5"):
                        v = np.array([r["decoders"][dec][basis][e][fit][0] for r in sel])
                        q[e] = [float(v.mean()), float(v.std(ddof=1) / sqrt(len(v)))]
                    lam = np.array([r["decoders"][dec][basis]["lambda"][fit] for r in sel])
                    if np.all(np.isfinite(lam[:, 1])):
                        m, se, spread = pooled(lam[:, 0], lam[:, 1])
                        q["lambda"] = [m, se]
                        q["lambda_chi2_per_dof"] = spread
                    else:
                        q["lambda"] = [float(lam[:, 0].mean()), float("nan")]
                    q["lambda_plain_mean"] = float(lam[:, 0].mean())
                    out[f"placement {pl} {dec} {basis} {fit}"] = q
    return out


def paired_differences(per_job: dict, boots: dict, rows: list[dict], dec: str, ref: str) -> dict:
    """dec - ref per placement, basis and fit: eps3, eps5 (plain means over days) and Lambda
    (inverse-variance means, each decoder's weights held at their point values), with the
    paired-bootstrap SE over shots."""
    out: dict = {}
    for pl in (1, 2):
        sel = [r for r in rows if r["placement"] == pl]
        for basis in "XZ":
            for fit in ("wls", "binomial"):
                reps_d = {"eps3": [], "eps5": [], "lam": {dec: [], ref: []}}
                point = {}
                for e, p in (("eps3", "3"), ("eps5", "5")):
                    diffs = []
                    for r in sel:
                        b = boots[r["jobs"][p]]
                        diffs.append(b[dec][f"{basis}{p}"][fit] - b[ref][f"{basis}{p}"][fit])
                    reps_d[e] = np.mean(diffs, axis=0)
                    point[e] = float(
                        np.mean(
                            [
                                r["decoders"][dec][basis][e][fit][0]
                                - r["decoders"][ref][basis][e][fit][0]
                                for r in sel
                            ]
                        )
                    )
                lam_rep = {}
                for d in (dec, ref):
                    lv = np.array([r["decoders"][d][basis]["lambda"][fit] for r in sel])
                    w = 1 / lv[:, 1] ** 2 if np.all(np.isfinite(lv[:, 1])) else np.ones(len(sel))
                    reps_l = []
                    for r in sel:
                        b3 = boots[r["jobs"]["3"]][d][f"{basis}3"][fit]
                        b5 = boots[r["jobs"]["5"]][d][f"{basis}5"][fit]
                        reps_l.append(b3 / b5)
                    lam_rep[d] = (np.array(reps_l).T * w).sum(1) / w.sum()
                    point[f"lambda {d}"] = float((lv[:, 0] * w).sum() / w.sum())
                dl = lam_rep[dec] - lam_rep[ref]
                out[f"placement {pl} {basis} {fit}"] = {
                    "eps3": [point["eps3"], float(np.nanstd(reps_d["eps3"], ddof=1))],
                    "eps5": [point["eps5"], float(np.nanstd(reps_d["eps5"], ddof=1))],
                    "lambda": [
                        point[f"lambda {dec}"] - point[f"lambda {ref}"],
                        float(np.nanstd(dl, ddof=1)),
                    ],
                }
    return out

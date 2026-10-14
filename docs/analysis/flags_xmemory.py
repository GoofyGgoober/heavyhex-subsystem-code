"""Flag and relay records in both memory bases, and the persistent state's statistics.

Flags and relays are uncomputed, so they read 0 without faults in X memory as in Z memory
(common.readout_model; checked here on every X-memory circuit with stim). X memory therefore
gives 320 more series, from the same 38 qubits in the same fifteen jobs, to test whether the
persistent state depends on the memory basis. Every series has n readouts in an n-round
circuit (X memory's preparation half is left out). This script puts the bases side by side:

(a) The re-firing excess Delta(l), l = 1..7 rounds, of each series' 8-round record (refiring.py's
    statistic), averaged over the 320 series. The model's comes from refiring*.json.
(b) hmm.py's fits in both bases, paired by series (same job, same qubit): ln(a_X/a_Z),
    ln(b_X/b_Z), e_P,X - e_P,Z and e_N,X - e_N,Z, averaged per qubit and day.
(c) Delta(l) = A r^l + C, least squares on the mean over series. A shot whose firing rate never
    changes adds the same amount at every lag, so C bounds such static heterogeneity. The
    two-state hidden Markov model gives r = 1 - a - b.
(d) The hidden Markov model against two models in which a shot's firing rate never changes:
    a two-class mixture (the same model with a = b = 0; pi, e_N, e_P) and a beta-binomial
    (a rate per shot drawn from Beta(alpha, beta)). Log-likelihoods of the same pooled 4-, 6-
    and 8-round shots, with hmm.py's parameters for the hidden Markov model.
(e) The onset rate by role in X memory, per round, per CZ and per readout or reset
    (hmm_causes.py has Z memory's), and the eight qubits that served in both roles.
(f) Co-streak excesses (costreak.json) with qubit-dyadic standard errors, and a t test with
    11 degrees of freedom for the 12 physical pairs of a gauge's two flags.
(g) An example: one flag whose fitted parameters are close to the medians, the first 120 shots
    of its 8-round Z-memory record, with the posterior probability of the persistent state.

Intervals are 95%, from a bootstrap over the 38 physical qubits (the same draws for both
bases where they are compared). Paper: Sec. VII (persistence, the replication in X memory, the
alternatives, the co-streaks), Fig. leak and App. Fig. hmm_trace.

Inputs: results/hmm.json, hmm_x.json, refiring.json, refiring_x.json, hmm_bootstrap.json,
hmm_causes_summary.json, costreak.json. The device series are cached in
cache/flag_series_{z,x}.npz: 4 minutes to build; with them, 45 s (the bootstraps and the
static models' fits). Under 0.5 GB.

    python docs/analysis/refiring.py --basis X
    python docs/analysis/hmm.py --basis X --recompute
    python docs/analysis/flags_xmemory.py       # writes results/flags_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import betaln
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from common import (
    CACHE,
    ROUND_US,
    Job,
    job_folders,
    memory_index,
    readout_model,
    readout_series,
    result,
    write_json,
)
from hmm_causes import operations
from refiring import excess

BASES = ("Z", "X")
LENGTHS = (4, 6, 8)
LAGS = np.arange(1, 8)
BOOT, SEED = 2000, 1
TRACE_SHOTS = 120


# ------------------------------------------------------------------ the device's series
def load_series(basis: str) -> tuple[dict, list[dict]]:
    """codes[T]: series x shots, bit t set if the readout fired in round t + 1; meta per series."""
    path = CACHE / f"flag_series_{basis.lower()}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {T: z[f"codes{T}"] for T in LENGTHS}, json.loads(str(z["meta"]))
    codes: dict = {T: [] for T in LENGTHS}
    meta: list = []
    for job in map(Job, job_folders()):  # one job in memory at a time
        per: dict = defaultdict(dict)
        ops = {}
        for T in LENGTHS:
            idx = memory_index(basis, T)
            model, records, labels = readout_model(job, idx)
            if basis == "X":  # stim refuses a detector that is not deterministic
                model.without_noise().detector_error_model()
            chip, _ = job.chip_detectors(idx, model, records)
            for key, cols in readout_series(labels).items():
                x = chip[:, cols]
                assert x.shape[1] == T
                per[key][T] = (x.astype(np.int64) << np.arange(T)).sum(1).astype(np.int16)
            if T in (4, 8):
                ops[T] = operations(job, idx)
        for key, d in per.items():
            q = job.readout_qubit(*key)
            meta.append(
                {
                    "job": job.name,
                    "day": job.day,
                    "rep": job.rep,
                    "patch": key[0],
                    "readout": key[1],
                    "k": key[2],
                    "qubit": q,
                    # steady state per round, as hmm_causes.py
                    "cz_per_round": (ops[8][0][q] - ops[4][0][q]) / 4,
                    "events_per_round": (ops[8][1][q] - ops[4][1][q]) / 4,
                }
            )
            for T in LENGTHS:
                codes[T].append(d[T])
        print(basis, job.name, "loaded", flush=True)
    codes = {T: np.stack(v) for T, v in codes.items()}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **{f"codes{T}": codes[T] for T in LENGTHS})
    return codes, meta


def patterns(T: int) -> np.ndarray:
    """Every 0/1 record of T rounds, P x T, row p = the bits of p."""
    return ((np.arange(1 << T)[:, None] >> np.arange(T)) & 1).astype(bool)


def decode(codes: np.ndarray, T: int) -> np.ndarray:
    return ((codes[..., None].astype(np.int64) >> np.arange(T)) & 1).astype(bool)


def counts_of(codes: np.ndarray, T: int) -> np.ndarray:
    """Series x 2^T: how many shots show each record."""
    off = (np.arange(len(codes)) << T)[:, None]
    flat = np.bincount((codes.astype(np.int64) + off).ravel(), minlength=len(codes) << T)
    return flat.reshape(len(codes), 1 << T).astype(float)


def key_of(m: dict) -> tuple:
    return (m["job"], m["patch"], m["readout"], m["qubit"])


# ------------------------------------------------------------------ likelihoods
def hmm_logprob(T: int, p: dict) -> np.ndarray:
    """ln P(record) of every T-round record under each series' hidden Markov parameters,
    series x 2^T (forward algorithm)."""
    pi, a, b, eN, eP = (np.asarray(p[k])[:, None] for k in ("pi", "a", "b", "e_N", "e_L"))
    pat = patterns(T)
    fN = (1 - pi) * np.where(pat[:, 0], eN, 1 - eN)
    fP = pi * np.where(pat[:, 0], eP, 1 - eP)
    out = np.zeros(fN.shape)
    for t in range(1, T):
        c = fN + fP
        out += np.log(c)
        fN, fP = fN / c, fP / c
        fN, fP = (
            (fN * (1 - a) + fP * b) * np.where(pat[:, t], eN, 1 - eN),
            (fN * a + fP * (1 - b)) * np.where(pat[:, t], eP, 1 - eP),
        )
    return out + np.log(fN + fP)


def hmm_loglik(counts: dict, p: dict) -> np.ndarray:
    """Log-likelihood of each series under its hidden Markov parameters (arrays over series)."""
    return sum((cnt * hmm_logprob(T, p)).sum(1) for T, cnt in counts.items())


def hmm_refire(p: dict, T: int = 8) -> np.ndarray:
    """Delta(l) that each series' fitted hidden Markov model implies for T-round records, pooled
    over rounds as refiring.excess pools them, in the limit of many shots; series x lags."""
    prob = np.exp(hmm_logprob(T, p))
    pat = patterns(T)
    out = []
    for lag in LAGS:
        now, later = pat[:, :-lag], pat[:, lag:]
        fired = prob @ (now & later).sum(1) / (prob @ now.sum(1))
        quiet = prob @ (~now & later).sum(1) / (prob @ (~now).sum(1))
        out.append(fired - quiet)
    return np.array(out).T


def independent_loglik(counts: dict) -> np.ndarray:
    """Independent rounds, each with its own rate: the calibration model's shape (hmm.py's)."""
    ll = 0.0
    for T, cnt in counts.items():
        pat = patterns(T)
        n = cnt.sum(1, keepdims=True)
        k = cnt @ pat  # firings per round
        q = np.clip(k / n, 1e-9, 1 - 1e-9)
        ll = ll + (k * np.log(q) + (n - k) * np.log(1 - q)).sum(1)
    return ll


def by_firings(counts: dict) -> dict:
    """Series x (T + 1): how many shots fire in k of their T rounds. A model in which a shot's
    rate never changes depends on a record only through k."""
    return {
        T: np.stack([cnt[:, patterns(T).sum(1) == k].sum(1) for k in range(T + 1)], 1)
        for T, cnt in counts.items()
    }


def mixture_loglik(by_k: dict, iters: int = 5000, tol: float = 1e-9) -> np.ndarray:
    """Two classes of shot, with rates e_N and e_P that never change within a shot; by EM,
    best of three starts."""
    S = len(next(iter(by_k.values())))
    rate = sum(n @ np.arange(T + 1) for T, n in by_k.items()) / sum(
        n.sum(1) * T for T, n in by_k.items()
    )
    best = np.full(S, -np.inf)
    for w0, scale, e1 in ((0.1, 1.0, 0.5), (0.3, 0.7, 0.25), (0.02, 1.0, 0.7)):
        w, eN, eP = np.full(S, w0), np.clip(rate * scale, 1e-4, 0.5), np.full(S, e1)
        last = np.full(S, -np.inf)
        for _ in range(iters):
            ll = np.zeros(S)
            nw = n_all = kP = tP = kN = tN = 0.0
            for T, n in by_k.items():
                k = np.arange(T + 1)[None]
                lN = (
                    np.log(1 - w)[:, None]
                    + k * np.log(eN)[:, None]
                    + (T - k) * np.log(1 - eN)[:, None]
                )
                lP = (
                    np.log(w)[:, None] + k * np.log(eP)[:, None] + (T - k) * np.log(1 - eP)[:, None]
                )
                m = np.logaddexp(lN, lP)
                r = np.exp(lP - m)
                ll += (n * m).sum(1)
                nw = nw + (n * r).sum(1)
                n_all = n_all + n.sum(1)
                kP = kP + (n * r * k).sum(1)
                tP = tP + (n * r).sum(1) * T
                kN = kN + (n * (1 - r) * k).sum(1)
                tN = tN + (n * (1 - r)).sum(1) * T
            w = np.clip(nw / n_all, 1e-9, 1 - 1e-9)
            eP = np.clip(kP / np.maximum(tP, 1e-12), 1e-9, 1 - 1e-9)
            eN = np.clip(kN / np.maximum(tN, 1e-12), 1e-9, 1 - 1e-9)
            if np.all(ll - last < tol):
                break
            last = ll
        best = np.maximum(best, ll)
    return best


def betabinomial_nll(x: np.ndarray, series: dict) -> float:
    """-ln L of one series, {T: shots with k = 0..T firings}, at (ln alpha, ln beta)."""
    al, be = np.exp(x)
    k = {T: np.arange(T + 1) for T in series}
    return -sum(n @ (betaln(al + k[T], be + T - k[T]) - betaln(al, be)) for T, n in series.items())


def betabinomial_loglik(by_k: dict) -> np.ndarray:
    """A firing rate per shot drawn from Beta(alpha, beta), fixed within the shot; by maximum
    likelihood, best of three starts."""
    S = len(next(iter(by_k.values())))
    out = np.empty(S)
    for i in range(S):
        series = {T: n[i] for T, n in by_k.items()}
        m = sum(n @ np.arange(T + 1) for T, n in series.items()) / sum(
            n.sum() * T for T, n in series.items()
        )
        fits = [
            minimize(
                betabinomial_nll,
                np.log([m * s, (1 - m) * s]),
                args=(series,),
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 4000},
            )
            for s in (1.0, 10.0, 100.0)
        ]
        out[i] = -min(f.fun for f in fits)
    return out


def posterior(pat: np.ndarray, p: dict) -> np.ndarray:
    """P(persistent in round t | the shot's record), records x T, for one series' parameters."""
    T = pat.shape[1]
    A = np.array([[1 - p["a"], p["a"]], [p["b"], 1 - p["b"]]])
    E = np.stack([np.where(pat, p["e_N"], 1 - p["e_N"]), np.where(pat, p["e_L"], 1 - p["e_L"])], 2)
    al = np.empty(E.shape)
    al[:, 0] = np.array([1 - p["pi"], p["pi"]]) * E[:, 0]
    al[:, 0] /= al[:, 0].sum(1, keepdims=True)
    for t in range(1, T):
        al[:, t] = (al[:, t - 1] @ A) * E[:, t]
        al[:, t] /= al[:, t].sum(1, keepdims=True)
    be = np.ones(E.shape)
    for t in range(T - 2, -1, -1):
        be[:, t] = (E[:, t + 1] * be[:, t + 1]) @ A.T
        be[:, t] /= be[:, t].sum(1, keepdims=True)
    g = al * be
    return g[:, :, 1] / g.sum(2)


# ------------------------------------------------------------------ helpers
def geometric(l, A, r, C):
    return A * r**l + C


def fit_decay(curve: np.ndarray) -> dict:
    (A, r, C), _ = curve_fit(geometric, LAGS, curve, p0=(curve[0] / 0.75, 0.75, 0.0), maxfev=20000)
    (_, r0), _ = curve_fit(
        lambda l, A, r: A * r**l, LAGS, curve, p0=(curve[0] / 0.75, 0.75), maxfev=20000
    )
    return {"A": A, "r": r, "C": C, "tau_us": -ROUND_US / np.log(r), "r_without_C": r0}


def ci(values) -> list[float]:
    return np.percentile(values, [2.5, 97.5]).tolist()


def qubit_draws(qubits: list[int], rng) -> list[np.ndarray]:
    """BOOT resamples of the physical qubits, as arrays of qubit labels."""
    labels = np.array(sorted(set(qubits)))
    return [labels[rng.integers(0, len(labels), len(labels))] for _ in range(BOOT)]


def rows_of(qubits: list[int]) -> dict:
    out = defaultdict(list)
    for i, q in enumerate(qubits):
        out[q].append(i)
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    rng = np.random.default_rng(SEED)
    data = {B: load_series(B) for B in BASES}
    meta = {B: data[B][1] for B in BASES}
    order = {B: {key_of(m): i for i, m in enumerate(meta[B])} for B in BASES}
    assert set(order["Z"]) == set(order["X"]) and len(order["Z"]) == len(meta["Z"]) == 320
    qubits = [m["qubit"] for m in meta["Z"]]
    xi = np.array([order["X"][key_of(m)] for m in meta["Z"]])  # X series in Z's order
    draws = qubit_draws(qubits, rng)
    where = rows_of(qubits)
    out: dict = {
        "script": "docs/analysis/flags_xmemory.py",
        "series_per_basis": len(qubits),
        "physical_qubits": len(where),
        "x_memory_detectors_deterministic": "checked with stim on all 45 X-memory circuits",
        "bootstrap": f"{BOOT} resamples of the {len(where)} physical qubits, seed {SEED}",
    }

    # (a), (c): Delta(l) per series, from the 8-round records, and its geometric fit
    delta = {}
    for B in BASES:
        x = decode(data[B][0][8], 8)
        delta[B] = np.array([[excess(s, int(l)) for l in LAGS] for s in x])
    delta["X"] = delta["X"][xi]
    ref = {"Z": result("refiring.json"), "X": result("refiring_x.json")}
    for B in BASES:
        for l in range(1, 7):
            assert abs(delta[B][:, l - 1].mean() - ref[B]["lags"][str(l)]["chip"]) < 1e-12
    boot_curves = {
        B: np.array([delta[B][np.concatenate([where[q] for q in d])].mean(0) for d in draws])
        for B in BASES
    }
    fits = {B: fit_decay(delta[B].mean(0)) for B in BASES}
    boot_fits = {B: [fit_decay(c) for c in boot_curves[B]] for B in BASES}
    hmm = {"Z": result("hmm.json"), "X": result("hmm_x.json")}
    out["refiring"] = {}
    for B in BASES:
        lam = np.array([1 - r["chip"]["a"] - r["chip"]["b"] for r in hmm[B]])
        implied = hmm_refire(
            {k: np.array([r["chip"][k] for r in hmm[B]]) for k in ("pi", "a", "b", "e_N", "e_L")}
        ).mean(0)
        out["refiring"][B] = {
            "lags": LAGS.tolist(),
            "lag_us": (LAGS * ROUND_US).tolist(),
            "device": delta[B].mean(0).tolist(),
            "device_ci95": [ci(boot_curves[B][:, j]) for j in range(len(LAGS))],
            "device_above_model_at_lag_1": ref[B]["chip_above_model_at_lag_1"],
            "model_lags_1_6": [ref[B]["lags"][str(l)]["model"] for l in range(1, 7)],
            "base_odds": ref[B]["base_odds"],
            "runs_of_5": ref[B]["runs_of_5"],
            "fit": {
                **fits[B],
                **{
                    f"{k}_ci95": ci([f[k] for f in boot_fits[B]])
                    for k in ("A", "r", "C", "tau_us", "r_without_C")
                },
                "model": "Delta(l) = A r^l + C, l = 1..7 rounds, unweighted least squares on the mean over series",
            },
            "hmm_one_minus_a_minus_b_median": float(np.median(lam)),
            # what the per-series fits imply for the same pooled statistic; its fit shows how
            # much of C a spread of decay rates over the series produces on its own
            "hmm_implied": {
                "device_minus_hmm": (delta[B].mean(0) - implied).tolist(),
                "curve": implied.tolist(),
                "fit": fit_decay(implied),
            },
        }
    out["refiring"]["X_minus_Z"] = {
        "delta_1": float(delta["X"][:, 0].mean() - delta["Z"][:, 0].mean()),
        "delta_1_ci95": ci(boot_curves["X"][:, 0] - boot_curves["Z"][:, 0]),
        "r": fits["X"]["r"] - fits["Z"]["r"],
        "r_ci95": ci([fx["r"] - fz["r"] for fx, fz in zip(boot_fits["X"], boot_fits["Z"])]),
        "C": fits["X"]["C"] - fits["Z"]["C"],
        "C_ci95": ci([fx["C"] - fz["C"] for fx, fz in zip(boot_fits["X"], boot_fits["Z"])]),
        "series_with_X_above_Z_at_lag_1": int((delta["X"][:, 0] > delta["Z"][:, 0]).sum()),
    }

    # (b): hidden Markov fits in both bases, paired by series, then per qubit and day
    fit = {B: {key_of(r): r for r in hmm[B]} for B in BASES}
    summ = {"Z": result("hmm_summary.json"), "X": result("hmm_x_summary.json")}
    keys = [key_of(m) for m in meta["Z"]]
    par = {
        B: {
            k: np.array([fit[B][key]["chip"][k] for key in keys])
            for k in ("a", "b", "e_N", "e_L", "pi")
        }
        for B in BASES
    }
    model_eN = {B: np.array([fit[B][key]["model"]["e_N"] for key in keys]) for B in BASES}
    paired = {
        "ln_a_ratio": np.log(par["X"]["a"] / par["Z"]["a"]),
        "ln_b_ratio": np.log(par["X"]["b"] / par["Z"]["b"]),
        "e_P_difference": par["X"]["e_L"] - par["Z"]["e_L"],
        "e_N_difference": par["X"]["e_N"] - par["Z"]["e_N"],
        "e_N_minus_model_difference": (par["X"]["e_N"] - model_eN["X"])
        - (par["Z"]["e_N"] - model_eN["Z"]),
    }
    units = defaultdict(list)  # (qubit, day) -> series
    for i, m in enumerate(meta["Z"]):
        units[(m["qubit"], m["day"])].append(i)
    ukeys = sorted(units)
    uq = np.array([q for q, _ in ukeys])
    unit_rows = rows_of(list(uq))
    out["hmm"] = {"medians": {}, "paired_by_qubit_and_day": {"units": len(ukeys)}}
    for B in BASES:
        s = summ[B]["chip"]
        tau = -ROUND_US / np.log(1 - par[B]["b"])
        med = {
            "a": par[B]["a"],
            "b": par[B]["b"],
            "e_P": par[B]["e_L"],
            "e_N": par[B]["e_N"],
            "tau_P_us": tau,
        }
        out["hmm"]["medians"][B] = {
            **{k: float(np.median(v)) for k, v in med.items()},
            **{
                f"{k}_ci95_over_qubits": ci(
                    [np.median(v[np.concatenate([where[q] for q in d])]) for d in draws]
                )
                for k, v in med.items()
            },
            "llr_vs_independent_rounds_median": s["llr_median"],
            "llr_above_10": s["llr_above_10"],
            "model_shots_llr_max": summ[B]["model"]["llr_max"],
            "model_shots_llr_median": summ[B]["model"]["llr_median"],
            "model_shots_a_median": summ[B]["model"]["a_median"],
            "capped_at_300": s.get("capped_at_300", result("hmm_bootstrap.json")["capped_at_300"]),
        }
    P = out["hmm"]["paired_by_qubit_and_day"]
    for name, v in paired.items():
        u = np.array([v[units[k]].mean() for k in ukeys])
        boot = [u[np.concatenate([unit_rows[q] for q in d])].mean() for d in draws]
        P[name] = {
            "mean": float(u.mean()),
            "ci95": ci(boot),
            "median": float(np.median(u)),
            "positive": int((u > 0).sum()),
        }
        if name.startswith("ln_"):
            P[name.replace("ln_", "geometric_mean_")] = {
                "value": float(np.exp(u.mean())),
                "ci95": np.exp(ci(boot)).tolist(),
            }
    for k in ("a", "b", "e_L"):
        ux = np.array([par["X"][k][units[u]].mean() for u in ukeys])
        uz = np.array([par["Z"][k][units[u]].mean() for u in ukeys])
        rho = spearmanr(ux, uz)[0]
        boot = []
        for d in draws:
            j = np.concatenate([unit_rows[q] for q in d])
            boot.append(spearmanr(ux[j], uz[j])[0])
        P[f"spearman_{k}_X_vs_Z"] = {"rho": float(rho), "ci95": ci(boot)}

    # (d): the hidden Markov model against shot-static alternatives, both bases
    out["alternatives"] = {}
    for B in BASES:
        codes, m = data[B]
        counts = {T: counts_of(codes[T], T) for T in LENGTHS}
        by_k = by_firings(counts)
        p = {
            k: np.array([fit[B][key_of(r)]["chip"][k] for r in m])
            for k in ("pi", "a", "b", "e_N", "e_L")
        }
        ll = {
            "hmm": hmm_loglik(counts, p),
            "independent_rounds": independent_loglik(counts),
            "mixture": mixture_loglik(by_k),
            "beta_binomial": betabinomial_loglik(by_k),
        }
        llr_saved = np.array([fit[B][key_of(r)]["chip"]["llr"] for r in m])
        out["alternatives"][B] = {
            "parameters": {"hmm": 5, "mixture": 3, "beta_binomial": 2, "independent_rounds": 18}
        }
        for alt in ("mixture", "beta_binomial", "independent_rounds"):
            d = ll["hmm"] - ll[alt]
            out["alternatives"][B][f"hmm_minus_{alt}"] = {
                "median": float(np.median(d)),
                "iqr": np.percentile(d, [25, 75]).tolist(),
                "min": float(d.min()),
                "above_10": int((d > 10).sum()),
                "below_0": int((d < 0).sum()),
            }
        out["alternatives"][B]["hmm_llr_check_max_abs"] = float(
            np.abs(ll["hmm"] - ll["independent_rounds"] - llr_saved).max()
        )

    # (e): onset rate by role in X memory, and the qubits in both roles
    causes = result("hmm_causes_summary.json")
    roles = {}
    for role in ("z_flag", "relay"):
        sel = [i for i, mm in enumerate(meta["Z"]) if mm["readout"].startswith(role)]
        a = par["X"]["a"][sel]
        cz = np.array([meta["X"][xi[i]]["cz_per_round"] for i in sel])
        ev = np.array([meta["X"][xi[i]]["events_per_round"] for i in sel])
        roles[role] = {
            "series": len(sel),
            "onset_per_round_median": float(np.median(a)),
            "onset_per_cz_median": float(np.median(a / cz)),
            "onset_per_event_median": float(np.median(a / ev)),
            "cz_per_round": sorted(set(cz.tolist())),
            "events_per_round": sorted(set(ev.tolist())),
        }
    both = defaultdict(lambda: defaultdict(list))
    for i, mm in enumerate(meta["Z"]):
        both[mm["qubit"]][mm["readout"].split()[0]].append(par["X"]["a"][i])
    ratio = {
        q: float(np.mean(d["z_flag"]) / np.mean(d["relay"])) for q, d in both.items() if len(d) == 2
    }
    logs = np.log(list(ratio.values()))
    geo = [np.exp(logs[rng.integers(0, len(logs), len(logs))].mean()) for _ in range(BOOT)]
    out["roles_x_memory"] = roles | {
        "both_roles": {
            "qubits": len(ratio),
            "flag_over_relay_onset": ratio,
            "median_ratio": float(np.median(list(ratio.values()))),
            "geometric_mean_ratio": float(np.exp(logs.mean())),
            "geometric_mean_ratio_ci95_over_qubits": ci(geo),
            "flag_above_relay": int((logs > 0).sum()),
        },
        "z_memory_from_hmm_causes": {k: causes[k] for k in ("by_role", "both_roles")},
    }

    # (f): co-streaks with qubit-dyadic errors; t with 11 degrees of freedom for 12 pairs
    co = result("costreak.json")["classes"]
    out["costreak"] = {}
    for kind, c in co.items():
        row = {
            "rows": c["rows"],
            "physical_pairs": c["physical_pairs"],
            "chip": c["chip"],
            "se_pair_clustered": c["chip_se_pair_clustered"],
            "se_qubit_dyadic": c["chip_se_qubit_dyadic"],
            "model_exact": c["model_exact"],
            "z_dyadic": (c["chip"] - c["model_exact"]) / c["chip_se_qubit_dyadic"],
            "per_slot": {
                k: c["per_slot_definition"][k]
                for k in ("chip", "chip_se_pair_clustered", "chip_se_qubit_dyadic", "model_exact")
            },
        }
        if c["physical_pairs"] < 30:
            dof = c["physical_pairs"] - 1
            for se in ("se_pair_clustered", "se_qubit_dyadic"):
                tv = (c["chip"] - c["model_exact"]) / row[se]
                row[f"t_{se}"] = tv
                row[f"p_two_sided_t{dof}_{se}"] = float(2 * student_t.sf(abs(tv), dof))
        out["costreak"][kind] = row

    # (g): an example record
    flags = [i for i, mm in enumerate(meta["Z"]) if mm["readout"].startswith("z_flag")]
    med = {k: np.median(par["Z"][k]) for k in ("a", "b", "e_L", "e_N")}
    score = (
        np.abs(np.log(par["Z"]["a"] / med["a"]))
        + np.abs(np.log(par["Z"]["b"] / med["b"]))
        + np.abs(par["Z"]["e_L"] - med["e_L"]) / 0.05
        + np.abs(par["Z"]["e_N"] - med["e_N"]) / 0.02
    )
    i = flags[int(np.argmin(score[flags]))]
    p = {k: float(par["Z"][k][i]) for k in ("pi", "a", "b", "e_N", "e_L")}
    rec = decode(data["Z"][0][8][i, :TRACE_SHOTS], 8)
    post = posterior(rec, p)
    out["example_trace"] = {
        **{k: meta["Z"][i][k] for k in ("job", "day", "patch", "readout", "qubit")},
        "basis": "Z",
        "rounds": 8,
        "shots": f"the first {TRACE_SHOTS} of the job's 8-round circuit",
        "parameters": p,
        "chosen_as": "the flag whose a, b, e_P and e_N are closest to the medians",
        "fired": rec.astype(int).tolist(),
        "posterior_persistent": np.round(post, 4).tolist(),
    }

    write_json("flags_summary.json", out)
    report(out)


def report(o: dict) -> None:
    for B in BASES:
        r, f = o["refiring"][B], o["refiring"][B]["fit"]
        print(
            f"{B} memory: Delta(1..7) {' '.join(f'{100 * v:.1f}' for v in r['device'])} points; "
            f"r {f['r']:.3f} ({f['r_ci95'][0]:.3f}-{f['r_ci95'][1]:.3f}), tau {f['tau_us']:.1f} us "
            f"({f['tau_us_ci95'][0]:.1f}-{f['tau_us_ci95'][1]:.1f}), C {100 * f['C']:+.2f} points "
            f"({100 * f['C_ci95'][0]:+.2f} to {100 * f['C_ci95'][1]:+.2f}); HMM 1-a-b median {r['hmm_one_minus_a_minus_b_median']:.3f}"
        )
        h = r["hmm_implied"]
        print(
            f"   implied by the HMM fits: {' '.join(f'{100 * v:.1f}' for v in h['curve'])}; r {h['fit']['r']:.3f}, "
            f"C {100 * h['fit']['C']:+.2f} points"
        )
    d = o["refiring"]["X_minus_Z"]
    print(
        f"X - Z: Delta(1) {100 * d['delta_1']:+.2f} points ({100 * d['delta_1_ci95'][0]:+.2f} to {100 * d['delta_1_ci95'][1]:+.2f}), "
        f"r {d['r']:+.3f} ({d['r_ci95'][0]:+.3f} to {d['r_ci95'][1]:+.3f}), "
        f"C {100 * d['C']:+.2f} points ({100 * d['C_ci95'][0]:+.2f} to {100 * d['C_ci95'][1]:+.2f})"
    )
    for B in BASES:
        m = o["hmm"]["medians"][B]
        print(
            f"{B} HMM medians: a {100 * m['a']:.2f}% ({100 * m['a_ci95_over_qubits'][0]:.2f}-{100 * m['a_ci95_over_qubits'][1]:.2f}), "
            f"b {m['b']:.3f} ({m['b_ci95_over_qubits'][0]:.3f}-{m['b_ci95_over_qubits'][1]:.3f}), e_P {m['e_P']:.3f}, e_N {m['e_N']:.3f}, "
            f"tau_P {m['tau_P_us']:.1f} us; gain over independent rounds {m['llr_vs_independent_rounds_median']:.0f} "
            f"(>10 in {m['llr_above_10']}); model shots max {m['model_shots_llr_max']:.1f}; capped {m['capped_at_300']}"
        )
    P = o["hmm"]["paired_by_qubit_and_day"]
    print(
        f"paired over {P['units']} qubit-days: a X/Z {P['geometric_mean_a_ratio']['value']:.3f} "
        f"({P['geometric_mean_a_ratio']['ci95'][0]:.3f}-{P['geometric_mean_a_ratio']['ci95'][1]:.3f}), "
        f"b X/Z {P['geometric_mean_b_ratio']['value']:.3f} ({P['geometric_mean_b_ratio']['ci95'][0]:.3f}-{P['geometric_mean_b_ratio']['ci95'][1]:.3f}), "
        f"e_P X-Z {P['e_P_difference']['mean']:+.3f} ({P['e_P_difference']['ci95'][0]:+.3f} to {P['e_P_difference']['ci95'][1]:+.3f}); "
        f"rho(a) {P['spearman_a_X_vs_Z']['rho']:.2f} ({P['spearman_a_X_vs_Z']['ci95'][0]:.2f}-{P['spearman_a_X_vs_Z']['ci95'][1]:.2f})"
    )
    for B in BASES:
        A = o["alternatives"][B]
        print(
            f"{B} HMM minus mixture: median {A['hmm_minus_mixture']['median']:.1f} (>10 in {A['hmm_minus_mixture']['above_10']}, <0 in {A['hmm_minus_mixture']['below_0']}); "
            f"minus beta-binomial {A['hmm_minus_beta_binomial']['median']:.1f} (>10 in {A['hmm_minus_beta_binomial']['above_10']}, <0 in {A['hmm_minus_beta_binomial']['below_0']}); "
            f"llr check {A['hmm_llr_check_max_abs']:.1e}"
        )
    R = o["roles_x_memory"]
    print(
        f"X roles: flags a {100 * R['z_flag']['onset_per_round_median']:.2f}% (per CZ {100 * R['z_flag']['onset_per_cz_median']:.2f}%, per event "
        f"{100 * R['z_flag']['onset_per_event_median']:.2f}%), relays {100 * R['relay']['onset_per_round_median']:.2f}% (per CZ "
        f"{100 * R['relay']['onset_per_cz_median']:.2f}%, per event {100 * R['relay']['onset_per_event_median']:.2f}%); "
        f"dual-role geometric mean {R['both_roles']['geometric_mean_ratio']:.2f} ({R['both_roles']['geometric_mean_ratio_ci95_over_qubits'][0]:.2f}-"
        f"{R['both_roles']['geometric_mean_ratio_ci95_over_qubits'][1]:.2f}), above 1 in {R['both_roles']['flag_above_relay']}"
    )
    for kind, c in o["costreak"].items():
        extra = "".join(f", {k} {v:.4f}" for k, v in c.items() if k.startswith("p_two"))
        print(
            f"co-streak {kind}: {1e4 * c['chip']:.2f} +- {1e4 * c['se_qubit_dyadic']:.2f} (dyadic), pair {1e4 * c['se_pair_clustered']:.2f}, z {c['z_dyadic']:.1f}{extra}"
        )
    e = o["example_trace"]
    print(
        f"example: {e['job']} q{e['qubit']} {e['readout']}, "
        + ", ".join(f"{k} {v:.3f}" for k, v in e["parameters"].items())
    )


if __name__ == "__main__":
    main()

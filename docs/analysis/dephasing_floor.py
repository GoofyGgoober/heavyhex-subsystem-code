"""Is the idle test's dephasing a multiple of IBM's, or IBM's plus a floor? (App. f)

f is defined by 1/T2_DD = 1/(2 T1) + f Gamma_phi, with Gamma_phi = 1/T2 - 1/(2 T1) from IBM's
echo T2 (capped at 2 T1). So Gamma_dd = f Gamma_phi is the dephasing rate beyond T1 that the
idle test measures, per data qubit and job (idle_f_t2.qubit_jobs, the 329 that it pins). Three
descriptions of it:

  multiple   Gamma_dd = c Gamma_phi              (the model: one f for every qubit)
  floor      Gamma_dd = Gamma_0
  additive   Gamma_dd = k Gamma_phi + Gamma_0

fitted (i) to the medians per physical qubit (36 points) by least squares, with a bootstrap
over physical qubits, also without the three qubits with T2 < 20 us; (ii) to every qubit-job
with weights 1 / (sigma_f Gamma_phi)^2, with chi2 per degree of freedom and an F test for the
floor (the qubits scatter far beyond the idle test's noise, so the F test, not chi2, compares).

The floor as a phase flip: p = (1 - exp(-Gamma_0 t)) / 2 over the decoupled idle t of a data
qubit in one round, from the circuits as sent (X memory, eight rounds minus four, over four),
beside the model's own decoupled phase flip per round at the job's f and the one the idle test
measures. Also the model's whole idle error per data qubit and round at the job's f, per patch.

Seconds, one process.

    python docs/analysis/dephasing_floor.py     # writes results/dephasing_floor.json
"""

from __future__ import annotations

from collections import defaultdict
from math import exp, log

import numpy as np
from idle_f_t2 import qubit_jobs
from scipy import stats

from heavyhex.experiment.circuits import H_FRAME

from common import Job, job_folders, memory_index, write_json

BOOT, SEED = 2000, 1
SHORT_T2_US = 20


# ------------------------------------------------------------------------------ the idles
def decoupled_idle_us(job) -> dict[int, float]:
    """Each data qubit's decoupled wait per round (us): the delays that noisy_scheduled gives
    decoupled decay (next to an X pulse on the same qubit, not in |0> or the H frame)."""
    data = {
        q for d, chip in job.info["chips"].items() for q in chip[: int(d) ** 2]
    }  # data qubit q is the flagged circuit's qubit q
    per: dict[int, list[float]] = defaultdict(list)
    for rounds in (4, 8):
        circuit = job.circuits[memory_index("X", rounds)]
        ops = [(item, [circuit.find_bit(q).index for q in item.qubits]) for item in circuit.data]
        last: dict[int, int] = {}
        decoupled: set[int] = set()
        for i, (item, qubits) in enumerate(ops):
            if item.operation.name == "barrier":
                continue
            for q in qubits:
                if q in last:
                    pair = {item.operation.name, ops[last[q]][0].operation.name}
                    if pair == {"delay", "x"}:
                        decoupled.add(i if item.operation.name == "delay" else last[q])
                last[q] = i
        total: dict[int, float] = defaultdict(float)
        in_ground, h_frame = set(range(circuit.num_qubits)), set()
        for i, (item, qubits) in enumerate(ops):
            name = item.operation.name
            if name == "barrier" and item.operation.label == H_FRAME:
                h_frame.update(qubits)
            elif name == "delay":
                (q,) = qubits
                if q in in_ground or q in h_frame:
                    h_frame.discard(q)
                elif i in decoupled and q in data:
                    total[q] += item.operation.duration * job.info["dt"] * 1e6
            elif name in ("sx", "x", "cz"):
                in_ground.difference_update(qubits)
            elif name == "reset":
                in_ground.update(qubits)
        for q in data:
            per[q].append(total[q])
    return {q: (v[1] - v[0]) / 4 for q, v in per.items()}


def phase_flip(t: float, t1: float, rate_beyond_t1: float) -> float:
    """pz of noisy.idle_error over t with 1/T2 = 1/(2 T1) + rate_beyond_t1 (us, 1/us)."""
    px = (1 - exp(-t / t1)) / 4
    return (1 - exp(-t * (1 / (2 * t1) + rate_beyond_t1))) / 2 - px


def model_idle_error(job: Job, idle: dict[int, float]) -> dict[str, dict[str, float]]:
    """The model's idle error per data qubit and round over its decoupled wait, at the job's f,
    medians over each patch's data qubits: every Pauli (px + py + pz), and the Z or Y part that
    flips X memory (pz + py). T2 is capped at 2 T1, as in the model."""
    cal = job.calibration
    out = {}
    for d, chip in job.info["chips"].items():
        total, phase = [], []
        for q in chip[: int(d) ** 2]:
            t1 = cal.t1_us[q]
            gamma_phi = 1 / min(cal.t2_us[q], 2 * t1) - 1 / (2 * t1)
            px = (1 - exp(-idle[q] / t1)) / 4
            pz = phase_flip(idle[q], t1, job.f * gamma_phi)
            total.append(2 * px + pz)
            phase.append(pz + px)
        out[f"d{d}"] = {"all_paulis": float(np.median(total)), "z_or_y": float(np.median(phase))}
    return out


# ------------------------------------------------------------------------------- the fits
def least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """The three descriptions by (weighted) least squares: parameters, residual sum, AIC."""
    w = np.ones_like(x) if w is None else w
    n = len(x)
    out = {}
    designs = {
        "multiple": np.c_[x],
        "floor": np.c_[np.ones(n)],
        "additive": np.c_[x, np.ones(n)],
    }
    for name, X in designs.items():
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        ss = float((w * (y - X @ beta) ** 2).sum())
        p = X.shape[1]
        out[name] = {
            "params": beta.tolist(),
            "residual_sum": ss,
            "dof": n - p,
            "aic": n * log(ss / n) + 2 * p,
        }
    tot = float((w * (y - (w * y).sum() / w.sum()) ** 2).sum())
    for v in out.values():
        v["r2_about_the_mean"] = 1 - v["residual_sum"] / tot
    ss_m, ss_a = out["multiple"]["residual_sum"], out["additive"]["residual_sum"]
    F = (ss_m - ss_a) / (ss_a / (n - 2))
    out["F_test_floor_over_multiple"] = {
        "F": F,
        "dof": [1, n - 2],
        "p": float(stats.f.sf(F, 1, n - 2)),
    }
    return out


def by_qubit(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Per physical qubit: median Gamma_phi, median Gamma_dd, median T2."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["physical_qubit"]].append(r)
    qubits = sorted(groups)
    g_phi = np.array([np.median([r["gamma_phi_per_us"] for r in groups[q]]) for q in qubits])
    g_dd = np.array([np.median([r["gamma_dd_per_us"] for r in groups[q]]) for q in qubits])
    t2 = np.array([np.median([r["t2_us"] for r in groups[q]]) for q in qubits])
    return g_phi, g_dd, t2, qubits


def fit_qubits(g_phi: np.ndarray, g_dd: np.ndarray) -> dict:
    fit = least_squares(g_phi, g_dd)
    idx = np.random.default_rng(SEED).integers(0, len(g_phi), size=(BOOT, len(g_phi)))
    boot = np.array(
        [least_squares(g_phi[i], g_dd[i])["additive"]["params"] for i in idx]
    )  # BOOT x (k, Gamma_0)
    mult = np.array([least_squares(g_phi[i], g_dd[i])["multiple"]["params"][0] for i in idx])
    fit["additive"]["ci95_bootstrap_over_qubits"] = {
        "k": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        "gamma_0_per_us": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
    }
    fit["multiple"]["ci95_bootstrap_over_qubits"] = np.percentile(mult, [2.5, 97.5]).tolist()
    fit["physical_qubits"] = len(g_phi)
    return fit


def main() -> None:
    rows = [r for r in qubit_jobs() if r["measured"]]
    for r in rows:
        r["gamma_dd_per_us"] = r["f"] * r["gamma_phi_per_us"]
    idle, f_job, budget = {}, {}, {}
    for folder in job_folders():  # one job at a time: its circuits take memory
        j = Job(folder)
        idle[folder.name], f_job[folder.name] = decoupled_idle_us(j), j.f
        budget[j.name] = model_idle_error(j, idle[folder.name])
    t_round = [idle[r["job"]][r["physical_qubit"]] for r in rows]
    t_med = float(np.median(t_round))

    def flip(g0: float) -> float:
        return (1 - exp(-g0 * t_med)) / 2

    g_phi, g_dd, t2, qubits = by_qubit(rows)
    long_t2 = t2 >= SHORT_T2_US
    per_qubit = fit_qubits(g_phi, g_dd)
    per_qubit_long = fit_qubits(g_phi[long_t2], g_dd[long_t2])
    for fit in (per_qubit, per_qubit_long):
        a = fit["additive"]
        a["floor_phase_flip_per_round"] = flip(a["params"][1])
        a["floor_phase_flip_per_round_ci95"] = [
            flip(g) for g in a["ci95_bootstrap_over_qubits"]["gamma_0_per_us"]
        ]

    x = np.array([r["gamma_phi_per_us"] for r in rows])
    y = np.array([r["gamma_dd_per_us"] for r in rows])
    s = np.array([r["sigma_f"] * r["gamma_phi_per_us"] for r in rows])
    weighted = least_squares(x, y, 1 / s**2)
    for v in weighted.values():
        if "residual_sum" in v:
            v["chi2_per_dof"] = v["residual_sum"] / v["dof"]
    weighted["additive"]["floor_phase_flip_per_round"] = flip(weighted["additive"]["params"][1])

    # each qubit-job's phase flip per round: the model's, the idle test's, the additive fit's
    k_add, g0 = per_qubit["additive"]["params"]
    flips = defaultdict(list)
    for r, t in zip(rows, t_round):
        t1, gp = r["t1_us"], r["gamma_phi_per_us"]
        flips["model_at_job_f"].append(phase_flip(t, t1, f_job[r["job"]] * gp))
        flips["idle_test"].append(phase_flip(t, t1, r["gamma_dd_per_us"]))
        flips["additive_fit"].append(phase_flip(t, t1, k_add * gp + g0))
        flips["T1_only"].append(phase_flip(t, t1, 0.0))
    out = {
        "what": "Gamma_dd = f Gamma_phi (1/us), the idle test's dephasing beyond T1, against IBM's "
        "Gamma_phi: one multiple, a floor, or both. Phase flips per round over the decoupled idle "
        "of a data qubit (X memory as sent).",
        "qubit_jobs": len(rows),
        "per_physical_qubit (medians, least squares)": per_qubit,
        f"per_physical_qubit, T2 >= {SHORT_T2_US} us": per_qubit_long,
        "per_qubit_job (weights 1/(sigma_f Gamma_phi)^2)": weighted,
        "gamma_dd_median_per_us": float(np.median(g_dd)),
        "gamma_phi_median_per_us": float(np.median(g_phi)),
        "floor_share_of_median_gamma_dd": g0 / float(np.median(g_dd)),
        "decoupled_idle_per_round_us": {
            "median": t_med,
            "range": [float(min(t_round)), float(max(t_round))],
        },
        "phase_flip_per_round_median": {k: float(np.median(v)) for k, v in flips.items()},
        "phase_flip_per_round_mean": {k: float(np.mean(v)) for k, v in flips.items()},
        "model_idle_error_per_round": {
            "what": "the model's idle error of a data qubit per round over its decoupled wait, "
            "at the job's f; medians over each patch's data qubits, then their range over the "
            "twenty patch-jobs",
            "by_job": budget,
            "range": {
                kind: [
                    min(p[kind] for b in budget.values() for p in b.values()),
                    max(p[kind] for b in budget.values() for p in b.values()),
                ]
                for kind in ("all_paulis", "z_or_y")
            },
        },
    }
    write_json("dephasing_floor.json", out)

    print(f"{len(rows)} qubit-jobs, {len(qubits)} physical qubits")
    for label, fit in (
        ("per physical qubit", per_qubit),
        (f"T2 >= {SHORT_T2_US} us", per_qubit_long),
    ):
        a, m, f0 = fit["additive"], fit["multiple"], fit["floor"]
        ci = a["ci95_bootstrap_over_qubits"]
        lo, hi = a["floor_phase_flip_per_round_ci95"]
        print(
            f"{label}: additive k = {a['params'][0]:.2f} ({ci['k'][0]:.2f}-{ci['k'][1]:.2f}), "
            f"Gamma_0 = {a['params'][1]:.4f}/us ({ci['gamma_0_per_us'][0]:.4f}-{ci['gamma_0_per_us'][1]:.4f}), "
            f"floor flip per round {100 * a['floor_phase_flip_per_round']:.2f}% ({100 * lo:.2f}-{100 * hi:.2f}%), "
            f"R2 {a['r2_about_the_mean']:.2f}, AIC {a['aic']:.1f}; multiple c = {m['params'][0]:.2f}, "
            f"R2 {m['r2_about_the_mean']:.2f}, AIC {m['aic']:.1f}; floor R2 {f0['r2_about_the_mean']:.2f}, "
            f"AIC {f0['aic']:.1f}; F = {fit['F_test_floor_over_multiple']['F']:.1f}, "
            f"p = {fit['F_test_floor_over_multiple']['p']:.2g}"
        )
    w = weighted
    print(
        f"qubit-jobs weighted: chi2/nu multiple {w['multiple']['chi2_per_dof']:.1f}, floor "
        f"{w['floor']['chi2_per_dof']:.1f}, additive {w['additive']['chi2_per_dof']:.1f} "
        f"(k = {w['additive']['params'][0]:.2f}, Gamma_0 = {w['additive']['params'][1]:.4f}/us, "
        f"floor flip per round {100 * w['additive']['floor_phase_flip_per_round']:.2f}%); "
        f"F = {w['F_test_floor_over_multiple']['F']:.1f}, p = {w['F_test_floor_over_multiple']['p']:.2g}"
    )
    print(
        f"decoupled idle per round {t_med:.2f} us; median phase flip per round: "
        + ", ".join(f"{k} {100 * v:.2f}%" for k, v in out["phase_flip_per_round_median"].items())
    )
    r = out["model_idle_error_per_round"]["range"]
    print(
        "model idle error per data qubit and round (patch medians): every Pauli "
        f"{100 * r['all_paulis'][0]:.1f}-{100 * r['all_paulis'][1]:.1f}%, Z or Y "
        f"{100 * r['z_or_y'][0]:.1f}-{100 * r['z_or_y'][1]:.1f}%"
    )


if __name__ == "__main__":
    main()

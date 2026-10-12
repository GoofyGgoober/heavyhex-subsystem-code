"""Did the chip heat up during a job? Error rates against time within each ~15 s job.

Exploratory, for the revision; uses no QPU time and changes nothing in src/ or runs/.
Reads per_shot.npz and per_shot_meta.json (written by extract.py) and writes heating.json.

Time order. The README's assumption (H1): IBM runs a job shot by shot across all its
pieces, so a piece's shot index j (0..624) is time order and every piece's shot j runs at
about the same moment. Each memory circuit has 4 pieces of 625 shots, each idle readout 2.
Bin b of 10 holds shots j with floor(10 j / 625) = b (62 or 63 shots of every piece), so
bin b is the b-th tenth of the job's shots in time, about 1.24 s each (IBM's rule puts
the shots at 14.43 s - 2 s of loading = 12.4 s). t_b = (b + 0.5) / 10.
The alternative (H2): IBM runs the pieces one after another; then the time order is the
piece's position k in the job (0..67, run.json "order") and t_k = (k + 0.5) / 68. Both
are tested; neither can be checked against per-shot timestamps, which the job does not
keep (job.json has only IBM's created/running/finished times, and the SamplerV2 result
metadata, which may hold execution spans, was not saved). Section "ordering" tests which
one the data support: (1) correlation of consecutive shots under each ordering, (2) do
different pieces of one job move together over j (permutation test, against pairs of
pieces from different jobs), (3) a qubit that switches abruptly mid-job: does every piece
step at the same shot index?

Series. For each job, circuit and patch, a per-shot value v (stabilizer detection events,
flag/relay firings in Z memory, decoded logical failure, idle-test ancilla 1s, idle-test
data 1s in the X readout). Bin means y_b are divided by the series mean, so a series reads
1 on average and the slope s of y against t is the fractional change from the start
(t = 0) to the end (t = 1) of the job. Each bin's standard error is sqrt(var / n_b) /
mean, var the within-bin pooled per-shot variance of v: binomial for a 0/1 value, and for
a count it includes the correlation between detectors in one shot (leakage bursts), which
plain binomial noise per detector would miss (the ratio is reported as "overdispersion").

Trend. One slope common to the series, each with its own intercept, by weighted least
squares; its standard error is 1/sqrt(sum w (t - tbar)^2), multiplied by sqrt(chi2/dof) of
the residuals when that is above 1. Per job the same, over that job's series. Across the 15
jobs: inverse-variance (fixed) and DerSimonian-Laird random-effects means, the plain mean of
the 15 slopes with Student's t (14 dof), the count of positive slopes (sign test) and each
job's z (two-sided p; Bonferroni over 15 jobs: |z| > 2.94).
Null: the same statistics on the model's own drift-free shots (stim, same shot counts).

    python docs/diagnosis/revision/heating/heating.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "docs" / "diagnosis"))

from heavyhex.experiment.circuits import ROUNDS, SETTINGS  # noqa: E402
from heavyhex.simulation.noisy import fit_per_round  # noqa: E402

NBINS = 10
BONFERRONI_Z = float(stats.norm.isf(0.05 / 15 / 2))  # 15 jobs, two-sided 5%
LOAD_SECONDS = 2.0  # IBM's rule (run.py): the rest of the estimate is the shots

saved = np.load(HERE / "per_shot.npz")
META = json.loads((HERE / "per_shot_meta.json").read_text())["jobs"]
JOBS = list(META)


def arr(job: str, k: int, p: str, what: str) -> np.ndarray:
    return saved[f"{job}/{k}/{p}/{what}"]


# ---------------------------------------------------------------- series ----------------

def memory_series(what: str, basis: str | None = None, null: bool = False):
    """(job, label, V) with V pieces x shots, for every memory circuit and patch."""
    out = []
    for job in JOBS:
        for i, s in enumerate(SETTINGS):
            if s.kind != "memory" or (basis and s.basis != basis):
                continue
            if what == "flag" and s.basis != "Z":
                continue
            info = META[job]["settings"][str(i)]
            for p in info["patches"]:
                key = ("null_" if null else "") + what
                V = np.stack([arr(job, k, p, key) for k in info["pieces"]]).astype(float)
                out.append((job, f"{s} d={p}", V, info["pieces"]))
    return out


def idle_series(what: str, readout: str | None = None):
    out = []
    for job in JOBS:
        for i, s in enumerate(SETTINGS):
            if s.kind != "idle" or (readout and s.readout != readout):
                continue
            info = META[job]["settings"][str(i)]
            for p in info["patches"]:
                if what == "anc":
                    V = np.stack([arr(job, k, p, "anc") for k in info["pieces"]]).astype(float)
                else:
                    V = np.stack([arr(job, k, p, "data").sum(axis=1) for k in info["pieces"]]).astype(float)
                out.append((job, f"{s} d={p}", V, info["pieces"]))
    return out


def binned(V: np.ndarray, nb: int = NBINS):
    """t, normalized bin means, their standard errors; and the pooled per-shot variance."""
    n = V.shape[1]
    b = (np.arange(n) * nb) // n
    means = np.array([V[:, b == q].mean() for q in range(nb)])
    counts = np.array([V[:, b == q].size for q in range(nb)])
    resid = V - means[b][None, :]
    var = float((resid**2).sum() / (V.size - nb))
    m = float(V.mean())
    t = (np.arange(nb) + 0.5) / nb
    if m == 0 or var == 0:
        return None
    return t, means / m, np.sqrt(var / counts) / m, var, m


def by_piece(V: np.ndarray, pieces: list[int], total: int):
    """t = the piece's position in the job (H2), normalized piece means, their errors."""
    m = float(V.mean())
    resid = V - V.mean(axis=1, keepdims=True)
    var = float((resid**2).sum() / (V.size - V.shape[0]))
    if m == 0 or var == 0:
        return None
    t = (np.array(pieces) + 0.5) / total
    return t, V.mean(axis=1) / m, np.full(len(t), sqrt(var / V.shape[1]) / m), var, m


def common_slope(rows):
    """rows of (t, y, sigma): slope shared by all rows, each with its own intercept."""
    num = den = 0.0
    keep = []
    for t, y, s in rows:
        w = 1 / s**2
        tb, yb = (w * t).sum() / w.sum(), (w * y).sum() / w.sum()
        num += (w * (t - tb) * (y - yb)).sum()
        den += (w * (t - tb) ** 2).sum()
        keep.append((t, y, w, tb, yb))
    slope = num / den
    chi2 = sum((w * (y - yb - slope * (t - tb)) ** 2).sum() for t, y, w, tb, yb in keep)
    dof = sum(len(t) - 1 for t, *_ in keep) - 1
    red = chi2 / dof if dof > 0 else 1.0
    se = sqrt(1 / den)
    return {"slope": float(slope), "se_shot_noise": se, "chi2_red": float(red),
            "se": se * sqrt(max(red, 1.0)), "series": len(rows)}


def pooled_curve(rows):
    """Inverse-variance mean over series of the normalized value in each bin."""
    y = np.array([r[1] for r in rows])
    w = 1 / np.array([r[2] for r in rows]) ** 2
    mean = (w * y).sum(axis=0) / w.sum(axis=0)
    se = 1 / np.sqrt(w.sum(axis=0))
    # extra scatter between series in a bin, as chi2/dof about the bin's mean
    red = ((w * (y - mean) ** 2).sum(axis=0) / max(len(rows) - 1, 1))
    se = se * np.sqrt(np.maximum(red, 1.0))
    return mean.tolist(), se.tolist()


def meta_analysis(per_job: dict[str, dict]) -> dict:
    s = np.array([v["slope"] for v in per_job.values()])
    e = np.array([v["se"] for v in per_job.values()])
    w = 1 / e**2
    fe = float((w * s).sum() / w.sum())
    fe_se = float(sqrt(1 / w.sum()))
    q = float((w * (s - fe) ** 2).sum())
    k = len(s)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    wr = 1 / (e**2 + tau2)
    re = float((wr * s).sum() / wr.sum())
    re_se = float(sqrt(1 / wr.sum()))
    tcrit = float(stats.t.ppf(0.975, k - 1))
    plain, plain_se = float(s.mean()), float(s.std(ddof=1) / sqrt(k))
    positive = int((s > 0).sum())
    z = s / e
    return {
        "fixed": {"slope": fe, "se": fe_se, "ci95": [fe - 1.96 * fe_se, fe + 1.96 * fe_se]},
        "random": {"slope": re, "se": re_se, "ci95": [re - 1.96 * re_se, re + 1.96 * re_se],
                   "tau": float(sqrt(tau2)), "Q": q, "Q_dof": k - 1,
                   "Q_p": float(stats.chi2.sf(q, k - 1))},
        "plain_mean_t": {"slope": plain, "se": plain_se,
                         "ci95": [plain - tcrit * plain_se, plain + tcrit * plain_se]},
        "jobs": k,
        "positive_slopes": positive,
        "sign_test_p_two_sided": float(stats.binomtest(positive, k, 0.5).pvalue),
        "z_max": float(z.max()), "z_min": float(z.min()),
        "jobs_rising_nominal_p05": [j for j, v in per_job.items() if v["z"] > 1.96],
        "jobs_falling_nominal_p05": [j for j, v in per_job.items() if v["z"] < -1.96],
        "jobs_rising_bonferroni": [j for j, v in per_job.items() if v["z"] > BONFERRONI_Z],
        "jobs_falling_bonferroni": [j for j, v in per_job.items() if v["z"] < -BONFERRONI_Z],
        "mean_z2": float((z**2).mean()),
        "stouffer_z": float(z.sum() / sqrt(k)),
    }


def trend(series, mode: str = "time", nb: int = NBINS) -> dict:
    """Pooled and per-job slope of the normalized value against time."""
    rows_by_job = defaultdict(list)
    over = []
    for job, label, V, pieces in series:
        if mode == "time":
            r = binned(V, nb)
        else:
            r = by_piece(V, pieces, len(META[job]["order"]))
        if r is None:
            continue
        t, y, s, var, m = r
        rows_by_job[job].append((t, y, s))
        over.append((job, label, V, var, m))
    allrows = [r for rows in rows_by_job.values() for r in rows]
    out = {"pooled": common_slope(allrows)}
    per_job = {}
    for job, rows in rows_by_job.items():
        c = common_slope(rows)
        c["z"] = c["slope"] / c["se"]
        c["p_two_sided"] = float(2 * stats.norm.sf(abs(c["z"])))
        per_job[job] = c
    out["per_job"] = per_job
    out["across_jobs"] = meta_analysis(per_job)
    if mode == "time" and nb == NBINS:
        mean, se = pooled_curve(allrows)
        out["pooled_bins"] = {"t": ((np.arange(nb) + 0.5) / nb).tolist(), "y": mean, "se": se}
        out["last_minus_first_bin"] = {"diff": mean[-1] - mean[0], "se": sqrt(se[-1] ** 2 + se[0] ** 2)}
    return out


def overdispersion(series, per: str) -> dict:
    """Per-shot variance of the count over what independent detectors would give."""
    ratios = []
    for job, label, V, pieces in series:
        n = per_count(job, label, per)
        if not n:
            continue
        p = V.mean() / n
        if 0 < p < 1:
            ratios.append(V.var() / (n * p * (1 - p)))
    return {"median": float(np.median(ratios)), "min": float(np.min(ratios)), "max": float(np.max(ratios))}


def per_count(job: str, label: str, per: str) -> int:
    for i, s in enumerate(SETTINGS):
        if label.startswith(str(s) + " d="):
            p = label.split("d=")[-1]
            info = META[job]["settings"][str(i)]
            if per in ("stab", "flag"):
                return info["columns"][p][per]
            return info["ancilla_bits"][p]
    return 0


# ---------------------------------------------------------------- epsilon per bin -------

def epsilon_series(null: bool = False, nb: int = NBINS):
    """eps per round in each time bin, by the repo's fit_per_round, per (job, basis, patch)."""
    rows_by_job = defaultdict(list)
    table = []
    for job in JOBS:
        for basis in "XZ":
            idx = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == basis]
            for p in META[job]["settings"][str(idx[0])]["patches"]:
                F = [np.stack([arr(job, k, p, ("null_" if null else "") + "fail")
                               for k in META[job]["settings"][str(i)]["pieces"]]) for i in idx]
                n = F[0].shape[1]
                b = (np.arange(n) * nb) // n
                full = fit_per_round(ROUNDS, [f.mean() for f in F], F[0].size)
                eps, unc = [], []
                for q in range(nb):
                    fit = fit_per_round(ROUNDS, [f[:, b == q].mean() for f in F], F[0][:, b == q].size)
                    eps.append(fit.per_round)
                    unc.append(fit.uncertainty)
                eps, unc = np.array(eps), np.array(unc)
                ok = np.isfinite(unc) & (unc > 0)
                if ok.sum() < 3:
                    continue
                t = (np.arange(nb) + 0.5) / nb
                # normalized by the mean over bins (a fit to a tenth of the shots sits a little below
                # the fit to all of them, the same in every bin); one error per series, the mean of
                # the bins' fit errors, so a bin's weight does not follow its own fluctuation
                norm = float(eps[ok].mean())
                rows_by_job[job].append((t[ok], eps[ok] / norm, np.full(ok.sum(), unc[ok].mean() / norm)))
                table.append({"job": job, "basis": basis, "patch": p, "eps_full": full.per_round,
                              "eps_full_se": full.uncertainty, "eps_bins": eps.tolist(),
                              "eps_bins_se": unc.tolist()})
    allrows = [r for rows in rows_by_job.values() for r in rows]
    out = {"pooled": common_slope(allrows)}
    per_job = {}
    for job, rows in rows_by_job.items():
        c = common_slope(rows)
        c["z"] = c["slope"] / c["se"]
        c["p_two_sided"] = float(2 * stats.norm.sf(abs(c["z"])))
        per_job[job] = c
    out["per_job"] = per_job
    out["across_jobs"] = meta_analysis(per_job)
    full_rows = [r for r in allrows if len(r[0]) == nb]
    mean, se = pooled_curve(full_rows)
    out["pooled_bins"] = {"t": ((np.arange(nb) + 0.5) / nb).tolist(), "y": mean, "se": se,
                          "series_with_all_bins": len(full_rows), "series": len(allrows)}
    out["last_minus_first_bin"] = {"diff": mean[-1] - mean[0], "se": sqrt(se[-1] ** 2 + se[0] ** 2)}
    out["table"] = table
    # slope per (job, basis, patch): the 40 series of the prespecified drift check
    each = []
    for job, rows in rows_by_job.items():
        for r in rows:
            c = common_slope([r])
            each.append(c["slope"] / c["se"])
    each = np.array(each)
    out["per_series_40"] = {"n": len(each), "positive": int((each > 0).sum()),
                            "stouffer_z": float(each.sum() / sqrt(len(each))),
                            "max_z": float(each.max()), "min_z": float(each.min())}
    return out


def reproduce_drift() -> dict:
    """analysis.json's first/second-half eps, rebuilt from the per-shot failures here."""
    worst = 0.0
    rows = []
    for job in JOBS:
        analysis = json.loads((REPO / "runs" / META[job]["folder"] / "analysis.json").read_text())
        halves = META[job]["halves"]
        for basis in "XZ":
            idx = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == basis]
            for p in halves[str(idx[0])]:
                fits = [fit_per_round(ROUNDS, [halves[str(i)][p][h] for i in idx], halves[str(idx[0])][p][2 + h])
                        for h in (0, 1)]
                ref = analysis["drift"][f"{basis} eps{p}"]
                diff = max(abs(fits[0].per_round - ref["early"]), abs(fits[1].per_round - ref["late"]))
                worst = max(worst, diff)
                rows.append({"job": job, "basis": basis, "patch": p, "early": fits[0].per_round,
                             "late": fits[1].per_round, "analysis_early": ref["early"],
                             "analysis_late": ref["late"], "analysis_change_sigma": ref["change"]})
    return {"max_abs_difference": worst, "rows": rows,
            "second_half_worse": sum(r["late"] > r["early"] for r in rows), "comparisons": len(rows)}


# ---------------------------------------------------------------- idle test extras -------

def idle_contrast(nb: int = NBINS, floor: float = 0.3) -> dict:
    """Contrast loss 1 - sqrt(<X>^2 + <Y>^2) of the idle test's data qubits in each bin.

    A drifting qubit frequency turns X into Y and moves the X- and Y-readout 1-rates in
    opposite directions without any loss; the contrast is blind to it, so it is the idle
    test's measure of dephasing. Per (job, rounds, patch): the mean over data qubits whose
    whole-job contrast is above `floor` (below it the delta-method error fails); the
    standard error by the delta method from each readout's binomial noise (2 pieces x
    62-63 shots per readout per bin), qubits taken as independent."""
    rows_by_job = defaultdict(list)
    used = 0
    for job in JOBS:
        for n in sorted({s.rounds for s in SETTINGS if s.kind == "idle"}):
            ix = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "X")
            iy = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "Y")
            for p in META[job]["settings"][str(ix)]["patches"]:
                X = np.stack([arr(job, k, p, "data") for k in META[job]["settings"][str(ix)]["pieces"]])
                Y = np.stack([arr(job, k, p, "data") for k in META[job]["settings"][str(iy)]["pieces"]])
                ex, ey = 1 - 2 * X.mean(axis=(0, 1)), 1 - 2 * Y.mean(axis=(0, 1))
                keep = np.sqrt(ex**2 + ey**2) > floor
                if keep.sum() == 0:
                    continue
                used += int(keep.sum())
                b = (np.arange(X.shape[1]) * nb) // X.shape[1]
                loss, se = [], []
                for q in range(nb):
                    x = 1 - 2 * X[:, b == q][..., keep].mean(axis=(0, 1))
                    y = 1 - 2 * Y[:, b == q][..., keep].mean(axis=(0, 1))
                    nx, ny = X[:, b == q].shape[0] * (b == q).sum(), Y[:, b == q].shape[0] * (b == q).sum()
                    c = np.sqrt(x**2 + y**2)
                    vc = (x**2 * (1 - x**2) / nx + y**2 * (1 - y**2) / ny) / np.maximum(c, 1e-9) ** 2
                    loss.append(float((1 - c).mean()))
                    se.append(float(np.sqrt(vc.sum()) / keep.sum()))
                t = (np.arange(nb) + 0.5) / nb
                norm = float(np.mean(loss))  # the per-bin contrast is noisier, so biased up, in every bin alike
                rows_by_job[job].append((t, np.array(loss) / norm, np.array(se) / norm))
    allrows = [r for rows in rows_by_job.values() for r in rows]
    out = {"pooled": common_slope(allrows), "qubit_settings_used": used, "contrast_floor": floor}
    per_job = {}
    for job, rows in rows_by_job.items():
        c = common_slope(rows)
        c["z"] = c["slope"] / c["se"]
        c["p_two_sided"] = float(2 * stats.norm.sf(abs(c["z"])))
        per_job[job] = c
    out["per_job"] = per_job
    out["across_jobs"] = meta_analysis(per_job)
    mean, se = pooled_curve(allrows)
    out["pooled_bins"] = {"t": ((np.arange(nb) + 0.5) / nb).tolist(), "y": mean, "se": se}
    out["last_minus_first_bin"] = {"diff": mean[-1] - mean[0], "se": sqrt(se[-1] ** 2 + se[0] ** 2)}
    return out


def idle_phase_drift(nb: int = NBINS, floor: float = 0.3, round_us: float = 7.9) -> dict:
    """The idle test's Bloch-vector angle against time: does the qubit frequency drift?

    Per (job, rounds n, patch, data qubit) with whole-job contrast above `floor`: the angle
    of (<X>_b + i <Y>_b) relative to the whole job's, per bin; its slope over the job by
    weighted least squares (delta-method errors), divided by n rounds of `round_us`, is a
    change of the qubit's frequency over the job, in Hz. Pooled by inverse variance."""
    slopes, ses = [], []
    for job in JOBS:
        for n in sorted({s.rounds for s in SETTINGS if s.kind == "idle"}):
            ix = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "X")
            iy = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "Y")
            for p in META[job]["settings"][str(ix)]["patches"]:
                X = np.stack([arr(job, k, p, "data") for k in META[job]["settings"][str(ix)]["pieces"]])
                Y = np.stack([arr(job, k, p, "data") for k in META[job]["settings"][str(iy)]["pieces"]])
                full = (1 - 2 * X.mean(axis=(0, 1))) + 1j * (1 - 2 * Y.mean(axis=(0, 1)))
                b = (np.arange(X.shape[1]) * nb) // X.shape[1]
                t = (np.arange(nb) + 0.5) / nb
                for q in np.flatnonzero(np.abs(full) > floor):
                    ang, var = [], []
                    for c in range(nb):
                        x = 1 - 2 * X[:, b == c, q].mean()
                        y = 1 - 2 * Y[:, b == c, q].mean()
                        m = X[:, b == c, q].size
                        z = (x + 1j * y) * np.conj(full[q]) / abs(full[q])
                        ang.append(np.angle(z))
                        c2 = max(x * x + y * y, 1e-6)
                        var.append((y * y * (1 - x * x) / m + x * x * (1 - y * y) / m) / c2**2)
                    w = 1 / np.array(var)
                    tb = (w * t).sum() / w.sum()
                    sxx = (w * (t - tb) ** 2).sum()
                    slope = (w * (t - tb) * (np.array(ang) - (w * np.array(ang)).sum() / w.sum())).sum() / sxx
                    hz = 1 / (2 * np.pi * n * round_us * 1e-6)
                    slopes.append(slope * hz)
                    ses.append(sqrt(1 / sxx) * hz)
    s, e = np.array(slopes), np.array(ses)
    w = 1 / e**2
    mean, se = float((w * s).sum() / w.sum()), float(sqrt(1 / w.sum()))
    z = s / e
    red = float((z**2).sum() / (len(z) - 1))
    return {"qubit_settings": len(s), "frequency_change_over_job_hz": mean,
            "se_hz": se * sqrt(max(red, 1.0)), "chi2_red": red,
            "positive": int((s > 0).sum()), "median_abs_hz": float(np.median(np.abs(s)))}


def idle_ancilla_per_qubit(nb: int = NBINS) -> dict:
    """The idle test's ancilla 1-rate against time, for each physical ancilla qubit.

    In the idle test every ancilla is only reset and read, so each 1 is a reset, readout
    or leakage error of that qubit. Raw bits from shots.npz; each register bit is mapped
    to its chip qubit through the job's own translated circuit. Global heating would push
    most qubits up together; a two-level-system or readout fluctuation moves one qubit."""
    from heavyhex.experiment import run as R

    from common import jobs as all_jobs

    rows = []
    for job in all_jobs():
        name = job.name
        circuits = job.circuits
        saved = np.load(job.folder / R.SHOTS)
        per_qubit = defaultdict(list)  # qubit -> [(t, y, s)] over idle settings
        for i, s in enumerate(SETTINGS):
            if s.kind != "idle":
                continue
            c = circuits[i]
            where = {}
            for inst in c.data:
                if inst.operation.name == "measure":
                    reg, idx = c.find_bit(inst.clbits[0]).registers[0]
                    where[(reg.name, idx)] = c.find_bit(inst.qubits[0]).index
            info = META[name]["settings"][str(i)]
            for p in info["patches"]:
                bits = np.stack([saved[f"{k}/{i}/{info_block(name, k)}/m{p}"] for k in info["pieces"]])
                qubits = np.array([where[(f"m{p}", b)] for b in range(bits.shape[2])])
                for q in np.unique(qubits):
                    V = bits[:, :, qubits == q].sum(axis=2).astype(float)
                    r = binned(V, nb)
                    if r is not None:
                        per_qubit[(p, int(q))].append(r[:3])
        for (p, q), rs in per_qubit.items():
            c = common_slope(rs)
            rows.append({"job": name, "patch": p, "qubit": q, "slope": c["slope"], "se": c["se"],
                         "z": c["slope"] / c["se"], "chi2_red": c["chi2_red"]})
    z = np.array([r["z"] for r in rows])
    return {
        "qubit_jobs": len(rows),
        "positive": int((z > 0).sum()),
        "z_above_3": int((z > 3).sum()), "z_below_minus_3": int((z < -3).sum()),
        "expected_beyond_3_each_side_if_no_drift": float(len(rows) * stats.norm.sf(3)),
        "mean_z2": float((z**2).mean()),
        "median_slope": float(np.median([r["slope"] for r in rows])),
        "largest": sorted(rows, key=lambda r: -abs(r["z"]))[:12],
        "rows": rows,
    }


def info_block(job: str, k: int) -> int:
    return META[job]["order"][k][1]


# ---------------------------------------------------------------- ordering checks -------

def piece_scores(job: str) -> np.ndarray:
    """Per piece, per shot: detection events + flags (memory) or ancilla 1s (idle), summed
    over the patches, standardized within the piece. Pieces x shots, in job order."""
    order = META[job]["order"]
    n = META[job]["shots_per_piece"]
    Z = np.zeros((len(order), n))
    for k, (i, _) in enumerate(order):
        info = META[job]["settings"][str(i)]
        v = np.zeros(n)
        for p in info["patches"]:
            if SETTINGS[i].kind == "memory":
                v += arr(job, k, p, "det") + arr(job, k, p, "flag")
            else:
                v += arr(job, k, p, "anc")
        Z[k] = (v - v.mean()) / v.std()
    return Z


def step_check(job_name: str = "2026-10-06-r2-d5", patch: str = "5", qubit: int = 51) -> dict:
    """One qubit whose readouts switch abruptly mid-job: when, and in how many pieces?

    Every readout of chip qubit `qubit` (flag, relay or idle ancilla: each should read 0)
    in every piece of the job, raw from shots.npz. The switch point j0 is the shot index
    that best splits the pooled 1-rate into two levels (binomial likelihood). Under H1
    every piece crosses j0 at the same moment, so every piece steps at j0; under H2 the
    switch falls inside one piece, and the others sit wholly before or after it, with no
    step at j0 beyond chance."""
    from heavyhex.experiment import run as R

    from common import jobs as all_jobs

    job = next(j for j in all_jobs() if j.name == job_name)
    saved = np.load(job.folder / R.SHOTS)
    order = META[job_name]["order"]
    per_piece = []
    for k, (i, block) in enumerate(order):
        c = job.circuits[i]
        where = {}
        for inst in c.data:
            if inst.operation.name == "measure":
                reg, idx = c.find_bit(inst.clbits[0]).registers[0]
                where[(reg.name, idx)] = c.find_bit(inst.qubits[0]).index
        bits = saved[f"{k}/{i}/{block}/m{patch}"]
        cols = [b for b in range(bits.shape[1]) if where[(f"m{patch}", b)] == qubit]
        if cols:
            per_piece.append((k, str(SETTINGS[i]), bits[:, cols].astype(float)))
    ones = sum(v.sum(axis=1) for _, _, v in per_piece)  # per shot index, over pieces
    reads = sum(v.shape[1] for _, _, v in per_piece)
    n = len(ones)

    def loglik(x, m):
        p = min(max(x / m, 1e-9), 1 - 1e-9)
        return x * np.log(p) + (m - x) * np.log(1 - p)

    best = max(range(20, n - 20),
               key=lambda j0: loglik(ones[:j0].sum(), reads * j0) + loglik(ones[j0:].sum(), reads * (n - j0)))
    drops = [(k, s, float(v[:best].mean()), float(v[best:].mean())) for k, s, v in per_piece]
    down = sum(a > b for _, _, a, b in drops)
    return {
        "job": job_name, "qubit": qubit, "switch_shot_index": int(best),
        "switch_fraction_of_job": best / n,
        "rate_before": float(ones[:best].sum() / (reads * best)),
        "rate_after": float(ones[best:].sum() / (reads * (n - best))),
        "pieces": len(drops), "pieces_lower_after": down,
        "sign_test_p": float(stats.binomtest(down, len(drops), 0.5).pvalue),
        "pieces_detail": drops,
    }


def adjacency() -> dict:
    """Correlation of consecutive shots under each ordering, pooled over jobs.

    H1 (shot-major): after (piece k, shot j) comes (k+1, j), and (0, j+1) after the last piece.
    H2 (piece-major): after (k, j) comes (k, j+1)."""
    acc = defaultdict(lambda: [0.0, 0])
    for job in JOBS:
        Z = piece_scores(job)
        K, n = Z.shape
        flat_h1 = Z.T.reshape(-1)  # time order under H1
        flat_h2 = Z.reshape(-1)  # time order under H2 (pieces in job order)
        for lag in (1, 2, 3, 5, 10):
            for name, flat in (("H1", flat_h1), ("H2", flat_h2)):
                prod = flat[:-lag] * flat[lag:]
                acc[f"{name} lag {lag}"][0] += prod.sum()
                acc[f"{name} lag {lag}"][1] += len(prod)
    return {key: {"corr": s / n, "se": 1 / sqrt(n), "pairs": n} for key, (s, n) in acc.items()}


def common_mode(nb: int = 25, perms: int = 2000, seed: int = 7) -> dict:
    """Do different pieces of one job vary together over the shot index j?

    Each (piece, patch) series of per-shot counts in nb bins of j, standardized across its
    own bins. Under H1 every piece's bin b runs at the same time, so any change of the
    chip on the scale of a bin (0.5 s) moves them together: positive mean pairwise
    correlation between series of the same job. Series of different jobs share no time,
    only a systematic pattern in j. Null: circular shifts of each series' bins."""
    rng = np.random.default_rng(seed)
    mats = {}
    for job in JOBS:
        order = META[job]["order"]
        rows = []
        n = META[job]["shots_per_piece"]
        b = (np.arange(n) * nb) // n
        for k, (i, _) in enumerate(order):
            info = META[job]["settings"][str(i)]
            for p in info["patches"]:
                if SETTINGS[i].kind == "memory":
                    v = (arr(job, k, p, "det") + arr(job, k, p, "flag")).astype(float)
                else:
                    v = arr(job, k, p, "anc").astype(float)
                m = np.array([v[b == q].mean() for q in range(nb)])
                if m.std() > 0:
                    rows.append((m - m.mean()) / m.std(ddof=1))
        mats[job] = np.array(rows)

    def mean_corr(Z):
        N = len(Z)
        S = Z.sum(axis=0)
        return float(((S**2).sum() - (Z**2).sum()) / ((nb - 1) * N * (N - 1)))

    same = {job: mean_corr(Z) for job, Z in mats.items()}
    null = {job: [] for job in mats}
    pooled_null = []
    for _ in range(perms):
        tot = 0.0
        for job, Z in mats.items():
            shifts = rng.integers(0, nb, len(Z))
            idx = (np.arange(nb)[None, :] + shifts[:, None]) % nb
            c = mean_corr(np.take_along_axis(Z, idx, axis=1))
            null[job].append(c)
            tot += c
        pooled_null.append(tot / len(mats))
    cross = []
    names = list(mats)
    for a in range(len(names)):
        for c in range(a + 1, len(names)):
            A, B = mats[names[a]], mats[names[c]]
            cross.append(float(A.sum(axis=0) @ B.sum(axis=0) / ((nb - 1) * len(A) * len(B))))
    per_job = {
        job: {"mean_pair_corr": same[job], "null_sd": float(np.std(null[job])),
              "p_one_sided": float((np.sum(np.array(null[job]) >= same[job]) + 1) / (perms + 1)),
              "series": len(mats[job]),
              "profile": (mats[job].mean(axis=0)).tolist()}
        for job in mats
    }
    mean_same = float(np.mean(list(same.values())))
    return {
        "bins": nb,
        "mean_pair_corr_same_job": mean_same,
        "null_mean": float(np.mean(pooled_null)), "null_sd": float(np.std(pooled_null)),
        "p_one_sided": float((np.sum(np.array(pooled_null) >= mean_same) + 1) / (perms + 1)),
        "mean_pair_corr_cross_job": float(np.mean(cross)),
        "cross_job_sd_over_pairs": float(np.std(cross) / sqrt(len(cross))),
        "per_job": per_job,
    }


# ---------------------------------------------------------------- main ------------------

def summary_line(name: str, r: dict) -> str:
    a = r["across_jobs"]
    re = a["random"]
    p = r["pooled"]
    return (f"{name:28s} pooled {100*p['slope']:+6.2f} ± {100*p['se']:.2f} %  (chi2/dof {p['chi2_red']:.2f}); "
            f"RE {100*re['slope']:+6.2f} [{100*re['ci95'][0]:+.2f}, {100*re['ci95'][1]:+.2f}] %; "
            f"positive {a['positive_slopes']}/{a['jobs']}; z range [{a['z_min']:+.2f}, {a['z_max']:+.2f}]; "
            f"Q p {re['Q_p']:.2f}")


def main() -> None:
    out: dict = {
        "what": "Fractional change of each error rate from the start (t=0) to the end (t=1) of a job's shots; "
                "slope of the series-normalized rate against time. See heating.py docstring.",
        "bins": NBINS,
        "bonferroni_z": BONFERRONI_Z,
        "seconds_per_job_of_shots": {
            job: META[job]["qpu_seconds_estimate"] - LOAD_SECONDS for job in JOBS},
        "job_wall_clock": {job: [META[job]["running_utc"], META[job]["finished_utc"],
                                 META[job]["qpu_charge_time_seconds"]] for job in JOBS},
    }
    det = memory_series("det")
    flag = memory_series("flag")
    fail = memory_series("fail")
    anc = idle_series("anc")
    dataX = idle_series("data", "X")
    dataY = idle_series("data", "Y")
    metrics = {
        "detection_events": det,
        "detection_events_X_memory": memory_series("det", "X"),
        "detection_events_Z_memory": memory_series("det", "Z"),
        "detection_events_d3": [s for s in det if s[1].endswith("d=3")],
        "detection_events_d5": [s for s in det if s[1].endswith("d=5")],
        "flags_relays_Z_memory": flag,
        "logical_failure": fail,
        "idle_ancilla_ones": anc,
        "idle_data_ones_X_readout": dataX,
        "idle_data_ones_Y_readout": dataY,
    }
    out["time_H1"] = {}
    for name, series in metrics.items():
        out["time_H1"][name] = trend(series, "time")
        print(summary_line(name, out["time_H1"][name]))
    out["time_H1"]["logical_eps_per_round"] = epsilon_series()
    print(summary_line("eps per round (fit/bin)", out["time_H1"]["logical_eps_per_round"]))
    out["time_H1"]["idle_contrast_loss"] = idle_contrast()
    print(summary_line("idle contrast loss", out["time_H1"]["idle_contrast_loss"]))
    ph = idle_phase_drift()
    out["time_H1"]["idle_phase_drift"] = ph
    print(f"idle phase drift: {ph['frequency_change_over_job_hz']:+.1f} ± {ph['se_hz']:.1f} Hz over the job "
          f"({ph['qubit_settings']} qubit-settings, chi2/dof {ph['chi2_red']:.2f}, {ph['positive']} positive)")
    pq = idle_ancilla_per_qubit()
    out["time_H1"]["idle_ancilla_per_qubit"] = pq
    print(f"idle ancilla per qubit: {pq['qubit_jobs']} qubit-jobs, {pq['positive']} rising, "
          f"z>3: {pq['z_above_3']}, z<-3: {pq['z_below_minus_3']} (expected {pq['expected_beyond_3_each_side_if_no_drift']:.2f} each), "
          f"mean z2 {pq['mean_z2']:.2f}, median slope {100*pq['median_slope']:+.2f}%")
    for r in pq["largest"]:
        print(f"    {r['job']} d={r['patch']} q{r['qubit']}: {100*r['slope']:+.1f} ± {100*r['se']:.1f} % (z {r['z']:+.2f})")

    out["unbinned_H1"] = {}
    for name in ("detection_events", "flags_relays_Z_memory", "logical_failure", "idle_ancilla_ones"):
        out["unbinned_H1"][name] = trend(metrics[name], "time", nb=META[JOBS[0]]["shots_per_piece"])
        print(summary_line("unbinned " + name, out["unbinned_H1"][name]))

    out["piece_order_H2"] = {}
    for name in ("detection_events", "flags_relays_Z_memory", "logical_failure",
                 "idle_ancilla_ones", "idle_data_ones_X_readout"):
        out["piece_order_H2"][name] = trend(metrics[name], "piece")
        print(summary_line("H2 " + name, out["piece_order_H2"][name]))

    out["null_model_shots"] = {}
    for name, series in (("detection_events", memory_series("det", null=True)),
                         ("flags_relays_Z_memory", memory_series("flag", null=True)),
                         ("logical_failure", memory_series("fail", null=True))):
        out["null_model_shots"][name] = trend(series, "time")
        print(summary_line("NULL " + name, out["null_model_shots"][name]))
    out["null_model_shots"]["logical_eps_per_round"] = epsilon_series(null=True)
    print(summary_line("NULL eps", out["null_model_shots"]["logical_eps_per_round"]))

    out["overdispersion_per_shot_variance_over_binomial"] = {
        "detection_events": overdispersion(det, "stab"),
        "flags_relays_Z_memory": overdispersion(flag, "flag"),
        "idle_ancilla_ones": overdispersion(anc, "anc"),
        "model_detection_events": overdispersion(memory_series("det", null=True), "stab"),
    }
    out["reproduce_registered_drift"] = reproduce_drift()
    print("drift check reproduced, max |diff|", out["reproduce_registered_drift"]["max_abs_difference"],
          "second half worse", out["reproduce_registered_drift"]["second_half_worse"], "/",
          out["reproduce_registered_drift"]["comparisons"])
    out["ordering"] = {"adjacent_shot_correlation": adjacency(), "common_mode": common_mode(),
                       "step": step_check()}
    st = out["ordering"]["step"]
    print(f"  step: {st['job']} q{st['qubit']} switches at shot {st['switch_shot_index']} "
          f"({st['rate_before']:.3f} -> {st['rate_after']:.3f}); lower after in {st['pieces_lower_after']}/{st['pieces']} pieces "
          f"(sign test p {st['sign_test_p']:.1e})")
    for key, v in out["ordering"]["adjacent_shot_correlation"].items():
        print(f"  {key}: {v['corr']:+.4f} ± {v['se']:.4f}")
    cm = out["ordering"]["common_mode"]
    print(f"  common mode: same job {cm['mean_pair_corr_same_job']:+.4f} (null {cm['null_mean']:+.4f} ± {cm['null_sd']:.4f}, p {cm['p_one_sided']:.4f}); "
          f"cross job {cm['mean_pair_corr_cross_job']:+.4f} ± {cm['cross_job_sd_over_pairs']:.4f}")
    for job, v in cm["per_job"].items():
        print(f"    {job}: {v['mean_pair_corr']:+.4f} (null sd {v['null_sd']:.4f}, p {v['p_one_sided']:.3f}, {v['series']} series)")
    (HERE / "heating.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

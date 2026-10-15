"""Rebuild the paper's exploratory raw-IQ summaries from Kingston NPZ records.

Refits two-state HMMs; no parameters are loaded from the exp3 analysis archive.
The joint posterior of each qubit's complete sequence selects readouts with
P(persistent | sequence)>0.9. These are descriptive, same-data associations,
not an independently trained classifier or identification of a transmon level.
Pair correlations use continuous posteriors at matched shot/cycle coordinates.
They are not onset correlations and no independent-pair uncertainty is claimed.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src:docs/analysis \
        python docs/analysis/kingston_iq.py

One process, no bootstrap or hardware access. The legacy three-state exploratory
fits are not reconstructed here and are not used by the paper.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

from followup import analysis
from followup import leakage_control as lc
from followup.placement import place
from heavyhex.patches.placement import Calibration

from common import REPO, RUNS, write_json


def posterior(observed, fit):
    """Scaled forward/backward probability of state 1, shots x cycles."""
    observed = np.asarray(observed, dtype=int)
    p, a, b = (fit[key] for key in ("pi", "a", "b"))
    transition = np.array([[1 - a, a], [b, 1 - b]])
    emission = np.array([fit["e_N"], fit["e_P"]])[:, observed].transpose(1, 2, 0)
    alpha = np.empty_like(emission)
    scale = np.empty(observed.shape)
    alpha[:, 0] = np.array([1 - p, p]) * emission[:, 0]
    scale[:, 0] = alpha[:, 0].sum(1)
    alpha[:, 0] /= scale[:, :1]
    for t in range(1, observed.shape[1]):
        alpha[:, t] = (alpha[:, t - 1] @ transition) * emission[:, t]
        scale[:, t] = alpha[:, t].sum(1)
        alpha[:, t] /= scale[:, t, None]
    beta = np.ones_like(alpha)
    for t in range(observed.shape[1] - 2, -1, -1):
        beta[:, t] = ((emission[:, t + 1] * beta[:, t + 1]) @ transition.T) / scale[:, t + 1, None]
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    return gamma[:, :, 1]


def project(points, clouds):
    """Along and perpendicular coordinates, in units of reference separation."""
    axis = clouds.one.mean - clouds.zero.mean
    norm = axis @ axis
    delta = np.stack([np.real(points), np.imag(points)], axis=-1) - clouds.zero.mean
    return delta @ axis / norm, (axis[0] * delta[..., 1] - axis[1] * delta[..., 0]) / norm


def pair_correlations(posteriors, partners):
    rows = []
    for q, r in combinations(sorted(posteriors), 2):
        x, y = posteriors[q].ravel(), posteriors[r].ravel()
        correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        rows.append(
            {
                "qubits": [q, r],
                "shared_partners": sorted(set(partners[q]) & set(partners[r])),
                "posterior_correlation": correlation,
            }
        )
    return rows


def reconstruct():
    folders = [RUNS / f"ibm_kingston-2026-10-14-exp3-{i}" for i in (1, 2, 3)]
    jobs, period, kerneled = analysis.exp3_jobs(folders)
    if not kerneled:
        raise ValueError("Raw IQ records are required")
    selection = lc.select(place(Calibration.load(folders[0] / "calibration.json"))[3].layout)
    partners = dict(zip(selection.qubits, selection.partners, strict=True))
    n = len(selection.qubits)
    rows, posteriors = [], {arm: {} for arm in "ABC"}
    for i, (q, role, czs) in enumerate(
        zip(selection.qubits, selection.roles, selection.czs, strict=True)
    ):
        row = {"qubit": q, "role": role, "czs": czs, "partners": list(partners[q]), "arms": {}}
        for job in jobs:
            clouds = lc.Clouds.fit(job["R0"][:, i], job["R1"][:, i])
            arm = next(arm for arm in "ABC" if arm in job)
            points = np.asarray(job[arm]).reshape(len(job[arm]), -1, n)[:, :, i]
            read = lc.classify(points, clouds)
            observed = lc.changes(read) if arm == "C" else read
            fit = lc.fit_hmm(observed)
            gamma = posterior(observed, fit)
            posteriors[arm][q] = gamma
            selected = gamma > 0.9
            edges = np.linspace(-1, 4, 41)
            along, across = project(points, clouds)
            histogram = {
                "edges": edges.tolist(),
                "state": np.histogram(along[selected], edges)[0].tolist(),
            }
            for reference in ("R0", "R1"):
                histogram[reference] = np.histogram(
                    project(job[reference][:, i], clouds)[0], edges
                )[0].tolist()
            row["arms"][arm] = {
                "hmm": fit,
                "shots": len(points),
                "cycles": points.shape[1],
                "lifetime_us": float(lc.lifetime_us(fit["b"], period)),
                "steady_occupancy": fit["a"] / (fit["a"] + fit["b"]),
                "posterior_by_cycle": gamma.mean(0).tolist(),
                "outside_fraction": float((read == 2).mean()),
                "third_cluster": lc.third_cluster(points, clouds),
                "confident": {
                    "readouts": int(selected.sum()),
                    "symbols": np.bincount(read[selected], minlength=3).tolist(),
                    "histogram": histogram,
                    "along_mean": float(along[selected].mean()),
                    "across_mean": float(across[selected].mean()),
                    "in_histogram": int(sum(histogram["state"])),
                },
            }
        row["steady"] = row["arms"]["A"]["hmm"]["e_P"][1] > 1 - lc.MIN_CONTRAST
        rows.append(row)
        print("q", q, "done", flush=True)
    pooled = {}
    pairs = {arm: pair_correlations(posteriors[arm], partners) for arm in "ABC"}
    for arm in "ABC":
        arms = [row["arms"][arm] for row in rows]
        fits = [a["hmm"] for a in arms]
        pooled[arm] = {
            "onset_median": float(np.median([f["a"] * f["present"] for f in fits])),
            "lifetime_us_median": float(np.median([a["lifetime_us"] for a in arms])),
            "state_reads_1_median": float(np.median([f["e_P"][1] for f in fits])),
            "pi_over_steady_median": float(
                np.median([f["pi"] * (f["a"] + f["b"]) / f["a"] for f in fits])
            ),
            "outside_range": [
                min(a["outside_fraction"] for a in arms),
                max(a["outside_fraction"] for a in arms),
            ],
            "third_cluster_accepted": [
                row["qubit"] for row in rows if row["arms"][arm]["third_cluster"]["accepted"]
            ],
            "correlation_medians": {
                kind: float(
                    np.median(
                        [
                            pair["posterior_correlation"]
                            for pair in pairs[arm]
                            if bool(pair["shared_partners"]) == shared
                            and pair["posterior_correlation"] is not None
                        ]
                    )
                )
                for kind, shared in (("shared", True), ("not_shared", False))
            },
        }

    def onset(row, arm):
        fit = row["arms"][arm]["hmm"]
        return fit["a"] * fit["present"]

    pooled["paired_cz_each_median"] = float(
        np.median([(onset(row, "B") - onset(row, "A")) / row["czs"] for row in rows])
    )
    nonsteady = [row for row in rows if not row["steady"]]
    pooled["reset_qubits"] = [row["qubit"] for row in nonsteady]
    pooled["paired_reset_median"] = float(
        np.median([onset(row, "A") - onset(row, "C") for row in nonsteady])
    )
    inputs = {}
    for folder in folders:
        for name in ("run.json", "results.npz", "calibration.json"):
            path = folder / name
            with path.open("rb") as stream:
                inputs[str(path.relative_to(REPO))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return {
        "method": __doc__,
        "input_sha256": inputs,
        "period_us": period,
        "posterior_cutoff": 0.9,
        "qubits": rows,
        "pairs": pairs,
        "pooled": pooled,
    }


if __name__ == "__main__":
    print(write_json("kingston_iq.json", reconstruct()))

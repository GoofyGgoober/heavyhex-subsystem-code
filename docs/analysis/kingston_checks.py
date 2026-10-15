"""Fixed-decoder Kingston comparisons and depth-level checks from saved local outputs.

First run the four followup.reproduce commands in followup/README.md with
--simulation-priors hardware. This script only reads their build/ outputs and the
immutable archive; it never submits or samples a circuit.

    PYTHONPATH=. python docs/analysis/kingston_checks.py
"""

from __future__ import annotations

import numpy as np

from followup import analysis

from common import REPO, read, write_json


def summarize_curve(curve: dict, shots: int) -> dict:
    n = np.arange(1, 5)
    k = np.array(curve["failures"])
    fitted = analysis.binomial(n, k, shots)
    p_fit = (1 - fitted.amplitude * (1 - 2 * fitted.eps) ** n) / 2
    return {
        "failures": k.tolist(),
        "shots": shots,
        "P": (k / shots).tolist(),
        "eps": fitted.eps,
        "A": fitted.amplitude,
        "dispersion": fitted.dispersion,
        "fit_P": p_fit.tolist(),
        "residual_P": (k / shots - p_fit).tolist(),
        "max_abs_residual_P": float(np.max(np.abs(k / shots - p_fit))),
    }


def main() -> None:
    archive = read(REPO / "runs" / "ibm_kingston-followup" / "analysis" / "exp1.json")
    variants = {}
    for name in ("ibm", "joint", "own_test", "measured_tq_readout"):
        source = read(REPO / "build" / "followup" / f"fixed-{name}.json")
        if source["simulation_priors"] != "hardware" or source["hardware_priors"] != "measured":
            raise ValueError("The comparison must use the measured-Tq decoder on both sides")
        variants[name] = {
            "input_sha256": source["input_sha256"],
            "hardware_priors": source["hardware_priors"],
            "simulation_priors": source["simulation_priors"],
            "shots": source["simulated_shots_per_circuit"],
            "seed": source["seed"],
            "memory": source["memory"],
        }
    variants["measured_tq"] = {
        "shots": archive["predicted_measured_tq"]["shots"],
        "memory": {
            "observed": archive["observed"]["0.1"],
            "predicted": archive["predicted_measured_tq"]["fits"]["0.1"],
            "observed_over_model": {
                b: {"binomial": archive["summary"]["o_over_m_measured_tq_cap0.1"][b]} for b in "XZ"
            },
        },
    }
    curves = {}
    for basis in "XZ":
        for d in "35":
            key = basis + d
            curves[key] = {
                "observed": summarize_curve(archive["observed"]["0.1"][basis]["eps"][d], 8192)
            }
            for name, row in variants.items():
                curves[key][name] = summarize_curve(
                    row["memory"]["predicted"][basis]["eps"][d], row["shots"]
                )
            obs = np.array(curves[key]["observed"]["P"])
            for name in variants:
                model = np.array(curves[key][name]["P"])
                curves[key][name]["observed_minus_model_P"] = (obs - model).tolist()
                curves[key][name]["observed_over_model_P"] = (obs / model).tolist()
    forecast = read(REPO / "runs/ibm_kingston-2026-10-14-exp1-1/prediction.json")
    stored = forecast["as_sent"]["decoders"]["0.1"]["logical_error"]
    stored_curves, job_curves = {}, {}
    for basis in "XZ":
        for d in "35":
            key = basis + d
            observed = summarize_curve(
                archive["observed_prerun_model_decoder"]["0.1"][basis]["eps"][d], 8192
            )
            prediction = np.array([stored[f"{basis}-{d}-{n}"] for n in range(1, 5)])
            stored_curves[key] = {
                "observed": observed,
                "model_P": prediction.tolist(),
                "observed_minus_model_P": (observed["P"] - prediction).tolist(),
            }
            for job, decoders in archive["observed_by_job"].items():
                observed = summarize_curve(decoders["0.1"][basis]["eps"][d], 4096)
                prediction = np.array(curves[key]["measured_tq"]["P"])
                job_curves.setdefault(job, {})[key] = {
                    "observed": observed,
                    "model_P": prediction.tolist(),
                    "observed_minus_model_P": (observed["P"] - prediction).tolist(),
                }
    out = {
        "what": "All device and simulated samples use the same capped measured-Tq decoder. "
        "New simulations use 200000 shots per depth and seed0; the measured-Tq baseline "
        "and device counts come from the archive. Fixed inputs, not held-out validation. "
        "Residuals quantify the exponential compression, not physical stationary rates.",
        "variants": variants,
        "curves": curves,
        "stored_curves": {
            "priors": "Original calibration, cap 0.1, on both sides",
            "curves": stored_curves,
        },
        "by_job_curves": {
            "priors": "Pooled measured-Tq, cap 0.1, on both sides; model inputs pooled across jobs",
            "jobs": job_curves,
        },
    }
    write_json("kingston_checks.json", out)
    for name, row in variants.items():
        print(name, row["memory"]["observed_over_model"])
    for key, row in curves.items():
        print(
            key,
            "measured-Tq fixed-depth differences:",
            row["measured_tq"]["observed_minus_model_P"],
        )


if __name__ == "__main__":
    main()

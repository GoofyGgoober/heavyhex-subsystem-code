"""Local reconstruction of the paper's follow-up input variants. Never submits jobs.

Each invocation writes a new result under build/, not into the archived runs/.
Use --variant inputs for the fitted decay inputs without Monte Carlo simulation.
The other variants use one process, cap 0.1 and a declared simulation seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from heavyhex.patches.placement import Calibration

from . import analysis as a
from . import circuits, decoder, model

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
MEMORY = [RUNS / f"ibm_kingston-2026-10-14-exp1-{i}" for i in (1, 2)]
Z_IDLE = RUNS / "ibm_kingston-2026-10-14-life-z"
VARIANTS = (
    "inputs",
    "ibm",
    "measured_tq",
    "joint",
    "own_test",
    "measured_tq_readout",
    "own_test_readout",
)


def clean(value):
    """Strict JSON, with nonfinite fitted times represented by null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to((REPO / "build").resolve()):
        raise ValueError("Reproduction outputs must stay inside this repository's build/ directory")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    return path


def matched_rates(
    target: float, along_z: float, equator: float, rate_q: float
) -> tuple[float, float]:
    """Match a basis-specific decay tz/T1 + te/Tq, holding Tq unless its 2*T1 cap binds."""
    if target <= 0 or along_z <= 0 or equator < 0 or rate_q <= 0:
        raise ValueError("Positive decay rates and longitudinal exposure are required")
    r1 = (target - equator * rate_q) / along_z
    if r1 < 0:
        raise ValueError("The target decay cannot be matched at the held transverse rate")
    if r1 > 2 * rate_q:
        r1 = target / (along_z + equator / 2)
        rate_q = r1 / 2
    return r1, rate_q


def unconstrained(xy: dict, z: dict) -> dict:
    """The same weighted design as analysis.together, without nonnegative-rate bounds."""
    r, vr = a.r_of(*(np.asarray(xy[k], float) for k in ("x", "y", "shots_x", "shots_y")))
    c, vc = a.c_of(*(np.asarray(z[k], float) for k in ("p0", "p1", "shots_0", "shots_1")))
    design, response, weights = [], [], []
    for which, (value, variance, row) in enumerate(((r, vr, xy), (c, vc, z))):
        keep = value > a.MIN_R
        te, tz, g = (np.asarray(row[k])[keep] for k in ("equator_us", "along_z_us", "pulses"))
        design.extend([1 - which, which, -zz - ee / 2, -ee] for zz, ee in zip(tz, te, strict=True))
        response.extend(np.log(value[keep] / g))
        weights.extend(1 / variance[keep])
    design, response, weights = np.array(design), np.array(response), np.array(weights)
    root = np.sqrt(weights)
    beta = np.linalg.lstsq(design * root[:, None], response * root, rcond=None)[0]
    covariance = np.linalg.inv((design.T * weights) @ design)
    return {
        "rate_t1": float(beta[2]),
        "rate_t_phi": float(beta[3]),
        "sigma_t_phi": float(np.sqrt(covariance[3, 3])),
    }


def inputs() -> tuple[a.Record, dict]:
    record = a.record(MEMORY)
    z_record = a.record([Z_IDLE])
    xy = a.idle(record.found, record.shots, record.calibration, record.dt)
    z = a.z_idle(z_record.found, z_record.shots, z_record.calibration, z_record.dt)
    by_xy = {row["qubit"]: row for row in xy["qubits"]}
    by_z = {row["qubit"]: row for row in z["qubits"]}
    joint = a.in_situ(record, z_record)
    free, own_t1, own_q = {}, {}, {}
    for q, row in by_z.items():
        p0, p1, n0, n1 = (np.asarray(row[k], float) for k in ("p0", "p1", "shots_0", "shots_1"))
        target = a.log_decay(
            np.asarray(z["rounds"]), *a.c_of(p0, p1, n0, n1), np.asarray(row["pulses"])
        )["rate"][0]
        tz, te = a._per_round(row, z["rounds"])
        # Hold the baseline model's effective Tq, not an uncapped fitted time.
        held_tq = min(xy["decay_us"][q], 2 * record.calibration.t1_us[q])
        r1, rq = matched_rates(target, tz, te, 1 / held_tq)
        own_t1[q], own_q[q] = 1 / r1 if r1 > 0 else float("inf"), 1 / rq
        free[q] = unconstrained(by_xy[q], row)
    bounds = sum(
        row["rate_t1"][0] == 0 or abs(row["rate_t_q"][0] - row["rate_t1"][0] / 2) < 1e-12
        for row in joint["qubits"]
    )
    return record, {
        "xy": xy,
        "z": z,
        "joint": joint,
        "unconstrained": free,
        "own_z_t1_us": own_t1,
        "own_z_tq_us": own_q,
        "summary": {
            "joint_on_boundary": bounds,
            "negative_pure_dephasing": sum(v["rate_t_phi"] < 0 for v in free.values()),
            "negative_beyond_2sigma": sum(
                v["rate_t_phi"] < -2 * v["sigma_t_phi"] for v in free.values()
            ),
        },
    }


def evaluate(
    variant: str,
    record: a.Record,
    fitted: dict,
    shots: int,
    seed: int,
    *,
    cap: float | None = decoder.PRIMARY,
    hardware_priors: str = "measured",
    simulation_priors: str = "variant",
) -> dict:
    calibration, dt = record.calibration, record.dt
    decay = fitted["xy"]["decay_us"]
    readout = Calibration.load(Z_IDLE / "calibration.json").readout
    observed_counts, simulated = {}, {}
    detection = {}
    for basis in "XZ":
        used, tq = calibration, decay
        if variant == "ibm":
            tq = None
        elif variant == "joint":
            used = model.with_t1(used, fitted["joint"]["t1_us"])
            tq = fitted["joint"]["decay_us"]
        elif variant.startswith("own_test") and basis == "Z":
            used = model.with_t1(used, fitted["own_z_t1_us"])
            tq = fitted["own_z_tq_us"]
        if variant.endswith("_readout"):
            used = replace(used, readout=readout)
        sent = {}
        for label, (setting, circuit) in record.found.items():
            if setting is None or not setting.decoded or setting.basis != basis:
                continue
            key = basis, setting.distance, setting.rounds
            # Hardware priors stay fixed across simulated input variants.
            observed = a.memory(
                setting,
                circuit,
                record.shots[label],
                calibration,
                dt,
                decay_us=decay if hardware_priors == "measured" else None,
                caps=[cap],
            )
            observed_counts[key] = observed["failures"][str(cap)]
            sent[key] = circuit, circuits.logical_circuit(setting)[1]
        for i, key in enumerate(sorted(sent)):
            noisy, labels = model.memory(*sent[key], used, dt, decay_us=tq)
            prior_model = noisy
            if simulation_priors == "hardware":
                prior_model, prior_labels = model.memory(
                    *sent[key],
                    calibration,
                    dt,
                    decay_us=decay if hardware_priors == "measured" else None,
                )
                if prior_labels != labels or prior_model.num_detectors != noisy.num_detectors:
                    raise ValueError("Fixed decoder and sample detector layouts differ")
            dem = prior_model.detector_error_model(decompose_errors=True)
            probabilities, rates = model.sample(
                noisy,
                decoder.decoders(dem, [cap]),
                shots,
                seed + i + (0 if basis == "X" else 8),
            )
            simulated[key] = probabilities[str(cap)]
            detection[str(circuits.Setting("memory", *key))] = model.detection_rates(
                labels, rates, key[2]
            )
    expected = a.fits({key: round(p * shots) for key, p in simulated.items()}, shots)
    # Idle and memory circuits have different shot counts.
    memory_label = next(
        label for label, (s, _) in record.found.items() if s is not None and s.decoded
    )
    n = len(next(iter(record.shots[memory_label].values())))
    observed = a.fits(observed_counts, n)
    return {
        "observed": observed,
        "predicted": expected,
        "observed_over_model": a.against(observed, expected),
        "detection": detection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder", choices=("capped", "uniform"), default="capped")
    parser.add_argument(
        "--hardware-priors", choices=("measured", "calibration"), default="measured"
    )
    parser.add_argument(
        "--simulation-priors",
        choices=("variant", "hardware"),
        default="variant",
        help="Use each simulated noise variant's priors, or hold the hardware decoder fixed",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.shots < 100:
        parser.error("--shots must be at least 100")
    path = output_path(args.output or REPO / "build" / "followup" / f"{args.variant}.json")
    record, fitted = inputs()
    cap = decoder.PRIMARY if args.decoder == "capped" else None
    source_files = [
        Path(__file__),
        *[
            folder / name
            for folder in [*MEMORY, Z_IDLE]
            for name in ("run.json", "calibration.json", "program.json.gz", "results.npz")
        ],
    ]
    result = {
        "variant": args.variant,
        "simulated_shots_per_circuit": args.shots,
        "seed": args.seed,
        "decoder_cap": cap,
        "hardware_priors": args.hardware_priors,
        "simulation_priors": args.simulation_priors,
        "input_sha256": {str(p.relative_to(REPO)): fingerprint(p) for p in source_files},
        "inputs": fitted,
    }
    if args.variant != "inputs":
        result["memory"] = evaluate(
            args.variant,
            record,
            fitted,
            args.shots,
            args.seed,
            cap=cap,
            hardware_priors=args.hardware_priors,
            simulation_priors=args.simulation_priors,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    print(path)
    print(json.dumps(result["inputs"]["summary"]))


if __name__ == "__main__":
    main()

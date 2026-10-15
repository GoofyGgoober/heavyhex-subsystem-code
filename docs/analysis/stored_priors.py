"""Fez's stored f=1 forecast evaluated with its own decoder on the saved device shots.

No simulation, IBM connection or raw-data writes. A job's f=1 model supplies both the
measurement-to-detector converter and the matching graph used by its stored prediction.
The first job is also decoded at the original measured f as a deterministic control.
Per-job counts are cached under build/; their input fingerprints are checked on reuse.
The complete result contains binomial fits, the five-day intervals used in the paper,
early-round comparisons and fit-window sensitivities. The old adaptive analysis is kept.

    python docs/analysis/stored_priors.py                 # all jobs, one worker
    python docs/analysis/stored_priors.py --jobs <name>    # counts for selected jobs only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from f_treatments import F1, QUANTITIES, observed, predicted, summary

from heavyhex.experiment.circuits import ROUNDS

from common import REPO, RESULTS, day_number, job_folders, read, write_json
from fits import fit_curve, redecode
from fitting import binomial_point

CACHE = REPO / "build" / "fez-stored-priors"
WINDOWS = {
    "1-8": (1, 2, 3, 4, 6, 8),
    "2-8": (2, 3, 4, 6, 8),
    "1-6": (1, 2, 3, 4, 6),
    "2-6": (2, 3, 4, 6),
    "1-4": (1, 2, 3, 4),
    "3-8": (3, 4, 6, 8),
}


def fingerprint(folder: Path) -> dict[str, str]:
    files = [
        folder / name
        for name in (
            "run.json",
            "calibration.json",
            "circuits.qpy.gz",
            "shots.npz",
            "analysis.json",
        )
    ]
    files.append(Path(__file__))
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def cached_counts(folder: Path, check_adaptive: bool = False) -> dict:
    """Only local translation and decoding; never calls submit, fetch or predict."""
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{folder.name}.json"
    inputs = fingerprint(folder)
    if target.exists():
        cached = read(target)
        if cached.get("inputs") == inputs and (
            not check_adaptive or cached.get("adaptive_control_matches") is True
        ):
            return cached
    analysis = read(folder / "analysis.json")
    counts = redecode(folder, 1.0)
    control = None
    if check_adaptive:
        decoded = redecode(folder, analysis["decoupled_dephasing"])
        from fits import memory_rows

        expected = {
            basis: {
                patch: [
                    round(row["logical_error"][k] * 2500)
                    for row in memory_rows(analysis["observed"]["settings"], basis)
                ]
                for k, patch in enumerate(analysis["patches"])
            }
            for basis in "XZ"
        }
        control = decoded == expected
        if not control:
            raise ValueError(f"Adaptive decoding does not reproduce the stored counts: {folder}")
    out = {
        "inputs": inputs,
        "shots": 2500,
        "decoder_f": 1.0,
        "counts": counts,
        "adaptive_control_matches": control,
    }
    # This generated cache is separate from the archived records in runs/.
    target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return out


def aggregate(fits: dict, counts: dict) -> dict:
    """Reuse the established day-block convention with a matched f=1 decoder."""
    reps, predictions = fits["per_rep"], fits["predictions"]
    keys = sorted(reps, key=lambda k: (reps[k]["day"], reps[k]["rep"]))
    days = np.array([day_number(reps[k]["day"]) for k in keys])
    placements = np.array([reps[k]["placement"] for k in keys])
    curves = {}
    for name, job in counts.items():
        curves[name] = {
            b: {d: fit_curve(ROUNDS, k, job["shots"]) for d, k in patches.items()}
            for b, patches in job["counts"].items()
        }

    def value(rep: dict, quantity: str, window: tuple[int, ...] | None = None) -> float:
        basis, name = quantity.split()

        def eps(d):
            job = rep["jobs"][d]
            curve = curves[job][basis][d]
            if window is None:
                return curve["binomial"]["eps"]
            ix = [list(ROUNDS).index(n) for n in window]
            return binomial_point(window, np.array(curve["k"])[ix], curve["N"])

        return eps("3") / eps("5") if name == "lambda" else eps(name[-1])

    primary = {}
    for quantity in QUANTITIES:
        o = np.array([value(reps[k], quantity) for k in keys])
        m = np.array([predicted(predictions, reps[k], F1, quantity) for k in keys])
        primary[quantity] = summary(np.log(o / m), int((o > m).sum()), days, placements)

    early_above, fitted_above, d5_worse = 0, 0, 0
    early_total = 0
    shifts = []
    stored_counts = read(RESULTS / "fits_counts.json")["jobs"]
    for key in keys:
        rep = reps[key]
        for basis in "XZ":
            for d in "35":
                curve = curves[rep["jobs"][d]][basis][d]
                model = predictions[rep["jobs"][d]][F1][basis][d]
                # Stored probabilities are fitted nowhere for this comparison.
                old_counts = stored_counts[rep["jobs"][d]]
                p_model = old_counts["predicted"][F1]["P"][basis][d]
                early_above += sum(o > m for o, m in zip(curve["P"][:4], p_model[:4]))
                early_total += 4
                fitted_above += int(curve["binomial"]["eps"] > model["binomial"]["eps"])
                shifts.append(curve["binomial"]["eps"] - observed(rep, f"{basis} eps{d}")[0])
        c3 = curves[rep["jobs"]["3"]]["X"]["3"]["P"]
        c5 = curves[rep["jobs"]["5"]]["X"]["5"]["P"]
        d5_worse += sum(p5 > p3 for p3, p5 in zip(c3, c5))

    windows = {}
    for label, window in WINDOWS.items():
        ix = [list(ROUNDS).index(n) for n in window]
        windows[label] = {}
        for quantity in (q for q in QUANTITIES if "eps" in q):
            basis, name = quantity.split()
            d = name[-1]
            o = np.array([value(reps[k], quantity, window) for k in keys])
            m = []
            for key in keys:
                model = stored_counts[reps[key]["jobs"][d]]["predicted"][F1]
                p = np.array(model["P"][basis][d])[ix]
                m.append(binomial_point(window, np.round(p * model["shots"]), model["shots"]))
            m = np.array(m)
            windows[label][quantity] = summary(np.log(o / m), int((o > m).sum()), days, placements)

    return {
        "what": "Stored f=1 noise and f=1 matching priors for both device and model. "
        "Device shots redecoded locally; stored simulated counts unchanged. Binomial fits; "
        "five daily log means and Bonferroni family-of-five intervals. Fixed inputs and patches.",
        "inputs": {name: job["inputs"] for name, job in counts.items()},
        "rounds": list(ROUNDS),
        "adaptive_control_matches": counts[next(iter(counts))]["adaptive_control_matches"],
        "curves": curves,
        "primary": primary,
        "windows": windows,
        "early_points_above": early_above,
        "early_points_total": early_total,
        "fitted_curves_above": fitted_above,
        "d5_worse_X_points": d5_worse,
        "eps_shift_from_adaptive": {
            "min": min(shifts),
            "max": max(shifts),
            "mean": float(np.mean(shifts)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", nargs="+", help="Cache only the named job folders")
    args = parser.parse_args()
    folders = job_folders()
    selected = set(args.jobs or [p.name for p in folders])
    if selected - {p.name for p in folders}:
        parser.error("Unknown Fez job")
    counts = {}
    for folder in folders:
        if folder.name not in selected:
            continue
        started = time.monotonic()
        counts[folder.name] = cached_counts(folder, check_adaptive=folder == folders[0])
        print(f"{folder.name}: {time.monotonic() - started:.1f} s", flush=True)
    if args.jobs:
        return
    out = aggregate(read(RESULTS / "fits.json"), counts)
    write_json("stored_priors.json", out)
    for name, row in out["primary"].items():
        print(name, row["ratio"], row["ci95_family_of_five"])
    print(
        "Counts:",
        {
            k: out[k]
            for k in (
                "early_points_above",
                "early_points_total",
                "fitted_curves_above",
                "d5_worse_X_points",
            )
        },
    )


if __name__ == "__main__":
    main()

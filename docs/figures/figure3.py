"""Figure 3: simulated Λ on Fez as placed, and with parts of the chip made better.

Every row runs what the run runs: the patches where fez_qubits puts them on the
given calibration, both in one circuit, translated by for_fez with the
calibration's gate times, simulated by run.simulated and decoded by MWPM. A
scenario changes only the calibration; the qubits and circuits stay as placed.
Λ in X memory at each end of decoupled_dephasing, and in Z memory averaged over
the two. The d=3 patch is also moved onto the d=5 patch's qubits, beside the usual
d=3, at every d=3 spot inside the d=5 footprint. The average over placements simulates
every clean d=3 and d=5 spot, each patch alone, in X memory. Runs offline.

    python docs/figures/figure3.py --calibration PATH [--out PATH.json] [--shots N]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from typing import Any

from heavyhex.experiment import run
from heavyhex.experiment.circuits import (
    DISTANCES,
    ROUNDS,
    Setting,
    for_fez,
    logical_circuit,
    with_durations,
)
from heavyhex.patches import placement as pl
from heavyhex.patches.layout import build_layout
from heavyhex.simulation.noisy import fit_per_round, typical

HERE = Path(__file__).resolve().parent
OUT_NAME = "figure3.json"
ALONE = "each placement simulated alone"
MOVED = "3in"  # the d=3 patch moved onto the d=5 patch's qubits


@dataclass(frozen=True)
class Alone(Setting):
    """Memory of one patch by itself, for the average over placements."""

    patch: str = "3"

    @property
    def patches(self) -> tuple[str, ...]:
        return (self.patch,)


@dataclass(frozen=True)
class Beside(Setting):
    """Memory of the usual d=3 patch and a second one moved onto the d=5 patch's qubits."""

    @property
    def patches(self) -> tuple[str, ...]:
        return ("3", MOVED)


def weak_at_median(calibration: pl.Calibration, qubits: set[int]) -> pl.Calibration:
    """The weak parts among these qubits, as placement.weak finds them, at the chip's median."""
    median = typical(calibration)

    def ours(part: int | tuple[int, int]) -> bool:
        return set(part if isinstance(part, tuple) else (part,)) <= qubits

    def fix(values: dict, medians: dict, weak: Callable[[float], bool]) -> dict:
        return {
            k: medians[k] if ours(k) and v is not None and weak(v) else v for k, v in values.items()
        }

    return replace(
        calibration,
        cz=fix(calibration.cz, median.cz, lambda e: pl.WEAK_CZ < e < pl.BROKEN),
        readout=fix(calibration.readout, median.readout, lambda e: pl.WEAK_READOUT < e < pl.BROKEN),
        t1_us=fix(calibration.t1_us, median.t1_us, lambda t: 0 < t < pl.WEAK_T1_US),
        t2_us=fix(calibration.t2_us, median.t2_us, lambda t: 0 < t < pl.WEAK_T2_US),
    )


def scaled(
    calibration: pl.Calibration, *, cz: float = 1.0, readout: float = 1.0, coherence: float = 1.0
) -> pl.Calibration:
    """Every CZ error, readout error, and T1 and T2 multiplied by these."""

    def times(values: dict, factor: float) -> dict:
        return {k: v * factor for k, v in values.items()}

    return replace(
        calibration,
        cz=times(calibration.cz, cz),
        readout=times(calibration.readout, readout),
        t1_us=times(calibration.t1_us, coherence),
        t2_us=times(calibration.t2_us, coherence),
    )


def scenarios(
    calibration: pl.Calibration, qubits: set[int]
) -> list[tuple[str, str, pl.Calibration]]:
    """(group, name, calibration) of each row but the placement test's."""
    median = typical(calibration)
    return [
        ("placed", "as placed", calibration),
        (
            "placed",
            "weak parts of both patches at the chip median",
            weak_at_median(calibration, qubits),
        ),
        ("median chip", "every part at the chip median", median),
        ("median chip", "CZ error halved", scaled(median, cz=0.5)),
        ("median chip", "readout error halved", scaled(median, readout=0.5)),
        ("median chip", "T1 and T2 doubled", scaled(median, coherence=2.0)),
    ]


def logical_error(
    setting: Setting,
    chips: dict[str, list[int]],
    calibration: pl.Calibration,
    target: Any,
    f: float,
    shots: int,
) -> list[float]:
    """Each patch's logical error rate: the run's circuit, sampled and decoded as run.predict does."""
    circuit = for_fez(logical_circuit(setting, chips, target.num_qubits)[0], target)
    model, _, _ = run.simulated(setting, circuit, calibration, target.dt, f)
    sampler = model.compile_detector_sampler(seed=setting.rounds)
    detectors, flips = sampler.sample(shots, separate_observables=True)
    return (run._decode(model, detectors) != flips).mean(axis=0).tolist()


def simulate(jobs: dict, target: Any) -> dict:
    """Every job's logical error rates, on all cores but two."""
    found = {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 3) - 2)) as pool:
        futures = {
            pool.submit(logical_error, setting, chips, calibration, target, f, shots): key
            for key, (setting, chips, calibration, f, shots) in jobs.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            found[futures[future]] = future.result()
            if done % 100 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} circuits simulated", flush=True)
    return found


def eps(errors: list[float], shots: int) -> list[float]:
    """[ε per round, its standard error] from the logical error after each of ROUNDS."""
    decay = fit_per_round(ROUNDS, errors, shots)
    return [decay.per_round, decay.uncertainty]


def ratio(a: list[float], b: list[float]) -> list[float]:
    """a / b and its standard error, for [value, standard error] pairs."""
    r = a[0] / b[0]
    return [r, r * sqrt((a[1] / a[0]) ** 2 + (b[1] / b[0]) ** 2)]


def figure3(calibration: pl.Calibration, shots: int) -> dict:
    """Every row, the d=3 patch at each spot inside the d=5 patch, and the average over placements."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    side = min(run.SIDE_SHOTS, shots)
    target = FakeFez().target
    if calibration.durations_ns:
        target = with_durations(target, calibration.durations_ns)
    device = json.loads(pl.FEZ_MAP.read_text())
    coords, edges = device["coords"], device["edges"]
    best = pl.best_spots(coords, edges, calibration)
    placed = {
        d: build_layout(d, coords, edges, origin=best[d].origin, direction=best[d].direction)
        for d in DISTANCES
    }
    chips = {str(d): pl.fez_qubits(d, calibration) for d in DISTANCES}
    inside = {
        spot: layout
        for spot, layout in pl.spots(3, coords, edges).items()
        if layout.physical_qubits <= placed[5].physical_qubits
    }
    clean = {
        d: {
            spot: layout
            for spot, layout in pl.spots(d, coords, edges).items()
            if not pl.problems(layout, calibration)
        }
        for d in DISTANCES
    }
    rows = scenarios(calibration, placed[3].physical_qubits | placed[5].physical_qubits)

    jobs: dict = {}
    for _, name, changed in rows:
        for basis in "XZ":
            for f in run.ENDS:
                for n in ROUNDS:
                    setting = Setting("memory", basis, n)
                    jobs["memory", name, basis, f, n] = (setting, chips, changed, f, shots)
    for k, layout in enumerate(inside.values()):
        test = {**chips, MOVED: pl.chip_qubits(layout)}
        for basis in "XZ":
            for f in run.ENDS:
                for n in ROUNDS:
                    setting = Beside("memory", basis, n)
                    jobs["inside", k, basis, f, n] = (setting, test, calibration, f, side)
    for d, found in clean.items():
        for k, layout in enumerate(found.values()):
            alone = {str(d): pl.chip_qubits(layout)}
            for f in run.ENDS:
                for n in ROUNDS:
                    setting = Alone("memory", "X", n, patch=str(d))
                    jobs["alone", d, k, f, n] = (setting, alone, calibration, f, side)
    p = simulate(jobs, target)

    table = []
    for group, name, _ in rows:
        row: dict = {"group": group, "scenario": name}
        for basis in "XZ":
            row[basis] = {}
            for f in run.ENDS:
                e3, e5 = (
                    eps([p["memory", name, basis, f, n][k] for n in ROUNDS], shots)
                    for k in range(2)
                )
                row[basis][str(f)] = {"eps3": e3, "eps5": e5, "lambda": ratio(e3, e5)}
        table.append(row)

    spots = []
    for k, (spot, layout) in enumerate(inside.items()):
        chip = pl.chip_qubits(layout)
        entry: dict = {"origin": spot.origin, "direction": spot.direction, "chip": chip}
        for basis in "XZ":
            entry[basis] = {}
            for f in run.ENDS:
                own, moved = (
                    eps([p["inside", k, basis, f, n][j] for n in ROUNDS], side) for j in range(2)
                )
                entry[basis][str(f)] = {"eps3": own, "eps3in": moved, "ratio": ratio(moved, own)}
        spots.append(entry)
    test_row: dict = {"group": "placed", "scenario": "d = 3 on qubits of the d = 5 patch"}
    for basis in "XZ":
        test_row[basis] = {}
        for f in run.ENDS:
            moved = [s[basis][str(f)]["eps3in"] for s in spots]
            mean = [
                statistics.mean(e for e, _ in moved),
                sqrt(sum(u**2 for _, u in moved)) / len(moved),
            ]
            e5 = table[0][basis][str(f)]["eps5"]
            test_row[basis][str(f)] = {
                "eps3in_mean": mean,
                "eps3in_range": [min(e for e, _ in moved), max(e for e, _ in moved)],
                "eps5": e5,
                "lambda": ratio(mean, e5),
            }
    table.insert(1, test_row)
    for row in table:
        row["lambda_x_f0"], row["lambda_x_f1"] = (row["X"][str(f)]["lambda"][0] for f in run.ENDS)
        row["lambda_z"] = statistics.mean(row["Z"][str(f)]["lambda"][0] for f in run.ENDS)

    placements = [
        {
            "d": d,
            "origin": spot.origin,
            "direction": spot.direction,
            "as placed": spot == best[d],
            "X": {str(f): eps([p["alone", d, k, f, n][0] for n in ROUNDS], side) for f in run.ENDS},
        }
        for d, found in clean.items()
        for k, spot in enumerate(found)
    ]
    average: dict = {
        "setup": f"{ALONE}: X memory of one patch on its chip qubits, every d=3 and d=5 spot "
        f"without broken parts, {side:,} shots per circuit; Λ from the mean ε₃ over the d=3 "
        "placements over the mean ε₅ over the d=5 placements",
        "d3_placements": len(clean[3]),
        "d5_placements": len(clean[5]),
        "placements": placements,
    }
    for f in run.ENDS:
        e = {d: [x["X"][str(f)][0] for x in placements if x["d"] == d] for d in DISTANCES}
        here = {x["d"]: x["X"][str(f)][0] for x in placements if x["as placed"]}
        m3, m5, med3 = statistics.mean(e[3]), statistics.mean(e[5]), statistics.median(e[3])
        average[str(f)] = {
            "mean_eps3": m3,
            "median_eps3": med3,
            "eps3_range": [min(e[3]), max(e[3])],
            "mean_eps5": m5,
            "eps5": e[5],
            "lambda_mean": m3 / m5,
            "lambda_median_d3": med3 / m5,
            "lambda_as_placed_alone": here[3] / here[5],
        }

    return {
        "calibration": {
            "backend": calibration.backend,
            "calibrated": calibration.calibrated,
            "pulled": calibration.pulled,
            "durations_ns": calibration.durations_ns,
        },
        "gate_times": "the calibration's" if calibration.durations_ns else "FakeFez's",
        "code": run.commit(),
        "setup": "every row: both patches in one circuit, as the run sends them; the run's "
        "circuits on the qubits fez_qubits picks from this calibration, translated by "
        "for_fez, simulated by run.simulated with the row's calibration, decoded by MWPM, "
        "sampler seed = rounds; ε per round fitted over ROUNDS",
        "shots": {"rows": shots, "inside spots and average": side},
        "rounds": list(ROUNDS),
        "decoupled_dephasing": list(run.ENDS),
        "chips": chips,
        "spots": {str(d): {"origin": s.origin, "direction": s.direction} for d, s in best.items()},
        "weak_parts": {str(d): pl.weak(placed[d], calibration) for d in DISTANCES},
        "rows": table,
        "inside_spots": spots,
        "average": average,
    }


def report(figure: dict) -> str:
    """A short table of the numbers."""
    cal = figure["calibration"]
    lines = [
        f"{cal['backend']} calibrated {cal['calibrated']}, timed with {figure['gate_times']} "
        f"gate times; {figure['shots']['rows']:,} shots per circuit",
        f"{'':58s}{'Λ_X f=0':>9s}{'Λ_X f=1':>9s}{'Λ_Z':>7s}",
    ]
    for row in figure["rows"]:
        lines.append(
            f"{row['group']:12s}{row['scenario']:46s}"
            f"{row['lambda_x_f0']:9.3f}{row['lambda_x_f1']:9.3f}{row['lambda_z']:7.3f}"
        )
    lines.append(
        f"d=3 on d=5 qubits, {figure['shots']['inside spots and average']:,} shots: "
        "X memory ε at f=0, 1, and its ratio to the usual d=3 patch's"
    )
    for s in figure["inside_spots"]:
        low, high = s["X"]["0.0"], s["X"]["1.0"]
        lines.append(
            f"  {str(tuple(s['origin'])):9s}{str(tuple(s['direction'])):9s}"
            f"{low['eps3in'][0]:7.2%}{high['eps3in'][0]:7.2%}"
            f"{low['ratio'][0]:7.2f}{high['ratio'][0]:6.2f}"
        )
    average = figure["average"]
    lines.append(
        f"average over placements, {ALONE} "
        f"({average['d3_placements']} d=3, {average['d5_placements']} d=5):"
    )
    for f in figure["decoupled_dephasing"]:
        a = average[str(f)]
        lines.append(
            f"  f={f:g}: mean ε₃ {a['mean_eps3']:.2%}, median ε₃ {a['median_eps3']:.2%}, "
            f"mean ε₅ {a['mean_eps5']:.2%}; Λ {a['lambda_mean']:.3f} from the mean, "
            f"{a['lambda_median_d3']:.3f} from the median d=3 placement"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate the numbers of Figure 3.")
    parser.add_argument("--calibration", type=Path, required=True, metavar="PATH")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / OUT_NAME,
        metavar="PATH",
        help=f"where to write the JSON (default: {OUT_NAME} next to this script)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=run.SIMULATED_SHOTS,
        metavar="N",
        help=f"shots per circuit for the rows (default {run.SIMULATED_SHOTS:,}); the inside "
        f"spots and the average take {run.SIDE_SHOTS:,}, or N if fewer",
    )
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".* uses weak parts")
        figure = figure3(pl.Calibration.load(args.calibration), args.shots)
    figure["calibration"]["path"] = str(args.calibration)
    args.out.write_text(json.dumps(figure, indent=1, ensure_ascii=False) + "\n")
    print(report(figure))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

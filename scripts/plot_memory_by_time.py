"""Plot logical-Z memory results against storage time, one curve per fixed round count.

Two figures come from one data set:
- logical-z-fidelity-vs-time: decoded logical-Z fidelity (1 - failure) for every n >= 1.
- logical-z-failure-vs-time: failure probability, adding the n = 0 readout-only control.

Chart contract: compare fixed-n curves over 13 durations from 10 to 200 µs
against an analytic unencoded-qubit reference. Static Matplotlib PNG/SVG;
linear time and probability axes; categorical colors in fixed slot order plus
a gray n = 0 control, distinct markers, a legend, and direct end labels.

--profile selects one of the library's named noise profiles (see
surface_code.simulation.profiles); --prep selects the logical-state preparation.
Results are resumable: rerunning with a larger grid only simulates new points.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from surface_code.simulation import PREPARATIONS, PROFILES, get_profile, run_cadence_experiment


ROOT = Path(__file__).resolve().parents[1]
TIMES_US = (10, 16, 20, 30, 40, 50, 65, 80, 100, 125, 150, 175, 200)
ROUNDS = (0, 1, 2, 4, 8, 16, 32, 64)
FIDELITY_ROUNDS = (1, 2, 4, 8, 16, 32, 64)


PROFILE_LABELS = {
    "baseline": "project baseline noise",
    "ibm-heron": "IBM Heron-like parameters",
    "google-willow": "Google Willow-like parameters",
}
PREPARATION_LABELS = {
    "encoder": "synthesized encoder prep",
    "product": "data qubits start in |0⟩",
}

# Chart chrome (light surface).
SURFACE = "#FCFCFB"
INK = "#0B0B0B"
INK_SECONDARY = "#52514E"
INK_MUTED = "#898781"
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green,
# violet): the order is the colorblind-safety mechanism, so n keeps its slot in
# every figure. n = 0 is a neutral gray control, not a series slot.
SERIES = {
    0: ("#898781", "o"),
    1: ("#2A78D6", "o"),
    2: ("#EB6834", "s"),
    4: ("#1BAF7A", "^"),
    8: ("#EDA100", "D"),
    16: ("#E87BA4", "v"),
    32: ("#008300", "P"),
    64: ("#4A3AA7", "X"),
}
REFERENCE = "#52514E"


def run_point(job: tuple[float, int, int, int, str, str]) -> dict:
    total_time, rounds, shots, seed, profile_name, preparation = job
    profile = get_profile(profile_name)
    experiment = run_cadence_experiment(
        total_time,
        (rounds,),
        round_duration_us=profile.round_duration_us,
        basis="Z",
        shots=shots,
        seed=seed,
        circuit_noise=profile.circuit_noise,
        idle_noise=profile.idle_noise,
        preparation=preparation,
    )
    return {
        "total_time_us": total_time,
        "basis": "Z",
        "seed": seed,
        "bare_qubit_failure_rate": experiment.bare_qubit_failure_rate,
        **asdict(experiment.points[0]),
    }


def save_data(path: Path, metadata: dict, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (row["rounds"], row["total_time_us"]))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "points": ordered}, indent=2) + "\n")
    temporary.replace(path)


def validate_data(metadata: dict, rows: list[dict]) -> None:
    expected = {
        (time, rounds)
        for time in metadata["times_us"]
        for rounds in metadata["rounds"]
        if time >= rounds * metadata["round_duration_us"]
    }
    actual = {(row["total_time_us"], row["rounds"]) for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise ValueError("saved results do not cover the grid exactly once")
    for row in rows:
        rate = row["logical_failure_rate"]
        consistent = (
            row["basis"] == "Z"
            and row["shots"] == metadata["shots_per_point"]
            and 0 <= row["logical_failures"] <= row["shots"]
            and math.isclose(rate, row["logical_failures"] / row["shots"])
            and 0 <= row["confidence_low"] <= rate <= row["confidence_high"] <= 1
            and row["idle_per_round_us"] >= 0
        )
        if not consistent:
            raise ValueError(f"inconsistent saved point: {row}")


def _series_rows(rows: list[dict], rounds: int) -> list[dict]:
    return sorted(
        (row for row in rows if row["rounds"] == rounds),
        key=lambda row: row["total_time_us"],
    )


def _values(rows: list[dict], key: str, *, fidelity: bool) -> list[float]:
    return [1 - row[key] if fidelity else row[key] for row in rows]


def _draw_series(ax, rows: list[dict], rounds: int, *, fidelity: bool) -> tuple[float, float]:
    """Draw one fixed-n curve; return its right-end point for direct labeling."""
    color, marker = SERIES[rounds]
    points = _series_rows(rows, rounds)
    times = [row["total_time_us"] for row in points]
    values = _values(points, "logical_failure_rate", fidelity=fidelity)
    ax.plot(
        times, values, label=f"n = {rounds}", color=color, linewidth=2.25,
        solid_joinstyle="round", solid_capstyle="round", marker=marker,
        markersize=6, markerfacecolor=color, markeredgecolor=SURFACE,
        markeredgewidth=1.5, zorder=3,
    )
    return times[-1], values[-1]


def _draw_reference(ax, rows: list[dict], *, fidelity: bool) -> tuple[float, float]:
    """Analytic unencoded-qubit reference under the same idle and readout noise."""
    points = _series_rows(rows, 0)
    times = [row["total_time_us"] for row in points]
    values = _values(points, "bare_qubit_failure_rate", fidelity=fidelity)
    ax.plot(
        times, values, label="Unencoded qubit (analytic)", color=REFERENCE,
        linewidth=1.6, linestyle=(0, (6, 3)), dash_capstyle="round", zorder=2,
    )
    return times[-1], values[-1]


def _label_line_ends(ax, ends: list[tuple[str, str, float, float]], *, x_text: float) -> None:
    """Direct-label each curve at its right end.

    Converging ends are spread apart vertically, centered on their true mean,
    and tied back to the line with a thin leader so no label detaches.
    """
    y_low, y_high = ax.get_ylim()
    min_gap = 0.038 * (y_high - y_low)
    ordered = sorted(ends, key=lambda end: end[3])
    positions: list[float] = []
    for *_, y_end in ordered:
        positions.append(y_end if not positions else max(y_end, positions[-1] + min_gap))
    drift = sum(positions) / len(positions) - sum(end[3] for end in ordered) / len(ordered)
    for (label, color, x_end, y_end), y_text in zip(ordered, positions):
        ax.annotate(
            label, xy=(x_end, y_end), xytext=(x_text, y_text - drift),
            textcoords="data", va="center", ha="left", fontsize=10.5,
            color=INK_SECONDARY, annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": color,
                        "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4},
        )


def _new_figure(plt, title: str, subtitle: tuple[str, str]):
    fig, ax = plt.subplots(figsize=(12, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.745, bottom=0.16)
    fig.text(0.08, 0.945, title, fontsize=21, weight="bold", color=INK)
    fig.text(0.08, 0.905, subtitle[0], fontsize=11.5, color=INK_SECONDARY)
    fig.text(0.08, 0.872, subtitle[1], fontsize=11.5, color=INK_SECONDARY)
    return fig, ax


def _finish_axes(ax, *, x_max: float, y_lim: tuple[float, float], y_step: float, y_label: str):
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    ax.set_xlabel("Storage time T (µs)", labelpad=9)
    ax.set_ylabel(y_label, labelpad=9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(*y_lim)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(color=AXIS, labelcolor=INK_SECONDARY, labelsize=11)
    # Two legend rows: the reference alone on top, every n on one row below.
    handles, labels = ax.get_legend_handles_labels()
    style = {"frameon": False, "fontsize": 10.5, "handlelength": 2.6,
             "columnspacing": 1.5, "labelcolor": INK_SECONDARY, "loc": "lower left"}
    reference = ax.legend(handles[:1], labels[:1], bbox_to_anchor=(-0.005, 1.10), **style)
    ax.add_artist(reference)
    ax.legend(handles[1:], labels[1:], bbox_to_anchor=(-0.005, 1.03), ncol=len(labels) - 1, **style)


def _footnotes(fig, lines: tuple[str, ...]) -> None:
    for index, text in enumerate(lines):
        fig.text(0.08, 0.06 - 0.03 * index, text, fontsize=9.5, color=INK_SECONDARY)


def _save(fig, plt, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot(metadata: dict, rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "svg.fonttype": "none",
    })
    profile = get_profile(metadata["profile"])
    shots = f"{metadata['shots_per_point']:,}"
    times = metadata["times_us"]
    tau = metadata["round_duration_us"]
    x_max = max(times) + 4
    x_text = max(times) + 7
    half_width = max((row["confidence_high"] - row["confidence_low"]) / 2 for row in rows)
    subtitle = (
        "Distance-3 surface code · n syndrome rounds spread evenly over T"
        f" · one round takes {tau:g} µs",
        f"{PROFILE_LABELS[profile.name]} · {PREPARATION_LABELS[metadata['prep']]}"
        f" · {shots} shots per point · 95% intervals within ±{half_width:.1%}",
    )
    footnotes = (profile.summary, profile.source)

    # Figure 1: fidelity, n >= 1, against the unencoded-qubit reference.
    fig, ax = _new_figure(plt, "Logical-Z fidelity vs storage time", subtitle)
    ends = [("Unencoded", REFERENCE, *_draw_reference(ax, rows, fidelity=True))]
    ends += [
        (f"n = {rounds}", SERIES[rounds][0], *_draw_series(ax, rows, rounds, fidelity=True))
        for rounds in FIDELITY_ROUNDS if rounds in metadata["rounds"]
    ]
    lowest = min(
        1 - row["logical_failure_rate"] for row in rows if row["rounds"] in FIDELITY_ROUNDS
    )
    y_low = min(0.5, math.floor((lowest - 0.02) * 10) / 10)
    _finish_axes(
        ax, x_max=x_max, y_lim=(y_low, 1.0), y_step=0.1, y_label="Decoded logical-Z fidelity"
    )
    _label_line_ends(ax, ends, x_text=x_text)
    _footnotes(fig, footnotes)
    _save(fig, plt, output, "logical-z-fidelity-vs-time")

    # Figure 2: failure probability, adding the n = 0 readout-only control.
    fig, ax = _new_figure(plt, "Logical-Z failure vs storage time", subtitle)
    ends = [("Unencoded", REFERENCE, *_draw_reference(ax, rows, fidelity=False))]
    ends += [
        (f"n = {rounds}", SERIES[rounds][0], *_draw_series(ax, rows, rounds, fidelity=False))
        for rounds in metadata["rounds"]
    ]
    top = math.ceil((max(row["logical_failure_rate"] for row in rows) + 0.03) / 0.05) * 0.05
    _finish_axes(
        ax, x_max=x_max, y_lim=(0, top), y_step=0.05,
        y_label="Decoded logical-Z failure probability",
    )
    _label_line_ends(ax, ends, x_text=x_text)
    _footnotes(fig, footnotes)
    _save(fig, plt, output, "logical-z-failure-vs-time")


GRID_KEYS = ("times_us", "rounds")


def source_digest() -> str:
    """One hash over the library's file contents, insensitive to file renames."""
    contents = sorted(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "src" / "surface_code").rglob("*.py")
    )
    return hashlib.sha256("\n".join(contents).encode()).hexdigest()


def _settings(saved: dict) -> dict:
    """Saved metadata minus the timestamp; older files predate profile/prep keys."""
    settings = {key: value for key, value in saved.items() if key != "created_utc"}
    settings.setdefault("profile", "baseline")
    settings.setdefault("prep", "encoder")
    return settings


def _point_seed(base_seed: int, time: float, rounds: int) -> int:
    """Unique per grid point, stable when the grid is extended later."""
    return base_seed + 1000 * ROUNDS.index(rounds) + TIMES_US.index(time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="baseline")
    parser.add_argument("--prep", choices=PREPARATIONS, default="encoder")
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026090600)
    parser.add_argument(
        "--output-dir", type=Path, help="default: artifacts/memory-by-time[-profile-prep]"
    )
    parser.add_argument("--plot-only", action="store_true", help="redraw existing saved data")
    parser.add_argument(
        "--ignore-source-change", action="store_true",
        help="extend saved data even though the library source hashes changed",
    )
    args = parser.parse_args()
    if args.shots <= 0 or args.workers <= 0:
        parser.error("shots and workers must be positive")
    profile = get_profile(args.profile)
    output = args.output_dir
    if output is None:
        default_run = (args.profile, args.prep) == ("baseline", "encoder")
        suffix = "" if default_run else f"-{args.profile}-{args.prep}"
        output = ROOT / "artifacts" / f"memory-by-time{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "data.json"
    if args.plot_only:
        saved = json.loads(data_path.read_text())
        metadata, rows = _settings(saved["metadata"]), saved["points"]
    else:
        metadata = {
            "profile": args.profile, "prep": args.prep,
            "basis": "Z", "times_us": list(TIMES_US), "rounds": list(ROUNDS),
            "round_duration_us": profile.round_duration_us, "shots_per_point": args.shots,
            "confidence": 0.95, "base_seed": args.seed,
            "circuit_noise": asdict(profile.circuit_noise),
            "idle_noise_per_us": asdict(profile.idle_noise),
            "versions": {name: version(name) for name in ("qiskit", "qiskit-aer", "matplotlib")},
            "source_digest": source_digest(),
        }
        rows = []
        if data_path.exists():
            saved = json.loads(data_path.read_text())
            old = _settings(saved["metadata"])
            ignored = set(GRID_KEYS) | ({"source_digest"} if args.ignore_source_change else set())
            old_settings = {key: value for key, value in old.items() if key not in ignored}
            new_settings = {key: value for key, value in metadata.items() if key not in ignored}
            if old_settings != new_settings:
                parser.error(
                    "Saved results have different settings or source; use a new output directory"
                )
            if any(not set(old[key]) <= set(metadata[key]) for key in GRID_KEYS):
                parser.error(
                    "Saved results cover points outside the current grid; "
                    "use a new output directory"
                )
            if old.get("source_digest") != metadata["source_digest"]:
                print("warning: extending data computed from different library source", flush=True)
            metadata = {
                **old,
                "created_utc": saved["metadata"]["created_utc"],
                **{key: metadata[key] for key in GRID_KEYS},
            }
            rows = saved["points"]
        else:
            metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
        grid = [
            (time, rounds) for time in TIMES_US for rounds in ROUNDS
            if time >= rounds * profile.round_duration_us
        ]
        completed = {(row["total_time_us"], row["rounds"]) for row in rows}
        jobs = [
            (
                time, rounds, args.shots, _point_seed(args.seed, time, rounds),
                args.profile, args.prep,
            )
            for time, rounds in grid
            if (time, rounds) not in completed
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_point, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                save_data(data_path, metadata, rows)
                print(
                    f"[{len(rows):2}/{len(grid)}] T={row['total_time_us']:3g} µs "
                    f"n={row['rounds']:2} failure={row['logical_failure_rate']:.2%}",
                    flush=True,
                )
    validate_data(metadata, rows)
    rows.sort(key=lambda row: (row["rounds"], row["total_time_us"]))
    # Derived fidelity columns for the table view; not written back to data.json.
    table = [
        {
            **row,
            "logical_fidelity": 1 - row["logical_failure_rate"],
            "fidelity_low": 1 - row["confidence_high"],
            "fidelity_high": 1 - row["confidence_low"],
        }
        for row in rows
    ]
    with (output / "data.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plot(metadata, rows, output)
    print(f"Saved figures and {len(rows)} data points to {output}", flush=True)


if __name__ == "__main__":
    main()

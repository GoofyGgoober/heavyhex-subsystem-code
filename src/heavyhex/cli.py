"""The heavyhex command.

Data qubit ids are 0-based (the paper's Q label is id + 1).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .core import Pauli
from .patches.operators import D3, D5, HeavyHexOperators

PATCHES: dict[int, HeavyHexOperators] = {3: D3, 5: D5}
BASES = ("X", "Z")
INSTALL_HINT = "python -m pip install -e '.[sim,matching]'"
HARDWARE_HINT = "python -m pip install -e '.[hardware]'"
_PAULI_WORD = re.compile(r"([XYZ])(\d+)", re.IGNORECASE)


def parse_pauli(text: str, patch: HeavyHexOperators) -> Pauli:
    """Parse 'X0 Z3' (0-based data ids), 'I', 'X_L', 'Z_L' or 'Y_L'."""
    code = patch.code
    words = text.split()
    if not words:
        raise ValueError("expected a Pauli expression such as 'X0 Z3'")
    if len(words) == 1:
        word = words[0].upper().replace("_", "")
        if word == "I":
            return Pauli()
        if word in ("XL", "YL", "ZL"):
            return {
                "XL": code.logical_x,
                "YL": code.logical_x * code.logical_z,
                "ZL": code.logical_z,
            }[word]
    x: set[int] = set()
    z: set[int] = set()
    for word in words:
        match = _PAULI_WORD.fullmatch(word)
        if match is None:
            raise ValueError(f"cannot parse Pauli word {word!r} in {text!r}")
        axis, qubit = match.group(1).upper(), int(match.group(2))
        if qubit not in code.data_qubits:
            raise ValueError(f"data qubit must be in 0..{code.n - 1}, got {qubit}")
        if axis in "XY":
            x.symmetric_difference_update((qubit,))
        if axis in "ZY":
            z.symmetric_difference_update((qubit,))
    return Pauli(frozenset(x), frozenset(z))


def pauli_words(pauli: Pauli) -> str:
    """Bare Pauli words, e.g. 'X0 Y3' or 'I'."""
    if not (pauli.x or pauli.z):
        return "I"
    words = []
    for qubit in sorted(pauli.x | pauli.z):
        axis = "Y" if qubit in pauli.x and qubit in pauli.z else "X" if qubit in pauli.x else "Z"
        words.append(f"{axis}{qubit}")
    return " ".join(words)


def format_pauli(pauli: Pauli, patch: HeavyHexOperators) -> str:
    """Pauli words, plus X_L, Y_L or Z_L when it is exactly that logical."""
    code = patch.code
    logicals = {
        code.logical_x: "X_L",
        code.logical_x * code.logical_z: "Y_L",
        code.logical_z: "Z_L",
    }
    name = logicals.get(pauli)
    text = pauli_words(pauli)
    return f"{text}  ({name})" if name else text


def _patch(args: argparse.Namespace) -> HeavyHexOperators:
    return PATCHES[args.distance]


def _emit(payload: dict[str, Any], lines: Sequence[str], *, as_json: bool) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True) if as_json else "\n".join(lines))
    return 0


def _usage(folder: Path) -> str:
    """The QPU time IBM counted for the run's job, against its limit."""
    record = json.loads((folder / "job.json").read_text()) if (folder / "job.json").exists() else {}
    usage = record.get("usage") or {}
    used = next(
        (usage[key] for key in ("qpu_charge_time_seconds", "quantum_seconds") if key in usage),
        None,
    )
    limit = record.get("qpu_seconds_limit")
    if used is None:
        return f"IBM hasn't reported the QPU time used yet (limit {limit} s)"
    if usage.get("status") != "complete":
        return (
            f"IBM has counted {used} s of QPU time so far and is still counting (limit {limit} s)"
        )
    return f"IBM counted {used} s of QPU time (limit {limit} s)"


def _not_sent_again(folder: Path) -> str:
    """What a failed submit left behind."""
    if not (folder / "job.json").exists():
        return "Nothing was sent."
    if json.loads((folder / "job.json").read_text()).get("job") is None:
        return (
            "A job may have been sent, but IBM's reply never came; analyze explains how to "
            "find it. This folder is not sent again."
        )
    return f"No shots. {_usage(folder)}; this folder is not sent again."


def _info_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    code = patch.code
    d = patch.distance
    payload = {
        "distance": d,
        "data_qubits": list(patch.data_qubits),
        "n_stabilizers": code.syndrome_size,
        "n_gauge_qubits": code.gauge_qubit_count(),
        "n_logical": code.logical_count(),
        "x_gauges": patch.x_gauges,
        "z_gauges": patch.z_gauges,
        "x_stabilizers": patch.x_stabilizers,
        "z_stabilizers": patch.z_stabilizers,
        "stabilizer_names": list(patch.stabilizer_names),
        "logical_x": pauli_words(code.logical_x),
        "logical_z": pauli_words(code.logical_z),
    }
    lines = [
        f"[[{d * d},1,{d}]] heavy-hex subsystem code, d={d}",
        f"Data ids: 0..{d * d - 1} (paper Q label = id + 1)",
        f"Stabilizers: {code.syndrome_size}, gauge qubits: {code.gauge_qubit_count()}",
        f"Logical X: {format_pauli(code.logical_x, patch)}",
        f"Logical Z: {format_pauli(code.logical_z, patch)}",
        f"X gauges ({len(patch.x_gauges)}), Z gauges ({len(patch.z_gauges)}) measured;",
        f"stabilizers ({', '.join(patch.stabilizer_names)}) inferred as products.",
    ]
    return _emit(payload, lines, as_json=args.json)


def _circuit_command(args: argparse.Namespace) -> int:
    from .circuits.flagged import memory_circuit_flagged  # qiskit is an optional extra

    patch = _patch(args)
    error = parse_pauli(args.error, patch) if args.error else None
    circuit, _ = memory_circuit_flagged(patch, basis=args.basis, rounds=args.rounds, error=error)
    print(circuit.draw(output="text"))
    return 0


def _calibrate_command(args: argparse.Namespace) -> int:
    from .patches.layout import build_layout
    from .patches.placement import Calibration, best_spots, calibrate, cost, problems, weak

    if args.offline:
        calibration = Calibration.load(args.file)
    else:
        calibration = calibrate(args.backend, args.file)
    device = json.loads(Path(args.map).read_text())
    spots = best_spots(device["coords"], device["edges"], calibration)
    picks = {}
    for d, spot in sorted(spots.items()):
        layout = build_layout(
            d, device["coords"], device["edges"], origin=spot.origin, direction=spot.direction
        )
        picks[d] = {
            "origin": list(spot.origin),
            "direction": list(spot.direction),
            "errors_per_round": round(cost(layout, calibration), 3),
            "broken": problems(layout, calibration),
            "weak": weak(layout, calibration),
        }
    lines = [
        f"{calibration.backend} calibrated {calibration.calibrated}, "
        f"pulled {calibration.pulled}" + ("" if args.offline else f", saved to {args.file}")
    ]
    for d, pick in picks.items():
        broken = f"; {', '.join(pick['broken'])}" if pick["broken"] else ""
        lines.append(
            f"d={d}: data qubit 1 at {tuple(pick['origin'])} facing {tuple(pick['direction'])}, "
            f"~{pick['errors_per_round']:.2f} errors per round{broken}"
        )
        lines += [f"    weak: {part}" for part in pick["weak"]]
    lines.append("Redraw with: python docs/figures/draw_blueprint.py")
    payload = {
        "backend": calibration.backend,
        "calibrated": calibration.calibrated,
        "pulled": calibration.pulled,
        "spots": picks,
    }
    return _emit(payload, lines, as_json=args.json)


def _experiment_command(args: argparse.Namespace) -> int:
    from datetime import date, datetime

    from .experiment import run  # qiskit, stim and pymatching are optional extras

    folder = (
        Path(args.run) if args.run else run.default_folder(args.backend, args.offline)
    ).resolve()
    if args.step == "prepare":
        return _emit_prepared(args, run, folder)
    if not (folder / "run.json").exists():
        raise ValueError(f"no prepared run in {folder}")
    info = json.loads((folder / "run.json").read_text())
    if args.step == "rehearse":
        dephasing = 0.5 if args.dephasing is None else args.dephasing
        run.rehearse(folder, args.shots, dephasing)
        payload = {"run": str(folder), "shots": args.shots, "decoupled_dephasing": dephasing}
        lines = [
            f"Simulated {args.shots} shots per memory circuit into {folder / run.REHEARSAL}",
            f"Analyze them with: heavyhex experiment analyze --rehearsal --run {folder}",
        ]
        return _emit(payload, lines, as_json=args.json)
    if args.step == "submit":
        if info["offline"]:
            raise ValueError("this run was prepared offline; prepare it again before using the QPU")
        prepared = datetime.fromisoformat(info["prepared"]).date()
        if prepared != date.today():
            raise ValueError(f"prepared on {prepared}; Fez recalibrates daily, prepare it again")
        run.check_settings(info)
        run.refuse_resubmit(folder)
        run.refuse_another_job()
        then, now = info.get("code"), run.commit()
        if not then or then["changed"] or not now or now["code"] != then["code"]:
            raise ValueError(
                "the run must be prepared, and sent, from committed code that hasn't changed "
                "since; commit the code, then prepare again"
            )
        unsaved = run.uncommitted(folder)
        if unsaved:
            raise ValueError(
                "commit and push the run folder first, so the prediction is on record "
                f"before the run: {', '.join(unsaved)}"
            )
        from qiskit_ibm_runtime import QiskitRuntimeService

        from .experiment.circuits import piece_shots

        each = piece_shots(args.shots)
        backend = QiskitRuntimeService().backend(info["backend"])  # read-only
        seconds = run.qpu_seconds(info, args.shots, backend)
        print(
            f"About to send {len(info['seconds'])} circuits ({len(info['order'])} pieces of "
            f"{each} shots) to {info['backend']}: about {seconds:.0f} s of QPU time by "
            f"IBM's rule. Limit {run.QPU_SECONDS_LIMIT} s: IBM cancels the job if it uses "
            "more, whatever has happened by then, and this folder is never sent again."
        )
        if seconds > run.QPU_SECONDS_LIMIT:
            raise ValueError(f"over the {run.QPU_SECONDS_LIMIT} s limit; not sent")
        if input("Type yes to send: ").strip() != "yes":
            print("Not sent.")
            return 1
        try:
            run.submit(folder, backend, args.shots)
        except Exception as error:  # the shots may be in; finish() can be retried by analyze
            if not (folder / run.SHOTS).exists():
                print(_not_sent_again(folder))
                raise
            print(f"Shots saved, but the calibration after the job wasn't: {error}")
        print(f"Done; shots saved in {folder / run.SHOTS}. {_usage(folder)}")
        if (folder / "calibration_after.json").exists():
            after = json.loads((folder / "calibration_after.json").read_text())["calibrated"]
            before = json.loads((folder / "calibration.json").read_text())["calibrated"]
            changed = "recalibrated" if after != before else "not recalibrated since"
            print(f"{info['backend']} {changed} the prepared calibration as the job finished")
        print("Commit the shots and calibration_after.json, then analyze.")
        return 0
    if args.rehearsal:
        if not (folder / run.REHEARSAL).exists():
            raise ValueError(f"no rehearsal in {folder}; rehearse first")
    elif not (folder / run.SHOTS).exists():
        if not (folder / "job.json").exists():
            raise ValueError(f"no shots in {folder}; submit first, or analyze --rehearsal")
        run.fetch(folder)
    elif (folder / "job.json").exists() and (
        not (folder / "calibration_after.json").exists() or run.usage_pending(folder)
    ):
        from qiskit_ibm_runtime import QiskitRuntimeService

        job_id = json.loads((folder / "job.json").read_text())["job"]
        run.finish(folder, QiskitRuntimeService().job(job_id))  # read-only
    analysis = run.analyze(folder, args.dephasing, rehearsal=args.rehearsal)
    return _emit_analysis(args, analysis, folder)


def _emit_prepared(args: argparse.Namespace, run: Any, folder: Path) -> int:
    from .patches.placement import LAST_CALIBRATION, Calibration, calibrate

    if args.offline:
        from qiskit_ibm_runtime.fake_provider import FakeFez

        from .experiment.circuits import with_durations

        calibration = Calibration.load(args.calibration or LAST_CALIBRATION)
        target = FakeFez().target
        if calibration.durations_ns:
            target = with_durations(target, calibration.durations_ns)
        timing = "the calibration's gate times" if calibration.durations_ns else "FakeFez's times"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService

        calibration = calibrate(args.backend)  # IBM's calibration in force now, read-only
        target = QiskitRuntimeService().backend(args.backend).target  # read-only
        timing = "the device's gate times"
    prediction = run.prepare(folder, calibration, target, offline=args.offline)
    low, high = (prediction[str(f)]["fits"] for f in run.ENDS)
    lines = [
        f"Run folder: {folder}",
        f"{calibration.backend} calibrated {calibration.calibrated}, pulled {calibration.pulled}; "
        f"timed with {timing}",
        "Predicted error per round, from decoupling leaving only T1 to leaving IBM's T2:",
    ]
    for basis in BASES:
        a, b = low[basis], high[basis]
        lines.append(
            f"  {basis} memory: d=3 {a['per_round']['3'][0]:.1%}-{b['per_round']['3'][0]:.1%}"
            f", d=5 {a['per_round']['5'][0]:.1%}-{b['per_round']['5'][0]:.1%}"
            f", Λ {a['lambda']:.2f}-{b['lambda']:.2f}"
        )
    lines += [
        "Redraw the blueprint with: python docs/figures/draw_blueprint.py",
        "Commit and push the run folder, so the prediction is on record before the run.",
        "Offline runs can't be sent; prepare again without --offline on the run day."
        if args.offline
        else f"Sending it uses the QPU: heavyhex experiment submit --run {folder}",
    ]
    payload = {"run": str(folder), "prediction": {f: p["fits"] for f, p in prediction.items()}}
    return _emit(payload, lines, as_json=args.json)


def _emit_analysis(args: argparse.Namespace, analysis: dict, folder: Path) -> int:
    from .experiment.circuits import SETTINGS
    from .experiment.run import DRIFT_BAR, LEAKAGE_BAR

    seen, expected = analysis["observed"], analysis["predicted_at_measured_dephasing"]
    after = analysis["predicted_from_calibration_after"]
    idle, table = analysis["idle_test"], analysis["dephasing_by_qubit"]
    code = analysis["code"]
    lines = []
    if analysis["shots_from"] != "shots.npz":
        lines.append(f"REHEARSAL: these are simulated shots from {analysis['shots_from']}")
    then, now = code["prepared"] or {}, code["analyzed"] or {}
    if then.get("code") != now.get("code") or then.get("changed") or now.get("changed"):
        lines.append(
            "The code differs from the prepared run's; the prediction below is "
            "recomputed with today's code (the frozen one is in prediction.json)"
        )
    if analysis["calibration_after_missing"]:
        lines.append(
            "No calibration_after.json: the agreement bounds leave out the calibration gap"
        )
    lines.append(
        f"Decoupling leaves {analysis['decoupled_dephasing']:.2f} of the calibrated dephasing "
        f"(0: only T1, 1: IBM's T2), "
        + (
            f"the median over data qubits, ± {idle['median_uncertainty']:.2f} "
            f"(from <X> alone: {idle['x_only_median']:.2f})"
            if table
            else "as given"
        )
    )
    spread = idle["uniformity"]
    if table and spread:
        low, high = spread["quartiles"]
        lines.append(
            f"Across {spread['qubits']} data qubits f runs {low:.2f}-{high:.2f} (quartiles); "
            f"χ²/dof {spread['chi2_per_dof']:.1f} about their weighted mean, "
            f"p = {spread['p_value']:.2g}"
        )
    if table:
        lines.append("Highest per data qubit (a side check; the median is used for every qubit):")
        lines += [
            f"  d={row['patch']} {row['data']} (q{row['qubit']}, T2 {row['t2_us']:.0f} us): "
            f"{row['dephasing']:.2f} ± {row['uncertainty']:.2f}, "
            + (
                f"phase {row['phase_per_round']:+.3f} rad per round"
                if row["phase_per_round"] == row["phase_per_round"]  # NaN: the angles don't fit
                else "phase unclear"
            )
            for row in table[:5]
            if row["measured"]
        ]
        unmeasured = [
            f"d={row['patch']} {row['data']} (q{row['qubit']})"
            for row in table
            if not row["measured"]
        ]
        if unmeasured:
            lines.append(f"The idle test can't pin these down: {', '.join(unmeasured)}")
    lines.append(f"{'':28s} {'observed':>18s} {'predicted':>18s}")
    for setting, got, want in zip(SETTINGS, seen["settings"], expected["settings"]):
        if setting.decoded:
            got3, got5 = got["logical_error"]
            want3, want5 = want["logical_error"]
            lines.append(f"{setting!s:28s} {got3:8.1%} {got5:8.1%}  {want3:8.1%} {want5:8.1%}")
    for basis in BASES:
        got, want = seen["fits"][basis], expected["fits"][basis]
        later = (
            f", {after['fits'][basis]['lambda']:.2f} from the later calibration" if after else ""
        )
        verdict = f", {got['verdict']}" if basis == "X" else ""
        lines.append(
            f"{basis} memory per round: d=3 {got['per_round']['3'][0]:.2%}, "
            f"d=5 {got['per_round']['5'][0]:.2%}, Λ {got['lambda']:.2f} ± "
            f"{got['lambda_uncertainty']:.2f}{verdict} "
            f"(predicted {want['per_round']['3'][0]:.2%}, {want['per_round']['5'][0]:.2%}, "
            f"Λ {want['lambda']:.2f}{later})"
        )
    agreement = analysis["agreement"]
    gaps = [row["calibration_gap"] for row in agreement["tests"].values()]
    lines.append(
        "Simulation matches the chip: "
        + ", ".join(
            f"{basis} {'yes' if ok else 'no'}" for basis, ok in agreement["matches"].items()
        )
        + " ("
        + ", ".join(
            f"{name} {'yes' if row['agrees'] else 'no'}" for name, row in agreement["tests"].items()
        )
        + ")"
        + ("" if any(gaps) else "; IBM didn't recalibrate during the job, so no calibration term")
    )
    for basis in BASES:
        window = seen["fits"][basis]["from_round_2"]
        lines.append(
            f"{basis} memory from round 2 (exploratory): Λ {window['lambda']:.2f} ± "
            f"{window['lambda_uncertainty']:.2f} (predicted "
            f"{expected['fits'][basis]['from_round_2']['lambda']:.2f})"
        )
    memory = analysis["dephasing_from_memory"]
    lines.append(
        f"f read off X memory (exploratory): {memory['from_eps3']:.2f} ± "
        f"{memory['from_eps3_uncertainty']:.2f} from ε₃, {memory['from_detector_rates']:.2f} "
        f"from detector rates; agree with the idle test: {memory['eps3_agrees']}, "
        f"{memory['detector_rates_agree']}"
    )
    lines.append(
        "Rise in detection rate over rounds beyond the prediction, in standard errors "
        f"(leakage above {LEAKAGE_BAR:.0f}): "
        + ", ".join(
            f"{name} {row['standard_errors']:+.1f}" for name, row in analysis["leakage"].items()
        )
    )
    lines.append(
        "Second half of the shots against the first, in standard errors: "
        + ", ".join(f"{name} {row['change']:+.1f}" for name, row in analysis["drift"].items())
        + (
            "; DRIFT"
            if any(row["drift"] for row in analysis["drift"].values())
            else f"; no drift (bar {DRIFT_BAR})"
        )
    )
    if analysis["calibration_after_filled"]:
        lines.append(
            "The calibration after the job lacked or broke qubits "
            f"{analysis['calibration_after_filled']}; their prepared values were used"
        )
    lines.append(
        f"{analysis['detectors_beyond_chance']} detectors fire beyond chance "
        f"(about {analysis['chance_excess']:.1f} standard errors) above the prediction"
        + (" from both calibrations" if after else "")
        + ("; most often:" if analysis["worst_detectors"] else "")
    )
    for row in analysis["worst_detectors"][:5]:
        d, r, name = row["detector"]
        lines.append(
            f"  {row['setting']}, d={d} round {r} {name}: {row['observed']:.1%} "
            f"vs {row['predicted']:.1%} ({row['ratio']:.1f} times, {row['excess']:+.1f} "
            "standard errors)"
        )
    saved = "analysis.json" if analysis["shots_from"] == "shots.npz" else "rehearsal_analysis.json"
    lines.append(f"Everything is in {folder / saved}")
    return _emit({"run": str(folder), **analysis}, lines, as_json=args.json)


COMMANDS = {
    "info": _info_command,
    "circuit": _circuit_command,
    "calibrate": _calibrate_command,
    "experiment": _experiment_command,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heavyhex", description="Heavy-hex subsystem code tools")
    parser.add_argument("--distance", type=int, choices=tuple(PATCHES), default=3)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Patch operators and parameters")

    circuit = sub.add_parser("circuit", help="Draw the flagged memory circuit (d=3 or d=5)")
    circuit.add_argument("--basis", type=str.upper, choices=BASES, default="Z")
    circuit.add_argument("--error", default=None, help="e.g. 'X0 Z3'")
    circuit.add_argument("--rounds", type=int, default=1)

    from .patches.placement import FEZ_MAP, LAST_CALIBRATION

    calibrate = sub.add_parser(
        "calibrate", help="Pull IBM's latest calibration (read-only) and pick the patch spots"
    )
    calibrate.add_argument("--backend", default="ibm_fez")
    calibrate.add_argument("--file", default=str(LAST_CALIBRATION))
    calibrate.add_argument("--map", default=str(FEZ_MAP))
    calibrate.add_argument(
        "--offline", action="store_true", help="use the last calibration instead of asking IBM"
    )

    experiment = sub.add_parser(
        "experiment",
        help="The hardware run: prepare, rehearse, submit (uses the QPU) or analyze",
    )
    experiment.add_argument("step", choices=("prepare", "rehearse", "submit", "analyze"))
    experiment.add_argument(
        "--run", default=None, help="run folder (default: runs/<backend>-<date>)"
    )
    experiment.add_argument("--backend", default="ibm_fez")
    experiment.add_argument(
        "--shots",
        type=int,
        default=5000,
        help="per memory circuit, a multiple of 4; each idle readout gets half",
    )
    experiment.add_argument(
        "--offline",
        action="store_true",
        help="prepare from the last calibration and IBM's snapshot",
    )
    experiment.add_argument(
        "--calibration", default=None, help="with --offline: a saved calibration file"
    )
    experiment.add_argument(
        "--rehearsal",
        action="store_true",
        help="with analyze: the rehearsal's simulated shots instead of the chip's",
    )
    experiment.add_argument(
        "--dephasing",
        type=float,
        default=None,
        help="decoupled_dephasing for rehearse (default 0.5) or analyze (default: idle test)",
    )
    return parser


def _install_hint(error: ImportError) -> str | None:
    """How to install the optional package that failed to import, if it is one."""
    module = (error.name or "").split(".", 1)[0]
    if module == "qiskit_ibm_runtime":
        return HARDWARE_HINT
    if module in ("qiskit", "pymatching", "numpy", "scipy"):
        return INSTALL_HINT
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as error:
        return int(error.code or 0)
    try:
        return COMMANDS[args.command](args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except ImportError as error:
        # qiskit, numpy and pymatching are optional and only imported where needed.
        hint = _install_hint(error)
        if hint is None:
            raise
        print(
            f"Error: the {args.command} command needs optional dependencies: {hint}",
            file=sys.stderr,
        )
        return 1

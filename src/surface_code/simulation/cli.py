"""Command-line tools for exploring the rotated distance-3 surface code."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterator, Sequence
from itertools import combinations, product
from typing import Any

from ..core import Pauli
from ..decoders import decode, z_basis_success
from ..patches import PATCH
from .aer import MAX_SEED, run_aer, shot_z_success, to_qiskit
from .noise import depolarizing_error

DEFAULT_SHOTS = 64
_COMMANDS = {"run", "interactive", "syndrome", "decode", "sweep", "circuit", "info"}
_TOKEN = re.compile(r"([XYZ])(_?L|\d+)", re.IGNORECASE)
_LOGICAL_WORD = re.compile(r"LOGICAL[\s_-]*([XYZ])", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s,*]+")

Syndrome = tuple[int, ...]
DataBits = tuple[int, ...]
Tallies = dict[tuple[Syndrome, DataBits], int]

MENU: tuple[tuple[str, str], ...] = (
    ("1 / x N", "Toggle physical X on data qubit N"),
    ("2 / z N", "Toggle physical Z on data qubit N"),
    ("6 / y N", "Toggle physical Y on data qubit N"),
    ("3 / xl", "Toggle logical X"),
    ("4 / zl", "Toggle logical Z"),
    ("add P", "Apply a Pauli expression, e.g. X0 Z3"),
    ("5 / measure", "Run the ideal Aer measurement"),
    ("d / draw", "Draw one random code-capacity Pauli error"),
    ("p / rate P", "Set the per-qubit draw probability"),
    ("s / shots N", "Set the number of Aer shots"),
    ("undo / reset", "Undo the last edit / clear the frame"),
    ("circuit / info", "Inspect the circuit / patch"),
    ("help / quit", "Show commands / exit"),
)


def parse_error(text: str) -> Pauli:
    """Parse a phase-free Pauli product such as ``X0 Y4 ZL``."""
    expanded = _LOGICAL_WORD.sub(r"\1L", text.strip())
    compact = _SEPARATORS.sub("", expanded).upper()
    if not compact or compact == "I":
        return Pauli()

    result = Pauli()
    position = 0
    while position < len(compact):
        match = _TOKEN.match(compact, position)
        if match is None:
            raise ValueError(f"could not parse Pauli {text!r} near {compact[position:]!r}")
        axis, target = match.groups()
        axis = axis.upper()
        if target.upper().endswith("L"):
            logical = {
                "X": PATCH.logical_x,
                "Y": PATCH.logical_x * PATCH.logical_z,
                "Z": PATCH.logical_z,
            }[axis]
            result = result * logical
        else:
            qubit = _data_qubit(target)
            if axis == "X":
                term = Pauli.x_on((qubit,))
            elif axis == "Z":
                term = Pauli.z_on((qubit,))
            else:
                term = Pauli.x_on((qubit,)) * Pauli.z_on((qubit,))
            result = result * term
        position = match.end()
    return result


def parse_syndrome(text: str) -> Syndrome:
    compact = re.sub(r"[\s,_]+", "", text)
    width = PATCH.code.syndrome_size
    if len(compact) != width or set(compact) - {"0", "1"}:
        raise ValueError(f"syndrome must contain exactly {width} binary bits, got {text!r}")
    return tuple(int(bit) for bit in compact)


def format_pauli(pauli: Pauli) -> str:
    if not (pauli.x or pauli.z):
        return "I"
    parts: list[str] = []
    for qubit in sorted(pauli.x | pauli.z):
        if qubit in pauli.x and qubit in pauli.z:
            parts.append(f"Y{qubit}")
        elif qubit in pauli.x:
            parts.append(f"X{qubit}")
        else:
            parts.append(f"Z{qubit}")
    text = " ".join(parts)
    logical_y = PATCH.logical_x * PATCH.logical_z
    if pauli == PATCH.logical_x:
        return f"{text}  (X_L)"
    if pauli == logical_y:
        return f"{text}  (Y_L)"
    if pauli == PATCH.logical_z:
        return f"{text}  (Z_L)"
    return text


def _plain_pauli(pauli: Pauli) -> str:
    return format_pauli(pauli).split("  (", 1)[0]


def _check_names() -> tuple[str, ...]:
    counters = {"X": 0, "Z": 0}
    names: list[str] = []
    for check in PATCH.checks:
        names.append(f"{check.basis}-check {counters[check.basis]}")
        counters[check.basis] += 1
    return tuple(names)


def _validate_syndrome(syndrome: Syndrome) -> None:
    width = PATCH.code.syndrome_size
    if len(syndrome) != width or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be {width} bits, got {syndrome!r}")


def format_syndrome(syndrome: Syndrome) -> str:
    _validate_syndrome(syndrome)
    fired = [name for name, bit in zip(_check_names(), syndrome) if bit]
    bits = "".join(str(bit) for bit in syndrome)
    grouped = f"{bits[:4]} {bits[4:]}"
    if not fired:
        return f"{grouped}  (trivial)"
    return f"{grouped}  ({', '.join(fired)})"


def _logical_class(error: Pauli, correction: Pauli) -> str:
    residual = correction * error
    representatives = (
        ("I_L", Pauli()),
        ("X_L", PATCH.logical_x),
        ("Y_L", PATCH.logical_x * PATCH.logical_z),
        ("Z_L", PATCH.logical_z),
    )
    for label, representative in representatives:
        if PATCH.code.in_stabilizer_group(residual * representative):
            return label
    return "unknown"


def _summarize_tallies(tallies: Tallies) -> dict[str, Any]:
    if not tallies:
        raise ValueError("simulator returned no outcomes")
    by_syndrome: dict[Syndrome, int] = {}
    survived = 0
    outcomes: list[dict[str, Any]] = []
    for (syndrome, data), count in tallies.items():
        _validate_syndrome(syndrome)
        if len(data) != len(PATCH.data_qubits) or any(bit not in (0, 1) for bit in data):
            raise ValueError(f"data result must be {len(PATCH.data_qubits)} bits, got {data!r}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"outcome count must be a positive integer, got {count!r}")
        success = shot_z_success(syndrome, data)
        by_syndrome[syndrome] = by_syndrome.get(syndrome, 0) + count
        survived += count * success
        outcomes.append(
            {
                "syndrome": "".join(map(str, syndrome)),
                "data": "".join(map(str, data)),
                "count": count,
                "decoded_z_success": bool(success),
            }
        )
    shots = sum(by_syndrome.values())
    dominant = min(by_syndrome, key=lambda syndrome: (-by_syndrome[syndrome], syndrome))
    outcomes.sort(key=lambda row: (-row["count"], row["syndrome"], row["data"]))
    return {
        "shots": shots,
        "decoded_z_plus": survived,
        "decoded_z_minus": shots - survived,
        "dominant_syndrome": dominant,
        "syndrome_count": len(by_syndrome),
        "outcomes": outcomes,
    }


def format_report(tallies: Tallies, error: Pauli, *, show_counts: bool = False) -> str:
    summary = _summarize_tallies(tallies)
    shots = summary["shots"]
    survived = summary["decoded_z_plus"]
    syndrome = summary["dominant_syndrome"]
    if survived == shots:
        logical_z = "+1"
    elif survived == 0:
        logical_z = "-1"
    else:
        logical_z = f"mixed ({survived}/{shots} at +1)"
    shot_word = "shot" if shots == 1 else "shots"
    correction = decode(syndrome)
    lines = [
        "Backend:      Aer stabilizer (ideal circuit)",
        f"Error:        {format_pauli(error)}",
        f"Syndrome:     {format_syndrome(syndrome)}",
        f"Correction:   {format_pauli(correction)}",
        f"Residual logical:  {_logical_class(error, correction)}",
        f"Decoded Z_L:  {logical_z}   ({shots} {shot_word})",
    ]
    if summary["syndrome_count"] > 1:
        lines.append(f"Syndromes:    {summary['syndrome_count']} distinct")
    if show_counts:
        lines.extend(("", "syndrome  data       count  decoded Z_L"))
        for outcome in summary["outcomes"]:
            eigenvalue = "+1" if outcome["decoded_z_success"] else "-1"
            lines.append(
                f"{outcome['syndrome']}  {outcome['data']}  {outcome['count']:5d}  {eigenvalue}"
            )
    return "\n".join(lines)


def _report_payload(tallies: Tallies, error: Pauli) -> dict[str, Any]:
    summary = _summarize_tallies(tallies)
    syndrome = summary.pop("dominant_syndrome")
    correction = decode(syndrome)
    return {
        "backend": "aer_stabilizer",
        "circuit_noise": "none",
        "error": _plain_pauli(error),
        "syndrome": "".join(map(str, syndrome)),
        "correction": _plain_pauli(correction),
        "residual_logical": _logical_class(error, correction),
        **summary,
    }


def _data_qubit(raw: str) -> int:
    try:
        qubit = int(raw)
    except ValueError as error:
        raise ValueError(f"data qubit must be an integer from 0 to 8, got {raw!r}") from error
    if qubit not in PATCH.data_qubits:
        raise ValueError(f"data qubit must be in {list(PATCH.data_qubits)}, got {qubit}")
    return qubit


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {raw!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def _seed(raw: str) -> int:
    value = _nonnegative_int(raw)
    if value > MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be at most {MAX_SEED}, got {value}")
    return value


def _probability(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        message = f"expected a probability in [0, 1], got {raw!r}"
        raise argparse.ArgumentTypeError(message) from error
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"expected a probability in [0, 1], got {raw!r}")
    return value


def _axes(raw: str) -> str:
    value = re.sub(r"[,\s]+", "", raw.upper())
    if not value or set(value) - {"X", "Y", "Z"}:
        raise argparse.ArgumentTypeError("axes must be a non-empty combination of X, Y, and Z")
    return "".join(axis for axis in "XYZ" if axis in value)


def _pauli_argument(raw: str) -> Pauli:
    try:
        return parse_error(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _syndrome_argument(raw: str) -> Syndrome:
    try:
        return parse_syndrome(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("error", nargs="?", type=_pauli_argument, default=Pauli(), metavar="PAULI")
    parser.add_argument(
        "-e",
        "--error",
        dest="extra_errors",
        action="append",
        type=_pauli_argument,
        default=[],
        metavar="PAULI",
        help="additional Pauli to compose; may be repeated",
    )
    parser.add_argument("-s", "--shots", type=_positive_int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=_seed, help="seed random errors and Aer sampling")
    parser.add_argument(
        "--counts", action="store_true", help="show every measured bit-string tally"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--draw-error",
        dest="noise",
        type=_probability,
        default=None,
        metavar="P",
        help="sample one code-capacity Pauli with per-qubit probability P",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        type=_probability,
        default=argparse.SUPPRESS,
        metavar="P",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draws",
        dest="steps",
        type=_nonnegative_int,
        default=None,
        metavar="N",
        help="random Paulis to compose (default: one with --draw-error)",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        type=_nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-code",
        usage="surface-code [-h] [--version] [COMMAND] ...",
        description=(
            "Explore an ideal [[9,1,3]] rotated surface-code syndrome round. "
            "With no command, open interactive mode."
        ),
        epilog=(
            "examples:\n"
            "  surface-code                         interactive mode\n"
            "  surface-code X4 --shots 128 --seed 7\n"
            "  surface-code syndrome 'X0 Z3'\n"
            "  surface-code decode '0000 1100'\n"
            "  surface-code sweep --weight 2 --failures-only\n"
            "  surface-code circuit XL"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = subparsers.add_parser(
        "run",
        help="run one ideal Aer syndrome round",
        description=(
            "Run one fixed injected error through the ideal circuit. Aer shots sample data "
            "readout strings; they do not resample the error."
        ),
    )
    _add_run_options(run)

    interactive = subparsers.add_parser("interactive", help="open the interactive error explorer")
    interactive.add_argument("-s", "--shots", type=_positive_int, default=DEFAULT_SHOTS)
    interactive.add_argument("--seed", type=_seed)
    interactive.add_argument(
        "--draw-error",
        dest="noise",
        type=_probability,
        default=0.0,
        metavar="P",
    )
    interactive.add_argument(
        "--noise",
        dest="noise",
        type=_probability,
        default=argparse.SUPPRESS,
        metavar="P",
        help=argparse.SUPPRESS,
    )

    syndrome = subparsers.add_parser("syndrome", help="inspect and decode a Pauli algebraically")
    syndrome.add_argument("error", type=_pauli_argument, metavar="PAULI")
    syndrome.add_argument("--json", action="store_true")

    decoder = subparsers.add_parser("decode", help="decode an eight-bit syndrome")
    decoder.add_argument("syndrome", type=_syndrome_argument, metavar="BITS")
    decoder.add_argument("--json", action="store_true")

    sweep = subparsers.add_parser("sweep", help="exhaustively decode ideal physical Pauli errors")
    sweep.add_argument("--weight", type=int, choices=(1, 2, 3), default=1)
    sweep.add_argument("--axes", type=_axes, default="XYZ", metavar="AXES")
    sweep.add_argument("--details", action="store_true", help="list individual errors")
    sweep.add_argument(
        "--failures-only", action="store_true", help="list only decoded Z_L failures"
    )
    sweep.add_argument("--json", action="store_true")

    circuit = subparsers.add_parser("circuit", help="draw the generated Qiskit circuit")
    circuit.add_argument("error", nargs="?", type=_pauli_argument, default=Pauli(), metavar="PAULI")

    info = subparsers.add_parser("info", help="show patch geometry, checks, and logical operators")
    info.add_argument("--json", action="store_true")
    return parser


def _combined_error(primary: Pauli, extras: Sequence[Pauli]) -> Pauli:
    result = primary
    for error in extras:
        result = result * error
    return result


def _call_aer(error: Pauli, *, shots: int, seed: int | None) -> Tallies:
    if seed is None:
        return run_aer(error, shots=shots)
    return run_aer(error, shots=shots, seed=seed)


def _run_command(args: argparse.Namespace) -> int:
    initial = _combined_error(args.error, args.extra_errors)
    frame = initial
    probability = 0.0 if args.noise is None else args.noise
    steps = int(args.noise is not None) if args.steps is None else args.steps
    rng = random.Random(args.seed)
    sampled = Pauli()
    for _ in range(steps):
        draw = depolarizing_error(probability, PATCH.data_qubits, rng)
        sampled = sampled * draw
        frame = frame * draw
    tallies = _call_aer(frame, shots=args.shots, seed=args.seed)
    if args.json:
        payload = _report_payload(tallies, frame)
        payload.update(
            {
                "initial_error": _plain_pauli(initial),
                "random_error_draw": {
                    "probability": probability,
                    "draws": steps,
                    "sampled_error": _plain_pauli(sampled),
                },
                "seed": args.seed,
            }
        )
        if not args.counts:
            payload.pop("outcomes")
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if steps:
            print(
                f"Random error: p={probability:g}, draws={steps}, seed={args.seed}, "
                f"sampled={_plain_pauli(sampled)} (fixed across shots)"
            )
        print(format_report(tallies, frame, show_counts=args.counts))
    return 0


def _syndrome_payload(error: Pauli) -> dict[str, Any]:
    syndrome = PATCH.code.syndrome(error)
    correction = decode(syndrome)
    return {
        "error": _plain_pauli(error),
        "syndrome": "".join(map(str, syndrome)),
        "fired_checks": [name for name, bit in zip(_check_names(), syndrome) if bit],
        "correction": _plain_pauli(correction),
        "residual_logical": _logical_class(error, correction),
        "decoded_z_success": bool(z_basis_success(error)),
    }


def _syndrome_command(args: argparse.Namespace) -> int:
    payload = _syndrome_payload(args.error)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        syndrome = parse_syndrome(payload["syndrome"])
        print(f"Error:        {format_pauli(args.error)}")
        print(f"Syndrome:     {format_syndrome(syndrome)}")
        print(f"Correction:   {format_pauli(decode(syndrome))}")
        print(f"Residual logical:  {payload['residual_logical']}")
        print(f"Decoded Z_L:  {'+1' if payload['decoded_z_success'] else '-1'}")
    return 0


def _decode_command(args: argparse.Namespace) -> int:
    correction = decode(args.syndrome)
    payload = {
        "syndrome": "".join(map(str, args.syndrome)),
        "fired_checks": [name for name, bit in zip(_check_names(), args.syndrome) if bit],
        "correction": _plain_pauli(correction),
        "weight": correction.weight(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Syndrome:    {format_syndrome(args.syndrome)}")
        print(f"Correction:  {format_pauli(correction)}")
        print(f"Weight:      {correction.weight()}")
    return 0


def _errors_of_weight(weight: int, axes: str) -> Iterator[Pauli]:
    for qubits in combinations(PATCH.data_qubits, weight):
        for chosen_axes in product(axes, repeat=weight):
            text = " ".join(f"{axis}{qubit}" for axis, qubit in zip(chosen_axes, qubits))
            yield parse_error(text)


def _sweep_command(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    classes = {"I_L": 0, "X_L": 0, "Y_L": 0, "Z_L": 0, "unknown": 0}
    successes = 0
    for error in _errors_of_weight(args.weight, args.axes):
        payload = _syndrome_payload(error)
        classes[payload["residual_logical"]] += 1
        successes += int(payload["decoded_z_success"])
        rows.append(payload)
    failures = len(rows) - successes
    result = {
        "weight": args.weight,
        "axes": args.axes,
        "total": len(rows),
        "decoded_z_plus": successes,
        "decoded_z_minus": failures,
        "residual_logical_counts": classes,
        "cases": rows,
    }
    if args.json:
        if args.failures_only:
            result["cases"] = [row for row in rows if not row["decoded_z_success"]]
        elif not args.details:
            result.pop("cases")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"Ideal Pauli sweep: weight={args.weight}, axes={args.axes}")
    print(f"Cases:              {len(rows)}")
    print(f"Decoded Z_L +1:     {successes}")
    print(f"Decoded Z_L -1:     {failures}")
    class_summary = ", ".join(f"{key}={value}" for key, value in classes.items() if value)
    print(f"Residual logical:   {class_summary}")
    if args.details or args.failures_only:
        print("\nerror       syndrome  correction  residual  Z_L")
        selected = rows
        if args.failures_only:
            selected = [row for row in rows if not row["decoded_z_success"]]
        for row in selected:
            eigenvalue = "+1" if row["decoded_z_success"] else "-1"
            print(
                f"{row['error']:<11} {row['syndrome']}  {row['correction']:<10} "
                f"{row['residual_logical']:<8}  {eigenvalue}"
            )
    return 0


def _info_payload() -> dict[str, Any]:
    grid = [
        list(PATCH.data_qubits[row * PATCH.distance : (row + 1) * PATCH.distance])
        for row in range(PATCH.distance)
    ]
    counters = {"X": 0, "Z": 0}
    checks: list[dict[str, Any]] = []
    for check in PATCH.checks:
        checks.append(
            {
                "name": f"{check.basis}-check {counters[check.basis]}",
                "basis": check.basis,
                "ancilla": check.ancilla,
                "data_qubits": sorted(check.support),
            }
        )
        counters[check.basis] += 1
    return {
        "parameters": {"n": len(PATCH.data_qubits), "k": 1, "distance": PATCH.distance},
        "data_qubit_grid": grid,
        "ancillas": list(PATCH.ancillas),
        "logical_x": _plain_pauli(PATCH.logical_x),
        "logical_z": _plain_pauli(PATCH.logical_z),
        "checks": checks,
    }


def format_info() -> str:
    payload = _info_payload()
    lines = [
        "[[9,1,3]] rotated planar surface code",
        "",
        "Data-qubit grid:",
        *("  " + "  ".join(str(qubit) for qubit in row) for row in payload["data_qubit_grid"]),
        "",
        f"Logical X:  {payload['logical_x']}",
        f"Logical Z:  {payload['logical_z']}",
        "",
        "Checks:",
    ]
    for check in payload["checks"]:
        support = ", ".join(map(str, check["data_qubits"]))
        lines.append(f"  {check['name']:<10} ancilla {check['ancilla']}: [{support}]")
    return "\n".join(lines)


def _info_command(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(_info_payload(), indent=2, sort_keys=True))
    else:
        print(format_info())
    return 0


def _circuit_command(args: argparse.Namespace) -> int:
    print(to_qiskit(args.error).draw(output="text"))
    return 0


def _print_menu() -> None:
    print("Commands:")
    for key, label in MENU:
        print(f"  {key:<14} {label}")


def _read(prompt: str, lines: Sequence[str] | None, index: list[int]) -> str:
    if lines is None:
        return input(prompt)
    if index[0] >= len(lines):
        raise EOFError
    value = lines[index[0]]
    index[0] += 1
    print(f"{prompt}{value}")
    return value


def _reject_extra(command: str, rest: str) -> None:
    if rest:
        raise ValueError(f"{command} does not accept an argument, got {rest!r}")


def _interactive_measure(
    frame: Pauli,
    shots: int,
    seed: int | None,
    *,
    counts: bool = False,
) -> None:
    print()
    print(format_report(_call_aer(frame, shots=shots, seed=seed), frame, show_counts=counts))


def _interactive_command(args: argparse.Namespace, lines: Sequence[str] | None) -> int:
    cursor = [0]
    frame = Pauli()
    noise_probability = args.noise
    shots = args.shots
    time = 0
    rng = random.Random(args.seed)
    initial_rng_state = rng.getstate()
    history: list[tuple[Pauli, int, tuple[Any, ...]]] = []

    print("[[9,1,3]] rotated surface-code error explorer")
    print("Aer circuit: ideal; optional draws inject a random data Pauli once.")
    print("Data qubits:")
    print("  0  1  2\n  3  4  5\n  6  7  8\n")
    _print_menu()
    while True:
        status = (
            f"draws={time} rate={noise_probability:g} shots={shots} "
            f"error={_plain_pauli(frame)}"
        )
        prompt = f"\n[{status}] > "
        try:
            raw = _read(prompt, lines, cursor).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130
        if not raw:
            continue
        command, _, rest = raw.partition(" ")
        command = command.lower()
        rest = rest.strip()
        try:
            if command in {"q", "quit", "exit"}:
                _reject_extra(command, rest)
                return 0
            if command in {"h", "help", "?"}:
                _reject_extra(command, rest)
                _print_menu()
                continue
            if command in {"show", "state"}:
                _reject_extra(command, rest)
                print(f"Injected error: {format_pauli(frame)}")
                continue
            if command in {"r", "reset", "clear"}:
                _reject_extra(command, rest)
                history.append((frame, time, rng.getstate()))
                frame, time = Pauli(), 0
                rng.setstate(initial_rng_state)
                print("Injected error and draw counter cleared.")
                continue
            if command in {"u", "undo"}:
                _reject_extra(command, rest)
                if not history:
                    print("Nothing to undo.")
                else:
                    frame, time, rng_state = history.pop()
                    rng.setstate(rng_state)
                    print(f"Injected error: {format_pauli(frame)}")
                continue
            if command in {"s", "shots"}:
                value = rest or _read(f"shots [{shots}]: ", lines, cursor).strip()
                if value:
                    shots = _positive_int(value)
                print(f"Shots: {shots}")
                continue
            if command in {"n", "noise", "p", "rate"}:
                value = rest or _read(f"p [{noise_probability:g}]: ", lines, cursor).strip()
                if value:
                    noise_probability = _probability(value)
                print(f"Random-error probability: {noise_probability:g}")
                continue
            if command in {"t", "step", "d", "draw"}:
                _reject_extra(command, rest)
                history.append((frame, time, rng.getstate()))
                injected = depolarizing_error(noise_probability, PATCH.data_qubits, rng)
                frame = frame * injected
                time += 1
                print(f"Random draw:    {format_pauli(injected)}")
                print(f"Injected error: {format_pauli(frame)}")
                continue
            if command in {"5", "m", "measure"}:
                if rest.lower() not in {"", "counts"}:
                    raise ValueError("measure accepts only the optional argument 'counts'")
                try:
                    _interactive_measure(frame, shots, args.seed, counts=rest.lower() == "counts")
                except ImportError as error:
                    if not _missing_qiskit(error):
                        raise
                    _print_qiskit_help()
                continue
            if command in {"c", "circuit"}:
                target = frame if not rest else parse_error(rest)
                try:
                    print(to_qiskit(target).draw(output="text"))
                except ImportError as error:
                    if not _missing_qiskit(error):
                        raise
                    _print_qiskit_help()
                continue
            if command in {"i", "info"}:
                _reject_extra(command, rest)
                print(format_info())
                continue

            edit: Pauli
            if command in {"1", "2", "6"}:
                axis = {"1": "X", "2": "Z", "6": "Y"}[command]
                qubit = _data_qubit(rest or _read("data qubit (0-8): ", lines, cursor).strip())
                edit = parse_error(f"{axis}{qubit}")
            elif command in {"x", "y", "z"}:
                qubit = _data_qubit(rest or _read("data qubit (0-8): ", lines, cursor).strip())
                edit = parse_error(f"{command}{qubit}")
            elif command in {"3", "xl"}:
                _reject_extra(command, rest)
                edit = PATCH.logical_x
            elif command in {"4", "zl"}:
                _reject_extra(command, rest)
                edit = PATCH.logical_z
            elif command == "yl":
                _reject_extra(command, rest)
                edit = PATCH.logical_x * PATCH.logical_z
            elif command in {"a", "add", "pauli"}:
                expression = rest or _read("Pauli: ", lines, cursor)
                edit = parse_error(expression)
            else:
                edit = parse_error(raw)
            history.append((frame, time, rng.getstate()))
            frame = frame * edit
            print(f"Injected error: {format_pauli(frame)}")
        except (ValueError, argparse.ArgumentTypeError) as error:
            print(f"Error: {error}", file=sys.stderr)
        except EOFError:
            print()
            return 0


def _normalized_argv(argv: Sequence[str]) -> list[str]:
    raw = list(argv)
    if not raw:
        return ["interactive"]
    if raw[0] == "help":
        if len(raw) == 1:
            return ["--help"]
        return [raw[1], "--help", *raw[2:]]
    if raw[0] in _COMMANDS or raw[0] in {"-h", "--help", "--version"}:
        return raw
    if raw[0].startswith("-"):
        return ["run", *raw]
    if _looks_like_pauli(raw[0]):
        return ["run", *raw]
    try:
        parse_error(raw[0])
    except ValueError:
        return raw
    return ["run", *raw]


def _looks_like_pauli(text: str) -> bool:
    return bool(re.match(r"(?:I$|[XYZ]|LOGICAL)", text, re.IGNORECASE))


def _missing_qiskit(error: ImportError) -> bool:
    if error.name is None:
        return False
    return error.name.split(".", 1)[0] in {"qiskit", "qiskit_aer"}


def _print_qiskit_help() -> None:
    print("Aer needs Qiskit: python -m pip install -e '.[sim]'", file=sys.stderr)


def main(argv: list[str] | None = None, *, lines: Sequence[str] | None = None) -> int:
    if lines is not None and argv is None:
        raw_argv = []
    else:
        raw_argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    try:
        args = parser.parse_args(_normalized_argv(raw_argv))
    except SystemExit as error:
        return int(error.code)

    handlers = {
        "run": _run_command,
        "syndrome": _syndrome_command,
        "decode": _decode_command,
        "sweep": _sweep_command,
        "circuit": _circuit_command,
        "info": _info_command,
    }
    try:
        if args.command == "interactive":
            return _interactive_command(args, lines)
        return handlers[args.command](args)
    except ImportError as error:
        if not _missing_qiskit(error):
            raise
        _print_qiskit_help()
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130

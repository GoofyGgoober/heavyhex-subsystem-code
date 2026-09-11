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

from .._validation import validate_binary_bits
from ..core import Pauli
from ..decoders import decode, z_basis_success
from ..patches import PATCH
from .aer import run_aer, shot_z_success, to_qiskit
from .cli_arguments import (
    DEFAULT_SHOTS as DEFAULT_SHOTS,
)
from .cli_arguments import (
    build_parser as build_parser,
)
from .cli_arguments import (
    parse_data_qubit,
    parse_positive_int,
    parse_probability,
)
from .cli_arguments import (
    parse_error as parse_error,
)
from .cli_arguments import (
    parse_syndrome as parse_syndrome,
)
from .noise import depolarizing_error

_COMMANDS = {
    "run",
    "interactive",
    "syndrome",
    "decode",
    "sweep",
    "circuit",
    "info",
}
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
    validate_binary_bits("syndrome", syndrome, PATCH.code.syndrome_size)


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
        validate_binary_bits("data result", data, len(PATCH.data_qubits))
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
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
    initial = _combined_error(args.error, (args.extra_errors or ()))
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
    draw_count = 0
    rng = random.Random(args.seed)
    initial_rng_state = rng.getstate()
    history: list[tuple[Pauli, int, tuple[Any, ...]]] = []

    print("[[9,1,3]] rotated surface-code error explorer")
    print("Aer circuit: ideal; optional draws inject a random data Pauli once.")
    print("Data qubits:")
    print("  0  1  2\n  3  4  5\n  6  7  8\n")
    _print_menu()
    while True:
        status = f"draws={draw_count} rate={noise_probability:g} shots={shots} error={_plain_pauli(frame)}"
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
        command, *arguments = raw.split(maxsplit=1)
        command = command.lower()
        rest = arguments[0].strip() if arguments else ""
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
                history.append((frame, draw_count, rng.getstate()))
                frame, draw_count = Pauli(), 0
                rng.setstate(initial_rng_state)
                print("Injected error and draw counter cleared.")
                continue
            if command in {"u", "undo"}:
                _reject_extra(command, rest)
                if not history:
                    print("Nothing to undo.")
                else:
                    frame, draw_count, rng_state = history.pop()
                    rng.setstate(rng_state)
                    print(f"Injected error: {format_pauli(frame)}")
                continue
            if command in {"s", "shots"}:
                value = rest or _read(f"shots [{shots}]: ", lines, cursor).strip()
                if value:
                    shots = parse_positive_int(value)
                print(f"Shots: {shots}")
                continue
            if command in {"n", "noise", "p", "rate"}:
                value = rest or _read(f"p [{noise_probability:g}]: ", lines, cursor).strip()
                if value:
                    noise_probability = parse_probability(value)
                print(f"Random-error probability: {noise_probability:g}")
                continue
            if command in {"t", "step", "d", "draw"}:
                _reject_extra(command, rest)
                history.append((frame, draw_count, rng.getstate()))
                injected = depolarizing_error(noise_probability, PATCH.data_qubits, rng)
                frame = frame * injected
                draw_count += 1
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
                qubit = parse_data_qubit(rest or _read("data qubit (0-8): ", lines, cursor).strip())
                edit = parse_error(f"{axis}{qubit}")
            elif command in {"x", "y", "z"}:
                qubit = parse_data_qubit(rest or _read("data qubit (0-8): ", lines, cursor).strip())
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
            history.append((frame, draw_count, rng.getstate()))
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
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except ImportError as error:
        if not _missing_qiskit(error):
            raise
        _print_qiskit_help()
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130

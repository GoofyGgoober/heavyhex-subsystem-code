"""Command-line tools for the heavy-hex subsystem code.

Data-qubit ids are 0-based (paper Q label = id + 1). Decoding, flagged circuits
and Aer runs are d=3 only; d=5 reports detection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import combinations, product
from typing import Any

from .core import Pauli
from .decoders.lookup import decode as lookup_decode
from .patches.operators import D3, D5, HeavyHexOperators

PATCHES: dict[int, HeavyHexOperators] = {3: D3, 5: D5}
AXES = "XYZ"
BASES = ("X", "Z")
INSTALL_HINT = "python -m pip install -e '.[sim]'"
_PAULI_WORD = re.compile(r"([XYZ])(\d+)", re.IGNORECASE)


def parse_error(text: str, patch: HeavyHexOperators) -> Pauli:
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


def parse_syndrome(text: str, patch: HeavyHexOperators) -> tuple[int, ...]:
    bits = text.replace(" ", "")
    size = patch.code.syndrome_size
    if len(bits) != size or any(bit not in "01" for bit in bits):
        raise ValueError(f"syndrome must be {size} bits, got {text!r}")
    return tuple(int(bit) for bit in bits)


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
    """Pauli words, annotated when the operator is exactly a bare logical."""
    code = patch.code
    logicals = {
        code.logical_x: "X_L",
        code.logical_x * code.logical_z: "Y_L",
        code.logical_z: "Z_L",
    }
    name = logicals.get(pauli)
    text = pauli_words(pauli)
    return f"{text}  ({name})" if name else text


def format_syndrome(syndrome: Sequence[int], patch: HeavyHexOperators) -> str:
    bits = "".join(str(bit) for bit in syndrome)
    fired = [name for name, bit in zip(patch.stabilizer_names, syndrome) if bit]
    return f"{bits}  ({', '.join(fired) if fired else 'trivial'})"


def _patch(args: argparse.Namespace) -> HeavyHexOperators:
    return PATCHES[args.distance]


def _require_d3(patch: HeavyHexOperators, what: str) -> None:
    if patch.distance != 3:
        raise ValueError(f"{what} is d=3 only")


def _emit(payload: dict[str, Any], lines: Sequence[str], *, as_json: bool) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True) if as_json else "\n".join(lines))
    return 0


def _info_payload(patch: HeavyHexOperators) -> dict[str, Any]:
    code = patch.code
    return {
        "distance": patch.distance,
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


def _info_lines(patch: HeavyHexOperators) -> list[str]:
    code = patch.code
    d = patch.distance
    return [
        f"[[{d * d},1,{d}]] heavy-hex subsystem code, d={d}",
        f"Data ids: 0..{d * d - 1} (paper Q label = id + 1)",
        f"Stabilizers: {code.syndrome_size}, gauge qubits: {code.gauge_qubit_count()}",
        f"Logical X: {format_pauli(code.logical_x, patch)}",
        f"Logical Z: {format_pauli(code.logical_z, patch)}",
        f"X gauges ({len(patch.x_gauges)}), Z gauges ({len(patch.z_gauges)}) measured;",
        f"stabilizers ({', '.join(patch.stabilizer_names)}) inferred as products.",
    ]


def _info_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    return _emit(_info_payload(patch), _info_lines(patch), as_json=args.json)


def _syndrome_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    code = patch.code
    error = parse_error(args.error, patch)
    syndrome = code.syndrome(error)
    payload: dict[str, Any] = {
        "error": pauli_words(error),
        "syndrome": "".join(map(str, syndrome)),
        "fired_checks": [n for n, b in zip(patch.stabilizer_names, syndrome) if b],
        "in_gauge_group": code.in_gauge_group(error),
    }
    lines = [
        f"Error:    {format_pauli(error, patch)}",
        f"Syndrome: {format_syndrome(syndrome, patch)}",
    ]
    if patch.distance == 3:
        correction = lookup_decode(syndrome, code)
        payload["correction"] = pauli_words(correction)
        payload["residual_in_gauge_group"] = code.in_gauge_group(correction * error)
        lines += [
            f"Correction: {format_pauli(correction, patch)}",
            f"Residual in gauge group: {payload['residual_in_gauge_group']}",
        ]
    else:
        lines.append(f"Lookup decoding is d=3 only; d={patch.distance} reports detection.")
    return _emit(payload, lines, as_json=args.json)


def _decode_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    _require_d3(patch, "lookup decoding")
    syndrome = parse_syndrome(args.syndrome, patch)
    correction = lookup_decode(syndrome, patch.code)
    payload = {
        "syndrome": "".join(map(str, syndrome)),
        "correction": pauli_words(correction),
        "weight": correction.weight(),
    }
    lines = [
        f"Syndrome:   {format_syndrome(syndrome, patch)}",
        f"Correction: {format_pauli(correction, patch)}",
        f"Weight:     {correction.weight()}",
    ]
    return _emit(payload, lines, as_json=args.json)


def _errors_of_weight(weight: int, axes: str, patch: HeavyHexOperators) -> Iterator[Pauli]:
    for qubits in combinations(patch.data_qubits, weight):
        for chosen in product(axes, repeat=weight):
            pairs = tuple(zip(chosen, qubits))
            x = frozenset(qubit for axis, qubit in pairs if axis in "XY")
            z = frozenset(qubit for axis, qubit in pairs if axis in "ZY")
            yield Pauli(x, z)


def _grade_errors(
    patch: HeavyHexOperators, weight: int, axes: str
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Grade every error of the given weight; rows cover the decoded cases."""
    code = patch.code
    counts = {"corrected": 0, "detected": 0, "harmless": 0, "harmful": 0}
    rows: list[dict[str, Any]] = []
    for error in _errors_of_weight(weight, axes, patch):
        syndrome = code.syndrome(error)
        if any(syndrome):
            if patch.distance != 3:
                counts["detected"] += 1
                continue
            ok = code.in_gauge_group(lookup_decode(syndrome, code) * error)
            counts["corrected" if ok else "harmful"] += 1
            rows.append({"error": format_pauli(error, patch), "ok": ok})
        elif code.in_gauge_group(error):
            counts["harmless"] += 1
        else:
            counts["harmful"] += 1
            rows.append({"error": format_pauli(error, patch), "ok": False})
    return counts, rows


def _sweep_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    axes = args.axes.upper()
    if not axes or any(axis not in AXES for axis in axes):
        raise ValueError(f"axes must be a non-empty selection from {AXES}, got {args.axes!r}")
    if not 0 <= args.weight <= patch.code.n:
        raise ValueError(f"weight must be in 0..{patch.code.n}, got {args.weight}")

    counts, rows = _grade_errors(patch, args.weight, axes)
    selected = [row for row in rows if not row["ok"]] if args.failures_only else rows
    payload: dict[str, Any] = {
        "distance": patch.distance,
        "weight": args.weight,
        "axes": axes,
        "total": sum(counts.values()),
        **counts,
    }
    if args.details or args.failures_only:
        payload["cases"] = selected
    lines = [
        f"Heavy-hex d={patch.distance} sweep: weight={args.weight}, axes={axes}",
        f"Cases:     {payload['total']}",
        f"Corrected: {counts['corrected']}  Detected: {counts['detected']}",
        f"Harmless:  {counts['harmless']}  Harmful:   {counts['harmful']}",
    ]
    if args.details or args.failures_only:
        lines += [f"  {row['error']:<12} {'ok' if row['ok'] else 'FAIL'}" for row in selected]
    return _emit(payload, lines, as_json=args.json)


def _run_command(args: argparse.Namespace) -> int:
    from .simulation.aer import run_memory_flagged  # qiskit-aer is an optional extra

    patch = _patch(args)
    _require_d3(patch, "flagged Aer runs")
    error = parse_error(args.error, patch) if args.error else None
    records = run_memory_flagged(
        patch, basis=args.basis, error=error, shots=args.shots, seed=args.seed
    )

    successes = sum(1 for record in records if record["success"])
    syndromes = Counter("".join(map(str, record["syndrome"])) for record in records)
    # Most frequent syndrome, ties broken lexicographically for reproducibility.
    dominant = min(syndromes, key=lambda bits: (-syndromes[bits], bits))
    payload: dict[str, Any] = {
        "backend": "aer_stabilizer_flagged",
        "basis": args.basis,
        "error": pauli_words(error) if error else "I",
        "shots": len(records),
        "successes": successes,
        "dominant_syndrome": dominant,
        "distinct_syndromes": len(syndromes),
    }
    if args.counts:
        payload["syndrome_counts"] = dict(syndromes)
    lines = [
        "Backend:  Aer stabilizer (flagged d=3 memory, one round)",
        f"Error:    {format_pauli(error, patch) if error else 'I'}",
        f"Shots:    {len(records)}  successes: {successes}",
        f"Syndrome: {dominant}  ({len(syndromes)} distinct)",
    ]
    return _emit(payload, lines, as_json=args.json)


def _circuit_command(args: argparse.Namespace) -> int:
    from .circuits.flagged import memory_circuit_flagged  # qiskit is an optional extra

    patch = _patch(args)
    _require_d3(patch, "flagged circuits")
    error = parse_error(args.error, patch) if args.error else None
    circuit, _ = memory_circuit_flagged(patch, basis=args.basis, error=error)
    print(circuit.draw(output="text"))
    return 0


COMMANDS = {
    "info": _info_command,
    "syndrome": _syndrome_command,
    "decode": _decode_command,
    "sweep": _sweep_command,
    "run": _run_command,
    "circuit": _circuit_command,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heavyhex", description="Heavy-hex subsystem code tools")
    parser.add_argument("--distance", type=int, choices=tuple(PATCHES), default=3)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Patch operators and parameters")

    syndrome = sub.add_parser("syndrome", help="Syndrome + correction for a Pauli error")
    syndrome.add_argument("error", help="e.g. 'X0 Z3', X_L, I")

    decode = sub.add_parser("decode", help="Lookup correction for a syndrome (d=3)")
    decode.add_argument("syndrome", help=f"{D3.code.syndrome_size} bits, X stabilizers then Z")

    sweep = sub.add_parser("sweep", help="Grade every error of a Pauli weight")
    sweep.add_argument("--weight", type=int, required=True)
    sweep.add_argument("--axes", default=AXES)
    sweep.add_argument("--details", action="store_true")
    sweep.add_argument("--failures-only", action="store_true")

    run = sub.add_parser("run", help="Flagged Aer memory round (d=3)")
    run.add_argument("--basis", type=str.upper, choices=BASES, default="Z")
    run.add_argument("--shots", type=int, default=128)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--error", default=None, help="e.g. 'X0 Z3'")
    run.add_argument("--counts", action="store_true")

    circuit = sub.add_parser("circuit", help="Draw the flagged memory circuit (d=3)")
    circuit.add_argument("--basis", type=str.upper, choices=BASES, default="Z")
    circuit.add_argument("--error", default=None, help="e.g. 'X0 Z3'")
    return parser


def _missing_extra(error: ImportError) -> bool:
    return (error.name or "").split(".", 1)[0] in ("qiskit", "qiskit_aer")


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
        # Qiskit only ever reaches us from the lazily imported simulator paths.
        if not _missing_extra(error):
            raise
        print(f"Error: the {args.command} command needs Qiskit: {INSTALL_HINT}", file=sys.stderr)
        return 1

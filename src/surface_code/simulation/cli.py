"""Command-line tools for the heavy-hex subsystem code (d=3 and d=5).

Data-qubit ids are 0-based; the paper's Q label is id + 1. Lookup decoding is
a d=3 tool (64 syndromes); d=5 syndrome commands report detection only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations, product
from typing import Any

from ..core import Pauli
from ..decoders.basis import normalize_measurement_basis
from ..decoders.heavyhex_lookup import decode as lookup_decode
from ..patches.heavyhex import D3, D5, HeavyHexOperators

PATCHES = {3: D3, 5: D5}
_ERROR_TOKEN = re.compile(r"\s*([XYZ])\s*(\d+)\s*", re.IGNORECASE)


def parse_error(text: str, patch: HeavyHexOperators) -> Pauli:
    """Parse 'X0 Z3' (0-based data ids), 'I', 'X_L', 'Z_L', 'Y_L'."""
    code = patch.code
    words = text.split()
    if len(words) == 1 and words[0].upper() in ("I", "II"):
        return Pauli()
    logicals = {
        "X_L": code.logical_x,
        "XL": code.logical_x,
        "Z_L": code.logical_z,
        "ZL": code.logical_z,
        "Y_L": code.logical_x * code.logical_z,
        "YL": code.logical_x * code.logical_z,
    }
    if len(words) == 1 and words[0].upper() in logicals:
        return logicals[words[0].upper()]
    x: set[int] = set()
    z: set[int] = set()
    for word in words:
        match = _ERROR_TOKEN.fullmatch(word)
        if match is None:
            raise ValueError(f"cannot parse Pauli word {word!r} in {text!r}")
        axis, qubit = match.group(1).upper(), int(match.group(2))
        if qubit not in code.data_qubits:
            raise ValueError(f"data qubit must be in 0..{code.n - 1}, got {qubit}")
        if axis in "XY":
            x.symmetric_difference_update((qubit,))
        if axis in "ZY":
            z.symmetric_difference_update((qubit,))
    if not words:
        raise ValueError("expected a Pauli expression such as 'X0 Z3'")
    return Pauli(frozenset(x), frozenset(z))


def parse_syndrome(text: str, patch: HeavyHexOperators) -> tuple[int, ...]:
    bits = text.replace(" ", "")
    size = patch.code.syndrome_size
    if len(bits) != size or any(bit not in "01" for bit in bits):
        raise ValueError(f"syndrome must be {size} bits, got {text!r}")
    return tuple(int(bit) for bit in bits)


def format_pauli(pauli: Pauli, patch: HeavyHexOperators) -> str:
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
    code = patch.code
    if pauli == code.logical_x:
        return f"{text}  (X_L)"
    if pauli == code.logical_x * code.logical_z:
        return f"{text}  (Y_L)"
    if pauli == code.logical_z:
        return f"{text}  (Z_L)"
    return text


def format_syndrome(syndrome: tuple[int, ...], patch: HeavyHexOperators) -> str:
    names = patch.stabilizer_names
    fired = [name for name, bit in zip(names, syndrome) if bit]
    bits = "".join(str(bit) for bit in syndrome)
    if not fired:
        return f"{bits}  (trivial)"
    return f"{bits}  ({', '.join(fired)})"


def _patch(args: argparse.Namespace) -> HeavyHexOperators:
    return PATCHES[args.distance]


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
        "logical_x": format_pauli(code.logical_x, patch).split("  (", 1)[0],
        "logical_z": format_pauli(code.logical_z, patch).split("  (", 1)[0],
    }


def _info_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    if args.json:
        print(json.dumps(_info_payload(patch), indent=2, sort_keys=True))
        return 0
    code = patch.code
    d = patch.distance
    print(f"[[{d * d},1,{d}]] heavy-hex subsystem code, d={d}")
    print(f"Data ids: 0..{d * d - 1} (paper Q label = id + 1)")
    print(f"Stabilizers: {code.syndrome_size}, gauge qubits: {code.gauge_qubit_count()}")
    print(f"Logical X: {format_pauli(code.logical_x, patch)}")
    print(f"Logical Z: {format_pauli(code.logical_z, patch)}")
    print(f"X gauges ({len(patch.x_gauges)}), Z gauges ({len(patch.z_gauges)}) measured;")
    print(f"stabilizers ({', '.join(patch.stabilizer_names)}) inferred as products.")
    return 0


def _syndrome_payload(error: Pauli, patch: HeavyHexOperators) -> dict[str, Any]:
    code = patch.code
    syndrome = code.syndrome(error)
    payload: dict[str, Any] = {
        "error": format_pauli(error, patch).split("  (", 1)[0],
        "syndrome": "".join(map(str, syndrome)),
        "fired_checks": [n for n, b in zip(patch.stabilizer_names, syndrome) if b],
        "in_gauge_group": code.in_gauge_group(error),
    }
    if patch.distance == 3:
        correction = lookup_decode(syndrome, code)
        payload["correction"] = format_pauli(correction, patch).split("  (", 1)[0]
        payload["residual_in_gauge_group"] = code.in_gauge_group(correction * error)
    return payload


def _syndrome_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    error = parse_error(args.error, patch)
    if args.json:
        print(json.dumps(_syndrome_payload(error, patch), indent=2, sort_keys=True))
        return 0
    payload = _syndrome_payload(error, patch)
    print(f"Error:    {format_pauli(error, patch)}")
    print(f"Syndrome: {format_syndrome(parse_syndrome(payload['syndrome'], patch), patch)}")
    if "correction" in payload:
        correction = lookup_decode(parse_syndrome(payload["syndrome"], patch), patch.code)
        print(f"Correction: {format_pauli(correction, patch)}")
        print(f"Residual in gauge group: {payload['residual_in_gauge_group']}")
    else:
        print("Lookup decoding is d=3 only; d=5 reports detection.")
    return 0


def _decode_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    if patch.distance != 3:
        print("Error: lookup decoding is d=3 only", file=sys.stderr)
        return 2
    syndrome = parse_syndrome(args.syndrome, patch)
    correction = lookup_decode(syndrome, patch.code)
    if args.json:
        print(
            json.dumps(
                {
                    "syndrome": "".join(map(str, syndrome)),
                    "correction": format_pauli(correction, patch).split("  (", 1)[0],
                    "weight": correction.weight(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"Syndrome:   {format_syndrome(syndrome, patch)}")
    print(f"Correction: {format_pauli(correction, patch)}")
    print(f"Weight:     {correction.weight()}")
    return 0


def _errors_of_weight(weight: int, axes: str, patch: HeavyHexOperators):
    for qubits in combinations(patch.data_qubits, weight):
        for chosen in product(axes, repeat=weight):
            yield parse_error(" ".join(f"{a}{q}" for a, q in zip(chosen, qubits)), patch)


def _sweep_command(args: argparse.Namespace) -> int:
    patch = _patch(args)
    code = patch.code
    rows: list[dict[str, Any]] = []
    counts = {"corrected": 0, "detected": 0, "harmless": 0, "harmful": 0}
    for error in _errors_of_weight(args.weight, args.axes, patch):
        syndrome = code.syndrome(error)
        if any(syndrome):
            if patch.distance == 3:
                ok = code.in_gauge_group(lookup_decode(syndrome, code) * error)
                counts["corrected" if ok else "harmful"] += 1
                rows.append({"error": format_pauli(error, patch), "ok": ok})
            else:
                counts["detected"] += 1
        elif code.in_gauge_group(error):
            counts["harmless"] += 1
        else:
            counts["harmful"] += 1
            rows.append({"error": format_pauli(error, patch), "ok": False})
    result = {
        "distance": patch.distance,
        "weight": args.weight,
        "axes": args.axes,
        "total": sum(counts.values()),
        **counts,
    }
    if args.json:
        if args.failures_only:
            result["cases"] = [row for row in rows if not row["ok"]]
        elif args.details:
            result["cases"] = rows
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"Heavy-hex d={patch.distance} sweep: weight={args.weight}, axes={args.axes}")
    print(f"Cases:     {result['total']}")
    print(f"Corrected: {counts['corrected']}  Detected: {counts['detected']}")
    print(f"Harmless:  {counts['harmless']}  Harmful:   {counts['harmful']}")
    if args.details or args.failures_only:
        selected = [row for row in rows if not row["ok"]] if args.failures_only else rows
        for row in selected:
            print(f"  {row['error']:<12} {'ok' if row['ok'] else 'FAIL'}")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    try:
        from .heavyhex_aer import run_memory_flagged
    except ImportError:
        print("Aer needs Qiskit: python -m pip install -e '.[sim]'", file=sys.stderr)
        return 1
    patch = _patch(args)
    if patch.distance != 3:
        print("Error: flagged Aer runs are d=3 only", file=sys.stderr)
        return 2
    basis = normalize_measurement_basis(args.basis)
    error = parse_error(args.error, patch) if args.error else None
    records = run_memory_flagged(patch, basis=basis, error=error, shots=args.shots, seed=args.seed)
    succeeded = sum(1 for record in records if record["success"])
    syndromes: dict[str, int] = {}
    for record in records:
        key = "".join(map(str, record["syndrome"]))
        syndromes[key] = syndromes.get(key, 0) + 1
    dominant = min(syndromes, key=lambda key: (-syndromes[key], key))
    if args.json:
        payload: dict[str, Any] = {
            "backend": "aer_stabilizer_flagged",
            "basis": basis,
            "error": format_pauli(error, patch).split("  (", 1)[0] if error else "I",
            "shots": len(records),
            "successes": succeeded,
            "dominant_syndrome": dominant,
            "distinct_syndromes": len(syndromes),
        }
        if args.counts:
            payload["syndrome_counts"] = syndromes
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("Backend:  Aer stabilizer (flagged d=3 memory, one round)")
    print(f"Error:    {format_pauli(error, patch) if error else 'I'}")
    print(f"Shots:    {len(records)}  successes: {succeeded}")
    print(f"Syndrome: {dominant}  ({len(syndromes)} distinct)")
    return 0


def _circuit_command(args: argparse.Namespace) -> int:
    try:
        from ..circuits.heavyhex_flagged import memory_circuit_flagged
    except ImportError as error:
        if error.name is None or error.name.split(".", 1)[0] != "qiskit":
            raise
        print("Circuits need Qiskit: python -m pip install -e '.[sim]'", file=sys.stderr)
        return 1
    patch = _patch(args)
    if patch.distance != 3:
        print("Error: flagged circuits are d=3 only", file=sys.stderr)
        return 2
    basis = normalize_measurement_basis(args.basis)
    error = parse_error(args.error, patch) if args.error else None
    circuit, _ = memory_circuit_flagged(patch, basis=basis, error=error)
    print(circuit.draw(output="text"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-code", description="Heavy-hex subsystem code tools"
    )
    parser.add_argument("--distance", type=int, choices=(3, 5), default=3)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Patch operators and parameters")

    syndrome = sub.add_parser("syndrome", help="Syndrome + correction for a Pauli error")
    syndrome.add_argument("error", help="e.g. 'X0 Z3', X_L, I")

    decode = sub.add_parser("decode", help="Lookup correction for a syndrome (d=3)")
    decode.add_argument("syndrome", help="6 bits, X stabilizers then Z")

    sweep = sub.add_parser("sweep", help="Grade every error of a Pauli weight")
    sweep.add_argument("--weight", type=int, required=True)
    sweep.add_argument("--axes", default="XYZ")
    sweep.add_argument("--details", action="store_true")
    sweep.add_argument("--failures-only", action="store_true")

    run = sub.add_parser("run", help="Flagged Aer memory round (d=3)")
    run.add_argument("--basis", default="Z")
    run.add_argument("--shots", type=int, default=128)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--error", default=None, help="e.g. 'X0 Z3'")
    run.add_argument("--counts", action="store_true")

    circuit = sub.add_parser("circuit", help="Draw the flagged memory circuit (d=3)")
    circuit.add_argument("--basis", default="Z")
    circuit.add_argument("--error", default=None, help="e.g. 'X0 Z3'")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as error:
        return int(error.code or 0)
    handlers = {
        "info": _info_command,
        "syndrome": _syndrome_command,
        "decode": _decode_command,
        "sweep": _sweep_command,
        "run": _run_command,
        "circuit": _circuit_command,
    }
    try:
        return handlers[args.command](args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

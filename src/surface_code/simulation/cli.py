"""Interactive menu: pick an error, run one Aer syndrome round."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence

from ..core import Pauli
from ..decoders import decode
from ..patches import PATCH
from .aer import run_aer, shot_z_success

_TOKEN = re.compile(r"([XYZ])(\d+)")

MENU: tuple[tuple[str, str], ...] = (
    ("1", "Physical X_i"),
    ("2", "Physical Z_i"),
    ("3", "Logical X"),
    ("4", "Logical Z"),
    ("5", "Measure"),
    ("r", "Reset"),
)


def parse_error(text: str) -> Pauli:
    compact = text.replace(" ", "")
    if not compact or compact == "I":
        return Pauli()
    leftover = _TOKEN.sub("", compact)
    if leftover:
        raise ValueError(f"could not parse Pauli {text!r}")
    x: set[int] = set()
    z: set[int] = set()
    for axis, index in _TOKEN.findall(compact):
        qubit = int(index)
        if axis in "XY":
            x.add(qubit)
        if axis in "YZ":
            z.add(qubit)
    return Pauli(frozenset(x), frozenset(z))


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
    if pauli == PATCH.logical_x:
        return f"{text}  (X_L)"
    if pauli == PATCH.logical_z:
        return f"{text}  (Z_L)"
    return text


def format_syndrome(syndrome: tuple[int, ...]) -> str:
    names = ("X-check 0", "X-check 1", "X-check 2", "X-check 3", "Z-check 0", "Z-check 1", "Z-check 2", "Z-check 3")
    fired = [name for name, bit in zip(names, syndrome) if bit]
    bits = "".join(str(bit) for bit in syndrome)
    if not fired:
        return f"{bits}  (trivial)"
    return f"{bits}  ({', '.join(fired)})"


def format_report(
    tallies: dict[tuple[tuple[int, ...], tuple[int, ...]], int],
    error: Pauli,
) -> str:
    shots = sum(tallies.values())
    by_syndrome: dict[tuple[int, ...], int] = {}
    survived = 0
    for (syndrome, data), count in tallies.items():
        by_syndrome[syndrome] = by_syndrome.get(syndrome, 0) + count
        survived += count * shot_z_success(syndrome, data)
    syndrome = max(by_syndrome, key=by_syndrome.__getitem__)
    if survived == shots:
        logical_z = "+1"
    elif survived == 0:
        logical_z = "-1"
    else:
        logical_z = f"mixed ({survived}/{shots})"
    lines = [
        f"Error:       {format_pauli(error)}",
        f"Syndrome:    {format_syndrome(syndrome)}",
        f"Correction:  {format_pauli(decode(syndrome))}",
        f"Logical Z:   {logical_z}   ({shots} shots)",
    ]
    if len(by_syndrome) > 1:
        lines.append(f"Syndromes:   {len(by_syndrome)}")
    return "\n".join(lines)


def _print_menu(frame: Pauli) -> None:
    print("\n[[9,1,3]] rotated surface code")
    print(f"State:  {format_pauli(frame)}\n")
    for key, label in MENU:
        print(f"  {key}  {label}")
    print("  q  Quit")


def _data_qubit(raw: str) -> int:
    qubit = int(raw)
    if qubit not in PATCH.data_qubits:
        raise ValueError(f"data qubit must be in {list(PATCH.data_qubits)}, got {qubit}")
    return qubit


def _read(prompt: str, lines: Sequence[str] | None, index: list[int]) -> str:
    if lines is None:
        return input(prompt)
    if index[0] >= len(lines):
        raise EOFError
    value = lines[index[0]]
    index[0] += 1
    print(f"{prompt}{value}")
    return value


def main(argv: list[str] | None = None, *, lines: Sequence[str] | None = None) -> int:
    del argv
    scripted = lines
    if scripted is None and not sys.stdin.isatty():
        scripted = [line.rstrip("\n") for line in sys.stdin]
    cursor = [0]
    frame = Pauli()
    try:
        while True:
            _print_menu(frame)
            choice = _read("> ", scripted, cursor).strip().lower()
            if choice in {"q", "quit"}:
                return 0
            if choice == "r":
                frame = Pauli()
                continue
            if choice == "1":
                frame = frame * Pauli.x_on((_data_qubit(_read("i (0-8): ", scripted, cursor)),))
            elif choice == "2":
                frame = frame * Pauli.z_on((_data_qubit(_read("i (0-8): ", scripted, cursor)),))
            elif choice == "3":
                frame = frame * PATCH.logical_x
            elif choice == "4":
                frame = frame * PATCH.logical_z
            elif choice != "5":
                print("Invalid selection.")
                continue
            print()
            print(format_report(run_aer(frame, shots=64), frame))
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ImportError:
        print("Aer needs qiskit: uv pip install -e '.[sim]'", file=sys.stderr)
        return 1

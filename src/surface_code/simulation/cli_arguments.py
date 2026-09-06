"""CLI argument definitions and text parsing; no simulation is run here."""

from __future__ import annotations

import argparse
import re
from importlib.metadata import PackageNotFoundError, version
from math import isfinite

from ..core import Pauli
from ..patches import PATCH
from .aer import MAX_SEED
from .profiles import PROFILES
from .record_shots import PREPARATIONS

DEFAULT_SHOTS = 64
_TOKEN = re.compile(r"([XYZ])(_?L|\d+)", re.IGNORECASE)
_LOGICAL_WORD = re.compile(r"LOGICAL[\s_-]*([XYZ])", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s,*]+")

Syndrome = tuple[int, ...]


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
            qubit = parse_data_qubit(target)
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


def parse_data_qubit(raw: str) -> int:
    try:
        qubit = int(raw)
    except ValueError as error:
        raise ValueError(f"data qubit must be an integer from 0 to 8, got {raw!r}") from error
    if qubit not in PATCH.data_qubits:
        raise ValueError(f"data qubit must be in {list(PATCH.data_qubits)}, got {qubit}")
    return qubit


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def parse_nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {raw!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def parse_seed(raw: str) -> int:
    value = parse_nonnegative_int(raw)
    if value > MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be at most {MAX_SEED}, got {value}")
    return value


def parse_probability(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        message = f"expected a probability in [0, 1], got {raw!r}"
        raise argparse.ArgumentTypeError(message) from error
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"expected a probability in [0, 1], got {raw!r}")
    return value


def parse_nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {raw!r}") from error
    if not isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {raw!r}")
    return value


def parse_positive_float(raw: str) -> float:
    value = parse_nonnegative_float(raw)
    if value == 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {raw!r}")
    return value


def parse_confidence(raw: str) -> float:
    value = parse_probability(raw)
    if value in {0.0, 1.0}:
        raise argparse.ArgumentTypeError("confidence must be strictly between 0 and 1")
    return value


def parse_memory_basis(raw: str) -> str:
    normalized = raw.upper()
    if normalized not in {"X", "Z", "BOTH"}:
        raise argparse.ArgumentTypeError("basis must be X, Z, or both")
    return normalized


def parse_axes(raw: str) -> str:
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
        default=None,
        metavar="PAULI",
        help="additional Pauli to compose; may be repeated",
    )
    parser.add_argument("-s", "--shots", type=parse_positive_int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=parse_seed, help="seed random errors and Aer sampling")
    parser.add_argument(
        "--counts", action="store_true", help="show every measured bit-string tally"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--draw-error",
        dest="noise",
        type=parse_probability,
        default=None,
        metavar="P",
        help="sample one code-capacity Pauli with per-qubit probability P",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        type=parse_probability,
        default=argparse.SUPPRESS,
        metavar="P",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draws",
        dest="steps",
        type=parse_nonnegative_int,
        default=None,
        metavar="N",
        help="random Paulis to compose (default: one with --draw-error)",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        type=parse_nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=argparse.SUPPRESS,
    )


def _add_circuit_noise_options(parser: argparse.ArgumentParser) -> None:
    for option, gate in (
        ("single-qubit", "one-qubit gate depolarizing"),
        ("two-qubit", "two-qubit gate depolarizing"),
        ("readout", "measurement flip"),
        ("reset", "reset flip"),
    ):
        parser.add_argument(
            f"--{option}-error", type=parse_probability, metavar="P", help=f"{gate} probability"
        )


def _add_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="baseline",
        help="named noise profile supplying the default rates (explicit rates override it)",
    )
    parser.add_argument(
        "--preparation",
        choices=PREPARATIONS,
        default="encoder",
        help="logical-state preparation: synthesized Clifford encoder or product state",
    )


def _package_version() -> str:
    try:
        return version("quantum-surface-code")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-code",
        usage="surface-code [-h] [--version] [COMMAND] ...",
        description=(
            "Explore the [[9,1,3]] rotated surface code: single syndrome rounds, "
            "noisy repeated-round memory, and fixed-duration cadence sweeps. "
            "With no command, open interactive mode."
        ),
        epilog=(
            "examples:\n"
            "  surface-code                         interactive mode\n"
            "  surface-code X4 --shots 128 --seed 7\n"
            "  surface-code syndrome 'X0 Z3'\n"
            "  surface-code decode '0000 1100'\n"
            "  surface-code memory --rounds 8 --two-qubit-error 0.01\n"
            "  surface-code cadence --time-us 10 --rounds 0 1 2 4 8\n"
            "  surface-code sweep --weight 2 --failures-only\n"
            "  surface-code circuit XL"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
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
    interactive.add_argument("-s", "--shots", type=parse_positive_int, default=DEFAULT_SHOTS)
    interactive.add_argument("--seed", type=parse_seed)
    interactive.add_argument(
        "--draw-error",
        dest="noise",
        type=parse_probability,
        default=0.0,
        metavar="P",
    )
    interactive.add_argument(
        "--noise",
        dest="noise",
        type=parse_probability,
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
    sweep.add_argument("--axes", type=parse_axes, default="XYZ", metavar="AXES")
    sweep.add_argument("--details", action="store_true", help="list individual errors")
    sweep.add_argument(
        "--failures-only", action="store_true", help="list only decoded Z_L failures"
    )
    sweep.add_argument("--json", action="store_true")

    memory = subparsers.add_parser(
        "memory",
        help="run repeated ancilla measurement/reset rounds",
        description=(
            "Run a repeated logical-zero memory circuit in Aer with a vendor-neutral "
            "baseline noise model."
        ),
    )
    memory.add_argument(
        "--rounds", type=parse_nonnegative_int, default=4, help="syndrome rounds to run"
    )
    _add_profile_options(memory)
    memory.add_argument(
        "-s", "--shots", type=parse_positive_int, default=DEFAULT_SHOTS, help="Aer shots"
    )
    memory.add_argument("--seed", type=parse_seed, help="seed Aer sampling")
    memory.add_argument(
        "-e",
        "--error",
        type=_pauli_argument,
        default=Pauli(),
        metavar="PAULI",
        help="Pauli injected after state preparation",
    )
    _add_circuit_noise_options(memory)
    memory.add_argument(
        "--ideal",
        action="store_true",
        help="start from zero noise (explicit error-rate options still override it)",
    )
    memory.add_argument("--json", action="store_true")

    cadence = subparsers.add_parser(
        "cadence",
        help="sweep syndrome frequency at one fixed storage time",
        description=(
            "Hold total memory time fixed, vary the number of syndrome rounds, "
            "and decode each complete syndrome history."
        ),
    )
    cadence.add_argument(
        "--time-us",
        type=parse_positive_float,
        default=10.0,
        metavar="T",
        help="fixed storage window in microseconds",
    )
    cadence.add_argument(
        "--rounds",
        type=parse_nonnegative_int,
        nargs="+",
        default=(0, 1, 2, 4, 8),
        metavar="N",
        help="numbers of syndrome rounds to compare (0 is the readout-only control)",
    )
    cadence.add_argument(
        "--round-duration-us",
        type=parse_nonnegative_float,
        metavar="TAU",
        help="modeled duration of one syndrome round (default: the profile's value)",
    )
    _add_profile_options(cadence)
    cadence.add_argument(
        "-s", "--shots", type=parse_positive_int, default=1024, help="Aer shots per point"
    )
    cadence.add_argument("--seed", type=parse_seed, help="seed Aer sampling")
    cadence.add_argument(
        "--basis",
        type=parse_memory_basis,
        default="BOTH",
        help="logical basis to store and read out: Z, X, or BOTH",
    )
    cadence.add_argument(
        "--confidence", type=parse_confidence, default=0.95, help="Wilson interval confidence level"
    )
    _add_circuit_noise_options(cadence)
    for axis in "xyz":
        cadence.add_argument(
            f"--idle-{axis}-rate",
            type=parse_nonnegative_float,
            metavar="RATE",
            help=f"idle {axis.upper()}-event rate per microsecond on each data qubit",
        )
    cadence.add_argument(
        "--ideal-circuit",
        action="store_true",
        help="start circuit faults at zero before applying explicit overrides",
    )
    cadence.add_argument(
        "--ideal-idle",
        action="store_true",
        help="start idle rates at zero before applying explicit overrides",
    )
    cadence.add_argument("--json", action="store_true")

    circuit = subparsers.add_parser("circuit", help="draw the generated Qiskit circuit")
    circuit.add_argument("error", nargs="?", type=_pauli_argument, default=Pauli(), metavar="PAULI")
    circuit.add_argument(
        "--rounds",
        type=parse_nonnegative_int,
        help="draw a repeated-round memory circuit instead of the one-round circuit",
    )
    circuit.add_argument(
        "--preparation",
        choices=PREPARATIONS,
        default="encoder",
        help="logical-state preparation for the repeated-round circuit",
    )

    info = subparsers.add_parser("info", help="show patch geometry, checks, and logical operators")
    info.add_argument("--json", action="store_true")
    return parser

"""Weight-1 lookup decoder.

8 syndrome bits → a data Pauli. Rows are I and every single-qubit X/Y/Z.
If two errors share a syndrome they differ by a stabilizer; we keep the first.
Syndromes that never show up for a weight-1 error are missing (None).
"""

from .pauli import Pauli
from .qubits import DATA_QUBITS
from .syndrome import extract_syndrome

Syndrome = tuple[int, ...]


def _single_qubit_errors() -> tuple[Pauli, ...]:
    errors: list[Pauli] = []
    for qubit in DATA_QUBITS:
        x = Pauli.x_on((qubit,))
        z = Pauli.z_on((qubit,))
        errors.extend((x, z, x * z))
    return tuple(errors)


def _build_table() -> dict[Syndrome, Pauli]:
    table: dict[Syndrome, Pauli] = {(0, 0, 0, 0, 0, 0, 0, 0): Pauli()}
    for error in _single_qubit_errors():
        table.setdefault(extract_syndrome(error), error)
    return table


LOOKUP = _build_table()


def decode(syndrome: Syndrome) -> Pauli | None:
    if len(syndrome) != 8 or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be 8 bits, got {syndrome!r}")
    return LOOKUP.get(syndrome)

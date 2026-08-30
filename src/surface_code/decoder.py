"""Weight-1 decoder: 8 syndrome bits → a data Pauli.

All zeros is I. Otherwise the first single-qubit X/Y/Z with that syndrome.
No match → None.
"""

from .logicals import LOGICAL_Z
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


def decode(syndrome: Syndrome) -> Pauli | None:
    if len(syndrome) != 8 or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be 8 bits, got {syndrome!r}")
    if syndrome == (0, 0, 0, 0, 0, 0, 0, 0):
        return Pauli()
    for error in _single_qubit_errors():
        if extract_syndrome(error) == syndrome:
            return error
    return None


def z_basis_success(error: Pauli) -> int | None:
    """1 if leftover commutes with Z_L, 0 if it is an X_L. None if undecoded."""
    correction = decode(extract_syndrome(error))
    if correction is None:
        return None
    leftover = correction * error
    return int(leftover.commutes(LOGICAL_Z))

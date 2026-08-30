"""Weight-1 decoder: 8 syndrome bits → a data Pauli.

All zeros is I. Otherwise the first single-qubit X/Y/Z with that syndrome.
No match → None.
"""

from ..circuits import extract_syndrome
from ..core import Pauli
from ..patches import PATCH

Syndrome = tuple[int, ...]


def _single_qubit_errors() -> tuple[Pauli, ...]:
    errors: list[Pauli] = []
    for qubit in PATCH.data_qubits:
        x = Pauli.x_on((qubit,))
        z = Pauli.z_on((qubit,))
        errors.extend((x, z, x * z))
    return tuple(errors)


def decode(syndrome: Syndrome) -> Pauli | None:
    if len(syndrome) != PATCH.code.syndrome_size or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be {PATCH.code.syndrome_size} bits, got {syndrome!r}")
    if syndrome == (0,) * PATCH.code.syndrome_size:
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
    return int(leftover.commutes(PATCH.logical_z))

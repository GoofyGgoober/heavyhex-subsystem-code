"""Lookup decoder for the heavy-hex code: syndrome -> minimum-weight correction.

Success means the residual (correction * error) lies in the gauge group.
Syndrome bits follow SubsystemCode.stabilizers order: X stabilizers, then Z.
"""

from functools import cache

from ..core import Pauli
from ..core.subsystem import SubsystemCode
from ..patches.operators import D3

Syndrome = tuple[int, ...]


@cache
def _table(code: SubsystemCode) -> dict[Syndrome, Pauli]:
    """Minimum-weight representative per syndrome; only viable for d=3."""
    if code.syndrome_size > 8:
        raise ValueError(
            f"lookup over {1 << code.syndrome_size} syndromes is infeasible; "
            "larger distances need a matching decoder"
        )
    size = 1 << code.syndrome_size
    table: dict[Syndrome, Pauli] = {}
    for weight in range(code.n + 1):
        for error in code.paulis_of_weight(weight):
            table.setdefault(code.syndrome(error), error)
        if len(table) == size:
            return table
    raise RuntimeError(f"filled {len(table)} of {size} syndromes")


def decode(syndrome: Syndrome, code: SubsystemCode | None = None) -> Pauli:
    code = D3.code if code is None else code
    if len(syndrome) != code.syndrome_size or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be {code.syndrome_size} bits, got {syndrome!r}")
    return _table(code)[syndrome]


def residual_in_gauge_group(error: Pauli, code: SubsystemCode | None = None) -> bool:
    """True when lookup-correcting error leaves the logical factor untouched."""
    code = D3.code if code is None else code
    return code.in_gauge_group(decode(code.syndrome(error), code) * error)

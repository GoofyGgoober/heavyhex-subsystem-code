"""Lookup decoder: for every syndrome, the lightest error that produces it.

The table only fits at d=3. Syndrome bits follow the code's stabilizer order, X then Z.
"""

from functools import cache

from ..core import Pauli
from ..core.subsystem import SubsystemCode
from ..patches.operators import D3

Syndrome = tuple[int, ...]


@cache
def _table(code: SubsystemCode) -> dict[Syndrome, Pauli]:
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

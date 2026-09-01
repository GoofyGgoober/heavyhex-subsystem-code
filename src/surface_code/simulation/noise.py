"""Code-capacity depolarizing noise on data qubits."""

from __future__ import annotations

import random

from ..core import Pauli


def depolarizing_error(
    p: float,
    qubits: tuple[int, ...],
    rng: random.Random | None = None,
) -> Pauli:
    """Each qubit: I with probability 1-p, else X, Y, or Z with probability p/3."""
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if p == 0:
        return Pauli()
    rng = random.Random() if rng is None else rng
    x: set[int] = set()
    z: set[int] = set()
    for qubit in qubits:
        if rng.random() >= p:
            continue
        axis = rng.choice("XYZ")
        if axis in "XY":
            x.add(qubit)
        if axis in "YZ":
            z.add(qubit)
    return Pauli(frozenset(x), frozenset(z))

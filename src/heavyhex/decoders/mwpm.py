"""MWPM for X and Z memory at d=3 and d=5, under a simple noise model.

Data qubits flip independently before each round and before readout, and each
stabilizer value can come out wrong. There are no gate faults, so flags and
hook errors are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log, log1p
from typing import TYPE_CHECKING

from .._validation import validate_probability
from ..core import Pauli
from ..patches.operators import HeavyHexOperators
from .shots import Memory, Shot

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MemoryNoise:
    """Error rates behind the edge weights: data flip per round, wrong outcome, readout flip."""

    data: float = 0.01
    measurement: float = 0.01
    readout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("data", "measurement", "readout"):
            value = getattr(self, name)
            validate_probability(name, value)
            if value >= 0.5 or (name != "readout" and value == 0):
                raise ValueError(
                    f"{name} must be {'0 <= p' if name == 'readout' else '0 < p'} < 0.5"
                )

    @property
    def final_data(self) -> float:
        """A data flip or a readout flip before the last row, but not both."""
        return self.data + self.readout - 2 * self.data * self.readout


class MWPMDecoder:
    """Matching graph for one memory run. It only uses the memory-basis stabilizers."""

    def __init__(
        self,
        patch: HeavyHexOperators,
        *,
        basis: str = "Z",
        rounds: int = 1,
        noise: MemoryNoise = MemoryNoise(),
    ) -> None:
        if patch.distance not in (3, 5):
            raise ValueError("MWPM supports d=3 and d=5")
        self.memory = Memory(patch, basis, rounds)
        from pymatching import Matching

        self._matching = Matching()
        stabs = patch.x_stabilizers if basis == "X" else patch.z_stabilizers
        supports = tuple(frozenset(q - 1 for q in support) for support in stabs.values())
        # Data qubits that trip the same checks differ by a gauge, so they share
        # one edge: the chance that an odd number of them flipped.
        groups: dict[tuple[int, ...], list[int]] = {}
        for q in patch.data_qubits:
            endpoints = tuple(i for i, support in enumerate(supports) if q in support)
            if len(endpoints) not in (1, 2):
                raise ValueError("each data fault must produce one or two checks")
            groups.setdefault(endpoints, []).append(q)
        width = len(supports)
        for r in range(rounds + 1):
            p = noise.data if r < rounds else noise.final_data
            for endpoints, qubits in groups.items():
                logical_effects = {q in self.memory.logical_support for q in qubits}
                if len(logical_effects) != 1:
                    raise ValueError(
                        "parallel faults with different logical effects cannot be merged"
                    )
                # log((1-p_odd)/p_odd), p_odd=(1-(1-2p)**k)/2.
                odd_probability = -expm1(len(qubits) * log1p(-2 * p)) / 2
                weight = log1p(-odd_probability) - log(odd_probability)
                nodes = tuple(r * width + i for i in endpoints)
                if len(nodes) == 1:
                    self._matching.add_boundary_edge(nodes[0], fault_ids={qubits[0]}, weight=weight)
                else:
                    self._matching.add_edge(*nodes, fault_ids={qubits[0]}, weight=weight)
        weight = log1p(-noise.measurement) - log(noise.measurement)
        for r in range(rounds):
            for i in range(width):
                self._matching.add_edge(r * width + i, (r + 1) * width + i, weight=weight)
        self._matching.ensure_num_fault_ids(patch.code.n)

    def decode_batch(self, detectors: ArrayLike) -> NDArray[np.uint8]:
        """Detectors, one row per shot, to data-qubit correction masks."""
        import numpy as np

        values = np.asarray(detectors)
        width = len(self.memory.labels)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"detectors must have shape (shots, {width})")
        if values.dtype.kind not in "biu" or not np.all((values == 0) | (values == 1)):
            raise ValueError("detectors must be binary integers")
        if len(values) == 0:
            return np.empty((0, self.memory.patch.code.n), dtype=np.uint8)
        used = np.ascontiguousarray(values[:, self.memory.basis_indices], dtype=np.uint8)
        return self._matching.decode_batch(used)

    def decode(self, detectors: tuple[int, ...]) -> Pauli:
        mask = self.decode_batch([detectors])[0]
        support = frozenset(q for q, bit in enumerate(mask) if bit)
        return Pauli.x_on(support) if self.memory.basis == "Z" else Pauli.z_on(support)

    def fails(self, shot: Shot) -> bool:
        """True if MWPM's correction leaves the wrong logical value."""
        if shot.memory != self.memory:
            raise ValueError("shot is from a different distance, basis or number of rounds")
        return shot.logical_error(self.decode(shot.detectors))

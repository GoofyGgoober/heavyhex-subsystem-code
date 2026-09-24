"""Sample the MWPM decoder's own noise model directly, with no circuit."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

from .._validation import validate_shots_and_seed
from ..decoders.mwpm import MemoryNoise, MWPMDecoder
from ..decoders.shots import Memory
from ..patches.operators import HeavyHexOperators

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MemorySamples:
    memory: Memory
    detectors: NDArray[np.uint8]
    data_bits: NDArray[np.uint8]
    logical_flips: NDArray[np.uint8]


def sample_memory(
    patch: HeavyHexOperators,
    *,
    basis: str = "Z",
    rounds: int = 3,
    shots: int = 1024,
    seed: int | None = None,
    noise: MemoryNoise = MemoryNoise(),
) -> MemorySamples:
    """Draw random flips from the noise model and return the detectors.

    Only memory-basis stabilizers are simulated, so the other detectors stay 0.
    """
    import numpy as np

    validate_shots_and_seed(shots, seed)
    memory = Memory(patch, basis, rounds)
    rng = np.random.default_rng(seed)
    stabs = patch.x_stabilizers if basis == "X" else patch.z_stabilizers
    matrix = np.array(
        [[q + 1 in support for q in patch.data_qubits] for support in stabs.values()],
        dtype=np.uint8,
    )
    data = np.zeros((shots, patch.code.n), dtype=np.uint8)
    previous = np.zeros((shots, len(stabs)), dtype=np.uint8)
    detectors = np.zeros((shots, len(memory.labels)), dtype=np.uint8)
    indices = np.array(memory.basis_indices).reshape(rounds + 1, len(stabs))
    for r in range(rounds + 1):
        data ^= (rng.random(data.shape) < noise.data).astype(np.uint8)
        if r == rounds:
            data ^= (rng.random(data.shape) < noise.readout).astype(np.uint8)
        checks = (data @ matrix.T) % 2
        if r < rounds:
            checks ^= (rng.random(checks.shape) < noise.measurement).astype(np.uint8)
        detectors[:, indices[r]] = checks ^ previous
        previous = checks
    flips = data[:, sorted(memory.logical_support)].sum(axis=1).astype(np.uint8) % 2
    return MemorySamples(memory, detectors, data, flips)


def benchmark_memory(
    patch: HeavyHexOperators,
    *,
    basis: str = "Z",
    rounds: int = 3,
    shots: int = 1024,
    seed: int | None = None,
    noise: MemoryNoise = MemoryNoise(),
) -> dict:
    """Decode sampled shots and count logical failures, with a 95% interval."""
    import numpy as np
    import pymatching

    samples = sample_memory(patch, basis=basis, rounds=rounds, shots=shots, seed=seed, noise=noise)
    decoder = MWPMDecoder(patch, basis=basis, rounds=rounds, noise=noise)
    corrections = decoder.decode_batch(samples.detectors)
    predictions = corrections[:, sorted(samples.memory.logical_support)].sum(axis=1) % 2
    failures = int(np.count_nonzero(predictions != samples.logical_flips))
    # Wilson score interval.
    z = 1.959963984540054
    rate = failures / shots
    denominator = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / denominator
    radius = z * sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots)) / denominator
    return {
        "model": "phenomenological_independent_stabilizer_errors",
        "distance": patch.distance,
        "basis": basis,
        "rounds": rounds,
        "shots": shots,
        "seed": seed,
        "noise": {"data": noise.data, "measurement": noise.measurement, "readout": noise.readout},
        "raw_logical_flips": int(np.count_nonzero(samples.logical_flips)),
        "logical_failures": failures,
        "logical_error_rate": rate,
        "logical_error_rate_95ci": [max(0.0, center - radius), min(1.0, center + radius)],
        "pymatching_version": pymatching.__version__,
        "numpy_version": np.__version__,
    }

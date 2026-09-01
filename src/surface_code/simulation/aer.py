"""Qiskit Aer backend: translate the patch circuit, run shots, parse counts.

Qiskit is imported only when you call these functions so the rest of the
package still runs without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..circuits.operations import CX, H, MeasureZ
from ..circuits.syndrome import SYNDROME_CIRCUIT
from ..core import Pauli
from ..decoders import decode
from ..patches import PATCH

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

NUM_QUBITS = max(PATCH.data_qubits + PATCH.ancillas) + 1
MAX_SEED = (1 << 63) - 1


def _pauli_label(pauli: Pauli, n: int) -> str:
    """Qiskit Pauli string: leftmost char is the highest-index qubit."""
    chars = ["I"] * n
    for qubit in pauli.x:
        chars[qubit] = "Y" if qubit in pauli.z else "X"
    for qubit in pauli.z:
        if qubit not in pauli.x:
            chars[qubit] = "Z"
    return "".join(reversed(chars))


def _logical_zero_circuit():
    """Clifford that takes |0>^9 to |0>_L (stabilizers and Z_L all +1)."""
    from qiskit.quantum_info import StabilizerState

    generators = [_pauli_label(s, len(PATCH.data_qubits)) for s in PATCH.stabilizers]
    generators.append(_pauli_label(PATCH.logical_z, len(PATCH.data_qubits)))
    return StabilizerState.from_stabilizer_list(generators).clifford.to_circuit()


def _apply_error(circuit: QuantumCircuit, error: Pauli) -> None:
    PATCH.code.validate_data_pauli(error, name="error")
    for qubit in error.x - error.z:
        circuit.x(qubit)
    for qubit in error.z - error.x:
        circuit.z(qubit)
    for qubit in error.x & error.z:
        circuit.y(qubit)


def to_qiskit(error: Pauli | None = None) -> QuantumCircuit:
    """Syndrome round, then Z-measure the 9 data qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(NUM_QUBITS, "q")
    syn = ClassicalRegister(len(PATCH.ancillas), "syn")
    data = ClassicalRegister(len(PATCH.data_qubits), "data")
    circuit = QuantumCircuit(qubits, syn, data)
    circuit.compose(_logical_zero_circuit(), PATCH.data_qubits, inplace=True)
    if error is not None:
        _apply_error(circuit, error)

    ancilla_bit = {ancilla: i for i, ancilla in enumerate(PATCH.ancillas)}
    for op in SYNDROME_CIRCUIT:
        if isinstance(op, H):
            circuit.h(op.qubit)
        elif isinstance(op, CX):
            circuit.cx(op.control, op.target)
        else:
            circuit.measure(op.qubit, syn[ancilla_bit[op.qubit]])

    # Data is still entangled with the ancillas until those measures finish.
    circuit.barrier()
    for i, qubit in enumerate(PATCH.data_qubits):
        circuit.measure(qubit, data[i])
    return circuit


def _bits_le(bitstring: str) -> tuple[int, ...]:
    """Qiskit prints a register with index 0 on the right."""
    return tuple(int(bit) for bit in bitstring[::-1])


def _parse_shot(key: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # get_counts prints the last-added register first: "data syn".
    data_str, syn_str = key.split()
    if len(data_str) != len(PATCH.data_qubits):
        data_str, syn_str = syn_str, data_str
    return _bits_le(syn_str), _bits_le(data_str)


def run_aer(
    error: Pauli | None = None,
    *,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], int]:
    """Return {(syndrome, data_bits): count} from a noiseless stabilizer sim."""
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed > MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, or None, got {seed!r}")

    from qiskit_aer import AerSimulator

    circuit = to_qiskit(error)
    run_options: dict[str, int] = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed
    result = AerSimulator(method="stabilizer").run(circuit, **run_options).result()
    tallies: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for key, count in result.get_counts().items():
        parsed = _parse_shot(key)
        tallies[parsed] = tallies.get(parsed, 0) + count
    return tallies


def shot_z_success(syndrome: tuple[int, ...], data_bits: tuple[int, ...]) -> int:
    """Hardware-shaped Z success: correct the data bits, then test Z_L == 0."""
    correction = decode(syndrome)
    bits = list(data_bits)
    for qubit in correction.x:
        bits[qubit] ^= 1
    return int(sum(bits[qubit] for qubit in PATCH.logical_z.z) % 2 == 0)

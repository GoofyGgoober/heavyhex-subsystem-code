"""Qiskit Aer backend: translate the patch circuit, run shots, parse counts.

Qiskit is imported only when you call these functions so the rest of the
package still runs without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._validation import validate_binary_bits
from ..circuits.operations import CX, H
from ..circuits.syndrome import SYNDROME_CIRCUIT
from ..core import Pauli
from ..decoders import decode
from ..decoders.basis import normalize_measurement_basis
from ..patches import PATCH

if TYPE_CHECKING:
    from qiskit import ClassicalRegister, QuantumCircuit

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


def _logical_state_circuit(basis: str = "Z") -> QuantumCircuit:
    """Prepare the +1 logical eigenstate for ``basis`` from ``|0>^9``."""
    from qiskit.quantum_info import StabilizerState

    basis = normalize_measurement_basis(basis)
    generators = [_pauli_label(s, len(PATCH.data_qubits)) for s in PATCH.stabilizers]
    logical = PATCH.logical_x if basis == "X" else PATCH.logical_z
    generators.append(_pauli_label(logical, len(PATCH.data_qubits)))
    return StabilizerState.from_stabilizer_list(generators).clifford.to_circuit()


def _apply_error(circuit: QuantumCircuit, error: Pauli) -> None:
    PATCH.code.validate_data_pauli(error, name="error")
    for qubit in error.x - error.z:
        circuit.x(qubit)
    for qubit in error.z - error.x:
        circuit.z(qubit)
    for qubit in error.x & error.z:
        circuit.y(qubit)


def _append_syndrome_round(circuit: QuantumCircuit, syn: ClassicalRegister) -> None:
    """Append one check-extraction round into the supplied classical register."""
    ancilla_bit = {ancilla: i for i, ancilla in enumerate(PATCH.ancillas)}
    for op in SYNDROME_CIRCUIT:
        if isinstance(op, H):
            circuit.h(op.qubit)
        elif isinstance(op, CX):
            circuit.cx(op.control, op.target)
        else:
            circuit.measure(op.qubit, syn[ancilla_bit[op.qubit]])


def to_qiskit(error: Pauli | None = None) -> QuantumCircuit:
    """Syndrome round, then Z-measure the 9 data qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(NUM_QUBITS, "q")
    syn = ClassicalRegister(len(PATCH.ancillas), "syn")
    data = ClassicalRegister(len(PATCH.data_qubits), "data")
    circuit = QuantumCircuit(qubits, syn, data)
    circuit.compose(_logical_state_circuit("Z"), PATCH.data_qubits, inplace=True)
    if error is not None:
        _apply_error(circuit, error)

    _append_syndrome_round(circuit, syn)

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
    syndrome, data_bits = _bits_le(syn_str), _bits_le(data_str)
    validate_binary_bits("syndrome", syndrome, len(PATCH.ancillas))
    validate_binary_bits("data readout", data_bits, len(PATCH.data_qubits))
    return syndrome, data_bits


def _validate_aer_options(shots: int, seed: int | None) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, or None, got {seed!r}")


def run_aer(
    error: Pauli | None = None,
    *,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], int]:
    """Return {(syndrome, data_bits): count} from a noiseless stabilizer sim."""
    _validate_aer_options(shots, seed)

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
    validate_binary_bits("data readout", data_bits, len(PATCH.data_qubits))
    correction = decode(syndrome)
    bits = list(data_bits)
    for qubit in correction.x:
        bits[qubit] ^= 1
    return int(sum(bits[qubit] for qubit in PATCH.logical_z.z) % 2 == 0)

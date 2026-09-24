"""Flagged d=3 and d=5 memory circuits as they would run on the chip.

Qubits are reset and reused: 23 at d=3, 65 at d=5, laid out as in Sundaresan et
al. 2023, Fig. 4. The X ancillas double as flags for the Z4 gauges, and the
boundary Z2 gauges reach their data through relays. Flags and relays are undone
after use, so they read 0 unless a fault hit them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import TYPE_CHECKING

from .._validation import validate_binary_bits
from ..core import Pauli
from ..patches.operators import D3, HeavyHexOperators
from .ideal import _apply_pauli, _checks, _halves

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


# Gate order of a Z4 gadget, as (arm, step): step 0 or 1 is a CX from that data
# qubit of the arm's pair onto its flag, "pass" a CX from the flag onto the ancilla.
# Each flag passes its data qubits one at a time and the two flags take turns,
# so a single Z fault on a flag or the ancilla spreads to at most one data qubit,
# or two in the same column. Undone this way, every flag ends back at 0.
Z4_GATES = (
    (0, 0), (0, "pass"), (1, 0), (1, "pass"),
    (0, 0), (0, 1), (0, "pass"), (1, 0), (1, 1), (1, "pass"),
    (0, 1), (1, 1),
)  # fmt: skip


@dataclass(frozen=True)
class Roles:
    """Circuit qubit indices; where they sit on the chip is a separate mapping."""

    num_data: int
    num_qubits: int
    x_ancillas: dict[str, int]
    z_ancillas: dict[str, int]
    z4_arms: dict[str, tuple[tuple[tuple[int, int], str], ...]]  # ((data pair), flag gauge)
    z2_arms: dict[str, tuple[tuple[int, int], ...]]  # (data, relay)


def qubit_roles(patch: HeavyHexOperators = D3) -> Roles:
    """Number the qubits: data, then X ancillas, Z ancillas and relays.

    Each Z4 gauge borrows the X ancillas of the two rows it spans as flags, upper row first.
    """
    if patch.distance not in (3, 5):
        raise ValueError("flagged circuits support d=3 and d=5 only")
    n = patch.distance**2
    xs = sorted(patch.x_gauges, key=lambda name: patch.x_gauges[name])
    zs = sorted(patch.z_gauges, key=lambda name: patch.z_gauges[name])
    x_ancillas = {name: n + i for i, name in enumerate(xs)}
    z_ancillas = {name: n + len(xs) + i for i, name in enumerate(zs)}
    by_support = {tuple(sorted(support)): name for name, support in patch.x_gauges.items()}
    z4, z2 = {}, {}
    next_qubit = n + len(xs) + len(zs)
    for name in zs:
        support = sorted(patch.z_gauges[name])
        if len(support) == 2:
            z2[name] = tuple((q - 1, next_qubit + i) for i, q in enumerate(support))
            next_qubit += 2
        else:
            a, b, c, d = support
            z4[name] = tuple(
                (tuple(q - 1 for q in pair), by_support[pair]) for pair in ((a, c), (b, d))
            )
    return Roles(n, next_qubit, x_ancillas, z_ancillas, z4, z2)


@dataclass(frozen=True)
class Measurement:
    round: int  # -1 for the prep half
    half: str  # "X" or "Z"
    kind: str  # "z_syn", "z_flag", "relay", "x_gauge"
    gauge: str
    bit: int


@dataclass(frozen=True)
class Gadget:
    """The gates that measure one gauge: circuit.data[start:end]."""

    round: int
    half: str
    gauge: str
    start: int
    end: int


@dataclass(frozen=True)
class FlaggedSchedule:
    patch: HeavyHexOperators
    basis: str
    rounds: int
    measurements: tuple[Measurement, ...]
    gadgets: tuple[Gadget, ...]
    half_order: tuple[tuple[int, str], ...]

    @cached_property
    def roles(self) -> Roles:
        return qubit_roles(self.patch)

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        """Each stabilizer's value per round, the XOR of its gauge outcomes."""
        validate_binary_bits("gauge outcomes", gauge_bits, len(self.measurements))
        outcomes = {
            (m.round, m.half, m.gauge): gauge_bits[m.bit]
            for m in self.measurements
            if m.kind in ("z_syn", "x_gauge")
        }
        return _checks(self.patch, outcomes)


def memory_circuit_flagged(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, FlaggedSchedule]:
    """Flagged memory on 23 (d=3) or 65 (d=5) reused qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    roles = qubit_roles(patch)
    prep_half, halves = _halves(patch, rounds, basis, error, inject_at)

    total = 3 * len(roles.z_ancillas) * (rounds + (prep_half == "Z")) + len(roles.x_ancillas) * (
        rounds + (prep_half == "X")
    )
    qubits = QuantumRegister(roles.num_qubits, "q")
    gauge = ClassicalRegister(total, "m")
    data = ClassicalRegister(roles.num_data, "d")
    circuit = QuantumCircuit(qubits, gauge, data)

    if basis == "X":
        circuit.h(list(range(roles.num_data)))

    measurements: list[Measurement] = []
    gadgets: list[Gadget] = []
    half_order: list[tuple[int, str]] = []
    bit = 0

    def record(round: int, half: str, kind: str, gauge_name: str, qubit: int) -> None:
        nonlocal bit
        circuit.measure(qubit, gauge[bit])
        measurements.append(Measurement(round, half, kind, gauge_name, bit))
        bit += 1

    def measure_z4(round: int, name: str) -> None:
        # Each data pair's parity goes through its flag to the ancilla, in the
        # order Z4_GATES gives. The flags read 0 unless something went wrong.
        (pair_a, gauge_a), (pair_b, gauge_b) = roles.z4_arms[name]
        flag_a, flag_b = roles.x_ancillas[gauge_a], roles.x_ancillas[gauge_b]
        ancilla = roles.z_ancillas[name]
        for qubit in (ancilla, flag_a, flag_b):
            circuit.reset(qubit)
        arms = ((pair_a, flag_a), (pair_b, flag_b))
        for arm, step in Z4_GATES:
            pair, flag = arms[arm]
            if step == "pass":
                circuit.cx(flag, ancilla)
            else:
                circuit.cx(pair[step], flag)
        record(round, "Z", "z_flag", gauge_a, flag_a)
        record(round, "Z", "z_flag", gauge_b, flag_b)
        record(round, "Z", "z_syn", name, ancilla)

    def measure_z2(round: int, name: str) -> None:
        # Each data qubit reaches the ancilla through a relay, which is then reset to |0>.
        (data_a, relay_a), (data_b, relay_b) = roles.z2_arms[name]
        ancilla = roles.z_ancillas[name]
        for qubit in (ancilla, relay_a, relay_b):
            circuit.reset(qubit)
        for qubit, relay in ((data_a, relay_a), (data_b, relay_b)):
            circuit.cx(qubit, relay)
            circuit.cx(relay, ancilla)
            circuit.cx(qubit, relay)
        record(round, "Z", "relay", name, relay_a)
        record(round, "Z", "relay", name, relay_b)
        record(round, "Z", "z_syn", name, ancilla)

    def measure_x2(round: int, name: str) -> None:
        ancilla = roles.x_ancillas[name]
        circuit.reset(ancilla)
        circuit.h(ancilla)
        for label in patch.x_gauges[name]:
            circuit.cx(ancilla, label - 1)
        circuit.h(ancilla)
        record(round, "X", "x_gauge", name, ancilla)

    def run_half(round: int, half: str) -> None:
        half_order.append((round, half))
        for name in roles.z_ancillas if half == "Z" else roles.x_ancillas:
            start = len(circuit.data)
            if half == "X":
                measure_x2(round, name)
            elif name in roles.z4_arms:
                measure_z4(round, name)
            else:
                measure_z2(round, name)
            gadgets.append(Gadget(round, half, name, start, len(circuit.data)))

    run_half(-1, prep_half)
    circuit.barrier()
    if inject_at == "after_prep":
        _apply_pauli(circuit, error)
    for r in range(rounds):
        for half in halves:
            run_half(r, half)
            circuit.barrier()
            if inject_at == f"after_{half.lower()}{r}":
                _apply_pauli(circuit, error)

    if basis == "X":
        circuit.h(list(range(roles.num_data)))
    for i in range(roles.num_data):
        circuit.measure(i, data[i])

    return circuit, FlaggedSchedule(
        patch, basis, rounds, tuple(measurements), tuple(gadgets), tuple(half_order)
    )


def with_fault(circuit: QuantumCircuit, index: int, fault: dict[int, str]) -> QuantumCircuit:
    """Copy of the circuit with `fault` inserted after instruction `index` (-1 puts it first)."""
    from qiskit import QuantumCircuit

    if any(axis not in ("X", "Y", "Z") for axis in fault.values()):
        raise ValueError(f"fault Paulis must be X, Y or Z, got {fault!r}")
    new = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    def append_fault() -> None:
        for qubit, axis in fault.items():
            getattr(new, axis.lower())(new.qubits[qubit])

    if index == -1:
        append_fault()
    for position, instruction in enumerate(circuit.data):
        new.append(instruction.operation, instruction.qubits, instruction.clbits)
        if position == index:
            append_fault()
    return new


def single_faults(circuit: QuantumCircuit, gadget: Gadget) -> Iterator[tuple[int, dict[int, str]]]:
    """Every single fault in a gadget, as (index it goes after, fault).

    X, Y or Z after an H, X after a reset, any of the 15 two-qubit Paulis after
    a CX, and X just before a measurement.
    """
    for index in range(gadget.start, gadget.end):
        name, qubits = _instruction(circuit, index)
        if name == "h":
            (qubit,) = qubits
            for axis in ("X", "Y", "Z"):
                yield index, {qubit: axis}
        elif name == "reset":
            (qubit,) = qubits
            yield index, {qubit: "X"}
        elif name == "cx":
            control, target = qubits
            for axes in product("IXYZ", repeat=2):
                if axes == ("I", "I"):
                    continue
                yield (
                    index,
                    {qubit: axis for qubit, axis in zip((control, target), axes) if axis != "I"},
                )
        elif name == "measure":
            (qubit,) = qubits
            yield index - 1, {qubit: "X"}
        else:  # pragma: no cover - gadgets contain only the above
            raise ValueError(f"unexpected instruction in gadget: {name}")


def propagate_fault(
    circuit: QuantumCircuit,
    gadget: Gadget,
    fault_index: int,
    fault: dict[int, str],
) -> tuple[Pauli, frozenset[int]]:
    """Push a fault through the rest of its gadget.

    Returns the Pauli left on the data and the measured qubits whose outcome
    flipped. Signs are ignored, and whatever is left on ancillas is dropped
    because they are reset before reuse.
    """
    x_set: set[int] = set()
    z_set: set[int] = set()
    for qubit, axis in fault.items():
        if axis in ("X", "Y"):
            x_set.add(qubit)
        if axis in ("Z", "Y"):
            z_set.add(qubit)
    flipped: set[int] = set()
    for position in range(fault_index + 1, gadget.end):
        name, qubits = _instruction(circuit, position)
        if name == "h":
            (qubit,) = qubits
            in_x, in_z = qubit in x_set, qubit in z_set
            x_set.difference_update((qubit,))
            z_set.difference_update((qubit,))
            if in_z:
                x_set.add(qubit)
            if in_x:
                z_set.add(qubit)
        elif name == "cx":
            control, target = qubits
            if control in x_set:
                x_set.symmetric_difference_update((target,))
            if target in z_set:
                z_set.symmetric_difference_update((control,))
        elif name == "reset":
            (qubit,) = qubits
            x_set.difference_update((qubit,))
            z_set.difference_update((qubit,))
        elif name == "measure":
            (qubit,) = qubits
            if qubit in x_set:
                flipped.add(qubit)
            x_set.difference_update((qubit,))
            z_set.difference_update((qubit,))
        else:  # pragma: no cover - gadgets contain only the above
            raise ValueError(f"unexpected instruction in gadget: {name}")
    # Data qubits come first, one per bit of the "d" register.
    data_register = next(register for register in circuit.cregs if register.name == "d")
    data = set(range(len(data_register)))
    return Pauli(frozenset(x_set & data), frozenset(z_set & data)), frozenset(flipped)


def _instruction(circuit: QuantumCircuit, position: int) -> tuple[str, list[int]]:
    item = circuit.data[position]
    return item.operation.name, [circuit.find_bit(qubit).index for qubit in item.qubits]

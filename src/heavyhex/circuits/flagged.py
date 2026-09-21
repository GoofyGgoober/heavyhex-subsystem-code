"""Device-faithful flagged d=3 heavy-hex circuits (Sundaresan et al. 2023, Fig. 4).

23 qubits with reset reuse: 9 data + 6 in-row flags (X gauges) + 4 Z syndromes +
4 boundary relays. Flags catch hook errors in the weight-4 Z gadgets; the CX
orders below are verified by single-fault enumeration in the test suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from ..core import Pauli
from ..patches.operators import D3, HeavyHexOperators
from .ideal import _apply_pauli, _half_order, _stabilizer_checks

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

NUM_DATA = 9
NUM_QUBITS = 23

# Abstract role indices: data 0..8 (id = label - 1), then flags, syndromes, relays.
FLAG_OF_X = {"X1X4": 9, "X2X5": 10, "X3X6": 11, "X4X7": 12, "X5X8": 13, "X6X9": 14}
SYN_OF_Z = {"Z1Z2": 15, "Z2Z3Z5Z6": 16, "Z4Z5Z7Z8": 17, "Z8Z9": 18}
# Weight-4 Z gauge -> two arms of (data pair, flag gauge); each flag sits between its pair.
Z4_ARMS = {
    "Z4Z5Z7Z8": (((3, 6), "X4X7"), ((4, 7), "X5X8")),
    "Z2Z3Z5Z6": (((1, 4), "X2X5"), ((2, 5), "X3X6")),
}
# Weight-2 Z gauge -> two arms of (data id, relay index).
Z2_ARMS = {"Z1Z2": ((0, 19), (1, 20)), "Z8Z9": ((7, 21), (8, 22))}
# Order gadgets run in within a half (not the patch's gauge order).
Z_HALF_ORDER = ("Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9")
X_HALF_ORDER = ("X1X4", "X2X5", "X3X6", "X4X7", "X5X8", "X6X9")
FLAGS_OF_Z4 = {"Z2Z3Z5Z6": ("X2X5", "X3X6"), "Z4Z5Z7Z8": ("X4X7", "X5X8")}
# Deflagging (paper rule, translated): flag gauge -> data id taking the virtual
# Z correction when exactly that flag fires within its weight-4 Z gauge.
DEFLAG = {"X2X5": 1, "X3X6": 5, "X4X7": 3, "X5X8": 7}
# Abstract index -> Falcon-27 device id (Fig. 4a); fez ids via docs/figures/d3_fez_layout.json.
FALCON = {
    0: 2,
    1: 10,
    2: 17,
    3: 5,
    4: 13,
    5: 21,
    6: 9,
    7: 16,
    8: 24,
    9: 3,
    10: 12,
    11: 18,
    12: 8,
    13: 14,
    14: 23,
    15: 4,
    16: 15,
    17: 11,
    18: 22,
    19: 1,
    20: 7,
    21: 19,
    22: 25,
}

# The flag's other data qubit: back-action is Z_a Z_b, DEFLAG takes one factor.
_RESIDUAL_OF_FLAG = {
    flag: next(q for q in pair if q != DEFLAG[flag])
    for arms in Z4_ARMS.values()
    for pair, flag in arms
}
_Z_HALF_BITS = 3 * len(Z_HALF_ORDER)  # two flags or relays + one syndrome per gauge
_X_HALF_BITS = len(X_HALF_ORDER)


@dataclass(frozen=True)
class FlaggedMeasurement:
    round: int  # -1 for the prep half
    half: str  # "X" or "Z"
    kind: str  # "z_syn", "z_flag", "relay", "x_gauge"
    gauge: str
    bit: int


@dataclass(frozen=True)
class Fragment:
    """One gadget's gate range [start, end) into circuit.data; barriers sit outside."""

    name: str
    gauge: str
    half: str
    start: int
    end: int


@dataclass(frozen=True)
class FlaggedSchedule:
    patch: HeavyHexOperators
    basis: str
    rounds: int
    measurements: tuple[FlaggedMeasurement, ...]
    fragments: tuple[Fragment, ...]
    half_order: tuple[tuple[int, str], ...]

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        # Relay outcomes share their gauge's key and are overwritten by the syndrome bit.
        outcomes = {(m.round, m.half, m.gauge): gauge_bits[m.bit] for m in self.measurements}
        return _stabilizer_checks(self.patch, outcomes)

    def deflag_corrections(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Virtual Z corrections for lone flags, tagged by (round, half) of their Z half."""
        return [
            ((round, "Z"), Pauli.z_on((DEFLAG[flag],)))
            for round, flag in self._lone_flags(gauge_bits)
        ]

    def backaction_residuals(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Weight-1 Z left on the flag's other data qubit once DEFLAG cancels one factor.

        Deterministic given the flags, so single-shot decoding subtracts them
        instead of handing the lookup a weight-2 error.
        """
        return [
            ((round, "Z"), Pauli.z_on((_RESIDUAL_OF_FLAG[flag],)))
            for round, flag in self._lone_flags(gauge_bits)
        ]

    def _lone_flags(self, gauge_bits: tuple[int, ...]) -> list[tuple[int, str]]:
        """(round, flag gauge) per weight-4 Z gauge where exactly one flag fired.

        Both flags firing leaves the full gauge operator, which is harmless.
        """
        bits = {
            (m.round, m.gauge): gauge_bits[m.bit] for m in self.measurements if m.kind == "z_flag"
        }
        rounds = sorted({m.round for m in self.measurements if m.half == "Z"})
        return [
            (round, first if bits[round, first] else second)
            for round in rounds
            for first, second in FLAGS_OF_Z4.values()
            if bool(bits[round, first]) != bool(bits[round, second])
        ]


def memory_circuit_flagged(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, FlaggedSchedule]:
    """Flagged memory circuit on 23 reused qubits; same shape as ideal.memory_circuit."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if patch.distance != 3:
        raise ValueError("flagged gadgets are implemented for d=3 only")
    prep_half, halves = _half_order(patch, rounds, basis, error, inject_at)

    total = _Z_HALF_BITS * (rounds + (prep_half == "Z")) + _X_HALF_BITS * (
        rounds + (prep_half == "X")
    )
    qubits = QuantumRegister(NUM_QUBITS, "q")
    gauge = ClassicalRegister(total, "m")
    data = ClassicalRegister(NUM_DATA, "d")
    circuit = QuantumCircuit(qubits, gauge, data)

    if basis == "X":
        circuit.h(list(range(NUM_DATA)))

    measurements: list[FlaggedMeasurement] = []
    fragments: list[Fragment] = []
    half_order: list[tuple[int, str]] = []
    bit = 0

    def record(round: int, half: str, kind: str, gauge_name: str, qubit: int) -> None:
        nonlocal bit
        circuit.measure(qubit, gauge[bit])
        measurements.append(FlaggedMeasurement(round, half, kind, gauge_name, bit))
        bit += 1

    def measure_z4(round: int, name: str) -> None:
        (pair_a, gauge_a), (pair_b, gauge_b) = Z4_ARMS[name]
        syndrome = SYN_OF_Z[name]
        flag_a, flag_b = _z4_fragment(
            circuit, ((pair_a, FLAG_OF_X[gauge_a]), (pair_b, FLAG_OF_X[gauge_b])), syndrome
        )
        record(round, "Z", "z_flag", gauge_a, flag_a)
        record(round, "Z", "z_flag", gauge_b, flag_b)
        record(round, "Z", "z_syn", name, syndrome)

    def measure_z2(round: int, name: str) -> None:
        syndrome = SYN_OF_Z[name]
        relay_a, relay_b = _z2_fragment(circuit, Z2_ARMS[name], syndrome)
        record(round, "Z", "relay", name, relay_a)
        record(round, "Z", "relay", name, relay_b)
        record(round, "Z", "z_syn", name, syndrome)

    def measure_x2(round: int, name: str) -> None:
        first, second = (q - 1 for q in patch.x_gauges[name])
        _x2_fragment(circuit, (first, second), FLAG_OF_X[name])
        record(round, "X", "x_gauge", name, FLAG_OF_X[name])

    def run_half(round: int, half: str) -> None:
        half_order.append((round, half))
        for name in Z_HALF_ORDER if half == "Z" else X_HALF_ORDER:
            start = len(circuit.data)
            if half == "X":
                measure_x2(round, name)
            elif name in Z4_ARMS:
                measure_z4(round, name)
            else:
                measure_z2(round, name)
            fragments.append(
                Fragment(f"{half}:{name}@{round}", name, half, start, len(circuit.data))
            )

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
        circuit.h(list(range(NUM_DATA)))
    for i in range(NUM_DATA):
        circuit.measure(i, data[i])

    return circuit, FlaggedSchedule(
        patch, basis, rounds, tuple(measurements), tuple(fragments), tuple(half_order)
    )


def with_fault(circuit: QuantumCircuit, index: int, fault: dict[int, str]) -> QuantumCircuit:
    """Copy of circuit with a Pauli fault inserted after instruction index (-1: before all)."""
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


def single_faults(
    circuit: QuantumCircuit, fragment: Fragment
) -> Iterator[tuple[int, dict[int, str]]]:
    """Yield (insert-after index, fault) for every single fault inside a fragment.

    After each 1Q gate: X, Y, Z. After each reset: X. After each 2Q gate: the 15
    nontrivial Paulis. Before each measurement: X.
    """
    for index in range(fragment.start, fragment.end):
        name, qubits = _instruction(circuit, index)
        if name in ("h", "x", "y", "z"):
            (qubit,) = qubits
            for axis in ("X", "Y", "Z"):
                yield index, {qubit: axis}
        elif name == "reset":
            (qubit,) = qubits
            yield index, {qubit: "X"}
        elif name in ("cx", "cz"):
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
        else:  # pragma: no cover - fragments contain only the above
            raise ValueError(f"unexpected instruction in fragment: {name}")


def propagate_fault(
    circuit: QuantumCircuit,
    fragment: Fragment,
    fault_index: int,
    fault: dict[int, str],
) -> tuple[Pauli, frozenset[int]]:
    """Propagate one fault through the fragment gates (fault_index, fragment.end).

    Returns the data-qubit Pauli leaving the fragment and the measured qubits
    whose outcome flips. Phase-free; reset wipes errors; ancilla residue is
    dropped (ancillas are reset before every reuse).
    """
    x_set: set[int] = set()
    z_set: set[int] = set()
    for qubit, axis in fault.items():
        if axis in ("X", "Y"):
            x_set.add(qubit)
        if axis in ("Z", "Y"):
            z_set.add(qubit)
    flipped: set[int] = set()
    for position in range(fault_index + 1, fragment.end):
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
        elif name == "cz":
            control, target = qubits
            if control in x_set:
                z_set.symmetric_difference_update((target,))
            if target in x_set:
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
        else:  # pragma: no cover - fragments contain only the above
            raise ValueError(f"unexpected instruction in fragment: {name}")
    data = set(range(NUM_DATA))
    return Pauli(frozenset(x_set & data), frozenset(z_set & data)), frozenset(flipped)


def _x2_fragment(circuit: QuantumCircuit, data: tuple[int, int], flag: int) -> None:
    """Weight-2 X gauge measured directly on its flag qubit, CXs in ascending data order."""
    circuit.reset(flag)
    circuit.h(flag)
    circuit.cx(flag, data[0])
    circuit.cx(flag, data[1])
    circuit.h(flag)


def _z4_fragment(
    circuit: QuantumCircuit,
    arms: tuple[tuple[tuple[int, int], int], tuple[tuple[int, int], int]],
    syndrome: int,
) -> tuple[int, int]:
    """Flagged weight-4 Z gauge; returns the two flag qubits in arm order."""
    (pair_a, flag_a), (pair_b, flag_b) = arms
    circuit.reset(syndrome)
    circuit.reset(flag_a)
    circuit.reset(flag_b)
    circuit.cx(pair_a[0], flag_a)
    circuit.cx(pair_a[1], flag_a)
    circuit.cx(flag_a, syndrome)
    circuit.cx(pair_b[0], flag_b)
    circuit.cx(pair_b[1], flag_b)
    circuit.cx(flag_b, syndrome)
    circuit.h(flag_a)
    circuit.h(flag_b)
    return flag_a, flag_b


def _z2_fragment(
    circuit: QuantumCircuit, arms: tuple[tuple[int, int], tuple[int, int]], syndrome: int
) -> tuple[int, int]:
    """Weight-2 Z gauge relayed through two qubits that uncompute back to |0>."""
    (data_a, relay_a), (data_b, relay_b) = arms
    circuit.reset(syndrome)
    circuit.reset(relay_a)
    circuit.reset(relay_b)
    circuit.cx(data_a, relay_a)
    circuit.cx(relay_a, syndrome)
    circuit.cx(data_a, relay_a)
    circuit.cx(data_b, relay_b)
    circuit.cx(relay_b, syndrome)
    circuit.cx(data_b, relay_b)
    return relay_a, relay_b


def _instruction(circuit: QuantumCircuit, position: int) -> tuple[str, list[int]]:
    item = circuit.data[position]
    return item.operation.name, [circuit.find_bit(qubit).index for qubit in item.qubits]

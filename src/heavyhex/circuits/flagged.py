"""Device-faithful flagged d=3 and d=5 heavy-hex circuits (Sundaresan et al. 2023, Fig. 4).

Reset reuse needs 23 qubits at d=3 and 65 at d=5: data, in-row X-gauge
ancillas (also Z-gadget flags), Z syndromes, and boundary relays. Gadget-local
CX orders are checked by single-fault enumeration; full circuit-level distance
with noisy decoding remains to be validated.
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


@dataclass(frozen=True)
class FlaggedRoles:
    """Abstract circuit indices, independent of the physical device placement."""

    num_data: int
    num_qubits: int
    flag_of_x: dict[str, int]
    syn_of_z: dict[str, int]
    z4_arms: dict[str, tuple[tuple[tuple[int, int], str], ...]]
    z2_arms: dict[str, tuple[tuple[int, int], ...]]
    deflag: dict[str, int]
    residual: dict[str, int]


def flagged_roles(patch: HeavyHexOperators = D3) -> FlaggedRoles:
    """Build the blueprint's data, in-row flags, Z syndromes and boundary relays.

    Numerical support ordering preserves the historical d=3 circuit ordering.
    Each Z plaquette uses its upper and lower X-gauge ancillas in that order;
    lone flags receive the upper-left / lower-right virtual Z correction.
    """
    if patch.distance not in (3, 5):
        raise ValueError("flagged circuits support d=3 and d=5 only")
    n = patch.distance**2
    xs = sorted(patch.x_gauges, key=lambda name: patch.x_gauges[name])
    zs = sorted(patch.z_gauges, key=lambda name: patch.z_gauges[name])
    flags = {name: n + i for i, name in enumerate(xs)}
    syndromes = {name: n + len(xs) + i for i, name in enumerate(zs)}
    by_support = {tuple(sorted(support)): name for name, support in patch.x_gauges.items()}
    z4, z2, deflag, residual = {}, {}, {}, {}
    next_qubit = n + len(xs) + len(zs)
    for name in zs:
        support = sorted(patch.z_gauges[name])
        if len(support) == 2:
            z2[name] = tuple((q - 1, next_qubit + i) for i, q in enumerate(support))
            next_qubit += 2
        else:
            a, b, c, d = support
            arms = []
            for i, pair in enumerate(((a, c), (b, d))):
                flag = by_support[pair]
                data = tuple(q - 1 for q in pair)
                arms.append((data, flag))
                deflag[flag] = data[i]
                residual[flag] = data[1 - i]
            z4[name] = tuple(arms)
    return FlaggedRoles(n, next_qubit, flags, syndromes, z4, z2, deflag, residual)


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

    @cached_property
    def roles(self) -> FlaggedRoles:
        return flagged_roles(self.patch)

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        validate_binary_bits("gauge outcomes", gauge_bits, len(self.measurements))
        outcomes = {
            (m.round, m.half, m.gauge): gauge_bits[m.bit]
            for m in self.measurements
            if m.kind in ("z_syn", "x_gauge")
        }
        return _stabilizer_checks(self.patch, outcomes)

    def deflag_corrections(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Virtual Z corrections for lone flags, tagged by (round, half) of their Z half."""
        return [
            ((round, "Z"), Pauli.z_on((self.roles.deflag[flag],)))
            for round, flag in self._lone_flags(gauge_bits)
        ]

    def backaction_residuals(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Weight-1 Z left on the flag's other data qubit once DEFLAG cancels one factor.

        This is the ideal gadget back-action used for noiseless and injected
        data-error validation. Noisy flags require circuit-aware decoding;
        these residuals must not be treated as known faults in that setting.
        """
        return [
            ((round, "Z"), Pauli.z_on((self.roles.residual[flag],)))
            for round, flag in self._lone_flags(gauge_bits)
        ]

    def _lone_flags(self, gauge_bits: tuple[int, ...]) -> list[tuple[int, str]]:
        """(round, flag gauge) per weight-4 Z gauge where exactly one flag fired.

        Both flags firing leaves the full gauge operator, which is harmless.
        """
        validate_binary_bits("gauge outcomes", gauge_bits, len(self.measurements))
        bits = {
            (m.round, m.gauge): gauge_bits[m.bit] for m in self.measurements if m.kind == "z_flag"
        }
        rounds = sorted({m.round for m in self.measurements if m.half == "Z"})
        return [
            (round, first if bits[round, first] else second)
            for round in rounds
            for arms in self.roles.z4_arms.values()
            for first, second in [(arms[0][1], arms[1][1])]
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
    """Flagged memory on 23 (d=3) or 65 (d=5) reused qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    roles = flagged_roles(patch)
    prep_half, halves = _half_order(patch, rounds, basis, error, inject_at)

    total = 3 * len(roles.syn_of_z) * (rounds + (prep_half == "Z")) + len(roles.flag_of_x) * (
        rounds + (prep_half == "X")
    )
    qubits = QuantumRegister(roles.num_qubits, "q")
    gauge = ClassicalRegister(total, "m")
    data = ClassicalRegister(roles.num_data, "d")
    circuit = QuantumCircuit(qubits, gauge, data)

    if basis == "X":
        circuit.h(list(range(roles.num_data)))

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
        (pair_a, gauge_a), (pair_b, gauge_b) = roles.z4_arms[name]
        syndrome = roles.syn_of_z[name]
        flag_a, flag_b = _z4_fragment(
            circuit,
            ((pair_a, roles.flag_of_x[gauge_a]), (pair_b, roles.flag_of_x[gauge_b])),
            syndrome,
        )
        record(round, "Z", "z_flag", gauge_a, flag_a)
        record(round, "Z", "z_flag", gauge_b, flag_b)
        record(round, "Z", "z_syn", name, syndrome)

    def measure_z2(round: int, name: str) -> None:
        syndrome = roles.syn_of_z[name]
        relay_a, relay_b = _z2_fragment(circuit, roles.z2_arms[name], syndrome)
        record(round, "Z", "relay", name, relay_a)
        record(round, "Z", "relay", name, relay_b)
        record(round, "Z", "z_syn", name, syndrome)

    def measure_x2(round: int, name: str) -> None:
        first, second = (q - 1 for q in patch.x_gauges[name])
        _x2_fragment(circuit, (first, second), roles.flag_of_x[name])
        record(round, "X", "x_gauge", name, roles.flag_of_x[name])

    def run_half(round: int, half: str) -> None:
        half_order.append((round, half))
        for name in roles.syn_of_z if half == "Z" else roles.flag_of_x:
            start = len(circuit.data)
            if half == "X":
                measure_x2(round, name)
            elif name in roles.z4_arms:
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
        circuit.h(list(range(roles.num_data)))
    for i in range(roles.num_data):
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
    # Builders place data first and expose their width in the final-readout register.
    data_register = next(register for register in circuit.cregs if register.name == "d")
    data = set(range(len(data_register)))
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

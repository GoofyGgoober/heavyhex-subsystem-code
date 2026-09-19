"""Device-faithful flagged heavy-hex circuits for d=3 (Sundaresan et al. 2023).

23 qubits with reset reuse: 9 data + 6 in-row flags (X gauges) + 4 Z syndromes
+ 4 boundary relays. Weight-4 Z gauges go through their two in-row flags
(Chamberland-style: flags catch hook errors); weight-2 Z gauges go through
relay chains with uncompute (relay outcomes discarded); weight-2 X gauges are
measured directly on their flag qubit.

CX orders are stated explicitly below and verified by single-fault
enumeration in tests/test_heavyhex_flagged.py: every single fault inside a
gadget must leave a deflag+lookup-correctable signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

from ..core import Pauli
from ..patches.heavyhex import D3, HeavyHexOperators

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

# Abstract role indices: data 0..8 (id = label - 1), then flags, syndromes, relays.
FLAG_OF_X = {"X1X4": 9, "X2X5": 10, "X3X6": 11, "X4X7": 12, "X5X8": 13, "X6X9": 14}
SYN_OF_Z = {"Z1Z2": 15, "Z2Z3Z5Z6": 16, "Z4Z5Z7Z8": 17, "Z8Z9": 18}
# Weight-4 Z gauge -> ((data pair, flag gauge name), (data pair, flag gauge name)),
# 0-based data ids. Each flag sits between its pair.
Z4_ARMS = {
    "Z4Z5Z7Z8": (((3, 6), "X4X7"), ((4, 7), "X5X8")),
    "Z2Z3Z5Z6": (((1, 4), "X2X5"), ((2, 5), "X3X6")),
}
# Weight-2 Z gauge -> ((data id, relay idx), (data id, relay idx)).
Z2_ARMS = {"Z1Z2": ((0, 19), (1, 20)), "Z8Z9": ((7, 21), (8, 22))}
Z_HALF_ORDER = ("Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9")
X_HALF_ORDER = ("X1X4", "X2X5", "X3X6", "X4X7", "X5X8", "X6X9")
# Deflagging (paper rule, translated): flag gauge -> data id taking the virtual
# Z correction when exactly that flag fires within its weight-4 Z gauge.
DEFLAG = {"X2X5": 1, "X3X6": 5, "X4X7": 3, "X5X8": 7}
FLAGS_OF_Z4 = {"Z2Z3Z5Z6": ("X2X5", "X3X6"), "Z4Z5Z7Z8": ("X4X7", "X5X8")}
# Abstract index -> Falcon-27 device id (Fig. 4a; fez ids via d3_fez_layout.json).
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

NUM_QUBITS = 23


@dataclass(frozen=True)
class FlaggedMeasurement:
    round: int  # -1 for the prep half
    half: str  # "X" or "Z"
    kind: str  # "z_syn", "z_flag", "relay", "x_gauge"
    gauge: str
    bit: int


@dataclass(frozen=True)
class Fragment:
    name: str
    gauge: str
    half: str
    start: int  # index range into circuit.data (gates only, no barriers)
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
        factors = self.patch.stabilizer_gauge_factors()
        outcomes = {(m.round, m.half, m.gauge): gauge_bits[m.bit] for m in self.measurements}
        checks: dict[tuple[int, str], int] = {}
        rounds = sorted({m.round for m in self.measurements})
        for stab_name, gauge_names in factors.items():
            half = "X" if stab_name.startswith("X") else "Z"
            for round in rounds:
                try:
                    bits = [outcomes[round, half, name] for name in gauge_names]
                except KeyError:
                    continue
                checks[round, stab_name] = sum(bits) % 2
        return checks

    def deflag_corrections(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Virtual Z corrections tagged by (round, half) of their Z half."""
        flags = {
            (m.round, m.gauge): gauge_bits[m.bit] for m in self.measurements if m.kind == "z_flag"
        }
        corrections: list[tuple[tuple[int, str], Pauli]] = []
        rounds = sorted({m.round for m in self.measurements if m.half == "Z"})
        for round in rounds:
            for gauge, (first, second) in FLAGS_OF_Z4.items():
                bit_first, bit_second = flags[round, first], flags[round, second]
                if bit_first and not bit_second:
                    corrections.append(((round, "Z"), Pauli.z_on((DEFLAG[first],))))
                elif bit_second and not bit_first:
                    corrections.append(((round, "Z"), Pauli.z_on((DEFLAG[second],))))
        return corrections

    def backaction_residuals(
        self, gauge_bits: tuple[int, ...]
    ) -> list[tuple[tuple[int, str], Pauli]]:
        """Known weight-1 leftovers of flag back-action, tagged by Z half.

        A lone flag firing means back-action Z_a Z_b on its pair; the virtual
        correction removes one factor, leaving a weight-1 Z residual on the
        other. Both-flags-firing leaves the full gauge operator, which is in
        the gauge group (harmless). Residuals are deterministic given flags,
        so single-shot decoding subtracts them instead of leaving them for the
        lookup (which would face weight-2 errors). Multi-round matching
        decoders instead decode residuals across rounds; see DEFLAG.
        """
        flags = {
            (m.round, m.gauge): gauge_bits[m.bit] for m in self.measurements if m.kind == "z_flag"
        }
        pair_of = {flag: pair for arms in Z4_ARMS.values() for pair, flag in arms}
        residuals: list[tuple[tuple[int, str], Pauli]] = []
        rounds = sorted({m.round for m in self.measurements if m.half == "Z"})
        for round in rounds:
            for gauge, (first, second) in FLAGS_OF_Z4.items():
                bit_first, bit_second = flags[round, first], flags[round, second]
                if bit_first and not bit_second:
                    lone = first
                elif bit_second and not bit_first:
                    lone = second
                else:
                    continue
                other = next(q for q in pair_of[lone] if q != DEFLAG[lone])
                residuals.append(((round, "Z"), Pauli.z_on((other,))))
        return residuals


def _x2_fragment(circuit: QuantumCircuit, data: tuple[int, int], ancilla: int) -> None:
    """Direct weight-2 X measurement: H, CX pair in ascending order, H, measure."""
    circuit.reset(ancilla)
    circuit.h(ancilla)
    circuit.cx(ancilla, data[0])
    circuit.cx(ancilla, data[1])
    circuit.h(ancilla)


def _z4_fragment(
    circuit: QuantumCircuit,
    arms: tuple[tuple[tuple[int, int], int], tuple[tuple[int, int], int]],
    syndrome: int,
) -> tuple[int, int]:
    """Flagged weight-4 Z measurement.

    Per arm: CX(data pair, ascending) onto the flag, then CX(flag, syndrome).
    Arms run in the given order; flags are H-wrapped and measured (X basis).
    Returns the two flag qubit indices in arm order.
    """
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
    """Relay weight-2 Z measurement with uncompute.

    Per arm: CX(data, relay), CX(relay, syndrome), CX(data, relay). Relays
    return to |0> and their outcomes are discarded.
    """
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


def _gauge_data(patch: HeavyHexOperators, half: str, name: str) -> tuple[int, ...]:
    gauges = patch.x_gauges if half == "X" else patch.z_gauges
    return tuple(q - 1 for q in gauges[name])


def memory_circuit_flagged(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, FlaggedSchedule]:
    """Flagged memory circuit on 23 qubits with reset reuse.

    Same experiment shape as circuits.heavyhex.memory_circuit; halves run the
    flagged gadgets sequentially in Z_HALF_ORDER / X_HALF_ORDER.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if patch.distance != 3:
        raise ValueError("flagged gadgets are implemented for d=3 only")
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be X or Z, got {basis!r}")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")

    prep_half = "X" if basis == "Z" else "Z"
    halves = ("Z", "X") if basis == "Z" else ("X", "Z")
    stages = ["after_prep"] + [f"after_{half.lower()}{r}" for r in range(rounds) for half in halves]
    if inject_at not in stages:
        raise ValueError(f"inject_at must be one of {stages}, got {inject_at!r}")
    if error is not None:
        patch.code.validate_data_pauli(error, name="error")

    n = 9
    qubits = QuantumRegister(NUM_QUBITS, "q")
    total = 12 * (rounds + (prep_half == "Z")) + 6 * (rounds + (prep_half == "X"))
    gauge = ClassicalRegister(total, "m")
    data = ClassicalRegister(n, "d")
    circuit = QuantumCircuit(qubits, gauge, data)

    if basis == "X":
        circuit.h(list(range(n)))

    measurements: list[FlaggedMeasurement] = []
    fragments: list[Fragment] = []
    half_order: list[tuple[int, str]] = []
    bit = 0

    def record(round: int, half: str, kind: str, gauge_name: str, qubit: int) -> None:
        nonlocal bit
        circuit.measure(qubit, gauge[bit])
        measurements.append(FlaggedMeasurement(round, half, kind, gauge_name, bit))
        bit += 1

    def fragment(round: int, half: str, gauge_name: str, build) -> None:
        start = len(circuit.data)
        build()
        fragments.append(
            Fragment(f"{half}:{gauge_name}@{round}", gauge_name, half, start, len(circuit.data))
        )

    def z4_build(gauge_name: str, round: int) -> None:
        syndrome = SYN_OF_Z[gauge_name]
        spec = Z4_ARMS[gauge_name]
        arms = tuple((pair, FLAG_OF_X[flag]) for pair, flag in spec)
        flag_a, flag_b = _z4_fragment(circuit, arms, syndrome)
        record(round, "Z", "z_flag", spec[0][1], flag_a)
        record(round, "Z", "z_flag", spec[1][1], flag_b)
        record(round, "Z", "z_syn", gauge_name, syndrome)

    def z2_build(gauge_name: str, round: int) -> None:
        syndrome = SYN_OF_Z[gauge_name]
        relay_a, relay_b = _z2_fragment(circuit, Z2_ARMS[gauge_name], syndrome)
        record(round, "Z", "relay", gauge_name, relay_a)
        record(round, "Z", "relay", gauge_name, relay_b)
        record(round, "Z", "z_syn", gauge_name, syndrome)

    def x2_build(gauge_name: str, round: int) -> None:
        data_ids = _gauge_data(patch, "X", gauge_name)
        assert len(data_ids) == 2
        _x2_fragment(circuit, (data_ids[0], data_ids[1]), FLAG_OF_X[gauge_name])
        record(round, "X", "x_gauge", gauge_name, FLAG_OF_X[gauge_name])

    def z_half(round: int) -> None:
        for gauge_name in Z_HALF_ORDER:
            build = z4_build if gauge_name in Z4_ARMS else z2_build
            fragment(round, "Z", gauge_name, partial(build, gauge_name, round))

    def x_half(round: int) -> None:
        for gauge_name in X_HALF_ORDER:
            fragment(round, "X", gauge_name, partial(x2_build, gauge_name, round))

    def maybe_inject(stage: str) -> None:
        if error is not None and inject_at == stage:
            for qubit in error.x - error.z:
                circuit.x(qubit)
            for qubit in error.z - error.x:
                circuit.z(qubit)
            for qubit in error.x & error.z:
                circuit.y(qubit)

    half_order.append((-1, prep_half))
    (x_half if prep_half == "X" else z_half)(-1)
    circuit.barrier()
    maybe_inject("after_prep")
    for r in range(rounds):
        for half in halves:
            half_order.append((r, half))
            (x_half if half == "X" else z_half)(r)
            circuit.barrier()
            maybe_inject(f"after_{half.lower()}{r}")

    if basis == "X":
        circuit.h(list(range(n)))
    for i in range(n):
        circuit.measure(i, data[i])

    return circuit, FlaggedSchedule(
        patch, basis, rounds, tuple(measurements), tuple(fragments), tuple(half_order)
    )


def with_fault(circuit: QuantumCircuit, index: int, fault: dict[int, str]) -> QuantumCircuit:
    """Copy of circuit with a Pauli fault inserted after instruction index.

    index=-1 inserts before the first instruction. fault maps circuit qubit
    index -> one of X, Y, Z.
    """
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


def propagate_fault(
    circuit: QuantumCircuit,
    start: int,
    end: int,
    fault_index: int,
    fault: dict[int, str],
) -> tuple[Pauli, bool]:
    """Propagate one fault forward through fragment gates [fault_index+1, end).

    Returns (data-qubit Pauli leaving the fragment, measured qubits whose
    outcome flips). Phase-free; reset wipes errors; ancilla residue is
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
    for position in range(fault_index + 1, end):
        instruction = circuit.data[position]
        name = instruction.operation.name
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
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
    data = set(range(9))
    return Pauli(frozenset(x_set & data), frozenset(z_set & data)), frozenset(flipped)


def single_faults(circuit: QuantumCircuit, start: int, end: int):
    """Yield (index, fault) for every single fault inside [start, end).

    After each 1Q gate: X, Y, Z. After each reset: X. After each 2Q gate: the
    15 nontrivial Paulis. Before each measurement: X (as insert-after-prev).
    """
    for index in range(start, end):
        instruction = circuit.data[index]
        name = instruction.operation.name
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
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

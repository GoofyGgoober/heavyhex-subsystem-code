"""Run 1's flagged memory and idle circuits, without mid-circuit resets.

Each circuit runs one patch. Every gate, readout and barrier of run 1 stays, in the same
order; only the resets go, so each readout still has its own bit. A qubit that is not reset
starts its next gadget in X^m|0>, m being its last readout. Pushed through the gadget, that
X flips some of the gadget's readouts and none of the data, so each value is a readout XOR
earlier readouts, its carry:

    X gauge       g_t = x_t ⊕ a                 a: the ancilla's last readout, flag or gauge
    Z4 gauge      g_t = z_t ⊕ z_(t-1)           each flag's X reaches the ancilla twice
    Z2 gauge      g_t = z_t ⊕ z_(t-1) ⊕ r ⊕ r'  r, r': each relay's last readout
    flag, relay   F_t = f_t ⊕ f'                f': the qubit's last readout; 0 without faults

carries() finds them by pushing that X through run 1's gadgets. The detectors are run 1's
on these values: a stabilizer's value XOR its value the round before, and the last round
against the data readout. As in run 1, the prep half's gauge values go into no detector, but
its readouts are carried into round 0.

The idle test is run 1's, read in X or Y, and also in Z for life-1: its data then start in
|0> or |1> instead of |+>, and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose, pi
from typing import TYPE_CHECKING

from heavyhex._validation import validate_binary_bits
from heavyhex.circuits.flagged import FlaggedSchedule, memory_circuit_flagged, propagate_fault
from heavyhex.experiment import circuits as run1
from heavyhex.patches.operators import HeavyHexOperators, build_operators
from heavyhex.simulation.scheduled import RZ

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim
    from qiskit import QuantumCircuit

    from heavyhex.core import Pauli

DISTANCES = (3, 5)
ROUNDS = (1, 2, 3, 4)
IDLE_ROUNDS = run1.IDLE_ROUNDS
IDLE_DISTANCE = 5  # the d=5 footprint holds every data qubit of the d=3 patch
GATES = {
    "h": "H", "cx": "CX", "cz": "CZ", "x": "X", "y": "Y", "z": "Z", "sx": "SQRT_X",
    "measure": "M", "reset": "R",
}  # fmt: skip


@dataclass(frozen=True)
class Setting:
    """One circuit: memory, or the idle test read in X, Y or Z."""

    kind: str  # "memory" or "idle"
    basis: str
    distance: int
    rounds: int
    readout: str = "X"
    start: str = "+"  # the idle test's data start in |+>, or, read in Z, in |0> or |1>

    def __str__(self) -> str:
        text = f"d={self.distance} {run1.Setting(self.kind, self.basis, self.rounds, self.readout)}"
        return text if self.start == "+" else f"{text}, from |{self.start}>"

    @property
    def decoded(self) -> bool:
        return self.kind == "memory"


SETTINGS = tuple(
    Setting("memory", basis, d, n) for d in DISTANCES for basis in "XZ" for n in ROUNDS
) + tuple(Setting("idle", "X", IDLE_DISTANCE, n, readout) for readout in "XY" for n in IDLE_ROUNDS)


@dataclass(frozen=True)
class Schedule(FlaggedSchedule):
    """Run 1's schedule, and each gauge bit's carry: the earlier gauge bits it is XORed with."""

    carries: tuple[tuple[int, ...], ...] = ()

    def values(self, gauge_bits: Sequence[int]) -> tuple[int, ...]:
        """Each gauge, flag and relay value: its readout XOR its carry."""
        validate_binary_bits("gauge outcomes", gauge_bits, len(self.measurements))
        return tuple(
            (bit + sum(gauge_bits[k] for k in carry)) % 2
            for bit, carry in zip(gauge_bits, self.carries, strict=True)
        )

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        """Each stabilizer's value per round, from the readouts."""
        return super().checks(self.values(gauge_bits))


def memory_circuit(
    patch: HeavyHexOperators,
    *,
    rounds: int,
    basis: str,
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, Schedule]:
    """Run 1's flagged memory circuit on the patch's own qubits, without its resets.

    error and inject_at put a data Pauli into it, as in run 1.
    """
    with_resets, schedule = memory_circuit_flagged(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    circuit, moved = without_resets(with_resets)
    gadgets = tuple(
        replace(g, gates=tuple(moved[i] for i in g.gates if i in moved)) for g in schedule.gadgets
    )
    return circuit, Schedule(
        patch,
        basis,
        rounds,
        schedule.measurements,
        gadgets,
        schedule.half_order,
        carries=carries(with_resets, schedule),
    )


def logical_circuit(
    setting: Setting, chip: Sequence[int] | None = None, num_qubits: int | None = None
) -> tuple[QuantumCircuit, Schedule]:
    """Run 1's circuit for the setting, one patch on its chip qubits, without its resets.

    The registers are m{d} (gauges) and d{d} (data), as in run 1. Without a chip, qubit i
    of the flagged circuit is qubit i. The idle test gets its X memory's schedule, which
    says what each bit is.
    """
    patch = build_operators(setting.distance)
    circuit, schedule = memory_circuit(patch, rounds=setting.rounds, basis=setting.basis)
    chip = range(circuit.num_qubits) if chip is None else chip
    one = run1.Setting(setting.kind, setting.basis, setting.rounds, setting.readout)
    with_resets, _ = run1.logical_circuit(
        one, {str(setting.distance): list(chip)}, num_qubits or max(chip) + 1
    )
    circuit = without_resets(with_resets)[0]
    return (read_in_z(circuit, setting.start) if setting.readout == "Z" else circuit), schedule


def read_in_z(circuit: QuantumCircuit, start: str) -> QuantumCircuit:
    """The idle test read in X, its data starting in |start> and read in Z instead: each data
    qubit's first H, which takes it to |+>, dropped for |0> or made an X for |1>, and its last,
    before its readout, dropped. Every wait, pulse and readout of the other qubits stays."""
    if start not in ("0", "1"):
        raise ValueError(f"the data start in |0> or |1> to be read in Z, not |{start}>")
    gates: dict[int, list[int]] = {q: [] for q in read_into(circuit, "d").values()}
    for i, item in enumerate(circuit.data):
        if item.operation.name not in ("barrier", "measure"):
            for q in (circuit.find_bit(bit).index for bit in item.qubits):
                if q in gates:
                    gates[q].append(i)
    first = {found[0] for found in gates.values()}
    last = {found[-1] for found in gates.values()}
    if any(circuit.data[i].operation.name != "h" for i in first | last):
        raise ValueError("the data's first and last gates are not the H gates of a read in X")
    out = circuit.copy_empty_like()
    for i, item in enumerate(circuit.data):
        if i in first and start == "1":
            out.x(item.qubits[0])
        elif i not in first | last:
            out.append(item)
    return out


def read_into(circuit: QuantumCircuit, prefix: str) -> dict[int, int]:
    """The qubit read into each bit of the register whose name starts with prefix."""
    found = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            register, i = circuit.find_bit(item.clbits[0]).registers[0]
            if register.name.startswith(prefix):
                found[i] = circuit.find_bit(item.qubits[0]).index
    return dict(sorted(found.items()))


def without_resets(circuit: QuantumCircuit) -> tuple[QuantumCircuit, dict[int, int]]:
    """The circuit with its resets taken out, and where each kept instruction moved to."""
    out = circuit.copy_empty_like()
    moved = {}
    for i, item in enumerate(circuit.data):
        if item.operation.name != "reset":
            moved[i] = len(out.data)
            out.append(item)
    return out, moved


def carries(circuit: QuantumCircuit, schedule: FlaggedSchedule) -> tuple[tuple[int, ...], ...]:
    """Each gauge bit's carry once run 1's resets go, found on run 1's circuit.

    Without its reset a qubit holds X^m|0>, m being its last readout, since nothing acts
    on it in between. propagate_fault pushes that X from the reset through the gadget.
    """
    previous: dict[int, str] = {}
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        for q in (circuit.find_bit(bit).index for bit in item.qubits):
            if item.operation.name == "reset" and previous.get(q, "measure") != "measure":
                raise ValueError(f"qubit {q} has gates between its readout and its reset")
            previous[q] = item.operation.name

    last: dict[int, int] = {}  # each qubit's latest readout, as a gauge bit
    found: list[set[int]] = [set() for _ in schedule.measurements]
    for gadget in schedule.gadgets:
        resets, reads = [], {}
        for i in gadget.gates:
            item = circuit.data[i]
            qubit = circuit.find_bit(item.qubits[0]).index
            if item.operation.name == "reset":
                resets.append((i, qubit))
            elif item.operation.name == "measure":
                reads[qubit] = circuit.find_bit(item.clbits[0]).registers[0][1]
        for i, qubit in resets:
            if qubit not in last:
                continue  # still in |0> from the start of the shot
            data, flipped = propagate_fault(circuit, gadget, i, {qubit: "X"})
            if data.x or data.z:
                raise ValueError(f"the X that qubit {qubit} carries reaches the data")
            for q in flipped:
                found[reads[q]] ^= {last[qubit]}
        last |= reads
    return tuple(tuple(sorted(carry)) for carry in found)


def add_detectors(
    out: stim.Circuit,
    schedule: Schedule,
    records: dict[tuple[str, int], int],
    observable: int = 0,
    labels: list | None = None,
    *,
    flags: bool | None = None,
) -> None:
    """Run 1's detectors and logical, on the gauge values instead of the readouts.

    records maps (register "m" or "d", bit) to its measurement in out. A detector is a
    stabilizer's value XOR its value the round before; the last round's is compared with
    the data readout. flags makes each flag and relay value from round 0 on a detector too;
    by default only in Z memory, as in run 1. labels, if given, gets (round, stabilizer or
    "kind gauge") for each detector.
    """
    import stim

    patch, basis, rounds = schedule.patch, schedule.basis, schedule.rounds
    if flags is None:
        flags = basis == "Z"
    total = out.num_measurements
    bits = {(m.round, m.half, m.kind, m.gauge): m.bit for m in schedule.measurements}
    kind = "x_gauge" if basis == "X" else "z_syn"
    stabilizers = patch.x_stabilizers if basis == "X" else patch.z_stabilizers
    found: list[tuple[int, str]] = []

    def value(bit: int) -> set[int]:
        return {bit, *schedule.carries[bit]}

    def stabilizer(r: int, name: str) -> set[int]:
        parity: set[int] = set()
        for gauge in patch.stabilizer_gauge_factors[name]:
            parity ^= value(bits[r, basis, kind, gauge])
        return parity

    def detector(label: tuple[int, str], gauge_bits: set[int], data: Sequence[int] = ()) -> None:
        targets = [records["m", b] for b in sorted(gauge_bits)] + [records["d", q] for q in data]
        out.append("DETECTOR", [stim.target_rec(k - total) for k in targets])
        found.append(label)

    for r in range(rounds):
        for name in stabilizers:
            previous = stabilizer(r - 1, name) if r else set()
            detector((r, name), stabilizer(r, name) ^ previous)
    for name, support in stabilizers.items():
        detector((rounds, name), stabilizer(rounds - 1, name), [q - 1 for q in sorted(support)])
    if flags:
        for m in schedule.measurements:
            if m.kind in ("z_flag", "relay") and m.round >= 0:
                detector((m.round, f"{m.kind} {m.gauge}"), value(m.bit))
    if labels is not None:
        labels += found
    logical = patch.code.logical_x if basis == "X" else patch.code.logical_z
    support = sorted(logical.x or logical.z)
    out.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(records["d", q] - total) for q in support],
        observable,
    )


def to_stim(circuit: QuantumCircuit) -> tuple[stim.Circuit, dict[tuple[str, int], int]]:
    """The circuit in stim without noise, and the measurement each (register, bit) lands in.

    Takes the logical gates and the native ones; delays and the idle test's waits do nothing.
    """
    import stim

    out = stim.Circuit()
    records: dict[tuple[str, int], int] = {}
    for item in circuit.data:
        name, op = item.operation.name, item.operation
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            out.append("TICK", [])
        elif name == "rz":
            quarters = float(op.params[0]) / (pi / 2)
            if not isclose(quarters, round(quarters), abs_tol=1e-9):
                raise ValueError(f"rz({op.params[0]}) is not a Clifford gate")
            gate = RZ[round(quarters) % 4]
            if gate:
                out.append(gate, qubits)
        elif name in GATES:
            out.append(GATES[name], qubits)
            if name == "measure":
                register, index = circuit.find_bit(item.clbits[0]).registers[0]
                records[register.name, index] = out.num_measurements - 1
        elif name not in ("delay", "wait"):
            raise ValueError(f"unexpected instruction: {name}")
    return out, records


def same_action(a: QuantumCircuit, b: QuantumCircuit) -> bool:
    """Whether two circuits act alike without noise: the same stabilizer flows, signs
    included, each readout matched by its (register, bit). Delays and waits do nothing."""
    import stim

    (one, one_records), (other, other_records) = to_stim(a), to_stim(b)
    if set(one_records) != set(other_records):
        return False
    used = sorted(_qubits(one) | _qubits(other))
    one, other = _compact(one, used), _compact(other, used)
    there = {one_records[k]: other_records[k] for k in one_records}
    back = {v: k for k, v in there.items()}

    def renamed(flow: stim.Flow, names: dict[int, int]) -> stim.Flow:
        measured = sorted(names[m] for m in flow.measurements_copy())
        return stim.Flow(input=flow.input_copy(), output=flow.output_copy(), measurements=measured)

    return other.has_all_flows([renamed(f, there) for f in one.flow_generators()]) and (
        one.has_all_flows([renamed(f, back) for f in other.flow_generators()])
    )


def _qubits(circuit: stim.Circuit) -> set[int]:
    return {t.value for item in _steps(circuit) for t in item.targets_copy() if t.is_qubit_target}


def _compact(circuit: stim.Circuit, used: Sequence[int]) -> stim.Circuit:
    """The circuit on qubits 0, 1, ... in place of `used`: unused qubits only add flows."""
    import stim

    index = {q: i for i, q in enumerate(used)}
    out = stim.Circuit()
    for item in _steps(circuit):
        targets = [
            stim.GateTarget(index[t.value]) if t.is_qubit_target else t for t in item.targets_copy()
        ]
        out.append(item.name, targets, item.gate_args_copy())
    return out


def _steps(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """The circuit's instructions; to_stim writes no repeat blocks."""
    import stim

    return [item for item in circuit.flattened() if isinstance(item, stim.CircuitInstruction)]


def stim_circuit(setting: Setting, *, flags: bool | None = None) -> stim.Circuit:
    """The setting's circuit in stim without noise; memory gets its detectors and logical."""
    circuit, schedule = logical_circuit(setting)
    out, records = to_stim(circuit)
    if setting.decoded:
        own = {(name[0], i): k for (name, i), k in records.items()}  # "m5" -> "m", as in run 1
        add_detectors(out, schedule, own, flags=flags)
    return out


def noiseless_check(circuit: stim.Circuit, shots: int = 256, seed: int = 0) -> None:
    """Raise unless, without noise, every detector and observable is 0 in every shot.

    Stim's error analysis refuses a detector or observable that some measurement makes
    random; sampling the readouts then checks each parity is 0, not just fixed.
    """
    ideal = circuit.without_noise()
    ideal.detector_error_model()
    measured = ideal.compile_sampler(seed=seed).sample(shots)
    fired, flips = ideal.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=measured, separate_observables=True
    )
    if fired.any() or flips.any():
        raise ValueError(
            f"{int(fired.any(axis=0).sum())} detectors and {int(flips.any(axis=0).sum())} "
            "observables fire without noise"
        )

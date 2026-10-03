"""What runs on the chip: both patches side by side, in Fez's native gates.

Memory in X and Z at each number of rounds, and an idle test: X memory with
the CZ of every CX replaced by a wait of the same length, so the data qubits
wait in |+>, with the same waits, pulses and H gates, while their ancillas are
reset and read out. Half the idle test reads the data in Y instead of X, so a
frequency offset, which turns X into Y, isn't taken for dephasing.
Patch p writes registers m{p} (gauges) and d{p} (data), p being "3" or "5".
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi
from typing import TYPE_CHECKING

from ..circuits.flagged import FlaggedSchedule, memory_circuit_flagged
from ..patches.operators import build_operators
from ..simulation.noisy import DECOUPLE_NS

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

DISTANCES = (3, 5)
ROUNDS = (1, 2, 3, 4, 6, 8)
IDLE_ROUNDS = (1, 2, 4, 8, 16)
BLOCKS = 2  # the job lists every circuit's pieces twice, each time in its own shuffled order
H_FRAME = "h frame"  # marks a wait the target spends between the H gates of a waited-out CX


@dataclass(frozen=True)
class Setting:
    """One circuit of the run."""

    kind: str  # "memory" or "idle"
    basis: str
    rounds: int
    readout: str = "X"  # the idle test reads its data in X or in Y

    def __str__(self) -> str:
        rounds = f"{self.rounds} round{'s' * (self.rounds > 1)}"
        if self.kind == "idle":
            return f"idle, {rounds}, read in {self.readout}"
        return f"{self.basis} {self.kind}, {rounds}"

    @property
    def patches(self) -> tuple[str, ...]:
        """The patches it runs, by register suffix."""
        return tuple(map(str, DISTANCES))

    @property
    def decoded(self) -> bool:
        """Memory is decoded; the idle test isn't."""
        return self.kind != "idle"

    @property
    def copies(self) -> int:
        """Pieces per block: two for memory, one per idle readout.

        Every piece takes the same shots, so the idle test splits a memory circuit's
        share between its X and Y readouts.
        """
        return 2 if self.decoded else 1


SETTINGS = (
    tuple(Setting("memory", basis, n) for basis in "XZ" for n in ROUNDS)
    + tuple(Setting("idle", "X", n) for n in IDLE_ROUNDS)
    + tuple(Setting("idle", "X", n, readout="Y") for n in IDLE_ROUNDS)
)


def job_order(settings: tuple[Setting, ...], seed: int = 0) -> list[tuple[int, int]]:
    """(setting, block) for each piece of the job: each block holds every circuit's
    pieces, shuffled. IBM runs the job shot by shot across all the pieces, so the
    blocks are not early and late; drift shows in the order of each piece's shots.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = []
    for b in range(BLOCKS):
        pieces = [i for i, s in enumerate(settings) for _ in range(s.copies)]
        order += [(int(i), b) for i in rng.permutation(pieces)]
    return order


def piece_shots(shots: int) -> int:
    """Shots in every piece of the job, for `shots` asked per memory circuit."""
    if shots % (2 * BLOCKS):
        raise ValueError(f"shots per memory circuit must be a multiple of {2 * BLOCKS}")
    return shots // (2 * BLOCKS)


def distance(patch: str) -> int:
    return int(patch[0])


def logical_circuit(
    setting: Setting, chips: dict[str, list[int]], num_qubits: int
) -> tuple[QuantumCircuit, dict[str, FlaggedSchedule]]:
    """Both patches on their chip qubits, before translation to native gates.

    chips maps each patch ("3", "5") to its chip qubits.
    """
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit.circuit import Instruction

    pair = QuantumCircuit(num_qubits)
    schedules = {}
    for patch in setting.patches:
        circuit, schedules[patch] = memory_circuit_flagged(
            build_operators(distance(patch)), rounds=setting.rounds, basis=setting.basis
        )
        if setting.kind == "idle":
            last = max(i for i, item in enumerate(circuit.data) if item.operation.name == "barrier")
            kept = circuit.copy_empty_like()
            for i, item in enumerate(circuit.data):
                if item.operation.name == "cx":
                    kept.append(Instruction("wait", 2, 0, []), item.qubits)
                    continue
                if setting.readout == "Y" and i > last and item.operation.name == "h":
                    kept.rz(-pi / 2, item.qubits[0])  # S^dag, so the closing H reads Y
                kept.append(item)
            circuit = kept
        gauges, data = (ClassicalRegister(r.size, f"{r.name}{patch}") for r in circuit.cregs)
        pair.add_register(gauges)
        pair.add_register(data)
        pair.compose(circuit, qubits=chips[patch], clbits=[*gauges, *data], inplace=True)
    return pair, schedules


def for_fez(circuit: QuantumCircuit, target: Target) -> QuantumCircuit:
    """The circuit in Fez's native gates, with every wait written out as a delay.

    Timing is the simulator's: an instruction starts once its qubits are free, a
    barrier waits for all of them, and a CX is H, CZ, H on its target back to back.
    A reset, and the gates after it, wait until just before the qubit's first CX,
    so a fresh qubit sits in |0>. Waits of at least DECOUPLE_NS get two X pulses.
    A two-qubit "wait" is a CX with a delay in place of its CZ: the target's H
    gates stay, so both qubits spend the CX as they would in memory.
    """
    if target.granularity != 1 or target.pulse_alignment != 1:
        raise ValueError("delays would need aligning to the device's time grid")

    def length(name: str, *qubits: int) -> int:
        return round(target[name][qubits].duration / target.dt)

    shortest = ceil(DECOUPLE_NS * 1e-9 / target.dt)
    out = circuit.copy_empty_like()
    free: dict[int, int] = {}  # when each qubit is next free, in dt
    held: dict[int, list] = {}  # a reset qubit's instructions, waiting for its first CX

    def wait(q: int, until: int) -> None:
        gap = until - free.get(q, 0)
        if gap >= shortest and q in free:  # an unused qubit is still in |0>
            pulse = length("x", q)
            slack = gap - 2 * pulse
            for piece in (slack // 4, None, slack - 2 * (slack // 4), None, slack // 4):
                if piece is None:
                    out.x(q)
                elif piece:
                    out.delay(piece, q, unit="dt")
        elif gap > 0:
            out.delay(gap, q, unit="dt")
        free[q] = max(free.get(q, 0), until)

    def h(q: int) -> None:
        out.rz(pi / 2, q)
        out.sx(q)
        out.rz(pi / 2, q)

    def held_length(q: int) -> int:
        return sum(length("sx" if i.operation.name == "h" else "reset", q) for i in held[q])

    def ready(q: int) -> int:
        return free.get(q, 0) + (held_length(q) if q in held else 0)

    def release(q: int, end: int) -> None:
        """Put a reset qubit's held instructions right before `end`."""
        wait(q, end - held_length(q))
        for item in held.pop(q):
            if item.operation.name == "h":
                h(q)
            else:
                out.append(item)
        free[q] = end

    for item in circuit.data:
        name = item.operation.name
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if name == "reset" or (name == "h" and qubits[0] in held):
            held.setdefault(qubits[0], []).append(item)
            continue
        start = max(ready(q) for q in qubits)
        for q in qubits:
            if q in held:
                release(q, start)
            else:
                wait(q, start)
        if name == "barrier":
            out.append(item)
            continue
        if name == "h":
            (q,) = qubits
            h(q)
            free[q] = start + length("sx", q)
        elif name == "cx":
            control, target_qubit = qubits
            turn = length("sx", target_qubit)
            h(target_qubit)
            out.delay(turn, control, unit="dt")
            out.cz(control, target_qubit)
            h(target_qubit)
            out.delay(turn, control, unit="dt")
            free[control] = free[target_qubit] = (
                start + 2 * turn + length("cz", control, target_qubit)
            )
        elif name == "wait":
            control, target_qubit = qubits
            turn, gap = length("sx", target_qubit), length("cz", control, target_qubit)
            h(target_qubit)
            out.delay(turn, control, unit="dt")
            out.delay(gap, control, unit="dt")
            out.barrier(target_qubit, label=H_FRAME)  # a data target sits in |0> meanwhile
            out.delay(gap, target_qubit, unit="dt")
            h(target_qubit)
            out.delay(turn, control, unit="dt")
            free[control] = free[target_qubit] = start + 2 * turn + gap
        elif name == "rz":
            out.append(item)  # virtual, takes no time
        elif name == "measure":
            (q,) = qubits
            out.append(item)
            free[q] = start + length("measure", q)
        else:
            raise ValueError(f"unexpected instruction: {name}")
    return out


def with_durations(target: Target, durations_ns: dict[str, int]) -> Target:
    """A copy of target whose gates take the calibration's median times.

    Qiskit's offline FakeFez is slower than the chip (readout 1.56 vs 1.66 us, CZ
    84 vs 68 ns), so an offline prediction uses the times the calibration saved.
    """
    from copy import deepcopy

    from qiskit.transpiler import InstructionProperties

    timed = deepcopy(target)
    for name, key in (
        ("x", "x"),
        ("sx", "x"),
        ("cz", "cz"),
        ("measure", "measure"),
        ("reset", "reset"),
    ):
        if key not in durations_ns:
            continue
        for qargs, properties in target[name].items():
            if properties is not None:
                timed.update_instruction_properties(
                    name,
                    qargs,
                    InstructionProperties(
                        duration=durations_ns[key] * 1e-9, error=properties.error
                    ),
                )
    return timed

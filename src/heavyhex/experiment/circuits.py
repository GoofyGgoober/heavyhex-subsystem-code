"""What runs on the chip: both patches side by side, in Fez's native gates.

Memory in X and Z at each number of rounds, and an idle test: the same rounds
with every CX removed, so the data qubits just wait in |+> while their ancillas
are reset and read out. Patch d writes registers m{d} (gauges) and d{d} (data).
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


@dataclass(frozen=True)
class Setting:
    """One circuit of the run."""

    kind: str  # "memory" or "idle"
    basis: str
    rounds: int

    def __str__(self) -> str:
        return f"{self.basis} {self.kind}, {self.rounds} round{'s' * (self.rounds > 1)}"


SETTINGS = tuple(Setting("memory", basis, n) for basis in "XZ" for n in ROUNDS) + tuple(
    Setting("idle", "X", n) for n in IDLE_ROUNDS
)


def logical_circuit(
    setting: Setting, chips: dict[int, list[int]], num_qubits: int
) -> tuple[QuantumCircuit, dict[int, FlaggedSchedule]]:
    """Both patches on their chip qubits, before translation to native gates."""
    from qiskit import ClassicalRegister, QuantumCircuit

    pair = QuantumCircuit(num_qubits)
    schedules = {}
    for d in DISTANCES:
        circuit, schedules[d] = memory_circuit_flagged(
            build_operators(d), rounds=setting.rounds, basis=setting.basis
        )
        if setting.kind == "idle":
            kept = circuit.copy_empty_like()
            for item in circuit.data:
                if item.operation.name != "cx":
                    kept.append(item)
            circuit = kept
        gauges, data = (ClassicalRegister(r.size, f"{r.name}{d}") for r in circuit.cregs)
        pair.add_register(gauges)
        pair.add_register(data)
        pair.compose(circuit, qubits=chips[d], clbits=[*gauges, *data], inplace=True)
    return pair, schedules


def for_fez(circuit: QuantumCircuit, target: Target) -> QuantumCircuit:
    """The circuit in Fez's native gates, with every wait written out as a delay.

    Timing is the simulator's: an instruction starts once its qubits are free, a
    barrier waits for all of them, and a CX is H, CZ, H on its target back to back.
    A reset, and the gates after it, wait until just before the qubit's first CX,
    so a fresh qubit sits in |0>. Waits of at least DECOUPLE_NS get two X pulses.
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
        elif name == "measure":
            (q,) = qubits
            out.append(item)
            free[q] = start + length("measure", q)
        else:
            raise ValueError(f"unexpected instruction: {name}")
    return out

"""A circuit in Fez's native gates, scheduled as sent, simulated in stim.

The noise is noisy.py's: gate and readout errors from the calibration, and T1/T2
decay during each delay. A delay next to an X pulse is decoupled; every X in
these circuits is a decoupling pulse. A qubit in |0>, from the start or a reset,
doesn't decay until a gate takes it out, and neither does a delay after a
barrier labelled "h frame": the idle test's data qubit between the H gates of a
waited-out CX, which sits in |0>.
"""

from __future__ import annotations

from math import isclose, pi
from typing import TYPE_CHECKING

from ..patches.placement import Calibration
from .noisy import decoupled_t2, idle_error, single_qubit_error

if TYPE_CHECKING:
    from collections.abc import Mapping

    import stim
    from qiskit import QuantumCircuit

RZ = (None, "S", "Z", "S_DAG")  # rz by 0, 1, 2 or 3 quarter turns, up to a global phase


def noisy_scheduled(
    circuit: QuantumCircuit,
    calibration: Calibration,
    dt: float,
    *,
    decoupled_dephasing: float = 0.0,
    dephasing_by_qubit: Mapping[int, float] | None = None,
) -> tuple[stim.Circuit, dict[tuple[str, int], int]]:
    """The stim circuit, and the measurement each (register, bit) lands in.

    dt is the device's time step in seconds; delays are counted in it.
    dephasing_by_qubit gives chip qubits their own decoupled_dephasing.
    """
    import stim

    ops = [(item, [circuit.find_bit(q).index for q in item.qubits]) for item in circuit.data]
    decoupled: set[int] = set()  # positions of delays next to an X pulse
    last: dict[int, int] = {}
    for i, (item, qubits) in enumerate(ops):
        if item.operation.name == "barrier":
            continue
        for q in qubits:
            if q in last:
                pair = {item.operation.name, ops[last[q]][0].operation.name}
                if pair == {"delay", "x"}:
                    decoupled.add(i if item.operation.name == "delay" else last[q])
            last[q] = i

    out = stim.Circuit()
    records: dict[tuple[str, int], int] = {}
    in_ground = set(range(circuit.num_qubits))
    h_frame: set[int] = set()  # qubits whose next delay is spent in the H frame
    for i, (item, qubits) in enumerate(ops):
        name, op = item.operation.name, item.operation
        if name == "barrier" and op.label == "h frame":
            h_frame.update(qubits)
        elif name == "barrier":
            out.append("TICK")
        elif name == "delay":
            (q,) = qubits
            if q in in_ground or q in h_frame:
                h_frame.discard(q)
                continue
            if op.unit != "dt":
                raise ValueError(f"expected delays in dt, got {op.unit}")
            t1 = calibration.t1_us[q] * 1e3
            t2 = min(calibration.t2_us[q] * 1e3, 2 * t1)
            if i in decoupled:
                share = (dephasing_by_qubit or {}).get(q, decoupled_dephasing)
                t2 = decoupled_t2(t1, t2, share)
            px, pz = idle_error(op.duration * dt * 1e9, t1, t2)
            out.append("PAULI_CHANNEL_1", [q], [px, px, pz])
        elif name == "rz":
            quarters = op.params[0] / (pi / 2)
            if not isclose(quarters, round(quarters), abs_tol=1e-9):
                raise ValueError(f"rz({op.params[0]}) is not a Clifford gate")
            if RZ[round(quarters) % 4]:
                out.append(RZ[round(quarters) % 4], qubits)
        elif name in ("sx", "x"):
            out.append("SQRT_X" if name == "sx" else "X", qubits)
            # IBM quotes average gate infidelity; a depolarizing channel needs 3/2 of it.
            error = min(1.5 * single_qubit_error(calibration, qubits[0]), 0.75)
            out.append("DEPOLARIZE1", qubits, error)
            in_ground.difference_update(qubits)
        elif name == "cz":
            out.append("CZ", qubits)
            # IBM quotes average gate infidelity; a depolarizing channel needs 5/4 of it.
            out.append(
                "DEPOLARIZE2", qubits, min(1.25 * calibration.cz[tuple(sorted(qubits))], 15 / 16)
            )
            in_ground.difference_update(qubits)
        elif name == "reset":
            out.append("R", qubits)
            out.append("X_ERROR", qubits, calibration.readout[qubits[0]])
            in_ground.update(qubits)
        elif name == "measure":
            out.append("M", qubits, calibration.readout[qubits[0]])
            register, index = circuit.find_bit(item.clbits[0]).registers[0]
            records[register.name, index] = out.num_measurements - 1
        else:
            raise ValueError(f"unexpected instruction: {name}")
    return out, records

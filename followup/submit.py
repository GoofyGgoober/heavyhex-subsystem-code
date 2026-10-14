"""Compile, price and send the follow-up's jobs on ibm_kingston, one job at a time.

    python -m followup.submit                   build every job, print it with its cost (--dry)
    python -m followup.submit --predict         the model's prediction for Exp 1, as sent
    python -m followup.submit --submit exp3-1   send that one job, wait for it, save it, stop
    python -m followup.submit --fetch exp3-1    collect a job whose wait was lost (read-only)

The jobs, in the plan's order; each goes only once the one before it has finished and passed
its checks:

    exp3-1  Exp 3: references and arm A, every readout kerneled, untwirled
    exp3-2  Exp 3: references and arm B
    exp3-3  Exp 3: references and arm C
    exp1-1  Exp 1: 16 memory circuits x 4096 shots (16 randomizations x 256), 10 idle x 2048
    life-1  lifetimes: the idle test read in Z, 10 circuits x 1024 shots (4 x 256), as Exp 1's
            goes; T1 and T2 on the d=5 footprint, 12 circuits x 1024 (4 pieces of 256).
            IBM cancelled it at its limit, having charged 10 s; life-z and life-t split it
    exp1-2  Exp 1 again
    life-z  life-1's idle test read in Z alone
    life-t  T1 and T2 alone, 12 circuits x 512 (2 pieces of 256), untwirled, for run 1's sampler
    exp3-d  Exp 3: references and arm D, optional, after the plan's jobs and within PLAN_SECONDS

Exp 1's circuits are circuits.py's on the placed qubits, in the device's gates as run 1 wrote
them (H = rz sx rz, CX = H CZ H on its target), with run 1's barriers between the halves. They
are twirled on IBM's client-side route: samplomatic boxes each layer of CZs, twirling the CZs'
qubits with Paulis, and builds a template and a samplex that draws the Paulis. The template is
scheduled ALAP and XY4 fills every idle period of the patch's qubits after their first gate
(Y = rz(pi/2) X rz(-pi/2), so it takes X's time). The idle test is boxed as its X memory is,
each CZ then replaced by a delay as long: the delays sit between the barriers that close each
layer, so its data wait as in memory, through the same pulses, and the Paulis pass them
unchanged. The idle test read in Z goes as Exp 1's does; the T1 and T2 circuits and Exp
3's are lifetimes.py's and leakage_control.py's, sent as they are. IBM adds no
decoupling and twirls no readout, rep_delay is 250 us, and IBM returns each circuit's schedule.

A job's QPU time is IBM's billing rule, as run 1 was billed: LOAD_S, then N (t + 253 us) for
each circuit run N times and t long once scheduled. IBM may load a job of more than
MAX_EXPERIMENTS circuits (each twirl one) in parts, LOAD_S each, so the caps take that bound.
A job is refused above JOB_SECONDS (the lifetimes' above their own caps, in SECONDS), if the
follow-up's jobs so far and it would pass TOTAL_SECONDS (an optional job PLAN_SECONDS), or if
IBM's usage counter would fall below COUNTER_FLOOR_S; IBM cancels it at its limit, 1.5 times the bound and 2 s, at most its
cap, and while it is queued it is cancelled if IBM's own estimate is over that. The ledger
claims a job before it is sent and counts it at its limit until IBM has counted it, so no
job goes twice and the total holds whatever happens; a send whose outcome is unknown stays
claimed until --fetch settles it.
Every job carries TAG, and a ledger that misses one of IBM's jobs with it refuses to go on.

If IBM refuses the client-side route, or fails the job, --untwirled sends Exp 1 without
twirling (memory in two pieces of the idle's shots); if kerneled data are refused or
malformed, --classified sends Exp 3 with IBM's 0/1 readout. --sampler legacy sends through
run 1's SamplerV2 instead of the executor, untwirled only. A job goes again only with a
fallback it hasn't failed with, and the next job of its experiment goes as it went; life-z
goes as exp1-1 went, since its idle test is Exp 1's, so after a twirled exp1-1 it has no
fallback. life-t, untwirled, goes with --sampler legacy. The job after one IBM ended goes
only with --despite-checks.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from math import ceil, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heavyhex.patches.placement import Calibration, Spot

from . import circuits, decoder, lifetimes
from . import leakage_control as exp3
from .placement import BACKEND, Placement, chip_map, place

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.transpiler import Target
    from qiskit_ibm_runtime import QuantumProgram
    from qiskit_ibm_runtime.options_models import ExecutorOptions
    from samplomatic.samplex import Samplex

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
LEDGER = RUNS / f"{BACKEND}-followup" / "ledger.json"
CODE = ("src", "pyproject.toml", "followup")  # what the circuits and their analysis depend on
# src/'s tree and pyproject.toml's blob, as every job of run 1 records them.
RUN1_CODE = ("a233d4069719d0cc396c71f29f6429bffd40f113", "f8e44350df9165b40fb5a3259f75ba9c9f697fee")
TAG = "heavyhex-followup"  # on every job, so IBM's own list of them needs no ledger

TOTAL_SECONDS = 90  # QPU time approved for the whole follow-up
PLAN_SECONDS = 80  # the plan's jobs within it, leaving room to send one again; optional jobs too
JOB_SECONDS = 30  # and for any one job
SECONDS = {"life-1": 14, "life-z": 7, "life-t": 7}  # and for these, approved, at most JOB_SECONDS
MARGIN_SECONDS = 10  # IBM's counter must leave this much beyond a job's bound
# IBM's counter when the 90 s were approved (375 s), less them. IBM gives run 1's time back
# from 31 October, after which the floor only loosens; the ledger still holds the total.
COUNTER_FLOOR_S = 285
PER_SHOT_US = exp3.REP_DELAY_US + exp3.SHOT_OVERHEAD_US
MAX_EXPERIMENTS = 300  # Kingston's max_experiments
RELEASE_AFTER = timedelta(minutes=15)  # a claim IBM still doesn't list is released after this
FIND_TRIES, FIND_EVERY_S = 6, 10.0  # looking for a job by its tag, as IBM's list can lag
WATCH_EVERY_S = 30.0  # how often IBM's estimate is read while the job is queued

RANDOMIZATIONS = {"memory": 16, "idle": 8}
SHOTS_PER_RANDOMIZATION = 256  # the executor runs every circuit of a job the same shots
TWIRLING = "active"  # each CZ layer twirls its own qubits; the rest keep their XY4

# The placement approved on 13 October 2026 and its parts beyond the thresholds, with their
# values then (us for T1 and T2, error rates otherwise); the d=3 patch's are among them. A
# spot that moves, any other weak part, a weak part WORSE times worse, or a d=3 data qubit
# with T2 below MIN_D3_DATA_T2_US waits for a new approval.
APPROVED_SPOTS = {5: Spot((7, 9), (1, -1)), 3: Spot((11, 9), (1, -1))}
APPROVED_WEAK: dict[tuple[int | tuple[int, int], str], float] = {
    (8, "T2"): 74, (9, "T2"): 41, (10, "T2"): 25, (12, "readout"): 0.185, (18, "T1"): 76,
    (25, "T2"): 26, (32, "T1"): 61, (32, "readout"): 0.042, (35, "readout"): 0.030,
    (37, "T2"): 28, (48, "T2"): 21, (52, "T2"): 56, (52, "readout"): 0.033,
    (54, "readout"): 0.045, (72, "readout"): 0.202, (78, "T2"): 32, (78, "readout"): 0.044,
    (85, "readout"): 0.033, (86, "T2"): 24, (87, "T1"): 54, (87, "T2"): 36, (88, "T2"): 19,
    ((85, 86), "CZ"): 0.047, ((86, 87), "CZ"): 0.057,
    # Calibration of 14 October 2026, 20:27 (+03:00), approved to run as it stands.
    # (12 and 72 readout were raised to their values in it, above.)
    (30, "T2"): 76, (31, "readout"): 0.042, (46, "T2"): 79, (65, "readout"): 0.034,
    (92, "readout"): 0.064,
}  # fmt: skip
WORSE = 1.5
MIN_D3_DATA_T2_US = 50

# Each job: what it runs, the job that must have finished first, and the job whose way it
# must go (twirled or --untwirled, kerneled or --classified), if any.
JOBS = {
    "exp3-1": ("A", None, None),
    "exp3-2": ("B", "exp3-1", "exp3-1"),
    "exp3-3": ("C", "exp3-2", "exp3-2"),
    "exp1-1": ("memory", "exp3-3", None),
    "life-1": ("lifetimes", "exp1-1", "exp1-1"),  # cancelled by IBM, kept for the record
    "exp1-2": ("memory", "life-1", "exp1-1"),
    "life-z": ("z-idle", "exp1-2", "exp1-1"),
    "life-t": ("t1t2", "life-z", None),
    "exp3-d": ("D", "life-t", "exp3-3"),  # last, so that it can't take the plan's room
}
OPTIONAL = ("exp3-d",)
LIFETIMES = {"lifetimes": "ZT", "z-idle": "Z", "t1t2": "T"}  # the parts of life() each runs
TWIRLED = ("memory", "lifetimes", "z-idle")  # twirled or --untwirled; Exp 3's arms are kerneled
DECOUPLING = {
    "memory": "XY4",
    "lifetimes": "XY4 in the idle test; T1 and T2 none, but the echo",
    "z-idle": "XY4",
    "t1t2": "none, but the echo",
}
FLIPS = "measurement_flips."  # the executor's prefix for readout flips a twirl leaves

# The checks a job's data must pass before the next job goes.
IQ_DISTINCT = 0.99  # the share of shots whose IQ points all differ, at least
SCHEDULE_GAP = 0.01  # IBM's length of a circuit against ours, at most this apart
# A memory circuit's detectors may fire as the model's would with its errors this many times
# larger, 1 - 2 r >= (1 - 2 r_model)^MAX_EXCESS; run 1 was 1.2-1.75 times the model.
MAX_EXCESS = 3
# The lifetimes' circuits that should read as prepared (lifetimes.prepared) do so this often,
# at least: 0.92-0.94 in the model, about 0 with a pulse missing.
READ_AS_PREPARED = 0.75


class Refused(Exception):
    """A job that may not be sent."""


@dataclass(frozen=True, eq=False)
class Item:
    """One circuit of a job, sent as it is or as a template its samplex fills in."""

    label: str
    circuit: QuantumCircuit
    seconds: float  # its length once scheduled
    shots: int  # per randomization when twirled
    samplex: Samplex | None = None
    randomizations: int = 0
    setting: circuits.Setting | None = None
    chip: tuple[int, ...] = ()  # the chip qubit of each qubit of the flagged circuit

    @property
    def executions(self) -> int:
        return self.shots * max(self.randomizations, 1)


@dataclass(frozen=True)
class Job:
    name: str
    items: tuple[Item, ...]
    meas_type: str = "classified"

    @property
    def twirled(self) -> bool:
        return any(item.samplex is not None for item in self.items)

    @property
    def instances(self) -> int:
        """The circuits IBM runs, each twirl one."""
        return sum(max(item.randomizations, 1) for item in self.items)

    @property
    def seconds(self) -> float:
        """QPU time by IBM's billing rule."""
        return exp3.LOAD_S + sum(
            item.executions * (item.seconds + PER_SHOT_US * 1e-6) for item in self.items
        )

    @property
    def bound(self) -> float:
        """QPU time should IBM load the job in parts of MAX_EXPERIMENTS circuits."""
        return self.seconds + exp3.LOAD_S * (ceil(self.instances / MAX_EXPERIMENTS) - 1)

    @property
    def billed(self) -> int:
        return ceil(self.seconds)


# Compiling


def native(circuit: QuantumCircuit) -> QuantumCircuit:
    """The circuit in the device's gates as run 1 wrote them: H = rz sx rz, CX = H CZ H on its
    target. The idle test's waits become CZs too, until waited() swaps them for delays."""
    out = circuit.copy_empty_like()

    def h(qubit: Any) -> None:
        out.rz(pi / 2, qubit)
        out.sx(qubit)
        out.rz(pi / 2, qubit)

    for item in circuit.data:
        name, qubits = item.operation.name, item.qubits
        if name == "h":
            h(qubits[0])
        elif name in ("cx", "wait"):
            h(qubits[1])
            out.cz(*qubits)
            h(qubits[1])
        elif name in ("rz", "x", "measure", "reset", "barrier"):
            out.append(item)
        else:
            raise ValueError(f"unexpected instruction: {name}")
    return out


def boxed(circuit: QuantumCircuit) -> QuantumCircuit:
    """Each layer of CZs in a box that twirls its qubits; run 1's barriers stay."""
    from samplomatic.transpiler import generate_boxing_pass_manager

    return generate_boxing_pass_manager(
        enable_gates=True,
        enable_measures=False,
        twirling_strategy=TWIRLING,
        remove_barriers="never",
    ).run(circuit)


def xy4() -> tuple[list[Gate], list[float]]:
    """XY4 as IBM builds it: X, Y, -X, -Y at the middles of four equal spans, each Y and -X a
    turned X, rz(a) X rz(-a), so that it takes X's time."""
    from qiskit.circuit.library import RZGate, XGate

    def turned(angle: float) -> list[Gate]:
        return [RZGate(angle), XGate(), RZGate(-angle)]

    gates = [XGate(), *turned(pi / 2), *turned(pi), *turned(-pi / 2)]
    return gates, [1 / 8, 1 / 4, 0, 0, 1 / 4, 0, 0, 1 / 4, 0, 0, 1 / 8]


def decoupled(
    circuit: QuantumCircuit, target: Target, qubits: Sequence[int], *, decouple: bool = True
) -> QuantumCircuit:
    """Scheduled ALAP, with XY4 in every idle period of `qubits` after their first gate.

    A period too short for XY4 stays a delay, as does every period without `decouple`. The
    qubits not used get no delay.
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        ALAPScheduleAnalysis,
        PadDelay,
        PadDynamicalDecoupling,
        TimeUnitConversion,
    )

    sequence, spacing = xy4()
    pad = (
        PadDynamicalDecoupling(
            target=target,
            dd_sequence=sequence,
            spacing=spacing,
            qubits=list(qubits),
            skip_reset_qubits=True,
        )
        if decouple
        else PadDelay(target=target)
    )
    scheduled = PassManager(
        [TimeUnitConversion(target=target), ALAPScheduleAnalysis(target=target), pad]
    ).run(circuit)
    used = set(qubits)
    out = scheduled.copy_empty_like()
    for item in scheduled.data:
        if item.operation.name != "delay" or scheduled.find_bit(item.qubits[0]).index in used:
            out.append(item)
    return out


def waited(
    circuit: QuantumCircuit, target: Target, chip: Sequence[int] | None = None
) -> QuantumCircuit:
    """Every CZ, in boxes too, replaced by a delay as long on each of its qubits.

    chip gives each qubit's place on the device, if it isn't the qubit's index.
    """
    from qiskit.circuit import BoxOp

    out = circuit.copy_empty_like()
    for item in circuit.data:
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        on_chip = [chip[q] for q in qubits] if chip is not None else qubits
        if isinstance(item.operation, BoxOp):
            body = waited(item.operation.body, target, on_chip)
            out.append(item.operation.replace_blocks([body]), item.qubits, item.clbits)
        elif item.operation.name == "cz":
            for qubit in item.qubits:
                out.delay(_length(target, "cz", *on_chip), qubit, unit="dt")
        else:
            out.append(item)
    return out


def scheduled(
    logical: QuantumCircuit,
    target: Target,
    chip: Sequence[int],
    *,
    idle: bool = False,
    twirl: bool = True,
    decouple: bool = True,
) -> tuple[QuantumCircuit, Samplex | None]:
    """A logical circuit as it is sent: in the device's gates, twirled into a template and the
    samplex that fills it, scheduled, with XY4 on the chip's qubits.

    idle swaps each CZ for a wait as long. Twirled, the waits sit between the barriers that
    close each layer, so the Paulis pass them unchanged; untwirled, they are CZs while the
    circuit is scheduled. Either way no XY4 falls inside them. twirl and decouple undo a fix
    each, for the model.
    """
    circuit, samplex = native(logical), None
    if twirl:
        from samplomatic import build

        boxes = boxed(circuit)
        circuit, samplex = build(waited(boxes, target) if idle else boxes)
    circuit = decoupled(circuit, target, chip, decouple=decouple)
    if idle and not twirl:
        circuit = waited(circuit, target)
    return circuit, samplex


def twirled(
    setting: circuits.Setting,
    chip: Sequence[int],
    target: Target,
    randomizations: int | None = None,
) -> Item:
    """The setting's template, scheduled and decoupled, and the samplex that twirls it; by
    default with RANDOMIZATIONS of its kind."""
    logical, _ = circuits.logical_circuit(setting, chip, target.num_qubits)
    circuit, samplex = scheduled(logical, target, chip, idle=setting.kind == "idle")
    return _item(
        str(setting),
        circuit,
        target,
        SHOTS_PER_RANDOMIZATION,
        samplex,
        randomizations or RANDOMIZATIONS[setting.kind],
        setting,
        chip,
    )


def untwirled(
    setting: circuits.Setting,
    chip: Sequence[int],
    target: Target,
    randomizations: int | None = None,
    piece: int = RANDOMIZATIONS["idle"],
) -> list[Item]:
    """The setting's circuit, scheduled and decoupled, with the shots its randomizations
    would get, in pieces of `piece` randomizations' shots (by default, the idle test's)."""
    logical, _ = circuits.logical_circuit(setting, chip, target.num_qubits)
    circuit, _ = scheduled(logical, target, chip, idle=setting.kind == "idle", twirl=False)
    shots = piece * SHOTS_PER_RANDOMIZATION
    pieces = (randomizations or RANDOMIZATIONS[setting.kind]) // piece
    return [_item(str(setting), circuit, target, shots, setting=setting, chip=chip)] * pieces


def exp1(
    placements: Mapping[int, Placement], target: Target, *, twirl: bool = True
) -> tuple[Item, ...]:
    """Exp 1's circuits: memory, then the idle test, each on its patch's qubits."""
    items: list[Item] = []
    for setting in circuits.SETTINGS:
        chip = placements[setting.distance].qubits
        if twirl:
            items.append(twirled(setting, chip, target))
        else:
            items += untwirled(setting, chip, target)
    return tuple(items)


def life(
    placements: Mapping[int, Placement], target: Target, *, twirl: bool = True, part: str = "ZT"
) -> tuple[Item, ...]:
    """The lifetimes' circuits on the d=5 footprint. With Z in `part`, the idle test read in
    Z, built as Exp 1's idle test is, lifetimes.SHOTS shots each in randomizations or pieces
    of SHOTS_PER_RANDOMIZATION; with T, T1 and T2 on every qubit of the footprint, in pieces
    of that, lifetimes.T_SHOTS shots each, or SHOTS beside the idle test, as life-1 sent them."""
    chip = placements[5].qubits
    pieces = lifetimes.SHOTS // SHOTS_PER_RANDOMIZATION
    items: list[Item] = []
    if "Z" in part:
        for setting in lifetimes.Z_IDLE:
            if twirl:
                items.append(twirled(setting, chip, target, pieces))
            else:
                items += untwirled(setting, chip, target, pieces, piece=1)
    if "T" in part:
        shots = lifetimes.SHOTS if "Z" in part else lifetimes.T_SHOTS
        for label, circuit in lifetimes.circuits(sorted(chip), target):
            item = _item(label, circuit, target, SHOTS_PER_RANDOMIZATION)
            items += [item] * (shots // SHOTS_PER_RANDOMIZATION)
    return tuple(items)


def exp3_items(arm: str, selection: exp3.Selection, target: Target) -> tuple[Item, ...]:
    """The references and one arm, each circuit SHOTS["R0"] shots; the arm in pieces of that."""
    shots = exp3.SHOTS["R0"]
    items: list[Item] = []
    for name in (*exp3.REFERENCES, arm):
        if exp3.SHOTS[name] % shots:
            raise ValueError(f"{name}'s shots are not a multiple of {shots}")
        circuit = exp3.circuit(name, selection, target)
        items += [_item(name, circuit, target, shots)] * (exp3.SHOTS[name] // shots)
    return tuple(items)


def build(
    names: Sequence[str],
    placements: Mapping[int, Placement],
    target: Target,
    *,
    twirl: bool = True,
    kerneled: bool = True,
) -> dict[str, Job]:
    """The named jobs; Exp 1's circuits are built once for both its jobs."""
    selection = exp3.select(placements[3].layout)
    memory: tuple[Item, ...] = ()
    jobs = {}
    for name in names:
        runs = JOBS[name][0]
        if runs == "memory":
            memory = memory or exp1(placements, target, twirl=twirl)
            jobs[name] = Job(name, memory)
        elif runs in LIFETIMES:
            jobs[name] = Job(name, life(placements, target, twirl=twirl, part=LIFETIMES[runs]))
        else:
            meas_type = exp3.MEAS_TYPE if kerneled else "classified"
            jobs[name] = Job(name, exp3_items(runs, selection, target), meas_type)
    return jobs


def _item(
    label: str,
    circuit: QuantumCircuit,
    target: Target,
    shots: int,
    samplex: Samplex | None = None,
    randomizations: int = 0,
    setting: circuits.Setting | None = None,
    chip: Sequence[int] = (),
) -> Item:
    for item in circuit.data:
        qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
        name = item.operation.name
        if name != "barrier" and not target.instruction_supported(name, qubits):
            raise ValueError(f"{label}: {name} on {qubits} is not native to the device")
    seconds = circuit.estimate_duration(target, unit="s")
    return Item(label, circuit, seconds, shots, samplex, randomizations, setting, tuple(chip))


def _length(target: Target, name: str, *qubits: int) -> int:
    properties = target[name].get(qubits) or target[name].get(qubits[::-1])
    if properties is None or properties.duration is None:
        raise ValueError(f"the target has no {name} on {qubits}")
    return round(properties.duration / target.dt)


def _wires(circuit: QuantumCircuit) -> dict[int, list[tuple]]:
    """What each qubit sees, in order; equal for circuits that differ only in the order of
    instructions on different qubits."""
    found: dict[int, list[tuple]] = {}
    for item in circuit.data:
        step = (item.operation.name, *map(str, item.operation.params))
        for qubit in item.qubits:
            found.setdefault(circuit.find_bit(qubit).index, []).append(step)
    return found


# Checking without noise


def check(item: Item, randomizations: int = 2, seed: int = 0) -> None:
    """Raise unless, in each of a few twirls drawn (or as it is, untwirled), the circuit acts
    as its logical circuit does without noise (stim's flows, signs included), a memory
    circuit's detectors and logical are 0, and the data of an idle test read in X or Z read
    as they started: 0 from |+> and |0>, 1 from |1>."""
    if item.setting is None:
        return
    if not item.chip:
        raise ValueError(f"{item.label}: no chip qubits to build its logical circuit on")
    logical, schedule = circuits.logical_circuit(item.setting, item.chip, item.circuit.num_qubits)
    for bound in instances(item, randomizations, seed):
        if not circuits.same_action(bound, logical):
            raise ValueError(f"{item.label}: the circuit sent does not act as its logical one")
        model, records = circuits.to_stim(bound)
        if item.setting.decoded:
            own = {(name[0], i): k for (name, i), k in records.items()}  # "m5" -> "m"
            circuits.add_detectors(model, schedule, own, flags=True)
            circuits.noiseless_check(model)
        elif item.setting.readout in ("X", "Z"):
            data = [k for (name, _), k in records.items() if name.startswith("d")]
            start = item.setting.start == "1"
            if (model.compile_sampler(seed=seed).sample(64)[:, data] != start).any():
                raise ValueError(f"{item.label}: the data don't read {start:d} without noise")


def instances(item: Item, randomizations: int = 2, seed: int = 0) -> list[QuantumCircuit]:
    """The circuit with a few twirls drawn and bound, each angle a whole quarter turn."""
    if item.samplex is None:
        return [item.circuit]
    drawn = item.samplex.sample(
        item.samplex.inputs().bind(), num_randomizations=randomizations, rng=seed
    )
    if any(key.startswith(FLIPS) for key in drawn):
        raise ValueError(f"{item.label}: the twirl flips readouts, which nothing undoes here")
    values = np.asarray(drawn["parameter_values"], dtype=float)
    quarters = np.round(values / (pi / 2))
    if not np.allclose(values, quarters * pi / 2, atol=1e-5):
        raise ValueError(f"{item.label}: a twirl drew an angle that isn't a quarter turn")
    return [item.circuit.assign_parameters(row * pi / 2) for row in quarters]


# The model, before the run


def prediction(
    job: Job,
    placements: Mapping[int, Placement],
    target: Target,
    calibration: Calibration,
    *,
    shots: int | None = None,
    seed: int = 0,
) -> dict:
    """What the model expects of Exp 1's memory circuits as sent, each data qubit decaying at
    IBM's T2, by every decoder, and of run 1's design on the same qubits and calibration. Its
    fits at the top are decoder.PRIMARY's, which leads; "rule" is where the single-fault rule
    (decoder.CAPS) ends."""
    from . import model, native

    sent = {}
    for item in job.items:
        s = item.setting
        if s is not None and s.decoded and (s.basis, s.distance, s.rounds) not in sent:
            schedule = circuits.logical_circuit(s)[1]
            sent[s.basis, s.distance, s.rounds] = (instances(item, 1, seed)[0], schedule)
    shots = shots or model.SHOTS
    chips = {d: p.qubits for d, p in placements.items()}
    old = native.run1_circuits(chips, target)
    return {
        "calibrated": calibration.calibrated,
        "placements": {d: _spot(p.spot) for d, p in sorted(placements.items())},
        "twirled": job.twirled,
        "shots": shots,
        "as_sent": model.predict(sent, calibration, target.dt, shots=shots, seed=seed),
        "run1_design": model.predict(old, calibration, target.dt, shots=shots, seed=seed),
    }


# Sending


def program(job: Job) -> tuple[QuantumProgram, ExecutorOptions]:
    """The job as the executor takes it, and its options."""
    from qiskit_ibm_runtime import QuantumProgram
    from qiskit_ibm_runtime.options_models import ExecutorOptions

    shots = {item.shots for item in job.items}
    if len(shots) != 1:
        raise ValueError(f"{job.name}: the executor runs every circuit the same shots")
    sent = QuantumProgram(shots=shots.pop(), meas_level=job.meas_type)  # type: ignore[arg-type]
    for item in job.items:
        if item.samplex is None:
            sent.append_circuit_item(item.circuit)
        else:
            sent.append_samplex_item(
                item.circuit, samplex=item.samplex, shape=(item.randomizations,)
            )
    options = ExecutorOptions()
    options.execution.init_qubits = True
    options.execution.rep_delay = exp3.REP_DELAY_US * 1e-6
    options.execution.scheduler_timing = True
    return sent, options


def payload(job: Job) -> dict:
    """What the executor would send for the job, encoded as it is sent. Contacts no one."""
    from qiskit_ibm_runtime.quantum_program.params_converters import (
        QUANTUM_PROGRAM_PARAMS_CONVERTERS,
    )

    sent, options = program(job)
    return QUANTUM_PROGRAM_PARAMS_CONVERTERS["v2.0"].encoder(sent, options).model_dump(mode="json")


def send(backend: Any, job: Job, limit: int, tags: Sequence[str], sampler: str = "executor") -> Any:
    """Send the job. Uses QPU time: IBM cancels it past `limit` seconds."""
    if sampler == "executor":
        from qiskit_ibm_runtime import Executor

        sent, options = program(job)
        options.environment.max_execution_time = limit
        options.environment.job_tags = list(tags)
        return Executor(mode=backend, options=options).run(sent)
    from qiskit_ibm_runtime import SamplerV2

    if job.twirled:
        raise Refused("run 1's sampler takes no samplex; add --untwirled")
    sampler_v2 = SamplerV2(mode=backend)
    options: Any = sampler_v2.options
    options.max_execution_time = limit
    options.environment.job_tags = list(tags)
    options.dynamical_decoupling.enable = False
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False
    options.execution.rep_delay = exp3.REP_DELAY_US * 1e-6
    options.execution.meas_type = job.meas_type
    options.experimental = {"execution": {"scheduler_timing": True}}
    pubs: list[Any] = [(item.circuit, None, item.shots) for item in job.items]
    return sampler_v2.run(pubs)


def submit(
    name: str,
    *,
    twirl: bool = True,
    kerneled: bool = True,
    sampler: str = "executor",
    despite: bool = False,
    service: Any = None,
) -> Path:
    """Send one planned job, wait for it, save everything to runs/ and stop.

    Uses QPU time. Refuses, before anything is sent: a job out of the plan's order or sent
    already, or after a job that IBM ended or that failed its checks (unless `despite`); a
    placement other than the one approved; code not committed, or src/ no longer run 1's; a
    circuit that doesn't act as it should without noise; and a job beyond the caps. Before
    exp1-1, saves the model's prediction for its circuits. Returns the job's folder.
    """
    if name not in JOBS:
        raise Refused(f"{name} is not a job of the plan: {', '.join(JOBS)}")
    if sampler == "legacy" and JOBS[name][0] in TWIRLED and twirl:
        raise Refused("run 1's sampler takes no samplex; add --untwirled")
    flags = {"twirl": twirl, "kerneled": kerneled, "sampler": sampler}
    if service is None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
    backend = service.backend(BACKEND)
    calibration = device_calibration(backend)
    placements = place(calibration)
    if problems := approval(placements, calibration):
        raise Refused("the placement isn't the one approved: " + "; ".join(problems))
    code = provenance()
    if code is None or code["changed"]:
        raise Refused(f"commit the code ({', '.join(CODE)}) first: the record names its commit")
    if tuple(code["code"]) != RUN1_CODE:
        raise Refused("src/ or pyproject.toml is no longer run 1's, which the follow-up imports")
    with _locked():
        attempts = settle(service, ledger())
        if attempts:
            _save_ledger(attempts)
        reconcile(service, attempts)
        check_order(name, attempts, flags, despite=despite)
    job = build([name], placements, backend.target, twirl=twirl, kerneled=kerneled)[name]
    for item in dict.fromkeys(job.items):
        check(item)
    payload(job)  # encoded as it is sent, so an error here sends nothing
    expected = (
        prediction(job, placements, backend.target, calibration) if name == "exp1-1" else None
    )

    with _locked():
        attempts = ledger()
        check_order(name, attempts, flags, despite=despite)
        limit = job_limit(job, attempts, remaining(service))
        folder = RUNS / _folder(name, flags)
        if folder.exists():
            raise Refused(f"{folder} exists already")
        tags = [TAG, f"heavyhex:{folder.name}"]
        first = next((a for a in attempts if a["status"] != "refused"), None)
        if first and first.get("calibrated") != calibration.calibrated:
            print(
                f"note: IBM's calibration is of {calibration.calibrated}; "
                f"{first['job']} went under {first.get('calibrated')}",
                file=sys.stderr,
            )
        attempts.append(
            {
                "job": name,
                "folder": folder.name,
                "tags": tags,
                "flags": flags,
                "claimed_utc": _utc(),
                "calibrated": calibration.calibrated,
                "estimate_s": job.seconds,
                "billed_s": job.billed,
                "bound_s": job.bound,
                "limit_s": limit,
                "id": None,
                "status": "claimed",  # counts at its limit until IBM has counted it
                "usage": None,
            }
        )
        _save_ledger(attempts)

    folder.mkdir(parents=True)
    record = {
        "job": None,  # until IBM's reply gives the job's id
        "tag": tags[-1],
        "tags": tags,
        "backend": BACKEND,
        "plan_job": name,
        "flags": flags,
        "submitted": datetime.now().astimezone().isoformat(timespec="seconds"),
        "submitted_utc": attempts[-1]["claimed_utc"],
        "code": code,
        "local": False,
        "qpu_seconds_limit": limit,
        "qpu_seconds_estimate": job.seconds,
        "qpu_seconds_bound": job.bound,
    }
    _write(folder / "job.json", record)
    _write(folder / "run.json", describe(job, placements, backend.target, calibration))
    calibration.save(folder / "calibration.json")
    save_circuits(folder, job)
    if expected is not None:
        _write(folder / "prediction.json", _plain(expected))
    try:
        sent = send(backend, job, limit, tags, sampler)
    except Exception as error:
        if _never_sent(error):
            shutil.rmtree(folder)
            _update(folder.name, status="refused", reason=_reason(error))
            raise Refused(
                f"IBM refused {name} ({error}); it may go again, if need be with --untwirled "
                "(twirling), --classified (IQ data) or --sampler legacy (the executor)"
            ) from error
        sent = _find(service, tags[-1])
        if sent is None:
            _update(folder.name, reason=_reason(error))
            raise Refused(
                f"sending {name} failed ({error}) and IBM doesn't list it; it counts at its "
                f"{limit} s limit until --fetch {name} collects it, or releases the claim "
                f"once IBM hasn't listed it for {RELEASE_AFTER.seconds // 60} min"
            ) from error
    return _collect(folder, sent, service)


def fetch(name: str, *, service: Any = None) -> Path:
    """Collect the job of `name` that is out: wait for it, save it and check it, as submit
    does; or release its claim if IBM still doesn't list it RELEASE_AFTER after. Sends
    nothing."""
    if service is None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
    with _locked():
        attempts = settle(service, ledger())
        if attempts:
            _save_ledger(attempts)
    out = [a for a in attempts if a["job"] == name and a["status"] in ("claimed", "sent")]
    if not out:
        raise Refused(f"{name} has no job out")
    attempt = out[-1]
    folder = RUNS / attempt["folder"]
    job = service.job(attempt["id"]) if attempt["id"] else _find(service, attempt["tags"][-1], 1)
    if job is None:
        released = datetime.fromisoformat(attempt["claimed_utc"]) + RELEASE_AFTER
        if datetime.now(timezone.utc) < released:
            raise Refused(f"IBM doesn't list {name} yet; ask again after {released:%H:%M} UTC")
        shutil.rmtree(folder, ignore_errors=True)
        _update(folder.name, status="refused", reason="IBM never listed it")
        raise Refused(f"IBM never listed {name}: its claim is released and it may go again")
    return _collect(folder, job, service)


def _collect(folder: Path, job: Any, service: Any) -> Path:
    """Record the job's id, cancel it while queued if IBM estimates it over its limit, wait for
    it, save its readouts and IBM's count of it, and check them. A job IBM ends without
    results is failed; a wait that drops leaves the job out, for --fetch."""
    from heavyhex.experiment.run import finish

    record = _read(folder / "job.json")
    name = record["plan_job"]
    record["job"] = job.job_id()
    _write(folder / "job.json", record)
    _update(folder.name, id=job.job_id(), status="sent")
    if ibm := _watch(service, job.job_id(), record["qpu_seconds_limit"]):
        record["qpu_seconds_ibm_estimate"] = ibm
        _write(folder / "job.json", record)
    try:
        result = job.result()
    except Exception as error:
        status = _status(job)
        if status not in ("ERROR", "CANCELLED"):
            raise Refused(f"the wait for {name} dropped ({error}); --fetch {name}") from error
        finish(folder, job)
        record = _read(folder / "job.json")
        reason = record.get("reason") or _reason(error)
        _update(folder.name, status="failed", usage=record.get("usage"), reason=reason)
        raise Refused(
            f"IBM ended {name}, {status} ({reason}); it may go again with a fallback it "
            "hasn't failed with, if one is left, or the next job with --despite-checks"
        ) from error
    save_results(folder, result)
    finish(folder, job)  # when it ran, IBM's count of it, and the calibration then in force
    record = _read(folder / "job.json")
    record["checks"] = inspect(folder)
    _write(folder / "job.json", record)
    _update(folder.name, status="done", usage=record.get("usage"), checks=record["checks"])
    return folder


def _watch(service: Any, job_id: str, limit: float) -> float | None:
    """While the job is queued, IBM's own estimate of it, read afresh every WATCH_EVERY_S;
    the job is cancelled, free then, if that is over its limit. Returns the highest seen."""
    highest = None
    while True:
        try:
            job = service.job(job_id)  # a new handle: a job reads IBM's estimate once
            status = _status(job)
            ibm = (job.usage_estimation or {}).get("quantum_seconds")
        except Exception:  # only a second look; IBM enforces the limit either way
            return highest
        if ibm is not None:
            highest = max(highest or 0.0, float(ibm))
        if status not in ("QUEUED", "INITIALIZING"):
            return highest
        if ibm is not None and ibm > limit:
            job.cancel()
            return highest
        time.sleep(WATCH_EVERY_S)


def _never_sent(error: BaseException) -> bool:
    """Whether IBM can't have the job: refused before anything was posted, or the post was
    answered 4xx. Anything else, a dropped connection or a 5xx, may have left a job."""
    from qiskit_ibm_runtime.api.exceptions import RequestsApiError
    from qiskit_ibm_runtime.exceptions import IBMInputValueError, IBMNotAuthorizedError

    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, RequestsApiError):
            return 400 <= seen.status_code < 500
        seen = seen.__cause__ or seen.__context__
    return isinstance(error, (IBMInputValueError, IBMNotAuthorizedError, Refused))


def _find(service: Any, tag: str, tries: int = FIND_TRIES) -> Any:
    """IBM's job with this tag, None if IBM lists none after a few looks."""
    for k in range(tries):
        try:
            found = service.jobs(job_tags=[tag], backend_name=BACKEND, limit=1)
        except Exception:
            found = []
        if found:
            return found[0]
        if k + 1 < tries:
            time.sleep(FIND_EVERY_S)
    return None


def _status(job: Any) -> str:
    try:
        return str(job.status())
    except Exception:
        return "unknown"


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _folder(name: str, flags: Mapping[str, Any]) -> str:
    """The job's folder in runs/, the fallbacks it goes with in its name."""
    twirled = JOBS[name][0] in TWIRLED
    parts = [f"{BACKEND}-{datetime.now(timezone.utc):%Y-%m-%d}-{name}"]
    parts += ["untwirled"] if twirled and not flags["twirl"] else []
    parts += ["classified"] if not twirled and not flags["kerneled"] else []
    parts += ["legacy"] if flags["sampler"] == "legacy" else []
    return "-".join(parts)


def device_calibration(device: Any) -> Calibration:
    """IBM's latest numbers for the device. Read-only; refuses a coupling map other than Fez's
    and a device that loads fewer circuits at once than MAX_EXPERIMENTS."""
    couplers = {tuple(sorted(edge)) for edge in device.coupling_map.get_edges()}
    if couplers != {tuple(sorted(edge)) for edge in chip_map()[1]}:
        raise ValueError(f"{device.name}'s couplers differ from Fez's map")
    most = getattr(device.configuration(), "max_experiments", MAX_EXPERIMENTS)
    if most < MAX_EXPERIMENTS:
        raise ValueError(f"{device.name} runs {most} circuits at once, not {MAX_EXPERIMENTS}")
    return Calibration.from_device(device)


def approval(placements: Mapping[int, Placement], calibration: Calibration) -> list[str]:
    """What sets today's placement apart from the one approved; empty if nothing does."""
    found = [
        f"d={d} moved to {_spot(placements[d].spot)}"
        for d, spot in APPROVED_SPOTS.items()
        if placements[d].spot != spot
    ]
    for d in sorted(placements, reverse=True):
        for s in placements[d].shortfalls:
            approved = APPROVED_WEAK.get((s.part, s.quantity))
            lifetime = s.quantity in ("T1", "T2")
            if approved is None:
                found.append(f"{s}, not approved")
            elif s.value < approved / WORSE if lifetime else s.value > approved * WORSE:
                was = f"{approved:.0f} us" if lifetime else f"{approved:.1%}"
                found.append(f"{s}, approved at {was}")
    found += [
        f"d=3 data {q}: T2 {calibration.t2_us[q]:.0f} us, below {MIN_D3_DATA_T2_US} us"
        for q in placements[3].roles["data"]
        if calibration.t2_us[q] < MIN_D3_DATA_T2_US
    ]
    return list(dict.fromkeys(found))


def provenance() -> dict | None:
    """The commit; src/'s tree and pyproject.toml's blob in it, as run 1 records them, and
    followup/'s tree; and whether the working tree's code differs from it."""

    def git(*words: str) -> str:
        return subprocess.run(
            ["git", *words], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        trees = git("rev-parse", *(f"HEAD:{path}" for path in CODE)).split()
        changed = git("status", "--porcelain", "--", *CODE)
        return {"commit": git("rev-parse", "HEAD"), "code": trees[:2], "followup": trees[2],
                "changed": bool(changed)}  # fmt: skip
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


# The ledger


def ledger(path: Path | None = None) -> list[dict]:
    """Every attempt to send a job, in order."""
    path = path or LEDGER
    return json.loads(path.read_text())["attempts"] if path.exists() else []


def committed(attempts: Sequence[Mapping]) -> float:
    """QPU seconds the follow-up has used: IBM's count of a job it has finished counting, and
    the job's limit otherwise, a job claimed but never named by IBM's reply included."""
    total = 0.0
    for attempt in attempts:
        if attempt["status"] != "refused":
            charged = _charged(attempt.get("usage"))
            total += attempt["limit_s"] if charged is None else charged
    return total


def check_order(
    name: str, attempts: Sequence[Mapping], flags: Mapping[str, Any], *, despite: bool = False
) -> None:
    """Refuse a job that has gone, unless each time it failed (IBM ended it, or its data
    failed their checks) and it now goes with a fallback it hasn't failed with; and a job
    whose predecessor hasn't gone or is out, or last ended by IBM, failed its checks or cost
    over its estimate (unless `despite`, so that a job with no fallback left holds up no
    other). A job goes as the job JOBS names went: an experiment's next job as the one before
    it, life-1, exp1-2 and life-z as exp1-1."""
    runs, after, like = JOBS[name]
    twirled = runs in TWIRLED
    keys = ("twirl" if twirled else "kerneled", "sampler")
    for attempt in attempts:
        if attempt["job"] != name or attempt["status"] == "refused":
            continue
        if attempt["status"] in ("claimed", "sent"):
            raise Refused(f"{name} is out already ({attempt['folder']}): --fetch {name}")
        if attempt["status"] == "done" and attempt["checks"]["passed"]:
            raise Refused(f"{name} has gone already ({attempt['folder']}); nothing goes twice")
        if all(attempt["flags"][k] == flags[k] for k in keys):
            raise Refused(
                f"{name} failed so already ({attempt['folder']}); it goes again only with "
                "another fallback"
            )
    if after is None:
        return
    tried = [a for a in attempts if a["job"] == after and a["status"] != "refused"]
    if not tried or any(a["status"] in ("claimed", "sent") for a in tried):
        raise Refused(f"{name} goes after {after}, which hasn't finished")
    guide = next(
        (a for a in reversed(attempts) if a["job"] == like and a["status"] == "done"), None
    )
    if guide is not None and guide["flags"][keys[0]] != flags[keys[0]]:
        fallback = "--untwirled" if twirled else "--classified"
        went = "without" if guide["flags"][keys[0]] else "with"
        raise Refused(f"{name} goes as {like} went, {went} {fallback}")
    last = tried[-1]
    if last["status"] == "done":
        what = "failed its checks"
        problems = list(last["checks"]["failed"])
        charged = _charged(last.get("usage"))
        if charged is not None and charged > last["billed_s"] + 1:
            problems.append(
                f"IBM charged {charged:.0f} s against an estimate of {last['billed_s']} s"
            )
    else:
        what = "was ended by IBM"
        problems = [last.get("reason") or "no reason given"]
    if problems and not despite:
        raise Refused(
            f"{after} {what} ({'; '.join(problems)}): send it again with a fallback it hasn't "
            "failed with, if one is left, or go on with --despite-checks"
        )


def job_limit(job: Job, attempts: Sequence[Mapping], remaining_s: float | None) -> int:
    """The job's QPU limit once it passes the caps: 1.5 times its bound and 2 s, at most its
    cap and what TOTAL_SECONDS has left. remaining_s is what IBM's counter leaves."""
    bound, cap = ceil(job.bound), job_cap(job.name)
    if bound > cap:
        raise Refused(f"{job.name} may need {bound} s, over its {cap} s")
    used = committed(attempts)
    if used + bound > TOTAL_SECONDS:
        raise Refused(
            f"{job.name} may need {bound} s and {used:.0f} s are counted already, "
            f"over the {TOTAL_SECONDS} s approved"
        )
    if job.name in OPTIONAL and used + bound > PLAN_SECONDS:
        raise Refused(
            f"{job.name} is optional: it may need {bound} s and {used:.0f} s are counted "
            f"already, past the plan's {PLAN_SECONDS} s"
        )
    limit = int(min(cap, TOTAL_SECONDS - used, ceil(1.5 * job.bound) + 2))
    if remaining_s is None or remaining_s < bound + MARGIN_SECONDS:
        raise Refused(
            f"IBM's counter leaves {remaining_s} s, less than the {bound} s the job may "
            f"need and {MARGIN_SECONDS} s to spare"
        )
    if remaining_s - limit < COUNTER_FLOOR_S:
        raise Refused(
            f"IBM's counter leaves {remaining_s:.0f} s: a {limit} s job could take it below "
            f"{COUNTER_FLOOR_S} s, past the {TOTAL_SECONDS} s approved"
        )
    return limit


def job_cap(name: str) -> int:
    """The most QPU seconds a job may take: its own cap, else JOB_SECONDS, never more."""
    return min(SECONDS.get(name, JOB_SECONDS), JOB_SECONDS)


def remaining(service: Any) -> float | None:
    """The QPU seconds IBM's usage counter leaves this instance. Read-only."""
    left = service.usage().get("usage_remaining_seconds")
    return None if left is None else float(left)


def settle(service: Any, attempts: list[dict]) -> list[dict]:
    """IBM's count of every job it hadn't finished counting (read-only), into the attempts."""
    for attempt in attempts:
        if attempt["id"] and _charged(attempt.get("usage")) is None:
            metrics = service.job(attempt["id"]).metrics() or {}
            attempt["usage"] = metrics.get("usage")
    return attempts


def reconcile(service: Any, attempts: Sequence[Mapping]) -> None:
    """Refuse to go on if IBM lists a job with TAG the ledger doesn't hold (read-only): another
    checkout's, or one whose claim was lost."""
    known = {a["id"] for a in attempts if a["id"]}
    claimed = {a["tags"][-1] for a in attempts if not a["id"] and a["status"] == "claimed"}
    found = service.jobs(job_tags=[TAG], backend_name=BACKEND, limit=None)
    stray = [
        job.job_id()
        for job in found
        if job.job_id() not in known and not set(getattr(job, "tags", []) or []) & claimed
    ]
    if stray:
        raise Refused(
            f"IBM lists follow-up jobs this ledger doesn't hold ({', '.join(stray)}): "
            f"another checkout's ledger? Settle them in {LEDGER} first"
        )


def _charged(usage: Mapping | None) -> float | None:
    if not usage or usage.get("status") != "complete":
        return None
    for key in ("qpu_charge_time_seconds", "quantum_seconds"):
        if key in usage:
            return float(usage[key])
    return None


@contextmanager
def _locked() -> Iterator[None]:
    """The ledger to this process alone, while it reads and writes it."""
    lock = LEDGER.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        held = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise Refused(f"{lock} is held: delete it if no submission is running") from None
    with os.fdopen(held, "w") as f:
        f.write(f"{os.getpid()} {_utc()}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _save_ledger(attempts: Sequence[Mapping]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    _write(LEDGER, {"approved_s": TOTAL_SECONDS, "attempts": list(attempts)})


def _update(folder: str, **fields: Any) -> None:
    """Change the attempt sent to this folder."""
    with _locked():
        attempts = ledger()
        attempt = next(a for a in reversed(attempts) if a["folder"] == folder)
        attempt.update(fields)
        _save_ledger(attempts)


# The record


def describe(
    job: Job, placements: Mapping[int, Placement], target: Target, calibration: Calibration
) -> dict:
    """What the job runs, on which qubits, and what it should cost."""
    return {
        "job": job.name,
        "backend": BACKEND,
        "dt": target.dt,
        "calibrated": calibration.calibrated,
        "placements": {
            str(d): {
                "spot": [list(p.spot.origin), list(p.spot.direction)],
                "qubits": p.qubits,
                "shortfalls": [str(s) for s in p.shortfalls],
            }
            for d, p in sorted(placements.items())
        },
        "meas_type": job.meas_type,
        "rep_delay_us": exp3.REP_DELAY_US,
        "twirling": TWIRLING if job.twirled else None,
        "decoupling": DECOUPLING.get(JOBS[job.name][0]),
        "items": [
            {
                "label": item.label,
                "setting": asdict(item.setting) if item.setting else None,
                "randomizations": item.randomizations,
                "shots": item.shots,
                "executions": item.executions,
                "bits": {reg.name: reg.size for reg in item.circuit.cregs},
                "seconds": item.seconds,
                "duration_dt": round(item.seconds / target.dt),
            }
            for item in job.items
        ],
        "instances": job.instances,
        "qpu_seconds_estimate": job.seconds,
        "qpu_seconds_billed_estimate": job.billed,
        "qpu_seconds_bound": job.bound,
    }


def save_circuits(folder: Path, job: Job) -> None:
    """The circuits sent, and the executor's encoding of the whole job with its samplexes."""
    from qiskit import qpy

    sent = BytesIO()
    qpy.dump([item.circuit for item in job.items], sent)
    (folder / "circuits.qpy.gz").write_bytes(gzip.compress(sent.getvalue()))
    with gzip.open(folder / "program.json.gz", "wt") as f:
        json.dump(payload(job), f)


def save_results(folder: Path, result: Any) -> None:
    """Every circuit's readouts as IBM returned them, keyed circuit/register, and IBM's
    schedule of each circuit and of the job."""
    arrays: dict[str, Any] = {}
    schedules = []
    for k, out in enumerate(result):
        if hasattr(out, "data"):  # run 1's sampler: a pub result
            data = {
                name: v.to_bool_array(order="little") if hasattr(v, "to_bool_array") else v
                for name, v in out.data.items()
            }
            schedules.append(out.metadata.get("compilation", {}).get("scheduler_timing"))
        else:  # the executor's: each register's array
            data = dict(out)
            timing = out.metadata.scheduler_timing
            schedules.append(asdict(timing) if timing else None)
        arrays |= {f"{k}/{name}": np.asarray(values) for name, values in data.items()}
    np.savez_compressed(folder / "results.npz", **arrays)
    chunks = [
        {"start": str(s.start), "stop": str(s.stop), "parts": [asdict(p) for p in s.parts]}
        for s in getattr(getattr(result, "metadata", None), "chunk_timing", None) or []
    ]
    with gzip.open(folder / "schedule.json.gz", "wt") as f:
        json.dump({"circuits": schedules, "chunks": chunks}, f)


def load(folder: Path) -> dict[str, dict[str, np.ndarray]]:
    """Each circuit's readouts by register, shots x bits: twirls' shots in a row, any readout
    flip undone, a circuit sent in pieces joined, and kerneled registers without "_iq"."""
    items = _read(folder / "run.json")["items"]
    saved = np.load(folder / "results.npz")
    by_item: list[dict[str, np.ndarray]] = [{} for _ in items]
    for key in saved.files:
        k, name = key.split("/", 1)
        by_item[int(k)][name] = saved[key]
    found: dict[str, dict[str, list[np.ndarray]]] = {}
    for item, data in zip(items, by_item, strict=True):
        for flips in [name for name in data if name.startswith(FLIPS)]:
            data[flips[len(FLIPS) :]] ^= data.pop(flips)
        for name, values in data.items():
            parts = found.setdefault(item["label"], {}).setdefault(name.removesuffix("_iq"), [])
            parts.append(values.reshape(-1, values.shape[-1]))
    return {
        label: {name: np.concatenate(parts) for name, parts in registers.items()}
        for label, registers in found.items()
    }


def inspect(folder: Path) -> dict:
    """The checks a job's data must pass before the next job goes: every circuit's shots and
    bits as planned; IQ points where kerneled, each shot's all distinct in IQ_DISTINCT of the
    shots; IBM's schedule of every circuit, less the resets it adds before and after, within SCHEDULE_GAP of ours; each memory
    circuit's detectors firing no more than the model's would with MAX_EXCESS times its
    errors: a readout that doesn't leave its qubit in |m>, or bits read in the wrong order,
    make them fire about half the time; and the lifetimes' circuits that should read as
    prepared doing so READ_AS_PREPARED of the time."""
    run = _read(folder / "run.json")
    data = load(folder)
    failed: list[str] = []
    planned: dict[str, dict] = {}
    for item in run["items"]:
        row = planned.setdefault(item["label"], {"shots": 0, "bits": item["bits"]})
        row["shots"] += item["executions"]
    found: dict[str, Any] = {"circuits": {}}
    for label, plan in planned.items():
        rows = {}
        for name, values in data.get(label, {}).items():
            row: dict[str, Any] = {"shots": values.shape[0], "bits": values.shape[1]}
            if np.iscomplexobj(values):
                distinct = [len(np.unique(shot)) == shot.size for shot in values[:200]]
                row["iq_distinct_per_shot"] = float(np.mean(distinct))
                if row["iq_distinct_per_shot"] < IQ_DISTINCT:
                    failed.append(f"{label}: IQ points repeat within a shot")
            elif run["meas_type"] == exp3.MEAS_TYPE:
                failed.append(f"{label}: no IQ points in {name}")
            rows[name] = row
        found["circuits"][label] = rows
        got = {name: (row["shots"], row["bits"]) for name, row in rows.items()}
        if got != {name: (plan["shots"], bits) for name, bits in plan["bits"].items()}:
            failed.append(f"{label}: {got} shots and bits, not {plan['shots']} x {plan['bits']}")
    with gzip.open(folder / "schedule.json.gz", "rt") as f:
        schedules = json.load(f)["circuits"]
    gaps = []
    for item, s in zip(run["items"], schedules, strict=True):
        if not s or "timing" not in s:
            failed.append(f"{item['label']}: IBM returned no schedule")
            continue
        gaps.append(_body(s) - item["duration_dt"])
        if abs(gaps[-1]) > SCHEDULE_GAP * item["duration_dt"]:
            failed.append(f"{item['label']}: IBM's schedule is {gaps[-1]} dt off ours")
    found["schedule_returned"] = len(gaps)
    found["schedule_gap_dt"] = [min(gaps), max(gaps)] if gaps else None
    found["detection"] = detection(folder, data)
    failed += [
        f"{label}: detectors fire {rate['chip']:.1%} of the time, the model's {rate['model']:.1%}"
        for label, rate in found["detection"].items()
        if 1 - 2 * rate["chip"] < (1 - 2 * rate["model"]) ** MAX_EXCESS
    ]
    found["prepared"] = lifetimes.prepared(run["items"], data)
    failed += [
        f"{label}: reads as prepared {share:.0%} of the time"
        for label, share in found["prepared"].items()
        if share < READ_AS_PREPARED
    ]
    found["failed"] = list(dict.fromkeys(failed))
    found["passed"] = not failed
    return found


def detection(
    folder: Path, data: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, dict[str, float]]:
    """Each memory circuit's mean detection rate on the chip and in the model at IBM's T2,
    its detectors from the circuit as sent."""
    from . import analysis

    run = _read(folder / "run.json")
    if not any(item["setting"] and item["setting"]["kind"] == "memory" for item in run["items"]):
        return {}
    calibration = Calibration.load(folder / "calibration.json")
    rates = {}
    for label, (setting, circuit) in analysis.sent(folder).items():
        if setting is not None and setting.decoded:
            _, schedule = circuits.logical_circuit(setting)
            noisy, records, _ = analysis.memory_model(circuit, schedule, calibration, run["dt"])
            fired, _ = analysis.detectors(noisy, records, data[label])
            expected = np.asarray(noisy.compile_detector_sampler(seed=0).sample(4096)).mean()
            rates[label] = {"chip": float(fired.mean()), "model": float(expected)}
    return rates


# The command


def _body(schedule: Mapping[str, Any]) -> int:
    """IBM's length of a circuit less the resets it adds before (INIT_*) and after (FINI_*) it."""
    rows = [line.split(",") for line in schedule["timing"].splitlines()]
    init = [int(r[3]) + int(r[4]) for r in rows if r[1].startswith("INIT_")]
    fini = [int(r[3]) for r in rows if r[1].startswith("FINI_")]
    return min(fini, default=schedule["circuit_duration"]) - max(init, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m followup.submit",
        description="Build, price and send the follow-up's jobs on ibm_kingston.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry", action="store_true", help="build every job and print it with its cost (default)"
    )
    action.add_argument(
        "--predict", action="store_true", help="the model's prediction for Exp 1, as sent"
    )
    action.add_argument(
        "--submit", metavar="JOB", choices=list(JOBS), help="send this one job, wait, save, stop"
    )
    action.add_argument(
        "--fetch", metavar="JOB", choices=list(JOBS), help="collect this job, if it is out"
    )
    parser.add_argument(
        "--untwirled",
        action="store_true",
        help="Exp 1 and life-z untwirled, if IBM refuses the twirling",
    )
    parser.add_argument(
        "--classified", action="store_true", help="Exp 3 read as 0/1, if IQ data fail"
    )
    parser.add_argument(
        "--sampler",
        choices=("executor", "legacy"),
        default="executor",
        help="legacy: run 1's SamplerV2, untwirled only",
    )
    parser.add_argument(
        "--despite-checks",
        action="store_true",
        help="send the job though the one before it failed its checks or was ended by IBM",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="--dry or --predict on this saved calibration and FakeKingston, without IBM",
    )
    parser.add_argument("--out", type=Path, help="--predict: also write it to this file")
    args = parser.parse_args(argv)
    if args.calibration and (args.submit or args.fetch):
        parser.error("--calibration is for --dry and --predict only")
    twirl, kerneled = not args.untwirled, not args.classified
    try:
        if args.submit or args.fetch:
            name = args.submit or args.fetch
            folder = (
                submit(
                    name,
                    twirl=twirl,
                    kerneled=kerneled,
                    sampler=args.sampler,
                    despite=args.despite_checks,
                )
                if args.submit
                else fetch(name)
            )
            checks = _read(folder / "job.json")["checks"]
            print(f"{name} done: {folder}")
            print(json.dumps(checks, indent=1))
            return 0 if checks["passed"] else 2
        if args.predict:
            found = predicted(args.calibration, twirl=twirl)
            if args.out:
                _write(args.out, _plain(found))
            print(table(found))
        else:
            print(dry(args.calibration, twirl=twirl, kerneled=kerneled))
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    return 0


def _device(calibration_path: Path | None) -> tuple[Any, Target, Calibration]:
    """IBM's service, Kingston's target and its calibration (read-only); or, given a saved
    calibration, no service and FakeKingston's target with its gates' lengths."""
    if calibration_path is None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backend: Any = service.backend(BACKEND)
        return service, backend.target, device_calibration(backend)
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    from heavyhex.experiment.circuits import with_durations

    calibration = Calibration.load(calibration_path)
    return None, with_durations(FakeKingston().target, calibration.durations_ns or {}), calibration


def predicted(calibration_path: Path | None = None, *, twirl: bool = True) -> dict:
    """prediction() for exp1-1 as it would go now. Sends nothing."""
    _, target, calibration = _device(calibration_path)
    placements = place(calibration)
    job = build(["exp1-1"], placements, target, twirl=twirl)["exp1-1"]
    return prediction(job, placements, target, calibration)


def table(found: Mapping) -> str:
    """A prediction as ε per round (%) and Λ for each design and decoder."""
    lines = [
        f"calibration {found['calibrated']}, d=5 at {found['placements'][5]}, d=3 at "
        f"{found['placements'][3]}; {found['shots']:,} shots a circuit; cap "
        f"{decoder.PRIMARY} leads, the single-fault rule ends at {found['as_sent']['rule']}",
        f"{'design':12} {'decoder':11} {'ε3 X':>6} {'ε5 X':>6} {'Λ X':>6} "
        f"{'ε3 Z':>6} {'ε5 Z':>6} {'Λ Z':>6}",
    ]
    for design in ("as_sent", "run1_design"):
        for name, row in found[design]["decoders"].items():
            cells = []
            for basis in "XZ":
                fit = row["fits"][basis]
                cells += [f"{100 * fit['eps'][d][0]:6.2f}" for d in (3, 5)]
                cells.append(f"{fit['lambda'][0]:6.3f}")
            lines.append(f"{design:12} {name:11} " + " ".join(cells))
    return "\n".join(lines)


def dry(calibration_path: Path | None = None, *, twirl: bool = True, kerneled: bool = True) -> str:
    """Every job built and priced, the caps it meets, and every circuit checked without noise.

    Reads IBM's calibration, target, usage counter and the follow-up's jobs (read-only), or,
    given a saved calibration, contacts no one. Sends nothing and writes nothing.
    """
    service, target, calibration = _device(calibration_path)
    usage_left = remaining(service) if service else None
    attempts = settle(service, ledger()) if service else ledger()
    stray = None
    if service:
        try:
            reconcile(service, attempts)
        except Refused as refusal:
            stray = str(refusal)
    placements = place(calibration)
    jobs = build(list(JOBS), placements, target, twirl=twirl, kerneled=kerneled)
    for item in dict.fromkeys(item for job in jobs.values() for item in job.items):
        check(item)
    code = provenance()
    used = committed(attempts)
    lines = [
        f"{BACKEND}, IBM's calibration of {calibration.calibrated}"
        + ("" if service else f" (saved, {calibration_path}; FakeKingston's gates)"),
        f"QPU time: {used:.0f} s of the follow-up's {TOTAL_SECONDS} s counted; "
        + (
            f"IBM's counter leaves {usage_left:.0f} s (floor {COUNTER_FLOOR_S} s)"
            if usage_left is not None
            else "IBM not asked"
        ),
    ]
    if stray:
        lines.append(f"ledger: {stray}")
    for distance, placement in sorted(placements.items(), reverse=True):
        qubits = placement.qubits
        lines.append(
            f"d={distance} at {_spot(placement.spot)}: {len(qubits)} qubits {min(qubits)}-"
            f"{max(qubits)}, {len(placement.shortfalls)} beyond the thresholds"
        )
    problems = approval(placements, calibration)
    lines.append("placement: " + ("as approved" if not problems else "; ".join(problems)))
    lines.append(
        "code: "
        + ("no commit holds it" if code is None else f"commit {code['commit'][:10]}")
        + (", working tree differs" if code and code["changed"] else "")
        + ("" if code is None or tuple(code["code"]) == RUN1_CODE else ", src/ is not run 1's")
    )
    lines.append(
        "every circuit checked without noise: acts as its logical circuit (2 twirls each); "
        "detectors and logical 0, idle data read as they started"
    )
    lines += [
        "",
        f"{'job':8} {'circuits':>8} {'executions':>10} {'QPU s':>6} {'bound':>6} {'limit':>5}"
        "  runs",
    ]
    planned = bound = 0
    for job in jobs.values():
        circuits_run = ", ".join(dict.fromkeys(item.label for item in job.items))
        if JOBS[job.name][0] in (*TWIRLED, *LIFETIMES):
            circuits_run = f"{len(dict.fromkeys(job.items))} circuits, " + (
                "twirled" if job.twirled else "untwirled"
            )
        limit = int(min(job_cap(job.name), ceil(1.5 * job.bound) + 2))
        lines.append(
            f"{job.name:8} {job.instances:>8} {sum(i.executions for i in job.items):>10,} "
            f"{job.seconds:>6.2f} {job.bound:>6.2f} {limit:>5}  {circuits_run}, "
            f"{job.meas_type}" + (" (optional)" if job.name in OPTIONAL else "")
        )
        if job.name not in OPTIONAL:
            planned += job.billed
            bound += ceil(job.bound)
    extra = sum(jobs[name].billed for name in OPTIONAL)
    own = ", ".join(f"{name} {cap} s" for name, cap in SECONDS.items())
    lines.append(
        f"planned: {planned} s ({bound} s if IBM loads in parts), {planned + extra} s with "
        f"{', '.join(OPTIONAL)}, which goes last and only within {PLAN_SECONDS} s; caps "
        f"{JOB_SECONDS} s a job ({own}), {TOTAL_SECONDS} s in all, IBM's counter "
        f"{MARGIN_SECONDS} s over"
    )
    shown = ("exp3-1", "exp1-1", "life-z", "life-t")
    for name in shown:
        job = jobs[name]
        lines += ["", f"{name}: {'circuit':40} {'randomizations x shots':>23} {'length us':>10}"]
        for item in dict.fromkeys(job.items):
            count = job.items.count(item)
            shots = f"{item.randomizations} x {item.shots}" if item.samplex else f"{item.shots}"
            shots += f" (x{count})" if count > 1 else ""
            lines.append(f"        {item.label:40} {shots:>23} {item.seconds * 1e6:>10.3f}")
    sizes = {name: len(json.dumps(payload(jobs[name]))) for name in shown}
    lines += [
        "",
        "encoded for the executor (nothing sent): "
        + ", ".join(f"{name} {size / 1e6:.1f} MB" for name, size in sizes.items()),
    ]
    return "\n".join(lines)


def _spot(spot: Spot) -> str:
    return f"{spot.origin} facing {spot.direction}"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(value: Any) -> Any:
    """The value as JSON takes it: tuple keys joined, numpy numbers as Python's."""
    if isinstance(value, Mapping):
        return {
            "-".join(map(str, k)) if isinstance(k, tuple) else str(k): _plain(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, payload: Mapping) -> None:
    """Written whole or not at all."""
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    os.replace(part, path)


if __name__ == "__main__":
    sys.exit(main())

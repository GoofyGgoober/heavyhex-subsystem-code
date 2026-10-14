"""The follow-up's memory circuits for the model: as sent, with a fix undone, and run 1's.

The compile is submit.py's (submit.scheduled), so the model runs on the circuits as they are
sent; twirl, xy4 and resets undo a fix each, to split the change from run 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from heavyhex.experiment import circuits as run1

from . import submit
from .circuits import SETTINGS, Schedule, Setting, logical_circuit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
    from samplomatic.samplex import Samplex

    from heavyhex.circuits.flagged import FlaggedSchedule


@dataclass(frozen=True)
class Template:
    """A scheduled circuit; when twirled, its layers are parameters that samplex fills."""

    setting: Setting
    circuit: QuantumCircuit
    samplex: Samplex | None
    schedule: Schedule | FlaggedSchedule  # run 1's, when it keeps run 1's resets


def template(
    setting: Setting,
    chip: Sequence[int],
    target: Target,
    *,
    twirl: bool = True,
    xy4: bool = True,
    resets: bool = False,
) -> Template:
    """The setting's circuit on its chip qubits as submit.py sends it, or with a fix undone:
    resets keeps run 1's resets, and its detectors."""
    if resets:
        patch = str(setting.distance)
        one = run1.Setting(setting.kind, setting.basis, setting.rounds, setting.readout)
        logical, schedules = run1.logical_circuit(one, {patch: list(chip)}, target.num_qubits)
        schedule: Schedule | FlaggedSchedule = schedules[patch]
    else:
        logical, schedule = logical_circuit(setting, chip, target.num_qubits)
    circuit, samplex = submit.scheduled(
        logical, target, chip, idle=setting.kind == "idle", twirl=twirl, decouple=xy4
    )
    return Template(setting, circuit, samplex, schedule)


def instance(t: Template, seed: int = 0) -> QuantumCircuit:
    """One randomization of the template, every angle a whole number of quarter turns."""
    item = submit.Item(str(t.setting), t.circuit, 0.0, 1, t.samplex, 1, t.setting)
    return submit.instances(item, 1, seed)[0]


def memory_circuits(
    chips: Mapping[int, Sequence[int]],
    target: Target,
    *,
    twirl: bool = True,
    xy4: bool = True,
    resets: bool = False,
    seed: int = 0,
) -> dict[tuple[str, int, int], tuple[QuantumCircuit, Schedule | FlaggedSchedule]]:
    """One randomization of each memory circuit on its patch's chip qubits, keyed by
    (basis, distance, rounds). chips maps each distance to its chip qubits."""
    found = {}
    for k, setting in enumerate(s for s in SETTINGS if s.decoded):
        chip = chips[setting.distance]
        t = template(setting, chip, target, twirl=twirl, xy4=xy4, resets=resets)
        found[setting.basis, setting.distance, setting.rounds] = (instance(t, seed + k), t.schedule)
    return found


def run1_circuits(
    chips: Mapping[int, Sequence[int]], target: Target
) -> dict[tuple[str, int, int], tuple[QuantumCircuit, FlaggedSchedule]]:
    """Run 1's design of the same memory circuits on the same qubits: resets, an X pair in
    every idle of at least 500 ns, no twirling (heavyhex.experiment.circuits.for_fez)."""
    found = {}
    for setting in (s for s in SETTINGS if s.decoded):
        d = str(setting.distance)
        circuit, schedules = run1.logical_circuit(
            run1.Setting("memory", setting.basis, setting.rounds),
            {d: list(chips[setting.distance])},
            target.num_qubits,
        )
        key = setting.basis, setting.distance, setting.rounds
        found[key] = (run1.for_fez(circuit, target), schedules[d])
    return found

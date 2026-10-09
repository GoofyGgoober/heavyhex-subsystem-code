"""Shared loading for the post-run diagnosis of why the calibration model misses the chip.

Exploratory: nothing here was pre-registered, and none of it changes the registered
analysis in src/. Every script reads the committed run folders and uses no QPU time.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import SETTINGS
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import Calibration

REPO = Path(__file__).resolve().parents[2]
ROUND_US = 7.9  # one round at both distances (paper, The setup)


def memory_index(basis: str, rounds: int) -> int:
    return next(
        i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == basis and s.rounds == rounds
    )


@dataclass
class Job:
    folder: Path
    info: dict = field(init=False)
    analysis: dict = field(init=False)

    def __post_init__(self) -> None:
        self.info = json.loads((self.folder / "run.json").read_text())
        self.analysis = json.loads((self.folder / "analysis.json").read_text())

    @property
    def name(self) -> str:
        return self.folder.name.removeprefix("ibm_fez-")

    @property
    def day(self) -> str:
        return self.info["rep"]["day"]

    @property
    def rep(self) -> int:
        return self.info["rep"]["rep"]

    @property
    def f(self) -> float:
        return self.analysis["decoupled_dephasing"]

    @cached_property
    def calibration(self) -> Calibration:
        return Calibration.load(self.folder / "calibration.json")

    @cached_property
    def circuits(self) -> list:
        return R.load_circuits(self.folder)

    @cached_property
    def shots(self) -> list:
        return R.load_shots(self.folder)

    def model(self, idx: int, calibration: Calibration | None = None, f: float | None = None):
        """(stim model, records, labels) for setting idx, as the analysis built it."""
        return R.simulated(
            SETTINGS[idx], self.circuits[idx], calibration or self.calibration, self.info["dt"],
            self.f if f is None else f,
        )

    def chip_detectors(self, idx: int, model=None, records=None):
        """The chip's detector bits and observable flips for setting idx."""
        if model is None:
            model, records, _ = self.model(idx)
        measured = R.in_record_order(self.shots[idx], records)
        return model.compile_m2d_converter().convert(measurements=measured, separate_observables=True)

    def readout_qubit(self, patch: str, name: str, k: int) -> int:
        """Chip qubit of a flag ("z_flag Xa..") or the k-th relay ("relay Z..") readout."""
        roles = qubit_roles(build_operators(int(patch)))
        kind, gauge = name.split()
        index = roles.x_ancillas[gauge] if kind == "z_flag" else roles.z2_arms[gauge][k][1]
        return self.info["chips"][patch][index]


def jobs() -> list[Job]:
    return [
        Job(f)
        for f in sorted((REPO / "runs").glob("ibm_fez-2026-10-*-r*"))
        if not f.name.endswith("-offline") and (f / "analysis.json").exists()
    ]


def readout_series(labels) -> dict:
    """Flag and relay detectors -> {(patch, name, k): [detector column per round]}."""
    seen = collections.Counter()
    out: dict = collections.defaultdict(dict)
    for col, (patch, rnd, name) in enumerate(labels):
        if " " not in name:
            continue
        k = seen[(patch, rnd, name)]
        seen[(patch, rnd, name)] += 1
        out[(str(patch), name, k)][rnd] = col
    return {key: [cols[r] for r in sorted(cols)] for key, cols in out.items()}


def marginals(model) -> np.ndarray:
    """Each detector's firing probability, exactly, from the model's independent error mechanisms."""
    dem = model.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    prod = np.ones(dem.num_detectors)
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        p = ins.args_copy()[0]
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                prod[t.val] *= 1 - 2 * p
    return (1 - prod) / 2

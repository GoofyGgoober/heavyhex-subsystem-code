"""Shared loading for the analyses done after the run.

Paths, the fifteen jobs (run folder, analysis, calibration, circuits, shots and the stim model
the analysis built), the flag and relay readout series, which data qubits a readout qubit
touches, and a strict JSON writer. Nothing here changes src/ or runs/, and nothing uses QPU time.
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from heavyhex.circuits.flagged import memory_circuit_flagged, qubit_roles
from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import SETTINGS
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import Calibration

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"  # committed outputs, one or more per script
CACHE = HERE / "cache"  # large intermediates the scripts rebuild (git-ignored)
ROUND_US = 7.9  # one round at both distances
WORKERS = 2  # default process count of the heavy steps; more needs more memory


def memory_index(basis: str, rounds: int) -> int:
    return next(
        i
        for i, s in enumerate(SETTINGS)
        if s.kind == "memory" and s.basis == basis and s.rounds == rounds
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
            SETTINGS[idx],
            self.circuits[idx],
            calibration or self.calibration,
            self.info["dt"],
            self.f if f is None else f,
        )

    def chip_detectors(self, idx: int, model=None, records=None):
        """The chip's detector bits and observable flips for setting idx."""
        if model is None:
            model, records, _ = self.model(idx)
        assert records is not None, "pass the records with the model"
        measured = R.in_record_order(self.shots[idx], records)
        return model.compile_m2d_converter().convert(
            measurements=measured, separate_observables=True
        )

    def readout_qubit(self, patch: str, name: str, k: int) -> int:
        """Chip qubit of a flag ("z_flag Xa..") or the k-th relay ("relay Z..") readout."""
        roles = qubit_roles(build_operators(int(patch)))
        kind, gauge = name.split()
        index = roles.x_ancillas[gauge] if kind == "z_flag" else roles.z2_arms[gauge][k][1]
        return self.info["chips"][patch][index]


def jobs() -> list[Job]:
    """The fifteen jobs that ran on the chip, in folder order."""
    return [
        Job(f)
        for f in sorted(RUNS.glob("ibm_fez-2026-10-*-r*"))
        if not f.name.endswith("-offline") and (f / "analysis.json").exists()
    ]


def job_folders() -> list[Path]:
    """The same fifteen folders, without loading them (combine() skips the offline ones too)."""
    found = []
    for folder in sorted(RUNS.glob("ibm_fez-2026-10-*-r*")):
        if not (folder / "analysis.json").exists():
            continue
        info = read(folder / "run.json")
        if not info.get("rep") or info.get("offline"):
            continue
        found.append(folder)
    return found


def day_number(day: str) -> int:
    """Folder dates 2026-10-04..08 are the paper's days 1-5."""
    return int(day[-2:]) - 3


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


def readout_model(job: Job, idx: int):
    """job.model(idx) with a detector on every flag and relay readout, in either basis.

    Flags and relays are uncomputed, so they read 0 without faults in X memory as well as in Z
    memory, where the analysis already made each readout a detector. In X memory they are added
    here for rounds 1..n; the preparation half's readouts are left out, so that a series has n
    readouts in both bases. Labels as job.model's, (patch, round, "kind gauge").
    """
    import stim

    model, records, labels = job.model(idx)
    setting = SETTINGS[idx]
    if setting.basis == "Z":
        return model, records, labels
    labels = list(labels)
    for patch in ("3", "5"):
        if (f"m{patch}", 0) not in records:
            continue
        _, schedule = memory_circuit_flagged(
            build_operators(int(patch)), rounds=setting.rounds, basis="X"
        )
        for m in schedule.measurements:
            if m.kind in ("z_flag", "relay") and m.round >= 0:
                k = records[f"m{patch}", m.bit] - model.num_measurements
                model.append("DETECTOR", [stim.target_rec(k)])
                labels.append((patch, m.round, f"{m.kind} {m.gauge}"))
    return model, records, labels


def data_read(patch: str, name: str, k: int) -> set[int]:
    """Data qubit labels a readout qubit is coupled to by CXs."""
    kind, gauge = name.split()
    if kind == "z_flag":  # an X gauge's ancilla, flagging a Z4 gauge: its two data qubits
        return {int(x) for x in gauge.lstrip("X").split("X")}
    roles = qubit_roles(build_operators(int(patch)))
    return {roles.z2_arms[gauge][k][0] + 1}


def z4_of(patch: str, name: str) -> str | None:
    """The Z4 gauge a flag serves (None for a relay)."""
    if not name.startswith("z_flag"):
        return None
    roles = qubit_roles(build_operators(int(patch)))
    gauge = name.split()[1]
    return next((z for z, arms in roles.z4_arms.items() if any(g == gauge for _, g in arms)), None)


def streaks(x: np.ndarray) -> np.ndarray:
    """Fired at round r - 1 and at round r."""
    return x[:, 1:] & x[:, :-1]


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


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result(name: str) -> dict:
    """A committed output: results/<name>."""
    return read(RESULTS / name)


def clean(x):
    """Strict JSON: numpy scalars to Python, NaN and inf to null."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(name: str, data) -> Path:
    """results/<name>, strict JSON."""
    path = RESULTS / name
    path.write_text(json.dumps(clean(data), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return path

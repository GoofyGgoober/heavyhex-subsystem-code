"""Where the patches sit on ibm_kingston: the d=5 footprint, and the d=3 patch inside it.

A patch is within the thresholds when every qubit it uses has T1 >= 80 us, T2 >= 50 us
(80 us on data qubits) and readout error <= 3%, and every CZ it uses has error <= 1%.
Spots are ranked by the errors one round expects,

    S = sum e_CZ + sum e_readout + sum over data of t_round / (2 T2),

each CZ and readout counted as often as a round uses it. Among spots within the
thresholds the lowest S wins; if there is none, the lowest S wins and every part beyond
them is named. A spot that uses anything IBM marks broken is never used. The d=3 patch
takes the best spot inside the d=5 footprint, so both distances run on the same qubits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

from heavyhex.circuits.flagged import one_round, qubit_roles
from heavyhex.patches.layout import HeavyHexLayout
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import FEZ_MAP, Calibration, Spot, chip_qubits, problems, spots

BACKEND = "ibm_kingston"
MIN_T1_US = 80
MIN_T2_US = 50
MIN_DATA_T2_US = 80
MAX_READOUT = 0.03
MAX_CZ = 0.01
# A twirled round without resets, as submit.py schedules it on Kingston's target of 13
# October: the twirl's one-qubit layers carry the H gates, so they add little.
ROUND_US = {3: 5.81, 5: 5.96}


class Shortfall(NamedTuple):
    """A part beyond its threshold."""

    part: int | tuple[int, int]  # a qubit, or a coupler (low, high)
    role: str  # "data", "X ancilla", "flag", "Z ancilla", "relay" or "coupler"
    quantity: str  # "T1", "T2", "readout" or "CZ"
    value: float  # us for T1 and T2, otherwise an error rate

    def __str__(self) -> str:
        part = "-".join(map(str, self.part)) if isinstance(self.part, tuple) else self.part
        value = f"{self.value:.0f} us" if self.quantity in ("T1", "T2") else f"{self.value:.1%}"
        return f"{self.role} {part}: {self.quantity} {value}"


@dataclass(frozen=True)
class Placement:
    spot: Spot
    layout: HeavyHexLayout
    score: float  # S
    shortfalls: tuple[Shortfall, ...]

    @property
    def usable(self) -> bool:
        return not self.shortfalls

    @property
    def qubits(self) -> list[int]:
        """The chip qubit for each qubit of the flagged circuit."""
        return chip_qubits(self.layout)

    @property
    def roles(self) -> dict[str, list[int]]:
        return by_role(self.layout)


@cache
def chip_map() -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Kingston's qubit coordinates and couplers, which are Fez's."""
    device = json.loads(FEZ_MAP.read_text())
    return device["coords"], [(a, b) for a, b in device["edges"]]


def fetch_calibration(path: str | Path, backend: str = BACKEND) -> Calibration:
    """Pull the backend's latest calibration and save it to `path`. Read-only, no QPU time."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    device = QiskitRuntimeService().backend(backend)
    couplers = {tuple(sorted(edge)) for edge in device.coupling_map.get_edges()}
    if couplers != {tuple(sorted(edge)) for edge in chip_map()[1]}:
        raise ValueError(f"{backend}'s couplers differ from {FEZ_MAP.name}")
    calibration = Calibration.from_device(device)
    calibration.save(path)
    return calibration


def place(calibration: Calibration) -> dict[int, Placement]:
    """The best d=5 spot, and the best d=3 spot inside its footprint."""
    fives = rank(5, calibration)
    if not fives:
        raise ValueError(f"every d=5 spot on {calibration.backend} uses a broken part")
    footprint = fives[0].layout.physical_qubits
    threes = rank(3, calibration, inside=footprint)
    if not threes:
        raise ValueError(f"every d=3 spot inside {sorted(footprint)} uses a broken part")
    return {5: fives[0], 3: threes[0]}


def rank(
    distance: int,
    calibration: Calibration,
    *,
    inside: set[int] | None = None,
    round_us: float | None = None,
) -> list[Placement]:
    """Every spot with no broken part, best first; only those within `inside` if given."""
    found = []
    for spot, layout in spots(distance, *chip_map()).items():
        if inside is not None and not layout.physical_qubits <= inside:
            continue
        if problems(layout, calibration):
            continue
        s = score(layout, calibration, round_us)
        found.append(Placement(spot, layout, s, tuple(shortfalls(layout, calibration))))
    # Rounded so that a footprint's two orientations tie (their sums differ in the 16th
    # digit) and the spots' order decides, as in run 1.
    return sorted(found, key=lambda p: (not p.usable, round(p.score, 9), p.spot))


def score(layout: HeavyHexLayout, calibration: Calibration, round_us: float | None = None) -> float:
    """S: the CZs and readouts of one round, plus the data's dephasing over it.

    There are no resets in the follow-up, so they cost nothing.
    """
    t = ROUND_US[layout.distance] if round_us is None else round_us
    qubits = chip_qubits(layout)
    total = 0.0
    for name, *used in _round(layout.distance):
        if name == "cx":
            total += calibration.cz.get(tuple(sorted(qubits[q] for q in used))) or 1.0
        elif name == "measure":
            total += calibration.readout.get(qubits[used[0]], 1.0)
    return total + sum(t / (2 * calibration.t2_us[q]) for q in layout.data.values())


def shortfalls(layout: HeavyHexLayout, calibration: Calibration) -> list[Shortfall]:
    """Every part the patch uses beyond the thresholds: qubits, then couplers."""
    found = []
    for q, role in sorted(_roles(layout).items()):
        t1, t2, readout = calibration.t1_us[q], calibration.t2_us[q], calibration.readout[q]
        if t1 < MIN_T1_US:
            found.append(Shortfall(q, role, "T1", t1))
        if t2 < (MIN_DATA_T2_US if role == "data" else MIN_T2_US):
            found.append(Shortfall(q, role, "T2", t2))
        if readout > MAX_READOUT:
            found.append(Shortfall(q, role, "readout", readout))
    for pair in sorted({(min(a, b), max(a, b)) for _, a, b in layout.couplings}):
        error = calibration.cz.get(pair) or 1.0
        if error > MAX_CZ:
            found.append(Shortfall(pair, "coupler", "CZ", error))
    return found


def by_role(layout: HeavyHexLayout) -> dict[str, list[int]]:
    """The chip qubits in each role. The flags are the X ancillas that flag a Z4 gauge."""
    z4_arms = qubit_roles(build_operators(layout.distance)).z4_arms
    flags = {layout.x_ancillas[gauge] for arms in z4_arms.values() for _, gauge in arms}
    return {
        "data": sorted(layout.data.values()),
        "x_ancillas": sorted(layout.x_ancillas.values()),
        "z_ancillas": sorted(layout.z_ancillas.values()),
        "flags": sorted(flags),
        "relays": sorted(layout.relays),
    }


def _roles(layout: HeavyHexLayout) -> dict[int, str]:
    qubits = by_role(layout)
    role = dict.fromkeys(qubits["data"], "data")
    role |= dict.fromkeys(qubits["x_ancillas"], "X ancilla")
    role |= dict.fromkeys(qubits["flags"], "flag")
    role |= dict.fromkeys(qubits["z_ancillas"], "Z ancilla")
    return role | dict.fromkeys(qubits["relays"], "relay")


@cache
def _round(distance: int) -> tuple[tuple, ...]:
    return tuple(one_round(build_operators(distance)))

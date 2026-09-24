"""Pull IBM's calibration and pick where the patches sit on the chip.

IBM recalibrates about once a day. `calibrate()` pulls the latest numbers and
keeps them as the last calibration. Hardware runs get their qubits from
`fez_qubits()`, which pulls again first if the last calibration isn't from
today. Anything IBM marks broken can't be used; among the spots that avoid it
we take the one expected to make the fewest errors.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

from .layout import HeavyHexLayout, build_layout

BROKEN = 0.5  # IBM reports a coupler or readout it can't use with error 1.0
# Working but weak: several times worse than Fez's median (CZ ~0.3%, readout ~1%,
# T1 ~130 us, T2 ~95 us).
WEAK_CZ = 0.01
WEAK_READOUT = 0.05
WEAK_T1_US = 50
WEAK_T2_US = 30
ROUND_US = 9  # one round at d=3 or d=5, from compiling the flagged circuit for Fez
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# The repo's docs/figures (the package is installed editable).
FIGURES = Path(__file__).resolve().parents[3] / "docs" / "figures"
LAST_CALIBRATION = FIGURES / "fez_calibration.json"
FEZ_MAP = FIGURES / "fez_map.json"


@dataclass(frozen=True)
class Calibration:
    backend: str
    calibrated: str  # when IBM last calibrated
    pulled: str  # when we fetched it, local time
    cz: dict[tuple[int, int], float | None]  # coupler (low, high) -> CZ error
    readout: dict[int, float]
    t1_us: dict[int, float]
    t2_us: dict[int, float]
    broken_qubits: tuple[int, ...] = ()

    @classmethod
    def fetch(cls, backend: str) -> Calibration:
        """Ask IBM for the latest numbers. Read-only, uses no QPU time."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        device = QiskitRuntimeService().backend(backend)
        target = device.target
        qubits = range(device.num_qubits)
        return cls(
            backend=backend,
            calibrated=str(device.properties().last_update_date),
            pulled=datetime.now().astimezone().isoformat(timespec="seconds"),
            cz={tuple(sorted(pair)): gate.error for pair, gate in target["cz"].items()},
            readout={q: target["measure"][(q,)].error for q in qubits},
            t1_us={q: (target.qubit_properties[q].t1 or 0) * 1e6 for q in qubits},
            t2_us={q: (target.qubit_properties[q].t2 or 0) * 1e6 for q in qubits},
            broken_qubits=tuple(device.properties().faulty_qubits()),
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        saved = json.loads(Path(path).read_text())
        return cls(
            backend=saved["backend"],
            calibrated=saved["calibrated"],
            pulled=saved["pulled"],
            cz={tuple(map(int, pair.split("-"))): error for pair, error in saved["cz"].items()},
            readout={int(q): error for q, error in saved["readout"].items()},
            t1_us={int(q): t for q, t in saved["t1_us"].items()},
            t2_us={int(q): t for q, t in saved["t2_us"].items()},
            broken_qubits=tuple(saved["broken_qubits"]),
        )

    def save(self, path: str | Path) -> None:
        saved = {
            "backend": self.backend,
            "calibrated": self.calibrated,
            "pulled": self.pulled,
            "broken_qubits": list(self.broken_qubits),
            "cz": {f"{a}-{b}": error for (a, b), error in sorted(self.cz.items())},
            "readout": {str(q): error for q, error in sorted(self.readout.items())},
            "t1_us": {str(q): t for q, t in sorted(self.t1_us.items())},
            "t2_us": {str(q): t for q, t in sorted(self.t2_us.items())},
        }
        Path(path).write_text(json.dumps(saved, indent=1) + "\n")


def calibrate(backend: str = "ibm_fez", path: str | Path = LAST_CALIBRATION) -> Calibration:
    """Pull IBM's latest numbers (read-only) and keep them as the last calibration."""
    calibration = Calibration.fetch(backend)
    calibration.save(path)
    return calibration


def todays_calibration(
    backend: str = "ibm_fez", path: str | Path = LAST_CALIBRATION
) -> Calibration:
    """The last calibration if it was pulled today, otherwise a fresh one."""
    if Path(path).exists():
        calibration = Calibration.load(path)
        pulled = datetime.fromisoformat(calibration.pulled).date()
        if calibration.backend == backend and pulled == date.today():
            return calibration
    return calibrate(backend, path)


class Spot(NamedTuple):
    origin: tuple[int, int]
    direction: tuple[int, int]


def spots(distance: int, coords: list[list[int]], edges: list) -> dict[Spot, HeavyHexLayout]:
    """Every place, facing every way, where the patch fits on the chip."""
    found = {}
    for direction in DIRECTIONS:
        for x, y in sorted(map(tuple, coords)):
            try:
                layout = build_layout(distance, coords, edges, origin=(x, y), direction=direction)
            except ValueError:
                continue
            found[Spot((x, y), direction)] = layout
    return found


def broken_coupler(error: float | None) -> bool:
    return error is None or error >= BROKEN


def broken_qubit(q: int, calibration: Calibration) -> bool:
    return (
        q in calibration.broken_qubits
        or calibration.readout.get(q, 1) >= BROKEN
        or not calibration.t1_us.get(q)
        or not calibration.t2_us.get(q)
    )


def problems(layout: HeavyHexLayout, calibration: Calibration) -> list[str]:
    """Everything the patch uses that IBM marks broken."""
    found = [
        f"coupler {a}-{b} is broken"
        for a, b in _couplers(layout)
        if broken_coupler(calibration.cz.get((a, b)))
    ]
    found += [
        f"qubit {q} is broken"
        for q in sorted(layout.physical_qubits)
        if broken_qubit(q, calibration)
    ]
    return found


def weak(layout: HeavyHexLayout, calibration: Calibration) -> list[str]:
    """Everything the patch uses that works but is much worse than the rest of the chip."""
    found = []
    for a, b in _couplers(layout):
        error = calibration.cz.get((a, b))
        if error is not None and WEAK_CZ < error < BROKEN:
            found.append(f"coupler {a}-{b}: CZ error {error:.1%}")
    for q in sorted(layout.physical_qubits):
        readout = calibration.readout.get(q, 0)
        t1, t2 = calibration.t1_us.get(q), calibration.t2_us.get(q)
        if WEAK_READOUT < readout < BROKEN:
            found.append(f"qubit {q}: readout error {readout:.1%}")
        if t1 and t1 < WEAK_T1_US:
            found.append(f"qubit {q}: T1 {t1:.0f} us")
        if t2 and t2 < WEAK_T2_US:
            found.append(f"qubit {q}: T2 {t2:.0f} us")
    return found


def cost(layout: HeavyHexLayout, calibration: Calibration) -> float:
    """Rough errors per round: CZ and readout errors, plus dephasing while idle (t / 2T2)."""
    total = sum(calibration.cz.get(c) or 1.0 for c in _couplers(layout))
    for q in layout.physical_qubits:
        t2 = calibration.t2_us.get(q)
        total += calibration.readout.get(q, 1.0) + (ROUND_US / (2 * t2) if t2 else 1.0)
    return total


def best_spots(coords: list[list[int]], edges: list, calibration: Calibration) -> dict[int, Spot]:
    """The d=3 and d=5 spots with the fewest broken parts between them, then the lowest cost.

    The two patches can't share a qubit or sit right next to each other.
    """
    scored = {
        d: {
            spot: (layout, len(problems(layout, calibration)), cost(layout, calibration))
            for spot, layout in spots(d, coords, edges).items()
        }
        for d in (3, 5)
    }
    pairs = []
    for d5, (layout5, broken5, cost5) in scored[5].items():
        near = set(layout5.physical_qubits)
        for a, b in edges:
            if a in layout5.physical_qubits or b in layout5.physical_qubits:
                near |= {a, b}
        for d3, (layout3, broken3, cost3) in scored[3].items():
            if not layout3.physical_qubits & near:
                pairs.append((broken3 + broken5, cost3 + cost5, d3, d5))
    if not pairs:
        raise ValueError("the d=3 and d=5 patches don't fit on the chip together")
    _, _, d3, d5 = min(pairs)
    return {3: d3, 5: d5}


def fez_qubits(distance: int, calibration: Calibration | None = None) -> list[int]:
    """The chip qubit for each qubit of the flagged circuit, from today's calibration.

    Hardware runs get their qubits here, so they always use today's numbers.
    Refuses a placement that uses anything IBM marks broken, and warns about
    parts that work but are weak.
    """
    from ..circuits.flagged import qubit_roles
    from .operators import build_operators

    calibration = calibration or todays_calibration()
    device = json.loads(FEZ_MAP.read_text())
    coords, edges = device["coords"], device["edges"]
    spot = best_spots(coords, edges, calibration)[distance]
    layout = build_layout(distance, coords, edges, origin=spot.origin, direction=spot.direction)
    broken = problems(layout, calibration)
    if broken:
        raise ValueError(
            f"d={distance} on {calibration.backend} would use broken parts: {', '.join(broken)}"
        )
    weak_parts = weak(layout, calibration)
    if weak_parts:
        warnings.warn(
            f"d={distance} on {calibration.backend} uses weak parts: {'; '.join(weak_parts)}",
            stacklevel=2,
        )
    roles = qubit_roles(build_operators(distance))
    qubits = {label - 1: q for label, q in layout.data.items()}
    qubits.update({i: layout.x_ancillas[name] for name, i in roles.x_ancillas.items()})
    qubits.update({i: layout.z_ancillas[name] for name, i in roles.z_ancillas.items()})
    bonds = {frozenset(edge) for edge in edges}
    for name, arms in roles.z2_arms.items():
        ancilla = layout.z_ancillas[name]
        for data, relay in arms:
            (qubits[relay],) = (
                q
                for q in layout.relays
                if frozenset((qubits[data], q)) in bonds and frozenset((q, ancilla)) in bonds
            )
    return [qubits[i] for i in range(roles.num_qubits)]


def _couplers(layout: HeavyHexLayout) -> list[tuple[int, int]]:
    return sorted({tuple(sorted((a, b))) for _, a, b in layout.couplings})

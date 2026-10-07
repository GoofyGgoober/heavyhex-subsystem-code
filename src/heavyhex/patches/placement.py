"""Pull IBM's calibration and pick where the patches sit on the chip.

IBM updates its calibration one or more times a day. `calibrate()` pulls the
latest numbers and keeps them as the last calibration. Anything IBM marks broken
can't be used. `day_plan()` turns a calibration into the day's reps: one per
clean place for the d=5 patch, best first, each with the d=3 patch at a clean
place of its own. Hardware runs get their qubits from `fez_qubits()`, which
pulls again first if the last calibration isn't from today and refuses broken
parts.
"""

from __future__ import annotations

import json
import statistics
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

from ..circuits.flagged import one_round, qubit_roles
from .layout import HeavyHexLayout, build_layout
from .operators import build_operators

BROKEN = 0.5  # IBM reports a coupler or readout it can't use with error 1.0
# Working but weak: several times worse than Fez's median (CZ ~0.3%, readout ~1%,
# T1 ~130 us, T2 ~95 us).
WEAK_CZ = 0.01
WEAK_READOUT = 0.05
WEAK_T1_US = 50
WEAK_T2_US = 30
ROUND_US = 9  # one round at d=3 or d=5, from compiling the flagged circuit for Fez
MAX_REPS = 2  # reps a day at most: one per clean place for the d=5 patch, best first
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
    durations_ns: dict[str, int] | None = None  # median x, cz, measure and reset times
    single_qubit: dict[int, float] | None = None  # each qubit's SX error; older files lack it

    @classmethod
    def fetch(cls, backend: str) -> Calibration:
        """Ask IBM for the latest numbers. Read-only, uses no QPU time."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        return cls.from_device(QiskitRuntimeService().backend(backend))

    @classmethod
    def from_device(cls, device, at: datetime | None = None) -> Calibration:
        """The numbers IBM had in force at a time, or its latest. Read-only.

        Gate times are always the device's current ones.
        """
        target = device.target
        qubits = range(device.num_qubits)

        def median_ns(gate: str) -> int:
            return round(statistics.median(p.duration for p in target[gate].values() if p) * 1e9)

        pulled = datetime.now().astimezone().isoformat(timespec="seconds")
        durations = {gate: median_ns(gate) for gate in ("x", "cz", "measure", "reset")}
        if at is None:
            return cls(
                backend=device.name,
                calibrated=str(device.properties().last_update_date),
                pulled=pulled,
                cz={tuple(sorted(pair)): gate.error for pair, gate in target["cz"].items()},
                readout={q: target["measure"][(q,)].error for q in qubits},
                t1_us={q: (target.qubit_properties[q].t1 or 0) * 1e6 for q in qubits},
                t2_us={q: (target.qubit_properties[q].t2 or 0) * 1e6 for q in qubits},
                broken_qubits=tuple(device.properties().faulty_qubits()),
                durations_ns=durations,
                single_qubit={
                    q: sx.error
                    for q in qubits
                    if (sx := target["sx"].get((q,))) and sx.error is not None
                },
            )
        properties = device.properties(datetime=at)
        if properties is None:
            raise ValueError(f"IBM has no calibration of {device.name} for {at}")

        def known(read, *args) -> float:
            try:
                return read(*args) or 0.0
            except Exception:  # IBM leaves out what it didn't measure
                return 0.0

        pairs = {tuple(sorted(g.qubits)) for g in properties.gates if g.gate == "cz"}
        return cls(
            backend=device.name,
            calibrated=str(properties.last_update_date),
            pulled=pulled,
            cz={
                pair: properties.gate_error("cz", list(pair))
                if properties.is_gate_operational("cz", list(pair))
                else 1.0
                for pair in pairs
            },
            readout={q: known(properties.readout_error, q) or 1.0 for q in qubits},
            t1_us={q: known(properties.t1, q) * 1e6 for q in qubits},
            t2_us={q: known(properties.t2, q) * 1e6 for q in qubits},
            broken_qubits=tuple(properties.faulty_qubits()),
            durations_ns=durations,
            single_qubit={
                q: error for q in qubits if (error := known(properties.gate_error, "sx", [q]))
            },
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
            durations_ns=saved.get("durations_ns"),
            single_qubit={int(q): e for q, e in saved["single_qubit"].items()}
            if saved.get("single_qubit")
            else None,
        )

    def save(self, path: str | Path) -> None:
        saved = {
            "backend": self.backend,
            "calibrated": self.calibrated,
            "pulled": self.pulled,
            "broken_qubits": list(self.broken_qubits),
            "durations_ns": self.durations_ns,
            "cz": {f"{a}-{b}": error for (a, b), error in sorted(self.cz.items())},
            "readout": {str(q): error for q, error in sorted(self.readout.items())},
            "t1_us": {str(q): t for q, t in sorted(self.t1_us.items())},
            "t2_us": {str(q): t for q, t in sorted(self.t2_us.items())},
            "single_qubit": {str(q): e for q, e in sorted(self.single_qubit.items())}
            if self.single_qubit
            else None,
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
        or (calibration.single_qubit or {}).get(q, 0) >= BROKEN
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
    """Rough errors per round: every CZ, measurement and reset a round makes, plus decay of the data.

    Parts count as often as a round uses them. The data idles through every
    ancilla measurement and reset; with decoupling it decays at the T1 rate.
    """
    qubits = chip_qubits(layout)
    total = 0.0
    for name, *used in one_round(build_operators(layout.distance)):
        if name == "cx":
            total += calibration.cz.get(tuple(sorted(qubits[q] for q in used))) or 1.0
        elif name in ("measure", "reset"):
            total += calibration.readout.get(qubits[used[0]], 1.0)
    for q in layout.data.values():
        t1 = calibration.t1_us.get(q)
        total += ROUND_US / (2 * t1) if t1 else 1.0
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
        near = neighbourhood(layout5, edges)
        for d3, (layout3, broken3, cost3) in scored[3].items():
            if not layout3.physical_qubits & near:
                # Rounded so that equal costs summed in a different order tie (a footprint's
                # two orientations differ in the 16th digit), and the spots' order decides.
                pairs.append((broken3 + broken5, round(cost3 + cost5, 9), d3, d5))
    if not pairs:
        raise ValueError("the d=3 and d=5 patches don't fit on the chip together")
    _, _, d3, d5 = min(pairs)
    return {3: d3, 5: d5}


def neighbourhood(layout: HeavyHexLayout, edges: list) -> set[int]:
    """The patch's qubits and every qubit coupled to one of them."""
    near = set(layout.physical_qubits)
    for a, b in edges:
        if a in layout.physical_qubits or b in layout.physical_qubits:
            near |= {a, b}
    return near


def clean_places(
    distance: int, coords: list[list[int]], edges: list, calibration: Calibration
) -> list[tuple[Spot, HeavyHexLayout]]:
    """Every place the patch fits with no broken part, at its best spot, best first.

    A place is a set of chip qubits; its two orientations are two spots on it. Best
    means the lowest cost, with equal costs settled by the spots' order.
    """
    best: dict[frozenset[int], tuple[tuple[float, Spot], HeavyHexLayout]] = {}
    for spot, layout in spots(distance, coords, edges).items():
        if problems(layout, calibration):
            continue
        key = (round(cost(layout, calibration), 9), spot)
        place = frozenset(layout.physical_qubits)
        if place not in best or key < best[place][0]:
            best[place] = (key, layout)
    return [(key[1], layout) for key, layout in sorted(best.values(), key=lambda kept: kept[0])]


def day_plan(
    coords: list[list[int]], edges: list, calibration: Calibration, reps: int = MAX_REPS
) -> list[list[dict[int, Spot]]]:
    """The day's reps: one per clean d=5 place, best first, at most `reps`.

    Each rep puts the d=3 patch on a clean place no earlier rep that day used: the
    best one that doesn't touch the d=5 patch, both in one job. If there is none,
    the d=5 patch runs alone and the d=3 patch in a job of its own straight after,
    at the best clean place not yet used that day. Returns each rep's jobs, in the
    order they are sent, each job as {distance: spot}.
    """
    threes = clean_places(3, coords, edges, calibration)
    if not threes:
        raise ValueError(f"no clean place for the d=3 patch on {calibration.backend}")
    used: set[frozenset[int]] = set()
    plan = []
    for spot5, layout5 in clean_places(5, coords, edges, calibration)[:reps]:
        near = neighbourhood(layout5, edges)
        fresh = [t for t in threes if frozenset(t[1].physical_qubits) not in used] or threes
        beside = [t for t in fresh if not t[1].physical_qubits & near]
        spot3, layout3 = (beside or fresh)[0]
        used.add(frozenset(layout3.physical_qubits))
        plan.append([{3: spot3, 5: spot5}] if beside else [{5: spot5}, {3: spot3}])
    return plan


def fez_plan(calibration: Calibration | None = None, reps: int = MAX_REPS) -> list:
    """day_plan on Fez, from today's calibration unless one is given."""
    calibration = calibration or todays_calibration()
    device = json.loads(FEZ_MAP.read_text())
    return day_plan(device["coords"], device["edges"], calibration, reps)


def fez_qubits(
    distance: int, calibration: Calibration | None = None, spot: Spot | None = None
) -> list[int]:
    """The chip qubit for each qubit of the flagged circuit, from today's calibration.

    At `spot`, or the best spot of the best pair if none is given. Hardware runs
    get their qubits here, so they always use today's numbers. Refuses a placement
    that uses anything IBM marks broken, and warns about parts that work but are weak.
    """
    calibration = calibration or todays_calibration()
    device = json.loads(FEZ_MAP.read_text())
    coords, edges = device["coords"], device["edges"]
    spot = spot or best_spots(coords, edges, calibration)[distance]
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
    return chip_qubits(layout)


def chip_qubits(layout: HeavyHexLayout) -> list[int]:
    """The chip qubit for each qubit of the flagged circuit."""
    roles = qubit_roles(build_operators(layout.distance))
    qubits = {label - 1: q for label, q in layout.data.items()}
    qubits.update({i: layout.x_ancillas[name] for name, i in roles.x_ancillas.items()})
    qubits.update({i: layout.z_ancillas[name] for name, i in roles.z_ancillas.items()})
    couplings = {frozenset((a, b)) for _, a, b in layout.couplings}
    for name, arms in roles.z2_arms.items():
        ancilla = layout.z_ancillas[name]
        for data, relay in arms:
            (qubits[relay],) = (
                q
                for q in layout.relays
                if {frozenset((qubits[data], q)), frozenset((q, ancilla))} <= couplings
            )
    return [qubits[i] for i in range(roles.num_qubits)]


def _couplers(layout: HeavyHexLayout) -> list[tuple[int, int]]:
    return sorted({tuple(sorted((a, b))) for _, a, b in layout.couplings})

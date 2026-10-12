"""Check qubit overlap of the two d=5 placements and which d=5 places were clean each day.
Read-only use of the repo's placement code and saved calibrations (no network)."""
import json
from pathlib import Path
from heavyhex.patches.placement import FEZ_MAP, Calibration, build_layout, clean_places

ROOT = Path(__file__).resolve().parents[4]
dev = json.loads(FEZ_MAP.read_text())
coords, edges = dev["coords"], dev["edges"]
def qubits(d, origin, direction):
    lay = build_layout(d, coords, edges, origin=tuple(origin), direction=tuple(direction))
    return set(lay.physical_qubits), lay
q1, _ = qubits(5, (3, 7), (1, 1))
q2, _ = qubits(5, (3, 13), (1, -1))
print("d5 p1 qubits", len(q1), "p2", len(q2), "shared", len(q1 & q2))
for o, dirn in [((5, 15), (1, -1)), ((3, 11), (1, 1)), ((7, 13), (1, -1)), ((3, 5), (1, -1))]:
    q3, _ = qubits(3, o, dirn)
    print("d3", o, dirn, len(q3), "within p1 d5:", q3 <= q1, "overlap p1", len(q3 & q1), "overlap p2", len(q3 & q2))
for day in ["2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]:
    cal = Calibration.load(ROOT / "runs" / f"ibm_fez-{day}-r1" / "calibration.json")
    places = clean_places(5, coords, edges, cal)
    print(day, "clean d5 places:", [(tuple(s.origin), tuple(s.direction)) for s, _ in places])

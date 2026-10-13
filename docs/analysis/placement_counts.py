"""How many placements Fez has, and how the chosen distance-3 patches rank (Sec. design).

The prespecified rule (src/heavyhex/patches/placement.py) searches the patch in the two
orientations that line up with the chip's lattice: `spots()` tries four directions, two per
qubit set. A patch also fits wherever its coupling graph embeds in the chip's, in orientations
the rule never tries. This script counts both, with the rule's own test for broken parts, on
each day's calibration:

  searched      qubit sets the rule searches, and how many are usable (no broken part)
  all           qubit sets of every embedding of the patch's coupling graph (networkx subgraph
                monomorphisms), and how many are usable
  rank_d3_rep2  repetition 2's distance-3 patch ranked by the rule's score among all usable
                sets of the day, the set repetition 1 used that day left out; each set at its
                best embedding
  stricter      distance-5 placements that would pass bars stricter than the rule's: every
                CZ error below 5%, every readout error below 10%, every T2 above 10 us
  relay_q51     placement 2's relay q51: its reported readout error by day, and on each day
                the fraction of eight-round Z-memory rounds in which it read 1 and in which the
                detector of the stabilizer it serves fired (runs/*/analysis.json)
  weak          every patch's weak parts on every day, by the rule's own test (placement.weak:
                CZ error above 1%, readout error above 5%, T1 below 50 us, T2 below 30 us), and
                the reported values of the parts that recur in placement 1
  pair2_d3      where pair 2's distance-3 patch sat: inside placement 1, its overlap with
                placement 2, and whether it used day 1's qubits
  round_us      the length of one round in each job, from the stored circuit durations:
                (8-round minus 4-round X-memory circuit) / 4

The score of an embedding is the rule's cost() with every chip qubit moved by the embedding.
Reads runs/ and docs/figures/fez_map.json; seconds.

    python docs/analysis/placement_counts.py      # writes results/placement_counts.json
"""

from __future__ import annotations

import json

import networkx as nx
from networkx.algorithms import isomorphism

from heavyhex.circuits.flagged import one_round
from heavyhex.patches.layout import build_layout
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import (
    ROUND_US,
    Calibration,
    broken_coupler,
    broken_qubit,
    chip_qubits,
    spots,
    weak,
)

from common import REPO, RUNS, jobs, memory_index, read, write_json

DAYS = ("2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")


def embedded_cost(distance: int, qubits: list[int], data: list[int], cal: Calibration) -> float:
    """cost() of placement.py for a patch whose circuit qubit i sits on chip qubit qubits[i]."""
    total = 0.0
    for name, *used in one_round(build_operators(distance)):
        if name == "cx":
            total += cal.cz.get(tuple(sorted(qubits[q] for q in used))) or 1.0
        elif name in ("measure", "reset"):
            total += cal.readout.get(qubits[used[0]], 1.0)
    for q in data:
        t1 = cal.t1_us.get(q)
        total += ROUND_US / (2 * t1) if t1 else 1.0
    return total


STRICT = {"CZ error < 5%": 0.05, "readout error < 10%": 0.10, "T2 > 10 us": 10.0}


def stricter(qubit_set, couplers, cal: Calibration) -> dict[str, bool]:
    """Which of the stricter bars the placement passes, and all of them together."""
    passes = {
        "CZ error < 5%": all((cal.cz.get(c) or 1.0) < STRICT["CZ error < 5%"] for c in couplers),
        "readout error < 10%": all(
            cal.readout.get(q, 1.0) < STRICT["readout error < 10%"] for q in qubit_set
        ),
        "T2 > 10 us": all((cal.t2_us.get(q) or 0.0) > STRICT["T2 > 10 us"] for q in qubit_set),
    }
    return passes | {"all three": all(passes.values())}


def data_labels(name: str) -> set[int]:
    """Z4Z5Z9 -> {4, 5, 9}."""
    return {int(x) for x in name.split("Z") if x}


def relay_q51() -> dict:
    """Relay q51 of placement 2's distance-5 patch, day by day."""
    out = {}
    idx = memory_index("Z", 8)
    for job in jobs():
        if not job.name.endswith("r2-d5"):
            continue
        setting = job.analysis["observed"]["settings"][idx]
        rows = list(zip(setting["detectors"], setting["detector_rates"]))
        seen: dict = {}
        read1, gauge = [], ""
        for (patch, rnd, name), rate in rows:
            if name.startswith("relay"):
                k = seen.get((patch, rnd, name), 0)
                seen[(patch, rnd, name)] = k + 1
                if job.readout_qubit(str(patch), name, k) == 51:
                    read1.append(rate)
                    gauge = name.split()[1]
        stabilizer = [
            rate
            for (_, _, name), rate in rows
            if " " not in name and data_labels(gauge) <= data_labels(name)
        ]
        out[job.day] = {
            "reported_readout_error": job.calibration.readout[51],
            "gauge": gauge,
            "read_1_fraction_by_round": read1,
            "stabilizer_detection_fraction_by_round": stabilizer,
        }
    return out


# The parts of placement 1 that were weak on every day (Sec. III).
RECURRING = {
    "cz": ["63-64", "102-103", "148-149", "149-150"],
    "readout": [83, 102],
    "t2_us": [149, 150],
}


def patches(day: str, coords: list, edges: list) -> dict:
    """The day's four patches as laid out: pair 1's two and pair 2's two."""
    out = {}
    for name, folder, d in (
        ("placement 1 (d5)", "r1", "5"),
        ("pair 1 d3", "r1", "3"),
        ("placement 2 (d5)", "r2-d5", "5"),
        ("pair 2 d3", "r2-d3", "3"),
    ):
        run = read(RUNS / f"ibm_fez-{day}-{folder}" / "run.json")
        origin, direction = run["rep"]["spots"][d]
        layout = build_layout(
            int(d), coords, edges, origin=tuple(origin), direction=tuple(direction)
        )
        assert set(layout.physical_qubits) == set(run["chips"][d])
        out[name] = layout
    return out


def weak_by_day(coords: list, edges: list) -> dict:
    """Each patch's weak parts by the rule's test, and placement 1's recurring ones, by day."""
    out: dict = {}
    for day in DAYS:
        cal = Calibration.load(RUNS / f"ibm_fez-{day}-r1" / "calibration.json")
        out[day] = {
            "weak": {
                name: weak(layout, cal) for name, layout in patches(day, coords, edges).items()
            },
            "recurring": {
                "cz": {c: cal.cz.get(tuple(map(int, c.split("-")))) for c in RECURRING["cz"]},
                "readout": {q: cal.readout[q] for q in RECURRING["readout"]},
                "t2_us": {q: cal.t2_us[q] for q in RECURRING["t2_us"]},
            },
        }
    return out


def pair2_d3(coords: list, edges: list) -> dict:
    """Where pair 2's distance-3 patch sat relative to the two distance-5 placements."""
    out = {}
    first = None
    for day in DAYS:
        q = {name: set(lay.physical_qubits) for name, lay in patches(day, coords, edges).items()}
        first = first or q["pair 2 d3"]
        out[day] = {
            "inside placement 1": q["pair 2 d3"] <= q["placement 1 (d5)"],
            "shared with placement 2": len(q["pair 2 d3"] & q["placement 2 (d5)"]),
            "qubits": len(q["pair 2 d3"]),
            "same qubits as day 1": q["pair 2 d3"] == first,
            "placements 1 and 2 share": len(q["placement 1 (d5)"] & q["placement 2 (d5)"]),
            "pair 1 d3 shared with placement 1": len(q["pair 1 d3"] & q["placement 1 (d5)"]),
        }
    return out


def round_us() -> dict:
    """One round's length in each job, from run.json's per-circuit durations, in us."""
    i4, i8 = memory_index("X", 4), memory_index("X", 8)
    return {
        job.name: (job.info["seconds"][i8] - job.info["seconds"][i4]) / 4 * 1e6 for job in jobs()
    }


def usable(qubit_set, couplers, cal: Calibration) -> bool:
    """The rule's test: no broken coupler and no broken qubit."""
    return not any(broken_coupler(cal.cz.get(c)) for c in couplers) and not any(
        broken_qubit(q, cal) for q in qubit_set
    )


def main() -> None:
    device = json.loads((REPO / "docs/figures/fez_map.json").read_text())
    coords, edges = device["coords"], device["edges"]
    chip = nx.Graph([tuple(e) for e in edges])
    cals = {day: Calibration.load(RUNS / f"ibm_fez-{day}-r1" / "calibration.json") for day in DAYS}
    out: dict = {"chip": {"qubits": chip.number_of_nodes(), "couplers": chip.number_of_edges()}}

    for d in (3, 5):
        found = spots(d, coords, edges)
        searched = {frozenset(layout.physical_qubits) for layout in found.values()}
        reference = next(iter(found.values()))
        circuit = chip_qubits(reference)  # circuit qubit -> chip qubit
        data_idx = [circuit.index(q) for q in reference.data.values()]
        pattern = nx.Graph(
            [tuple(sorted((a, b))) for _, a, b in reference.couplings]
        )  # nodes are the reference's chip qubits
        embeddings = []
        for m in isomorphism.GraphMatcher(chip, pattern).subgraph_monomorphisms_iter():
            to_chip = {v: k for k, v in m.items()}
            qubits = [to_chip[q] for q in circuit]
            couplers = sorted({tuple(sorted((to_chip[a], to_chip[b]))) for a, b in pattern.edges})
            embeddings.append((frozenset(qubits), qubits, couplers))
        sets = {e[0] for e in embeddings}
        block: dict = {
            "searched_spots": len(found),
            "searched_sets": len(searched),
            "embeddings": len(embeddings),
            "all_sets": len(sets),
            "searched_subset_of_all": searched <= sets,
            "pattern_qubits": pattern.number_of_nodes(),
            "pattern_couplers": pattern.number_of_edges(),
            "per_day": {},
        }
        for day, cal in cals.items():
            best: dict = {}
            for qset, qubits, couplers in embeddings:
                if not usable(qset, couplers, cal):
                    continue
                c = embedded_cost(d, qubits, [qubits[i] for i in data_idx], cal)
                best[qset] = min(c, best.get(qset, c))
            block["per_day"][day] = {
                "usable_all": len(best),
                "usable_searched": sum(s in best for s in searched),
            }
            if d == 5:
                bars = [stricter(qset, couplers, cal) for qset, _, couplers in embeddings]
                block["per_day"][day]["stricter (usable distance-5 placements passing)"] = {
                    bar: len({e[0] for e, b in zip(embeddings, bars) if b[bar] and e[0] in best})
                    for bar in bars[0]
                }
            if d == 3:
                rep1 = frozenset(read(RUNS / f"ibm_fez-{day}-r1" / "run.json")["chips"]["3"])
                rep2 = frozenset(read(RUNS / f"ibm_fez-{day}-r2-d3" / "run.json")["chips"]["3"])
                ranked = sorted((c, sorted(s)) for s, c in best.items() if s != rep1)
                rank = 1 + [frozenset(s) for _, s in ranked].index(rep2)
                block["per_day"][day]["rank_d3_rep2"] = {
                    "rank_among_all_usable": rank,
                    "of": len(ranked),
                    "chosen_cost": best[rep2],
                    "best_cost": ranked[0][0],
                    "best_is_searched": frozenset(ranked[0][1]) in searched,
                }
        out[f"d{d}"] = block
    out["relay_q51"] = relay_q51()
    out["weak"] = weak_by_day(coords, edges)
    out["pair2_d3"] = pair2_d3(coords, edges)
    rounds = round_us()
    out["round_us"] = {"by_job": rounds, "range": [min(rounds.values()), max(rounds.values())]}
    write_json("placement_counts.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

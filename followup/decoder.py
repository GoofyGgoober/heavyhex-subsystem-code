"""Matching on the model's detector error model, with every edge's probability capped.

pymatching merges the model's errors into one edge per pair of detectors, or per detector and
the boundary, weighted w = ln[(1 - p)/p]. Here each merged p is capped first, at 0.1, so that
w >= 2.2: in run 1, merged edges at p = 0.13-0.20 made two-edge detours cheaper than single
faults. Each single fault of the model, one line of its decomposed error model, is then
decoded alone. If any is miscorrected, the cap drops to 0.05, and then every edge gets the
same weight (CAPS): the rule's cap. CORRELATED, pymatching's correlated matching on the same
model uncapped, is reported beside it; it reads a fault that fires a flag or relay as one,
where decomposed matching sees its pieces.
"""

from __future__ import annotations

import heapq
from functools import partial, reduce
from math import inf
from operator import xor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np
    import pymatching
    import stim

CAPS = (0.1, 0.05, None)  # None: every edge weighs the same
CORRELATED = "correlated"
UNIFORM_P = 0.01  # any p gives equal weights
# Limits of stim's search for the circuit's smallest undetectable logical error.
SEARCH = {
    "dont_explore_detection_event_sets_with_size_above": 6,
    "dont_explore_edges_with_degree_above": 6,
    "dont_explore_edges_increasing_symptom_degree": False,
}


def matching(dem: stim.DetectorErrorModel, cap: float | None = CAPS[0]) -> pymatching.Matching:
    """pymatching's graph for the decomposed model, each edge's p capped (uniform for None).

    The model is first resplit, so that every fault's parts carry its observables.
    """
    import pymatching
    import stim

    merged = pymatching.Matching.from_detector_error_model(resplit(dem))
    lines = []
    for u, v, edge in merged.edges():
        p = UNIFORM_P if cap is None else min(edge["error_probability"], cap)
        detectors = f"D{u}" if v is None else f"D{u} D{v}"
        observables = "".join(f" L{k}" for k in sorted(edge["fault_ids"]))
        lines.append(f"error({p}) {detectors}{observables}")
    lines += [f"detector D{d}" for d in range(dem.num_detectors)]
    lines += [f"logical_observable L{k}" for k in range(dem.num_observables)]
    return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))


def resplit(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """The decomposed model, with each fault that stim splits into edges plus a part holding
    only an observable split into pairs of its detectors instead.

    Matching drops a part without detectors, so such a fault's edges flip the wrong
    observables, whatever their weights: at d=3 in Z memory, a data X between the CZs of a
    Z2 gadget fires a stabilizer twice, a relay and a flag, and only boundary edges join
    them. Its detectors are paired (an odd one with the boundary) as near one another in the
    graph of the other faults as they can be; each pair carries the observables of its
    shortest path there, but the farthest, which carries what is left of the fault's.
    """
    import stim

    lines = _lines(dem)
    split = [line for line in lines if _split_badly(line)]
    if not split:
        return dem
    whole = [line for line in lines if not _split_badly(line)]
    boundary = dem.num_detectors
    graph: dict[int, list[tuple[int, int]]] = {}
    for line in whole:
        for detectors, mask in _parts(line) if line.type == "error" else []:
            a, b = (*detectors, boundary)[:2]
            graph.setdefault(a, []).append((b, mask))
            graph.setdefault(b, []).append((a, mask))
    text = [str(line) for line in whole]
    for line in split:
        detectors, mask = _symptom(line)
        nodes = sorted(detectors) + [boundary] * (len(detectors) % 2)
        paths = {n: _paths(graph, n) for n in nodes}
        pairs = min(_pairings(nodes), key=lambda ps: sum(paths[a][b][0] for a, b in ps))
        masks = [paths[a][b][1] for a, b in pairs]
        far = max(range(len(pairs)), key=lambda k: paths[pairs[k][0]][pairs[k][1]][0])
        masks[far] ^= mask ^ reduce(xor, masks, 0)
        pieces = [
            " ".join(
                [f"D{n}" for n in pair if n != boundary]
                + [f"L{k}" for k in range(dem.num_observables) if masks[i] >> k & 1]
            )
            for i, pair in enumerate(pairs)
        ]
        text.append(f"error({line.args_copy()[0]}) " + " ^ ".join(pieces))
    return stim.DetectorErrorModel("\n".join(text))


def _lines(dem: stim.DetectorErrorModel) -> list[stim.DemInstruction]:
    """The model's instructions, repeat blocks unrolled."""
    import stim

    flat = dem.flattened()
    every = (flat[k] for k in range(len(flat)))
    return [line for line in every if isinstance(line, stim.DemInstruction)]


def _parts(line: stim.DemInstruction) -> list[tuple[list[int], int]]:
    """The error line's parts: each one's detectors and its observables as a bit mask."""
    import stim

    parts: list[tuple[list[int], int]] = [([], 0)]
    for target in line.targets_copy():
        if not isinstance(target, stim.DemTarget):
            continue
        detectors, mask = parts[-1]
        if target.is_separator():
            parts.append(([], 0))
        elif target.is_relative_detector_id():
            detectors.append(target.val)
        elif target.is_logical_observable_id():
            parts[-1] = (detectors, mask ^ 1 << target.val)
    return parts


def _symptom(line: stim.DemInstruction) -> tuple[set[int], int]:
    """The detectors a fault fires and the observables it flips: its parts XORed."""
    detectors: set[int] = set()
    mask = 0
    for found, part in _parts(line):
        detectors ^= set(found)
        mask ^= part
    return detectors, mask


def _split_badly(line: stim.DemInstruction) -> bool:
    """An error that stim split into parts with detectors and a part with none."""
    parts = _parts(line) if line.type == "error" else []
    return any(not d for d, _ in parts) and any(d for d, _ in parts)


def _paths(graph: dict[int, list[tuple[int, int]]], start: int) -> dict[int, tuple[float, int]]:
    """Breadth first from start: each node's distance in edges and the observables of one
    shortest path to it."""
    found = {start: (0.0, 0)}
    queue = [start]
    for node in queue:
        d, mask = found[node]
        for other, m in graph.get(node, []):
            if other not in found:
                found[other] = (d + 1, mask ^ m)
                queue.append(other)
    return found


def _pairings(nodes: list[int]):
    """Every way to split the nodes, an even number, into pairs."""
    if not nodes:
        yield []
        return
    first, rest = nodes[0], nodes[1:]
    for k, other in enumerate(rest):
        for more in _pairings(rest[:k] + rest[k + 1 :]):
            yield [(first, other), *more]


def decode(m: pymatching.Matching, detectors: np.ndarray, correlated: bool = False) -> np.ndarray:
    """The predicted observable flips of each shot."""
    import numpy as np

    return np.asarray(m.decode_batch(detectors, enable_correlations=correlated), dtype=bool)


def decoders(
    dem: stim.DetectorErrorModel, caps: Iterable[float | str | None]
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Each cap's decoder (or CORRELATED's), keyed str(cap): the shots' detectors to their
    predicted observable flips."""
    import pymatching

    found = {}
    for cap in caps:
        if cap == CORRELATED:
            m = pymatching.Matching.from_detector_error_model(
                resplit(dem), enable_correlations=True
            )
            found[CORRELATED] = partial(decode, m, correlated=True)
        else:
            found[str(cap)] = partial(decode, matching(dem, cap))  # type: ignore[arg-type]
    return found


def faults(dem: stim.DetectorErrorModel) -> tuple[np.ndarray, np.ndarray]:
    """Every single fault of the model: the detectors it fires and the observables it flips.

    One per error line; a line's parts (^) are XORed, giving the fault's whole symptom.
    """
    import numpy as np

    lines = [line for line in _lines(dem) if line.type == "error"]
    fired = np.zeros((len(lines), dem.num_detectors), dtype=bool)
    flips = np.zeros((len(lines), dem.num_observables), dtype=bool)
    for i, line in enumerate(lines):
        detectors, mask = _symptom(line)
        fired[i, sorted(detectors)] = True
        flips[i] = [mask >> k & 1 for k in range(dem.num_observables)]
    return fired, flips


def miscorrected(dem: stim.DetectorErrorModel, m: pymatching.Matching) -> np.ndarray:
    """The single faults the decoder gets wrong, by error line; an undetectable logical
    fault (no detector, an observable flipped) counts too."""
    import numpy as np

    fired, flips = faults(dem)
    return np.flatnonzero((decode(m, fired) != flips).any(axis=1))


def capped(
    dems: Iterable[stim.DetectorErrorModel], caps: Sequence[float | None] = CAPS
) -> tuple[float | None, dict]:
    """The first cap whose decoder corrects every single fault of every model, and how many
    each cap tried miscorrects per model. Raises if the last one miscorrects any."""
    dems = list(dems)
    counts: dict = {}
    for cap in caps:
        counts[cap] = [len(miscorrected(dem, matching(dem, cap))) for dem in dems]
        if not any(counts[cap]):
            return cap, counts
    raise ValueError(f"single faults miscorrected at every cap: {counts}")


def graph_distance(m: pymatching.Matching, observable: int = 0, weighted: bool = False) -> float:
    """The fewest edges (or least weight) in a logical error of the matching graph: a cycle,
    through the boundary or not, that flips the observable an odd number of times."""
    boundary = m.num_detectors
    neighbours: list[list[tuple[int, float, int]]] = [[] for _ in range(boundary + 1)]
    for u, v, edge in m.edges():
        v = boundary if v is None else v
        w = edge["weight"] if weighted else 1.0
        odd = int(observable in edge["fault_ids"])
        neighbours[u].append((v, w, odd))
        neighbours[v].append((u, w, odd))
    best = inf
    for start in range(boundary + 1):
        dist = {(start, 0): 0.0}
        heap = [(0.0, start, 0)]
        while heap:
            d, node, parity = heapq.heappop(heap)
            if d >= best or d > dist[node, parity]:
                continue
            if node == start and parity:
                best = d
                break
            for other, w, odd in neighbours[node]:
                key = (other, parity ^ odd)
                if d + w < dist.get(key, inf):
                    dist[key] = d + w
                    heapq.heappush(heap, (d + w, *key))
    return best


def circuit_distance(circuit: stim.Circuit) -> tuple[int, int]:
    """The fewest faults in an undetectable logical error: graphlike (stim's exact search on
    the decomposed graph), and over all faults (stim's search, within SEARCH's limits)."""
    graphlike = len(circuit.shortest_graphlike_error())
    found = circuit.search_for_undetectable_logical_errors(**SEARCH)
    return graphlike, len(found)

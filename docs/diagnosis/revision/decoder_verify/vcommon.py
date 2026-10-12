"""Shared helpers for the independent verification (own code; uses only the cache and repo models)."""

from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CACHE = HERE / "cache"
sys.path[:0] = [str(REPO / "src")]

ROUNDS = (1, 2, 3, 4, 6, 8)
MEM = {("X", n): i for i, n in enumerate(ROUNDS)} | {("Z", n): 6 + i for i, n in enumerate(ROUNDS)}


def folders() -> list[str]:
    return sorted(p.name for p in CACHE.iterdir() if (p / "meta.json").exists())


def meta(folder: str) -> dict:
    return json.loads((CACHE / folder / "meta.json").read_text())


@functools.lru_cache(maxsize=2)
def _job_inputs(folder: str):
    from heavyhex.experiment import run as R
    from heavyhex.patches.placement import Calibration

    path = REPO / "runs" / folder
    run = json.loads((path / "run.json").read_text())
    analysis = json.loads((path / "analysis.json").read_text())
    return R.load_circuits(path), Calibration.load(path / "calibration.json"), run["dt"], analysis["decoupled_dephasing"]


def model(folder: str, idx: int) -> stim.Circuit:
    """The repo's exact stim model (rebuilt; the cached .stim text rounds probabilities to 6 digits,
    which breaks stim's exact PAULI_CHANNEL_1 handling, so it is NOT used)."""
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS

    circuits, cal, dt, f = _job_inputs(folder)
    m, _, _ = R.simulated(SETTINGS[idx], circuits[idx], cal, dt, f)
    return m


def chip(folder: str, idx: int):
    z = np.load(CACHE / folder / f"chip_{idx}.npz")
    return z["det"], z["obs"]


def cal_matching(m: stim.Circuit) -> pymatching.Matching:
    """The prespecified decoder: pymatching on stim's decomposed DEM, default options."""
    return pymatching.Matching.from_detector_error_model(m.detector_error_model(decompose_errors=True))


def uni_matching(cal: pymatching.Matching) -> pymatching.Matching:
    """Same graph (nodes, edges, boundary edges, fault_ids), every weight 1.

    Built differently from the audited code: a fresh Matching with add_edge /
    add_boundary_edge copied edge by edge (no DEM round-trip).
    """
    m = pymatching.Matching()
    for u, v, a in cal.edges():
        if v is None:
            m.add_boundary_edge(u, fault_ids=set(a["fault_ids"]), weight=1.0)
        else:
            m.add_edge(u, v, fault_ids=set(a["fault_ids"]), weight=1.0)
    assert m.num_detectors == cal.num_detectors, (m.num_detectors, cal.num_detectors)
    assert m.num_edges == cal.num_edges
    return m


def patch_of_detector(labels) -> np.ndarray:
    return np.array([str(p) for p, _, _ in labels])


def weight(p: float) -> float:
    return math.log((1 - p) / p)

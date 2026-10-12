"""Shared helpers for the decoder audit (revision, exploratory; no QPU use).

Everything is built with the repository's own code: the stim model of each saved
memory circuit comes from docs/diagnosis/common.py's Job.model, which calls
heavyhex.experiment.run.simulated with the job's calibration.json, dt and the job's
measured f, exactly as heavyhex.experiment.run.analyze does. The calibrated-weight
decoder is the repository's run._decode (pymatching on the decomposed stim DEM).
The uniform-weight decoder is the same matching graph (same nodes, same edges, same
boundary edges, same observable labels per edge) with every edge weight set equal.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "docs" / "diagnosis"), str(REPO / "src")]

import numpy as np  # noqa: E402
import pymatching  # noqa: E402
import stim  # noqa: E402

from common import jobs, memory_index  # noqa: E402,F401
from heavyhex.experiment import run as R  # noqa: E402,F401
from heavyhex.experiment.circuits import ROUNDS, SETTINGS  # noqa: E402,F401

UNIFORM_P = 0.01  # any constant p gives equal weights; only the ordering matters to matching


def calibrated_matching(model: stim.Circuit) -> pymatching.Matching:
    """Exactly what run._decode builds."""
    return pymatching.Matching.from_detector_error_model(model.detector_error_model(decompose_errors=True))


def uniform_matching(model: stim.Circuit, calibrated: pymatching.Matching | None = None) -> pymatching.Matching:
    """The calibrated matching graph with every edge (and boundary edge) given the same weight.

    Built as a DEM with one error line per edge of the already-merged pymatching graph, so
    nothing merges again and all weights are ln((1-p)/p) for the same p.
    """
    calibrated = calibrated or calibrated_matching(model)
    lines = []
    for u, v, attr in calibrated.edges():
        obs = " ".join(f"L{k}" for k in sorted(attr["fault_ids"]))
        dets = f"D{u}" if v is None else f"D{u} D{v}"
        lines.append(f"error({UNIFORM_P}) {dets} {obs}".rstrip())
    for d in range(model.num_detectors):
        lines.append(f"detector D{d}")
    for k in range(model.num_observables):
        lines.append(f"logical_observable L{k}")
    m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))
    ws = {round(a["weight"], 9) for _, _, a in m.edges()}
    assert len(ws) == 1, ws
    assert m.num_edges == calibrated.num_edges
    return m


def mechanisms(model: stim.Circuit):
    """Each error mechanism of the decomposed DEM that the decoder uses.

    Returns (probabilities, detector-sets, observable-masks, components) where a
    mechanism's detector set is the XOR of its ^-separated components (the full symptom
    of the one circuit fault class) and its observable mask is the XOR of theirs.
    """
    dem = model.detector_error_model(decompose_errors=True)
    probs, dets, obs, comps = [], [], [], []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        d: set[int] = set()
        o = 0
        parts = [[]]
        for t in ins.targets_copy():
            if t.is_separator():
                parts.append([])
                continue
            if t.is_relative_detector_id():
                d ^= {t.val}
                parts[-1].append(f"D{t.val}")
            elif t.is_logical_observable_id():
                o ^= 1 << t.val
                parts[-1].append(f"L{t.val}")
        probs.append(ins.args_copy()[0])
        dets.append(sorted(d))
        obs.append(o)
        comps.append(" ^ ".join(" ".join(p) for p in parts))
    return np.array(probs), dets, np.array(obs, dtype=np.int64), comps


def dense(det_sets, n: int) -> np.ndarray:
    out = np.zeros((len(det_sets), n), dtype=np.uint8)
    for i, ds in enumerate(det_sets):
        out[i, ds] = 1
    return out


def patch_of_observable(job, labels) -> dict[int, str]:
    """Observable k -> patch ("3" or "5"): logical k is the circuit's k-th patch present."""
    present = list(dict.fromkeys(str(p) for p, _, _ in labels))
    return dict(enumerate(present))


def detector_patch(labels) -> np.ndarray:
    return np.array([int(str(p)) for p, _, _ in labels])


def single_observable_circuit(model: stim.Circuit, keep: int) -> stim.Circuit:
    """The same circuit with only observable `keep`, renumbered to L0."""
    out = stim.Circuit()
    for ins in model.flattened():
        if ins.name == "OBSERVABLE_INCLUDE":
            if int(ins.gate_args_copy()[0]) != keep:
                continue
            out.append("OBSERVABLE_INCLUDE", ins.targets_copy(), [0])
        else:
            out.append(ins)
    return out

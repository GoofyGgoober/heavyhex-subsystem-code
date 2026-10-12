"""The leakage model's grid fit with the Pauli-frame simulator, as first done.

Every qubit leaks with the same probability per CZ (leak), a leaked qubit's CZ gives its
partner a Pauli error with probability strength (an X for "bitflip", a random Pauli for
"depolarize") and spreads the leakage to it with probability spread; the lifetime and the
leaked readout are the hidden Markov fits' (hmm.py), resets and readouts IBM's (properties.py).
The three are fitted together to six statistics of the 8-round circuits of four jobs
(leakage.FIT_JOBS, leakage.stats): the mean detection rates of X-memory stabilizers, Z-memory
stabilizers and Z-memory flags and relays; the flags' and relays' repeat excess; and the
co-streak excess of pairs that read a common data qubit and of pairs on one Z gauge. Logical
errors and Lambda are left out of the fit. 8,000 shots per circuit, seed 21.

    python docs/analysis/leakage_frame_grid.py bitflip      # results/leakage_frame_grid_bitflip.json
    python docs/analysis/leakage_frame_grid.py depolarize   # results/leakage_frame_grid_depolarize.json

With --mode the same fit runs with the left-out CZ handled another way (frame_sim.py):
"reference" as above, "dephase" a random phase on the partner, "exact-identity" and
"exact-dephase" the exact simulation (tableau_ref.py, cut to leakage.EXACT_SUBSET because it
is slow); --grid wide widens the grid. These write
results/leakage_frame_grid_<kind>_<mode>_<grid>.json. Each run is heavy (8 circuits x 8,000
shots per grid point; the exact modes take hours).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from collections import defaultdict

import numpy as np

import frame_sim
from common import RESULTS, ROUND_US, WORKERS, jobs, memory_index
from leakage import EXACT_SUBSET, FIT_JOBS, AllQubits, hmm_inputs, ibm_errors, stats

KINDS = ("depolarize", "bitflip", "phase")  # what a leaked qubit's CZ does to its partner
MODES = ("reference", "dephase", "exact-identity", "exact-dephase")
GRIDS = {
    "paper": {
        "bitflip": ((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05)),
        "phase": ((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05)),
        "depolarize": ((0.001, 0.002, 0.003, 0.0045), (0.25, 0.5, 1.0), (0.0, 0.05, 0.15)),
    },
    "wide": dict.fromkeys(
        ("bitflip", "depolarize"),
        ((0.0005, 0.001, 0.0015, 0.002, 0.003), (0.0, 0.1, 0.25, 0.5, 1.0), (0.0, 0.05, 0.15)),
    ),
}
SHOTS = 8000
SEED = 21


def published(kind: str) -> None:
    """The fit as first run: the frame simulator, the CZ handled as "reference"."""
    life, read = hmm_inputs()
    js = [j for j in jobs() if j.name in FIT_JOBS]
    cases = []
    for j in js:
        for basis in "XZ":
            idx = memory_index(basis, 8)
            model, records, labels = j.model(idx)
            chip, _ = j.chip_detectors(idx, model, records)
            cases.append(
                (
                    j,
                    basis,
                    model,
                    labels,
                    {f"{basis} {k}": v for k, v in stats(chip, labels).items()},
                )
            )
    chip_tot = {}
    for *_, s in cases:
        for k, v in s.items():
            chip_tot.setdefault(k, []).append(v)
    chip_mean = {k: float(np.nanmean(v)) for k, v in chip_tot.items()}
    print("chip:", "  ".join(f"{k} {v:.4g}" for k, v in sorted(chip_mean.items())), flush=True)
    results = []
    for leak, strength, spread in itertools.product(*GRIDS["paper"][kind]):
        tot = {}
        for j, basis, model, labels, _ in cases:
            reset, readout = ibm_errors(j.day)
            lk = frame_sim.Leakage(
                # Used within this iteration, so the late-bound leak is this one.
                leak=AllQubits(defaultdict(lambda: leak)),  # noqa: B023
                lifetime_us=life,
                half_round_us=ROUND_US / 2,
                read_leaked=read,
                partner_phase=strength,
                partner_kind=kind,
                transfer=spread,
                reset_error=reset,
                readout=readout,
            )
            det, _ = frame_sim.simulate(model, SHOTS, lk, seed=SEED)
            for k, v in stats(det, labels).items():
                tot.setdefault(f"{basis} {k}", []).append(v)
        sim = {k: float(np.nanmean(v)) for k, v in tot.items()}
        rel = {
            k: (sim[k] - chip_mean[k]) / abs(chip_mean[k])
            for k in chip_mean
            if np.isfinite(chip_mean[k])
        }
        loss = float(np.sqrt(np.mean([v**2 for v in rel.values()])))
        results.append(
            {"leak": leak, "strength": strength, "spread": spread, "loss": loss, "sim": sim}
        )
        print(
            f"leak {leak:<6g} strength {strength:<4g} spread {spread:<4g} loss {loss:.3f}  "
            + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in sorted(rel.items())),
            flush=True,
        )
    best = min(results, key=lambda r: r["loss"])
    print(
        f"\nbest: leak {best['leak']} per CZ, strength {best['strength']}, spread {best['spread']} (loss {best['loss']:.3f})"
    )
    out = {
        "chip": chip_mean,
        "lifetime_us": life,
        "read_leaked": read,
        "grid": results,
        "best": best,
    }
    (RESULTS / f"leakage_frame_grid_{kind}.json").write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------ the CZ handled other ways
_CASES: list = []


def _init():
    life, read = hmm_inputs()
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        reset, readout = ibm_errors(j.day)
        for basis in "XZ":
            model, _records, labels = j.model(memory_index(basis, 8))
            _CASES.append((j.name, basis, model, labels, reset, readout, life, read))
    assert len(_CASES) == 2 * len(FIT_JOBS), len(_CASES)


def _case(args):
    kind, mode, leak, strength, spread, ci = args
    _name, basis, model, labels, reset_error, readout, life, read = _CASES[ci]
    lk = frame_sim.Leakage(
        leak=AllQubits(defaultdict(lambda: leak)),
        lifetime_us=life,
        half_round_us=ROUND_US / 2,
        read_leaked=read,
        partner_phase=strength,
        partner_kind=kind,
        transfer=spread,
        reset_error=reset_error,
        readout=readout,
        skipped_cz="dephase" if mode.startswith("exact-") else mode,
    )
    if mode.startswith("exact-"):
        import tableau_ref

        det, _ = tableau_ref.simulate(model, SHOTS, lk, mode.removeprefix("exact-"), seed=SEED)
    else:
        det, _ = frame_sim.simulate(model, SHOTS, lk, seed=SEED)
    return {f"{basis} {k}": v for k, v in stats(det, labels).items()}


def other_mode(kind: str, mode: str, grid: str, workers: int) -> None:
    chip = json.loads((RESULTS / f"leakage_frame_grid_{kind}.json").read_text())["chip"]
    pts = list(itertools.product(*GRIDS[grid][kind]))
    if mode.startswith("exact-") and grid == "paper":
        pts = EXACT_SUBSET
        print("exact mode: paper grid cut to", pts, flush=True)
    ncases = 2 * len(FIT_JOBS)
    tasks = [(kind, mode, *p, ci) for p in pts for ci in range(ncases)]
    with mp.Pool(workers, initializer=_init) as pool:
        per_case = pool.map(_case, tasks, chunksize=1)
    results = []
    for i, (leak, strength, spread) in enumerate(pts):
        tot = {}
        for st in per_case[i * ncases : (i + 1) * ncases]:
            for k, v in st.items():
                tot.setdefault(k, []).append(v)
        sim = {k: float(np.nanmean(v)) for k, v in tot.items()}
        rel = {k: (sim[k] - chip[k]) / abs(chip[k]) for k in chip if np.isfinite(chip[k])}
        loss = float(np.sqrt(np.mean([v**2 for v in rel.values()])))
        results.append(
            {
                "leak": leak,
                "strength": strength,
                "spread": spread,
                "loss": loss,
                "sim": sim,
                "rel": rel,
            }
        )
        print(
            f"leak {leak:<6g} strength {strength:<4g} spread {spread:<4g} loss {loss:.3f}  "
            + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in sorted(rel.items())),
            flush=True,
        )
    best = min(results, key=lambda r: r["loss"])
    print(
        f"\nbest ({kind}, {mode}, {grid}): leak {best['leak']} strength {best['strength']} spread {best['spread']} loss {best['loss']:.3f}"
    )
    print("  " + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in sorted(best["rel"].items())))
    (RESULTS / f"leakage_frame_grid_{kind}_{mode}_{grid}.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "mode": mode,
                "grid": grid,
                "chip": chip,
                "points": results,
                "best": best,
            },
            indent=1,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("kind", nargs="?", choices=KINDS, default=KINDS[0])
    ap.add_argument("--mode", choices=MODES, help="handle the left-out CZ another way (see above)")
    ap.add_argument("--grid", choices=tuple(GRIDS), default="paper")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.mode is None:
        published(a.kind)
    else:
        other_mode(a.kind, a.mode, a.grid, a.workers)


if __name__ == "__main__":
    main()

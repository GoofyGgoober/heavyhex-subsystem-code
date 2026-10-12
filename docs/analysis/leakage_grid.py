"""The leakage model's grid fit (leakage_frame_grid.py), redone with an exact simulation.

Same four jobs (leakage.FIT_JOBS), same eight-round circuits, same six detection statistics
(leakage.stats), same 8000 shots per circuit, same loss (root-mean-square relative deviation
from the device), same lifetime and leaked readout (the hidden Markov fits) and IBM
reset/readout errors. Only the simulator differs: exact_sim.py (a per-shot stabilizer
simulation, checked against tableau_ref.py in leakage_checks.py), so a CZ with a leaked
operand really does not act. CHANNEL picks what it does instead:

  identity  nothing (the paper's stated model)
  dephase   the partner gets a uniformly random phase (Z with probability 1/2) at each such CZ
  pi        the partner gets Z (conditional phase pi)

For "dephase" the Pauli-frame simulator (frame_sim.py, skipped_cz="dephase") is also exact
(tracing out the leaked qubit, dephase o CZ = dephase) and about 30 times faster, so
--sim frame uses it for wide scans.

Each point also gets the co-streak excesses computed per round slot and averaged
(P(A&B|slot) - P(A|slot)P(B|slot)), the round-conditioned statistic of the paper's spatial
analysis, and the loss with those in place of the pooled ones ("loss_slot").

    python docs/analysis/leakage_grid.py CHANNEL GRID [--seed 21] [--shots 8000] [--workers 2] [--sim exact|frame]
      GRID: a name in GRIDS, or explicit points "leak,strength,spread;leak,strength,spread;..."

Very heavy: hours per grid. Results accumulate in results/leakage_grid_<CHANNEL>[_frame].json
(one entry per point, seed and shot count; points already there are skipped unless --redo).
Device statistics: results/leakage_grid_device.json (checked against
results/leakage_frame_grid_bitflip.json).
"""

from __future__ import annotations

import os

# one BLAS thread per process: OpenBLAS otherwise commits ~0.5 GB per process on Windows,
# which the multiprocessing pools below cannot afford
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict

import numpy as np

from common import RESULTS, ROUND_US, WORKERS, data_read, jobs, memory_index, readout_series, z4_of
from leakage import FIT_JOBS, hmm_inputs, ibm_errors, stats

PAPER = ((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05))  # the published bit-flip grid
GRIDS = {
    "paper": list(itertools.product(*PAPER)),
    # around the published optimum (0.2%, 25%, 5%), whose strength and transfer sit on edges
    "ext": [
        p
        for p in itertools.product(
            (0.0015, 0.002, 0.0025), (0.1, 0.175, 0.25), (0.025, 0.05, 0.075, 0.1)
        )
        if p != (0.002, 0.25, 0.05)
    ],
    # random phase: the paper-grid optimum is at its low corner, so go lower
    "low": list(
        itertools.product(
            (0.00025, 0.0005, 0.00075, 0.001), (0.0, 0.1, 0.25, 0.5), (0.0, 0.05, 0.1, 0.15)
        )
    ),
    # random phase, second extension: "low" put the optimum at its lowest leak, highest transfer
    "low2": list(itertools.product((0.0001, 0.00025, 0.0005), (0.5, 0.75, 1.0), (0.15, 0.2, 0.3))),
    # random phase, third: still at the lowest leak and highest transfer; stop at 50% transfer
    "low3": list(itertools.product((0.00005, 0.0001), (0.25, 0.5), (0.3, 0.4, 0.5))),
    # phase pi (supplementary): a first coarse grid in the random-phase optimum's region
    "pi1": list(itertools.product((0.00025, 0.0005, 0.001), (0.25, 0.5), (0.05, 0.15))),
    # phase pi, around pi1's optimum (0.05%, 25%, 15%: strength and transfer on edges)
    "pi2": list(itertools.product((0.0005, 0.00065), (0.1, 0.25), (0.1, 0.15, 0.2))),
}
CHUNK = 1000
_W: dict = {}


def fit_cases():
    """[(job, basis, model, labels, reset_error, readout)] for the fit's 8 circuits."""
    out = []
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        reset, readout = ibm_errors(j.day)
        for basis in "XZ":
            model, _records, labels = j.model(memory_index(basis, 8))
            out.append((j, basis, model, labels, reset, readout))
    return out


def hmm_params() -> tuple[float, float, float]:
    """Lifetime (us) and leaked readout (the medians of the hidden Markov fits), half round (us)."""
    return (*hmm_inputs(), ROUND_US / 2)


def leakage(leak, strength, spread, reset_error, readout, hmm):
    import exact_sim as E

    return E.Leakage(
        leak=E.Uniform(leak),
        lifetime_us=hmm[0],
        half_round_us=hmm[2],
        read_leaked=hmm[1],
        partner_phase=strength,
        partner_kind="bitflip",
        transfer=spread,
        reset_error=reset_error,
        readout=readout,
    )


def _init(sim, light, hmm):
    # light: [(basis, circuit instructions, reset_error, readout)] from the parent, so that the
    # workers need neither the run folders nor qiskit
    import exact_sim as E

    _W["cases"] = [(b, E.rebuild(data), re, ro) for b, data, re, ro in light]
    _W["hmm"], _W["sim"], _W["progs"] = hmm, sim, {}


def _task(args):
    channel, leak, strength, spread, ci, chunk, n, seed = args
    import exact_sim as E

    _basis, model, reset_error, readout = _W["cases"][ci]
    lk = leakage(leak, strength, spread, reset_error, readout, _W["hmm"])
    s = seed * 1000 + chunk
    if _W["sim"] == "frame":
        import frame_sim as F

        assert channel == "dephase", (
            "the frame simulator is exact only for the random-phase channel"
        )
        flk = F.Leakage(
            **{k: getattr(lk, k) for k in lk.__dataclass_fields__}, skipped_cz="dephase"
        )
        det, _ = F.simulate(model, n, flk, seed=s)
    else:
        if ci not in _W["progs"]:
            _W["progs"][ci] = E.Program(model, lk)
        det, _ = E.simulate(model, n, lk, channel, seed=s, program=_W["progs"][ci])
    return (leak, strength, spread, ci, chunk), np.packbits(det, axis=1), det.shape


def slot_costreak(det, labels) -> dict:
    """Co-streak excesses per round slot, averaged over slots (the round-conditioned version)."""
    series = readout_series(labels)
    if not series:
        return {}
    st = {}
    for key, cols in series.items():
        x = det[:, cols]
        st[key] = x[:, :-1] & x[:, 1:]
    acc = defaultdict(list)
    for a, b in itertools.combinations(series, 2):
        if a[0] != b[0]:
            continue
        sa, sb = st[a], st[b]
        v = float(np.mean((sa & sb).mean(0) - sa.mean(0) * sb.mean(0)))
        if data_read(*a) & data_read(*b):
            acc["co_data_slot"].append(v)
        elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
            acc["co_gauge_slot"].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def case_stats(det, basis, labels) -> dict:
    s = stats(det, labels)
    s.update(slot_costreak(det, labels))
    return {f"{basis} {k}": v for k, v in s.items()}


def merge(per_case) -> dict:
    tot = defaultdict(list)
    for st in per_case:
        for k, v in st.items():
            tot[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in tot.items()}


POOLED = ["X stab", "Z stab", "Z flag", "Z repeat", "Z co_data", "Z co_gauge"]
SLOT = ["X stab", "Z stab", "Z flag", "Z repeat", "Z co_data_slot", "Z co_gauge_slot"]


def loss(sim: dict, dev: dict, keys) -> tuple[float, dict]:
    rel = {k: (sim[k] - dev[k]) / abs(dev[k]) for k in keys if np.isfinite(dev[k])}
    return float(np.sqrt(np.mean([v**2 for v in rel.values()]))), rel


def device_stats(cases) -> dict:
    path = RESULTS / "leakage_grid_device.json"
    if path.exists():
        return json.loads(path.read_text())
    per = []
    for j, basis, _model, labels, *_ in cases:
        chip, _ = j.chip_detectors(memory_index(basis, 8))
        per.append(case_stats(chip, basis, labels))
    dev = merge(per)
    published = json.loads((RESULTS / "leakage_frame_grid_bitflip.json").read_text())["chip"]
    assert all(
        abs(dev[k] - published[k]) <= 1e-12 * max(1, abs(published[k])) for k in published
    ), (dev, published)
    path.write_text(json.dumps(dev, indent=1) + "\n")
    return dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", choices=["identity", "dephase", "pi"])
    ap.add_argument("grid")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--sim", choices=["exact", "frame"], default="exact")
    ap.add_argument("--redo", action="store_true", help="rerun points already in the results file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pts = (
        GRIDS[a.grid]
        if a.grid in GRIDS
        else [tuple(float(x) for x in p.split(",")) for p in a.grid.split(";")]
    )
    cases = fit_cases()
    dev = device_stats(cases)
    labels = [(c[1], c[3]) for c in cases]
    out_path = RESULTS / f"leakage_grid_{a.channel}{'_frame' if a.sim == 'frame' else ''}.json"
    done = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {"channel": a.channel, "sim": a.sim, "device": dev, "points": []}
    )
    have = {
        (q["leak"], q["strength"], q["spread"])
        for q in done["points"]
        if q["seed"] == a.seed and q["shots"] == a.shots
    }
    if not a.redo:
        pts = [p for p in pts if tuple(p) not in have]
    nchunk = (a.shots + CHUNK - 1) // CHUNK
    weight = [m.num_detectors * len(m) for _, _, m, *_ in cases]
    order = sorted(range(len(cases)), key=lambda ci: -weight[ci])
    tasks = [
        (a.channel, *p, ci, c, min(CHUNK, a.shots - c * CHUNK), a.seed)
        for p in pts
        for ci in order
        for c in range(nchunk)
    ]
    need = defaultdict(lambda: len(cases) * nchunk)
    got: dict = defaultdict(dict)
    t0 = time.time()
    print(f"{a.channel} {a.grid}: {len(pts)} points, {len(tasks)} tasks, sim {a.sim}", flush=True)
    import exact_sim as E

    light = []
    for _, basis, model, _, re, ro in cases:
        data = E.instructions(model)
        assert E.rebuild(data) == model.flattened()
        light.append((basis, data, re, ro))
    with mp.Pool(a.workers, initializer=_init, initargs=(a.sim, light, hmm_params())) as pool:
        for key, packed, shape in pool.imap_unordered(_task, tasks, chunksize=1):
            p, ci, c = key[:3], key[3], key[4]
            got[p][(ci, c)] = np.unpackbits(packed, axis=1, count=shape[1]).astype(bool)
            need[p] -= 1
            if need[p]:
                continue
            per = []
            for ci2 in range(len(cases)):
                det = np.concatenate([got[p][(ci2, c2)] for c2 in range(nchunk)])
                per.append(case_stats(det, *labels[ci2]))
            del got[p]
            sim = merge(per)
            lp, rel = loss(sim, dev, POOLED)
            ls, rel_s = loss(sim, dev, SLOT)
            rec = {
                "leak": p[0],
                "strength": p[1],
                "spread": p[2],
                "seed": a.seed,
                "shots": a.shots,
                "loss": lp,
                "loss_slot": ls,
                "rel": rel,
                "rel_slot": rel_s,
                "sim": sim,
            }
            done["points"] = [
                q
                for q in done["points"]
                if (q["leak"], q["strength"], q["spread"], q["seed"], q["shots"])
                != (*p, a.seed, a.shots)
            ] + [rec]
            out_path.write_text(json.dumps(done, indent=1) + "\n")
            print(
                f"leak {p[0]:<7g} strength {p[1]:<5g} spread {p[2]:<5g} loss {lp:.3f} (slot {ls:.3f})  "
                + "  ".join(f"{k} {100 * v:+.0f}%" for k, v in rel.items())
                + f"   [{time.time() - t0:.0f}s]",
                flush=True,
            )
    best = min((q for q in done["points"] if q["seed"] == a.seed), key=lambda q: q["loss"])
    print(
        f"best so far (seed {a.seed}): leak {best['leak']} strength {best['strength']} spread {best['spread']} loss {best['loss']:.3f}"
    )


if __name__ == "__main__":
    main()

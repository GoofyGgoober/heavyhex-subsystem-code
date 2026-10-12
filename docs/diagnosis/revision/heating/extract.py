"""Per-shot quantities for the heating test, in the order the shots ran within each piece.

Exploratory, for the revision. Reads the committed run folders, uses no QPU time and
changes nothing in src/ or runs/. Writes per_shot.npz next to this file, which
heating.py reads.

For every job and every piece (one PUB of 625 shots; a memory circuit has 4 pieces, an
idle readout 2), keyed "<job>/<k>/<patch>/<what>" with k the piece's position in the job:

  memory pieces (decoded as analyze() decodes them: the job's own stim model at the
  job's measured f, R.in_record_order -> m2d converter -> R._decode, pymatching):
    det    stabilizer detection events in the shot (int16)
    flag   flag and relay firings in the shot (Z memory; 0 columns in X memory)
    fail   the decoded logical failed (bool)
    and the same three from the model's own shots ("null_det", "null_flag", "null_fail"),
    sampled with stim, 625 per piece, so the statistics can be checked on drift-free data
  idle pieces:
    anc    gauge-ancilla readouts that read 1 (every one should read 0 without errors)
    data   data-qubit bits, shots x qubits (X or Y readout)

Meta (job, piece -> setting, block, detector counts, the decoded logical error of each
half, to check against analysis.json's "drift") goes to per_shot_meta.json.

    python docs/diagnosis/revision/heating/extract.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "docs" / "diagnosis"))

from common import jobs  # noqa: E402

from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import SETTINGS  # noqa: E402


def pieces_of(job) -> dict[int, dict[str, np.ndarray]]:
    saved = np.load(job.folder / R.SHOTS)
    out: dict[int, dict[str, np.ndarray]] = {}
    order = job.info["order"]
    for key in saved.files:
        k, i, block, name = key.split("/")
        k, i, block = int(k), int(i), int(block)
        assert order[k] == [i, block], (job.name, key, order[k])
        out.setdefault(k, {})[name] = saved[key]
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {"jobs": {}}
    for jn, job in enumerate(jobs()):
        t0 = time.time()
        pieces = pieces_of(job)
        order = job.info["order"]
        record = json.loads((job.folder / "job.json").read_text())
        each = record["shots_per_piece"]
        info = {
            "folder": job.folder.name,
            "day": job.day,
            "rep": job.rep,
            "pairing": job.info["rep"]["pairing"],
            "f": job.f,
            "order": order,
            "shots_per_piece": each,
            "qpu_seconds_estimate": record.get("qpu_seconds_estimate"),
            "qpu_charge_time_seconds": (record.get("usage") or {}).get("qpu_charge_time_seconds"),
            "running_utc": record.get("running_utc"),
            "finished_utc": record.get("finished_utc"),
            "circuit_seconds": job.info["seconds"],
            "settings": {},
            "halves": {},
        }
        for i, setting in enumerate(SETTINGS):
            ks = [k for k, (ii, _) in enumerate(order) if ii == i]
            if setting.kind == "memory":
                model, records, labels = job.model(i)
                present = list(dict.fromkeys(str(p) for p, _, _ in labels))
                conv = model.compile_m2d_converter()
                dets, obss = [], []
                for k in ks:
                    assert all(len(b) == each for b in pieces[k].values())
                    measured = R.in_record_order(pieces[k], records)
                    d, o = conv.convert(measurements=measured, separate_observables=True)
                    dets.append(d)
                    obss.append(o)
                chip_d, chip_o = np.concatenate(dets), np.concatenate(obss)
                chip_fail = R._decode(model, chip_d) != chip_o
                sampler = model.compile_detector_sampler(seed=1000 * jn + i)
                null_d, null_o = sampler.sample(len(chip_d), separate_observables=True)
                null_fail = R._decode(model, null_d) != null_o
                cols = {}
                for pi, p in enumerate(present):
                    stab = [c for c, (pp, _, name) in enumerate(labels) if str(pp) == p and " " not in name]
                    flag = [c for c, (pp, _, name) in enumerate(labels) if str(pp) == p and " " in name]
                    cols[p] = {"stab": len(stab), "flag": len(flag)}
                    for src, D, F in (("", chip_d, chip_fail), ("null_", null_d, null_fail)):
                        det = D[:, stab].sum(axis=1).astype(np.int16)
                        flg = D[:, flag].sum(axis=1).astype(np.int16) if flag else np.zeros(len(D), np.int16)
                        for n, k in enumerate(ks):
                            sl = slice(n * each, (n + 1) * each)
                            arrays[f"{job.name}/{k}/{p}/{src}det"] = det[sl]
                            arrays[f"{job.name}/{k}/{p}/{src}flag"] = flg[sl]
                            arrays[f"{job.name}/{k}/{p}/{src}fail"] = F[sl, pi]
                    # the decoded logical error in each half of every piece, as analyze() splits them
                    middle = each // 2
                    f3 = chip_fail[:, pi].reshape(len(ks), each)
                    info["halves"].setdefault(str(i), {})[p] = [
                        float(f3[:, :middle].mean()), float(f3[:, middle:].mean()),
                        int(f3[:, :middle].size), int(f3[:, middle:].size),
                    ]
                info["settings"][str(i)] = {"pieces": ks, "patches": present, "columns": cols}
            else:
                present = [p for p in ("3", "5") if f"m{p}" in pieces[ks[0]]]
                for k in ks:
                    for p in present:
                        arrays[f"{job.name}/{k}/{p}/anc"] = pieces[k][f"m{p}"].sum(axis=1).astype(np.int16)
                        arrays[f"{job.name}/{k}/{p}/data"] = pieces[k][f"d{p}"]
                info["settings"][str(i)] = {
                    "pieces": ks,
                    "patches": present,
                    "ancilla_bits": {p: int(pieces[ks[0]][f"m{p}"].shape[1]) for p in present},
                }
        meta["jobs"][job.name] = info
        print(f"{job.name}: {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(HERE / "per_shot.npz", **arrays)
    (HERE / "per_shot_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()

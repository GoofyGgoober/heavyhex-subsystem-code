"""Independent re-extraction of per-shot quantities for the heating verification.

Written from scratch (not copied from ../heating/extract.py). Uses only the repo's
own detector/decoder building blocks (R.simulated, R.in_record_order, the stim m2d
converter, pymatching on the model's DEM) so that detection events and decoded
failures are defined exactly as analyze() defines them. Everything else (piece
bookkeeping, binning, statistics) is written independently.

Output: v_per_shot.npz with, for each job and piece k (position in the job's PUB
list), arrays of length 625 in shot order:
  mem/<job>/<k>/<patch>/det    stabilizer detection events per shot (int)
  mem/<job>/<k>/<patch>/flag   flag+relay detection events per shot (int; Z memory)
  mem/<job>/<k>/<patch>/fail   decoded logical failure per shot (bool)
  idle/<job>/<k>/<patch>/anc   gauge-ancilla bits per shot that read 1 (int)
  idle/<job>/<k>/<patch>/data  data bits (shots x n_data, bool)
  q/<job>/<k>/<qubit>          per-shot fraction of that physical qubit's readouts that read 1
and v_meta.json with piece -> setting, counts, plus a check against analysis.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pymatching

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]

from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import SETTINGS  # noqa: E402
from heavyhex.patches.placement import Calibration  # noqa: E402


def clbit_qubits(qc):
    """(register, index) -> physical qubit, from the translated circuit's measures."""
    out = {}
    for it in qc.data:
        if it.operation.name != "measure":
            continue
        q = qc.find_bit(it.qubits[0]).index
        for reg, idx in qc.find_bit(it.clbits[0]).registers:
            out[(reg.name, idx)] = q
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    folders = sorted(p for p in (REPO / "runs").glob("ibm_fez-2026-10-0*-r*") if p.is_dir())
    for folder in folders:
        t0 = time.time()
        job = folder.name.removeprefix("ibm_fez-")
        run = json.loads((folder / "run.json").read_text())
        ana = json.loads((folder / "analysis.json").read_text())
        jj = json.loads((folder / "job.json").read_text())
        f = ana["decoupled_dephasing"]
        cal = Calibration.load(folder / "calibration.json")
        circuits = R.load_circuits(folder)
        saved = np.load(folder / "shots.npz")
        pieces: dict[int, dict] = {}
        for key in saved.files:
            k, i, b, name = key.split("/")
            p = pieces.setdefault(int(k), {"i": int(i), "block": int(b), "bits": {}})
            assert p["i"] == int(i) and p["block"] == int(b)
            p["bits"][name] = saved[key]
        assert sorted(pieces) == list(range(len(run["order"])))
        for k, p in pieces.items():
            assert run["order"][k] == [p["i"], p["block"]]
        patches = sorted(run["chips"])  # "3", "5"
        jm = {
            "folder": folder.name,
            "day": run["rep"]["day"],
            "rep": run["rep"]["rep"],
            "patches": patches,
            "f": f,
            "running_utc": jj.get("running_utc"),
            "finished_utc": jj.get("finished_utc"),
            "qpu_seconds_estimate": jj.get("qpu_seconds_estimate"),
            "pieces": {},
            "check": {},
        }
        # memory
        for i, s in enumerate(SETTINGS):
            mine = sorted(k for k, p in pieces.items() if p["i"] == i)
            qmap = clbit_qubits(circuits[i])
            for k in mine:
                bits = pieces[k]["bits"]
                qubits = sorted(set(qmap.values()))
                for q in qubits:
                    cols = {}
                    for (reg, idx), qq in qmap.items():
                        if qq == q:
                            cols.setdefault(reg, []).append(idx)
                    tot = np.zeros(len(next(iter(bits.values()))), dtype=float)
                    n = 0
                    for reg, idxs in cols.items():
                        tot += bits[reg][:, idxs].sum(axis=1)
                        n += len(idxs)
                    arrays[f"q/{job}/{k}/{q}"] = (tot / n).astype(np.float32)
            if not s.decoded:
                for k in mine:
                    bits = pieces[k]["bits"]
                    jm["pieces"][k] = {"i": i, "block": pieces[k]["block"], "kind": "idle",
                                       "readout": s.readout, "rounds": s.rounds}
                    for patch in patches:
                        m = bits[f"m{patch}"]
                        # only measured columns of m (register may be larger than measured)
                        measured = sorted(idx for (reg, idx) in qmap if reg == f"m{patch}")
                        arrays[f"idle/{job}/{k}/{patch}/anc"] = m[:, measured].sum(axis=1).astype(np.int16)
                        jm["pieces"][k][f"n_anc_{patch}"] = len(measured)
                        arrays[f"idle/{job}/{k}/{patch}/data"] = bits[f"d{patch}"].astype(bool)
                continue
            model, records, labels = R.simulated(s, circuits[i], cal, run["dt"], f)
            conv = model.compile_m2d_converter()
            matching = pymatching.Matching.from_detector_error_model(
                model.detector_error_model(decompose_errors=True)
            )
            present = list(dict.fromkeys(str(pp) for pp, _, _ in labels))
            stab = {pp: np.array([c for c, (q, r, nm) in enumerate(labels) if str(q) == pp and " " not in nm]) for pp in present}
            flg = {pp: np.array([c for c, (q, r, nm) in enumerate(labels) if str(q) == pp and " " in nm], dtype=int) for pp in present}
            all_det, all_fail = [], []
            for k in mine:
                bits = pieces[k]["bits"]
                meas = R.in_record_order(bits, records)
                det, obs = conv.convert(measurements=meas, separate_observables=True)
                pred = matching.decode_batch(det).astype(bool)
                fail = pred != obs
                all_det.append(det)
                all_fail.append(fail)
                jm["pieces"][k] = {"i": i, "block": pieces[k]["block"], "kind": "memory",
                                   "basis": s.basis, "rounds": s.rounds}
                for o, pp in enumerate(present):
                    arrays[f"mem/{job}/{k}/{pp}/det"] = det[:, stab[pp]].sum(axis=1).astype(np.int16)
                    arrays[f"mem/{job}/{k}/{pp}/flag"] = (
                        det[:, flg[pp]].sum(axis=1).astype(np.int16) if len(flg[pp]) else np.zeros(len(det), np.int16)
                    )
                    arrays[f"mem/{job}/{k}/{pp}/fail"] = fail[:, o].astype(bool)
                    jm["pieces"][k][f"n_stab_{pp}"] = int(len(stab[pp]))
                    jm["pieces"][k][f"n_flag_{pp}"] = int(len(flg[pp]))
            # check against analysis.json (pieces concatenated in k order, as load_shots does)
            det = np.concatenate(all_det)
            fail = np.concatenate(all_fail)
            obs_ana = ana["observed"]["settings"][i]
            jm["check"][str(i)] = {
                "max_abs_det_rate_diff": float(np.max(np.abs(det.mean(0) - np.array(obs_ana["detector_rates"])))),
                "logical_error_mine": fail.mean(0).tolist(),
                "logical_error_ana": obs_ana["logical_error"],
                "present": present,
            }
        meta[job] = jm
        print(job, f"{time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(OUT / "v_per_shot.npz", **arrays)
    (OUT / "v_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

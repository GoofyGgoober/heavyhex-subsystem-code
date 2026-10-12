"""Re-decode memory-circuit shots from shots.npz with the repo's decoder pipeline
(stim model at the job's recorded f, pymatching) and compare the failure counts
with analysis.json. Read-only on runs/. Usage: python v_redecode.py [folder ...]
"""
import json
import sys
from pathlib import Path


from heavyhex.experiment import run as R

REPO = Path(__file__).resolve().parents[4]


def check(folder: Path):
    info = json.loads((folder / "run.json").read_text())
    an = json.loads((folder / "analysis.json").read_text())
    cal = R.Calibration.load(folder / "calibration.json")
    circuits = R.load_circuits(folder)
    shots = R.load_shots(folder)
    f = an["decoupled_dephasing"]
    bad = 0
    for i, (setting, circuit) in enumerate(zip(R.SETTINGS, circuits)):
        if not setting.decoded:
            continue
        model, records, labels = R.simulated(setting, circuit, cal, info["dt"], f)
        meas = R.in_record_order(shots[i], records)
        fired, flips = model.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
        failed = R._decode(model, fired) != flips
        mine = failed.sum(axis=0).tolist()
        stored = [round(p * failed.shape[0]) for p in an["observed"]["settings"][i]["logical_error"]]
        ok = mine == stored
        bad += not ok
        print(folder.name, i, setting.basis, setting.rounds, failed.shape[0], mine, stored, "OK" if ok else "MISMATCH", flush=True)
    return bad


if __name__ == "__main__":
    folders = [Path(a) for a in sys.argv[1:]] or sorted((REPO / "runs").glob("ibm_fez-*"))
    total = sum(check(f) for f in folders)
    print("mismatching circuits:", total)

"""Independent verifier: per-round logical-error counts for all 40 hardware curves
and the stored per-round predictions, read straight from runs/*/analysis.json and
runs/*/prediction.json. Checks that p*shots is an integer and that settings are in
the expected order. Writes v_counts.json next to this file.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "runs"
OUT = Path(__file__).resolve().parent / "v_counts.json"
DAYS = ["2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
ROUNDS = [1, 2, 3, 4, 6, 8]


def load(p):
    return json.loads(Path(p).read_text())


def curves_from(folder: Path):
    """{(basis, d): dict(k, N, pred..)} for one job folder."""
    run = load(folder / "run.json")
    an = load(folder / "analysis.json")
    job = load(folder / "job.json")
    pred = load(folder / "prediction.json")
    settings = run["settings"]
    patches = an["patches"]
    N = job["shots"]
    out = {}
    for basis, off in (("X", 0), ("Z", 6)):
        idx = list(range(off, off + 6))
        for i, n in zip(idx, ROUNDS):
            s = settings[i]
            assert s["kind"] == "memory" and s["basis"] == basis and s["rounds"] == n, (folder, i, s)
        for k, d in enumerate(patches):
            ps = [an["observed"]["settings"][i]["logical_error"][k] for i in idx]
            ks = [round(p * N) for p in ps]
            assert all(abs(p * N - kk) < 1e-6 for p, kk in zip(ps, ks)), (folder, ps)
            pm = [an["predicted_at_measured_dephasing"]["settings"][i]["logical_error"][k] for i in idx]
            pa = ([an["predicted_from_calibration_after"]["settings"][i]["logical_error"][k] for i in idx]
                  if an.get("predicted_from_calibration_after") else None)
            p0 = [pred["0.0"]["settings"][i]["logical_error"][k] for i in idx]
            p1 = [pred["1.0"]["settings"][i]["logical_error"][k] for i in idx]
            out[(basis, d)] = {
                "folder": folder.name, "k": ks, "N": N, "p": ps,
                "pred_measured_f": pm, "pred_after": pa, "pred_f0": p0, "pred_f1": p1,
                "f": an["decoupled_dephasing"],
                "sigma_f": an["idle_test"]["median_uncertainty"],
                "stored_fit": an["observed"]["fits"][basis]["per_round"][d],
                "stored_pred_fit": an["predicted_at_measured_dephasing"]["fits"][basis]["per_round"][d],
                "stored_window": an["observed"]["fits"][basis]["from_round_2"]["per_round"][d],
                "stored_pred_window": an["predicted_at_measured_dephasing"]["fits"][basis]["from_round_2"]["per_round"][d],
                "stored_tests": {kk: v for kk, v in an["agreement"]["tests"].items()
                                 if kk.startswith(basis + " eps" + d) or (kk == basis + " lambda")},
            }
    return out, run["rep"]


def main():
    reps = []
    for di, day in enumerate(DAYS):
        for rep in (1, 2):
            folders = ([RUNS / f"ibm_fez-{day}-r1"] if rep == 1
                       else [RUNS / f"ibm_fez-{day}-r2-d3", RUNS / f"ibm_fez-{day}-r2-d5"])
            entry = {"day": day, "day_index": di + 1, "rep": rep, "placement": rep, "curves": {}}
            for f in folders:
                cur, info = curves_from(f)
                assert info["day"] == day and info["rep"] == rep, (f, info)
                for (b, d), v in cur.items():
                    entry["curves"][f"{b}{d}"] = v
            assert sorted(entry["curves"]) == ["X3", "X5", "Z3", "Z5"], entry["curves"].keys()
            reps.append(entry)
    OUT.write_text(json.dumps(reps, indent=1))
    n_ge = sum(1 for r in reps for c in r["curves"].values() for p in c["p"] if p >= 0.5)
    where = sorted({(r["day"], r["rep"], key) for r in reps for key, c in r["curves"].items() if max(c["p"]) >= 0.5})
    print("curves:", sum(len(r["curves"]) for r in reps), "shots:", {c["N"] for r in reps for c in r["curves"].values()})
    print("points with P>=0.5:", n_ge, "in curves:", where)


if __name__ == "__main__":
    main()

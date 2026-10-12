"""Detection-event fractions, device against model, by detector class and round (Fig. detfrac, Sec. agreement).

The detection-event fraction of a detector is the share of shots in which it fires. For every
job and memory circuit it is read from analysis.json: the device's ("observed") and the
model's at the job's measured f ("predicted_at_measured_dephasing", 200,000 simulated shots).
Classes (detectors.detector_class): X stabilizers of weight 4 (bulk) and 2 (boundary), Z
stabilizers (weight 2d), flags, relays. Round t = 1..n compares two measured rounds, except
that t = 1 compares with the prepared value; t = n + 1 is the final layer from the data readout.

  rounds     per placement (or both), basis, distance and class, the 8-round circuits: the
             mean fraction over the class's detectors in each round t, device and model, mean
             over the days (plus or minus the SE over the days) (Fig. detfrac)
  bulk       the same over rounds t = 2..n of the 8-round circuits: device / model per class
  detectors  every detector of every memory circuit of the 15 jobs (14,280): device / model per
             detector by class, basis and distance: median, quartiles, share above one, and
             the ratio of summed fractions; and by position in time (t = 1, bulk, final layer).
             It replaces the count of detectors beyond sqrt(2 ln N) standard errors.
  rise       device - model at t = 2 and at t = n of the 8-round circuits, by class

Reads runs/ only; seconds.

    python docs/analysis/detection_events.py     # writes results/detection_events.json
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from heavyhex.experiment.circuits import SETTINGS

import detectors as DT
from common import jobs, write_json

PRED = "predicted_at_measured_dephasing"
LONGEST = 8
CLASSES = {"X": ("X weight 4", "X weight 2"), "Z": ("Z", "flag", "relay")}


def collect() -> list[dict]:
    """One row per detector: job, placement, day, basis, rounds, patch, class, t, device, model."""
    rows = []
    for job in jobs():
        obs, pred = job.analysis["observed"]["settings"], job.analysis[PRED]["settings"]
        for i, s in enumerate(SETTINGS):
            if s.kind != "memory":
                continue
            for lab, o, p in zip(
                obs[i]["detectors"], obs[i]["detector_rates"], pred[i]["detector_rates"]
            ):
                patch, r, name = lab
                rows.append(
                    {
                        "job": job.name,
                        "placement": job.rep,
                        "day": job.day,
                        "basis": s.basis,
                        "rounds": s.rounds,
                        "patch": str(patch),
                        "class": DT.detector_class(name),
                        "t": int(r) + 1,
                        "device": o,
                        "model": p,
                    }
                )
    return rows


def per_round(rows: list[dict]) -> dict:
    """Mean over a class's detectors per (day, placement, basis, patch, class, t), 8 rounds;
    then the mean and SE over days, per placement and for both placements together."""
    cell = defaultdict(lambda: ([], []))
    for r in rows:
        if r["rounds"] != LONGEST:
            continue
        k = (r["placement"], r["day"], r["basis"], r["patch"], r["class"], r["t"])
        cell[k][0].append(r["device"])
        cell[k][1].append(r["model"])
    means = {k: (np.mean(d), np.mean(m)) for k, (d, m) in cell.items()}
    out = {}
    for basis, classes in CLASSES.items():
        for patch in ("3", "5"):
            for c in classes:
                for pl in (1, 2, "both"):
                    ts = sorted({k[5] for k in means if k[2:5] == (basis, patch, c)})
                    if not ts:
                        continue
                    dev, mod = [], []
                    for t in ts:
                        sel = [
                            v
                            for k, v in means.items()
                            if k[2:6] == (basis, patch, c, t) and (pl == "both" or k[0] == pl)
                        ]
                        a = np.array(sel)
                        dev.append([a[:, 0].mean(), a[:, 0].std(ddof=1) / np.sqrt(len(a))])
                        mod.append([a[:, 1].mean(), a[:, 1].std(ddof=1) / np.sqrt(len(a))])
                    out[f"{basis} d{patch} {c} placement {pl}"] = {
                        "t": ts,
                        "repetitions": len(sel),
                        "device": dev,
                        "model": mod,
                    }
    return out


def bulk(curves: dict) -> dict:
    """Device / model of the mean fraction over rounds t = 2..n of the 8-round circuits."""
    out = {}
    for key, c in curves.items():
        keep = [j for j, t in enumerate(c["t"]) if 2 <= t <= LONGEST]
        d = np.mean([c["device"][j][0] for j in keep])
        m = np.mean([c["model"][j][0] for j in keep])
        out[key] = {"device": d, "model": m, "ratio": d / m}
    return out


def ratios(rows: list[dict]) -> dict:
    """Per-detector device / model, by class (and basis, distance, position in time)."""
    groups = defaultdict(list)
    for r in rows:
        pos = "t = 1" if r["t"] == 1 else "final layer" if r["t"] == r["rounds"] + 1 else "bulk"
        for key in (
            "all",
            f"{r['class']}",
            f"{r['basis']} d{r['patch']} {r['class']}",
            f"{r['class']}, {pos}",
            f"{r['basis']} d{r['patch']}",
        ):
            groups[key].append((r["device"], r["model"]))
    out = {}
    for key, v in sorted(groups.items()):
        a = np.array(v)
        q = a[:, 0] / a[:, 1]
        out[key] = {
            "detectors": len(a),
            "median_ratio": float(np.median(q)),
            "quartiles": [float(np.percentile(q, 25)), float(np.percentile(q, 75))],
            "share_above_1": float((q > 1).mean()),
            "ratio_of_sums": float(a[:, 0].sum() / a[:, 1].sum()),
            "mean_device": float(a[:, 0].mean()),
            "mean_model": float(a[:, 1].mean()),
        }
    return out


def rise(curves: dict) -> dict:
    out = {}
    for key, c in curves.items():
        if not key.endswith("both"):
            continue
        at = {t: (c["device"][j][0] - c["model"][j][0]) for j, t in enumerate(c["t"])}
        out[key.removesuffix(" placement both")] = {
            "excess_t2": at[2],
            "excess_t8": at[LONGEST],
            "change": at[LONGEST] - at[2],
        }
    return out


def main() -> None:
    rows = collect()
    curves = per_round(rows)
    out = {
        "method": __doc__,
        "detectors": len(rows),
        "rounds": curves,
        "bulk": bulk(curves),
        "per_detector": ratios(rows),
        "rise": rise(curves),
    }
    write_json("detection_events.json", out)
    print(f"{len(rows)} detectors")
    for k, v in out["per_detector"].items():
        if k == "all" or k.split(",")[0] in DT.DETECTOR_CLASSES:
            print(
                f"  {k:24s} n={v['detectors']:5d} median {v['median_ratio']:.3f}"
                f" IQR {v['quartiles'][0]:.2f}-{v['quartiles'][1]:.2f} >1 {v['share_above_1']:.2f}"
                f" sums {v['ratio_of_sums']:.3f}"
            )
    for k, v in out["bulk"].items():
        if k.endswith("both"):
            print(
                f"  bulk {k}: device {v['device']:.3f} model {v['model']:.3f} ratio {v['ratio']:.2f}"
            )
    for k, v in out["rise"].items():
        print(f"  rise {k}: {v}")


if __name__ == "__main__":
    main()

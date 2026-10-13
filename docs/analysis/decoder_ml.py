"""Exact maximum-likelihood decoding of the distance-3 X memory, against matching (App. decoder).

For the X-memory distance-3 patch of every job at n = 1-4 rounds, the joint probability of
each syndrome and logical value is computed exactly from the model's detector error model
(not decomposed), which takes 2^(4(n+1)+1) numbers, at most 2^21. The maximum-likelihood
decoder picks the likelier logical value of each syndrome; matching is run._decode's graph,
evaluated on every syndrome. The two are compared on

* the model's logical error, exactly, and eps3 from the slope of ln(1 - 2P) over n = 1-4;
* the single-fault mechanisms of decoder_faults.py (decomposed DEM, full symptom) that
  matching miscorrects: how much of their probability maximum likelihood also assigns to the
  wrong logical value, and the posterior of the right one;
* the device's shots, decoded both ways (matching reproduces the stored logical error).

A few minutes on 2 workers, under 1 GB each.

    python docs/analysis/decoder_ml.py      # writes results/decoder_ml.json
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import WORKERS, day_number, jobs, memory_index, write_json
from matching import calibrated_matching, mechanisms, patch_of_observable

ROUNDS = (1, 2, 3, 4)
PATCH = "3"
CHUNK = 1 << 16


def patch_errors(model, dets: list[int], k: int) -> list[tuple[float, int, int]]:
    """(p, syndrome bit mask over dets, flips logical k) for each mechanism of the patch."""
    pos = {d: i for i, d in enumerate(dets)}
    out = []
    for ins in model.detector_error_model(decompose_errors=False).flattened():
        if ins.type != "error":
            continue
        ds = [t.val for t in ins.targets_copy() if t.is_relative_detector_id()]
        obs = 0
        for t in ins.targets_copy():
            if t.is_logical_observable_id():
                obs ^= 1 << t.val
        inside = [d in pos for d in ds]
        if not any(inside) and not (obs >> k) & 1:
            continue
        assert all(inside) and obs & ~(1 << k) == 0, ins  # the patches share no mechanism
        mask = 0
        for d in ds:
            mask ^= 1 << pos[d]
        out.append((ins.args_copy()[0], mask, (obs >> k) & 1))
    return out


def joint(errors, nd: int) -> np.ndarray:
    """P[o, s]: probability that the logical value flips (o) with syndrome s."""
    v = np.zeros(1 << (nd + 1))
    v[0] = 1.0
    index = np.arange(v.size)
    for p, mask, o in errors:
        v = (1 - p) * v + p * v[index ^ (mask | o << nd)]
    return v.reshape(2, -1)


def matching_table(model, dets: list[int], k: int) -> np.ndarray:
    """Matching's logical prediction for every syndrome of the patch."""
    matching = calibrated_matching(model)
    nd = len(dets)
    out = np.empty(1 << nd, dtype=np.uint8)
    bits = np.arange(nd)
    for start in range(0, 1 << nd, CHUNK):
        s = np.arange(start, min(1 << nd, start + CHUNK))
        syndromes = np.zeros((len(s), model.num_detectors), dtype=np.uint8)
        syndromes[:, dets] = (s[:, None] >> bits) & 1
        out[s] = matching.decode_batch(syndromes)[:, k]
    return out


def eps(p: list[float]) -> float:
    """Logical error per round from the slope of ln(1 - 2P) over the round counts."""
    slope = np.polyfit(ROUNDS, np.log(1 - 2 * np.asarray(p)), 1)[0]
    return float((1 - np.exp(slope)) / 2)


def one(job_index: int) -> tuple[str, dict]:
    job = jobs()[job_index]
    rows = []
    for n in ROUNDS:
        idx = memory_index("X", n)
        model, records, labels = job.model(idx)
        k = {p: i for i, p in patch_of_observable(job, labels).items()}[PATCH]
        dets = [i for i, (p, _, _) in enumerate(labels) if str(p) == PATCH]
        nd = len(dets)
        P = joint(patch_errors(model, dets, k), nd)
        mwpm = matching_table(model, dets, k)
        ml = (P[1] > P[0]).astype(np.uint8)
        s = np.arange(1 << nd)

        # single faults, as decoder_faults.py counts them
        probs, mdets, mobs, _ = mechanisms(model)
        pos = {d: i for i, d in enumerate(dets)}
        miscorrected = both = ml_wrong = 0.0
        count = [0, 0]
        posterior = []
        for p, ds, o in zip(probs, mdets, mobs):
            if not ds or any(d not in pos for d in ds):
                continue
            mask = sum(1 << pos[d] for d in ds)
            flip = int(o >> k) & 1
            ml_wrong += p * (ml[mask] != flip)
            if mwpm[mask] == flip:
                continue
            count[0] += 1
            miscorrected += p
            if ml[mask] != flip:
                count[1] += 1
                both += p
            posterior.append((p, P[flip, mask] / P[:, mask].sum()))

        fired, flips = job.chip_detectors(idx, model, records)
        syn = (fired[:, dets].astype(np.int64) << np.arange(nd)).sum(axis=1)
        actual = flips[:, k].astype(np.uint8)
        device_mwpm = float((mwpm[syn] != actual).mean())
        stored = job.analysis["observed"]["settings"][idx]["logical_error"]
        assert abs(device_mwpm - stored[k]) < 1e-12, (job.name, n, device_mwpm, stored)
        w = np.array([q for q, _ in posterior])
        rows.append(
            {
                "rounds": n,
                "detectors": nd,
                "model_matching": float(P[1 - mwpm, s].sum()),
                "model_ml": float(np.minimum(P[0], P[1]).sum()),
                "model_stored_200000_shots": job.analysis["predicted_at_measured_dephasing"][
                    "settings"
                ][idx]["logical_error"][k],
                "device_matching": device_mwpm,
                "device_ml": float((ml[syn] != actual).mean()),
                "single_faults_matching_miscorrects": {"count": count[0], "p": miscorrected},
                "of_which_ml_also_wrong": {"count": count[1], "p": both},
                "single_faults_ml_wrong": ml_wrong,
                "right_value_posterior_weighted_mean": float(
                    (w * np.array([q for _, q in posterior])).sum() / w.sum()
                )
                if posterior
                else None,
            }
        )
    out = {
        "day": day_number(job.day),
        "pair": job.rep,
        "by_rounds": rows,
        "eps3": {
            key: eps([r[key] for r in rows])
            for key in ("model_matching", "model_ml", "device_matching", "device_ml")
        },
    }
    e = out["eps3"]
    out["model_ml_over_matching"] = e["model_ml"] / e["model_matching"]
    out["device_ml_minus_matching_pp"] = 100 * (e["device_ml"] - e["device_matching"])
    last = rows[-1]
    out["n4_share_of_miscorrected_p_ml_also_wrong"] = (
        last["of_which_ml_also_wrong"]["p"] / last["single_faults_matching_miscorrects"]["p"]
    )
    print(
        job.name,
        f"model eps3 ML/matching {out['model_ml_over_matching']:.3f};",
        f"device ML - matching {out['device_ml_minus_matching_pp']:+.2f} pp;",
        f"n=4 ML also wrong on {out['n4_share_of_miscorrected_p_ml_also_wrong']:.0%}",
        flush=True,
    )
    return job.name, out


def main() -> None:
    picked = [i for i, job in enumerate(jobs()) if PATCH in job.info["chips"]]
    with ProcessPoolExecutor(WORKERS) as pool:
        patches = dict(pool.map(one, picked))
    n4 = [v["by_rounds"][-1] for v in patches.values()]
    every = [r for v in patches.values() for r in v["by_rounds"]]
    mis4 = sum(r["single_faults_matching_miscorrects"]["p"] for r in n4)
    both4 = sum(r["of_which_ml_also_wrong"]["p"] for r in n4)
    mis = sum(r["single_faults_matching_miscorrects"]["p"] for r in every)
    both = sum(r["of_which_ml_also_wrong"]["p"] for r in every)
    ratio = [v["model_ml_over_matching"] for v in patches.values()]
    shift = [v["device_ml_minus_matching_pp"] for v in patches.values()]
    stored_z = [  # the stored predictions are 200,000-shot samples of the same model
        abs(r["model_stored_200000_shots"] - r["model_matching"])
        / np.sqrt(r["model_matching"] * (1 - r["model_matching"]) / 200_000)
        for r in every
    ]
    write_json(
        "decoder_ml.json",
        {
            "what": "exact maximum likelihood against matching, X memory, distance 3, n = 1-4",
            "summary": {
                "patch_days": len(patches),
                "model_eps3_drop_with_ml_range": [1 - max(ratio), 1 - min(ratio)],
                "device_eps3_change_pp_range": [min(shift), max(shift)],
                "device_eps3_change_pp_mean": float(np.mean(shift)),
                "n4_miscorrected_single_fault_p": [mis4, both4, both4 / mis4],
                "n1to4_miscorrected_single_fault_p": [mis, both, both / mis],
                "n4_share_ml_also_wrong_range": [
                    min(v["n4_share_of_miscorrected_p_ml_also_wrong"] for v in patches.values()),
                    max(v["n4_share_of_miscorrected_p_ml_also_wrong"] for v in patches.values()),
                ],
                "stored_prediction_against_exact_matching_max_abs_z": max(stored_z),
            },
            "patches": patches,
        },
    )


if __name__ == "__main__":
    main()

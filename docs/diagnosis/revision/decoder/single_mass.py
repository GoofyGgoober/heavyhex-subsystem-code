"""How much probability the miscorrected single faults carry, under the calibration model.

For each job, basis, rounds and patch: the summed DEM probability of the single
mechanisms each decoder miscorrects (first-order contribution to the logical error
of the whole circuit), beside the summed probability of all the patch's mechanisms.
Pairs (d=5): the summed p_a * p_b of miscorrected pairs (second-order contribution).

    python docs/diagnosis/revision/decoder/single_mass.py
"""

from __future__ import annotations

import json

from dec_common import (
    HERE, ROUNDS, calibrated_matching, dense, jobs, mechanisms, memory_index, np,
    patch_of_observable, uniform_matching,
)


def main() -> None:
    out = {}
    for job in jobs():
        out[job.name] = {}
        for basis in "XZ":
            for rounds in ROUNDS:
                model, _, labels = job.model(memory_index(basis, rounds))
                cal = calibrated_matching(model)
                uni = uniform_matching(model, cal)
                probs, dets, obs, _ = mechanisms(model)
                det_patch = np.array([int(str(p)) for p, _, _ in labels])
                S_all = dense(dets, model.num_detectors)
                row = {}
                for k, patch in patch_of_observable(job, labels).items():
                    sel = np.array([i for i, ds in enumerate(dets) if ds and det_patch[ds[0]] == int(patch)
                                    or (not ds and (obs[i] >> k) & 1)])
                    S, O, p = S_all[sel], ((obs[sel] >> k) & 1).astype(np.uint8), probs[sel]
                    r = {"total_p": float(p.sum())}
                    for name, m in (("calibrated", cal), ("uniform", uni)):
                        bad = m.decode_batch(S)[:, k] != O
                        r[f"miscorrected_single_p_{name}"] = float(p[bad].sum())
                    if patch == "5" and rounds in (1, 2, 3):
                        a, b = np.triu_indices(len(sel), 1)
                        syn, act, pp = S[a] ^ S[b], O[a] ^ O[b], p[a] * p[b]
                        r["total_pair_p"] = float(pp.sum())
                        for name, m in (("calibrated", cal), ("uniform", uni)):
                            bad = (m.decode_batch(syn)[:, k] != act) & (syn.sum(1) > 0)
                            r[f"miscorrected_pair_p_{name}"] = float(pp[bad].sum())
                    row[patch] = r
                out[job.name][f"{basis}{rounds}"] = row
                print(job.name, f"{basis}{rounds}", " | ".join(
                    f"d{p}: sum p {r['total_p']:.3f}, miscorr single p cal {r['miscorrected_single_p_calibrated']:.2e}"
                    f" uni {r['miscorrected_single_p_uniform']:.2e}"
                    + (f"; pair p cal {r['miscorrected_pair_p_calibrated']:.2e} uni {r['miscorrected_pair_p_uniform']:.2e}"
                       f" of {r['total_pair_p']:.2e}" if "total_pair_p" in r else "")
                    for p, r in row.items()), flush=True)
    (HERE / "single_mass.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

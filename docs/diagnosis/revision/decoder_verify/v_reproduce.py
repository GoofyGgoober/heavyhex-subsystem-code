"""(1) Reproduce the two decoder counterexamples of the external audit by a different route.

Route A (no sampling): stim's explain_detector_error_model_errors maps each circuit
fault location (instruction offset, qubit, Pauli) to its detector/observable symptom;
pick the forced locations and XOR their symptoms.
Route B (direct): the noiseless model with Z_ERROR(1)/X_ERROR(1) inserted before the
named noise instruction (one-based in the flattened circuit, as the audit numbered them),
sampled with the detector sampler (deterministic).
Decoders: the repo's run._decode (calibrated) and an independent uniform-weight copy of its graph.
"""

from __future__ import annotations

import json

import numpy as np
import stim

from vcommon import HERE, MEM, cal_matching, meta, model, uni_matching

FOLDER = "ibm_fez-2026-10-08-r1"
CASES = [("X", [(798, "Z", 24)]), ("Z", [(3907, "X", 122), (4127, "X", 86)])]


def inject(m: stim.Circuit, faults) -> stim.Circuit:
    flat = m.flattened()
    out = stim.Circuit()
    where = {n: (p, q) for n, p, q in faults}
    for n, ins in enumerate(flat, start=1):
        if n in where:
            p, q = where[n]
            assert q in [t.value for t in ins.targets_copy()], (n, str(ins))
            out.append(f"{p}_ERROR", [q], 1.0)
        c = stim.Circuit()
        c.append(ins)
        out += c.without_noise()
    return out


def symptom_from_explain(m: stim.Circuit, faults):
    flat = m.flattened()
    dets, obs = set(), set()
    for n, p, q in faults:
        found = None
        for e in flat.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
            for loc in e.circuit_error_locations:
                if loc.stack_frames[0].instruction_offset != n - 1:
                    continue
                paulis = [(t.gate_target.pauli_type, t.gate_target.value) for t in loc.flipped_pauli_product]
                if paulis == [(p, q)]:
                    found = e
        assert found is not None, (n, p, q)
        d = {t.dem_target.val for t in found.dem_error_terms if t.dem_target.is_relative_detector_id()}
        o = {t.dem_target.val for t in found.dem_error_terms if t.dem_target.is_logical_observable_id()}
        dets ^= d
        obs ^= o
    return sorted(dets), sorted(obs)


def edges_used(mt, syndrome):
    arr = mt.decode_to_edges_array(syndrome)
    out = []
    for u, v in arr:
        u, v = int(u), int(v)
        v = None if v == -1 else v
        u, v = (v, None) if u == -1 else (u, v)
        d = mt.get_edge_data(u, v) if v is not None else mt.get_boundary_edge_data(u)
        out.append({"edge": [u, v], "p": d["error_probability"], "w": d["weight"], "L": sorted(d["fault_ids"])})
    return out


def main():
    from heavyhex.experiment import run as R

    res = {"folder": FOLDER, "f": meta(FOLDER)["f"], "cases": []}
    for basis, faults in CASES:
        idx = MEM[(basis, 2)]
        m = model(FOLDER, idx)
        labels = meta(FOLDER)["settings"][str(idx)]["labels"]
        dA, oA = symptom_from_explain(m, faults)
        forced = inject(m, faults)
        det, obs = forced.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert (det == det[0]).all()
        dB = np.flatnonzero(det[0]).tolist()
        oB = np.flatnonzero(obs[0]).tolist()
        cal = cal_matching(m)
        uni = uni_matching(cal)
        syn = det[:1]
        pred_repo = R._decode(m, syn)[0].astype(int).tolist()
        pred_uni = uni.decode_batch(syn)[0].astype(int).tolist()
        e_cal = edges_used(cal, det[0])
        e_uni = edges_used(uni, det[0])
        # calibrated weight of uniform's chosen edges
        def calw(es):
            tot = 0
            for e in es:
                u, v = e["edge"]
                d = cal.get_edge_data(u, v) if v is not None else cal.get_boundary_edge_data(u)
                tot += d["weight"]
            return tot
        case = {
            "basis": basis, "faults": faults,
            "detectors_explain": dA, "observables_explain": oA,
            "detectors_inject": dB, "observables_inject": oB,
            "detector_labels": {d: labels[d] for d in dB},
            "pred_repo_decoder": pred_repo, "pred_uniform": pred_uni,
            "calibrated_edges": e_cal, "calibrated_total_w": sum(e["w"] for e in e_cal),
            "uniform_edges": e_uni, "uniform_choice_calibrated_w": calw(e_uni),
        }
        res["cases"].append(case)
        print(json.dumps(case, indent=None, default=str))
    (HERE / "v_reproduce.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

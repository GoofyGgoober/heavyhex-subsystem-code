"""(1) Reproduce the two decoder counterexamples of the external audit with the repo's own code.

Day 5 (runs/ibm_fez-2026-10-08-r1), two-round memory circuits, model built by
Job.model (run.simulated with the job's calibration and measured f), decoded by
run._decode. Faults are forced into a noiseless copy of the circuit at the
audit's instruction positions (1-based in the flattened stim circuit), sampled,
converted to detectors against the ORIGINAL model's reference, and decoded.
Then the matching is opened up: which edges the decoder chose, with what
probabilities, versus the edges of the fault that actually happened; and the
uniform-weight decoder on the same syndrome.

    python docs/diagnosis/revision/decoder/counterexamples.py
"""

from __future__ import annotations

import json

from dec_common import HERE, R, calibrated_matching, jobs, memory_index, np, stim, uniform_matching

EXAMPLES = [
    ("X", [(798, "Z", 24)], 0),
    ("Z", [(3907, "X", 122), (4127, "X", 86)], 1),
]


def edge_table(matching, pairs):
    rows = []
    for u, v in pairs:
        if v == -1:
            a = matching.get_boundary_edge_data(int(u))
        else:
            a = matching.get_edge_data(int(u), int(v))
        rows.append({"edge": [int(u), None if v == -1 else int(v)], "p": a["error_probability"],
                     "weight": a["weight"], "observables": sorted(a["fault_ids"])})
    return rows


def main() -> None:
    job = next(j for j in jobs() if j.name == "2026-10-08-r1")
    out = {"job": job.name, "f": job.f, "examples": []}
    print(f"job {job.name}, measured f = {job.f:.4f}")
    for basis, faults, target in EXAMPLES:
        idx = memory_index(basis, 2)
        model, records, labels = job.model(idx)
        flat = list(model.flattened())
        forced = stim.Circuit()
        where = {n: (p, q) for n, p, q in faults}
        replaced = []
        for number, ins in enumerate(flat, start=1):
            if number in where:
                p, q = where[number]
                assert ins.name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"), ins
                assert q in [t.value for t in ins.targets_copy()], ins
                replaced.append(f"#{number} {ins} -> {p}{q}")
                forced.append(f"{p}_ERROR", [q], 1.0)
            else:
                tmp = stim.Circuit()
                tmp.append(ins)
                forced += tmp.without_noise()
        meas = forced.compile_sampler(seed=1).sample(16)
        dets, flips = model.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
        assert (dets == dets[0]).all() and (flips == flips[0]).all()
        syndrome = dets[:1]
        fired = np.flatnonzero(syndrome[0]).tolist()
        decoded_repo = R._decode(model, syndrome)[0]  # the repository's decoder, unmodified
        cal = calibrated_matching(model)
        uni = uniform_matching(model, cal)
        dec_uni = uni.decode_batch(syndrome)[0]
        chosen = cal.decode_to_edges_array(syndrome[0].astype(bool))
        chosen_uni = uni.decode_to_edges_array(syndrome[0].astype(bool))
        # the edges of the actual fault(s): the DEM mechanisms whose symptoms match each forced fault
        actual_edges = []
        for n, p, q in faults:
            one = stim.Circuit()
            for number, ins in enumerate(flat, start=1):
                if number == n:
                    one.append(f"{p}_ERROR", [q], 1.0)
                else:
                    tmp = stim.Circuit(); tmp.append(ins); one += tmp.without_noise()
            m1 = one.compile_sampler(seed=1).sample(1)
            d1, o1 = model.compile_m2d_converter().convert(measurements=m1, separate_observables=True)
            actual_edges.append({"fault": f"{p}{q} at #{n}", "detectors": np.flatnonzero(d1[0]).tolist(),
                                 "observables": np.flatnonzero(o1[0]).tolist()})
        # the probability the DEM gives the forced fault's own mechanism(s)
        dem = model.detector_error_model(decompose_errors=True)
        own = []
        for a in actual_edges:
            want = set(a["detectors"])
            for ins in dem.flattened():
                if ins.type != "error":
                    continue
                d = set(); o = set()
                for t in ins.targets_copy():
                    if t.is_relative_detector_id():
                        d ^= {t.val}
                    elif t.is_logical_observable_id():
                        o ^= {t.val}
                if d == want and o == set(a["observables"]):
                    own.append({"fault": a["fault"], "dem_line": str(ins), "p": ins.args_copy()[0]})
        lab = {i: [str(x) for x in labels[i]] for i in fired}
        row = {
            "basis": basis, "rounds": 2, "target_patch_distance": (3, 5)[target],
            "replaced": replaced,
            "fired_detectors": fired, "detector_labels": lab,
            "actual_observable_flips": flips[0].astype(int).tolist(),
            "repo_decoder_prediction": decoded_repo.astype(int).tolist(),
            "uniform_decoder_prediction": dec_uni.astype(int).tolist(),
            "actual_fault_symptoms": actual_edges,
            "actual_fault_dem_mechanisms": own,
            "calibrated_matching_edges": edge_table(cal, chosen),
            "uniform_matching_edges": edge_table(cal, chosen_uni),  # shown with calibrated p for comparison
        }
        row["calibrated_correction_weight"] = sum(e["weight"] for e in row["calibrated_matching_edges"])
        row["uniform_choice_calibrated_weight"] = sum(e["weight"] for e in row["uniform_matching_edges"])
        out["examples"].append(row)
        print(f"\n{basis} memory, 2 rounds, target d={(3, 5)[target]}")
        for r in replaced:
            print("  forced", r)
        print("  fired detectors", fired, lab)
        print("  actual flips [L0,L1]", row["actual_observable_flips"],
              " repo decoder", row["repo_decoder_prediction"], " uniform decoder", row["uniform_decoder_prediction"])
        for o in own:
            print(f"  fault's own DEM mechanism: p = {o['p']:.3g}   {o['dem_line']}")
        print(f"  calibrated MWPM chose (total weight {row['calibrated_correction_weight']:.3f}):")
        for e in row["calibrated_matching_edges"]:
            print(f"     {e['edge']}  p={e['p']:.4f}  w={e['weight']:.3f}  L={e['observables']}")
        print(f"  uniform MWPM chose (calibrated weight of that choice {row['uniform_choice_calibrated_weight']:.3f}):")
        for e in row["uniform_matching_edges"]:
            print(f"     {e['edge']}  p={e['p']:.4f}  w={e['weight']:.3f}  L={e['observables']}")
    (HERE / "counterexamples.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()

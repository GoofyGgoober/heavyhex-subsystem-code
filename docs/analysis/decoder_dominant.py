"""Which single faults the calibrated decoder miscorrects most, by probability (App. decoder).

decoder_faults.py counts the single faults that calibrated matching gets wrong; here they are
weighted by their probability and traced to the circuit. For the X-memory distance-3 patch of
every job, eight-round circuit, each mechanism of the decomposed detector error model is
decoded alone (as in decoder_faults.py). The miscorrected ones are grouped by the circuit
fault behind them (stim's explanation: Pauli, qubit and instruction), and the largest group is
kept with the matching edges the decoder chose instead. Shares are of the patch's predicted
eps3 (runs/combined.json). Under a minute.

    python docs/analysis/decoder_dominant.py      # writes results/decoder_dominant.json
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import stim

from common import RUNS, jobs, memory_index, read, write_json
from matching import calibrated_matching, mechanisms, patch_of_observable

ROUNDS = 8


def explain(model: stim.Circuit, dem_line: str) -> str:
    """One representative circuit fault of a DEM mechanism, e.g. 'Z on q44 (DEPOLARIZE1)'."""
    dem = stim.DetectorErrorModel(dem_line.replace(" ^ ", " "))
    found = model.explain_detector_error_model_errors(
        dem_filter=dem, reduce_to_one_representative_error=True
    )
    loc = found[0].circuit_error_locations[0]
    paulis = ", ".join(
        f"{t.gate_target.pauli_type} on q{t.gate_target.qubit_value}"
        for t in loc.flipped_pauli_product
    )
    return f"{paulis} ({loc.instruction_targets.gate})"


def main() -> None:
    combined = read(RUNS / "combined.json")
    predicted = {}
    for rep in combined["reps"]:
        day = rep["day"]
        predicted[(day, rep["rep"])] = rep["tests"]["X eps3"]["predicted"]
    out: dict = {"rounds": ROUNDS, "patches": {}}
    for job in jobs():
        if "3" not in job.info["chips"]:
            continue
        model, _, labels = job.model(memory_index("X", ROUNDS))
        matching = calibrated_matching(model)
        probs, dets, obs, comps = mechanisms(model)
        k = {p: i for i, p in patch_of_observable(job, labels).items()}["3"]
        det_patch = np.array([int(str(p)) for p, _, _ in labels])
        groups: dict[str, float] = defaultdict(float)
        chosen: dict[str, list] = {}
        for p, ds, o, comp in zip(probs, dets, obs, comps):
            if not ds or any(det_patch[d] != 3 for d in ds):
                continue
            syndrome = np.zeros(model.num_detectors, dtype=np.uint8)
            syndrome[ds] = 1
            if matching.decode(syndrome)[k] == (o >> k) & 1:
                continue
            fault = explain(model, f"error({p}) {comp}")
            groups[fault] += float(p)
            if fault not in chosen:
                edges = matching.decode_to_edges_array(syndrome)
                chosen[fault] = [
                    matching.get_boundary_edge_data(int(u))["error_probability"]
                    if v == -1
                    else matching.get_edge_data(int(u), int(v))["error_probability"]
                    for u, v in edges
                ]
        total = sum(groups.values())
        top = max(groups, key=groups.get) if groups else None
        eps3 = predicted[(job.day, job.rep)]
        out["patches"][job.name] = {
            "predicted_eps3": eps3,
            "miscorrected_per_round": total / ROUNDS,
            "share_of_predicted_eps3": total / ROUNDS / eps3,
            "largest": None
            if top is None
            else {
                "fault": top,
                "p_per_round": groups[top] / ROUNDS,
                "share_of_miscorrected": groups[top] / total,
                "decoder_chose_edges_p": chosen[top],
            },
        }
        print(
            job.name, f"{total / ROUNDS / eps3:.0%} of eps3;", out["patches"][job.name]["largest"]
        )
    shares = [v["share_of_predicted_eps3"] for v in out["patches"].values()]
    out["share_range"] = [min(shares), max(shares)]
    write_json("decoder_dominant.json", out)


if __name__ == "__main__":
    main()

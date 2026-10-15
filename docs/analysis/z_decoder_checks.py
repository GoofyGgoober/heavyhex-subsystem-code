"""Z-memory decoder sensitivity at stored f=1, using only saved shots.

Remove flag/relay DETECTOR declarations, not gates, measurements or observables.
Compare the original matching decoder, matching on the reduced detector set,
and exact maximum likelihood for that reduced set under the undecomposed DEM.
ML is model-optimal for these retained bits, not for all hardware information.
Both decoders see the very same simulated and device shots. No hardware access.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:docs/analysis \
        python docs/analysis/z_decoder_checks.py

One worker; --job restricts to a folder suffix, --shots controls model sampling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import stim
from decoder_graph import shortest_logical

from common import jobs, memory_index, write_json
from matching import calibrated_matching, patch_of_observable


def stabilizers_only(model, labels):
    """Drop auxiliary detector declarations, retaining their physical readouts."""
    if len(labels) != model.num_detectors:
        raise ValueError("One label is required for every detector")
    keep = np.array([" " not in label[2] for label in labels])
    out, col = stim.Circuit(), 0
    for instruction in model.flattened():
        if instruction.name == "DETECTOR":
            if keep[col]:
                out.append(instruction)
            col += 1
        else:
            out.append(instruction)
    return out, [label for label, use in zip(labels, keep, strict=True) if use], keep


def walsh(values):
    """Unnormalized Walsh-Hadamard transform, in natural XOR-mask order."""
    values = np.array(values, dtype=float, copy=True)
    n = len(values)
    if n == 0 or n & (n - 1):
        raise ValueError("The transform length must be a positive power of two")
    width = 1
    while width < n:
        blocks = values.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    return values


def xor_distribution(bits, mechanisms):
    """Exact distribution of independent Bernoulli XOR masks via Fourier products.

    A mask m contributes log(1-2p) on characters with odd parity against m.
    Transforming the sparse log weights evaluates all character products at once.
    Half-probability and >half-probability errors are handled explicitly.
    """
    if not 0 <= bits <= 21:
        raise ValueError("Exact tables are limited to 21 bits (20 detectors plus logical)")
    weights = np.zeros(1 << bits)
    offset, halves = 0, []
    for mask, probability in mechanisms:
        if not 0 <= mask < len(weights) or not 0 <= probability <= 1:
            raise ValueError("Invalid error mask or probability")
        if not mask:
            continue
        if probability > 0.5:
            offset ^= mask
            probability = 1 - probability
        if probability == 0.5:
            halves.append(mask)
        else:
            weights[mask] += np.log1p(-2 * probability)
    characters = np.exp(0.5 * (weights.sum() - walsh(weights)))
    probability = walsh(characters) / len(weights)
    if probability.min() < -1e-12:
        raise ArithmeticError("Negative probability beyond transform roundoff")
    probability = np.maximum(probability, 0)
    indices = np.arange(len(weights))
    for mask in halves:
        probability = (probability + probability[indices ^ mask]) / 2
    probability = probability[indices ^ offset]
    probability /= probability.sum()
    return probability


def ml_table(dem, detectors, observable):
    """P[syndrome, logical], marginalizing any other patch's variables."""
    positions = {detector: bit for bit, detector in enumerate(detectors)}
    if len(positions) != len(detectors):
        raise ValueError("Detector columns must be unique")
    mechanisms = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id() and target.val in positions:
                mask ^= 1 << positions[target.val]
            elif target.is_logical_observable_id() and target.val == observable:
                mask ^= 1 << len(detectors)
        mechanisms.append((mask, instruction.args_copy()[0]))
    return xor_distribution(len(detectors) + 1, mechanisms).reshape(2, -1).T


def ml_decode(table, detections):
    syndrome = np.asarray(detections, dtype=np.int64) @ (1 << np.arange(detections.shape[1]))
    return table[syndrome, 1] > table[syndrome, 0]


def compare(job, rounds, shots, seed):
    index = memory_index("Z", rounds)
    model, records, labels = job.model(index, f=1.0)
    reduced, reduced_labels, keep = stabilizers_only(model, labels)
    original_matching, reduced_matching = map(calibrated_matching, (model, reduced))
    device, flips = job.chip_detectors(index, model, records)
    red_device, red_flips = job.chip_detectors(index, reduced, records)
    np.testing.assert_array_equal(red_device, device[:, keep])
    np.testing.assert_array_equal(red_flips, flips)
    simulated, sim_flips = model.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    pairs = [(device, flips), (simulated, sim_flips)]
    original = [original_matching.decode_batch(d) for d, _ in pairs]
    reduced_pred = [reduced_matching.decode_batch(d[:, keep]) for d, _ in pairs]
    dem = reduced.detector_error_model(decompose_errors=False)
    rows = []
    for observable, patch in patch_of_observable(job, labels).items():
        columns = [i for i, label in enumerate(reduced_labels) if str(label[0]) == patch]
        table = ml_table(dem, columns, observable)
        row = {
            "job": job.name,
            "distance": int(patch),
            "rounds": rounds,
            "retained_detectors": len(columns),
            "seed": seed,
            "exact_model_ml_failure": float(table.min(axis=1).sum()),
        }
        for name, matching in (("original", original_matching), ("reduced", reduced_matching)):
            path = shortest_logical(matching.edges(), matching.num_detectors, observable)
            row[name + "_graph_distance"] = None if path is None else len(path)
        for j, (detections, truth) in enumerate(pairs):
            pred = {
                "original": original[j][:, observable],
                "reduced": reduced_pred[j][:, observable],
                "ml": ml_decode(table, detections[:, keep][:, columns]),
            }
            baseline = pred["original"] != truth[:, observable]
            row[("device", "model")[j]] = {
                "shots": len(truth),
                "failures": {
                    name: int((p != truth[:, observable]).sum()) for name, p in pred.items()
                },
                "paired_change": {
                    name: {
                        "rescued": int((baseline & (p == truth[:, observable])).sum()),
                        "spoiled": int((~baseline & (p != truth[:, observable])).sum()),
                    }
                    for name, p in pred.items()
                    if name != "original"
                },
            }
        rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["distance"], row["rounds"])].append(row)
    out = []
    for (distance, rounds), group in sorted(groups.items()):
        item = {"distance": distance, "rounds": rounds, "patch_jobs": len(group)}
        for sample in ("device", "model"):
            n = sum(row[sample]["shots"] for row in group)
            item[sample] = {
                "shots": n,
                "failure_fraction": {
                    name: sum(row[sample]["failures"][name] for row in group) / n
                    for name in ("original", "reduced", "ml")
                },
            }
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", default="z_decoder_checks.json")
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    rows = []
    selected = [job for job in jobs() if args.job is None or job.name in args.job]
    if not selected or (args.job and set(args.job) != {job.name for job in selected}):
        parser.error("Unknown job selection")
    for job in selected:
        for rounds in (1, 2, 3, 4):
            rows.extend(compare(job, rounds, args.shots, args.seed + rounds))
        print(job.name, "done", flush=True)
    print(
        write_json(
            args.output,
            {
                "method": __doc__,
                "seed": args.seed,
                "model_shots_per_circuit": args.shots,
                "f": 1.0,
                "rows": rows,
                "summary": summarize(rows),
            },
        )
    )


if __name__ == "__main__":
    main()

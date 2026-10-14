"""Three checks of the decoder's matching graph (App. decoder): Stim's observable-only pieces,
single faults on a uniform chip, and the shortest logical path in edges.

Pieces. With decompose_errors=True Stim writes a fault that fires more than two detectors as
pieces of at most two detectors each. In distance-3 Z memory one fault per round has no split
into edges whose observables add up to its own: an X or Y on data qubit Q2 partway through the
round (at its CZ with the Z1Z2 relay or the X2X5 ancilla, or idle between them), which fires
one of Q2's two stabilizers in that round and the other in the next, the Z1Z2 relay and the
X2X5 flag, and flips no logical value. Its pieces are the boundary edges of the four detectors
as the graph already has them, and those flip the logical value once, so Stim adds a piece
that holds only the observable (`D1 ^ D2 L0 ^ D7 ^ D8 ^ L0`). PyMatching 2.4
ignores a piece without detectors (and refuses one with correlations), so in the graph the
fault's own pieces flip the logical value. For every memory circuit of every job: the pieces per
patch and their probability; the graph with them removed (unchanged); edges whose pieces carry
different observables (none, so the label PyMatching keeps when it merges parallel pieces, the
first it reads, is never at stake); each such fault decoded alone with the calibrated and the
uniform weights, and whether matching then uses exactly the fault's own pieces; and the fault
with its observable folded into each of its detector pieces in turn: the graph, and the
device's shots and 10,000 of the model's (refit.sim_seed, as corr_matching.py) decoded with
and without correlations, against the paper's decoder and corr_matching.py's (pieces dropped).
The distance-3 Z-memory single-fault scan of decoder_faults.py is repeated to see which of its
miscorrections are these faults.

Uniform chip. Every coupler and qubit at the day's median (noisy.typical), f = 1 and decoupled
idles, as in uniform_scale.py (day 5) and figure3.py (day 1): each single fault of the
decomposed detector error model decoded alone, in the pre-run simulator (noisy.noisy_circuit,
one patch) with f = 1, f = 0 and the idle noise removed, and in the per-job model of the day's
first job (common.Job.model with the uniform calibration). Mechanisms are counted as
decoder_faults.py counts them, lines of the decomposed model, and also by distinct symptom.
The miscorrected probability per round over the simulated eps3 (noisy.decay, 20,000 shots, as
uniform_scale.py) is their share of the uniform chip's eps3.

Path. The fewest edges of the matching graph that flip the logical value without a detection
event (breadth-first over node and parity, as decoder_weights.py's hops), every circuit, with
one such path and the detector-error-model lines its edges come from. At distance 5 in Z memory
two of its three edges are boundary edges of interior stabilizers that exist only as pieces of
the same kind of fault (a data X error seen by a flag and a relay).

About 3.5 minutes on 2 workers, 0.5 GB each.

    python docs/analysis/decoder_graph.py      # writes results/decoder_graph.json
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from multiprocessing import Pool

import numpy as np
import pymatching
import stim

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.patches.operators import build_operators
from heavyhex.simulation.noisy import decay, noisy_circuit, typical

import refit
from common import RUNS, WORKERS, jobs, memory_index, read, write_json
from matching import calibrated_matching, mechanisms, patch_of_observable, uniform_matching

FOLDS = 4  # the faults with an observable-only piece have four detector pieces
DAYS = ("2026-10-04-r1", "2026-10-08-r1")  # the uniform chip at day 1's and day 5's medians


# --- the detector error model, line by line


def parts(ins: stim.DemInstruction) -> list[list[stim.DemTarget]]:
    out: list[list[stim.DemTarget]] = [[]]
    for t in ins.targets_copy():
        if t.is_separator():
            out.append([])
        else:
            out[-1].append(t)
    return out


def detectors(part) -> tuple[int, ...]:
    return tuple(sorted(t.val for t in part if t.is_relative_detector_id()))


def mask(part) -> int:
    m = 0
    for t in part:
        if t.is_logical_observable_id():
            m ^= 1 << t.val
    return m


def join(pieces: list[tuple[tuple[int, ...], int]]) -> list[stim.DemTarget]:
    out: list[stim.DemTarget] = []
    for j, (ds, m) in enumerate(pieces):
        out += [stim.target_separator()] if j else []
        out += [stim.target_relative_detector_id(d) for d in ds]
        out += [stim.target_logical_observable_id(k) for k in range(m.bit_length()) if m >> k & 1]
    return out


def lonely(ins: stim.DemInstruction) -> bool:
    """An error line with a piece that fires no detector."""
    return ins.type == "error" and any(not detectors(p) for p in parts(ins))


def rewrite(dem: stim.DetectorErrorModel, fold: int | None) -> stim.DetectorErrorModel:
    """The model with each observable-only piece dropped (fold None) or its observable moved
    onto the fold-th detector piece of the same line."""
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if not lonely(ins):
            out.append(ins)
            continue
        ps = [(detectors(p), mask(p)) for p in parts(ins)]
        kept = [list(x) for x in ps if x[0]]
        if fold is not None:
            for _, m in (x for x in ps if not x[0]):
                kept[fold % len(kept)][1] ^= m
        out.append("error", ins.args_copy(), join([(ds, m) for ds, m in kept]))
    return out


def graph(m: pymatching.Matching) -> list:
    return sorted(
        (u, -1 if v is None else v, tuple(sorted(a["fault_ids"])), round(a["weight"], 12))
        for u, v, a in m.edges()
    )


def mixed_labels(dem: stim.DetectorErrorModel) -> int:
    """Detector sets (graph edges) whose pieces carry different observables."""
    seen: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for ins in dem.flattened():
        if ins.type == "error":
            for p in parts(ins):
                if detectors(p):
                    seen[detectors(p)].add(mask(p))
    return sum(len(v) > 1 for v in seen.values())


def edge(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v)) if v != -1 else (u, -1)


# --- roles of the chip's qubits


def role(job, q: int) -> str:
    """'data Q2', 'X2X5 ancilla', 'Z1Z2 relay' or 'Z2Z3Z5Z6 ancilla' for chip qubit q."""
    for patch, chips in job.info["chips"].items():
        if q in chips:
            return local_role(int(patch), chips.index(q))
    return f"q{q}"


def local_role(d: int, i: int) -> str:
    r = qubit_roles(build_operators(d))
    if i < r.num_data:
        return f"data Q{i + 1}"
    names = {a: f"{g} ancilla" for g, a in {**r.x_ancillas, **r.z_ancillas}.items()}
    names |= {relay: f"{g} relay" for g, arms in r.z2_arms.items() for _, relay in arms}
    return names[i]


def explain(model: stim.Circuit, symptom: list[int], flips: int, name) -> list[str]:
    """Every circuit fault behind one symptom: 'X on data Q2, Y on X2X5 ancilla (DEPOLARIZE2)'."""
    line = " ".join(f"D{d}" for d in symptom)
    line += "".join(f" L{k}" for k in range(flips.bit_length()) if flips >> k & 1)
    found = model.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(f"error(0.1) {line}"),
        reduce_to_one_representative_error=False,
    )
    out = []
    for e in found:
        for loc in e.circuit_error_locations:
            pauli = ", ".join(
                f"{t.gate_target.pauli_type} on {name(t.gate_target.qubit_value)}"
                for t in loc.flipped_pauli_product
            )
            out.append(f"{pauli} ({loc.instruction_targets.gate})")
    return sorted(set(out))


# --- the shortest logical path in edges


def shortest_logical(edges: list, boundary: int, k: int) -> list | None:
    """One logical with the fewest edges: an odd-parity closed walk over (node, parity),
    each node tried as its start, the search cut at the best length so far."""
    adj: dict[int, list] = defaultdict(list)
    for u, v, a in edges:
        v = boundary if v is None else v
        flip = 1 if k in a["fault_ids"] else 0
        adj[u].append((v, flip, a))
        adj[v].append((u, flip, a))
    best: list | None = None
    for s in [boundary, *sorted(n for n in adj if n != boundary)]:
        prev: dict = {(s, 0): None}
        queue, depth = deque([(s, 0)]), {(s, 0): 0}
        while queue:
            x, p = queue.popleft()
            if (x, p) == (s, 1):
                break
            if best is not None and depth[(x, p)] >= len(best) - 1:
                continue
            for y, flip, a in adj[x]:
                st = (y, p ^ flip)
                if st not in prev:
                    prev[st] = ((x, p), (x, y), a)
                    depth[st] = depth[(x, p)] + 1
                    queue.append(st)
        if (s, 1) in prev and (best is None or depth[(s, 1)] < len(best)):
            path, st = [], (s, 1)
            while prev[st] is not None:
                st, xy, a = prev[st]
                path.append((xy, a))
            best = path[::-1]
    return best


def path_rows(job, model, dem, labels, path, boundary) -> list[dict]:
    """Each edge of a path: its detectors, probability and observable, and the model's lines
    it comes from (whole, or as a piece of a decomposed line)."""
    whole: dict[tuple[int, ...], float] = defaultdict(float)
    within: dict[tuple[int, ...], list] = defaultdict(list)
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        ps = parts(ins)
        for p in ps:
            if len(ps) == 1:
                whole[detectors(p)] += ins.args_copy()[0]
            else:
                within[detectors(p)].append(ins)
    rows = []
    for (x, y), a in path:
        ds = tuple(sorted(d for d in (x, y) if d != boundary))
        row = {
            "detectors": [list(map(str, labels[d])) for d in ds],
            "p": a["error_probability"],
            "flips_logical": bool(a["fault_ids"]),
            "p_as_a_whole_line": whole.get(ds, 0.0),
            "lines_with_it_as_a_piece": len(within.get(ds, [])),
        }
        if ds not in whole and within.get(ds):
            top = max(within[ds], key=lambda i: i.args_copy()[0])
            full: set[int] = set()
            flips = 0
            for p in parts(top):
                full ^= set(detectors(p))
                flips ^= mask(p)
            row["largest_line"] = str(top)
            row["largest_line_detectors"] = [list(map(str, labels[d])) for d in sorted(full)]
            row["largest_line_faults"] = explain(model, sorted(full), flips, lambda q: role(job, q))
        rows.append(row)
    return rows


# --- per job: pieces, folds, decoding, single faults and paths


def job_scan(k: int) -> dict:
    t0 = time.time()
    job = jobs()[k]
    out: dict = {"name": job.name, "patches": job.analysis["patches"], "circuits": {}}
    for basis in "XZ":
        for n in ROUNDS:
            i = memory_index(basis, n)
            model, records, labels = job.model(i)
            dem = model.detector_error_model(decompose_errors=True)
            obs_patch = patch_of_observable(job, labels)
            det_patch = np.array([int(str(p)) for p, _, _ in labels])
            paper = calibrated_matching(model)
            row: dict = {
                "pieces": Counter(),
                "pieces_p": defaultdict(float),
                "graph_unchanged_without_pieces": graph(
                    pymatching.Matching.from_detector_error_model(rewrite(dem, None))
                )
                == graph(paper),
                "edges_with_mixed_observables": mixed_labels(dem),
            }
            lines = [ins for ins in dem.flattened() if lonely(ins)]
            for ins in lines:
                for p in parts(ins):
                    if not detectors(p):
                        patch = obs_patch[mask(p).bit_length() - 1]
                        row["pieces"][patch] += 1
                        row["pieces_p"][patch] += ins.args_copy()[0]
            if lines:
                row |= pieces_decoded(job, i, model, records, dem, paper, lines, obs_patch)
                if n == 1:
                    ins = lines[0]
                    full = sorted(set().union(*[set(detectors(p)) for p in parts(ins)]))
                    row["example_line"] = str(ins)
                    row["example_detectors"] = [list(map(str, labels[d])) for d in full]
                    row["example_faults"] = explain(model, full, 0, lambda q: role(job, q))
            if basis == "Z" and "3" in obs_patch.values():
                row["z_d3_single_scan"] = single_scan(model, paper, obs_patch, det_patch)
            row["shortest_logical_edges"] = {}
            for kk, patch in obs_patch.items():
                mine = [e for e in paper.edges() if det_patch[e[0]] == int(patch)]
                path = shortest_logical(mine, model.num_detectors, kk)
                assert path is not None
                row["shortest_logical_edges"][patch] = len(path)
                if basis == "Z" and patch == "5" and n in (1, 2):
                    row["z_d5_path"] = path_rows(job, model, dem, labels, path, model.num_detectors)
            out["circuits"][f"{basis}{n}"] = row
    out["seconds"] = time.time() - t0
    print(f"{job.name}: {out['seconds']:.0f} s", flush=True)
    return out


def pieces_decoded(job, i, model, records, dem, paper, lines, obs_patch) -> dict:
    """The faults with an observable-only piece decoded alone, and every shot decoded with the
    piece folded into each of the fault's detector pieces, with and without correlations."""
    uni = uniform_matching(model, paper)
    faults = []
    for ins in lines:
        ps = parts(ins)
        full: set[int] = set()
        flips = 0
        for p in ps:
            full ^= set(detectors(p))
            flips ^= mask(p)
        kk = next(kk for kk in range(len(obs_patch)) if any(mask(p) >> kk & 1 for p in ps))
        syndrome = np.zeros(model.num_detectors, dtype=np.uint8)
        syndrome[sorted(full)] = 1
        own = {
            edge(*detectors(p)) if len(detectors(p)) == 2 else (detectors(p)[0], -1)
            for p in ps
            if detectors(p)
        }
        row = {"p": ins.args_copy()[0], "patch": obs_patch[kk], "flips": flips >> kk & 1}
        for name, m in (("calibrated", paper), ("uniform", uni)):
            row[f"{name}_wrong"] = bool(m.decode(syndrome)[kk] != row["flips"])
            used = {edge(int(u), int(v)) for u, v in m.decode_to_edges_array(syndrome)}
            row[f"{name}_uses_its_own_pieces"] = used == own
        faults.append(row)
    try:
        pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        refused = None
    except Exception as ex:  # PyMatching says why it cannot
        refused = str(ex).splitlines()[0][:160]
    folds = [rewrite(dem, j) for j in range(FOLDS)]
    plain = [pymatching.Matching.from_detector_error_model(f) for f in folds]
    corr = [
        pymatching.Matching.from_detector_error_model(f, enable_correlations=True) for f in folds
    ]
    dropped = pymatching.Matching.from_detector_error_model(
        rewrite(dem, None), enable_correlations=True
    )
    chip = job.chip_detectors(i, model, records)
    sim = model.compile_detector_sampler(seed=refit.sim_seed(i)).sample(
        refit.SIM_SHOTS, separate_observables=True
    )
    differ: dict = {}
    for name, (dets, flips) in (("device", chip), ("model", sim)):
        base = paper.decode_batch(dets)
        base_corr = dropped.decode_batch(dets, enable_correlations=True)
        differ[name] = {
            "shots": len(dets),
            "matching_folded_vs_paper": max(
                int((m.decode_batch(dets) != base).any(1).sum()) for m in plain
            ),
            "correlated_folded_vs_dropped": max(
                int((m.decode_batch(dets, enable_correlations=True) != base_corr).any(1).sum())
                for m in corr
            ),
            "failures_paper": (base != flips).sum(0).tolist(),
            "failures_correlated": (base_corr != flips).sum(0).tolist(),
        }
    return {
        "faults": faults,
        "correlations_refuse_the_piece": refused,
        "graph_unchanged_folded": [graph(m) == graph(paper) for m in plain],
        "shots_decoded_differently": differ,
    }


def single_scan(model, paper, obs_patch, det_patch) -> dict:
    """decoder_faults.py's single-fault scan of the distance-3 patch, with the faults that have
    an observable-only piece marked."""
    k = {p: kk for kk, p in obs_patch.items()}["3"]
    probs, dets, obs, comps = mechanisms(model)
    sel = [j for j, ds in enumerate(dets) if ds and det_patch[ds[0]] == 3]
    lone = np.array(
        [any(not q.strip().startswith("D") for q in comps[j].split(" ^ ")) for j in sel]
    )
    syn = np.zeros((len(sel), model.num_detectors), dtype=np.uint8)
    for r, j in enumerate(sel):
        syn[r, dets[j]] = 1
    actual = (obs[sel] >> k) & 1
    uni = uniform_matching(model, paper)
    out = {"mechanisms": len(sel), "with_observable_only_piece": int(lone.sum())}
    for name, m in (("calibrated", paper), ("uniform", uni)):
        wrong = m.decode_batch(syn)[:, k] != actual
        out[f"{name}_wrong"] = int(wrong.sum())
        out[f"{name}_wrong_with_observable_only_piece"] = int((wrong & lone).sum())
        out[f"{name}_wrong_p"] = float(probs[sel][wrong].sum())
        out[f"{name}_wrong_p_with_observable_only_piece"] = float(probs[sel][wrong & lone].sum())
    return out


# --- the uniform chip


def without_idles(circuit: stim.Circuit) -> stim.Circuit:
    """The same circuit without the idle decay (PAULI_CHANNEL_1, in both simulators)."""
    out = stim.Circuit()
    for ins in circuit.flattened():
        if ins.name != "PAULI_CHANNEL_1":
            out.append(ins)
    return out


def singles(
    model: stim.Circuit, patch_of_det: np.ndarray, patch_k: dict, name, examples: int = 0
) -> dict:
    """Each single fault decoded alone, per patch: counts by line and by distinct symptom, the
    miscorrected probability, the boundary edges' probabilities, and a few examples."""
    m = calibrated_matching(model)
    probs, dets, obs, comps = mechanisms(model)
    bnd = defaultdict(list)
    for u, v, a in m.edges():
        if v is None:
            bnd[int(patch_of_det[u])].append(a["error_probability"])
    out = {}
    for patch, k in patch_k.items():
        sel = [j for j, ds in enumerate(dets) if ds and patch_of_det[ds[0]] == patch]
        syn = np.zeros((len(sel), model.num_detectors), dtype=np.uint8)
        for r, j in enumerate(sel):
            syn[r, dets[j]] = 1
        actual = (obs[sel] >> k) & 1
        wrong = np.flatnonzero(m.decode_batch(syn)[:, k] != actual)
        symptom = [(tuple(dets[j]), int(a)) for j, a in zip(sel, actual)]
        shown = []
        for r in wrong[:examples]:
            j = sel[r]
            used = m.decode_to_edges_array(syn[r])
            shown.append(
                {
                    "line": f"error({probs[j]:.6g}) {comps[j]}",
                    "faults": explain(model, dets[j], int(obs[j]), name)[:3],
                    "decoder_used_edges_p": [
                        (
                            m.get_boundary_edge_data(int(u))
                            if v == -1
                            else m.get_edge_data(int(u), int(v))
                        )["error_probability"]
                        for u, v in used
                    ],
                }
            )
        out[str(patch)] = {
            "mechanisms": len(sel),
            "symptoms": len(set(symptom)),
            "miscorrected": len(wrong),
            "miscorrected_symptoms": len({symptom[r] for r in wrong}),
            "miscorrected_p": float(probs[sel][wrong].sum()),
            "boundary_p": [min(bnd[patch]), max(bnd[patch])],
            "examples": shown,
        }
    return out


def uniform_chip() -> dict:
    out: dict = {}
    for job in (j for j in jobs() if j.name in DAYS):
        cal = typical(job.calibration)
        day: dict = {"pre-run simulator": {}, "per-job model": {}}
        for variant, f, idles in (
            ("f=1", 1.0, True),
            ("f=0", 0.0, True),
            ("f=1 no idle", 1.0, False),
        ):
            rows = {}
            for basis in "XZ":
                for d in (3, 5):
                    for n in ROUNDS:
                        c = noisy_circuit(d, basis, n, cal, decoupling=True, decoupled_dephasing=f)
                        c = c if idles else without_idles(c)
                        dets = np.full(c.num_detectors, d)
                        rows[f"{basis} d{d} n={n}"] = singles(
                            c, dets, {d: 0}, lambda q, d=d: local_role(d, q), 6 * (n == 1)
                        )[str(d)]
            day["pre-run simulator"][variant] = rows
        for basis in "XZ":
            for n in ROUNDS:
                model, _, labels = job.model(memory_index(basis, n), calibration=cal, f=1.0)
                det_patch = np.array([int(str(p)) for p, _, _ in labels])
                patch_k = {int(p): kk for kk, p in patch_of_observable(job, labels).items()}
                for p, r in singles(
                    model, det_patch, patch_k, lambda q, job=job: role(job, q)
                ).items():
                    day["per-job model"][f"{basis} d{p} n={n}"] = r
        day["eps3_simulated"] = {
            basis: decay(
                3, basis, cal, shots=20_000, seed=5, decoupling=True, decoupled_dephasing=1.0
            ).per_round
            for basis in "XZ"
        }
        out[job.day] = day
        print(f"uniform chip, {job.day}", flush=True)
    return out


# --- summary


def summary(per_job: list[dict], uniform: dict) -> dict:
    combined = {(r["day"], r["rep"]): r for r in read(RUNS / "combined.json")["reps"]}
    names = {j.name: j for j in jobs()}
    pieces: Counter = Counter()
    p_line, faults, scan = [], [], Counter()
    graph_ok, folds_ok, mixed = True, True, 0
    differ: Counter = Counter()
    refused, examples = set(), {}
    share = []
    hops: dict[str, set] = defaultdict(set)
    only_pieces = []
    for j in per_job:
        for setting, row in j["circuits"].items():
            basis = setting[0]
            for patch, c in row["pieces"].items():
                pieces[f"{basis} d{patch}"] += c
            graph_ok &= row["graph_unchanged_without_pieces"]
            mixed = max(mixed, row["edges_with_mixed_observables"])
            for patch, h in row["shortest_logical_edges"].items():
                hops[f"{basis} d{patch}"].add(h)
            if "z_d5_path" in row:
                only_pieces.append(sum(e["p_as_a_whole_line"] == 0 for e in row["z_d5_path"]))
            if "z_d3_single_scan" in row:
                scan.update(row["z_d3_single_scan"])
            if "faults" not in row:
                continue
            faults += row["faults"]
            p_line += [f["p"] for f in row["faults"]]
            folds_ok &= all(row["graph_unchanged_folded"])
            refused.add(row["correlations_refuse_the_piece"])
            for shots, v in row["shots_decoded_differently"].items():
                differ[f"{shots} shots"] += v["shots"]
                for key in ("matching_folded_vs_paper", "correlated_folded_vs_dropped"):
                    differ[f"{shots} {key}"] += v[key]
            if "example_faults" in row:
                examples[j["name"]] = row["example_faults"]
            if setting == "Z8":
                job = names[j["name"]]
                mass = sum(f["p"] for f in row["faults"] if f["calibrated_wrong"])
                eps3 = combined[(job.day, job.rep)]["tests"]["Z eps3"]["predicted"]
                share.append(
                    {
                        "job": j["name"],
                        "miscorrected_p_per_round": mass / 8,
                        "all_p_per_round": sum(f["p"] for f in row["faults"]) / 8,
                        "share_of_predicted_Z_eps3": mass / 8 / eps3,
                    }
                )
    cal_wrong = [f for f in faults if f["calibrated_wrong"]]
    uni_wrong = [f for f in faults if f["uniform_wrong"]]
    pieces_out = {
        "count_by_basis_and_patch": dict(pieces),
        "per_circuit": "one per round, distance-3 Z memory only",
        "fault_probability_range": [min(p_line), max(p_line)],
        "faults_example_day1_pair1": examples.get("2026-10-04-r1"),
        "graph_unchanged_without_pieces_every_circuit": graph_ok,
        "edges_with_mixed_observables_max": mixed,
        "graph_unchanged_folded_into_any_piece_every_circuit": folds_ok,
        "correlated_matching_on_the_paper_model": sorted(map(str, refused)),
        "shots_decoded_differently": dict(differ),
        "faults_decoded_alone": {
            "faults": len(faults),
            "calibrated_wrong": len(cal_wrong),
            "calibrated_wrong_using_exactly_its_own_pieces": sum(
                f["calibrated_uses_its_own_pieces"] for f in cal_wrong
            ),
            "uniform_wrong": len(uni_wrong),
            "uniform_wrong_using_exactly_its_own_pieces": sum(
                f["uniform_uses_its_own_pieces"] for f in uni_wrong
            ),
        },
        "z_d3_single_scan (decoder_faults.py's, repeated)": dict(scan),
        "per_round_8_rounds": share,
        "share_of_predicted_Z_eps3_range": [
            min(s["share_of_predicted_Z_eps3"] for s in share),
            max(s["share_of_predicted_Z_eps3"] for s in share),
        ],
    }
    uni_out = {}
    for day, u in uniform.items():
        pre, per = u["pre-run simulator"], u["per-job model"]
        row = {}
        for basis in "XZ":
            eps = u["eps3_simulated"][basis]
            for src, rows in (("pre-run simulator f=1", pre["f=1"]), ("per-job model", per)):
                for d in (3, 5):
                    one, two, eight = (rows[f"{basis} d{d} n={n}"] for n in (1, 2, 8))
                    cell = {
                        "n=1": [one["miscorrected"], one["mechanisms"]],
                        "n=2": [two["miscorrected"], two["mechanisms"]],
                        "n=2 by symptom": [two["miscorrected_symptoms"], two["symptoms"]],
                        "n=8": [eight["miscorrected"], eight["mechanisms"]],
                        "boundary_p": eight["boundary_p"],
                    }
                    if d == 3:
                        cell["share_of_eps3"] = eight["miscorrected_p"] / 8 / eps
                    row[f"{basis} d{d} {src}"] = cell
            for variant in ("f=0", "f=1 no idle"):
                r = pre[variant][f"{basis} d3 n=8"]
                row[f"{basis} d3 pre-run simulator {variant}"] = {
                    "n=8": [r["miscorrected"], r["mechanisms"]],
                    "boundary_p": r["boundary_p"],
                }
            row[f"{basis} d3 n=1 examples (pre-run simulator f=1)"] = pre["f=1"][f"{basis} d3 n=1"][
                "examples"
            ]
            row[f"{basis} eps3_simulated"] = eps
        uni_out[day] = row
    return {
        "pieces": pieces_out,
        "uniform_chip": uni_out,
        "path": {
            "shortest_logical_edges_seen": {k: sorted(v) for k, v in sorted(hops.items())},
            "z_d5_path_edges_never_a_whole_line (n = 1, 2)": sorted(set(only_pieces)),
            "z_d5_example_day1_pair1_n1": next(
                j["circuits"]["Z1"]["z_d5_path"] for j in per_job if j["name"] == "2026-10-04-r1"
            ),
        },
    }


def pymatching_behaviour() -> dict:
    """Two toy models: a piece without detectors, and two parallel edges with different
    observables."""

    def labels(text: str) -> list:
        m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(text))
        return [(u, v, sorted(a["fault_ids"])) for u, v, a in m.edges()]

    return {
        "error(0.1) D0 ^ L0": labels("error(0.1) D0 ^ L0"),
        "error(0.01) D0, then error(0.1) D0 L0": labels("error(0.01) D0\nerror(0.1) D0 L0"),
        "error(0.1) D0 L0, then error(0.01) D0": labels("error(0.1) D0 L0\nerror(0.01) D0"),
        "pymatching": pymatching.__version__,
        "stim": stim.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(args.workers) as pool:
        per_job = pool.map(job_scan, range(len(jobs())))
    uniform = uniform_chip()
    out = {
        "method": __doc__,
        "pymatching_behaviour": pymatching_behaviour(),
        "summary": summary(per_job, uniform),
        "uniform_chip": uniform,
        "jobs": {j["name"]: j["circuits"] for j in per_job},
        "seconds": time.time() - t0,
    }
    write_json("decoder_graph.json", out)
    s = out["summary"]
    print(s["pieces"])
    print(s["path"]["shortest_logical_edges_seen"])
    for day, row in s["uniform_chip"].items():
        print(day, {k: v for k, v in row.items() if "examples" not in k})
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

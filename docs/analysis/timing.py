"""One round of X memory as it ran: what each qubit does, and when (Fig. timing).

The compiled circuits write every wait as a delay, so each qubit's timeline follows from the
circuit alone: an instruction starts when its qubits are free, a barrier waits for all of them,
rz takes no time, and x, sx, measure and reset take the calibration's median times. A CZ takes
its coupler's own time, which calibration.json does not keep; the delays fix it (cz_lengths:
68 ns, the median, on every coupler but two of the distance-5 patch, 88 ns). With these, every
circuit of the job comes out exactly as long as run.json recorded from the device's gate times.

The round shown is the middle one of the 3-round X-memory circuit of day 1's first job (pair 1),
on the distance-3 patch, between the barriers that close the rounds. Each qubit's time is split
into readout, reset, CZ, single-qubit gates (the sx of each H), decoupled idling (a wait of
500 ns or more, holding two X pulses, counted with them) and short waits (shorter, undecoupled
waits, such as a CX control's 24 ns beside its target's H). Four qubits are drawn, those of
the gadget figure where it has them: data Q5 (q24); q23, the X ancilla of X2X5 and a flag of
Z2Z3Z5Z6; that gauge's Z ancilla q16; and q21, a relay of an edge Z gauge. Every qubit of both
patches is summarized by role, with the X ancillas that no Z gauge borrows apart from the flags.

Each half of a round runs from a barrier to the next: a reset, gates, a readout. The gates are
what is left of the half after one reset and one readout; that some qubit's reset starts the
half and some qubit's readout ends it is checked. With every reset taking no time (no_reset.py)
the round is two resets shorter.

A few seconds.

    python docs/analysis/timing.py      # writes results/timing.json
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.experiment.circuits import H_FRAME
from heavyhex.patches.operators import build_operators
from heavyhex.simulation.noisy import DECOUPLE_NS

from common import RUNS, Job, jobs, memory_index, result, write_json

FOLDER = RUNS / "ibm_fez-2026-10-04-r1"
ROUNDS = 3  # the circuit; its middle round is drawn
SHOWN = 1  # 0-based: the second of three rounds
DRAWN = {"data": 24, "flag": 23, "z_ancilla": 16, "relay": 21}
ROLES = ("data", "flag", "x_ancilla", "z_ancilla", "relay")
KINDS = ("readout", "reset", "cz", "single_qubit", "idle_decoupled", "short_wait")
GATE = {"measure": "readout", "reset": "reset", "cz": "cz", "sx": "single_qubit"}


def lengths_ns(durations_ns: dict) -> dict:
    return {
        "x": durations_ns["x"],
        "sx": durations_ns["x"],
        "cz": durations_ns["cz"],
        "measure": durations_ns["measure"],
        "reset": durations_ns["reset"],
        "rz": 0,
    }


def cz_lengths(circuit, durations_ns: dict, dt_ns: float) -> dict:
    """Each coupler's CZ length in ns, read back from the circuit's delays.

    calibration.json keeps only the median CZ time, but the circuits were compiled with each
    coupler's own. Every CZ and barrier starts when all its qubits are free, so the qubits'
    clocks must agree there; with the median elsewhere, that fixes each coupler's extra time
    exactly (a linear system of full rank, solved by least squares and checked).
    """
    length = lengths_ns(durations_ns)
    const: dict[int, float] = defaultdict(float)
    count: dict[int, Counter] = defaultdict(Counter)
    rows, rhs, couplers = [], [], {}
    for item in circuit.data:
        name = item.operation.name
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        first, *rest = qubits
        for q in rest:
            diff = count[q].copy()
            diff.subtract(count[first])
            if any(diff.values()) or const[q] != const[first]:
                rows.append(diff)
                rhs.append(const[first] - const[q])
            const[q], count[q] = const[first], count[first].copy()
        if name == "barrier":
            continue
        if name == "cz":
            pair = tuple(sorted(qubits))
            couplers.setdefault(pair, len(couplers))
            for q in qubits:
                count[q][pair] += 1
        took = item.operation.duration * dt_ns if name == "delay" else length[name]
        for q in qubits:
            const[q] += took
    a = np.zeros((len(rows), len(couplers)))
    for i, row in enumerate(rows):
        for pair, n in row.items():
            a[i, couplers[pair]] = n
    extra, *_ = np.linalg.lstsq(a, np.array(rhs), rcond=None)
    assert np.linalg.matrix_rank(a) == len(couplers), "some CZ lengths are not fixed"
    assert np.allclose(a @ extra, rhs, atol=1e-6), "no CZ lengths fit the delays"
    assert np.allclose(extra, np.round(extra / dt_ns) * dt_ns, atol=1e-6)
    return {pair: length["cz"] + round(extra[i]) for pair, i in couplers.items()}


def timelines(circuit, durations_ns: dict, dt_ns: float, cz_ns: dict) -> tuple[dict, list, float]:
    """Each qubit's instructions as (name, start, length) in ns, the barriers as (qubits, time),
    and the circuit's length. Every wait is a delay in the circuit, which is checked."""
    length = lengths_ns(durations_ns)
    clock: dict[int, float] = defaultdict(float)
    line: dict[int, list] = defaultdict(list)
    barriers = []
    for item in circuit.data:
        name = item.operation.name
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        start = max(clock[q] for q in qubits)
        if name == "barrier":
            if item.operation.label != H_FRAME:
                barriers.append((frozenset(qubits), start))
            for q in qubits:
                clock[q] = start
            continue
        if name == "delay":
            assert item.operation.unit == "dt"
            took = item.operation.duration * dt_ns
        elif name == "cz":
            took = cz_ns[tuple(sorted(qubits))]
        else:
            took = length[name]
        for q in qubits:
            assert clock[q] == start or not line[q], f"an unwritten wait on q{q}"
            line[q].append((name, start, took))
            clock[q] = start + took
    return line, barriers, max(clock.values())


def segments(line: list, t0: float, t1: float) -> tuple[list, list]:
    """One qubit's time from t0 to t1 as (kind, start, length) from t0, and its X pulses.

    A delay next to an X pulse is a decoupled wait, as in the simulator (scheduled.py); the run
    delay-X-delay-X-delay is one decoupled idle period, its pulses included.
    """
    out, pulses = [], []
    for i, (name, start, took) in enumerate(line):
        a, b = max(start, t0) - t0, min(start + took, t1) - t0
        if name == "rz" or b <= a:
            continue
        if name == "x":
            pulses.append(a)
            kind = "idle_decoupled"
        elif name == "delay":
            near = {line[j][0] for j in (i - 1, i + 1) if 0 <= j < len(line)}
            kind = "idle_decoupled" if "x" in near else "short_wait"
        else:
            kind = GATE[name]
        last = out[-1] if out else None
        if last and last[0] == kind == "idle_decoupled" and abs(last[1] + last[2] - a) < 1e-6:
            out[-1] = (kind, last[1], b - last[1])
        else:
            out.append((kind, a, b - a))
    return out, pulses


def totals(segs: list) -> dict:
    t = dict.fromkeys(KINDS, 0.0)
    for kind, _, took in segs:
        t[kind] += took
    return t


def roles_on_chip(patch: str, chip: list[int]) -> dict[int, str]:
    roles = qubit_roles(build_operators(int(patch)))
    flags = {roles.x_ancillas[g] for arms in roles.z4_arms.values() for _, g in arms}
    role = {chip[i]: "data" for i in range(roles.num_data)}
    role |= {chip[i]: "flag" if i in flags else "x_ancilla" for i in roles.x_ancillas.values()}
    role |= {chip[i]: "z_ancilla" for i in roles.z_ancillas.values()}
    role |= {chip[r]: "relay" for arms in roles.z2_arms.values() for _, r in arms}
    assert len(role) == len(chip) == roles.num_qubits
    return role


def spread(values: list[float]) -> dict:
    return {"min": min(values), "median": float(np.median(values)), "max": max(values)}


def main() -> None:
    job = Job(FOLDER)
    durations = job.calibration.durations_ns
    reset, readout = durations["reset"], durations["measure"]
    dt_ns = job.info["dt"] * 1e9
    circuits = job.circuits

    idx = memory_index("X", ROUNDS)
    cz_ns = cz_lengths(circuits[idx], durations, dt_ns)
    # The reconstruction against the lengths run.json recorded with the device's gate times.
    lengths = [timelines(c, durations, dt_ns, cz_ns)[2] for c in circuits]
    recorded = [s * 1e9 for s in job.info["seconds"]]
    worst = max(abs(a - b) for a, b in zip(lengths, recorded))
    assert worst < 1e-3, f"the reconstruction misses run.json by {worst} ns"

    line, barriers, _ = timelines(circuits[idx], durations, dt_ns, cz_ns)
    patches = {}
    for patch, chip in job.info["chips"].items():
        role_of = roles_on_chip(patch, chip)
        # The patch's own barriers: after the preparation half, then after each half.
        times = [t for qs, t in barriers if qs == frozenset(chip)]
        assert len(times) == 1 + 2 * ROUNDS
        rounds = [times[2 * r + 2] - times[2 * r] for r in range(ROUNDS)]
        t0, mid, t1 = times[2 * SHOWN], times[2 * SHOWN + 1], times[2 * SHOWN + 2]
        per_qubit = {}
        for q in chip:
            segs, pulses = segments(line[q], t0, t1)
            assert abs(sum(d for _, _, d in segs) - (t1 - t0)) < 1e-6, f"q{q} has a gap"
            per_qubit[q] = (segs, pulses, totals(segs))
        by_role = {}
        for role in ROLES:
            qs = [q for q in chip if role_of[q] == role]
            share = {k: [per_qubit[q][2][k] / (t1 - t0) for q in qs] for k in KINDS}
            idle = [s + w for s, w in zip(share["idle_decoupled"], share["short_wait"])]
            busy = [r + s for r, s in zip(share["readout"], share["reset"])]
            by_role[role] = {
                "qubits": len(qs),
                "share_of_round": {k: spread(v) for k, v in share.items()},
                "share_idle_any": spread(idle),
                "share_readout_and_reset": spread(busy),
                "x_pulses": sorted({len(per_qubit[q][1]) for q in qs}),
            }
        halves = []
        for half, (a, b) in (("X", (t0, mid)), ("Z", (mid, t1))):
            starts = any(n == "reset" and abs(s - a) < 1e-6 for q in chip for n, s, _ in line[q])
            ends = any(
                n == "measure" and abs(s + d - b) < 1e-6 for q in chip for n, s, d in line[q]
            )
            assert starts and ends, f"the {half} half does not run from a reset to a readout"
            halves.append(
                {"half": half, "us": (b - a) / 1e3, "gates_us": (b - a - reset - readout) / 1e3}
            )
        patches[patch] = {
            "round_us_each": [r / 1e3 for r in rounds],
            "shown_window_ns": [t0, t1],
            "halves": halves,
            "by_role": by_role,
            "drawn": {
                role: {
                    "qubit": q,
                    "segments_ns": [[k, a, d] for k, a, d in per_qubit[q][0]],
                    "x_pulses_ns": per_qubit[q][1],
                    "total_us": {k: v / 1e3 for k, v in per_qubit[q][2].items()},
                }
                for role, q in DRAWN.items()
                if q in chip
            },
        }
        for role, q in DRAWN.items():
            assert q not in chip or role_of[q] == role, f"q{q} is not a {role}"

    d3 = patches["3"]
    round_ns = d3["shown_window_ns"][1] - d3["shown_window_ns"][0]
    path = [
        step
        for h in d3["halves"]
        for step in (("reset", reset / 1e3), ("gates", h["gates_us"]), ("readout", readout / 1e3))
    ]
    no_reset = result("no_reset.json")["placement"]["no_reset_time"]["round_us"]
    zero_reset = (round_ns - 2 * reset) / 1e3
    assert abs(zero_reset - no_reset) < 1e-6, (zero_reset, no_reset)
    every_job = [j.calibration.durations_ns for j in jobs()]

    out = {
        "what": "one round (the middle one) of the 3-round X-memory circuit of day 1's first "
        "job, from the compiled circuit as sent: what each qubit does, and when",
        "folder": FOLDER.name,
        "circuit_index": idx,
        "setting": job.info["settings"][idx],
        "dt_ns": dt_ns,
        "durations_ns": durations,
        "cz_ns_off_median": {f"{a}-{b}": t for (a, b), t in cz_ns.items() if t != durations["cz"]},
        "same_median_durations_in_every_job": all(d == durations for d in every_job),
        "decouple_threshold_ns": DECOUPLE_NS,
        "reconstruction_vs_run_json_max_abs_ns": worst,
        "round_us": round_ns / 1e3,
        "critical_path_us": path,
        "readout_and_reset_us": 2 * (reset + readout) / 1e3,
        "readout_and_reset_share_of_round": 2 * (reset + readout) / round_ns,
        "round_us_zero_duration_resets": zero_reset,
        "patches": patches,
    }
    write_json("timing.json", out)

    print(f"reconstruction matches run.json to {worst:.1g} ns over {len(circuits)} circuits")
    print(
        f"round {round_ns / 1e3:.3f} us (d=5: {patches['5']['round_us_each'][SHOWN]:.3f}); "
        f"readout and reset {out['readout_and_reset_us']:.3f} us "
        f"({100 * out['readout_and_reset_share_of_round']:.0f}%); without reset time "
        f"{zero_reset:.3f} us; path " + " ".join(f"{k} {v:.3f}" for k, v in path)
    )
    for patch, p in patches.items():
        print(f"d={patch}")
        for role, r in p["by_role"].items():
            s = r["share_of_round"]
            print(
                f"  {role:9s} {r['qubits']:2d}: idle any {r['share_idle_any']['min']:.3f}-"
                f"{r['share_idle_any']['max']:.3f}, decoupled "
                f"{s['idle_decoupled']['min']:.3f}-{s['idle_decoupled']['max']:.3f}, "
                f"readout+reset {r['share_readout_and_reset']['min']:.3f}-"
                f"{r['share_readout_and_reset']['max']:.3f}, cz {s['cz']['min']:.3f}-"
                f"{s['cz']['max']:.3f}, X pulses {r['x_pulses']}"
            )
    for role, r in d3["drawn"].items():
        shown = ", ".join(f"{k} {v:.3f}" for k, v in r["total_us"].items() if v)
        print(f"  q{r['qubit']} ({role}): {shown}")


if __name__ == "__main__":
    main()

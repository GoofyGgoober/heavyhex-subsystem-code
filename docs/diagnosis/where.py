"""Which qubits leak: the readout qubits themselves, or the data qubits next to them?

A flag keeps firing if it is leaked itself, or if a data qubit it reads is leaked: a CX
from a leaked control puts a random phase between the target's H gates, so the flag
flips at random. If the data qubits leak, two flags that read the same data qubit fire in
streaks together; if the flags leak, their streaks are independent. Relays read one data
qubit each, so the same test holds for flag-relay pairs.

For every pair of readout series in a job's 8-round Z memory, the co-streak excess is
P(both in a streak at round r) - P(A in a streak) P(B in a streak), where "in a streak" is
firing at r-1 and at r, averaged over rounds; set against the same from the model's
shots, for pairs that read a common data qubit, pairs on the same Z gauge (they share
only its ancilla) and unrelated pairs.

    python docs/diagnosis/where.py
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np

from common import REPO, jobs, memory_index, readout_series
from heavyhex.circuits.flagged import qubit_roles
from heavyhex.patches.operators import build_operators


def data_read(patch: str, name: str, k: int) -> set[int]:
    """Data qubit labels a readout qubit is coupled to by CXs."""
    kind, gauge = name.split()
    if kind == "z_flag":  # an X gauge's ancilla, flagging a Z4 gauge: its two data qubits
        return {int(x) for x in gauge.lstrip("X").split("X")}
    roles = qubit_roles(build_operators(int(patch)))
    return {roles.z2_arms[gauge][k][0] + 1}


def z4_of(patch: str, name: str) -> str | None:
    if not name.startswith("z_flag"):
        return None
    roles = qubit_roles(build_operators(int(patch)))
    gauge = name.split()[1]
    return next((z for z, arms in roles.z4_arms.items() if any(g == gauge for _, g in arms)), None)


def streaks(x: np.ndarray) -> np.ndarray:
    return x[:, 1:] & x[:, :-1]


def main() -> None:
    idx = memory_index("Z", 8)
    acc = defaultdict(lambda: defaultdict(list))
    for job in jobs():
        model, records, labels = job.model(idx)
        chip, _ = job.chip_detectors(idx, model, records)
        sim = model.compile_detector_sampler(seed=4).sample(20000)
        series = readout_series(labels)
        keys = list(series)
        st = {src: {k: streaks(x[:, series[k]]) for k in keys} for src, x in (("chip", chip), ("model", sim))}
        for a, b in itertools.combinations(keys, 2):
            if a[0] != b[0]:
                kind = "different patches"
            elif data_read(*a) & data_read(*b):
                kind = "share a data qubit"
            elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
                kind = "same Z gauge, no shared data"
            else:
                kind = "same patch, unrelated"
            for src in ("chip", "model"):
                sa, sb = st[src][a], st[src][b]
                acc[kind][src].append(float((sa & sb).mean() - sa.mean() * sb.mean()))
        print(job.name, "done", flush=True)
    out = {}
    print("\nco-streak excess, P(both) - P(A)P(B), mean over pairs:")
    for kind, d in acc.items():
        c, m = np.array(d["chip"]), np.array(d["model"])
        out[kind] = {"pairs": len(c), "chip": float(c.mean()), "model": float(m.mean()),
                     "chip_se": float(c.std(ddof=1) / np.sqrt(len(c)))}
        print(f"  {kind:30s} pairs {len(c):5d}  chip {1e4*c.mean():+7.2f}e-4 (±{1e4*c.std(ddof=1)/np.sqrt(len(c)):.2f})  model {1e4*m.mean():+7.2f}e-4")
    (REPO / "docs/diagnosis/results/where.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

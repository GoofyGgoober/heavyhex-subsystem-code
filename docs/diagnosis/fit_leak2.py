"""Fit a leakage model in which every qubit can leak and a leaked qubit disturbs its CZ partners.

where.py shows that flags and relays fire in streaks together when they read a common
data qubit, or sit on the same Z gauge: leakage is not confined to the readout qubits.
Here every qubit leaks with the same probability per CZ (leak), a leaked qubit's CZ gives
its partner a random Pauli with probability strength and spreads the leakage to it with
probability spread; the lifetime and the leaked readout are the hidden Markov fit's.
The three are fitted together to six statistics of the 8-round runs on four jobs:
the mean detection rates of X-memory stabilizers, Z-memory stabilizers and Z-memory
flags and relays; the flags' and relays' repeat excess (P(fire at r+1 | fired at r) -
P(fire at r+1 | quiet at r)); and the co-streak excess of pairs that read a common data
qubit and of pairs on the same Z gauge. Logical errors and Λ are left out of the fit.

    python docs/diagnosis/fit_leak2.py [depolarize | bitflip | phase]
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from common import REPO, ROUND_US, jobs, memory_index, readout_series
import leaky
import rerun
from where import data_read, z4_of

FIT_JOBS = ("2026-10-08-r1", "2026-10-08-r2-d5", "2026-10-08-r2-d3", "2026-10-06-r1")
KIND = sys.argv[1] if len(sys.argv) > 1 else "depolarize"
LEAKS = (0.001, 0.002, 0.003, 0.0045) if KIND == "depolarize" else (0.001, 0.002, 0.003)
STRENGTHS = (0.25, 0.5, 1.0)
SPREADS = (0.0, 0.05, 0.15) if KIND == "depolarize" else (0.0, 0.05)
SHOTS = 8000


def stats(det: np.ndarray, labels) -> dict:
    out = {}
    stab = [c for c, (p, r, n) in enumerate(labels) if " " not in n and 1 <= r <= 7]
    out["stab"] = float(det[:, stab].mean())
    series = readout_series(labels)
    if series:
        flags = [c for cols in series.values() for c in cols]
        out["flag"] = float(det[:, flags].mean())
        rep = []
        st = {}
        for key, cols in series.items():
            x = det[:, cols]
            now, nxt = x[:, :-1], x[:, 1:]
            rep.append((now & nxt).sum() / max(now.sum(), 1) - (~now & nxt).sum() / (~now).sum())
            st[key] = now & nxt
        out["repeat"] = float(np.mean(rep))
        share, gauge = [], []
        for a, b in itertools.combinations(series, 2):
            if a[0] != b[0]:
                continue
            v = float((st[a] & st[b]).mean() - st[a].mean() * st[b].mean())
            if data_read(*a) & data_read(*b):
                share.append(v)
            elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
                gauge.append(v)
        out["co_data"] = float(np.mean(share)) if share else np.nan
        out["co_gauge"] = float(np.mean(gauge)) if gauge else np.nan
    return out


def main() -> None:
    hmm = json.loads((REPO / "docs/diagnosis/results/leakage_hmm.json").read_text())
    life = float(np.median([1 / max(r["chip"]["b"], 1e-6) for r in hmm])) * ROUND_US
    read = float(np.median([r["chip"]["e_L"] for r in hmm]))
    js = [j for j in jobs() if j.name in FIT_JOBS]
    cases = []
    for j in js:
        for basis in "XZ":
            idx = memory_index(basis, 8)
            model, records, labels = j.model(idx)
            chip, _ = j.chip_detectors(idx, model, records)
            cases.append((j, basis, model, labels, {f"{basis} {k}": v for k, v in stats(chip, labels).items()}))
    chip_tot = {}
    for *_, s in cases:
        for k, v in s.items():
            chip_tot.setdefault(k, []).append(v)
    chip_mean = {k: float(np.nanmean(v)) for k, v in chip_tot.items()}
    print("chip:", "  ".join(f"{k} {v:.4g}" for k, v in sorted(chip_mean.items())), flush=True)
    results = []
    for leak, strength, spread in itertools.product(LEAKS, STRENGTHS, SPREADS):
        tot = {}
        for j, basis, model, labels, _ in cases:
            ro = rerun.variant(j, "leak+ro", *rerun.leakage_params(), 1.0)
            lk = leaky.Leakage(leak=rerun._AllQubits(__import__("collections").defaultdict(lambda: leak)),
                               lifetime_us=life, half_round_us=ROUND_US / 2, read_leaked=read,
                               partner_phase=strength, partner_kind=KIND, transfer=spread,
                               reset_error=ro.reset_error, readout=ro.readout)
            det, _ = leaky.simulate(model, SHOTS, lk, seed=21)
            for k, v in stats(det, labels).items():
                tot.setdefault(f"{basis} {k}", []).append(v)
        sim = {k: float(np.nanmean(v)) for k, v in tot.items()}
        rel = {k: (sim[k] - chip_mean[k]) / abs(chip_mean[k]) for k in chip_mean if np.isfinite(chip_mean[k])}
        loss = float(np.sqrt(np.mean([v ** 2 for v in rel.values()])))
        results.append({"leak": leak, "strength": strength, "spread": spread, "loss": loss, "sim": sim})
        print(f"leak {leak:<6g} strength {strength:<4g} spread {spread:<4g} loss {loss:.3f}  " +
              "  ".join(f"{k} {100*v:+.0f}%" for k, v in sorted(rel.items())), flush=True)
    best = min(results, key=lambda r: r["loss"])
    print(f"\nbest: leak {best['leak']} per CZ, strength {best['strength']}, spread {best['spread']} (loss {best['loss']:.3f})")
    (REPO / f"docs/diagnosis/results/fit_leak2_{KIND}.json").write_text(
        json.dumps({"chip": chip_mean, "lifetime_us": life, "read_leaked": read, "grid": results, "best": best}, indent=1) + "\n")


if __name__ == "__main__":
    main()

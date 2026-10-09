"""Summarize rerun.py: how much of the chip's excess over the calibration model a variant closes.

    python docs/diagnosis/rerun_summary.py docs/diagnosis/results/rerun_model-fit_cz1.5_d1_p0.5.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def main(path: str) -> None:
    res = json.loads(open(path).read())
    jobs = res["jobs"]
    variants = [v for v in next(iter(jobs.values())) if v != "chip"]
    eps = defaultdict(lambda: defaultdict(list))
    reps = defaultdict(lambda: defaultdict(dict))  # (day, rep) -> src -> {(basis, d): eps}
    for name, r in jobs.items():
        day, rep = name[:10], name[11:13]
        for src in ["chip", *variants]:
            for b in "XZ":
                for p, v in r[src][b].items():
                    eps[(b, p)][src].append(v[0])
                    reps[(day, rep)][src][(b, p)] = v[0]
    print("mean error per round over the jobs, and the share of the chip's excess over the model a variant closes:")
    out = {"eps": {}, "lambda": {}}
    for (b, p), d in sorted(eps.items()):
        chip, model = np.mean(d["chip"]), np.mean(d["model"])
        line = f"  {b} memory d={p}: chip {100*chip:.1f}%  model {100*model:.1f}%"
        out["eps"][f"{b} d{p}"] = {"chip": chip, "model": model}
        for v in variants[1:]:
            m = np.mean(d[v]); closed = (m - model) / (chip - model)
            line += f"  {v} {100*m:.1f}% (closes {100*closed:.0f}% of the gap)"
            out["eps"][f"{b} d{p}"][v] = m
        print(line)
    print("Λ per rep, chip | " + " | ".join(variants))
    lam = defaultdict(list)
    for key in sorted(reps):
        row = []
        for b in "XZ":
            vals = []
            for src in ["chip", *variants]:
                e = reps[key][src]
                if (b, "3") in e and (b, "5") in e:
                    vals.append(e[(b, "3")] / e[(b, "5")]); lam[(b, src)].append(vals[-1])
            row.append(f"{b}: " + " ".join(f"{v:.2f}" for v in vals))
        print(f"  {key[0]} {key[1]}  " + "   ".join(row))
    for b in "XZ":
        print(f"  {b} Λ mean over reps: " + ", ".join(f"{src} {np.mean(lam[(b, src)]):.2f}" for src in ["chip", *variants]))
        out["lambda"][b] = {src: float(np.mean(lam[(b, src)])) for src in ["chip", *variants]}
    open(path.replace(".json", "_summary.json"), "w").write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])

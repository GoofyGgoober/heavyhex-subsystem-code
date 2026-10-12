"""Print the summary of refits.json (written by refit.py).

    python docs/diagnosis/revision/refits/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOCKS = ("X", "Z")


def main(path: Path = HERE / "refits.json") -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    per_rep, summary, gof = result["per_rep"], result["summary"], result["goodness_of_fit"]
    check = result["replication_check"]
    agreement_out = {est: {"model": m} for est, m in result["agreement"].items()}
    keys = sorted(per_rep)
    print("\nreplication check:", json.dumps(check))
    print("\nper rep (WLS ± se | binomial [profile 68%] {quasi 68%})")
    for k in keys:
        r = per_rep[k]
        line = f"{k}:"
        for b in STOCKS:
            for q in ("eps3", "eps5"):
                x = r[b][q]
                line += f" {b}{q[-1]} {100*x['wls'][0]:.2f}±{100*x['wls'][1]:.2f}|{100*x['binomial']:.2f}"
            x = r[b]["lambda"]
            line += (f" Λ{b} {x['wls'][0]:.3f}±{x['wls'][1]:.3f}|{x['binomial']:.3f}"
                     f"[{x['binomial_lo68']:.3f},{x['binomial_hi68']:.3f}]{{{x['binomial_lo68_quasi']:.3f},"
                     f"{x['binomial_hi68_quasi']:.3f}}}")
        print(line)
    print("\nunweighted means over the ten reps (± sd/sqrt(10)):")
    for k, v in summary["unweighted_means"].items():
        s = 100 if "eps" in k else 1
        print(f"  {k}: {s*v['mean']:.3f} ± {s*v['se']:.3f}")
    print("\nplacement pooled (widened):")
    for k, v in summary["placement_pooled"].items():
        s = 100 if "eps" in k else 1
        print(f"  {k}: {s*v['pooled']:.3f} ± {s*v['se_widened']:.3f} (chi2/dof {v['chi2_per_dof']:.2f})")
    print("\npooled Λ:")
    for k, v in summary["pooled_lambda"].items():
        print(f"  {k}: {json.dumps(v)}")
    print("\npredicted means:")
    for k, v in summary["predicted_means"].items():
        s = 100 if "eps" in k else 1
        print(f"  {k}: {s*v['mean']:.3f}  pl1 {s*v['placement 1']:.3f}  pl2 {s*v['placement 2']:.3f}")
    print("\nagreement chi2 (bar 18.31):")
    for est, v in agreement_out.items():
        print(f"  {est}: " + ", ".join(f"{q} {m['chi2']:.1f} ({m['reps_agreeing']}/10)" for q, m in v["model"].items()))
    print("\ngoodness of fit:", json.dumps(gof))


if __name__ == "__main__":
    main()

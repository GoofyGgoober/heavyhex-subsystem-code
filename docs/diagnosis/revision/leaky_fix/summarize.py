"""Table IV and gap closures, before and after the skipped-CZ fix.

    python summarize.py rerun_A.json [rerun_B.json ...]

Uses heldout.py's grouping (means over repetitions; closure = (leak - model)/(device - model)).
The "before" column is the published rerun (docs/diagnosis/results/rerun_model-fit-bitflip...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "heldout"))
import heldout as H  # noqa: E402

CELLS = [("X", "3"), ("X", "5"), ("Z", "3"), ("Z", "5")]


def table(data, reps):
    g = H.group(data, reps)
    return g


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fit = sorted({H.rep_of(j) for j in H.fit_jobs()})
    runs = {"published (original sim)": H.load_wls(H.RESULTS / "rerun_model-fit-bitflip_cz1.5_d1_p0.5.json", "fit-bitflip")}
    for p in sys.argv[1:]:
        runs[Path(p).stem] = H.load_wls(Path(p), "leak")
    out = {}
    for name, data in runs.items():
        reps = sorted(data)
        groups = {"all": reps, "fit": [r for r in reps if r in fit], "heldout": [r for r in reps if r not in fit]}
        res = {gname: H.group(data, rs) for gname, rs in groups.items() if rs}
        out[name] = res
        print(f"\n== {name} ({len(reps)} reps) ==")
        for b, p in CELLS:
            line = f"  {b} d{p}:"
            for gname in res:
                x = res[gname][f"{b} d{p}"]
                line += (f"  [{gname}] dev {100*x['device']:.1f} model {100*x['model']:.1f} leak {100*x['leak']:.1f}"
                         f" closure {100*x['closure']:.0f}% (leak-dev {100*x['leak_minus_device']:+.1f}pp)")
            print(line)
        for b in "XZ":
            lam = res["all"][f"{b} lambda"]
            print(f"  Lambda {b} (all): device {lam['device']:.2f} model {lam['model']:.2f} leak {lam['leak']:.2f}")
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()

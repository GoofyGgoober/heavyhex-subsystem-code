"""Arithmetic re-check of the 'mc' and 'redecoded' chi^2 using the perqubit_f analysis's raw
mc_from_f.json SDs and redecode.json fits, with this folder's own test assembly (verify.py logic).
Also: how far redecoding moves each observed X quantity, in units of sigma_O.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify as V  # reuses verify.py's loaders; runs its prints once

PQ = str(HERE.parent / "perqubit_f") + "/"
mc = V.load(PQ + "mc_from_f.json")["jobs"]
rd = V.load(PQ + "redecode.json")


def build(mode):
    R = {}
    for name, J in V.jobs.items():
        info = J["run"]["rep"]
        t = V.job_tests(name, "pq")
        if mode == "mc":
            for k, row in t.items():
                b, q = k.split()
                if b == "X":
                    row["from_f"] = mc[name]["sd"][q]
        if mode == "rd":
            fits = rd[name]["per_qubit_f"]
            for b in "XZ":
                for q, (v, s) in V.quantities(fits, b).items():
                    t[f"{b} {q}"]["O"], t[f"{b} {q}"]["sO"] = v, s
        R.setdefault((info["day"], info["rep"]), {}).update(t)
    for tests in R.values():
        for b in "XZ":
            if f"{b} lambda" not in tests:
                tests[f"{b} lambda"] = V.ratio(tests[f"{b} eps3"], tests[f"{b} eps5"])
        for t in tests.values():
            V.finish(t)
    return dict(sorted(R.items()))


for mode in ("mc", "rd"):
    R = build(mode)
    c = V.chi2_table(R)
    print(mode, "; ".join(f"{t}: {c[t]['chi2']:.1f} [{c[t]['n_agree']}]" for t in V.TESTS))
    if mode == "mc":
        for k in R:
            print("  ", k, {t: round(R[k][t]["from_f"] * (100 if "eps" in t else 1), 4) for t in ("X eps3", "X eps5", "X lambda")})

# redecode shifts in sigma_O
print("redecode shift (pq-decode minus median-decode)/sigma_O, X memory:")
med = V.reps("median")
Rr = build("rd")
for k in med:
    print("  ", k, {t: round((Rr[k][t]["O"] - med[k][t]["O"]) / med[k][t]["sO"], 2) for t in ("X eps3", "X eps5", "X lambda")})

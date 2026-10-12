"""Check the skipped-CZ fix of leaky.py against an exact stabilizer simulation.

1. The reviewer's three-qubit counterexample (q0 forced to leak in |1>, partner in |+>).
2. Random small Clifford circuits with leakage (forced and stochastic, every partner
   kind, transfer, lifetime): the joint distribution of all measurement flips from
   (a) original leaky.py, (b) fixed leaky.py and the exact tableau simulation with
   (c) the CZ left out ("identity") and (d) the CZ replaced by a random phase ("dephase").
   Expect (b) == (d) within shot noise, and (a) != (c), (a) != (d) in general.

    python docs/diagnosis/revision/leaky_fix/check_fix.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import leaky as fixed  # noqa: E402
import tableau_ref  # noqa: E402

spec = importlib.util.spec_from_file_location("leaky_orig", REPO / "docs/diagnosis/leaky.py")
orig = importlib.util.module_from_spec(spec)
sys.modules["leaky_orig"] = orig
spec.loader.exec_module(orig)

CE = """
R 0 1 2
X 0
CZ 0 2
S 1
SQRT_X 1
S 1
CZ 0 1
S 1
SQRT_X 1
S 1
M 1
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def leak(mod, **kw):
    return mod.Leakage(**kw)


def counterexample(shots=4000):
    c = stim.Circuit(CE)
    kw = dict(leak={0: 1.0}, partner_phase=0.0, transfer=0.0)
    out = {
        "original leaky.py": orig.simulate(c, shots, leak(orig, **kw), seed=1)[1].mean(),
        "fixed leaky.py (dephase)": fixed.simulate(c, shots, leak(fixed, **kw), seed=1)[1].mean(),
        "exact, CZ left out": tableau_ref.simulate(c, shots, leak(fixed, **kw), "identity", seed=1)[1].mean(),
        "exact, CZ -> random phase": tableau_ref.simulate(c, shots, leak(fixed, **kw), "dephase", seed=1)[1].mean(),
    }
    print("counterexample: observable flip rate (1.0 = target 0, the omitted-CZ answer)")
    for k, v in out.items():
        print(f"  {k:28s} {v:.3f}")
    return {k: float(v) for k, v in out.items()}


def random_circuit(rng, n=4, layers=8):
    c = stim.Circuit()
    c.append("R", range(n))
    for q in range(n):
        if rng.random() < 0.5:
            c.append("S", [q]); c.append("SQRT_X", [q]); c.append("S", [q])  # H
    nm = 0
    for _ in range(layers):
        for q in range(n):
            g = rng.choice(["I", "S", "SQRT_X", "X", "H"])
            if g == "H":
                c.append("S", [q]); c.append("SQRT_X", [q]); c.append("S", [q])
            elif g != "I":
                c.append(g, [q])
        perm = rng.permutation(n)
        pairs = [int(x) for x in perm[: 2 * (n // 2)]]
        c.append("CZ", pairs)
        c.append("DEPOLARIZE2", pairs, [0.01])
        if rng.random() < 0.3:
            q = int(rng.integers(n))
            c.append("M", [q], [0.01]); c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm]); nm += 1
            c.append("R", [q]); c.append("X_ERROR", [q], [0.01])
        c.append("TICK")
    for q in range(n):
        if rng.random() < 0.5:
            c.append("S", [q]); c.append("SQRT_X", [q]); c.append("S", [q])
        c.append("M", [q]); c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm]); nm += 1
    return c


def joint(obs):
    k = obs.shape[1]
    idx = obs.astype(int) @ (1 << np.arange(k))
    return np.bincount(idx, minlength=1 << k) / len(obs)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def random_checks(ncirc=24, shots=6000):
    rng = np.random.default_rng(7)
    rows = []
    kinds = ["bitflip", "depolarize", "phase"]
    for i in range(ncirc):
        c = random_circuit(rng)
        kind = kinds[i % 3]
        strength = [0.0, 0.25, 0.5, 1.0][i % 4]
        kw = dict(leak={q: (0.3 if q == 0 else 0.02) for q in range(4)}, partner_phase=strength,
                  partner_kind=kind, transfer=0.1 if i % 2 else 0.0, lifetime_us=8.0, half_round_us=1.0,
                  read_leaked=0.5)
        a = joint(orig.simulate(c, shots, leak(orig, **kw), seed=i)[1])
        b = joint(fixed.simulate(c, shots, leak(fixed, **kw), seed=i)[1])
        b2 = joint(fixed.simulate(c, shots, leak(fixed, **kw), seed=1000 + i)[1])
        ident = joint(tableau_ref.simulate(c, shots, leak(fixed, **kw), "identity", seed=i)[1])
        deph = joint(tableau_ref.simulate(c, shots, leak(fixed, **kw), "dephase", seed=i)[1])
        r = {"circuit": i, "kind": kind, "strength": strength, "transfer": kw["transfer"],
             "TV fixed vs exact-dephase": tv(b, deph), "TV fixed vs fixed (other seed)": tv(b, b2),
             "TV original vs exact-dephase": tv(a, deph), "TV original vs exact-identity": tv(a, ident),
             "TV exact-identity vs exact-dephase": tv(ident, deph)}
        rows.append(r)
        print(f"circ {i:2d} {kind:10s} s={strength:<4} t={kw['transfer']:<4} TV: fixed~deph {r['TV fixed vs exact-dephase']:.3f} "
              f"(noise {r['TV fixed vs fixed (other seed)']:.3f})  orig~deph {r['TV original vs exact-dephase']:.3f}  "
              f"orig~ident {r['TV original vs exact-identity']:.3f}  ident~deph {r['TV exact-identity vs exact-dephase']:.3f}",
              flush=True)
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"counterexample": counterexample()}
    # no-leakage branch unchanged: fixed == original bit for bit
    c = random_circuit(np.random.default_rng(3))
    d1 = orig.simulate(c, 2000, seed=5); d2 = fixed.simulate(c, 2000, seed=5)
    out["no_leak_identical"] = bool((d1[1] == d2[1]).all())
    print("no-leakage output identical to original:", out["no_leak_identical"])
    out["random"] = random_checks()
    keys = [k for k in out["random"][0] if k.startswith("TV")]
    out["random_mean"] = {k: float(np.mean([r[k] for r in out["random"]])) for k in keys}
    out["random_max"] = {k: float(np.max([r[k] for r in out["random"]])) for k in keys}
    print("mean TV:", json.dumps(out["random_mean"], indent=1))
    print("max TV:", json.dumps(out["random_max"], indent=1))
    (HERE / "check_fix.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

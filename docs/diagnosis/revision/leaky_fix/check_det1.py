"""Does the original simulator differ from 'CZ left out' only when the leaked operand is a
deterministic |1> in the noiseless reference? Random 4-qubit circuits, split by whether any CZ
operand is deterministic-1 (peek_z == -1); total-variation distance of the joint outcome
distribution, original leaky.py vs the exact tableau simulation of 'CZ left out'.
"""
import sys
from pathlib import Path
import numpy as np, stim
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leaky as fixed, tableau_ref
from check_fix import orig, joint, tv

def circ(rng, allow_x):
    c = stim.Circuit(); n = 4
    c.append("R", range(n))
    nm = 0
    for _ in range(8):
        for q in range(n):
            g = rng.choice(["I", "S", "SQRT_X", "H"] + (["X"] if allow_x else []))
            if g == "H":
                c.append("S", [q]); c.append("SQRT_X", [q]); c.append("S", [q])
            elif g != "I":
                c.append(g, [q])
        pairs = [int(x) for x in rng.permutation(n)]
        c.append("CZ", pairs); c.append("DEPOLARIZE2", pairs, [0.01])
        if rng.random() < 0.3:
            q = int(rng.integers(n)); c.append("M", [q], [0.01]); c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm]); nm += 1
            c.append("R", [q])
        c.append("TICK")
    for q in range(n):
        c.append("M", [q]); c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [nm]); nm += 1
    return c

def det1(c):
    sim = stim.TableauSimulator(); k = 0
    for ins in c.without_noise().flattened():
        if ins.name == "CZ":
            k += sum(sim.peek_z(t.value) == -1 for t in ins.targets_copy())
        if ins.name not in ("OBSERVABLE_INCLUDE", "TICK"):
            sim.do(ins)
    return k

rng = np.random.default_rng(11)
rows = {True: [], False: []}
while min(len(v) for v in rows.values()) < 10:
    c = circ(rng, allow_x=True)
    has = det1(c) > 0
    if len(rows[has]) >= 10:
        continue
    kw = dict(leak={q: (0.3 if q == 0 else 0.02) for q in range(4)}, partner_phase=0.0, transfer=0.0,
              lifetime_us=8.0, half_round_us=1.0, read_leaked=0.5)
    s = 6000
    a = joint(orig.simulate(c, s, orig.Leakage(**kw), seed=1)[1])
    a2 = joint(orig.simulate(c, s, orig.Leakage(**kw), seed=2)[1])
    e = joint(tableau_ref.simulate(c, s, fixed.Leakage(**kw), "identity", seed=3)[1])
    rows[has].append((tv(a, e), tv(a, a2)))
    print(f"det1 operands present={has}: TV orig~exact-identity {tv(a, e):.3f}  (seed noise {tv(a, a2):.3f})", flush=True)
for k, v in rows.items():
    v = np.array(v)
    print(f"det1 present={k}: mean TV {v[:,0].mean():.3f}, mean noise {v[:,1].mean():.3f}, max excess {np.max(v[:,0]-v[:,1]):+.3f}")

"""The circuit generator of leaky_fix/check_det1.py (which runs its check on import), for
validate_exact.py: random 4-qubit circuits, and the number of CZ operands that are a
deterministic |1> in the noiseless circuit (where the original simulator is wrong)."""

import stim


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

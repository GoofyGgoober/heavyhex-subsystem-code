"""Where can the original simulator differ from 'the CZ does not act'? Only through the leaked
qubit's Z value in stim's reference trajectory. Count, over the noiseless memory circuits of the
four fit jobs, the CZ operands whose Z value is deterministically 1 (as in the reviewer's
counterexample), deterministically 0, or random, using stim.TableauSimulator.peek_z.

    python peek_reference.py
"""
import sys
from collections import Counter
from pathlib import Path

import stim

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from common import jobs, memory_index  # noqa: E402
from fit_leak2 import FIT_JOBS  # noqa: E402
from heavyhex.experiment.circuits import ROUNDS  # noqa: E402

total = Counter()
for j in [j for j in jobs() if j.name in FIT_JOBS]:
    for basis in "XZ":
        for rounds in (ROUNDS[0], ROUNDS[-1]):
            model, _, _ = j.model(memory_index(basis, rounds))
            clean = model.without_noise()
            sim = stim.TableauSimulator()
            c = Counter()
            for ins in clean.flattened():
                if ins.name == "CZ":
                    t = [x.value for x in ins.targets_copy()]
                    for q in t:
                        c[{1: "det0", -1: "det1", 0: "random"}[sim.peek_z(q)]] += 1
                if ins.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
                    continue
                sim.do(ins)
            total.update(c)
            print(j.name, basis, rounds, dict(c), flush=True)
print("total", dict(total))

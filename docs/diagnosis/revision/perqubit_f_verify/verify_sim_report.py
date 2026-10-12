"""Summarize verify_sim.json against analysis.json and the perqubit_f analysis's outputs (../perqubit_f/)."""
import json, math
from pathlib import Path
import numpy as np

OUT = str(Path(__file__).resolve().parent) + "/"
REPO = str(Path(OUT).parents[3])
S = json.load(open(OUT + "verify_sim.json"))
theirs_mc = json.load(open(REPO + "/docs/diagnosis/revision/perqubit_f/mc_from_f.json"))["jobs"]
theirs_rd = json.load(open(REPO + "/docs/diagnosis/revision/perqubit_f/redecode.json"))
lines = []


def A(name):
    return json.load(open(f"{REPO}/runs/{name}/analysis.json"))


lines.append("A. Redecode of chip shots")
for name, res in S["redecode"].items():
    a = A(name)
    obs = a["observed"]["fits"]
    for b in "XZ":
        for p, (v, s) in res["median"][b].items():
            reg = obs[b]["per_round"][p]
            lines.append(f"  {name} {b} eps{p}: median-decode {v:.6f} (prespecified {reg[0]:.6f}, diff {v-reg[0]:.1e}) | "
                         f"pq-decode {res['pq'][b][p][0]:.6f} (perqubit_f {theirs_rd[name]['per_qubit_f'][b]['per_round'][p][0]:.6f}) "
                         f"shift/sigma_O {(res['pq'][b][p][0]-reg[0])/reg[1]:+.2f}")
    if "3" in res["pq"]["X"] and "5" in res["pq"]["X"]:
        lam = res["pq"]["X"]["3"][0] / res["pq"]["X"]["5"][0]
        lines.append(f"  {name} Lambda_X pq-decode {lam:.4f} (prespecified {obs['X']['lambda']:.4f})")

lines.append("B. Rerun of stored X-memory predictions")
for key, r in S["repro"].items():
    name, which = key.split("|")
    a = A(name)
    stored = a["predicted_at_measured_dephasing" if which == "median" else "predicted_with_each_qubits_dephasing"]
    for p, (v, s) in r["fits"].items():
        st = stored["fits"]["X"]["per_round"][p]
        lines.append(f"  {name} {which} eps{p}: rerun {v:.6f} stored {st[0]:.6f} +- {st[1]:.6f} (diff {(v-st[0])/st[1]:+.2f} SE)")
    for i, raw in r["raw"].items():
        st = stored["settings"][int(i)]["logical_error"]
        lines.append(f"     setting {i}: rerun {raw} stored {st} identical={raw == st}")

lines.append("C. Mini Monte Carlo spread (these draws) vs perqubit_f's")
for name, draws in S["mc"].items():
    keys = sorted(draws[0])
    sd = {f"eps{p}": float(np.std([d[p][0] for d in draws], ddof=1)) for p in keys}
    if len(keys) == 2:
        sd["lambda"] = float(np.std([d["3"][0] / d["5"][0] for d in draws], ddof=1))
    n = len(draws)
    rel = 1 / math.sqrt(2 * (n - 1))
    lines.append(f"  {name} n={n}: mine " + ", ".join(f"{k} {v:.5f}" for k, v in sd.items())
                 + f" (rel. SE ~{rel:.0%}) | theirs " + ", ".join(f"{k} {v:.5f}" for k, v in theirs_mc[name]["sd"].items()))
txt = "\n".join(lines)
print(txt)
open(OUT + "verify_sim_report.txt", "w").write(txt)

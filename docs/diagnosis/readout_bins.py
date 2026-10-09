"""Flags and relays against IBM's quoted readout error: a floor, or regression to the mean?

Each flag's and relay's firing rate, averaged over the Z-memory runs, set against the
model's prediction and binned by the qubit's quoted readout error, first the day's own and
then the other days' (which the day's choice of qubits cannot have selected on). Also how
steady the quotes are from day to day, and which quote predicts the flags better.

    python docs/diagnosis/readout_bins.py
"""

import json, glob, collections, statistics as st
import numpy as np
from heavyhex.circuits.flagged import qubit_roles
from heavyhex.patches.operators import build_operators

folders = [f for f in sorted(glob.glob("runs/ibm_fez-2026-10-0*-r*")) if not f.endswith("offline")]
cal_by_day = {}
for f in folders:
    day = f.split("fez-")[1][:10]
    cal_by_day.setdefault(day, json.load(open(f"{f}/calibration.json")))
days = sorted(cal_by_day)
rows = []  # (day, kind, qubit, obs, pred, ro_same, ro_other)
for f in folders:
    day = f.split("fez-")[1][:10]
    run = json.load(open(f"{f}/run.json")); a = json.load(open(f"{f}/analysis.json"))
    o = a["observed"]["settings"]; p = a["predicted_at_measured_dephasing"]["settings"]
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for i, s in enumerate(run["settings"]):
        if s["kind"] != "memory" or s["basis"] != "Z": continue
        seen = collections.Counter()
        for (patch, rnd, name), ro, rp in zip(o[i]["detectors"], o[i]["detector_rates"], p[i]["detector_rates"]):
            if not (name.startswith("z_flag") or name.startswith("relay")): continue
            roles = qubit_roles(build_operators(int(patch)))
            if name.startswith("z_flag"):
                ci = roles.x_ancillas[name.split()[1]]
            else:
                k = seen[(patch, rnd, name)]; seen[(patch, rnd, name)] += 1
                ci = roles.z2_arms[name.split()[1]][k][1]
            q = run["chips"][patch][ci]
            x = acc[(name.split()[0], q)]; x[0] += ro; x[1] += rp; x[2] += 1
    for (kind, q), (ob, pr, n) in acc.items():
        same = cal_by_day[day]["readout"][str(q)]
        other = st.mean(cal_by_day[d]["readout"][str(q)] for d in days if d != day)
        rows.append((day, kind, q, ob / n, pr / n, same, other))

print(f"{len(rows)} (day, qubit) readouts: {sum(r[1]=='z_flag' for r in rows)} flags, {sum(r[1]=='relay' for r in rows)} relays, days {days}")
# 1. how stable are IBM's quoted readout errors for these qubits, day to day?
qs = sorted({r[2] for r in rows})
M = np.array([[cal_by_day[d]["readout"][str(q)] for d in days] for q in qs])
L = np.log(M)
cors = [np.corrcoef(L[:, i], L[:, j])[0, 1] for i in range(len(days)) for j in range(i + 1, len(days))]
print(f"quoted readout, same qubits across days: day-to-day correlation of log values {min(cors):.2f}-{max(cors):.2f}; median relative day-to-day change {np.median(np.abs(np.diff(M, axis=1)) / M[:, :-1]):.0%}")
# 2. excess binned by the same day's quote and by the other days' quotes
def bins(idx, label):
    print(f"binned by {label}:")
    for lo, hi in ((0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1)):
        sel = [r for r in rows if lo <= r[idx] < hi]
        if sel:
            print(f"   {lo:.0%}-{hi:.0%}: n={len(sel):3d}  observed {st.mean(r[3] for r in sel):.3f}  predicted {st.mean(r[4] for r in sel):.3f}  excess {st.mean(r[3]-r[4] for r in sel):+.3f}")
bins(5, "the same day's quoted readout")
bins(6, "the OTHER days' mean quoted readout (independent of selection)")
# 3. consistently good qubits: quoted below 1.5% on every day
good = [r for r in rows if max(cal_by_day[d]["readout"][str(r[2])] for d in days) < 0.015]
print(f"qubits quoted under 1.5% on all {len(days)} days: n={len(good)}, excess {st.mean(r[3]-r[4] for r in good):+.3f} (each a flag/relay readout averaged over rounds)")
# 4. which quote predicts the observed rate better?
obs = np.array([r[3] for r in rows]); pred = np.array([r[4] for r in rows])
for idx, label in ((5, "same-day"), (6, "other-days")):
    x = np.log([r[idx] for r in rows]); c = np.corrcoef(x, obs)[0, 1]
    print(f"correlation of observed rate with log {label} quote: {c:.2f}")
print(f"correlation of observed with the model's prediction: {np.corrcoef(obs, pred)[0,1]:.2f}")
# 5. implied effective readout+reset error: excess/2 on top of the quote (a flag passes one reset and one readout)
eff = [(r[5], r[5] + (r[3] - r[4]) / 2) for r in rows if r[5] < 0.02]
print(f"for qubits quoted under 2%: quoted median {np.median([e[0] for e in eff]):.2%}, implied effective per reset/readout {np.median([e[1] for e in eff]):.2%}")

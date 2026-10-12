"""Independent check of the perqubit_f analysis (own code; reads only runs/*.json).

Does not import heavyhex or the perqubit_f scripts.
"""
import json, math, glob, os
from pathlib import Path
import numpy as np
from scipy.stats import chi2

OUT = str(Path(__file__).resolve().parent)
REPO = str(Path(OUT).parents[3])


def load(p):
    with open(p) as fh:
        return json.load(fh)


jobs = {}
for f in sorted(glob.glob(REPO + "/runs/ibm_fez-*/analysis.json")):
    folder = os.path.dirname(f)
    name = os.path.basename(folder)
    jobs[name] = dict(run=load(folder + "/run.json"), a=load(f), frozen=load(folder + "/prediction.json"))


def quantities(fits, basis):
    q = {f"eps{d}": tuple(v) for d, v in fits[basis]["per_round"].items()}
    if basis == "X" and "lambda" in fits[basis]:
        q["lambda"] = (fits[basis]["lambda"], fits[basis]["lambda_uncertainty"])
    return q


def slope(frozen, expected_fits, f, basis, name):
    pts = [(0.0, frozen["0.0"]["fits"]), (1.0, frozen["1.0"]["fits"])]
    if 0.05 < f < 0.95:
        pts.insert(1, (f, expected_fits))
    x = [p for p, _ in pts]
    y = [quantities(ft, basis)[name][0] for _, ft in pts]
    if len(pts) == 2:
        return (y[1] - y[0]) / (x[1] - x[0])
    x0, x1, x2 = x
    y0, y1, y2 = y
    return (y0 * ((f - x1) + (f - x2)) / ((x0 - x1) * (x0 - x2))
            + y1 * ((f - x0) + (f - x2)) / ((x1 - x0) * (x1 - x2))
            + y2 * ((f - x0) + (f - x1)) / ((x2 - x0) * (x2 - x1)))


def job_tests(name, which, fromf_mode="carry"):
    J = jobs[name]
    a = J["a"]
    f = a["decoupled_dephasing"]
    sf = a["idle_test"]["median_uncertainty"]
    obs = a["observed"]["fits"]
    med = a["predicted_at_measured_dephasing"]["fits"]
    pq = a["predicted_with_each_qubits_dephasing"]["fits"]
    aft = a["predicted_from_calibration_after"]
    aft = aft["fits"] if aft else None
    pred = med if which == "median" else pq
    out = {}
    for basis in "XZ":
        so, sm, sp = quantities(obs, basis), quantities(med, basis), quantities(pred, basis)
        sa = quantities(aft, basis) if aft else {}
        for nm, (val, sig) in so.items():
            guess, noise = sp[nm]
            gap = abs(sm[nm][0] - sa[nm][0]) if sa else 0.0  # gap of the median-f prediction
            fromf = 0.0 if fromf_mode == "zero" else slope(J["frozen"], med, f, basis, nm) * sf
            out[f"{basis} {nm}"] = dict(O=val, sO=sig, M=guess, noise=noise, from_f=fromf, gap=gap)
    return out


def ratio(e3, e5):
    O = e3["O"] / e5["O"]
    M = e3["M"] / e5["M"]
    rel = lambda k, ag: math.sqrt(sum((r[k] / r[ag]) ** 2 for r in (e3, e5)))
    return dict(O=O, M=M, sO=O * rel("sO", "O"), noise=M * rel("noise", "M"),
                from_f=M * rel("from_f", "M"), gap=M * rel("gap", "M"))


def finish(t):
    t["half"] = math.sqrt(t["sO"] ** 2 + t["noise"] ** 2 + t["from_f"] ** 2 + t["gap"] ** 2 / 2)
    t["sM"] = math.sqrt(t["noise"] ** 2 + t["from_f"] ** 2 + t["gap"] ** 2 / 2)
    t["z"] = (t["O"] - t["M"]) / t["half"]
    return t


def reps(which, **kw):
    R = {}
    for name, J in jobs.items():
        info = J["run"]["rep"]
        R.setdefault((info["day"], info["rep"]), {}).update(job_tests(name, which, **kw))
    for tests in R.values():
        for b in "XZ":
            if f"{b} lambda" not in tests:
                tests[f"{b} lambda"] = ratio(tests[f"{b} eps3"], tests[f"{b} eps5"])
        for t in tests.values():
            finish(t)
    return dict(sorted(R.items()))


TESTS = ["X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5", "Z lambda"]


def chi2_table(R, keys=None):
    keys = keys or list(R)
    res = {}
    for t in TESTS:
        z = np.array([R[k][t]["z"] for k in keys])
        s = float((z ** 2).sum())
        res[t] = dict(chi2=s, p=float(chi2.sf(s, len(z))), bar=float(chi2.ppf(0.95, len(z))),
                      n_agree=int((np.abs(z) < 2).sum()))
    return res


med = reps("median")
pq = reps("pq")
pq0 = reps("pq", fromf_mode="zero")
med0 = reps("median", fromf_mode="zero")

lines = []
comb = load(REPO + "/runs/combined.json")
maxdiff = 0.0
for e in comb["reps"]:
    k = (e["day"], e["rep"])
    for t, row in e["tests"].items():
        mine = med[k][t]
        for a_, b_ in (("observed", "O"), ("predicted", "M"), ("observed_uncertainty", "sO"),
                       ("simulator_noise", "noise"), ("calibration_gap", "gap")):
            maxdiff = max(maxdiff, abs(row[a_] - mine[b_]))
        maxdiff = max(maxdiff, abs(abs(row["from_f"]) - abs(mine["from_f"])))
        maxdiff = max(maxdiff, abs(row["bound"] / 2 - mine["half"]))
lines.append(f"max |diff| median rebuild vs combined.json components: {maxdiff:.3g}")
cm = chi2_table(med)
lines.append("chi2 median rebuild vs combined: " + str({t: (round(cm[t]["chi2"], 6), round(comb["model"][t]["chi2"], 6)) for t in TESTS}))

pct = lambda x: 100 * x
lines.append("day,rep | X eps3 O | Mmed (z) | Mpq (z) | X eps5 ... | LX ...")
for k in med:
    m, p = med[k], pq[k]
    s = f"{k[0]} r{k[1]}"
    for t in ("X eps3", "X eps5"):
        s += (f" | {pct(m[t]['O']):.2f}+-{pct(m[t]['sO']):.2f} | {pct(m[t]['M']):.2f} ({m[t]['z']:+.2f}) sM={pct(m[t]['sM']):.2f}"
              f" | {pct(p[t]['M']):.2f} ({p[t]['z']:+.2f}) sM={pct(p[t]['sM']):.2f}")
    t = "X lambda"
    s += (f" | {m[t]['O']:.3f}+-{m[t]['sO']:.3f} | {m[t]['M']:.4f} ({m[t]['z']:+.2f}) sM={m[t]['sM']:.4f}"
          f" | {p[t]['M']:.4f} ({p[t]['z']:+.2f}) sM={p[t]['sM']:.4f}")
    lines.append(s)
lines.append("Z memory")
for k in med:
    m, p = med[k], pq[k]
    s = f"{k[0]} r{k[1]}"
    for t in ("Z eps3", "Z eps5"):
        s += f" | {pct(m[t]['O']):.2f}+-{pct(m[t]['sO']):.2f} | {pct(m[t]['M']):.2f} ({m[t]['z']:+.2f}) | {pct(p[t]['M']):.2f} ({p[t]['z']:+.2f})"
    t = "Z lambda"
    s += f" | {m[t]['O']:.3f}+-{m[t]['sO']:.3f} | {m[t]['M']:.4f} ({m[t]['z']:+.2f}) | {p[t]['M']:.4f} ({p[t]['z']:+.2f})"
    lines.append(s)

lines.append("")
for label, R in (("median", med), ("pq (noise own, from_f+gap carried)", pq),
                 ("pq, from_f=0", pq0), ("median, from_f=0", med0)):
    c = chi2_table(R)
    lines.append(f"chi2 {label}: " + "; ".join(f"{t}: {c[t]['chi2']:.1f} p={c[t]['p']:.2g} [{c[t]['n_agree']}]" for t in TESTS))
    allx = sum(all(abs(R[k][t]["z"]) < 2 for t in ("X eps3", "X eps5", "X lambda")) for k in R)
    lines.append(f"   reps agreeing on all three X: {allx}")
    for rep in (1, 2):
        keys = [k for k in R if k[1] == rep]
        c = chi2_table(R, keys)
        lines.append(f"   placement {rep}: " + "; ".join(f"{t}: {c[t]['chi2']:.1f}" for t in ("X eps3", "X eps5", "X lambda")) + f" (bar {c['X eps3']['bar']:.2f})")
for label, R in (("median", med), ("pq", pq)):
    z2 = {k: R[k]["X lambda"]["z"] ** 2 for k in R}
    lines.append(f"Lambda_X z^2 {label}: " + ", ".join(f"{k[0][5:]}r{k[1]}:{v:.2f}" for k, v in z2.items()))

lines.append("")


def wmean(vals, ses):
    w = 1 / np.array(ses) ** 2
    v = np.array(vals)
    m = (w * v).sum() / w.sum()
    sp = (w * (v - m) ** 2).sum() / (len(v) - 1)
    return m, math.sqrt(1 / w.sum()) * math.sqrt(max(sp, 1)), sp


sd5 = lambda x: x.std(ddof=1) / math.sqrt(len(x))
for basis in "XZ":
    for rep in (1, 2):
        keys = [k for k in med if k[1] == rep]
        for q in ("eps3", "eps5", "lambda"):
            t = f"{basis} {q}"
            sc = 100 if q != "lambda" else 1
            g = lambda R, f_: np.array([R[k][t][f_] for k in keys]) * sc
            O, sO, Mm, Mp, nm_, npq = g(med, "O"), g(med, "sO"), g(med, "M"), g(pq, "M"), g(med, "noise"), g(pq, "noise")
            d = Mp - Mm
            wm, wse, sp = wmean(O, sO)
            shift_chi2 = float(((d / np.sqrt(nm_ ** 2 + npq ** 2)) ** 2).sum())
            lines.append(f"{t} P{rep}: O {O.mean():.4f}+-{sd5(O):.4f} (pooled {wm:.4f}+-{wse:.4f}, chi2/nu {sp:.2f}) | "
                         f"Mmed {Mm.mean():.4f}+-{sd5(Mm):.4f} ({Mm.min():.4f}-{Mm.max():.4f}) | "
                         f"Mpq {Mp.mean():.4f}+-{sd5(Mp):.4f} ({Mp.min():.4f}-{Mp.max():.4f}) | "
                         f"shift {d.mean():+.4f}+-{sd5(d):.4f} | sim-noise on mean shift {math.sqrt((nm_**2 + npq**2).sum())/5:.4f} | shift chi2 {shift_chi2:.2f}")
            if basis == "X" and q == "lambda" and rep == 1:
                lines.append("   P1 LX daily Mmed: " + " ".join(f"{x:.4f}" for x in Mm) + f" mean {Mm.mean():.5f}")
                lines.append("   P1 LX daily Mpq : " + " ".join(f"{x:.4f}" for x in Mp) + f" mean {Mp.mean():.5f}")

lines.append("")
tot = exc = neg = 0


def arr(lists):
    return np.array([np.nan if s is None else s for v in lists for s in v], dtype=float)


for name, J in jobs.items():
    a = J["a"]
    raw = arr(a["idle_test"]["per_qubit"].values())
    unc = arr(a["idle_test"]["uncertainty"].values())
    e = int((~np.isfinite(raw)).sum())
    n = int((np.isfinite(raw) & (raw < 0)).sum())
    tab = a["dephasing_by_qubit"]
    e2 = sum(not r["measured"] for r in tab)
    n2 = sum(r["measured"] and r["dephasing"] == 0 for r in tab)
    tot += len(raw); exc += e; neg += n
    by = {}
    for r in tab:
        by.setdefault(r["patch"], []).append(r["dephasing"])
    rawby = {d: arr([v]) for d, v in a["idle_test"]["per_qubit"].items()}
    extra = "; ".join(f"d{d}: med(table) {np.median(v):.3f} mean(table) {np.mean(v):.3f} med(raw) {np.nanmedian(rawby[str(d)]):.3f}" for d, v in sorted(by.items()))
    fin = raw[np.isfinite(raw) & np.isfinite(unc)]
    lines.append(f"{name}: n={len(raw)} excluded={e} (table {e2}) negative={n} (table zero {n2}) | f={a['decoupled_dephasing']:.4f} "
                 f"recomputed {np.median(fin):.4f} sf {1.25*np.std(fin)/math.sqrt(len(fin)):.4f} vs {a['idle_test']['median_uncertainty']:.4f} | {extra}")
lines.append(f"TOTAL n={tot} excluded={exc} negative={neg}")

txt = "\n".join(lines)
print(txt)
open(OUT + "/verify_output.txt", "w", encoding="utf-8").write(txt)

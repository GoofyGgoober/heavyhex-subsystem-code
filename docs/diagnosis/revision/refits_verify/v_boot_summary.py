"""Verifier: bias of both estimators in the parametric bootstrap (v_boot.json), and
bootstrap 68% percentile intervals for Lambda compared with the quasi-profile ones."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "v_results.json").read_text())
boot = json.loads((HERE / "v_boot.json").read_text())

print("Bias in eps (pp): mean over 10 reps of [mean(replicate) - true]; range over reps")
for key in ("X3", "X5", "Z3", "Z5"):
    bw, bm, medw = [], [], []
    for i, r in enumerate(res):
        t = r["curves"][key]["mle"]["eps"]
        b = boot[f"{i}:{key}"]
        w, m = np.array(b["wls"]), np.array(b["mle"])
        bw.append((w.mean() - t) * 100)
        bm.append((m.mean() - t) * 100)
        medw.append((np.median(w) - t) * 100)
    print(f"  {key}: WLS {np.mean(bw):+.2f} (range {min(bw):+.2f} .. {max(bw):+.2f}; median-based {np.mean(medw):+.2f})"
          f"   MLE {np.mean(bm):+.3f} (max |.| {max(map(abs, bm)):.3f})")

print("\nBias in Lambda: mean over reps of [mean(replicate ratio) - true ratio]")
dev = []
for b_ in "XZ":
    bw, bm = [], []
    for i, r in enumerate(res):
        t = r["curves"][f"{b_}3"]["mle"]["eps"] / r["curves"][f"{b_}5"]["mle"]["eps"]
        w = np.array(boot[f"{i}:{b_}3"]["wls"]) / np.array(boot[f"{i}:{b_}5"]["wls"])
        m = np.array(boot[f"{i}:{b_}3"]["mle"]) / np.array(boot[f"{i}:{b_}5"]["mle"])
        bw.append(w.mean() - t)
        bm.append(m.mean() - t)
        lo, hi = np.percentile(m, [15.865, 84.135])
        q = r["lam"][b_]["quasi"]
        p = r["lam"][b_]["prof"]
        dev.append((max(abs(lo - p[0]), abs(hi - p[1])), max(abs(lo - q[0]), abs(hi - q[1])), r["day"], r["rep"], b_))
    print(f"  Lambda_{b_}: WLS {np.mean(bw):+.4f} (range {min(bw):+.3f}..{max(bw):+.3f})   MLE {np.mean(bm):+.4f} (max |.| {max(map(abs, bm)):.4f})")
print("\nmax |bootstrap 68% endpoint - pure-profile endpoint| for Lambda:", round(max(d[0] for d in dev), 4))
print("max |bootstrap 68% endpoint - quasi-profile endpoint| for Lambda:", round(max(d[1] for d in dev), 4), max(dev, key=lambda d: d[1]))

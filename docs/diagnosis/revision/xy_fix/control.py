"""Control: run.analyze, unpatched, on a scratch copy of one job; must reproduce its analysis.json."""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
from heavyhex.experiment import run as R
name, scratch = sys.argv[1], Path(sys.argv[2])
dest = scratch / name
dest.mkdir(parents=True, exist_ok=True)
for f in ("run.json", "calibration.json", "calibration_after.json", "circuits.qpy.gz", "job.json", "prediction.json", "shots.npz"):
    shutil.copy2(REPO / "runs" / name / f, dest / f)
new = R.analyze(dest)
old = json.loads((REPO / "runs" / name / "analysis.json").read_text())
for t in old["agreement"]["tests"]:
    a, b = old["agreement"]["tests"][t], new["agreement"]["tests"][t]
    print(t, {k: (a[k], b[k]) for k in ("observed", "predicted", "bound")})
print("f", old["decoupled_dephasing"], new["decoupled_dephasing"])

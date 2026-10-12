"""Collect the verification's key numbers into verify_summary.json."""
import json
from vcommon import HERE
J = lambda n: json.loads((HERE / n).read_text())
rd, sim, fs = J("v_redecode.json"), J("v_redecode_sim20000.json"), J("v_faults_summary.json")
out = {
    "counterexamples": J("v_reproduce.json"),
    "remnants_and_boundary_feed": J("v_remnants.json"),
    "fault_scan": fs,
    "uniform_tie_check_3seeds": J("v_ties.json"),
    "uniform_tie_check_12seeds": J("v_ties12.json"),
    "hardware": {k: rd[k] for k in ("pooled", "mean_eps", "mcnemar")},
    "hardware_points_dropped_P_ge_half": sim.get("points_dropped_P_ge_half"),
    "hardware_reps": rd["reps"],
    "model_sim_20000_pooled": sim["sim"]["pooled"],
}
(HERE / "verify_summary.json").write_text(json.dumps(out, indent=1, default=str))
print("ok")

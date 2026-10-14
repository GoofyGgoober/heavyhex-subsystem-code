"""The light pass: every summary step that reads committed results or runs/, then the ledger.

Each step takes seconds to about a minute and rewrites its results file; the heavy steps that
feed them (README, "Runtime") are not run, and neither are fits.py and perqubit_f.py, whose
committed outputs come from another machine and would move in their last digits. Elsewhere
too, outputs that come from optimizers or seeded samplers can move in their last digits.

    python docs/analysis/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time

from common import HERE

LIGHT = [
    ["hmm.py"],
    ["hmm.py", "--basis", "X"],
    ["hmm_causes.py"],
    ["readout.py"],
    ["placements.py"],
    ["model_vs_device.py"],
    ["effect_sizes.py"],
    ["sensitivity.py"],
    ["dependence.py"],
    ["prespecified_tests.py"],
    ["f_models.py"],
    ["placement_counts.py"],
    ["timing.py"],
    ["calibration_inputs.py"],
    ["detection_events.py"],
    ["idle_f.py"],
    ["idle_f_t2.py"],
    ["idle_test_scope.py"],
    ["fit_bias.py"],
    ["cz_scaled.py"],
    ["error_scale.py"],
    ["error_budget.py"],
    ["channels.py"],
    ["dephasing_floor.py"],
    ["f_weighted_mean.py"],
    ["f_treatments.py"],
    ["input_resolution.py"],
    ["input_noise.py"],
    ["stat_checks.py"],
    ["leakage_fit.py"],
    ["leakage_checks.py", "A", "E"],
    ["decoder.py"],
    ["decoder_dominant.py"],
    ["decoder_ml.py"],
    ["ledger.py"],
]


def main() -> None:
    for step in LIGHT:
        t0 = time.time()
        done = subprocess.run(
            [sys.executable, str(HERE / step[0]), *step[1:]], capture_output=True, text=True
        )
        if done.returncode:
            sys.exit(f"{' '.join(step)} failed:\n{done.stderr}")
        print(f"{' '.join(step):28s} {time.time() - t0:5.1f} s", flush=True)


if __name__ == "__main__":
    main()

# Why the model misses: the post-run diagnosis

Exploratory analysis, decided after the run, of why the simulation from IBM's calibration
under-predicts the chip's errors. It reads the committed run folders and IBM's published
calibrations, uses no QPU time, and changes nothing in `src/` or in the registered results.
The paper's section "Why the model misses" reports it.

Run from the repository root, in this order (each writes to `results/`):

| Script | What it does |
| --- | --- |
| `fetch_properties.py` | IBM's per-qubit readout and reset errors and per-coupler CZ errors for each run day (read-only API) |
| `repeats.py` | Do flags and relays fire again after firing? The model says no; the chip says yes |
| `leakage_hmm.py` | A two-state hidden Markov fit per flag and relay: leak rate, lifetime, leaked readout |
| `why.py` | What the fitted leakage follows: CZ count, CZ error, readout, neighbouring CZs |
| `where.py` | Do flags that share a data qubit, or a Z gauge, fire in streaks together? |
| `readout_bins.py` | Flags against IBM's quoted readout, by the day's own and the other days' quotes |
| `stats.py` | Λ by placement, other estimators, and the agreement test with the calibration's day-to-day scatter |
| `uniform_scale.py` | Where these circuits would reach Λ = 1 on a uniform chip, every error scaled together |
| `no_reset.py` | What the resets cost: the calibration model with every reset taking no time |
| `leaky.py` | A Pauli-frame simulator for the stim model with a leaked state per qubit (matches stim without leakage) |
| `fit_leak.py`, `fit_leak2.py` | Fit the leakage model's inputs to detection statistics only |
| `rerun.py`, `rerun_summary.py` | Rerun every job's memory circuits with the fitted model, decoded as the chip's shots were |

`common.py` holds the shared loading.

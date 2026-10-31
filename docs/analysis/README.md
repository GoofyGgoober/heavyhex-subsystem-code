# Analyses

These scripts analyse the saved records in `runs/`. They use no QPU time and change neither
`src/` nor `runs/`. Run them from the repository root in the pinned environment (see the
[main README](../../README.md#install)):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/analysis/stored_priors.py   # about 3 min
python docs/analysis/run_all.py      # every light step, then ledger.py: a few minutes
```

Each script writes `results/<script>[_<variant>].json`, and the results are committed. A
script whose name has a suffix (`decoder_faults.py`) is a heavy step that feeds the script
named by its prefix (`decoder.py`). `run_all.py` runs only the light steps. It leaves out
`stored_priors.py`, run on its own above, and `fits.py` and `perqubit_f.py`, whose committed
outputs come from another machine and would move in their last digits. Large intermediates go
to `cache/` (git-ignored) and are rebuilt when needed. Heavy steps run with `--workers 2` by
default, at about 0.5–2 GB per worker.

## Where each result comes from

Sections, figures and tables are those of the write-up, [paper/main.pdf](../../paper/main.pdf).
Scripts and outputs named without a folder are in `docs/analysis/` and `docs/analysis/results/`;
other paths start at the repository root. *Stored priors* are the decoder priors of the f = 1
prediction stored before each job; *in-job priors* are those at the job's median idle-test f.

| Result | Script | Output | Runtime |
| --- | --- | --- | --- |
| Sec. 2, Fig. 1: the Fez layouts; rings mark the qubits the placement rule calls weak (`make_figures.weak_qubits`) | `docs/figures/figure1.py`, via `paper/figures/make_figures.py` | `paper/figures/layout.pdf` | seconds |
| Sec. 2: placements searched and usable by day (18 at distance 3, 2 at distance 5); the stricter cutoffs that leave no distance-5 placement; every patch's weak parts on every day by the rule's own test (`placement.weak`); relay q51; each job's round length. Also the qubit sets of all embeddings, and where pair 2's distance-3 patch ranked and sat | `placement_counts.py` | `placement_counts.json` | seconds |
| Fig. 2(a): the bulk Z-gauge gadget | `paper/figures/gadget.tex` (`latexmk -pdf gadget.tex` in `paper/figures/`) | `paper/figures/gadget.pdf` | seconds |
| Sec. 2, Fig. 2(b): one round of X memory as compiled (day 1, pair 1, distance 3, the middle round of the 3-round circuit), one qubit per role, and every qubit's time by kind and role in both patches; the critical path (two resets of 1.684 µs and two readouts of 1.660 µs, 6.69 of the 7.896 µs, the rest gates and short waits: 0.28 µs in the X half, 0.93 µs in the Z half); the round with zero-duration resets (4.528 µs); each coupler's CZ time read back from the delays (68 ns; 88 ns on 102–103 and 146–147). Harper et al.'s 3.2 µs round is taken from their paper | `timing.py`; figure by `paper/figures/fig_timing.py` | `timing.json`, `paper/figures/timing.pdf` | a few seconds |
| Secs. 2 and 5: circuit distance from exhaustive and Stim fault searches; the single faults that matching with calibrated priors miscorrects (none with uniform weights); the device's shots decoded with calibrated and with uniform weights | `decoder.py` (with `decoder_faults.py`, `decoder_weights.py`, `decoder_mass.py`, `decoder_examples.py`, `decoder_shots.py`, `decoder_ties.py`) | `decoder*.json` | `decoder.py` seconds; `decoder_ties.py` 2 min; `decoder_faults.py` and `decoder_shots.py` heavy |
| Sec. 3, Fig. 3: the idle-test decay against its f = 0 and f = 1 model curves for three data qubits (the patch's shortest-T2 qubit, the one with f closest to the job's median, and the one with the patch's median reduced χ², 0.90); f against IBM's reported T2 per physical qubit; f from the memory runs against the idle test; decoding with per-qubit f | `idle_f_t2.py`; figure by `paper/figures/fig_C_idle_dephasing.py` | `idle_f_t2.json`, `paper/figures/C_idle_dephasing.pdf` | seconds |
| Sec. 3: each data qubit's f from the idle test, with the variance of each point when ⟨X⟩ and ⟨Y⟩ are sampled separately | `idle_f.py` (with `idle_f_reanalyze.py`) | `idle_f.json`, `idle_f_*.json` | seconds; the reanalysis 15 min at 1–2 GB per job |
| Sec. 3: q150 and the choice of f per qubit. Pair 1's predicted ε₃, ε₅ and Λ_X per day, and pair 1 minus pair 2 predicted Λ_X against the observed 0.17 ± 0.03, with every qubit at the job's median f (0.19), each data qubit at its own f (0.32), the median except q150 at its own f (0.20), and each data qubit's f shrunk toward the median by its idle-test noise (0.31; 0.29 with that noise scaled by the fit's reduced χ²), all simulated at 50,000 shots with shared seeds; q150's Z-or-Y error per round in the model (32–39% at the median f, 4–13% at its own on days 1, 2, 3 and 5) | `f_models.py` (reads `runs/`, `placements.json`, `sensitivity.json`, `idle_f_inputs.json`, `perqubit_f_decode.json`, `fits.json`, `fits_counts.json`) | `f_models.json`, `f_models_runs.json` | seconds; `--recompute` 6 min on 2 workers, 0.4 GB each |
| Sec. 3: the prediction with each data qubit at its own measured f | `perqubit_f.py` (with `perqubit_f_mc.py`, `perqubit_f_decode.py`) | `perqubit_f*.json` | seconds; `perqubit_f_mc.py` about 20 min on 14 workers, `perqubit_f_decode.py` minutes |
| Sec. 3, Eq. (10): the idle test's extra dephasing as a multiple of IBM's, a constant floor, or both, and the floor as a phase flip per round; the model's idle error per data qubit and round | `dephasing_floor.py` | `dephasing_floor.json` | seconds |
| Sec. 3: what the idle test sees. Its ancilla, flag and relay readouts read 1 in 1.7–5.7% of shots (job means), against 28.5–32.6% in memory; the 11 qubit-jobs left out of the job's median f have a median reported T2 of 166 µs, against 85 µs for the 329 used | `idle_test_scope.py` (reads the shots and `idle_f_t2.json`) | `idle_test_scope.json` | seconds |
| Sec. 3: the age of the calibration snapshot in force, and of IBM's previous one, at each day's first job, and which values they share; IBM's median single-qubit error, and the model's noise on the two SX gates around a CZ as a share of the CZ's own (26–28%). Not in the write-up: Pr(0\|1) − Pr(1\|0) on days 1–3 and the reported T1 of the flag and relay qubits | `calibration_inputs.py` | `calibration_inputs.json` | a second |
| Sec. 3: the calibrations pulled while each day was prepared, before the one in force, are in the git history. On day 1 every value equals the 17:05 UTC snapshot's (268 min before the first job, commit `9b2993e`), and every value but the readout errors equals the 16:09 UTC one's (324 min, `186b8f5`; 154 of 156 readout errors differ). On day 2 every value, single-qubit errors included, equals the 16:50 UTC snapshot's (75 min, `501ac84`) | `git show <commit>:runs/<folder>/calibration.json`, compared with the `calibration.json` in force | none | seconds |
| Sec. 3: IBM's readout errors are whole multiples of 1/8192 (4096 shots per prepared state); their relative standard error (11% at the median) | `input_resolution.py` | `input_resolution.json` | a second |
| Sec. 3: the model with IBM's initialization error for the resets and its separate P(1\|0) and P(0\|1) for the readouts (`readout_ibm.py`). Not in the write-up: flag and relay firing against the reported readout error (`readout.py`) | `readout_ibm.py`, `readout.py` | `readout_ibm.json`, `readout.json` | seconds; `readout_ibm.py` 6 min, 1.1 GB |
| Sec. 3: the model with Harper et al.'s mid-circuit reset (7.5%) and readout (4.2%) errors, day 1 | `reset_harper.py` (after `pij.py`) | `reset_harper.json` | 20 s |
| Input to `readout_ibm.py` and others: IBM's separate readout errors and initialization error per qubit, and where they may be used ([below](#ibms-properties-snapshot)) | `properties.py` (fetching needs an IBM account) | `properties.json` | not rerun |
| Sec. 4: the device against the stored prediction, both decoded with the stored priors and fitted by binomial maximum likelihood: ε above the prediction on 39 of 40 curves; the geometric-mean factors (1.27 and 1.19 in X memory, 2.06 and 1.84 in Z) with five-day intervals; other fit windows; the failure counts at every depth. It also checks that the first job, decoded with its in-job priors, reproduces its stored analysis | `stored_priors.py` (saved shots only, no new sampling) | `stored_priors.json`; count caches in `build/fez-stored-priors/` | about 3 min, one process, 0.41 GB |
| Sec. 4, Fig. 4: failure probability at one to four rounds, averaged within each day and then over the five days, with simultaneous t intervals over sixteen cells and leave-one-day-out sensitivity. Sec. 5: the change in the device's distance-3 X-memory failure under exact maximum-likelihood (ML) decoding | `early_depths.py` (reads `stored_priors.json`, `fits_counts.json`, `fits.json`, `decoder_ml.json`); figure by `paper/figures/fig_depth_errors.py` | `early_depths.json`, `paper/figures/depth_errors.pdf` | under a second |
| Sec. 4: the device/model ε factors with one day left out (four-day t intervals: distance-5 X memory's includes one after leaving out day 2, 3 or 4, the other three stay above one) and at the fit-window boundaries | `day_sensitivity.py` (reads the five saved daily means) | `day_sensitivity.json` | under a second |
| Sec. 4, Fig. 5(a): curve fits with in-job priors, per pair-day and per placement | `fits.py`, then `placements.py` | `fits.json`, `fits_per_rep.csv`, `fits_counts.json`, `placements.json` | about 5 min with the saved bootstrap; `--boot 2000` adds about 30 min |
| Sec. 4: what drives the swing of pair 1's predicted Λ_X (each reported value set to its median over the other days) | `day_swing.py` | `day_swing.json` | 5 min on 2 workers |
| Fig. 5: pair 1's Λ_X by day with in-job priors; flag and relay persistence with the hidden-Markov curves | `paper/figures/fig_diagnostics.py` (reads `fits.json`, `flags_summary.json`) | `paper/figures/diagnostics.pdf` | seconds |
| Sec. 5, Table 1: each job's model simulated with each of the 31 nonempty subsets of its five channels (CZ, single-qubit gates, readout and reset, T1 decay while idle, dephasing while idle), the noiseless case taken as zero. From these, each channel's ε alone, the drop with it off and its Shapley share, by basis, distance and placement; Λ with each channel off, also with a decoder matched to it; and each channel alone scaled until the model gives the device's ε and its mean stabilizer detection rate over rounds 2–8 of the eight-round circuits. Not in the write-up: every idle shortened to the zero-reset round or to Harper et al.'s 3.2 µs round. A parallel X/Z schedule is not simulated, since the X ancillas double as the bulk Z gauges' flags; it would need another gadget, such as Harper et al.'s | `error_budget.py` (reads `error_scale.json`, `fits.json`, `no_reset.json`, `uniform_scale.json`, `docs/figures/figure3.json`) | `error_budget.json`, `error_budget_runs.json` | 75 s; `--recompute` 25 min on 2 workers, 0.4 GB each (5 min a day with `--jobs`) |
| Sec. 5: one-channel factors with Birge-widened 95% intervals; whether one factor on a channel fits several targets; two channels fitted together (relaxation × 2.63 and dephasing × 0.60 for the four decay summaries); IBM's observed spread. The intervals leave out the correlations from shared shots, days and overlapping placements | `channels.py` (reads `error_budget.json`, `error_budget_runs.json`, `error_scale.json`, `f_models.json`, the jobs' calibrations) | `channels.json`, `channels_runs.json` | 35 s; `--recompute` 7 min on 2 workers, 0.3 GB each |
| Sec. 5: one factor k on every error of the model (CZ, single-qubit, readout and reset, idle decay at the job's f) that gives the device's ε, and the one that gives its stabilizer detection rate, per basis, distance and placement; the model's Λ at that k | `error_scale.py` | `error_scale.json`, `error_scale_runs.json` | 30 s; `--recompute` 7 min on 2 workers, 0.4 GB |
| Sec. 5: the matching graphs of the jobs as run. Graphlike logical paths have 3 and 5 edges in X memory at distances 3 and 5, but 3 in distance-5 Z memory. Stim's observable-only components hold 100 of the 328 distance-3 Z single faults that matching miscorrects; folding their labels into other components changes neither the graph nor the decoding of any device shot or of 10,000 model shots per circuit. The file's `uniform` entry keeps per-qubit single-gate errors and is not in the write-up | `decoder_graph.py` | `decoder_graph.json` | 3.5 min on 2 workers, 0.5 GB each |
| Sec. 5: Z memory at one to four rounds in all fifteen jobs with the flag and relay detectors removed (gates, readouts and the logical observable kept): matching with all detectors and with the reduced set, and exact ML on the reduced syndrome, with the stored priors on device and model shots alike | `PYTHONPATH=src:docs/analysis OPENBLAS_NUM_THREADS=1 python docs/analysis/z_decoder_checks.py` | `z_decoder_checks.json` | about 2 min, one process, under 0.5 GB |
| Sec. 5: exact maximum-likelihood decoding of distance-3 X memory (one to four rounds, every patch-day) against matching on the same in-job detector error model: the model's logical error under each; the share of the single-fault probability miscorrected by matching that ML also assigns to the wrong value (80% at four rounds; 86–100% on pair 1 and on pair 2's day 3, 16–23% on pair 2's other days); the device's shots decoded both ways. Matching reproduces the stored logical error exactly | `decoder_ml.py` | `decoder_ml.json` | 40 s on 2 workers, under 0.5 GB each |
| Sec. 5: edge probabilities p_ij learned from the detection events, by edge class; detector pairs the model never joins; every edge with p > 0.1 and the faults behind it | `pij.py` | `pij.json` | 1 min on 2 workers |
| Sec. 5: the device decoded with the learned priors, and how much of the device–model gap that closes, with a control on the model's shots | `pij_decode.py` (after `pij.py`) | `pij_decode.json` | 5 min on 2 workers |
| Sec. 6: how often a flag or relay that read 1 reads 1 again in the next round (23.0%, against 8.2% after a 0 and 6.9% in the model); the persistence Δ(ℓ) of Eq. (13) by lag | `refiring.py` (`--basis X` for X memory) | `refiring.json`, `refiring_x.json` | 30 s each |
| Sec. 6, Eq. (14): two-state hidden Markov fits to the flag and relay records: onset and exit rates, the readout in the state, its lifetime, and its occupation in steady state and over the rounds measured. Not in the write-up: onset rate by role, per CZ and per readout or reset; onset and exit rates of the qubits in both roles; correlations over qubits | `hmm.py` (`--basis X` for X memory), `hmm_causes.py`, `hmm_bootstrap.py` | `hmm*.json` | seconds; `hmm.py --recompute` 3 min per basis; `hmm_bootstrap.py` hours |
| Sec. 6, Fig. 5(b): the persistence in both bases; the fit Δ(ℓ) = Δ₀κ^ℓ + Δ_∞ (`A`, `r`, `C` in the file); the hidden Markov model against models whose rates differ only from shot to shot; co-streaks over qubits | `flags_xmemory.py` (after `refiring.py --basis X` and `hmm.py --basis X --recompute`) | `flags_summary.json` | 4 min the first time (the series go to `cache/`), then 45 s |
| Sec. 6: what a streak (a flag or relay reading 1 twice in a row) predicts: detection events on the nearby and the other stabilizers from two rounds before to three rounds after it, within qubit, role, job, circuit and round, against the model's exact values; the logical failure of shots with a streak, by job, circuit and patch, also at a fixed number of flag and relay 1s | `streak_effects.py` | `streak_effects.json` | 3 min on 2 workers, under 0.5 GB each |
| Sec. 6: the leakage simulation fitted to the detector statistics, and its logical errors in both memories for each assumed action of a leaked qubit on its CZ partner | `leakage_fit.py` (with `leakage_grid.py`, `leakage_rerun.py`, `leakage_grid_bootstrap.py`, `leakage_frame_grid.py`, `leakage_frame_rerun.py`, `leakage_frame_counts.py`) | `leakage_fit.json`, `leakage_grid_*.json`, `leakage_rerun_*.json`, `leakage_frame_*.json` | `leakage_fit.py` 15 s; a grid point 2.3 min and a rerun 5 min per day on 2 workers, 0.5 GB |
| Not in the write-up: checks of the leakage simulators, among them the left-out CZ ([below](#the-left-out-cz-in-frame_simpy)); recovery tests of the fit and of the lifetime (the simulated lifetime whose fitted persistence matches the device's) | `leakage_checks.py`, `leakage_roundcond.py`, `leakage_recovery.py`, `leakage_lifetime.py`, `leakage_heldout.py` | `leakage_checks.json` and one file each | parts A and E seconds; `leakage_lifetime.py` 3.5 min on 2 workers; the rest heavy (cached in `cache/`) |
| Sec. 7: Kingston memory rates and Λ, the stored prediction's device/model ratios (Table 2, first row), the idle-test T_q, the lifetime and Z idle-test fits, and two-state fits to the raw readout. `exp3_analysis` writes only part of `exp3.json`: the `pooled`, `findings` and `beside` sections and the three-state fits are not written by it, and the write-up's raw-readout numbers come from `kingston_iq.py` | `followup/analysis.py` (`exp1`, `life`, `exp3_analysis`), reading `runs/ibm_kingston-2026-10-14-*` | `runs/ibm_kingston-followup/analysis/exp1.json`, `life.json`, `exp3.json` | minutes |
| Sec. 7, Table 2 (rows below the stored prediction), Fig. 6 (top): the device and every simulated input variant decoded with one capped decoder built from the pooled measured T_q; probabilities by depth, Pearson dispersion and exponential residuals; the re-prediction with the later readout errors; each job against the pooled model | four `python -m followup.reproduce` runs with `--simulation-priors hardware`, then `PYTHONPATH=. python docs/analysis/kingston_checks.py` ([commands](../../followup/README.md#fixed-decoder-comparisons)) | `kingston_checks.json`; the simulations in `build/followup/fixed-*.json` | about 80 s per variant, one process, under 0.75 GB; the collector seconds |
| Fig. 6 (top) | `paper/figures/fig_kingston_curves.py` (reads `kingston_checks.json`) | `paper/figures/kingston.pdf`; also `paper/figures/F_kingston_curves.pdf`, the simulated curves' exponential residuals, not in the write-up | seconds |
| Sec. 7, Fig. 6(a, b): the joint idle fit's log residuals for all 25 data qubits, with a median and the worst qubit; a check of the count estimator on planted counts (2000 replicates per qubit and basis, seed 1313) | `PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python docs/analysis/idle_checks.py` (reads `build/followup/fixed-joint.json`); figure by `paper/figures/fig_idle_checks.py`, which takes Fig. 6(c) from `kingston_iq.json` | `idle_checks.json`, `paper/figures/F_idle_checks.pdf` | about 5 s, one process |
| Sec. 7, Fig. 6(c): two-state hidden Markov fits refitted from the raw IQ records; readouts outside both reference clouds; the signal in the fitted state, projected on each qubit's 0-to-1 axis; median onset and lifetime by arm; initial and steady occupation; occupation correlations between qubits that share a CZ partner. It reads the raw records, not `exp3.json` | `PYTHONPATH=.:src:docs/analysis OPENBLAS_NUM_THREADS=1 python docs/analysis/kingston_iq.py` | `kingston_iq.json` | about 3 min, one process |
| Not in the write-up: each Kingston input variant decoded with its own priors (bounded and unconstrained joint idle fits, separate X/Y and Z fits, the later readout errors), compared with `life.json` and `exp1.json` | `python -m followup.reproduce --variant <name>` ([variants](../../followup/README.md#reconstructing-the-input-variants)) | `build/followup/<variant>.json` | about 1 min per variant, one process, under 0.75 GB |
| Not in the write-up: day 5's pair-2 distance-3 Z memory (ε₃ 25.5% against 8.1% predicted) and the reported readout error of q90 in that patch, 14.4% (`runs/ibm_fez-2026-10-08-r2-d3/calibration.json`) | `fits.py` | `fits_per_rep.csv` | with `fits.py` |
| Not in the write-up: drift within a job; the rise of the excess detection rate with round count, beyond a 3-standard-error slope on 25 of 40 curves; calibration and idle-test diagnostics | `model_vs_device.py` | `model_vs_device.json`; per-job `analysis.json` for each curve's slope | seconds |
| Not in the write-up: heating within a job (the detection-event rate from the start to the end of a job changes by −0.21%, 95% interval −0.57% to +0.15%) and the time order of the shots (relay q51 steps at shot 371 in all 68 sub-batches of 625 shots of its day-3 job) | `heating.py` (with `heating_extract.py`) | `heating.json`, `heating_extract.json` | extraction 10 min to `cache/`, then minutes |
| Not in the write-up: day 1's stored predictions derived again with the frozen code | `repredict.py` | `repredict.json` | 2 min on 2 workers, 1 GB |
| Not in the write-up: device/model per placement with in-job priors, propagated input errors, fit windows and tracking across days | `effect_sizes.py` | `effect_sizes.json` | 15 s |
| Not in the write-up: input uncertainty, fit windows, interpolation and statistical power | `sensitivity.py` | `sensitivity.json` | 40 s, 0.3 GB |
| Not in the write-up: correlations from shared days, placement-specific summaries, day-block and hierarchical estimates | `dependence.py` | `dependence.json` | 30 s, 0.25 GB |
| Not in the write-up: the agreement tests specified before the run | `prespecified_tests.py` | `prespecified_tests.json` | 2 s, 0.1 GB |
| Not in the write-up: binomial against log-linear fits, multiplicity, fit windows and detector thresholds | `stat_checks.py` | `stat_checks.json` | 3 s |
| Not in the write-up: how the rates respond to each calibration input, and the propagated input errors | `input_noise.py` | `input_noise.json`, `input_noise_runs.json` | 2 s; `--recompute` about 3 min on 2 workers |
| Not in the write-up: device/model under each treatment of f (f = 0, f = 1, the job's median, per qubit, the inverse-variance mean), with day-block intervals (`ci95_family_of_five`). Not every treatment holds the decoder fixed; the Sec. 4 comparison does (`stored_priors.py`) | `f_treatments.py` | `f_treatments.json` | a second |
| Not in the write-up: device/model with every qubit at the inverse-variance mean of its job's per-qubit f instead of the median; f from the memory runs' detection rates against the idle test | `f_weighted_mean.py` | `f_weighted_mean.json`, `f_weighted_mean_runs.json` | 40 s; `--recompute` 2 min on 2 workers, 0.4 GB |
| Not in the write-up: each curve's device ε against the stored prediction (f = 1 on Fez, IBM's T2 on Kingston) and against the prediction at the measured dephasing (the job's f on Fez, each data qubit's T_q on Kingston) | `paper/figures/fig_obs_pred.py`, via `make_figures.py` (reads `stored_priors.json`, `fits.json`, `effect_sizes.json`, `runs/ibm_kingston-followup/analysis/exp1.json`) | `paper/figures/obs_pred.pdf` | seconds |
| Not in the write-up: parametric-bootstrap bias of the log-linear fit, with weights at the observed or at the true probabilities | `fit_bias.py` | `fit_bias.json` | seconds |
| Not in the write-up: the isolated single faults that matching miscorrects, per round; the largest group (Z on q44 on day 1: 3.3% per round over 15 mechanisms, 85% of the miscorrected probability), with its largest mechanism (p = 0.0355, edge weight ln[(1−p)/p] = 3.30) matched instead through edges of 0.1344 and 0.2042 (total weight 3.22) | `decoder_dominant.py` | `decoder_dominant.json` | 20 s |
| Not in the write-up: detection-event fraction by detector class and round, device against model | `detection_events.py` | `detection_events.json` | a second |
| Not in the write-up: correlated matching on device and model shots; Λ_Z per pair-day | `corr_matching.py` | `corr_matching.json` | 3.5 min on 2 workers |
| Not in the write-up: whether flags and relays that read a common data qubit fire in streaks together | `costreak.py` | `costreak.json` | 30 s |
| Not in the write-up: the CZ error alone raised until the mean bulk stabilizer detection rate matches the device's, in the four jobs of days 3 and 5 (`follow`, `cz_factor_at_chip_rate`: 1.2–4 times in 10 of 12 patch-bases, beyond the largest simulated 4 in the other two) | `cz_scaled.py` | `cz_scaled.json`, `cz_scaled_runs.json` | seconds; `--recompute` heavy |
| Not in the write-up: the model with zero-duration resets; `timing.py` checks its round length and `error_budget.py` reads it | `no_reset.py` | `no_reset.json` | heavy |
| Not in the write-up: the model with CZ, readout, T1 and T2 set to the day-5 chip medians, single-gate errors still per qubit, scaled together; its rates are log-linear fits | `uniform_scale.py` | `uniform_scale.json` | a few min |
| Not in the write-up: the model on day 1's calibration before the run, with weak parts repaired, nested patches, single inputs halved or doubled, and a preset at chip medians (single-gate errors still per qubit); its Λ_X of 0.86 at f = 1 and 0.94 at f = 0 come from log-linear fits | `docs/figures/figure3.py` | `docs/figures/figure3.json` | heavy |
| Not in the write-up: the power to detect a Λ_X offset of 0.06 with f known (`sensitivity.py` adds the uncertainty in f) | `docs/power.py` (command below) | `power.json` | about 5 min |
| Not in the write-up: a calibration-scatter check for f | `docs/calibration_scatter.py` | printed | heavy |
| Every figure of the write-up but Fig. 2(a), and `pn.pdf`, `obs_pred.pdf` and `F_kingston_curves.pdf` | `paper/figures/make_figures.py`, which calls the `fig_*.py` scripts | `paper/figures/*.pdf` | 15 s |
| Summary numbers from the results above, each with its source | `ledger.py` | `paper_numbers.json`, `paper_numbers.txt` | a second |

`docs/power.py` uses the stored f = 0 and f = 1 predictions, with seed 1 and 2,000 simulated
experiments per rate by default:

```bash
python docs/power.py runs/ibm_fez-2026-10-0{4,5,6,7,8}-r{1,2-d3,2-d5} --out docs/analysis/results/power.json
```

Shared modules, not run on their own: `common.py` (paths, the jobs, readout series),
`fitting.py` (the weighted-log and binomial fits), `matching.py` (the decoder's graph),
`leakage.py` (the leakage fit's jobs, statistics and cached simulations), `frame_sim.py`,
`exact_sim.py` and `tableau_ref.py` (the three leakage simulators), `detectors.py` (detector
classes, edge types, the p_ij estimator and learned priors) and `refit.py` (fits of re-decoded
shots, paired bootstrap).

## IBM's properties snapshot

`calibration.json` keeps each qubit's readout error, the mean of P(1|0) and P(0|1), but neither
of the two on its own, nor IBM's initialization error; `properties.json` adds them. It was
fetched after the run with each day's calibration time as the query time. IBM serves the latest
snapshot strictly older than the time asked, so each day's entry is the snapshot before the one
the jobs ran under (`served`, 36 to 84 minutes before it). Its readout errors equal the ones in
force on all 156 qubits on days 1–3, but on only one qubit on each of days 4 and 5 (relay q51
has 4.7% in it on day 4, against 48.6% in force). The analyses therefore read it through
`properties.py`, which keeps a qubit's P(1|0), P(0|1) and initialization error only where its
readout error matches the one in force, taking the initialization error to come with the
readout calibration: every qubit on days 1–3, one on each of days 4 and 5. Elsewhere resets and
readouts keep the readout error in force, as in the model. Run as a script, `properties.py`
fetches the snapshot in force and checks it against `calibration.json`; that needs an IBM
account, so the committed file is the one described here.

These results use the filtered inputs: `readout_ibm.py`, `readout.py`, `calibration_inputs.py`,
`hmm_causes.py`, the exact leakage reruns (`leakage_rerun.py --jobs` on days 4 and 5; days 1–3
are the same either way), `leakage_grid_bootstrap.py`, `leakage_lifetime.py`, and the leakage
grids, which hold only the points near their optima: `leakage_grid_identity.json` the 14 best
of 53 points (loss up to 0.23), two more neighbours of the best point and seeds 22–24 at the
four best; `leakage_grid_pi.json` the 3 best of 23; `leakage_grid_dephase.json` both points;
`leakage_grid_dephase_frame.json` the 6 best of 100. The frame simulator's files
(`leakage_frame_*`), `leakage_recovery.py`, `leakage_roundcond.py`, `leakage_heldout.py` and
parts C, D and X of `leakage_checks.py` take the served snapshot for every qubit on days 4 and
5; they compare simulators on the same inputs.

## The left-out CZ in frame_sim.py

A leaked qubit's CZ should not act. `frame_sim.py` propagates Pauli frames against a reference
sample of the full circuit, so for a leaked operand it applies the CZ with that operand's
noiseless Z value instead of leaving it out. The two differ only when that value is a definite
1, and no CZ operand in the fit jobs' 1- and 8-round memory circuits has a definite Z value (0
of 24,960; `leakage_checks.py`, part E). An exact simulation of the left-out CZ (`exact_sim.py`,
checked against `tableau_ref.py`) reproduces the frame simulator's grid losses and reruns
within Monte Carlo noise (`leakage_fit.json`). Every frame-simulator result uses
`skipped_cz="reference"`, the default, except the `leakage_frame_*` files named after another
mode.

## Reproducibility

Counts and rates decoded from the chip's shots repeat exactly on any machine, and fits to them
to floating-point rounding, except the binomial fits, which stop at an optimizer tolerance and
repeat to about 1e-7. Results of seeded sampling (stim, and `tableau_ref.py`, which draws from
stim) repeat exactly only on the same platform and stim build, and otherwise within their Monte
Carlo noise. The committed results were computed on Windows x86-64 (CPython 3.14.3) and on
Linux; the header of `requirements-lock.txt` lists the packages whose Windows versions differ
from the pinned ones.

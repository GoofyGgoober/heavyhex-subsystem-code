# Follow-up on ibm_kingston

Eight jobs ran on `ibm_kingston` on 14 October 2026: two memory jobs (`exp1-1`, `exp1-2`),
three raw-readout jobs on flag, relay and X-ancilla qubits (`exp3-1` to `exp3-3`), and three
lifetime jobs (`life-1`, `life-z`, `life-t`). They used 86 s of QPU time by IBM's count.
`life-1` ran past its time limit and was cancelled without data, after IBM had charged 10 s;
`life-z` and `life-t` then ran its two parts separately. "Fez" below means the main
experiment on `ibm_fez`. See [Results](#results) and [Checks](#checks). The code imports the
Fez package in `src/` unchanged. Each job's records are in
`runs/ibm_kingston-<UTC date>-<job>/`.

| Module | What it does |
| --- | --- |
| `circuits.py` | the Fez memory circuits without mid-circuit resets, and their detectors |
| `placement.py` | the d=5 footprint and the d=3 patch inside it: cutoffs, the placement score S (with T2), weak parts |
| `submit.py` | the circuits as sent, their cost and caps, the ledger, and each job sent and checked |
| `native.py` | the memory circuits for the model: as sent, with one of the three changes from Fez (twirling, XY4, no resets) undone, and as on Fez |
| `model.py` | the calibration model on the scheduled circuits, and the predictions |
| `decoder.py` | matching with capped edge probabilities, the rule that lowers the cap until every single fault is corrected, correlated matching, distances |
| `leakage_control.py` | the raw-readout jobs: circuits, readout clouds, the two-state model and the split by cause |
| `lifetimes.py` | life-z and life-t: T1 and T2 circuits, their fits, the flips per round, the comparison |
| `analysis.py` | every job from its folder in `runs/` |
| `reproduce.py` | the model variants rebuilt from the saved records ([below](#reconstructing-the-input-variants)) |

## Inspecting the circuits

The environment is the repository's pinned one (`requirements-lock.txt`: qiskit 2.5.2,
qiskit-ibm-runtime 0.49.0, samplomatic 0.21.0). Only `--submit` uses QPU time.

```bash
python -m followup.submit                  # dry run: build, check and estimate job costs
python -m followup.submit --predict        # the model's prediction for the memory jobs
```

Both commands read IBM's latest calibration, so they do not reproduce the stored predictions;
the commands [below](#reconstructing-the-input-variants) work from the saved records. The
stored `exp1-1` prediction used the calibration available before that job, with decoder
weights capped at 0.1.

`--submit` sends one job, and only after the previous job's data passed their checks: shot
counts and readout format as planned, IBM's schedule within 1% of ours, detection rates
consistent with the model, and the idle and lifetime circuits reading as prepared at their
shortest waits. It also caps each job's QPU time and the follow-up's total, and records every
job in `runs/ibm_kingston-followup/ledger.json` before sending it.

## Lifetimes (life-z, life-t)

The lifetime jobs compare IBM's reported T1 and T2 with direct measurements and with the decay
inside the memory schedule. life-z runs the idle test read in Z, twirled like the memory jobs.
life-t measures T1 and Hahn-echo T2 at 512 shots per circuit, untwirled, through SamplerV2
(`--sampler legacy`). IBM charged 5 s for each.

- **The idle test read in Z.** The memory jobs' idle test (each CZ replaced by a wait as long,
  with the same pulses, XY4 and twirl), with the data starting in |0> or |1> instead of |+>
  and read in Z; rounds 1, 2, 4, 8 and 16, 4 twirls × 256 shots. Twirled, its template matches
  the X test's wire for wire, data qubits included, because the twirl's layers carry the data's
  first and last pulses; only the random twirl angles differ.
- **T1 and T2 as IBM measures them**, on all 65 qubits of the footprint at once: X, a wait
  t, readout; and Hahn's echo, SX, t/2, X, t/2, SX†, readout, the T2 IBM reports. Six waits
  each, 20–450 µs and 12–240 µs, from 0.1 to 2 times the footprint's median T1 and T2 on 13
  October (223 and 121 µs); no decoupling but the echo's X; 2 pieces × 256 shots. The echo
  ends in |1> (SX† after the X), where qiskit-experiments' ends in |0>: P(1) =
  (1 + e^(−t/T2))/2, the same T2, with a static detuning refocused. Because all qubits are
  echoed at once, neighbours flip together and the ZZ between them is not refocused: a factor
  cos(ζt/2) per neighbour, 0.99 at 240 µs for ζ/2π = 200 Hz and 0.73 for 1 kHz. If IBM
  measures T2 with the neighbours idle, ours can read low on the worst pairs; T1 is unaffected.

`analysis.life` reads both back from their folders, with the memory jobs' folders for the
in-situ values:

- **T1 and T2** per qubit from P(1) = A e^(−t/T) + B, weighted by the binomial variance at
  the fit, with σ widened by √(χ²/dof) where that is above 1. At the median T1 and T2, σ is
  about 13% of T1 (P(1) falls from 1 to 0) and 30% of T2 (from 1 to ½), √2 times life-1's at
  1024 shots. For T beyond the longest wait, σ is too small on the low side (at T2 of 1–3
  times 240 µs, 10–18% of fits land more than 2σ low), while the rate's σ/T² holds (under 5%).
  So T is compared through its rate: shown with its 1σ range 1/(1/T ± σ/T²), and its distance
  from IBM's as (1/T_IBM − 1/T)/(σ/T²).
- **Excited share at rest**, π = (B − r)/(1 − 2r) from T1's B, with r IBM's readout error; a
  readout that errs δ more often from |1> than from |0> reads it about δ/2 low.
- **Flips per round** of each data qubit: P(1) after n rounds from |s> fitted to
  M + D_s λ^n, so p↓ + p↑ = 1 − λ, the decay per round that Z memory's data see, pulses
  included. Only the sum belongs to the qubit: the twirl's Paulis and XY4 hold a qubit
  prepared in |1> in |0> for half of its time along Z (0.45 with XY4 alone), so the memory's
  frame settles at (M − r)/(1 − 2r) ≈ ½ whatever the qubit's own split, and that ½ checks the
  frame rather than measuring the qubit. A share π of the flips is taken to go |0> → |1>, with
  π from T1's B.
- **In situ**, each data qubit's T1 and T_q (its decay time on the equator, item 7 below)
  from both idle tests at once (`together`):
  ln r = k_X + ln G − t_eq/T_q − t_Z/T1 read in X and Y, ln c = k_Z + ln G − t_Z/T1 −
  t_eq/T_q read in Z, with c = P(1 | 1) − P(1 | 0), G what the model's pulses and readout
  leave, and 1/T1 ≥ 0, 1/T_q ≥ 1/(2 T1) as the model takes them. A data qubit spends about
  2.9 µs of a round on one axis and 2.3 µs on the other, swapped between the tests, so the
  tests pin down the decay per round each memory's data see, t_Z/T1 + t_eq/T_q with that
  memory's t_Z and t_eq (to 6% in Z and 5% in X per qubit with both memory jobs, in
  simulation), but T1 and T_q apart only to about ±50% and ±40%, correlated near −1.
- **The comparison:** `lifetimes.table` prints each qubit's T1 and T2 against IBM's (the
  calibration in force when life-t ended), and π; then each data qubit in situ: its flips
  and |0> → |1> flips per round, the decay per round each memory sees, and T1 and T_q apart;
  then the medians of the ratios, among them each memory's decay in situ over the one at
  IBM's T1 and T2. Read the decay, not the split: an ∞ in the split is a rate on its bound,
  with its range beside it.
- **The model again:** `analysis.exp1(folders, life=folder)`, with life-z's folder, runs the
  model with the data's in-situ T1 and T_q and gives the device/model ratio (O/M) against it,
  for Z memory and X.

## Circuit and decoder details

1. **Twirls.** Each memory job runs 16 randomizations × 256 shots of each memory circuit and
   8 × 256 of each idle circuit: 336 circuits per job. The estimated cost is 26.0 s in one
   batch, or 28.1 s if loaded in batches of 300.
2. **Round length.** A twirled round is 5.81 µs at d=3 and 5.96 µs at d=5 (three slower
   couplers in the d=5 patch). The twirl's one-qubit layers carry the H gates.
3. **Misreads.** Without resets a misread of m_t fires D_t and D_(t+2) only when the
   qubit's next readout has the same role: Z ancillas in Z memory, and the X ancillas that
   are not flags in X memory. Otherwise:

   | Readout misread | Detectors it fires |
   | --- | --- |
   | X ancilla that is also a flag, X memory | D_t, D_(t+1), F_t |
   | flag, X memory | D_(t+1), D_(t+2), F_t |
   | flag in the preparation half-round, X memory | D_0, D_1 |
   | relay, Z memory | F_t, F_(t+1), D_(t+1), D_(t+2) |
   | flag, Z memory | F_t |
   | last round, second to last | D_(T−1) or D_(T−2), and the final detector |

4. **Preparation half-round.** The other-basis gauges measured once before the first round
   enter no detector, but their readouts are carried into round 0's values, so a misread
   there fires round-0 detectors.
5. **Decoder.** The primary decoder (`decoder.PRIMARY`) caps every edge probability at 0.1;
   cap 0.05, uniform weights and correlated matching are reported alongside. Uniform weights
   correct every single fault. At d=3, caps 0.1 and 0.05 miscorrect a few single faults: data
   qubits 54 and 72 read out with 4.5–5.2% error, and for each such fault the decoder picks a
   fault set that the model makes likelier than the single fault. In the model, uniform weights
   raise ε₃,X by about 35% and ε₅,X by about 58% over cap 0.1 (ε₅,X from 5.6% to 8.8%), so
   correcting every single fault does not give the decoder with the lowest total error.
6. **Z at d=5.** The decoding graph has a logical path of 3 edges, two of them pieces of relay
   and flag faults that stim splits into single detectors, as on Fez, while the circuit
   distance is 5. Correlated matching, reported alongside, decodes each such fault's pieces
   together.
7. **T_q.** Between a waited CX's H gates a data qubit sits on the Z axis, where only T1
   acts, so the idle test is fitted as ln r = c + ln G(n) − t_eq(n)/T_q, with t_eq the
   delay time on the equator and G what the model's gates, readout and T1 leave of r. That
   T1 is IBM's, and an error ΔΓ₁ in it moves 1/T_q by about 0.8 ΔΓ₁; with life-z, T_q and
   T1 are fitted together instead (above).
8. **Flag and relay records.** A flag or relay value is F_t = f_t ⊕ (its previous readout):
   a misread fires it twice, and a state that keeps reading 1 fires it only where the state
   starts and ends. Their persistence and two-state fits are compared with the model's own
   shots, not with Fez's or the raw-readout jobs'.
9. **Readout in the model.** A readout flips the record, not the qubit, and the carried
   qubit does not decay during it. Adding T1 over the whole readout raises P_L by at most
   1–5%.

## Checks

Circuits and decoders:

- With two twirls per circuit, Stim's signed flows match the intended logical circuit.
  Without noise no detector fires, the logical outcome is correct, and the idle-test data
  return their prepared value in the measured basis.
- The Z idle test uses the X test's template wire for wire, data qubits included. The T1 and
  T2 circuits read 1 without noise, and their waits hold no pulse but the Hahn echo's X.
- The circuit distance is d in both memories at d = 3 and 5, checked at 2 and 4 rounds; the
  matching graph's logical paths have 3 and 5 edges in X memory but 3 and 3 in Z memory
  (item 6 above).
- At d=3, cap 0.1 miscorrects 2, 3, 5 and 7 single-fault classes in X memory at 1–4 rounds,
  and 1, 1, 2 and 3 in Z memory; cap 0.05 also miscorrects some, and uniform weights none
  (item 5 above).

Recovery of planted values from data the model itself generated, a test of the analysis code:

- The memory analysis recovers T_q on the 25 data qubits with pulls −0.18 ± 0.95; fitted ε and
  Λ are within one standard error of the planted values, and O/M is 0.97–1.04.
- At life-1's 1024 shots, the lifetime fits recover 1/T1 and 1/T2 on the 65 qubits with pulls
  −0.19 ± 0.73 and −0.07 ± 0.88, with one of the 130 estimates beyond two standard errors.
  Taken in T instead of the rate, two T2 above 1.5 times the longest wait came out 3.3σ low
  (1.4σ high in the rate). The in-situ decay per round has pulls −0.06 ± 0.90 in Z and
  +0.23 ± 0.88 in X over the 25 data qubits, and the simulated job passes its checks.
- With both simulated memory jobs, the model at the fitted in-situ T1 and T_q is within 3% of
  the model at the planted values, and O/M is 0.94–1.05.

Twirling and XY4 leave the idle test sensitive only to the sum of the upward and downward
flip rates. The in-situ T1 and T_q are strongly anticorrelated, so the decay per round is
better determined than either lifetime.

## Reconstructing the input variants

`followup/reproduce.py` rebuilds the Kingston model variants from the saved records, one
variant per command, from the repository root in the pinned environment:

```bash
python -m followup.reproduce --variant inputs
python -m followup.reproduce --variant ibm
python -m followup.reproduce --variant measured_tq
python -m followup.reproduce --variant joint
python -m followup.reproduce --variant own_test
python -m followup.reproduce --variant measured_tq_readout
python -m followup.reproduce --variant own_test_readout
```

Each writes `build/followup/<variant>.json` and refuses to overwrite an existing file; pass
`--output build/followup/<name>.json` for a repeat. It makes no connection to IBM. Defaults
are one process, 200,000 simulated shots per circuit and seed 0; the output records `--shots`,
`--seed` and SHA-256 fingerprints of the script and its input records.

| Variant | Inputs and calculation | Compared with |
| --- | --- | --- |
| `inputs` | Re-reads both memory jobs and life-z; fits the X/Y and Z idle decay; bounded joint fit and an unconstrained rate check | `life.json`: `z_idle.summary.joint_on_bound`, `unconstrained_negative`, and per-qubit fits |
| `ibm` | The memory jobs' calibration, with IBM's T2 | `life.json`: `model.variants.ibm` |
| `measured_tq` | Each data qubit's T2 replaced by its pooled X/Y idle-test T_q | `exp1.json`: `predicted_measured_tq` |
| `joint` | Data T1 and T_q replaced by their bounded joint fit | `life.json`: `model.variants.in_situ` |
| `own_test` | X memory at the measured T_q; for Z memory, each data qubit's T1 solved so that the model matches the Z idle test's decay per round, holding the baseline T_q unless that would break T_q ≤ 2T1 | `life.json`: `model.variants.matched` |
| `measured_tq_readout`, `own_test_readout` | The `measured_tq` and `own_test` models with only the readout errors replaced by those of the 18:55 UTC calibration, the one life-z ran under | `life.json`: `model.variants.measured_tq_readout_2155` and `matched_readout_2155` |

Times here are UTC; the run records give them in Israel time (UTC+3), so the 18:55 UTC
calibration is `2155` in field names.

The device's shots are decoded for every variant with priors from the memory jobs'
calibration with the measured T_q, capped at 0.1. By default (`--simulation-priors variant`)
each simulated sample is decoded with its own variant's priors and the same cap, as in
`life.json` and `exp1.json`. `--variant ibm --hardware-priors calibration --output
build/followup/ibm-stored-priors.json` decodes the device with the stored prediction's priors
instead (Table 2, first row), and `--variant measured_tq --decoder uniform --output
build/followup/uniform.json` uses uniform weights on device and simulated shots. No decoder is
fitted to the logical outcomes.

The outputs hold fresh fits, O/M and detection rates in their own format, and `life.json` and
`exp1.json` also hold variants that this script does not rebuild. Twirl sampling and Monte
Carlo noise can change the last digits, so compare central values together with their
statistical errors. `inputs` takes about 47 s and each 200,000-shot variant about 71–73 s,
under 0.75 GB.

### Fixed-decoder comparisons

Table 2's rows below the stored prediction, Fig. 6 and the readout re-prediction in the
write-up decode the device and every simulated variant with one decoder, built from the
pooled measured T_q:

```bash
for variant in ibm joint own_test measured_tq_readout; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m followup.reproduce \
    --variant "$variant" --simulation-priors hardware \
    --output "build/followup/fixed-$variant.json"
done
PYTHONPATH=. python docs/analysis/kingston_checks.py
```

`kingston_checks.py` reads the four `build/followup/fixed-*.json` files, takes the device's
counts and the measured-T_q baseline from `runs/ibm_kingston-followup/analysis/exp1.json`, and
writes `docs/analysis/results/kingston_checks.json`: probabilities by depth, exponential
residuals, fitted rates and source fingerprints. `docs/analysis/idle_checks.py` (Fig. 6,
bottom) also reads `build/followup/fixed-joint.json`. Each simulation takes about 80 s and
0.75 GB; the collector takes seconds.

## Results

The eight jobs went on 14 October between 18:08 and 19:41 UTC: exp3-1 to exp3-3 (4 s each),
exp1-1 and exp1-2 (27 s each), life-1 (cancelled, 10 s charged, no shots), life-z and life-t
(5 s each), 86 s by IBM's count (the ledger, rounding life-z's estimate up, has 87 s). Each
job's record is in `runs/ibm_kingston-2026-10-14-<job>/` (life-t's ends in `-life-t-legacy`);
the ledger and the analyses are in `runs/ibm_kingston-followup/`.

| File | What it holds |
| --- | --- |
| `analysis/exp1.json` | ε and Λ by decoder, the model before the run and at the measured T_q, O/M, detection rates, flag and relay records, Fez refitted on rounds 1–4 |
| `analysis/exp3.json`, `exp3_iq_A.png`, `exp3_iq_B.png` | readout clouds, two- and three-state fits, onsets by cause, the hypotheses |
| `analysis/life.json` | T1 and Hahn T2 against IBM's, the Z idle test's flips per round, the data's in-situ T1 and T_q, the memory jobs' model at them |

`life.json` holds seven T1 values written as `Infinity` (Python's json, not strict JSON):
in-situ fits on the bound 1/T1 = 0. Strict parsers need them mapped to null.

**Memory.** Binomial fits over rounds 1–4, cap 0.1, both jobs pooled (8192 shots a circuit),
Λ = ε₃/ε₅:

| | ε₃ | ε₅ | Λ |
| --- | --- | --- | --- |
| X memory | 6.49 ± 0.27% | 9.47 ± 0.32% | 0.686 ± 0.036 |
| Z memory | 6.84 ± 0.26% | 11.92 ± 0.40% | 0.574 ± 0.029 |
| O/M, prediction before the run (X; Z) | 1.98; 1.51 | 1.62; 1.12 | 1.22; 1.35 |
| O/M, model at the measured T_q (X; Z) | 1.69; 1.25 | 1.69; 1.22 | 1.00; 1.02 |

Here the device is decoded with the measured-T_q priors in both O/M rows; Table 2's first row
decodes it with the stored prediction's own priors.

- Fez fitted the same way gives 12.4% and 22.7% in X memory and 15.9% and 29.5% in Z, so ε
  here is 0.40–0.53 times Fez's. On Kingston's calibration the model puts the effect of the
  circuit changes alone at 0.71 and 0.61 (X) and 0.80 and 1.08 (Z). The chip and the circuit
  changed together, so the measured reduction cannot be split between them.
- With the measured T_q the model's Λ is close to the device's, but its ε is low by factors
  of 1.2–1.7 (1.18–1.34 with uniform weights). IBM's 17:27 UTC calibration gave data qubits
  72 and 12 readout errors of 20.2% and 18.5%; the 18:55 UTC calibration gave 5.1% and 4.4%,
  and the idle tests put both near 5%. With the later readout errors (`readout_sensitivity`
  in `exp1.json`), Λ_X's O/M moves from 1.00 ± 0.07 to 0.92 ± 0.05, a change of 0.08 from the
  inputs alone that the quoted statistical errors do not include. The fitted T_q also moves
  between the jobs.
- T_q in the circuit (XY4, twirled) is 0.74 of IBM's T2 (median); qubits 86, 88, 48 and 10,
  weak by IBM's T2, are not weak in the circuit.
- Flags still persist without resets: lag-1 persistence 0.12 against the model's 0, onset
  1.3–1.5% per round, lifetime 24–30 µs (T_p = −τ/ln(1 − b), as on Fez).

**Lifetimes.** T1 agrees with IBM's: median ratio 0.97 (quartiles 0.75–1.23), with 14 of 65
qubits beyond 3σ. The Hahn T2, echoed on all 65 qubits at once, is 0.31 of IBM's, with an added
Gaussian decay consistent with unrefocused ZZ coupling; no model uses it. In the Z idle test
the data qubits flip with probability 5.3% per round (median), 1.12 times the memory jobs'
model. One physical fit to both idle tests puts 17 of the 25 data qubits on a constraint
boundary; without the constraints, 14 need negative pure-dephasing rates, seven of them by more
than two standard errors. The constrained fit cannot reproduce both decays, and with it the
Z-memory O/M is 1.13 and 1.15. Giving each memory its own idle test's decay per round
(`matched`) brings Z memory's O/M to 1.00 and 1.08; this holds the X-memory model fixed and fits
a separate effective decay for Z memory, so it is not one physical model of both tests. X
memory's own test is the one T_q comes from, so the measured idle decay does not explain X
memory's excess. These O/M use each variant's own decoder priors; Table 2 holds one decoder
fixed. Stabilizer detectors fire 1.2–1.5 times as often as in the model in both memories, and
final detectors about as often as in the model (X memory: 1.07 and 1.10 with the 18:55 UTC
readout errors). The excess is consistent with errors during the rounds, including on the
ancillas, that the idle test cannot see, since it has no active neighbours and no data–ancilla
gates. In Z memory the extra detections cost little ε, so detection rates alone cannot locate X
memory's excess.

**Raw readout.** Ten flag, relay and X-ancilla qubits ran 30 cycles of H, a wait, H, readout and
reset, with the raw IQ signal recorded. Arm A is the reference, arm B adds CZs with each qubit's
usual partners, and arm C drops the resets. Readouts outside the 0 and 1 clouds are
0.02–0.18%, and a cluster test finds a third cluster on one to four of the ten qubits per arm.
In the persistent state the signal reads as 1 about 60% of the time and as 0 about 30%, inside
the 0 and 1 clouds; without a calibrated |2> reference this does not rule out leakage of the
measured qubit. The state starts in 0.26% of cycles without CZs and 0.71% with them (0.09% per
CZ), and in 0.11% without resets; a reset adds about 0.04%. It lives 32–40 µs (Fez: 31 µs).
Qubits that share a CZ partner have correlated fitted occupations (0.13, against 0.009 for
other pairs), consistent with the partner's state or with a frequency shift, for example from a
two-level-system (TLS) defect. The source is still open. `exp3.json` also holds three-state
fits that the write-up does not use.

The write-up's raw-readout numbers come from `docs/analysis/kingston_iq.py`, which refits the
two-state hidden Markov models from the three jobs' `results.npz` files:

```bash
PYTHONPATH=.:src:docs/analysis OPENBLAS_NUM_THREADS=1 python docs/analysis/kingston_iq.py
```

It writes `docs/analysis/results/kingston_iq.json`, with both reference histograms and the
same-shot and same-cycle correlations between qubits. It does not read `exp3.json`, part of
which `exp3_analysis` does not write.

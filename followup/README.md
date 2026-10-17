# Follow-up on ibm_kingston

Memory experiments (Exp 1), flag and relay controls (Exp 3), and qubit-lifetime
measurements (life-z and life-t) on `ibm_kingston`, 14 October 2026. Eight jobs used
86 s of QPU time by IBM's count, including 10 s for the cancelled life-1 job; seven
returned data. See [Results](#results) and the [circuit and analysis checks](READINESS.md).
The code imports the unchanged Fez package in `src/`. Each job's records are in
`runs/ibm_kingston-<UTC date>-<job>/`.

| Module | What it does |
| --- | --- |
| `circuits.py` | run 1's circuits without mid-circuit resets, and their detectors |
| `placement.py` | the d=5 footprint and the d=3 patch inside it: thresholds, the score S, weak parts |
| `submit.py` | the circuits as sent, their cost and caps, the ledger, and each job sent and checked |
| `native.py` | `submit.py`'s circuits with a fix undone, and run 1's design, for the model |
| `model.py` | the calibration model on the scheduled circuits, and the predictions |
| `decoder.py` | matching with capped weights, the single-fault rule, correlated matching, distances |
| `leakage_control.py` | Exp 3: circuits, readout clouds, the two-state model and the split by cause |
| `lifetimes.py` | life-z and life-t: T1 and T2 circuits, their fits, the flips per round, the comparison |
| `analysis.py` | every job from its folder in `runs/` |

## Inspecting the circuits

The environment is the repository's pinned one (`requirements-lock.txt`: qiskit 2.5.2,
qiskit-ibm-runtime 0.49.0, samplomatic 0.21.0). Only `--submit` uses QPU time.

```bash
python -m followup.submit                  # dry run: build, check and estimate job costs
python -m followup.submit --predict        # simulate Exp 1 using the current calibration
```

These commands may access IBM's current calibration; they do not reproduce the stored
forecasts. For the paper, use the saved-input commands below. The `--submit` option sends
new hardware jobs and is not needed for reproduction. Its checks enforce job order,
placement and code consistency, noiseless circuit behaviour, and QPU-time caps.
The stored `exp1-1` prediction uses the calibration available before submission, with
decoder weights capped at 0.1.

The next job goes only once the one before it has passed its checks: shots and bits as
planned, distinct IQ points in kerneled readouts, IBM's schedule within 1% of ours, each
memory circuit's mean detection rate r within what the model's r_M gives with its errors
three times larger, 1 − 2r ≥ (1 − 2r_M)³ (r_M is 9–19%; a readout that doesn't leave its
qubit in |m>, or bits read in the wrong order, give r near 1/2), and T1 and T2 at their
shortest waits, and the idle test read in Z after one round, reading as prepared at least
75% of the time (92–94% in the model). `--despite-checks` overrides that, by hand. If IBM
refuses or fails a job, or its data fail their checks, it goes again with a fallback it
hasn't failed with: `--untwirled` (Exp 1), `--classified` (Exp 3) or `--sampler legacy`
(untwirled only); the experiment's next job then goes the same way, and `exp1-2` and life-z
go as `exp1-1` went. So life-z has no fallback of its own: none after a twirled `exp1-1`,
`--sampler legacy` after an untwirled one. life-t, untwirled, goes with `--sampler legacy`,
the executor its fallback. A job after one that IBM ended, with
no fallback left or none wanted, goes with `--despite-checks`; never while that one is
out.

## Caps

The QPU time of a job is IBM's billing rule, 2.1 s + Σ N (t + 253 µs), and its bound adds
2.1 s for each 300 circuits beyond the first 300 (each twirl a circuit), should IBM load
the job in parts. A job may need at most 30 s, life-z and life-t at most 7 s each (life-1
14 s); the follow-up's jobs at most 90 s, and an optional job may not take them past 80 s,
which keeps 10 s for a job sent again. IBM cancels a job at 1.5 times its bound plus 2 s, at
most its cap (8 s for Exp 3, 30 s for Exp 1, 7 s for life-z and life-t), and while it is
queued it is cancelled if IBM's own
estimate is over that. The ledger, `runs/ibm_kingston-followup/ledger.json`, claims a job
before it is sent and counts it at its limit until IBM has counted it; a send whose outcome
is unknown stays counted until `--fetch` finds the job or, 15 minutes on, releases the
claim. Every job is tagged `heavyhex-followup`, and a ledger that misses one of IBM's jobs
with that tag refuses to go on. Independently of the ledger, a job is refused if IBM's
counter could fall below 285 s (375 s when the 90 s were approved, less them).

## Lifetimes (life-z, life-t)

The lifetime measurements compare IBM's reported T1 and T2 with direct measurements and
decay inside the memory schedule. life-z measures the idle test in Z, with the same
twirling as Exp 1. life-t measures T1 and Hahn-echo T2 at 512 shots per circuit,
untwirled, through the legacy sampler. IBM charged 5 s for each job. The cancelled
combined job, life-1, returned no data.

- **The idle test read in Z.** Exp 1's idle test, each CZ a wait as long, with the same
  pulses, XY4 and twirl, its data starting in |0> or |1> instead of |+> and read in Z;
  rounds 1, 2, 4, 8, 16, 4 twirls × 256 shots. Twirled, its template is the X test's wire
  for wire, the data's included: the twirl's layers carry the data's first and last pulses,
  so only the angles its samplex draws differ.
- **T1 and T2 as IBM measures them**, on all 65 qubits of the footprint at once: X, a wait
  t, readout; and Hahn's echo, SX, t/2, X, t/2, SX†, readout, the T2 IBM reports. Six waits
  each, 20–450 µs and 12–240 µs, from 0.1 to 2 times the footprint's median T1 and T2 on 13
  October (223 and 121 µs); no decoupling but the echo's X; 2 pieces × 256 shots. The echo
  ends in |1> (SX† after the X), where qiskit-experiments' ends in |0>: P(1) =
  (1 + e^(−t/T2))/2, the same T2, a static detuning refocused. Echoed all at once,
  neighbours flip together, so the ZZ between them is not refocused: a factor cos(ζt/2) a
  neighbour, 0.99 at 240 µs for ζ/2π = 200 Hz, 0.73 for 1 kHz. Where IBM measures T2 with
  its neighbours quiet, ours can read low on the worst pairs; T1 is untouched.

`analysis.life` reads both back from their folders, with Exp 1's for the in-situ values:

- **T1 and T2** per qubit from P(1) = A e^(−t/T) + B, weighted by the binomial variance at
  the fit, σ widened by √(χ²/dof) above 1. At the median T1 and T2, σ is about 13% of T1
  (P(1) falls from 1 to 0) and 30% of T2 (from 1 to ½), √2 times life-1's at 1024 shots.
  Beyond the longest wait T's σ is
  too small on the low side (at T2 of 1–3 times 240 µs, 10–18% of fits land over 2σ low),
  while the rate's, σ/T², holds (under 5%). So T is compared in its rate: shown with its
  1σ range 1/(1/T ± σ/T²), and its distance from IBM's as (1/T_IBM − 1/T)/(σ/T²).
- **Excited share at rest**, π = (B − r)/(1 − 2r) from T1's B, r IBM's readout error; a
  readout that errs δ more often from |1> than from |0> reads it about δ/2 low.
- **Flips per round** of each data qubit: P(1) after n rounds from |s> fitted to
  M + D_s λ^n, so p↓ + p↑ = 1 − λ, the decay a round Z memory's data see, pulses included.
  Only the sum is the qubit's: the twirl's Paulis and XY4 hold a qubit prepared in |1> in
  |0> half of its time along Z (0.45 with XY4 alone), so the memory's frame settles at
  (M − r)/(1 − 2r) ≈ ½ whatever the qubit's own split; that ½ is a check, not a finding.
  The flips that go |0> → |1> are taken as π of them, π from T1's B.
- **In situ**, each data qubit's T1 and T_q from both idle tests at once (`together`):
  ln r = k_X + ln G − t_eq/T_q − t_Z/T1 read in X and Y, ln c = k_Z + ln G − t_Z/T1 −
  t_eq/T_q read in Z, with c = P(1 | 1) − P(1 | 0), G what the model's pulses and readout
  leave, and 1/T1 ≥ 0, 1/T_q ≥ 1/(2 T1) as the model takes them. A data qubit spends about
  2.9 µs of a round on one axis and 2.3 µs on the other, swapped between the tests, so the
  tests pin down the decay a round each memory's data see, t_Z/T1 + t_eq/T_q with that
  memory's t_Z and t_eq (to 6% in Z and 5% in X per qubit with both Exp 1 jobs, in
  simulation), but T1 and T_q apart only to about ±50% and ±40%, correlated near −1.
- **The comparison:** `lifetimes.table` prints each qubit's T1 and T2 against IBM's (the
  calibration in force when life-t ended), and π; then each data qubit in situ: its flips
  and |0> → |1> flips a round, the decay a round each memory sees, and T1 and T_q apart;
  then the medians of the ratios, among them each memory's decay in situ over the one at
  IBM's T1 and T2. Read the decay, not the split: an ∞ in the split is a rate on its bound,
  its range beside it.
- **The model again:** `analysis.exp1(folders, life=folder)`, life-z's folder, runs it with
  the data's in-situ T1 and T_q and gives O/M against it, for Z memory and X.

Simulated from the model at planted lifetimes and read back as sent, at life-1's 1024 shots,
the fits return them:
pulls of 1/T1 −0.19 ± 0.73 and 1/T2 −0.07 ± 0.88 over the 65 qubits (in T, two T2 above
1.5 times the longest wait came out 3.3σ low; in the rate, 1.4σ high), and of the decay a
round in situ −0.06 ± 0.90 in Z and +0.23 ± 0.88 in X over the 25 data qubits; the job
passes its checks. With both simulated Exp 1 jobs, the model at the in-situ values is
within 3% of the model at the planted ones, and O/M is 0.94–1.05 for each ε.

## Circuit and decoder details

1. **Twirls.** Each Exp 1 job runs 16 randomizations × 256 shots of each memory circuit and
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
   | flag in the prep half, X memory | D_0, D_1 |
   | relay, Z memory | F_t, F_(t+1), D_(t+1), D_(t+2) |
   | flag, Z memory | F_t |
   | last round, second to last | D_(T−1) or D_(T−2), and the final detector |

4. **Prep half.** Its gauge values enter no detector, but its readouts are carried into
   round 0's values, so a misread there fires round-0 detectors.
5. **Decoder.** Weights capped at 0.1 define the primary decoder (`decoder.PRIMARY`).
   The alternatives are cap 0.05, uniform weights and correlated matching. Uniform
   weights correct every single fault; at d=3 caps 0.1 and 0.05 miscorrect a
   few fault classes, because data qubits 54 and 72 read out with 4.5–5.2% error, and each
   time the decoder takes a fault set the model makes likelier than the single fault. The
   model expects uniform weights to cost ε₃,X about 35% and ε₅,X about 58% against cap 0.1
   (ε₅,X 5.6% → 8.8%). Single-fault correction alone therefore does not determine the
   decoder with the lowest total error under this model.
6. **Z at d=5.** The decoding graph has a logical of 3 edges, two of them remnants of relay
   and flag faults that stim splits into single detectors, as in run 1; the circuit
   distance is 5. Correlated matching, reported beside, decodes each such fault's parts
   together.
7. **T_q.** Between a waited CX's H gates a data qubit sits on the Z axis, where only T1
   acts, so the idle test is fitted as ln r = c + ln G(n) − t_eq(n)/T_q, with t_eq the
   delay time on the equator and G what the model's gates, readout and T1 leave of r. That
   T1 is IBM's, and an error ΔΓ₁ in it moves 1/T_q by about 0.8 ΔΓ₁; with life-z, T_q and
   T1 are fitted together instead (above).
8. **Flag and relay records.** A flag or relay value is F_t = f_t ⊕ (its previous readout):
   a misread fires it twice and a state that keeps reading 1 only at its ends. Their
   re-firing and two-state fits are compared with the model's own shots, not with run 1's
   or Exp 3's.
9. **Exp 3 with 0/1 readout.** Arm C counts changes, so it can't see a state that reads a
   steady 1: for such a qubit only the CZ part separates, and arm C gives bounds on the
   reset and readout parts.
10. **Readout in the model.** A readout flips the record, not the qubit, and the carried
    qubit doesn't decay during it. Adding T1 over the whole readout raises P_L by at most
    1–5%.

## Reconstruct the paper's input comparisons

From the repository root in the pinned environment, run one variant per command:

```bash
python -m followup.reproduce --variant inputs
python -m followup.reproduce --variant ibm
python -m followup.reproduce --variant measured_tq
python -m followup.reproduce --variant joint
python -m followup.reproduce --variant own_test
python -m followup.reproduce --variant measured_tq_readout
python -m followup.reproduce --variant own_test_readout
```

Outputs go to `build/followup/<variant>.json`, never `runs/`. Existing outputs are not
overwritten; choose a new `--output build/followup/<name>.json` for a repeat. The driver
does not initialize an IBM service or submit jobs. It uses one process, 200,000 simulated
shots per circuit and seed 0 by default. `--shots` and `--seed` are recorded in the output,
along with SHA256 fingerprints of its source and the input records.

| Variant | Inputs and calculation | Archived comparison |
| --- | --- | --- |
| `inputs` | Re-read both Exp 1 jobs and life-z, fit X/Y and Z idle decay, bounded joint fit and unconstrained rate check | `life.json`: `z_idle.summary.joint_on_bound`, `unconstrained_negative`, and per-qubit fits |
| `ibm` | Exp 1's original calibration and IBM T2 | Original-prior archive: `life.json`, `model.variants.ibm` |
| `measured_tq` | Replace each data qubit's T2 by the pooled X/Y idle-test T_q | `exp1.json`: `predicted_measured_tq` |
| `joint` | Replace data T1 and T_q by their bounded joint fit | Original-prior archive: `life.json`, `model.variants.in_situ` |
| `own_test` | X memory stays at measured T_q; Z memory solves its own decay per round for T1 at the baseline model's effective T_q (already capped at twice IBM's T1), respecting the physical cap again after the change | Original-prior archive: `life.json`, `model.variants.matched` |
| `measured_tq_readout`, `own_test_readout` | The corresponding model with only readout replaced by life-z's 21:55 calibration | App. F; `life.json`: `model.variants.measured_tq_readout_2155` and `matched_readout_2155` |

The hardware decoder defaults to original calibration plus measured T_q, with cap 0.1,
held fixed across the simulated variants. Each simulated sample uses its own variant's
priors with the same cap. The Appendix F stored-prior check instead uses
`--variant ibm --hardware-priors calibration --output build/followup/ibm-stored-priors.json`.
The uniform-weight comparison uses
`--variant measured_tq --decoder uniform --output build/followup/uniform.json` on both
hardware and simulated shots. The decoder never learns from the logical outcomes.

The outputs contain fresh fits, O/M and detector rates, not a rewritten copy of the archival
summary format. The archived summaries include additional exploratory variants not selected
here. This driver reconstructs the methods and checks numerical agreement; it is not a
bit-for-bit regenerator of the historical summary files. Representative twirls and Monte
Carlo samples can change the last digits. Keep the archived inputs and results unchanged,
and compare both central values and their conditional statistical errors.

On this PC the input reconstruction took about 47 s and the tested 200,000-shot variants
about 71–73 s, with peak RSS below 0.75 GB. These are observed runtimes, not bounds.

### Fixed-decoder comparisons used in the paper

The paper's Kingston input columns hold the pooled measured-T_q decoder fixed on both
hardware and every simulated variant. The original pre-job forecast, which used its own
calibration priors, is a separate archived control. Reconstruct the fixed-decoder columns
and the readout sensitivity locally with one process:

```bash
for variant in ibm joint own_test measured_tq_readout; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m followup.reproduce \
    --variant "$variant" --simulation-priors hardware \
    --output "build/followup/fixed-$variant.json"
done
PYTHONPATH=. python docs/analysis/kingston_checks.py
```

The output guard refuses existing files; use a fresh checkout or preserve the earlier
outputs and choose new paths for a repeat. The collector's four `fixed-*.json` inputs are
the outputs named above. It writes `docs/analysis/results/kingston_checks.json`, including
fixed-depth probabilities, exponential residuals, fitted rates and source fingerprints.
Each simulation takes about 80 seconds and 0.75 GB here; the collector takes seconds.
The measured-T_q baseline and device counts are read from the immutable archive.
The default `--simulation-priors variant` retains the original archival reconstruction
policy described above; it must not be confused with this controlled comparison.

## Results

The eight jobs went on 14 October between 18:08 and 19:41 UTC: exp3-1 to exp3-3 (4 s each),
exp1-1 and exp1-2 (27 s each), life-1 (cancelled by IBM, 10 s charged, no shots), life-z and
life-t (5 s each), 86 s by IBM's count (the ledger, rounding life-z's estimate up, has 87 s).
Each job's record is in `runs/ibm_kingston-2026-10-14-<job>/` (life-t's ends in
`-life-t-legacy`); the ledger and the analyses are in `runs/ibm_kingston-followup/`:

| File | What it holds |
| --- | --- |
| `analysis/exp1.json` | ε and Λ by decoder, the model before the run and at the measured T_q, O/M, detection rates, flag and relay records, run 1 refitted on rounds 1–4 |
| `analysis/exp3.json`, `exp3_iq_A.png`, `exp3_iq_B.png` | readout clouds, two- and three-state fits, onsets by cause, the hypotheses |
| `analysis/life.json` | T1 and Hahn T2 against IBM's, the Z idle test's flips per round, the data's in-situ T1 and T_q, Exp 1's model at them |

`life.json` holds seven T1 values written as `Infinity` (Python's json, not strict JSON):
in-situ fits on the bound 1/T1 = 0. Strict parsers need them mapped to null.

**Exp 1.** Binomial fits over rounds 1–4, cap 0.1, both jobs pooled (8192 shots a circuit),
Λ = ε₃/ε₅:

| | ε₃ | ε₅ | Λ |
| --- | --- | --- | --- |
| X memory | 6.49 ± 0.27% | 9.47 ± 0.32% | 0.686 ± 0.036 |
| Z memory | 6.84 ± 0.26% | 11.92 ± 0.40% | 0.574 ± 0.029 |
| O/M, prediction before the run (X; Z) | 1.98; 1.51 | 1.62; 1.12 | 1.22; 1.35 |
| O/M, model at the measured T_q (X; Z) | 1.69; 1.25 | 1.69; 1.22 | 1.00; 1.02 |

- Run 1 refitted the same way gives 12.4% and 22.7% (X), 15.9% and 29.5% (Z): ε here is
  0.40–0.53 of run 1's. On Kingston's calibration the model puts the design change alone at
  0.71 and 0.61 (X), 0.80 and 1.08 (Z). Chip and circuit changed together, so the hardware
  reduction cannot be assigned to individual changes.
- The model at the measured T_q puts Λ's central value near the device's and misses ε by
  1.2–1.7 (1.18–1.34 with uniform weights). These are diagnostics using measured decay,
  distinct from the stored calibration prediction. IBM's 20:27 readout errors for
  data 72 and 12 (20.2% and 18.5%) did not hold
  during the jobs (21:55: 5.1% and 4.4%, and the idle tests put them near 5%); the model with
  21:55's readout is in `readout_sensitivity`. With it, Λ_X's O/M changes from
  1.00 ± 0.07 to 0.92 ± 0.05: a shift of 0.08 from changed inputs, separate from the
  conditional statistical errors. The fitted T_q also moves between jobs.
- T_q in the circuit (XY4, twirled) is 0.74 of IBM's T2 (median); qubits 86, 88, 48 and 10,
  weak by IBM's T2, are not weak in the circuit.
- Flags still persist without resets: lag-1 re-firing 0.12 against the model's 0, onset
  1.3–1.5% a round, lifetime 24–30 µs (T_p = −τ/ln(1 − b), as for run 1).

**Lifetimes.** T1 agrees with IBM's: median ratio 0.97 (quartiles 0.75–1.23), 14 of 65 beyond
3σ. Hahn T2, echoed on all 65 qubits at once, is 0.31 of IBM's, with an added Gaussian
decay consistent with unrefocused ZZ coupling. No model uses it. The Z idle test's data lose 5.3% a
round (median), 1.12 times the model. A common physical fit to both idle tests puts 17 of
25 data qubits on a boundary; unconstrained, 14 have negative pure-dephasing rates, seven
beyond two standard errors. The constrained fit cannot reproduce both decays. Its Z-memory
O/M is 1.13 and 1.15. Each memory at its own test's decay per round
(`matched`) brings Z memory's O/M to 1.00 and 1.08; X memory's own test is the one T_q comes
from, so that measured idle decay does not account for its excess. `matched` holds the
X-memory model fixed and solves a separate Z-memory effective decay, not one physical
model that fits both tests. Stabilizer detectors fire
1.2–1.5 times the
model's in both memories and final detectors about as often (X: 1.07 and 1.10 with 21:55's
readout). This is consistent with errors during the rounds, including ancilla-side
processes, but the idle test lacks active neighbours and data–ancilla gates. In Z memory
the extra detections cost little ε, so the detection rates alone do not locate X's excess.

**Exp 3.** Readouts outside the 0 and 1 clouds are 0.02–0.18%, and the cluster test finds a
third cluster on one to four of the ten qubits per arm. The persistent state
reads as 1 about 60% and 0 about 30% of the time, on the computational clouds. There is no
calibrated |2> reference, so this does not exclude leakage of the measured qubit.
Onset a cycle: 0.26% without CZs, 0.71% with them (0.09%
per CZ), 0.11% without reset; a reset adds about 0.04%. Lifetime 32–40 µs (run 1: 31 µs).
Qubits that share a CZ partner have correlated fitted occupations (0.13 against 0.009), which
is consistent with a partner's state or a frequency shift, including a TLS. The source
remains unresolved. The historical three-state estimates are not used in the paper.

Reconstruct the paper's retained raw-IQ summaries directly from the three NPZ archives:

```bash
PYTHONPATH=.:src:docs/analysis OPENBLAS_NUM_THREADS=1 python docs/analysis/kingston_iq.py
```

This refits the two-state HMMs and writes `docs/analysis/results/kingston_iq.json`, including
both reference histograms and same-shot/cycle posterior correlations. It does not read
the extended `exp3.json` summaries, which `exp3_analysis` did not itself generate.
The legacy archive is preserved unchanged. Neither a two-state fit nor a one-dimensional
IQ projection identifies a physical transmon level.

# Heavy-hex subsystem code on IBM Fez

Code and data for *Calibration-based predictions versus measured logical errors for heavy-hex
memories on IBM Fez* (Nadav Abramson). The heavy-hex subsystem code
([Chamberland et al. 2020](https://arxiv.org/abs/1907.09528), as run by [Sundaresan et al.
2023](https://arxiv.org/abs/2203.07205)) ran at distances 3 and 5 side by side on
`ibm_fez` on five days. The stored calibration-based forecasts underpredict the early-depth
failure probabilities in all 160 comparisons. Averaging placements within each day and then
five days gives mean gaps of 3.8–7.6 percentage points in X memory and 8.8–16.6 in Z memory.
Approximate simultaneous intervals cover sixteen depth/basis/distance cells, conditional on
fixed forecasts and sampled placements; fifteen exclude zero. These are not 160 independent
experiments or prediction intervals for future jobs. Secondary binomial rate fits give X
O/M of 1.27 and 1.19 at distances 3 and 5, and about two in Z memory. Distance 5 is worse than
3 on these implementations; no threshold is claimed.

## Data and reproducibility

The fifteen jobs ran on 4–8 October 2026 by Israel time (3–8 October UTC; the folder names
use the Israel date), and their folders, `runs/ibm_fez-*`, are final. The code that prepared
and analysed them is tag v0.2.0: every job records the git hash of `src/` (tree `a233d40`) and
of `pyproject.toml`, and both are unchanged in this release, which a test checks.
[`docs/commit-map.tsv`](docs/commit-map.tsv) maps the commit IDs recorded by the jobs to
this repository's history. The paper's post-run analyses and reproduction commands are in
[`docs/analysis/`](docs/analysis/README.md).

`src/` and `pyproject.toml` are frozen with the run; release tags identify repository
versions. `runs/combined.json` is the output of `heavyhex experiment combine` from that
same code. `requirements-lock.txt` pins the reproduction environment.

## Install

Python 3.14. The pinned environment is the one that analysed the runs:

```bash
python -m venv .venv            # then activate it
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip install -r requirements-dev.txt   # tests, ruff and pyright
```

## Check

```bash
pytest                            # about 10 min; the re-analysis below is skipped
pytest --reproduce                # also re-analyses the 15 jobs (about 15 min, 4 workers)
ruff check . && ruff format --check . && pyright
```

Each re-analysis worker takes 1–2 GB; on a small machine set `HEAVYHEX_WORKERS=1`.
`pytest --reproduce --strict-floats` asks for every `analysis.json` and `runs/combined.json`
back byte for byte, apart from line endings and the record of which commit re-analysed them
(`code.analyzed`). That holds on the environment of `requirements-lock.txt` (Linux x86-64).
Elsewhere the results decoded from the chip's shots still come back exactly and the fits to
them to within 1e-11, but the predictions, which stim samples with fixed seeds, agree only
within their simulator noise: the C math library can change a noise probability in its last
bit, which reshuffles the samples.

## Reproduce the paper

The paper cites this repository at the tag `paper`.

| What | Command | Output |
| --- | --- | --- |
| Each job's analysis and the pooled record | `python scripts/reproduce.py analyze`, then `python scripts/reproduce.py compare` | `build/reproduce/` |
| Main Fez comparison | `python docs/analysis/stored_priors.py`, then `python docs/analysis/early_depths.py` | `docs/analysis/results/stored_priors.json`, `early_depths.json` |
| Other analyses and Kingston comparisons | Commands and source-to-result map in [`docs/analysis/README.md`](docs/analysis/README.md) | `docs/analysis/results/` |
| Figures | `python paper/figures/make_figures.py`; Fig. gadget: `latexmk -pdf gadget.tex` in `paper/figures/` | `paper/figures/*.pdf` |
| The paper | `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` | `paper/main.pdf` |
| The arXiv upload | `paper/make_arxiv.sh` | `paper/arxiv.tar.gz` |

`heavyhex experiment analyze --run runs/<folder>` and `heavyhex experiment combine` rewrite
the stored files in `runs/`; `scripts/reproduce.py` works on copies.

## The code

```bash
heavyhex info                      # the d=3 patch's operators
heavyhex --distance 5 circuit      # draw the d=5 flagged memory circuit
heavyhex --help
```

Data qubit ids are 0-based; Sundaresan et al.'s Q label is id + 1. d = 3 takes 23 qubits and
d = 5 takes 65, and both fit on Fez at once ([layout](docs/figures/heavyhex-blueprint.png)).

## The hardware runs

Each day `heavyhex calibrate` read IBM's calibration and planned the day's repetitions (one
per usable placement of the d=5 patch, at most two); `heavyhex experiment prepare` placed the
patches, translated the circuits and stored the predictions at f = 0 and f = 1;
`heavyhex experiment submit --run runs/<folder>` sent each job; `analyze` and `combine`
followed. Each job ran X and Z memory at 1–8 rounds and an idle test, 2,500 shots per memory
circuit. `submit` uses QPU time and asks first, and the code caps it: at most 50 s a job, and
300 s and 20 jobs in all (fifteen jobs used 225 s). Day 1 was sent by hand and days 2–5 by
`scripts/daily_run.sh`. That script submits hardware jobs; use the reproduction commands
above to analyse the saved data without QPU time.

## Follow-up on ibm_kingston (14 October 2026)

Eight follow-up jobs were submitted to `ibm_kingston`; seven returned data. IBM's reported
charges total 86 s, including 10 s for the cancelled job. The budget ledger totals 87 s:
it counts 6 s for `life-z` by rounding its estimate up, while IBM reported a 5 s charge.
The code in [`followup/`](followup/README.md) imports `src/` unchanged. Records are in
`runs/ibm_kingston-2026-10-14-*/`; the ledger and analyses are in `runs/ibm_kingston-followup/`.

- **Exp 1**, memory circuits with no mid-circuit resets, XY4
  decoupling and Pauli twirling, the d=3 patch inside the d=5 one, placed by T2; rounds 1–4,
  8192 shots a circuit. ε per round (binomial fit) is 6.49 ± 0.27% and 9.47 ± 0.32% in X
  memory, 6.84 ± 0.26% and 11.92 ± 0.40% in Z, 0.40–0.53 of run 1's refitted the same way;
  Λ_X = 0.686 ± 0.036 and Λ_Z = 0.574 ± 0.029. The model at the T_q measured in the circuit
  gives Λ central values near the device's (O/M 1.00 ± 0.07 and 1.02 ± 0.06), but these
  conditional errors omit input uncertainty. A common error cancels in the ratio; ε remains
  underpredicted (O/M 1.69 in X, 1.22–1.25 in Z).
- **Lifetimes** (life-z, life-t): T1 measured as IBM measures it agrees with IBM's (median
  ratio 0.97). In an idle test read in Z the data lose 5.3% a round, 1.12 times the model;
  a separate basis-specific fit reduces the fitted Z-memory discrepancy and not X memory's.
  A stationary shared-parameter fit reaches a boundary on 17 of 25 data qubits. The tests
  ran at different times and the rates are poorly separated, so drift can contribute to
  its failure to reproduce both decays.
- **Exp 3**, a leakage control on the flags, relays and X ancillas: the state that persists on
  them reads as 0 or 1 on the computational clouds, but without a calibrated |2> response
  this does not exclude leakage of the measured qubit. Readouts outside both reference
  clouds account for 0.02–0.18%; the separate cluster test accepts a third cluster on one
  to four qubits per arm. Fitted onset is 0.26% a cycle without CZs and 0.71% with them.
  Qubits sharing a CZ partner show more strongly correlated inferred state occupancy.

## Layout

```
src/heavyhex/        the package: code, patches and placement, circuits, simulation, the run
tests/               pytest; test_provenance.py and test_reproduce.py check runs/ against the code
scripts/             reproduce.py (re-analysis in a scratch folder), daily_run.sh (the run's record)
runs/                the fifteen jobs: calibration, circuits, prediction, shots, analysis;
                     the Kingston jobs, their ledger and analyses (ibm_kingston-*)
followup/            the Kingston follow-up: circuits, model, decoder, submission, analysis
docs/analysis/       the paper analyses, their results, and reproduction commands
docs/figures/        the layout drawings and the calibration they were drawn from
docs/                calibration and power diagnostics, commit-map.tsv
paper/               the paper (LaTeX), figures/make_figures.py and make_arxiv.sh
```

## Citation and license

MIT, see [LICENSE](LICENSE). [CITATION.cff](CITATION.cff) says how to cite it.

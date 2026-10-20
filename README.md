# Testing quantum memories on IBM hardware

How well do IBM's calibration data predict the errors of a logical quantum memory, and does a
heavy-hex code get better from distance 3 to 5? This repository holds the code, the raw
hardware records and the analysis of a project that tested both on IBM's Heron processors.

**Write-up:** [paper/main.pdf](paper/main.pdf) (8 pages). A longer version with every check is
in [paper/full/main.pdf](paper/full/main.pdf).

- The heavy-hex subsystem code ([Chamberland et al. 2020](https://arxiv.org/abs/1907.09528), as
  run by [Sundaresan et al. 2023](https://arxiv.org/abs/2203.07205)) at distances 3 and 5, X and
  Z memory: fifteen jobs over five days on `ibm_fez`, and a follow-up on `ibm_kingston`.
- A noise model built from IBM's calibration, with its predictions stored before each job,
  underestimates the logical error: by about 1.2–1.3 in X memory and up to 2 in Z memory on Fez.
- Distance 5 did worse than distance 3 on both chips.
- On Kingston, a circuit without mid-circuit resets, with XY4 decoupling and twirling, went with
  roughly half the error per round (the chip and placement changed too).
- Flag and relay qubits show a persistent state that the noise model lacks.

## Data and reproducibility

The fifteen jobs ran on 4–8 October 2026 by Israel time (3–8 October UTC; the folder names
use the Israel date), and their folders, `runs/ibm_fez-*`, are final. The code that prepared
and analysed them is tag v0.2.0: every job records the git hash of `src/` (tree `a233d40`) and
of `pyproject.toml`, and both are unchanged in this release, which a test checks.
[`docs/commit-map.tsv`](docs/commit-map.tsv) maps the commit IDs recorded by the jobs to
this repository's history. The post-run analyses and reproduction commands are in
[`docs/analysis/`](docs/analysis/README.md); its figure and section numbers refer to the full version.

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

## Reproduce the write-up

The write-up cites this repository at the tag `paper`.

| What | Command | Output |
| --- | --- | --- |
| Each job's analysis and the pooled record | `python scripts/reproduce.py analyze`, then `python scripts/reproduce.py compare` | `build/reproduce/` |
| Main Fez comparison | `python docs/analysis/stored_priors.py`, then `python docs/analysis/early_depths.py` | `docs/analysis/results/stored_priors.json`, `early_depths.json` |
| Other analyses and Kingston comparisons | Commands and source-to-result map in [`docs/analysis/README.md`](docs/analysis/README.md) | `docs/analysis/results/` |
| Figures | `python paper/figures/make_figures.py`; Fig. gadget: `latexmk -pdf gadget.tex` in `paper/figures/` | `paper/figures/*.pdf` |
| The write-up | `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` | `paper/main.pdf` |
| The full version | the same commands in `paper/full/` | `paper/full/main.pdf` |
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

Eight jobs ran on `ibm_kingston` for 86 s of QPU; IBM cancelled one, which returned no data.
The code is in [`followup/`](followup/README.md) and imports `src/` unchanged. The records are
in `runs/ibm_kingston-2026-10-14-*/`, the budget ledger and analyses in
`runs/ibm_kingston-followup/`.

- **Memory** without mid-circuit resets, with XY4 decoupling and twirling, the d=3 patch inside
  the d=5 one, rounds 1–4, 8192 shots a circuit: ε per round 6.49 ± 0.27% and 9.47 ± 0.32% in X
  memory, 6.84 ± 0.26% and 11.92 ± 0.40% in Z, 0.40–0.53 of Fez's fitted the same way;
  Λ_X = 0.69 ± 0.04 and Λ_Z = 0.57 ± 0.03. With the measured idle decay the model gets Λ about
  right but still underestimates ε, by 1.69 in X and 1.22–1.25 in Z.
- **Lifetimes:** T1 measured as IBM measures it agrees with IBM's (median ratio 0.97). In the
  circuit, the data qubits lose 5.3% a round in Z, 1.12 times the model.
- **Raw readout signals** on the flags, relays and X ancillas: the persistent state reads mostly
  as ordinary 0 and 1, with 0.02–0.18% of readouts outside both reference clouds. Its onset
  rises from 0.26% to 0.71% a cycle when the CZs are added.

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
paper/               the write-up (LaTeX), figures/make_figures.py and make_arxiv.sh
paper/full/          the full-length version
```

## Citation and license

MIT, see [LICENSE](LICENSE). [CITATION.cff](CITATION.cff) says how to cite it.

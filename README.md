# Heavy-hex subsystem code

Measures the error-suppression factor Λ = p_L(3) / p_L(5) of the heavy-hex
subsystem code, the ratio of logical error per round at d=3 and d=5, by running both side
by side on `ibm_fez`, and compares it with a simulation built from IBM's
calibration.

The code is the heavy-hex subsystem code of [Chamberland et al.
2020](https://arxiv.org/abs/1907.09528), as run by [Sundaresan et al.
2023](https://arxiv.org/abs/2203.07205). d=3 takes 23 qubits and d=5 takes 65,
and both fit on Fez at once ([layout](docs/figures/heavyhex-blueprint.png)).

```bash
pip install -e '.[sim,matching]'
heavyhex info                          # the patch's operators
heavyhex --distance 5 circuit          # draw its flagged memory circuit
heavyhex --help
```

Data qubit ids are 0-based; the paper's Q label is id + 1.

The hardware run is X and Z memory at 1-8 rounds with both patches at once, and
an idle test, read in X and in Y, that measures f, how much of the calibrated
dephasing the decoupling pulses leave. IBM runs a job shot by shot across all its
circuits, so each circuit's first and second halves of shots are the job's first
and second halves in time, and comparing them shows drift during the job. Each run
keeps its calibration, circuits, prediction and shots in
`runs/<backend>-<date>`:

```bash
heavyhex experiment prepare                      # place, translate, freeze the prediction
heavyhex experiment rehearse --run runs/ibm_fez-<date>   # simulated shots, into rehearsal.npz
heavyhex experiment analyze --rehearsal --run runs/ibm_fez-<date>   # try the analysis on them
heavyhex experiment submit --run runs/ibm_fez-<date>     # uses the QPU, asks first
heavyhex experiment analyze --run runs/ibm_fez-<date>
```

## Run day

This needs `pip install -e '.[sim,matching,hardware]'` and saved IBM
credentials. Do it all on one day: submit refuses a run prepared on another.
`<folder>` is the one prepare prints, `ibm_fez-<date>`.

1. `heavyhex calibrate` pulls today's Fez numbers and picks where the patches
   go (read-only).
2. `heavyhex experiment prepare` places the patches, translates the circuits and
   freezes the prediction.
3. Redraw the layout with `python docs/figures/draw_blueprint.py`, and the
   paper's Figure 1 with `python docs/figures/figure1.py --out docs/figures/figure1.png`.
4. Commit and push `runs/<folder>`, with the new calibration and layout in
   `docs/figures/` (submit refuses a run that isn't committed and pushed). Note
   the commit hash (`git rev-parse HEAD`) and the UTC time (`date -u`) for the paper.
5. `heavyhex experiment submit --run runs/<folder>` uses the QPU, and asks first.
   The job may use at most 50 s of QPU time (`QPU_SECONDS_LIMIT` in run.py): submit
   won't send a job estimated to need more, cancels it before it runs if IBM's own
   estimate, read as it's queued, is higher, and IBM cancels it if it uses more. The
   run day's job is about 27 s by IBM's rule. job.json claims the folder before
   anything is sent, so a folder is never sent twice, and once one job has gone
   (`QPU_JOBS_LIMIT`) no other folder is sent either.
   It saves the shots and IBM's calibration as the job finished; if the wait is
   cut short, `analyze` fetches both later. It refuses a run that already has a job.
6. Commit the shots (`shots.npz`, `job.json`) and `calibration_after.json`.
7. `heavyhex experiment analyze --run runs/<folder>`, then commit `analysis.json`.
8. Regenerate Figure 3 with `python docs/figures/figure3.py --calibration
   runs/<folder>/calibration.json --out docs/figures/figure3.json`.
9. Publish a release on GitHub (a release, not just a tag), so Zenodo mints a DOI.

The paper's predictions come from the run folder `runs/ibm_fez-2026-10-03`,
prepared from IBM's calibration of 3 October 2026, 17:05 UTC.

## License

MIT, see [LICENSE](LICENSE). [CITATION.cff](CITATION.cff) says how to cite it.

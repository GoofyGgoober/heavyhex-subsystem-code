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

The hardware run is X and Z memory at 1-8 rounds, and an idle test, read in X
and in Y, that measures f, how much of the calibrated dephasing the decoupling
pulses leave. It runs on five days. Each day has one rep per clean place for the
d=5 patch (at most two), best first. The d=3 patch goes on a clean place of its
own each rep: beside the d=5 patch in the same job when one fits without touching
it, otherwise in a job of its own straight after. Every job is 2,500 shots per
memory circuit. IBM runs a job shot by shot across all its circuits, so each
circuit's first and second halves of shots are the job's first and second halves
in time, and comparing them shows drift during the job. Each job keeps its
calibration, circuits, prediction and shots in `runs/<backend>-<date>-r<rep>`,
with `-d5` or `-d3` for a patch that runs alone:

```bash
heavyhex calibrate                                   # today's numbers and the day's reps
heavyhex experiment prepare                          # a folder per job, prediction frozen
heavyhex experiment rehearse --run runs/<folder>     # simulated shots, into rehearsal.npz
heavyhex experiment analyze --rehearsal --run runs/<folder>   # try the analysis on them
heavyhex experiment submit --run runs/<folder>       # uses the QPU, asks first
heavyhex experiment analyze --run runs/<folder>
heavyhex experiment combine                          # Λ per rep, and over every rep
```

## Run day

This needs `pip install -e '.[sim,matching,hardware]'` and saved IBM
credentials. Do it all on one day: submit refuses a job prepared on another.

1. `heavyhex calibrate` pulls today's Fez numbers and lists the day's reps
   (read-only). With no clean place for the d=5 patch there is no rep; the day is
   skipped.
2. `heavyhex experiment prepare` prepares a folder for each job: it places the
   patches, translates the circuits and freezes the prediction.
3. Redraw the layout with `python docs/figures/draw_blueprint.py`, and the
   paper's Figure 1 with `python docs/figures/figure1.py --out docs/figures/figure1.png`.
4. Commit and push the day's `runs/` folders, with the new calibration and layout in
   `docs/figures/` (submit refuses a job that isn't committed and pushed). Note
   the commit hash (`git rev-parse HEAD`) and the UTC time (`date -u`) for the paper.
5. `heavyhex experiment submit --run runs/<folder>` for each job, in the order
   prepare lists them; each uses the QPU and asks first. A job may use at most 50 s
   of QPU time (`QPU_SECONDS_LIMIT` in run.py), and every job together at most
   300 s (`QPU_TOTAL_SECONDS`): each job's limit is cut to what the total has left,
   counting a job IBM hasn't finished counting at its whole limit. Submit won't send
   a job estimated to need more than its limit, cancels it before it runs if IBM's
   own estimate, read as it's queued, is higher, and IBM cancels it if it uses more.
   A job is about 15 s by IBM's rule. job.json claims the folder before anything is
   sent, so a folder is never sent twice. It saves the shots and IBM's calibration as
   the job finished; if the wait is cut short, `analyze` fetches both later.
6. Commit the shots (`shots.npz`, `job.json`) and `calibration_after.json`.
7. `heavyhex experiment analyze --run runs/<folder>` for each job, then
   `heavyhex experiment combine`, and commit `analysis.json` and `runs/combined.json`.
8. Regenerate Figure 3 with `python docs/figures/figure3.py --calibration
   runs/<folder>/calibration.json --out docs/figures/figure3.json`.
9. Publish a release on GitHub (a release, not just a tag), so Zenodo mints a DOI.

## The run, 4–8 October 2026

All five days ran: fifteen jobs, 225 s of QPU time, every day's folders committed and
pushed before its first job (`runs/`, with `runs/combined.json` pooling the ten reps).
Day 1 was sent by hand; days 2–5 were prepared and sent unattended by
`scripts/daily_run.sh`, under the author's written approval and the same caps. Only
day 1's design was released (v0.2.0, archived by Zenodo four minutes before the first
job); the later days rest on the git history and GitHub's push log.

`docs/diagnosis/` holds the exploratory analysis, done after the run, of why the
simulation from IBM's calibration under-predicts the chip's errors (mostly leakage),
with a leakage-aware simulator; see its README.

## History

Two force-pushes rewrote `main` before any run data: on 24 September 2026, replacing
three earlier commits with a new line of work, and on 2 October 2026, rewording commit
messages only (the files are identical). From 3 October the history matches GitHub's
push events.

## License

MIT, see [LICENSE](LICENSE). [CITATION.cff](CITATION.cff) says how to cite it.

# Heavy-hex subsystem code

Measures the error-suppression factor Λ = p_L(3) / p_L(5) of the heavy-hex
subsystem code, the ratio of logical error per round at d=3 and d=5, by running both side
by side on `ibm_fez`, and compares it with a simulation built from IBM's
calibration.

The code is the heavy-hex subsystem code of [Chamberland et al.
2020](https://arxiv.org/abs/1907.09528), as run by [Sundaresan et al.
2023](https://arxiv.org/abs/2203.07205). d=3 takes 23 qubits and d=5 takes 65,
and both fit on Fez at once ([layout](docs/figures/heavyhex-blueprint.png)).

So far there are both patches, their flagged circuits, and Aer runs decoded with
a lookup table (d=3) or [MWPM](docs/mwpm-decoder.md). Nothing has run on the QPU yet.

```bash
pip install -e '.[sim,matching]'
heavyhex info
heavyhex --distance 5 run --decoder mwpm --rounds 3 --error X0
heavyhex --help
```

Data qubit ids are 0-based; the paper's Q label is id + 1.

Before anything goes to hardware, run `heavyhex calibrate` to pull today's Fez
numbers and pick where the patches go, then redraw with
`python docs/figures/draw_blueprint.py`. Hardware runs pull fresh numbers
themselves if the last ones aren't from today.

The hardware run is X and Z memory at 1-8 rounds, both patches at once, plus an
idle test that measures how much dephasing the decoupling pulses leave. Each run
keeps its calibration, circuits, prediction and shots in `runs/<backend>-<date>`:

```bash
heavyhex experiment prepare                      # place, translate, freeze the prediction
heavyhex experiment rehearse --run runs/ibm_fez-<date>   # simulated shots, to try the analysis
heavyhex experiment submit --run runs/ibm_fez-<date>     # uses the QPU, asks first
heavyhex experiment analyze --run runs/ibm_fez-<date>
```

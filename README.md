# Heavy-hex scalability check

Does the heavy-hex code get better from d=3 to d=5 on IBM hardware? The plan is
to run both side by side on `ibm_fez` and compare logical error rates,
Λ = p_L(3) / p_L(5).

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

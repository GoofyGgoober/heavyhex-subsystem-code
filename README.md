# Heavy-hex scalability check

Does the heavy-hex code get better as it grows? The plan is to run d=3 and
d=5 side by side on IBM Heron (`ibm_fez`) and compare logical error rates,
Λ = p_L(3) / p_L(5). The end goal is a logical qubit that beats the best
physical qubit on the same chip.

The code is the heavy-hex subsystem code ([Chamberland et al., PRX
2020](https://arxiv.org/abs/1907.09528); on hardware, [Sundaresan et al.,
Nat. Commun. 2023](https://arxiv.org/abs/2203.07205)). d=3 uses 23 qubits
(9 data, 14 ancillas and relays). d=5 uses 65 (25 data, 20 X ancillas, 12 Z
ancillas, 8 relays). The two patches sit on Fez without sharing a qubit; see
the [blueprint](docs/figures/heavyhex-blueprint.png).

## Status

Done:

- Gauges, stabilizers and logicals at d=3 and d=5, placed on existing Fez couplers.
- Ideal and flagged memory circuits at both distances. Every single fault
  inside one gadget stays correctable. The distance of the whole circuit is
  not checked yet.
- Aer runs graded by a lookup decoder (d=3) or by MWPM under a simple noise
  model ([docs/mwpm-decoder.md](docs/mwpm-decoder.md)).

Not done: decoding that uses the flags under real circuit noise, compiling to
native gates, anything on the QPU. A learned decoder is sketched in
[docs/ml-decoder.md](docs/ml-decoder.md).

## Run

```bash
python -m pip install -e '.[sim,matching]'
heavyhex info
heavyhex syndrome 'X0 Z3'
heavyhex sweep --weight 2 --failures-only
heavyhex run --shots 128 --seed 7 --error X0
heavyhex --distance 5 run --decoder mwpm --rounds 3 --error X0
heavyhex --distance 5 circuit
heavyhex --distance 5 mwpm-sim --shots 10000 --seed 7
```

Data qubit ids are 0-based; the paper's Q label is id + 1. The lookup decoder
only works at d=3, so use `--decoder mwpm` at d=5. `heavyhex --help` lists the
rest, and `./sim` runs the same thing from `.venv`.

Redraw the blueprint with `python docs/figures/draw_blueprint.py` (needs Matplotlib).

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check .
```

QPU jobs need explicit permission for each run (see `AGENTS.md`).

# MWPM decoder

MWPM for X and Z memory at d=3 and d=5, under a simple noise model: each data
qubit flips independently before every round and before readout, each
stabilizer value can come out wrong, and readout can flip. Gate faults, flag
errors, leakage and real IBM calibration are not modelled.

## Run

```bash
python -m pip install -e '.[sim,matching]'
# Flagged circuit with perfect gates, X0 injected after the last X half.
heavyhex --distance 5 --json run --decoder mwpm --rounds 3 --error X0 --inject-at after_x2
# The noise model on its own, no circuit.
heavyhex --distance 5 --json mwpm-sim --rounds 3 --readout-error 0.01 --shots 10000 --seed 7
```

`run` still defaults to `--decoder lookup` (d=3, one round, error after prep).

## Detectors

A stabilizer's value in a round is the XOR of its gauge outcomes, and a round's
values are its syndrome. A detector is a stabilizer's value XOR its value the
round before.
`Memory(patch, basis, rounds).labels` gives the order: round by round, in
`patch.stabilizer_names` order, then one last row of memory-basis stabilizers
computed from the data readout.

Memory-basis stabilizers start at 0. The other basis is measured once during
prep (round -1), where it comes out random, so round 0 is compared with that.
Only the memory basis is checked against the readout, so MWPM only uses those
stabilizers.

`read_shot(schedule, gauge_bits, data_bits)` returns a `Shot` with the detectors
and the data bits. In the flagged circuit the flags and relays are undone after
use, so they read 0 unless a fault hit them. This decoder ignores them; the
circuit-level decoder in `simulation/noisy.py` uses them.

## Graph

One node per stabilizer per round. A data flip joins the one or two
stabilizers it touches (one means an edge to the boundary). A wrong stabilizer
value joins that stabilizer in two neighbouring rounds. Weights are
`log((1-p)/p)`.

In `MemoryNoise(data, measurement, readout)`, `data` is per qubit per round
(before each round and before readout), `measurement` is per stabilizer value,
and `readout` adds to the last data flip. Data qubits that touch the same
stabilizers, common on the Z strips, share one edge: the chance that an odd
number of them flipped.

```python
shot = read_shot(schedule, gauge_bits, data_bits)
decoder = MWPMDecoder(schedule.patch, basis=schedule.basis, rounds=schedule.rounds)
decoder.fails(shot)  # True if the logical readout is still wrong after correction
```

`decode_batch(detectors)` takes a `(shots, len(decoder.memory.labels))` array of
0s and 1s and returns one data-qubit correction mask per shot.

## Tested

- Noiseless ideal and flagged circuits, d=3 and d=5, both bases: no detector fires.
- Every single-qubit Pauli on every data qubit, between any two halves, over
  three flagged rounds.
- Every correctable data error with perfect stabilizer values.
- Every pair of faults at d=5 under the noise model.

Corrections only need to be right up to a gauge.

`simulation/noisy.py` goes further: it simulates the flagged circuit in stim with
Fez's calibrated noise and matches on the circuit's own error model. Next: IBM data.

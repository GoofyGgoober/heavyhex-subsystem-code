# ML decoder (planned)

**Goal.** Predict a shot's logical flip from its gauge outcomes and final
readout, as an alternative to MWPM.

**Why.** MWPM assumes independent errors with known rates. Noise on the chip
is correlated, biased and drifts, and a learned decoder can fit that.

**Training data.** Simulated shots labelled with their true logical flip.
`sample_memory` gives these for the simple noise model; there is no noisy
circuit simulation yet.

**Compare against** [MWPM](mwpm-decoder.md) that uses the flags, once it
exists, on logical error rate and time per shot. Keep it only if it wins when
the noise model is wrong, or is much faster.

**Not planned.** Real-time decoding on an FPGA.

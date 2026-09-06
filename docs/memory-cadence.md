# Fixed-duration syndrome-cadence experiment

## Question

For a fixed logical-memory window of length \(T\), how does the number \(n\) of
stabilizer-extraction rounds affect the decoded logical failure probability?
The quantity to minimize is

\[
p_L(n,T)=\Pr(\text{decoded logical outcome differs from the prepared outcome}).
\]

The experiment does not assume that the optimum is internal. A boundary result
such as \(n=0\) or \(n=1\) is a valid measurement.

## Protocol

One shot prepares either \(|0_L\rangle\), for a logical-Z test, or
\(|+_L\rangle\), for a logical-X test. Two preparations are available. The
*encoder* applies a synthesized Clifford that maps \(|0\rangle^{\otimes 9}\) to
the exact logical eigenstate, so every stabilizer is deterministic from the
first round, at the price of 18 noisy two-qubit gates before any check. The
*product* preparation starts every data qubit in \(|0\rangle\) (or
\(|+\rangle\)), as hardware experiments do: the stabilizers of the measured type
and the logical operator are already \(+1\), while the other type's first
outcomes are uniformly random and only define a reference frame that the
decoder never reads. For \(n>0\), the memory window is divided
into equal slices of duration

\[
\Delta t=T/n.
\]

A syndrome round of modeled duration \(\tau_s\) occurs at the end of each
slice. The free-evolution duration in that slice is

\[
\Delta t_{\mathrm{idle}}=T/n-\tau_s,
\]

so a point is feasible only when \(n\tau_s\le T\). For \(n=0\), the data idle
for the whole interval and are then destructively measured.

Every checked point uses the same \(T\): increasing \(n\) replaces idle time
with additional syndrome circuitry rather than extending the experiment.

## Idle process

Each data qubit has independent continuous-time Pauli-event rates
\(\gamma_X,\gamma_Y,\gamma_Z\). If \(g\in\{I,X,Y,Z\}\), the probabilities after
an interval are obtained from the exact continuous-time random walk on the
Pauli group modulo phase. In particular, define

\[
a=e^{-2(\gamma_X+\gamma_Y)t},\quad
b=e^{-2(\gamma_Z+\gamma_Y)t},\quad
c=e^{-2(\gamma_X+\gamma_Z)t}.
\]

Then

\[
\begin{aligned}
p_I&=(1+a+b+c)/4,\\
p_X&=(1-a+b-c)/4,\\
p_Y&=(1-a-b+c)/4,\\
p_Z&=(1+a-b-c)/4.
\end{aligned}
\]

The Aer runner attaches this Pauli channel to one delay per data qubit and time
slice. Syndrome extraction separately uses one-qubit depolarizing, two-qubit
depolarizing, reset, and readout channels.

## Observations and terminal boundary

Each shot records the eight-bit syndromes

\[
s_1,\ldots,s_n\in\mathbf F_2^8
\]

and the final nine data measurements. Syndrome changes are

\[
d_1=s_1,\qquad d_r=s_r\oplus s_{r-1}.
\]

The destructive data measurement supplies a final stabilizer boundary. In the
Z-basis experiment, the decoder uses the four Z checks and corrects the logical
Z readout. In the X-basis experiment, it uses the four X checks and corrects the
logical X readout. The raw logical bit is never shown to the decoder when it
chooses a correction; doing so would leak the known benchmark answer.

## Syndrome-history decoder

For one CSS basis, an error is reduced to four syndrome bits and one logical
flip bit. There are therefore only

\[
2^4\times2=32
\]

classes. For an independent per-data-qubit fault probability \(p\), the decoder
enumerates all \(2^9\) binary error patterns and sums their probability into
these classes. It then propagates the complete 32-state probability
distribution through every time slice, conditions it on each noisy syndrome,
and finally conditions on the stabilizer values inferred from data readout.

The chosen correction is the more likely of the two logical classes at that
final syndrome. This is exact Bayesian logical-class decoding for the stated
independent CSS model. Circuit-level ancilla propagation and correlated faults
are represented only through approximate decoder weights, so it is not an
exact maximum-likelihood decoder for the Aer circuit.

## Reported quantities

For \(N\) shots and \(F\) decoded failures, the main estimate is

\[
\widehat p_L=F/N.
\]

The report includes a Wilson binomial confidence interval, raw logical failure,
and mean detection events per shot on the checks the decoder reads; the JSON
output adds the decoder probabilities used at each cadence. It reports the
best sampled \(n\), rather than fitting an optimum between sampled values.

The sweep is run in both logical bases because storing only \(|0_L\rangle\)
tests bit preservation but cannot detect logical phase loss. An analytic bare
physical-qubit failure rate under the same idle and terminal-readout model is
included as a reference. State-preparation faults are excluded from that bare
reference and must be considered when making a hardware break-even claim.

## Circuit schedule

The ideal patch has no connectivity constraints, so the eight checks run one
after another and the round duration \(\tau_s\) is a model parameter rather than
a derived schedule. The CNOT order inside each weight-4 check still matters: an
ancilla fault after the second CNOT leaves a weight-2 "hook" error on the two
data qubits not yet visited. Z-type checks visit their qubits column by column
and X-type checks row by row, so every hook lies across the logical operator
of its own type rather than along it, and a single fault cannot reduce the
effective distance to 2.

## Interpretation and limitations

More frequent checks can separate faults that would otherwise be ambiguous in
one long interval. They also add faulty CNOTs, resets, and measurements. The
competition may yield an internal optimum, but no such outcome is guaranteed.
Whether the code beats an unencoded qubit at all depends on the noise rates:
under the project baseline it does not, while under Heron-like parameters it
does at every storage time, with more rounds always better.

Before treating a result as a hardware prediction, replace project baseline
rates with measured parameters and add the target topology, a scheduled circuit
with duration-aware relaxation during every operation, leakage/crosstalk models,
and a decoder error model derived from that circuit. The current preparation
circuit is synthesized for simulation and is not fault-tolerant.

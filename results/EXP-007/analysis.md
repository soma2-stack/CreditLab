# EXP-007 — Is recurrent learning necessary?

**Status: complete. EXPERIMENTAL RESULT.** The forward models are unchanged.
Only the training regime changed. Full training updates every weight. Readout-only
training freezes the input and recurrent weights at their random starting
values and trains the output layer. `B` stayed 4. These runs are not added
to the EXP-006 counts.

The same ten seeds and data base seed 2000 were used. All 120 training runs
and 120 bit-flip audits finished. There were no numerical failures and no
unstarted runs. Accounted runtime was 753 seconds, inside the 20-minute cap.
An earlier process stopped during delay 128; the unfinished runs were
completed without repeating finished ones. Python 3.11.9, PyTorch 2.13.0+cpu.
Training code began at revision `facb570`.

Success means final test accuracy of at least 0.95. The full additive control
succeeded on 10 of 10 delay-64 seeds, so fresh delay-128 training was run.

## How many seeds succeeded

| Model | Regime | Delay 64 | Delay 128 |
|---|---|---:|---:|
| Vanilla | Full | 3/10 | 1/10 |
| Vanilla | Readout only | 0/10 | 0/10 |
| Additive | Full | 10/10 | 10/10 |
| Additive | Readout only | 0/10 | 0/10 |
| Bounded | Full | 6/10 | 5/10 |
| Bounded | Readout only | 0/10 | 0/10 |

For every model, readout-only was worse than full training on the paired
seeds. Readout-only never succeeded on a seed where full training failed.

## Additive accuracies

Delay 64, full: 1.000 on every seed.

Delay 64, readout only: 0.871, 0.854, 0.871, 0.926, 0.896, 0.865, 0.873,
0.801, 0.885, 0.832.

Delay 128, full: 1.000 on every seed.

Delay 128, readout only: 0.801, 0.846, 0.840, 0.885, 0.840, 0.859, 0.855,
0.768, 0.902, 0.637.

Seed order is 101, 113, 127, 139, 151, 163, 179, 191, 211, 223.

Before any training, the additive hidden state already carried a partial
linear trace of the original bit. The validation probe R^2 at initialization
was about 0.36 to 0.70 at delay 64, averaging about 0.53, and about 0.29 to
0.58 at delay 128, averaging about 0.45. Those probe values were unchanged
after readout-only training, and the recurrent weights and hidden states
stayed the same on all 60 readout-only runs. The readout could use part of
that initial trace, which is why accuracy rose to roughly 0.80–0.93, but it
never reached the 0.95 bar.

## Bounded accuracies

Delay 64, full: 1.000, 1.000, 1.000, 0.896, 0.512, 1.000, 1.000, 0.482, 0.500, 1.000.

Delay 64, readout only: 0.838, 0.723, 0.826, 0.781, 0.703, 0.535, 0.850, 0.658, 0.590, 0.609.

Delay 128, full: 0.523, 0.504, 1.000, 1.000, 0.510, 0.500, 1.000, 0.492, 1.000, 1.000.

Delay 128, readout only: 0.764, 0.564, 0.697, 0.584, 0.570, 0.525, 0.777, 0.594, 0.580, 0.621.

Freezing the bounded model did not make it more reliable. It removed every
success.

## Bit-flip audit

Full additive runs and the successful full bounded runs changed the final
state and the prediction when only the original bit was flipped, and both
members of the pair were correct.

Readout-only additive runs also changed the final state on every pair, and
both answers were correct on roughly 55 to 83 percent of pairs. The recorded
bit gradient was not an exact computational zero on those pairs. The initial
dynamics already carry usable bit information. Training only the readout does
not turn that into reliable task success under this schedule.

The full bounded misses looked like the earlier audit. Some had exactly
identical final states. Delay-64 seed 139 again changed many pairs despite a
recorded bit gradient of 0.0.

## Interpretation

This is outcome B for the additive model.

Full training reproduced the additive result on all 10 seeds at both delays.
Readout-only additive did not succeed on any seed. Learning the recurrent and
input weights matters for reliable success under this matched schedule. That
does not prove another optimizer or a longer budget could never train the
readout alone, and it does not name a specific credit mechanism.

Outcome A is not supported. Outcome C is not supported: freezing hurt the
bounded model. Outcome D fits the readout-only bounded result, while full
bounded training remained only partly reliable, as in EXP-006. Outcome E does
not apply.

Gradient clipping for readout-only runs was computed over the readout weights
only. That is a real difference from full training and is not claimed to be
irrelevant. These results are not a new architecture and do not validate the
broader theory.

## Files

Metrics, loss curves, checkpoint diagnostics, fingerprints, checkpoints, and
audit rows are in this folder. Frozen EXP-001 through EXP-006 hashes matched
after the run.

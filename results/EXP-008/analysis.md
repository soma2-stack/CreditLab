# EXP-008 — Which parameter group enables additive learning?

**Status: complete. EXPERIMENTAL RESULT.** Only the existing additive model was
used. The forward equation was not changed. The bounded model was not trained
or tuned. These runs are not added to earlier seed counts.

Four training regimes shared the same starting weights, datasets, and
minibatches:

- Full: train input-side, recurrent-side, and readout.
- Readout only: freeze both input-side and recurrent-side weights.
- Input-side plus readout: train `W_x`, `b_x`, and the readout. Freeze `W_h` and `b_h`.
- Recurrent-side plus readout: train `W_h`, `b_h`, and the readout. Freeze `W_x` and `b_x`.

`b_x` and `b_h` are added in the same preactivation, so they are not separate
memory mechanisms. `W_x` is used at every timestep, so input-side learning is
not only learning to store the first bit.

Seeds were 101, 113, 127, 139, 151, 163, 179, 191, 211, and 223. Data base
seed was 2000. Success means final test accuracy of at least 0.95. The full
control succeeded on all 10 delay-64 seeds, so fresh delay-128 training was
run. All 80 training runs and 80 bit-flip audits finished in 625 seconds.
There were no numerical failures and no unstarted runs. Python 3.11.9,
PyTorch 2.13.0+cpu. Training code `906a7bd4d375`.

Frozen parameters stayed at their initial values in every run. Readout-only
hidden states on a fixed batch also stayed unchanged. Frozen EXP-001 through
EXP-007 hashes matched after the run.

## Delay 64

| Seed | Full | Readout only | Input + readout | Recurrent + readout |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.871 | 1.000 | 1.000 |
| 113 | 1.000 | 0.854 | 0.998 | 1.000 |
| 127 | 1.000 | 0.871 | 0.998 | 1.000 |
| 139 | 1.000 | 0.926 | 1.000 | 1.000 |
| 151 | 1.000 | 0.896 | 1.000 | 1.000 |
| 163 | 1.000 | 0.865 | 0.998 | 1.000 |
| 179 | 1.000 | 0.873 | 1.000 | 1.000 |
| 191 | 1.000 | 0.801 | 0.998 | 1.000 |
| 211 | 1.000 | 0.885 | 0.998 | 1.000 |
| 223 | 1.000 | 0.832 | 1.000 | 1.000 |

Success counts: full 10/10, readout only 0/10, input plus readout 10/10,
recurrent plus readout 10/10.

## Delay 128

| Seed | Full | Readout only | Input + readout | Recurrent + readout |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.801 | 1.000 | 1.000 |
| 113 | 1.000 | 0.846 | 1.000 | 1.000 |
| 127 | 1.000 | 0.840 | 1.000 | 1.000 |
| 139 | 1.000 | 0.885 | 0.963 | 0.998 |
| 151 | 1.000 | 0.840 | 1.000 | 1.000 |
| 163 | 1.000 | 0.859 | 0.998 | 1.000 |
| 179 | 1.000 | 0.855 | 1.000 | 1.000 |
| 191 | 1.000 | 0.768 | 0.996 | 1.000 |
| 211 | 1.000 | 0.902 | 0.998 | 1.000 |
| 223 | 1.000 | 0.637 | 1.000 | 1.000 |

Success counts are again 10/10, 0/10, 10/10, and 10/10. Delay 128 was fresh
training, not a longer test of the delay-64 weights.

## Bit-flip check

Full, input-plus-readout, and recurrent-plus-readout runs changed the final
state when only the original bit was flipped. Final states were not exactly
equal on those pairs. Both answers were correct on at least 93 percent of
pairs, and on at least 99.8 percent for full and recurrent-side runs. The
recorded bit gradient was not an exact computational zero alongside a finite
logit change.

Readout-only runs also changed the final state, and both answers were correct
on a substantial fraction of pairs, but test accuracy stayed below 0.95.
That matches EXP-007: the initial additive state already carries partial bit
information, and a trained readout uses some of it.

## Interpretation

This is outcome C.

Both partial routes succeeded on every seed at both delays. Learning the
hidden-to-hidden weights is not necessary for reliable success here, because
input-side weights plus the readout were enough while `W_h` and `b_h` stayed
fixed. Recurrent-side weights plus the readout were also enough while `W_x`
and `b_x` stayed fixed. Either group can provide a sufficient adaptation
route under this schedule.

Outcome A is included in that result. Outcome B is not, because input-side
training did not fail. Outcome D is not, because full training was not the
only reliable route. Outcome E is not: the full control succeeded, and
readout-only still did not reach 0.95.

This does not prove another budget could never change the readout-only
result. It does not say the two biases are separate mechanisms. It does not
claim a new architecture or validate the broader theory.

## Files

Metrics, curves, checkpoint diagnostics, fingerprints, checkpoints, and audit
rows are in this folder. The EXP-007 execution note is
`exp007_execution_audit.md`.

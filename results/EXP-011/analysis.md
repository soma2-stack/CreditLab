# EXP-011 — Gradient horizon × clipping policy

**Status: complete. EXPERIMENTAL RESULT.** The forward equation was not changed. K stayed 16. These runs are fresh controls and are not added to earlier seed counts.

Four conditions used the same starting weights, data, and minibatches:

- Full history, clipping at norm 5
- Final 16 steps, clipping at norm 5
- Full history, no gradient rescaling
- Final 16 steps, no gradient rescaling

No clipping means Adam received the computed gradients unchanged. It does not mean the numerical checks were removed. A raw gradient larger than 5 was recorded in those runs, and it was not used to rescale anything.

Seeds were 101, 113, 127, 139, 151, 163, 179, 191, 211, and 223. Data base seed was 2000. Success means final test accuracy of at least 0.95. The clipped full-history control succeeded on all 10 delay-64 seeds, so fresh delay-128 training was run. All 80 training runs and 80 bit-flip audits finished in 1032 seconds. There were no numerical failures, no timeouts, and no unstarted runs. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `39b57cd499ece`.

The frequent clipping in the earlier final-16 runs is a result of those larger gradients. It does not make that earlier comparison invalid. This experiment asks whether the full-history advantage still appears when clipping is turned off.

## Delay 64

| Seed | Full, clip 5 | Final 16, clip 5 | Full, no clip | Final 16, no clip |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.494 | 0.998 | 0.879 |
| 113 | 1.000 | 0.578 | 1.000 | 0.832 |
| 127 | 1.000 | 0.924 | 1.000 | 0.760 |
| 139 | 1.000 | 0.744 | 1.000 | 0.877 |
| 151 | 1.000 | 0.773 | 1.000 | 0.795 |
| 163 | 1.000 | 0.652 | 0.998 | 0.729 |
| 179 | 1.000 | 0.605 | 1.000 | 0.986 |
| 191 | 1.000 | 0.627 | 1.000 | 0.834 |
| 211 | 1.000 | 0.539 | 1.000 | 0.992 |
| 223 | 1.000 | 0.998 | 1.000 | 0.846 |

Success counts: 10/10, 1/10, 10/10, and 2/10. The clipped final-16 success is seed 223. The unclipped final-16 successes are seeds 179 and 211.

## Delay 128

These were fresh training runs.

| Seed | Full, clip 5 | Final 16, clip 5 | Full, no clip | Final 16, no clip |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.477 | 1.000 | 0.787 |
| 113 | 1.000 | 0.639 | 1.000 | 0.777 |
| 127 | 1.000 | 0.488 | 1.000 | 0.834 |
| 139 | 1.000 | 0.811 | 0.902 | 0.791 |
| 151 | 1.000 | 0.850 | 1.000 | 0.582 |
| 163 | 1.000 | 0.523 | 0.762 | 0.764 |
| 179 | 1.000 | 0.469 | 1.000 | 0.594 |
| 191 | 1.000 | 0.492 | 1.000 | 0.732 |
| 211 | 1.000 | 0.842 | 0.709 | 0.869 |
| 223 | 1.000 | 0.637 | 1.000 | 0.902 |

Success counts: 10/10, 0/10, 7/10, and 0/10. The unclipped full-history misses are seeds 139, 163, and 211.

## Checks

The clipped accuracies match the earlier final-16 experiment, including the single delay-64 success on seed 223. No unclipped run rescaled a gradient. On those runs the raw gradient was still larger than 5 on most final-16 updates, often about 90 percent or more, and that fact was only recorded.

Every final-16 training graph had a computational-zero gradient at the original event. The separate full-forward measurement still changed with the bit. Full-history runs kept a large forward bit sensitivity and got both flip answers right on nearly every pair when they had solved the ordinary test.

Unclipped final-16 successes at delay 64 also passed the flip check: both answers were correct on about 96 percent and 98 percent of pairs. The misses still changed the final state. Several stayed well above chance without reaching 0.95. A few clipped final-16 runs were near chance on the pair test even though the state still changed.

## Interpretation

This is outcome D. The effect depends on the delay.

At delay 64, the full-history advantage remains with clipping off: 10 of 10 versus 2 of 10. Turning clipping off raised many final-16 scores, but it did not make that regime reliable. The gap is not only an artifact of the norm-5 rule, and it is also not untouched by that rule.

At delay 128, clipped full-history training still solved every seed and clipped final-16 training solved none. Without clipping, full-history training fell to 7 of 10, while final-16 training still solved none. Removing clipping made the longer full-history runs less consistent. It did not rescue final-16 training.

No unclipped run produced an invalid number. That does not mean unclipped training is always safe. The Adam step is not the raw gradient times the learning rate, and the recorded update sizes are separate from the gradient sizes.

This does not claim a new architecture or a result about the broader theory.

## Files

Metrics, per-update gradient records, curves, checkpoints, and audit rows are in this folder. Frozen EXP-001 through EXP-010 hashes matched after the run.

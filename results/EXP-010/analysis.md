# EXP-010 — Does additive learning need long-horizon backpropagation?

**Status: complete. EXPERIMENTAL RESULT.** The forward equation was not changed. Only the training gradient was cut. These runs are not added to earlier seed counts.

Two regimes shared the same starting weights, data, and minibatches. Both trained every existing parameter. Full-history training sent the terminal loss back through the whole sequence. Final-16 training carried the state forward through the whole sequence, then detached that state once, without changing its value, and sent the loss back only through the last 16 steps. The last step is the query. K stayed 16.

Seeds were 101, 113, 127, 139, 151, 163, 179, 191, 211, and 223. Data base seed was 2000. Success means final test accuracy of at least 0.95. Full-history training succeeded on all 10 delay-64 seeds, so fresh delay-128 training was run. All 40 training runs and 40 bit-flip audits finished in 472 seconds. There were no numerical failures and no unstarted runs. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `af1c7fbfeaf8`.

The same weights are used at every step. A final-16 update can still change how the early steps behave on the next minibatch. A success here does not mean the early steps never learn. A miss does not prove the network lacks room to store the bit.

## Delay 64

| Seed | Full history | Final 16 |
|---|---:|---:|
| 101 | 1.000 | 0.494 |
| 113 | 1.000 | 0.578 |
| 127 | 1.000 | 0.924 |
| 139 | 1.000 | 0.744 |
| 151 | 1.000 | 0.773 |
| 163 | 1.000 | 0.652 |
| 179 | 1.000 | 0.605 |
| 191 | 1.000 | 0.627 |
| 211 | 1.000 | 0.539 |
| 223 | 1.000 | 0.998 |

Success counts: full history 10/10, final 16 steps 1/10. The only final-16 success is seed 223.

## Delay 128

These were fresh training runs.

| Seed | Full history | Final 16 |
|---|---:|---:|
| 101 | 1.000 | 0.477 |
| 113 | 1.000 | 0.639 |
| 127 | 1.000 | 0.488 |
| 139 | 1.000 | 0.811 |
| 151 | 1.000 | 0.850 |
| 163 | 1.000 | 0.523 |
| 179 | 1.000 | 0.469 |
| 191 | 1.000 | 0.492 |
| 211 | 1.000 | 0.842 |
| 223 | 1.000 | 0.637 |

Success counts: full history 10/10, final 16 steps 0/10.

## Two different gradient measurements

On every final-16 run, the training-graph gradient at the original event was a computational zero on all 512 test examples. The inputs before the cutoff contributed no training gradient. That is the cutoff working, not evidence that the forward pass forgot the bit.

The separate full-forward measurement, with no detach, still had a nonzero original-bit sensitivity on every test example. For the weak runs that sensitivity was small. For delay-64 seed 223 it was large. Full-history runs had large event gradients, because their training graph is the full sequence.

Final-16 runs were gradient-clipped on most updates, often more than 90 percent. Full-history runs were clipped on about 7 to 28 percent. The shorter horizon changes the gradient size, so clipping is not isolated from the cutoff. The clip limit stayed 5.0.

## Bit-flip check

This check uses the learned forward pass, with no detach and no training step. Final states were compared coordinate by coordinate.

Full-history runs changed the final state and got both answers right on at least 99.8 percent of pairs.

Final-16 runs also changed the final state, and the final logits were not identical. That is not the same as answering reliably. Delay-64 seed 223 got both answers right on 99.2 percent of pairs. Delay-64 seed 127, which scored 0.924 on the ordinary test, got both answers right on 84.6 percent of pairs. Several other final-16 runs changed the state but rarely flipped the prediction. Delay-64 seed 101 and several delay-128 runs were near chance on the pair test even though the state still changed.

A linear readout fitted to the hidden state sometimes still found partial bit information on runs whose own answers were near chance. Retention and a correct answer are different measurements.

## Interpretation

This is outcome B, with one seed-level exception and some partial information.

Full-history training reproduced the earlier additive result: every seed succeeded at both delays. Cutting explicit backpropagation off before the original event made reliable success much less common. Long-horizon training feedback improved learning under this schedule.

It was not required for every run. Delay-64 seed 223 reached 0.998 with the cutoff. Several other seeds kept partial bit information without reaching 0.95. That is not a proof that another horizon or a longer budget could never succeed, and it is not a proof that the network cannot hold the bit.

This does not claim a new architecture, a new learning rule, or a result about the broader theory.

## Files

Metrics, curves, diagnostics, checkpoints, and audit rows are in this folder. Frozen EXP-001 through EXP-009 hashes matched after the run.

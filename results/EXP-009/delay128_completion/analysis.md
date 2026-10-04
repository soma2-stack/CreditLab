# EXP-009 delay-128 continuation

**Status: the planned delay-128 cohort is now complete through an authorized continuation.**
The original training session remains interrupted. It did not finish normally, and this measured cost does not show that the original 15-minute cap was met. The first bias-plus-readout attempt for seed 101 still has no saved result and no known duration.

This continuation reused the two finished original runs, seed 101 full and seed 101 readout-only. It did not retrain them. It restarted seed 101 bias-plus-readout once, from initialization, and trained the other 37 conditions once. All 38 jobs finished. There were no numerical failures and no unstarted jobs. Python 3.11.9, PyTorch 2.13.0+cpu. Continuation code `3c5e02f64be3`. Measured continuation time was 381.534 seconds.

## Delay 128 accuracy

Success means final test accuracy of at least 0.95. Seed 101 full and seed 101 readout-only are the original-session results. Every other row is from this continuation.

| Seed | Full | Readout only | Bias + readout | Matrices + readout |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.801 | 0.895 | 1.000 |
| 113 | 1.000 | 0.846 | 0.809 | 1.000 |
| 127 | 1.000 | 0.840 | 0.971 | 1.000 |
| 139 | 1.000 | 0.885 | 0.850 | 1.000 |
| 151 | 1.000 | 0.840 | 0.930 | 1.000 |
| 163 | 1.000 | 0.859 | 0.834 | 1.000 |
| 179 | 1.000 | 0.855 | 0.947 | 1.000 |
| 191 | 1.000 | 0.768 | 0.898 | 1.000 |
| 211 | 1.000 | 0.902 | 0.992 | 1.000 |
| 223 | 1.000 | 0.637 | 0.949 | 1.000 |

Success counts: full 10/10, readout only 0/10, bias plus readout 2/10, matrices plus readout 10/10.

The two bias successes are seeds 127 and 211. Seeds 179 and 223 are just below the line, at 0.947 and 0.949. Being just below 0.95 is not evidence that the bit was forgotten.

Full training and readout-only training match the earlier delay-128 pattern: full training solves every seed, and readout-only stays below 0.95, including the low seed 223 score. There is no control discrepancy to investigate before reading the partial results.

## Checks

The two reused checkpoints matched their saved hashes and their saved correct counts, 512 of 512 for full and 410 of 512 for readout-only. Frozen parameters stayed fixed on every new run. Readout-only hidden states on a fixed batch stayed unchanged. Historical files matched the snapshot taken before this continuation.

## Bit-flip audit

All 40 delay-128 models were audited, including the two reused ones. Final states were compared coordinate by coordinate. No pair had exactly equal final states. No pair had a computational-zero bit gradient alongside a finite state or logit change. This is not a new pass threshold.

The two bias successes depend on the original bit. Both answers were correct on 0.924 of pairs for seed 127 and 0.986 for seed 211.

The eight bias misses also still change the final state when only the original bit is flipped. Both answers were correct on about 0.66 to 0.90 of pairs. The weakest of those pair scores are seeds 113 and 163, around 0.66. Seeds 179 and 223, which missed the ordinary line by a small margin, had both answers correct on about 0.89 and 0.90 of pairs.

Every matrix-trained run changed the final state and the logit. Both answers were correct on every pair.

## Interpretation

At delay 128, one learned preactivation bias plus the readout was enough to cross 0.95 on 2 of 10 seeds while both matrices stayed fixed. It was not enough on most seeds under this 400-update schedule. The misses still kept partial bit information.

Matrix learning plus the readout was enough on all 10 seeds while both preactivation biases stayed fixed.

This does not show that bias learning explains EXP-008. It does not show that bias adaptation always works, or that matrix learning is always necessary. It does not show that clipping caused the misses. It is not a new architecture and it does not validate the broader theory.

The original unfinished attempt remains an unknown cost, separate from these 381.534 seconds.

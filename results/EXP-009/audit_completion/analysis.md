# EXP-009 delay-64 bit-flip audit

**Status: diagnostic continuation only.** Training remains interrupted. The delay-64 cohort was already complete. The delay-128 cohort remains incomplete. This audit does not complete EXP-009.

All 40 saved delay-64 checkpoints were loaded. None were retrained. Python 3.11.9, PyTorch 2.13.0+cpu. Audit code `b8a42b7d8bb62e`. The diagnostic run took 10.649 seconds by `time.perf_counter`. That clock does not fill in the unfinished training run, and it does not show that the original 15-minute training cap was met.

A first launch stopped before any checkpoint check because the saved configuration file was read as JSON. No audit number came from that launch. The loader was corrected before this run.

## Verification

Each of the 40 conditions had one saved training record with status ok, a matching checkpoint hash, matching metadata, and the same integer correct count on the original 512 test examples. The additive equation was unchanged. Parameter hashes did not change during evaluation. Pairs differed only at the original bit. Historical EXP-001 through EXP-009 files still matched the snapshot taken before this audit.

## Pair results

These are not a new success threshold. Ordinary test success stays the saved training result. The audit asks whether flipping only the original bit changes the final state and the answer.

Final states were compared coordinate by coordinate. No pair had exactly equal final states. No pair had identical final logits. No pair had a computational-zero bit gradient alongside a finite state or logit change. Marker-channel sensitivity was stored separately and was not used as the bit measure.

| Regime | Both answers correct | Notes |
|---|---|---|
| Full | 0.998 to 1.000 | Seed 191 is the 0.998 case. |
| Readout only | 0.645 to 0.826 | The state still changes. This is partial use of the bit, below the ordinary 0.95 test line. |
| Bias plus readout | 0.686 to 0.984 | The eight ordinary successes are 0.908 to 0.984. |
| Matrices plus readout | 0.998 to 1.000 | Seed 191 is the 0.998 case. |

## Bias-only runs

The eight ordinary successes still depend on the original bit when later inputs are held fixed. Both answers were correct on about 91 to 98 percent of pairs. The prediction flipped on those same pairs. The final state was never exactly equal, and the bit gradient was not a computational zero.

Seed 151 missed the ordinary test line at 0.945. On the flip pairs its accuracy was 0.947. Both answers were correct on 0.895 of pairs, exactly one was correct on 0.105, and neither pair was wrong on both members. The state and the logit still changed.

Seed 191 missed the ordinary test line at 0.846. On the flip pairs its accuracy was 0.843. Both answers were correct on 0.686 of pairs and exactly one on 0.314. It still changed state and logit when only the original bit changed. It is weaker than the other bias runs, and it is not empty of bit information.

An ordinary score of at least 0.95 does not mean every flip pair is answered correctly. Some bias successes still miss about 4 to 9 percent of pairs.

## Matrix-trained runs

Every matrix-plus-readout delay-64 success changed the final state and the logit when only the original bit was flipped. Both answers were correct on at least 99.8 percent of pairs. The recorded bit gradient was not a computational zero.

## What this does not say

Bias adaptation does not, by itself, explain EXP-008. Matrix learning is not shown to be necessary in every setting. Clipping is not shown to have caused the two bias misses. Delay 128 remains unfinished, so no delay-128 bias or matrix result is established. This is not a new architecture and it does not validate the broader theory.

The original process stop still has no recorded error. Its cause is unknown.

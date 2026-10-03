# EXP-006 — Fresh-seed replication

**Status: complete. EXPERIMENTAL RESULT.** The models are the unchanged
EXP-004 vanilla, additive, and B=4 bounded-additive networks. Seeds 17, 29,
and 43 are not included in these counts. Nothing in EXP-001 through EXP-005
was changed.

Ten new training seeds were used: 101, 113, 127, 139, 151, 163, 179, 191,
211, and 223. The data base seed was 2000, so the datasets are new as well
as the starting weights. Success means final test accuracy of at least 0.95.
The additive model succeeded on all 10 delay-64 seeds, so fresh delay-128
training was run. All 60 training runs and all 60 bit-flip audits finished.
None failed numerically, none hit the five-minute cap, and none were left
unstarted. Total time was 529 seconds, inside the 15-minute cap. Python
3.11.9, PyTorch 2.13.0+cpu. Training code revision `53d4d63`.

## Delay 64 accuracy

| Seed | Vanilla | Additive | Bounded |
|---|---:|---:|---:|
| 101 | 0.521 | 1.000 | 1.000 |
| 113 | 1.000 | 1.000 | 1.000 |
| 127 | 0.490 | 1.000 | 1.000 |
| 139 | 0.680 | 1.000 | 0.896 |
| 151 | 1.000 | 1.000 | 0.512 |
| 163 | 1.000 | 1.000 | 1.000 |
| 179 | 0.486 | 1.000 | 1.000 |
| 191 | 0.498 | 1.000 | 0.482 |
| 211 | 0.502 | 1.000 | 0.500 |
| 223 | 0.686 | 1.000 | 1.000 |

Success counts: vanilla 3/10, additive 10/10, bounded 6/10.

Exact 95 percent intervals for those success rates, which are not a paired
test: vanilla 0.067 to 0.652, additive 0.692 to 1.000, bounded 0.262 to 0.878.
Mean accuracy was 0.686 for vanilla, 1.000 for additive, and 0.839 for bounded.

Paired outcomes: additive and bounded both succeeded on 6 seeds. Additive
succeeded and bounded failed on 4 seeds. Bounded never succeeded when
additive failed.

## Delay 128 accuracy

| Seed | Vanilla | Additive | Bounded |
|---|---:|---:|---:|
| 101 | 0.467 | 1.000 | 0.523 |
| 113 | 0.367 | 1.000 | 0.504 |
| 127 | 0.521 | 1.000 | 1.000 |
| 139 | 0.490 | 1.000 | 1.000 |
| 151 | 0.506 | 1.000 | 0.510 |
| 163 | 1.000 | 1.000 | 0.500 |
| 179 | 0.484 | 1.000 | 1.000 |
| 191 | 0.479 | 1.000 | 0.492 |
| 211 | 0.488 | 1.000 | 1.000 |
| 223 | 0.490 | 1.000 | 1.000 |

Success counts: vanilla 1/10, additive 10/10, bounded 5/10.

Intervals: vanilla 0.003 to 0.445, additive 0.692 to 1.000, bounded 0.187 to
0.813. Paired outcomes: both succeeded on 5 seeds, additive only on 5, bounded
only on 0.

These delay-128 numbers are fresh training runs, not a test of the delay-64
weights on a longer sequence.

## Bit-flip audit

Every model that met the 0.95 accuracy bar also answered both members of the
fresh bit-flip pairs correctly, or essentially so. Flipping only the original
bit changed the final state and the prediction. The recorded bit gradient was
not an exact computational zero on those successful runs.

Some bounded runs that missed the 0.95 bar still kept partial dependence.
At delay 64, seed 139 scored 0.896. On that audit, 400 of 512 pairs changed
the final state and the logit even though the recorded bit gradient was 0.0
on both members of every pair. Both answers were correct on about 78 percent
of pairs. At delay 128, seed 101 did not flip its prediction, while about
one third of the final states were not exactly equal.

Other bounded misses, including delay-64 seeds 151, 191, and 211, and several
delay-128 seeds, had exactly identical final states on all 512 pairs and a
recorded bit gradient of 0.0. For those runs the local gradient and the finite
flip agree that the original bit no longer changes the answer.

## Interpretation

This is closest to outcome A, with a clear paired limit.

The additive result repeated on all 10 new seeds at both delays. Bounded
success also occurred on several new seeds, so it is not confined to the old
seed 17. Six of 10 at delay 64 and 5 of 10 at delay 128 is the measured rate
in this cohort. The paired counts do not support parity: whenever the bounded
model failed, the additive model still succeeded, and the reverse did not
happen.

Ten seeds leave a wide interval, especially for the bounded model. These
rates are not a universal property of the architecture. The audit does not
explain why particular seeds succeed, and it does not show that clipping
caused the misses. This is not a new architecture and it does not validate
the broader theory.

## Files

Training metrics, loss curves, checkpoint diagnostics, fingerprints,
checkpoints, audit rows, and the delay-64 summary written before delay 128
are in this folder. Frozen EXP-001 through EXP-005 hashes matched after the
run.

# EXP-021 — Independent replication of clean-control overwrite

**Status: complete. EXPERIMENTAL RESULT.** This is a new cohort, not a pool with EXP-020. The models, the reset rule, the task, and the training settings were unchanged. All 40 runs finished in 612 seconds. The preflight check passed, with state, score, and gradient gaps of 0. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `9dd7a3c81f21`.

## Hold task

Both versions scored 1.000 on all ten seeds, 512 of 512. Their losses match to the recorded digits, as they did in EXP-020.

## Selective task

Ordinary additive accuracy, then replace-when-equal, replace-when-different, keep-when-equal, and keep-when-different. Each subgroup has 128 examples.

| Seed | Overall | Replace, same | Replace, conflict | Hold, same | Hold, conflict |
|---|---:|---:|---:|---:|---:|
| 601 | 0.703 | 0.922 | 0.141 | 0.875 | 0.875 |
| 607 | 0.752 | 1.000 | 0.008 | 1.000 | 1.000 |
| 613 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 617 | 0.566 | 0.578 | 0.422 | 0.633 | 0.633 |
| 619 | 0.584 | 0.672 | 0.266 | 0.711 | 0.688 |
| 631 | 0.637 | 0.766 | 0.188 | 0.797 | 0.797 |
| 641 | 0.559 | 0.617 | 0.398 | 0.625 | 0.594 |
| 643 | 0.742 | 0.984 | 0.016 | 0.984 | 0.984 |
| 647 | 0.557 | 0.641 | 0.312 | 0.656 | 0.617 |
| 653 | 0.543 | 0.562 | 0.469 | 0.594 | 0.547 |

None cleared 0.95. The separate readout was 0.75 on every one of these seeds. The conflicting-replacement cell was the miss.

Reset additive scored 1.000, 512 of 512, on nine seeds, with 128 of 128 in every subgroup. Seed 607 scored 0.998, 511 of 512. Its conflicting-replacement cell was 127 of 128, which is still above 0.95. The other three subgroups were 128 of 128. The separate readout agreed. The four bit-flip audits were at or above 0.996 for every reset seed.

## Relative to EXP-020

The success pattern repeated: reset 10 of 10, ordinary additive 0 of 10, hold 10 of 10 for both. A few ordinary-additive seeds here scored higher than the near-chance seeds in EXP-020, but none crossed the line, and conflicting replacement still failed. The two cohorts are not pooled.

## Interpretation

The fixed reset again solved replacement when both models received the same clean write signal. Ordinary additive recurrence again did not. The wipe of the old state is still a hard-coded rule, not a learned gate. Saturation was lower after the reset, as before, and it is not isolated as the cause. This is one memory episode. It does not show that other memories would be preserved, and it does not establish video capability.

## Possible video-model relevance

A reliable update instruction can replace a stale attribute. Wiping the whole state cannot be assumed to preserve other memories. Recognizing a visual change, and generating video, remain untested.

## Files

Scores and hashes are in this folder. Frozen EXP-001 through EXP-020 files, including the EXP-019 stop record, were unchanged.

# EXP-017 — Selective overwrite versus continued retention

**Status: complete. EXPERIMENTAL RESULT.** This is a new task, not another copy of the original delayed-bit test. Nothing new was added to the model. All 40 runs finished in 471.9 seconds. No numerical failures and no unfinished jobs. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `d3b223fb3dd0`.

The hold task always asks for the first bit, A. The selective task asks for a later bit, B, only when that bit is marked as an update. Otherwise it still asks for A. Each subgroup below has 128 test examples.

## Hold task

Additive accuracy was 1.000 on all ten seeds. Vanilla accuracy was 1.000, 0.504, 0.498, 0.535, 0.531, 0.482, 0.488, 0.520, 0.506, and 0.494. Only seed 503 cleared 0.95.

## Selective task, original readout

Overall accuracy, then the four subgroups: replacement with A equal to B, replacement with A different from B, hold with A equal to B, and hold with A different from B.

Additive:

| Seed | Overall | Replace, same | Replace, conflict | Hold, same | Hold, conflict |
|---|---:|---:|---:|---:|---:|
| 503 | 0.686 | 0.883 | 0.141 | 0.867 | 0.852 |
| 509 | 0.615 | 0.734 | 0.242 | 0.758 | 0.727 |
| 521 | 0.520 | 0.562 | 0.438 | 0.508 | 0.570 |
| 523 | 0.613 | 0.734 | 0.211 | 0.773 | 0.734 |
| 541 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 547 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 557 | 0.646 | 0.758 | 0.250 | 0.820 | 0.758 |
| 563 | 0.625 | 0.719 | 0.234 | 0.789 | 0.758 |
| 569 | 0.723 | 0.945 | 0.062 | 0.945 | 0.938 |
| 571 | 0.570 | 0.648 | 0.328 | 0.656 | 0.648 |

Vanilla overall accuracy was 0.479, 0.592, 0.482, 0.500, 0.527, 0.502, 0.482, 0.504, 0.510, and 0.479. Its subgroups stayed near one half. No selective run, additive or vanilla, met the rule that the overall score and all four subgroups reach 0.95.

## What the separate readout showed

On every additive selective model, the separate readout scored 0.750 overall. It was perfect when the answer was still A, and it scored 0.000 when a marked update required a different bit. That is an always-A readout. It does not solve selective replacement. On seeds 541 and 547 the jointly trained readout was at chance, while this separate readout still read A perfectly.

## Audits

For the additive hold models, flipping A changed the answer, and flipping B did not. That matches the task they were trained on. They were not trained to treat a marked B as a replacement, so those audit rows are not selective-overwrite failures.

For the additive selective models, an unmarked B was ignored on the separate readout in every pair. A marked B was not followed: the answer did not switch, and on several seeds the final state did not change either. After a marked update, flipping the old A still changed the answer. The model did not set the old bit aside.

## Interpretation

The additive model can keep the first bit. On this schedule it does not reliably replace that bit when the later bit is marked and disagrees. Keeping a conflicting distractor is not the failure. Using the new bit is. A separate linear readout makes that pattern sharper. It does not show that the old bit was erased, and it does not show that no other training schedule could learn the update. This is not a reason, by itself, to add a new gate.

## Possible video-model relevance

The useful split is whether a model can keep an attribute through distractions, replace it after a real change, and ignore an appearance that is not a change. Success or failure on this symbolic bit does not establish video capability.

## Files

Scores, audits, and hashes are in this folder. Frozen EXP-001 through EXP-016 files, including the interruption and continuation records, still match their hashes.

# EXP-020 — Selective overwrite with a clean write control

**Status: complete. EXPERIMENTAL RESULT.** This is not a rerun of EXP-017. Both models received an extra on/off write channel that the earlier task did not provide. The ordinary additive model and the reset model saw the same four-channel input. The reset is still a fixed rule, not a learned gate.

All 40 runs finished in 587 seconds. The preflight check passed: the new channel contains only exact 0 and 1, and the two hold-task models matched exactly before training. State gap, score gap, and gradient gap were all 0. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `398ea44b4474`. Each model has 1,249 parameters. The earlier three-channel models had 1,217. The extra 32 are the weights for the new channel.

EXP-017 did not save a separate data-fingerprint file. The first three channels were regenerated from that same generator and left unchanged when the fourth channel was added.

## Hold task

Both versions scored 1.000 on all ten seeds, 512 of 512. Their losses match to the recorded digits. The hold task never turns the clean control on after the first step, and that first step starts from zero, so the reset does not change hold training.

## Selective task

Ordinary additive accuracy, then replace-when-equal, replace-when-different, keep-when-equal, and keep-when-different. Each subgroup has 128 examples.

| Seed | Overall | Replace, same | Replace, conflict | Hold, same | Hold, conflict |
|---|---:|---:|---:|---:|---:|
| 503 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 509 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 521 | 0.543 | 0.586 | 0.445 | 0.586 | 0.555 |
| 523 | 0.500 | 0.516 | 0.445 | 0.539 | 0.500 |
| 541 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 547 | 0.533 | 0.570 | 0.375 | 0.602 | 0.586 |
| 557 | 0.568 | 0.633 | 0.406 | 0.617 | 0.617 |
| 563 | 0.510 | 0.516 | 0.477 | 0.523 | 0.523 |
| 569 | 0.652 | 0.797 | 0.188 | 0.812 | 0.812 |
| 571 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

None of these cleared 0.95. The separate readout was again about 0.75, perfect when the answer was still the old bit and about 0 on a conflicting replacement.

Reset additive scored 1.000 on nine seeds, with 128 of 128 in every subgroup. Seed 547 scored 0.998, 511 of 512. Its weakest subgroup was keep-when-equal, 127 of 128, which is still above 0.95. Conflicting replacement was 128 of 128 on every seed, including 547. The separate readout agreed.

## What the reset changed

On a marked write, the carried state is wiped before the update. Its magnitude after that wipe is 0. The new bit then changes the state immediately, on every pair, and the difference is still large at the question. The saved answer follows it.

The ordinary additive model, given the same clean control, often shows little or no state change when the marked bit flips. Its candidates stay mostly near plus or minus one, with saturation fractions about 0.89 to 1. After the reset, that fraction falls to about 0.37 to 0.49. Saturation moved with the intervention. The intervention also changes the old-state path, the size of the state, and the gradients together, so saturation is not isolated as the cause.

A hold-trained reset model was not trained to replace. If the clean control is turned on anyway, the wipe is structural. That is not learned erasure.

## Interpretation

The explicit control by itself did not make ordinary additive replacement succeed. The same control plus the fixed reset did, on all ten seeds, while hold performance stayed perfect. This is not a learned gate. It is also not a claim about the noisy EXP-017 interface, because the model was given a cleaner instruction. It does not establish video capability.

## Possible video-model relevance

A reliable update instruction is a different problem from recognizing a visual change. Wiping the whole state can replace a stale attribute and can also discard unrelated scene information. These symbolic results do not establish video capability.

## Files

Scores and hashes are in this folder. Frozen EXP-001 through EXP-019 files, including the EXP-019 stop record, were unchanged.

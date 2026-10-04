# EXP-022 — Update one memory without losing another

**Status: complete. EXPERIMENTAL RESULT.** This is a new two-memory task with six input channels. It is not a rerun of EXP-021. All 30 runs finished. None met the success rule. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `4fbbe75e7be108519d2abd2c7b12c765bf32fac5`.

## Question

Can a fixed reset of one half of the state update one stored bit and keep the other?

Three matched models shared the same inputs, the same starting weights, and the same minibatches:

- ordinary additive
- whole-state reset additive
- addressed-reset additive

## Contract checks

The preflight check passed before training.

- Clean write and address channels were exactly 0 or 1. True writes used a one-hot address. The query address appeared only at the final step. Noisy channels did not trigger a reset.
- All 64 combinations of the two stored bits, the candidate bit, the replacement flag, the update address, and the query address were balanced. Train had 24 of each, validation and test had 8 of each.
- Distractors skipped the two true writes and the candidate. The three models had equal parameter counts, 1,313, and matching initial weights and minibatch lists on every seed.
- Seeds 701, 709, 719, 727, 733, 739, 743, 751, 757, and 761 were unused. Data base seed 9000.

Before training, a counterfactual pair flipped a bit that appeared only before a later whole-state wipe. From the wipe onward, the hidden states were exactly equal. The largest absolute difference was 0. The whole-state model cannot recover that bit from the later suffix. That is a structural limit of the wipe, recorded before any score.

The addressed reset clears only the 16 coordinates named by the current address, and it does that before the candidate is added. The other 16 coordinates are unchanged by the reset itself. The candidate and the dense recurrence can still change them afterward. The half-state names are a fixed reset convention. They do not show that each bit lives only in its half.

## Scores

Success required overall accuracy of at least 0.95 and every subgroup at least 0.95. The test set has 512 examples. The six subgroups partition it: 128, 128, 64, 64, 64, and 64. Success count: 0 of 30.

Columns after the overall score are: no replacement and query slot 0; no replacement and query slot 1; replacement when the old bit already equals the new bit; replacement when they differ; replacement while querying the untouched slot 0; replacement while querying the untouched slot 1.

### Ordinary additive

| Seed | Overall | Keep 0 | Keep 1 | Replace, same | Replace, conflict | Untouched 0 | Untouched 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 701 | 256/512 (0.500) | 64/128 | 64/128 | 32/64 | 32/64 | 32/64 | 32/64 |
| 709 | 256/512 (0.500) | 64/128 | 64/128 | 32/64 | 32/64 | 32/64 | 32/64 |
| 719 | 281/512 (0.549) | 74/128 | 69/128 | 37/64 | 28/64 | 37/64 | 36/64 |
| 727 | 355/512 (0.693) | 112/128 | 80/128 | 50/64 | 13/64 | 55/64 | 45/64 |
| 733 | 356/512 (0.695) | 112/128 | 86/128 | 46/64 | 18/64 | 55/64 | 39/64 |
| 739 | 352/512 (0.688) | 104/128 | 91/128 | 46/64 | 16/64 | 53/64 | 42/64 |
| 743 | 308/512 (0.602) | 83/128 | 80/128 | 41/64 | 23/64 | 39/64 | 42/64 |
| 751 | 348/512 (0.680) | 99/128 | 90/128 | 50/64 | 16/64 | 49/64 | 44/64 |
| 757 | 256/512 (0.500) | 64/128 | 64/128 | 32/64 | 32/64 | 32/64 | 32/64 |
| 761 | 333/512 (0.650) | 86/128 | 91/128 | 46/64 | 19/64 | 44/64 | 47/64 |

Updated-slot accuracy by address, 64 examples each, stayed between 29/64 and 34/64 on every seed.

### Whole-state reset

| Seed | Overall | Keep 0 | Keep 1 | Replace, same | Replace, conflict | Untouched 0 | Untouched 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 701 | 256/512 (0.500) | 64/128 | 64/128 | 32/64 | 32/64 | 32/64 | 32/64 |
| 709 | 276/512 (0.539) | 64/128 | 64/128 | 40/64 | 41/64 | 33/64 | 34/64 |
| 719 | 373/512 (0.729) | 68/128 | 116/128 | 57/64 | 62/64 | 33/64 | 37/64 |
| 727 | 366/512 (0.715) | 62/128 | 122/128 | 59/64 | 57/64 | 34/64 | 32/64 |
| 733 | 382/512 (0.746) | 64/128 | 127/128 | 64/64 | 63/64 | 31/64 | 33/64 |
| 739 | 264/512 (0.516) | 64/128 | 64/128 | 36/64 | 38/64 | 31/64 | 31/64 |
| 743 | 280/512 (0.547) | 64/128 | 64/128 | 47/64 | 44/64 | 28/64 | 33/64 |
| 751 | 288/512 (0.562) | 64/128 | 64/128 | 47/64 | 42/64 | 36/64 | 35/64 |
| 757 | 350/512 (0.684) | 64/128 | 99/128 | 63/64 | 58/64 | 36/64 | 30/64 |
| 761 | 267/512 (0.521) | 64/128 | 64/128 | 36/64 | 37/64 | 32/64 | 34/64 |

On the stronger seeds, the updated slot was high for both addresses. Seed 733 was 64/64 on address 0 and 63/64 on address 1. The untouched cells stayed near 32/64. A high updated-slot score did not carry the untouched memory.

### Addressed reset

| Seed | Overall | Keep 0 | Keep 1 | Replace, same | Replace, conflict | Untouched 0 | Untouched 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 701 | 351/512 (0.686) | 96/128 | 95/128 | 45/64 | 23/64 | 48/64 | 44/64 |
| 709 | 282/512 (0.551) | 64/128 | 64/128 | 41/64 | 31/64 | 40/64 | 42/64 |
| 719 | 359/512 (0.701) | 100/128 | 90/128 | 36/64 | 29/64 | 64/64 | 40/64 |
| 727 | 353/512 (0.689) | 104/128 | 88/128 | 38/64 | 26/64 | 51/64 | 46/64 |
| 733 | 375/512 (0.732) | 102/128 | 87/128 | 33/64 | 39/64 | 63/64 | 51/64 |
| 739 | 284/512 (0.555) | 70/128 | 71/128 | 39/64 | 28/64 | 32/64 | 44/64 |
| 743 | 317/512 (0.619) | 93/128 | 95/128 | 32/64 | 30/64 | 34/64 | 33/64 |
| 751 | 326/512 (0.637) | 96/128 | 95/128 | 35/64 | 27/64 | 41/64 | 32/64 |
| 757 | 310/512 (0.605) | 86/128 | 89/128 | 34/64 | 28/64 | 32/64 | 41/64 |
| 761 | 337/512 (0.658) | 96/128 | 96/128 | 38/64 | 27/64 | 42/64 | 38/64 |

Updated-slot accuracy by address stayed between 30/64 and 40/64. Neither address was solved. The best conflicting-replacement cell was 39/64 on seed 733.

## Audits

Each audit used 256 matched pairs and the saved readout. No further training was done on those pairs. On every seed, 128 of the 256 query-switch histories had different current values in the two slots.

Flipping the new bit on a marked update, then querying the updated slot, produced both answers correct on 0 of the pairs for every addressed-reset seed and every ordinary-additive seed. The two answers usually agreed, so the readout gave the same decision for both signs of the new bit. On the addressed model the hidden state usually did change. A state change here did not become a usable replacement.

The same flip on the whole-state model changed the final state on every pair. Several seeds then followed the new bit: 0.988, 0.898, 0.887, and 0.812 of pairs on seeds 733, 719, 727, and 757. That is the bit written at the wipe, which the model can still see.

Flipping the untouched original bit left the whole-state model’s final state exactly equal on every seed, and both answers were correct on 0 pairs. The same equality held when the obsolete old bit of the updated slot was flipped. Those equalities are the structural wipe. They are not learned erasure.

On addressed reset, seeds 719 and 733 followed the flipped untouched bit on 0.977 and 0.973 of pairs. The other eight seeds were at or below 0.559. Changing only the query address left the answer unchanged on every addressed pair: prediction agreement was 1.0 on all ten seeds. The model did not switch slots when the question changed.

## Separate readout

The fixed diagnostic classifier was fit on training states and the current queried bit only. It did not move any seed to the success line.

Ordinary additive diagnostic accuracy was 0.668 to 0.717. Conflicting replacement stayed weak, 13/64 to 20/64.

Whole-state diagnostic accuracy sat at about 0.73 to 0.75. Four seeds were exactly 384/512. On those seeds the classifier was perfect on the bit written at the latest wipe and at chance on the wiped slot. That 0.75 pattern is the structural ceiling of remembering only what survives the last whole wipe.

Addressed diagnostic accuracy was 0.686 to 0.740. Conflicting replacement stayed at 16/64 to 29/64. On seeds 719, 727, and 733 the classifier scored 64/64 on untouched slot 0, so that bit was linearly readable, while replacement still failed. A classifier miss does not prove a bit is absent. A classifier hit on one cell does not make the two-memory task solved.

## Diagnostics

Training used 400 updates on every run. Validation was recorded at 0, 50, 100, 200, and 400. Gradients were clipped on 97.75 to 100 percent of updates. No loss was non-finite. Candidate saturation was high for ordinary additive, about 0.78 to 1.00, and for addressed reset, about 0.81 to 0.95. On marked whole-state steps the saturation fraction was 0, because that candidate is computed from a cleared state. Both hidden halves stayed large after the candidate step in every model. Saturation is not isolated as the cause.

## Interpretation

Letter E. No model solved the task under the registered rule.

The addressed reset did not achieve both correct replacement and preservation of the other memory. Conflicting replacement stayed near chance on every seed, and the query address did not change the answer. Two seeds showed partial retention of one untouched bit. That is persistence without controlled updating, and it stayed below the success line on the other cells.

Ordinary additive also stayed below the line. The strongest overall score was 356/512.

The whole-state model can report a bit written at the wipe, and the separate readout often does. It cannot report a bit that was written only before the wipe. Those untouched failures were expected from the pretraining equality check.

What is hard-coded: the clean write channel, the two address channels, and which 16 coordinates the addressed reset clears. What was left for learning: the shared weights, including a dense recurrence that can still change both halves. Those weights did not learn to update one slot and answer the other.

## Possible video-model relevance

Updating one attribute, such as shirt color, while keeping another attribute would require both replacement and retention. This experiment did not produce that behavior. The addresses here are explicit symbols supplied with the input. They are not a discovery of visual objects. A whole-state wipe and a half-state wipe are fixed rules in this model. They are not yet video mechanisms.

## Cost and files

The session took 640.441 seconds. The longest condition took 25.9 seconds. All 30 jobs exited 0. Nothing was left unstarted. There were no retries. Historical hashes before and after matched on 1,655 paths. Frozen EXP-001 through EXP-021 files were unchanged.

Scores, loss curves, checkpoints, classifiers, and audits are in this folder.

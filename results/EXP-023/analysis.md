# EXP-023 — Are both current memory values independently readable?

**Status: complete. EXPERIMENTAL RESULT.** This is a diagnostic on the 30 frozen EXP-022 models. No recurrent weight was trained. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Diagnostic code `4358f3e6fad6f960bf9e2ff00298754f8374e179`.

## Question

Before the final query, can two separately fitted linear classifiers recover the current values of both memory slots from the same hidden state? If so, does choosing the classifier with the query address that was already supplied improve the task score?

The features are the 32 hidden coordinates after input index 128, before the query at index 129. Head 0 is trained to read slot 0. Head 1 is trained to read slot 1. At test time the real query address selects one head. That selection is a fixed rule. It is not learned attention.

## Verification

All 30 checkpoints matched the saved file hashes and reproduced the saved integer correct counts of the original readout. Training inputs, initial weights, and minibatch lists still match within each seed. The pre-query state matches a rollout that stops at index 128. Replacing the query token with zeros, or changing only its address, leaves that state unchanged. The original task label is the current value of the queried slot. All 60 fits converged, with iteration counts from 31 to 300. No fit was inconclusive.

## Scores

Success required routed test accuracy of at least 0.95 and every one of the six original subgroups at least 0.95. That is 487 of 512. Success count: 0 of 30.

The columns are slot 0, slot 1, both slots on the same example, the routed task score, and the original saved readout. Losses are mean logistic loss.

### Ordinary additive

Slot 0 was 448/512 on every seed. Its three conditions were 256/256 with no replacement, 128/128 when the other slot was replaced, and 64/128 when slot 0 itself was replaced. The updated cell is the chance level of a readout that still reports the original bit.

| Seed | Slot 0 | Slot 1 | Both | Routed | Original |
|---|---:|---:|---:|---:|---:|
| 701 | 448/512 | 448/512 | 384/512 | 448/512 | 256/512 |
| 709 | 448/512 | 448/512 | 384/512 | 448/512 | 256/512 |
| 719 | 448/512 | 439/512 | 375/512 | 443/512 | 281/512 |
| 727 | 448/512 | 438/512 | 377/512 | 444/512 | 355/512 |
| 733 | 448/512 | 441/512 | 379/512 | 445/512 | 356/512 |
| 739 | 448/512 | 448/512 | 384/512 | 448/512 | 352/512 |
| 743 | 448/512 | 406/512 | 348/512 | 429/512 | 308/512 |
| 751 | 448/512 | 426/512 | 368/512 | 436/512 | 348/512 |
| 757 | 448/512 | 448/512 | 384/512 | 448/512 | 256/512 |
| 761 | 448/512 | 442/512 | 381/512 | 442/512 | 333/512 |

Slot 0 loss was 0.378 to 0.406. Slot 1 loss was 0.389 to 0.488. Routed loss was 0.383 to 0.428. On eight seeds the conflicting-replacement subgroup was 0/64. The other routed subgroups were at or next to the ceiling. The routed score of 448/512 is what follows from answering the original bit of the requested slot: the only misses are the 64 conflicting replacements.

### Whole-state reset

| Seed | Slot 0 | Slot 1 | Both | Routed | Original |
|---|---:|---:|---:|---:|---:|
| 701 | 319/512 | 446/512 | 255/512 | 381/512 | 256/512 |
| 709 | 311/512 | 433/512 | 244/512 | 375/512 | 276/512 |
| 719 | 307/512 | 434/512 | 243/512 | 374/512 | 373/512 |
| 727 | 313/512 | 447/512 | 255/512 | 382/512 | 366/512 |
| 733 | 320/512 | 448/512 | 256/512 | 384/512 | 382/512 |
| 739 | 320/512 | 449/512 | 257/512 | 385/512 | 264/512 |
| 743 | 321/512 | 446/512 | 256/512 | 383/512 | 280/512 |
| 751 | 318/512 | 448/512 | 254/512 | 382/512 | 288/512 |
| 757 | 320/512 | 448/512 | 256/512 | 384/512 | 350/512 |
| 761 | 314/512 | 447/512 | 250/512 | 381/512 | 267/512 |

Slot 0 loss was 0.661 to 0.682. Slot 1 loss was 0.365 to 0.426. Routed loss was 0.516 to 0.551. Slot 0 stays near chance when it was not the latest write: no-replacement was 124/256 to 130/256, and untouched during a later wipe was 61/128 to 67/128. The value written at a wipe was readable: the updated cell was 112/128 to 128/128 for slot 0 and 118/128 to 128/128 for slot 1. Slot 1 with no later wipe was 241/256 to 256/256. Both slots were correct together on about half of the examples. The routed score sits at about 384/512.

### Addressed reset

| Seed | Slot 0 | Slot 1 | Both | Routed | Original |
|---|---:|---:|---:|---:|---:|
| 701 | 436/512 | 376/512 | 322/512 | 407/512 | 351/512 |
| 709 | 445/512 | 386/512 | 333/512 | 419/512 | 282/512 |
| 719 | 448/512 | 382/512 | 331/512 | 421/512 | 359/512 |
| 727 | 438/512 | 409/512 | 347/512 | 424/512 | 353/512 |
| 733 | 450/512 | 392/512 | 339/512 | 420/512 | 375/512 |
| 739 | 387/512 | 364/512 | 257/512 | 388/512 | 284/512 |
| 743 | 439/512 | 443/512 | 373/512 | 434/512 | 317/512 |
| 751 | 449/512 | 457/512 | 394/512 | 449/512 | 326/512 |
| 757 | 408/512 | 352/512 | 272/512 | 382/512 | 310/512 |
| 761 | 441/512 | 396/512 | 345/512 | 422/512 | 337/512 |

Slot 0 loss was 0.202 to 0.467. Slot 1 loss was 0.304 to 0.532. Routed loss was 0.280 to 0.474. The best routed score was 449/512, on seed 751. Its no-replacement cells were 256/256 for both slots, and its untouched cells were 128/128 and 126/128. Its updated cells were 65/128 and 75/128. Across seeds, the updated cell stayed between 54/128 and 75/128. Conflicting replacement stayed between 11/64 and 37/64. Explicit routing raised the score above the original readout and remained below 0.95.

Validation counts moved with the test counts and were not used to choose a model or a setting.

## Audits

Each pair audit has 256 pairs. The query-switch count is both heads correct on the histories whose two current values differ. There were 128 such histories in every audit set. Nothing was fit on these examples.

Ordinary additive followed the untouched bit on 256/256 pairs for every seed, and followed a marked new bit on at most 1/256. Flipping the obsolete old bit changed the answer: both members stayed correct on 0/256 pairs, and the pre-query states were not equal. An unmarked candidate was ignored on 256/256 pairs for every seed. Both heads were correct on 74/128 to 96/128 of the histories whose current values differed.

Whole-state reset followed the marked new bit on 229/256 to 256/256 pairs, and the pre-query states differed. The untouched-bit flip left the pre-query state exactly equal on 256/256 pairs for every seed, and both answers were correct on 0 pairs. The same exact equality held for the obsolete-bit flip. Those equalities are the wipe established before EXP-022 training. They are not a new decoding result and they are not learned erasure. Both heads were correct on at most 6/128 differing histories.

Addressed reset followed a marked new bit on at most 1/256 pairs. The pre-query state usually changed. It followed the untouched bit on 256/256 pairs for seeds 719, 727, 733, and 743, on 255/256 for seed 751, and on 235/256 to 243/256 for seeds 701, 709, and 761. Seeds 739 and 757 were lower, at 165/256 and 197/256. Both heads were correct on 27/128 to 103/128 differing histories. The best joint query-switch count was 103/128, on seed 751.

## Interpretation

Letter C for ordinary additive and for addressed reset. The original stored bits are linearly readable, including the slot that was not named by a later replacement. The replacement itself is not. A marginal score of 448/512 is the original-bit ceiling, and the joint count is lower, 384/512 on the clean additive seeds, because an update makes one of the two original-bit answers wrong. High marginal scores do not solve the task.

Letter B for whole-state reset. The value written at the latest wipe is readable. A value written only before that wipe is not. Slot 0 and the untouched slot during a later wipe stay at chance. That split was required by the wipe. It is not two-memory competence. The routed score near 384/512 is that structural ceiling.

Explicit routing did not rescue the task for any checkpoint. Where an original bit was linearly present, the routed heads scored higher than the saved single readout. That shows the saved readout underused information the two new heads could read. It does not show that query selection alone caused the original miss, because the updated value is still not recovered. The two heads and the separate fitting procedure are not an equal-budget comparison with the original readout.

The whole-state miss on a pre-wipe bit is a property of the state: the audit states match exactly. The additive and addressed miss on the new bit is a miss of this linear diagnostic. On the addressed models the new bit usually changes the state, and the head still does not follow it. A state change is not a usable current value. A linear miss does not prove the new bit is absent.

The half-state names remain a fixed reset convention. This diagnostic does not show that each bit lives only in its half. The route from query address to head is hard-coded. It is not learned addressing.

## Possible video-model relevance

Keeping several attributes and then answering the one that was asked are separate steps. Here the original attributes were often readable, and a supplied address could pick one of them. The updated attribute was not recovered by this readout. The address is an explicit symbol in the input. It is not learned visual addressing. No visual or video capability was tested.

## Cost and files

The diagnostic session took 24.502 seconds. The longest seed batch took 2.654 seconds. All 10 batches exited 0. Nothing was left unstarted. There were no retries. Recurrent weights were unchanged after fitting. Historical hashes before and after matched on 1,790 paths, including the frozen EXP-001 through EXP-022 files.

Scores, fitted heads, and audits are in this folder.

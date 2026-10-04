# EXP-014 — Independent replication of representation versus readout

**Status: complete. EXPERIMENTAL RESULT.** This is a new cohort. It is not added to the earlier seed counts. The model, the cutoff, and the classifier were not changed after these scores were seen.

Seeds were 307, 311, 313, 317, 331, 337, 347, 349, 353, and 359. Data base seed was 3000. Delay was 128 only. All 20 training runs and all 30 classifier fits finished in 340 seconds. There were no numerical failures and no unstarted runs. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `76d7b76dc488`.

The final-16 training gradient at the original event was blocked on every seed. A separate full-forward check, with no cutoff, still changed with the bit. Those two measurements were kept apart.

## Original readout and new readout

Success means test accuracy of at least 0.95.

| Seed | Init diagnostic | Full original | Full diagnostic | Final-16 original | Final-16 diagnostic |
|---|---:|---:|---:|---:|---:|
| 307 | 0.990 | 1.000 | 1.000 | 0.904 | 0.984 |
| 311 | 0.986 | 1.000 | 1.000 | 0.967 | 1.000 |
| 313 | 0.922 | 1.000 | 1.000 | 0.643 | 1.000 |
| 317 | 0.969 | 1.000 | 1.000 | 0.543 | 1.000 |
| 331 | 0.842 | 1.000 | 1.000 | 0.848 | 0.945 |
| 337 | 0.977 | 1.000 | 1.000 | 0.895 | 1.000 |
| 347 | 0.861 | 1.000 | 1.000 | 0.855 | 0.998 |
| 349 | 0.893 | 1.000 | 1.000 | 0.713 | 1.000 |
| 353 | 0.943 | 1.000 | 1.000 | 0.887 | 0.922 |
| 359 | 0.887 | 1.000 | 1.000 | 0.678 | 1.000 |

Success counts:

- Initialized diagnostic: 4/10
- Full-history original readout: 10/10
- Full-history diagnostic: 10/10
- Final-16 original readout: 1/10, seed 311
- Final-16 diagnostic: 8/10

The two final-16 states that stayed below 0.95 were seeds 331 at 0.945 and 353 at 0.922. Both were still above their untrained diagnostic scores for seed 331, and seed 353 was slightly below its untrained score of 0.943. Full-history training raised both to 1.000.

## Bit-flip check

States changed on every pair. Where the ordinary score was at least 0.95, both answers were correct on about 93 to 100 percent of pairs. The two final-16 diagnostic misses still changed the score on every pair, and both answers were correct on about 85 percent and 86 percent of pairs. Being below 0.95 did not mean the bit was absent.

## Interpretation

The readout-rescue pattern mostly replicated on this new cohort. Final-16 training again left an original readout below 0.95 on 9 of 10 seeds, while a separate linear classifier reached 0.95 on 8 of 10 of those states. It did not repeat for every seed.

Untrained states were already enough on 4 of 10 seeds. That rate is in the same range as the earlier delay-128 check, which was 5 of 10. Some success does not require recurrent training. It is not the typical outcome on this cohort.

Full-history learning raised every weaker untrained state to 1.000, and its own original readout also succeeded on every seed. The full-history control did not weaken. Final-16 learning improved most of the weaker untrained states without sending a training gradient back to the original event. The same weights are still reused at every step, so a late update can change earlier processing on the next batch.

This does not prove the representations are equivalent just because many diagnostic scores are high. It does not prove a miss is linearly unreadable. It does not claim a new architecture or a result about the broader theory.

## Files

Training records, classifiers, audits, and hashes are in this folder. The pre-run note was not rewritten. Frozen EXP-001 through EXP-013 files matched after the run.

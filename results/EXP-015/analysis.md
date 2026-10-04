# EXP-015 — Frozen-model length generalization

**Status: complete. EXPERIMENTAL RESULT.** Nothing was trained and no classifier was refit. The saved delay-128 models and the saved diagnostic classifiers were only scored again.

The fresh delay-128 numbers are a new draw, with data base seed 4000. They are the reference for this check. They are not the original EXP-014 test set. Longer sequences also contain more distractors, so this is not a pure test of elapsed time.

Seeds are 307, 311, 313, 317, 331, 337, 347, 349, 353, and 359. Success means accuracy of at least 0.95. All 30 checkpoints and 30 classifiers reproduced their saved correct counts. All 150 endpoint scores finished in 15.8 seconds. None were numerically invalid. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Evaluation code `114e74ffefe`.

Hidden magnitude was recorded after the event (index 0), near the middle of the distractors (index delay / 2), just before the query (index delay), and after the query (index delay + 1).

## Accuracy

| Seed | Init diagnostic | Full original | Full diagnostic | Final-16 original | Final-16 diagnostic |
|---|---:|---:|---:|---:|---:|
| **Delay 128** | | | | | |
| 307 | 0.990 | 1.000 | 1.000 | 0.887 | 0.994 |
| 311 | 0.965 | 1.000 | 1.000 | 0.959 | 0.998 |
| 313 | 0.949 | 1.000 | 1.000 | 0.594 | 1.000 |
| 317 | 0.969 | 1.000 | 1.000 | 0.545 | 1.000 |
| 331 | 0.814 | 1.000 | 1.000 | 0.840 | 0.924 |
| 337 | 0.959 | 1.000 | 1.000 | 0.875 | 1.000 |
| 347 | 0.865 | 1.000 | 1.000 | 0.855 | 0.994 |
| 349 | 0.910 | 1.000 | 1.000 | 0.760 | 1.000 |
| 353 | 0.920 | 1.000 | 1.000 | 0.863 | 0.926 |
| 359 | 0.896 | 1.000 | 1.000 | 0.732 | 0.984 |
| **Delay 256** | | | | | |
| 307 | 0.535 | 0.762 | 0.535 | 0.953 | 0.535 |
| 311 | 0.938 | 1.000 | 1.000 | 0.826 | 0.480 |
| 313 | 0.496 | 0.998 | 0.951 | 0.529 | 0.484 |
| 317 | 0.512 | 1.000 | 1.000 | 0.512 | 0.488 |
| 331 | 0.627 | 0.992 | 1.000 | 0.738 | 0.484 |
| 337 | 0.512 | 1.000 | 0.492 | 0.688 | 0.512 |
| 347 | 0.480 | 0.783 | 0.475 | 0.604 | 0.525 |
| 349 | 0.494 | 0.953 | 0.510 | 0.631 | 0.490 |
| 353 | 0.752 | 1.000 | 0.502 | 0.664 | 0.543 |
| 359 | 0.854 | 1.000 | 0.506 | 0.555 | 0.500 |
| **Delay 512** | | | | | |
| 307 | 0.482 | 0.482 | 0.482 | 0.928 | 0.482 |
| 311 | 0.830 | 0.992 | 0.957 | 0.539 | 0.475 |
| 313 | 0.492 | 0.836 | 0.816 | 0.496 | 0.510 |
| 317 | 0.518 | 1.000 | 1.000 | 0.516 | 0.484 |
| 331 | 0.510 | 0.961 | 1.000 | 0.484 | 0.557 |
| 337 | 0.488 | 1.000 | 0.512 | 0.508 | 0.488 |
| 347 | 0.475 | 0.488 | 0.488 | 0.496 | 0.512 |
| 349 | 0.535 | 0.748 | 0.477 | 0.477 | 0.523 |
| 353 | 0.707 | 1.000 | 0.369 | 0.561 | 0.516 |
| 359 | 0.779 | 1.000 | 0.473 | 0.582 | 0.529 |

Success counts, all 10 seeds:

| Endpoint | 128 | 256 | 512 |
|---|---:|---:|---:|
| Initialized diagnostic | 4 | 0 | 0 |
| Full-history original | 10 | 8 | 6 |
| Full-history diagnostic | 10 | 4 | 3 |
| Final-16 original | 1 | 1 | 0 |
| Final-16 diagnostic | 8 | 0 | 0 |

Among the models that were already successful on this fresh delay-128 draw, the number that stayed successful was 0 and 0 for the initialized diagnostic, 8 and 6 for the full-history original readout, 4 and 3 for the full-history diagnostic readout, and 0 and 0 for both final-16 readouts. The one final-16 original success at delay 256 is seed 307, which had not cleared 0.95 at delay 128. It is not a retained success.

## Bit flips and magnitude

At every delay, flipping only the original bit still changed the final state. No pair had exactly equal final states. The diagnostic score changed on every pair. Correct answers did not survive with that dependence. At delay 256 and 512, the diagnostic readouts are mostly near chance even though the state still depends on the bit.

The largest hidden value grew with delay for every representation, roughly in line with the number of steps: about 128, 256, and 512. Untrained and trained models grew in the same way. A large state is what this additive rule does. It is not, by itself, evidence of memory.

## Interpretation

The saved diagnostic readout does not carry the delay-128 result to 256 or 512. The full-history original readout does so on many seeds: 8 of 10 at delay 256 and 6 of 10 at delay 512. The two readouts are not equally robust. No new fit was used to obtain that difference.

Final-16 models, and the untrained models, do not keep a reliable answer at the longer lengths. Their states still change with the original bit, so the bit has not disappeared. This does not prove that no readout could succeed at those lengths.

This is not unlimited memory, not video memory, and not a proof of the broader theory.

## Files

Scores, magnitude records, audit rows, and hashes are in this folder. Frozen EXP-001 through EXP-014 files matched after the run. The scientific checkpoint was not rewritten.

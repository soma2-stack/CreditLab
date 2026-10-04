# EXP-013 — Does initialized additive memory already suffice?

**Status: complete. DIAGNOSTIC RESULT.** No recurrent weights were trained. This is an exploratory check of saved models, not a new seed count.

Three frozen states were compared for each of the ten seeds at delays 64 and 128:

- Initialized additive dynamics, from the EXP-007 readout-only checkpoints. Their input and recurrent weights still match a freshly seeded model, with no optimizer step. Their old readout was not used for the new fit.
- Full-history states from EXP-011, with clipping.
- Final-16 states from EXP-011, with clipping.

The classifier is the same fixed L2 logistic regression used in EXP-012. All 60 fits converged, in at most 176 iterations. The full-history and final-16 fits reproduced the EXP-012 test predictions exactly. The diagnostic took 11.3 seconds. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Diagnostic code `7b132c6ce698`.

EXP-007 does not have a complete saved fingerprint file. Those missing rows were not filled in. The data tensors used here were regenerated from the registered rule and matched the saved EXP-011 fingerprints.

## Diagnostic accuracy

Success means test accuracy of at least 0.95. A score below that line can still carry partial information.

| Seed | Delay 64 init | Delay 64 full | Delay 64 final 16 | Delay 128 init | Delay 128 full | Delay 128 final 16 |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 0.896 | 1.000 | 1.000 | 0.840 | 1.000 | 1.000 |
| 113 | 0.861 | 1.000 | 1.000 | 0.854 | 1.000 | 1.000 |
| 127 | 0.986 | 1.000 | 0.988 | 0.979 | 1.000 | 1.000 |
| 139 | 0.947 | 1.000 | 1.000 | 0.939 | 1.000 | 0.998 |
| 151 | 0.965 | 1.000 | 1.000 | 0.904 | 1.000 | 0.998 |
| 163 | 0.998 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 |
| 179 | 0.932 | 1.000 | 1.000 | 0.895 | 1.000 | 1.000 |
| 191 | 0.996 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 |
| 211 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 223 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Initialized states reached 0.95 on 6 of 10 seeds at delay 64 and 5 of 10 at delay 128. Full-history and final-16 states reached it on all 10 seeds at both delays.

The original EXP-007 readouts on these same initialized models stayed below 0.95, mostly about 0.80 to 0.93. The new classifier is what changed for the seeds that now succeed.

## Bit-flip check

Initialized states never stayed exactly the same when only the original bit was flipped, and the new readout’s score changed on every pair. Seeds that crossed 0.95 had both answers correct on about 93 to 100 percent of pairs. Seeds below 0.95 still had both answers correct on about 73 to 87 percent of pairs. The trained states had both answers correct on at least 98.4 percent of pairs.

## Interpretation

This is outcome C.

Untrained additive dynamics are already enough for a reliable answer on some seeds under this diagnostic, and not on others. Earlier readout-only failure came from that original fitting schedule. It did not show that the untrained state lacks a readable answer.

Where the untrained state stayed below 0.95, both full-history learning and final-16 learning raised the diagnostic score to about 0.99 or 1.00. Final-16 learning did that even though its training gradient did not reach back to the original event. That is an improvement in the state, not only in the original readout.

A miss by this one regularized classifier does not prove that no linear classifier could succeed. This does not claim a new architecture or a result about the broader theory.

## Files

Verification, provenance, metrics, classifiers, and hashes are in this folder. Recurrent hashes were unchanged. Frozen EXP-001 through EXP-012 files matched after the run.

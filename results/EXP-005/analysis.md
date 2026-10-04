# EXP-005 — Counterfactual early-bit retention audit

**Status: complete. EXPERIMENTAL RESULT.** This audit does not add a model and
does not change EXP-001 through EXP-004 files. EXP-004 saved no trained
weights. The eighteen models were rebuilt by replaying the EXP-004 training
procedure. Every replay matched the saved EXP-004 correct count, validation
loss, test loss, retention probe, and hidden-state magnitude within the
tolerances locked before this audit. Matching those numbers is consistency
evidence, not proof that every weight bit is identical.

The question was: if every later input stays the same, does flipping only the
original event bit change the final hidden state and the prediction?

Fresh data used 512 sequences per seed and delay, with split index 3. Each
sequence was copied twice. Only the original bit was set to +1 or -1. Python
3.11.9, PyTorch 2.13.0+cpu. Audit code revision `3ebc12f0e389`. Total runtime
162.5 seconds. No run failed and none hit the five-minute replay cap.

A recorded gradient of 0.0 is an exact computational zero in float32. It is
not claimed to be a mathematical zero.

## What flipping the bit did

| Delay | Model | Seed | Both correct | Prediction flips | Final states exactly equal | Recorded bit gradient |
|---|---|---:|---:|---:|---:|---|
| 64 | Vanilla | 17 | 0.000 | 0.000 | 0.000 | tiny, never exact zero |
| 64 | Vanilla | 29 | 0.000 | 0.000 | 0.000 | tiny, never exact zero |
| 64 | Vanilla | 43 | 1.000 | 1.000 | 0.000 | nonzero |
| 64 | Additive | 17 | 0.998 | 0.998 | 0.000 | nonzero |
| 64 | Additive | 29 | 1.000 | 1.000 | 0.000 | nonzero |
| 64 | Additive | 43 | 1.000 | 1.000 | 0.000 | nonzero |
| 64 | Bounded | 17 | 1.000 | 1.000 | 0.000 | nonzero |
| 64 | Bounded | 29 | 0.000 | 0.000 | 1.000 | exact zero on every example |
| 64 | Bounded | 43 | 0.000 | 0.000 | 1.000 | exact zero on every example |
| 128 | Vanilla | 17 | 0.080 | 0.102 | 0.000 | nonzero |
| 128 | Vanilla | 29 | 0.000 | 0.000 | 0.000 | tiny, never exact zero |
| 128 | Vanilla | 43 | 0.000 | 0.000 | 0.000 | tiny, never exact zero |
| 128 | Additive | 17 | 1.000 | 1.000 | 0.000 | nonzero |
| 128 | Additive | 29 | 1.000 | 1.000 | 0.000 | nonzero |
| 128 | Additive | 43 | 1.000 | 1.000 | 0.000 | nonzero |
| 128 | Bounded | 17 | 1.000 | 1.000 | 0.000 | mostly nonzero |
| 128 | Bounded | 29 | 0.000 | 0.000 | 1.000 | exact zero on every example |
| 128 | Bounded | 43 | 0.232 | 0.232 | 0.768 | exact zero on every example |

## Successful models

The additive model and the bounded seed-17 model change with the original bit.
Final states differ on every pair. The signed logit gap is large and points
the right way. Predictions flip, and both members of the pair are correct.
On the bounded successes, about 92 to 93 percent of final coordinates sit
exactly at +4 or -4, but the boundary sign pattern differs between the two
members of every pair. Equal magnitude does not mean equal state.

The one successful vanilla run, seed 43 at delay 64, behaves the same way:
the bit changes the state, the logit, and the prediction.

## Bounded failures that truly ignore the bit

Bounded seed 29 at both delays, and bounded seed 43 at delay 64, are the
same pattern. Flipping the bit leaves the final state exactly equal on all
512 pairs, including the pattern of +4 and -4 signs. Every final coordinate
is on the boundary. The logits are exactly equal. The recorded bit-channel
gradient is 0.0 on every example. The local gradient and the finite flip
agree: neither sees a change. Accuracy is 0.5 because one fixed prediction
is correct for exactly one member of each pair.

## Bounded seed 43 at delay 128

This case is mixed, and it is not the same as seed 29.

On 393 of 512 pairs, the final states and boundary sign patterns are
identical, the logits are identical, and the prediction does not flip.

On the other 119 pairs, the final state changes, the boundary sign pattern
changes, the logit changes, and the prediction flips in the correct
direction. Both answers are correct on those pairs. The recorded gradient
of the final logit with respect to the original bit channel is still 0.0
on both members of all 512 pairs, including those 119. So a zero local
gradient does not fully describe the finite change for this seed. That is
compatible with a saturated sign pattern, but it does not prove a specific
internal code.

## Vanilla failures

Failed vanilla runs usually do not flip their prediction. Their final states
are not exactly equal, but the distances are tiny and the sign patterns
usually match. Their bit gradients are very small and were not recorded as
exact zeros. That is a weak residue, not the exact identity seen in the
bounded seed-29 failures.

## Interpretation

- Where the model succeeded, flipping the original bit changed the state and
  the answer. For the bounded successes, the information was in the pattern
  of boundary signs, not in growing past 4.
- Bounded seed 29, and bounded seed 43 at delay 64, have lost usable
  dependence on the original bit. The finite flip and the local gradient
  agree.
- Bounded seed 43 at delay 128 still depends on the bit for 119 pairs even
  though the recorded local bit gradient is 0.0. Local sensitivity missed
  that dependence.
- These failed bounded models do not all forget in the same way.
- This audit does not show that clipping caused the failures. It also does
  not validate the broader theory or propose a new architecture.

## Files

Recovered weights are in `checkpoints/`. Verification is in
`verification_report.json`. Pair metrics are in `audit_metrics.jsonl` and
`summary.csv`. State trajectories are in `state_trajectories.jsonl`.

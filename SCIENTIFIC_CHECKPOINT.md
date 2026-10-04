# Scientific checkpoint

The project has a repeated result on one small synthetic task. An additive recurrent network, which keeps the previous state and adds a bounded update, learns a delayed bit more reliably than a vanilla network when both are trained the same way. Later checks separated three different things that are easy to mix together: whether information is still in the hidden state, whether a fixed linear readout can see it, and whether the readout trained jointly with the network actually uses it. Full-sequence training feedback makes the joint readout reliable. Cutting that feedback usually hurts the joint readout, but the state often remains readable by a different linear fit. Some seeds are readable even before any recurrent training. Some failures remain.

This is a saved-record audit. Nothing was retrained or refit for this note.

## Evidence

| Claim | Support | Setting | Seeds | Kind of evidence | Limit | Where to look |
|---|---|---|---|---|---|---|
| Additive training beats vanilla on this task | EXP-002, then EXP-006 | HARD-v2, noise 0.3, competitor rate 0.4, hidden 32, Adam, 400 updates, clip 5 | 17/29/43, then 101–223 with data base seed 2000 | Training | Not pooled across cohorts. EXP-001 used a different Python environment | `results/EXP-002/`, `results/EXP-006/`. Commits `f30b598`, `c351788` |
| A half-and-half mix does not keep that advantage | EXP-003 | Same task. Update is `0.5 * previous + 0.5 * candidate` | 17/29/43 | Training | This is not the additive model. The factor 0.5 also fades a pure copy | `results/EXP-003/`. Commit `b297b99` |
| A hard cap at 4 is mixed | EXP-004, EXP-006 | Additive state clamped to `[-4, 4]` | 17/29/43, then 101–223 | Training, then bit-flip replay | B was not tuned. EXP-004 weights were replayed in EXP-005, not saved originally | `results/EXP-004/`, `results/EXP-005/`, `results/EXP-006/` |
| Flipping the original bit can change the state even when the local gradient is a recorded zero | EXP-005, later audits | 512 pairs, split index 3, only the original bit flipped | Saved models from the runs above | Replay and frozen evaluation | A recorded 0.0 is a computational zero, not a proof of no dependence | `results/EXP-005/`. Commit `ae544c6` |
| Either input-side or recurrent-side weights, plus the readout, can be enough | EXP-008 | Additive model only. One group frozen at initialization | 101–223, base seed 2000 | Training | The two biases add into one preactivation. Not a necessity proof | `results/EXP-008/`. Commit `c252b44` |
| Bias-only learning is enough on some seeds and not others; matrix learning was enough on all tested seeds | EXP-009 | One bias trained, or both matrices trained, readout trained | Same cohort. Delay 128 was finished in a later authorized continuation | Training | Original session was interrupted. That attempt’s duration is unknown | `results/EXP-009/`, `results/EXP-009/delay128_completion/`. Commits `7fb416c`, `4f239bd` |
| Full-history training is more reliable than final-16 training for the original readout | EXP-010, repeated under clipping in EXP-011 and on new seeds in EXP-014 | K = 16. State is carried forward, then detached once | 101–223, base seed 2000; EXP-014 uses 307–359, base seed 3000, delay 128 only | Training | Late updates can still change early processing later, because the weights are shared. Not a proof that no other budget could work | `results/EXP-010/`, `results/EXP-011/`, `results/EXP-014/`. Commits `acce854`, `10fd5ed`, `0b1722c` |
| Turning clipping off does not by itself explain the final-16 gap | EXP-011 | Same model, clip 5 or no rescaling | 101–223, base seed 2000 | Training | At delay 128, unclipped full-history training itself became less consistent (7/10) | `results/EXP-011/` |
| Many final-16 states are linearly readable even when the original readout misses | EXP-012 on the old cohort; EXP-014 on a new cohort | Fixed L2 logistic fit, C = 1, lbfgs, training data only | Old: 101–223. New: 307–359, delay 128, clip 5 | Diagnostic refit, not new recurrent training | The fit changes scaling, solver, regularization, precision, and stopping together. See the denominator note below | `results/EXP-012/`, `results/EXP-014/`. Commits `b59adc3`, `0b1722c` |
| Never-trained additive states are already enough on some seeds under that same fit | EXP-013, and the initialized arm of EXP-014 | EXP-007 readout-only weights for the old cohort; a saved initial model for the new cohort | Old delay 128: 5/10. New delay 128: 4/10 | Diagnostic refit | Not pooled. Below 0.95 is not “no information” | `results/EXP-013/`, `results/EXP-014/`. Commit `515ebdf` |

## What is not the same measurement

Information in the state means the final hidden state still changes when only the original bit changes. Linear readability means the fixed diagnostic classifier, fit on training states only, can score the test set. Original readout performance is the head that was trained jointly, or left at initialization. Diagnostic refitting is an extra fitting procedure. It is not an equal-budget substitute for end-to-end training. Long-horizon feedback means the terminal loss is sent back through the whole sequence. A computational zero in that training graph means the cutoff blocked the early gradient. It does not mean the forward state forgot the bit.

## EXP-014 rescue count

The saved EXP-014 metrics, re-read for this note, say:

- Final-16 original readout at or above 0.95: 1 of 10 seeds. That seed is 311.
- Final-16 diagnostic readout at or above 0.95: 8 of 10 seeds.
- Original misses: 9.
- Of those 9, the diagnostic readout reached 0.95 for 7.
- Two original misses stayed below 0.95: seeds 331 (0.848 original, 0.945 diagnostic) and 353 (0.887 original, 0.922 diagnostic).

Rescued seeds: 307, 313, 317, 337, 347, 349, 359.

Seed 311 was already successful with the original readout. It is one of the 8 diagnostic successes. It is not a rescue.

“8 of 10 after refitting” and “7 of 9 misses rescued” answer different questions. The frozen EXP-014 analysis was not edited. The clarification is appended in the project ledger.

EXP-012’s “36 of 37” covers both delays and both clipping settings on the older seeds. It is not the same estimate as EXP-014’s “7 of 9.” The closest saved subgroup is EXP-012, delay 128, final-16, clipping at 5, seeds 101–223, data base seed 2000: 10 original misses, 10 rescued, none still below 0.95. That subgroup is a different cohort. The two fractions are not pooled.

## Current interpretation

Additive recurrence has a repeated performance advantage under this synthetic task and this training recipe. The additive rule is `h_t = h_(t-1) + tanh(...)`. It is not the half-and-half mixture, which failed, and it is not the clamped version, which was mixed.

Some initialized additive dynamics already support a successful prediction with the fixed diagnostic readout. Learning improves task-useful linear information on many seeds. Final-16 learning can improve that representation even though the training gradient does not flow back to the original event. Full-history feedback improves reliable performance of the jointly trained readout under the tested schedule.

Many final-16 failures of that joint readout are rescuable by the fixed diagnostic fit. Some are not. The rescue does not isolate one optimizer defect: the diagnostic changed scaling, solver, regularization, precision, and stopping together.

A score below 0.95 is not total forgetting. A computational-zero local gradient is not proof of no finite dependence. Matching success counts are not proof that two representations are the same.

## Limits

The task is one delayed bit. The effective distractor noise is 0.3 and the competitor rate is 0.4. Hidden size, learning rate, batch size, and the 400-update budget stayed fixed. Delays 64 and 128 were trained separately, not by stretching one trained model. Seeds 101–223 and base seed 2000 were reused for several exploratory follow-ups. EXP-014 is the independent new-cohort replication. Diagnostic refits are not new recurrent-training seeds, and they are not equal-budget controls. Shared weights let a late update change earlier processing on the next batch. Thresholded success is not the same as continuous accuracy. EXP-007 is missing a complete fingerprint file; those rows were not invented. The interrupted EXP-009 attempt has no recorded finish time. None of this is a proof of the broader CreditLab theory or a new architecture.

## Three open questions

1. **Why do a few final-16 states stay below 0.95 for this one classifier?** Seeds 331 and 353 on the new cohort, and delay-128 unclipped seed 179 on the old cohort, improved or stayed partial without crossing the line. The bit-flip checks still showed dependence. What is unknown is whether another preregistered readout, fixed before looking, would read those states, or whether the linear representation is genuinely weaker there. That distinction changes whether the remaining failures are readout limits or representation limits.

2. **How much of the full-history joint-training advantage is long-horizon feedback, and how much is the size of the update?** With clipping, full-history training was 10 of 10 at delay 128 and final-16 training was 0 of 10 on the original readout. Without clipping, full-history training fell to 7 of 10 and final-16 training stayed at 0 of 10. A future comparison would need to separate horizon from update size in one preregistered design. That changes whether the mechanism claim is about credit flowing back to the event or about optimization.

3. **How often is the untrained additive state already enough?** The reused cohort was 5 of 10 at delay 128. The new cohort was 4 of 10. Ten seeds leave a wide range. A larger new cohort, with the same classifier declared in advance, would estimate that rate. It would change whether recurrent learning is usually needed for this readout or often unnecessary.

No runner for those questions is included here. EXP-015 has not been authorized.

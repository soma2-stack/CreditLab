# Task List

## Phase 0 — repository foundation

- [x] Create the standalone CreditLab repository and organized folders.
- [x] Add project scope, safety, and reproducibility guidance.
- [x] Add deterministic CPU seed utilities.
- [x] Add a small experiment JSONL logger.
- [x] Add a tiny vanilla RNN baseline module.
- [x] Add quick foundation tests and run them on CPU.
- [ ] Check installation in a clean environment.

## Phase 1 — vanilla-RNN baseline review

- [x] Implement the synthetic delayed-bit generator and original EASY/HARD tasks.
- [x] Complete and freeze EXP-001; preserve its results.
- [x] Diagnose whether HARD competitors correlate with the target.
- [x] Add corrected HARD-v2 with independent competitor bits.
- [x] Add competitor-conditioned accuracy and per-example early-event loss-gradient diagnostics.
- [x] Re-run EASY, original HARD, and HARD-v2 at delays 16, 32, 64, 128, and 256 with seeds 17, 29, and 43.
- [x] Review hidden-state retention, early-event sensitivity, loss gradients, and seed variation.
- [x] Document EXP-001 and EXP-001B separately.

## Phase 2 — one residual-path test

- [x] Preregister one change: a fixed identity skip around the tanh recurrent update, scale 1.0, chosen before any EXP-002 test result.
- [x] Train the vanilla RNN and the residual RNN on corrected HARD-v2 at delay 64, with seeds 17, 29, and 43 and the EXP-001B training budget.
- [x] After those runs were saved, repeat the same pair at delay 128.
- [x] Record accuracy, loss, retention, event sensitivity, loss gradient, step gain, and numerical stability.
- [x] Leave EXP-001 and EXP-001B result files unchanged.
- [x] Record that those frozen runs used the unrecovered Python 3.12 environment, and that EXP-002 conclusions use the vanilla and residual runs from the same Python 3.11 / PyTorch 2.13 process.

## Phase 3 — bounded mixture test

- [x] Preregister one change: a fixed 0.5/0.5 mix of the previous state and the tanh candidate, chosen before any EXP-003 test result.
- [x] Train the vanilla RNN and the bounded mixture on corrected HARD-v2 at delays 64 and 128, with seeds 17, 29, and 43 and the EXP-002 training budget.
- [x] Record accuracy, loss, retention, credit diagnostics, hidden-state size, and numerical stability.
- [x] Compare state size with the saved EXP-002 additive residual without rerunning EXP-002.
- [x] Leave EXP-001, EXP-001B, and EXP-002 result files unchanged.

## Phase 4 — fixed-bound additive test

- [x] Preregister B = 4, the three model equations, the 0.95 success rule, and the clipping definition before any EXP-004 accuracy result.
- [x] Train matched vanilla, additive, and bounded-additive models at delay 64 on corrected HARD-v2, seeds 17, 29, and 43.
- [x] Save the delay-64 summary before delay 128. Run delay 128 only because the additive control had at least two successes.
- [x] Record accuracy, retention, early-event diagnostics, clipping before the clamp, magnitudes, and stability.
- [x] Leave EXP-001 through EXP-003 artifacts unchanged and check their hashes.

Do not start another experiment from EXP-004. Do not retune B.

## Phase 5 — original-bit flip audit

- [x] Preregister recovery tolerances and the counterfactual pair rule before any audit result.
- [x] Replay the eighteen EXP-004 models, verify them against saved metrics, and save the recovered weights outside `results/EXP-004/`.
- [x] Measure whether flipping only the original bit changes the final state, the boundary sign pattern, the logit, and the prediction.
- [x] Compare those finite changes with the recorded local bit gradient.
- [x] Leave EXP-001 through EXP-004 artifacts unchanged.

Do not start EXP-006 from this audit. Do not add a model. Do not retune B.

## Phase 6 — ten-seed replication

- [x] Preregister seeds 101 through 223, base seed 2000, and the additive gate of 8 out of 10 before any EXP-006 accuracy.
- [x] Train matched vanilla, additive, and bounded models at delay 64, then fresh delay-128 runs because the gate passed.
- [x] Run the bit-flip audit on the completed models.
- [x] Keep historical seeds out of the primary counts and leave EXP-001 through EXP-005 artifacts unchanged.

Do not start EXP-007. Do not tune B. Do not add a model.

## Phase 7 — readout-only comparison

- [x] Preregister full training versus readout-only training before any EXP-007 accuracy.
- [x] Train all six conditions at delay 64, then fresh delay-128 runs because full additive succeeded on 10 of 10 seeds.
- [x] Check that frozen recurrent weights and hidden states stayed unchanged.
- [x] Run the bit-flip audit on the completed conditions.
- [x] Leave EXP-001 through EXP-006 artifacts unchanged.

Do not start EXP-008. Do not tune B. Do not change the forward equations.

## Phase 8 — additive parameter groups

- [x] Clarify that EXP-007 froze both input-side and recurrent-side weights, and record the interrupted-run execution note without editing saved EXP-007 measurements.
- [x] Preregister the four additive regimes before any EXP-008 accuracy.
- [x] Train all four regimes at delay 64, then fresh delay-128 runs because full training succeeded on 10 of 10 seeds.
- [x] Run the bit-flip audit and confirm frozen parameters stayed fixed.
- [x] Leave EXP-001 through EXP-007 artifacts unchanged.

Do not start EXP-009. Do not tune B. Do not change the forward equation.

## Phase 9 — bias versus matrix adaptation

- [x] Preregister one trainable preactivation bias, `b_x`, and the four additive regimes before any EXP-009 accuracy.
- [x] Train all four regimes at delay 64. Full training succeeded on 10 of 10 seeds.
- [x] Run the delay-64 bit-flip audit on the 40 saved checkpoints. This was a diagnostic continuation, not a training resume.
- [x] Finish the planned delay-128 comparison through an authorized continuation. The original session remains interrupted. Seed 101 bias-plus-readout was restarted once from initialization, and the other 37 conditions were trained once.
- [x] Audit all 40 delay-128 models, including the two reused checkpoints.
- [x] Leave EXP-001 through the earlier EXP-009 training and audit files unchanged.

Do not start EXP-010. Do not subdivide parameter groups further. Do not tune B. Do not change the forward equation.

## Phase 10 — final-16 gradient cutoff

- [x] Preregister K = 16 and the two additive training regimes before any EXP-010 accuracy.
- [x] Train full-history and final-16 regimes at delay 64, then fresh delay-128 runs because full-history training succeeded on 10 of 10 seeds.
- [x] Confirm that the final-16 training graph blocks the original event, and keep that measurement separate from the full forward pass.
- [x] Run the bit-flip audit on the completed models.
- [x] Leave EXP-001 through EXP-009 artifacts unchanged.

Do not try another horizon. Do not build another model. Do not start EXP-011.

## Phase 11 — horizon and clipping

- [x] Preregister the four conditions, with K = 16 and clipping either at norm 5 or off, before any EXP-011 accuracy.
- [x] Train all four conditions at delay 64, then fresh delay-128 runs because clipped full-history training succeeded on 10 of 10 seeds.
- [x] Record raw gradient norms, whether clipping was applied, and the actual Adam update size.
- [x] Run the bit-flip audit on the completed models.
- [x] Leave EXP-001 through EXP-010 artifacts unchanged.

Do not try another horizon. Do not tune a clipping threshold. Do not build another model. Do not start EXP-012.

## Phase 12 — frozen readout refit

- [x] Preregister the fixed L2 logistic classifier before any refitted accuracy.
- [x] Verify all 80 EXP-011 checkpoints against their saved test counts.
- [x] Fit one classifier per checkpoint on training hidden states only, and compare it with the original readout.
- [x] Run the bit-flip check on the original and refitted readouts using the same frozen states.
- [x] Leave EXP-001 through EXP-011 artifacts unchanged.

Do not train recurrent weights. Do not tune C. Do not try another horizon. Do not start EXP-013.

## Phase 13 — initialized additive states

- [x] Preregister the three frozen representations and the unchanged EXP-012 classifier.
- [x] Verify the EXP-007 readout-only weights against a freshly seeded model, with no optimizer step.
- [x] Confirm that the full-history and final-16 refits reproduce EXP-012.
- [x] Compare diagnostic accuracy and the bit-flip check across the three states.
- [x] Leave EXP-001 through EXP-012 artifacts unchanged, including the incomplete EXP-007 fingerprint file.

Do not train recurrent weights. Do not tune C. Do not build a model. Do not start EXP-014.

## Phase 14 — new-seed replication

- [x] Confirm the new seeds were unused, and preregister them with the unchanged training and classifier rules.
- [x] Record the EXP-010 through EXP-013 interpretation before any EXP-014 accuracy.
- [x] Train full-history and final-16 models at delay 128, and keep the untrained initial state.
- [x] Fit the fixed diagnostic classifier and run the bit-flip check.
- [x] Leave EXP-001 through EXP-013 artifacts unchanged.

Do not build a model. Do not change the task distribution. Do not tune the classifier or the horizon. Do not start EXP-015.

## Checkpoint — saved-record audit

- [x] Write `SCIENTIFIC_CHECKPOINT.md` from the saved EXP-001 through EXP-014 records.
- [x] Correct the EXP-014 rescue denominator in new documentation: 7 of 9 original misses, not 8 of 10.
- [x] Leave every saved experiment artifact unchanged, including `results/EXP-014/analysis.md`.
- [x] Record that no experiment is running and EXP-015 is not authorized.

Do not start EXP-015. Do not train, refit, or evaluate another model.

## Phase 15 — frozen length generalization

- [x] Preregister evaluation of the saved EXP-014 models at delays 128, 256, and 512 before any new score.
- [x] Verify the 30 checkpoints and 30 classifiers against their saved correct counts.
- [x] Score the five frozen endpoints on fresh data, with no training and no refitting.
- [x] Record bit-flip results and hidden-state magnitude.
- [x] Leave EXP-001 through EXP-014 artifacts, and the scientific checkpoint, unchanged.

Do not test delay 1024. Do not retrain or refit. Do not start EXP-016.

## Phase 16 — stronger distractor noise

- [ ] Confirm the new seeds are unused, and preregister the four matched conditions before any EXP-016 accuracy.
- [ ] Train vanilla and additive models at delay 128, at distractor noise 0.3 and 1.0, on the new seeds.
- [ ] Fit the fixed diagnostic classifier separately at each noise level.
- [ ] Run the bit-flip audit on the completed models.
- [ ] Leave EXP-001 through EXP-015 artifacts, and the scientific checkpoint, unchanged.

Do not raise the training budget. Do not try another noise level. Do not start EXP-017.

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

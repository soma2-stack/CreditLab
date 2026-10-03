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

## Phase 4 — not started

- [ ] Write down one way to keep a full-strength residual copy while bounding state size, before looking at any new test result.
- [ ] Only after that design is fixed, test it against the frozen EXP-003 vanilla control and the frozen EXP-002 additive result.

Do not start that follow-up as part of EXP-003. Do not retune the 0.5 weights.

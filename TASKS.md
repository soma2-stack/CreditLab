# Task List

## Phase 0 — repository foundation

- [x] Create the standalone CreditLab repository and organized folders.
- [x] Add project scope, safety, and reproducibility guidance.
- [x] Add deterministic CPU seed utilities.
- [x] Add a small experiment JSONL logger.
- [x] Add a tiny vanilla RNN baseline module.
- [x] Add quick foundation tests and run them on CPU.
- [ ] Review dependency setup on a clean environment.

## Phase 1 — detect delayed learning credit

- [ ] Implement the synthetic delayed-bit task with controllable delay and distractor difficulty.
- [ ] Train the tiny standard RNN baseline using the frozen EXP-001 config.
- [ ] Measure loss/accuracy and gradient magnitude as a function of delay.
- [ ] Measure relevant hidden-state retention and simple recurrent contraction indicators.
- [ ] Define and evaluate simple credit-difficulty signals without using task IDs.
- [ ] Review whether the baseline results justify a minimal adaptive-credit prototype.
- [ ] If justified, compare that prototype with matched standard-RNN and truncated-credit baselines.

No large-scale training is scheduled.
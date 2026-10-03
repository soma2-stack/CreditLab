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

## Phase 2 — next experiment design only

- [ ] Preregister one small EXP-002 comparing the vanilla baseline with a simple way to send learning feedback to the original event, at delay 64 on HARD-v2.
- [ ] Implement or run EXP-002 only after that design is reviewed.

No EXP-002 run or new architecture is part of this work.

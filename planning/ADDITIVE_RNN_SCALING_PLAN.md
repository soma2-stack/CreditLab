# Proposed additive-RNN scaling study — preparation only

**Status:** PLAN ONLY — no training authorized or started. This document does not supersede the EXP-023 stop recommendation. **No EXP-024 designation** is assigned.

## Motivation and verified starting evidence

- EXP-011: clipped full-history additive training passed 10/10 seeds at delay 128; clipped final-16 training passed 0/10. Full-history without clipping passed 7/10.
- EXP-015: among ten frozen full-history additive models trained at delay 128, original readout passed 10/10 at 128, 8/10 at 256 and 6/10 at 512 without retraining. Longer sequences also increase distractors.
- EXP-020: clean-control whole-state reset solved single-memory replacement but is **not** a suitable selective two-memory solution. Preserve this as a separate control, not the main scaling target.
- EXP-022/023: two-memory replacement/readout remains unsuccessful; do not interpret the scaling study as resolving it.

## Scientific question

Does increasing recurrent width improve the *out-of-training-delay* accuracy of the previously successful full-history additive RNN, at what compute/memory cost, and how does that compare to matched vanilla and GRU baselines? This is a finite-size study, **not** a direct test of the theory's asymptotic learning-credit dimension D.

## Preregistered proposed matrix (requires approval before execution)

- Primary architecture: reproduce EXP-011 full-history additive recurrence, clipping norm 5, original trained readout.
- Widths: 16, 32 (historical reference), 64, 128. Keep task, data generator and optimization schedule fixed except for documented width effects.
- Training delay: 128 only. Evaluation delays: 128, 256, 512, 1024; no retraining or readout refit at evaluation delays.
- Seeds: ten fixed seeds 307, 311, 313, 317, 331, 337, 347, 349, 353, 359. Separate training and evaluation data RNG streams; prevent overlap with historical test data.
- Controls: matched-width vanilla RNN and GRU with equal training data and budget; report both parameter counts and measured resource cost rather than claiming perfect parameter matching.
- Primary endpoint: fraction of seeds reaching >=0.95 test accuracy at each delay; include per-seed accuracy and confidence intervals. Do not select checkpoints based on test results.
- Secondary: state norms, bit-flip sensitivity, per-example gradient diagnostics, wall time, peak RAM/VRAM, parameter count, forward/backward FLOP estimates and numerical stability.
- Important confound: greater delay means more distractors. An optional follow-up can isolate time from distractor count, but must be separately preregistered.

## Preparation / execution gates

1. Verify existing experiment entrypoints, config schema, saved checkpoint loading and baseline parity with EXP-011/015 **before writing or running new training code**.
2. Implement dry-run configuration validation and deterministic unit tests. Tests must not initiate training.
3. Record exact git revision, environment, seeds, dataset fingerprint, and complete configs. Preserve all EXP-001–023 outputs and historical logs unchanged.
4. Before execution, obtain explicit authorization, available compute device, time/VRAM budget and safety limits. Start with one small pilot only after approval.
5. Stop on NaN/Inf, OOM, crashes or unsafe temperature. Never silently change batch size, training steps or optimizer settings.
6. Compare observed finite-size trends to the theory separately; do not claim a proof of D=Omega(n) or sub-n^(3/2) scaling from classification accuracy.

## Current state

Planning document only. No experiment started, no checkpoints generated, no result files changed, no theoretical claim validated.

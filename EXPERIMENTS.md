# Experiment Registry

Registry entries describe completed experiments or plans. Every run saves its exact configuration, seeds, code revision, measurements, and compute use under `results/<experiment_id>/`.

## EXP-001 — Delayed-bit vanilla RNN baseline (completed, frozen)

- **Question:** How do task performance and available learning signal change as the delay between a relevant bit and the final prediction grows?
- **Baseline:** Small vanilla tanh RNN trained with a terminal binary prediction loss.
- **Dataset/task:** Original synthetic delayed-marked-bit task with EASY and HARD modes.
- **Seeds / configuration:** Seeds 17, 29, and 43; frozen config revision 2. The completed run includes delays through 256.
- **Result:** Completed. EASY remains successful through delay 32, then shows a seed split at 64 and is near chance by 128. Original HARD stays above chance at long delays, but EXP-001B found a target-correlated competitor shortcut in this generator.
- **Interpretation:** Keep original EXP-001 measurements unchanged. See `results/EXP-001/analysis.md`.
- **Follow-up:** EXP-001B checks the HARD shortcut and the vanilla-RNN failure region.

## EXP-001B — HARD shortcut and credit diagnostic follow-up (completed)

- **Question:** Does the original HARD task leak target information through competitors, why can the old early-event gradient be tiny, and where does the baseline begin failing?
- **Baseline:** Same vanilla tanh RNN and training settings as EXP-001.
- **Change tested:** Added corrected HARD-v2 with competitor bits independent of the target; measured accuracy by first competitor agreement/disagreement; added per-example loss-gradient norms while retaining the original event-sensitivity metric.
- **Delays / modes:** 16, 32, 64, 128, and 256; EASY, original HARD, and HARD-v2.
- **Seeds / compute:** 17, 29, and 43; 400 Adam updates per run, 45 runs, 0 numerical failures, 273 seconds summed model runtime. No tuning.
- **Result:** Original HARD competitors agree with the target about 75% of the time, versus about 50% in HARD-v2. At delay 256, mean accuracy is 0.723 on original HARD and 0.491 on HARD-v2. EASY and HARD-v2 show a one-success/two-failure split at delay 64.
- **Interpretation:** The original HARD long-delay advantage is partly a shortcut. Failed EASY/HARD-v2 runs generally lack a linearly readable target in the final hidden state, and have very small early-event sensitivity and loss gradients. Loss gradients can also be small on already-correct examples, so they are not a standalone failure detector.
- **Files:** Frozen config at `configs/exp001b_hard_shortcut_credit.yaml`; raw and aggregated records plus analysis under `results/EXP-001B/`.
- **Follow-up:** Design only, not run: test whether one simple way to send learning feedback to the original event helps at delay 64 on corrected HARD-v2.

## EXP-002 — Focused early-event feedback test (proposed; not run)

- **Question:** At delay 64 on HARD-v2, does a simple additional learning signal to the original event improve success across seeds over the matched vanilla RNN?
- **Baseline:** EXP-001B vanilla RNN on HARD-v2, using seeds 17, 29, and 43.
- **Change tested:** To be selected and preregistered as one minimal intervention; no architecture or implementation has been selected.
- **Metrics:** Test accuracy, original-event hidden-state retention, event sensitivity, and per-example loss-gradient norm.
- **Stopping / compute:** Freeze after the design is reviewed; match EXP-001B training settings and CPU budget.
- **Result:** Not run.

## EXP-003 — Minimal adaptive-credit prototype (unstarted)

- **Question / design:** Not yet specified. Do not begin until a focused EXP-002 result justifies it.
- **Result:** Not run.

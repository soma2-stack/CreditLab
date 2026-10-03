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
- **Follow-up:** EXP-002 tested one fixed residual state path on HARD-v2. EXP-001B files were not changed.

## EXP-002 — Fixed residual recurrent path (completed)

- **Question:** Does a direct residual copy through recurrent state make success at delay 64 on corrected HARD-v2 more reliable across seeds than the vanilla tanh RNN?
- **Hypothesis:** The vanilla failure happens because state and learning signal must repeatedly pass through the tanh recurrent map. A fixed identity skip around that map should make delay-64 success more reliable.
- **Baseline:** The same vanilla tanh RNN, hidden size 32, Adam settings, data sizes, seeds 17, 29, and 43, and diagnostics as EXP-001B. The vanilla arm was retrained in this run; frozen EXP-001 and EXP-001B files were not changed.
- **Change tested:** One residual recurrence, `h_t = h_{t-1} + candidate_t`, with fixed scale 1.0 chosen before any EXP-002 test result. Same parameter count and initialization family. No LSTM, GRU, attention, external memory, or extra labels.
- **Dataset/task:** Corrected HARD-v2 only, through the unchanged EXP-001B generator. Primary delay 64. Confirmation delay 128 only after the delay-64 records were saved.
- **Metrics:** Test accuracy, train and validation loss, hidden-state retention probe, event sensitivity, per-example loss gradient, step-to-step gain, and numerical stability. Residual runs also record the size of the tanh candidate and the hidden state.
- **Success rule, locked beforehand:** Accuracy of at least 0.75 counts as success. Probe R^2 of at least 0.5 counts as clearly readable retention.
- **Seeds / compute:** 17, 29, and 43; 400 updates; 12 runs; 0 numerical failures; 64.7 seconds wall time. Python 3.11.9, PyTorch 2.13.0+cpu. Code revision `b17fa60c8e3c`. The scale was not tuned.
- **Result:** At delay 64, vanilla accuracies were 0.4961, 0.4570, and 1.0000. Residual accuracies were 1.0000, 1.0000, and 1.0000. Residual probe R^2 was about 0.98 on every seed; vanilla probe R^2 was near zero on the two failures and 0.995 on the success. At delay 128, vanilla accuracies were 0.5098, 0.5039, and 0.4590; residual accuracies were 1.0000 on all three seeds.
- **Interpretation:** Outcome A for this task. The direct path made success and retention reliable, and the smallest step-to-step gain stayed near 0.7–0.9 instead of near 0. The loss gradient stayed small because the residual answers were already correct. The residual state also grew to about one unit per step, so state size changed along with the path. This does not prove the broader theory or establish an architecture.
- **Files:** `configs/exp002_residual_recurrent.yaml` and `results/EXP-002/`.
- **Follow-up:** Not run. The next experiment should test one preregistered bounded version of this same skip, so a direct path can be separated from unbounded state growth.

## EXP-003 — Next mechanism check (unstarted)

- **Question:** Does one bounded version of the same residual copy still make delay-64 HARD-v2 success reliable when the hidden state cannot grow with the delay?
- **Baseline:** Frozen EXP-002 vanilla and scale-1 residual results. Do not retune the scale-1 weight.
- **Change tested:** Not selected yet. One bound must be written down before any new test result.
- **Result:** Not run. Do not start it from the EXP-002 run automatically.

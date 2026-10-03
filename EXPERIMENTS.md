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
- **Baseline:** The vanilla tanh RNN retrained inside EXP-002, on the same machine process as the residual model. Hidden size 32, Adam settings, data sizes, seeds 17, 29, and 43, and diagnostics match EXP-001B. Frozen EXP-001 and EXP-001B files were not changed. Those frozen runs used the earlier Python 3.12 environment (EXP-001B records Python 3.12.14 and PyTorch 2.14.1+cpu). That interpreter could not be recovered, so EXP-002 used Python 3.11.9 and PyTorch 2.13.0+cpu for both models. Differences between the frozen files and this vanilla rerun are environment differences, not architecture effects.
- **Change tested:** One residual recurrence, `h_t = h_{t-1} + candidate_t`, with fixed scale 1.0 chosen before any EXP-002 test result. Same parameter count and initialization family. No LSTM, GRU, attention, external memory, or extra labels.
- **Dataset/task:** Corrected HARD-v2 only, through the unchanged EXP-001B generator. Primary delay 64. Confirmation delay 128 only after the delay-64 records were saved.
- **Metrics:** Test accuracy, train and validation loss, hidden-state retention probe, event sensitivity, per-example loss gradient, step-to-step gain, and numerical stability. Residual runs also record the size of the tanh candidate and the hidden state.
- **Success rule, locked beforehand:** Accuracy of at least 0.75 counts as success. Probe R^2 of at least 0.5 counts as clearly readable retention.
- **Seeds / compute:** 17, 29, and 43; 400 updates; 12 runs; 0 numerical failures; 64.7 seconds wall time. Both models: Python 3.11.9, PyTorch 2.13.0+cpu, code revision `b17fa60c8e3c`. The scale was not tuned. No second control run was required.
- **Result:** At delay 64, vanilla accuracies were 0.4961, 0.4570, and 1.0000. Residual accuracies were 1.0000, 1.0000, and 1.0000. Residual probe R^2 was about 0.98 on every seed; vanilla probe R^2 was near zero on the two failures and 0.995 on the success. At delay 128, vanilla accuracies were 0.5098, 0.5039, and 0.4590; residual accuracies were 1.0000 on all three seeds.
- **Interpretation:** Outcome A against the matched Python 3.11 vanilla control. The direct path made success and retention reliable, and the smallest step-to-step gain stayed near 0.7–0.9 instead of near 0. The loss gradient stayed small because the residual answers were already correct. The residual state also grew to about one unit per step, so state size changed along with the path. This does not prove the broader theory or establish an architecture. Frozen-file differences are not part of this conclusion.
- **Files:** `configs/exp002_residual_recurrent.yaml` and `results/EXP-002/`.
- **Follow-up:** EXP-003 tested one fixed half-and-half mixture. EXP-002 files were not changed.

## EXP-003 — Bounded half-and-half mixture (completed)

- **Question:** Did EXP-002’s direct path help because it preserves information through time, or did that result depend on hidden-state magnitude growing much larger?
- **Hypothesis:** A fixed mix, `h_t = 0.5 h_{t-1} + 0.5 candidate_t`, keeps a direct path while preventing additive growth. If it still beats the matched vanilla RNN, the direct path itself is useful here. If it does not, magnitude or the additive update was an important part of EXP-002.
- **Baseline:** Vanilla tanh RNN trained in the same Python 3.11.9 / PyTorch 2.13.0+cpu process. Same hidden size, data, optimizer, learning rate, update budget, seeds, and diagnostics as EXP-002.
- **Change tested:** One bounded mixture with both weights fixed at 0.5 before any EXP-003 test result. No second mechanism. EXP-002 was not rerun; its saved state sizes were copied for comparison.
- **Dataset/task:** Corrected HARD-v2. Delays 64 and 128. Seeds 17, 29, and 43.
- **Rules locked beforehand:** Success is test accuracy of at least 0.75. Readable retention is probe R^2 of at least 0.5. Controlled magnitude is a largest absolute hidden entry of at most 1.01.
- **Seeds / compute:** 12 runs, 0 numerical failures, 111.9 seconds. Code revision `ec69be58e3c8`. Weights were not tuned.
- **Result:** Delay-64 vanilla accuracies were 0.4961, 0.4570, and 1.0000. Bounded accuracies were 0.5293, 0.4805, and 0.4688. Delay-128 vanilla accuracies were 0.5098, 0.5039, and 0.4590. Bounded accuracies were 0.4902, 0.5098, and 0.4805. Bounded probe R^2 stayed near zero. Largest absolute hidden values stayed between about 0.41 and 0.83, versus about 65 and 129 for the saved EXP-002 additive model.
- **Interpretation:** The EXP-002 advantage did not survive this bound. The half-and-half model kept state size controlled and did not preserve the bit or beat the matched vanilla model at either delay. A factor of 0.5 each step also fades a pure copy of the original state, so this does not isolate magnitude from a full-strength copy. It does not prove the broader theory or establish an architecture.
- **Files:** `configs/exp003_bounded_mixture.yaml` and `results/EXP-003/`.
- **Follow-up:** Not run. The next question is whether a full-strength copy can be kept while a single preregistered bound stops the state from growing with the delay.

## EXP-004 — Fixed-bound additive recurrence (preregistered; training not started)

- **Question:** Does the additive update keep its delayed-learning advantage when one fixed clamp limits hidden-state size, without shrinking the previous state on every step?
- **Models:** Vanilla `h_t = candidate_t`; additive `h_t = h_{t-1} + candidate_t`; bounded additive `h_t = clamp(h_{t-1} + candidate_t, -4, 4)`. `B = 4` is fixed before any EXP-004 accuracy result and is not learned.
- **Task:** Corrected HARD-v2 with the effective settings stated explicitly: independent competitor bits, competitor rate 0.4, distractor noise standard deviation 0.3. Historical YAML files are not edited.
- **Matching:** For each seed and delay, all three models share dataset tensors, initial parameters, minibatch indices, and the training step. Fingerprints are saved.
- **Success rule:** Final-checkpoint test accuracy at least 0.95. Delay 128 is fresh training and runs only if the additive control succeeds on at least two of three delay-64 seeds.
- **Active clipping convention:** At least 1 percent of validation hidden coordinates are outside [-4, 4] before the clamp at the final checkpoint. The continuous fraction is always saved.
- **Result:** Not collected. Do not interpret this entry as an outcome.

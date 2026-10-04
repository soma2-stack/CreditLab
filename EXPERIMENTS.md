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

## EXP-004 — Fixed-bound additive recurrence (completed)

- **Question:** Does the additive update keep its delayed-learning advantage when one fixed clamp limits hidden-state size, without shrinking the previous state on every step?
- **Models:** Vanilla, additive `h_t = h_{t-1} + candidate_t`, and bounded additive `h_t = clamp(h_{t-1} + candidate_t, -4, 4)`. `B = 4` was fixed before any EXP-004 accuracy result.
- **Task:** Corrected HARD-v2 with explicit noise standard deviation 0.3 and competitor rate 0.4. Historical configuration files were not edited. Three models in each seed and delay shared data, initialization, and minibatches.
- **Rules:** Success is final test accuracy of at least 0.95. Delay 128 ran because the additive control succeeded on 3 of 3 delay-64 seeds. Active clipping means at least 1 percent of validation coordinates were outside [-4, 4] before the clamp. Python 3.11.9, PyTorch 2.13.0+cpu, training code `14bdc4786549`.
- **Result:** Eighteen runs, zero failures. Delay-64 accuracy was vanilla 0.4961, 0.4570, 1.0000; additive 1.0000 on all three seeds; bounded 1.0000, 0.5098, 0.5195. Delay-128 accuracy was vanilla 0.5098, 0.5039, 0.4590; additive 1.0000 on all three seeds; bounded 1.0000, 0.5098, 0.6289. Bounded probe R^2 was about 0.996 and 0.997 on seed 17 and about zero on the other delay-64 seeds. Clipping fraction on the bounded test rollouts was about 0.71 to 0.94. Stored bounded states were capped at 4. Additive states reached about 65 and 129.
- **Interpretation:** Mixed, so inconclusive. The additive control reproduced. Clipping was active even on the bounded success, so growth to 65–129 was not required for that one seed. The same bound did not succeed on the other seeds. Co-occurrence of saturation and failure is not treated as proof that clipping caused the failure. This is not an architecture result and it does not validate the broader theory.
- **Files:** `configs/exp004_bounded_additive.yaml` and `results/EXP-004/`.
- **Follow-up:** Not started. Do not change B from these results, and do not start another experiment from this run.

## EXP-005 — Counterfactual early-bit retention audit (completed)

- **Question:** When later inputs are held fixed, does flipping only the original event bit change the final hidden state and the prediction?
- **Models:** Replayed EXP-004 vanilla, additive, and bounded-additive models. No new architecture. `B` stayed 4. EXP-004 had saved no checkpoints.
- **Recovery:** All 18 replays matched the saved EXP-004 correct counts, losses, retention, and hidden magnitudes within the pre-registered tolerances. Python 3.11.9, PyTorch 2.13.0+cpu. Audit code `3ebc12f0e389`.
- **Fresh data:** 512 sequences per seed and delay, split index 3. Only the original bit differed within a pair.
- **Result:** Successful additive runs and bounded seed 17 changed state, boundary sign pattern, logit, and prediction when the bit flipped. Bounded seed 29 at both delays, and bounded seed 43 at delay 64, produced exactly identical states, identical boundary signs, identical logits, and recorded bit gradients of 0.0. Bounded seed 43 at delay 128 was mixed: 393 pairs were identical, and 119 pairs changed state, logit, and prediction despite a recorded bit gradient of 0.0 on every example.
- **Interpretation:** EXPERIMENTAL RESULT. Zero local sensitivity agreed with no finite change for some bounded failures, and missed a real finite change for part of bounded seed 43 at delay 128. Failed bounded models did not all forget in the same way. This is not an architecture result and it does not validate the broader theory.
- **Files:** `configs/exp005_counterfactual_audit.yaml` and `results/EXP-005/`.
- **Follow-up:** Not started.

## EXP-006 — Fresh-seed replication (completed)

- **Question:** How consistently do the unchanged additive and B=4 bounded models learn delayed memory across ten new seeds and fresh datasets, compared with matched vanilla controls?
- **Seeds:** 101, 113, 127, 139, 151, 163, 179, 191, 211, 223. Data base seed 2000. Not pooled with seeds 17, 29, and 43.
- **Models:** Unchanged EXP-004 vanilla, additive, and bounded-additive equations. `B` stayed 4.
- **Result:** All 60 training runs and 60 bit-flip audits finished in 529 seconds. No failures and no unstarted runs. Delay 64 successes were vanilla 3/10, additive 10/10, and bounded 6/10. The additive gate passed, so fresh delay-128 training ran. Delay 128 successes were vanilla 1/10, additive 10/10, and bounded 5/10. Paired counts at delay 64 were 6 both-succeed and 4 additive-only. Exact 95 percent intervals for the delay-64 success rates were about 0.07–0.65 for vanilla, 0.69–1.00 for additive, and 0.26–0.88 for bounded.
- **Audit:** Models that reached 0.95 accuracy also passed the bit-flip check. Some bounded misses still changed state on part of the pairs despite a recorded bit gradient of 0.0, notably delay-64 seed 139. Other bounded misses had exactly identical final states.
- **Interpretation:** EXPERIMENTAL RESULT. Additive success repeated on every new seed. Bounded success is not confined to the old seed 17, and it was less reliable than additive in the paired comparison. Ten seeds leave a wide interval. This is not an architecture result and it does not validate the broader theory.
- **Files:** `configs/exp006_replication.yaml` and `results/EXP-006/`.
- **Follow-up:** Not started.

## EXP-007 — Readout-only versus full training (completed)

- **Question:** Does additive delayed-memory success require learning the recurrent weights, or is the initial recurrent dynamics enough if only the readout is trained?
- **Change:** No new forward equation. `B` stayed 4. Each model was trained either fully or with input and recurrent weights frozen at initialization.
- **Result:** All 120 runs and 120 audits finished. Full additive succeeded on 10/10 seeds at both delays. Readout-only additive succeeded on 0/10 at both delays, with accuracy mostly about 0.80–0.93. Full bounded succeeded on 6/10 and 5/10. Readout-only bounded succeeded on 0/10 at both delays. Frozen recurrent weights and hidden states stayed unchanged in every readout-only run. The initial additive probe R^2 was about 0.53 at delay 64 and about 0.45 at delay 128, and it did not change when only the readout was trained.
- **Interpretation:** EXPERIMENTAL RESULT, outcome B. Under this schedule, learning the recurrent weights is needed for reliable additive success. Freezing did not repair bounded reliability. This is not an architecture result and it does not validate the broader theory.
- **Clarification added later:** The readout-only condition froze both the input-side parameters (`W_x`, `b_x`) and the recurrent-side parameters (`W_h`, `b_h`). The supported claim is that training the input/recurrent module improves reliable additive success under this schedule. It does not show that learning the hidden-to-hidden matrix by itself is necessary. Fewer successful seeds is also not the same as lower accuracy on every paired seed. On 18 of 60 paired comparisons, readout-only accuracy was higher than full training.
- **Files:** `configs/exp007_readout_only.yaml` and `results/EXP-007/`.
- **Follow-up:** EXP-008 separates input-side training from recurrent-side training. The EXP-007 analysis file was not rewritten.

## EXP-008 — Which parameter group enables additive learning? (preregistered; training not started)

- **Question:** Can reliable additive delayed learning happen while the hidden-to-hidden weights stay fixed, if the input-side weights and the readout are trained? Can the reverse, recurrent-side weights plus the readout, also succeed?
- **Model:** The existing additive equation only. Four regimes: full, readout only, input-side plus readout, and recurrent-side plus readout. `B` is not part of this comparison and is not tuned.
- **Seeds:** 101 through 223, data base seed 2000. Not pooled with earlier counts.
- **Gate:** Fresh delay-128 training runs only if full additive succeeds on at least 8 of 10 delay-64 seeds.
- **Budget:** Five minutes per run and 15 minutes total.
- **Result:** Not collected.

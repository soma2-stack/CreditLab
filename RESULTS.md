# Results Ledger

Chronological record. Append new entries; never overwrite old results. Distinguish measured observations from interpretations and note replication status.

## 2026-10-03 — Repository foundation

- **Experiment ID:** None (scaffolding only).
- **Configuration:** Python 3.11.9; PyTorch and pytest were available locally. Tests and smoke check were run with CUDA hidden and CPU tensors. No training or benchmark run was performed.
- **Baseline:** Not applicable.
- **Result:** Seven foundation tests passed, and the no-training CPU smoke check passed. Tests cover deterministic initialization, model shapes and CPU execution, input validation, and append-only JSONL logging.
- **Uncertainty:** A clean-environment dependency installation has not been checked; EXP-001 has not run.
- **Replicated:** Not applicable.
- **What changed:** Phase 0 foundation is ready; EXP-001 remains the next experiment.
## 2026-10-03 — EXP-001 completed (frozen)

- **Configuration:** `configs/exp001_delayed_credit_v2.yaml`; seeds 17, 29, and 43; CPU vanilla tanh RNN; completed delay sweep includes 256.
- **Result:** The saved EXP-001 analysis and measurements show EASY solving through delay 32, a seed split at delay 64, and near-chance EASY performance by delay 128. Original HARD remains above chance at long delays.
- **Interpretation:** The HARD result required follow-up because competitors might leak target information. EXP-001 result files were not modified during EXP-001B.
- **Details:** See `results/EXP-001/analysis.md`, `summary.csv`, `summary.json`, and `raw_metrics.jsonl`.

## 2026-10-03 — EXP-001B HARD shortcut and credit follow-up

- **Configuration:** `configs/exp001b_hard_shortcut_credit.yaml`; delays 16, 32, 64, 128, and 256; modes EASY, original HARD, HARD-v2; seeds 17, 29, and 43; same 32-unit vanilla tanh RNN, 400 Adam updates, learning rate 0.003, batch size 64. PyTorch 2.14.1+cpu, Python 3.12.14. Code revision for training records: `acd8bf3987f5`.
- **Result:** 45/45 runs completed with finite values. Original HARD's first competitor agreed with the target about 75% of the time; HARD-v2 measured about 50%. Mean accuracy at delay 256 was 0.723 for original HARD, 0.491 for HARD-v2, and 0.485 for EASY. At delay 64, EASY and HARD-v2 each had one successful seed and two near-chance seeds. EASY seed outcomes match EXP-001 (seed 29 succeeds; 17 and 43 fail).
- **Credit/retention:** Failed EASY and HARD-v2 runs generally show near-zero target-bit retention in the final hidden state, plus tiny event sensitivity and per-example loss gradients. The loss-gradient norm also becomes small when examples are already correct, so use it together with sensitivity, accuracy, and retention.
- **Interpretation:** EXP-001's long-delay HARD advantage is partly caused by a generator shortcut. A genuine vanilla-RNN difficulty remains on EASY and corrected HARD-v2 from delay 64 onward. This supports testing a way to improve learning feedback, but does not validate an adaptive-credit architecture.
- **Compute and deviations:** 273 seconds summed per-run model runtime. No tuning or retries. A summary-only empty-group handling issue was fixed after all runs; summaries were regenerated from the saved raw records, with no training repeated. Competitor-target agreement was computed afterward from the deterministic test examples and saved latent metadata.
- **Files:** `results/EXP-001B/analysis.md`, `raw_metrics.jsonl`, `summary.csv`, `summary.json`, `config_used.yaml`, and `environment.txt`.
- **Replication:** Three seeds per condition; this is limited replication on one synthetic setup.

## 2026-10-03 — EXP-002 fixed residual path

- **Experiment ID:** EXP-002.
- **Configuration:** `configs/exp002_residual_recurrent.yaml`; corrected HARD-v2; delays 64 then 128; seeds 17, 29, and 43; hidden size 32; 400 Adam updates; learning rate 0.003; batch size 64; 1,536 / 512 / 512 train, validation, and test examples. Residual rule `h_t = h_{t-1} + candidate_t` with fixed scale 1.0, chosen before the run and not tuned. Python 3.11.9, PyTorch 2.13.0+cpu. Code revision `b17fa60c8e3c`.
- **Baseline:** Vanilla tanh RNN trained in the same Python 3.11.9 / PyTorch 2.13.0+cpu process as the residual model. Frozen EXP-001 and EXP-001B result files were not modified. Those frozen runs belong to the earlier Python 3.12 environment. EXP-001B records Python 3.12.14 and PyTorch 2.14.1+cpu. That interpreter could not be recovered.
- **Result:** Twelve of twelve runs finished with finite values in 64.7 seconds. Conclusions use this matched pair. At delay 64 the vanilla accuracies were 0.4961, 0.4570, and 1.0000, and the residual accuracies were 1.0000, 1.0000, and 1.0000, with residual probe R^2 about 0.98 on every seed. At delay 128 the vanilla accuracies were 0.5098, 0.5039, and 0.4590, and the residual accuracies were 1.0000 on every seed. Numeric differences from the frozen Python 3.12 files are environment differences, not architecture effects.
- **Retention and credit:** Failed vanilla runs kept probe R^2 near zero and event sensitivity near zero. Residual runs kept the bit readable and kept the smallest one-step gain around 0.7–0.9, versus about 0.002–0.02 for the vanilla RNN. The loss gradient stayed tiny on residual runs because those answers were already correct. Residual states grew to a maximum absolute value of about 65 at delay 64 and about 129 at delay 128. About 9–16% of residual updates were gradient-clipped. No NaN or Inf.
- **Interpretation:** EXPERIMENTAL RESULT, outcome A for this toy task. A fixed direct copy made delayed success reliable and preserved a readable state. State magnitude also changed, so the result does not isolate the copy from a larger state, and it does not establish an architecture or prove the broader theory.
- **Deviations:** HARD-v2 still uses the EXP-001B generator, including its 0.3 distractor noise. EXP-002 could not use the original Python 3.12 interpreter, so both of its models used Python 3.11.9 and PyTorch 2.13.0+cpu. The residual scale was not changed after looking at test accuracy. No extra rerun was needed, because vanilla and residual were already environment-matched.
- **Files:** `results/EXP-002/analysis.md`, `raw_metrics.jsonl`, `summary.csv`, `summary.json`, `primary_delay64_summary.json`, `config_used.yaml`, and `environment.txt`.
- **Replication:** Three seeds, one synthetic task, one fixed scale.

## 2026-10-03 — EXP-003 bounded half-and-half mixture

- **Experiment ID:** EXP-003.
- **Configuration:** `configs/exp003_bounded_mixture.yaml`; corrected HARD-v2; delays 64 then 128; seeds 17, 29, and 43; hidden size 32; 400 Adam updates; learning rate 0.003; batch size 64; 1,536 / 512 / 512 examples. Mixture `h_t = 0.5 h_{t-1} + 0.5 candidate_t`, fixed before the run and not tuned. Python 3.11.9, PyTorch 2.13.0+cpu. Code revision `ec69be58e3c8`.
- **Baseline:** Vanilla tanh RNN trained in that same process. Frozen EXP-001, EXP-001B, and EXP-002 result files were not modified. EXP-002 was not rerun. Its saved additive state sizes were copied for comparison.
- **Result:** Twelve of twelve runs finished with finite values in 111.9 seconds. Delay-64 vanilla accuracy was 0.4961, 0.4570, and 1.0000. Bounded accuracy was 0.5293, 0.4805, and 0.4688. Delay-128 vanilla accuracy was 0.5098, 0.5039, and 0.4590. Bounded accuracy was 0.4902, 0.5098, and 0.4805. Bounded probe R^2 stayed near zero. Largest absolute hidden entries on the bounded model were about 0.41–0.83. The saved EXP-002 additive model reached about 65 at delay 64 and about 129 at delay 128.
- **Credit and stability:** Bounded event sensitivity and loss gradients stayed near zero on runs that had not solved the task. The smallest one-step gain was about 0.07–0.21, above the failed vanilla values and below the additive EXP-002 range of about 0.7–0.9. No bounded run was gradient-clipped. No NaN or Inf.
- **Interpretation:** EXPERIMENTAL RESULT. The EXP-002 advantage did not survive this bound. The half-and-half path kept state size controlled and did not preserve delayed memory better than the matched vanilla RNN. Because a factor of 0.5 also fades across 64 steps, this does not prove that magnitude alone caused EXP-002. It does not establish an architecture or prove the broader theory.
- **Files:** `results/EXP-003/analysis.md`, `raw_metrics.jsonl`, `summary.csv`, `summary.json`, `primary_delay64_summary.json`, `exp002_magnitude_reference.json`, `config_used.yaml`, and `environment.txt`.
- **Replication:** Three seeds, one synthetic task, one fixed mixture.

## 2026-10-03 — EXP-004 fixed-bound additive recurrence

- **Experiment ID:** EXP-004.
- **Configuration:** `configs/exp004_bounded_additive.yaml`; corrected HARD-v2 with explicit distractor noise 0.3 and competitor rate 0.4; delays 64 then 128; seeds 17, 29, and 43; hidden size 32; 400 Adam updates; learning rate 0.003. Bounded rule `h_t = clamp(h_{t-1} + candidate_t, -4, 4)`, with `B = 4` fixed before accuracy was collected. Python 3.11.9, PyTorch 2.13.0+cpu. Training code revision `14bdc4786549`.
- **Baseline:** Fresh vanilla and additive models trained on the same tensors, initial weights, and minibatches as the bounded model. Frozen EXP-001 through EXP-003 files were not modified. Their hashes matched after this run.
- **Result:** Eighteen of eighteen runs finished with finite values. No budget stop. Success means test accuracy of at least 0.95. Delay 64: vanilla 0.4961, 0.4570, 1.0000 (1 success); additive 1.0000, 1.0000, 1.0000 (3 successes); bounded 1.0000, 0.5098, 0.5195 (1 success). Delay 128 ran because additive had three delay-64 successes. Delay 128: vanilla 0.5098, 0.5039, 0.4590 (0 successes); additive 1.0000 on all three seeds; bounded 1.0000, 0.5098, 0.6289 (1 success).
- **Retention and credit:** Additive probe R^2 stayed high on every seed. Bounded probe R^2 was 0.996 and 0.997 on seed 17 and about zero on the failed seeds, except 0.147 for delay-128 seed 43. Event sensitivity was about 1.2 to 1.7 on the bounded successes and exactly zero on the bounded failures.
- **Clipping and magnitude:** Every bounded run was above the 1 percent clipping convention. Final test clipping fractions were about 0.71 to 0.94. Proposed states reached 5 before the clamp and stored states stopped at 4. Failed bounded states had root-mean-square size exactly 4. Additive states reached about 65 at delay 64 and about 129 at delay 128. The successful bounded runs were also heavily clipped, so clipping and failure occurring together is not treated as a cause.
- **Interpretation:** EXPERIMENTAL RESULT, mixed and therefore inconclusive. The additive control reproduced. One bounded seed showed that a state capped at 4 can still solve the delay. The other seeds did not. This does not establish an architecture or validate the broader theory.
- **Files:** `results/EXP-004/`.
- **Replication:** Three seeds, one synthetic task, one fixed bound.

## 2026-10-03 — EXP-005 counterfactual early-bit audit

- **Experiment ID:** EXP-005.
- **Configuration:** `configs/exp005_counterfactual_audit.yaml`. No new model. `B` remained 4. Fresh HARD-v2 pairs used split index 3, 512 sequences per seed and delay, with only the original bit flipped inside each pair. Python 3.11.9, PyTorch 2.13.0+cpu. Audit code revision `3ebc12f0e389`.
- **Recovery:** EXP-004 saved no checkpoints. Eighteen models were replayed with the EXP-004 procedure and saved under `results/EXP-005/checkpoints/`. All 18 matched the saved EXP-004 integer correct counts and passed the pre-registered loss, retention, and magnitude tolerances. Nothing was written into `results/EXP-004/`. Frozen EXP-001 through EXP-004 hashes matched after the audit.
- **Result:** On successful additive runs and on bounded seed 17, flipping the original bit changed the final state, the logit, and the prediction, and both pair members were correct. Bounded successes also changed the pattern of +4 and -4 signs. Bounded seed 29 at both delays, and bounded seed 43 at delay 64, kept exactly the same final state, the same boundary signs, and the same logit, with a recorded bit gradient of 0.0. Bounded seed 43 at delay 128 changed state and gave both answers correctly on 119 of 512 pairs, while the recorded bit gradient was 0.0 on every example; the other 393 pairs were unchanged.
- **Interpretation:** EXPERIMENTAL RESULT. Some bounded failures have no finite dependence on the original bit. Seed 43 at delay 128 shows that a recorded local gradient of 0.0 can miss a real finite change. These failures are not all the same. The audit does not show that clipping caused them, and it does not validate the broader theory.
- **Compute:** 162.5 seconds for replay plus audit. No numerical failures and no budget stops.
- **Files:** `results/EXP-005/`.
- **Replication:** Three seeds, one synthetic task, recovered EXP-004 weights.

## 2026-10-03 — EXP-006 fresh-seed replication

- **Experiment ID:** EXP-006.
- **Configuration:** `configs/exp006_replication.yaml`. Unchanged EXP-004 vanilla, additive, and B=4 bounded models. New seeds 101, 113, 127, 139, 151, 163, 179, 191, 211, 223. Data base seed 2000. HARD-v2 noise 0.3 and competitor rate 0.4. Success is final test accuracy of at least 0.95. Python 3.11.9, PyTorch 2.13.0+cpu. Training code revision `53d4d63`. Historical seeds were not pooled.
- **Result:** Sixty of sixty training runs and sixty bit-flip audits finished in 529 seconds. No numerical failures, budget stops, or unstarted runs. Delay-64 successes: vanilla 3/10, additive 10/10, bounded 6/10. Additive cleared the gate of 8, so fresh delay-128 training ran. Delay-128 successes: vanilla 1/10, additive 10/10, bounded 5/10. Paired delay-64 counts were 6 both-succeed and 4 additive-only. Clopper-Pearson 95 percent intervals for the delay-64 success rates were about 0.07–0.65, 0.69–1.00, and 0.26–0.88.
- **Audit:** Successful models passed the bit-flip check. Bounded delay-64 seed 139 scored 0.896 and still changed state and logit on 400 of 512 pairs despite a recorded bit gradient of 0.0. Several other bounded misses had exactly identical final states and a recorded bit gradient of 0.0.
- **Interpretation:** EXPERIMENTAL RESULT. The additive advantage reproduced on every new seed. Bounded success is not limited to the old seed 17, and the paired comparison shows it was less reliable than additive. Ten seeds leave a wide interval. This does not identify why individual seeds succeed, does not show that clipping caused the misses, and does not validate the broader theory.
- **Files:** `results/EXP-006/`.
- **Replication:** Ten new seeds, one synthetic task, one fixed bound. Not pooled with the earlier three seeds.

## 2026-10-03 — EXP-007 readout-only versus full training

- **Experiment ID:** EXP-007.
- **Configuration:** `configs/exp007_readout_only.yaml`. Same three forward equations and `B = 4`. Same ten seeds and data base seed 2000 as EXP-006, but these full-training runs are a new matched control and are not pooled with EXP-006. Readout-only training freezes input and recurrent weights at initialization and trains the output layer with the same Adam settings and 400 updates. Python 3.11.9, PyTorch 2.13.0+cpu.
- **Result:** 120/120 training runs and 120 bit-flip audits finished. Accounted runtime 753 seconds. No numerical failures and no unstarted runs. Full additive succeeded on 10/10 seeds at delays 64 and 128. Readout-only additive succeeded on 0/10 at both delays. Full bounded succeeded on 6/10 and 5/10. Readout-only bounded succeeded on 0/10 at both delays. Full vanilla succeeded on 3/10 and 1/10. Readout-only vanilla succeeded on 0/10.
- **Memory before training:** At initialization, additive probe R^2 averaged about 0.53 at delay 64 and about 0.45 at delay 128. Those values were unchanged after readout-only training. Recurrent parameter hashes and hidden states on a fixed batch were unchanged in all 60 readout-only runs.
- **Interpretation:** EXPERIMENTAL RESULT, outcome B. Reliable additive success under this schedule uses learning in the recurrent weights. The initial dynamics already carry partial bit information, and a trained readout uses some of it, but not enough to reach 0.95 accuracy. Freezing the bounded model removed its successes rather than making it more reliable. This does not prove another budget could never train the readout alone, and it does not validate the broader theory.
- **Files:** `results/EXP-007/`.
- **Replication:** Ten seeds, two training regimes, not pooled with EXP-006.

## 2026-10-03 — EXP-007 interpretation clarification

- **Clarification:** The EXP-007 readout-only condition froze both input-side parameters (`W_x` and `b_x`) and recurrent-side parameters (`W_h` and `b_h`). The supported conclusion is that training that combined input/recurrent module improves reliable additive success under the registered schedule. It does not establish that learning the hidden-to-hidden matrix specifically is necessary.
- **Paired accuracy:** Fewer successful seeds is not the same as lower accuracy on every paired seed. Across the 60 paired comparisons, readout-only accuracy was higher than full training on 18 pairs, mostly where full training had not solved the task. The saved EXP-007 analysis file was not rewritten. The execution record of the interrupted run is `results/EXP-008/exp007_execution_audit.md`.

## 2026-10-03 — EXP-008 additive parameter groups

- **Experiment ID:** EXP-008.
- **Configuration:** `configs/exp008_parameter_groups.yaml`. Additive model only. Four regimes: full; readout only; input-side (`W_x`, `b_x`) plus readout with `W_h` and `b_h` frozen; recurrent-side (`W_h`, `b_h`) plus readout with `W_x` and `b_x` frozen. Same ten seeds and data base seed 2000. Not pooled with earlier counts. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `906a7bd4d375`.
- **Result:** 80/80 training runs and 80 bit-flip audits finished in 625 seconds. No failures and no unstarted runs. Success counts at both delays were full 10/10, readout only 0/10, input-side plus readout 10/10, and recurrent-side plus readout 10/10. Frozen parameters did not change. Readout-only hidden states did not change.
- **Interpretation:** EXPERIMENTAL RESULT, outcome C. Either trained parameter group, together with the readout, was enough for reliable success while the other group stayed at initialization. Learning the hidden-to-hidden matrix was not necessary in this setup. Readout-only training still stayed below 0.95, consistent with EXP-007. This does not prove another budget could never train the readout alone, and it does not validate the broader theory.
- **Files:** `results/EXP-008/`.
- **Replication:** Ten seeds, four training regimes, not pooled with EXP-006 or EXP-007.

## 2026-10-03 — EXP-009 bias versus matrix adaptation (interrupted)

- **Experiment ID:** EXP-009.
- **Configuration:** `configs/exp009_bias_ablation.yaml`. Additive model only. Regime C trains `b_x` and the readout, with both matrices and `b_h` frozen. Regime D trains both matrices and the readout, with both preactivation biases frozen. Same ten seeds and data base seed 2000. Not pooled with earlier counts. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `573bd668d03ead`.
- **Result:** The delay-64 cohort finished: full 10/10, readout only 0/10, bias plus readout 8/10, matrices plus readout 10/10. Bias misses were seed 151 at 0.945 and seed 191 at 0.846. Frozen parameters stayed fixed. The process stopped during delay-128 seed 101 bias-plus-readout. Only delay-128 seed 101 full and readout-only were saved. No audit was run. The run was not resumed. The last recorded session mark is 229.085 seconds, at the start of the unfinished run. That unfinished run has no recorded duration.
- **Interpretation:** EXPERIMENTAL RESULT for delay 64 only. Learning the matrices was sufficient while both biases stayed fixed. Learning one preactivation bias was sufficient on 8 of 10 seeds while both matrices stayed fixed. This does not prove EXP-008 used the bias-only route, and it does not validate the broader theory.
- **Files:** `results/EXP-009/`.
- **Replication:** Ten seeds at delay 64, four regimes, not pooled with earlier counts. Delay 128 is incomplete.

## 2026-10-03 — EXP-009 delay-64 bit-flip audit (diagnostic continuation)

- **Experiment ID:** EXP-009 audit continuation. Training remains interrupted.
- **Configuration:** The 40 saved delay-64 checkpoints only. No new training. Pair rule: 512 base sequences, split index 3, data base seed 2000, flip only the original bit. Python 3.11.9, PyTorch 2.13.0+cpu. Audit code `b8a42b7d8bb62e`.
- **Result:** All 40 checkpoints reproduced their saved integer test counts. Audit runtime was 10.649 seconds. Final states were never exactly equal after the bit flip. Bias-only ordinary successes had both pair members correct on about 0.91 to 0.98 of pairs. Seed 151, ordinary accuracy 0.945, had both members correct on 0.895 of pairs. Seed 191, ordinary accuracy 0.846, had both members correct on 0.686 of pairs. Matrix-trained runs had both members correct on at least 0.998 of pairs. No computational-zero bit gradient hid a finite change. Preexisting files were unchanged.
- **Interpretation:** DIAGNOSTIC RESULT. The pair check supports finite dependence on the original bit for the bias-only successes and for the matrix-trained runs. The two ordinary misses still carry partial information. This does not prove a bias explanation of EXP-008, does not finish delay 128, and does not validate the broader theory. The cause of the original process stop remains unknown.
- **Files:** `results/EXP-009/audit_completion/`.
- **Compute:** Separate diagnostic allowance. It is not evidence about the original 15-minute training cap.

## 2026-10-03 — EXP-009 delay-128 continuation

- **Experiment ID:** EXP-009 delay-128 continuation. The original session remains interrupted.
- **Configuration:** Unchanged additive model and training settings. Two original conditions were reused: delay 128, seed 101, full and readout-only. Seed 101 bias-plus-readout was restarted once from initialization. The other 37 delay-128 conditions were trained once. Python 3.11.9, PyTorch 2.13.0+cpu. Continuation code `3c5e02f64be3`.
- **Result:** 38/38 new jobs finished in 381.534 seconds. No numerical failures and no unstarted jobs. Counting each condition once, delay-128 successes were full 10/10, readout only 0/10, bias plus readout 2/10, and matrices plus readout 10/10. Bias successes were seeds 127 at 0.971 and 211 at 0.992. The nearest misses were seeds 223 at 0.949 and 179 at 0.947. All 40 models changed final state when only the original bit was flipped. Matrix-trained runs had both pair members correct on every pair. Bias misses still had both members correct on about 0.66 to 0.90 of pairs. Frozen parameters and historical files stayed unchanged.
- **Interpretation:** EXPERIMENTAL RESULT. At delay 128, bias-only adaptation was not sufficient for most seeds, while matrix adaptation was sufficient on every seed with both biases fixed. Scores just below 0.95 still carried bit information. This does not explain EXP-008, does not make matrix learning universally necessary, and does not validate the broader theory. The original unfinished attempt still has no recorded duration.
- **Files:** `results/EXP-009/delay128_completion/`.
- **Compute:** Separate 10-minute allowance. Measured time was 381.534 seconds. This is not the complete historical EXP-009 cost.

## 2026-10-03 — EXP-010 final-16 gradient cutoff

- **Experiment ID:** EXP-010.
- **Configuration:** `configs/exp010_truncated_bptt.yaml`. Unchanged additive model, all parameters trainable. Full-history training versus one detach before the final 16 steps. K = 16, fixed beforehand. Same ten seeds and data base seed 2000. Not pooled with earlier counts. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `af1c7fbfeaf8`.
- **Result:** 40/40 training runs and 40 bit-flip audits finished in 472 seconds. No failures and no unstarted runs. Full-history success was 10/10 at both delays. Final-16 success was 1/10 at delay 64, seed 223 at 0.998, and 0/10 at delay 128. On every final-16 run the training-graph gradient at the original event was a computational zero. The separate full-forward measurement still changed with the bit, and the bit-flip check changed the final state. Final-16 clipping was much more frequent than full-history clipping.
- **Interpretation:** EXPERIMENTAL RESULT, outcome B. Sending the training gradient back through the whole sequence improved reliable success under this schedule. The cutoff was not fatal for every seed, and several misses kept partial information. This does not prove another budget or horizon could never work, and it does not validate the broader theory.
- **Files:** `results/EXP-010/`.
- **Replication:** Ten seeds, two training regimes, not pooled with earlier counts.

## 2026-10-03 — EXP-011 horizon and clipping

- **Experiment ID:** EXP-011.
- **Configuration:** `configs/exp011_clipping_horizon.yaml`. Same additive model. Four fresh conditions: full history or final 16 steps, each with norm-5 clipping or with no gradient rescaling. K stayed 16. Same ten seeds and data base seed 2000. Not pooled with EXP-010. Python 3.11.9, PyTorch 2.13.0+cpu. Training code `39b57cd499ece`.
- **Result:** 80/80 training runs and 80 bit-flip audits finished in 1032 seconds. No numerical failures, timeouts, or unstarted runs. Delay-64 successes were 10/10, 1/10, 10/10, and 2/10 for full-clip, final-16-clip, full-unclipped, and final-16-unclipped. Delay-128 successes were 10/10, 0/10, 7/10, and 0/10. The clipped accuracies matched EXP-010. Unclipped runs never rescaled a gradient, although their raw gradients often exceeded 5. The final-16 training-graph event gradient stayed a computational zero.
- **Interpretation:** EXPERIMENTAL RESULT, outcome D. At delay 64 the full-history advantage remains after clipping is removed. At delay 128, removing clipping makes full-history training less consistent and does not make final-16 training reliable. The earlier clipped comparison is not invalidated by its higher clipping rate. This does not validate the broader theory.
- **Files:** `results/EXP-011/`.
- **Replication:** Ten seeds, four conditions, two delays trained separately. Not pooled with earlier counts.

## 2026-10-03 — EXP-012 frozen readout refit

- **Experiment ID:** EXP-012.
- **Configuration:** `configs/exp012_readout_refit.yaml`. Exploratory diagnostic on all 80 saved EXP-011 checkpoints. No recurrent training. L2 logistic regression, C = 1.0, lbfgs, tolerance 1e-8, 2000 iterations. scikit-learn 1.9.1 expresses that L2 setting as `l1_ratio = 0`. Standardization uses the training mean and population standard deviation. Python 3.11.9, PyTorch 2.13.0+cpu. Diagnostic code `bc4fcdd1e130`. scikit-learn was installed because it was not already present.
- **Result:** All 80 checkpoints reproduced their saved test counts. All 80 classifiers converged, in at most 170 iterations, in 15.6 seconds. Of 37 final-16 misses, 36 reached at least 0.95 with the new readout. Their bit-flip both-correct fractions were at least 0.977. The remaining miss, delay 128 seed 179 final-16 without clipping, rose from 0.594 to 0.881. Every full-history refit was at least 0.973. Recurrent hashes and historical files stayed unchanged.
- **Interpretation:** DIAGNOSTIC RESULT, outcome C. Most unsuccessful final-16 states already contained enough linearly readable task information for this fixed classifier. The original readout did not fully use it. One state was only partly improved, which does not prove that no linear classifier could succeed. This does not remove the joint-training advantage of full-history learning, and it does not validate the broader theory.
- **Files:** `results/EXP-012/`.
- **Replication:** Not a new recurrent-training replication. The refits are not new seeds.

## 2026-10-03 — EXP-007 schedule clarification

- **Clarification:** EXP-007 showed an advantage for training the input and recurrent weights under its original readout-training schedule. It did not establish that recurrent training is necessary no matter how the readout is fit. The saved EXP-007 analysis was not rewritten.

## 2026-10-03 — EXP-013 initialized additive states

- **Experiment ID:** EXP-013.
- **Configuration:** `configs/exp013_initialized_readout.yaml`. Same EXP-012 classifier on three frozen states: EXP-007 additive readout-only initialization, EXP-011 full-history with clipping, and EXP-011 final-16 with clipping. Ten seeds, delays 64 and 128. No recurrent training. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Diagnostic code `7b132c6ce698`.
- **Result:** All 60 checkpoints verified. Initialized recurrent weights matched a freshly seeded model exactly. The full-history and final-16 fits reproduced the EXP-012 predictions. All 60 classifiers converged. Initialized diagnostic success was 6/10 at delay 64 and 5/10 at delay 128. Full-history and final-16 diagnostic success was 10/10 at both delays. The run took 11.3 seconds. EXP-007’s incomplete fingerprint file was not altered.
- **Interpretation:** DIAGNOSTIC RESULT, outcome C. Untrained additive dynamics are already enough for this readout on some seeds. Learning, including final-16 learning, raises the remaining seeds to a reliable score. The earlier readout-only misses were not proof that the untrained state had no readable answer. This does not validate the broader theory.
- **Files:** `results/EXP-013/`.
- **Replication:** Exploratory diagnostic on the existing cohort. These refits are not new recurrent-training seeds.

## 2026-10-03 — EXP-014 new-seed replication

- **Experiment ID:** EXP-014.
- **Configuration:** `configs/exp014_replication.yaml`. New seeds 307, 311, 313, 317, 331, 337, 347, 349, 353, and 359. Data base seed 3000. Delay 128 only. Unchanged additive model, final-16 cutoff, norm-5 clipping, and the EXP-012 classifier. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `76d7b76dc488`.
- **Result:** 20/20 training runs and 30/30 classifier fits finished in 340 seconds. No failures and no unstarted runs. Success counts were initialized diagnostic 4/10, full-history original 10/10, full-history diagnostic 10/10, final-16 original 1/10, and final-16 diagnostic 8/10. The final-16 diagnostic misses were seeds 331 at 0.945 and 353 at 0.922. The early training gradient was blocked. Historical files were unchanged.
- **Interpretation:** EXPERIMENTAL RESULT. The separation between a weak original readout and a readable final-16 state mostly replicated on a new cohort. It did not replicate for every seed. Untrained states were already enough on 4 of 10 seeds. Full-history training stayed reliable with its original readout. This does not validate the broader theory.
- **Files:** `results/EXP-014/`. The pre-run note is `research_checkpoint.md`.
- **Replication:** Ten new seeds, one delay, one task distribution. Not pooled with the earlier cohort.

## 2026-10-03 — EXP-014 rescue denominator

- **Clarification:** Re-counted from `results/EXP-014/diagnostic_metrics.jsonl`. Final-16 original success is 1/10, seed 311. Diagnostic success is 8/10. Original misses are 9. Rescued misses are 7: seeds 307, 313, 317, 337, 347, 349, and 359. Seeds 331 and 353 remained below 0.95. Seed 311 is not a rescue. The frozen analysis was not edited.
- **Separate cell:** EXP-012 delay 128, final-16, clipping at 5, seeds 101–223: 10 original misses and 10 rescues. That is not pooled with 7/9.

## 2026-10-03 — Scientific checkpoint

- **Scope:** Saved-record audit of EXP-001 through EXP-014. No new training, refitting, or evaluation.
- **Document:** `SCIENTIFIC_CHECKPOINT.md` and `DOCUMENTATION_AUDIT.md`.
- **Result:** The hashed experiment artifacts were unchanged by the documentation edit. The main wording gap was the EXP-014 rescue denominator above.
- **Next:** No experiment is running. EXP-015 awaits coordinator authorization.

## 2026-10-04 — EXP-015 frozen length generalization

- **Experiment ID:** EXP-015.
- **Configuration:** `configs/exp015_length_generalization.yaml`. Saved EXP-014 models and classifiers only. No training and no refitting. Fresh HARD-v2 draws at delays 128, 256, and 512, data base seed 4000, evaluation split 4, audit split 5. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Evaluation code `114e74ffefe`.
- **Result:** All 30 checkpoints and 30 classifiers reproduced their saved correct counts. Evaluation took 15.8 seconds. Fresh delay-128 success counts were initialized diagnostic 4/10, full-history original 10/10, full-history diagnostic 10/10, final-16 original 1/10, and final-16 diagnostic 8/10. Delay 256 was 0, 8, 4, 1, and 0. Delay 512 was 0, 6, 3, 0, and 0. Of the models already successful at fresh delay 128, retained success was 0, 8, 4, 0, and 0 at delay 256, and 0, 6, 3, 0, and 0 at delay 512. Final states still differed on every bit-flip pair. Hidden magnitude grew with the number of steps for untrained and trained models alike. No numerical failures.
- **Interpretation:** EXPERIMENTAL RESULT. The frozen diagnostic readout does not remain reliable at delays 256 and 512. The full-history original readout does on many seeds. Surviving bit dependence is not the same as a correct answer. This is not a pure elapsed-time test, because longer sequences contain more distractors. It does not validate the broader theory.
- **Files:** `results/EXP-015/`.
- **Replication:** Evaluation of the EXP-014 cohort on new longer sequences. Not a new training replication.
- **Length note:** A fixed-length readout rescue did not establish length robustness. The saved diagnostic readout failed at the longer delays even where the state still depended on the bit.

## 2026-10-04 — EXP-016 stronger noise, interrupted

- **Experiment ID:** EXP-016.
- **Configuration:** `configs/exp016_noise_robustness.yaml`. New seeds 401, 409, 419, 421, 431, 433, 439, 443, 449, and 457. Data base seed 6000. Delay 128. Vanilla and additive models, each at distractor noise 0.3 and 1.0, with shared latent draws. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `8ccb69e9e447`.
- **Result:** Interrupted during seed 439, vanilla, noise 1.0. No score was saved for that condition. A second start was refused. Finished original-readout accuracies were, in seed order: vanilla noise 0.3: 0.494, 0.518, 0.506, 0.730, 0.506, 0.543, 0.506. Additive noise 0.3: 1.000 on those same seven seeds. Vanilla noise 1.0: 0.506, 0.482, 0.529, 0.570, 0.533, 0.457. Additive noise 1.0: 1.000 on those six seeds. The diagnostic readout on the six fully finished seeds was about 0.5 for vanilla and 1.0 for additive, at both noise levels. Seeds 443, 449, and 457 were not started. The last recorded session time was 218 seconds. Historical files were unchanged.
- **Interpretation:** The finished runs are consistent with the additive advantage surviving this stronger noise, on both the original readout and the separate diagnostic readout. This is not a completed ten-seed result. A vanilla miss was not always the same: some vanilla states became identical after a bit flip, and one partial vanilla run did not. Noise 1.0 does not replace the frozen noise-0.3 experiments. This does not validate the broader theory.
- **Files:** `results/EXP-016/`.
- **Replication:** Incomplete new cohort. Not pooled with earlier seeds.

## 2026-10-04 — EXP-016 completion

- **Experiment ID:** EXP-016, completed through an authorized continuation. Not a normally completed original session.
- **Configuration:** The original `configs/exp016_noise_robustness.yaml` settings were unchanged. Twenty-six finished conditions were reused. Fourteen jobs were trained under `results/EXP-016/completion/`. Seed 439, vanilla, noise 1.0, was restarted once from initialization. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Completion code `4fa96b9058fa`.
- **Result:** All 26 reused checkpoints reproduced their saved correct counts. All 14 jobs exited 0. Continuation time was 170.4 seconds. Scored conditions are 40. Original-readout success was additive 10/10 at noise 0.3 and 10/10 at noise 1.0. Vanilla was 1/10 at noise 0.3, seed 457 only, and 0/10 at noise 1.0. The diagnostic readout agreed on every success line. No further interruption and no numerical failure. The original attempt’s duration and cause remain unknown. Historical files outside the completion folder were unchanged.
- **Interpretation:** EXPERIMENTAL RESULT. The additive advantage remains at this stronger noise under the fixed budget. One vanilla seed shows that noise 0.3 is not impossible for that model. This does not establish general robustness or video capability. Overwrite remains untested.
- **Files:** `results/EXP-016/completion/`.
- **Replication:** The preregistered ten-seed cohort is now scored. The original session and the continuation are not pooled into one runtime.

## 2026-10-04 — EXP-017 selective overwrite

- **Experiment ID:** EXP-017.
- **Configuration:** `configs/exp017_selective_overwrite.yaml`. New seeds 503, 509, 521, 523, 541, 547, 557, 563, 569, and 571. Data base seed 7000. Hold versus selective overwrite, vanilla and additive. Not a HARD-v2 replication. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `d3b223fb3dd0`.
- **Result:** 40/40 runs finished in 471.9 seconds. Additive hold succeeded on 10/10. Vanilla hold succeeded on 1/10, seed 503. Selective success was 0/10 for both models. On the additive selective task, the conflicting replacement subgroup was the weak one, from 0.062 to 0.500. The separate readout scored 0.750 on every additive selective seed by answering A, including 0.000 when the marked bit differed.
- **Interpretation:** EXPERIMENTAL RESULT. The additive model retained the first bit and did not reliably replace it. The separate readout made that retention sharper. It did not establish selective updating, erasure, or a need for a new gate. This does not validate the broader theory.
- **Files:** `results/EXP-017/`.
- **Replication:** Ten new seeds, one new task, one budget. Not pooled with the delayed-bit cohorts.

## 2026-10-04 — EXP-018 locate the overwrite failure

- **Experiment ID:** EXP-018.
- **Configuration:** `configs/exp018_failure_location.yaml`. Frozen EXP-017 checkpoints only. Fresh split index 4, 512 sequences per seed. No training and no new fit. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Diagnostic code `252565c7f858`.
- **Result:** All 40 checkpoints and classifiers reproduced their saved correct counts. The run took 11.4 seconds. Additive selective seeds 503, 521, 557, and 569 were usually unchanged in ordinary precision when the marked bit flipped, with median difference 0. Seeds 523 and 541 changed by median 0.392 and 0.153, and that change remained at the query. Seed 509 changed and about half of the differences were gone by the query. Both saved readouts followed the marked bit on 0 of the pairs for every selective model. Additive candidates were mostly near plus or minus one. Float64 changed many exact-equality calls without making the typical tiny differences large.
- **Interpretation:** NUMERICAL EVIDENCE. Weak entry, later disappearance, and unused persistent differences all occur. Saturation is associated with weaker entry and is not shown to be the cause. A lasting state difference is not evidence that another readout could use the new bit. This does not validate the broader theory.
- **Files:** `results/EXP-018/`.
- **Replication:** Diagnostic of the EXP-017 models. Not a new training replication.

## 2026-10-04 — EXP-019 marked reset, stopped before training

- **Experiment ID:** EXP-019.
- **Configuration:** `configs/exp019_marker_reset.yaml`. The reset uses the write channel with no threshold and no new parameter. Training was allowed only if hold-task trajectories matched within 0.00001. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Code `1ce3059676bd`.
- **Result:** The check failed in 0.3 seconds. All ten seeds disagree from the first distractor step. About 40 percent of interior steps are competitors, and about 99 percent of interior write-channel values are not exactly 0 or 1. The largest hidden-state gap was 145 and the largest score gap was 166. Training runs: 0. No accuracy was collected. Historical files were unchanged.
- **Interpretation:** The hold-match assumption is false for the existing task generator. The reset is not confined to the original event. This is not a selective-overwrite accuracy result and not evidence that a cleaner marker could not help. No threshold was added after the check.
- **Files:** `results/EXP-019/`.
- **Replication:** Not a training replication. Not pooled with earlier success rates.

## 2026-10-04 — EXP-019 interpretation, stated before EXP-020

- **Correction:** EXP-019 collected no training results. Its clean-marker assumption was false. EXP-017 tested overwrite under its actual noisy input encoding, not under an unambiguous binary write-control interface. EXP-019 is not a model failure. The saved EXP-019 analysis was not rewritten.

## 2026-10-04 — EXP-020 clean write control

- **Experiment ID:** EXP-020.
- **Configuration:** `configs/exp020_clean_write.yaml`. Same seeds and base seed as EXP-017, plus a fourth clean write channel. Four-input additive and four-input reset additive. 1,249 parameters, 32 more than the three-channel models. Not an unchanged-task replication. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `398ea44b4474`.
- **Result:** Preflight passed with state, score, and gradient gaps of 0. 40/40 runs finished in 587 seconds. Hold success was 10/10 for both versions. Selective success was 0/10 for ordinary additive and 10/10 for reset additive. Conflicting replacement was 128/128 on every reset seed. Seed 547 was 511/512 overall. No numerical failures.
- **Interpretation:** EXPERIMENTAL RESULT. The explicit control alone did not make ordinary additive replacement succeed. The fixed reset did. This is not learned gating and is not pooled with EXP-017. It does not establish video capability.
- **Files:** `results/EXP-020/`.
- **Replication:** Matched intervention on the existing cohort. Not an independent new-seed replication.

## 2026-10-04 — EXP-021 clean-control replication

- **Experiment ID:** EXP-021.
- **Configuration:** `configs/exp021_replication.yaml`. Same procedure as EXP-020. New seeds 601, 607, 613, 617, 619, 631, 641, 643, 647, and 653. Data base seed 8000. Not pooled with EXP-020. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `9dd7a3c81f21`.
- **Result:** Preflight passed with gaps of 0. 40/40 runs finished in 612 seconds. Hold success was 10/10 for both versions. Selective success was 0/10 for ordinary additive and 10/10 for the reset. Seed 607 was 511/512, with conflicting replacement at 127/128. Audits for the reset were at or above 0.996. No numerical failures. Historical files were unchanged.
- **Interpretation:** EXPERIMENTAL RESULT. The reset advantage replicated. Ordinary additive recurrence remained below the line, including on conflicting replacement. This is not learned gating and not a video result.
- **Files:** `results/EXP-021/`.
- **Replication:** Independent cohort. Not pooled with EXP-020.

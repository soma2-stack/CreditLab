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

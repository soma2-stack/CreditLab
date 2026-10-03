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

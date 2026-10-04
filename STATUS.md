# Project Status

- **Current phase:** EXP-009 training is still interrupted. The delay-64 bit-flip audit is done. No training is running.
- **What was tested:** The 40 saved delay-64 checkpoints, without retraining. The question was whether flipping only the original bit changes the final state and the answer.
- **Latest result:** All 40 checkpoints verified. Bias-only successes depend on the original bit. Seeds 151 and 191 still show partial dependence. Matrix-trained runs do too. Delay 128 is still incomplete.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not resume delay 128, do not tune the bound of 4, and do not start EXP-010.

See `results/EXP-009/audit_completion/analysis.md`. The training interruption note was not rewritten.

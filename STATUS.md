# Project Status

- **Current phase:** The planned EXP-009 delay-128 comparison is complete through an authorized continuation. The original training session remains interrupted. No training is running.
- **What was tested:** The same additive model at delay 128. Two finished runs were reused. The unfinished bias run was restarted once from initialization, and the other 37 runs were trained once.
- **Latest result:** Full training and matrix-plus-readout training succeeded on all 10 seeds. Bias-plus-readout training succeeded on 2 of 10. Readout-only training succeeded on none. The misses still showed partial dependence on the original bit.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not start EXP-010, do not split the parameter groups further, and do not tune the bound of 4.

See `results/EXP-009/delay128_completion/analysis.md`. The original interruption note was not rewritten.

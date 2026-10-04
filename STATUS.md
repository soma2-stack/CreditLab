# Project Status

- **Current phase:** EXP-011 is complete. No further experiment is running.
- **What was tested:** The same additive model under four fixed settings: full-history or final-16 training, each with norm-5 clipping or with clipping turned off.
- **Latest result:** With clipping, full-history training succeeded on all 10 seeds at both delays and final-16 training succeeded on 1 and then 0. Without clipping, full-history training still succeeded on all 10 seeds at delay 64 but only 7 at delay 128. Final-16 training succeeded on 2 seeds at delay 64 and none at delay 128.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not try another horizon, do not tune a clipping threshold, and do not start EXP-012.

See `results/EXP-011/analysis.md`. This is one factorial comparison, not a proof of the theory.

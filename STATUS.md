# Project Status

- **Current phase:** EXP-010 is complete. No further experiment is running.
- **What was tested:** The same additive model, trained either through the whole sequence or with the training gradient cut off so that only the final 16 steps send a gradient.
- **Latest result:** Full-sequence training succeeded on all 10 seeds at delays 64 and 128. Final-16 training succeeded on 1 of 10 seeds at delay 64 and on none at delay 128. The training gradient at the original event was blocked in the final-16 runs. The forward pass could still change with that bit.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not try another horizon, do not add a model, and do not start EXP-011.

See `results/EXP-010/analysis.md`. This is one training comparison, not a proof of the theory.

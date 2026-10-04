# Project Status

- **Current phase:** EXP-014 is complete. No further experiment is running.
- **What was tested:** The same additive model and the same fixed linear classifier on ten new seeds at delay 128. Each seed had an untrained state, a full-history trained state, and a final-16 trained state.
- **Latest result:** The original final-16 readout succeeded on 1 of 10 seeds. A separate classifier read 8 of those 10 states reliably. Untrained states succeeded on 4 of 10. Full-history training succeeded on all 10 with its original readout.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not change the task, the classifier, or the horizon, and do not start EXP-015.

See `results/EXP-014/analysis.md`. This is one new cohort, not a proof of the theory.

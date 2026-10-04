# Project Status

- **Current phase:** EXP-007 is complete. No further experiment is running.
- **What was tested:** The same vanilla, additive, and B=4 bounded models, each trained either fully or with only the readout allowed to learn.
- **Latest result:** Full additive training succeeded on all 10 seeds at both delays. Readout-only additive training succeeded on none. The initial additive state already held a partial trace of the original bit, and that trace did not change when the recurrent weights were frozen. Full bounded training succeeded on 6 of 10 and 5 of 10 seeds. Readout-only bounded training succeeded on none.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not tune the bound of 4, and do not start another experiment from this comparison.

See `results/EXP-007/analysis.md`. This is one matched training comparison, not a proof of the theory.

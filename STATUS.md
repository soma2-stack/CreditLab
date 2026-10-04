# Project Status

- **Current phase:** EXP-009 was interrupted. No training is running.
- **What was tested:** The same additive model under four regimes: full training, readout only, one preactivation bias (`b_x`) plus readout, and both matrices plus readout.
- **Latest result:** At delay 64, full training and matrix-plus-readout training succeeded on all 10 seeds. Bias-plus-readout training succeeded on 8 of 10. Readout-only training succeeded on none. Delay 128 and the bit-flip audit did not finish.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Ask the coordinator before any continuation. Do not resume the interrupted run automatically, do not tune the bound of 4, and do not start EXP-010.

See `results/EXP-009/analysis.md` and `results/EXP-009/interruption_note.md`.

# Project Status

- **Current phase:** EXP-008 is complete. No further experiment is running.
- **What was tested:** The same additive model under four training regimes: full training, readout only, input-side plus readout, and recurrent-side plus readout.
- **Latest result:** Full training and both partial regimes succeeded on all 10 seeds at delays 64 and 128. Readout-only training succeeded on none. Learning the hidden-to-hidden weights was not required when the input-side weights and readout could learn. Learning the input-side weights was not required when the recurrent-side weights and readout could learn.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not tune the bound of 4, and do not start another experiment from this comparison.

See `results/EXP-008/analysis.md`. This is one matched training comparison, not a proof of the theory.

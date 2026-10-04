# Project Status

- **Current phase:** EXP-012 is complete. No further experiment is running.
- **What was tested:** A new linear classifier was fit to the frozen final hidden states of all 80 EXP-011 models. The recurrent weights were not trained again.
- **Latest result:** 36 of 37 unsuccessful final-16 models reached at least 0.95 with the new readout, and those rescues held up when only the original bit was flipped. One model, delay 128 seed 179 without clipping, rose from 0.594 to 0.881 and stayed below 0.95. Every full-history model stayed strong.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not train the recurrent weights again, do not tune the classifier, and do not start EXP-013.

See `results/EXP-012/analysis.md`. This is a diagnostic on models that were already trained, not a proof of the theory.

# Project Status

- **Current phase:** EXP-013 is complete. No further experiment is running.
- **What was tested:** The same fixed linear classifier on three frozen additive states: never-trained input and recurrent weights, full-history training, and final-16 training.
- **Latest result:** Never-trained states reached 0.95 on 6 of 10 seeds at delay 64 and 5 of 10 at delay 128. Both trained states reached it on all 10 seeds at both delays. The earlier readout-only misses were not proof that the untrained state had no readable answer.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not train recurrent weights, do not tune the classifier, and do not start EXP-014.

See `results/EXP-013/analysis.md`. This is a diagnostic on saved models, not a proof of the theory.

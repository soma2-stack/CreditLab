# Project Status

- **Current phase:** EXP-015 is complete. No experiment is running.
- **What was tested:** The saved delay-128 models and their saved readouts, scored again at delays 128, 256, and 512. Nothing was trained or refit.
- **Latest result:** The separate linear readout did not stay reliable at the longer delays. The full-history model’s own readout still succeeded on 8 of 10 seeds at delay 256 and 6 of 10 at delay 512. The hidden state still changed when the original bit was flipped.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not test a longer delay, do not retrain, and do not start EXP-016.

See `results/EXP-015/analysis.md`. The scientific checkpoint was not rewritten.

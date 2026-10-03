# Project Status

- **Current phase:** EXP-006 is complete. No further experiment is running.
- **What was replicated:** The unchanged additive and B=4 bounded models, plus matched vanilla controls, on ten new seeds and fresh datasets.
- **Latest result:** At delay 64 the additive model succeeded on 10 of 10 seeds and the bounded model on 6 of 10. At delay 128, after a fresh training run, additive again succeeded on 10 of 10 and bounded on 5 of 10. Vanilla succeeded on 3 of 10 and 1 of 10. These counts are separate from the older seeds 17, 29, and 43.
- **Audit:** Models that reached the accuracy bar also passed the bit-flip check. Some bounded misses still changed when the original bit was flipped, and others did not change at all.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not tune the bound of 4, and do not start another experiment from this replication.

See `results/EXP-006/analysis.md`. Ten seeds still leave a wide range of uncertainty. This is not a proof of the theory.

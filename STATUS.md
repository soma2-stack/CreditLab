# Project Status

- **Current phase:** EXP-005 is complete. No further experiment is running.
- **What exists:** Frozen EXP-001 through EXP-004 results, plus a bit-flip audit of replayed EXP-004 models. No new architecture was added.
- **Recovery:** EXP-004 had no saved weights. The replay matched those saved metrics within the pre-registered tolerances. Checkpoints now live under `results/EXP-005/checkpoints/`.
- **Latest result:** Successful models, including bounded seed 17, change their final state and answer when only the original bit is flipped. Bounded seed 29, and bounded seed 43 at delay 64, do not change at all. Bounded seed 43 at delay 128 changes on 119 of 512 pairs even though the recorded bit gradient is 0.0 on every example.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not tune the bound of 4, and do not start another experiment from this audit.

See `results/EXP-005/analysis.md`. This is a diagnostic on one synthetic task, not a proof of the theory.

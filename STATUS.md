# Project Status

- **Current phase:** EXP-004 is complete. No further experiment is running.
- **What exists:** Frozen EXP-001 through EXP-003 results, plus an additive RNN whose state is clamped to the fixed interval from -4 to 4.
- **Environments:** EXP-004 used Python 3.11.9 and PyTorch 2.13.0+cpu for the vanilla, additive, and bounded models together. Training code revision `14bdc4786549`.
- **Completed experiments:** EXP-001 through EXP-003 result files are unchanged, and their hashes still match. EXP-004 compared the three fresh models on corrected HARD-v2 at delays 64 and 128.
- **Latest result:** The additive model scored 1.000 on all three seeds at both delays. The clamped model scored 1.000 only on seed 17 at both delays. The other clamped seeds stayed near chance, except delay-128 seed 43 at 0.629. Vanilla still had one delay-64 success and no delay-128 success.
- **Clipping:** The clamp was active on every bounded run, including the success. Stored states stopped at 4. Additive states still grew to about 65 and 129. Failed clamped states sat entirely on the boundary. That pattern is not treated as proof that clipping caused the failures.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. Do not change the bound of 4 from these results, and do not start another experiment until a new design is written down first.

See `results/EXP-004/analysis.md`. This is a small synthetic result, not a proof of the theory.

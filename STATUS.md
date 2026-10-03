# Project Status

- **Current phase:** Phase 1 baseline review complete; no adaptive-credit architecture has been implemented.
- **What exists:** A reproducible vanilla tanh RNN baseline, the EXP-001 delayed-bit task, and the separate EXP-001B follow-up with corrected HARD-v2 task and added credit diagnostics.
- **Completed experiments:** EXP-001 is frozen and complete. EXP-001B is complete. The original EXP-001 result files remain unchanged.
- **Latest result:** The original HARD competitor events correlate with the answer (about 75% agreement); corrected HARD-v2 removes this. The original HARD long-delay advantage largely disappears on HARD-v2. EASY and HARD-v2 begin splitting by seed at delay 64, and are mostly at chance by delay 128–256.
- **Credit and retention:** Failed EASY/HARD-v2 runs generally have little linear evidence of the original bit in the final hidden state, plus very small event sensitivity and loss gradients. Correct predictions can also have tiny loss gradients, so the metrics need to be read together.
- **Currently running:** Nothing.
- **Next recommended action:** Write a narrow EXP-002 design for one test of whether a simple way of sending learning feedback to the original event improves the vanilla baseline at delay 64 on corrected HARD-v2. Do not implement it as part of EXP-001B.

See `results/EXP-001/analysis.md` and `results/EXP-001B/analysis.md`. These are experimental observations on small synthetic runs, not architecture or theory results.

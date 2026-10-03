# Project Status

- **Current phase:** Phase 1 baseline is frozen. One small mechanism test, EXP-002, is complete. No adaptive-credit architecture has been built.
- **What exists:** The frozen EXP-001 and EXP-001B vanilla-RNN results, plus a residual tanh RNN that copies the previous hidden state forward and adds the usual tanh update. The copy weight is fixed at 1.
- **Completed experiments:** EXP-001 and EXP-001B are unchanged. EXP-002 compared that residual path with the vanilla RNN on corrected HARD-v2 at delay 64, then checked delay 128.
- **Latest result:** At delay 64, vanilla test accuracy was 0.4961, 0.4570, and 1.0000 across seeds 17, 29, and 43. The residual model scored 1.0000 on all three seeds and kept the original bit readable in the final state. At delay 128 the residual model again scored 1.0000 on all three seeds, while the vanilla model stayed near chance on all three in this run.
- **Credit and retention:** The residual path kept a strong step-to-step copy. The vanilla failures still showed almost no readable memory and almost no sensitivity to the original event. The residual loss gradient stayed small because those answers were already correct. Residual hidden states grew to about the length of the sequence, and training stayed finite.
- **Currently running:** Nothing.
- **Next recommended action:** Design, but do not start, one follow-up that keeps this same skip and adds a single bound chosen before any new test result. The question is whether the direct path still helps when the hidden state cannot grow with the delay. Do not retune the scale of 1 against the EXP-002 test set, and do not treat EXP-002 as an architecture result.

See `results/EXP-002/analysis.md`. These are small synthetic observations, not a proof of the theory.

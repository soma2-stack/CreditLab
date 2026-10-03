# Project Status

- **Current phase:** Phase 1 baseline is frozen. One small mechanism test, EXP-002, is complete. No adaptive-credit architecture has been built.
- **What exists:** The frozen EXP-001 and EXP-001B vanilla-RNN results, plus a residual tanh RNN that copies the previous hidden state forward and adds the usual tanh update. The copy weight is fixed at 1.
- **Environments:** Frozen EXP-001 and EXP-001B used the earlier Python 3.12 environment. EXP-001B records Python 3.12.14 and PyTorch 2.14.1+cpu. That interpreter could not be recovered. EXP-002 ran both the vanilla control and the residual model under Python 3.11.9 and PyTorch 2.13.0+cpu. EXP-002 conclusions use that matched pair. Differences from the frozen files are not treated as architecture effects.
- **Completed experiments:** EXP-001 and EXP-001B result files are unchanged. EXP-002 compared the residual path with a vanilla RNN trained in the same run, on corrected HARD-v2 at delay 64, then delay 128.
- **Latest result:** In the matched Python 3.11 run, delay-64 vanilla accuracy was 0.4961, 0.4570, and 1.0000 across seeds 17, 29, and 43. The residual model scored 1.0000 on all three seeds and kept the original bit readable. At delay 128 the residual model again scored 1.0000 on all three seeds, and the matched vanilla model stayed near chance on all three.
- **Credit and retention:** The residual path kept a strong step-to-step copy. The vanilla failures still showed almost no readable memory and almost no sensitivity to the original event. The residual loss gradient stayed small because those answers were already correct. Residual hidden states grew to about the length of the sequence, and training stayed finite.
- **Currently running:** Nothing.
- **Next recommended action:** Design, but do not start, one follow-up that keeps this same skip and adds a single bound chosen before any new test result. The question is whether the direct path still helps when the hidden state cannot grow with the delay. Do not retune the scale of 1 against the EXP-002 test set, and do not treat EXP-002 as an architecture result.

See `results/EXP-002/analysis.md`. These are small synthetic observations, not a proof of the theory.

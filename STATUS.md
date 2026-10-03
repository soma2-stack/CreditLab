# Project Status

- **Current phase:** EXP-003 is complete. No adaptive-credit architecture has been built.
- **What exists:** Frozen EXP-001, EXP-001B, and EXP-002 results, plus a bounded recurrent mixture that takes half the previous hidden state and half the usual tanh update.
- **Environments:** Frozen EXP-001 and EXP-001B used the earlier Python 3.12 environment. EXP-002 and EXP-003 both used Python 3.11.9 and PyTorch 2.13.0+cpu. Each experiment’s conclusion uses the vanilla and altered model trained together in that experiment.
- **Completed experiments:** EXP-001, EXP-001B, and EXP-002 result files are unchanged. EXP-003 compared the 0.5/0.5 mixture with a vanilla RNN on corrected HARD-v2 at delays 64 and 128.
- **Latest result:** The bounded mixture stayed near chance on all three seeds at both delays. The matched vanilla model still succeeded on seed 43 at delay 64 and failed at delay 128. The bounded hidden state stayed below 1 in absolute value. The saved EXP-002 additive model had reached about 65 and 129.
- **Credit and retention:** The bounded model did not keep a readable copy of the original bit, and the original event had almost no effect on its answer. Its one-step path was somewhat stronger than a failed vanilla step, and much weaker than the additive EXP-002 path.
- **Currently running:** Nothing.
- **Next recommended action:** Do not start a new run yet. The next question is whether a full-strength copy can be kept while one bound, chosen before any new test, stops the state from growing with the delay. Do not retune the 0.5 weights. Do not treat EXP-003 as an architecture result.

See `results/EXP-003/analysis.md`. These are small synthetic observations, not a proof of the theory.

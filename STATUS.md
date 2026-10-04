# Project Status

- **Current phase:** The research phase through EXP-014 is consolidated. No experiment is running.
- **What was learned:** On this delayed-bit task, additive recurrence is more reliable than a vanilla network under the same training recipe. Full-sequence training makes the joint readout reliable. Cutting that training feedback usually hurts the joint readout, but the hidden state is often still readable by a separate linear fit. Some seeds are readable even before recurrent training. A few are not.
- **Latest correction:** EXP-014’s final-16 diagnostic success is 8 of 10. Of the 9 original misses, 7 were rescued. Seed 311 was already successful and is not a rescue. Seeds 331 and 353 stayed below 0.95.
- **Currently running:** Nothing.
- **Next recommended action:** Stop. EXP-015 is not authorized.

See `SCIENTIFIC_CHECKPOINT.md`.

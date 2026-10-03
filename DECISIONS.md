# Project Decisions

## 2026-10-03 — Start with Adaptive Credit Memory as a question

- **Decision:** Use Adaptive Credit Memory as the first experimental direction, not as a settled architecture.
- **Why:** The theory motivation suggests that useful credit retention may vary with sequence dynamics. A small delayed-credit task can measure this need before committing to a mechanism.
- **Consequence:** Characterize a standard RNN first. Prototype adaptive allocation only if baseline measurements identify a repeatable opportunity.

## 2026-10-03 — Keep the first test small and synthetic

- **Decision:** Begin with a controllable delayed-bit task and a tiny vanilla RNN.
- **Why:** Delay, distractor difficulty, and target labels are directly controlled, making it easier to interpret gradient and memory measurements.
- **Consequence:** Results will be diagnostic toy-task evidence, not evidence of broad task ability or architectural novelty.

## 2026-10-03 — Keep theory and experiment repositories separate

- **Decision:** CreditLab is an independent Git repository beside AI-Architecture-Research.
- **Why:** CreditLab tests mechanisms experimentally and must not silently edit, synchronize with, or absorb the proof repository.
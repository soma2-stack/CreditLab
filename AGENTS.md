# CreditLab Agent Instructions

CreditLab is an experimental sandbox. Keep all work inside this repository. It is separate from AI-Architecture-Research: never edit that repository, copy large proofs into CreditLab, add it as a submodule, or create automatic synchronization between the repositories. The short motivation in `THEORY_NOTES.md` is sufficient context.

## Before work

Read `README.md`, `STATUS.md`, `TASKS.md`, `EXPERIMENTS.md`, and `RESULTS.md` before starting a task. Check the current Git status and preserve existing outputs.

## Research discipline

- Do not describe speculation as a theorem.
- Label claims as `EXPERIMENTAL RESULT`, `NUMERICAL EVIDENCE`, `HYPOTHESIS`, or `FAILED IDEA`.
- Prefer small controlled experiments before large models. Establish matched baselines before proposing improvements.
- Keep experiments reproducible. Record configurations, seeds, code version, measurements, compute budget, and deviations.
- Never hide failed experiments. Add results chronologically; do not overwrite old entries.
- Do not tune a benchmark or method after inspecting protected test results.
- Do not start expensive training without a clear question, a baseline, and a budget.
- Keep code modular and add quick tests for important mechanisms.
- Update `STATUS.md`, `TASKS.md`, and `RESULTS.md` after meaningful work.
- Stop on NaN/Inf, repeated crashes, CUDA out-of-memory, unsafe temperatures, or obviously invalid results. Record the stop and preserve useful logs.
- Never claim architecture success from one run. Separate a useful experimental finding from an architecture claim.

## Communication

The human user does not need to know the mathematics or coding details. Explain outcomes in plain English, then include technical details that help them check the work.

## Scope

The initial direction is Adaptive Credit Memory: test whether a recurrent model can decide when it needs extra learning-memory. This direction is provisional and may be rejected by evidence. Theory motivates experiments. Experiments do not modify or automatically validate the theory.
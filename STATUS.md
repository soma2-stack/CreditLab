# Project Status

- **Current phase:** Phase 0 — repository foundation / no architecture result yet.
- **What exists:** Project documentation, a minimal CPU-friendly PyTorch vanilla RNN, deterministic CPU seed helper, JSONL result logger, an EXP-001 design config, and quick foundation tests.
- **What has been tested:** Seven foundation tests pass; the CPU smoke check passes. Tests cover deterministic initialization, model output shapes, CPU execution, input validation, and JSONL logging. No task experiment has run.
- **Currently running:** Nothing.
- **Latest important result:** Foundation validation only; no experimental result has been collected.
- **Blockers:** None known. A clean-environment dependency installation has not been checked. EXP-001 still needs a small training/evaluation runner and a baseline run.
- **Next recommended action:** Implement the delayed-bit data generator and a minimal training/evaluation script for EXP-001; review its config and tests, then run within its small CPU budget.

Theory notes are motivation only. This repository contains no proof results.
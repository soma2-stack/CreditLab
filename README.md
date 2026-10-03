# CreditLab

CreditLab is a small experimental sandbox for recurrent learning, online credit assignment, and adaptive memory. It is a separate Git repository from AI-Architecture-Research. It does not change or automatically validate that project's theory.

## The starting idea: Adaptive Credit Memory

A recurrent model keeps ordinary hidden state to process its sequence. Adaptive Credit Memory asks whether it also needs extra state to remember information that will matter for a later learning update. The model might use little extra credit memory when old information fades quickly and preserve more when a sequence has long-lived dependencies.

This is a research question, not a settled architecture. The first work will use small synthetic tasks and ordinary baselines before adding a minimal adaptive mechanism.

## First experiment

EXP-001 is a tiny vanilla RNN baseline on a synthetic delayed-bit task. The task varies how many distractor steps separate a marked bit from the final prediction, and varies distractor difficulty. We will measure prediction accuracy/loss, gradient size by delay, hidden-state retention, and simple recurrent contraction indicators. This establishes whether useful learning credit becomes measurably harder to assign as delay grows. No large training run is planned.

## Setup

Use Python 3.10 or newer. From the repository root, install the package and test tools:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts/smoke_check.py
```

Experiment code and saved configurations will live in `experiments/` and `configs/`. Generated outputs belong in `results/<experiment_id>/`, never beside source code. See `AGENTS.md` before contributing.

**Theory motivates experiments. Experiments do not modify or automatically validate the theory.**
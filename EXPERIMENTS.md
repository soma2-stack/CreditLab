# Experiment Registry

Registry entries are plans until a dated result is added to `RESULTS.md`. Each run must save its exact configuration, seeds, code revision, measurements, compute use, and deviations under `results/<experiment_id>/`.

## EXP-001 — Delayed-bit vanilla RNN baseline

- **Question:** How do task performance and available learning signal change as the delay between a relevant bit and the final prediction grows?
- **Hypothesis:** Longer delays and stronger distractors will make the standard RNN baseline harder to train and may reduce useful gradient/hidden-state signal. This is a hypothesis, not an expected result.
- **Baseline:** Small vanilla tanh RNN trained with a terminal binary prediction loss.
- **Change tested:** None; baseline characterization only.
- **Dataset/task:** Synthetic delayed-bit sequence. One marked event carries a random bit; distractor steps follow; a terminal query marker requests the bit. Delay and distractor noise are controlled. No task ID is supplied.
- **Metrics:** Loss, accuracy by delay/difficulty, gradient norm by event age, relevant-bit decodability from hidden state, and a simple recurrent-Jacobian/contraction diagnostic where practical.
- **Seeds:** 17, 29, 43 (fixed in `configs/exp001_delayed_credit.yaml`).
- **Compute budget:** Small CPU pilot, at most 5 CPU-minutes per seed/config initially. Stop on invalid numerics or repeated failures; do not silently expand.
- **Stopping criteria:** Complete the registered tiny sweep or stop at the budget; preserve partial results and explain the stop.
- **Result:** Not run.
- **Interpretation:** Pending.
- **Follow-up:** Only after baseline analysis, decide whether to define EXP-002 credit-difficulty predictors.

## EXP-002 — Credit-difficulty signals

- **Question:** Can online signals such as delay, gradient decay, contraction, or event sparsity predict which sequences need longer-lived learning credit?
- **Hypothesis:** At least one measurable signal will separate easy and difficult delay cases beyond task metadata alone.
- **Baseline:** EXP-001 measurements and a simple delay-only predictor.
- **Change tested:** Compare candidate signals as predictors; no adaptive mechanism yet.
- **Dataset/task:** Reuse the EXP-001 task under a separately saved config.
- **Metrics:** Prediction quality for a preregistered difficulty label, calibration, and compute overhead.
- **Seeds / compute / stopping:** To be preregistered before running.
- **Result / interpretation / follow-up:** Not run.

## EXP-003 — Minimal adaptive-credit prototype

- **Question:** Does allocating a small amount of extra credit memory based on a validated difficulty signal improve the matched baseline tradeoff?
- **Hypothesis:** A simple adaptive allocation may preserve performance at lower average credit-state cost on mixed easy/hard sequences.
- **Baseline:** Standard RNN and a fixed-budget credit-memory control.
- **Change tested:** One minimal adaptive-credit rule, selected only after EXP-001/002.
- **Dataset/task:** Not yet frozen.
- **Metrics / seeds / compute / stopping:** Must be preregistered before implementation evaluation.
- **Result / interpretation / follow-up:** Not run.
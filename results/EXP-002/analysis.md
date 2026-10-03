# EXP-002 — Fixed residual path on corrected HARD-v2

**Status: complete. EXPERIMENTAL RESULT.** This run does not change any file in
`results/EXP-001/` or `results/EXP-001B/`. It asks one question: does a fixed
direct copy of the previous hidden state make delay-64 learning more reliable
than the vanilla tanh RNN?

The only model change is

```text
candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
h_t = h_{t-1} + 1.0 * candidate_t
```

The copy weight is 1. It is not learned, and it was not changed after these
test results. A half-and-half mix was not used, because multiplying the copy
by one half at every step would erase it long before delay 64. Both models
have the same hidden size (32), the same number of parameters (1,217), and the
same initial weights when the seed matches. Training matches EXP-001B: Adam,
learning rate 0.003, batch size 64, 400 updates, gradient clip 5, 1,536
training examples, 512 validation and 512 test examples, seeds 17, 29, and 43.

## What was compared

Primary delay: 64. Confirmation delay, run only after every delay-64 record
was saved: 128. Task: corrected HARD-v2 only.

The data generator is the same helper used for EXP-001B. That helper gives
HARD-v2 independent competitor bits, but it applies noise standard deviation
0.3 rather than the 1.0 written under HARD-v2 in the YAML. Measured noise on
the query channel of the distractor steps was about 0.30 in every run. First
competitor agreement with the target was about 0.48–0.54, close to a coin
flip. This generator was not edited.

A seed counts as successful when test accuracy is at least 0.75. Retention
counts as clearly readable when the probe R^2 is at least 0.5. Both cutoffs
were taken from the gap in the frozen EXP-001B HARD-v2 results and were locked
in `configs/exp002_residual_recurrent.yaml` before this run.

Software for this run: Python 3.11.9, PyTorch 2.13.0+cpu. Code revision
`b17fa60c8e3c`. Twelve runs, zero numerical failures, 64.7 seconds wall time.
EXP-001B was recorded on Python 3.12.14 and PyTorch 2.14.1+cpu.

## Delay 64

| Model | Seed 17 | Seed 29 | Seed 43 | Seeds at or above 0.75 |
|---|---:|---:|---:|---:|
| Vanilla | 0.4961 | 0.4570 | 1.0000 | 1 of 3 |
| Residual | 1.0000 | 1.0000 | 1.0000 | 3 of 3 |

The vanilla accuracies match the frozen EXP-001B HARD-v2 delay-64 accuracies
exactly (0.4961, 0.4570, 1.0000). Seed 17 training loss matches the frozen
value exactly, and seed 29 differs by 0.000003. Seed 43 remains the only
vanilla success, with small numeric drift in its loss and gradient-clip rate
relative to the frozen file.

Hidden-state probe R^2 at delay 64:

| Model | Seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| Vanilla | -0.001 | -0.037 | 0.995 |
| Residual | 0.977 | 0.978 | 0.976 |

On the two vanilla failures, a linear readout cannot recover the original bit
from the final state. On every residual seed, it can. Residual accuracy is
1.000 on examples whose first competitor agrees with the target and on
examples where it disagrees, so the result is not the old competitor shortcut.

## Credit and the residual path

The smallest step-to-step gain (minimum singular value of the true one-step
Jacobian, averaged over sampled steps) stays near zero for the vanilla RNN and
stays well away from zero for the residual RNN:

| Delay | Model | Minimum step gain, three seeds |
|---|---|---|
| 64 | Vanilla | 0.002, 0.009, 0.003 |
| 64 | Residual | 0.829, 0.714, 0.793 |
| 128 | Vanilla | 0.014, 0.023, 0.022 |
| 128 | Residual | 0.856, 0.812, 0.851 |

When the recurrent weight is zero, that residual gain is exactly 1, because
the update is a pure copy. After training it is about 0.7–0.9, so the direct
copy is still the main step-to-step path.

Sensitivity of the final answer to the original event is about 0 on failed
vanilla runs, 0.62 on the one successful vanilla run at delay 64, and about
8.5–22 on the three residual runs at delay 64. Part of that increase comes
from the much larger residual state, so the raw size should not be read as a
pure “how many times more credit” figure. The direction is clear: the original
event still moves the residual model’s answer, and it does not move the failed
vanilla models.

The per-example loss gradient stays tiny on the residual runs (about 0.0000001
to 0.00004 at delay 64). That does not contradict the sensitivity result.
EXP-001B already found that this loss gradient shrinks once the answers are
correct. These residual runs are correct, so a small loss gradient is
expected. The loss-gradient number by itself did not get larger.

## Delay 128 confirmation

| Model | Seed 17 | Seed 29 | Seed 43 | Seeds at or above 0.75 |
|---|---:|---:|---:|---:|
| Vanilla | 0.5098 | 0.5039 | 0.4590 | 0 of 3 |
| Residual | 1.0000 | 1.0000 | 1.0000 | 3 of 3 |

Residual probe R^2 values are 0.935, 0.965, and 0.947. Vanilla probe R^2
values are about 0. All three residual seeds are again perfect on both
competitor-agreement and competitor-disagreement examples.

This vanilla arm does not reproduce every frozen EXP-001B delay-128 number.
Seed 29 matches the frozen accuracy 0.5039. Seed 17 stays near chance
(0.5098 here, 0.4961 in the frozen file). Seed 43 was a success in the frozen
file (0.9648) and is a failure here (0.4590). Delay 64 reproduced the frozen
accuracies on all three seeds, so this is not a different task. The longer
delay is sensitive to the Python and PyTorch versions. The comparison that
matters for EXP-002 is the vanilla-versus-residual pair trained together in
this run.

## Numerical stability

Every recorded value was finite. No run was stopped for NaN or Inf. The
residual state is not bounded by 1. Its largest absolute entry was about 65
after delay 64 and about 129 after delay 128, roughly one unit per step. The
tanh candidate itself was usually large (mean absolute value about 0.90–0.96),
so most of the stored state is the copied history rather than a small
correction. Gradient clipping fired on about 9–16% of residual updates, and
on 0–6% of vanilla updates. Training still finished inside the original
400-update budget.

## Interpretation

This is outcome A from the preregistered list.

The residual model succeeded on all three seeds at delay 64. The vanilla model
succeeded on one. The residual model kept the original bit linearly readable,
and its direct step-to-step path stayed strong. That supports the specific
hypothesis that a direct recurrent copy makes this delayed-credit failure more
reliable. It does not prove the broader CreditLab theory, and it is not a
finished architecture.

The caveat is that the copy also makes the hidden state much larger. This
experiment does not separate “the direct path helped” from “a larger state
helped”. The loss-gradient diagnostic stayed small because the residual
answers were already correct.

## Files

- `raw_metrics.jsonl`: per-seed measurements.
- `primary_delay64_summary.json`: summary written before delay 128 started.
- `summary.csv`, `summary.json`: both delays.
- `config_used.yaml`: the preregistered configuration. The residual scale was
  not edited after the run.
- `environment.txt`: Python, PyTorch, and code revision.

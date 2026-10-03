# EXP-003 — Bounded half-and-half mixture

**Status: complete. EXPERIMENTAL RESULT.** Saved files for EXP-001, EXP-001B,
and EXP-002 were not changed, and EXP-002 was not retrained. The question is
whether EXP-002 succeeded because a direct state path preserves information,
or because its additive update let the hidden state grow very large.

The only new model is

```text
candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
h_t = 0.5 * h_{t-1} + 0.5 * candidate_t
```

Both weights were fixed at 0.5 before this run and were not changed afterward.
Hidden size stays 32. The parameter count stays 1,217. A zero start cannot
grow past 1 in absolute value, because each new state is a halfway mix of the
previous state and a tanh value.

One design fact, written down before the test: repeating a factor of 0.5 for
64 steps leaves almost nothing of the original state. This mixture keeps a
direct path at each step, but it does not keep a full-strength copy of the
first state all the way to the query.

## What was compared

Both models were trained in one process: Python 3.11.9, PyTorch 2.13.0+cpu,
code revision `ec69be58e3c8`. Corrected HARD-v2, seeds 17, 29, and 43, delays
64 then 128, Adam learning rate 0.003, 400 updates, batch size 64, and the
same data sizes as EXP-002. Twelve runs, zero numerical failures, 111.9
seconds. The generator was not edited, so HARD-v2 still uses noise standard
deviation 0.3.

The primary comparison is this matched pair. Saved EXP-002 magnitudes are
quoted only to describe how large the additive state was. Differences from
the frozen Python 3.12 files are not used as evidence.

A seed counts as successful at test accuracy of at least 0.75. Retention
counts as readable at probe R^2 of at least 0.5. Magnitude counts as
controlled when the largest absolute hidden entry is at most 1.01. These
cutoffs were locked in `configs/exp003_bounded_mixture.yaml` before the run.

## Delay 64

| Model | Seed 17 | Seed 29 | Seed 43 | Seeds at or above 0.75 |
|---|---:|---:|---:|---:|
| Vanilla | 0.4961 | 0.4570 | 1.0000 | 1 of 3 |
| Bounded mixture | 0.5293 | 0.4805 | 0.4688 | 0 of 3 |

Validation loss stayed near 0.69 on every bounded seed, the same region as a
model that has not learned the bit. The vanilla seed 43 run is the only
success in this pair. Its accuracy, 1.0000, matches the saved EXP-002 vanilla
run from this same Python 3.11 process. That is a same-environment repeat, not
a comparison with the older Python 3.12 files.

Probe R^2 at delay 64:

| Model | Seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| Vanilla | -0.001 | -0.037 | 0.995 |
| Bounded mixture | 0.000 | -0.028 | 0.003 |

The bounded model did not keep a readable copy of the original bit.

## Delay 128

| Model | Seed 17 | Seed 29 | Seed 43 | Seeds at or above 0.75 |
|---|---:|---:|---:|---:|
| Vanilla | 0.5098 | 0.5039 | 0.4590 | 0 of 3 |
| Bounded mixture | 0.4902 | 0.5098 | 0.4805 | 0 of 3 |

Bounded probe R^2 values are about 0 on all three seeds. Delay 128 agrees with
delay 64: the bounded model does not beat the matched vanilla model.

## Hidden-state size

Largest absolute hidden entry:

| Delay | Vanilla seeds 17, 29, 43 | Bounded mixture | Saved EXP-002 additive residual |
|---|---|---|---|
| 64 | 0.94, 0.81, 1.00 | 0.64, 0.78, 0.72 | 64.8, 64.2, 64.8 |
| 128 | 0.97, 0.92, 0.98 | 0.69, 0.41, 0.83 | 128.7, 126.9, 128.9 |

Every bounded run stayed under the 1.01 cutoff. The additive EXP-002 states
were about as large as the delay. Final root-mean-square size for the bounded
model was about 0.15–0.22, smaller than the successful vanilla seed and far
smaller than EXP-002.

## Credit and stability

Event sensitivity on the bounded runs was about 0.002 or smaller. The
per-example loss gradient was also about 0. These runs were not already
correct, so the small loss gradient here means the original event was not
receiving a useful training signal. The smallest one-step gain was about
0.07–0.21, higher than the failed vanilla values near 0.002–0.02, but far
below the 0.7–0.9 range of the additive EXP-002 model. That one-step change
did not produce readable memory or accuracy.

Every value was finite. No bounded update was gradient-clipped. The vanilla
seed 43 delay-64 run clipped about 5.5% of updates, as in EXP-002.

## Interpretation

This is the preregistered “advantage lost” outcome.

The bounded mixture kept the hidden state small and did not solve delay 64 or
delay 128 on any seed. The matched vanilla model still had its one delay-64
success. Removing additive growth also removed the EXP-002 result, where the
additive model scored 1.000 on every seed.

That does not prove magnitude was the only cause. A factor of 0.5 at every
step also erases a pure copy of the original state long before the query. The
supported statement is narrower: this fixed half-and-half path did not carry
the delayed bit, while the earlier additive path did, and the additive path
was the one whose state grew with the delay. This is not a proof of the
broader theory and it is not a new architecture.

## Files

- `raw_metrics.jsonl`: per-seed measurements.
- `primary_delay64_summary.json`: summary written before delay 128 started.
- `summary.csv`, `summary.json`: both delays.
- `exp002_magnitude_reference.json`: copied EXP-002 magnitudes, not a rerun.
- `config_used.yaml`: preregistered configuration. The weights were not edited
  after the run.
- `environment.txt`: Python, PyTorch, and code revision.

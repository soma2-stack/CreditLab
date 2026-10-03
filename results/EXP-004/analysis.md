# EXP-004 — Fixed-bound additive recurrence

**Status: complete. EXPERIMENTAL RESULT.** Saved EXP-001, EXP-001B, EXP-002,
and EXP-003 files were not changed. EXP-002 and EXP-003 were not rerun.

The question was whether the additive update can keep its delayed-learning
advantage when one fixed clamp limits the hidden state, without multiplying
the previous state by a number smaller than 1.

```text
candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
vanilla:   h_t = candidate_t
additive:  h_t = h_{t-1} + candidate_t
bounded:   h_t = clamp(h_{t-1} + candidate_t, min=-4, max=4)
```

`B = 4` was locked before any EXP-004 accuracy was collected and was not
changed. The clamp uses ordinary autograd. A coordinate outside the interval
gets zero local gradient through the clamp.

## Setup

All three models were trained in one process: Python 3.11.9, PyTorch
2.13.0+cpu. Training code revision `14bdc4786549`. Corrected HARD-v2 used
independent competitor bits, competitor rate 0.4, and distractor noise
standard deviation 0.3, passed explicitly into the generator. Measured noise
on the distractor query channel was about 0.300 in every run. For each seed
and delay, the three models shared the same data tensors, the same initial
parameters, and the same minibatch index list. Those fingerprints match.
Hidden size 32, 1,217 trainable parameters, Adam learning rate 0.003, 400
updates, batch size 64, gradient clip 5, 1,536 / 512 / 512 examples. Eighteen
runs, zero numerical failures, zero budget stops. The final checkpoint is the
reported model. Test accuracy was not computed during training.

Success means final test accuracy of at least 0.95. Clipping counts as active
when at least 1 percent of validation hidden coordinates are outside [-4, 4]
before the clamp at update 400. That 1 percent rule is only a reporting
convention. The continuous fractions are below.

Delay 64 was saved before delay 128 started. The additive control succeeded
on 3 of 3 delay-64 seeds, so fresh delay-128 training was run.

## Accuracy

| Delay | Model | Seed 17 | Seed 29 | Seed 43 | Successes |
|---|---|---:|---:|---:|---:|
| 64 | Vanilla | 0.4961 | 0.4570 | 1.0000 | 1 of 3 |
| 64 | Additive | 1.0000 | 1.0000 | 1.0000 | 3 of 3 |
| 64 | Bounded additive | 1.0000 | 0.5098 | 0.5195 | 1 of 3 |
| 128 | Vanilla | 0.5098 | 0.5039 | 0.4590 | 0 of 3 |
| 128 | Additive | 1.0000 | 1.0000 | 1.0000 | 3 of 3 |
| 128 | Bounded additive | 1.0000 | 0.5098 | 0.6289 | 1 of 3 |

The fresh additive control reproduced the earlier pattern: perfect accuracy
on every seed at both delays. The bounded model did that on seed 17 only.

## Retention and early-event diagnostics

Final probe R^2, fit on validation and scored on test:

| Delay | Model | Seed 17 | Seed 29 | Seed 43 |
|---|---|---:|---:|---:|
| 64 | Vanilla | -0.001 | -0.037 | 0.995 |
| 64 | Additive | 0.977 | 0.978 | 0.976 |
| 64 | Bounded additive | 0.996 | 0.000 | 0.000 |
| 128 | Vanilla | 0.004 | -0.005 | 0.002 |
| 128 | Additive | 0.935 | 0.965 | 0.947 |
| 128 | Bounded additive | 0.997 | 0.000 | 0.147 |

Where the bounded model scored 1.000, the original bit was readable and the
final answer still moved when the original event input changed (sensitivity
about 1.69 at delay 64 and 1.22 at delay 128). Its loss gradient was small
but not zero, which fits an already-correct answer. Where it failed, probe
R^2 was about zero and both the event sensitivity and the loss gradient were
exactly zero. Seed 43 at delay 128 is in between on accuracy (0.629) and
retention (0.147), and its event sensitivity was still exactly zero. These
input gradients are proxies for credit, not a direct measurement of the
parameter update.

## Clipping and magnitude

On every bounded run, including the successes, clipping was active. At the
final test rollout the fraction of hidden coordinates whose proposed value
was outside [-4, 4] was:

| Delay | Seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| 64 | 0.713 | 0.886 | 0.881 |
| 128 | 0.741 | 0.927 | 0.942 |

Every validation and test example had at least one clipped coordinate. The
first timestep was never clipped, because the state starts at zero and one
tanh step cannot reach 4. By the middle of the sequence, clipping was common
even at initialization. After training, the failed runs had late timesteps
essentially fully clipped. The successful runs still had roughly 15 to 25
percent of the late coordinates inside the interval.

The largest proposed value before the clamp was 5. That is one saturated tanh
step past a state already sitting on 4. After the clamp, the largest absolute
stored value was 4 on every bounded run. Final root-mean-square size was
about 3.82 and 3.73 on the two successes, and exactly 4 on the failures,
meaning those failed states were entirely on the boundary. The matched
additive states reached about 65 at delay 64 and about 129 at delay 128.
Vanilla states stayed at or below 1.

Clipping and failure happened together on four of the six bounded runs. That
co-occurrence is not, by itself, evidence that clipping caused the failure.
The successful runs were also heavily clipped.

## Stability

No NaN or Inf. No run hit the five-minute cap. Parameter-gradient clipping
fired on about 6 to 26 percent of bounded updates and about 9 to 16 percent
of additive updates. Training and validation loss on the bounded successes
fell near zero. On the bounded failures it stayed near 0.69, except seed 43
at delay 128, which ended near 0.60.

Frozen EXP-001 through EXP-003 hashes matched after the run.

## Interpretation

This is mixed, so the mechanism conclusion is **inconclusive**.

The additive control did reproduce: 3 of 3 seeds at both delays. Outcome D
does not apply. Outcome C does not apply either, because clipping was active
on every bounded run, far above the 1 percent convention.

Outcome A would require the bounded model to succeed while clipping is
active. That happened for seed 17 at both delays: accuracy 1.000, readable
memory, and a stored state capped at 4 rather than 65 or 129. It did not
happen for the other seeds. Outcome B would require the bounded model to
fail once the additive control had succeeded. That happened for seeds 29 and
43, not for seed 17.

So a stored state as large as 65 to 129 was not necessary on the one
successful bounded seed. The same fixed clamp did not reliably keep the
additive advantage. Whether the failures come from the magnitude limit, from
the zero gradient on clipped coordinates, or from both, is not settled here.
This is not a new architecture and it does not validate the broader theory.

## Files

- `raw_metrics.jsonl`, `loss_curves.jsonl`, `checkpoint_diagnostics.jsonl`
- `clipping_records.jsonl`, `comparison_table.csv`, `summary.csv`, `summary.json`
- `delay64_summary.json` written before delay 128
- `fingerprints_delay64.json`, `fingerprints.json`
- `frozen_hashes_before.json`, `frozen_hashes_after.json`
- `config_used.yaml`, `environment.txt`, `environment_preregistered.txt`
- `pre_run_tests.txt`

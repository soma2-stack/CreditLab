# EXP-001B — HARD shortcut and vanilla-RNN follow-up

**Status: complete. EXPERIMENTAL RESULT.** This follow-up leaves all files in
`results/EXP-001/` unchanged. It uses the same 32-unit vanilla tanh RNN,
training schedule, and seeds as EXP-001, over delays 16, 32, 64, 128, and 256.
Each of EASY, original HARD, and corrected HARD-v2 was run with seeds 17, 29,
and 43: 45 runs total, 0 numerical failures. Training settings were 400 Adam
updates, batch size 64, learning rate 0.003, and 1,536 training examples per
run. The recorded model-run times sum to 273 seconds. No tuning was done.

The input gradient for the original event and the new per-example terminal-loss
gradient were both saved. The new loss-gradient measure reports the mean,
median, and 95th percentile of each example's gradient norm before averaging
examples. It therefore cannot hide large and small example gradients by
cancelling them together. Its size also depends on how much prediction error
remains: a confident correct prediction has little loss gradient. The existing
event sensitivity measure is kept alongside it for this reason.

## HARD shortcut investigation

The original generator does not make competitor bits independent of the target.
When the target is −1, every competitor bit is −1. When the target is +1, a
competitor is +1 only half the time. On test examples that contain competitors,
the first competitor's bit therefore agrees with the answer about 75% of the
time. In HARD-v2, the measured agreement is about 50%, as intended. The
per-example rates are in `raw_metrics.jsonl` and their averages are in
`summary.csv`.

| Delay | Original HARD accuracy | HARD-v2 accuracy | EASY accuracy | HARD first competitor agrees with target |
|---:|---:|---:|---:|---:|
| 16  | 0.900 | 1.000 | 1.000 | 0.758 |
| 32  | 0.686 | 1.000 | 1.000 | 0.745 |
| 64  | 0.745 | 0.651 | 0.661 | 0.749 |
| 128 | 0.766 | 0.655 | 0.502 | 0.749 |
| 256 | 0.723 | 0.491 | 0.485 | 0.750 |

The original HARD advantage at the longest delays is substantially reduced when
that correlation is removed. At delay 256, for example, the model averaged
0.723 accuracy on original HARD and 0.491 on HARD-v2. Original HARD prediction
accuracy at delay 256 was 0.833 on examples whose first competitor agreed with
the answer, and 0.394 when it disagreed. That is strong evidence that the
shortcut contributes to the result. It does not prove that the RNN used only
the shortcut: the original event still helps on some examples and some seeds.

At delays 16 and 32, HARD-v2 is easier than original HARD in this run. The
shortcut is not the only difference between these randomly generated datasets,
and three seeds give limited precision. The clean conclusion concerns the
long-delay advantage, where original HARD remains well above chance while
HARD-v2 is near chance.

## Where the vanilla RNN fails

EASY and HARD-v2 were learned by all seeds through delay 32. At delay 64, each
has one successful seed and two seeds near chance. At delay 128, EASY has no
successful seeds; HARD-v2 has one successful seed and two near chance. At
delay 256, both are near chance for all seeds. So the first unreliable point
is delay 64, and performance is mostly lost by delay 128. This is consistent
with EXP-001's delay-64 seed split; its exact EASY per-seed pattern repeats
(seed 29 succeeds, seeds 17 and 43 fail).

The hidden-state probe measures whether a simple linear readout can recover
the original target from the last hidden state. It is near zero on failed EASY
runs at delays 64–256 and on failed HARD-v2 runs, while successful seeds have
high probe scores. In these failures, the measured original information is
mostly gone from the final hidden state. HARD-v2 at delay 128 illustrates the
seed split: two failures have probe scores near zero, while the successful
seed's score is 0.865. A probe only checks linear readability; it cannot rule
out every possible nonlinear code.

## Did learning credit vanish?

The old event-sensitivity metric remains near zero for several runs that still
score well on original HARD. The per-example loss-gradient measure is also
near zero on these runs. This fits the shortcut finding: if the model can
answer from competitors, little training signal needs to travel back to the
original event. On failed EASY and HARD-v2 runs, both event sensitivity and
loss gradient are generally near zero, alongside near-zero hidden-state
retention. In successful runs, event sensitivity is often substantial, while
the loss gradient may be small because the answers are already correct.

This improved diagnostic clarifies why the old measure alone was misleading:
small input sensitivity can mean either that the model has already solved the
task and is using a shortcut, or that learning failed. The paired loss-gradient
and hidden-state measurements help separate those cases. They support, but do
not establish, that a vanished learning signal caused forgetting; the run does
not show which happened first.

## Answers to the registered questions

- **Was HARD long-delay performance partly caused by a shortcut?** Yes. The
  generator's first competitor predicts the target with 75% accuracy, and the
  long-delay advantage mostly disappears in HARD-v2.
- **When does the vanilla RNN begin failing?** At delay 64, where seed outcomes
  split. By delay 128, EASY fails for all seeds and corrected HARD-v2 fails for
  two of three; by 256 both are near chance for all seeds.
- **Is the target information gone when it fails?** Usually it is not linearly
  readable from the final state in failed EASY/HARD-v2 runs. Successful seeds
  retain it.
- **Does the improved credit diagnostic show vanished signal?** Failed runs
  generally have tiny event sensitivity and tiny per-example loss gradients.
  The loss gradient alone is also tiny on easy, already-correct examples, so
  it must be read with sensitivity, accuracy, and retention.
- **Is delay 64 an optimization-luck region?** Yes for this small experiment:
  one of three same-seed runs succeeds on EASY and on HARD-v2, while two fail.
  EXP-001 showed the same EASY seeds succeeding and failing.

## Limits and files

These are small synthetic CPU runs with three seeds. They establish a generator
correlation and a repeatable pattern in this setup, not a general result about
RNNs. The competitor agreement statistic was recomputed from the deterministic
test examples and recorded latent event metadata after training; no model was
retrained for this statistic. `summary.json` and `summary.csv` were regenerated
from all 45 saved raw records after correcting a summary-only empty-group
handling issue. Training was not repeated.

- `raw_metrics.jsonl`: per-seed measurements and competitor-conditioned
  accuracies.
- `summary.csv`, `summary.json`: means and standard deviations across seeds.
- `config_used.yaml`: frozen run configuration.
- `environment.txt`: runtime versions and recorded code revision.

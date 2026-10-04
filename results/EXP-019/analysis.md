# EXP-019 — Explicit marked reset

**Status: stopped before training.** No accuracy was collected. The hold-task check failed on all ten seeds, so the forty training runs were not started.

The reset is a fixed rule, not a learned gate. It reads the write channel already present in the input and scales the carried state by one minus that number before the usual additive update. There is no threshold and no new parameter.

On a clean marked step, with the write channel exactly 1, the previous state is blocked. On a clean unmarked step, including the query, the update matches the ordinary additive model. Those local checks passed.

The hold task does not give the model a clean write channel. About 40 percent of the interior steps are competitor events, and the generator then adds distractor noise to that same channel. About 99 percent of interior write-channel values are not exactly 0 or 1. The only clean zero is the designated later bit, which the hold task leaves unmarked.

Because of that, the reset is not confined to the original event. The two models, started from the same weights, already disagree at the first distractor step. The largest hidden-state gap was about 145, and the largest final-score gap was about 166. The registered tolerance was 0.00001. The check took 0.3 seconds.

Training was not started. There is no seed accuracy, no replacement result, and no saturation comparison after learning. The ordinary additive control was not rerun as a scored experiment.

What is hard-coded is the reset formula. Nothing about selectivity was learned, because learning did not begin. A later experiment would need a preregistered definition of the marker that does not treat distractor noise as a reset. That definition was not invented here.

## Possible video-model relevance

An explicit update might help replace a stale attribute. Wiping the whole state on a noisy or incidental mark could also throw away unrelated scene information. This check did not reach a symbolic success, and it does not establish video capability.

## Files

The mismatch record is `investigation.json`. Frozen EXP-001 through EXP-018 files were unchanged. Code revision `1ce3059676bd`.

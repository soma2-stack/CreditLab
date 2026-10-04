# EXP-016 — Additive advantage under stronger noise

**Status: interrupted. Not a completed 10-seed result.**

The training process stopped during seed 439, vanilla, noise 1.0. It left a start record and no finish record, no checkpoint, and no score for that condition. A second start was refused because measurement files already existed. Nothing was resumed, retried, or replaced.

The last durable session clock was 218 seconds, recorded when that interrupted condition started. There is no verified finish time after that. No numerical failure was written. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Training code `8ccb69e9e447`.

Noise 1.0 is a new condition. It does not correct or replace the frozen noise-0.3 experiments.

## What finished

Seeds 401, 409, 419, 421, 431, and 433 finished all four conditions, including the separate diagnostic readout and the bit-flip check. Seed 439 finished only the two noise-0.3 trainings. Those two do not yet have a diagnostic readout.

Accuracy of the jointly trained readout:

| Seed | Vanilla 0.3 | Additive 0.3 | Vanilla 1.0 | Additive 1.0 |
|---|---:|---:|---:|---:|
| 401 | 0.494 | 1.000 | 0.506 | 1.000 |
| 409 | 0.518 | 1.000 | 0.482 | 1.000 |
| 419 | 0.506 | 1.000 | 0.529 | 1.000 |
| 421 | 0.730 | 1.000 | 0.570 | 1.000 |
| 431 | 0.506 | 1.000 | 0.533 | 1.000 |
| 433 | 0.543 | 1.000 | 0.457 | 1.000 |
| 439 | 0.506 | 1.000 | not run | not run |

Success means at least 0.95. On the finished runs, additive succeeded 7 of 7 times at noise 0.3 and 6 of 6 times at noise 1.0. Vanilla succeeded 0 of 7 and 0 of 6. The diagnostic readout, fit separately at each noise level, told the same story on the six fully finished seeds: additive at 1.0, vanilla near 0.5. Higher noise did not weaken either additive readout on those seeds. It also did not make vanilla succeed.

## Bit flips, on the six finished seeds

Additive final states still changed when only the original bit changed. No additive pair was exactly equal, and both pair members were answered correctly on nearly every pair, at both noise levels.

Vanilla was different, and not in only one way. Several vanilla models barely changed their prediction when the bit flipped. Seed 431 at noise 0.3 and seed 433 at noise 1.0 had identical final states on most pairs. Seed 421 at noise 0.3 still changed on about half of the pairs and scored 0.73. A vanilla miss is not always the same kind of miss.

## Unfinished

- Seed 439, vanilla, noise 1.0: started, then the process stopped. No score.
- Seed 439, additive, noise 1.0: not started.
- Seed 439 diagnostics and bit-flip checks: not started.
- Seeds 443, 449, and 457: none of the four conditions started.

## Limits

These counts are not out of ten. The missing seeds are not failures and are not successes. The completed additive runs are consistent with the advantage extending to noise 1.0, but the preregistered cohort is incomplete, so that is not the full result. This does not show general robustness, and it does not validate the broader theory.

Saved EXP-001 through EXP-015 files, and the scientific checkpoint, still match their hashes.

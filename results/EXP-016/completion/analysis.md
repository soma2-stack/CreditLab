# EXP-016 completion — full ten-seed noise comparison

**Status: complete through an authorized continuation.** The original session remains interrupted. Its missing time was not reconstructed, and it is not counted as a model failure.

All 26 saved checkpoints reproduced their saved correct counts. Fourteen new training jobs then ran, each in its own process, all with exit code 0. Seed 439, vanilla, noise 1.0, was restarted once from initialization. That was attempt 2. The interrupted attempt still has no score. Continuation time was 170.4 seconds, including verification and the missing diagnostics. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Completion code `4fa96b9058fa`.

## Original readout

| Seed | Vanilla 0.3 | Additive 0.3 | Vanilla 1.0 | Additive 1.0 |
|---|---:|---:|---:|---:|
| 401 | 0.494 | 1.000 | 0.506 | 1.000 |
| 409 | 0.518 | 1.000 | 0.482 | 1.000 |
| 419 | 0.506 | 1.000 | 0.529 | 1.000 |
| 421 | 0.730 | 1.000 | 0.570 | 1.000 |
| 431 | 0.506 | 1.000 | 0.533 | 1.000 |
| 433 | 0.543 | 1.000 | 0.457 | 1.000 |
| 439 | 0.506 | 1.000 | 0.506 | 1.000 |
| 443 | 0.482 | 1.000 | 0.516 | 1.000 |
| 449 | 0.494 | 1.000 | 0.449 | 1.000 |
| 457 | 1.000 | 1.000 | 0.490 | 1.000 |

Success means at least 0.95. Additive succeeded on 10 of 10 seeds at both noise levels. Vanilla succeeded on 1 of 10 at noise 0.3, seed 457, and on 0 of 10 at noise 1.0. The separate diagnostic readout crossed the same line on every one of the 40 conditions. Higher noise did not remove an additive success. It did remove the one vanilla success.

## Retention and bit flips

The validation-fit retention probe was high for every additive model, about 0.84 to 1.00. It was near zero for the vanilla misses. Seed 457’s successful vanilla model was the exception, with probe R^2 about 0.995. Seed 421’s partial vanilla model at noise 0.3 was about 0.30.

Flipping only the original bit still changed every additive final state, and both members of the pair were answered correctly. The same was true for the successful vanilla model. Some vanilla misses were different: seed 431 at noise 0.3 and seed 433 at noise 1.0 had identical final states on most pairs. Seed 457’s vanilla model at noise 1.0 still changed its state, but it no longer answered both members correctly.

## Interpretation

EXPERIMENTAL RESULT. Under this fixed budget, the additive advantage remains at distractor noise 1.0: 10 of 10 versus 0 of 10. Noise 0.3 is not a pure vanilla failure, because seed 457 solved it. This is not general robustness, and noise 1.0 does not replace the older noise-0.3 experiments. The two noise levels were trained separately. Nothing here is a transfer test from one noise level to the other.

## Possible video-model relevance

Learning under stronger interference is a reason to test visual persistence later. Success on this symbolic bit does not establish visual or video capability. Overwrite, where a later event must replace an earlier one, remains untested.

## Limits

No further interruption and no unfinished job. No numerical failure. Frozen EXP-001 through EXP-016 files, outside this completion folder, still match their hashes. The original interruption’s duration and cause remain unknown.

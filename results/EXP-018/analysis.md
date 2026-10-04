# EXP-018 — Where selective overwrite fails

**Status: complete. NUMERICAL EVIDENCE from frozen models.** Nothing was trained and no new readout was fit. All 40 saved checkpoints and classifiers reproduced their saved correct counts. The new measurements used fresh split index 4 and took 11.4 seconds. Python 3.11.9, PyTorch 2.13.0+cpu, scikit-learn 1.9.1. Diagnostic code `252565c7f858`.

The question is where a marked new bit fails: it never enters, it enters and later disappears, or it remains in the state while the saved answers ignore it. Exact coordinate equality and the size of the difference are reported separately. A float64 rerun does not repair the float32 result. Casting the saved weights cannot recover precision already lost.

## Did the marked new bit enter?

For the additive models trained on the selective task, the median size of the immediate state change was:

| Seed | Fraction exactly equal | Median difference | Still equal at the query |
|---|---:|---:|---:|
| 503 | 0.943 | 0 | 0.988 |
| 509 | 0.000 | 0.183 | 0.547 |
| 521 | 0.930 | 0 | 0.936 |
| 523 | 0.000 | 0.392 | 0.002 |
| 541 | 0.000 | 0.153 | 0.000 |
| 547 | 0.191 | 0.00017 | 0.266 |
| 557 | 0.816 | 0 | 0.859 |
| 563 | 0.092 | 0.0025 | 0.115 |
| 569 | 0.986 | 0 | 0.996 |
| 571 | 0.000 | 0.015 | 0.137 |

Seeds 503, 521, 557, and 569 usually show no ordinary-precision change, and the typical difference stays tiny in float64 as well. Seeds 523 and 541 do change, and that change is still there at the query. Seed 509 changes and then about half of those differences are gone by the query. Seeds 547, 563, and 571 change only a little.

The vanilla selective models are different. The new bit changes the state immediately on every seed, with a median difference around 0.5 to 1.4. On nine seeds that difference is still present at the query. On seed 547 it has disappeared by the query.

## Did the saved answers follow the new bit?

No. On every selective model, vanilla and additive, both saved readouts followed the marked new bit on none of the 512 pairs. That includes the seeds whose states still differed at the end. A difference that lasts is not evidence that some other classifier could read the new bit. The separate readout was already fit; it also does not follow the marked new bit.

When the later bit is unmarked, the separate additive readout keeps the old answer on essentially every pair. The jointly trained additive readout does that on many pairs and not all of them. Flipping only the update marker never switches the saved answer from the old bit to the new one. On the more saturated seeds the marker often fails to change the state as well.

The hold-trained additive models still answer the old bit, which is what they were trained to do. They are not scored here as failed replacement models.

## Saturation

At the update, additive candidates are usually pressed near plus or minus one. About 83 to 99 percent of additive coordinates meet the fixed reporting rule `1 - candidate^2 < 1e-6`. Vanilla candidates do not. Within an additive seed, the more saturated half of the examples is more often exactly unchanged. That is an association. It is not proof that saturation caused the training failure. Some highly saturated seeds, especially 523, still show a moderate state change.

## Float64

Exact equality is sensitive to precision. On several additive seeds, float32 calls most pairs identical while float64 calls many of them different. The typical size of those differences does not become large in float64. A recorded zero in float32 is not a mathematical zero. It is also not a hidden large effect.

## What kind of failure

- Weak entry, with saturated candidates: additive seeds 503, 521, 557, and 569, and the typical case for 547 and 563.
- Entry followed by later loss on many pairs: additive seed 509, and vanilla seed 547.
- A state change that lasts, while both saved readouts still ignore the new bit: additive seeds 523 and 541, and most vanilla seeds.

These descriptions can overlap inside one seed. They do not show that the old bit was erased, and they do not show a universal limit of the model.

## Possible video-model relevance

A visual system would need to write an actual change, keep the changed attribute, and decode the current value. These symbolic measurements do not establish that video capability.

## Files

Measurements and hashes are in this folder. Frozen EXP-001 through EXP-017 files were unchanged.

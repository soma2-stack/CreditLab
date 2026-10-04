# EXP-007 execution audit

This note records how the interrupted EXP-007 run was finished. It does not
change any saved EXP-007 measurement.

## Code revisions

- Preregistration commit: `281061a` (`Preregister full training versus readout-only training for EXP-007`).
- Plain-text test-log commit: `23221a6`. That commit did not change training code.
- Forward-check commit: `facb570b7886` (`Allow a tiny paired-forward tolerance before EXP-007 training`).
- Resume commit: `a637771d92ed` (`Resume EXP-007 without repeating completed runs`).
- Results commit: `200f273`.

The saved raw records use two training revisions:

- `facb570b7886`: all 60 delay-64 runs, plus 46 delay-128 runs through additive readout-only for seed 191.
- `a637771d92ed`: the remaining 14 delay-128 runs, from bounded full and readout-only for seed 191 through seed 223.

There are 120 records and 120 unique condition keys. No duplicate condition was found.

## What the code change did

`facb570` changed only the pre-training check that compares a full model with its readout-only copy. Copied vanilla modules have identical weights, but their float32 forwards are not always bitwise identical. The check was relaxed from an exact output hash to a maximum logit gap of 1e-5. The forward equations, Adam settings, update count, data, and clipping threshold were not changed.

`a637771` added a resume path. It skips condition keys already present in `raw_metrics.jsonl` and does not retrain them. It does not change the model equation or the optimizer. New records from that commit are only the 14 unfinished delay-128 conditions listed above.

## Fingerprints

`results/EXP-007/fingerprints.json` contains 3 rows, all for delay 128. A delay-64 fingerprint file was not saved. The missing delay-64 fingerprints were not reconstructed, and the saved EXP-007 records were not edited to fill that gap.

## Runtime

The exact wall-clock cost of the interrupted process is not available. The resume code estimated prior elapsed time from file timestamps. That estimate is not an independently verified cumulative execution time.

What is directly available is the sum of the per-run `runtime_seconds` fields written before the interruption: 612.6 seconds. The final `summary.json` field `elapsed_seconds` is 752.788. That number comes from the runner clock after the timestamp estimate. It should not be treated as a stopwatch total across both processes.

No numerical failure or per-run budget stop is recorded. All 120 planned training conditions and 120 audits are present.

## Effect on the comparison

The training comparison is not compromised. Every condition finished, none was duplicated, and the resume did not change the update rule. The missing delay-64 fingerprint file is a record-keeping gap, not evidence that those runs used different data or different initial weights.

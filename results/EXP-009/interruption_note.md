# EXP-009 interruption note

The training process stopped during delay 128. Automatic resume is disabled.
These files were left as the process wrote them. Nothing was retrained.

## What finished

All 40 delay-64 runs have a saved terminal record with status `ok`.
`delay64_summary.json` was written before delay 128 began.
Full training succeeded on 10 of 10 delay-64 seeds, so the gate allowed delay 128.

## Where it stopped

The runtime log has a start event and no finish event for:

- delay 128
- seed 101
- regime `bias_plus_readout`

That run has no raw-metric row and no checkpoint.
The two delay-128 runs saved before it are seed 101 full and seed 101 readout only.
No other delay-128 run was started.
No bit-flip audit was started. There is no `audit_metrics.jsonl` and no runner-written `summary.json`.

## Time

Each finished run recorded its own `runtime_seconds` from `time.perf_counter`.
The last session mark in `runtime_log.jsonl` is 229.085 seconds, at the start of the unfinished run.
The unfinished run has no recorded finish time.
File timestamps were not used as a runtime estimate.

## Historical files

A comparison after the stop found that the 397 frozen EXP-001 through EXP-008 hashes still matched `frozen_hashes_before.json`.
The runner did not get as far as writing `frozen_hashes_after.json`.

## Continuation

Do not continue this session automatically.
Coordinator review is required before any further EXP-009 training or audit.

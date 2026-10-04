# Documentation audit

This pass read saved records. It did not retrain, refit, or regenerate historical summaries.

## Hash check

Before the new documents were written, 933 saved experiment files under `results/EXP-001` through `results/EXP-014` and `configs/exp*.yaml` were hashed. The same set was hashed again after the documentation edits. The two manifests match. Those artifacts were not modified.

The manifests are `documentation_audit/historical_hashes_before.json` and `documentation_audit/historical_hashes_after.json`.

## EXP-014 denominator

Re-counted from `results/EXP-014/diagnostic_metrics.jsonl`, final-16 rows only:

- Original readout at or above 0.95: 1/10 (seed 311).
- Diagnostic readout at or above 0.95: 8/10.
- Original misses: 9.
- Rescued misses: 7. Seeds 307, 313, 317, 337, 347, 349, 359.
- Still below 0.95: seeds 331 and 353.

The frozen `results/EXP-014/analysis.md` correctly says the diagnostic succeeded on 8 of 10 and names seeds 331 and 353. It does not say that 8 misses were rescued. The shorter project entries were easy to read that way. The clarification is appended in `EXPERIMENTS.md` and `RESULTS.md`. The analysis file was not edited.

## EXP-012 subgroup, not pooled

Re-counted from `results/EXP-012/metrics.jsonl`, delay 128 and `final_16_clip5` only: 10 original misses, 10 rescued, none still below 0.95. Seeds are 101–223 and the data base seed is 2000. EXP-014’s 7/9 uses seeds 307–359 and base seed 3000. These are not one sample.

The headline EXP-012 figure, 36 of 37, also includes delay 64 and the no-clipping condition. The one miss in that broader count is delay 128, seed 179, final-16 without clipping. That is a different cell from EXP-014.

## Other gaps left as they are

- EXP-001 and EXP-001B record Python 3.12.14 and PyTorch 2.14.1. That interpreter was not recovered. Later comparisons use matched runs under Python 3.11.9 and PyTorch 2.13.0.
- Corrected HARD-v2 in the early generator used distractor noise 0.3 in practice. A historical YAML field says 1.0. That field was not rewritten. Later experiments pass 0.3 explicitly.
- EXP-004 did not save checkpoints. EXP-005 replayed the eighteen models and matched the saved metrics within preregistered tolerances. The audit states that this is consistency evidence, not proof that every weight bit matches.
- EXP-007 was interrupted and resumed. Its accounted runtime uses file timestamps and is not a verified stopwatch total. Delay-64 fingerprints were not saved and were not invented later.
- EXP-009’s first session stopped during delay 128, seed 101, bias plus readout. That attempt has no finish time. A later authorized continuation finished the planned delay-128 comparison. The two are separate.
- EXP-007 froze both input-side and recurrent-side weights in the readout-only condition. A later clarification says that result is about training the combined module, not about the hidden-to-hidden matrix alone. The original EXP-007 analysis file was left as written.
- Additive recurrence is `h_t = h_(t-1) + tanh(...)`. The half-and-half model in EXP-003 is a different equation and it failed. The clamp at 4 is a third equation and was mixed.

No contradiction was found between the EXP-014 jsonl counts and the success totals already printed in the EXP-014 ledger. The gap is the rescue denominator, not the underlying rows.

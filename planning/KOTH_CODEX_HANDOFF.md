# Codex handoff: four-cell HARD_V2 benchmark

Work only on branch `planning/additive-width-delay-scaling` of `soma2-stack/CreditLab`. Do not modify `main` or prior experiment branches. Do not begin GPU training without explicit user approval.

## Four cells
- Original HARD_V2, light noise 0.3, seed 17
- Original HARD_V2, light noise 0.3, seed 29
- Corrected HARD_V2, heavy noise 1.0, seed 17
- Corrected HARD_V2, heavy noise 1.0, seed 29

Runner: `experiments/run_koth_four_cell.py`.
Checks: `tests/test_koth_four_cell.py`.

## Outstanding correction
The original `src/creditlab/delayed_task.py` must remain untouched. Its HARD_V2 implementation draws a random `flip` tensor that is unused, advancing its RNG. The corrected generator must skip that draw, in addition to using noise 1.0. Implement corrected logic in a separate module or a versioned function and call it only for the fixed cells. Add regression tests proving both the original and corrected variants are reproducible and that the corrected generator does not consume the unused random draw.

## Checks before any training
1. Install dependencies and run `python -m unittest tests/test_koth_four_cell.py` (or `python -m unittest discover -s tests -p 'test_koth_four_cell.py'`).
2. Run `python experiments/run_koth_four_cell.py` without `--run`; it must print preflight only and start no training.
3. Verify distinct train, validation and test sets; seed consistency; correct competitor distribution and noise settings.
4. Verify the four runs use identical model architecture, delay, hidden width, optimizer, update count, batch size and data sizes.
5. Verify CUDA availability and temperature monitoring. Abort above 86 C, on OOM, or nonfinite values. Do not silently proceed if the temperature sensor is unavailable.
6. Preserve existing result files, capture environment/commit/configuration, and report pass/fail with any unresolved issues.

**Stop after preflight and report results. Do not use `--run` until the user explicitly approves GPU training.**

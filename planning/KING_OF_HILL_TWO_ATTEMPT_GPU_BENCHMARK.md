# King of the Hill — two-attempt RTX 3060 benchmark
Status: PREPARATION ONLY. No training started.

## Objective
Measure elapsed training time, peak allocated/reserved CUDA VRAM, GPU temperature, and held-out delayed-bit accuracy for two matched attempts with seeds 17 and 29. This is a timing pilot, not a final tournament/elimination policy.

## Guardrails
- Preserve all previous CreditLab experiments, configurations, checkpoints, and results.
- Run on a separate working branch with unique output paths per attempt; never overwrite.
- No training in CI; no automatic start. Obtain user's explicit approval before local training.
- Stop for CUDA OOM, NaN/Inf, temperature above 86 C, or repeated errors; prefer operating at or below 80 C.
- Record exact commit, environment, PyTorch/CUDA versions, GPU name, seeds, configuration, wall time, peak VRAM, temperature sampling method, and failures.
- Train/validation/test splits must be distinct; don't tune to final test.
- HARD_V2 caveat: make_split uses easy_noise_std for HARD_V2, and generate_batch consumes unused flip draws. Do not silently change the benchmark generator; request a decision before fixing or selecting another mode.

## Implementation checklist for local Codex (do not launch training yet)
1. Inspect existing delayed_task.py, model definitions, training scripts and tests on the experiment branch.
2. Create a standalone GPU-capable two-seed benchmark runner and a fixed, versioned config without editing original runners or generator.
3. Check CUDA availability; refuse CPU fallback for the GPU timing pilot. Record torch.cuda.max_memory_allocated and max_memory_reserved. Sample GPU temperature via NVML or nvidia-smi if available, and record unavailable sensors honestly.
4. Add dry-run/preflight that checks paths, config, dataset splits, seed separation, model shape, finite forward/backward on a tiny synthetic batch, GPU temperature guard, and output collision refusal; preflight must not execute training updates.
5. Add unit tests for deterministic generation, split independence, no train/test mixing, same settings across seeds, unique outputs, and safety stops. Run only tests and dry-run before user approval.
6. Once user explicitly approves local execution, run exactly two matched attempts; save separate metrics and compare both without cherry-picking.

## User decisions to request when they matter
- Whether to use HARD_V2 unchanged for timing comparability or first repair its noise-mode mismatch (which changes benchmark definition).
- Training budget (maximum minutes or updates) before starting actual runs.
- Whether failed attempts should later be eliminated or retried in a full competition. Two runs here are a timing pilot only.

## Completion report format
What was implemented; tests passed/failed; no training versus training actually executed; confirmed issues; exact files/commit; remaining decisions and next action.

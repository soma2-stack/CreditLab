"""Synthetic delayed-marked-bit task used by EXP-001.

Sequence layout (length = delay + 2):
    t = 0            important event:  x[:, 0] = bit in {-1, +1}, x[:, 1] = 1
    t = 1 .. T-2     distractors:      noise only (EASY) or noise plus
                                       competing marked events (HARD)
    t = T-1          terminal query:   x[:, 2] = 1, other channels near zero

The target label is the bit from the timestep-0 event.  The number of
distractor steps between the event and the query equals ``delay``.

All randomness comes from an explicitly seeded ``torch.Generator`` so runs are
reproducible on CPU.  This module intentionally contains no task-ID channel;
the model must infer from structure alone.
"""

from __future__ import annotations

import torch

MODES = ("EASY", "HARD")


def sequence_length(delay: int) -> int:
    """Total sequence length for a given delay (event step + distractors + query)."""
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
        raise ValueError("delay must be a positive integer")
    return delay + 2


def generate_batch(
    num_samples: int,
    delay: int,
    mode: str,
    *,
    seed: int,
    distractor_noise_std: float,
    competitor_rate: float = 0.0,
    competitor_flip_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one deterministic batch of the delayed-marked-bit task.

    Returns ``(inputs, targets)`` with shapes ``[N, delay+2, 3]`` and
    ``[N]`` (float values in {-1.0, +1.0}).

    EASY mode uses simple Gaussian distractors (competitor_rate ignored when
    0).  HARD mode additionally injects competing marked events into
    distractor positions; competitors carry the opposite bit with probability
    ``competitor_flip_prob`` when the target bit is +1 (and the same bit
    otherwise), so they conflict with the true answer at the query step.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if distractor_noise_std < 0:
        raise ValueError("distractor_noise_std must be >= 0")
    if not 0.0 <= competitor_rate <= 1.0:
        raise ValueError("competitor_rate must be in [0, 1]")

    length = sequence_length(delay)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    bits = torch.where(
        torch.rand(num_samples, generator=gen) < 0.5, -1.0, 1.0
    )

    inputs = torch.zeros(num_samples, length, 3)
    # Channel 0: event bit at t=0, random ±bit-scaled noise elsewhere.
    inputs[:, 0, 0] = bits
    inputs[:, 1:, 0] = (
        torch.randint(0, 2, (num_samples, length - 1), generator=gen).float() * 2 - 1
    )
    # Channel 1: event marker only at t=0.
    inputs[:, 0, 1] = 1.0
    # Channel 2: query marker only at the final step.
    inputs[:, -1, 2] = 1.0

    # HARD-mode competing marked events in distractor positions are applied
    # BEFORE noise so that distractor steps carry no marker at all in EASY.
    if competitor_rate > 0 and length > 2:
        comp_mask = (
            torch.rand(num_samples, length - 2, generator=gen) < competitor_rate
        )
        flip = (
            torch.rand(num_samples, length - 2, generator=gen) < competitor_flip_prob
        )
        # Competitor bit: opposite of target when target bit is +1 and flip is
        # active; same as target otherwise.  Net effect: roughly half of the
        # competitors on +1-target trials point to the wrong answer.
        comp_bit = torch.where(
            flip & (bits.unsqueeze(1) > 0), -1.0, bits.unsqueeze(1).expand(-1, length - 2)
        )
        inputs[:, 1:-1, 0] = torch.where(comp_mask, comp_bit, inputs[:, 1:-1, 0])
        inputs[:, 1:-1, 1] = torch.where(comp_mask, 1.0, inputs[:, 1:-1, 1])

    # Gaussian distractor noise on every non-event, non-query step.
    if distractor_noise_std > 0 and length > 2:
        noise = torch.randn(num_samples, length - 2, 3, generator=gen)
        inputs[:, 1:-1, :] += distractor_noise_std * noise

    return inputs, bits


def make_split(
    num_samples: int,
    delay: int,
    mode: str,
    *,
    run_seed: int,
    split_index: int,
    base_seed: int = 1000,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-split data pull derived from (run_seed, split, delay, mode)."""
    cfg = config or {}
    easy_std = cfg.get("easy_noise_std", 0.3)
    hard_std = cfg.get("hard_noise_std", 1.0)
    competitor_rate = cfg.get("competitor_rate", 0.4 if mode == "HARD" else 0.0)
    if mode == "EASY":
        competitor_rate = 0.0
    combined_seed = (
        (int(run_seed) * 1_000_003 + int(base_seed) * 9_973 + delay * 31 + split_index)
        % (2**31 - 1)
    )
    return generate_batch(
        num_samples,
        delay,
        mode,
        seed=combined_seed,
        distractor_noise_std=hard_std if mode == "HARD" else easy_std,
        competitor_rate=competitor_rate,
        competitor_flip_prob=cfg.get("competitor_flip_prob", 0.5),
    )

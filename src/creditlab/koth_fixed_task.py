"""Corrected HARD_V2 generator; original remains unchanged."""
from __future__ import annotations
import torch
from creditlab.delayed_task import MODES, sequence_length

def generate_batch_fixed(
    num_samples: int,
    delay: int,
    mode: str,
    *,
    seed: int,
    distractor_noise_std: float,
    competitor_rate: float = 0.0,
    competitor_flip_prob: float = 0.5,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Generate one deterministic batch of the delayed-marked-bit task.

    Returns ``(inputs, targets)`` with shapes ``[N, delay+2, 3]`` and
    ``[N]`` (float values in {-1.0, +1.0}).

    EASY mode uses simple Gaussian distractors. HARD reproduces the original
    EXP-001 competitor rule. HARD_V2 uses the same events and noise, but each
    competitor bit is sampled independently of the target. Optional metadata
    returns the latent competitor mask and bits for diagnostics.
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

    comp_mask = torch.zeros(num_samples, max(length - 2, 0), dtype=torch.bool)
    comp_bits = torch.zeros(num_samples, max(length - 2, 0))
    # Competing marked events in distractor positions are applied
    # BEFORE noise so that distractor steps carry no marker at all in EASY.
    if competitor_rate > 0 and length > 2:
        comp_mask = (
            torch.rand(num_samples, length - 2, generator=gen) < competitor_rate
        )
        if mode == "HARD_V2":
            # Independent balanced bits remove target information from a
            # competitor while preserving the original event rate.
            comp_bit = torch.where(
                torch.rand(num_samples, length - 2, generator=gen) < 0.5,
                -1.0, 1.0,
            )
        else:
            flip = torch.rand(num_samples, length - 2, generator=gen) < competitor_flip_prob
            comp_bit = torch.where(flip & (bits.unsqueeze(1) > 0), -1.0, bits.unsqueeze(1).expand(-1, length - 2))
        comp_bits = comp_bit
        inputs[:, 1:-1, 0] = torch.where(comp_mask, comp_bit, inputs[:, 1:-1, 0])
        inputs[:, 1:-1, 1] = torch.where(comp_mask, 1.0, inputs[:, 1:-1, 1])

    # Gaussian distractor noise on every non-event, non-query step.
    if distractor_noise_std > 0 and length > 2:
        noise = torch.randn(num_samples, length - 2, 3, generator=gen)
        inputs[:, 1:-1, :] += distractor_noise_std * noise

    if return_metadata:
        return inputs, bits, {"competitor_mask": comp_mask, "competitor_bits": comp_bits}
    return inputs, bits


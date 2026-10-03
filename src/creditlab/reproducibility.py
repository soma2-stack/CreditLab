"""Small deterministic-seeding helpers for CPU experiments."""

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch's CPU generator.

    This helper does not initialize, seed, or use CUDA. Call it before model
    construction and data generation for repeatable CPU experiments.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
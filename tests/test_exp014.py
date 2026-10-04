"""Checks for EXP-014. These do not use new-cohort performance to choose settings."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp010 import forward_regime  # noqa: E402
from run_exp014 import OLD_SEEDS, SEEDS, build_initialized, load_config


def test_new_seeds_do_not_collide_and_delay_is_128_only():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["data"]["base_seed"] == 3000
    assert cfg["delay"] == 128
    assert not set(cfg["seeds"]) & OLD_SEEDS
    assert cfg["classifier"]["C"] == 1.0
    assert cfg["classifier"]["l1_ratio"] == 0.0
    assert cfg["budget"]["max_cpu_seconds_total"] == 600


def test_copies_match_and_the_cutoff_preserves_forward_values():
    cfg = load_config()
    model = build_initialized(cfg, 307)
    other = build_initialized(cfg, 307)
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(left, right)
    inputs = torch.randn(2, 130, 3)
    with torch.no_grad():
        reference, _ = model(inputs)
        detached, _states, cut = forward_regime(model, inputs, "final_16")
    assert cut == 114
    assert torch.equal(reference, detached)

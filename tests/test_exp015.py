"""Checks for EXP-015. These do not score the length-generalization sets."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp004 import load_hard_v2  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp015 import SEEDS, DELAYS, magnitude_indices, load_config


def test_delays_endpoints_and_new_data_rules_are_locked():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["delays"] == {"reference": 128, "primary": 256, "secondary": 512}
    assert DELAYS == (128, 256, 512)
    assert cfg["data"]["base_seed"] == 4000
    assert cfg["data"]["evaluation_split_index"] == 4
    assert cfg["data"]["audit_split_index"] == 5
    assert cfg["budget"]["no_training"] is True
    assert cfg["budget"]["no_refit"] is True
    assert "initialized_diagnostic" in cfg["endpoints"]
    assert "initialized_original" not in cfg["endpoints"]


def test_magnitude_indices_and_query_location():
    assert magnitude_indices(128) == {
        "after_event": 0, "halfway_distractor": 64, "before_query": 128, "after_query": 129,
    }
    assert magnitude_indices(512)["halfway_distractor"] == 256
    inputs, targets = load_hard_v2(
        4, 8, 307, 4, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=4000,
    )
    assert inputs.shape[1] == 10
    assert torch.equal(inputs[:, -1, 2], torch.ones(4))
    assert torch.equal(inputs[:, 0, 1], torch.ones(4))
    assert torch.equal(inputs[:, 0, 0], targets)
    positive, negative = make_counterfactual_pairs(inputs)
    assert pairs_differ_only_on_original_bit(positive, negative)

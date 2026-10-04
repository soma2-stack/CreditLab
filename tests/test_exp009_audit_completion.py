"""Checks for the EXP-009 audit continuation. These do not load checkpoints."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp009_audit_completion import (  # noqa: E402
    DELAY,
    REGIMES,
    SEEDS,
    coordinate_equal_stats,
    saved_correct_count,
)


def test_registration_constants_cover_the_delay64_cohort():
    assert DELAY == 64
    assert SEEDS == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert REGIMES == ["full", "readout_only", "bias_plus_readout", "matrices_plus_readout"]
    assert len(SEEDS) * len(REGIMES) == 40


def test_saved_accuracy_converts_to_an_integer_count():
    assert saved_correct_count(1.0) == 512
    assert saved_correct_count(0.87109375) == 446
    assert saved_correct_count(0.9453125) == 484


def test_coordinate_equality_is_not_an_l2_proxy():
    positive = torch.zeros(2, 3)
    negative = torch.zeros(2, 3)
    negative[1, 0] = 1.0
    stats = coordinate_equal_stats(positive, negative)
    assert stats["final_states_coordinate_equal_count"] == 1
    assert stats["n_pairs"] == 2


def test_pairs_change_only_the_original_bit():
    base = torch.randn(4, 6, 3)
    positive, negative = make_counterfactual_pairs(base)
    assert pairs_differ_only_on_original_bit(positive, negative)
    positive[:, 0, 0] = 0
    negative[:, 0, 0] = 0
    assert torch.equal(positive, negative)

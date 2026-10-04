"""Checks for EXP-013. These do not score the real task to choose settings."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.reproducibility import seed_everything  # noqa: E402
from creditlab.models import ResidualTanhRNN  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp013 import REPRESENTATIONS, load_config, recurrent_equal, reconstruct_initialized


def test_config_locks_three_representations_and_the_exp012_classifier():
    cfg = load_config()
    assert [item["name"] for item in cfg["representations"]] == ["initialized", "full_history", "final_16"]
    assert REPRESENTATIONS == ("initialized", "full_history", "final_16")
    assert cfg["classifier"]["C"] == 1.0
    assert cfg["classifier"]["l1_ratio"] == 0.0
    assert cfg["classifier"]["solver"] == "lbfgs"
    assert cfg["classifier"]["max_iterations"] == 2000
    assert cfg["budget"]["max_cpu_seconds_total"] == 300
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]


def test_reconstruction_is_exact_and_has_no_optimizer():
    seed_everything(101)
    first = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    second = reconstruct_initialized(101)
    assert recurrent_equal(first, second)


def test_pairs_change_only_the_original_bit():
    base = torch.randn(3, 5, 3)
    positive, negative = make_counterfactual_pairs(base)
    assert pairs_differ_only_on_original_bit(positive, negative)

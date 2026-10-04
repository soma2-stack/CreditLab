"""Checks for EXP-022. These do not train the two-memory comparison."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp022 import (  # noqa: E402
    JOBS,
    SEEDS,
    build_models,
    load_config,
    make_two_memory,
    reset_equation_checks,
    structural_whole_reset,
    subgroup_masks,
)


def test_design_balance_and_structural_limit():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["data"]["base_seed"] == 9000
    assert len(JOBS) == 30
    data = make_two_memory(512, 701, 2)
    assert data["inputs"].shape == (512, 130, 6)
    groups = subgroup_masks(data["A"], data["B"], data["C"], data["F"], data["S"], data["Q"])
    assert sum(int(mask.sum()) for mask in groups.values()) == 512
    reset_equation_checks()
    whole = build_models(1)["whole_reset_additive"]
    report = structural_whole_reset(whole)
    assert report["structural_independence"] is True
    addressed = build_models(1)["addressed_reset_additive"]
    hidden = torch.randn(2, 32)
    marked = torch.zeros(2, 6)
    marked[:, 3] = 1
    marked[:, 4] = 1
    carried = addressed.carried_state(marked, hidden)
    assert torch.equal(carried[:, :16], torch.zeros(2, 16))
    assert torch.equal(carried[:, 16:], hidden[:, 16:])

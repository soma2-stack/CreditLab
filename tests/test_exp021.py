"""Checks for EXP-021. These do not train the replication."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp017 import make_latent, task_views  # noqa: E402
from run_exp020 import hold_match, inspect_control, with_control  # noqa: E402
from run_exp021 import BASE_SEED, JOBS, SEEDS, load_config, query_does_not_reset  # noqa: E402


def test_new_cohort_is_locked():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["data"]["base_seed"] == 8000
    assert BASE_SEED == 8000
    assert len(JOBS) == 40
    assert cfg["replicates"] == "EXP-020"
    query_does_not_reset()


def test_new_base_seed_keeps_the_clean_control_and_hold_match():
    old = make_latent(64, 601, 0, 7000)
    new = make_latent(64, 601, 0, 8000)
    assert not torch.equal(old["A"], new["A"])
    hold_x, hold_y, selective_x, selective_y = task_views(new)
    selective = with_control(selective_x, new["timing"], new["F"])
    inspect_control(selective, new["timing"], new["F"], selective_y, new["A"], new["B"])
    assert torch.equal(selective[:, :, :3], selective_x)
    report = hold_match(601, 8000)
    assert report["matched"] is True
    assert report["state_gap"] <= 1e-5 and report["gradient_gap"] <= 1e-5
    assert torch.equal(with_control(hold_x, new["timing"], torch.zeros(64))[:, 129, 3], torch.zeros(64))

"""Checks for EXP-020. These do not train the clean-control comparison."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp017 import make_latent, task_views  # noqa: E402
from run_exp020 import (  # noqa: E402
    JOBS,
    hold_match,
    inspect_control,
    load_config,
    local_rules,
    with_control,
)


def test_contract_and_hold_match():
    cfg = load_config()
    assert cfg["models"]["additive_parameter_count"] == 1249
    assert cfg["models"]["exp017_parameter_count"] == 1217
    assert cfg["preflight"]["hold_match_tolerance"] == 1e-5
    assert len(JOBS) == 40
    local_rules()
    report = hold_match(503)
    assert report["matched"] is True
    assert report["parameter_count"] == 1249
    latent = make_latent(64, 503, 2)
    hold_x, _y, selective_x, _sy = task_views(latent)
    selective = with_control(selective_x, latent["timing"], latent["F"])
    assert torch.equal(selective[:, :, :3], selective_x)
    inspect_control(selective, latent["timing"], latent["F"], _sy, latent["A"], latent["B"])
    assert torch.equal(with_control(hold_x, latent["timing"], torch.zeros(64))[:, :, :3], hold_x)

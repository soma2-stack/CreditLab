"""Checks for EXP-017. These do not train the overwrite comparison."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp016 import verify_equations  # noqa: E402
from run_exp017 import (  # noqa: E402
    AUDITS,
    JOBS,
    SEEDS,
    differs_only_at,
    load_config,
    make_audit_pair,
    make_latent,
    subgroups,
    task_views,
)


def test_design_is_a_new_task_with_forty_jobs():
    cfg = load_config()
    assert cfg["not_a_hard_v2_replication"] is True
    assert cfg["seeds"] == SEEDS
    assert cfg["data"]["base_seed"] == 7000
    assert len(JOBS) == 40
    assert JOBS[0] == {"seed": 503, "model_type": "vanilla_tanh_rnn", "task": "hold", "attempt": 1}
    assert cfg["success"]["selective"].startswith("overall accuracy >= 0.95")
    assert cfg["budget"]["max_cpu_seconds_total"] == 900
    verify_equations()


def test_balance_pairing_noise_and_markers():
    latent = make_latent(512, 503, 0)
    hold_x, hold_y, selective_x, selective_y = task_views(latent)
    assert hold_x.shape == (512, 130, 3)
    assert torch.equal(hold_y, latent["A"])
    assert torch.equal(selective_y, torch.where(latent["F"] == 1, latent["B"], latent["A"]))
    rows = torch.arange(512)
    assert torch.equal(hold_x[rows, latent["timing"], 0], latent["B"])
    assert torch.equal(hold_x[rows, latent["timing"], 1], torch.zeros(512))
    assert torch.equal(selective_x[rows, latent["timing"], 1], latent["F"])
    assert torch.equal(hold_x[:, 0, 1], torch.ones(512))
    assert torch.equal(hold_x[:, 129, 2], torch.ones(512))
    assert int(latent["timing"].min()) >= 32 and int(latent["timing"].max()) <= 96
    groups = subgroups(latent["A"], latent["B"], latent["F"])
    assert all(int(mask.sum()) == 128 for mask in groups.values())
    covered = torch.zeros(512, dtype=torch.bool)
    for mask in groups.values():
        assert not bool((covered & mask).any())
        covered |= mask
    assert bool(covered.all())


def test_audits_change_only_the_intended_bit():
    latent = make_latent(256, 503, 3)
    rows = torch.arange(256)
    expectations = {
        "retained_initial": (torch.zeros(256, dtype=torch.long), 0),
        "replacement_bit": (latent["timing"], 0),
        "ignore_unmarked_later": (latent["timing"], 0),
        "ignore_obsolete_initial": (torch.zeros(256, dtype=torch.long), 0),
    }
    assert set(AUDITS) == set(expectations)
    for name, (positions, channel) in expectations.items():
        pair = make_audit_pair(latent, name)
        assert differs_only_at(pair["positive"], pair["negative"], rows, positions, channel)
    replacement = make_audit_pair(latent, "replacement_bit")
    assert torch.equal(replacement["selective_targets"][0], torch.ones(256))
    assert torch.equal(replacement["hold_targets"][0], latent["A"])
    ignore = make_audit_pair(latent, "ignore_unmarked_later")
    assert torch.equal(ignore["selective_targets"][0], ignore["selective_targets"][1])
    obsolete = make_audit_pair(latent, "ignore_obsolete_initial")
    assert torch.equal(obsolete["selective_targets"][0], latent["B"])
    assert torch.equal(obsolete["hold_targets"][0], torch.ones(256))

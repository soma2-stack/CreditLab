"""Correctness checks for EXP-008. These do not choose a parameter group."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp008 import (  # noqa: E402
    apply_freeze,
    build_four,
    group_fingerprint,
    load_config,
    load_hard_v2,
    measured_noise_std,
)


def test_config_locks_four_regimes_and_gate():
    cfg = load_config()
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["data"]["base_seed"] == 2000
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_total"] == 900
    assert cfg["task"]["distractor_noise_std"] == 0.3
    assert set(cfg["frozen_by_regime"]["input_plus_readout"]) == {
        "recurrent.weight_hh_l0", "recurrent.bias_hh_l0",
    }
    assert set(cfg["frozen_by_regime"]["recurrent_plus_readout"]) == {
        "recurrent.weight_ih_l0", "recurrent.bias_ih_l0",
    }


def test_regimes_share_initialization_and_freeze_the_named_groups():
    cfg = load_config()
    models = build_four(cfg, 101)
    reference = {name: value.detach().clone() for name, value in models["full"].named_parameters()}
    for regime, model in models.items():
        for name, value in model.named_parameters():
            assert torch.equal(value, reference[name])
        frozen = set(cfg["frozen_by_regime"][regime])
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable.isdisjoint(frozen)
        assert frozen <= {name for name, _ in model.named_parameters()}
    inputs = torch.randn(4, 6, 3)
    targets = torch.ones(4)
    partial = models["input_plus_readout"]
    before = group_fingerprint(partial, cfg["frozen_by_regime"]["input_plus_readout"])
    optimizer = torch.optim.Adam((p for p in partial.parameters() if p.requires_grad), lr=0.003)
    logits, _ = partial(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], targets)
    loss.backward()
    trained = [name for name, parameter in partial.named_parameters() if parameter.grad is not None]
    assert "recurrent.weight_ih_l0" in trained
    assert "recurrent.weight_hh_l0" not in trained
    optimizer.step()
    assert group_fingerprint(partial, cfg["frozen_by_regime"]["input_plus_readout"]) == before


def test_readout_only_states_stay_fixed():
    cfg = load_config()
    model = build_four(cfg, 113)["readout_only"]
    inputs = torch.randn(3, 5, 3)
    with torch.no_grad():
        before = model.recurrent_states(inputs).clone()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.003)
    logits, _ = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], torch.ones(3))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = model.recurrent_states(inputs)
    assert torch.equal(before, after)


def test_noise_and_bit_pairs():
    inputs, targets, meta = load_hard_v2(
        1200, 8, 101, 0, noise_std=0.3, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=2000, return_metadata=True,
    )
    assert 0.25 < measured_noise_std(inputs) < 0.36
    assert torch.equal(inputs[:, 0, 0], targets)
    values = meta["competitor_bits"][meta["competitor_mask"] & (targets[:, None] < 0)]
    assert 0.4 < float((values > 0).float().mean()) < 0.6
    positive, negative = make_counterfactual_pairs(inputs[:5])
    assert pairs_differ_only_on_original_bit(positive, negative)

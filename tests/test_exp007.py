"""Correctness checks for EXP-007. These do not tune the readout-only regime."""

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp007 import (  # noqa: E402
    build_matched_models,
    forward_fingerprint,
    freeze_recurrent,
    hidden_fingerprint,
    load_config,
    load_hard_v2,
    measured_noise_std,
    recurrent_fingerprint,
    trainable_names,
)


def test_config_locks_regimes_gate_and_budget():
    cfg = load_config()
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["bound"]["B"] == 4.0
    assert cfg["data"]["base_seed"] == 2000
    assert cfg["regimes"] == ["full", "readout_only"]
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_total"] == 1200
    assert cfg["frozen_parameter_names"] == [
        "recurrent.weight_ih_l0", "recurrent.weight_hh_l0",
        "recurrent.bias_ih_l0", "recurrent.bias_hh_l0",
    ]
    assert cfg["trainable_readout_parameter_names"] == ["readout.weight", "readout.bias"]


def test_paired_regimes_match_before_training_and_freeze_is_correct():
    cfg = load_config()
    full = build_matched_models(cfg, 101)["additive_tanh_rnn"]
    readout = copy.deepcopy(full)
    frozen = freeze_recurrent(readout)
    assert frozen == cfg["frozen_parameter_names"]
    assert trainable_names(readout) == cfg["trainable_readout_parameter_names"]
    inputs = torch.randn(4, 7, 3)
    assert forward_fingerprint(full, inputs) == forward_fingerprint(readout, inputs)
    assert hidden_fingerprint(full, inputs) == hidden_fingerprint(readout, inputs)
    before = recurrent_fingerprint(readout)
    optimizer = torch.optim.Adam((p for p in readout.parameters() if p.requires_grad), lr=0.003)
    logits, _ = readout(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], torch.ones(4))
    loss.backward()
    assert all(parameter.grad is not None for parameter in readout.readout.parameters())
    optimizer.step()
    assert recurrent_fingerprint(readout) == before
    assert hidden_fingerprint(readout, inputs) == hidden_fingerprint(full, inputs)
    assert not torch.equal(readout.readout.weight, full.readout.weight)


def test_noise_and_pairs():
    inputs, targets, meta = load_hard_v2(
        1500, 8, 101, 0, noise_std=0.3, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=2000, return_metadata=True,
    )
    assert 0.25 < measured_noise_std(inputs) < 0.36
    assert torch.equal(inputs[:, 0, 0], targets)
    values = meta["competitor_bits"][meta["competitor_mask"] & (targets[:, None] > 0)]
    assert 0.4 < float((values > 0).float().mean()) < 0.6
    positive, negative = make_counterfactual_pairs(inputs[:4])
    assert pairs_differ_only_on_original_bit(positive, negative)
    positive[:, 0, 0] = 0
    negative[:, 0, 0] = 0
    assert torch.equal(positive, negative)

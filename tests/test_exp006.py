"""Correctness checks for EXP-006. These do not select seeds or the bound."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import BoundedAdditiveTanhRNN, ResidualTanhRNN, TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp006 import (  # noqa: E402
    build_matched_models,
    clopper_pearson,
    frozen_manifest,
    load_config,
    load_hard_v2,
    measured_noise_std,
)


def test_config_locks_seeds_bound_and_gate():
    cfg = load_config()
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["bound"]["B"] == 4.0
    assert cfg["data"]["base_seed"] == 2000
    assert cfg["data"]["audit_split_index"] == 3
    assert cfg["task"]["distractor_noise_std"] == 0.3
    assert cfg["task"]["competitor_rate"] == 0.4
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_per_run"] == 300
    assert cfg["budget"]["max_cpu_seconds_total"] == 900
    assert cfg["environment_requirement"]["python"] == "3.11.9"


def test_equations_and_matched_initialization():
    seed_everything(101)
    vanilla = TinySequenceRNN(3, 8, 1)
    seed_everything(101)
    additive = ResidualTanhRNN(3, 8, 1, residual_scale=1.0)
    seed_everything(101)
    bounded = BoundedAdditiveTanhRNN(3, 8, 1, bound=4.0)
    assert all(torch.equal(a, b) for a, b in zip(vanilla.parameters(), additive.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(vanilla.parameters(), bounded.parameters(), strict=True))
    inputs = torch.randn(2, 6, 3)
    h = torch.zeros(2, 8)
    cell = additive.recurrent
    pre = inputs[:, 0] @ cell.weight_ih_l0.T + cell.bias_ih_l0 + h @ cell.weight_hh_l0.T + cell.bias_hh_l0
    candidate = torch.tanh(pre)
    state, returned, proposed = bounded.step_parts(inputs[:, 0], h)
    assert torch.allclose(returned, candidate, atol=1e-6)
    assert torch.allclose(proposed, h + candidate, atol=1e-6)
    assert torch.allclose(state, proposed.clamp(-4, 4), atol=1e-6)
    models = build_matched_models(load_config(), 101)
    counts = {sum(p.numel() for p in model.parameters()) for model in models.values()}
    assert counts == {1217}


def test_new_base_seed_uses_noise_0_3_and_independent_competitors():
    inputs, targets, meta = load_hard_v2(
        2000, 8, 101, 0, noise_std=0.3, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=2000, return_metadata=True,
    )
    assert 0.25 < measured_noise_std(inputs) < 0.36
    assert torch.equal(inputs[:, 0, 0], targets)
    for sign in (-1.0, 1.0):
        values = meta["competitor_bits"][meta["competitor_mask"] & (targets[:, None] == sign)]
        assert values.numel() > 100
        assert 0.4 < float((values > 0).float().mean()) < 0.6


def test_pairs_and_negative_control():
    base = torch.randn(4, 5, 3)
    positive, negative = make_counterfactual_pairs(base)
    assert pairs_differ_only_on_original_bit(positive, negative)
    positive[:, 0, 0] = 0
    negative[:, 0, 0] = 0
    assert torch.equal(positive, negative)


def test_checkpoint_roundtrip(tmp_path):
    from run_exp006 import save_checkpoint

    cfg = load_config()
    model = build_matched_models(cfg, 113)["bounded_additive_tanh_rnn"]
    record = {
        "model_type": "bounded_additive_tanh_rnn",
        "delay": 64,
        "seed": 113,
        "bound": 4.0,
        "status": "ok",
        "code_revision": "test",
    }
    # save_checkpoint writes under the given directory's checkpoints folder.
    original_root_behavior = save_checkpoint(tmp_path, model, record)
    assert "bounded_additive_tanh_rnn_delay64_seed113.pt" in original_root_behavior
    payload = torch.load(tmp_path / "checkpoints" / "bounded_additive_tanh_rnn_delay64_seed113.pt", weights_only=False)
    reloaded = BoundedAdditiveTanhRNN(3, 32, 1, bound=4.0)
    reloaded.load_state_dict(payload["state_dict"])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), reloaded.parameters(), strict=True))


def test_clopper_pearson_edges():
    empty = clopper_pearson(0, 10)
    full = clopper_pearson(10, 10)
    assert empty["lower"] == 0.0
    assert 0.25 < empty["upper"] < 0.35
    assert full["upper"] == 1.0
    assert 0.65 < full["lower"] < 0.75


def test_frozen_manifest_is_stable():
    assert frozen_manifest() == frozen_manifest()
    saved = ROOT / "results" / "EXP-006" / "frozen_hashes_before.json"
    if saved.exists():
        import json
        assert json.loads(saved.read_text(encoding="utf-8")) == frozen_manifest()

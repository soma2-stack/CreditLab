"""Pre-training checks for EXP-004. These do not select hyperparameters."""

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import BoundedAdditiveTanhRNN, ResidualTanhRNN  # noqa: E402
from run_exp004 import (  # noqa: E402
    MODEL_TYPES,
    assert_outputs_available,
    build_matched_models,
    frozen_manifest,
    historical_hard_v2,
    load_config,
    load_hard_v2,
    measured_noise_std,
    minibatch_indices,
    parameter_fingerprint,
    train_one,
)

CONFIG_PATH = ROOT / "configs" / "exp004_bounded_additive.yaml"


def _tiny_cfg():
    cfg = load_config()
    cfg["training"]["updates_per_run"] = 2
    cfg["data"]["train_samples"] = 32
    cfg["data"]["val_samples"] = 16
    cfg["data"]["test_samples"] = 16
    cfg["budget"]["max_cpu_seconds_per_run"] = 60
    return cfg


def test_config_locks_bound_noise_and_rules():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["bound"]["B"] == 4.0
    assert cfg["bound"]["learned"] is False
    assert cfg["bound"]["chosen_before_test_results"] is True
    assert cfg["task"]["distractor_noise_std"] == 0.3
    assert cfg["task"]["competitor_rate"] == 0.4
    assert cfg["task"]["run_mode"] == "HARD_V2"
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["interpretation"]["active_clipping_fraction_at_least"] == 0.01
    assert cfg["diagnostics"]["test_accuracy_during_training"] is False
    assert cfg["training"]["diagnostic_updates"] == [0, 50, 100, 200, 400]
    assert cfg["training"]["updates_per_run"] == 400
    assert cfg["model"]["hidden_size"] == 32
    assert cfg["seeds"] == [17, 29, 43]
    assert cfg["budget"]["max_cpu_seconds_per_run"] == 300
    assert cfg["outputs"]["refuse_overwrite"] is True


def test_three_models_share_initialization_and_parameter_count():
    models = build_matched_models(load_config(), 17)
    assert set(models) == set(MODEL_TYPES)
    fingerprints = {name: parameter_fingerprint(model) for name, model in models.items()}
    assert len(set(fingerprints.values())) == 1
    counts = {sum(p.numel() for p in model.parameters()) for model in models.values()}
    assert counts == {1217}


def test_explicit_hard_v2_matches_historical_generator_and_uses_noise_0_3():
    explicit = load_hard_v2(
        256, 16, 17, 2, noise_std=0.3, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=1000, return_metadata=True,
    )
    historical = historical_hard_v2(256, 16, 17, 2, return_metadata=True)
    assert torch.equal(explicit[0], historical[0])
    assert torch.equal(explicit[1], historical[1])
    assert torch.equal(explicit[2]["competitor_bits"], historical[2]["competitor_bits"])
    std = measured_noise_std(explicit[0])
    assert 0.25 < std < 0.36
    louder = load_hard_v2(
        256, 16, 17, 2, noise_std=1.0, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=1000,
    )
    assert not torch.equal(explicit[0], louder[0])


def test_labels_and_independent_competitor_bits():
    inputs, targets, meta = load_hard_v2(
        3000, 8, 29, 0, noise_std=0.0, competitor_rate=0.4,
        competitor_flip_prob=0.5, base_seed=1000, return_metadata=True,
    )
    assert torch.equal(inputs[:, 0, 0], targets)
    mask = meta["competitor_mask"]
    bits = meta["competitor_bits"]
    for sign in (-1.0, 1.0):
        selected = mask & (targets[:, None] == sign)
        values = bits[selected]
        assert values.numel() > 200
        positive = float((values > 0).float().mean())
        agreement = float((values == sign).float().mean())
        assert 0.42 < positive < 0.58
        assert 0.42 < agreement < 0.58


def test_minibatch_indices_repeat():
    first = minibatch_indices(17, 64, 100, 8, 5)
    second = minibatch_indices(17, 64, 100, 8, 5)
    assert first == second
    assert minibatch_indices(29, 64, 100, 8, 5) != first


def test_inactive_clamp_matches_additive_and_active_clamp_stays_inside_bounds():
    torch.manual_seed(3)
    additive = ResidualTanhRNN(3, 8, 1, residual_scale=1.0)
    bounded = BoundedAdditiveTanhRNN(3, 8, 1, bound=4.0)
    bounded.load_state_dict(additive.state_dict())
    bounded.bound = 1_000_000.0
    inputs = torch.randn(4, 12, 3)
    assert torch.allclose(
        additive.recurrent_states(inputs), bounded.recurrent_states(inputs), atol=1e-5,
    )
    bounded.bound = 4.0
    saturated = torch.ones(2, 30, 3)
    states = bounded.recurrent_states(saturated)
    assert float(states.detach().abs().max()) <= 4.0 + 1e-5


def test_clamp_gradient_matches_autograd_away_from_and_on_the_boundary():
    proposed = torch.tensor([0.25, 5.5], requires_grad=True)
    proposed.clamp(-4.0, 4.0).sum().backward()
    assert torch.equal(proposed.grad, torch.tensor([1.0, 0.0]))

    additive = ResidualTanhRNN(3, 4, 1)
    bounded = BoundedAdditiveTanhRNN(3, 4, 1, bound=4.0)
    with torch.no_grad():
        for param in additive.parameters():
            param.mul_(0.05)
    bounded.load_state_dict(additive.state_dict())
    inputs = torch.randn(2, 3, 3) * 0.05
    additive_states = additive.recurrent_states(inputs)
    bounded_states = bounded.recurrent_states(inputs)
    assert float(bounded_states.detach().abs().max()) < 4.0
    assert torch.allclose(additive_states, bounded_states, atol=1e-5)
    names = ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")
    additive_grads = torch.autograd.grad(
        additive_states.sum(), [getattr(additive.recurrent, name) for name in names],
    )
    bounded_grads = torch.autograd.grad(
        bounded_states.sum(), [getattr(bounded.recurrent, name) for name in names],
    )
    assert all(torch.allclose(left, right, atol=1e-5) for left, right in zip(additive_grads, bounded_grads, strict=True))


def test_short_runs_stay_finite_for_all_three_models():
    cfg = _tiny_cfg()
    from run_exp004 import build_matched_models, dataset_bundle, dataset_fingerprints, index_fingerprint

    bundle = dataset_bundle(cfg, 4, 17)
    fingerprints = dataset_fingerprints(bundle)
    indices = minibatch_indices(17, 4, 32, cfg["training"]["batch_size"], 2)
    models = build_matched_models(cfg, 17)
    init_hash = parameter_fingerprint(models["vanilla_tanh_rnn"])
    index_hash = index_fingerprint(indices)
    for model_type, model in models.items():
        record, curve, checks, _clip = train_one(
            cfg, model, model_type, bundle, indices, 4, 17, "primary",
            init_hash, fingerprints, index_hash,
        )
        assert record["status"] == "ok", record.get("stop_reason")
        assert record["all_finite"] is True
        assert len(curve) == 2
        assert checks[0]["update"] == 0
        assert math_isfinite(record["test_loss"])
        if model_type == "bounded_additive_tanh_rnn":
            assert record["final_test_clipping"]["actual_max_abs"] <= 4.0 + 1e-5


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def test_output_writer_refuses_existing_raw_metrics(tmp_path):
    raw = tmp_path / "raw_metrics.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    try:
        assert_outputs_available(tmp_path, "raw_metrics.jsonl")
    except FileExistsError:
        return
    raise AssertionError("existing raw metrics were not refused")


def test_frozen_manifest_is_stable():
    assert frozen_manifest() == frozen_manifest()
    saved = ROOT / "results" / "EXP-004" / "frozen_hashes_before.json"
    if saved.exists():
        assert json.loads(saved.read_text(encoding="utf-8")) == frozen_manifest()

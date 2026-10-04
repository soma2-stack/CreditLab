"""Correctness checks for EXP-011. These do not train the cohort."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp010 import forward_regime, rollout  # noqa: E402
from run_exp011 import (  # noqa: E402
    apply_clipping,
    build_four,
    float64_grad_norm,
    load_config,
    save_checkpoint,
)


def _model():
    seed_everything(5)
    return ResidualTanhRNN(3, 32, 1, residual_scale=1.0)


def test_config_locks_four_conditions_and_k():
    cfg = load_config()
    assert [item["name"] for item in cfg["conditions"]] == [
        "full_history_clip5", "final_16_clip5", "full_history_noclip", "final_16_noclip",
    ]
    assert cfg["gradient_horizon"]["k"] == 16
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["budget"]["max_cpu_seconds_total"] == 1200
    assert cfg["budget"]["no_automatic_resume"] is True
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95


def test_four_models_share_initialization_and_forward_values():
    models = build_four(load_config(), 101)
    reference = {name: value.detach().clone() for name, value in next(iter(models.values())).named_parameters()}
    inputs = torch.randn(2, 20, 3)
    with torch.no_grad():
        reference_logits, _ = next(iter(models.values()))(inputs)
    for model in models.values():
        for name, value in model.named_parameters():
            assert torch.equal(value, reference[name])
            assert value.requires_grad
        with torch.no_grad():
            logits, _ = model(inputs)
        assert torch.equal(logits, reference_logits)
        detached, _ = rollout(model, inputs, 4)
        assert torch.equal(detached, reference_logits)


def test_final_16_blocks_early_gradients_and_keeps_parameters_trainable():
    model = _model()
    inputs = torch.randn(2, 20, 3, requires_grad=True)
    logits, _states, cut = forward_regime(model, inputs, "final_16")
    assert cut == 4
    logits[:, -1, 0].sum().backward()
    assert torch.count_nonzero(inputs.grad[:, :cut]) == 0
    assert torch.count_nonzero(inputs.grad[:, cut:]) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_clipping_on_matches_norm_5_and_clipping_off_is_unchanged():
    model = _model()
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    before = [parameter.grad.detach().clone() for parameter in model.parameters()]
    recorded = float64_grad_norm(list(model.parameters()))
    assert torch.equal(model.parameters().__iter__().__next__().grad, before[0])
    off = apply_clipping(list(model.parameters()), enabled=False)
    for parameter, saved in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter.grad, saved)
    assert off["clipping_applied"] is False
    assert off["applied_clip_scale"] is None
    assert off["raw_grad_norm_float64"] == recorded
    copy_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]
    on = apply_clipping(list(model.parameters()), enabled=True)
    reference = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    for parameter, saved in zip(reference.parameters(), copy_grads, strict=True):
        parameter.grad = saved.clone()
    torch.nn.utils.clip_grad_norm_(list(reference.parameters()), 5.0)
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)
    if on["implementation_grad_norm"] > 5.0:
        assert on["clipping_applied"] is True
        assert on["applied_clip_scale"] < 1.0


def test_checkpoint_reload_and_overwrite_protection(tmp_path):
    model = _model()
    record = {
        "condition": "full_history_clip5", "horizon": "full_history", "clip": True,
        "delay": 64, "seed": 101, "code_revision": "test",
    }
    save_checkpoint(tmp_path, model, record)
    try:
        save_checkpoint(tmp_path, model, record)
        raise AssertionError("overwrite was allowed")
    except FileExistsError:
        pass
    payload = torch.load(tmp_path / "checkpoints" / "additive_full_history_clip5_delay64_seed101.pt", weights_only=False)
    reloaded = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    reloaded.load_state_dict(payload["state_dict"])
    inputs = torch.randn(2, 5, 3)
    with torch.no_grad():
        assert torch.equal(model(inputs)[0], reloaded(inputs)[0])

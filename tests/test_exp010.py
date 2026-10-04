"""Correctness checks for EXP-010. These do not train the cohort."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.delayed_task import sequence_length  # noqa: E402
from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp010 import (  # noqa: E402
    K,
    build_pair,
    cut_index,
    forward_regime,
    load_config,
    reference_final_logit,
    rollout,
)


def _model():
    seed_everything(5)
    return ResidualTanhRNN(3, 32, 1, residual_scale=1.0)


def test_config_locks_k_and_the_gate():
    cfg = load_config()
    assert cfg["gradient_horizon"]["k"] == 16
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["data"]["base_seed"] == 2000
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_total"] == 600
    assert cfg["budget"]["no_automatic_resume"] is True
    assert cfg["forward_match"] == "direct_equality"
    assert cfg["regimes"] == ["full_history", "final_16"]
    assert K == 16


def test_delay_cuts_cover_exactly_sixteen_steps_and_exclude_the_event():
    assert sequence_length(64) == 66
    assert sequence_length(128) == 130
    assert cut_index(66) == 50
    assert cut_index(130) == 114
    assert 66 - 50 == 16
    assert 130 - 114 == 16
    assert cut_index(10) == 0


def test_detach_matches_the_full_forward_and_does_not_change_the_boundary():
    model = _model()
    inputs = torch.randn(3, 20, 3)
    with torch.no_grad():
        reference_logits, _ = model(inputs)
        reference_states = model.recurrent_states(inputs)
    full_logits, full_states = rollout(model, inputs, 0)
    cut_logits, cut_states = rollout(model, inputs, 4)
    assert torch.equal(reference_logits, full_logits)
    assert torch.equal(reference_logits, cut_logits)
    assert torch.equal(reference_states, full_states)
    assert torch.equal(reference_states, cut_states)
    hidden = inputs.new_zeros(3, 32)
    for time_index in range(4):
        hidden, _candidate = model.candidate_step(inputs[:, time_index, :], hidden)
    assert torch.equal(hidden, hidden.detach())


def test_final_16_blocks_early_input_gradients_and_keeps_late_ones():
    model = _model()
    inputs = torch.randn(2, 20, 3, requires_grad=True)
    logits, _states, cut = forward_regime(model, inputs, "final_16")
    assert cut == 4
    logits[:, -1, 0].sum().backward()
    assert torch.count_nonzero(inputs.grad[:, :cut, :]) == 0
    assert torch.count_nonzero(inputs.grad[:, cut:, :]) > 0
    short = torch.randn(2, 10, 3, requires_grad=True)
    short_logits, _states, short_cut = forward_regime(model, short, "final_16")
    assert short_cut == 0
    full_logits, _ = model(short)
    assert torch.equal(short_logits, full_logits)


def test_reference_gradients_match_and_every_parameter_is_trainable():
    model = _model()
    sequence = torch.randn(2, 6, 3)
    sequence.requires_grad_(True)
    parameters = [
        model.recurrent.weight_ih_l0,
        model.recurrent.weight_hh_l0,
        model.recurrent.bias_ih_l0,
        model.recurrent.bias_hh_l0,
        model.readout.weight,
        model.readout.bias,
    ]
    logits, _ = model(sequence)
    reference = reference_final_logit(sequence, *parameters)
    assert torch.equal(logits[:, -1, :], reference)
    model_grads = torch.autograd.grad(logits[:, -1, 0].sum(), parameters + [sequence])
    reference_grads = torch.autograd.grad(reference[:, 0].sum(), parameters + [sequence])
    for left, right in zip(model_grads, reference_grads, strict=True):
        assert torch.equal(left, right)
    truncated = torch.randn(2, 20, 3)
    truncated_logits, _states, _cut = forward_regime(model, truncated, "final_16")
    truncated_logits[:, -1, 0].sum().backward()
    assert all(parameter.requires_grad and parameter.grad is not None for parameter in model.parameters())


def test_regimes_share_initialization_and_take_one_terminal_update():
    cfg = load_config()
    models = build_pair(cfg, 101)
    reference = {name: value.detach().clone() for name, value in models["full_history"].named_parameters()}
    for model in models.values():
        for name, value in model.named_parameters():
            assert torch.equal(value, reference[name])
            assert value.requires_grad
    model = models["final_16"]
    inputs = torch.randn(4, 12, 3)
    targets = torch.ones(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    calls = {"steps": 0}
    original = optimizer.step

    def counted_step(*args, **kwargs):
        calls["steps"] += 1
        return original(*args, **kwargs)

    optimizer.step = counted_step
    logits, _states, _cut = forward_regime(model, inputs, "final_16")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], targets)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], targets)
    assert torch.equal(loss, manual)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert calls["steps"] == 1


def test_bit_pairs_change_only_the_original_bit():
    base = torch.randn(4, 8, 3)
    positive, negative = make_counterfactual_pairs(base)
    assert pairs_differ_only_on_original_bit(positive, negative)

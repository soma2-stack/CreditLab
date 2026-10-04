"""Correctness checks for EXP-009. These do not choose a training result."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp009 import (  # noqa: E402
    apply_freeze,
    build_four,
    group_fingerprint,
    load_config,
    load_hard_v2,
    measured_noise_std,
    measurement_blockers,
    save_checkpoint,
    verify_regime_lists,
)


def test_config_locks_bias_choice_counts_and_gate():
    cfg = load_config()
    assert cfg["seeds"] == [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
    assert cfg["data"]["base_seed"] == 2000
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_total"] == 900
    assert cfg["budget"]["no_automatic_resume"] is True
    assert cfg["budget"]["no_timestamp_runtime_inference"] is True
    assert cfg["task"]["distractor_noise_std"] == 0.3
    assert cfg["single_trainable_preactivation_bias"] == "recurrent.bias_ih_l0"
    assert cfg["expected_trainable_counts"] == {
        "full": 1217,
        "readout_only": 33,
        "bias_plus_readout": 65,
        "matrices_plus_readout": 1153,
    }
    assert cfg["trainable_by_regime"]["bias_plus_readout"] == [
        "recurrent.bias_ih_l0", "readout.weight", "readout.bias",
    ]
    assert set(cfg["frozen_by_regime"]["bias_plus_readout"]) == {
        "recurrent.weight_ih_l0", "recurrent.weight_hh_l0", "recurrent.bias_hh_l0",
    }
    assert set(cfg["frozen_by_regime"]["matrices_plus_readout"]) == {
        "recurrent.bias_ih_l0", "recurrent.bias_hh_l0",
    }


def test_additive_equation_and_separate_biases():
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    names = {name for name, _ in model.named_parameters()}
    assert "recurrent.bias_ih_l0" in names and "recurrent.bias_hh_l0" in names
    assert sum(parameter.numel() for parameter in model.parameters()) == 1217
    x = torch.randn(2, 3)
    hidden = torch.randn(2, 32)
    cell = model.recurrent
    pre = (
        x.matmul(cell.weight_ih_l0.T)
        + cell.bias_ih_l0
        + hidden.matmul(cell.weight_hh_l0.T)
        + cell.bias_hh_l0
    )
    expected = hidden + torch.tanh(pre)
    got, candidate = model.candidate_step(x, hidden)
    assert torch.equal(got, expected)
    assert torch.equal(candidate, torch.tanh(pre))


def test_regimes_share_initialization_and_freeze_the_named_groups():
    cfg = load_config()
    models = build_four(cfg, 101)
    verify_regime_lists(cfg, models)
    reference = {name: value.detach().clone() for name, value in models["full"].named_parameters()}
    probe = torch.randn(4, 6, 3)
    with torch.no_grad():
        reference_logits, _ = models["full"](probe)
    for regime, model in models.items():
        for name, value in model.named_parameters():
            assert torch.equal(value, reference[name])
        with torch.no_grad():
            logits, _ = model(probe)
        assert float((logits - reference_logits).abs().max()) <= 1e-5
        frozen = set(cfg["frozen_by_regime"][regime])
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable.isdisjoint(frozen)


def test_bias_regime_updates_only_bx_and_readout():
    cfg = load_config()
    model = build_four(cfg, 127)["bias_plus_readout"]
    frozen_names = cfg["frozen_by_regime"]["bias_plus_readout"]
    before = group_fingerprint(model, frozen_names)
    inputs = torch.randn(4, 6, 3)
    with torch.no_grad():
        hidden_before = model.recurrent_states(inputs).clone()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.003)
    logits, _ = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], torch.ones(4))
    loss.backward()
    graded = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert graded == {"recurrent.bias_ih_l0", "readout.weight", "readout.bias"}
    optimizer.step()
    assert group_fingerprint(model, frozen_names) == before
    with torch.no_grad():
        hidden_after = model.recurrent_states(inputs)
    assert hidden_before.shape == hidden_after.shape


def test_matrix_regime_keeps_both_biases_fixed():
    cfg = load_config()
    model = build_four(cfg, 139)["matrices_plus_readout"]
    frozen_names = cfg["frozen_by_regime"]["matrices_plus_readout"]
    before = group_fingerprint(model, frozen_names)
    inputs = torch.randn(4, 6, 3)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.003)
    logits, _ = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], torch.ones(4))
    loss.backward()
    graded = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert "recurrent.weight_ih_l0" in graded and "recurrent.weight_hh_l0" in graded
    assert "recurrent.bias_ih_l0" not in graded and "recurrent.bias_hh_l0" not in graded
    optimizer.step()
    assert group_fingerprint(model, frozen_names) == before


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


def test_checkpoint_reload_and_overwrite_protection(tmp_path):
    cfg = load_config()
    model = build_four(cfg, 151)["full"]
    record = {
        "regime": "full", "delay": 64, "seed": 151,
        "code_revision": "test", "init_fingerprint": "test",
        "trainable_parameter_names": cfg["trainable_by_regime"]["full"],
        "frozen_parameter_names": [],
    }
    save_checkpoint(tmp_path, model, record)
    try:
        save_checkpoint(tmp_path, model, record)
        raise AssertionError("overwrite was allowed")
    except FileExistsError:
        pass
    payload = torch.load(tmp_path / "checkpoints" / "additive_full_delay64_seed151.pt", weights_only=False)
    reloaded = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    reloaded.load_state_dict(payload["state_dict"])
    inputs = torch.randn(2, 5, 3)
    with torch.no_grad():
        original, _ = model(inputs)
        again, _ = reloaded(inputs)
    assert torch.equal(original, again)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "raw_metrics.jsonl").write_text("", encoding="utf-8")
    assert measurement_blockers(blocked) == ["raw_metrics.jsonl"]


def test_noise_labels_and_bit_pairs():
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
    zero_a = positive.clone()
    zero_b = negative.clone()
    zero_a[:, 0, 0] = 0
    zero_b[:, 0, 0] = 0
    assert torch.equal(zero_a, zero_b)
    model = build_four(load_config(), 163)["full"]
    with torch.no_grad():
        logits_a, _ = model(zero_a)
        logits_b, _ = model(zero_b)
    assert torch.equal(logits_a, logits_b)

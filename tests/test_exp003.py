"""Checks for the EXP-003 bounded mixture before any test-set training run."""

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.diagnostics import step_jacobian_stats  # noqa: E402
from creditlab.models import BoundedMixtureTanhRNN, ResidualTanhRNN, TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp003_bounded_mixture.yaml"


def test_config_locks_equal_mixture_before_results():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["mixture"]["previous_weight"] == 0.5
    assert cfg["mixture"]["candidate_weight"] == 0.5
    assert cfg["mixture"]["weights_are_learned"] is False
    assert cfg["mixture"]["weights_chosen_before_test_results"] is True
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.75
    assert cfg["interpretation"]["controlled_max_abs_hidden_at_most"] == 1.01
    assert cfg["task"]["run_mode"] == "HARD_V2"
    assert cfg["task"]["primary_delay"] == 64
    assert cfg["task"]["confirmation_delay"] == 128
    assert cfg["seeds"] == [17, 29, 43]
    assert cfg["training"]["updates_per_run"] == 400
    assert cfg["model"]["hidden_size"] == 32
    assert cfg["outputs"]["directory"] == "results/EXP-003"


def test_mixture_matches_vanilla_initialization_and_stays_bounded():
    seed_everything(17)
    vanilla = TinySequenceRNN(3, 32, 1)
    seed_everything(17)
    bounded = BoundedMixtureTanhRNN(3, 32, 1)
    assert all(
        torch.equal(a, b)
        for a, b in zip(vanilla.parameters(), bounded.parameters(), strict=True)
    )
    sequence = torch.randn(8, 40, 3)
    states = bounded.recurrent_states(sequence)
    vanilla_states = vanilla.recurrent_states(sequence)
    assert torch.allclose(states[:, 0], 0.5 * vanilla_states[:, 0], atol=1e-5)
    assert float(states.detach().abs().max()) <= 1.0 + 1e-5
    logits, final_hidden = bounded(sequence)
    assert logits.shape == (8, 40, 1)
    assert final_hidden.shape == (1, 8, 32)


def test_additive_residual_formula_is_unchanged():
    seed_everything(5)
    residual = ResidualTanhRNN(3, 8, 1, residual_scale=1.0)
    sequence = torch.randn(2, 3, 3)
    h = torch.zeros(2, 8)
    expected = []
    for t in range(3):
        cell = residual.recurrent
        pre = (
            sequence[:, t].matmul(cell.weight_ih_l0.T)
            + cell.bias_ih_l0
            + h.matmul(cell.weight_hh_l0.T)
            + cell.bias_hh_l0
        )
        candidate = torch.tanh(pre)
        h = h + candidate
        expected.append(h)
    actual = residual.recurrent_states(sequence)
    assert torch.allclose(actual, torch.stack(expected, dim=1), atol=1e-6)


def test_zero_recurrent_weight_has_half_step_gain():
    model = BoundedMixtureTanhRNN(3, 6, 1)
    with torch.no_grad():
        model.recurrent.weight_hh_l0.zero_()
    stats = step_jacobian_stats(model, torch.randn(3, 5, 3), num_steps=4)
    assert abs(stats["spectral_norm"] - 0.5) < 1e-5
    assert abs(stats["min_singular"] - 0.5) < 1e-5


def test_bounded_training_step_stays_finite_and_repeats():
    import run_exp003

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["training"]["updates_per_run"] = 3
    cfg["data"]["train_samples"] = 32
    cfg["data"]["val_samples"] = 16
    cfg["data"]["test_samples"] = 16
    first = run_exp003.train_one_run(cfg, 4, "bounded_mixture_tanh_rnn", 17, "primary")
    second = run_exp003.train_one_run(cfg, 4, "bounded_mixture_tanh_rnn", 17, "primary")
    assert first["status"] == "ok" and second["status"] == "ok"
    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["previous_weight"] == 0.5
    assert first["candidate_weight"] == 0.5
    assert first["max_abs_hidden"] <= 1.01
    vanilla = run_exp003.train_one_run(cfg, 4, "vanilla_tanh_rnn", 17, "primary")
    assert vanilla["status"] == "ok"
    assert vanilla["parameter_count"] == first["parameter_count"]

"""Checks for the EXP-002 residual path before any test-set training run."""

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.delayed_task import make_split  # noqa: E402
from creditlab.diagnostics import hidden_states, step_jacobian_stats  # noqa: E402
from creditlab.models import ResidualTanhRNN, TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp002_residual_recurrent.yaml"


def test_config_locks_scale_and_cutoffs_before_results():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["residual"]["residual_scale"] == 1.0
    assert cfg["residual"]["scale_is_learned"] is False
    assert cfg["residual"]["scale_chosen_before_test_results"] is True
    assert cfg["interpretation"]["success_test_accuracy_at_least"] == 0.75
    assert cfg["interpretation"]["retention_clearly_readable_r2_at_least"] == 0.5
    assert cfg["task"]["run_mode"] == "HARD_V2"
    assert cfg["task"]["primary_delay"] == 64
    assert cfg["task"]["confirmation_delay"] == 128
    assert cfg["seeds"] == [17, 29, 43]
    assert cfg["training"]["updates_per_run"] == 400
    assert cfg["training"]["learning_rate"] == 0.003
    assert cfg["model"]["hidden_size"] == 32


def test_same_seed_gives_identical_parameters_and_count():
    seed_everything(17)
    vanilla = TinySequenceRNN(3, 32, 1)
    seed_everything(17)
    residual = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    left = list(vanilla.parameters())
    right = list(residual.parameters())
    assert len(left) == len(right)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    assert sum(p.numel() for p in left) == sum(p.numel() for p in right)
    assert not any(p.requires_grad and p.ndim == 0 for p in residual.parameters())


def test_residual_matches_vanilla_on_first_step_then_diverges():
    seed_everything(17)
    vanilla = TinySequenceRNN(3, 8, 1)
    seed_everything(17)
    residual = ResidualTanhRNN(3, 8, 1, residual_scale=1.0)
    sequence = torch.randn(4, 5, 3)
    vanilla_states = vanilla.recurrent_states(sequence)
    residual_states = residual.recurrent_states(sequence)
    assert torch.allclose(vanilla_states[:, 0], residual_states[:, 0], atol=1e-5)
    assert not torch.allclose(vanilla_states[:, 1], residual_states[:, 1], atol=1e-4)
    logits, final_hidden = residual(sequence)
    assert logits.shape == (4, 5, 1)
    assert final_hidden.shape == (1, 4, 8)


def test_zero_recurrent_weight_leaves_an_identity_step():
    model = ResidualTanhRNN(3, 6, 1, residual_scale=1.0)
    with torch.no_grad():
        model.recurrent.weight_hh_l0.zero_()
    inputs = torch.randn(3, 5, 3)
    stats = step_jacobian_stats(model, inputs, num_steps=4)
    assert abs(stats["spectral_norm"] - 1.0) < 1e-5
    assert abs(stats["min_singular"] - 1.0) < 1e-5


def test_vanilla_hidden_states_match_the_rnn_cell():
    seed_everything(3)
    model = TinySequenceRNN(3, 8, 1)
    inputs = torch.randn(4, 6, 3)
    expected, _ = model.recurrent(inputs)
    assert torch.equal(hidden_states(model, inputs), expected)


def test_hard_v2_split_keeps_the_exp001b_noise_path():
    """HARD-v2 in make_split uses easy noise. EXP-002 must not silently change it."""
    cfg = {
        "easy_noise_std": 0.3,
        "hard_noise_std": 1.0,
        "competitor_rate": 0.4,
        "competitor_flip_prob": 0.5,
    }
    hard_v2, _ = make_split(1024, 16, "HARD_V2", run_seed=17, split_index=2, config=cfg)
    hard, _ = make_split(1024, 16, "HARD", run_seed=17, split_index=2, config=cfg)
    v2_std = float(hard_v2[:, 1:-1, 2].std())
    hard_std = float(hard[:, 1:-1, 2].std())
    assert 0.2 < v2_std < 0.45
    assert hard_std > 0.7


def test_residual_training_step_stays_finite_and_repeats():
    import run_exp002

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["training"]["updates_per_run"] = 3
    cfg["data"]["train_samples"] = 32
    cfg["data"]["val_samples"] = 16
    cfg["data"]["test_samples"] = 16
    first = run_exp002.train_one_run(cfg, 4, "residual_tanh_rnn", 17, "primary")
    second = run_exp002.train_one_run(cfg, 4, "residual_tanh_rnn", 17, "primary")
    assert first["status"] == "ok" and second["status"] == "ok"
    assert first["all_finite"] is True
    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["retention_probe_r2"] == second["retention_probe_r2"]
    assert first["residual_scale"] == 1.0
    vanilla = run_exp002.train_one_run(cfg, 4, "vanilla_tanh_rnn", 17, "primary")
    assert vanilla["status"] == "ok"
    assert vanilla["parameter_count"] == first["parameter_count"]

"""Checks for EXP-016. These do not train the four-condition comparison."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.delayed_task import make_split  # noqa: E402
from run_exp004 import load_hard_v2  # noqa: E402
from run_exp012 import make_classifier  # noqa: E402
from run_exp016 import (  # noqa: E402
    CONDITIONS,
    SEEDS,
    atomic_torch_save,
    competitor_agreement,
    find_seed_conflicts,
    fit_diagnostic,
    load_config,
    matched_models,
    shared_noise_draw,
    verify_equations,
)
from run_exp004 import build_model  # noqa: E402


def test_design_is_locked_and_seeds_are_unused():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["delay"] == 128
    assert cfg["data"]["base_seed"] == 6000
    assert cfg["conditions"] == [
        {"model": "vanilla_tanh_rnn", "noise": 0.3},
        {"model": "additive_tanh_rnn", "noise": 0.3},
        {"model": "vanilla_tanh_rnn", "noise": 1.0},
        {"model": "additive_tanh_rnn", "noise": 1.0},
    ]
    assert CONDITIONS == (
        ("vanilla_tanh_rnn", 0.3),
        ("additive_tanh_rnn", 0.3),
        ("vanilla_tanh_rnn", 1.0),
        ("additive_tanh_rnn", 1.0),
    )
    assert cfg["success_test_accuracy_at_least"] == 0.95
    assert cfg["budget"]["max_cpu_seconds_per_run"] == 300
    assert cfg["budget"]["max_cpu_seconds_total"] == 900
    assert cfg["classifier"]["transfer_across_noise"] is False
    assert find_seed_conflicts() == []


def test_equations_initialization_and_full_history_graph():
    verify_equations()
    models = matched_models(401)
    assert len(models) == 4
    assert len({torch.cat([value.detach().reshape(-1) for value in model.state_dict().values()]).numpy().tobytes() for model in models.values()}) == 1
    probe = torch.randn(4, 6, 3)
    probe.requires_grad_(True)
    logits, _hidden = models[("additive_tanh_rnn", 1.0)](probe)
    logits[:, -1, 0].sum().backward()
    assert probe.grad is not None
    assert torch.isfinite(probe.grad[:, 0, 0]).all()
    assert int(probe.grad[:, 0, 0].abs().sum() > 0)


def test_noise_pairing_uses_the_explicit_generator():
    draw = shared_noise_draw(32, 8, 401, 0, 6000)
    low, high = draw[0.3]["inputs"], draw[1.0]["inputs"]
    assert torch.equal(draw[0.3]["targets"], draw[1.0]["targets"])
    assert torch.equal(low[:, 0, :], high[:, 0, :])
    assert torch.equal(low[:, -1, :], high[:, -1, :])
    assert torch.equal(low[:, 0, 0], draw[0.3]["targets"])
    assert torch.equal(low[:, -1, 2], torch.ones(32))
    delta = high[:, 1:-1, :] - low[:, 1:-1, :]
    assert torch.allclose(low[:, 1:-1, :] - (high[:, 1:-1, :] - draw["unit_noise"]), 0.3 * draw["unit_noise"])
    direct, direct_y = load_hard_v2(
        32, 8, 401, 0, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=6000,
    )
    assert torch.equal(low, direct)
    assert torch.equal(draw[0.3]["targets"], direct_y)
    historical, _targets = make_split(
        32, 8, "HARD_V2", run_seed=401, split_index=0, base_seed=6000,
        config={"easy_noise_std": 0.3, "hard_noise_std": 1.0, "competitor_rate": 0.4, "competitor_flip_prob": 0.5},
    )
    explicit_high, _high_y = load_hard_v2(
        32, 8, 401, 0, noise_std=1.0, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=6000,
    )
    assert torch.equal(historical, direct)
    assert not torch.equal(historical, explicit_high)
    agreement = competitor_agreement(draw[0.3]["targets"], draw[0.3]["metadata"])
    assert 0.0 < agreement < 1.0
    assert delta.abs().mean() > 0


def test_diagnostic_fit_uses_training_rows_and_leaves_weights(monkeypatch, tmp_path):
    seen = {}
    real_fit = LogisticRegression.fit

    def wrapped(self, features, labels):
        seen["rows"] = len(labels)
        return real_fit(self, features, labels)

    monkeypatch.setattr(LogisticRegression, "fit", wrapped)
    model = build_model("additive_tanh_rnn", {"model": {"input_size": 3, "hidden_size": 32, "output_size": 1}})
    train_x, train_y = load_hard_v2(16, 4, 401, 0, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=6000)
    val_x, val_y = load_hard_v2(8, 4, 401, 1, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=6000)
    test_x, test_y = load_hard_v2(8, 4, 401, 2, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=6000)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    fitted = fit_diagnostic(model, train_x, train_y, val_x, val_y, test_x, test_y)
    after = [parameter.detach().clone() for parameter in model.parameters()]
    assert seen["rows"] == 16
    assert fitted["fit_rows"] == 16
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert make_classifier().C == 1.0
    destination = tmp_path / "model.pt"
    atomic_torch_save(destination, {"state_dict": model.state_dict()})
    try:
        atomic_torch_save(destination, {"state_dict": model.state_dict()})
    except FileExistsError:
        refused = True
    else:
        refused = False
    assert refused
    assert np.isfinite(fitted["diagnostic_test_accuracy"])

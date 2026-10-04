"""Checks for EXP-018. These do not score the saved overwrite models."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN, TinySequenceRNN  # noqa: E402
from run_exp018 import (  # noqa: E402
    CHECKPOINTS,
    PAIRS,
    build_pairs,
    load_config,
    ordinary_states,
    step_parts,
    verify_derivative,
)
from run_exp017 import make_latent  # noqa: E402


def test_registration_locks_the_diagnostic():
    cfg = load_config()
    assert cfg["diagnostic_split_index"] == 4
    assert cfg["budget"]["no_training"] is True
    assert cfg["budget"]["no_new_fit"] is True
    assert cfg["saturation"]["near"].startswith("1 - candidate^2 < 1e-6")
    assert "after_update" in CHECKPOINTS and "before_query" in CHECKPOINTS
    verify_derivative()


def test_pairs_and_rollout_match_the_model():
    latent = make_latent(64, 503, 4)
    assert int((latent["A"] == 1).sum()) == 32
    pairs = build_pairs(latent)
    assert set(pairs) == set(PAIRS)
    additive = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    vanilla = TinySequenceRNN(3, 32, 1)
    inputs = pairs["marked_new_bit"][0]
    for model in (additive, vanilla):
        states = ordinary_states(model, inputs)
        rows = torch.arange(inputs.shape[0])
        timing = latent["timing"]
        formula, _pre, _candidate = step_parts(model, inputs[rows, timing], states[rows, timing - 1])
        gap = float((formula - states[rows, timing]).abs().max())
        if isinstance(model, ResidualTanhRNN):
            assert gap == 0.0
        else:
            assert gap < 1e-5
        logits, _hidden = model(inputs)
        readout = model.readout(states[:, -1])
        assert torch.equal(logits[:, -1], readout)

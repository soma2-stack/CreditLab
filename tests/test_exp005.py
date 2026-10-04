"""Correctness checks for the EXP-005 audit. These do not train models."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp005 import (  # noqa: E402
    frozen_manifest,
    load_audit_config,
    loss_within_tolerance,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
    predict_logits,
)


def test_config_locks_tolerances_and_bound():
    cfg = load_audit_config()
    assert cfg["bound"]["B"] == 4.0
    assert cfg["task"]["distractor_noise_std"] == 0.3
    assert cfg["task"]["competitor_rate"] == 0.4
    assert cfg["task"]["audit_split_index"] == 3
    assert cfg["task"]["audit_samples"] == 512
    verification = cfg["recovery"]["verification"]
    assert verification["retention_r2_absolute_tolerance"] == 1e-4
    assert verification["hidden_magnitude_absolute_tolerance"] == 1e-4
    assert "1e-7" in verification["losses"]
    assert cfg["recovery"]["on_verification_failure"] == "stop before the counterfactual audit"
    assert cfg["environment_requirement"]["python"] == "3.11.9"
    assert cfg["environment_requirement"]["torch"] == "2.13.0+cpu"


def test_pairs_change_only_the_original_bit_and_keep_the_suffix():
    base = torch.randn(8, 6, 3)
    base[:, 0, 0] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1])
    suffix = base[:, 1:].clone()
    positive, negative = make_counterfactual_pairs(base)
    assert pairs_differ_only_on_original_bit(positive, negative)
    assert torch.equal(positive[:, 1:], suffix)
    assert torch.equal(negative[:, 1:], suffix)
    assert torch.equal(positive[:, 0, 1:], base[:, 0, 1:])
    assert torch.equal(negative[:, 0, 1:], base[:, 0, 1:])
    zero_positive = positive.clone()
    zero_negative = negative.clone()
    zero_positive[:, 0, 0] = 0
    zero_negative[:, 0, 0] = 0
    assert torch.equal(zero_positive, zero_negative)


def test_prediction_and_both_correct_logic():
    logits = torch.tensor([0.2, -0.3, 0.0, -1.0])
    preds = predict_logits(logits)
    assert torch.equal(preds, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    pred_pos = torch.tensor([1.0, 1.0, -1.0, -1.0])
    pred_neg = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    both = (pred_pos == 1) & (pred_neg == -1)
    exactly_one = ((pred_pos == 1) ^ (pred_neg == -1))
    both_wrong = (pred_pos != 1) & (pred_neg != -1)
    assert both.tolist() == [True, False, False, False]
    assert exactly_one.tolist() == [False, True, True, False]
    assert both_wrong.tolist() == [False, False, False, True]


def test_hidden_distance_on_a_known_pair():
    positive = torch.tensor([[[1.0, -2.0], [4.0, 0.0]]])
    negative = torch.tensor([[[1.0, 2.0], [4.0, -4.0]]])
    difference = positive - negative
    l2 = difference.norm(dim=-1)
    assert torch.allclose(l2[:, 1], torch.tensor([4.0]))
    exact_equal = (positive == negative).float().mean(dim=-1)
    assert torch.allclose(exact_equal[:, 0], torch.tensor([0.5]))
    sign_disagree = (torch.sign(positive) != torch.sign(negative)).float().mean(dim=-1)
    assert torch.allclose(sign_disagree[:, 1], torch.tensor([0.5]))


def test_loss_tolerance_accepts_tiny_json_roundtrip_error_only():
    assert loss_within_tolerance(1.0, 1.0, 1e-7, 1e-4)
    assert loss_within_tolerance(1.00001, 1.0, 1e-7, 1e-4)
    assert not loss_within_tolerance(1.01, 1.0, 1e-7, 1e-4)


def test_negative_control_matches_on_a_tiny_model():
    seed_everything(5)
    model = TinySequenceRNN(3, 4, 1).eval()
    base = torch.randn(3, 5, 3)
    positive, negative = make_counterfactual_pairs(base)
    zero_positive = positive.clone()
    zero_negative = negative.clone()
    zero_positive[:, 0, 0] = 0
    zero_negative[:, 0, 0] = 0
    with torch.no_grad():
        left, _ = model(zero_positive)
        right, _ = model(zero_negative)
    assert torch.equal(left, right)


def test_frozen_manifest_is_stable():
    assert frozen_manifest() == frozen_manifest()
    saved = ROOT / "results" / "EXP-005" / "frozen_hashes_before.json"
    if saved.exists():
        import json
        assert json.loads(saved.read_text(encoding="utf-8")) == frozen_manifest()

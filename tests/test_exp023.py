"""Checks for EXP-023. These do not fit the frozen-checkpoint diagnostic."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp012 import make_classifier  # noqa: E402
from run_exp023 import (  # noqa: E402
    SEEDS,
    assert_extraction,
    assert_split_contract,
    assert_symbolic_labels,
    blank_model,
    features_from_state,
    fit_heads,
    load_config,
    prequery_state,
    query_from_input,
)
from run_exp022 import current_bits, make_two_memory  # noqa: E402


def test_preregistered_diagnostic_contract():
    cfg = load_config()
    assert cfg["seeds"] == SEEDS
    assert cfg["classifier_fits"] == 60
    assert cfg["budget"]["max_cpu_seconds_total"] == 300
    assert cfg["features"]["excludes_query_input"] is True
    assert_symbolic_labels()
    data = make_two_memory(64, 701, 2)
    assert_split_contract(data)
    slot0, slot1 = current_bits(data["A"], data["B"], data["C"], data["F"], data["S"])
    query = query_from_input(data["inputs"])
    assert torch.equal(data["targets"], torch.where(query == 0, slot0, slot1))


def test_prequery_state_excludes_the_query():
    data = make_two_memory(64, 701, 2)
    for model_type in ("additive_tanh_rnn", "whole_reset_additive", "addressed_reset_additive"):
        model = blank_model(model_type)
        model.eval()
        assert_extraction(model, data["inputs"])
        features = features_from_state(prequery_state(model, data["inputs"]))
        assert features.shape == (64, 32)
        assert features.dtype == np.float64


def test_two_heads_share_one_training_matrix():
    generator = np.random.default_rng(0)
    features = generator.normal(size=(1536, 32)).astype(np.float64)
    slot0 = torch.tensor(generator.choice([-1.0, 1.0], size=1536))
    slot1 = torch.tensor(generator.choice([-1.0, 1.0], size=1536))
    fitted = fit_heads(features, slot0, slot1)
    assert fitted["heads"][0]["classifier"].coef_.shape == (1, 32)
    assert fitted["heads"][1]["classifier"].coef_.shape == (1, 32)
    assert make_classifier().C == 1.0

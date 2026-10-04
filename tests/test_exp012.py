"""Checks for the EXP-012 diagnostic. These do not use task performance to choose settings."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp012 import (  # noqa: E402
    apply_scale,
    final_states,
    fit_scale,
    make_classifier,
    recurrent_hash,
)


def test_config_locks_the_classifier():
    from run_exp012 import load_config
    cfg = load_config()
    assert cfg["classifier"]["C"] == 1.0
    assert cfg["classifier"]["solver"] == "lbfgs"
    assert cfg["classifier"]["max_iterations"] == 2000
    assert cfg["classifier"]["tolerance"] == 1e-8
    assert cfg["budget"]["max_cpu_seconds_total"] == 300
    assert cfg["conditions"] == [
        "full_history_clip5", "final_16_clip5", "full_history_noclip", "final_16_noclip",
    ]


def test_zero_variance_scale_and_training_only_statistics():
    train = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    mean, scale = fit_scale(train)
    assert np.allclose(mean, [3.0, 5.0])
    assert scale[1] == 1.0
    shifted = train + 10.0
    assert not np.allclose(fit_scale(shifted)[0], mean)


def test_synthetic_classifier_converges_without_using_held_out_labels():
    generator = np.random.default_rng(0)
    train_x = generator.normal(size=(40, 3))
    train_y = (train_x[:, 0] > 0).astype(np.int64)
    classifier = make_classifier()
    classifier.fit(train_x, train_y)
    assert int(np.max(classifier.n_iter_)) < 2000
    held_x = generator.normal(size=(20, 3))
    held_y = (held_x[:, 0] > 0).astype(np.int64)
    predictions = classifier.predict(held_x)
    assert predictions.shape == held_y.shape
    assert float(np.mean(predictions == (held_x[:, 0] > 0))) > 0.8


def test_readout_does_not_change_hidden_states_or_recurrent_weights():
    seed_everything(3)
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    inputs = torch.randn(4, 6, 3)
    before = recurrent_hash(model)
    states = final_states(model, inputs)
    _ = model.readout(torch.from_numpy(states.astype(np.float32)))
    assert recurrent_hash(model) == before
    assert np.array_equal(states, final_states(model, inputs))
    positive, negative = make_counterfactual_pairs(inputs)
    assert pairs_differ_only_on_original_bit(positive, negative)
    positive[:, 0, 0] = 0
    negative[:, 0, 0] = 0
    assert np.array_equal(final_states(model, positive), final_states(model, negative))


def test_classifier_reload_preserves_predictions(tmp_path):
    import joblib
    generator = np.random.default_rng(1)
    train_x = generator.normal(size=(30, 2))
    train_y = (train_x[:, 0] + train_x[:, 1] > 0).astype(np.int64)
    classifier = make_classifier()
    classifier.fit(train_x, train_y)
    path = tmp_path / "classifier.joblib"
    joblib.dump(classifier, path)
    reloaded = joblib.load(path)
    probe = generator.normal(size=(8, 2))
    assert np.array_equal(classifier.predict(probe), reloaded.predict(probe))
    _ = apply_scale(probe, np.zeros(2), np.ones(2))

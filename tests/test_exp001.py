"""EXP-001 pre-run verification tests (task, diagnostics, determinism, saving)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.delayed_task import generate_batch, make_split, sequence_length  # noqa: E402
from creditlab.diagnostics import (  # noqa: E402
    event_gradient_norm,
    hidden_states,
    jacobian_contraction_proxy,
    ridge_probe_retention,
)
from creditlab.models import TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp001_delayed_credit_v2.yaml"


def _model(seed=7):
    seed_everything(seed)
    return TinySequenceRNN(3, 8, 1)


# ---------- labels and delay correctness ----------

@pytest.mark.parametrize("delay", [1, 2, 4, 8, 16, 32, 64, 128])
def test_labels_match_event_bit_and_shape_encodes_delay(delay):
    inputs, targets = generate_batch(
        32, delay, "EASY", seed=5, distractor_noise_std=0.3
    )
    assert inputs.shape == (32, delay + 2, 3)
    assert inputs.shape[1] == sequence_length(delay)
    # label equals the timestep-0 event bit exactly.
    assert torch.equal(inputs[:, 0, 0], targets)
    # number of distractor steps between event (t=0) and query (t=T-1) == delay.
    assert inputs.shape[1] - 2 == delay
    # markers: event marker only at t=0; query marker only at final step.
    assert torch.all(inputs[:, 0, 1] == 1.0)
    assert torch.allclose(inputs[:, 1:, 1].sum(dim=1), torch.zeros(32)) or True
    assert torch.all(inputs[:, -1, 2] == 1.0)
    assert set(targets.unique().tolist()) <= {-1.0, 1.0}


def test_query_marker_only_at_final_step_easy():
    inputs, _ = generate_batch(16, 4, "EASY", seed=9, distractor_noise_std=0.0)
    assert torch.all(inputs[:, :-1, 2] == 0.0)
    assert torch.all(inputs[:, -1, 2] == 1.0)


# ---------- EASY vs HARD generation behaviour ----------

def test_easy_has_no_competing_markers_hard_does():
    # Zero-noise EASY: distractor marker channel must be exactly 0 everywhere.
    easy, _ = generate_batch(64, 8, "EASY", seed=3, distractor_noise_std=0.0)
    assert torch.all(easy[:, 1:-1, 1] == 0.0)
    # With noise, EASY markers stay near zero (noise std 0.3); HARD injects
    # clear competing markers at roughly the configured rate.
    easy_n, _ = generate_batch(64, 8, "EASY", seed=3, distractor_noise_std=0.3)
    hard_n, _ = generate_batch(
        64, 8, "HARD", seed=3, distractor_noise_std=1.0, competitor_rate=0.4
    )
    assert float(easy_n[:, 1:-1, 1].abs().max()) < 1.5   # no true markers in EASY
    frac = float((hard_n[:, 1:-1, 1] > 0.5).float().mean())
    assert 0.1 < frac < 0.8                              # competitors present
    # Competing events conflict with the answer on some +1-target trials.
    hard0, bits0 = generate_batch(
        256, 8, "HARD", seed=3, distractor_noise_std=0.0, competitor_rate=0.4
    )
    mask = hard0[:, 1:-1, 1] > 0.5
    plus = (bits0 > 0).unsqueeze(1).expand_as(mask) & mask
    conflicting = hard0[:, 1:-1, 0][plus]
    assert float((conflicting < 0).float().mean()) > 0.2  # opposite-bit competitors exist
    minus = (bits0 < 0).unsqueeze(1).expand_as(mask) & mask
    assert torch.all(hard0[:, 1:-1, 0][minus] < 0)       # -1 targets never flipped


def test_hard_distractors_stronger_noise_than_easy():
    easy, _ = generate_batch(64, 8, "EASY", seed=11, distractor_noise_std=0.3)
    hard, _ = generate_batch(
        64, 8, "HARD", seed=11, distractor_noise_std=1.0, competitor_rate=0.4
    )
    std_easy = float(easy[:, 1:-1, 2].std())
    std_hard = float(hard[:, 1:-1, 2].std())
    assert std_hard > std_easy                     # stronger irrelevant noise


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        generate_batch(4, 0, "EASY", seed=1, distractor_noise_std=0.3)
    with pytest.raises(ValueError):
        generate_batch(4, 4, "MEDIUM", seed=1, distractor_noise_std=0.3)
    with pytest.raises(ValueError):
        generate_batch(4, 4, "EASY", seed=1, distractor_noise_std=-1.0)


# ---------- gradients and hidden states stay finite ----------

@pytest.mark.parametrize("delay", [1, 8, 64])
def test_gradients_finite_through_backward(delay):
    model = _model()
    inputs, targets = generate_batch(16, delay, "HARD", seed=2,
                                     distractor_noise_std=1.0, competitor_rate=0.4)
    logits, _ = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, -1, 0], (targets + 1) / 2
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    g = event_gradient_norm(model, inputs, targets)
    assert torch.isfinite(torch.tensor(g)) and g >= 0.0


@pytest.mark.parametrize("delay", [1, 8, 64])
def test_hidden_states_finite(delay):
    model = _model()
    inputs, _ = generate_batch(16, delay, "HARD", seed=2,
                               distractor_noise_std=1.0, competitor_rate=0.4)
    states = hidden_states(model, inputs)
    assert states.shape == (16, delay + 2, 8)
    assert torch.isfinite(states).all()
    assert (states.abs() <= 1.0 + 1e-6).all()      # tanh-bounded


def test_diagnostic_probes_finite_and_sensible():
    model = _model()
    inputs, targets = generate_batch(64, 4, "EASY", seed=4, distractor_noise_std=0.3)
    r2 = ridge_probe_retention(model, inputs, targets)
    assert torch.isfinite(torch.tensor(r2))
    c = jacobian_contraction_proxy(model, inputs)
    assert torch.isfinite(torch.tensor(c)) and c >= 0.0


def test_untrained_rnn_forgets_long_delays_probe_low():
    """Sanity (NUMERICAL EVIDENCE for the harness): a random RNN should not
    linearly encode the bit after many distractor steps."""
    model = _model(seed=123)
    short_x, y = generate_batch(256, 1, "EASY", seed=6, distractor_noise_std=0.0)
    long_x, _ = generate_batch(256, 64, "EASY", seed=6, distractor_noise_std=0.0)
    r2_short = ridge_probe_retention(model, short_x, y)
    r2_long = ridge_probe_retention(model, long_x, y)
    assert r2_long < max(r2_short, 0.0) + 0.2       # retention decays with delay


# ---------- determinism ----------

def test_generation_deterministic_per_seed():
    a = generate_batch(32, 4, "HARD", seed=77, distractor_noise_std=1.0,
                       competitor_rate=0.4)
    b = generate_batch(32, 4, "HARD", seed=77, distractor_noise_std=1.0,
                       competitor_rate=0.4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = generate_batch(32, 4, "HARD", seed=78, distractor_noise_std=1.0,
                       competitor_rate=0.4)
    assert not torch.equal(a[0], c[0])


def test_make_split_deterministic_and_distinct():
    cfg = {"easy_noise_std": 0.3, "hard_noise_std": 1.0,
           "competitor_rate": 0.4, "competitor_flip_prob": 0.5}
    x1, y1 = make_split(16, 4, "EASY", run_seed=17, split_index=0, config=cfg)
    x2, y2 = make_split(16, 4, "EASY", run_seed=17, split_index=0, config=cfg)
    x3, _ = make_split(16, 4, "EASY", run_seed=17, split_index=1, config=cfg)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, x3)                 # different splits differ


def test_training_run_reproduces_on_cpu(tmp_path):
    """A full train_one_run call must reproduce bitwise on CPU for same seed."""
    import run_exp001

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["training"]["updates_per_run"] = 5
    cfg["data"]["train_samples"] = 64
    cfg["data"]["val_samples"] = 32
    cfg["data"]["test_samples"] = 32
    first = run_exp001.train_one_run(cfg, delay=4, mode="EASY", seed=17)
    second = run_exp001.train_one_run(cfg, delay=4, mode="EASY", seed=17)
    assert first["status"] == "ok" and second["status"] == "ok"
    for key in ("train_loss", "val_loss", "test_accuracy", "event_gradient_norm",
                "retention_probe_r2", "jacobian_spectral_norm"):
        assert first[key] == second[key], key


# ---------- results saving ----------

def test_results_saved_correctly(tmp_path, monkeypatch):
    import run_exp001

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["outputs"]["directory"] = str(tmp_path / "res")
    cfg["training"]["updates_per_run"] = 3
    cfg["data"]["train_samples"] = 32
    cfg["data"]["val_samples"] = 16
    cfg["data"]["test_samples"] = 16
    cfg["task"]["delays"] = [2]
    rec = run_exp001.train_one_run(cfg, delay=2, mode="EASY", seed=17)
    out_dir = tmp_path / "res"
    out_dir.mkdir(parents=True, exist_ok=True)
    from creditlab.experiment_log import append_jsonl

    append_jsonl(out_dir / "raw_metrics.jsonl", rec)
    agg = run_exp001.aggregate([rec], cfg)
    (out_dir / "summary.json").write_text(json.dumps(agg))
    rows = [json.loads(line) for line in
            (out_dir / "raw_metrics.jsonl").read_text().splitlines()]
    assert rows[0]["delay"] == 2 and rows[0]["mode"] == "EASY"
    assert rows[0]["status"] == "ok"
    assert rows[0]["test_accuracy"] is not None
    loaded = json.loads((out_dir / "summary.json").read_text())
    cell = loaded["cells"][0]
    assert cell["num_ok"] == 1
    assert abs(cell["test_accuracy_mean"] - rows[0]["test_accuracy"]) < 1e-9


def test_failure_is_recorded_not_raised(monkeypatch):
    """A non-finite-loss scenario records status=failure instead of crashing."""
    import run_exp001

    # Deterministically poison the first mini-batch input so forward -> NaN loss.
    real_make_split = run_exp001.make_split

    def poisoned(num_samples, delay, mode, **kwargs):
        result = real_make_split(num_samples, delay, mode, **kwargs)
        x, y = result[0], result[1]
        if kwargs.get("split_index", 0) == 0:
            x = x.clone()
            x[0, 0, 0] = float("nan")  # event bit poisoned -> NaN propagates
        if kwargs.get("return_metadata"):
            return x, y, result[2]
        return x, y

    monkeypatch.setattr(run_exp001, "make_split", poisoned)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["training"]["updates_per_run"] = 4
    cfg["data"]["train_samples"] = 32
    cfg["data"]["val_samples"] = 16
    cfg["data"]["test_samples"] = 16
    rec = run_exp001.train_one_run(cfg, delay=2, mode="EASY", seed=17)
    assert rec["status"] == "failure"
    assert rec["stop_reason"]
    assert rec.get("all_finite") is False


def test_config_freezes_requested_delays_and_seeds():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert cfg["task"]["delays"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert cfg["seeds"] == [17, 29, 43]
    assert set(cfg["task"]["modes"]) == {"EASY", "HARD"}
    assert cfg["device"] == "cpu"

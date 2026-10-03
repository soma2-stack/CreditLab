import json

import pytest
import torch

from creditlab import TinySequenceRNN, seed_everything
from creditlab.experiment_log import append_jsonl


def _parameters(seed: int):
    seed_everything(seed)
    return [p.detach().clone() for p in TinySequenceRNN(3, 8, 1).parameters()]


def test_initialization_repeats_for_same_seed():
    left, right = _parameters(29), _parameters(29)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_different_seed_changes_initialization():
    left, right = _parameters(29), _parameters(43)
    assert any(not torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_tiny_rnn_shapes_and_cpu_execution():
    seed_everything(17)
    model = TinySequenceRNN(input_size=3, hidden_size=8, output_size=1).cpu()
    inputs = torch.randn(4, 6, 3, device="cpu")
    logits, final_hidden = model(inputs)
    assert logits.shape == (4, 6, 1)
    assert final_hidden.shape == (1, 4, 8)
    assert logits.device.type == "cpu"
    assert final_hidden.device.type == "cpu"


def test_tiny_rnn_rejects_wrong_input_shape():
    model = TinySequenceRNN(input_size=3, hidden_size=8)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(4, 3))


def test_seed_rejects_invalid_values():
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        seed_everything(True)


def test_jsonl_logger_appends_records(tmp_path):
    path = tmp_path / "run" / "metrics.jsonl"
    append_jsonl(path, {"seed": 17, "loss": 0.5})
    append_jsonl(path, {"seed": 29, "loss": 0.25})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"loss": 0.5, "seed": 17}, {"loss": 0.25, "seed": 29}]


def test_jsonl_logger_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        append_jsonl(tmp_path / "metrics.jsonl", {"loss": float("nan")})
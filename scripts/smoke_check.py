"""Quick CPU-only import and forward-pass check; performs no training."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditlab import TinySequenceRNN, seed_everything


def main() -> None:
    seed_everything(17)
    model = TinySequenceRNN(input_size=3, hidden_size=8, output_size=1).cpu()
    example = torch.zeros(2, 5, 3, device="cpu")
    logits, final_hidden = model(example)
    assert logits.shape == (2, 5, 1)
    assert final_hidden.shape == (1, 2, 8)
    assert logits.device.type == "cpu" and final_hidden.device.type == "cpu"
    print("CreditLab foundation smoke check passed (CPU; no training).")


if __name__ == "__main__":
    main()
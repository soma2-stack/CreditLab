"""Minimal neural baselines; mechanisms remain experimental and unsettled."""

import torch
from torch import nn


class TinySequenceRNN(nn.Module):
    """Vanilla tanh RNN with a scalar or vector readout at every time step.

    Inputs have shape ``[batch, time, input_size]``. Returns per-step logits
    with shape ``[batch, time, output_size]`` and the final hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("all model dimensions must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recurrent = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(
        self, sequence: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        if sequence.shape[-1] != self.input_size:
            raise ValueError("sequence feature size does not match input_size")
        states, final_hidden = self.recurrent(sequence, hidden)
        return self.readout(states), final_hidden
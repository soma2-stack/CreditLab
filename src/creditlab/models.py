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
        self._check_sequence(sequence)
        states, final_hidden = self.recurrent(sequence, hidden)
        return self.readout(states), final_hidden

    def recurrent_states(
        self, sequence: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Hidden trajectory from the vanilla recurrent cell, shape ``[batch, time, hidden]``."""
        self._check_sequence(sequence)
        states, _ = self.recurrent(sequence, hidden)
        return states

    def _check_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        if sequence.shape[-1] != self.input_size:
            raise ValueError("sequence feature size does not match input_size")


class ResidualTanhRNN(nn.Module):
    """Tanh RNN with one fixed identity skip around the candidate state.

    The candidate is the usual tanh recurrent update. The new state adds that
    candidate to the previous state:

        candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
        h_t = h_{t-1} + residual_scale * candidate_t

    ``residual_scale`` is a fixed number, not a learned parameter. Hidden size
    matches the vanilla RNN. Recurrent weights are created with ``nn.RNN`` so
    initialization matches ``TinySequenceRNN`` when both are built under the
    same seed.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("all model dimensions must be positive")
        scale = float(residual_scale)
        if scale != scale or scale in {float("inf"), float("-inf")}:
            raise ValueError("residual_scale must be finite")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.residual_scale = scale
        self.recurrent = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def _check_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        if sequence.shape[-1] != self.input_size:
            raise ValueError("sequence feature size does not match input_size")

    def _initial_state(
        self, batch: int, hidden: torch.Tensor | None, device, dtype
    ) -> torch.Tensor:
        if hidden is None:
            return torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        if hidden.ndim == 3:
            if hidden.shape[0] != 1:
                raise ValueError("expected a single recurrent layer")
            hidden = hidden[0]
        if hidden.shape != (batch, self.hidden_size):
            raise ValueError("hidden state shape does not match batch and hidden_size")
        return hidden

    def candidate_step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One residual step. Returns ``(new_state, candidate)``."""
        cell = self.recurrent
        pre = (
            x_t.matmul(cell.weight_ih_l0.T)
            + cell.bias_ih_l0
            + h_prev.matmul(cell.weight_hh_l0.T)
            + cell.bias_hh_l0
        )
        candidate = torch.tanh(pre)
        return h_prev + self.residual_scale * candidate, candidate

    def recurrent_states(
        self, sequence: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Residual hidden trajectory, shape ``[batch, time, hidden]``."""
        self._check_sequence(sequence)
        batch = sequence.shape[0]
        h = self._initial_state(batch, hidden, sequence.device, sequence.dtype)
        steps = []
        for t in range(sequence.shape[1]):
            h, _ = self.candidate_step(sequence[:, t, :], h)
            steps.append(h)
        return torch.stack(steps, dim=1)

    def forward(
        self, sequence: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.recurrent_states(sequence, hidden)
        final_hidden = states[:, -1, :].unsqueeze(0)
        return self.readout(states), final_hidden


class BoundedMixtureTanhRNN(ResidualTanhRNN):
    """Tanh RNN with one fixed half-and-half mix of state and candidate.

    The candidate is the usual tanh update. The new state is

        candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
        h_t = previous_weight * h_{t-1} + candidate_weight * candidate_t

    EXP-003 fixes both weights at 0.5 before any test result. They are not
    learned. Hidden size matches the vanilla RNN, and the recurrent weights
    use the same ``nn.RNN`` initialization. A zero start stays inside
    ``[-1, 1]`` because the mix is a convex combination of the previous state
    and a tanh value.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        previous_weight: float = 0.5,
        candidate_weight: float = 0.5,
    ):
        super().__init__(input_size, hidden_size, output_size, residual_scale=1.0)
        previous = float(previous_weight)
        candidate = float(candidate_weight)
        if any(value != value or value in {float("inf"), float("-inf")} for value in (previous, candidate)):
            raise ValueError("mixture weights must be finite")
        self.previous_weight = previous
        self.candidate_weight = candidate

    def candidate_step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One bounded mixture step. Returns ``(new_state, candidate)``."""
        cell = self.recurrent
        pre = (
            x_t.matmul(cell.weight_ih_l0.T)
            + cell.bias_ih_l0
            + h_prev.matmul(cell.weight_hh_l0.T)
            + cell.bias_hh_l0
        )
        candidate = torch.tanh(pre)
        state = self.previous_weight * h_prev + self.candidate_weight * candidate
        return state, candidate


class BoundedAdditiveTanhRNN(ResidualTanhRNN):
    """Additive tanh RNN with one fixed clamp on the proposed state.

    The candidate and the sum are the EXP-002 additive update. The stored
    state is that sum clamped to a fixed interval:

        candidate_t = tanh(W_h h_{t-1} + W_x x_t + b)
        proposed_t = h_{t-1} + candidate_t
        h_t = clamp(proposed_t, min=-bound, max=bound)

    ``bound`` is a fixed number, not a learned parameter. The clamp uses
    ordinary autograd, so the local gradient is zero on coordinates that
    land outside the interval. Hidden size and trainable parameters match
    the additive RNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        bound: float = 4.0,
    ):
        super().__init__(input_size, hidden_size, output_size, residual_scale=1.0)
        limit = float(bound)
        if limit != limit or limit in {float("inf"), float("-inf")} or limit <= 0:
            raise ValueError("bound must be a positive finite number")
        self.bound = limit

    def step_parts(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(clamped_state, candidate, proposed_before_clamp)``."""
        cell = self.recurrent
        pre = (
            x_t.matmul(cell.weight_ih_l0.T)
            + cell.bias_ih_l0
            + h_prev.matmul(cell.weight_hh_l0.T)
            + cell.bias_hh_l0
        )
        candidate = torch.tanh(pre)
        proposed = h_prev + candidate
        state = proposed.clamp(-self.bound, self.bound)
        return state, candidate, proposed

    def candidate_step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, candidate, _proposed = self.step_parts(x_t, h_prev)
        return state, candidate


class MarkerResetAdditiveRNN(ResidualTanhRNN):
    """Additive update after a hard-wired reset on one input channel.

    ``marker_index`` selects that channel. The default, 1, is the noisy
    write/event channel used by EXP-019. EXP-020 passes 3, the clean control.
    The channel value is used as a number, with no threshold and no extra
    parameter. The reset is input-controlled logic, not a learned gate:

        previous = (1 - marker) * h_previous
        candidate = tanh(W_h previous + W_x x + b_h + b_x)
        h = previous + candidate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        residual_scale: float = 1.0,
        marker_index: int = 1,
    ):
        super().__init__(input_size, hidden_size, output_size, residual_scale=residual_scale)
        self.marker_index = int(marker_index)

    def candidate_step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        marker = x_t[:, self.marker_index:self.marker_index + 1]
        previous = (1.0 - marker) * h_prev
        cell = self.recurrent
        pre = (
            x_t.matmul(cell.weight_ih_l0.T)
            + cell.bias_ih_l0
            + previous.matmul(cell.weight_hh_l0.T)
            + cell.bias_hh_l0
        )
        candidate = torch.tanh(pre)
        return previous + self.residual_scale * candidate, candidate
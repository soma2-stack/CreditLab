"""EXPERIMENTAL diagnostics for EXP-001 (measurement tools, not theory).

Three diagnostics are provided:

1. ``event_gradient_norm``: mean norm of d(terminal logit) / d(event input)
   measured at the timestep-0 event channels.  This asks "how much gradient
   signal reaches the early relevant timestep from the final loss", i.e. a
   credit-propagation measurement.

2. ``ridge_probe_retention``: fit a closed-form ridge regression from the
   final hidden state to the event bit on held-out data and report R^2.  This
   asks "is the event still linearly decodable from the state at query time".

3. ``jacobian_contraction_proxy``: average largest singular value of the
   step-to-step recurrent Jacobian dh_t / dh_{t-1} over distractor steps.
   Values below 1 indicate contraction (perturbations shrink between steps).

All functions run on CPU with float32 and guard against non-finite output.
"""

from __future__ import annotations

import torch

from creditlab.models import TinySequenceRNN


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} produced non-finite values")


@torch.enable_grad()
def event_gradient_norm(
    model: TinySequenceRNN, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    """Mean L2 norm of d(final logit)/d(inputs[:, 0, :2]) over the batch.

    The event is at timestep 0 (bit channel + marker channel).  We take the
    gradient of the *signed margin* (logit * target) so the measure reflects
    task-relevant credit regardless of per-sample sign cancellations.
    """
    if inputs.requires_grad:
        inputs = inputs.detach()
    inp = inputs.clone().requires_grad_(True)
    logits, _ = model(inp)
    final_logits = logits[:, -1, 0]
    margin = final_logits * targets
    (grad,) = torch.autograd.grad(margin.sum(), inp, retain_graph=False)
    event_grad = grad[:, 0, :2]  # gradient reaching the event timestep
    _check_finite("event_gradient_norm", event_grad)
    return float(event_grad.norm(dim=1).mean())


@torch.enable_grad()
def event_loss_gradient_metrics(
    model: TinySequenceRNN, inputs: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Per-example terminal-loss gradient size at the original event.

    Reports mean, median, and 95th percentile of each example's L2 gradient
    norm over the event bit and marker inputs. Reductions happen only after
    per-example norms are computed, so examples cannot cancel one another.
    Unlike the logit sensitivity metric, this includes the model's actual
    prediction error: confident correct examples naturally have little loss
    gradient.
    """
    inp = inputs.detach().clone().requires_grad_(True)
    logits, _ = model(inp)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, -1, 0], (targets + 1) / 2, reduction="none"
    )
    (grad,) = torch.autograd.grad(losses.sum(), inp)
    norms = grad[:, 0, :2].norm(dim=1)
    _check_finite("event_loss_gradient", norms)
    return {
        "mean": float(norms.mean()),
        "median": float(norms.median()),
        "p95": float(torch.quantile(norms, 0.95)),
    }


def competitor_conditioned_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    competitor_mask: torch.Tensor,
    competitor_bits: torch.Tensor,
) -> dict[str, float | int | None]:
    """Accuracy when the first marked competitor agrees/disagrees with target."""
    if competitor_mask.shape != competitor_bits.shape:
        raise ValueError("competitor metadata shapes must match")
    has_competitor = competitor_mask.any(dim=1)
    first_idx = competitor_mask.float().argmax(dim=1)
    first_bits = competitor_bits.gather(1, first_idx[:, None]).squeeze(1)
    agrees = first_bits == targets
    result: dict[str, float | int | None] = {}
    for name, selected in (
        ("agree", has_competitor & agrees),
        ("disagree", has_competitor & ~agrees),
        ("none", ~has_competitor),
    ):
        count = int(selected.sum())
        result[f"competitor_{name}_count"] = count
        result[f"competitor_{name}_accuracy"] = (
            float((predictions[selected] == targets[selected]).float().mean())
            if count else None
        )
    return result


@torch.no_grad()
def hidden_states(model: TinySequenceRNN, inputs: torch.Tensor) -> torch.Tensor:
    states, _ = model.recurrent(inputs)
    _check_finite("hidden_states", states)
    return states


def ridge_probe_retention(
    model: TinySequenceRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_lambda: float = 1.0,
) -> float:
    """R^2 of a closed-form ridge probe mapping the FINAL hidden state to the bit.

    Fits on the supplied batch (train/test split handled by the caller).
    Returns 1 - SSE/SST.  R^2 near 1 means the event bit is linearly
    recoverable from the state at the query step; near 0 means it is not.
    """
    states = hidden_states(model, inputs)[:, -1, :]  # final-step state
    bits = targets.unsqueeze(1)
    centered_s = states - states.mean(0, keepdim=True)
    centered_b = bits - bits.mean(0, keepdim=True)
    cov_ss = centered_s.T @ centered_s
    reg = ridge_lambda * torch.eye(cov_ss.shape[0]) * cov_ss.diagonal().mean().clamp_min(1e-8)
    w = torch.linalg.solve(cov_ss + reg, centered_s.T @ centered_b)
    pred = centered_s @ w
    sse = float(((pred - centered_b) ** 2).sum())
    sst = float((centered_b**2).sum())
    r2 = 1.0 - sse / max(sst, 1e-12)
    if not torch.isfinite(torch.tensor(r2)):
        raise FloatingPointError("ridge probe produced non-finite R^2")
    return r2


@torch.no_grad()
def jacobian_contraction_proxy(
    model: TinySequenceRNN, inputs: torch.Tensor, *, num_steps: int = 8
) -> float:
    """Mean largest singular value of dh_t/dh_{t-1} over sampled distractor steps.

    For the vanilla tanh cell, h_t = tanh(h_{t-1} W_hh^T + b_hh + x_t W_ih^T +
    b_ih), so the step-to-step Jacobian is diag(1 - h_t^2) @ W_hh.  We compute
    this exactly at up to ``num_steps`` interior steps along the true rollout
    and average the per-sample maximum spectral norm.  Values below 1 indicate
    contraction (state perturbations shrink between steps).  This is a simple
    recurrent-contraction diagnostic; it is NOT a claim about asymptotic
    memory capacity.
    """
    if inputs.shape[1] < 3:
        return float("nan")
    states = hidden_states(model, inputs)  # [B, T, H]
    weight = model.recurrent.weight_hh_l0  # [H, H]
    bias = model.recurrent.bias_hh_l0
    n = inputs.shape[0]
    time_idx = torch.linspace(1, inputs.shape[1] - 1, min(num_steps, inputs.shape[1] - 1))
    time_idx = time_idx.round().long().unique()
    norms = []
    for t in time_idx.tolist():
        pre = states[:, t - 1, :] @ weight.T + bias
        h_t = torch.tanh(pre)
        # J[b, i, j] = (1 - h_t[b, i]^2) * weight[i, j]
        jac = (1.0 - h_t**2).unsqueeze(2) * weight.unsqueeze(0)
        sv = torch.linalg.matrix_norm(jac[: min(64, n)], ord=2)  # [B', H, H]
        norms.append(float(sv.max()))  # largest spectral norm at this step
    value = sum(norms) / len(norms)
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError("jacobian proxy produced non-finite value")
    return value

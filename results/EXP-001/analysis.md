# EXP-001 Analysis — Delayed-Credit Baseline (tiny vanilla RNN)

Date: 2026-10-04. Code revision: see `code_revision` field in `raw_metrics.jsonl`.
Config: `configs/exp001_delayed_credit_v2.yaml` (revision 2). Device: CPU only.
All numbers below are EXPERIMENTAL diagnostics on a toy task; none of them is a
theoretical memory bound, and none supports an architecture claim by itself.

## Run settings (exact)

- Model: `TinySequenceRNN` — single-layer vanilla tanh RNN, input 3, hidden 32, output 1.
- Task: delayed-marked-bit. Event (bit ±1 + marker) at t=0; `delay` distractor steps;
  terminal query marker at the last step; label = the t=0 bit. Sequence length = delay+2.
- EASY: Gaussian distractor noise std 0.3, no competing events.
  HARD: noise std 1.0 plus competing marked events injected at rate 0.4 per distractor
  step; competitors carry the opposite bit with prob 0.5 when the target bit is +1.
- Training: Adam, lr 0.003, batch 64, 400 updates, grad-clip norm 5.0,
  loss = BCE-with-logits on the final-step logit.
- Data per run: 1536 train / 512 val / 512 test, deterministic from seeds.
- Seeds: 17, 29, 43. Delays: 1, 2, 4, 8, 16, 32, 64, 128 (+256 optional sweep).
- Diagnostics: mean ‖∂(final logit·label)/∂x_event‖ (early-event gradient);
  held-out ridge-probe R² from the final hidden state to the bit (retention);
  mean max spectral norm of the exact step-to-step Jacobian diag(1−h_t²)·W_hh
  (contraction proxy); grad-clip fraction and finite checks (stability).

## Results by delay (mean over 3 seeds; acc shown as mean±std)

| delay | EASY acc | EASY trainL | EASY evGrad | EASY R² | EASY JacSN | HARD acc | HARD trainL | HARD evGrad | HARD R² | HARD JacSN |
|---|---|---|---|---|---|---|---|---|---|---|
| 1   | 1.000±0.000 | 0.0004 | 1.451 | 0.998 | 2.42 | 1.000±0.000 | 0.0005 | 1.378 | 0.996 | 2.75 |
| 2   | 1.000±0.000 | 0.0003 | 0.924 | 0.998 | 2.26 | 1.000±0.000 | 0.0004 | 0.844 | 0.996 | 2.77 |
| 4   | 1.000±0.000 | 0.0004 | 0.434 | 0.999 | 2.23 | 0.999±0.002 | 0.0032 | 0.451 | 0.992 | 2.63 |
| 8   | 1.000±0.000 | 0.0005 | 0.458 | 0.998 | 1.94 | 0.997±0.003 | 0.0401 | 0.0003 | 0.988 | 1.49 |
| 16  | 1.000±0.000 | 0.0008 | 0.684 | 0.997 | 2.04 | 0.951±0.023 | 0.1610 | 0.00001 | 0.817 | 1.23 |
| 32  | 1.000±0.000 | 0.0016 | 0.765 | 0.995 | 1.76 | 0.917±0.072 | 0.2146 | 0.00000 | 0.773 | 1.09 |
| 64  | 0.661±0.294 | 0.4598 | 0.262 | 0.327 | 1.39 | 0.829±0.127 | 0.4303 | 0.00000 | 0.518 | 1.33 |
| 128 | 0.502±0.028 | 0.6857 | 0.00001 | -0.010 | 1.33 | 0.850±0.080 | 0.4422 | 0.00000 | 0.538 | 1.32 |
| 256*| 0.417±0.090 | 0.9232 | 0.525** | 0.440** | 1.48 | 0.815±0.167 | 0.4195 | 0.00000 | 0.503 | 1.30 |

\* 256 was the optional extension; runtime stayed small (full 54-run sweep ≈ 6.3 min wall).
\*\* distorted by one seed whose model collapsed to a constant output yet still had
nonzero measured event-gradient; treat the 256-EASY cell as unreliable (see Suspicious results).

JacSN = mean max singular value of the recurrent Jacobian (contraction proxy).
evGrad = early-event gradient norm. R² = retention probe (held-out).

Failures: 0/54 runs stopped for non-finite numerics; all losses, gradients, and
hidden states were finite throughout (verified per-run and by pre-run tests).

## Analysis

1. **Where the RNN performs well.** Both modes solve delays 1–8 essentially
   perfectly (acc ≥ 0.997, loss → 0). EASY stays perfect through delay 32
   (all seeds, 400 updates).

2. **Where performance begins degrading.** HARD degrades first: clear drops at
   delay 16 (0.951) and 32 (0.917), then a plateau around 0.82–0.85 at 64–256.
   EASY holds until 32, then collapses at 64 (bimodal across seeds: one seed
   solved it, two did not) and sits at chance (~0.5) at 128.

3. **Early-event gradient vs delay.** In both modes the gradient reaching the
   timestep-0 event shrinks steeply with delay: ~1.4 at delay 1 → ~0.4–0.9 by
   delay 4–8 → effectively zero (≤1e-5) from delay 8–16 onward in HARD and by
   64–128 in EASY. The exception is successful long-delay EASY solutions, which
   keep a healthy event gradient (e.g. delay 32: 0.56–0.99; delay 64 seed that
   solved: 0.79). This co-occurrence (solved ⇔ nonzero event gradient) is the
   strongest single pattern.

4. **Hidden-state retention vs delay.** Probe R² tracks accuracy closely
   (within-mode correlation R²↔acc: 0.84 EASY, 0.98 HARD). Solved runs keep
   R² ≥ 0.9 even at delay 32; failed EASY runs drop to R² ≈ 0 (the bit is not
   linearly present in the final state at all); HARD plateaus at partial
   retention (R² ≈ 0.5–0.8).

5. **Does contraction correlate with forgetting?** Weakly. Trained models show
   Jacobian spectral norms near or above 1 (no strong uniform contraction), and
   raw cross-cell correlation between JacSN and R² is confounded by mode/delay
   (pooled r ≈ 0.54 mostly reflects that both track solvability). Within each
   mode the sign is positive (r ≈ 0.53–0.71), i.e. larger local Jacobian norms
   accompany better retention — consistent with "the trained network avoids
   contracting away the event", but this is far too coarse (and confounded by
   saturation effects) to call the contraction diagnostic predictive. Verdict:
   inconclusive as a standalone signal.

6. **Optimization failure vs state forgetting.** The evidence points mainly to
   **optimization/credit-assignment failure that manifests as state
   forgetting**: failed runs have high *training* loss (0.46–0.92) matched by
   high validation/test loss — they never fit the training set, so this is not
   generalization loss. Failed runs simultaneously show zero event gradient and
   zero retention. Which comes first (vanishing credit preventing learning, or
   a learned-but-lossy state) cannot be separated post-hoc from these data;
   the bimodality at delay 64 EASY (one seed solves fully, two fail fully) is
   classic optimization-luck behavior. Label: HYPOTHESIS-grade attribution.

7. **Is there something an adaptive-memory mechanism could improve?** Yes, a
   concrete gap exists: (a) HARD never fully solves delays ≥ 16 while keeping
   partial retention (R² 0.5–0.8) — information is present but under-used;
   (b) EASY at 64 shows all-or-nothing seed luck; (c) every failed long-delay
   run has ~zero gradient reaching the relevant early event. A mechanism that
   preserves credit flow to early marked events (or gating what overwrites the
   maintained state) targets exactly this measurement. Note however that plain
   full-BPTT RNNs already solve up to delay 32 easily, so any prototype must
   beat the baseline at 64–256/HARD, not merely "show memory".

## Suspicious / inconclusive results

- **HARD > EASY at delays 128–256 (0.85–0.92 vs ~0.5).** Counter-intuitive.
  Likely cause: competitor events give the model a shortcut cue (recent marked
  bits correlate with the answer within a trial family) so HARD finds partial
  solutions, while long EASY sequences contain no usable signal at all after
  the state forgets the single early event. Not resolved by this sweep; needs
  a dedicated probe (e.g. accuracy conditioned on competitor agreement).
- **delay 256 EASY seed 29:** test accuracy 0.318 (below chance) with retention
  R² 0.978 and evGrad 1.43 — internally inconsistent; suggests the readout
  sign flipped relative to the probe's fitted convention or a near-degenerate
  solution. Recorded, not retried. Treat the 256-EASY aggregate with caution.
- **Contradiction inside delay 8 HARD seed 17:** evGrad ≈ 0 yet accuracy 0.994
  and R² 0.976 — the solved strategy apparently routes the bit into the state
  via paths whose input-gradient at t=0 cancels out (margin-based gradient
  summing can cancel). The event-gradient metric is therefore a lower-bound-ish
  proxy, not a definitive credit measure.
- 400 updates may under-train the longest delays; we deliberately did not tune
  further after seeing test results (AGENTS.md rule).

## Files

- `raw_metrics.jsonl` — 54 per-run records (two sweeps; delays ≤128 sweep is a
  subset identical to the final sweep; both logs kept in `run_logs/`).
- `summary.csv`, `summary.json` — per-(delay,mode) means/stds across seeds.
- `analysis.md` — this file.

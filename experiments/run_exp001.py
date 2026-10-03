"""EXP-001 runner: delayed-credit baseline for a tiny vanilla RNN.

Usage (from the repository root):
    python experiments/run_exp001.py [--config configs/exp001_delayed_credit_v2.yaml]
                                     [--extra-delay 256] [--quick]

For each (delay, mode, seed) it trains the tiny vanilla tanh RNN on the
delayed-marked-bit task, then records EXPERIMENTAL diagnostics:
train/val loss, test accuracy, early-event gradient norm, hidden-state
retention (ridge probe R^2), a recurrent-Jacobian contraction proxy, and
training-stability flags.  Raw per-run records go to
results/EXP-001/raw_metrics.jsonl; aggregated summaries go to
results/EXP-001/summary.csv and summary.json.  Failures are recorded with
status="failure" rather than retried.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditlab.diagnostics import (  # noqa: E402
    event_gradient_norm,
    jacobian_contraction_proxy,
    ridge_probe_retention,
)
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.delayed_task import make_split, sequence_length  # noqa: E402
from creditlab.models import TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def evaluate(model, inputs, targets, batch_size):
    """Terminal-step BCE loss and accuracy over the full tensor in chunks."""
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            chunk_x = inputs[start : start + batch_size]
            chunk_y = targets[start : start + batch_size]
            logits, _ = model(chunk_x)
            final = logits[:, -1, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                final, (chunk_y + 1) / 2, reduction="sum"
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite evaluation loss")
            total_loss += float(loss)
            preds = torch.sign(final)
            preds = torch.where(preds == 0, torch.ones_like(preds), preds)
            correct += int((preds == chunk_y).sum())
            count += chunk_y.numel()
    return total_loss / count, correct / count


def train_one_run(cfg: dict, delay: int, mode: str, seed: int) -> dict:
    """Train and measure one (delay, mode, seed) cell.  Returns a raw record."""
    started = time.time()
    record = {
        "experiment_id": cfg["experiment_id"],
        "config_revision": cfg["config_revision"],
        "code_revision": git_revision(),
        "delay": delay,
        "mode": mode,
        "seed": seed,
        "sequence_length": sequence_length(delay),
        "device": "cpu",
        "status": "ok",
        "stop_reason": None,
    }
    tcfg, mcfg, dcfg = cfg["task"], cfg["model"], cfg["data"]
    trcfg = cfg["training"]
    task_cfg = {
        "easy_noise_std": tcfg["modes"]["EASY"]["distractor_noise_std"],
        "hard_noise_std": tcfg["modes"]["HARD"]["distractor_noise_std"],
        "competitor_rate": tcfg["modes"]["HARD"]["competitor_rate"],
        "competitor_flip_prob": tcfg["modes"]["HARD"]["competitor_flip_prob"],
    }

    # Data first, from explicit generators (independent of global RNG order).
    train_x, train_y = make_split(dcfg["train_samples"], delay, mode,
                                  run_seed=seed, split_index=0,
                                  base_seed=dcfg["base_seed"], config=task_cfg)
    val_x, val_y = make_split(dcfg["val_samples"], delay, mode,
                              run_seed=seed, split_index=1,
                              base_seed=dcfg["base_seed"], config=task_cfg)
    test_x, test_y = make_split(dcfg["test_samples"], delay, mode,
                                run_seed=seed, split_index=2,
                                base_seed=dcfg["base_seed"], config=task_cfg)

    # Model init uses the global CPU RNG (seeded deterministically).
    seed_everything(seed)
    model = TinySequenceRNN(mcfg["input_size"], mcfg["hidden_size"], mcfg["output_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=trcfg["learning_rate"])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    batch_size = trcfg["batch_size"]
    n = train_x.shape[0]
    rng = random.Random(seed * 7919 + delay)

    best_val = math.inf
    clipped = 0
    loss_curve = []
    try:
        for update in range(trcfg["updates_per_run"]):
            idx = [rng.randrange(n) for _ in range(batch_size)]
            xb, yb = train_x[idx], train_y[idx]
            logits, _ = model(xb)
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), trcfg["grad_clip_norm"]
            )
            if not torch.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm at update {update}")
            if float(gnorm) > trcfg["grad_clip_norm"]:
                clipped += 1
            optimizer.step()
            loss_curve.append(float(loss))
            if (update + 1) % 50 == 0 or update == trcfg["updates_per_run"] - 1:
                vloss, vacc = evaluate(model, val_x, val_y, trcfg["eval_batch_size"])
                if vloss < best_val:
                    best_val = vloss
        train_loss, train_acc = evaluate(model, train_x, train_y, trcfg["eval_batch_size"])
        val_loss, val_acc = evaluate(model, val_x, val_y, trcfg["eval_batch_size"])
        test_loss, test_acc = evaluate(model, test_x, test_y, trcfg["eval_batch_size"])
        diag_grad = event_gradient_norm(model, test_x, test_y)
        # Probe fit on val states, scored on test states (held-out R^2).
        retention_r2 = probe_fit_score(model, val_x, val_y, test_x, test_y)
        contraction = jacobian_contraction_proxy(model, test_x[:64])
        record.update(
            train_loss=round(train_loss, 6),
            train_loss_final_window=round(
                sum(loss_curve[-20:]) / len(loss_curve[-20:]), 6
            ),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 6),
            val_accuracy=round(val_acc, 4),
            best_val_loss=round(best_val, 6),
            test_loss=round(test_loss, 6),
            test_accuracy=round(test_acc, 4),
            event_gradient_norm=round(diag_grad, 6),
            retention_probe_r2=round(retention_r2, 6),
            jacobian_spectral_norm=round(contraction, 6),
            grad_clip_fraction=round(clipped / trcfg["updates_per_run"], 4),
            all_finite=True,
            runtime_seconds=round(time.time() - started, 2),
            loss_curve_first_last=[
                round(loss_curve[0], 4), round(loss_curve[-1], 4)
            ],
        )
    except (FloatingPointError, RuntimeError) as exc:
        record["status"] = "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["all_finite"] = False
        record["runtime_seconds"] = round(time.time() - started, 2)
        record["train_loss_curve_len"] = len(loss_curve)
    return record


def probe_fit_score(model, fit_x, fit_y, score_x, score_y):
    """Fit ridge probe on final hidden states of (fit_x), report R^2 on (score_x)."""
    from creditlab.diagnostics import hidden_states

    with torch.no_grad():
        s_fit = hidden_states(model, fit_x)[:, -1, :]
        s_score = hidden_states(model, score_x)[:, -1, :]
    mu_s, mu_b = s_fit.mean(0, keepdim=True), fit_y.mean()
    sc = s_fit - mu_s
    bc = (fit_y - mu_b).unsqueeze(1)
    cov = sc.T @ sc
    reg = torch.eye(cov.shape[0]) * (cov.diagonal().mean().clamp_min(1e-8))
    w = torch.linalg.solve(cov + reg, sc.T @ bc)
    pred = (s_score - mu_s) @ w
    resid = pred - (score_y - mu_b).unsqueeze(1)
    sse = float((resid**2).sum())
    sst = float(((score_y - mu_b) ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-12)
    if not math.isfinite(r2):
        raise FloatingPointError("probe R^2 non-finite")
    return r2


def aggregate(records: list[dict], cfg: dict) -> dict:
    """Mean/std across seeds per (delay, mode); only successful runs feed means."""
    cells = {}
    for rec in records:
        key = (rec["delay"], rec["mode"])
        cells.setdefault(key, []).append(rec)
    summary_rows = []
    metric_names = [
        "train_loss", "val_loss", "test_accuracy", "event_gradient_norm",
        "retention_probe_r2", "jacobian_spectral_norm", "grad_clip_fraction",
        "runtime_seconds",
    ]
    for (delay, mode), recs in sorted(cells.items()):
        ok = [r for r in recs if r["status"] == "ok"]
        row = {"delay": delay, "mode": mode, "num_seeds": len(recs),
               "num_ok": len(ok), "num_failed": len(recs) - len(ok)}
        for name in metric_names:
            vals = [r[name] for r in ok if name in r]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0
                row[f"{name}_mean"] = round(mean, 6)
                row[f"{name}_std"] = round(math.sqrt(var), 6)
            else:
                row[f"{name}_mean"] = None
                row[f"{name}_std"] = None
        summary_rows.append(row)
    return {"cells": summary_rows, "seeds": cfg["seeds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp001_delayed_credit_v2.yaml")
    parser.add_argument("--extra-delay", type=int, default=None,
                        help="optional extra delay (e.g. 256) appended to the sweep")
    parser.add_argument("--quick", action="store_true",
                        help="pilot mode: fewer updates/samples, delays up to 32")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out_dir = ROOT / cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        cfg["training"]["updates_per_run"] = 120
        cfg["data"]["train_samples"] = 384
        cfg["data"]["val_samples"] = 128
        cfg["data"]["test_samples"] = 128
        cfg["task"]["delays"] = [d for d in cfg["task"]["delays"] if d <= 32]
        out_dir = out_dir / "pilot_quick"
        out_dir.mkdir(parents=True, exist_ok=True)

    delays = list(cfg["task"]["delays"])
    if args.extra_delay:
        delays.append(args.extra_delay)
    modes = list(cfg["task"]["modes"].keys())

    raw_path = out_dir / cfg["outputs"]["raw_metrics"]
    if raw_path.exists():
        raw_path.unlink()  # fresh file per invocation; history kept in RESULTS.md

    records = []
    wall_start = time.time()
    for delay in delays:
        for mode in modes:
            for seed in cfg["seeds"]:
                rec = train_one_run(cfg, delay, mode, seed)
                records.append(rec)
                append_jsonl(raw_path, rec)
                print(json.dumps({k: rec[k] for k in
                                  ("delay", "mode", "seed", "status",
                                   "test_accuracy", "event_gradient_norm",
                                   "retention_probe_r2", "jacobian_spectral_norm",
                                   "runtime_seconds")}, sort_keys=True), flush=True)
                if rec["status"] != "ok":
                    print(f"  recorded failure for delay={delay} mode={mode} "
                          f"seed={seed}: {rec['stop_reason']}", flush=True)

    elapsed = time.time() - wall_start
    agg = aggregate(records, cfg)
    agg["total_wall_seconds"] = round(elapsed, 1)
    agg["config_file"] = args.config
    agg["config_revision"] = cfg["config_revision"]
    agg["updates_per_run"] = cfg["training"]["updates_per_run"]

    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2, sort_keys=True))
    rows = agg["cells"]
    flat = sorted({k for row in rows for k in row})
    with (out_dir / cfg["outputs"]["summary_metrics"]).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat)
        writer.writeheader()
        writer.writerows(rows)
    print(f"EXP-001 sweep finished in {elapsed:.1f}s wall time; "
          f"{len(records)} runs, {sum(r['status'] == 'failure' for r in records)} failures.")


if __name__ == "__main__":
    main()

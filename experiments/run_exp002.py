"""EXP-002: one fixed residual skip versus the vanilla tanh RNN.

The comparison uses corrected HARD-v2, seeds 17, 29, and 43, and the EXP-001B
training settings. Delay 64 is the primary comparison. Delay 128 runs only
after every delay-64 record has been written.

Usage (from the repository root):
    python experiments/run_exp002.py
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.delayed_task import make_split, sequence_length  # noqa: E402
from creditlab.diagnostics import (  # noqa: E402
    competitor_conditioned_accuracy,
    event_gradient_norm,
    event_loss_gradient_metrics,
    hidden_magnitude_stats,
    jacobian_contraction_proxy,
    residual_candidate_mean_abs,
    step_jacobian_stats,
)
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.models import ResidualTanhRNN, TinySequenceRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate, probe_fit_score  # noqa: E402

MODEL_TYPES = {
    "vanilla_tanh_rnn": TinySequenceRNN,
    "residual_tanh_rnn": ResidualTanhRNN,
}


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def train_one_run(cfg: dict, delay: int, model_type: str, seed: int, phase: str) -> dict:
    """Train one matched (delay, model, seed) cell on HARD-v2."""
    started = time.time()
    mode = cfg["task"]["run_mode"]
    scale = float(cfg["residual"]["residual_scale"])
    record = {
        "experiment_id": cfg["experiment_id"],
        "config_revision": cfg["config_revision"],
        "code_revision": git_revision(),
        "phase": phase,
        "delay": delay,
        "mode": mode,
        "model_type": model_type,
        "residual_scale": scale if model_type == "residual_tanh_rnn" else None,
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
    train_x, train_y = make_split(
        dcfg["train_samples"], delay, mode, run_seed=seed, split_index=0,
        base_seed=dcfg["base_seed"], config=task_cfg,
    )
    val_x, val_y = make_split(
        dcfg["val_samples"], delay, mode, run_seed=seed, split_index=1,
        base_seed=dcfg["base_seed"], config=task_cfg,
    )
    test_x, test_y, test_meta = make_split(
        dcfg["test_samples"], delay, mode, run_seed=seed, split_index=2,
        base_seed=dcfg["base_seed"], config=task_cfg, return_metadata=True,
    )
    if test_x.shape[1] > 2:
        record["distractor_query_channel_std"] = round(float(test_x[:, 1:-1, 2].std()), 4)

    seed_everything(seed)
    model_cls = MODEL_TYPES[model_type]
    if model_cls is ResidualTanhRNN:
        model = model_cls(
            mcfg["input_size"], mcfg["hidden_size"], mcfg["output_size"],
            residual_scale=scale,
        )
    else:
        model = model_cls(mcfg["input_size"], mcfg["hidden_size"], mcfg["output_size"])
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
            loss_curve.append(float(loss.detach()))
            if (update + 1) % 50 == 0 or update == trcfg["updates_per_run"] - 1:
                vloss, _vacc = evaluate(model, val_x, val_y, trcfg["eval_batch_size"])
                if vloss < best_val:
                    best_val = vloss
        train_loss, train_acc = evaluate(model, train_x, train_y, trcfg["eval_batch_size"])
        val_loss, val_acc = evaluate(model, val_x, val_y, trcfg["eval_batch_size"])
        test_loss, test_acc = evaluate(model, test_x, test_y, trcfg["eval_batch_size"])
        diag_grad = event_gradient_norm(model, test_x, test_y)
        loss_grad = event_loss_gradient_metrics(model, test_x, test_y)
        final_logits, _ = model(test_x)
        predictions = torch.where(final_logits[:, -1, 0] >= 0, 1.0, -1.0)
        competitor_stats = competitor_conditioned_accuracy(
            predictions, test_y, test_meta["competitor_mask"], test_meta["competitor_bits"],
        )
        retention_r2 = probe_fit_score(model, val_x, val_y, test_x, test_y)
        legacy_jacobian = (
            jacobian_contraction_proxy(model, test_x[:64])
            if model_type == "vanilla_tanh_rnn" else None
        )
        step_stats = step_jacobian_stats(model, test_x[:64])
        magnitude = hidden_magnitude_stats(model, test_x)
        candidate_abs = (
            residual_candidate_mean_abs(model, test_x)
            if model_type == "residual_tanh_rnn" else None
        )
        record.update(
            train_loss=round(train_loss, 6),
            train_loss_final_window=round(sum(loss_curve[-20:]) / len(loss_curve[-20:]), 6),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 6),
            val_accuracy=round(val_acc, 4),
            best_val_loss=round(best_val, 6),
            test_loss=round(test_loss, 6),
            test_accuracy=round(test_acc, 4),
            event_gradient_norm=round(diag_grad, 6),
            event_loss_gradient_norm_mean=round(loss_grad["mean"], 8),
            event_loss_gradient_norm_median=round(loss_grad["median"], 8),
            event_loss_gradient_norm_p95=round(loss_grad["p95"], 8),
            **competitor_stats,
            retention_probe_r2=round(retention_r2, 6),
            jacobian_spectral_norm=None if legacy_jacobian is None else round(legacy_jacobian, 6),
            step_jacobian_spectral_norm=round(step_stats["spectral_norm"], 6),
            step_jacobian_min_singular=round(step_stats["min_singular"], 6),
            final_hidden_rms=round(magnitude["final_hidden_rms"], 6),
            max_abs_hidden=round(magnitude["max_abs_hidden"], 6),
            mean_abs_candidate=None if candidate_abs is None else round(candidate_abs, 6),
            grad_clip_fraction=round(clipped / trcfg["updates_per_run"], 4),
            all_finite=True,
            runtime_seconds=round(time.time() - started, 2),
            loss_curve_first_last=[round(loss_curve[0], 4), round(loss_curve[-1], 4)],
            parameter_count=sum(p.numel() for p in model.parameters()),
        )
    except (FloatingPointError, RuntimeError) as exc:
        record["status"] = "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["all_finite"] = False
        record["runtime_seconds"] = round(time.time() - started, 2)
        record["train_loss_curve_len"] = len(loss_curve)
    return record


def aggregate(records: list[dict], cfg: dict) -> dict:
    cells = {}
    for rec in records:
        key = (rec["phase"], rec["delay"], rec["model_type"])
        cells.setdefault(key, []).append(rec)
    cutoff = float(cfg["interpretation"]["success_test_accuracy_at_least"])
    retention_cutoff = float(cfg["interpretation"]["retention_clearly_readable_r2_at_least"])
    metric_names = [
        "train_loss", "val_loss", "test_loss", "test_accuracy",
        "event_gradient_norm", "event_loss_gradient_norm_mean",
        "event_loss_gradient_norm_median", "event_loss_gradient_norm_p95",
        "retention_probe_r2", "jacobian_spectral_norm",
        "step_jacobian_spectral_norm", "step_jacobian_min_singular",
        "final_hidden_rms", "max_abs_hidden", "mean_abs_candidate",
        "grad_clip_fraction", "runtime_seconds",
        "competitor_first_bit_agreement_rate",
    ]
    rows = []
    for (phase, delay, model_type), recs in sorted(cells.items()):
        ok = [r for r in recs if r["status"] == "ok"]
        row = {
            "phase": phase,
            "delay": delay,
            "model_type": model_type,
            "num_seeds": len(recs),
            "num_ok": len(ok),
            "num_failed": len(recs) - len(ok),
            "num_success_at_cutoff": sum(
                1 for r in ok if r.get("test_accuracy", 0.0) >= cutoff
            ),
            "num_retention_clear": sum(
                1 for r in ok if r.get("retention_probe_r2", 0.0) >= retention_cutoff
            ),
        }
        for seed in cfg["seeds"]:
            match = [r for r in ok if r["seed"] == seed]
            row[f"test_accuracy_seed_{seed}"] = match[0]["test_accuracy"] if match else None
            row[f"retention_r2_seed_{seed}"] = match[0]["retention_probe_r2"] if match else None
        for name in metric_names:
            vals = [r[name] for r in ok if r.get(name) is not None]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0
                row[f"{name}_mean"] = round(mean, 6)
                row[f"{name}_std"] = round(math.sqrt(var), 6)
            else:
                row[f"{name}_mean"] = None
                row[f"{name}_std"] = None
        rows.append(row)
    return {
        "cells": rows,
        "seeds": cfg["seeds"],
        "success_test_accuracy_at_least": cutoff,
        "retention_clearly_readable_r2_at_least": retention_cutoff,
        "residual_scale": cfg["residual"]["residual_scale"],
    }


def write_summary(out_dir: Path, summary_name: str, agg: dict, cfg: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2, sort_keys=True))
    rows = agg["cells"]
    flat = sorted({k for row in rows for k in row})
    with (out_dir / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat)
        writer.writeheader()
        writer.writerows(rows)


def write_environment(out_dir: Path, revision: str) -> None:
    lines = [
        f"Python: {platform.python_version()}",
        f"PyTorch: {torch.__version__}",
        f"CPU architecture: {platform.machine()}",
        f"Recorded training code revision: {revision}",
        "All runs used CPU.",
        "Residual scale 1.0 was fixed before these runs and was not tuned.",
    ]
    (out_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp002_residual_recurrent.yaml").read_text())
    out_dir = ROOT / cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(
        (ROOT / "configs" / "exp002_residual_recurrent.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    raw_path = out_dir / cfg["outputs"]["raw_metrics"]
    if raw_path.exists():
        raw_path.unlink()
    revision = git_revision()
    write_environment(out_dir, revision)
    records = []
    wall_start = time.time()
    phases = [
        ("primary", int(cfg["task"]["primary_delay"])),
        ("confirmation", int(cfg["task"]["confirmation_delay"])),
    ]
    for phase, delay in phases:
        for model_type in cfg["models"]:
            for seed in cfg["seeds"]:
                rec = train_one_run(cfg, delay, model_type, seed, phase)
                records.append(rec)
                append_jsonl(raw_path, rec)
                print(json.dumps({
                    k: rec.get(k) for k in (
                        "phase", "delay", "model_type", "seed", "status",
                        "test_accuracy", "retention_probe_r2",
                        "event_gradient_norm", "event_loss_gradient_norm_mean",
                        "step_jacobian_min_singular", "final_hidden_rms",
                        "max_abs_hidden", "runtime_seconds", "stop_reason",
                    )
                }, sort_keys=True), flush=True)
        if phase == "primary":
            partial = aggregate(records, cfg)
            partial["note"] = "primary delay only; confirmation delay has not run"
            (out_dir / "primary_delay64_summary.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True)
            )
    elapsed = time.time() - wall_start
    agg = aggregate(records, cfg)
    agg["total_wall_seconds"] = round(elapsed, 1)
    agg["config_revision"] = cfg["config_revision"]
    agg["code_revision"] = revision
    agg["updates_per_run"] = cfg["training"]["updates_per_run"]
    write_summary(out_dir, cfg["outputs"]["summary_metrics"], agg, cfg)
    failures = sum(r["status"] == "failure" for r in records)
    print(
        f"EXP-002 finished in {elapsed:.1f}s wall time; "
        f"{len(records)} runs, {failures} failures."
    )


if __name__ == "__main__":
    main()

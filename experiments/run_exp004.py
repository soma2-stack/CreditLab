"""EXP-004: additive RNN versus the same update clamped at a fixed bound.

Delay 64 is fully saved before any delay-128 training. Delay 128 runs only
when the fresh additive control has at least two successes at delay 64.

Usage (from the repository root):
    python experiments/run_exp004.py --write-frozen-hashes
    python experiments/run_exp004.py
"""

from __future__ import annotations

import csv
import hashlib
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

from creditlab.delayed_task import generate_batch, make_split  # noqa: E402
from creditlab.diagnostics import (  # noqa: E402
    event_gradient_norm,
    event_loss_gradient_metrics,
    hidden_magnitude_stats,
)
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.models import (  # noqa: E402
    BoundedAdditiveTanhRNN,
    ResidualTanhRNN,
    TinySequenceRNN,
)
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate, probe_fit_score  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp004_bounded_additive.yaml"
MODEL_TYPES = (
    "vanilla_tanh_rnn",
    "additive_tanh_rnn",
    "bounded_additive_tanh_rnn",
)
FROZEN_PATHS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "configs/exp001_delayed_credit.yaml",
    "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml",
    "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml",
)
DIAGNOSTIC_UPDATES = (0, 50, 100, 200, 400)


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def working_tree_dirty() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def combined_seed(run_seed: int, delay: int, split_index: int, base_seed: int) -> int:
    return (
        int(run_seed) * 1_000_003 + int(base_seed) * 9_973 + delay * 31 + split_index
    ) % (2**31 - 1)


def load_hard_v2(
    num_samples: int,
    delay: int,
    run_seed: int,
    split_index: int,
    *,
    noise_std: float,
    competitor_rate: float,
    competitor_flip_prob: float,
    base_seed: int,
    return_metadata: bool = False,
):
    """Explicit HARD-v2 draw. Noise and competitor rate come from the caller."""
    return generate_batch(
        num_samples,
        delay,
        "HARD_V2",
        seed=combined_seed(run_seed, delay, split_index, base_seed),
        distractor_noise_std=noise_std,
        competitor_rate=competitor_rate,
        competitor_flip_prob=competitor_flip_prob,
        return_metadata=return_metadata,
    )


def historical_hard_v2(num_samples: int, delay: int, run_seed: int, split_index: int, **kwargs):
    """The EXP-002/003 make_split path, used only to verify the explicit route."""
    task_cfg = {
        "easy_noise_std": 0.3,
        "hard_noise_std": 1.0,
        "competitor_rate": 0.4,
        "competitor_flip_prob": 0.5,
    }
    return make_split(
        num_samples, delay, "HARD_V2", run_seed=run_seed, split_index=split_index,
        base_seed=1000, config=task_cfg, **kwargs,
    )


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parameter_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def index_fingerprint(indices: list[list[int]]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_manifest() -> dict[str, str]:
    records = {}
    for relative in FROZEN_PATHS:
        target = ROOT / relative
        if target.is_file():
            records[relative.replace("\\", "/")] = file_sha256(target)
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file():
                key = path.relative_to(ROOT).as_posix()
                records[key] = file_sha256(path)
    return records


def write_frozen_hashes(destination: Path) -> dict[str, str]:
    manifest = frozen_manifest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def hashes_match(saved: Path) -> tuple[bool, str]:
    if not saved.exists():
        return False, "frozen hash manifest is missing"
    expected = json.loads(saved.read_text(encoding="utf-8"))
    current = frozen_manifest()
    if current != expected:
        missing = sorted(set(expected) ^ set(current))[:8]
        return False, f"frozen artifact hashes changed; sample keys {missing}"
    return True, "frozen artifact hashes match"


def build_model(model_type: str, cfg: dict) -> torch.nn.Module:
    size = cfg["model"]
    common = (size["input_size"], size["hidden_size"], size["output_size"])
    if model_type == "vanilla_tanh_rnn":
        return TinySequenceRNN(*common)
    if model_type == "additive_tanh_rnn":
        return ResidualTanhRNN(*common, residual_scale=1.0)
    if model_type == "bounded_additive_tanh_rnn":
        return BoundedAdditiveTanhRNN(*common, bound=float(cfg["bound"]["B"]))
    raise ValueError(f"unknown model type {model_type}")


def build_matched_models(cfg: dict, seed: int) -> dict[str, torch.nn.Module]:
    models = {}
    for model_type in MODEL_TYPES:
        seed_everything(seed)
        models[model_type] = build_model(model_type, cfg)
    fingerprints = {name: parameter_fingerprint(model) for name, model in models.items()}
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError(f"initial parameters do not match: {fingerprints}")
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"parameter counts do not match: {counts}")
    return models


def minibatch_indices(seed: int, delay: int, n_examples: int, batch_size: int, updates: int) -> list[list[int]]:
    rng = random.Random(seed * 7919 + delay)
    return [[rng.randrange(n_examples) for _ in range(batch_size)] for _ in range(updates)]


def dataset_bundle(cfg: dict, delay: int, seed: int) -> dict:
    task = cfg["task"]
    data = cfg["data"]
    noise = float(task["distractor_noise_std"])
    rate = float(task["competitor_rate"])
    flip = float(task["competitor_flip_prob"])
    base = int(data["base_seed"])
    kwargs = dict(
        noise_std=noise, competitor_rate=rate, competitor_flip_prob=flip, base_seed=base,
    )
    train_x, train_y = load_hard_v2(data["train_samples"], delay, seed, 0, **kwargs)
    val_x, val_y = load_hard_v2(data["val_samples"], delay, seed, 1, **kwargs)
    test_x, test_y, test_meta = load_hard_v2(
        data["test_samples"], delay, seed, 2, return_metadata=True, **kwargs,
    )
    return {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "test_x": test_x, "test_y": test_y, "test_meta": test_meta,
    }


def dataset_fingerprints(bundle: dict) -> dict[str, str]:
    return {
        "train_x": tensor_fingerprint(bundle["train_x"]),
        "train_y": tensor_fingerprint(bundle["train_y"]),
        "val_x": tensor_fingerprint(bundle["val_x"]),
        "val_y": tensor_fingerprint(bundle["val_y"]),
        "test_x": tensor_fingerprint(bundle["test_x"]),
        "test_y": tensor_fingerprint(bundle["test_y"]),
        "competitor_mask": tensor_fingerprint(bundle["test_meta"]["competitor_mask"]),
        "competitor_bits": tensor_fingerprint(bundle["test_meta"]["competitor_bits"]),
    }


def measured_noise_std(inputs: torch.Tensor) -> float:
    if inputs.shape[1] <= 2:
        return float("nan")
    return float(inputs[:, 1:-1, 2].std())


def clipping_rollout(model: BoundedAdditiveTanhRNN, inputs: torch.Tensor) -> dict:
    """Clipping statistics from proposed states before the clamp."""
    batch = inputs.shape[0]
    h = model._initial_state(batch, None, inputs.device, inputs.dtype)
    outside_steps = []
    proposed_sq = inputs.new_zeros(())
    proposed_max = 0.0
    count = 0
    any_example = torch.zeros(batch, dtype=torch.bool)
    actual_steps = []
    for t in range(inputs.shape[1]):
        state, _candidate, proposed = model.step_parts(inputs[:, t, :], h)
        outside = proposed.detach().abs() > model.bound
        outside_steps.append(outside)
        any_example |= outside.flatten(1).any(dim=1)
        proposed_sq = proposed_sq + (proposed.detach() ** 2).sum()
        proposed_max = max(proposed_max, float(proposed.detach().abs().max()))
        count += proposed.numel()
        actual_steps.append(state.detach())
        h = state.detach()
    outside_all = torch.stack(outside_steps, dim=1)
    by_time = [float(value) for value in outside_all.float().mean(dim=(0, 2)).tolist()]
    actual = torch.stack(actual_steps, dim=1)
    fraction = float(outside_all.float().mean())
    return {
        "clipping_fraction": fraction,
        "clipping_fraction_by_timestep": by_time,
        "example_any_clip_fraction": float(any_example.float().mean()),
        "proposed_max_abs": proposed_max,
        "proposed_rms": float((proposed_sq / count).sqrt()),
        "actual_max_abs": float(actual.abs().max()),
        "actual_rms": float(actual.pow(2).mean().sqrt()),
    }


def checkpoint_record(
    model, model_type: str, val_x, val_y, update: int, cfg: dict,
) -> dict:
    half = val_x.shape[0] // 2
    retention = probe_fit_score(model, val_x[:half], val_y[:half], val_x[half:], val_y[half:])
    sensitivity = event_gradient_norm(model, val_x, val_y)
    loss_grad = event_loss_gradient_metrics(model, val_x, val_y)
    magnitude = hidden_magnitude_stats(model, val_x)
    val_loss, val_acc = evaluate(model, val_x, val_y, cfg["training"]["eval_batch_size"])
    record = {
        "update": update,
        "split": "validation",
        "retention_probe_r2": retention,
        "event_gradient_norm": sensitivity,
        "event_loss_gradient_norm_mean": loss_grad["mean"],
        "event_loss_gradient_norm_median": loss_grad["median"],
        "event_loss_gradient_norm_p95": loss_grad["p95"],
        "max_abs_hidden": magnitude["max_abs_hidden"],
        "final_hidden_rms": magnitude["final_hidden_rms"],
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    if model_type == "bounded_additive_tanh_rnn":
        record["clipping"] = clipping_rollout(model, val_x)
    return record


def json_safe(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("non-finite diagnostic value")
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def train_one(
    cfg: dict,
    model: torch.nn.Module,
    model_type: str,
    bundle: dict,
    indices: list[list[int]],
    delay: int,
    seed: int,
    phase: str,
    init_fingerprint: str,
    data_fingerprints: dict[str, str],
    index_hash: str,
) -> tuple[dict, list[float], list[dict], dict | None]:
    started = time.perf_counter()
    budget = float(cfg["budget"]["max_cpu_seconds_per_run"])
    updates_target = int(cfg["training"]["updates_per_run"])
    record = {
        "experiment_id": cfg["experiment_id"],
        "config_revision": cfg["config_revision"],
        "code_revision": git_revision()[:12],
        "phase": phase,
        "delay": delay,
        "mode": "HARD_V2",
        "model_type": model_type,
        "bound": float(cfg["bound"]["B"]) if model_type == "bounded_additive_tanh_rnn" else None,
        "seed": seed,
        "device": "cpu",
        "status": "ok",
        "stop_reason": None,
        "init_fingerprint": init_fingerprint,
        "dataset_fingerprints": data_fingerprints,
        "minibatch_fingerprint": index_hash,
        "distractor_query_channel_std": round(measured_noise_std(bundle["test_x"]), 6),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    tr = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss_curve: list[float] = []
    checkpoints: list[dict] = []
    clipped = 0
    try:
        checkpoints.append(json_safe(checkpoint_record(
            model, model_type, bundle["val_x"], bundle["val_y"], 0, cfg,
        )))
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > budget:
                raise TimeoutError(f"CPU budget of {budget:.0f}s exhausted at update {update}")
            xb = bundle["train_x"][idx]
            yb = bundle["train_y"][idx]
            logits, _ = model(xb)
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at update {update}")
            if float(grad_norm) > float(tr["grad_clip_norm"]):
                clipped += 1
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in DIAGNOSTIC_UPDATES:
                checkpoints.append(json_safe(checkpoint_record(
                    model, model_type, bundle["val_x"], bundle["val_y"], update + 1, cfg,
                )))
        train_loss, train_acc = evaluate(
            model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"],
        )
        val_loss, val_acc = evaluate(
            model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"],
        )
        test_loss, test_acc = evaluate(
            model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"],
        )
        sensitivity = event_gradient_norm(model, bundle["test_x"], bundle["test_y"])
        loss_grad = event_loss_gradient_metrics(model, bundle["test_x"], bundle["test_y"])
        retention = probe_fit_score(
            model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"],
        )
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        final_clip = None
        if model_type == "bounded_additive_tanh_rnn":
            final_clip = json_safe(clipping_rollout(model, bundle["test_x"]))
        success_cutoff = float(cfg["interpretation"]["success_test_accuracy_at_least"])
        record.update(
            updates_completed=updates_target,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
            success=bool(test_acc >= success_cutoff),
            retention_probe_r2=retention,
            event_gradient_norm=sensitivity,
            event_loss_gradient_norm_mean=loss_grad["mean"],
            event_loss_gradient_norm_median=loss_grad["median"],
            event_loss_gradient_norm_p95=loss_grad["p95"],
            max_abs_hidden=magnitude["max_abs_hidden"],
            final_hidden_rms=magnitude["final_hidden_rms"],
            grad_clip_fraction=clipped / updates_target,
            all_finite=True,
            runtime_seconds=round(time.perf_counter() - started, 3),
            final_test_clipping=final_clip,
        )
    except (FloatingPointError, RuntimeError, TimeoutError) as exc:
        record["status"] = "budget_stop" if isinstance(exc, TimeoutError) else "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["all_finite"] = False if not isinstance(exc, TimeoutError) else record.get("all_finite", True)
        record["updates_completed"] = len(loss_curve)
        record["success"] = False
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
        record["grad_clip_fraction"] = clipped / max(len(loss_curve), 1)
    return record, loss_curve, checkpoints, record.get("final_test_clipping")


def assert_outputs_available(out_dir: Path, raw_name: str) -> None:
    raw_path = out_dir / raw_name
    if raw_path.exists():
        raise FileExistsError(
            f"refusing to overwrite completed or partial output {raw_path}"
        )


def environment_lines(revision: str) -> list[str]:
    return [
        f"Python: {platform.python_version()}",
        f"PyTorch: {torch.__version__}",
        f"CPU architecture: {platform.machine()}",
        f"Training code revision: {revision}",
        "Device: cpu",
        "Bound B=4.0 was fixed before training and was not tuned.",
        "All three models used this same process.",
    ]


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
        "phase", "delay", "seed", "model_type", "status", "success",
        "test_accuracy", "test_loss", "val_loss", "retention_probe_r2",
        "event_gradient_norm", "event_loss_gradient_norm_mean",
        "max_abs_hidden", "final_hidden_rms", "updates_completed",
        "runtime_seconds", "stop_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict], cfg: dict) -> dict:
    cells = {}
    for rec in records:
        cells.setdefault((rec["phase"], rec["delay"], rec["model_type"]), []).append(rec)
    rows = []
    for key, group in sorted(cells.items()):
        phase, delay, model_type = key
        successes = sum(1 for rec in group if rec.get("success"))
        failures = sum(1 for rec in group if rec["status"] != "ok")
        rows.append({
            "phase": phase,
            "delay": delay,
            "model_type": model_type,
            "num_seeds": len(group),
            "num_success": successes,
            "num_not_success": len(group) - successes,
            "num_numerical_or_budget_failures": failures,
        })
    return {
        "cells": rows,
        "success_test_accuracy_at_least": cfg["interpretation"]["success_test_accuracy_at_least"],
        "bound": cfg["bound"]["B"],
        "seeds": cfg["seeds"],
    }


def additive_successes(records: list[dict], delay: int) -> int:
    return sum(
        1 for rec in records
        if rec["delay"] == delay and rec["model_type"] == "additive_tanh_rnn" and rec.get("success")
    )


def run_delay(cfg, out_dir, delay, phase, records, curves, checkpoints, clipping_rows, fingerprint_rows):
    print(f"starting delay {delay}", flush=True)
    for seed in cfg["seeds"]:
        bundle = dataset_bundle(cfg, delay, seed)
        data_fp = dataset_fingerprints(bundle)
        indices = minibatch_indices(
            seed, delay, bundle["train_x"].shape[0],
            cfg["training"]["batch_size"], cfg["training"]["updates_per_run"],
        )
        index_hash = index_fingerprint(indices)
        models = build_matched_models(cfg, seed)
        init_hash = parameter_fingerprint(models["vanilla_tanh_rnn"])
        fingerprint_rows.append({
            "delay": delay,
            "seed": seed,
            "init_fingerprint": init_hash,
            "minibatch_fingerprint": index_hash,
            "dataset_fingerprints": data_fp,
            "models_matched": True,
        })
        for model_type in MODEL_TYPES:
            record, curve, checks, _final_clip = train_one(
                cfg, models[model_type], model_type, bundle, indices,
                delay, seed, phase, init_hash, data_fp, index_hash,
            )
            records.append(record)
            append_jsonl(out_dir / cfg["outputs"]["raw_metrics"], record)
            curves.append({
                "delay": delay, "seed": seed, "model_type": model_type,
                "loss_curve": curve,
            })
            for item in checks:
                checkpoints.append({
                    "delay": delay, "seed": seed, "model_type": model_type, **item,
                })
                if "clipping" in item:
                    clipping_rows.append({
                        "delay": delay, "seed": seed, "model_type": model_type,
                        "update": item["update"], "split": item["split"],
                        **item["clipping"],
                    })
            if record.get("final_test_clipping"):
                clipping_rows.append({
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "update": "final_test", "split": "test",
                    **record["final_test_clipping"],
                })
            print(json.dumps({
                "delay": delay, "seed": seed, "model_type": model_type,
                "status": record["status"], "success": record.get("success"),
                "test_accuracy": record.get("test_accuracy"),
                "retention_probe_r2": record.get("retention_probe_r2"),
                "max_abs_hidden": record.get("max_abs_hidden"),
                "runtime_seconds": record.get("runtime_seconds"),
                "stop_reason": record.get("stop_reason"),
            }, sort_keys=True), flush=True)
            append_jsonl(out_dir / "loss_curves.jsonl", {
                "delay": delay, "seed": seed, "model_type": model_type, "loss_curve": curve,
            })
            for item in checks:
                append_jsonl(out_dir / "checkpoint_diagnostics.jsonl", {
                    "delay": delay, "seed": seed, "model_type": model_type, **item,
                })
            for item in checks:
                if "clipping" in item:
                    append_jsonl(out_dir / "clipping_records.jsonl", {
                        "delay": delay, "seed": seed, "model_type": model_type,
                        "update": item["update"], "split": item["split"],
                        **item["clipping"],
                    })
            if record.get("final_test_clipping"):
                append_jsonl(out_dir / "clipping_records.jsonl", {
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "update": "final_test", "split": "test",
                    **record["final_test_clipping"],
                })


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        destination = ROOT / "results" / "EXP-004" / "frozen_hashes_before.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        write_frozen_hashes(destination)
        print(f"wrote {destination}")
        return

    cfg = load_config()
    out_dir = ROOT / cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_outputs_available(out_dir, cfg["outputs"]["raw_metrics"])
    dirty = working_tree_dirty()
    if dirty:
        raise RuntimeError(f"working tree is dirty; resolve it before training:\n{dirty}")
    hash_path = out_dir / "frozen_hashes_before.json"
    ok, message = hashes_match(hash_path)
    if not ok:
        raise RuntimeError(message)
    expected_env = out_dir / "environment_preregistered.txt"
    if not expected_env.exists():
        raise RuntimeError("preregistered environment file is missing")
    current_versions = f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\n"
    saved_versions = expected_env.read_text(encoding="utf-8")
    if current_versions not in saved_versions:
        raise RuntimeError(
            "training environment does not match the preregistered environment:\n"
            f"current:\n{current_versions}\nsaved:\n{saved_versions}"
        )
    revision = git_revision()
    (out_dir / "environment.txt").write_text(
        "\n".join(environment_lines(revision)) + "\n", encoding="utf-8",
    )
    (out_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    records = []
    curves = []
    checkpoints = []
    clipping_rows = []
    fingerprint_rows = []
    wall = time.perf_counter()
    run_delay(cfg, out_dir, int(cfg["task"]["primary_delay"]), "primary",
              records, curves, checkpoints, clipping_rows, fingerprint_rows)
    decision = (
        "will run"
        if additive_successes(records, 64) >= 2
        else "not run: additive control had fewer than two successes at delay 64"
    )
    primary_summary = summarize(records, cfg)
    primary_summary["additive_successes"] = additive_successes(records, 64)
    primary_summary["delay_128"] = decision
    primary_summary["written_before_delay_128"] = True
    (out_dir / "delay64_summary.json").write_text(
        json.dumps(primary_summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    (out_dir / "fingerprints_delay64.json").write_text(
        json.dumps(fingerprint_rows, indent=2, sort_keys=True), encoding="utf-8",
    )
    if additive_successes(records, 64) >= 2:
        run_delay(cfg, out_dir, int(cfg["task"]["secondary_delay"]), "secondary",
                  records, curves, checkpoints, clipping_rows, fingerprint_rows)
    (out_dir / "fingerprints.json").write_text(
        json.dumps(fingerprint_rows, indent=2, sort_keys=True), encoding="utf-8",
    )
    summary = summarize(records, cfg)
    summary["total_wall_seconds"] = round(time.perf_counter() - wall, 3)
    summary["code_revision"] = revision
    summary["additive_delay64_successes"] = additive_successes(records, 64)
    summary["delay_128_ran"] = any(rec["delay"] == 128 for rec in records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_table(out_dir / "comparison_table.csv", records)
    flat_rows = summary["cells"]
    if flat_rows:
        fields = sorted({key for row in flat_rows for key in row})
        with (out_dir / cfg["outputs"]["summary_metrics"]).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
    after = frozen_manifest()
    before = json.loads(hash_path.read_text(encoding="utf-8"))
    (out_dir / "frozen_hashes_after.json").write_text(
        json.dumps({"match": after == before, "hashes": after}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if after != before:
        raise RuntimeError("frozen artifacts changed during EXP-004 training")
    print(
        f"EXP-004 finished; additive delay-64 successes "
        f"{summary['additive_delay64_successes']}; delay 128 ran: {summary['delay_128_ran']}"
    )


if __name__ == "__main__":
    main()

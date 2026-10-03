"""EXP-006: ten new seeds for vanilla, additive, and B=4 bounded models.

Delay 128 starts only after the delay-64 summary is saved and the additive
model has at least 8 successes. A 15-minute total cap covers training and
the bit-flip audit.

Usage (from the repository root):
    python experiments/run_exp006.py --write-frozen-hashes
    python experiments/run_exp006.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp004 import (  # noqa: E402
    build_matched_models,
    build_model,
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    load_hard_v2,
    measured_noise_std,
    minibatch_indices,
    parameter_fingerprint,
    train_one,
)
from run_exp005 import (  # noqa: E402
    audit_one,
    bit_and_marker_sensitivity,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
)

CONFIG_PATH = ROOT / "configs" / "exp006_replication.yaml"
FROZEN_PATHS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "results/EXP-004",
    "results/EXP-005",
    "configs/exp001_delayed_credit.yaml",
    "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml",
    "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml",
    "configs/exp004_bounded_additive.yaml",
    "configs/exp005_counterfactual_audit.yaml",
)


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


def frozen_manifest() -> dict[str, str]:
    records = {}
    for relative in FROZEN_PATHS:
        target = ROOT / relative
        if target.is_file():
            records[relative.replace("\\", "/")] = file_sha256(target)
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_frozen_hashes(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frozen_manifest(), indent=2, sort_keys=True), encoding="utf-8")


def binom_log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if k == n else float("-inf")
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_choose + k * math.log(p) + (n - k) * math.log(1.0 - p)


def binom_tail_ge(k: int, n: int, p: float) -> float:
    logs = [binom_log_pmf(j, n, p) for j in range(k, n + 1)]
    top = max(logs)
    return math.exp(top) * sum(math.exp(item - top) for item in logs)


def binom_tail_le(k: int, n: int, p: float) -> float:
    logs = [binom_log_pmf(j, n, p) for j in range(0, k + 1)]
    top = max(logs)
    return math.exp(top) * sum(math.exp(item - top) for item in logs)


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> dict[str, float]:
    """Exact central binomial interval. This is not a paired test."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    target = alpha / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if binom_tail_ge(successes, trials, mid) > target:
                hi = mid
            else:
                lo = mid
        lower = (lo + hi) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if binom_tail_le(successes, trials, mid) > target:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return {"lower": lower, "upper": upper, "successes": successes, "trials": trials}


def bit_loss_gradient(model, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    """Per-example absolute loss gradient on the original bit channel only."""
    probe = inputs.detach().clone().requires_grad_(True)
    logits, _ = model(probe)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, -1, 0], (targets + 1) / 2, reduction="none",
    )
    (grad,) = torch.autograd.grad(losses.sum(), probe)
    values = grad[:, 0, 0].detach().abs()
    marker = grad[:, 0, 1].detach().abs()
    return {
        "bit_loss_gradient_mean": float(values.mean()),
        "bit_loss_gradient_median": float(values.median()),
        "bit_loss_gradient_p95": float(torch.quantile(values, 0.95)),
        "bit_loss_gradient_exact_zeros": int((values == 0).sum()),
        "marker_loss_gradient_mean": float(marker.mean()),
    }


def final_channel_sensitivity(model, inputs: torch.Tensor) -> dict[str, float | int]:
    bit, marker = bit_and_marker_sensitivity(model, inputs)
    return {
        "bit_sensitivity_mean": float(bit.mean()),
        "bit_sensitivity_mean_abs": float(bit.abs().mean()),
        "bit_sensitivity_exact_zeros": int((bit == 0).sum()),
        "bit_sensitivity_max_abs": float(bit.abs().max()),
        "marker_sensitivity_mean_abs": float(marker.abs().mean()),
        "marker_sensitivity_exact_zeros": int((marker == 0).sum()),
    }


def save_checkpoint(out_dir: Path, model, record: dict) -> str:
    destination = out_dir / "checkpoints" / (
        f"{record['model_type']}_delay{record['delay']}_seed{record['seed']}.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    torch.save({
        "experiment_id": "EXP-006",
        "model_type": record["model_type"],
        "delay": record["delay"],
        "seed": record["seed"],
        "bound": record.get("bound"),
        "state_dict": model.state_dict(),
        "status": record["status"],
        "code_revision": record["code_revision"],
    }, destination)
    digest = file_sha256(destination)
    try:
        relative = destination.relative_to(ROOT).as_posix()
    except ValueError:
        relative = destination.as_posix()
    return relative + ":" + digest


def cohort_summary(records: list[dict], model_type: str, delay: int) -> dict:
    group = [row for row in records if row["model_type"] == model_type and row["delay"] == delay and row["status"] == "ok"]
    accuracies = [float(row["test_accuracy"]) for row in group]
    successes = sum(1 for row in group if row.get("success"))
    summary = {
        "model_type": model_type,
        "delay": delay,
        "completed": len(group),
        "successes": successes,
        "accuracies": {str(row["seed"]): row.get("test_accuracy") for row in records if row["model_type"] == model_type and row["delay"] == delay},
    }
    if accuracies:
        mean = sum(accuracies) / len(accuracies)
        summary["accuracy_mean"] = mean
        ordered = sorted(accuracies)
        mid = len(ordered) // 2
        summary["accuracy_median"] = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2
        if len(accuracies) > 1:
            var = sum((value - mean) ** 2 for value in accuracies) / (len(accuracies) - 1)
            summary["accuracy_sample_std"] = math.sqrt(var)
        else:
            summary["accuracy_sample_std"] = None
        summary["success_interval_95"] = clopper_pearson(successes, len(group))
    return summary


def paired_counts(records: list[dict], delay: int) -> dict[str, int]:
    by_seed = {}
    for row in records:
        if row["delay"] != delay or row["status"] != "ok":
            continue
        by_seed.setdefault(row["seed"], {})[row["model_type"]] = bool(row.get("success"))
    counts = {
        "both_succeed": 0,
        "additive_only": 0,
        "bounded_only": 0,
        "neither": 0,
        "vanilla_and_additive": 0,
        "vanilla_only_vs_additive": 0,
        "seeds_compared": 0,
    }
    for outcomes in by_seed.values():
        if "additive_tanh_rnn" not in outcomes or "bounded_additive_tanh_rnn" not in outcomes:
            continue
        counts["seeds_compared"] += 1
        additive = outcomes["additive_tanh_rnn"]
        bounded = outcomes["bounded_additive_tanh_rnn"]
        if additive and bounded:
            counts["both_succeed"] += 1
        elif additive:
            counts["additive_only"] += 1
        elif bounded:
            counts["bounded_only"] += 1
        else:
            counts["neither"] += 1
        if "vanilla_tanh_rnn" in outcomes:
            if outcomes["vanilla_tanh_rnn"] and additive:
                counts["vanilla_and_additive"] += 1
            elif outcomes["vanilla_tanh_rnn"] and not additive:
                counts["vanilla_only_vs_additive"] += 1
    return counts


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        write_frozen_hashes(ROOT / "results" / "EXP-006" / "frozen_hashes_before.json")
        print("wrote frozen hash manifest")
        return
    cfg = load_config()
    if platform.python_version() != cfg["environment_requirement"]["python"]:
        raise RuntimeError(f"Python {platform.python_version()} is not the required environment")
    if torch.__version__ != cfg["environment_requirement"]["torch"]:
        raise RuntimeError(f"PyTorch {torch.__version__} is not the required environment")
    out_dir = ROOT / cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "raw_metrics.jsonl").exists():
        raise FileExistsError("refusing to overwrite existing EXP-006 outputs")
    dirty = working_tree_dirty()
    if dirty:
        raise RuntimeError(f"working tree is dirty:\n{dirty}")
    before_path = out_dir / "frozen_hashes_before.json"
    if json.loads(before_path.read_text(encoding="utf-8")) != frozen_manifest():
        raise RuntimeError("frozen artifact hashes changed before EXP-006")
    revision = git_revision()
    (out_dir / "environment.txt").write_text(
        "\n".join([
            f"Python: {platform.python_version()}",
            f"PyTorch: {torch.__version__}",
            f"Training code revision: {revision}",
            "Bound B=4 was not tuned.",
            "Historical seeds are not pooled with this cohort.",
        ]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    total_cap = float(cfg["budget"]["max_cpu_seconds_total"])
    clock = time.perf_counter()
    records = []
    fingerprint_rows = []
    planned = [(64, seed, model) for seed in cfg["seeds"] for model in cfg["models"]]
    unstarted = []

    def expired() -> bool:
        return time.perf_counter() - clock >= total_cap

    def run_delay(delay: int, phase: str) -> None:
        for seed in cfg["seeds"]:
            if expired():
                for model_type in cfg["models"]:
                    unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "reason": "total cap"})
                continue
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
                "delay": delay, "seed": seed, "init_fingerprint": init_hash,
                "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
                "noise_std": measured_noise_std(bundle["test_x"]),
            })
            for model_type in cfg["models"]:
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "reason": "total cap"})
                    continue
                print(f"training {model_type} delay {delay} seed {seed}", flush=True)
                model = models[model_type]
                record, curve, checks, _clip = train_one(
                    cfg, model, model_type, bundle, indices, delay, seed, phase,
                    init_hash, data_fp, index_hash,
                )
                if record["status"] == "ok":
                    record.update(final_channel_sensitivity(model, bundle["test_x"]))
                    record.update(bit_loss_gradient(model, bundle["test_x"], bundle["test_y"]))
                    record["checkpoint_sha256"] = save_checkpoint(out_dir, model, record)
                    record["audit_ready"] = True
                else:
                    record["audit_ready"] = False
                records.append(record)
                append_jsonl(out_dir / "raw_metrics.jsonl", record)
                append_jsonl(out_dir / "loss_curves.jsonl", {
                    "delay": delay, "seed": seed, "model_type": model_type, "loss_curve": curve,
                })
                for item in checks:
                    append_jsonl(out_dir / "checkpoint_diagnostics.jsonl", {
                        "delay": delay, "seed": seed, "model_type": model_type, **item,
                    })
                print(json.dumps({
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "status": record["status"], "success": record.get("success"),
                    "test_accuracy": record.get("test_accuracy"),
                    "stop_reason": record.get("stop_reason"),
                }, sort_keys=True), flush=True)

    run_delay(64, "primary")
    additive_ok = sum(
        1 for row in records
        if row["delay"] == 64 and row["model_type"] == "additive_tanh_rnn" and row.get("success")
    )
    gate = additive_ok >= 8
    delay64 = {
        "additive_successes": additive_ok,
        "delay_128": "will run" if gate else "not run: additive had fewer than 8 delay-64 successes",
        "written_before_delay_128": True,
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "cohorts": [cohort_summary(records, model, 64) for model in cfg["models"]],
        "paired": paired_counts(records, 64),
    }
    (out_dir / "delay64_summary.json").write_text(json.dumps(delay64, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "fingerprints_delay64.json").write_text(
        json.dumps([row for row in fingerprint_rows if row["delay"] == 64], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if gate and not expired():
        run_delay(128, "secondary")
    elif gate:
        for seed in cfg["seeds"]:
            for model_type in cfg["models"]:
                unstarted.append({"delay": 128, "seed": seed, "model_type": model_type, "reason": "total cap before delay 128"})
    audit_rows = []
    if not expired():
        for delay in sorted({row["delay"] for row in records}):
            for seed in cfg["seeds"]:
                ready = [row for row in records if row["delay"] == delay and row["seed"] == seed and row.get("audit_ready")]
                if len(ready) < 3:
                    continue
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "model_type": "audit", "reason": "total cap"})
                    continue
                base, _y, _meta = load_hard_v2(
                    cfg["data"]["audit_samples"], delay, seed, cfg["data"]["audit_split_index"],
                    noise_std=cfg["task"]["distractor_noise_std"],
                    competitor_rate=cfg["task"]["competitor_rate"],
                    competitor_flip_prob=cfg["task"]["competitor_flip_prob"],
                    base_seed=cfg["data"]["base_seed"],
                    return_metadata=True,
                )
                positive, negative = make_counterfactual_pairs(base)
                if not pairs_differ_only_on_original_bit(positive, negative):
                    raise RuntimeError(f"pair construction failed at delay {delay} seed {seed}")
                for model_type in cfg["models"]:
                    if expired():
                        unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "reason": "total cap during audit"})
                        continue
                    path = out_dir / "checkpoints" / f"{model_type}_delay{delay}_seed{seed}.pt"
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    model = build_model(model_type, cfg)
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    print(f"auditing {model_type} delay {delay} seed {seed}", flush=True)
                    audited = audit_one(model, model_type, positive, negative)
                    compact = {
                        "delay": delay, "seed": seed, "model_type": model_type,
                        "accuracy": audited["accuracy"],
                        "both_correct_fraction": audited["both_correct_fraction"],
                        "prediction_flip_fraction": audited["prediction_flip_fraction"],
                        "identical_logits_fraction": audited["identical_logits_fraction"],
                        "final_states_exactly_equal_fraction": audited["final_states_exactly_equal_fraction"],
                        "final_sign_pattern_identical_fraction": audited["final_sign_pattern_identical_fraction"],
                        "pairs_zero_local_grad_but_logit_changes": audited["pairs_zero_local_grad_but_logit_changes"],
                        "pairs_zero_local_grad_but_state_changes": audited["pairs_zero_local_grad_but_state_changes"],
                        "pairs_both_bit_grads_exact_zero": audited["pairs_both_bit_grads_exact_zero"],
                        "boundary": audited.get("boundary"),
                    }
                    audit_rows.append(compact)
                    append_jsonl(out_dir / "audit_metrics.jsonl", compact)
    else:
        for row in records:
            unstarted.append({
                "delay": row["delay"], "seed": row["seed"], "model_type": row["model_type"],
                "reason": "audit not started because the total cap was reached",
            })
    (out_dir / "fingerprints.json").write_text(json.dumps(fingerprint_rows, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "code_revision": revision,
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "total_cap_seconds": total_cap,
        "cap_reached": expired(),
        "additive_delay64_successes": additive_ok,
        "delay_128_ran": any(row["delay"] == 128 for row in records),
        "planned_delay64": len(planned),
        "completed_runs": sum(1 for row in records if row["status"] == "ok"),
        "failed_runs": sum(1 for row in records if row["status"] != "ok"),
        "unstarted": unstarted,
        "cohorts": [cohort_summary(records, model, delay) for delay in (64, 128) for model in cfg["models"] if any(row["delay"] == delay for row in records)],
        "paired_delay64": paired_counts(records, 64),
        "paired_delay128": paired_counts(records, 128) if any(row["delay"] == 128 for row in records) else None,
        "audit_rows": len(audit_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "budget_log.json").write_text(json.dumps({
        "elapsed_seconds": summary["elapsed_seconds"],
        "cap_reached": summary["cap_reached"],
        "unstarted": unstarted,
        "failed": [
            {"delay": row["delay"], "seed": row["seed"], "model_type": row["model_type"], "stop_reason": row.get("stop_reason")}
            for row in records if row["status"] != "ok"
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")
    table_fields = [
        "delay", "seed", "model_type", "status", "success", "test_accuracy", "test_loss",
        "val_loss", "train_loss", "retention_probe_r2", "max_abs_hidden", "final_hidden_rms",
        "bit_sensitivity_exact_zeros", "bit_sensitivity_mean_abs",
        "runtime_seconds", "stop_reason",
    ]
    with (out_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delay", "seed", "model_type", "success", "test_accuracy"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    after = frozen_manifest()
    before = json.loads(before_path.read_text(encoding="utf-8"))
    matched = after == before
    (out_dir / "frozen_hashes_after.json").write_text(json.dumps({"match": matched}, indent=2), encoding="utf-8")
    if not matched:
        raise RuntimeError("frozen artifacts changed during EXP-006")
    print(
        f"EXP-006 finished in {summary['elapsed_seconds']}s; "
        f"additive delay-64 successes {additive_ok}; unstarted {len(unstarted)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

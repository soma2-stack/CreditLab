"""EXP-008: which additive parameter group is trained.

Usage (from the repository root):
    python experiments/run_exp008.py --write-frozen-hashes
    python experiments/run_exp008.py
"""

from __future__ import annotations

import csv
import hashlib
import json
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

from creditlab.diagnostics import hidden_magnitude_stats  # noqa: E402
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.models import ResidualTanhRNN  # noqa: E402
from run_exp001 import evaluate, probe_fit_score  # noqa: E402
from run_exp004 import (  # noqa: E402
    DIAGNOSTIC_UPDATES,
    clipping_rollout,
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    json_safe,
    load_hard_v2,
    measured_noise_std,
    minibatch_indices,
    parameter_fingerprint,
)
from run_exp005 import (  # noqa: E402
    audit_one,
    bit_and_marker_sensitivity,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
)
from run_exp006 import bit_loss_gradient, clopper_pearson  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp008_parameter_groups.yaml"
FROZEN_PATHS = (
    "results/EXP-001", "results/EXP-001B", "results/EXP-002", "results/EXP-003",
    "results/EXP-004", "results/EXP-005", "results/EXP-006", "results/EXP-007",
    "configs/exp001_delayed_credit.yaml", "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml", "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml", "configs/exp004_bounded_additive.yaml",
    "configs/exp005_counterfactual_audit.yaml", "configs/exp006_replication.yaml",
    "configs/exp007_readout_only.yaml",
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
        elif target.exists():
            for path in sorted(item for item in target.rglob("*") if item.is_file()):
                records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_frozen_hashes(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frozen_manifest(), indent=2, sort_keys=True), encoding="utf-8")


def apply_freeze(model, frozen_names: list[str]) -> None:
    present = {name for name, _ in model.named_parameters()}
    missing = [name for name in frozen_names if name not in present]
    if missing:
        raise RuntimeError(f"missing parameters: {missing}")
    frozen = set(frozen_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in frozen)


def group_fingerprint(model, names: list[str]) -> str:
    tensors = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_four(cfg: dict, seed: int) -> dict:
    from creditlab.reproducibility import seed_everything
    models = {}
    for regime, frozen_names in cfg["frozen_by_regime"].items():
        seed_everything(seed)
        model = ResidualTanhRNN(
            cfg["model"]["input_size"], cfg["model"]["hidden_size"], cfg["model"]["output_size"],
            residual_scale=1.0,
        )
        apply_freeze(model, frozen_names)
        models[regime] = model
    return models


def checkpoint_plus(model, val_x, val_y, update: int, cfg: dict) -> dict:
    record = {
        "update": update,
        "split": "validation",
        "retention_probe_r2": probe_fit_score(model, val_x[:256], val_y[:256], val_x[256:], val_y[256:]),
        "val_loss": None,
        "val_accuracy": None,
    }
    val_loss, val_acc = evaluate(model, val_x, val_y, cfg["training"]["eval_batch_size"])
    record["val_loss"] = val_loss
    record["val_accuracy"] = val_acc
    bit, marker = bit_and_marker_sensitivity(model, val_x)
    record["bit_sensitivity_mean_abs"] = float(bit.abs().mean())
    record["bit_sensitivity_exact_zeros"] = int((bit == 0).sum())
    record["marker_sensitivity_mean_abs"] = float(marker.abs().mean())
    record.update(bit_loss_gradient(model, val_x, val_y))
    magnitude = hidden_magnitude_stats(model, val_x)
    record["max_abs_hidden"] = magnitude["max_abs_hidden"]
    record["final_hidden_rms"] = magnitude["final_hidden_rms"]
    return json_safe(record)


def train_regime(cfg, model, regime, bundle, indices, delay, seed, phase, init_fingerprint, data_fp, index_hash):
    started = time.perf_counter()
    budget = float(cfg["budget"]["max_cpu_seconds_per_run"])
    updates_target = int(cfg["training"]["updates_per_run"])
    frozen_names = list(cfg["frozen_by_regime"][regime])
    apply_freeze(model, frozen_names)
    frozen_before = group_fingerprint(model, frozen_names) if frozen_names else None
    hidden_before = None
    if regime == "readout_only":
        with torch.no_grad():
            hidden_before = model.recurrent_states(bundle["val_x"][:32]).clone()
    record = {
        "experiment_id": "EXP-008",
        "regime": regime,
        "delay": delay,
        "seed": seed,
        "phase": phase,
        "status": "ok",
        "stop_reason": None,
        "code_revision": git_revision()[:12],
        "init_fingerprint": init_fingerprint,
        "dataset_fingerprints": data_fp,
        "minibatch_fingerprint": index_hash,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "distractor_query_channel_std": round(measured_noise_std(bundle["test_x"]), 6),
    }
    tr = cfg["training"]
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(tr["learning_rate"]))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss_curve = []
    checkpoints = []
    clipped = 0
    try:
        checkpoints.append(checkpoint_plus(model, bundle["val_x"], bundle["val_y"], 0, cfg))
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > budget:
                raise TimeoutError(f"CPU budget of {budget:.0f}s exhausted at update {update}")
            xb, yb = bundle["train_x"][idx], bundle["train_y"][idx]
            logits, _ = model(xb)
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(tr["grad_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at update {update}")
            if float(grad_norm) > float(tr["grad_clip_norm"]):
                clipped += 1
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in DIAGNOSTIC_UPDATES:
                checkpoints.append(checkpoint_plus(model, bundle["val_x"], bundle["val_y"], update + 1, cfg))
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"])
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"])
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"])
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        bit, marker = bit_and_marker_sensitivity(model, bundle["test_x"])
        bit_loss = bit_loss_gradient(model, bundle["test_x"], bundle["test_y"])
        frozen_after = group_fingerprint(model, frozen_names) if frozen_names else None
        hidden_unchanged = None
        if hidden_before is not None:
            with torch.no_grad():
                hidden_after = model.recurrent_states(bundle["val_x"][:32])
            hidden_unchanged = bool(torch.equal(hidden_before, hidden_after))
        success = bool(test_acc >= float(cfg["interpretation"]["success_test_accuracy_at_least"]))
        record.update(
            updates_completed=updates_target,
            train_loss=train_loss, train_accuracy=train_acc,
            val_loss=val_loss, val_accuracy=val_acc,
            test_loss=test_loss, test_accuracy=test_acc, success=success,
            retention_probe_r2=retention,
            max_abs_hidden=magnitude["max_abs_hidden"],
            final_hidden_rms=magnitude["final_hidden_rms"],
            bit_sensitivity_mean_abs=float(bit.abs().mean()),
            bit_sensitivity_exact_zeros=int((bit == 0).sum()),
            marker_sensitivity_mean_abs=float(marker.abs().mean()),
            grad_clip_fraction=clipped / updates_target,
            frozen_parameters_unchanged=frozen_before == frozen_after,
            hidden_states_unchanged=hidden_unchanged,
            all_finite=True,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
        record.update(bit_loss)
    except (FloatingPointError, RuntimeError, TimeoutError) as exc:
        record["status"] = "budget_stop" if isinstance(exc, TimeoutError) else "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["updates_completed"] = len(loss_curve)
        record["success"] = False
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return record, loss_curve, checkpoints


def save_checkpoint(out_dir: Path, model, record: dict) -> str:
    destination = out_dir / "checkpoints" / f"additive_{record['regime']}_delay{record['delay']}_seed{record['seed']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save({
        "experiment_id": "EXP-008",
        "regime": record["regime"],
        "delay": record["delay"],
        "seed": record["seed"],
        "state_dict": model.state_dict(),
    }, destination)
    return file_sha256(destination)


def cohort_summary(records, regime, delay):
    group = [row for row in records if row["regime"] == regime and row["delay"] == delay and row["status"] == "ok"]
    accuracies = [float(row["test_accuracy"]) for row in group]
    successes = sum(1 for row in group if row.get("success"))
    summary = {
        "regime": regime, "delay": delay, "completed": len(group), "successes": successes,
        "accuracies": {str(row["seed"]): row.get("test_accuracy") for row in records if row["regime"] == regime and row["delay"] == delay},
    }
    if accuracies:
        mean = sum(accuracies) / len(accuracies)
        ordered = sorted(accuracies)
        mid = len(ordered) // 2
        summary["accuracy_mean"] = mean
        summary["accuracy_median"] = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        summary["success_interval_95"] = clopper_pearson(successes, len(group))
    return summary


def git_revision() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
    return out.stdout.strip()


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        write_frozen_hashes(ROOT / "results" / "EXP-008" / "frozen_hashes_before.json")
        print("wrote frozen hash manifest")
        return
    cfg = load_config()
    if platform.python_version() != cfg["environment_requirement"]["python"] or torch.__version__ != cfg["environment_requirement"]["torch"]:
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    out_dir = ROOT / "results" / "EXP-008"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_metrics.jsonl"
    resuming = raw_path.exists()
    if working_tree_dirty() and not resuming:
        raise RuntimeError("working tree is dirty")
    before_path = out_dir / "frozen_hashes_before.json"
    if json.loads(before_path.read_text(encoding="utf-8")) != frozen_manifest():
        raise RuntimeError("frozen artifact hashes changed")
    revision = git_revision()
    if not resuming:
        (out_dir / "environment.txt").write_text(
            f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nTraining code revision: {revision}\n",
            encoding="utf-8",
        )
        (out_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] if resuming else []
    done = {(row["delay"], row["seed"], row["regime"]) for row in records}
    already = 0.0
    if resuming:
        created = raw_path.stat().st_ctime
        last_write = max(path.stat().st_mtime for path in out_dir.rglob("*") if path.is_file())
        already = max(0.0, last_write - created)
    clock = time.perf_counter() - already
    total_cap = float(cfg["budget"]["max_cpu_seconds_total"])
    unstarted = []

    def expired() -> bool:
        return time.perf_counter() - clock >= total_cap

    def run_delay(delay: int, phase: str) -> None:
        for seed in cfg["seeds"]:
            if all((delay, seed, regime) in done for regime in cfg["regimes"]):
                continue
            if expired():
                for regime in cfg["regimes"]:
                    if (delay, seed, regime) not in done:
                        unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "total cap"})
                continue
            bundle = dataset_bundle(cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(seed, delay, bundle["train_x"].shape[0], cfg["training"]["batch_size"], cfg["training"]["updates_per_run"])
            index_hash = index_fingerprint(indices)
            models = build_four(cfg, seed)
            reference = parameter_fingerprint(models["full"])
            if any(parameter_fingerprint(model) != reference for model in models.values()):
                raise RuntimeError(f"initial parameters differ at delay {delay} seed {seed}")
            probe = bundle["train_x"][:4]
            with torch.no_grad():
                reference_logits, _ = models["full"](probe)
            for regime, model in models.items():
                with torch.no_grad():
                    logits, _ = model(probe)
                if float((logits - reference_logits).abs().max()) > float(cfg["diagnostics"]["paired_forward_tolerance"]):
                    raise RuntimeError(f"initial outputs differ for {regime}")
            append_jsonl(out_dir / "fingerprints.jsonl", {
                "delay": delay, "seed": seed, "init_fingerprint": reference,
                "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
                "noise_std": measured_noise_std(bundle["test_x"]),
            })
            for regime in cfg["regimes"]:
                if (delay, seed, regime) in done:
                    continue
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "total cap"})
                    continue
                print(f"training {regime} delay {delay} seed {seed}", flush=True)
                record, curve, checks = train_regime(
                    cfg, models[regime], regime, bundle, indices, delay, seed, phase, reference, data_fp, index_hash,
                )
                if regime != "readout_only" and record.get("hidden_states_unchanged") is False:
                    pass
                if record.get("frozen_parameters_unchanged") is False or (
                    regime == "readout_only" and record.get("hidden_states_unchanged") is False
                ):
                    record["status"] = "implementation_bug"
                    record["success"] = False
                    record["stop_reason"] = "a frozen parameter changed"
                if record["status"] == "ok":
                    record["checkpoint_sha256"] = save_checkpoint(out_dir, models[regime], record)
                records.append(record)
                done.add((delay, seed, regime))
                append_jsonl(raw_path, record)
                append_jsonl(out_dir / "loss_curves.jsonl", {"delay": delay, "seed": seed, "regime": regime, "loss_curve": curve})
                for item in checks:
                    append_jsonl(out_dir / "checkpoint_diagnostics.jsonl", {"delay": delay, "seed": seed, "regime": regime, **item})
                print(json.dumps({
                    "delay": delay, "seed": seed, "regime": regime, "status": record["status"],
                    "success": record.get("success"), "test_accuracy": record.get("test_accuracy"),
                }, sort_keys=True), flush=True)

    run_delay(64, "primary")
    full_successes = sum(1 for row in records if row["delay"] == 64 and row["regime"] == "full" and row.get("success"))
    gate = full_successes >= 8
    if not (out_dir / "delay64_summary.json").exists():
        (out_dir / "delay64_summary.json").write_text(json.dumps({
            "full_successes": full_successes,
            "delay_128": "will run" if gate else "not run",
            "written_before_delay_128": True,
            "cohorts": [cohort_summary(records, regime, 64) for regime in cfg["regimes"]],
        }, indent=2, sort_keys=True), encoding="utf-8")
    if gate and not expired():
        run_delay(128, "secondary")
    elif gate:
        for seed in cfg["seeds"]:
            for regime in cfg["regimes"]:
                if (128, seed, regime) not in done:
                    unstarted.append({"delay": 128, "seed": seed, "regime": regime, "reason": "total cap"})
    audit_count = 0
    for delay in sorted({row["delay"] for row in records}):
        for seed in cfg["seeds"]:
            ready = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["status"] == "ok"]
            if len(ready) < len(cfg["regimes"]):
                continue
            if expired():
                unstarted.append({"delay": delay, "seed": seed, "regime": "audit", "reason": "total cap"})
                continue
            base, _y, _meta = load_hard_v2(
                cfg["data"]["audit_samples"], delay, seed, cfg["data"]["audit_split_index"],
                noise_std=cfg["task"]["distractor_noise_std"], competitor_rate=cfg["task"]["competitor_rate"],
                competitor_flip_prob=cfg["task"]["competitor_flip_prob"], base_seed=cfg["data"]["base_seed"],
                return_metadata=True,
            )
            positive, negative = make_counterfactual_pairs(base)
            if not pairs_differ_only_on_original_bit(positive, negative):
                raise RuntimeError("pair construction failed")
            for regime in cfg["regimes"]:
                path = out_dir / "checkpoints" / f"additive_{regime}_delay{delay}_seed{seed}.pt"
                if not path.exists() or expired():
                    if not path.exists():
                        continue
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "audit cap"})
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                from creditlab.reproducibility import seed_everything
                seed_everything(0)
                model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                before = parameter_fingerprint(model)
                audited = audit_one(model, "additive_tanh_rnn", positive, negative)
                if parameter_fingerprint(model) != before:
                    raise RuntimeError("audit changed parameters")
                append_jsonl(out_dir / "audit_metrics.jsonl", {
                    "delay": delay, "seed": seed, "regime": regime,
                    "accuracy": audited["accuracy"],
                    "both_correct_fraction": audited["both_correct_fraction"],
                    "prediction_flip_fraction": audited["prediction_flip_fraction"],
                    "final_states_exactly_equal_fraction": audited["final_states_exactly_equal_fraction"],
                    "identical_logits_fraction": audited["identical_logits_fraction"],
                    "pairs_zero_local_grad_but_logit_changes": audited["pairs_zero_local_grad_but_logit_changes"],
                    "pairs_zero_local_grad_but_state_changes": audited["pairs_zero_local_grad_but_state_changes"],
                })
                audit_count += 1
    summary = {
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "full_delay64_successes": full_successes,
        "delay_128_ran": any(row["delay"] == 128 for row in records),
        "completed_runs": sum(1 for row in records if row["status"] == "ok"),
        "failed_runs": sum(1 for row in records if row["status"] != "ok"),
        "unstarted": unstarted,
        "audit_rows": audit_count,
        "cohorts": [cohort_summary(records, regime, delay) for delay in (64, 128) for regime in cfg["regimes"] if any(row["delay"] == delay for row in records)],
        "code_revision": revision,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delay", "seed", "regime", "status", "success", "test_accuracy", "retention_probe_r2"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    matched = frozen_manifest() == json.loads(before_path.read_text(encoding="utf-8"))
    (out_dir / "frozen_hashes_after.json").write_text(json.dumps({"match": matched}, indent=2), encoding="utf-8")
    if not matched:
        raise RuntimeError("frozen artifacts changed during EXP-008")
    print(f"EXP-008 finished in {summary['elapsed_seconds']}s; full successes {full_successes}; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

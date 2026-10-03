"""EXP-007: full training versus readout-only training on the same models.

Usage (from the repository root):
    python experiments/run_exp007.py --write-frozen-hashes
    python experiments/run_exp007.py
"""

from __future__ import annotations

import copy
import csv
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

from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp004 import (  # noqa: E402
    DIAGNOSTIC_UPDATES,
    build_matched_models,
    build_model,
    checkpoint_record,
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    json_safe,
    load_hard_v2,
    measured_noise_std,
    minibatch_indices,
    parameter_fingerprint,
    train_one,
)
from run_exp005 import (  # noqa: E402
    audit_one,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
)
from run_exp006 import bit_loss_gradient, clopper_pearson, final_channel_sensitivity  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp007_readout_only.yaml"
FROZEN_PATHS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "results/EXP-004",
    "results/EXP-005",
    "results/EXP-006",
    "configs/exp001_delayed_credit.yaml",
    "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml",
    "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml",
    "configs/exp004_bounded_additive.yaml",
    "configs/exp005_counterfactual_audit.yaml",
    "configs/exp006_replication.yaml",
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


def recurrent_fingerprint(model) -> str:
    digest_source = []
    for name, value in sorted(model.recurrent.state_dict().items()):
        digest_source.append(name.encode("utf-8") + value.detach().cpu().contiguous().numpy().tobytes())
    return __import__("hashlib").sha256(b"".join(digest_source)).hexdigest()


def freeze_recurrent(model) -> list[str]:
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith("recurrent."):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def trainable_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def forward_fingerprint(model, inputs: torch.Tensor) -> str:
    with torch.no_grad():
        logits, _ = model(inputs)
    return __import__("hashlib").sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hidden_fingerprint(model, inputs: torch.Tensor) -> str:
    with torch.no_grad():
        states = model.recurrent_states(inputs)
    return __import__("hashlib").sha256(states.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def train_readout_only(cfg, model, model_type, bundle, indices, delay, seed, phase, init_fingerprint, data_fp, index_hash, hidden_before):
    started = time.perf_counter()
    budget = float(cfg["budget"]["max_cpu_seconds_per_run"])
    updates_target = int(cfg["training"]["updates_per_run"])
    frozen_names = freeze_recurrent(model)
    trainable = trainable_names(model)
    recurrent_before = recurrent_fingerprint(model)
    record = {
        "experiment_id": cfg["experiment_id"],
        "config_revision": cfg["config_revision"],
        "code_revision": git_revision()[:12],
        "phase": phase,
        "regime": "readout_only",
        "delay": delay,
        "mode": "HARD_V2",
        "model_type": model_type,
        "bound": float(cfg["bound"]["B"]) if model_type == "bounded_additive_tanh_rnn" else None,
        "seed": seed,
        "device": "cpu",
        "status": "ok",
        "stop_reason": None,
        "init_fingerprint": init_fingerprint,
        "dataset_fingerprints": data_fp,
        "minibatch_fingerprint": index_hash,
        "distractor_query_channel_std": round(measured_noise_std(bundle["test_x"]), 6),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable,
        "recurrent_fingerprint_before": recurrent_before,
    }
    tr = cfg["training"]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(tr["learning_rate"]))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss_curve = []
    checkpoints = []
    clipped = 0
    try:
        checkpoints.append(json_safe(checkpoint_record(model, model_type, bundle["val_x"], bundle["val_y"], 0, cfg)))
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(tr["grad_clip_norm"]),
            )
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
        from run_exp001 import evaluate, probe_fit_score
        from creditlab.diagnostics import event_gradient_norm, event_loss_gradient_metrics, hidden_magnitude_stats
        from run_exp004 import clipping_rollout
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"])
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"])
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"])
        sensitivity = event_gradient_norm(model, bundle["test_x"], bundle["test_y"])
        loss_grad = event_loss_gradient_metrics(model, bundle["test_x"], bundle["test_y"])
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        final_clip = clipping_rollout(model, bundle["test_x"]) if model_type == "bounded_additive_tanh_rnn" else None
        success_cutoff = float(cfg["interpretation"]["success_test_accuracy_at_least"])
        hidden_after = hidden_fingerprint(model, bundle["val_x"][:64])
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
            final_test_clipping=None if final_clip is None else json_safe(final_clip),
            recurrent_fingerprint_after=recurrent_fingerprint(model),
            recurrent_parameters_unchanged=recurrent_fingerprint(model) == recurrent_before,
            hidden_states_unchanged=hidden_after == hidden_before,
        )
        record.update(final_channel_sensitivity(model, bundle["test_x"]))
        record.update(bit_loss_gradient(model, bundle["test_x"], bundle["test_y"]))
    except (FloatingPointError, RuntimeError, TimeoutError) as exc:
        record["status"] = "budget_stop" if isinstance(exc, TimeoutError) else "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["updates_completed"] = len(loss_curve)
        record["success"] = False
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
        record["recurrent_fingerprint_after"] = recurrent_fingerprint(model)
        record["recurrent_parameters_unchanged"] = record["recurrent_fingerprint_after"] == recurrent_before
    return record, loss_curve, checkpoints


def save_checkpoint(out_dir: Path, model, record: dict) -> str:
    destination = out_dir / "checkpoints" / (
        f"{record['model_type']}_{record['regime']}_delay{record['delay']}_seed{record['seed']}.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    torch.save({
        "experiment_id": "EXP-007",
        "model_type": record["model_type"],
        "regime": record["regime"],
        "delay": record["delay"],
        "seed": record["seed"],
        "state_dict": model.state_dict(),
        "status": record["status"],
    }, destination)
    return destination.relative_to(out_dir).as_posix() + ":" + file_sha256(destination)


def cohort_summary(records, model_type, regime, delay):
    group = [row for row in records if row["model_type"] == model_type and row["regime"] == regime and row["delay"] == delay and row["status"] == "ok"]
    accuracies = [float(row["test_accuracy"]) for row in group]
    successes = sum(1 for row in group if row.get("success"))
    summary = {
        "model_type": model_type, "regime": regime, "delay": delay,
        "completed": len(group), "successes": successes,
        "accuracies": {
            str(row["seed"]): row.get("test_accuracy")
            for row in records if row["model_type"] == model_type and row["regime"] == regime and row["delay"] == delay
        },
    }
    if accuracies:
        mean = sum(accuracies) / len(accuracies)
        ordered = sorted(accuracies)
        mid = len(ordered) // 2
        summary["accuracy_mean"] = mean
        summary["accuracy_median"] = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        summary["success_interval_95"] = clopper_pearson(successes, len(group))
    return summary


def paired_regimes(records, delay):
    counts = {"full_only": 0, "readout_only": 0, "both": 0, "neither": 0, "seeds": 0}
    by_model = {}
    grouped = {}
    for row in records:
        if row["delay"] != delay or row["status"] != "ok":
            continue
        grouped.setdefault((row["model_type"], row["seed"]), {})[row["regime"]] = bool(row.get("success"))
    for (model_type, _seed), outcomes in grouped.items():
        if "full" not in outcomes or "readout_only" not in outcomes:
            continue
        by_model.setdefault(model_type, {"full_only": 0, "readout_only": 0, "both": 0, "neither": 0, "seeds": 0})
        bucket = by_model[model_type]
        bucket["seeds"] += 1
        full, readout = outcomes["full"], outcomes["readout_only"]
        if full and readout:
            bucket["both"] += 1
        elif full:
            bucket["full_only"] += 1
        elif readout:
            bucket["readout_only"] += 1
        else:
            bucket["neither"] += 1
    return by_model


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        write_frozen_hashes(ROOT / "results" / "EXP-007" / "frozen_hashes_before.json")
        print("wrote frozen hash manifest")
        return
    cfg = load_config()
    if platform.python_version() != cfg["environment_requirement"]["python"] or torch.__version__ != cfg["environment_requirement"]["torch"]:
        raise RuntimeError("environment does not match the preregistered Python 3.11.9 / PyTorch 2.13.0+cpu")
    out_dir = ROOT / cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "raw_metrics.jsonl").exists():
        raise FileExistsError("refusing to overwrite existing EXP-007 outputs")
    dirty = working_tree_dirty()
    if dirty:
        raise RuntimeError(f"working tree is dirty:\n{dirty}")
    before_path = out_dir / "frozen_hashes_before.json"
    if json.loads(before_path.read_text(encoding="utf-8")) != frozen_manifest():
        raise RuntimeError("frozen artifact hashes changed before EXP-007")
    revision = git_revision()
    (out_dir / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (out_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "parameter_sets.json").write_text(json.dumps({
        "frozen_when_readout_only": cfg["frozen_parameter_names"],
        "trainable_when_readout_only": cfg["trainable_readout_parameter_names"],
        "total_parameter_count": 1217,
        "readout_trainable_count": 33,
    }, indent=2), encoding="utf-8")
    clock = time.perf_counter()
    total_cap = float(cfg["budget"]["max_cpu_seconds_total"])
    records = []
    fingerprint_rows = []
    unstarted = []
    stop_for_bug = False

    def expired() -> bool:
        return time.perf_counter() - clock >= total_cap

    def run_delay(delay: int, phase: str) -> None:
        nonlocal stop_for_bug
        for seed in cfg["seeds"]:
            if stop_for_bug or expired():
                for model_type in cfg["models"]:
                    for regime in cfg["regimes"]:
                        unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "regime": regime, "reason": "stopped"})
                continue
            bundle = dataset_bundle(cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(seed, delay, bundle["train_x"].shape[0], cfg["training"]["batch_size"], cfg["training"]["updates_per_run"])
            index_hash = index_fingerprint(indices)
            full_models = build_matched_models(cfg, seed)
            init_hash = parameter_fingerprint(full_models["vanilla_tanh_rnn"])
            fingerprint_rows.append({
                "delay": delay, "seed": seed, "init_fingerprint": init_hash,
                "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
                "noise_std": measured_noise_std(bundle["test_x"]),
            })
            for model_type in cfg["models"]:
                full = full_models[model_type]
                readout = copy.deepcopy(full)
                freeze_recurrent(readout)
                probe = bundle["train_x"][:8]
                with torch.no_grad():
                    full_logits, _ = full(probe)
                    readout_logits, _ = readout(probe)
                forward_gap = float((full_logits - readout_logits).abs().max())
                if forward_gap > 1e-5:
                    raise RuntimeError(
                        f"paired regimes differ before training at delay {delay} seed {seed}: {forward_gap}"
                    )
                hidden_before = hidden_fingerprint(readout, bundle["val_x"][:64])
                for regime, model in (("full", full), ("readout_only", readout)):
                    if expired() or stop_for_bug:
                        unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "regime": regime, "reason": "stopped"})
                        continue
                    print(f"training {model_type} {regime} delay {delay} seed {seed}", flush=True)
                    if regime == "full":
                        record, curve, checks, _clip = train_one(
                            cfg, model, model_type, bundle, indices, delay, seed, phase, init_hash, data_fp, index_hash,
                        )
                        record["regime"] = "full"
                        record["parameter_count"] = sum(p.numel() for p in model.parameters())
                        record["trainable_parameter_count"] = record["parameter_count"]
                        if record["status"] == "ok":
                            record.update(final_channel_sensitivity(model, bundle["test_x"]))
                            record.update(bit_loss_gradient(model, bundle["test_x"], bundle["test_y"]))
                    else:
                        record, curve, checks = train_readout_only(
                            cfg, model, model_type, bundle, indices, delay, seed, phase, init_hash, data_fp, index_hash, hidden_before,
                        )
                        if record["status"] == "ok" and not (record["recurrent_parameters_unchanged"] and record["hidden_states_unchanged"]):
                            record["status"] = "implementation_bug"
                            record["stop_reason"] = "frozen recurrent state changed"
                            record["success"] = False
                            stop_for_bug = True
                    if record["status"] == "ok":
                        record["checkpoint_sha256"] = save_checkpoint(out_dir, model, record)
                        record["audit_ready"] = True
                    else:
                        record["audit_ready"] = False
                    records.append(record)
                    append_jsonl(out_dir / "raw_metrics.jsonl", record)
                    append_jsonl(out_dir / "loss_curves.jsonl", {
                        "delay": delay, "seed": seed, "model_type": model_type, "regime": regime, "loss_curve": curve,
                    })
                    for item in checks:
                        append_jsonl(out_dir / "checkpoint_diagnostics.jsonl", {
                            "delay": delay, "seed": seed, "model_type": model_type, "regime": regime, **item,
                        })
                    print(json.dumps({
                        "delay": delay, "seed": seed, "model_type": model_type, "regime": regime,
                        "status": record["status"], "success": record.get("success"),
                        "test_accuracy": record.get("test_accuracy"), "stop_reason": record.get("stop_reason"),
                    }, sort_keys=True), flush=True)

    run_delay(64, "primary")
    additive_ok = sum(1 for row in records if row["delay"] == 64 and row["model_type"] == "additive_tanh_rnn" and row["regime"] == "full" and row.get("success"))
    gate = additive_ok >= 8 and not stop_for_bug
    (out_dir / "delay64_summary.json").write_text(json.dumps({
        "full_additive_successes": additive_ok,
        "delay_128": "will run" if gate else "not run",
        "written_before_delay_128": True,
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "cohorts": [cohort_summary(records, model, regime, 64) for model in cfg["models"] for regime in cfg["regimes"]],
        "paired": paired_regimes(records, 64),
    }, indent=2, sort_keys=True), encoding="utf-8")
    if gate and not expired():
        run_delay(128, "secondary")
    elif gate:
        for seed in cfg["seeds"]:
            for model_type in cfg["models"]:
                for regime in cfg["regimes"]:
                    unstarted.append({"delay": 128, "seed": seed, "model_type": model_type, "regime": regime, "reason": "total cap"})
    audit_count = 0
    if not stop_for_bug:
        for delay in sorted({row["delay"] for row in records}):
            for seed in cfg["seeds"]:
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "model_type": "audit", "regime": "all", "reason": "total cap"})
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
                for model_type in cfg["models"]:
                    for regime in cfg["regimes"]:
                        match = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["model_type"] == model_type and row["regime"] == regime and row.get("audit_ready")]
                        if not match:
                            continue
                        if expired():
                            unstarted.append({"delay": delay, "seed": seed, "model_type": model_type, "regime": regime, "reason": "total cap during audit"})
                            continue
                        path = out_dir / "checkpoints" / f"{model_type}_{regime}_delay{delay}_seed{seed}.pt"
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        model = build_model(model_type, cfg)
                        model.load_state_dict(payload["state_dict"])
                        model.eval()
                        for parameter in model.parameters():
                            parameter.requires_grad_(False)
                        before = parameter_fingerprint(model)
                        audited = audit_one(model, model_type, positive, negative)
                        if parameter_fingerprint(model) != before:
                            raise RuntimeError("audit changed parameters")
                        append_jsonl(out_dir / "audit_metrics.jsonl", {
                            "delay": delay, "seed": seed, "model_type": model_type, "regime": regime,
                            "accuracy": audited["accuracy"],
                            "both_correct_fraction": audited["both_correct_fraction"],
                            "prediction_flip_fraction": audited["prediction_flip_fraction"],
                            "final_states_exactly_equal_fraction": audited["final_states_exactly_equal_fraction"],
                            "identical_logits_fraction": audited["identical_logits_fraction"],
                            "pairs_zero_local_grad_but_logit_changes": audited["pairs_zero_local_grad_but_logit_changes"],
                            "pairs_zero_local_grad_but_state_changes": audited["pairs_zero_local_grad_but_state_changes"],
                            "boundary": audited.get("boundary"),
                        })
                        audit_count += 1
    summary = {
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "full_additive_delay64_successes": additive_ok,
        "delay_128_ran": any(row["delay"] == 128 for row in records),
        "completed_runs": sum(1 for row in records if row["status"] == "ok"),
        "failed_runs": sum(1 for row in records if row["status"] != "ok"),
        "unstarted": unstarted,
        "audit_rows": audit_count,
        "cohorts": [cohort_summary(records, model, regime, delay) for delay in (64, 128) for model in cfg["models"] for regime in cfg["regimes"] if any(row["delay"] == delay for row in records)],
        "paired": {str(delay): paired_regimes(records, delay) for delay in (64, 128)},
        "code_revision": revision,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "fingerprints.json").write_text(json.dumps(fingerprint_rows, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delay", "seed", "model_type", "regime", "status", "success", "test_accuracy", "retention_probe_r2"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    after_match = frozen_manifest() == json.loads(before_path.read_text(encoding="utf-8"))
    (out_dir / "frozen_hashes_after.json").write_text(json.dumps({"match": after_match}, indent=2), encoding="utf-8")
    if not after_match:
        raise RuntimeError("frozen artifacts changed during EXP-007")
    print(f"EXP-007 finished in {summary['elapsed_seconds']}s; full additive successes {additive_ok}; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

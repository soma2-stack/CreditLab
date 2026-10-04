"""EXP-011: gradient horizon crossed with clipping on or off.

Registration, before any accuracy:
    python experiments/run_exp011.py --write-registration

Training, after that registration is committed:
    python experiments/run_exp011.py
"""

from __future__ import annotations

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

from creditlab.diagnostics import hidden_magnitude_stats  # noqa: E402
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate, probe_fit_score  # noqa: E402
from run_exp004 import (  # noqa: E402
    DIAGNOSTIC_UPDATES,
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
from run_exp006 import bit_loss_gradient  # noqa: E402
from run_exp010 import (  # noqa: E402
    diagnose,
    forward_regime,
    training_graph_event,
)

CONFIG_PATH = ROOT / "configs" / "exp011_clipping_horizon.yaml"
OUT_DIR = ROOT / "results" / "EXP-011"
K = 16
CLIP_NORM = 5.0
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
BLOCKED = {
    "raw_metrics.jsonl", "runtime_session.json", "runtime_log.jsonl",
    "loss_curves.jsonl", "gradient_updates.jsonl", "summary.json",
}
ARTIFACT_DIRS = (
    "results/EXP-001", "results/EXP-001B", "results/EXP-002", "results/EXP-003",
    "results/EXP-004", "results/EXP-005", "results/EXP-006", "results/EXP-007",
    "results/EXP-008", "results/EXP-009", "results/EXP-010",
)
ARTIFACT_CONFIGS = (
    "configs/exp001_delayed_credit.yaml",
    "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml",
    "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml",
    "configs/exp004_bounded_additive.yaml",
    "configs/exp005_counterfactual_audit.yaml",
    "configs/exp006_replication.yaml",
    "configs/exp007_readout_only.yaml",
    "configs/exp008_parameter_groups.yaml",
    "configs/exp009_bias_ablation.yaml",
    "configs/exp010_truncated_bptt.yaml",
)


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def conditions(cfg: dict | None = None) -> list[dict]:
    source = load_config() if cfg is None else cfg
    return list(source["conditions"])


def float64_grad_norm(parameters) -> float:
    """Global L2 norm in float64. This does not change the gradients."""
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError(f"{parameter.shape} did not receive a training gradient")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite parameter gradient")
        flat = parameter.grad.detach().reshape(-1).double()
        total = total + torch.dot(flat, flat)
    value = torch.sqrt(total)
    if not torch.isfinite(value):
        raise FloatingPointError("non-finite gradient norm")
    return float(value)


def apply_clipping(parameters, enabled: bool) -> dict:
    raw = float64_grad_norm(parameters)
    record = {
        "raw_grad_norm_float64": raw,
        "would_exceed_5": bool(raw > CLIP_NORM),
        "clipping_applied": False,
        "applied_clip_scale": None,
        "implementation_grad_norm": None,
    }
    if not enabled:
        return record
    total = torch.nn.utils.clip_grad_norm_(list(parameters), CLIP_NORM)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite implementation gradient norm")
    total_value = float(total)
    scale = min(1.0, CLIP_NORM / (total_value + 1e-6))
    record["implementation_grad_norm"] = total_value
    record["applied_clip_scale"] = scale
    record["clipping_applied"] = bool(scale < 1.0)
    return record


def parameter_delta_norm(before: list[torch.Tensor], parameters) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for old, parameter in zip(before, parameters, strict=True):
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("non-finite parameter after update")
        flat = (parameter.detach() - old).reshape(-1).double()
        total = total + torch.dot(flat, flat)
    value = torch.sqrt(total)
    if not torch.isfinite(value):
        raise FloatingPointError("non-finite Adam update norm")
    return float(value)


def build_four(cfg: dict, seed: int) -> dict[str, ResidualTanhRNN]:
    models = {}
    for condition in conditions(cfg):
        seed_everything(seed)
        models[condition["name"]] = ResidualTanhRNN(
            cfg["model"]["input_size"], cfg["model"]["hidden_size"], cfg["model"]["output_size"],
            residual_scale=float(cfg["model"]["residual_scale"]),
        )
    return models


def train_one(cfg, model, condition, bundle, indices, delay, seed, init_fingerprint, data_fp, index_hash, per_run_cap):
    started = time.perf_counter()
    name = condition["name"]
    horizon = condition["horizon"]
    clip = bool(condition["clip"])
    updates_target = int(cfg["training"]["updates_per_run"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-011",
        "condition": name,
        "horizon": horizon,
        "clip": clip,
        "delay": delay,
        "seed": seed,
        "status": "ok",
        "stop_reason": None,
        "code_revision": git_revision(),
        "k": K,
        "init_fingerprint": init_fingerprint,
        "dataset_fingerprints": data_fp,
        "minibatch_fingerprint": index_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "distractor_query_channel_std": round(measured_noise_std(bundle["test_x"]), 6),
    }
    loss_curve = []
    checkpoints = []
    clipped = 0
    would_exceed = 0
    try:
        checkpoints.append(diagnose(model, bundle["val_x"], bundle["val_y"], 0, horizon, cfg, None))
        if horizon == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
            raise RuntimeError("the original event was not blocked in the final-16 training graph")
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > per_run_cap:
                raise TimeoutError(f"CPU budget of {per_run_cap:.0f}s exhausted at update {update}")
            xb, yb = bundle["train_x"][idx], bundle["train_y"][idx]
            logits, states, cut = forward_regime(model, xb, horizon)
            if horizon == "final_16" and cut != max(xb.shape[1] - K, 0):
                raise RuntimeError("final-16 cut does not match the registered rule")
            if not torch.isfinite(states).all():
                raise FloatingPointError(f"non-finite hidden state at update {update}")
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            clip_record = apply_clipping(list(model.parameters()), clip)
            if clip and clip_record["clipping_applied"]:
                clipped += 1
            if (not clip) and clip_record["clipping_applied"]:
                raise RuntimeError("clipping-off applied a gradient rescale")
            if clip_record["would_exceed_5"]:
                would_exceed += 1
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            update_norm = parameter_delta_norm(before, list(model.parameters()))
            loss_curve.append(float(loss.detach()))
            append_jsonl(OUT_DIR / "gradient_updates.jsonl", {
                "delay": delay,
                "seed": seed,
                "condition": name,
                "update": update + 1,
                "loss": float(loss.detach()),
                "adam_update_l2": update_norm,
                **clip_record,
            })
            if (update + 1) in DIAGNOSTIC_UPDATES:
                checkpoints.append(diagnose(
                    model, bundle["val_x"], bundle["val_y"], update + 1, horizon, cfg, None,
                ))
                if horizon == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
                    raise RuntimeError("the final-16 training graph stopped blocking the early inputs")
        tr = cfg["training"]
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"])
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"])
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"])
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        bit, marker = bit_and_marker_sensitivity(model, bundle["test_x"])
        forward_loss = bit_loss_gradient(model, bundle["test_x"], bundle["test_y"])
        graph = training_graph_event(model, bundle["test_x"], horizon)
        success = bool(test_acc >= float(cfg["interpretation"]["success_test_accuracy_at_least"]))
        record.update(
            updates_completed=updates_target,
            optimizer_steps=updates_target,
            train_loss=train_loss, train_accuracy=train_acc,
            val_loss=val_loss, val_accuracy=val_acc,
            test_loss=test_loss, test_accuracy=test_acc, success=success,
            retention_probe_r2=retention,
            max_abs_hidden=magnitude["max_abs_hidden"],
            final_hidden_rms=magnitude["final_hidden_rms"],
            forward_bit_sensitivity_mean_abs=float(bit.abs().mean()),
            forward_bit_sensitivity_exact_zeros=int((bit == 0).sum()),
            forward_marker_sensitivity_mean_abs=float(marker.abs().mean()),
            forward_bit_loss_gradient_mean=forward_loss["bit_loss_gradient_mean"],
            forward_bit_loss_gradient_exact_zeros=forward_loss["bit_loss_gradient_exact_zeros"],
            grad_clip_events=clipped,
            grad_clip_fraction=clipped / updates_target,
            would_exceed_5_events=would_exceed,
            would_exceed_5_fraction=would_exceed / updates_target,
            all_finite=True,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
        record.update(graph)
    except (FloatingPointError, TimeoutError, RuntimeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, TimeoutError):
            status = "budget_stop"
        elif "blocked" in str(exc) or "cut does not" in str(exc) or "did not receive" in str(exc) or "clipping-off applied" in str(exc):
            status = "implementation_bug"
        elif isinstance(exc, FloatingPointError) or "non-finite" in str(exc):
            status = "numerical_failure"
        else:
            status = "failure"
        record["status"] = status
        record["stop_reason"] = message
        record["updates_completed"] = len(loss_curve)
        record["optimizer_steps"] = len(loss_curve)
        record["success"] = False
        record["grad_clip_events"] = clipped
        record["would_exceed_5_events"] = would_exceed
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return record, loss_curve, checkpoints


def save_checkpoint(out_dir: Path, model, record: dict) -> str:
    destination = out_dir / "checkpoints" / f"additive_{record['condition']}_delay{record['delay']}_seed{record['seed']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_suffix(".pt.partial")
    torch.save({
        "experiment_id": "EXP-011",
        "condition": record["condition"],
        "horizon": record["horizon"],
        "clip": record["clip"],
        "delay": record["delay"],
        "seed": record["seed"],
        "k": K,
        "code_revision": record.get("code_revision"),
        "state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(destination)
    return file_sha256(destination)


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    for relative in ARTIFACT_CONFIGS:
        records[relative] = file_sha256(ROOT / relative)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-011 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    registration = {
        "experiment_id": "EXP-011",
        "kind": "preregistration",
        "k": K,
        "conditions": conditions(cfg),
        "seeds": SEEDS,
        "clip_norm": CLIP_NORM,
        "budget_seconds": 1200,
        "per_run_seconds": 300,
        "note": "Four fixed conditions. Clipping is either the existing norm-5 operation or absent. K stays 16.",
    }
    destination.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")
    manifest = artifact_manifest()
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered EXP-011; hashed {len(manifest)} preexisting files")


def coordinate_audit(model, positive, negative) -> dict:
    with torch.no_grad():
        logit_positive = model(positive)[0][:, -1, 0]
        logit_negative = model(negative)[0][:, -1, 0]
        final_positive = model.recurrent_states(positive)[:, -1, :]
        final_negative = model.recurrent_states(negative)[:, -1, :]
    equal = (final_positive == final_negative).all(dim=-1)
    finite = torch.isfinite(logit_positive) & torch.isfinite(logit_negative) & (logit_positive != logit_negative)
    return {
        "final_states_coordinate_equal_fraction": float(equal.float().mean()),
        "finite_logit_difference_fraction": float(finite.float().mean()),
    }


def paired_counts(records, delay: int) -> dict:
    def success_map(condition: str) -> dict[int, bool | None]:
        found = {}
        for row in records:
            if row["delay"] == delay and row["condition"] == condition:
                found[int(row["seed"])] = bool(row.get("success")) if row["status"] == "ok" else None
        return found

    def compare(left_name: str, right_name: str) -> dict:
        left = success_map(left_name)
        right = success_map(right_name)
        both = left_only = right_only = neither = incomplete = 0
        for seed in SEEDS:
            a = left.get(seed)
            b = right.get(seed)
            if a is None or b is None:
                incomplete += 1
            elif a and b:
                both += 1
            elif a:
                left_only += 1
            elif b:
                right_only += 1
            else:
                neither += 1
        return {
            "left": left_name, "right": right_name,
            "both_succeed": both, "left_only": left_only, "right_only": right_only,
            "neither": neither, "incomplete": incomplete,
        }

    return {
        "delay": delay,
        "full_versus_final16_with_clipping": compare("full_history_clip5", "final_16_clip5"),
        "full_versus_final16_without_clipping": compare("full_history_noclip", "final_16_noclip"),
        "clipping_versus_none_within_full_history": compare("full_history_clip5", "full_history_noclip"),
        "clipping_versus_none_within_final_16": compare("final_16_clip5", "final_16_noclip"),
    }


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    if int(cfg["gradient_horizon"]["k"]) != K:
        raise RuntimeError("K was changed after preregistration")
    blockers = [name for name in BLOCKED if (OUT_DIR / name).exists()]
    if blockers:
        raise RuntimeError(f"EXP-011 outputs already exist ({', '.join(blockers)}). Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-011")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-011",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": 1200,
        "k": K,
    }, indent=2, sort_keys=True), encoding="utf-8")
    records = []
    unstarted = []
    stopped = None
    condition_list = conditions(cfg)

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def expired() -> bool:
        return time.perf_counter() - started >= 1200

    def run_delay(delay: int) -> None:
        nonlocal stopped
        for seed in cfg["seeds"]:
            pending = [condition for condition in condition_list if not any(
                row["delay"] == delay and row["seed"] == seed and row["condition"] == condition["name"] for row in records
            )]
            if expired() or stopped:
                for condition in pending:
                    unstarted.append({"delay": delay, "seed": seed, "condition": condition["name"], "reason": "implementation stop" if stopped else "total cap"})
                continue
            bundle = dataset_bundle(cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(
                seed, delay, bundle["train_x"].shape[0],
                cfg["training"]["batch_size"], cfg["training"]["updates_per_run"],
            )
            index_hash = index_fingerprint(indices)
            models = build_four(cfg, seed)
            reference = parameter_fingerprint(next(iter(models.values())))
            if any(parameter_fingerprint(model) != reference for model in models.values()):
                raise RuntimeError(f"initial parameters differ at delay {delay} seed {seed}")
            with torch.no_grad():
                probe = bundle["train_x"][:4]
                reference_logits, _ = next(iter(models.values()))(probe)
            for model in models.values():
                with torch.no_grad():
                    logits, _ = model(probe)
                if not torch.equal(logits, reference_logits):
                    raise RuntimeError("initial forward outputs differ")
            append_jsonl(OUT_DIR / "fingerprints.jsonl", {
                "delay": delay, "seed": seed, "init_fingerprint": reference,
                "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
            })
            for condition in condition_list:
                if expired() or stopped:
                    unstarted.append({"delay": delay, "seed": seed, "condition": condition["name"], "reason": "implementation stop" if stopped else "total cap"})
                    continue
                remaining = 1200 - (time.perf_counter() - started)
                per_run_cap = min(300.0, remaining)
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "start", "delay": delay, "seed": seed, "condition": condition["name"],
                    "started_unix": time.time(), "session_elapsed_seconds": elapsed(),
                    "applied_per_run_cap_seconds": per_run_cap,
                })
                print(f"training {condition['name']} delay {delay} seed {seed}", flush=True)
                record, curve, checks = train_one(
                    cfg, models[condition["name"]], condition, bundle, indices, delay, seed,
                    reference, data_fp, index_hash, per_run_cap,
                )
                if record["status"] == "implementation_bug":
                    stopped = {"delay": delay, "seed": seed, "condition": condition["name"], "reason": record["stop_reason"]}
                if record["status"] == "ok":
                    record["checkpoint_sha256"] = save_checkpoint(OUT_DIR, models[condition["name"]], record)
                record["session_elapsed_seconds"] = elapsed()
                records.append(record)
                append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
                append_jsonl(OUT_DIR / "loss_curves.jsonl", {
                    "delay": delay, "seed": seed, "condition": condition["name"], "loss_curve": curve,
                })
                for item in checks:
                    append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {
                        "delay": delay, "seed": seed, "condition": condition["name"], **item,
                    })
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "finish", "delay": delay, "seed": seed, "condition": condition["name"],
                    "status": record["status"], "updates_completed": record.get("updates_completed"),
                    "runtime_seconds": record.get("runtime_seconds"),
                    "session_elapsed_seconds": elapsed(),
                })
                print(json.dumps({
                    "delay": delay, "seed": seed, "condition": condition["name"], "status": record["status"],
                    "success": record.get("success"), "test_accuracy": record.get("test_accuracy"),
                }, sort_keys=True), flush=True)

    run_delay(64)
    control_successes = sum(
        1 for row in records
        if row["delay"] == 64 and row["condition"] == "full_history_clip5" and row.get("success")
    )
    delay64_terminal = sum(1 for row in records if row["delay"] == 64)
    gate = delay64_terminal == 40 and control_successes >= 8 and stopped is None
    (OUT_DIR / "delay64_summary.json").write_text(json.dumps({
        "full_history_clip5_successes": control_successes,
        "delay64_terminal_records": delay64_terminal,
        "delay_128": "will run" if gate else "not run",
        "written_before_delay_128": True,
        "accuracies": {
            condition["name"]: {
                str(row["seed"]): row.get("test_accuracy")
                for row in records if row["delay"] == 64 and row["condition"] == condition["name"]
            }
            for condition in condition_list
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    if gate and not expired():
        run_delay(128)
    elif gate:
        for seed in cfg["seeds"]:
            for condition in condition_list:
                unstarted.append({"delay": 128, "seed": seed, "condition": condition["name"], "reason": "total cap"})
    audit_rows = []
    if stopped is None:
        for delay in sorted({row["delay"] for row in records}):
            for seed in cfg["seeds"]:
                ready = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["status"] == "ok"]
                if not ready:
                    continue
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "condition": "audit", "reason": "total cap"})
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
                for row in ready:
                    if expired():
                        unstarted.append({"delay": delay, "seed": seed, "condition": row["condition"], "reason": "audit cap"})
                        continue
                    path = OUT_DIR / "checkpoints" / f"additive_{row['condition']}_delay{delay}_seed{seed}.pt"
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    seed_everything(0)
                    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    before_hash = parameter_fingerprint(model)
                    audited = audit_one(model, "additive_tanh_rnn", positive, negative)
                    direct = coordinate_audit(model, positive, negative)
                    if parameter_fingerprint(model) != before_hash:
                        raise RuntimeError("audit changed parameters")
                    audit_record = {
                        "delay": delay, "seed": seed, "condition": row["condition"], "status": "ok",
                        "measurement": "unmodified_forward",
                        "accuracy": audited["accuracy"],
                        "both_correct_fraction": audited["both_correct_fraction"],
                        "prediction_flip_fraction": audited["prediction_flip_fraction"],
                        "identical_logits_fraction": audited["identical_logits_fraction"],
                        **direct,
                    }
                    audit_rows.append(audit_record)
                    append_jsonl(OUT_DIR / "audit_metrics.jsonl", audit_record)
    delays_present = sorted({row["delay"] for row in records})
    summary = {
        "elapsed_seconds": elapsed(),
        "timing_method": "time.perf_counter inside this process",
        "full_history_clip5_delay64_successes": control_successes,
        "delay64_terminal_records": delay64_terminal,
        "delay_128_ran": any(row["delay"] == 128 for row in records),
        "completed_runs": sum(1 for row in records if row["status"] == "ok"),
        "numerical_failures": sum(1 for row in records if row["status"] == "numerical_failure"),
        "budget_stops": sum(1 for row in records if row["status"] == "budget_stop"),
        "failed_runs": sum(1 for row in records if row["status"] != "ok"),
        "unstarted": unstarted,
        "implementation_stop": stopped,
        "audit_rows": len(audit_rows),
        "paired": [paired_counts(records, delay) for delay in delays_present],
        "code_revision": revision,
        "k": K,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "paired_comparison.json").write_text(json.dumps(summary["paired"], indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["delay", "seed", "condition", "horizon", "clip", "status", "success", "test_accuracy", "test_loss", "retention_probe_r2", "grad_clip_fraction", "would_exceed_5_fraction", "runtime_seconds", "stop_reason"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-011")
    if stopped:
        raise RuntimeError(f"implementation stop: {stopped}")
    print(
        f"EXP-011 finished in {summary['elapsed_seconds']}s; "
        f"clip5 full-history successes {control_successes}; unstarted {len(unstarted)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

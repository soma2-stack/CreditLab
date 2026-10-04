"""EXP-010: full-history training versus a final-16-step gradient cutoff.

The forward equation is unchanged. Only the training graph is cut.

Registration, before any accuracy:
    python experiments/run_exp010.py --write-registration

Training, after that registration is committed:
    python experiments/run_exp010.py
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

CONFIG_PATH = ROOT / "configs" / "exp010_truncated_bptt.yaml"
OUT_DIR = ROOT / "results" / "EXP-010"
K = 16
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
REGIMES = ["full_history", "final_16"]
BLOCKED = {
    "raw_metrics.jsonl", "runtime_session.json", "runtime_log.jsonl",
    "loss_curves.jsonl", "summary.json",
}
ARTIFACT_DIRS = (
    "results/EXP-001", "results/EXP-001B", "results/EXP-002", "results/EXP-003",
    "results/EXP-004", "results/EXP-005", "results/EXP-006", "results/EXP-007",
    "results/EXP-008", "results/EXP-009",
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
)


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def cut_index(length: int, k: int = K) -> int:
    """First timestep inside the gradient horizon. Zero means no detach."""
    return max(int(length) - int(k), 0)


def rollout(model: ResidualTanhRNN, sequence: torch.Tensor, cut: int):
    """Additive rollout. A positive cut detaches the state once, without changing it."""
    batch, length, _ = sequence.shape
    hidden = sequence.new_zeros(batch, model.hidden_size)
    steps = []
    for time_index in range(length):
        if cut > 0 and time_index == cut:
            hidden = hidden.detach()
        hidden, _candidate = model.candidate_step(sequence[:, time_index, :], hidden)
        steps.append(hidden)
    states = torch.stack(steps, dim=1)
    return model.readout(states), states


def forward_regime(model, sequence: torch.Tensor, regime: str):
    cut = cut_index(sequence.shape[1]) if regime == "final_16" else 0
    logits, states = rollout(model, sequence, cut)
    return logits, states, cut


def reference_final_logit(sequence, weight_ih, weight_hh, bias_ih, bias_hh, readout_weight, readout_bias):
    """Explicit additive rollout used only to check gradients."""
    batch, length, _features = sequence.shape
    hidden = sequence.new_zeros(batch, weight_hh.shape[0])
    for time_index in range(length):
        current = sequence[:, time_index, :]
        preactivation = (
            current.matmul(weight_ih.T)
            + bias_ih
            + hidden.matmul(weight_hh.T)
            + bias_hh
        )
        hidden = hidden + torch.tanh(preactivation)
    return hidden.matmul(readout_weight.T) + readout_bias


def build_pair(cfg: dict, seed: int) -> dict[str, ResidualTanhRNN]:
    models = {}
    for _regime in REGIMES:
        seed_everything(seed)
        models[_regime] = ResidualTanhRNN(
            cfg["model"]["input_size"],
            cfg["model"]["hidden_size"],
            cfg["model"]["output_size"],
            residual_scale=float(cfg["model"]["residual_scale"]),
        )
    return models


def group_grad_norms(model) -> dict[str, float]:
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} did not receive a training gradient")
        norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach()))
    return norms


def training_graph_event(model, inputs: torch.Tensor, regime: str) -> dict:
    probe = inputs.detach().clone().requires_grad_(True)
    logits, _states, cut = forward_regime(model, probe, regime)
    (grad,) = torch.autograd.grad(logits[:, -1, 0].sum(), probe)
    event = grad[:, 0, 0]
    late = grad[:, cut:, :]
    early_max = 0.0 if cut == 0 else float(grad[:, :cut, :].abs().max())
    return {
        "gradient_cut": cut,
        "training_graph_event_bit_mean_abs": float(event.abs().mean()),
        "training_graph_event_bit_exact_zeros": int((event == 0).sum()),
        "training_graph_early_input_grad_max_abs": early_max,
        "training_graph_late_input_grad_mean_abs": float(late.abs().mean()),
        "training_graph_event_blocked": bool(cut > 0 and torch.count_nonzero(grad[:, :cut, :]) == 0),
    }


def diagnose(model, inputs, targets, update: int, regime: str, cfg: dict, step_norms: dict | None) -> dict:
    record = {"update": update, "regime": regime, "gradient_kind_note": "forward_ fields have no detach; training_graph_ fields use the regime graph"}
    record["retention_probe_r2"] = probe_fit_score(model, inputs[:256], targets[:256], inputs[256:], targets[256:])
    val_loss, val_acc = evaluate(model, inputs, targets, cfg["training"]["eval_batch_size"])
    record["val_loss"] = val_loss
    record["val_accuracy"] = val_acc
    bit, marker = bit_and_marker_sensitivity(model, inputs)
    record["forward_bit_sensitivity_mean_abs"] = float(bit.abs().mean())
    record["forward_bit_sensitivity_exact_zeros"] = int((bit == 0).sum())
    record["forward_marker_sensitivity_mean_abs"] = float(marker.abs().mean())
    forward_loss = bit_loss_gradient(model, inputs, targets)
    record["forward_bit_loss_gradient_mean"] = forward_loss["bit_loss_gradient_mean"]
    record["forward_bit_loss_gradient_exact_zeros"] = forward_loss["bit_loss_gradient_exact_zeros"]
    record["forward_marker_loss_gradient_mean"] = forward_loss["marker_loss_gradient_mean"]
    magnitude = hidden_magnitude_stats(model, inputs)
    record["max_abs_hidden"] = magnitude["max_abs_hidden"]
    record["final_hidden_rms"] = magnitude["final_hidden_rms"]
    record.update(training_graph_event(model, inputs, regime))
    if step_norms is not None:
        record["training_step_grad_norms"] = step_norms
    return json_safe(record)


def train_one(cfg, model, regime, bundle, indices, delay, seed, init_fingerprint, data_fp, index_hash, per_run_cap):
    started = time.perf_counter()
    updates_target = int(cfg["training"]["updates_per_run"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    if len(optimizer.param_groups[0]["params"]) != sum(1 for _parameter in model.parameters()):
        raise RuntimeError("Adam did not receive every parameter")
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-010",
        "regime": regime,
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
    try:
        checkpoints.append(diagnose(model, bundle["val_x"], bundle["val_y"], 0, regime, cfg, None))
        if regime == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
            raise RuntimeError("the original event was not blocked in the final-16 training graph")
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > per_run_cap:
                raise TimeoutError(f"CPU budget of {per_run_cap:.0f}s exhausted at update {update}")
            xb, yb = bundle["train_x"][idx], bundle["train_y"][idx]
            logits, _states, cut = forward_regime(model, xb, regime)
            if regime == "final_16" and cut != cut_index(xb.shape[1]):
                raise RuntimeError("final-16 cut does not match the registered rule")
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            step_norms = group_grad_norms(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()), float(cfg["training"]["grad_clip_norm"]),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at update {update}")
            if float(grad_norm) > float(cfg["training"]["grad_clip_norm"]):
                clipped += 1
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in DIAGNOSTIC_UPDATES:
                checkpoints.append(diagnose(
                    model, bundle["val_x"], bundle["val_y"], update + 1, regime, cfg, step_norms,
                ))
                if regime == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
                    raise RuntimeError("the final-16 training graph stopped blocking the early inputs")
        tr = cfg["training"]
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"])
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"])
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"])
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        bit, marker = bit_and_marker_sensitivity(model, bundle["test_x"])
        forward_loss = bit_loss_gradient(model, bundle["test_x"], bundle["test_y"])
        graph = training_graph_event(model, bundle["test_x"], regime)
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
            all_finite=True,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
        record.update(graph)
    except (FloatingPointError, TimeoutError, RuntimeError) as exc:
        record["status"] = "budget_stop" if isinstance(exc, TimeoutError) else "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["updates_completed"] = len(loss_curve)
        record["optimizer_steps"] = len(loss_curve)
        record["success"] = False
        record["grad_clip_events"] = clipped
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
        if any(phrase in str(exc) for phrase in ("blocked", "cut does not", "did not receive")):
            record["status"] = "implementation_bug"
    return record, loss_curve, checkpoints


def save_checkpoint(model, record: dict) -> str:
    destination = OUT_DIR / "checkpoints" / f"additive_{record['regime']}_delay{record['delay']}_seed{record['seed']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_suffix(".pt.partial")
    torch.save({
        "experiment_id": "EXP-010",
        "regime": record["regime"],
        "delay": record["delay"],
        "seed": record["seed"],
        "k": K,
        "code_revision": record.get("code_revision"),
        "init_fingerprint": record.get("init_fingerprint"),
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
        raise FileExistsError("refusing to overwrite the EXP-010 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registration = {
        "experiment_id": "EXP-010",
        "kind": "preregistration",
        "k": K,
        "regimes": REGIMES,
        "seeds": SEEDS,
        "forward_match": "direct_equality",
        "budget_seconds": 600,
        "per_run_seconds": 300,
        "note": "K stays 16. This is a backward cutoff, not a new model.",
    }
    destination.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")
    manifest = artifact_manifest()
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered EXP-010; hashed {len(manifest)} preexisting files")


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
        "final_states_coordinate_equal_count": int(equal.sum()),
        "finite_logit_difference_count": int(finite.sum()),
        "finite_logit_difference_fraction": float(finite.float().mean()),
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
        raise RuntimeError(f"EXP-010 outputs already exist ({', '.join(blockers)}). Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-010")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-010",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": 600,
        "k": K,
    }, indent=2, sort_keys=True), encoding="utf-8")
    records = []
    unstarted = []
    stopped = None

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def expired() -> bool:
        return time.perf_counter() - started >= 600

    def run_delay(delay: int) -> None:
        nonlocal stopped
        for seed in cfg["seeds"]:
            if expired() or stopped:
                for regime in REGIMES:
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "implementation stop" if stopped else "total cap"})
                continue
            bundle = dataset_bundle(cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(
                seed, delay, bundle["train_x"].shape[0],
                cfg["training"]["batch_size"], cfg["training"]["updates_per_run"],
            )
            index_hash = index_fingerprint(indices)
            models = build_pair(cfg, seed)
            reference = parameter_fingerprint(models["full_history"])
            if any(parameter_fingerprint(model) != reference for model in models.values()):
                raise RuntimeError(f"initial parameters differ at delay {delay} seed {seed}")
            with torch.no_grad():
                probe = bundle["train_x"][:4]
                full_logits, _ = models["full_history"](probe)
                other_logits, _states, _cut = forward_regime(models["final_16"], probe, "final_16")
            if not torch.equal(full_logits, other_logits):
                raise RuntimeError("forward outputs are not identical before training")
            append_jsonl(OUT_DIR / "fingerprints.jsonl", {
                "delay": delay, "seed": seed, "init_fingerprint": reference,
                "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
                "gradient_cut": cut_index(bundle["train_x"].shape[1]),
            })
            for regime in REGIMES:
                if expired() or stopped:
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "implementation stop" if stopped else "total cap"})
                    continue
                remaining = 600 - (time.perf_counter() - started)
                per_run_cap = min(300.0, remaining)
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "start", "delay": delay, "seed": seed, "regime": regime,
                    "started_unix": time.time(), "session_elapsed_seconds": elapsed(),
                    "applied_per_run_cap_seconds": per_run_cap,
                })
                print(f"training {regime} delay {delay} seed {seed}", flush=True)
                record, curve, checks = train_one(
                    cfg, models[regime], regime, bundle, indices, delay, seed,
                    reference, data_fp, index_hash, per_run_cap,
                )
                if record["status"] == "implementation_bug":
                    stopped = {"delay": delay, "seed": seed, "regime": regime, "reason": record["stop_reason"]}
                if record["status"] == "ok":
                    record["checkpoint_sha256"] = save_checkpoint(models[regime], record)
                record["session_elapsed_seconds"] = elapsed()
                records.append(record)
                append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
                append_jsonl(OUT_DIR / "loss_curves.jsonl", {
                    "delay": delay, "seed": seed, "regime": regime, "loss_curve": curve,
                })
                for item in checks:
                    append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {
                        "delay": delay, "seed": seed, "regime": regime, **item,
                    })
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "finish", "delay": delay, "seed": seed, "regime": regime,
                    "status": record["status"], "updates_completed": record.get("updates_completed"),
                    "runtime_seconds": record.get("runtime_seconds"),
                    "session_elapsed_seconds": elapsed(),
                })
                print(json.dumps({
                    "delay": delay, "seed": seed, "regime": regime, "status": record["status"],
                    "success": record.get("success"), "test_accuracy": record.get("test_accuracy"),
                }, sort_keys=True), flush=True)

    run_delay(64)
    full_successes = sum(
        1 for row in records
        if row["delay"] == 64 and row["regime"] == "full_history" and row.get("success")
    )
    delay64_terminal = sum(1 for row in records if row["delay"] == 64)
    gate = delay64_terminal == 20 and full_successes >= 8 and stopped is None
    (OUT_DIR / "delay64_summary.json").write_text(json.dumps({
        "full_history_successes": full_successes,
        "delay64_terminal_records": delay64_terminal,
        "delay_128": "will run" if gate else "not run",
        "written_before_delay_128": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    if gate and not expired():
        run_delay(128)
    elif gate:
        for seed in cfg["seeds"]:
            for regime in REGIMES:
                unstarted.append({"delay": 128, "seed": seed, "regime": regime, "reason": "total cap"})
    audit_rows = []
    if stopped is None:
        for delay in sorted({row["delay"] for row in records}):
            for seed in cfg["seeds"]:
                ready = [
                    row for row in records
                    if row["delay"] == delay and row["seed"] == seed and row["status"] == "ok"
                ]
                if len(ready) < 2:
                    continue
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "regime": "audit", "reason": "total cap"})
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
                    raise RuntimeError("pair construction failed")
                for regime in REGIMES:
                    if expired():
                        unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "audit cap"})
                        continue
                    path = OUT_DIR / "checkpoints" / f"additive_{regime}_delay{delay}_seed{seed}.pt"
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
                        "delay": delay, "seed": seed, "regime": regime, "status": "ok",
                        "measurement": "unmodified_forward",
                        "accuracy": audited["accuracy"],
                        "both_correct_fraction": audited["both_correct_fraction"],
                        "exactly_one_correct_fraction": audited["exactly_one_correct_fraction"],
                        "both_wrong_fraction": audited["both_wrong_fraction"],
                        "prediction_flip_fraction": audited["prediction_flip_fraction"],
                        "identical_logits_fraction": audited["identical_logits_fraction"],
                        **direct,
                    }
                    audit_rows.append(audit_record)
                    append_jsonl(OUT_DIR / "audit_metrics.jsonl", audit_record)
    paired = []
    for delay in sorted({row["delay"] for row in records}):
        for seed in cfg["seeds"]:
            item = {"delay": delay, "seed": seed}
            for regime in REGIMES:
                match = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["regime"] == regime]
                item[regime] = None if not match else {
                    "status": match[0]["status"],
                    "success": match[0].get("success"),
                    "test_accuracy": match[0].get("test_accuracy"),
                }
            paired.append(item)
    summary = {
        "elapsed_seconds": elapsed(),
        "timing_method": "time.perf_counter inside this process",
        "full_history_delay64_successes": full_successes,
        "delay64_terminal_records": delay64_terminal,
        "delay_128_ran": any(row["delay"] == 128 for row in records),
        "completed_runs": sum(1 for row in records if row["status"] == "ok"),
        "failed_runs": sum(1 for row in records if row["status"] != "ok"),
        "unstarted": unstarted,
        "implementation_stop": stopped,
        "audit_rows": len(audit_rows),
        "paired": paired,
        "code_revision": revision,
        "k": K,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "paired_comparison.json").write_text(json.dumps(paired, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["delay", "seed", "regime", "status", "success", "test_accuracy", "retention_probe_r2", "grad_clip_fraction", "runtime_seconds", "stop_reason"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-010")
    if stopped:
        raise RuntimeError(f"implementation stop: {stopped}")
    print(
        f"EXP-010 finished in {summary['elapsed_seconds']}s; "
        f"full-history delay-64 successes {full_successes}; unstarted {len(unstarted)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

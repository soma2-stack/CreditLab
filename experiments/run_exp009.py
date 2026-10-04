"""EXP-009: bias adaptation versus matrix adaptation on the additive model.

Usage (from the repository root, after the preregistration commit):
    python experiments/run_exp009.py --write-frozen-hashes
    python experiments/run_exp009.py

This runner does not resume an interrupted session and does not estimate
earlier runtime from file timestamps.
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
from run_exp006 import bit_loss_gradient, clopper_pearson  # noqa: E402

class ImplementationError(RuntimeError):
    """A training-rule failure. It is not caught as an ordinary numerical failure."""


CONFIG_PATH = ROOT / "configs" / "exp009_bias_ablation.yaml"
OUT_DIR = ROOT / "results" / "EXP-009"
MEASUREMENT_NAMES = {
    "raw_metrics.jsonl",
    "loss_curves.jsonl",
    "checkpoint_diagnostics.jsonl",
    "audit_metrics.jsonl",
    "fingerprints.jsonl",
    "bias_records.jsonl",
    "runtime_log.jsonl",
    "runtime_session.json",
    "summary.json",
    "summary.csv",
    "delay64_summary.json",
    "paired_comparison.json",
    "config_used.yaml",
    "environment.txt",
    "frozen_hashes_after.json",
}
FROZEN_PATHS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "results/EXP-004",
    "results/EXP-005",
    "results/EXP-006",
    "results/EXP-007",
    "results/EXP-008",
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


def measurement_blockers(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.name in MEASUREMENT_NAMES or path.name == "checkpoints")


def apply_freeze(model, frozen_names: list[str]) -> None:
    present = {name for name, _ in model.named_parameters()}
    missing = [name for name in frozen_names if name not in present]
    if missing:
        raise RuntimeError(f"missing parameters: {missing}")
    frozen = set(frozen_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in frozen)


def trainable_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def group_fingerprint(model, names: list[str]) -> str:
    tensors = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_four(cfg: dict, seed: int) -> dict:
    models = {}
    for regime, frozen_names in cfg["frozen_by_regime"].items():
        seed_everything(seed)
        model = ResidualTanhRNN(
            cfg["model"]["input_size"],
            cfg["model"]["hidden_size"],
            cfg["model"]["output_size"],
            residual_scale=float(cfg["model"]["residual_scale"]),
        )
        apply_freeze(model, frozen_names)
        models[regime] = model
    return models


def verify_regime_lists(cfg: dict, models: dict) -> None:
    all_names = [name for name, _ in models["full"].named_parameters()]
    if all_names != cfg["all_parameters"]:
        raise RuntimeError(f"parameter names differ from the preregistration: {all_names}")
    total = sum(parameter.numel() for parameter in models["full"].parameters())
    if total != int(cfg["expected_total_parameters"]):
        raise RuntimeError(f"total parameter count is {total}")
    for regime, model in models.items():
        expected = list(cfg["trainable_by_regime"][regime])
        actual = trainable_names(model)
        if actual != expected:
            raise RuntimeError(f"{regime} trainable names are {actual}")
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if count != int(cfg["expected_trainable_counts"][regime]):
            raise RuntimeError(f"{regime} trainable count is {count}")
        frozen_actual = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if frozen_actual != list(cfg["frozen_by_regime"][regime]):
            raise RuntimeError(f"{regime} frozen names are {frozen_actual}")


def describe_parameter_sets(cfg: dict) -> dict:
    models = build_four(cfg, int(cfg["seeds"][0]))
    verify_regime_lists(cfg, models)
    return {
        "seed_used_for_count_check": int(cfg["seeds"][0]),
        "total_parameters": int(cfg["expected_total_parameters"]),
        "single_trainable_preactivation_bias": cfg["single_trainable_preactivation_bias"],
        "regimes": {
            regime: {
                "trainable": list(cfg["trainable_by_regime"][regime]),
                "frozen": list(cfg["frozen_by_regime"][regime]),
                "trainable_count": int(cfg["expected_trainable_counts"][regime]),
            }
            for regime in cfg["regimes"]
        },
    }


def write_parameter_sets(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(describe_parameter_sets(load_config()), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def checkpoint_plus(model, val_x, val_y, update: int, cfg: dict, clipped: int) -> dict:
    record = {
        "update": update,
        "split": "validation",
        "retention_probe_r2": probe_fit_score(model, val_x[:256], val_y[:256], val_x[256:], val_y[256:]),
        "grad_clip_events_so_far": clipped,
        "grad_clip_fraction_so_far": 0.0 if update == 0 else clipped / update,
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


def change_norms(before: dict[str, torch.Tensor], model) -> dict[str, float]:
    current = dict(model.named_parameters())
    return {
        name: float(torch.linalg.vector_norm(current[name].detach() - value))
        for name, value in before.items()
    }


def bias_record(cfg: dict, model, before: dict[str, torch.Tensor], regime: str, delay: int, seed: int) -> dict:
    current = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    bx_name = cfg["parameter_names"]["b_x"]
    bh_name = cfg["parameter_names"]["b_h"]
    wx_name = cfg["parameter_names"]["W_x"]
    wh_name = cfg["parameter_names"]["W_h"]
    effective_before = before[bx_name] + before[bh_name]
    effective_after = current[bx_name] + current[bh_name]
    record = {
        "delay": delay,
        "seed": seed,
        "regime": regime,
        "preactivation_bias_name": bx_name,
        "frozen_preactivation_bias_name": bh_name,
        "readout_bias_name": cfg["parameter_names"]["readout_bias"],
        "parameter_change_l2": change_norms(before, model),
        "W_x_hash_before": group_fingerprint_from(before, [wx_name]),
        "W_x_hash_after": group_fingerprint_from(current, [wx_name]),
        "W_h_hash_before": group_fingerprint_from(before, [wh_name]),
        "W_h_hash_after": group_fingerprint_from(current, [wh_name]),
        "b_x_hash_before": group_fingerprint_from(before, [bx_name]),
        "b_x_hash_after": group_fingerprint_from(current, [bx_name]),
        "b_h_hash_before": group_fingerprint_from(before, [bh_name]),
        "b_h_hash_after": group_fingerprint_from(current, [bh_name]),
    }
    if regime == "bias_plus_readout":
        record.update(
            initial_b_x=before[bx_name].tolist(),
            final_b_x=current[bx_name].tolist(),
            initial_effective_bias=effective_before.tolist(),
            final_effective_bias=effective_after.tolist(),
            effective_bias_change_l2=float(torch.linalg.vector_norm(effective_after - effective_before)),
            effective_bias_change_equals_b_x_change=bool(torch.equal(
                effective_after - effective_before, current[bx_name] - before[bx_name],
            )),
            matrices_unchanged=bool(
                torch.equal(current[wx_name], before[wx_name]) and torch.equal(current[wh_name], before[wh_name])
            ),
        )
    if regime == "matrices_plus_readout":
        record["both_preactivation_biases_unchanged"] = bool(
            torch.equal(current[bx_name], before[bx_name]) and torch.equal(current[bh_name], before[bh_name])
        )
    return record


def group_fingerprint_from(tensors: dict[str, torch.Tensor], names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_regime(cfg, model, regime, bundle, indices, delay, seed, phase, init_fingerprint, data_fp, index_hash):
    started = time.perf_counter()
    budget = float(cfg["budget"]["max_cpu_seconds_per_run"])
    updates_target = int(cfg["training"]["updates_per_run"])
    frozen_names = list(cfg["frozen_by_regime"][regime])
    apply_freeze(model, frozen_names)
    if trainable_names(model) != list(cfg["trainable_by_regime"][regime]):
        raise RuntimeError(f"{regime} trainable set does not match the preregistration")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    frozen_before = group_fingerprint(model, frozen_names) if frozen_names else None
    hidden_before = None
    if regime == "readout_only":
        with torch.no_grad():
            hidden_before = model.recurrent_states(bundle["val_x"][:32]).clone()
    record = {
        "experiment_id": "EXP-009",
        "regime": regime,
        "delay": delay,
        "seed": seed,
        "phase": phase,
        "status": "ok",
        "stop_reason": None,
        "code_revision": git_revision(),
        "init_fingerprint": init_fingerprint,
        "dataset_fingerprints": data_fp,
        "minibatch_fingerprint": index_hash,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "grad_clip_scope": cfg["training"]["grad_clip_scope"],
        "distractor_query_channel_std": round(measured_noise_std(bundle["test_x"]), 6),
    }
    tr = cfg["training"]
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(tr["learning_rate"]),
    )
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimized) != sum(1 for parameter in model.parameters() if parameter.requires_grad):
        raise ImplementationError("Adam received a parameter outside the trainable set")
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss_curve = []
    checkpoints = []
    clipped = 0
    try:
        checkpoints.append(checkpoint_plus(model, bundle["val_x"], bundle["val_y"], 0, cfg, clipped))
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
            unintended = [
                name for name, parameter in model.named_parameters()
                if (not parameter.requires_grad) and parameter.grad is not None
            ]
            if unintended:
                raise ImplementationError(f"frozen parameters received gradients: {unintended}")
            missing_grad = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
            if missing_grad:
                raise ImplementationError(f"trainable parameters missed gradients: {missing_grad}")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(tr["grad_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at update {update}")
            if float(grad_norm) > float(tr["grad_clip_norm"]):
                clipped += 1
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in DIAGNOSTIC_UPDATES:
                checkpoints.append(checkpoint_plus(model, bundle["val_x"], bundle["val_y"], update + 1, cfg, clipped))
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
            grad_clip_events=clipped,
            grad_clip_fraction=clipped / updates_target,
            parameter_change_l2=change_norms(before, model),
            frozen_hash_before=frozen_before,
            frozen_hash_after=frozen_after,
            frozen_parameters_unchanged=frozen_before == frozen_after,
            hidden_states_unchanged=hidden_unchanged,
            all_finite=True,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
        record.update(bit_loss)
        details = bias_record(cfg, model, before, regime, delay, seed)
    except (FloatingPointError, TimeoutError, RuntimeError) as exc:
        if isinstance(exc, ImplementationError):
            raise
        record["status"] = "budget_stop" if isinstance(exc, TimeoutError) else "failure"
        record["stop_reason"] = f"{type(exc).__name__}: {exc}"
        record["updates_completed"] = len(loss_curve)
        record["success"] = False
        record["grad_clip_events"] = clipped
        record["runtime_seconds"] = round(time.perf_counter() - started, 3)
        details = None
    return record, loss_curve, checkpoints, details


def save_checkpoint(out_dir: Path, model, record: dict) -> str:
    destination = out_dir / "checkpoints" / f"additive_{record['regime']}_delay{record['delay']}_seed{record['seed']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    torch.save({
        "experiment_id": "EXP-009",
        "regime": record["regime"],
        "delay": record["delay"],
        "seed": record["seed"],
        "code_revision": record.get("code_revision"),
        "init_fingerprint": record.get("init_fingerprint"),
        "trainable_parameter_names": record.get("trainable_parameter_names"),
        "frozen_parameter_names": record.get("frozen_parameter_names"),
        "state_dict": model.state_dict(),
    }, destination)
    return file_sha256(destination)


def cohort_summary(records, regime, delay):
    group = [row for row in records if row["regime"] == regime and row["delay"] == delay and row["status"] == "ok"]
    accuracies = [float(row["test_accuracy"]) for row in group]
    successes = sum(1 for row in group if row.get("success"))
    summary = {
        "regime": regime,
        "delay": delay,
        "completed": len(group),
        "terminal_records": sum(1 for row in records if row["regime"] == regime and row["delay"] == delay),
        "successes": successes,
        "accuracies": {
            str(row["seed"]): row.get("test_accuracy")
            for row in records if row["regime"] == regime and row["delay"] == delay
        },
        "clip_fractions": {
            str(row["seed"]): row.get("grad_clip_fraction")
            for row in records if row["regime"] == regime and row["delay"] == delay
        },
    }
    if accuracies:
        ordered = sorted(accuracies)
        mid = len(ordered) // 2
        summary["accuracy_mean"] = sum(accuracies) / len(accuracies)
        summary["accuracy_median"] = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        summary["success_interval_95"] = clopper_pearson(successes, len(group))
    return summary


def paired_comparison(cfg: dict, records, delay: int) -> dict:
    rows = []
    for seed in cfg["seeds"]:
        item = {"seed": seed}
        for regime in cfg["regimes"]:
            match = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["regime"] == regime]
            item[regime] = None if not match else {
                "status": match[0]["status"],
                "success": match[0].get("success"),
                "test_accuracy": match[0].get("test_accuracy"),
            }
        rows.append(item)
    both = bias_only = matrix_only = full_only = 0
    for item in rows:
        bias = item["bias_plus_readout"] or {}
        matrix = item["matrices_plus_readout"] or {}
        full = item["full"] or {}
        if bias.get("success") and matrix.get("success"):
            both += 1
        elif bias.get("success"):
            bias_only += 1
        elif matrix.get("success"):
            matrix_only += 1
        elif full.get("success"):
            full_only += 1
    return {
        "delay": delay,
        "seeds": rows,
        "both_restricted_regimes_succeed": both,
        "bias_plus_readout_only": bias_only,
        "matrices_plus_readout_only": matrix_only,
        "full_only_among_these": full_only,
    }


def coordinate_audit(model, positive: torch.Tensor, negative: torch.Tensor) -> dict:
    with torch.no_grad():
        logits_pos, _ = model(positive)
        logits_neg, _ = model(negative)
        final_pos = model.recurrent_states(positive)[:, -1, :]
        final_neg = model.recurrent_states(negative)[:, -1, :]
    logit_pos = logits_pos[:, -1, 0]
    logit_neg = logits_neg[:, -1, 0]
    equal = (final_pos == final_neg).all(dim=-1)
    finite_diff = torch.isfinite(logit_pos) & torch.isfinite(logit_neg) & (logit_pos != logit_neg)
    return {
        "final_states_coordinate_equal_fraction": float(equal.float().mean()),
        "final_states_coordinate_equal_count": int(equal.sum()),
        "finite_logit_difference_count": int(finite_diff.sum()),
        "finite_logit_difference_fraction": float(finite_diff.float().mean()),
    }


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        write_frozen_hashes(OUT_DIR / "frozen_hashes_before.json")
        write_parameter_sets(OUT_DIR / "parameter_sets.json")
        print("wrote frozen hash manifest and parameter sets")
        return
    cfg = load_config()
    if platform.python_version() != cfg["environment_requirement"]["python"] or torch.__version__ != cfg["environment_requirement"]["torch"]:
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    blockers = measurement_blockers(OUT_DIR)
    if blockers:
        raise RuntimeError(
            "EXP-009 measurements already exist "
            f"({', '.join(blockers)}). Automatic resume is disabled. "
            "Preserve the files and request coordinator review before continuing."
        )
    if working_tree_dirty():
        raise RuntimeError("working tree is dirty")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before_path = OUT_DIR / "frozen_hashes_before.json"
    if json.loads(before_path.read_text(encoding="utf-8")) != frozen_manifest():
        raise RuntimeError("frozen artifact hashes changed")
    revision = git_revision()
    session_start = time.perf_counter()
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-009",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "timestamp_inference": False,
        "automatic_resume": False,
    }, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    verify_regime_lists(cfg, build_four(cfg, int(cfg["seeds"][0])))
    records = []
    done = set()
    total_cap = float(cfg["budget"]["max_cpu_seconds_total"])
    unstarted = []
    stopped_for_bug = None

    def expired() -> bool:
        return time.perf_counter() - session_start >= total_cap

    def session_elapsed() -> float:
        return round(time.perf_counter() - session_start, 3)

    def run_delay(delay: int, phase: str) -> None:
        nonlocal stopped_for_bug
        for seed in cfg["seeds"]:
            if expired() or stopped_for_bug:
                for regime in cfg["regimes"]:
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "implementation stop" if stopped_for_bug else "total cap"})
                continue
            bundle = dataset_bundle(cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(
                seed, delay, bundle["train_x"].shape[0],
                cfg["training"]["batch_size"], cfg["training"]["updates_per_run"],
            )
            index_hash = index_fingerprint(indices)
            models = build_four(cfg, seed)
            verify_regime_lists(cfg, models)
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
            append_jsonl(OUT_DIR / "fingerprints.jsonl", {
                "delay": delay,
                "seed": seed,
                "init_fingerprint": reference,
                "minibatch_fingerprint": index_hash,
                "dataset_fingerprints": data_fp,
                "noise_std": measured_noise_std(bundle["test_x"]),
            })
            for regime in cfg["regimes"]:
                if expired() or stopped_for_bug:
                    unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "implementation stop" if stopped_for_bug else "total cap"})
                    continue
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "start", "delay": delay, "seed": seed, "regime": regime,
                    "session_elapsed_seconds": session_elapsed(),
                })
                print(f"training {regime} delay {delay} seed {seed}", flush=True)
                try:
                    record, curve, checks, details = train_regime(
                        cfg, models[regime], regime, bundle, indices, delay, seed, phase,
                        reference, data_fp, index_hash,
                    )
                except ImplementationError as exc:
                    record = {
                        "experiment_id": "EXP-009",
                        "regime": regime,
                        "delay": delay,
                        "seed": seed,
                        "phase": phase,
                        "status": "implementation_bug",
                        "success": False,
                        "stop_reason": f"ImplementationError: {exc}",
                        "code_revision": git_revision(),
                        "runtime_seconds": session_elapsed(),
                    }
                    curve, checks, details = [], [], None
                    stopped_for_bug = {"delay": delay, "seed": seed, "regime": regime}
                if record.get("frozen_parameters_unchanged") is False or (
                    regime == "readout_only" and record.get("hidden_states_unchanged") is False
                ):
                    record["status"] = "implementation_bug"
                    record["success"] = False
                    record["stop_reason"] = "a frozen parameter or readout-only hidden state changed"
                    stopped_for_bug = {"delay": delay, "seed": seed, "regime": regime}
                if regime == "matrices_plus_readout" and details and not details.get("both_preactivation_biases_unchanged"):
                    record["status"] = "implementation_bug"
                    record["success"] = False
                    record["stop_reason"] = "a preactivation bias changed in the matrix regime"
                    stopped_for_bug = {"delay": delay, "seed": seed, "regime": regime}
                if regime == "bias_plus_readout" and details and not details.get("matrices_unchanged"):
                    record["status"] = "implementation_bug"
                    record["success"] = False
                    record["stop_reason"] = "a weight matrix changed in the bias regime"
                    stopped_for_bug = {"delay": delay, "seed": seed, "regime": regime}
                if record["status"] == "ok":
                    record["checkpoint_sha256"] = save_checkpoint(OUT_DIR, models[regime], record)
                records.append(record)
                done.add((delay, seed, regime))
                append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
                append_jsonl(OUT_DIR / "loss_curves.jsonl", {
                    "delay": delay, "seed": seed, "regime": regime, "loss_curve": curve,
                })
                for item in checks:
                    append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {
                        "delay": delay, "seed": seed, "regime": regime, **item,
                    })
                if details is not None:
                    append_jsonl(OUT_DIR / "bias_records.jsonl", details)
                append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                    "event": "finish", "delay": delay, "seed": seed, "regime": regime,
                    "status": record["status"], "runtime_seconds": record["runtime_seconds"],
                    "session_elapsed_seconds": session_elapsed(),
                })
                print(json.dumps({
                    "delay": delay, "seed": seed, "regime": regime, "status": record["status"],
                    "success": record.get("success"), "test_accuracy": record.get("test_accuracy"),
                }, sort_keys=True), flush=True)

    try:
        run_delay(64, "primary")
        full_successes = sum(1 for row in records if row["delay"] == 64 and row["regime"] == "full" and row.get("success"))
        delay64_terminal = sum(1 for row in records if row["delay"] == 64)
        gate = delay64_terminal == 40 and full_successes >= 8 and stopped_for_bug is None
        (OUT_DIR / "delay64_summary.json").write_text(json.dumps({
            "full_successes": full_successes,
            "delay64_terminal_records": delay64_terminal,
            "delay_128": "will run" if gate else "not run",
            "written_before_delay_128": True,
            "cohorts": [cohort_summary(records, regime, 64) for regime in cfg["regimes"]],
        }, indent=2, sort_keys=True), encoding="utf-8")
        if gate and not expired():
            run_delay(128, "secondary")
        elif gate:
            for seed in cfg["seeds"]:
                for regime in cfg["regimes"]:
                    unstarted.append({"delay": 128, "seed": seed, "regime": regime, "reason": "total cap"})
        audit_count = 0
        for delay in sorted({row["delay"] for row in records}):
            for seed in cfg["seeds"]:
                ready = [row for row in records if row["delay"] == delay and row["seed"] == seed and row["status"] == "ok"]
                if len(ready) < len(cfg["regimes"]):
                    continue
                if expired() or stopped_for_bug:
                    unstarted.append({
                        "delay": delay, "seed": seed, "regime": "audit",
                        "reason": "implementation stop" if stopped_for_bug else "total cap",
                    })
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
                for regime in cfg["regimes"]:
                    path = OUT_DIR / "checkpoints" / f"additive_{regime}_delay{delay}_seed{seed}.pt"
                    if expired():
                        unstarted.append({"delay": delay, "seed": seed, "regime": regime, "reason": "audit cap"})
                        continue
                    append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                        "event": "audit_start", "delay": delay, "seed": seed, "regime": regime,
                        "session_elapsed_seconds": session_elapsed(),
                    })
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    seed_everything(0)
                    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    with torch.no_grad():
                        original_logits, _ = model(positive[:2])
                    reloaded = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
                    reloaded.load_state_dict(payload["state_dict"])
                    reloaded.eval()
                    with torch.no_grad():
                        reloaded_logits, _ = reloaded(positive[:2])
                    if not torch.equal(original_logits, reloaded_logits):
                        raise RuntimeError("checkpoint reload changed outputs")
                    before = parameter_fingerprint(model)
                    audited = audit_one(model, "additive_tanh_rnn", positive, negative)
                    direct = coordinate_audit(model, positive, negative)
                    if parameter_fingerprint(model) != before:
                        raise RuntimeError("audit changed parameters")
                    append_jsonl(OUT_DIR / "audit_metrics.jsonl", {
                        "delay": delay, "seed": seed, "regime": regime,
                        "accuracy": audited["accuracy"],
                        "both_correct_fraction": audited["both_correct_fraction"],
                        "prediction_flip_fraction": audited["prediction_flip_fraction"],
                        "final_states_exactly_equal_fraction": audited["final_states_exactly_equal_fraction"],
                        "final_states_coordinate_equal_fraction": direct["final_states_coordinate_equal_fraction"],
                        "final_states_coordinate_equal_count": direct["final_states_coordinate_equal_count"],
                        "identical_logits_fraction": audited["identical_logits_fraction"],
                        "finite_logit_difference_count": direct["finite_logit_difference_count"],
                        "finite_logit_difference_fraction": direct["finite_logit_difference_fraction"],
                        "pairs_zero_local_grad_but_logit_changes": audited["pairs_zero_local_grad_but_logit_changes"],
                        "pairs_zero_local_grad_but_state_changes": audited["pairs_zero_local_grad_but_state_changes"],
                    })
                    audit_count += 1
                    append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                        "event": "audit_finish", "delay": delay, "seed": seed, "regime": regime,
                        "session_elapsed_seconds": session_elapsed(),
                    })
        delays_present = sorted({row["delay"] for row in records})
        summary = {
            "elapsed_seconds": session_elapsed(),
            "timing_method": "time.perf_counter inside this process",
            "timestamp_inference": False,
            "full_delay64_successes": full_successes,
            "delay64_terminal_records": delay64_terminal,
            "delay_128_ran": any(row["delay"] == 128 for row in records),
            "completed_runs": sum(1 for row in records if row["status"] == "ok"),
            "failed_runs": sum(1 for row in records if row["status"] != "ok"),
            "unstarted": unstarted,
            "implementation_stop": stopped_for_bug,
            "audit_rows": audit_count,
            "cohorts": [
                cohort_summary(records, regime, delay)
                for delay in delays_present for regime in cfg["regimes"]
            ],
            "paired": [paired_comparison(cfg, records, delay) for delay in delays_present],
            "code_revision": revision,
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (OUT_DIR / "paired_comparison.json").write_text(
            json.dumps(summary["paired"], indent=2, sort_keys=True), encoding="utf-8",
        )
        with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "delay", "seed", "regime", "status", "success", "test_accuracy",
                    "retention_probe_r2", "grad_clip_fraction", "runtime_seconds", "stop_reason",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)
        after = frozen_manifest()
        matched = after == json.loads(before_path.read_text(encoding="utf-8"))
        (OUT_DIR / "frozen_hashes_after.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary["historical_hashes_match"] = matched
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if not matched:
            raise RuntimeError("frozen artifacts changed during EXP-009")
        if stopped_for_bug:
            raise RuntimeError(f"implementation stop: {stopped_for_bug}")
        print(
            f"EXP-009 finished in {summary['elapsed_seconds']}s; "
            f"full successes {full_successes}; unstarted {len(unstarted)}",
            flush=True,
        )
    except Exception:
        partial = {
            "elapsed_seconds": session_elapsed(),
            "timing_method": "time.perf_counter inside this process",
            "timestamp_inference": False,
            "completed_runs": sum(1 for row in records if row["status"] == "ok"),
            "failed_runs": sum(1 for row in records if row["status"] != "ok"),
            "unstarted": unstarted,
            "implementation_stop": stopped_for_bug,
            "partial": True,
            "code_revision": revision,
        }
        partial_path = OUT_DIR / "summary.json"
        if not partial_path.exists():
            partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

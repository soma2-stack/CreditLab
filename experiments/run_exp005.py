"""EXP-005: recover EXP-004 models and audit original-bit dependence.

No new architecture. Replay writes only under results/EXP-005/.

Usage (from the repository root):
    python experiments/run_exp005.py --write-frozen-hashes
    python experiments/run_exp005.py
"""

from __future__ import annotations

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
from run_exp001 import evaluate, probe_fit_score  # noqa: E402
from run_exp004 import (  # noqa: E402
    build_matched_models,
    build_model,
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    load_hard_v2,
    minibatch_indices,
    parameter_fingerprint,
    train_one,
)

CONFIG_PATH = ROOT / "configs" / "exp005_counterfactual_audit.yaml"
EXP004_CONFIG = ROOT / "configs" / "exp004_bounded_additive.yaml"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
FROZEN_PATHS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "results/EXP-004",
    "configs/exp001_delayed_credit.yaml",
    "configs/exp001_delayed_credit_v2.yaml",
    "configs/exp001b_hard_shortcut_credit.yaml",
    "configs/exp002_residual_recurrent.yaml",
    "configs/exp003_bounded_mixture.yaml",
    "configs/exp004_bounded_additive.yaml",
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


def load_audit_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_exp004_config() -> dict:
    return yaml.safe_load(EXP004_CONFIG.read_text(encoding="utf-8"))


def frozen_manifest() -> dict[str, str]:
    records = {}
    for relative in FROZEN_PATHS:
        target = ROOT / relative
        if target.is_file():
            records[relative.replace("\\", "/")] = file_sha256(target)
            continue
        if not target.exists():
            raise FileNotFoundError(target)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_frozen_hashes(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(frozen_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )


def hashes_match(saved: Path) -> tuple[bool, str]:
    if not saved.exists():
        return False, "frozen hash manifest is missing"
    expected = json.loads(saved.read_text(encoding="utf-8"))
    current = frozen_manifest()
    if current != expected:
        return False, "frozen EXP-001 through EXP-004 hashes changed"
    return True, "match"


def loss_within_tolerance(replay: float, saved: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(replay - saved) <= abs_tol + rel_tol * abs(saved)


def predict_logits(logits: torch.Tensor) -> torch.Tensor:
    """Match EXP-004 evaluation: sign, with exact zero mapped to +1."""
    preds = torch.sign(logits)
    return torch.where(preds == 0, torch.ones_like(preds), preds)


def correct_count(model, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int) -> int:
    correct = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            logits, _ = model(inputs[start : start + batch_size])
            preds = predict_logits(logits[:, -1, 0])
            correct += int((preds == targets[start : start + batch_size]).sum())
    return correct


def make_counterfactual_pairs(base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positive = base.clone()
    negative = base.clone()
    positive[:, 0, 0] = 1.0
    negative[:, 0, 0] = -1.0
    return positive, negative


def pairs_differ_only_on_original_bit(positive: torch.Tensor, negative: torch.Tensor) -> bool:
    if positive.shape != negative.shape:
        return False
    difference = positive - negative
    if not torch.equal(difference[:, 0, 0], torch.full_like(difference[:, 0, 0], 2.0)):
        return False
    rest = difference.clone()
    rest[:, 0, 0] = 0
    return bool(torch.count_nonzero(rest) == 0)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    points = torch.tensor(QUANTILES, dtype=torch.float32)
    scored = torch.quantile(values.detach().float().cpu(), points)
    return {f"{point:.2f}": float(item) for point, item in zip(QUANTILES, scored, strict=True)}


def checkpoint_path(out_dir: Path, delay: int, seed: int, model_type: str) -> Path:
    return out_dir / "checkpoints" / f"{model_type}_delay{delay}_seed{seed}.pt"


def load_exp004_metrics() -> dict[tuple, dict]:
    rows = [
        json.loads(line)
        for line in (ROOT / "results" / "EXP-004" / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return {(row["delay"], row["seed"], row["model_type"]): row for row in rows}


def load_exp004_fingerprints() -> dict[tuple, dict]:
    rows = json.loads((ROOT / "results" / "EXP-004" / "fingerprints.json").read_text(encoding="utf-8"))
    return {(row["delay"], row["seed"]): row for row in rows}


def replay_models(audit_cfg: dict, exp4_cfg: dict, out_dir: Path) -> list[dict]:
    metrics = load_exp004_metrics()
    fingerprints = load_exp004_fingerprints()
    abs_tol = 1e-7
    rel_tol = 1e-4
    retention_tol = float(audit_cfg["recovery"]["verification"]["retention_r2_absolute_tolerance"])
    magnitude_tol = float(audit_cfg["recovery"]["verification"]["hidden_magnitude_absolute_tolerance"])
    report = []
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for delay in audit_cfg["delays"]:
        phase = "primary" if delay == 64 else "secondary"
        for seed in audit_cfg["seeds"]:
            bundle = dataset_bundle(exp4_cfg, delay, seed)
            data_fp = dataset_fingerprints(bundle)
            indices = minibatch_indices(
                seed, delay, bundle["train_x"].shape[0],
                exp4_cfg["training"]["batch_size"], exp4_cfg["training"]["updates_per_run"],
            )
            index_hash = index_fingerprint(indices)
            saved_fp = fingerprints[(delay, seed)]
            if data_fp != saved_fp["dataset_fingerprints"] or index_hash != saved_fp["minibatch_fingerprint"]:
                raise RuntimeError(f"replay data or minibatch fingerprint mismatch at delay {delay} seed {seed}")
            models = build_matched_models(exp4_cfg, seed)
            init_hash = parameter_fingerprint(models["vanilla_tanh_rnn"])
            if init_hash != saved_fp["init_fingerprint"]:
                raise RuntimeError(f"replay initialization fingerprint mismatch at delay {delay} seed {seed}")
            for model_type in audit_cfg["models"]:
                print(f"replaying {model_type} delay {delay} seed {seed}", flush=True)
                model = models[model_type]
                before = parameter_fingerprint(model)
                record, _curve, _checks, _clip = train_one(
                    exp4_cfg, model, model_type, bundle, indices, delay, seed, phase,
                    init_hash, data_fp, index_hash,
                )
                if record["status"] != "ok":
                    report.append({
                        "delay": delay, "seed": seed, "model_type": model_type,
                        "verified": False, "reason": record.get("stop_reason"),
                    })
                    continue
                saved = metrics[(delay, seed, model_type)]
                replay_correct = correct_count(
                    model, bundle["test_x"], bundle["test_y"], exp4_cfg["training"]["eval_batch_size"],
                )
                saved_correct = int(round(float(saved["test_accuracy"]) * bundle["test_y"].numel()))
                loss, _acc = evaluate(
                    model, bundle["test_x"], bundle["test_y"], exp4_cfg["training"]["eval_batch_size"],
                )
                val_loss, _val_acc = evaluate(
                    model, bundle["val_x"], bundle["val_y"], exp4_cfg["training"]["eval_batch_size"],
                )
                retention = probe_fit_score(
                    model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"],
                )
                magnitude = hidden_magnitude_stats(model, bundle["test_x"])
                checks = {
                    "correct_count": replay_correct == saved_correct,
                    "test_loss": loss_within_tolerance(loss, float(saved["test_loss"]), abs_tol, rel_tol),
                    "val_loss": loss_within_tolerance(val_loss, float(saved["val_loss"]), abs_tol, rel_tol),
                    "retention_probe_r2": abs(retention - float(saved["retention_probe_r2"])) <= retention_tol,
                    "max_abs_hidden": abs(magnitude["max_abs_hidden"] - float(saved["max_abs_hidden"])) <= magnitude_tol,
                    "final_hidden_rms": abs(magnitude["final_hidden_rms"] - float(saved["final_hidden_rms"])) <= magnitude_tol,
                }
                destination = checkpoint_path(out_dir, delay, seed, model_type)
                if destination.exists():
                    raise FileExistsError(f"refusing to overwrite {destination}")
                torch.save({
                    "experiment_id": "EXP-005",
                    "source_experiment": "EXP-004",
                    "recovery": "deterministic replay of experiments/run_exp004.py train_one",
                    "model_type": model_type,
                    "delay": delay,
                    "seed": seed,
                    "bound": 4.0 if model_type == "bounded_additive_tanh_rnn" else None,
                    "state_dict": model.state_dict(),
                    "init_fingerprint": init_hash,
                    "trained_fingerprint": parameter_fingerprint(model),
                    "replay_code_revision": git_revision(),
                }, destination)
                row = {
                    "delay": delay,
                    "seed": seed,
                    "model_type": model_type,
                    "verified": all(checks.values()),
                    "checks": checks,
                    "replay_correct_count": replay_correct,
                    "saved_correct_count": saved_correct,
                    "replay_test_loss": loss,
                    "saved_test_loss": saved["test_loss"],
                    "replay_val_loss": val_loss,
                    "saved_val_loss": saved["val_loss"],
                    "replay_retention_probe_r2": retention,
                    "saved_retention_probe_r2": saved["retention_probe_r2"],
                    "replay_max_abs_hidden": magnitude["max_abs_hidden"],
                    "saved_max_abs_hidden": saved["max_abs_hidden"],
                    "replay_final_hidden_rms": magnitude["final_hidden_rms"],
                    "saved_final_hidden_rms": saved["final_hidden_rms"],
                    "checkpoint": destination.relative_to(ROOT).as_posix(),
                    "init_fingerprint_before_training": before,
                }
                report.append(row)
                append_jsonl(out_dir / "recovery_records.jsonl", row)
                print(json.dumps({
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "verified": row["verified"], "checks": checks,
                }, sort_keys=True), flush=True)
    return report


def load_recovered_model(exp4_cfg: dict, path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["model_type"], exp4_cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def bit_and_marker_sensitivity(model, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of each final logit w.r.t. the original bit and marker channels."""
    probe = inputs.detach().clone()
    # The input gradient is required. Parameters stay frozen.
    probe.requires_grad_(True)
    logits, _ = model(probe)
    final = logits[:, -1, 0]
    (grad,) = torch.autograd.grad(final.sum(), probe)
    return grad[:, 0, 0].detach(), grad[:, 0, 1].detach()


def summarize_values(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1)
    exact_zero = int((flat == 0).sum())
    nonzero = flat[flat != 0]
    return {
        "count": int(flat.numel()),
        "exact_computational_zeros": exact_zero,
        "nonzero_count": int(nonzero.numel()),
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "quantiles": quantiles(flat),
        "min_nonzero_abs": None if nonzero.numel() == 0 else float(nonzero.abs().min()),
        "max_abs": float(flat.abs().max()),
    }


def audit_one(model, model_type: str, positive: torch.Tensor, negative: torch.Tensor) -> dict:
    fingerprint_before = parameter_fingerprint(model)
    with torch.no_grad():
        zero_positive = positive.clone()
        zero_negative = negative.clone()
        zero_positive[:, 0, 0] = 0
        zero_negative[:, 0, 0] = 0
        if not torch.equal(zero_positive, zero_negative):
            raise RuntimeError("negative control inputs are not identical")
        zero_logits, _ = model(zero_positive)
        zero_logits_again, _ = model(zero_negative)
        zero_states = model.recurrent_states(zero_positive)
        zero_states_again = model.recurrent_states(zero_negative)
        if not torch.equal(zero_logits, zero_logits_again) or not torch.equal(zero_states, zero_states_again):
            raise RuntimeError("negative control states or logits diverged")
        logits_pos, _ = model(positive)
        logits_neg, _ = model(negative)
        states_pos = model.recurrent_states(positive)
        states_neg = model.recurrent_states(negative)
    logit_pos = logits_pos[:, -1, 0]
    logit_neg = logits_neg[:, -1, 0]
    pred_pos = predict_logits(logit_pos)
    pred_neg = predict_logits(logit_neg)
    targets_pos = torch.ones_like(pred_pos)
    targets_neg = -torch.ones_like(pred_neg)
    pos_correct = pred_pos == targets_pos
    neg_correct = pred_neg == targets_neg
    both_correct = pos_correct & neg_correct
    both_wrong = (~pos_correct) & (~neg_correct)
    exactly_one = pos_correct ^ neg_correct
    difference = states_pos - states_neg
    l2 = difference.norm(dim=-1)
    mean_abs = difference.abs().mean(dim=-1)
    exact_equal = (states_pos == states_neg).float().mean(dim=-1)
    sign_disagree = (torch.sign(states_pos) != torch.sign(states_neg)).float().mean(dim=-1)
    final_pos = states_pos[:, -1, :]
    final_neg = states_neg[:, -1, :]
    final_l2 = l2[:, -1]
    logit_delta = logit_pos - logit_neg
    bit_pos, marker_pos = bit_and_marker_sensitivity(model, positive)
    bit_neg, marker_neg = bit_and_marker_sensitivity(model, negative)
    both_grads_zero = (bit_pos == 0) & (bit_neg == 0)
    finite_logit = logit_pos != logit_neg
    finite_state = final_l2 != 0
    record = {
        "n_pairs": int(positive.shape[0]),
        "n_sequences": int(positive.shape[0] * 2),
        "accuracy": float(torch.cat([pos_correct, neg_correct]).float().mean()),
        "both_correct_fraction": float(both_correct.float().mean()),
        "exactly_one_correct_fraction": float(exactly_one.float().mean()),
        "both_wrong_fraction": float(both_wrong.float().mean()),
        "prediction_flip_fraction": float((pred_pos != pred_neg).float().mean()),
        "logit_difference": summarize_values(logit_delta),
        "positive_logit_difference_fraction": float((logit_delta > 0).float().mean()),
        "identical_logits_fraction": float((logit_pos == logit_neg).float().mean()),
        "final_l2": summarize_values(final_l2),
        "final_states_exactly_equal_fraction": float((final_l2 == 0).float().mean()),
        "final_sign_pattern_identical_fraction": float(
            (torch.sign(final_pos) == torch.sign(final_neg)).all(dim=-1).float().mean()
        ),
        "bit_sensitivity_positive": summarize_values(bit_pos),
        "bit_sensitivity_negative": summarize_values(bit_neg),
        "marker_sensitivity_positive_mean_abs": float(marker_pos.abs().mean()),
        "marker_sensitivity_negative_mean_abs": float(marker_neg.abs().mean()),
        "pairs_both_bit_grads_exact_zero": int(both_grads_zero.sum()),
        "pairs_zero_local_grad_but_logit_changes": int((both_grads_zero & finite_logit).sum()),
        "pairs_zero_local_grad_but_state_changes": int((both_grads_zero & finite_state).sum()),
        "trajectory": {
            "mean_l2": [float(value) for value in l2.mean(dim=0)],
            "median_l2": [float(value) for value in l2.median(dim=0).values],
            "mean_abs_coordinate_difference": [float(value) for value in mean_abs.mean(dim=0)],
            "exact_equal_fraction": [float(value) for value in exact_equal.mean(dim=0)],
            "sign_disagreement_fraction": [float(value) for value in sign_disagree.mean(dim=0)],
        },
    }
    if model_type == "bounded_additive_tanh_rnn":
        at_pos = final_pos.abs() == 4.0
        at_neg = final_neg.abs() == 4.0
        union = at_pos | at_neg
        has_boundary = union.any(dim=-1)
        signs_agree = torch.sign(final_pos) == torch.sign(final_neg)
        boundary_signs_agree = (signs_agree | ~union).all(dim=-1)
        record["boundary"] = {
            "mean_fraction_coordinates_at_boundary": float(union.float().mean()),
            "pairs_with_any_boundary_coordinate": int(has_boundary.sum()),
            "pairs_identical_boundary_sign_pattern": int((has_boundary & boundary_signs_agree).sum()),
            "pairs_different_boundary_sign_pattern": int((has_boundary & ~boundary_signs_agree).sum()),
            "pairs_no_boundary_coordinate": int((~has_boundary).sum()),
        }
    if parameter_fingerprint(model) != fingerprint_before:
        raise RuntimeError("audit changed model parameters")
    return record


def run_audit(audit_cfg: dict, exp4_cfg: dict, out_dir: Path) -> None:
    fingerprints = []
    for delay in audit_cfg["delays"]:
        pair_bank = {}
        for seed in audit_cfg["seeds"]:
            base, _targets, _meta = load_hard_v2(
                audit_cfg["task"]["audit_samples"], delay, seed,
                audit_cfg["task"]["audit_split_index"],
                noise_std=audit_cfg["task"]["distractor_noise_std"],
                competitor_rate=audit_cfg["task"]["competitor_rate"],
                competitor_flip_prob=audit_cfg["task"]["competitor_flip_prob"],
                base_seed=audit_cfg["task"]["base_seed"],
                return_metadata=True,
            )
            positive, negative = make_counterfactual_pairs(base)
            if not pairs_differ_only_on_original_bit(positive, negative):
                raise RuntimeError(f"pair construction failed for delay {delay} seed {seed}")
            pair_bank[seed] = (positive, negative)
            fingerprints.append({
                "delay": delay,
                "seed": seed,
                "run_seed": seed,
                "split_index": audit_cfg["task"]["audit_split_index"],
                "base_fingerprint": hashlib.sha256(base.contiguous().numpy().tobytes()).hexdigest(),
            })
        for seed in audit_cfg["seeds"]:
            positive, negative = pair_bank[seed]
            for model_type in audit_cfg["models"]:
                path = checkpoint_path(out_dir, delay, seed, model_type)
                model = load_recovered_model(exp4_cfg, path)
                print(f"auditing {model_type} delay {delay} seed {seed}", flush=True)
                started = time.perf_counter()
                record = audit_one(model, model_type, positive, negative)
                record.update({
                    "delay": delay,
                    "seed": seed,
                    "model_type": model_type,
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                })
                append_jsonl(out_dir / "audit_metrics.jsonl", {
                    key: value for key, value in record.items() if key != "trajectory"
                })
                append_jsonl(out_dir / "state_trajectories.jsonl", {
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "trajectory": record["trajectory"],
                    "final_l2": record["final_l2"],
                    "boundary": record.get("boundary"),
                })
                print(json.dumps({
                    "delay": delay, "seed": seed, "model_type": model_type,
                    "accuracy": record["accuracy"],
                    "both_correct_fraction": record["both_correct_fraction"],
                    "prediction_flip_fraction": record["prediction_flip_fraction"],
                    "final_states_exactly_equal_fraction": record["final_states_exactly_equal_fraction"],
                    "zero_grad_but_logit_changes": record["pairs_zero_local_grad_but_logit_changes"],
                    "zero_grad_but_state_changes": record["pairs_zero_local_grad_but_state_changes"],
                }, sort_keys=True), flush=True)
    (out_dir / "audit_data_fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True), encoding="utf-8",
    )


def summarize_audit(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {"rows": rows}


def main() -> None:
    if "--write-frozen-hashes" in sys.argv:
        write_frozen_hashes(ROOT / "results" / "EXP-005" / "frozen_hashes_before.json")
        print("wrote frozen hash manifest")
        return
    audit_cfg = load_audit_config()
    exp4_cfg = load_exp004_config()
    if audit_cfg["task"]["distractor_noise_std"] != exp4_cfg["task"]["distractor_noise_std"]:
        raise RuntimeError("EXP-005 noise does not match EXP-004")
    if audit_cfg["bound"]["B"] != exp4_cfg["bound"]["B"]:
        raise RuntimeError("EXP-005 bound does not match EXP-004")
    if platform.python_version() != audit_cfg["environment_requirement"]["python"]:
        raise RuntimeError(f"Python {platform.python_version()} does not match the required environment")
    if torch.__version__ != audit_cfg["environment_requirement"]["torch"]:
        raise RuntimeError(f"PyTorch {torch.__version__} does not match the required environment")
    out_dir = ROOT / audit_cfg["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "verification_report.json").exists() or (out_dir / "audit_metrics.jsonl").exists():
        raise FileExistsError("refusing to overwrite existing EXP-005 outputs")
    dirty = working_tree_dirty()
    if dirty:
        raise RuntimeError(f"working tree is dirty:\n{dirty}")
    ok, message = hashes_match(out_dir / "frozen_hashes_before.json")
    if not ok:
        raise RuntimeError(message)
    exp4_checkpoints = list((ROOT / "results" / "EXP-004").rglob("*.pt"))
    provenance = {
        "exp004_checkpoint_files": [path.relative_to(ROOT).as_posix() for path in exp4_checkpoints],
        "recovery": "replay" if not exp4_checkpoints else "load",
    }
    if exp4_checkpoints:
        raise RuntimeError(
            "EXP-004 checkpoints were found. This runner is authorized to replay only when "
            "they are absent. Inspect them before loading."
        )
    revision = git_revision()
    (out_dir / "environment.txt").write_text(
        "\n".join([
            f"Python: {platform.python_version()}",
            f"PyTorch: {torch.__version__}",
            f"CPU architecture: {platform.machine()}",
            f"Audit code revision: {revision}",
            "Recovery: deterministic replay because EXP-004 saved no checkpoints.",
            "No optimizer steps are taken during the audit.",
        ]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    wall = time.perf_counter()
    report = replay_models(audit_cfg, exp4_cfg, out_dir)
    verified = all(row.get("verified") for row in report) and len(report) == 18
    payload = {
        "provenance": provenance,
        "verified": verified,
        "rows": report,
        "replay_wall_seconds": round(time.perf_counter() - wall, 3),
    }
    (out_dir / "verification_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    if not verified:
        print("EXP-005 stopped: model recovery verification failed", flush=True)
        return
    audit_started = time.perf_counter()
    run_audit(audit_cfg, exp4_cfg, out_dir)
    summary = summarize_audit(out_dir / "audit_metrics.jsonl")
    summary["audit_wall_seconds"] = round(time.perf_counter() - audit_started, 3)
    summary["total_wall_seconds"] = round(time.perf_counter() - wall, 3)
    summary["code_revision"] = revision
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = summary["rows"]
    fields = [
        "delay", "seed", "model_type", "accuracy", "both_correct_fraction",
        "exactly_one_correct_fraction", "both_wrong_fraction", "prediction_flip_fraction",
        "identical_logits_fraction", "final_states_exactly_equal_fraction",
        "pairs_zero_local_grad_but_logit_changes", "pairs_zero_local_grad_but_state_changes",
    ]
    import csv
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    after = frozen_manifest()
    before = json.loads((out_dir / "frozen_hashes_before.json").read_text(encoding="utf-8"))
    matched = after == before
    (out_dir / "frozen_hashes_after.json").write_text(
        json.dumps({"match": matched}, indent=2), encoding="utf-8",
    )
    if not matched:
        raise RuntimeError("frozen artifacts changed during EXP-005")
    print(f"EXP-005 audit finished in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

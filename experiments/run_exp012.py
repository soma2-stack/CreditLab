"""EXP-012: refit a fixed linear readout on frozen EXP-011 hidden states.

Registration, before any refitted accuracy:
    python experiments/run_exp012.py --write-registration

Diagnostic, after that registration is committed:
    python experiments/run_exp012.py
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import file_sha256, load_hard_v2, tensor_fingerprint  # noqa: E402
from run_exp005 import correct_count, make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp009_audit_completion import saved_correct_count  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp012_readout_refit.yaml"
OUT_DIR = ROOT / "results" / "EXP-012"
EXP11 = ROOT / "results" / "EXP-011"
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
CONDITIONS = ["full_history_clip5", "final_16_clip5", "full_history_noclip", "final_16_noclip"]
DELAYS = [64, 128]
RECURRENT_NAMES = (
    "recurrent.weight_ih_l0",
    "recurrent.weight_hh_l0",
    "recurrent.bias_ih_l0",
    "recurrent.bias_hh_l0",
)
BLOCKED = {"metrics.jsonl", "runtime_session.json", "summary.json"}
ARTIFACT_DIRS = tuple(f"results/EXP-{name}" for name in (
    "001", "001B", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011",
))


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


def source_records() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (EXP11 / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("status") == "ok"]


def checkpoint_path(condition: str, delay: int, seed: int) -> Path:
    return EXP11 / "checkpoints" / f"additive_{condition}_delay{delay}_seed{seed}.pt"


def manifest_rows() -> list[dict]:
    saved = {(row["condition"], int(row["delay"]), int(row["seed"])): row for row in source_records()}
    rows = []
    for delay in DELAYS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                match = saved.get((condition, delay, seed))
                rows.append({
                    "delay": delay,
                    "seed": seed,
                    "condition": condition,
                    "checkpoint": checkpoint_path(condition, delay, seed).relative_to(ROOT).as_posix(),
                    "recorded_sha256": None if match is None else match.get("checkpoint_sha256"),
                    "recorded_test_accuracy": None if match is None else match.get("test_accuracy"),
                    "recorded_status": None if match is None else match.get("status"),
                })
    return rows


def load_model(path: Path) -> tuple[ResidualTanhRNN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def recurrent_hash(model) -> str:
    tensors = dict(model.named_parameters())
    import hashlib
    digest = hashlib.sha256()
    for name in RECURRENT_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def final_states(model, inputs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        states = model.recurrent_states(inputs)[:, -1, :]
    return states.detach().cpu().double().numpy()


def original_logits(model, states: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(states.astype(np.float32))
    with torch.no_grad():
        logits = model.readout(tensor)[:, 0]
    return logits.double().numpy()


def original_predictions(logits: np.ndarray) -> np.ndarray:
    preds = np.sign(logits)
    preds[preds == 0] = 1.0
    return preds


def fit_scale(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    variance = ((train - mean) ** 2).mean(axis=0)
    scale = np.sqrt(variance)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def apply_scale(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def binary_labels(targets: torch.Tensor) -> np.ndarray:
    return ((targets.detach().cpu().numpy() + 1.0) / 2.0).astype(np.int64)


def bce_from_decision(decision: np.ndarray, binary: np.ndarray) -> float:
    score = decision.astype(np.float64)
    target = binary.astype(np.float64)
    loss = np.maximum(score, 0.0) - score * target + np.log1p(np.exp(-np.abs(score)))
    return float(np.mean(loss))


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        tol=1e-8,
        max_iter=2000,
        fit_intercept=True,
        class_weight=None,
    )


def refit_predictions(decision: np.ndarray) -> np.ndarray:
    return np.where(decision > 0.0, 1.0, -1.0)


def pair_metrics(logits_positive, logits_negative, decision_rule) -> dict:
    pred_positive = decision_rule(logits_positive)
    pred_negative = decision_rule(logits_negative)
    pos_correct = pred_positive == 1.0
    neg_correct = pred_negative == -1.0
    both = pos_correct & neg_correct
    finite = np.isfinite(logits_positive) & np.isfinite(logits_negative) & (logits_positive != logits_negative)
    accuracy = float(np.mean(np.concatenate([pos_correct, neg_correct])))
    return {
        "accuracy": accuracy,
        "both_correct_fraction": float(np.mean(both)),
        "prediction_flip_fraction": float(np.mean(pred_positive != pred_negative)),
        "finite_logit_difference_fraction": float(np.mean(finite)),
    }


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    records["configs/exp011_clipping_horizon.yaml"] = file_sha256(ROOT / "configs" / "exp011_clipping_horizon.yaml")
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-012 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows()
    if len(rows) != 80 or any(row["recorded_sha256"] is None for row in rows):
        raise RuntimeError("the EXP-011 checkpoint manifest is not 80 complete records")
    registration = {
        "experiment_id": "EXP-012",
        "kind": "exploratory_diagnostic",
        "checkpoints": rows,
        "classifier": load_config()["classifier"],
        "preprocessing": load_config()["preprocessing"],
        "budget_seconds": 300,
        "note": "No recurrent training and no change to C.",
    }
    destination.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")
    manifest = artifact_manifest()
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered {len(rows)} checkpoints; hashed {len(manifest)} preexisting files")


def data_bundle(delay: int, seed: int, split_index: int, count: int):
    return load_hard_v2(
        count, delay, seed, split_index,
        noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=2000,
    )


def verify_all(saved: dict) -> list[dict]:
    reports = []
    for delay in DELAYS:
        for seed in SEEDS:
            test_x, test_y = data_bundle(delay, seed, 2, 512)
            for condition in CONDITIONS:
                row = saved[(condition, delay, seed)]
                path = checkpoint_path(condition, delay, seed)
                report = {"delay": delay, "seed": seed, "condition": condition, "status": "ok", "stop_reason": None}
                if not path.is_file():
                    report["status"] = "verification_failure"
                    report["stop_reason"] = "checkpoint missing"
                    reports.append(report)
                    return reports
                recomputed = file_sha256(path)
                report["checkpoint_sha256"] = recomputed
                if recomputed != row.get("checkpoint_sha256"):
                    report["status"] = "verification_failure"
                    report["stop_reason"] = "checkpoint hash does not match the saved record"
                    reports.append(report)
                    return reports
                if tensor_fingerprint(test_x) != row["dataset_fingerprints"]["test_x"]:
                    report["status"] = "verification_failure"
                    report["stop_reason"] = "regenerated test examples do not match"
                    reports.append(report)
                    return reports
                model, payload = load_model(path)
                if payload.get("condition") != condition or int(payload.get("delay")) != delay or int(payload.get("seed")) != seed:
                    report["status"] = "verification_failure"
                    report["stop_reason"] = "checkpoint metadata does not match"
                    reports.append(report)
                    return reports
                expected = saved_correct_count(row["test_accuracy"], 512)
                actual = correct_count(model, test_x, test_y, 256)
                report["saved_correct_count"] = expected
                report["recomputed_correct_count"] = actual
                if actual != expected:
                    report["status"] = "verification_failure"
                    report["stop_reason"] = f"correct count {actual} does not match saved {expected}"
                    reports.append(report)
                    return reports
                reports.append(report)
    return reports


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    blockers = [name for name in BLOCKED if (OUT_DIR / name).exists()]
    if blockers:
        raise RuntimeError(f"EXP-012 outputs already exist ({', '.join(blockers)}). Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-012")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nDiagnostic code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-012",
        "started_unix": time.time(),
        "code_revision": revision,
        "sklearn": sklearn_version(),
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": 300,
    }, indent=2, sort_keys=True), encoding="utf-8")
    saved = {(row["condition"], int(row["delay"]), int(row["seed"])): row for row in source_records()}
    if len(saved) != 80:
        raise RuntimeError(f"expected 80 EXP-011 records, found {len(saved)}")
    reports = verify_all(saved)
    (OUT_DIR / "checkpoint_verification.jsonl").write_text(
        "".join(json.dumps(report, sort_keys=True) + "\n" for report in reports),
        encoding="utf-8",
    )
    if any(report["status"] != "ok" for report in reports) or len(reports) != 80:
        raise RuntimeError(f"checkpoint verification failed: {[report for report in reports if report['status'] != 'ok'][:1]}")

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def expired() -> bool:
        return time.perf_counter() - started >= 300

    metrics = []
    unstarted = []
    for delay in DELAYS:
        for seed in SEEDS:
            if expired():
                for condition in CONDITIONS:
                    unstarted.append({"delay": delay, "seed": seed, "condition": condition, "reason": "total cap"})
                continue
            train_x, train_y = data_bundle(delay, seed, 0, 1536)
            val_x, val_y = data_bundle(delay, seed, 1, 512)
            test_x, test_y = data_bundle(delay, seed, 2, 512)
            audit_base, _audit_y = data_bundle(delay, seed, 3, 512)
            positive, negative = make_counterfactual_pairs(audit_base)
            if not pairs_differ_only_on_original_bit(positive, negative):
                raise RuntimeError("pair construction failed")
            zero_positive = positive.clone()
            zero_negative = negative.clone()
            zero_positive[:, 0, 0] = 0
            zero_negative[:, 0, 0] = 0
            for condition in CONDITIONS:
                if expired():
                    unstarted.append({"delay": delay, "seed": seed, "condition": condition, "reason": "total cap"})
                    continue
                row = saved[(condition, delay, seed)]
                model, _payload = load_model(checkpoint_path(condition, delay, seed))
                before_hash = recurrent_hash(model)
                train_state = final_states(model, train_x)
                val_state = final_states(model, val_x)
                test_state = final_states(model, test_x)
                if recurrent_hash(model) != before_hash:
                    raise RuntimeError("feature extraction changed recurrent parameters")
                zero_state_a = final_states(model, zero_positive[:8])
                zero_state_b = final_states(model, zero_negative[:8])
                if not np.array_equal(zero_state_a, zero_state_b):
                    raise RuntimeError("zero-bit controls did not match")
                direct_state = final_states(model, test_x[:4])
                if not np.array_equal(direct_state, test_state[:4]):
                    raise RuntimeError("extracted features do not match a second readout-free pass")
                mean, scale = fit_scale(train_state)
                train_binary = binary_labels(train_y)
                classifier = make_classifier()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(apply_scale(train_state, mean, scale), train_binary)
                iterations = int(np.max(classifier.n_iter_))
                converged = iterations < 2000 and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
                def decision(states: np.ndarray) -> np.ndarray:
                    return classifier.decision_function(apply_scale(states, mean, scale))
                train_decision = decision(train_state)
                val_decision = decision(val_state)
                test_decision = decision(test_state)
                original_test = original_logits(model, test_state)
                original_pred = original_predictions(original_test)
                refit_pred = refit_predictions(test_decision)
                test_targets = test_y.detach().cpu().numpy()
                original_accuracy = float(np.mean(original_pred == test_targets))
                refit_accuracy = float(np.mean(refit_pred == test_targets))
                original_binary = binary_labels(test_y)
                pos_state = final_states(model, positive)
                neg_state = final_states(model, negative)
                if recurrent_hash(model) != before_hash:
                    raise RuntimeError("audit extraction changed recurrent parameters")
                original_audit = pair_metrics(original_logits(model, pos_state), original_logits(model, neg_state), original_predictions)
                refit_audit = pair_metrics(decision(pos_state), decision(neg_state), refit_predictions)
                again = final_states(model, positive[:4])
                if not np.array_equal(again, pos_state[:4]):
                    raise RuntimeError("readout evaluation changed hidden states")
                record = {
                    "delay": delay,
                    "seed": seed,
                    "condition": condition,
                    "status": "ok" if converged else "inconclusive_not_converged",
                    "converged": converged,
                    "iterations": iterations,
                    "convergence_warnings": [str(item.message) for item in caught],
                    "original_test_accuracy": original_accuracy,
                    "refit_test_accuracy": refit_accuracy,
                    "accuracy_change": refit_accuracy - original_accuracy,
                    "original_test_loss": bce_from_decision(original_test, original_binary),
                    "refit_test_loss": bce_from_decision(test_decision, original_binary),
                    "original_success": bool(original_accuracy >= 0.95),
                    "refit_success": bool(refit_accuracy >= 0.95),
                    "train_accuracy": float(np.mean(refit_predictions(train_decision) == train_y.numpy())),
                    "train_loss": bce_from_decision(train_decision, train_binary),
                    "val_accuracy": float(np.mean(refit_predictions(val_decision) == val_y.numpy())),
                    "val_loss": bce_from_decision(val_decision, binary_labels(val_y)),
                    "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                    "intercept": float(classifier.intercept_[0]),
                    "zero_variance_coordinates": int(np.sum(np.sqrt(((train_state - mean) ** 2).mean(axis=0)) == 0.0)),
                    "recurrent_hash": before_hash,
                    "original_audit": original_audit,
                    "refit_audit": refit_audit,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
                classifier_path = OUT_DIR / "classifiers" / f"{condition}_delay{delay}_seed{seed}.joblib"
                classifier_path.parent.mkdir(parents=True, exist_ok=True)
                if classifier_path.exists():
                    raise FileExistsError(classifier_path)
                import joblib
                joblib.dump({
                    "classifier": classifier,
                    "mean": mean,
                    "scale": scale,
                    "condition": condition,
                    "delay": delay,
                    "seed": seed,
                }, classifier_path)
                record["classifier_sha256"] = file_sha256(classifier_path)
                metrics.append(record)
                with (OUT_DIR / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                with (OUT_DIR / "feature_fingerprints.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "delay": delay, "seed": seed, "condition": condition,
                        "train_state_sha256": tensor_fingerprint(torch.from_numpy(train_state)),
                        "test_state_sha256": tensor_fingerprint(torch.from_numpy(test_state)),
                    }) + "\n")
                print(json.dumps({
                    "delay": delay, "seed": seed, "condition": condition,
                    "original": round(original_accuracy, 4), "refit": round(refit_accuracy, 4),
                    "converged": converged,
                }), flush=True)
    summary = {
        "elapsed_seconds": elapsed(),
        "timing_method": "time.perf_counter inside this process",
        "verified_checkpoints": len(reports),
        "completed": len(metrics),
        "unstarted": unstarted,
        "inconclusive": sum(1 for row in metrics if not row["converged"]),
        "code_revision": revision,
        "sklearn": sklearn_version(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "delay", "seed", "condition", "status", "converged", "iterations",
            "original_test_accuracy", "refit_test_accuracy", "accuracy_change",
            "original_success", "refit_success", "original_test_loss", "refit_test_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-012")
    print(f"EXP-012 finished in {summary['elapsed_seconds']}s; completed {len(metrics)}; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

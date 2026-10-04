"""EXP-013: diagnostic readout on initialized, full-history, and final-16 states.

Registration, before any new accuracy:
    python experiments/run_exp013.py --write-registration

Diagnostic, after that registration is committed:
    python experiments/run_exp013.py
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

import joblib
import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import build_matched_models, file_sha256, load_hard_v2, tensor_fingerprint  # noqa: E402
from run_exp005 import correct_count, make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp009_audit_completion import saved_correct_count  # noqa: E402
from run_exp012 import (  # noqa: E402
    apply_scale,
    bce_from_decision,
    binary_labels,
    final_states,
    fit_scale,
    make_classifier,
    original_logits,
    original_predictions,
    pair_metrics,
    recurrent_hash,
    refit_predictions,
    sklearn_version,
)

CONFIG_PATH = ROOT / "configs" / "exp013_initialized_readout.yaml"
OUT_DIR = ROOT / "results" / "EXP-013"
EXP7_CFG = ROOT / "configs" / "exp007_readout_only.yaml"
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
DELAYS = [64, 128]
REPRESENTATIONS = ("initialized", "full_history", "final_16")
RECURRENT = (
    "recurrent.weight_ih_l0", "recurrent.weight_hh_l0",
    "recurrent.bias_ih_l0", "recurrent.bias_hh_l0",
)
BLOCKED = {"metrics.jsonl", "runtime_session.json", "summary.json"}
ARTIFACT_DIRS = tuple(
    f"results/EXP-{name}" for name in
    ("001", "001B", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012")
)


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_tail(value: str) -> str:
    return str(value).rsplit(":", 1)[-1]


def exp7_path(delay: int, seed: int) -> Path:
    return ROOT / "results" / "EXP-007" / "checkpoints" / f"additive_tanh_rnn_readout_only_delay{delay}_seed{seed}.pt"


def exp11_path(condition: str, delay: int, seed: int) -> Path:
    return ROOT / "results" / "EXP-011" / "checkpoints" / f"additive_{condition}_delay{delay}_seed{seed}.pt"


def source_rows() -> dict[tuple, dict]:
    exp7 = {
        (int(row["delay"]), int(row["seed"])): row
        for row in read_jsonl(ROOT / "results" / "EXP-007" / "raw_metrics.jsonl")
        if row.get("model_type") == "additive_tanh_rnn" and row.get("regime") == "readout_only" and row.get("status") == "ok"
    }
    exp11 = {
        (row["condition"], int(row["delay"]), int(row["seed"])): row
        for row in read_jsonl(ROOT / "results" / "EXP-011" / "raw_metrics.jsonl")
        if row.get("status") == "ok" and row.get("condition") in {"full_history_clip5", "final_16_clip5"}
    }
    exp12 = {
        (row["condition"], int(row["delay"]), int(row["seed"])): row
        for row in read_jsonl(ROOT / "results" / "EXP-012" / "metrics.jsonl")
        if row.get("condition") in {"full_history_clip5", "final_16_clip5"}
    }
    rows = {}
    for delay in DELAYS:
        for seed in SEEDS:
            old = exp7[(delay, seed)]
            rows[("initialized", delay, seed)] = {
                "path": exp7_path(delay, seed),
                "recorded_sha256": sha_tail(old["checkpoint_sha256"]),
                "recorded_test_accuracy": old["test_accuracy"],
                "dataset_fingerprints": None,
                "reproduce_exp12": False,
            }
            for name, condition in (("full_history", "full_history_clip5"), ("final_16", "final_16_clip5")):
                trained = exp11[(condition, delay, seed)]
                rows[(name, delay, seed)] = {
                    "path": exp11_path(condition, delay, seed),
                    "recorded_sha256": trained["checkpoint_sha256"],
                    "recorded_test_accuracy": trained["test_accuracy"],
                    "dataset_fingerprints": trained["dataset_fingerprints"],
                    "exp12_accuracy": exp12[(condition, delay, seed)]["refit_test_accuracy"],
                    "exp12_classifier": ROOT / "results" / "EXP-012" / "classifiers" / f"{condition}_delay{delay}_seed{seed}.joblib",
                    "reproduce_exp12": True,
                }
    return rows


def load_additive(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from creditlab.models import ResidualTanhRNN
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def reconstruct_initialized(seed: int):
    cfg = yaml.safe_load(EXP7_CFG.read_text(encoding="utf-8"))
    seed_everything(seed)
    return build_matched_models(cfg, seed)["additive_tanh_rnn"]


def recurrent_equal(left, right) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return all(torch.equal(left_state[name], right_state[name]) for name in RECURRENT)


def data_bundle(delay: int, seed: int, split_index: int, count: int):
    return load_hard_v2(
        count, delay, seed, split_index,
        noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=2000,
    )


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-013 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = source_rows()
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 conditions, found {len(rows)}")
    registration = {
        "experiment_id": "EXP-013",
        "kind": "exploratory_matched_diagnostic",
        "conditions": [
            {"representation": name, "delay": delay, "seed": seed, "checkpoint": rows[(name, delay, seed)]["path"].relative_to(ROOT).as_posix()}
            for delay in DELAYS for seed in SEEDS for name in REPRESENTATIONS
        ],
        "initialization_check": load_config()["initialization_check"],
        "reproduction_check": load_config()["reproduction_check"],
        "classifier": load_config()["classifier"],
        "budget_seconds": 300,
    }
    destination.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(artifact_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered {len(rows)} conditions")


def fit_one(train_state, train_y):
    mean, scale = fit_scale(train_state)
    classifier = make_classifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(apply_scale(train_state, mean, scale), binary_labels(train_y))
    iterations = int(np.max(classifier.n_iter_))
    converged = iterations < 2000 and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return classifier, mean, scale, iterations, converged, [str(item.message) for item in caught]


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match Python 3.11.9, PyTorch 2.13.0+cpu, and scikit-learn 1.9.1")
    blockers = [name for name in BLOCKED if (OUT_DIR / name).exists()]
    if blockers:
        raise RuntimeError(f"EXP-013 outputs already exist ({', '.join(blockers)}). Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-013")
    started = time.perf_counter()
    revision = git_revision()
    sources = source_rows()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nDiagnostic code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-013", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds": 300,
    }, indent=2, sort_keys=True), encoding="utf-8")
    sources = source_rows()
    reports = []
    for delay in DELAYS:
        for seed in SEEDS:
            test_x, test_y = data_bundle(delay, seed, 2, 512)
            fingerprints = {
                "train_x": tensor_fingerprint(data_bundle(delay, seed, 0, 1536)[0]),
                "test_x": tensor_fingerprint(test_x),
            }
            for name in REPRESENTATIONS:
                info = sources[(name, delay, seed)]
                path = info["path"]
                report = {"representation": name, "delay": delay, "seed": seed, "status": "ok", "stop_reason": None}
                digest = file_sha256(path)
                report["checkpoint_sha256"] = digest
                if digest != info["recorded_sha256"]:
                    report["status"] = "verification_failure"
                    report["stop_reason"] = "checkpoint hash does not match the saved record"
                    reports.append(report)
                    (OUT_DIR / "checkpoint_verification.jsonl").write_text("".join(json.dumps(item) + "\n" for item in reports), encoding="utf-8")
                    raise RuntimeError(report["stop_reason"] + f" for {name} delay {delay} seed {seed}")
                model, payload = load_additive(path)
                expected = saved_correct_count(info["recorded_test_accuracy"], 512)
                actual = correct_count(model, test_x, test_y, 256)
                report["saved_correct_count"] = expected
                report["recomputed_correct_count"] = actual
                if actual != expected:
                    report["status"] = "verification_failure"
                    report["stop_reason"] = f"correct count {actual} does not match saved {expected}"
                    reports.append(report)
                    (OUT_DIR / "checkpoint_verification.jsonl").write_text("".join(json.dumps(item) + "\n" for item in reports), encoding="utf-8")
                    raise RuntimeError(report["stop_reason"])
                if name != "initialized":
                    saved_fp = info["dataset_fingerprints"]
                    if saved_fp["test_x"] != fingerprints["test_x"] or saved_fp["train_x"] != fingerprints["train_x"]:
                        raise RuntimeError(f"regenerated data does not match EXP-011 fingerprints for seed {seed} delay {delay}")
                if name == "initialized":
                    fresh = reconstruct_initialized(seed)
                    matched = recurrent_equal(fresh, model)
                    report["initialization_exact_match"] = matched
                    if not matched:
                        report["status"] = "verification_failure"
                        report["stop_reason"] = "initialized recurrent parameters do not exactly match the checkpoint"
                        reports.append(report)
                        (OUT_DIR / "checkpoint_verification.jsonl").write_text("".join(json.dumps(item) + "\n" for item in reports), encoding="utf-8")
                        raise RuntimeError(report["stop_reason"] + f" at seed {seed} delay {delay}")
                reports.append(report)
    (OUT_DIR / "checkpoint_verification.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in reports), encoding="utf-8",
    )
    (OUT_DIR / "provenance_note.json").write_text(json.dumps({
        "exp007_fingerprints": "not backfilled; the saved EXP-007 fingerprint file is incomplete",
        "dataset_match": "regenerated tensors match the saved EXP-011 fingerprints for each seed and delay",
        "initialization": "exact tensor equality against a freshly seeded additive model, with no optimizer step",
    }, indent=2), encoding="utf-8")

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    metrics = []
    unstarted = []
    for delay in DELAYS:
        for seed in SEEDS:
            if time.perf_counter() - started >= 300:
                for name in REPRESENTATIONS:
                    unstarted.append({"delay": delay, "seed": seed, "representation": name, "reason": "total cap"})
                continue
            train_x, train_y = data_bundle(delay, seed, 0, 1536)
            val_x, val_y = data_bundle(delay, seed, 1, 512)
            test_x, test_y = data_bundle(delay, seed, 2, 512)
            audit_base, _audit_y = data_bundle(delay, seed, 3, 512)
            positive, negative = make_counterfactual_pairs(audit_base)
            if not pairs_differ_only_on_original_bit(positive, negative):
                raise RuntimeError("pair construction failed")
            zero_a = positive.clone()
            zero_b = negative.clone()
            zero_a[:, 0, 0] = 0
            zero_b[:, 0, 0] = 0
            with (OUT_DIR / "feature_fingerprints.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "delay": delay, "seed": seed,
                    "train_x": tensor_fingerprint(train_x), "test_x": tensor_fingerprint(test_x),
                }) + "\n")
            for name in REPRESENTATIONS:
                if time.perf_counter() - started >= 300:
                    unstarted.append({"delay": delay, "seed": seed, "representation": name, "reason": "total cap"})
                    continue
                info = sources[(name, delay, seed)]
                model, _payload = load_additive(info["path"])
                before_hash = recurrent_hash(model)
                train_state = final_states(model, train_x)
                val_state = final_states(model, val_x)
                test_state = final_states(model, test_x)
                if recurrent_hash(model) != before_hash:
                    raise RuntimeError("extraction changed recurrent parameters")
                if not np.array_equal(final_states(model, zero_a[:8]), final_states(model, zero_b[:8])):
                    raise RuntimeError("zero-bit controls did not match")
                classifier, mean, scale, iterations, converged, warnings_found = fit_one(train_state, train_y)
                decision = lambda states: classifier.decision_function(apply_scale(states, mean, scale))
                test_decision = decision(test_state)
                refit_pred = refit_predictions(test_decision)
                targets = test_y.detach().cpu().numpy()
                refit_accuracy = float(np.mean(refit_pred == targets))
                original_test = original_logits(model, test_state)
                original_accuracy = float(np.mean(original_predictions(original_test) == targets))
                if info["reproduce_exp12"]:
                    expected_count = saved_correct_count(info["exp12_accuracy"], 512)
                    actual_count = int(np.sum(refit_pred == targets))
                    saved_bundle = joblib.load(info["exp12_classifier"])
                    saved_decision = saved_bundle["classifier"].decision_function(
                        apply_scale(test_state, saved_bundle["mean"], saved_bundle["scale"])
                    )
                    saved_pred = refit_predictions(saved_decision)
                    if actual_count != expected_count or not np.array_equal(refit_pred, saved_pred):
                        failure = {
                            "representation": name, "delay": delay, "seed": seed,
                            "status": "reproduction_failure",
                            "expected_correct_count": expected_count,
                            "actual_correct_count": actual_count,
                            "prediction_agreement": bool(np.array_equal(refit_pred, saved_pred)),
                        }
                        with (OUT_DIR / "reproduction_checks.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(failure) + "\n")
                        raise RuntimeError(f"EXP-012 reproduction failed for {name} delay {delay} seed {seed}")
                    with (OUT_DIR / "reproduction_checks.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "representation": name, "delay": delay, "seed": seed, "status": "ok",
                            "correct_count": actual_count, "prediction_agreement": True,
                        }) + "\n")
                pos_state = final_states(model, positive)
                neg_state = final_states(model, negative)
                equal = np.all(pos_state == neg_state, axis=1)
                audit = pair_metrics(decision(pos_state), decision(neg_state), refit_predictions)
                audit["final_states_coordinate_equal_fraction"] = float(np.mean(equal))
                record = {
                    "representation": name,
                    "delay": delay,
                    "seed": seed,
                    "status": "ok" if converged else "inconclusive_not_converged",
                    "converged": converged,
                    "iterations": iterations,
                    "convergence_warnings": warnings_found,
                    "original_test_accuracy": original_accuracy,
                    "diagnostic_test_accuracy": refit_accuracy,
                    "diagnostic_test_loss": bce_from_decision(test_decision, binary_labels(test_y)),
                    "original_test_loss": bce_from_decision(original_test, binary_labels(test_y)),
                    "diagnostic_train_accuracy": float(np.mean(refit_predictions(decision(train_state)) == train_y.numpy())),
                    "diagnostic_val_accuracy": float(np.mean(refit_predictions(decision(val_state)) == val_y.numpy())),
                    "diagnostic_success": bool(refit_accuracy >= 0.95),
                    "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                    "intercept": float(classifier.intercept_[0]),
                    "recurrent_hash_unchanged": True,
                    "audit": audit,
                }
                destination = OUT_DIR / "classifiers" / f"{name}_delay{delay}_seed{seed}.joblib"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                joblib.dump({"classifier": classifier, "mean": mean, "scale": scale}, destination)
                metrics.append(record)
                with (OUT_DIR / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({
                    "delay": delay, "seed": seed, "representation": name,
                    "original": round(original_accuracy, 4), "diagnostic": round(refit_accuracy, 4),
                }), flush=True)
    summary = {
        "elapsed_seconds": elapsed(),
        "verified": len(reports),
        "completed": len(metrics),
        "unstarted": unstarted,
        "code_revision": revision,
        "sklearn": sklearn_version(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "delay", "seed", "representation", "status", "original_test_accuracy",
            "diagnostic_test_accuracy", "diagnostic_success", "iterations",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-013")
    print(f"EXP-013 finished in {summary['elapsed_seconds']}s; completed {len(metrics)}; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

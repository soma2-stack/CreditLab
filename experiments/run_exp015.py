"""EXP-015: evaluate saved EXP-014 models at delays 128, 256, and 512.

No training and no classifier fitting.

Registration, before any new score:
    python experiments/run_exp015.py --write-registration

Evaluation, after that registration is committed:
    python experiments/run_exp015.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.models import ResidualTanhRNN  # noqa: E402
from run_exp004 import file_sha256, load_hard_v2, tensor_fingerprint  # noqa: E402
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp009_audit_completion import saved_correct_count  # noqa: E402
from run_exp012 import (  # noqa: E402
    apply_scale,
    bce_from_decision,
    binary_labels,
    final_states,
    original_logits,
    original_predictions,
    pair_metrics,
    recurrent_hash,
    refit_predictions,
    sklearn_version,
)

CONFIG_PATH = ROOT / "configs" / "exp015_length_generalization.yaml"
OUT_DIR = ROOT / "results" / "EXP-015"
EXP14 = ROOT / "results" / "EXP-014"
SEEDS = [307, 311, 313, 317, 331, 337, 347, 349, 353, 359]
REPRESENTATIONS = ("initialized", "full_history", "final_16")
DELAYS = (128, 256, 512)
BLOCKED = {"metrics.jsonl", "runtime_session.json", "summary.json"}
ARTIFACT_DIRS = tuple(
    f"results/EXP-{name}" for name in
    ("001", "001B", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014")
)


def git_revision() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def magnitude_indices(delay: int) -> dict[str, int]:
    return {
        "after_event": 0,
        "halfway_distractor": delay // 2,
        "before_query": delay,
        "after_query": delay + 1,
    }


def model_path(representation: str, seed: int) -> Path:
    return EXP14 / "checkpoints" / f"additive_{representation}_delay128_seed{seed}.pt"


def classifier_path(representation: str, seed: int) -> Path:
    return EXP14 / "classifiers" / f"{representation}_delay128_seed{seed}.joblib"


def load_model(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def draw(count: int, delay: int, seed: int, split_index: int, base_seed: int):
    return load_hard_v2(
        count, delay, seed, split_index,
        noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=base_seed,
    )


def score_logits(logits: np.ndarray, targets: np.ndarray, rule) -> dict:
    if not np.isfinite(logits).all():
        raise FloatingPointError("non-finite logits")
    predictions = rule(logits)
    binary = ((targets + 1.0) / 2.0).astype(np.int64)
    correct = int(np.sum(predictions == targets))
    return {
        "correct_count": correct,
        "accuracy": correct / len(targets),
        "loss": bce_from_decision(logits, binary),
        "success": bool(correct / len(targets) >= 0.95),
        "finite": True,
    }


def hidden_report(states: torch.Tensor, delay: int) -> dict:
    if not torch.isfinite(states).all():
        raise FloatingPointError("non-finite hidden state")
    indices = magnitude_indices(delay)
    report = {
        "max_abs_hidden": float(states.abs().max()),
        "final_hidden_rms": float(torch.sqrt(torch.mean(states[:, -1, :].float() ** 2))),
        "indices": indices,
    }
    for name, index in indices.items():
        step = states[:, index, :]
        report[name] = {
            "index": index,
            "max_abs": float(step.abs().max()),
            "rms": float(torch.sqrt(torch.mean(step.float() ** 2))),
        }
    return report


def rollout(model, inputs: torch.Tensor):
    with torch.no_grad():
        states = model.recurrent_states(inputs)
    return states


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def saved_diagnostics() -> dict[tuple[str, int], dict]:
    rows = [
        json.loads(line)
        for line in (EXP14 / "diagnostic_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {(row["representation"], int(row["seed"])): row for row in rows}


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-015 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            items.append({
                "seed": seed,
                "representation": representation,
                "checkpoint": model_path(representation, seed).relative_to(ROOT).as_posix(),
                "classifier": classifier_path(representation, seed).relative_to(ROOT).as_posix(),
                "checkpoint_sha256": file_sha256(model_path(representation, seed)),
                "classifier_sha256": file_sha256(classifier_path(representation, seed)),
            })
    if len(items) != 30:
        raise RuntimeError(f"expected 30 saved pairs, found {len(items)}")
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-015",
        "delays": [128, 256, 512],
        "base_seed": 4000,
        "artifacts": items,
        "note": "Evaluation only. No training and no refitting.",
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(artifact_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered {len(items)} checkpoint-classifier pairs")


def verify(saved: dict[tuple[str, int], dict]) -> list[dict]:
    reports = []
    for seed in SEEDS:
        test_x, test_y = draw(512, 128, seed, 2, 3000)
        if test_x.shape[1] != 130 or not torch.equal(test_x[:, -1, 2], torch.ones(test_x.shape[0])):
            raise RuntimeError("original test sequences do not put the query at the final step")
        targets = test_y.detach().cpu().numpy()
        for representation in REPRESENTATIONS:
            row = saved[(representation, seed)]
            path = model_path(representation, seed)
            report = {"seed": seed, "representation": representation, "status": "ok", "stop_reason": None}
            digest = file_sha256(path)
            report["checkpoint_sha256"] = digest
            model, payload = load_model(path)
            if payload.get("experiment_id") != "EXP-014" or payload.get("representation") != representation or int(payload.get("seed")) != seed:
                report["status"] = "verification_failure"
                report["stop_reason"] = "checkpoint metadata does not match"
                reports.append(report)
                return reports
            if model.residual_scale != 1.0:
                report["status"] = "verification_failure"
                report["stop_reason"] = "additive scale is not 1"
                reports.append(report)
                return reports
            states = rollout(model, test_x)
            final = states[:, -1, :].double().numpy()
            original_count = int(np.sum(original_predictions(original_logits(model, final)) == targets))
            expected_original = saved_correct_count(row["original_readout_accuracy"], 512)
            report["original_correct_count"] = original_count
            report["expected_original_correct_count"] = expected_original
            bundle = joblib.load(classifier_path(representation, seed))
            decision = bundle["classifier"].decision_function(apply_scale(final, bundle["mean"], bundle["scale"]))
            diagnostic_count = int(np.sum(refit_predictions(decision) == targets))
            expected_diagnostic = saved_correct_count(row["diagnostic_test_accuracy"], 512)
            report["diagnostic_correct_count"] = diagnostic_count
            report["expected_diagnostic_correct_count"] = expected_diagnostic
            if original_count != expected_original or diagnostic_count != expected_diagnostic:
                report["status"] = "verification_failure"
                report["stop_reason"] = "saved correct count was not reproduced"
                reports.append(report)
                return reports
            reports.append(report)
    return reports


def evaluate_seed_delay(model, bundle, inputs, targets, delay: int) -> dict:
    states = rollout(model, inputs)
    final = states[:, -1, :].double().numpy()
    magnitude = hidden_report(states, delay)
    original = score_logits(original_logits(model, final), targets, original_predictions)
    diagnostic = score_logits(
        bundle["classifier"].decision_function(apply_scale(final, bundle["mean"], bundle["scale"])),
        targets,
        refit_predictions,
    )
    return {"final": final, "magnitude": magnitude, "original": original, "diagnostic": diagnostic}


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match Python 3.11.9, PyTorch 2.13.0+cpu, and scikit-learn 1.9.1")
    if [name for name in BLOCKED if (OUT_DIR / name).exists()]:
        raise RuntimeError("EXP-015 outputs already exist. Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-015")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nEvaluation code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-015", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds": 300,
        "no_optimizer": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    saved = saved_diagnostics()
    if len(saved) != 30:
        raise RuntimeError(f"expected 30 EXP-014 diagnostic records, found {len(saved)}")
    reports = verify(saved)
    (OUT_DIR / "checkpoint_verification.jsonl").write_text(
        "".join(json.dumps(report) + "\n" for report in reports), encoding="utf-8",
    )
    if any(report["status"] != "ok" for report in reports) or len(reports) != 30:
        raise RuntimeError("checkpoint verification failed")
    metrics = []
    audits = []
    unstarted = []

    def expired() -> bool:
        return time.perf_counter() - started >= 300

    for delay in DELAYS:
        for seed in SEEDS:
            if expired():
                unstarted.append({"delay": delay, "seed": seed, "reason": "total cap"})
                continue
            inputs, targets_tensor = draw(512, delay, seed, 4, 4000)
            if inputs.shape[1] != delay + 2 or not torch.equal(inputs[:, -1, 2], torch.ones(inputs.shape[0])):
                raise RuntimeError(f"query is not at the final step for delay {delay}")
            if not torch.equal(inputs[:, 0, 1], torch.ones(inputs.shape[0])):
                raise RuntimeError("event marker is not at the first step")
            targets = targets_tensor.detach().cpu().numpy()
            audit_base, _audit_y = draw(512, delay, seed, 5, 4000)
            positive, negative = make_counterfactual_pairs(audit_base)
            if not pairs_differ_only_on_original_bit(positive, negative):
                raise RuntimeError("pair construction failed")
            zero_a, zero_b = positive.clone(), negative.clone()
            zero_a[:, 0, 0] = 0
            zero_b[:, 0, 0] = 0
            with (OUT_DIR / "data_fingerprints.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "delay": delay, "seed": seed,
                    "evaluation": tensor_fingerprint(inputs),
                    "audit_base": tensor_fingerprint(audit_base),
                }) + "\n")
            for representation in REPRESENTATIONS:
                path = model_path(representation, seed)
                model, _payload = load_model(path)
                bundle = joblib.load(classifier_path(representation, seed))
                before_hash = recurrent_hash(model)
                mean_before = np.array(bundle["mean"], copy=True)
                scale_before = np.array(bundle["scale"], copy=True)
                try:
                    scored = evaluate_seed_delay(model, bundle, inputs, targets, delay)
                    if not np.array_equal(final_states(model, zero_a[:4]), final_states(model, zero_b[:4])):
                        raise RuntimeError("zero-bit controls did not match")
                    pos_states = rollout(model, positive)
                    neg_states = rollout(model, negative)
                    pos_final = pos_states[:, -1, :].double().numpy()
                    neg_final = neg_states[:, -1, :].double().numpy()
                    equal = np.all(pos_final == neg_final, axis=1)
                    original_audit = pair_metrics(
                        original_logits(model, pos_final), original_logits(model, neg_final), original_predictions,
                    ) if representation != "initialized" else None
                    diagnostic_decision_pos = bundle["classifier"].decision_function(apply_scale(pos_final, bundle["mean"], bundle["scale"]))
                    diagnostic_decision_neg = bundle["classifier"].decision_function(apply_scale(neg_final, bundle["mean"], bundle["scale"]))
                    diagnostic_audit = pair_metrics(diagnostic_decision_pos, diagnostic_decision_neg, refit_predictions)
                    diagnostic_audit["final_states_coordinate_equal_fraction"] = float(np.mean(equal))
                except FloatingPointError as exc:
                    record = {
                        "delay": delay, "seed": seed, "representation": representation,
                        "status": "numerical_failure", "stop_reason": str(exc),
                    }
                    metrics.append(record)
                    with (OUT_DIR / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    continue
                if recurrent_hash(model) != before_hash or not np.array_equal(bundle["mean"], mean_before) or not np.array_equal(bundle["scale"], scale_before):
                    raise RuntimeError("evaluation changed a frozen model or preprocessing vector")
                endpoints = []
                if representation != "initialized":
                    endpoints.append(("original", scored["original"], original_audit))
                endpoints.append(("diagnostic", scored["diagnostic"], diagnostic_audit))
                for readout, scores, audit in endpoints:
                    record = {
                        "delay": delay, "seed": seed, "representation": representation, "readout": readout,
                        "status": "ok", **scores, "magnitude": scored["magnitude"], "audit": audit,
                    }
                    metrics.append(record)
                    with (OUT_DIR / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                print(json.dumps({
                    "delay": delay, "seed": seed, "representation": representation,
                    "original": None if representation == "initialized" else round(scored["original"]["accuracy"], 4),
                    "diagnostic": round(scored["diagnostic"]["accuracy"], 4),
                }, sort_keys=True), flush=True)
    summary = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "verified": len(reports),
        "completed_endpoint_rows": len(metrics),
        "numerical_failures": sum(1 for row in metrics if row["status"] == "numerical_failure"),
        "unstarted": unstarted,
        "code_revision": revision,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-015")
    print(f"EXP-015 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

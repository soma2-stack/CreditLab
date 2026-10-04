"""EXP-014: new-seed replication of representation versus readout.

Registration, before any accuracy:
    python experiments/run_exp014.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp014.py
"""

from __future__ import annotations

import copy
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

from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import (  # noqa: E402
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    load_hard_v2,
    minibatch_indices,
    parameter_fingerprint,
)
from run_exp005 import make_counterfactual_pairs, pairs_differ_only_on_original_bit  # noqa: E402
from run_exp010 import diagnose, forward_regime, training_graph_event  # noqa: E402
from run_exp011 import apply_clipping  # noqa: E402
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
from run_exp001 import evaluate, probe_fit_score  # noqa: E402
from creditlab.diagnostics import hidden_magnitude_stats  # noqa: E402
from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp005 import bit_and_marker_sensitivity  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp014_replication.yaml"
OUT_DIR = ROOT / "results" / "EXP-014"
SEEDS = [307, 311, 313, 317, 331, 337, 347, 349, 353, 359]
OLD_SEEDS = {17, 29, 43, 101, 113, 127, 139, 151, 163, 179, 191, 211, 223}
REGIMES = ("full_history", "final_16")
REPRESENTATIONS = ("initialized", "full_history", "final_16")
BLOCKED = {"raw_metrics.jsonl", "runtime_session.json", "summary.json"}
ARTIFACT_DIRS = tuple(
    f"results/EXP-{name}" for name in
    ("001", "001B", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013")
)


def git_revision() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def build_initialized(cfg: dict, seed: int) -> ResidualTanhRNN:
    seed_everything(seed)
    return ResidualTanhRNN(
        cfg["model"]["input_size"], cfg["model"]["hidden_size"], cfg["model"]["output_size"],
        residual_scale=float(cfg["model"]["residual_scale"]),
    )


def train_one(cfg, model, regime, bundle, indices, seed, per_run_cap):
    started = time.perf_counter()
    updates_target = int(cfg["training"]["updates_per_run"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-014", "representation": regime, "delay": 128, "seed": seed,
        "status": "ok", "stop_reason": None, "code_revision": git_revision(),
    }
    loss_curve, checkpoints, clipped = [], [], 0
    try:
        checkpoints.append(diagnose(model, bundle["val_x"], bundle["val_y"], 0, regime, cfg, None))
        if regime == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
            raise RuntimeError("the original event was not blocked in the final-16 training graph")
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > per_run_cap:
                raise TimeoutError(f"CPU budget of {per_run_cap:.0f}s exhausted at update {update}")
            xb, yb = bundle["train_x"][idx], bundle["train_y"][idx]
            logits, states, cut = forward_regime(model, xb, regime)
            if regime == "final_16" and cut != max(xb.shape[1] - 16, 0):
                raise RuntimeError("final-16 cut does not match the registered rule")
            if not torch.isfinite(states).all() or not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite state or logit at update {update}")
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            clip_record = apply_clipping(list(model.parameters()), True)
            if clip_record["clipping_applied"]:
                clipped += 1
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError(f"non-finite parameter after update {update}")
            loss_curve.append(float(loss.detach()))
            if (update + 1) in (50, 100, 200, 400):
                checkpoints.append(diagnose(model, bundle["val_x"], bundle["val_y"], update + 1, regime, cfg, None))
                if regime == "final_16" and not checkpoints[-1]["training_graph_event_blocked"]:
                    raise RuntimeError("the final-16 training graph stopped blocking the early inputs")
        tr = cfg["training"]
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], tr["eval_batch_size"])
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], tr["eval_batch_size"])
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], tr["eval_batch_size"])
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        bit, marker = bit_and_marker_sensitivity(model, bundle["test_x"])
        graph = training_graph_event(model, bundle["test_x"], regime)
        record.update(
            updates_completed=updates_target, train_loss=train_loss, train_accuracy=train_acc,
            val_loss=val_loss, val_accuracy=val_acc, test_loss=test_loss, test_accuracy=test_acc,
            original_success=bool(test_acc >= 0.95), retention_probe_r2=retention,
            max_abs_hidden=magnitude["max_abs_hidden"], final_hidden_rms=magnitude["final_hidden_rms"],
            forward_bit_sensitivity_mean_abs=float(bit.abs().mean()),
            forward_marker_sensitivity_mean_abs=float(marker.abs().mean()),
            grad_clip_fraction=clipped / updates_target, runtime_seconds=round(time.perf_counter() - started, 3),
        )
        record.update(graph)
    except (FloatingPointError, TimeoutError, RuntimeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, TimeoutError):
            status = "budget_stop"
        elif "blocked" in str(exc) or "cut does not" in str(exc):
            status = "implementation_bug"
        elif isinstance(exc, FloatingPointError):
            status = "numerical_failure"
        else:
            status = "failure"
        record.update(status=status, stop_reason=message, updates_completed=len(loss_curve), original_success=False,
                      runtime_seconds=round(time.perf_counter() - started, 3))
    return record, loss_curve, checkpoints


def save_model(model, seed: int, representation: str) -> str:
    destination = OUT_DIR / "checkpoints" / f"additive_{representation}_delay128_seed{seed}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_suffix(".pt.partial")
    torch.save({"experiment_id": "EXP-014", "representation": representation, "delay": 128, "seed": seed, "state_dict": model.state_dict()}, temporary)
    temporary.replace(destination)
    return file_sha256(destination)


def fit_diagnostic(model, train_x, train_y, val_x, val_y, test_x, test_y):
    before = recurrent_hash(model)
    train_state = final_states(model, train_x)
    val_state = final_states(model, val_x)
    test_state = final_states(model, test_x)
    if recurrent_hash(model) != before:
        raise RuntimeError("diagnostic fitting changed recurrent parameters")
    mean, scale = fit_scale(train_state)
    classifier = make_classifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(apply_scale(train_state, mean, scale), binary_labels(train_y))
    iterations = int(np.max(classifier.n_iter_))
    converged = iterations < 2000 and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    decision = lambda states: classifier.decision_function(apply_scale(states, mean, scale))
    targets = test_y.detach().cpu().numpy()
    test_decision = decision(test_state)
    accuracy = float(np.mean(refit_predictions(test_decision) == targets))
    original = original_logits(model, test_state)
    original_accuracy = float(np.mean(original_predictions(original) == targets))
    return {
        "classifier": classifier, "mean": mean, "scale": scale, "decision": decision,
        "test_state": test_state, "iterations": iterations, "converged": converged,
        "warnings": [str(item.message) for item in caught],
        "diagnostic_test_accuracy": accuracy,
        "diagnostic_test_loss": bce_from_decision(test_decision, binary_labels(test_y)),
        "diagnostic_train_accuracy": float(np.mean(refit_predictions(decision(train_state)) == train_y.numpy())),
        "diagnostic_val_accuracy": float(np.mean(refit_predictions(decision(val_state)) == val_y.numpy())),
        "original_readout_accuracy": original_accuracy,
        "original_test_loss": bce_from_decision(original, binary_labels(test_y)),
        "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
        "intercept": float(classifier.intercept_[0]),
        "recurrent_hash": before,
    }


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if target.exists():
            for path in sorted(item for item in target.rglob("*") if item.is_file()):
                records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-014 registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    if set(cfg["seeds"]) & OLD_SEEDS:
        raise RuntimeError("a new seed collides with an earlier cohort")
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-014", "seeds": cfg["seeds"], "base_seed": cfg["data"]["base_seed"],
        "delay": 128, "k": 16, "classifier": cfg["classifier"], "budget_seconds": 600,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(artifact_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print("registered EXP-014")


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if [name for name in BLOCKED if (OUT_DIR / name).exists()]:
        raise RuntimeError("EXP-014 outputs already exist. Automatic resume is disabled.")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-014")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-014", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds": 600,
    }, indent=2, sort_keys=True), encoding="utf-8")
    records, unstarted, stopped = [], [], None

    def elapsed():
        return round(time.perf_counter() - started, 3)

    for seed in cfg["seeds"]:
        if time.perf_counter() - started >= 600 or stopped:
            unstarted.append({"seed": seed, "reason": "implementation stop" if stopped else "total cap"})
            continue
        bundle = dataset_bundle(cfg, 128, seed)
        data_fp = dataset_fingerprints(bundle)
        indices = minibatch_indices(seed, 128, bundle["train_x"].shape[0], cfg["training"]["batch_size"], cfg["training"]["updates_per_run"])
        index_hash = index_fingerprint(indices)
        initialized = build_initialized(cfg, seed)
        full = copy.deepcopy(initialized)
        truncated = copy.deepcopy(initialized)
        if parameter_fingerprint(initialized) != parameter_fingerprint(full) or parameter_fingerprint(initialized) != parameter_fingerprint(truncated):
            raise RuntimeError(f"training copies do not match initialization at seed {seed}")
        with torch.no_grad():
            probe = bundle["train_x"][:4]
            reference, _ = initialized(probe)
            other, _states, _cut = forward_regime(truncated, probe, "final_16")
        if not torch.equal(reference, other):
            raise RuntimeError("detached forward values do not match")
        append_jsonl(OUT_DIR / "fingerprints.jsonl", {
            "seed": seed, "delay": 128, "init_fingerprint": parameter_fingerprint(initialized),
            "minibatch_fingerprint": index_hash, "dataset_fingerprints": data_fp,
        })
        init_hash = recurrent_hash(initialized)
        init_record = {
            "experiment_id": "EXP-014", "representation": "initialized", "delay": 128, "seed": seed,
            "status": "ok", "updates_completed": 0, "checkpoint_sha256": save_model(initialized, seed, "initialized"),
        }
        if recurrent_hash(initialized) != init_hash:
            raise RuntimeError("saving the initialized model changed its weights")
        records.append(init_record)
        append_jsonl(OUT_DIR / "raw_metrics.jsonl", init_record)
        models = {"initialized": initialized, "full_history": full, "final_16": truncated}
        for regime in REGIMES:
            if time.perf_counter() - started >= 600 or stopped:
                unstarted.append({"seed": seed, "representation": regime, "reason": "implementation stop" if stopped else "total cap"})
                continue
            remaining = 600 - (time.perf_counter() - started)
            append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                "event": "start", "seed": seed, "representation": regime, "started_unix": time.time(),
                "session_elapsed_seconds": elapsed(),
            })
            print(f"training {regime} seed {seed}", flush=True)
            record, curve, checks = train_one(cfg, models[regime], regime, bundle, indices, seed, min(300.0, remaining))
            if record["status"] == "implementation_bug":
                stopped = {"seed": seed, "representation": regime, "reason": record["stop_reason"]}
            if record["status"] == "ok":
                record["checkpoint_sha256"] = save_model(models[regime], seed, regime)
            records.append(record)
            append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
            append_jsonl(OUT_DIR / "loss_curves.jsonl", {"seed": seed, "representation": regime, "loss_curve": curve})
            for item in checks:
                append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {"seed": seed, "representation": regime, **item})
            append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                "event": "finish", "seed": seed, "representation": regime, "status": record["status"],
                "runtime_seconds": record.get("runtime_seconds"), "session_elapsed_seconds": elapsed(),
            })
            print(json.dumps({"seed": seed, "representation": regime, "status": record["status"], "test_accuracy": record.get("test_accuracy")}, sort_keys=True), flush=True)
        if stopped or time.perf_counter() - started >= 600:
            for representation in REPRESENTATIONS:
                unstarted.append({"seed": seed, "representation": representation, "reason": "diagnostic cap"})
            continue
        audit_base, _y = load_hard_v2(512, 128, seed, 3, noise_std=0.3, competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=3000)
        positive, negative = make_counterfactual_pairs(audit_base)
        if not pairs_differ_only_on_original_bit(positive, negative):
            raise RuntimeError("pair construction failed")
        zero_a, zero_b = positive.clone(), negative.clone()
        zero_a[:, 0, 0] = 0
        zero_b[:, 0, 0] = 0
        for representation, model in models.items():
            if representation != "initialized" and not any(row["seed"] == seed and row["representation"] == representation and row["status"] == "ok" for row in records):
                continue
            fitted = fit_diagnostic(model, bundle["train_x"], bundle["train_y"], bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
            if not np.array_equal(final_states(model, zero_a[:8]), final_states(model, zero_b[:8])):
                raise RuntimeError("zero-bit controls did not match")
            pos_state = final_states(model, positive)
            neg_state = final_states(model, negative)
            audit = pair_metrics(
                fitted["decision"](pos_state), fitted["decision"](neg_state), refit_predictions,
            )
            audit["final_states_coordinate_equal_fraction"] = float(np.mean(np.all(pos_state == neg_state, axis=1)))
            if representation != "initialized":
                original_audit = pair_metrics(original_logits(model, pos_state), original_logits(model, neg_state), original_predictions)
            else:
                original_audit = None
            destination = OUT_DIR / "classifiers" / f"{representation}_delay128_seed{seed}.joblib"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            joblib.dump({"classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"]}, destination)
            diagnostic = {
                "seed": seed, "representation": representation, "converged": fitted["converged"],
                "iterations": fitted["iterations"], "warnings": fitted["warnings"],
                "original_readout_accuracy": fitted["original_readout_accuracy"],
                "original_test_loss": fitted["original_test_loss"],
                "diagnostic_test_accuracy": fitted["diagnostic_test_accuracy"],
                "diagnostic_test_loss": fitted["diagnostic_test_loss"],
                "diagnostic_train_accuracy": fitted["diagnostic_train_accuracy"],
                "diagnostic_val_accuracy": fitted["diagnostic_val_accuracy"],
                "diagnostic_success": bool(fitted["diagnostic_test_accuracy"] >= 0.95),
                "original_success": bool(fitted["original_readout_accuracy"] >= 0.95) if representation != "initialized" else None,
                "coefficient_l2": fitted["coefficient_l2"], "intercept": fitted["intercept"],
                "diagnostic_audit": audit, "original_audit": original_audit,
                "classifier_sha256": file_sha256(destination),
            }
            append_jsonl(OUT_DIR / "diagnostic_metrics.jsonl", diagnostic)
            print(json.dumps({
                "seed": seed, "representation": representation,
                "original": round(fitted["original_readout_accuracy"], 4),
                "diagnostic": round(fitted["diagnostic_test_accuracy"], 4),
            }, sort_keys=True), flush=True)
    summary = {
        "elapsed_seconds": elapsed(), "completed_training_records": len(records),
        "unstarted": unstarted, "implementation_stop": stopped, "code_revision": revision,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "representation", "status", "test_accuracy", "original_success", "retention_probe_r2", "grad_clip_fraction", "runtime_seconds", "stop_reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("preexisting artifacts changed during EXP-014")
    if stopped:
        raise RuntimeError(f"implementation stop: {stopped}")
    print(f"EXP-014 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

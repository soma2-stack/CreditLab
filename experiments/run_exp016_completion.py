"""EXP-016 completion: the 14 training jobs the interrupted session never scored.

Registration, before any new accuracy:
    python experiments/run_exp016_completion.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp016_completion.py

This does not call the original runner, so it does not delete or bypass
that runner's restart refusal. New files go under results/EXP-016/completion/.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp004 import (  # noqa: E402
    file_sha256,
    index_fingerprint,
    minibatch_indices,
    parameter_fingerprint,
)
from run_exp005 import make_counterfactual_pairs  # noqa: E402
from run_exp012 import apply_scale, sklearn_version  # noqa: E402
from run_exp016 import (  # noqa: E402
    CONDITIONS,
    ImplementationError,
    artifact_manifest,
    atomic_torch_save,
    audit_pair,
    build_bundle,
    condition_name,
    dataset_fingerprints,
    fit_diagnostic,
    load_saved_model,
    matched_models,
    noise_tag,
    shared_noise_draw,
    train_one,
)

ORIGINAL = ROOT / "results" / "EXP-016"
COMPLETION = ORIGINAL / "completion"
FULL_SEEDS = (401, 409, 419, 421, 431, 433)
REUSED = tuple(
    {"seed": seed, "model_type": model_type, "noise": noise}
    for seed in FULL_SEEDS
    for model_type, noise in CONDITIONS
) + (
    {"seed": 439, "model_type": "vanilla_tanh_rnn", "noise": 0.3},
    {"seed": 439, "model_type": "additive_tanh_rnn", "noise": 0.3},
)
JOBS = (
    {"seed": 439, "model_type": "vanilla_tanh_rnn", "noise": 1.0, "attempt": 2, "role": "restart_from_initialization"},
    {"seed": 439, "model_type": "additive_tanh_rnn", "noise": 1.0, "attempt": 1, "role": "previously_unstarted"},
) + tuple(
    {"seed": seed, "model_type": model_type, "noise": noise, "attempt": 1, "role": "previously_unstarted"}
    for seed in (443, 449, 457)
    for model_type, noise in CONDITIONS
)
DIAGNOSTIC_SEEDS = (439, 443, 449, 457)
BLOCKED = {"raw_metrics.jsonl", "runtime_session.json", "summary.json"}
TOTAL_BUDGET = 300.0
PER_RUN_BUDGET = 300.0


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def key_of(seed: int, model_type: str, noise: float) -> tuple:
    return (int(seed), model_type, noise_tag(float(noise)))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    folder = ORIGINAL
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        if "completion" in path.parts:
            continue
        records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def saved_integer(accuracy: float, count: int = 512) -> int:
    value = float(accuracy) * count
    nearest = round(value)
    if abs(value - nearest) > 1e-4:
        raise ImplementationError(f"saved accuracy {accuracy} is not an integer count out of {count}")
    return int(nearest)


def integer_correct(model, inputs: torch.Tensor, targets: torch.Tensor) -> int:
    correct = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], 256):
            logits, _hidden = model(inputs[start:start + 256])
            final = logits[:, -1, 0]
            predictions = torch.sign(final)
            predictions = torch.where(predictions == 0, torch.ones_like(predictions), predictions)
            correct += int((predictions == targets[start:start + 256]).sum())
    return correct


def interruption_cause() -> dict:
    rows = load_jsonl(ORIGINAL / "runtime_log.jsonl")
    starts = [row for row in rows if row.get("seed") == 439 and row.get("condition") == "vanilla_noise1p0" and row.get("event") == "start"]
    finishes = [row for row in rows if row.get("seed") == 439 and row.get("condition") == "vanilla_noise1p0" and row.get("event") == "finish"]
    stderr_files = [path.name for path in ORIGINAL.glob("*stderr*")]
    exception_text = any("Traceback" in json.dumps(row) or "Error" in json.dumps(row.get("status", "")) for row in rows)
    return {
        "condition": "vanilla_noise1p0",
        "seed": 439,
        "start_records": len(starts),
        "finish_records": len(finishes),
        "last_recorded_session_seconds": None if not starts else starts[-1].get("session_elapsed_seconds"),
        "stderr_files": stderr_files,
        "exception_text_in_runtime_log": exception_text,
        "cause": "unknown",
        "note": "A start record has no finish record. No exception text and no stderr file were saved. This is not evidence of a timeout or an out-of-memory event.",
    }


def write_registration() -> None:
    destination = COMPLETION / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-016 completion registration")
    if len(REUSED) != 26 or len(JOBS) != 14:
        raise ImplementationError("the continuation manifest is not 26 reused conditions and 14 jobs")
    reused_keys = {key_of(item["seed"], item["model_type"], item["noise"]) for item in REUSED}
    job_keys = {key_of(item["seed"], item["model_type"], item["noise"]) for item in JOBS}
    if reused_keys & job_keys:
        raise ImplementationError("a reused condition is also scheduled for training")
    COMPLETION.mkdir(parents=True, exist_ok=True)
    cause = interruption_cause()
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-016",
        "kind": "authorized_completion",
        "not_experiment": "EXP-017",
        "original_session": "interrupted, unknown duration and cause",
        "does_not_reconstruct_original_runtime": True,
        "does_not_claim_original_budget_compliance": True,
        "reused_completed_conditions": list(REUSED),
        "restarted_attempt": {
            "seed": 439,
            "model_type": "vanilla_tanh_rnn",
            "noise": 1.0,
            "attempt": 2,
            "from": "original initialization",
            "previous_attempt": "interrupted, no saved result or checkpoint, not a scored seed",
        },
        "jobs": list(JOBS),
        "settings": "unchanged EXP-016 configuration and implementation",
        "verification_rules": [
            "exactly one completed result per reused condition",
            "checkpoint hash and metadata must match",
            "integer test correct count must be reproduced",
            "saved initialization, data, and minibatch fingerprints must match",
            "seed 439 regeneration must match its completed noise-0.3 records",
            "existing 24 diagnostic fits and audits are reused, not repeated",
        ],
        "budget_seconds_total": TOTAL_BUDGET,
        "budget_seconds_per_training_condition": PER_RUN_BUDGET,
        "batching": "one training condition per process",
        "interruption_policy": "preserve records, identify the condition, stop, no automatic retry",
        "interruption_cause": cause,
    }, sort_keys=False), encoding="utf-8")
    (COMPLETION / "historical_hashes_before.json").write_text(
        json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print("registered EXP-016 completion; 14 jobs, 26 reused")


def original_rows() -> list[dict]:
    return [row for row in load_jsonl(ORIGINAL / "raw_metrics.jsonl") if row.get("status") == "ok"]


def verify_reused(bundles: dict) -> list[dict]:
    rows = original_rows()
    found: dict[tuple, dict] = {}
    for row in rows:
        identity = key_of(row["seed"], row["model_type"], row["noise"])
        if identity in found:
            raise ImplementationError(f"more than one completed result for {identity}")
        found[identity] = row
    expected = {key_of(item["seed"], item["model_type"], item["noise"]) for item in REUSED}
    if set(found) != expected:
        raise ImplementationError(f"reused results do not match the 26-condition list: {sorted(set(found) ^ expected)[:8]}")
    fingerprint_rows = load_jsonl(ORIGINAL / "fingerprints.jsonl")
    if len(fingerprint_rows) != len({int(row["seed"]) for row in fingerprint_rows}):
        raise ImplementationError("fingerprint rows are not one per seed")
    fingerprints = {int(row["seed"]): row for row in fingerprint_rows}
    checks = []
    for item in REUSED:
        identity = key_of(item["seed"], item["model_type"], item["noise"])
        row = found[identity]
        path = ORIGINAL / "checkpoints" / f"{row['condition']}_delay128_seed{row['seed']}.pt"
        digest = file_sha256(path)
        if digest != row["checkpoint_sha256"]:
            raise ImplementationError(f"checkpoint hash mismatch for {identity}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("experiment_id") != "EXP-016" or payload.get("seed") != row["seed"]:
            raise ImplementationError(f"checkpoint metadata mismatch for {identity}")
        if payload.get("model_type") != row["model_type"] or noise_tag(float(payload["noise"])) != noise_tag(float(row["noise"])):
            raise ImplementationError(f"checkpoint condition mismatch for {identity}")
        if int(row["seed"]) not in bundles:
            bundles[int(row["seed"])] = build_bundle(int(row["seed"]), 6000)
        bundle = bundles[int(row["seed"])][float(row["noise"])]
        model = load_saved_model(path, row["model_type"])
        counted = integer_correct(model, bundle["test_x"], bundle["test_y"])
        expected_count = saved_integer(row["test_accuracy"])
        if counted != expected_count:
            raise ImplementationError(f"correct count {counted} != saved {expected_count} for {identity}")
        checks.append({"seed": row["seed"], "condition": row["condition"], "sha256": digest, "correct": counted})
    for seed, saved in fingerprints.items():
        if seed not in bundles:
            bundles[seed] = build_bundle(seed, 6000)
        models = matched_models(seed)
        if parameter_fingerprint(models[("vanilla_tanh_rnn", 0.3)]) != saved["init_fingerprint"]:
            raise ImplementationError(f"initialization fingerprint mismatch for seed {seed}")
        if dataset_fingerprints(bundles[seed][0.3]) != saved["noise_0p3"]:
            raise ImplementationError(f"noise 0.3 fingerprint mismatch for seed {seed}")
        if dataset_fingerprints(bundles[seed][1.0]) != saved["noise_1p0"]:
            raise ImplementationError(f"noise 1.0 fingerprint mismatch for seed {seed}")
        indices = minibatch_indices(seed, 128, 1536, 64, 400)
        if index_fingerprint(indices) != saved["minibatch_fingerprint"]:
            raise ImplementationError(f"minibatch fingerprint mismatch for seed {seed}")
        if seed == 439:
            trained = load_saved_model(ORIGINAL / "checkpoints" / "vanilla_noise0p3_delay128_seed439.pt", "vanilla_tanh_rnn")
            if parameter_fingerprint(trained) == parameter_fingerprint(models[("vanilla_tanh_rnn", 1.0)]):
                raise ImplementationError("seed 439 restart matches a trained checkpoint")
    diagnostics = load_jsonl(ORIGINAL / "diagnostic_metrics.jsonl")
    expected_diagnostics = {(seed, condition_name(model_type, noise)) for seed in FULL_SEEDS for model_type, noise in CONDITIONS}
    found_diagnostics = {(int(row["seed"]), row["condition"]) for row in diagnostics}
    if found_diagnostics != expected_diagnostics or len(diagnostics) != 24:
        raise ImplementationError("the 24 reused diagnostic records are not exactly the first six seeds")
    for row in diagnostics:
        path = ORIGINAL / "classifiers" / f"{row['condition']}_delay128_seed{row['seed']}.joblib"
        if file_sha256(path) != row["classifier_sha256"]:
            raise ImplementationError(f"diagnostic classifier hash mismatch for seed {row['seed']} {row['condition']}")
    return checks


def save_completion_classifier(fitted: dict, seed: int, model_type: str, noise: float) -> str:
    destination = COMPLETION / "classifiers" / f"{condition_name(model_type, noise)}_delay128_seed{seed}.joblib"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".joblib.partial")
    joblib.dump({"classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"]}, temporary)
    temporary.replace(destination)
    loaded = joblib.load(destination)
    probe = fitted["mean"].reshape(1, -1)
    if not (loaded["classifier"].decision_function(apply_scale(probe, loaded["mean"], loaded["scale"]))
            == fitted["classifier"].decision_function(apply_scale(probe, fitted["mean"], fitted["scale"]))).all():
        raise ImplementationError("reloaded completion classifier changed its decision")
    return file_sha256(destination)


def train_job(job: dict, seconds_left: float) -> None:
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    seed = int(job["seed"])
    model_type = job["model_type"]
    noise = float(job["noise"])
    name = condition_name(model_type, noise)
    bundles = build_bundle(seed, 6000)
    models = matched_models(seed)
    fingerprints = {int(row["seed"]): row for row in load_jsonl(ORIGINAL / "fingerprints.jsonl")}
    if seed in fingerprints and parameter_fingerprint(models[(model_type, noise)]) != fingerprints[seed]["init_fingerprint"]:
        raise ImplementationError(f"seed {seed} regeneration does not match the saved initialization")
    if seed == 439 and model_type == "vanilla_tanh_rnn" and noise == 1.0:
        trained = load_saved_model(ORIGINAL / "checkpoints" / "vanilla_noise0p3_delay128_seed439.pt", "vanilla_tanh_rnn")
        if parameter_fingerprint(trained) == parameter_fingerprint(models[(model_type, noise)]):
            raise ImplementationError("restart would reuse the trained noise-0.3 model")
    indices = minibatch_indices(seed, 128, 1536, 64, 400)
    started = time.perf_counter()
    deadline = started + min(PER_RUN_BUDGET, seconds_left)
    record, curve, checks = train_one(
        models[(model_type, noise)], model_type, noise, bundles[noise], indices, seed, min(PER_RUN_BUDGET, seconds_left), deadline,
    )
    record["attempt"] = int(job["attempt"])
    record["role"] = job["role"]
    record["source_session"] = "completion"
    record["parent_revision"] = git_revision()
    if record["status"] == "ok":
        destination = COMPLETION / "checkpoints" / f"{name}_delay128_seed{seed}.pt"
        record["checkpoint_sha256"] = atomic_torch_save(destination, {
            "experiment_id": "EXP-016",
            "source_session": "completion",
            "attempt": int(job["attempt"]),
            "model_type": model_type,
            "noise": noise,
            "delay": 128,
            "seed": seed,
            "state_dict": models[(model_type, noise)].state_dict(),
        })
    append_jsonl(COMPLETION / "raw_metrics.jsonl", record)
    append_jsonl(COMPLETION / "loss_curves.jsonl", {"seed": seed, "condition": name, "attempt": job["attempt"], "loss_curve": curve})
    for item in checks:
        append_jsonl(COMPLETION / "checkpoint_diagnostics.jsonl", {"seed": seed, "condition": name, **item})
    print(json.dumps({"seed": seed, "condition": name, "status": record["status"], "test_accuracy": record.get("test_accuracy")}, sort_keys=True), flush=True)


def diagnose_one(seed: int, model_type: str, noise: float, model, bundles: dict, audits: dict) -> None:
    if seed not in bundles:
        bundles[seed] = build_bundle(seed, 6000)
    bundle = bundles[seed][noise]
    fitted = fit_diagnostic(model, bundle["train_x"], bundle["train_y"], bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
    digest = save_completion_classifier(fitted, seed, model_type, noise)
    if seed not in audits:
        audits[seed] = shared_noise_draw(512, 128, seed, 3, 6000)
    positive, negative = make_counterfactual_pairs(audits[seed][noise]["inputs"])
    audit = audit_pair(model, fitted, positive, negative)
    append_jsonl(COMPLETION / "diagnostic_metrics.jsonl", {
        "seed": seed,
        "condition": condition_name(model_type, noise),
        "model_type": model_type,
        "noise": noise,
        "source_session": "completion",
        "fit_rows": fitted["fit_rows"],
        "iterations": fitted["iterations"],
        "converged": fitted["converged"],
        "diagnostic_test_accuracy": fitted["diagnostic_test_accuracy"],
        "diagnostic_test_loss": fitted["diagnostic_test_loss"],
        "diagnostic_success": fitted["diagnostic_success"],
        "classifier_sha256": digest,
        "audit": audit,
    })


def model_for_diagnostic(seed: int, model_type: str, noise: float):
    name = condition_name(model_type, noise)
    completion_path = COMPLETION / "checkpoints" / f"{name}_delay128_seed{seed}.pt"
    original_path = ORIGINAL / "checkpoints" / f"{name}_delay128_seed{seed}.pt"
    path = completion_path if completion_path.exists() else original_path
    if not path.exists():
        raise ImplementationError(f"no checkpoint available for diagnostic {name} seed {seed}")
    return load_saved_model(path, model_type)


def combine(new_rows: list[dict], new_diagnostics: list[dict], elapsed: float, unstarted: list[dict]) -> dict:
    original = original_rows()
    original_diagnostics = load_jsonl(ORIGINAL / "diagnostic_metrics.jsonl")
    by_train = {}
    for row in original:
        item = dict(row)
        item["source_session"] = "original"
        item["attempt"] = 1
        by_train[key_of(row["seed"], row["model_type"], row["noise"])] = item
    for row in new_rows:
        identity = key_of(row["seed"], row["model_type"], row["noise"])
        if identity in by_train:
            raise ImplementationError(f"completion duplicated a scored condition {identity}")
        by_train[identity] = row
    by_diagnostic = {}
    for row in original_diagnostics:
        item = dict(row)
        item["source_session"] = "original"
        by_diagnostic[(int(row["seed"]), row["condition"])] = item
    for row in new_diagnostics:
        identity = (int(row["seed"]), row["condition"])
        if identity in by_diagnostic:
            raise ImplementationError(f"completion duplicated a diagnostic {identity}")
        by_diagnostic[identity] = row
    table = []
    for seed in (401, 409, 419, 421, 431, 433, 439, 443, 449, 457):
        entry = {"seed": seed}
        for model_type, noise in CONDITIONS:
            name = condition_name(model_type, noise)
            train = by_train.get(key_of(seed, model_type, noise))
            diagnostic = by_diagnostic.get((seed, name))
            entry[name] = None if train is None else {
                "source_session": train.get("source_session"),
                "attempt": train.get("attempt"),
                "status": train.get("status"),
                "test_accuracy": train.get("test_accuracy"),
                "original_success": train.get("original_success"),
                "retention_probe_r2": train.get("retention_probe_r2"),
                "diagnostic_test_accuracy": None if diagnostic is None else diagnostic.get("diagnostic_test_accuracy"),
                "diagnostic_success": None if diagnostic is None else diagnostic.get("diagnostic_success"),
                "diagnostic_source": None if diagnostic is None else diagnostic.get("source_session"),
                "audit": None if diagnostic is None else diagnostic.get("audit"),
            }
        table.append(entry)
    return {
        "experiment_id": "EXP-016",
        "kind": "authorized_completion",
        "elapsed_seconds": elapsed,
        "original_runtime_unknown": True,
        "scored_conditions": len(by_train),
        "unstarted": unstarted,
        "per_seed": table,
    }


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if "--train-one" in sys.argv:
        payload = json.loads(sys.argv[sys.argv.index("--train-one") + 1])
        train_job(payload["job"], float(payload["seconds_left"]))
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if any((COMPLETION / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-016 completion outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    before = json.loads((COMPLETION / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before the EXP-016 completion")
    started = time.perf_counter()
    deadline = started + TOTAL_BUDGET
    revision = git_revision()
    (COMPLETION / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\n"
        f"scikit-learn: {sklearn_version()}\nCompletion code revision: {revision}\n",
        encoding="utf-8",
    )
    (COMPLETION / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-016",
        "kind": "authorized_completion",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process, including verification",
        "budget_seconds_total": TOTAL_BUDGET,
        "original_runtime": "unknown; not reconstructed",
    }, indent=2, sort_keys=True), encoding="utf-8")
    bundles: dict = {}
    checks = verify_reused(bundles)
    (COMPLETION / "verification.json").write_text(json.dumps({
        "reused": len(checks),
        "checks": checks,
        "interruption_cause": interruption_cause(),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"verified {len(checks)} reused checkpoints", flush=True)
    unstarted = []
    for index, job in enumerate(JOBS):
        name = condition_name(job["model_type"], job["noise"])
        remaining = deadline - time.perf_counter()
        if remaining <= 1:
            unstarted.append({"seed": job["seed"], "condition": name, "reason": "continuation allowance exhausted"})
            continue
        log_dir = COMPLETION / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}_seed{job['seed']}_attempt{job['attempt']}.stdout.txt"
        stderr_path = log_dir / f"{name}_seed{job['seed']}_attempt{job['attempt']}.stderr.txt"
        started_job = time.perf_counter()
        append_jsonl(COMPLETION / "runtime_log.jsonl", {
            "event": "start", "seed": job["seed"], "condition": name, "attempt": job["attempt"],
            "session_elapsed_seconds": round(started_job - started, 3),
        })
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--train-one", json.dumps({"job": job, "seconds_left": remaining})],
                    cwd=str(ROOT), stdout=stdout, stderr=stderr, timeout=remaining,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        append_jsonl(COMPLETION / "runtime_log.jsonl", {
            "event": "finish", "seed": job["seed"], "condition": name, "attempt": job["attempt"],
            "exit_code": exit_code,
            "runtime_seconds": round(time.perf_counter() - started_job, 3),
            "session_elapsed_seconds": round(time.perf_counter() - started, 3),
            "stderr_path": stderr_path.relative_to(ROOT).as_posix(),
        })
        if exit_code != 0:
            unstarted.extend(
                {"seed": later["seed"], "condition": condition_name(later["model_type"], later["noise"]), "reason": f"stopped after {name} seed {job['seed']} exit {exit_code}"}
                for later in JOBS[index + 1:]
            )
            break
    new_rows = load_jsonl(COMPLETION / "raw_metrics.jsonl")
    audits: dict = {}
    for seed in DIAGNOSTIC_SEEDS:
        for model_type, noise in CONDITIONS:
            name = condition_name(model_type, noise)
            if time.perf_counter() >= deadline:
                unstarted.append({"seed": seed, "condition": name, "stage": "diagnostic", "reason": "continuation allowance exhausted"})
                continue
            if not any(row.get("seed") == seed and row.get("condition") == name and row.get("status") == "ok" for row in original_rows() + new_rows):
                unstarted.append({"seed": seed, "condition": name, "stage": "diagnostic", "reason": "training was not completed"})
                continue
            diagnose_one(seed, model_type, noise, model_for_diagnostic(seed, model_type, noise), bundles, audits)
            print(json.dumps({"seed": seed, "condition": name, "stage": "diagnostic"}, sort_keys=True), flush=True)
    summary = combine(new_rows, load_jsonl(COMPLETION / "diagnostic_metrics.jsonl"), round(time.perf_counter() - started, 3), unstarted)
    (COMPLETION / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["seed,condition,source_session,attempt,status,test_accuracy,original_success,diagnostic_test_accuracy,diagnostic_success,retention_probe_r2"]
    for entry in summary["per_seed"]:
        for model_type, noise in CONDITIONS:
            name = condition_name(model_type, noise)
            item = entry[name]
            if item is None:
                lines.append(f"{entry['seed']},{name},missing,,,,,,,")
                continue
            values = [
                entry["seed"], name, item.get("source_session"), item.get("attempt"), item.get("status"),
                item.get("test_accuracy"), item.get("original_success"), item.get("diagnostic_test_accuracy"),
                item.get("diagnostic_success"), item.get("retention_probe_r2"),
            ]
            lines.append(",".join("" if value is None else str(value) for value in values))
    (COMPLETION / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (COMPLETION / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during the EXP-016 completion")
    print(f"EXP-016 completion finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

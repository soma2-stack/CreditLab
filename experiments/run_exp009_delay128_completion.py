"""Authorized EXP-009 delay-128 continuation.

This wrapper does not call the original training runner and does not
overwrite saved EXP-009 records.

Registration, before any new training result:
    python experiments/run_exp009_delay128_completion.py --write-registration

Continuation, only after that registration is committed:
    python experiments/run_exp009_delay128_completion.py
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
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import (  # noqa: E402
    dataset_bundle,
    dataset_fingerprints,
    file_sha256,
    index_fingerprint,
    load_hard_v2,
    minibatch_indices,
    parameter_fingerprint,
    tensor_fingerprint,
)
from run_exp005 import (  # noqa: E402
    audit_one,
    correct_count,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
)
from run_exp009 import (  # noqa: E402
    ImplementationError,
    build_four,
    load_config,
    train_regime,
    verify_regime_lists,
)
from run_exp009_audit_completion import (  # noqa: E402
    artifact_manifest,
    compact_audit,
    direct_pair_stats,
    load_additive,
    saved_correct_count,
    verify_equation,
)

OUT_DIR = ROOT / "results" / "EXP-009" / "delay128_completion"
SAVED_DIR = ROOT / "results" / "EXP-009"
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
REGIMES = ["full", "readout_only", "bias_plus_readout", "matrices_plus_readout"]
DELAY = 128
REUSED = ((101, "full"), (101, "readout_only"))
RESTART = (101, "bias_plus_readout")
TOTAL_SECONDS = 600
PER_RUN_SECONDS = 300
BLOCKED_NAMES = {
    "raw_metrics.jsonl",
    "runtime_session.json",
    "runtime_log.jsonl",
    "loss_curves.jsonl",
    "checkpoint_diagnostics.jsonl",
    "parameter_records.jsonl",
    "audit_metrics.jsonl",
    "summary.json",
}


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def original_delay128_records() -> list[dict]:
    path = SAVED_DIR / "raw_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if int(row["delay"]) == DELAY]


def attempt_for(seed: int, regime: str) -> str:
    if (seed, regime) == RESTART:
        return "authorized_restart_1"
    return "continuation_1"


def planned_jobs() -> list[dict]:
    jobs = []
    for seed in SEEDS:
        for regime in REGIMES:
            if (seed, regime) in REUSED:
                continue
            jobs.append({
                "delay": DELAY,
                "seed": seed,
                "regime": regime,
                "attempt_id": attempt_for(seed, regime),
                "prior_attempt_id": "original_session" if (seed, regime) == RESTART else None,
                "prior_outcome": "unknown_no_saved_result" if (seed, regime) == RESTART else None,
                "starts_from": "initialization",
            })
    return jobs


def reused_rows() -> list[dict]:
    saved = {(int(row["seed"]), row["regime"]): row for row in original_delay128_records()}
    rows = []
    for seed, regime in REUSED:
        match = saved.get((seed, regime))
        rows.append({
            "delay": DELAY,
            "seed": seed,
            "regime": regime,
            "source": "original_session",
            "attempt_id": "original_session",
            "retrained": False,
            "checkpoint": f"results/EXP-009/checkpoints/additive_{regime}_delay{DELAY}_seed{seed}.pt",
            "checkpoint_sha256_recorded": None if match is None else match.get("checkpoint_sha256"),
            "saved_status": None if match is None else match.get("status"),
            "saved_test_accuracy": None if match is None else match.get("test_accuracy"),
        })
    return rows


def preexisting_manifest() -> dict[str, str]:
    records = artifact_manifest()
    extra_root = ROOT / "results" / "EXP-009" / "audit_completion"
    if extra_root.exists():
        for path in sorted(item for item in extra_root.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return {
        key: value
        for key, value in records.items()
        if not key.startswith("results/EXP-009/delay128_completion/")
    }


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the continuation registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = planned_jobs()
    if len(jobs) != 38:
        raise RuntimeError(f"expected 38 jobs, found {len(jobs)}")
    registration = {
        "experiment_id": "EXP-009",
        "kind": "authorized_delay128_continuation",
        "original_session": "interrupted",
        "original_budget_compliance": "not established; the unfinished attempt has no finish time",
        "authorized_training_jobs": jobs,
        "reused_completed_conditions": reused_rows(),
        "restart": {
            "delay": DELAY,
            "seed": 101,
            "regime": "bias_plus_readout",
            "new_attempt_id": "authorized_restart_1",
            "prior_attempt_id": "original_session",
            "prior_outcome": "unknown_no_saved_result",
            "starts_from": "initialization",
        },
        "settings": "unchanged EXP-009 additive equation, regimes, seeds, data, and Adam schedule",
        "success_test_accuracy_at_least": 0.95,
        "verification": [
            "reused delay-128 checkpoints match recorded hashes and integer test counts",
            "new runs regenerate the original dataset, minibatches, and initialization",
            "trainable and frozen names match the preregistration",
            "the restarted bias run does not load another trained model",
        ],
        "budget_seconds": TOTAL_SECONDS,
        "per_condition_cap_seconds": PER_RUN_SECONDS,
        "budget_note": "This allowance is separate. It is not the historical EXP-009 cost. The original unfinished attempt remains an unknown additional cost.",
        "failure_policy": "Save the affected attempt, do not retry it, and stop for review if verification or a freeze check fails. A numerical failure or timeout is unsuccessful and is not repeated.",
        "interpretation_limits": [
            "Do not say the original session finished normally.",
            "Do not say bias learning explains EXP-008.",
            "Do not say bias adaptation is always sufficient.",
            "Do not say matrix learning is always necessary.",
            "Do not say clipping caused a miss.",
            "Do not claim a new architecture or a broader theory result.",
            "Keep the 0.95 test rule separate from pair metrics.",
        ],
    }
    destination.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")
    manifest = preexisting_manifest()
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered {len(jobs)} jobs and {len(reused_rows())} reused conditions; hashed {len(manifest)} files")


def original_checkpoint(regime: str, seed: int) -> Path:
    return SAVED_DIR / "checkpoints" / f"additive_{regime}_delay{DELAY}_seed{seed}.pt"


def new_checkpoint(regime: str, seed: int) -> Path:
    return OUT_DIR / "checkpoints" / f"additive_{regime}_delay{DELAY}_seed{seed}.pt"


def save_new_checkpoint(model, record: dict) -> str:
    destination = new_checkpoint(record["regime"], int(record["seed"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_suffix(".pt.partial")
    torch.save({
        "experiment_id": "EXP-009",
        "continuation": "delay128_completion",
        "attempt_id": record.get("attempt_id"),
        "regime": record["regime"],
        "delay": record["delay"],
        "seed": record["seed"],
        "code_revision": record.get("code_revision"),
        "init_fingerprint": record.get("init_fingerprint"),
        "trainable_parameter_names": record.get("trainable_parameter_names"),
        "frozen_parameter_names": record.get("frozen_parameter_names"),
        "state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(destination)
    return file_sha256(destination)


def verify_reused(cfg: dict) -> list[dict]:
    saved = {(int(row["seed"]), row["regime"]): row for row in original_delay128_records()}
    if len(saved) != 2 or set(saved) != set(REUSED):
        raise RuntimeError("original delay-128 records are not exactly the two reused conditions")
    reports = []
    for seed, regime in REUSED:
        row = saved[(seed, regime)]
        path = original_checkpoint(regime, seed)
        report = {"delay": DELAY, "seed": seed, "regime": regime, "status": "ok", "stop_reason": None}
        if row.get("status") != "ok" or not path.is_file():
            report["status"] = "verification_failure"
            report["stop_reason"] = "saved delay-128 condition is missing or not ok"
            reports.append(report)
            return reports
        recomputed = file_sha256(path)
        report["checkpoint_sha256"] = recomputed
        if recomputed != row.get("checkpoint_sha256"):
            report["status"] = "verification_failure"
            report["stop_reason"] = "checkpoint hash does not match the saved record"
            reports.append(report)
            return reports
        test_x, test_y = load_hard_v2(
            cfg["data"]["test_samples"], DELAY, seed, 2,
            noise_std=cfg["task"]["distractor_noise_std"],
            competitor_rate=cfg["task"]["competitor_rate"],
            competitor_flip_prob=cfg["task"]["competitor_flip_prob"],
            base_seed=cfg["data"]["base_seed"],
        )
        if tensor_fingerprint(test_x) != row["dataset_fingerprints"]["test_x"] or tensor_fingerprint(test_y) != row["dataset_fingerprints"]["test_y"]:
            report["status"] = "verification_failure"
            report["stop_reason"] = "regenerated test examples do not match the saved fingerprint"
            reports.append(report)
            return reports
        model, payload = load_additive(path)
        verify_equation(model)
        if payload.get("regime") != regime or int(payload.get("seed")) != seed or int(payload.get("delay")) != DELAY:
            report["status"] = "verification_failure"
            report["stop_reason"] = "checkpoint metadata does not match the condition"
            reports.append(report)
            return reports
        expected = saved_correct_count(row["test_accuracy"], cfg["data"]["test_samples"])
        actual = correct_count(model, test_x, test_y, cfg["training"]["eval_batch_size"])
        report["saved_correct_count"] = expected
        report["recomputed_correct_count"] = actual
        if actual != expected:
            report["status"] = "verification_failure"
            report["stop_reason"] = f"correct count {actual} does not match saved {expected}"
            reports.append(report)
            return reports
        reports.append(report)
    return reports


def prepare_seed(cfg: dict, seed: int) -> dict:
    bundle = dataset_bundle(cfg, DELAY, seed)
    data_fp = dataset_fingerprints(bundle)
    indices = minibatch_indices(
        seed, DELAY, bundle["train_x"].shape[0],
        cfg["training"]["batch_size"], cfg["training"]["updates_per_run"],
    )
    index_hash = index_fingerprint(indices)
    models = build_four(cfg, seed)
    verify_regime_lists(cfg, models)
    reference = parameter_fingerprint(models["full"])
    if any(parameter_fingerprint(model) != reference for model in models.values()):
        raise RuntimeError(f"initial parameters differ at seed {seed}")
    probe = bundle["train_x"][:4]
    with torch.no_grad():
        reference_logits, _ = models["full"](probe)
    tolerance = float(cfg["diagnostics"]["paired_forward_tolerance"])
    for regime, model in models.items():
        with torch.no_grad():
            logits, _ = model(probe)
        if float((logits - reference_logits).abs().max()) > tolerance:
            raise RuntimeError(f"initial outputs differ for seed {seed} regime {regime}")
    saved = [row for row in original_delay128_records() if int(row["seed"]) == seed]
    for row in saved:
        if row.get("init_fingerprint") != reference:
            raise RuntimeError(f"initialization does not match the original delay-128 record for seed {seed}")
        if row.get("minibatch_fingerprint") != index_hash:
            raise RuntimeError(f"minibatches do not match the original delay-128 record for seed {seed}")
        if row.get("dataset_fingerprints") != data_fp:
            raise RuntimeError(f"dataset does not match the original delay-128 record for seed {seed}")
    append_jsonl(OUT_DIR / "fingerprints.jsonl", {
        "delay": DELAY,
        "seed": seed,
        "init_fingerprint": reference,
        "minibatch_fingerprint": index_hash,
        "dataset_fingerprints": data_fp,
    })
    return {
        "bundle": bundle,
        "indices": indices,
        "models": models,
        "reference": reference,
        "data_fp": data_fp,
        "index_hash": index_hash,
    }


def job_failed_freeze(record: dict, details: dict | None) -> str | None:
    regime = record["regime"]
    if record.get("frozen_parameters_unchanged") is False:
        return "a frozen parameter changed"
    if regime == "readout_only" and record.get("hidden_states_unchanged") is False:
        return "readout-only hidden states changed"
    if regime == "matrices_plus_readout" and details and not details.get("both_preactivation_biases_unchanged"):
        return "a preactivation bias changed in the matrix regime"
    if regime == "bias_plus_readout" and details and not details.get("matrices_unchanged"):
        return "a weight matrix changed in the bias regime"
    return None


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    if not (OUT_DIR / "continuation_registration.yaml").exists():
        raise RuntimeError("continuation registration is missing")
    blockers = [name for name in BLOCKED_NAMES if (OUT_DIR / name).exists()]
    if blockers:
        raise RuntimeError(
            "delay-128 continuation outputs already exist "
            f"({', '.join(blockers)}). Automatic restart is disabled."
        )
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preexisting_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before the continuation")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nContinuation code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "kind": "authorized_delay128_continuation",
        "original_session": "interrupted",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": TOTAL_SECONDS,
        "original_unfinished_attempt_runtime": "unknown",
    }, indent=2, sort_keys=True), encoding="utf-8")
    jobs = planned_jobs()
    records = []
    unstarted = []
    stopped = None

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def expired() -> bool:
        return time.perf_counter() - started >= TOTAL_SECONDS

    reports = verify_reused(cfg)
    (OUT_DIR / "reused_checkpoint_verification.jsonl").write_text(
        "".join(json.dumps(report, sort_keys=True) + "\n" for report in reports),
        encoding="utf-8",
    )
    if any(report["status"] != "ok" for report in reports) or len(reports) != 2:
        raise RuntimeError(f"reused checkpoint verification failed: {reports}")
    prepared: dict[int, dict] = {}
    for job in jobs:
        if expired() or stopped:
            unstarted.append({**job, "reason": "implementation stop" if stopped else "total cap"})
            continue
        seed = int(job["seed"])
        regime = job["regime"]
        if seed not in prepared:
            try:
                prepared[seed] = prepare_seed(cfg, seed)
            except Exception as exc:
                stopped = {"seed": seed, "regime": regime, "reason": f"{type(exc).__name__}: {exc}"}
                unstarted.append({**job, "reason": "verification failure"})
                continue
        remaining = TOTAL_SECONDS - (time.perf_counter() - started)
        job_cfg = copy.deepcopy(cfg)
        job_cfg["budget"]["max_cpu_seconds_per_run"] = min(PER_RUN_SECONDS, remaining)
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {
            "event": "start",
            "attempt_id": job["attempt_id"],
            "delay": DELAY,
            "seed": seed,
            "regime": regime,
            "started_unix": time.time(),
            "session_elapsed_seconds": elapsed(),
            "applied_per_run_cap_seconds": job_cfg["budget"]["max_cpu_seconds_per_run"],
        })
        print(f"training {regime} delay {DELAY} seed {seed} attempt {job['attempt_id']}", flush=True)
        ready = prepared[seed]
        try:
            record, curve, checks, details = train_regime(
                job_cfg, ready["models"][regime], regime, ready["bundle"], ready["indices"],
                DELAY, seed, "delay128_continuation", ready["reference"], ready["data_fp"], ready["index_hash"],
            )
        except ImplementationError as exc:
            record = {
                "experiment_id": "EXP-009",
                "delay": DELAY,
                "seed": seed,
                "regime": regime,
                "status": "implementation_bug",
                "success": False,
                "stop_reason": f"ImplementationError: {exc}",
                "updates_completed": None,
                "runtime_seconds": None,
            }
            curve, checks, details = [], [], None
            stopped = {"seed": seed, "regime": regime, "reason": record["stop_reason"]}
        freeze_reason = job_failed_freeze(record, details)
        if freeze_reason:
            record["status"] = "implementation_bug"
            record["success"] = False
            record["stop_reason"] = freeze_reason
            stopped = {"seed": seed, "regime": regime, "reason": freeze_reason}
        record["source"] = "delay128_completion"
        record["attempt_id"] = job["attempt_id"]
        record["prior_attempt_id"] = job["prior_attempt_id"]
        record["prior_outcome"] = job["prior_outcome"]
        record["session_elapsed_seconds"] = elapsed()
        if record["status"] == "ok":
            record["checkpoint_sha256"] = save_new_checkpoint(ready["models"][regime], record)
            record["complete"] = True
        else:
            record["complete"] = False
        records.append(record)
        append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
        append_jsonl(OUT_DIR / "loss_curves.jsonl", {
            "delay": DELAY, "seed": seed, "regime": regime, "attempt_id": job["attempt_id"], "loss_curve": curve,
        })
        for item in checks:
            append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {
                "delay": DELAY, "seed": seed, "regime": regime, "attempt_id": job["attempt_id"], **item,
            })
        if details is not None:
            details["attempt_id"] = job["attempt_id"]
            append_jsonl(OUT_DIR / "parameter_records.jsonl", details)
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {
            "event": "finish",
            "attempt_id": job["attempt_id"],
            "delay": DELAY,
            "seed": seed,
            "regime": regime,
            "status": record["status"],
            "updates_completed": record.get("updates_completed"),
            "runtime_seconds": record.get("runtime_seconds"),
            "session_elapsed_seconds": elapsed(),
            "complete": record["complete"],
        })
        print(json.dumps({
            "seed": seed, "regime": regime, "status": record["status"],
            "success": record.get("success"), "test_accuracy": record.get("test_accuracy"),
        }, sort_keys=True), flush=True)
    audit_rows = []
    available = []
    for seed, regime in REUSED:
        available.append((seed, regime, original_checkpoint(regime, seed), "original_session"))
    for record in records:
        if record.get("complete") and record.get("checkpoint_sha256"):
            available.append((
                int(record["seed"]), record["regime"],
                new_checkpoint(record["regime"], int(record["seed"])), record["attempt_id"],
            ))
    for seed in SEEDS:
        if stopped and stopped.get("stage") == "audit":
            break
        present = [item for item in available if item[0] == seed]
        if not present:
            continue
        if expired():
            for item in present:
                unstarted.append({
                    "delay": DELAY, "seed": item[0], "regime": item[1], "reason": "audit cap",
                })
            continue
        base, _targets, _meta = load_hard_v2(
            cfg["data"]["audit_samples"], DELAY, seed, cfg["data"]["audit_split_index"],
            noise_std=cfg["task"]["distractor_noise_std"],
            competitor_rate=cfg["task"]["competitor_rate"],
            competitor_flip_prob=cfg["task"]["competitor_flip_prob"],
            base_seed=cfg["data"]["base_seed"],
            return_metadata=True,
        )
        positive, negative = make_counterfactual_pairs(base)
        if not pairs_differ_only_on_original_bit(positive, negative):
            raise RuntimeError(f"pair construction failed for seed {seed}")
        append_jsonl(OUT_DIR / "audit_data_fingerprints.jsonl", {
            "delay": DELAY,
            "seed": seed,
            "base_fingerprint": tensor_fingerprint(base),
        })
        for item_seed, regime, path, attempt_id in present:
            if expired():
                unstarted.append({"delay": DELAY, "seed": item_seed, "regime": regime, "reason": "audit cap"})
                continue
            try:
                model, _payload = load_additive(path)
                before_hash = parameter_fingerprint(model)
                seed_everything(0)
                audited = audit_one(model, "additive_tanh_rnn", positive, negative)
                direct = direct_pair_stats(model, positive, negative)
                if parameter_fingerprint(model) != before_hash:
                    raise RuntimeError("evaluation changed parameters")
            except Exception as exc:
                append_jsonl(OUT_DIR / "audit_metrics.jsonl", {
                    "delay": DELAY, "seed": item_seed, "regime": regime, "attempt_id": attempt_id,
                    "status": "failure", "stop_reason": f"{type(exc).__name__}: {exc}",
                })
                stopped = {"stage": "audit", "seed": item_seed, "regime": regime, "reason": str(exc)}
                break
            audit_record = {
                "delay": DELAY,
                "seed": item_seed,
                "regime": regime,
                "attempt_id": attempt_id,
                "source": "original_session" if attempt_id == "original_session" else "delay128_completion",
                "status": "ok",
                **compact_audit(audited, direct),
            }
            audit_rows.append(audit_record)
            append_jsonl(OUT_DIR / "audit_metrics.jsonl", audit_record)
    by_new = {(int(row["seed"]), row["regime"]): row for row in records}
    by_old = {(int(row["seed"]), row["regime"]): row for row in original_delay128_records()}
    table = []
    for seed in SEEDS:
        for regime in REGIMES:
            if (seed, regime) in REUSED:
                row = by_old[(seed, regime)]
                table.append({
                    "delay": DELAY,
                    "seed": seed,
                    "regime": regime,
                    "source": "original_session",
                    "attempt_id": "original_session",
                    "status": row["status"],
                    "success": row.get("success"),
                    "test_accuracy": row.get("test_accuracy"),
                    "retention_probe_r2": row.get("retention_probe_r2"),
                })
            elif (seed, regime) in by_new:
                row = by_new[(seed, regime)]
                table.append({
                    "delay": DELAY,
                    "seed": seed,
                    "regime": regime,
                    "source": "delay128_completion",
                    "attempt_id": row["attempt_id"],
                    "status": row["status"],
                    "success": row.get("success"),
                    "test_accuracy": row.get("test_accuracy"),
                    "retention_probe_r2": row.get("retention_probe_r2"),
                })
            else:
                table.append({
                    "delay": DELAY,
                    "seed": seed,
                    "regime": regime,
                    "source": "not_run",
                    "attempt_id": None,
                    "status": "unstarted",
                    "success": None,
                    "test_accuracy": None,
                    "retention_probe_r2": None,
                })
    paired = []
    for seed in SEEDS:
        bias = next(row for row in table if row["seed"] == seed and row["regime"] == "bias_plus_readout")
        matrix = next(row for row in table if row["seed"] == seed and row["regime"] == "matrices_plus_readout")
        paired.append({
            "seed": seed,
            "bias_success": bias.get("success"),
            "bias_accuracy": bias.get("test_accuracy"),
            "matrix_success": matrix.get("success"),
            "matrix_accuracy": matrix.get("test_accuracy"),
        })
    summary = {
        "kind": "authorized_delay128_continuation",
        "original_session": "interrupted",
        "authorized_jobs": 38,
        "executed_jobs": len(records),
        "completed_jobs": sum(1 for row in records if row.get("complete")),
        "failed_jobs": sum(1 for row in records if row.get("status") != "ok"),
        "reused_conditions": 2,
        "unstarted": unstarted,
        "implementation_stop": stopped,
        "audit_rows": len(audit_rows),
        "elapsed_seconds": elapsed(),
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": TOTAL_SECONDS,
        "original_unfinished_attempt_runtime": "unknown",
        "code_revision": revision,
        "table": table,
        "paired": paired,
    }
    (OUT_DIR / "delay128_table.json").write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    after = preexisting_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8",
    )
    if after != before:
        raise RuntimeError("preexisting artifacts changed during the continuation")
    if stopped:
        raise RuntimeError(f"continuation stopped: {stopped}")
    print(
        f"delay-128 continuation finished in {summary['elapsed_seconds']}s; "
        f"executed {len(records)}; unstarted {len(unstarted)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

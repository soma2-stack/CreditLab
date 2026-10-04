"""Diagnostic-only delay-64 bit-flip audit for the saved EXP-009 checkpoints.

This file does not train and does not call the EXP-009 training runner.

Registration, before any audit result:
    python experiments/run_exp009_audit_completion.py --write-registration

Audit, only after that registration is committed:
    python experiments/run_exp009_audit_completion.py
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

from creditlab.models import ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import file_sha256, load_hard_v2, tensor_fingerprint  # noqa: E402
from run_exp005 import (  # noqa: E402
    audit_one,
    correct_count,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
    parameter_fingerprint,
)
from run_exp009 import load_config  # noqa: E402

OUT_DIR = ROOT / "results" / "EXP-009" / "audit_completion"
SAVED_DIR = ROOT / "results" / "EXP-009"
CONFIG_PATH = ROOT / "configs" / "exp009_bias_ablation.yaml"
SEEDS = [101, 113, 127, 139, 151, 163, 179, 191, 211, 223]
REGIMES = ["full", "readout_only", "bias_plus_readout", "matrices_plus_readout"]
DELAY = 64
AUDIT_SECONDS = 180
ARTIFACT_DIRS = (
    "results/EXP-001",
    "results/EXP-001B",
    "results/EXP-002",
    "results/EXP-003",
    "results/EXP-004",
    "results/EXP-005",
    "results/EXP-006",
    "results/EXP-007",
    "results/EXP-008",
    "results/EXP-009",
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


def saved_correct_count(accuracy: float, n_examples: int = 512) -> int:
    value = float(accuracy) * n_examples
    nearest = int(round(value))
    if abs(value - nearest) > 1e-4:
        raise ValueError(f"saved accuracy {accuracy} is not an integer count out of {n_examples}")
    return nearest


def coordinate_equal_stats(final_positive: torch.Tensor, final_negative: torch.Tensor) -> dict:
    equal = (final_positive == final_negative).all(dim=-1)
    return {
        "final_states_coordinate_equal_fraction": float(equal.float().mean()),
        "final_states_coordinate_equal_count": int(equal.sum()),
        "n_pairs": int(equal.numel()),
    }


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            key = path.relative_to(ROOT).as_posix()
            if key.startswith("results/EXP-009/audit_completion/"):
                continue
            records[key] = file_sha256(path)
    for relative in ARTIFACT_CONFIGS:
        records[relative] = file_sha256(ROOT / relative)
    return records


def checkpoint_path(regime: str, seed: int) -> Path:
    return SAVED_DIR / "checkpoints" / f"additive_{regime}_delay{DELAY}_seed{seed}.pt"


def load_delay64_records() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (SAVED_DIR / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["delay"] == DELAY]


def condition_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((int(row["seed"]), row["regime"]), []).append(row)
    conditions = []
    for seed in SEEDS:
        for regime in REGIMES:
            matches = grouped.get((seed, regime), [])
            conditions.append({
                "delay": DELAY,
                "seed": seed,
                "regime": regime,
                "saved_record_count": len(matches),
                "status": None if not matches else matches[0].get("status"),
                "saved_test_accuracy": None if not matches else matches[0].get("test_accuracy"),
                "saved_success": None if not matches else matches[0].get("success"),
                "checkpoint": checkpoint_path(regime, seed).relative_to(ROOT).as_posix(),
                "checkpoint_sha256_recorded": None if not matches else matches[0].get("checkpoint_sha256"),
                "code_revision_recorded": None if not matches else matches[0].get("code_revision"),
                "init_fingerprint_recorded": None if not matches else matches[0].get("init_fingerprint"),
            })
    return conditions


def write_registration() -> None:
    if (OUT_DIR / "continuation_registration.yaml").exists():
        raise FileExistsError("refusing to overwrite the continuation registration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = condition_rows(load_delay64_records())
    registration = {
        "experiment_id": "EXP-009",
        "kind": "post_interruption_diagnostic_amendment",
        "scope": "diagnostic_only",
        "training_status": "interrupted",
        "delay_64_cohort": "complete",
        "delay_128_cohort": "incomplete",
        "authorized": "bit-flip audit of the 40 saved delay-64 checkpoints",
        "not_authorized": [
            "resume delay-128 training",
            "restart the unfinished bias-plus-readout run",
            "retrain any completed condition",
            "change the model, parameters, or hyperparameters",
            "start EXP-010",
        ],
        "conditions": conditions,
        "checkpoint_provenance": "results/EXP-009/checkpoints/additive_{regime}_delay64_seed{seed}.pt saved by the interrupted EXP-009 training process",
        "pair_generation": {
            "task": "HARD_V2",
            "samples": 512,
            "delay": 64,
            "base_seed": 2000,
            "split_index": 3,
            "competitor_rate": 0.4,
            "competitor_flip_prob": 0.5,
            "distractor_noise_std": 0.3,
            "members": "original bit +1 and -1; every other input identical",
            "shared_across_regimes": True,
            "initial_state": "zero, reset independently",
            "optimizer_steps": 0,
        },
        "metrics": [
            "accuracy over 1024 sequences",
            "both members correct",
            "exactly one member correct",
            "both members wrong",
            "prediction flip fraction",
            "signed final-logit difference distribution",
            "identical final-logit fraction",
            "exact final-state equality by coordinate comparison",
            "local original-bit gradients at both members",
            "computational-zero bit gradients with finite state or logit changes",
            "marker-channel sensitivity stored separately",
        ],
        "verification": [
            "exactly one saved delay-64 result per condition",
            "status ok",
            "checkpoint exists and its hash matches the saved record",
            "checkpoint metadata matches the condition",
            "loaded weights reproduce the saved integer test correct count on 512 examples",
            "pairs differ only at inputs[:, 0, 0]",
            "zero-bit copies match",
            "parameter hashes do not change during evaluation",
        ],
        "budget_seconds": AUDIT_SECONDS,
        "budget_note": "This allowance is separate. It does not reconstruct the unfinished training run and does not show that the original 15-minute cap was met. That unfinished run has no finish time.",
        "no_new_pass_threshold": True,
        "equation": "candidate_t = tanh(W_h h_{t-1} + W_x x_t + b_h + b_x); h_t = h_{t-1} + candidate_t",
    }
    (OUT_DIR / "continuation_registration.yaml").write_text(
        yaml.safe_dump(registration, sort_keys=False), encoding="utf-8",
    )
    manifest = artifact_manifest()
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"registered {len(conditions)} conditions; hashed {len(manifest)} preexisting files")


def load_additive(path: Path) -> tuple[ResidualTanhRNN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def verify_equation(model: ResidualTanhRNN) -> None:
    if model.residual_scale != 1.0:
        raise RuntimeError("residual scale is not 1")
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, generator=generator)
    hidden = torch.randn(2, 32, generator=generator)
    cell = model.recurrent
    pre = (
        x.matmul(cell.weight_ih_l0.T)
        + cell.bias_ih_l0
        + hidden.matmul(cell.weight_hh_l0.T)
        + cell.bias_hh_l0
    )
    expected = hidden + torch.tanh(pre)
    with torch.no_grad():
        got, candidate = model.candidate_step(x, hidden)
    if not torch.equal(got, expected) or not torch.equal(candidate, torch.tanh(pre)):
        raise RuntimeError("loaded model does not use the additive equation")


def verify_checkpoints(cfg: dict, records: list[dict]) -> list[dict]:
    used = yaml.safe_load((SAVED_DIR / "config_used.yaml").read_text(encoding="utf-8"))
    if used["model_equation"] != cfg["model_equation"]:
        raise RuntimeError("saved configuration equation does not match the training configuration")
    by_key = {(int(row["seed"]), row["regime"]): row for row in records}
    if len(records) != 40 or len(by_key) != 40:
        raise RuntimeError(f"expected 40 unique delay-64 records, found {len(records)}")
    reports = []
    for seed in SEEDS:
        bundle_x, bundle_y = load_hard_v2(
            cfg["data"]["test_samples"], DELAY, seed, 2,
            noise_std=cfg["task"]["distractor_noise_std"],
            competitor_rate=cfg["task"]["competitor_rate"],
            competitor_flip_prob=cfg["task"]["competitor_flip_prob"],
            base_seed=cfg["data"]["base_seed"],
        )
        for regime in REGIMES:
            row = by_key[(seed, regime)]
            path = checkpoint_path(regime, seed)
            report = {
                "delay": DELAY,
                "seed": seed,
                "regime": regime,
                "status": "ok",
                "stop_reason": None,
                "checkpoint": path.relative_to(ROOT).as_posix(),
            }
            if row.get("status") != "ok":
                report["status"] = "verification_failure"
                report["stop_reason"] = f"saved status is {row.get('status')}"
                reports.append(report)
                return reports
            if not path.is_file():
                report["status"] = "verification_failure"
                report["stop_reason"] = "checkpoint file is missing"
                reports.append(report)
                return reports
            recomputed = file_sha256(path)
            report["checkpoint_sha256"] = recomputed
            if recomputed != row.get("checkpoint_sha256"):
                report["status"] = "verification_failure"
                report["stop_reason"] = "checkpoint hash does not match the saved training record"
                reports.append(report)
                return reports
            if tensor_fingerprint(bundle_x) != row["dataset_fingerprints"]["test_x"] or tensor_fingerprint(bundle_y) != row["dataset_fingerprints"]["test_y"]:
                report["status"] = "verification_failure"
                report["stop_reason"] = "regenerated test examples do not match the saved fingerprint"
                reports.append(report)
                return reports
            model, payload = load_additive(path)
            verify_equation(model)
            metadata_ok = (
                payload.get("experiment_id") == "EXP-009"
                and payload.get("regime") == regime
                and int(payload.get("delay")) == DELAY
                and int(payload.get("seed")) == seed
                and payload.get("code_revision") == row.get("code_revision")
                and payload.get("init_fingerprint") == row.get("init_fingerprint")
            )
            report["metadata_match"] = metadata_ok
            if not metadata_ok:
                report["status"] = "verification_failure"
                report["stop_reason"] = "checkpoint metadata does not match the saved condition"
                reports.append(report)
                return reports
            expected = saved_correct_count(row["test_accuracy"], cfg["data"]["test_samples"])
            actual = correct_count(model, bundle_x, bundle_y, cfg["training"]["eval_batch_size"])
            report["saved_correct_count"] = expected
            report["recomputed_correct_count"] = actual
            if actual != expected:
                report["status"] = "verification_failure"
                report["stop_reason"] = f"correct count {actual} does not match saved {expected}"
                reports.append(report)
                return reports
            reports.append(report)
    return reports


def finite_logit_stats(logit_positive: torch.Tensor, logit_negative: torch.Tensor) -> dict:
    difference = logit_positive - logit_negative
    finite = torch.isfinite(logit_positive) & torch.isfinite(logit_negative) & (logit_positive != logit_negative)
    return {
        "finite_logit_difference_count": int(finite.sum()),
        "finite_logit_difference_fraction": float(finite.float().mean()),
        "signed_logit_difference_mean": float(difference.mean()),
    }


def direct_pair_stats(model, positive: torch.Tensor, negative: torch.Tensor) -> dict:
    with torch.no_grad():
        logits_positive, _ = model(positive)
        logits_negative, _ = model(negative)
        final_positive = model.recurrent_states(positive)[:, -1, :]
        final_negative = model.recurrent_states(negative)[:, -1, :]
    stats = coordinate_equal_stats(final_positive, final_negative)
    stats.update(finite_logit_stats(logits_positive[:, -1, 0], logits_negative[:, -1, 0]))
    return stats


def compact_audit(audited: dict, direct: dict) -> dict:
    return {
        "n_pairs": audited["n_pairs"],
        "n_sequences": audited["n_sequences"],
        "accuracy": audited["accuracy"],
        "both_correct_fraction": audited["both_correct_fraction"],
        "exactly_one_correct_fraction": audited["exactly_one_correct_fraction"],
        "both_wrong_fraction": audited["both_wrong_fraction"],
        "prediction_flip_fraction": audited["prediction_flip_fraction"],
        "logit_difference": audited["logit_difference"],
        "positive_logit_difference_fraction": audited["positive_logit_difference_fraction"],
        "identical_logits_fraction": audited["identical_logits_fraction"],
        "finite_logit_difference_count": direct["finite_logit_difference_count"],
        "finite_logit_difference_fraction": direct["finite_logit_difference_fraction"],
        "final_states_coordinate_equal_fraction": direct["final_states_coordinate_equal_fraction"],
        "final_states_coordinate_equal_count": direct["final_states_coordinate_equal_count"],
        "final_l2_equal_fraction_not_used_as_equality": audited["final_states_exactly_equal_fraction"],
        "bit_sensitivity_positive": audited["bit_sensitivity_positive"],
        "bit_sensitivity_negative": audited["bit_sensitivity_negative"],
        "marker_sensitivity_positive_mean_abs": audited["marker_sensitivity_positive_mean_abs"],
        "marker_sensitivity_negative_mean_abs": audited["marker_sensitivity_negative_mean_abs"],
        "pairs_both_bit_grads_exact_zero": audited["pairs_both_bit_grads_exact_zero"],
        "pairs_zero_local_grad_but_logit_changes": audited["pairs_zero_local_grad_but_logit_changes"],
        "pairs_zero_local_grad_but_state_changes": audited["pairs_zero_local_grad_but_state_changes"],
    }


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("environment does not match Python 3.11.9 / PyTorch 2.13.0+cpu")
    if not (OUT_DIR / "continuation_registration.yaml").exists():
        raise RuntimeError("continuation registration is missing")
    if (OUT_DIR / "audit_metrics.jsonl").exists():
        raise RuntimeError("audit metrics already exist; automatic repetition is disabled")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before the audit")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nAudit code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "kind": "diagnostic_only",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": AUDIT_SECONDS,
        "does_not_reconstruct_original_training_runtime": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    records = load_delay64_records()
    reports = verify_checkpoints(cfg, records)
    (OUT_DIR / "checkpoint_verification.jsonl").write_text(
        "".join(json.dumps(report, sort_keys=True) + "\n" for report in reports),
        encoding="utf-8",
    )
    failed = [report for report in reports if report["status"] != "ok"]
    if failed or len(reports) != 40:
        raise RuntimeError(f"checkpoint verification stopped: {failed[:1]}")
    saved = {(int(row["seed"]), row["regime"]): row for row in records}
    unstarted = []
    audited_rows = []

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def expired() -> bool:
        return time.perf_counter() - started >= AUDIT_SECONDS

    for seed in SEEDS:
        if expired():
            for regime in REGIMES:
                unstarted.append({"delay": DELAY, "seed": seed, "regime": regime, "reason": "diagnostic cap"})
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
        with (OUT_DIR / "audit_data_fingerprints.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "delay": DELAY,
                "seed": seed,
                "split_index": 3,
                "base_fingerprint": hashlib.sha256(base.contiguous().numpy().tobytes()).hexdigest(),
                "positive_fingerprint": tensor_fingerprint(positive),
                "negative_fingerprint": tensor_fingerprint(negative),
            }, sort_keys=True) + "\n")
        for regime in REGIMES:
            if expired():
                unstarted.append({"delay": DELAY, "seed": seed, "regime": regime, "reason": "diagnostic cap"})
                continue
            path = checkpoint_path(regime, seed)
            with (OUT_DIR / "runtime_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "start", "delay": DELAY, "seed": seed, "regime": regime,
                    "session_elapsed_seconds": elapsed(),
                }, sort_keys=True) + "\n")
            model, _payload = load_additive(path)
            before_hash = parameter_fingerprint(model)
            try:
                seed_everything(0)
                audited = audit_one(model, "additive_tanh_rnn", positive, negative)
                direct = direct_pair_stats(model, positive, negative)
            except Exception as exc:
                failure = {
                    "delay": DELAY, "seed": seed, "regime": regime,
                    "status": "failure", "stop_reason": f"{type(exc).__name__}: {exc}",
                    "session_elapsed_seconds": elapsed(),
                }
                with (OUT_DIR / "audit_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                raise RuntimeError(f"audit failed for delay {DELAY} seed {seed} regime {regime}") from exc
            after_hash = parameter_fingerprint(model)
            if after_hash != before_hash:
                raise RuntimeError(f"evaluation changed parameters for seed {seed} regime {regime}")
            training = saved[(seed, regime)]
            record = {
                "delay": DELAY,
                "seed": seed,
                "regime": regime,
                "status": "ok",
                "checkpoint_sha256": training["checkpoint_sha256"],
                "saved_training_test_accuracy": training["test_accuracy"],
                "saved_training_success": training["success"],
                "parameter_hash_unchanged": True,
                **compact_audit(audited, direct),
            }
            audited_rows.append(record)
            with (OUT_DIR / "audit_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            with (OUT_DIR / "runtime_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "finish", "delay": DELAY, "seed": seed, "regime": regime,
                    "status": "ok", "session_elapsed_seconds": elapsed(),
                }, sort_keys=True) + "\n")
            print(json.dumps({
                "seed": seed, "regime": regime,
                "accuracy": record["accuracy"],
                "both_correct_fraction": record["both_correct_fraction"],
                "coordinate_equal_fraction": record["final_states_coordinate_equal_fraction"],
            }, sort_keys=True), flush=True)
    summary = {
        "kind": "diagnostic_only",
        "training_status": "interrupted",
        "delay_64_cohort": "complete",
        "delay_128_cohort": "incomplete",
        "verified_checkpoints": len(reports),
        "audited_conditions": len(audited_rows),
        "failed_conditions": 0,
        "unstarted": unstarted,
        "elapsed_seconds": elapsed(),
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds": AUDIT_SECONDS,
        "original_training_runtime_known": False,
        "code_revision": revision,
        "rows": [
            {
                "delay": row["delay"],
                "seed": row["seed"],
                "regime": row["regime"],
                "saved_training_success": row["saved_training_success"],
                "saved_training_test_accuracy": row["saved_training_test_accuracy"],
                "audit_accuracy": row["accuracy"],
                "both_correct_fraction": row["both_correct_fraction"],
                "exactly_one_correct_fraction": row["exactly_one_correct_fraction"],
                "both_wrong_fraction": row["both_wrong_fraction"],
                "prediction_flip_fraction": row["prediction_flip_fraction"],
                "identical_logits_fraction": row["identical_logits_fraction"],
                "final_states_coordinate_equal_fraction": row["final_states_coordinate_equal_fraction"],
                "pairs_zero_local_grad_but_logit_changes": row["pairs_zero_local_grad_but_logit_changes"],
                "pairs_zero_local_grad_but_state_changes": row["pairs_zero_local_grad_but_state_changes"],
            }
            for row in audited_rows
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary["rows"][0].keys()) if summary["rows"] else ["delay", "seed", "regime"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["rows"])
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8",
    )
    if after != before:
        raise RuntimeError("preexisting artifacts changed during the audit")
    print(f"audit continuation finished in {summary['elapsed_seconds']}s; audited {len(audited_rows)}; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

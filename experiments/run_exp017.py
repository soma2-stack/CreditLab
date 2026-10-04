"""EXP-017: selective overwrite versus continued retention.

Registration, before any accuracy:
    python experiments/run_exp017.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp017.py
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

from creditlab.diagnostics import hidden_magnitude_stats  # noqa: E402
from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp001 import evaluate  # noqa: E402
from run_exp004 import (  # noqa: E402
    combined_seed,
    file_sha256,
    index_fingerprint,
    minibatch_indices,
    parameter_fingerprint,
    tensor_fingerprint,
)
from run_exp011 import apply_clipping  # noqa: E402
from run_exp012 import (  # noqa: E402
    apply_scale,
    final_states,
    original_logits,
    original_predictions,
    refit_predictions,
    sklearn_version,
)
from run_exp016 import (  # noqa: E402
    ImplementationError,
    artifact_manifest,
    atomic_torch_save,
    fit_diagnostic,
    git_revision,
    matched_models,
    verify_equations,
)

CONFIG_PATH = ROOT / "configs" / "exp017_selective_overwrite.yaml"
OUT_DIR = ROOT / "results" / "EXP-017"
SEEDS = [503, 509, 521, 523, 541, 547, 557, 563, 569, 571]
MODELS = ("vanilla_tanh_rnn", "additive_tanh_rnn")
TASKS = ("hold", "selective")
JOBS = tuple(
    {"seed": seed, "model_type": model_type, "task": task, "attempt": 1}
    for seed in SEEDS
    for task in TASKS
    for model_type in MODELS
)
LENGTH = 130
BASE_SEED = 7000
BLOCKED = {"raw_metrics.jsonl", "runtime_session.json", "summary.json"}
TOTAL_BUDGET = 900.0
PER_RUN_BUDGET = 300.0
AUDITS = ("retained_initial", "replacement_bit", "ignore_unmarked_later", "ignore_obsolete_initial")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def condition_name(model_type: str, task: str) -> str:
    family = "vanilla" if model_type.startswith("vanilla") else "additive"
    return f"{family}_{task}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_latent(num_samples: int, run_seed: int, split_index: int, base_seed: int = BASE_SEED) -> dict:
    """One balanced draw of the overwrite task. The candidate token is not a distractor."""
    if num_samples % 8 != 0:
        raise ImplementationError("split size is not divisible by eight")
    generator = torch.Generator(device="cpu").manual_seed(combined_seed(run_seed, 128, split_index, base_seed))
    cells = [(a, b, f) for a in (-1.0, 1.0) for b in (-1.0, 1.0) for f in (0.0, 1.0)]
    repeated = [cell for cell in cells for _ in range(num_samples // 8)]
    order = torch.randperm(num_samples, generator=generator)
    ordered = [repeated[int(index)] for index in order]
    bit_a = torch.tensor([item[0] for item in ordered], dtype=torch.float32)
    bit_b = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
    flag = torch.tensor([item[2] for item in ordered], dtype=torch.float32)
    timing = torch.randint(32, 97, (num_samples,), generator=generator)
    inputs = torch.zeros(num_samples, LENGTH, 3)
    inputs[:, 0, 0] = bit_a
    inputs[:, 0, 1] = 1.0
    inputs[:, 129, 2] = 1.0
    inputs[:, 1:129, 0] = torch.randint(0, 2, (num_samples, 128), generator=generator).float() * 2 - 1
    competitor_mask = torch.rand(num_samples, 128, generator=generator) < 0.4
    torch.rand(num_samples, 128, generator=generator)
    competitor_bits = torch.where(torch.rand(num_samples, 128, generator=generator) < 0.5, -1.0, 1.0)
    rows = torch.arange(num_samples)
    interior = timing - 1
    competitor_mask[rows, interior] = False
    inputs[:, 1:129, 0] = torch.where(competitor_mask, competitor_bits, inputs[:, 1:129, 0])
    inputs[:, 1:129, 1] = torch.where(competitor_mask, torch.ones_like(competitor_bits), inputs[:, 1:129, 1])
    unit_noise = torch.randn(num_samples, 128, 3, generator=generator)
    unit_noise[rows, interior] = 0
    inputs[:, 1:129, :] += 0.3 * unit_noise
    inputs[rows, timing, 0] = bit_b
    inputs[rows, timing, 1] = 0.0
    inputs[rows, timing, 2] = 0.0
    if int(timing.min()) < 32 or int(timing.max()) > 96:
        raise ImplementationError("candidate index left the registered range")
    if not torch.equal(inputs[rows, timing, 0], bit_b) or not torch.equal(inputs[:, 0, 0], bit_a):
        raise ImplementationError("event or candidate bit was overwritten by a distractor")
    if not torch.equal(inputs[:, 129, 2], torch.ones(num_samples)) or not torch.equal(inputs[:, 129, 0], torch.zeros(num_samples)):
        raise ImplementationError("query marker is not confined to the final step")
    non_candidate = torch.ones(num_samples, 128, dtype=torch.bool)
    non_candidate[rows, interior] = False
    observed = inputs[:, 1:129, 2][non_candidate]
    expected = (0.3 * unit_noise[:, :, 2])[non_candidate]
    if not torch.equal(observed, expected):
        raise ImplementationError("distractor noise is not the registered 0.3 scale")
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            for f in (0.0, 1.0):
                count = int(((bit_a == a) & (bit_b == b) & (flag == f)).sum())
                if count != num_samples // 8:
                    raise ImplementationError("A, B, and F are not exactly balanced")
    return {
        "A": bit_a, "B": bit_b, "F": flag, "timing": timing,
        "hold_x": inputs, "competitor_mask": competitor_mask, "competitor_bits": competitor_bits,
    }


def task_views(latent: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hold_x = latent["hold_x"]
    hold_y = latent["A"].clone()
    selective_x = hold_x.clone()
    rows = torch.arange(hold_x.shape[0])
    selective_x[rows, latent["timing"], 1] = latent["F"]
    selective_y = torch.where(latent["F"] == 1, latent["B"], latent["A"])
    difference = selective_x - hold_x
    expected = torch.zeros_like(difference)
    expected[rows, latent["timing"], 1] = latent["F"]
    if not torch.equal(difference, expected):
        raise ImplementationError("hold and selective inputs differ outside the candidate write marker")
    target_difference = selective_y - hold_y
    changed = (latent["F"] == 1) & (latent["A"] != latent["B"])
    expected_targets = torch.where(changed, latent["B"] - latent["A"], torch.zeros_like(latent["A"]))
    if not torch.equal(target_difference, expected_targets):
        raise ImplementationError("targets changed where replacement does not change the bit")
    if not torch.equal(selective_x[rows, latent["timing"], 0], latent["B"]):
        raise ImplementationError("the candidate bit is not B")
    return hold_x, hold_y, selective_x, selective_y


def competitor_agreement(latent: dict, targets: torch.Tensor) -> float:
    mask = latent["competitor_mask"]
    bits = latent["competitor_bits"]
    if int(mask.sum()) < 5000:
        return float("nan")
    expected = targets.unsqueeze(1).expand_as(bits)
    return float((bits[mask] == expected[mask]).float().mean())


def bundle_for(seed: int, task: str) -> dict:
    bundle = {}
    for num_samples, split_index, name in ((1536, 0, "train"), (512, 1, "val"), (512, 2, "test")):
        latent = make_latent(num_samples, seed, split_index)
        hold_x, hold_y, selective_x, selective_y = task_views(latent)
        if task == "hold":
            bundle[f"{name}_x"], bundle[f"{name}_y"] = hold_x, hold_y
        else:
            bundle[f"{name}_x"], bundle[f"{name}_y"] = selective_x, selective_y
        if name == "train":
            agreement = competitor_agreement(latent, selective_y)
            if agreement == agreement and not 0.45 <= agreement <= 0.55:
                raise ImplementationError(f"competitor bits agree with the target at rate {agreement}")
        if name == "test":
            bundle["A"], bundle["B"], bundle["F"], bundle["timing"] = latent["A"], latent["B"], latent["F"], latent["timing"]
    return bundle


def subgroups(bit_a: torch.Tensor, bit_b: torch.Tensor, flag: torch.Tensor) -> dict[str, torch.Tensor]:
    same = bit_a == bit_b
    replacement = flag == 1
    groups = {
        "replacement_equal": replacement & same,
        "replacement_conflict": replacement & ~same,
        "hold_equal": (~replacement) & same,
        "hold_conflict": (~replacement) & ~same,
    }
    if sum(int(mask.sum()) for mask in groups.values()) != int(bit_a.shape[0]):
        raise ImplementationError("subgroups do not cover the split")
    return groups


def terminal_predictions(model, inputs: torch.Tensor) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], 256):
            final = model(inputs[start:start + 256])[0][:, -1, 0]
            predictions = torch.sign(final)
            predictions = torch.where(predictions == 0, torch.ones_like(predictions), predictions)
            pieces.append(predictions)
    return torch.cat(pieces)


def subgroup_record(predictions: torch.Tensor, targets: torch.Tensor, bit_a, bit_b, flag) -> dict:
    groups = subgroups(bit_a, bit_b, flag)
    record = {}
    for name, mask in groups.items():
        size = int(mask.sum())
        correct = int((predictions[mask] == targets[mask]).sum())
        record[name] = {"size": size, "correct": correct, "accuracy": correct / size}
    return record


def sensitivities(model, inputs: torch.Tensor, timing: torch.Tensor) -> dict:
    probe = inputs.detach().clone()
    probe.requires_grad_(True)
    final = model(probe)[0][:, -1, 0]
    (gradient,) = torch.autograd.grad(final.sum(), probe)
    rows = torch.arange(inputs.shape[0])
    return {
        "bit_a_mean_abs": float(gradient[:, 0, 0].abs().mean()),
        "bit_b_mean_abs": float(gradient[rows, timing, 0].abs().mean()),
    }


def train_job(job: dict, seconds_left: float) -> None:
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    verify_equations()
    seed = int(job["seed"])
    model_type = job["model_type"]
    task = job["task"]
    name = condition_name(model_type, task)
    bundle = bundle_for(seed, task)
    models = matched_models(seed)
    model = models[(model_type, 0.3)]
    if parameter_fingerprint(models[("vanilla_tanh_rnn", 0.3)]) != parameter_fingerprint(models[("additive_tanh_rnn", 0.3)]):
        raise ImplementationError("initial parameters do not match")
    indices = minibatch_indices(seed, 128, 1536, 64, 400)
    started = time.perf_counter()
    deadline = started + min(PER_RUN_BUDGET, seconds_left)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-017", "model_type": model_type, "task": task, "condition": name,
        "seed": seed, "attempt": 1, "status": "ok", "stop_reason": None, "code_revision": git_revision(),
    }
    loss_curve, checkpoints, grad_norms = [], [], []
    clipped = 0
    try:
        for update_mark in (0,):
            loss, accuracy = evaluate(model, bundle["val_x"], bundle["val_y"], 256)
            checkpoints.append({"update": update_mark, "val_loss": loss, "val_accuracy": accuracy})
        for update, rows in enumerate(indices):
            if time.perf_counter() - started > min(PER_RUN_BUDGET, seconds_left) or time.perf_counter() > deadline:
                raise TimeoutError(f"CPU budget exhausted at update {update}")
            logits, _hidden = model(bundle["train_x"][rows])
            loss = bce(logits[:, -1, 0], (bundle["train_y"][rows] + 1) / 2)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite training value at update {update}")
            optimizer.zero_grad()
            loss.backward()
            clip_record = apply_clipping(list(model.parameters()), True)
            grad_norms.append(clip_record["raw_grad_norm_float64"])
            clipped += int(clip_record["clipping_applied"])
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in (50, 100, 200, 400):
                val_loss, val_accuracy = evaluate(model, bundle["val_x"], bundle["val_y"], 256)
                checkpoints.append({"update": update + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
        test_loss, test_accuracy = evaluate(model, bundle["test_x"], bundle["test_y"], 256)
        predictions = terminal_predictions(model, bundle["test_x"])
        correct = int((predictions == bundle["test_y"]).sum())
        if correct != round(test_accuracy * bundle["test_y"].numel()):
            raise ImplementationError("prediction count does not match the evaluation accuracy")
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        record.update(
            updates_completed=400,
            test_loss=test_loss,
            test_accuracy=correct / int(bundle["test_y"].numel()),
            test_correct=correct,
            test_count=int(bundle["test_y"].numel()),
            max_abs_hidden=magnitude["max_abs_hidden"],
            final_hidden_rms=magnitude["final_hidden_rms"],
            mean_raw_grad_norm=float(np.mean(grad_norms)),
            grad_clip_fraction=clipped / 400,
            runtime_seconds=round(time.perf_counter() - started, 3),
            **sensitivities(model, bundle["test_x"], bundle["timing"]),
        )
        if task == "selective":
            record["subgroups"] = subgroup_record(predictions, bundle["test_y"], bundle["A"], bundle["B"], bundle["F"])
            record["selective_success"] = bool(
                record["test_accuracy"] >= 0.95 and all(item["accuracy"] >= 0.95 for item in record["subgroups"].values())
            )
        else:
            record["hold_success"] = bool(record["test_accuracy"] >= 0.95)
        destination = OUT_DIR / "checkpoints" / f"{name}_seed{seed}.pt"
        record["checkpoint_sha256"] = atomic_torch_save(destination, {
            "experiment_id": "EXP-017", "model_type": model_type, "task": task, "seed": seed,
            "attempt": 1, "state_dict": model.state_dict(),
        })
    except (FloatingPointError, TimeoutError) as exc:
        record.update(
            status="budget_stop" if isinstance(exc, TimeoutError) else "numerical_failure",
            stop_reason=f"{type(exc).__name__}: {exc}",
            updates_completed=len(loss_curve),
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
    append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
    append_jsonl(OUT_DIR / "loss_curves.jsonl", {"seed": seed, "condition": name, "loss_curve": loss_curve})
    for item in checkpoints:
        append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {"seed": seed, "condition": name, **item})
    print(json.dumps({"seed": seed, "condition": name, "status": record["status"], "test_accuracy": record.get("test_accuracy")}, sort_keys=True), flush=True)


def differs_only_at(positive, negative, rows, positions, channel) -> bool:
    difference = positive - negative
    changed = difference[rows, positions, channel]
    if not torch.equal(changed, torch.full_like(changed, 2.0)):
        return False
    cleared = difference.clone()
    cleared[rows, positions, channel] = 0
    return bool(torch.count_nonzero(cleared) == 0)


def make_audit_pair(latent: dict, audit: str) -> dict:
    base = latent["hold_x"]
    positive, negative = base.clone(), base.clone()
    rows = torch.arange(base.shape[0])
    timing = latent["timing"]
    bit_a, bit_b = latent["A"], latent["B"]
    if audit == "retained_initial":
        positive[rows, 0, 0] = 1
        negative[rows, 0, 0] = -1
        positions, channel = torch.zeros(base.shape[0], dtype=torch.long), 0
        selective_pos, selective_neg = torch.ones_like(bit_a), -torch.ones_like(bit_a)
        hold_pos, hold_neg = selective_pos.clone(), selective_neg.clone()
    elif audit == "replacement_bit":
        positive[rows, timing, 1] = 1
        negative[rows, timing, 1] = 1
        positive[rows, timing, 0] = 1
        negative[rows, timing, 0] = -1
        positions, channel = timing, 0
        selective_pos, selective_neg = torch.ones_like(bit_b), -torch.ones_like(bit_b)
        hold_pos, hold_neg = bit_a.clone(), bit_a.clone()
    elif audit == "ignore_unmarked_later":
        positive[rows, timing, 0] = 1
        negative[rows, timing, 0] = -1
        positions, channel = timing, 0
        selective_pos, selective_neg = bit_a.clone(), bit_a.clone()
        hold_pos, hold_neg = bit_a.clone(), bit_a.clone()
    elif audit == "ignore_obsolete_initial":
        positive[rows, timing, 1] = 1
        negative[rows, timing, 1] = 1
        positive[rows, 0, 0] = 1
        negative[rows, 0, 0] = -1
        positions, channel = torch.zeros(base.shape[0], dtype=torch.long), 0
        selective_pos, selective_neg = bit_b.clone(), bit_b.clone()
        hold_pos, hold_neg = torch.ones_like(bit_a), -torch.ones_like(bit_a)
    else:
        raise ImplementationError(f"unknown audit {audit}")
    if not differs_only_at(positive, negative, rows, positions, channel):
        raise ImplementationError(f"audit {audit} changes more than the intended bit")
    return {
        "positive": positive, "negative": negative,
        "selective_targets": (selective_pos, selective_neg),
        "hold_targets": (hold_pos, hold_neg),
    }


def score_audit(model, fitted, pair: dict, task: str) -> dict:
    positive, negative = pair["positive"], pair["negative"]
    target_pos, target_neg = pair[f"{task}_targets"]
    pos_state, neg_state = final_states(model, positive), final_states(model, negative)
    original_pos = original_logits(model, pos_state)
    original_neg = original_logits(model, neg_state)
    diagnostic_pos = fitted["decision"](pos_state)
    diagnostic_neg = fitted["decision"](neg_state)
    return {
        "original": readout_score(original_pos, original_neg, original_predictions(original_pos), original_predictions(original_neg), target_pos, target_neg),
        "diagnostic": readout_score(diagnostic_pos, diagnostic_neg, refit_predictions(diagnostic_pos), refit_predictions(diagnostic_neg), target_pos.numpy(), target_neg.numpy()),
        "final_states_coordinate_equal_fraction": float(np.mean(np.all(pos_state == neg_state, axis=1))),
    }


def readout_score(logits_pos, logits_neg, pred_pos, pred_neg, target_pos, target_neg) -> dict:
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_neg = np.asarray(target_neg, dtype=np.float64)
    pred_pos = np.asarray(pred_pos, dtype=np.float64)
    pred_neg = np.asarray(pred_neg, dtype=np.float64)
    difference = np.asarray(logits_pos, dtype=np.float64) - np.asarray(logits_neg, dtype=np.float64)
    finite = np.isfinite(difference)
    both = (pred_pos == target_pos) & (pred_neg == target_neg)
    return {
        "both_correct_fraction": float(np.mean(both)),
        "prediction_flip_fraction": float(np.mean(pred_pos != pred_neg)),
        "prediction_agreement_fraction": float(np.mean(pred_pos == pred_neg)),
        "finite_signed_difference_fraction": float(np.mean(finite)),
        "mean_signed_logit_difference": None if not finite.any() else float(difference[finite].mean()),
    }


def save_classifier(fitted: dict, seed: int, model_type: str, task: str) -> str:
    destination = OUT_DIR / "classifiers" / f"{condition_name(model_type, task)}_seed{seed}.joblib"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".joblib.partial")
    joblib.dump({"classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"]}, temporary)
    temporary.replace(destination)
    return file_sha256(destination)


def diagnose_seed(seed: int) -> None:
    from run_exp016 import load_saved_model
    latent = make_latent(256, seed, 3)
    task_views(latent)
    pairs = {name: make_audit_pair(latent, name) for name in AUDITS}
    for task in TASKS:
        bundle = bundle_for(seed, task)
        for model_type in MODELS:
            name = condition_name(model_type, task)
            path = OUT_DIR / "checkpoints" / f"{name}_seed{seed}.pt"
            model = load_saved_model(path, model_type)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            fitted = fit_diagnostic(model, bundle["train_x"], bundle["train_y"], bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
            after = [parameter.detach().clone() for parameter in model.parameters()]
            if any(not torch.equal(left, right) for left, right in zip(before, after)):
                raise ImplementationError("diagnostic fitting changed parameters")
            digest = save_classifier(fitted, seed, model_type, task)
            states = final_states(model, bundle["test_x"])
            diagnostic_predictions = torch.from_numpy(refit_predictions(fitted["decision"](states))).float()
            record = {
                "seed": seed, "condition": name, "task": task, "model_type": model_type,
                "fit_rows": fitted["fit_rows"], "iterations": fitted["iterations"], "converged": fitted["converged"],
                "diagnostic_test_accuracy": fitted["diagnostic_test_accuracy"],
                "diagnostic_test_loss": fitted["diagnostic_test_loss"],
                "classifier_sha256": digest,
            }
            if task == "selective":
                record["subgroups"] = subgroup_record(diagnostic_predictions, bundle["test_y"], bundle["A"], bundle["B"], bundle["F"])
                record["diagnostic_selective_success"] = bool(
                    fitted["diagnostic_test_accuracy"] >= 0.95 and all(item["accuracy"] >= 0.95 for item in record["subgroups"].values())
                )
            else:
                record["diagnostic_hold_success"] = bool(fitted["diagnostic_test_accuracy"] >= 0.95)
            record["audits"] = {audit: score_audit(model, fitted, pairs[audit], task) for audit in AUDITS}
            append_jsonl(OUT_DIR / "diagnostic_metrics.jsonl", record)


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    folder = ROOT / "results" / "EXP-016"
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def find_seed_conflicts() -> list[str]:
    hits = []
    needles = [f'"seed": {seed}' for seed in SEEDS]
    for folder in (ROOT / "results", ROOT / "configs", ROOT / "experiments"):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "EXP-017" in path.parts:
                continue
            if path.suffix.lower() not in {".jsonl", ".json", ".yaml", ".yml", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits.extend(f"{path.relative_to(ROOT).as_posix()}: {needle}" for needle in needles if needle in text)
    return hits


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-017 registration")
    conflicts = find_seed_conflicts()
    if conflicts:
        raise ImplementationError("seed conflict: " + "; ".join(conflicts[:8]))
    if len(JOBS) != 40:
        raise ImplementationError("the job list is not 40 conditions")
    verify_equations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-017",
        "kind": "task_capability",
        "not_a_hard_v2_replication": True,
        "seeds": SEEDS,
        "base_seed": BASE_SEED,
        "jobs": list(JOBS),
        "seed_conflicts": [],
        "success_hold": 0.95,
        "success_selective": "overall and four subgroups each >= 0.95",
        "budget_seconds_per_run": PER_RUN_BUDGET,
        "budget_seconds_total": TOTAL_BUDGET,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print("registered EXP-017; seeds unused")


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
    if any((OUT_DIR / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-017 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-017")
    started = time.perf_counter()
    deadline = started + TOTAL_BUDGET
    revision = git_revision()
    (OUT_DIR / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-017", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds_total": TOTAL_BUDGET,
    }, indent=2, sort_keys=True), encoding="utf-8")
    verify_equations()
    unstarted = []
    stopped = False
    for index, job in enumerate(JOBS):
        name = condition_name(job["model_type"], job["task"])
        remaining = deadline - time.perf_counter()
        if remaining <= 1 or stopped:
            unstarted.append({"seed": job["seed"], "condition": name, "reason": "not started"})
            continue
        log_dir = OUT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}_seed{job['seed']}.stdout.txt"
        stderr_path = log_dir / f"{name}_seed{job['seed']}.stderr.txt"
        job_started = time.perf_counter()
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "start", "seed": job["seed"], "condition": name, "session_elapsed_seconds": round(job_started - started, 3)})
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--train-one", json.dumps({"job": job, "seconds_left": remaining})],
                    cwd=str(ROOT), stdout=stdout, stderr=stderr, timeout=remaining,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {
            "event": "finish", "seed": job["seed"], "condition": name, "exit_code": exit_code,
            "runtime_seconds": round(time.perf_counter() - job_started, 3),
            "session_elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        print(json.dumps({"seed": job["seed"], "condition": name, "exit_code": exit_code}, sort_keys=True), flush=True)
        if exit_code != 0:
            stopped = True
            unstarted.extend(
                {"seed": later["seed"], "condition": condition_name(later["model_type"], later["task"]), "reason": f"stopped after {name} seed {job['seed']}"}
                for later in JOBS[index + 1:]
            )
    rows = load_jsonl(OUT_DIR / "raw_metrics.jsonl")
    diagnosed = set()
    for seed in SEEDS:
        names = [condition_name(model_type, task) for task in TASKS for model_type in MODELS]
        ready = all(any(row.get("seed") == seed and row.get("condition") == name and row.get("status") == "ok" for row in rows) for name in names)
        if not ready or time.perf_counter() >= deadline:
            for name in names:
                if (seed, name) not in diagnosed:
                    unstarted.append({"seed": seed, "condition": name, "stage": "diagnostic", "reason": "not started"})
            continue
        if seed in {row.get("seed") for row in load_jsonl(OUT_DIR / "diagnostic_metrics.jsonl")}:
            continue
        diagnose_seed(seed)
        diagnosed.add(seed)
        print(json.dumps({"seed": seed, "stage": "diagnostic"}, sort_keys=True), flush=True)
    summary = {
        "experiment_id": "EXP-017",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "unstarted": unstarted,
        "records": rows,
        "diagnostics": load_jsonl(OUT_DIR / "diagnostic_metrics.jsonl"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_lines = ["seed,condition,status,test_accuracy,test_correct,hold_success,selective_success"]
    for row in rows:
        csv_lines.append(",".join(str(row.get(field, "")) for field in (
            "seed", "condition", "status", "test_accuracy", "test_correct", "hold_success", "selective_success",
        )))
    (OUT_DIR / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-017")
    print(f"EXP-017 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

"""EXP-016: additive versus vanilla at two distractor-noise levels.

Registration, before any accuracy:
    python experiments/run_exp016.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp016.py
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

from creditlab.diagnostics import hidden_magnitude_stats  # noqa: E402
from creditlab.experiment_log import append_jsonl  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate, probe_fit_score  # noqa: E402
from run_exp004 import (  # noqa: E402
    build_model,
    combined_seed,
    file_sha256,
    index_fingerprint,
    load_hard_v2,
    minibatch_indices,
    parameter_fingerprint,
    tensor_fingerprint,
)
from run_exp005 import (  # noqa: E402
    bit_and_marker_sensitivity,
    make_counterfactual_pairs,
    pairs_differ_only_on_original_bit,
)
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

CONFIG_PATH = ROOT / "configs" / "exp016_noise_robustness.yaml"
OUT_DIR = ROOT / "results" / "EXP-016"
SEEDS = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457]
CONDITIONS = (
    ("vanilla_tanh_rnn", 0.3),
    ("additive_tanh_rnn", 0.3),
    ("vanilla_tanh_rnn", 1.0),
    ("additive_tanh_rnn", 1.0),
)
BLOCKED = {"raw_metrics.jsonl", "runtime_session.json", "summary.json"}
ARTIFACT_DIRS = tuple(
    f"results/EXP-{name}" for name in (
        "001", "001B", "002", "003", "004", "005", "006", "007", "008", "009",
        "010", "011", "012", "013", "014", "015",
    )
)
PRESERVED = ("SCIENTIFIC_CHECKPOINT.md", "DOCUMENTATION_AUDIT.md")
MODEL_CFG = {"model": {"input_size": 3, "hidden_size": 32, "output_size": 1}}


class ImplementationError(RuntimeError):
    """A correctness check failed. This is not a training outcome."""


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return out.stdout.strip()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def noise_tag(noise: float) -> str:
    if noise == 0.3:
        return "0p3"
    if noise == 1.0:
        return "1p0"
    raise ImplementationError(f"unregistered noise level {noise}")


def condition_name(model_type: str, noise: float) -> str:
    family = "vanilla" if model_type.startswith("vanilla") else "additive"
    return f"{family}_noise{noise_tag(noise)}"


def verify_equations() -> None:
    seed_everything(0)
    vanilla = build_model("vanilla_tanh_rnn", MODEL_CFG)
    seed_everything(0)
    additive = build_model("additive_tanh_rnn", MODEL_CFG)
    if parameter_fingerprint(vanilla) != parameter_fingerprint(additive):
        raise ImplementationError("vanilla and additive initialization differ")
    count = sum(parameter.numel() for parameter in vanilla.parameters())
    if count != 1217 or count != sum(parameter.numel() for parameter in additive.parameters()):
        raise ImplementationError(f"parameter count is {count}, not 1217")
    sequence = torch.tensor([[[0.2, 1.0, 0.0], [-0.4, 0.0, 0.0]]], dtype=torch.float32)
    hidden = torch.zeros(1, 32)
    vanilla_states = vanilla.recurrent_states(sequence)
    additive_state, candidate = additive.candidate_step(sequence[:, 0, :], hidden)
    if not torch.equal(vanilla_states[:, 0, :], candidate):
        raise ImplementationError("vanilla state is not the tanh candidate")
    if not torch.equal(additive_state, hidden + candidate):
        raise ImplementationError("additive state is not the previous state plus the candidate")
    cell = vanilla.recurrent
    previous = vanilla_states[:, 0, :]
    preactivation = (
        sequence[:, 1, :].matmul(cell.weight_ih_l0.T)
        + cell.bias_ih_l0
        + previous.matmul(cell.weight_hh_l0.T)
        + cell.bias_hh_l0
    )
    if not torch.allclose(vanilla_states[:, 1, :], torch.tanh(preactivation), atol=1e-6, rtol=0):
        raise ImplementationError("vanilla recurrence did not stay a plain tanh update")


def shared_noise_draw(num_samples: int, delay: int, seed: int, split_index: int, base_seed: int):
    """One latent draw, scaled only on the generator's distractor span."""
    common = dict(competitor_rate=0.4, competitor_flip_prob=0.5, base_seed=base_seed)
    clean_x, clean_y, clean_meta, unit_noise = latent_draw(num_samples, delay, seed, split_index, base_seed)
    low_x = clean_x.clone()
    high_x = clean_x.clone()
    low_x[:, 1:-1, :] += 0.3 * unit_noise
    high_x[:, 1:-1, :] += 1.0 * unit_noise
    direct_low_x, direct_low_y, direct_low_meta = load_hard_v2(
        num_samples, delay, seed, split_index, noise_std=0.3, return_metadata=True, **common,
    )
    direct_high_x, direct_high_y, direct_high_meta = load_hard_v2(
        num_samples, delay, seed, split_index, noise_std=1.0, return_metadata=True, **common,
    )
    if not torch.equal(low_x, direct_low_x) or not torch.equal(high_x, direct_high_x):
        raise ImplementationError("scaled noise does not match the direct generator")
    if not torch.equal(clean_y, direct_low_y) or not torch.equal(clean_y, direct_high_y):
        raise ImplementationError("noise scaling changed the labels")
    if not torch.equal(clean_meta["competitor_mask"], direct_low_meta["competitor_mask"]):
        raise ImplementationError("noise scaling changed competitor locations")
    if not torch.equal(clean_meta["competitor_mask"], direct_high_meta["competitor_mask"]):
        raise ImplementationError("noise scaling changed competitor locations")
    if not torch.equal(clean_meta["competitor_bits"], direct_high_meta["competitor_bits"]):
        raise ImplementationError("noise scaling changed competitor bits")
    if not torch.equal(clean_x[:, 0, :], high_x[:, 0, :]) or not torch.equal(clean_x[:, -1, :], high_x[:, -1, :]):
        raise ImplementationError("event or query changed when distractor noise was added")
    if not torch.equal(clean_x[:, -1, 2], torch.ones(num_samples)):
        raise ImplementationError("query marker is not at the final step")
    agreement = competitor_agreement(clean_y, clean_meta)
    competitor_count = int(clean_meta["competitor_mask"].sum())
    if agreement in (0.0, 1.0):
        raise ImplementationError("competitor bits are a copy of the target")
    if competitor_count >= 5000 and not 0.45 <= agreement <= 0.55:
        raise ImplementationError(f"competitor bits agree with the target at rate {agreement}")
    return {
        0.3: {"inputs": low_x, "targets": clean_y, "metadata": clean_meta},
        1.0: {"inputs": high_x, "targets": clean_y, "metadata": clean_meta},
        "unit_noise": unit_noise,
    }


def latent_draw(num_samples: int, delay: int, seed: int, split_index: int, base_seed: int):
    """Replay the HARD-v2 generator through its standard-normal distractor draw."""
    length = delay + 2
    generator = torch.Generator(device="cpu").manual_seed(combined_seed(seed, delay, split_index, base_seed))
    bits = torch.where(torch.rand(num_samples, generator=generator) < 0.5, -1.0, 1.0)
    inputs = torch.zeros(num_samples, length, 3)
    inputs[:, 0, 0] = bits
    inputs[:, 1:, 0] = torch.randint(0, 2, (num_samples, length - 1), generator=generator).float() * 2 - 1
    inputs[:, 0, 1] = 1.0
    inputs[:, -1, 2] = 1.0
    comp_mask = torch.rand(num_samples, length - 2, generator=generator) < 0.4
    torch.rand(num_samples, length - 2, generator=generator)
    comp_bits = torch.where(torch.rand(num_samples, length - 2, generator=generator) < 0.5, -1.0, 1.0)
    inputs[:, 1:-1, 0] = torch.where(comp_mask, comp_bits, inputs[:, 1:-1, 0])
    inputs[:, 1:-1, 1] = torch.where(comp_mask, torch.ones_like(comp_bits), inputs[:, 1:-1, 1])
    unit_noise = torch.randn(num_samples, length - 2, 3, generator=generator)
    return inputs, bits, {"competitor_mask": comp_mask, "competitor_bits": comp_bits}, unit_noise


def competitor_agreement(targets: torch.Tensor, metadata: dict) -> float:
    mask = metadata["competitor_mask"]
    bits = metadata["competitor_bits"]
    if int(mask.sum()) == 0:
        raise ImplementationError("a draw contained no competitors")
    expected = targets.unsqueeze(1).expand_as(bits)
    return float((bits[mask] == expected[mask]).float().mean())


def build_bundle(seed: int, base_seed: int) -> dict[float, dict]:
    sizes = ((1536, 0, "train"), (512, 1, "val"), (512, 2, "test"))
    bundles = {0.3: {}, 1.0: {}}
    for num_samples, split_index, name in sizes:
        draw = shared_noise_draw(num_samples, 128, seed, split_index, base_seed)
        for noise in (0.3, 1.0):
            bundles[noise][f"{name}_x"] = draw[noise]["inputs"]
            bundles[noise][f"{name}_y"] = draw[noise]["targets"]
            if name == "test":
                bundles[noise]["test_meta"] = draw[noise]["metadata"]
        if not torch.equal(bundles[0.3][f"{name}_y"], bundles[1.0][f"{name}_y"]):
            raise ImplementationError("the two noise levels do not share labels")
    return bundles


def dataset_fingerprints(bundle: dict) -> dict[str, str]:
    return {
        "train_x": tensor_fingerprint(bundle["train_x"]),
        "train_y": tensor_fingerprint(bundle["train_y"]),
        "val_x": tensor_fingerprint(bundle["val_x"]),
        "val_y": tensor_fingerprint(bundle["val_y"]),
        "test_x": tensor_fingerprint(bundle["test_x"]),
        "test_y": tensor_fingerprint(bundle["test_y"]),
        "competitor_mask": tensor_fingerprint(bundle["test_meta"]["competitor_mask"]),
        "competitor_bits": tensor_fingerprint(bundle["test_meta"]["competitor_bits"]),
    }


def matched_models(seed: int) -> dict[tuple[str, float], torch.nn.Module]:
    seed_everything(seed)
    vanilla = build_model("vanilla_tanh_rnn", MODEL_CFG)
    seed_everything(seed)
    additive = build_model("additive_tanh_rnn", MODEL_CFG)
    if parameter_fingerprint(vanilla) != parameter_fingerprint(additive):
        raise ImplementationError(f"seed {seed} initial parameters do not match")
    models = {}
    for model_type, source in (("vanilla_tanh_rnn", vanilla), ("additive_tanh_rnn", additive)):
        for noise in (0.3, 1.0):
            models[(model_type, noise)] = copy.deepcopy(source)
    fingerprints = {key: parameter_fingerprint(model) for key, model in models.items()}
    if len(set(fingerprints.values())) != 1:
        raise ImplementationError("noise copies do not share initialization")
    return models


def assert_full_history(model, inputs: torch.Tensor) -> None:
    probe = inputs[:2].detach().clone()
    probe.requires_grad_(True)
    logits, _hidden = model(probe)
    logits[:, -1, 0].sum().backward()
    if probe.grad is None or not torch.isfinite(probe.grad[:, 0, 0]).all():
        raise ImplementationError("full-history training graph did not reach the original bit")
    model.zero_grad(set_to_none=True)


def validation_checkpoint(model, val_x, val_y, update: int, eval_batch: int) -> dict:
    loss, accuracy = evaluate(model, val_x, val_y, eval_batch)
    return {"update": update, "val_loss": loss, "val_accuracy": accuracy}


def train_one(model, model_type, noise, bundle, indices, seed, per_run_cap, started_cap):
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-016",
        "model_type": model_type,
        "noise": noise,
        "condition": condition_name(model_type, noise),
        "delay": 128,
        "seed": seed,
        "status": "ok",
        "stop_reason": None,
        "code_revision": git_revision(),
    }
    loss_curve, checkpoints = [], []
    grad_norms, clipped, exceeded = [], 0, 0
    try:
        checkpoints.append(validation_checkpoint(model, bundle["val_x"], bundle["val_y"], 0, 256))
        for update, idx in enumerate(indices):
            if time.perf_counter() - started > per_run_cap or time.perf_counter() > started_cap:
                raise TimeoutError(f"CPU budget exhausted at update {update}")
            xb, yb = bundle["train_x"][idx], bundle["train_y"][idx]
            logits, _hidden = model(xb)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logit at update {update}")
            loss = bce(logits[:, -1, 0], (yb + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            clip_record = apply_clipping(list(model.parameters()), True)
            grad_norms.append(clip_record["raw_grad_norm_float64"])
            if clip_record["clipping_applied"]:
                clipped += 1
            if clip_record["would_exceed_5"]:
                exceeded += 1
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError(f"non-finite parameter after update {update}")
            loss_curve.append(float(loss.detach()))
            if (update + 1) in (50, 100, 200, 400):
                checkpoints.append(validation_checkpoint(model, bundle["val_x"], bundle["val_y"], update + 1, 256))
        train_loss, train_acc = evaluate(model, bundle["train_x"], bundle["train_y"], 256)
        val_loss, val_acc = evaluate(model, bundle["val_x"], bundle["val_y"], 256)
        test_loss, test_acc = evaluate(model, bundle["test_x"], bundle["test_y"], 256)
        retention = probe_fit_score(model, bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        bit, marker = bit_and_marker_sensitivity(model, bundle["test_x"])
        record.update(
            updates_completed=400,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
            original_success=bool(test_acc >= 0.95),
            retention_probe_r2=retention,
            max_abs_hidden=magnitude["max_abs_hidden"],
            final_hidden_rms=magnitude["final_hidden_rms"],
            forward_bit_sensitivity_mean_abs=float(bit.abs().mean()),
            forward_marker_sensitivity_mean_abs=float(marker.abs().mean()),
            mean_raw_grad_norm=float(np.mean(grad_norms)),
            max_raw_grad_norm=float(np.max(grad_norms)),
            grad_clip_fraction=clipped / 400,
            grad_exceeded_5_fraction=exceeded / 400,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
    except (FloatingPointError, TimeoutError) as exc:
        status = "budget_stop" if isinstance(exc, TimeoutError) else "numerical_failure"
        record.update(
            status=status,
            stop_reason=f"{type(exc).__name__}: {exc}",
            updates_completed=len(loss_curve),
            original_success=False,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )
    return record, loss_curve, checkpoints


def atomic_torch_save(destination: Path, payload: dict) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return file_sha256(destination)


def save_model(model, seed: int, model_type: str, noise: float) -> str:
    destination = OUT_DIR / "checkpoints" / f"{condition_name(model_type, noise)}_delay128_seed{seed}.pt"
    return atomic_torch_save(destination, {
        "experiment_id": "EXP-016",
        "model_type": model_type,
        "noise": noise,
        "delay": 128,
        "seed": seed,
        "state_dict": model.state_dict(),
    })


def load_saved_model(path: Path, model_type: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(model_type, MODEL_CFG)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def fit_diagnostic(model, train_x, train_y, val_x, val_y, test_x, test_y):
    before = recurrent_hash(model)
    train_state = final_states(model, train_x)
    val_state = final_states(model, val_x)
    test_state = final_states(model, test_x)
    if train_state.shape[0] != int(train_y.shape[0]) or train_state.shape[0] == int(test_y.shape[0]):
        raise ImplementationError("diagnostic fit is not on the training split")
    if recurrent_hash(model) != before:
        raise ImplementationError("extracting states changed recurrent parameters")
    mean, scale = fit_scale(train_state)
    classifier = make_classifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(apply_scale(train_state, mean, scale), binary_labels(train_y))
    if recurrent_hash(model) != before:
        raise ImplementationError("diagnostic fitting changed recurrent parameters")
    iterations = int(np.max(classifier.n_iter_))
    converged = iterations < 2000 and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    decision = lambda states: classifier.decision_function(apply_scale(states, mean, scale))
    targets = test_y.detach().cpu().numpy()
    test_decision = decision(test_state)
    with torch.no_grad():
        direct_logits = model(test_x)[0][:, -1, 0].detach().double().numpy()
    reconstructed = original_logits(model, test_state)
    if not np.array_equal(direct_logits, reconstructed):
        raise ImplementationError("original readout on extracted states does not match the forward pass")
    return {
        "classifier": classifier,
        "mean": mean,
        "scale": scale,
        "decision": decision,
        "iterations": iterations,
        "converged": converged,
        "warnings": [str(item.message) for item in caught],
        "fit_rows": int(train_state.shape[0]),
        "diagnostic_test_accuracy": float(np.mean(refit_predictions(test_decision) == targets)),
        "diagnostic_test_loss": bce_from_decision(test_decision, binary_labels(test_y)),
        "diagnostic_train_accuracy": float(np.mean(refit_predictions(decision(train_state)) == train_y.numpy())),
        "diagnostic_val_accuracy": float(np.mean(refit_predictions(decision(val_state)) == val_y.numpy())),
        "diagnostic_success": bool(float(np.mean(refit_predictions(test_decision) == targets)) >= 0.95),
        "recurrent_hash": before,
    }


def save_classifier(fitted: dict, seed: int, model_type: str, noise: float) -> str:
    destination = OUT_DIR / "classifiers" / f"{condition_name(model_type, noise)}_delay128_seed{seed}.joblib"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".joblib.partial")
    joblib.dump({
        "classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"],
    }, temporary)
    temporary.replace(destination)
    loaded = joblib.load(destination)
    replay = loaded["classifier"].decision_function(apply_scale(fitted["mean"].reshape(1, -1) * 0 + fitted["mean"], loaded["mean"], loaded["scale"]))
    original = fitted["classifier"].decision_function(apply_scale(fitted["mean"].reshape(1, -1), fitted["mean"], fitted["scale"]))
    if not np.array_equal(replay, original):
        raise ImplementationError("reloaded classifier did not preserve its decision")
    return file_sha256(destination)


def audit_pair(model, fitted, positive, negative) -> dict:
    if not pairs_differ_only_on_original_bit(positive, negative):
        raise ImplementationError("audit pairs change more than the original bit")
    zero_positive, zero_negative = positive.clone(), negative.clone()
    zero_positive[:, 0, 0] = 0
    zero_negative[:, 0, 0] = 0
    if not torch.equal(zero_positive, zero_negative):
        raise ImplementationError("zero-bit inputs are not identical")
    pos_state = final_states(model, positive)
    neg_state = final_states(model, negative)
    zero_state_a = final_states(model, zero_positive)
    zero_state_b = final_states(model, zero_negative)
    if not np.array_equal(zero_state_a, zero_state_b):
        raise ImplementationError("zero-bit states do not match")
    zero_logits_a = original_logits(model, zero_state_a)
    zero_logits_b = original_logits(model, zero_state_b)
    if not np.array_equal(zero_logits_a, zero_logits_b):
        raise ImplementationError("zero-bit outputs do not match")
    original = readout_audit(
        original_logits(model, pos_state), original_logits(model, neg_state), original_predictions,
    )
    diagnostic = readout_audit(fitted["decision"](pos_state), fitted["decision"](neg_state), refit_predictions)
    equality = float(np.mean(np.all(pos_state == neg_state, axis=1)))
    original["final_states_coordinate_equal_fraction"] = equality
    diagnostic["final_states_coordinate_equal_fraction"] = equality
    return {"original": original, "diagnostic": diagnostic}


def readout_audit(positive_scores, negative_scores, decision_rule) -> dict:
    metrics = pair_metrics(positive_scores, negative_scores, decision_rule)
    difference = positive_scores.astype(np.float64) - negative_scores.astype(np.float64)
    finite = np.isfinite(difference)
    metrics["finite_signed_difference_fraction"] = float(np.mean(finite))
    metrics["mean_signed_logit_difference"] = None if not finite.any() else float(difference[finite].mean())
    return metrics


def artifact_manifest() -> dict[str, str]:
    records = {}
    for relative in ARTIFACT_DIRS:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    for name in PRESERVED:
        records[name] = file_sha256(ROOT / name)
    return records


def find_seed_conflicts() -> list[str]:
    hits = []
    needles = [f'"seed": {seed}' for seed in SEEDS]
    roots = [ROOT / "results", ROOT / "configs", ROOT / "experiments"]
    for folder in roots:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "EXP-016" in path.parts:
                continue
            if path.suffix.lower() not in {".jsonl", ".json", ".yaml", ".yml", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    hits.append(f"{path.relative_to(ROOT).as_posix()}: {needle}")
    return hits


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-016 registration")
    conflicts = find_seed_conflicts()
    if conflicts:
        raise ImplementationError("seed conflict: " + "; ".join(conflicts[:8]))
    verify_equations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-016",
        "seeds": cfg["seeds"],
        "base_seed": cfg["data"]["base_seed"],
        "delay": 128,
        "conditions": cfg["conditions"],
        "noise_pairing": "shared latent draw scaled by 0.3 or 1.0 on distractor steps only",
        "success_test_accuracy_at_least": 0.95,
        "budget_seconds_per_run": 300,
        "budget_seconds_total": 900,
        "seed_conflicts": [],
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(artifact_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print("registered EXP-016; seeds unused")


def summarize(records: list[dict], unstarted: list[dict], elapsed: float, stopped) -> dict:
    def rows(model_type, noise):
        return [row for row in records if row["model_type"] == model_type and row["noise"] == noise and row["status"] == "ok"]

    def count(model_type, noise, field):
        selected = rows(model_type, noise)
        return int(sum(bool(row.get(field)) for row in selected)), len(selected)

    comparisons = []
    for seed in SEEDS:
        item = {"seed": seed}
        for model_type, noise in CONDITIONS:
            match = next((row for row in records if row["seed"] == seed and row["model_type"] == model_type and row["noise"] == noise), None)
            prefix = condition_name(model_type, noise)
            item[f"{prefix}_status"] = None if match is None else match["status"]
            item[f"{prefix}_test_accuracy"] = None if match is None else match.get("test_accuracy")
            item[f"{prefix}_original_success"] = None if match is None else match.get("original_success")
            item[f"{prefix}_diagnostic_accuracy"] = None if match is None else match.get("diagnostic_test_accuracy")
            item[f"{prefix}_diagnostic_success"] = None if match is None else match.get("diagnostic_success")
        comparisons.append(item)
    return {
        "experiment_id": "EXP-016",
        "elapsed_seconds": elapsed,
        "stopped": stopped,
        "unstarted": unstarted,
        "original_success_counts": {
            condition_name(model_type, noise): {
                "successes": count(model_type, noise, "original_success")[0],
                "completed": count(model_type, noise, "original_success")[1],
            }
            for model_type, noise in CONDITIONS
        },
        "diagnostic_success_counts": {
            condition_name(model_type, noise): {
                "successes": count(model_type, noise, "diagnostic_success")[0],
                "completed": count(model_type, noise, "diagnostic_success")[1],
            }
            for model_type, noise in CONDITIONS
        },
        "per_seed": comparisons,
        "numerical_failures": [row for row in records if row["status"] == "numerical_failure"],
        "budget_stops": [row for row in records if row["status"] == "budget_stop"],
    }


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    cfg = load_config()
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if [name for name in BLOCKED if (OUT_DIR / name).exists()]:
        raise RuntimeError("EXP-016 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    verify_equations()
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if artifact_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-016")
    started = time.perf_counter()
    deadline = started + 900
    revision = git_revision()
    (OUT_DIR / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\n"
        f"scikit-learn: {sklearn_version()}\nTraining code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-016",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "budget_seconds_per_run": 300,
        "budget_seconds_total": 900,
    }, indent=2, sort_keys=True), encoding="utf-8")
    records, unstarted, stopped = [], [], None

    def elapsed():
        return round(time.perf_counter() - started, 3)

    for seed in cfg["seeds"]:
        if stopped or time.perf_counter() >= deadline:
            for model_type, noise in CONDITIONS:
                unstarted.append({"seed": seed, "condition": condition_name(model_type, noise), "reason": "not started"})
            continue
        bundles = build_bundle(seed, int(cfg["data"]["base_seed"]))
        indices = minibatch_indices(seed, 128, 1536, 64, 400)
        index_hash = index_fingerprint(indices)
        models = matched_models(seed)
        for model in models.values():
            assert_full_history(model, bundles[0.3]["train_x"])
        append_jsonl(OUT_DIR / "fingerprints.jsonl", {
            "seed": seed,
            "delay": 128,
            "init_fingerprint": parameter_fingerprint(models[("vanilla_tanh_rnn", 0.3)]),
            "minibatch_fingerprint": index_hash,
            "noise_0p3": dataset_fingerprints(bundles[0.3]),
            "noise_1p0": dataset_fingerprints(bundles[1.0]),
            "shared_train_labels": torch.equal(bundles[0.3]["train_y"], bundles[1.0]["train_y"]),
            "shared_test_labels": torch.equal(bundles[0.3]["test_y"], bundles[1.0]["test_y"]),
        })
        trained = {}
        for model_type, noise in CONDITIONS:
            name = condition_name(model_type, noise)
            if stopped or time.perf_counter() >= deadline:
                unstarted.append({"seed": seed, "condition": name, "reason": "not started"})
                continue
            append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                "event": "start", "seed": seed, "condition": name,
                "session_elapsed_seconds": elapsed(),
            })
            print(f"training {name} seed {seed}", flush=True)
            remaining = min(300.0, deadline - time.perf_counter())
            record, curve, checks = train_one(
                models[(model_type, noise)], model_type, noise, bundles[noise], indices, seed, remaining, deadline,
            )
            if record["status"] == "ok":
                path = OUT_DIR / "checkpoints" / f"{name}_delay128_seed{seed}.pt"
                record["checkpoint_sha256"] = save_model(models[(model_type, noise)], seed, model_type, noise)
                reloaded = load_saved_model(path, model_type)
                with torch.no_grad():
                    current = models[(model_type, noise)](bundles[noise]["test_x"][:4])[0]
                    replay = reloaded(bundles[noise]["test_x"][:4])[0]
                if not torch.equal(current, replay):
                    raise ImplementationError(f"reloaded checkpoint changed outputs for {name} seed {seed}")
                if recurrent_hash(models[(model_type, noise)]) != recurrent_hash(reloaded):
                    raise ImplementationError("reloaded recurrent weights differ")
                trained[(model_type, noise)] = record
            records.append(record)
            append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
            append_jsonl(OUT_DIR / "loss_curves.jsonl", {"seed": seed, "condition": name, "loss_curve": curve})
            for item in checks:
                append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {"seed": seed, "condition": name, **item})
            append_jsonl(OUT_DIR / "runtime_log.jsonl", {
                "event": "finish", "seed": seed, "condition": name, "status": record["status"],
                "runtime_seconds": record.get("runtime_seconds"), "session_elapsed_seconds": elapsed(),
            })
            print(json.dumps({
                "seed": seed, "condition": name, "status": record["status"],
                "test_accuracy": record.get("test_accuracy"),
            }, sort_keys=True), flush=True)
        audit_draws = {
            noise: shared_noise_draw(512, 128, seed, 3, int(cfg["data"]["base_seed"]))
            for noise in (0.3, 1.0)
        }
        for model_type, noise in CONDITIONS:
            name = condition_name(model_type, noise)
            if (model_type, noise) not in trained:
                continue
            if time.perf_counter() >= deadline:
                unstarted.append({"seed": seed, "condition": name, "stage": "diagnostic", "reason": "not started"})
                continue
            model = models[(model_type, noise)]
            bundle = bundles[noise]
            fitted = fit_diagnostic(
                model, bundle["train_x"], bundle["train_y"], bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"],
            )
            classifier_sha = save_classifier(fitted, seed, model_type, noise)
            positive, negative = make_counterfactual_pairs(audit_draws[noise][noise]["inputs"])
            audit = audit_pair(model, fitted, positive, negative)
            diagnostic_record = {
                "seed": seed, "condition": name, "model_type": model_type, "noise": noise,
                "fit_rows": fitted["fit_rows"], "iterations": fitted["iterations"], "converged": fitted["converged"],
                "diagnostic_test_accuracy": fitted["diagnostic_test_accuracy"],
                "diagnostic_test_loss": fitted["diagnostic_test_loss"],
                "diagnostic_success": fitted["diagnostic_success"],
                "classifier_sha256": classifier_sha,
                "audit": audit,
            }
            for row in records:
                if row["seed"] == seed and row["model_type"] == model_type and row["noise"] == noise:
                    row["diagnostic_test_accuracy"] = fitted["diagnostic_test_accuracy"]
                    row["diagnostic_test_loss"] = fitted["diagnostic_test_loss"]
                    row["diagnostic_success"] = fitted["diagnostic_success"]
            append_jsonl(OUT_DIR / "diagnostic_metrics.jsonl", diagnostic_record)
            print(json.dumps({
                "seed": seed, "condition": name, "diagnostic_accuracy": fitted["diagnostic_test_accuracy"],
            }, sort_keys=True), flush=True)
    summary = summarize(records, unstarted, elapsed(), stopped)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "seed", "condition", "model_type", "noise", "status", "test_accuracy", "test_loss",
            "original_success", "diagnostic_test_accuracy", "diagnostic_success", "retention_probe_r2",
            "max_abs_hidden", "final_hidden_rms", "grad_clip_fraction", "runtime_seconds", "stop_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    after = artifact_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-016")
    print(f"EXP-016 finished in {elapsed()}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

"""EXP-018: locate the selective-overwrite failure in saved models.

Registration, before any new diagnostic score:
    python experiments/run_exp018.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp018.py
"""

from __future__ import annotations

import copy
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
from creditlab.experiment_log import append_jsonl  # noqa: E402
from run_exp004 import file_sha256, parameter_fingerprint  # noqa: E402
from run_exp012 import apply_scale, final_states, refit_predictions, sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, git_revision  # noqa: E402
from run_exp017 import (  # noqa: E402
    SEEDS,
    condition_name,
    make_latent,
    task_views,
    terminal_predictions,
)

CONFIG_PATH = ROOT / "configs" / "exp018_failure_location.yaml"
OUT_DIR = ROOT / "results" / "EXP-018"
EXP17 = ROOT / "results" / "EXP-017"
MODELS = ("vanilla_tanh_rnn", "additive_tanh_rnn")
TASKS = ("hold", "selective")
PAIRS = ("marked_new_bit", "unmarked_new_bit", "marker")
CHECKPOINTS = ("before_update", "after_update", "plus_1", "plus_8", "plus_16", "before_query", "after_query")
BLOCKED = {"summary.json", "metrics.jsonl", "runtime_session.json"}
BUDGET = 300.0


class ImplementationError(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_parts(model, x_t, h_prev):
    cell = model.recurrent
    pre = (
        x_t.matmul(cell.weight_ih_l0.T)
        + cell.bias_ih_l0
        + h_prev.matmul(cell.weight_hh_l0.T)
        + cell.bias_hh_l0
    )
    candidate = torch.tanh(pre)
    if isinstance(model, ResidualTanhRNN):
        state = h_prev + model.residual_scale * candidate
    else:
        state = candidate
    return state, pre, candidate


def ordinary_states(model, inputs):
    with torch.no_grad():
        return model.recurrent_states(inputs)


def checkpoint_index(name: str, timing: torch.Tensor) -> torch.Tensor:
    if name == "before_update":
        return timing - 1
    if name == "after_update":
        return timing
    if name == "plus_1":
        return timing + 1
    if name == "plus_8":
        return timing + 8
    if name == "plus_16":
        return timing + 16
    if name == "before_query":
        return torch.full_like(timing, 128)
    if name == "after_query":
        return torch.full_like(timing, 129)
    raise ImplementationError(name)


def gather(states, index):
    return states[torch.arange(states.shape[0]), index]


def pair_metrics(left, right):
    difference = left.double() - right.double()
    equal = torch.all(left == right, dim=-1)
    l2 = torch.linalg.vector_norm(difference, dim=-1)
    max_abs = difference.abs().amax(dim=-1)
    mean_abs = difference.abs().mean(dim=-1)
    return equal, l2, max_abs, mean_abs


def distribution(values: torch.Tensor) -> dict:
    quantiles = torch.quantile(values.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return {
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(values.float().mean()),
        "fraction_exact_zero": float((values == 0).float().mean()),
    }


def build_pairs(latent: dict) -> dict:
    base, timing, bit_a = latent["hold_x"], latent["timing"], latent["A"]
    rows = torch.arange(base.shape[0])

    def place(bit, marker):
        copied = base.clone()
        copied[rows, timing, 0] = bit
        copied[rows, timing, 1] = marker
        copied[rows, timing, 2] = 0
        return copied

    pairs = {
        "marked_new_bit": (place(1.0, 1.0), place(-1.0, 1.0)),
        "unmarked_new_bit": (place(1.0, 0.0), place(-1.0, 0.0)),
        "marker": (place(-bit_a, 0.0), place(-bit_a, 1.0)),
    }
    for name, (left, right) in pairs.items():
        difference = left - right
        channel = 0 if name != "marker" else 1
        changed = difference[rows, timing, channel]
        expected = torch.full_like(changed, 2.0 if channel == 0 else -1.0)
        if not torch.equal(changed, expected):
            raise ImplementationError(f"{name} does not change by the intended amount")
        cleared = difference.clone()
        cleared[rows, timing, channel] = 0
        if int(torch.count_nonzero(cleared)) != 0:
            raise ImplementationError(f"{name} changes more than the intended coordinate")
    return pairs


def original_predictions(model, states):
    logits = model.readout(states)[:, 0]
    predictions = torch.sign(logits)
    return torch.where(predictions == 0, torch.ones_like(predictions), predictions), logits


def diagnostic_predictions(classifier, mean, scale, states):
    values = states.detach().double().cpu().numpy()
    decision = classifier.decision_function(apply_scale(values, mean, scale))
    return torch.from_numpy(refit_predictions(decision)).float(), decision


def verify_derivative() -> None:
    torch.manual_seed(0)
    model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    x = torch.randn(1, 3, requires_grad=True)
    hidden = torch.randn(1, 32)
    _state, _pre, candidate = step_parts(model, x, hidden)
    gradients = []
    for coordinate in range(32):
        (gradient,) = torch.autograd.grad(candidate[0, coordinate], x, retain_graph=True)
        gradients.append(gradient[0, 0])
    expected = (1 - candidate.detach()[0] ** 2) * model.recurrent.weight_ih_l0[:, 0]
    if not torch.allclose(torch.stack(gradients), expected, atol=1e-6, rtol=0):
        raise ImplementationError("candidate derivative does not match autograd")
    omitted = x.matmul(model.recurrent.weight_ih_l0.T) + hidden.matmul(model.recurrent.weight_hh_l0.T)
    full = omitted + model.recurrent.bias_ih_l0 + model.recurrent.bias_hh_l0
    if torch.equal(omitted, full):
        raise ImplementationError("preactivation did not include the biases")


def verify_saved() -> list[dict]:
    metrics = { (row["seed"], row["condition"]): row for row in load_jsonl(EXP17 / "raw_metrics.jsonl") }
    diagnostics = { (row["seed"], row["condition"]): row for row in load_jsonl(EXP17 / "diagnostic_metrics.jsonl") }
    checks = []
    for seed in SEEDS:
        bundles = {task: task_views(make_latent(512, seed, 2)) for task in TASKS}
        for task in TASKS:
            _hold_x, hold_y, selective_x, selective_y = bundles[task]
            inputs, targets = (selective_x, selective_y) if task == "selective" else (_hold_x, hold_y)
            for model_type in MODELS:
                name = condition_name(model_type, task)
                row = metrics[(seed, name)]
                diagnostic = diagnostics[(seed, name)]
                if task == "selective":
                    for key in ("replacement_equal", "replacement_conflict", "hold_equal", "hold_conflict"):
                        if diagnostic["subgroups"][key]["size"] != 128 or row["subgroups"][key]["size"] != 128:
                            raise ImplementationError(f"subgroup {key} is not the saved size for {name} {seed}")
                for audit in ("retained_initial", "replacement_bit", "ignore_unmarked_later", "ignore_obsolete_initial"):
                    if audit not in diagnostic["audits"]:
                        raise ImplementationError(f"missing audit {audit}")
                checkpoint = EXP17 / "checkpoints" / f"{name}_seed{seed}.pt"
                classifier_path = EXP17 / "classifiers" / f"{name}_seed{seed}.joblib"
                if file_sha256(checkpoint) != row["checkpoint_sha256"]:
                    raise ImplementationError(f"checkpoint hash mismatch for {name} seed {seed}")
                if file_sha256(classifier_path) != diagnostic["classifier_sha256"]:
                    raise ImplementationError(f"classifier hash mismatch for {name} seed {seed}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0) if model_type.startswith("additive") else __import__("creditlab.models", fromlist=["TinySequenceRNN"]).TinySequenceRNN(3, 32, 1)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                predictions = terminal_predictions(model, inputs)
                correct = int((predictions == targets).sum())
                if correct != int(row["test_correct"]):
                    raise ImplementationError(f"original count {correct} != saved {row['test_correct']} for {name} {seed}")
                bundle = joblib.load(classifier_path)
                states = final_states(model, inputs)
                diagnostic_pred = torch.from_numpy(refit_predictions(bundle["classifier"].decision_function(apply_scale(states, bundle["mean"], bundle["scale"])))).float()
                diagnostic_correct = int((diagnostic_pred == targets).sum())
                expected = round(float(diagnostic["diagnostic_test_accuracy"]) * 512)
                if diagnostic_correct != expected:
                    raise ImplementationError(f"diagnostic count {diagnostic_correct} != saved {expected} for {name} {seed}")
                checks.append({"seed": seed, "condition": name, "original_correct": correct, "diagnostic_correct": diagnostic_correct})
    return checks


def score_model(model, classifier_bundle, latent, pairs) -> dict:
    before = parameter_fingerprint(model)
    timing = latent["timing"]
    bit_a = latent["A"]
    result = {"precision": str(next(model.parameters()).dtype).replace("torch.", "")}
    for pair_name, (left, right) in pairs.items():
        left_states = ordinary_states(model, left)
        right_states = ordinary_states(model, right)
        rows = torch.arange(left.shape[0])
        before_gap = (gather(left_states, timing - 1) - gather(right_states, timing - 1)).abs().max()
        if float(before_gap) != 0.0:
            raise ImplementationError("pair states differ before the changed input")
        with torch.no_grad():
            formula_left, pre_left, cand_left = step_parts(model, left[rows, timing], gather(left_states, timing - 1))
            formula_right, pre_right, cand_right = step_parts(model, right[rows, timing], gather(right_states, timing - 1))
        formula_gap = max(
            float((formula_left - gather(left_states, timing)).abs().max()),
            float((formula_right - gather(right_states, timing)).abs().max()),
        )
        if isinstance(model, ResidualTanhRNN) and formula_gap != 0.0:
            raise ImplementationError("additive formula state does not match the ordinary rollout")
        if formula_gap > 1e-5:
            raise ImplementationError("formula state disagrees with the ordinary rollout")
        saturation = ((1 - cand_left ** 2) < 1e-6).float().mean(dim=-1)
        exact = ((cand_left == 1) | (cand_left == -1)).float().mean(dim=-1)
        candidate_gap = cand_left.double() - cand_right.double()
        checkpoints = {}
        per_example = {}
        for name in CHECKPOINTS:
            index = checkpoint_index(name, timing)
            equal, l2, max_abs, mean_abs = pair_metrics(gather(left_states, index), gather(right_states, index))
            checkpoints[name] = {
                "equal_fraction": float(equal.float().mean()),
                "l2": distribution(l2),
                "max_abs": distribution(max_abs),
                "mean_abs": distribution(mean_abs),
            }
            per_example[name] = {
                "equal": equal.numpy().astype(np.uint8),
                "l2": l2.numpy(),
            }
        after_equal = torch.from_numpy(per_example["after_update"]["equal"].astype(bool))
        median = float(saturation.median())
        high = saturation >= median
        low = ~high
        high_fraction = None if int(high.sum()) == 0 else float(after_equal[high].float().mean())
        low_fraction = None if int(low.sum()) == 0 else float(after_equal[low].float().mean())
        final_left, logits_left = original_predictions(model, left_states[:, -1])
        final_right, logits_right = original_predictions(model, right_states[:, -1])
        diag_left, _ = diagnostic_predictions(classifier_bundle["classifier"], classifier_bundle["mean"], classifier_bundle["scale"], left_states[:, -1])
        diag_right, _ = diagnostic_predictions(classifier_bundle["classifier"], classifier_bundle["mean"], classifier_bundle["scale"], right_states[:, -1])
        result[pair_name] = {
            "formula_gap": formula_gap,
            "candidate_l2": distribution(torch.linalg.vector_norm(candidate_gap, dim=-1)),
            "candidate_equal_fraction": float(torch.all(cand_left == cand_right, dim=-1).float().mean()),
            "saturation_fraction": distribution(saturation),
            "exact_pm1_fraction": distribution(exact),
            "preactivation_abs_median": float(pre_left.abs().median()),
            "high_saturation_equal_fraction": high_fraction,
            "low_saturation_equal_fraction": low_fraction,
            "checkpoints": checkpoints,
            "readout": readout_record(pair_name, bit_a, final_left, final_right, diag_left, diag_right, logits_left, logits_right),
            "per_example": per_example,
        }
    if parameter_fingerprint(model) != before:
        raise ImplementationError("scoring changed parameters")
    return result


def readout_record(pair_name, bit_a, original_left, original_right, diagnostic_left, diagnostic_right, logits_left, logits_right):
    def both(left, right, target_left, target_right):
        return float(((left == target_left) & (right == target_right)).float().mean())

    ones = torch.ones_like(bit_a)
    if pair_name == "marked_new_bit":
        selective = (ones, -ones)
        hold = (bit_a, bit_a)
    elif pair_name == "unmarked_new_bit":
        selective = (bit_a, bit_a)
        hold = (bit_a, bit_a)
    else:
        selective = (bit_a, -bit_a)
        hold = (bit_a, bit_a)
    difference = logits_left.double() - logits_right.double()
    return {
        "original_selective_expectation": both(original_left, original_right, *selective),
        "original_hold_expectation": both(original_left, original_right, *hold),
        "diagnostic_selective_expectation": both(diagnostic_left, diagnostic_right, *selective),
        "diagnostic_hold_expectation": both(diagnostic_left, diagnostic_right, *hold),
        "original_flip_fraction": float((original_left != original_right).float().mean()),
        "finite_logit_difference_fraction": float(torch.isfinite(difference).float().mean()),
    }


def to_float64(model):
    return copy.deepcopy(model).double()


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-018 registration")
    verify_derivative()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for model_type in MODELS:
                name = condition_name(model_type, task)
                checkpoint = EXP17 / "checkpoints" / f"{name}_seed{seed}.pt"
                classifier = EXP17 / "classifiers" / f"{name}_seed{seed}.joblib"
                rows.append({
                    "seed": seed, "condition": name,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "classifier_sha256": file_sha256(classifier),
                })
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-018",
        "diagnostic_split_index": 4,
        "checkpoints": rows,
        "no_training": True,
        "no_new_fit": True,
        "budget_seconds": BUDGET,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"registered EXP-018; {len(rows)} checkpoints")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if key != "per_example"}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the recorded versions")
    if any((OUT_DIR / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-018 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to evaluate")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-018")
    started = time.perf_counter()
    deadline = started + BUDGET
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nDiagnostic code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-018", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds": BUDGET, "no_optimizer": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    verify_derivative()
    checks = verify_saved()
    (OUT_DIR / "verification.json").write_text(json.dumps({"checks": checks}, indent=2), encoding="utf-8")
    print(f"verified {len(checks)} checkpoints", flush=True)
    unstarted = []
    for seed in SEEDS:
        if time.perf_counter() >= deadline:
            unstarted.append({"seed": seed, "reason": "allowance exhausted"})
            continue
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "start", "seed": seed, "session_elapsed_seconds": round(time.perf_counter() - started, 3)})
        latent = make_latent(512, seed, 4)
        task_views(latent)
        pairs = build_pairs(latent)
        seed_records = []
        for task in TASKS:
            for model_type in MODELS:
                name = condition_name(model_type, task)
                payload = torch.load(EXP17 / "checkpoints" / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=False)
                model = ResidualTanhRNN(3, 32, 1, residual_scale=1.0) if model_type.startswith("additive") else __import__("creditlab.models", fromlist=["TinySequenceRNN"]).TinySequenceRNN(3, 32, 1)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                classifier_bundle = joblib.load(EXP17 / "classifiers" / f"{name}_seed{seed}.joblib")
                float32 = score_model(model, classifier_bundle, latent, pairs)
                shadow = score_model(to_float64(model), classifier_bundle, latent, {key: (value[0].double(), value[1].double()) for key, value in pairs.items()})
                example_path = OUT_DIR / "per_example" / f"{name}_seed{seed}.npz"
                example_path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {}
                for pair_name, item in float32.items():
                    if not isinstance(item, dict) or "per_example" not in item:
                        continue
                    for checkpoint, values in item["per_example"].items():
                        arrays[f"{pair_name}_{checkpoint}_equal"] = values["equal"]
                        arrays[f"{pair_name}_{checkpoint}_l2"] = values["l2"]
                temporary = example_path.with_suffix(".partial.npz")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                temporary.replace(example_path)
                record = {"seed": seed, "condition": name, "task": task, "model_type": model_type, "float32": json_ready(float32), "float64": json_ready(shadow)}
                seed_records.append(record)
                append_jsonl(OUT_DIR / "metrics.jsonl", record)
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "finish", "seed": seed, "session_elapsed_seconds": round(time.perf_counter() - started, 3)})
        print(json.dumps({"seed": seed, "done": True}, sort_keys=True), flush=True)
    summary = {"experiment_id": "EXP-018", "elapsed_seconds": round(time.perf_counter() - started, 3), "unstarted": unstarted, "verified": len(checks)}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["seed,condition,pair,precision,equal_after,equal_final,median_l2_after,selective_original,selective_diagnostic"]
    for row in load_jsonl(OUT_DIR / "metrics.jsonl"):
        for precision in ("float32", "float64"):
            block = row[precision]
            for pair_name in PAIRS:
                item = block[pair_name]
                lines.append(",".join(map(str, [
                    row["seed"], row["condition"], pair_name, precision,
                    item["checkpoints"]["after_update"]["equal_fraction"],
                    item["checkpoints"]["after_query"]["equal_fraction"],
                    item["checkpoints"]["after_update"]["l2"]["median"],
                    item["readout"]["original_selective_expectation"],
                    item["readout"]["diagnostic_selective_expectation"],
                ])))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-018")
    print(f"EXP-018 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

"""EXP-022: update one memory without losing the other.

Registration, before any accuracy:
    python experiments/run_exp022.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp022.py
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
from creditlab.models import AddressedResetAdditiveRNN, MarkerResetAdditiveRNN, ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate  # noqa: E402
from run_exp004 import combined_seed, file_sha256, index_fingerprint, minibatch_indices, parameter_fingerprint  # noqa: E402
from run_exp011 import apply_clipping  # noqa: E402
from run_exp012 import final_states, refit_predictions, sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, atomic_torch_save, fit_diagnostic, git_revision  # noqa: E402
from run_exp017 import terminal_predictions  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp022_two_memory.yaml"
OUT_DIR = ROOT / "results" / "EXP-022"
SEEDS = [701, 709, 719, 727, 733, 739, 743, 751, 757, 761]
BASE_SEED = 9000
LENGTH = 130
BUDGET = 1200.0
PER_RUN = 300.0
BLOCKED = {"summary.json", "runtime_session.json", "raw_metrics.jsonl"}
MODELS = ("additive_tanh_rnn", "whole_reset_additive", "addressed_reset_additive")
JOBS = tuple({"seed": seed, "model_type": model_type, "attempt": 1} for seed in SEEDS for model_type in MODELS)
SUBGROUPS = (
    "no_replacement_query0",
    "no_replacement_query1",
    "replacement_updated_same",
    "replacement_updated_conflict",
    "replacement_untouched_query0",
    "replacement_untouched_query1",
)


class ImplementationError(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def current_bits(bit_a, bit_b, bit_c, flag, slot):
    slot0 = torch.where((flag == 1) & (slot == 0), bit_c, bit_a)
    slot1 = torch.where((flag == 1) & (slot == 1), bit_c, bit_b)
    return slot0, slot1


def labels(bit_a, bit_b, bit_c, flag, slot, query):
    slot0, slot1 = current_bits(bit_a, bit_b, bit_c, flag, slot)
    return torch.where(query == 0, slot0, slot1)


def make_two_memory(num_samples: int, run_seed: int, split_index: int, base_seed: int = BASE_SEED) -> dict:
    if num_samples % 64 != 0:
        raise ImplementationError("split size is not divisible by 64")
    generator = torch.Generator(device="cpu").manual_seed(combined_seed(run_seed, 128, split_index, base_seed))
    cells = [
        (a, b, c, f, s, q)
        for a in (-1.0, 1.0)
        for b in (-1.0, 1.0)
        for c in (-1.0, 1.0)
        for f in (0.0, 1.0)
        for s in (0.0, 1.0)
        for q in (0.0, 1.0)
    ]
    repeated = [cell for cell in cells for _ in range(num_samples // 64)]
    order = torch.randperm(num_samples, generator=generator)
    ordered = [repeated[int(index)] for index in order]
    bit_a = torch.tensor([item[0] for item in ordered])
    bit_b = torch.tensor([item[1] for item in ordered])
    bit_c = torch.tensor([item[2] for item in ordered])
    flag = torch.tensor([item[3] for item in ordered])
    slot = torch.tensor([item[4] for item in ordered])
    query = torch.tensor([item[5] for item in ordered])
    timing = torch.randint(32, 97, (num_samples,), generator=generator)
    inputs = torch.zeros(num_samples, LENGTH, 6)
    inputs[:, 0, 0] = bit_a
    inputs[:, 0, 1] = 1.0
    inputs[:, 0, 3] = 1.0
    inputs[:, 0, 4] = 1.0
    inputs[:, 1, 0] = bit_b
    inputs[:, 1, 1] = 1.0
    inputs[:, 1, 3] = 1.0
    inputs[:, 1, 5] = 1.0
    span = 127
    inputs[:, 2:129, 0] = torch.randint(0, 2, (num_samples, span), generator=generator).float() * 2 - 1
    competitor_mask = torch.rand(num_samples, span, generator=generator) < 0.4
    torch.rand(num_samples, span, generator=generator)
    competitor_bits = torch.where(torch.rand(num_samples, span, generator=generator) < 0.5, -1.0, 1.0)
    rows = torch.arange(num_samples)
    offset = timing - 2
    competitor_mask[rows, offset] = False
    inputs[:, 2:129, 0] = torch.where(competitor_mask, competitor_bits, inputs[:, 2:129, 0])
    inputs[:, 2:129, 1] = torch.where(competitor_mask, torch.ones_like(competitor_bits), inputs[:, 2:129, 1])
    noise = torch.randn(num_samples, span, 3, generator=generator)
    noise[rows, offset] = 0
    inputs[:, 2:129, :3] += 0.3 * noise
    inputs[rows, timing, 0] = bit_c
    inputs[rows, timing, 1] = flag
    inputs[rows, timing, 2] = 0.0
    inputs[rows, timing, 3] = flag
    inputs[rows, timing, 4] = (slot == 0).float()
    inputs[rows, timing, 5] = (slot == 1).float()
    inputs[:, 129, 2] = 1.0
    inputs[:, 129, 4] = (query == 0).float()
    inputs[:, 129, 5] = (query == 1).float()
    targets = labels(bit_a, bit_b, bit_c, flag, slot, query)
    _assert_contract(inputs, timing, bit_a, bit_b, bit_c, flag, slot, query, targets, competitor_mask)
    return {
        "inputs": inputs, "targets": targets, "A": bit_a, "B": bit_b, "C": bit_c,
        "F": flag, "S": slot, "Q": query, "timing": timing, "competitor_mask": competitor_mask,
    }


def _assert_contract(inputs, timing, bit_a, bit_b, bit_c, flag, slot, query, targets, competitor_mask) -> None:
    rows = torch.arange(inputs.shape[0])
    control = inputs[:, :, 3]
    if not set(control.unique().tolist()) <= {0.0, 1.0}:
        raise ImplementationError("clean write control is not binary")
    if not torch.equal(control[:, 0], torch.ones(inputs.shape[0])) or not torch.equal(control[:, 1], torch.ones(inputs.shape[0])):
        raise ImplementationError("the initial writes are not marked")
    if not torch.equal(control[rows, timing], flag) or not torch.equal(control[:, 129], torch.zeros(inputs.shape[0])):
        raise ImplementationError("candidate or query write control is wrong")
    if not torch.equal(inputs[rows, timing, 4], (slot == 0).float()) or not torch.equal(inputs[:, 129, 4], (query == 0).float()):
        raise ImplementationError("addresses are not one-hot")
    if int((inputs[:, 129, 4] + inputs[:, 129, 5] != 1).sum()) != 0:
        raise ImplementationError("the query address is not one-hot")
    distractor = torch.ones_like(control, dtype=torch.bool)
    distractor[:, 0] = False
    distractor[:, 1] = False
    distractor[rows, timing] = False
    distractor[:, 129] = False
    if int(inputs[:, :, 3:6][distractor].abs().sum()) != 0:
        raise ImplementationError("a distractor carries a clean control")
    if not torch.equal(inputs[rows, timing, 0], bit_c) or not torch.equal(inputs[:, 0, 0], bit_a):
        raise ImplementationError("a true token was left noisy")
    if not torch.equal(targets, labels(bit_a, bit_b, bit_c, flag, slot, query)):
        raise ImplementationError("labels do not match the 64-combination rule")
    per = inputs.shape[0] // 64
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            for c in (-1.0, 1.0):
                for f in (0.0, 1.0):
                    for s in (0.0, 1.0):
                        for q in (0.0, 1.0):
                            count = int(((bit_a == a) & (bit_b == b) & (bit_c == c) & (flag == f) & (slot == s) & (query == q)).sum())
                            if count != per:
                                raise ImplementationError("the 64 combinations are not balanced")
    if int(competitor_mask[rows, timing - 2].sum()) != 0:
        raise ImplementationError("the candidate was left inside the distractor generator")


def subgroup_masks(bit_a, bit_b, bit_c, flag, slot, query) -> dict[str, torch.Tensor]:
    old = torch.where(slot == 0, bit_a, bit_b)
    updated_query = (flag == 1) & (query == slot)
    untouched_query = (flag == 1) & (query != slot)
    groups = {
        "no_replacement_query0": (flag == 0) & (query == 0),
        "no_replacement_query1": (flag == 0) & (query == 1),
        "replacement_updated_same": updated_query & (old == bit_c),
        "replacement_updated_conflict": updated_query & (old != bit_c),
        "replacement_untouched_query0": untouched_query & (query == 0),
        "replacement_untouched_query1": untouched_query & (query == 1),
    }
    covered = torch.zeros(bit_a.shape[0], dtype=torch.bool)
    for mask in groups.values():
        if bool((covered & mask).any()):
            raise ImplementationError("subgroups overlap")
        covered |= mask
    if not bool(covered.all()):
        raise ImplementationError("subgroups do not cover the split")
    return groups


def score_groups(predictions, targets, bit_a, bit_b, bit_c, flag, slot, query) -> dict:
    groups = subgroup_masks(bit_a, bit_b, bit_c, flag, slot, query)
    record = {}
    for name, mask in groups.items():
        size = int(mask.sum())
        correct = int((predictions[mask] == targets[mask]).sum())
        record[name] = {"size": size, "correct": correct, "accuracy": correct / size}
    updated = (flag == 1) & (query == slot)
    for address in (0.0, 1.0):
        mask = updated & (slot == address)
        size = int(mask.sum())
        correct = int((predictions[mask] == targets[mask]).sum()) if size else 0
        record[f"updated_slot_address_{int(address)}"] = {"size": size, "correct": correct, "accuracy": None if size == 0 else correct / size}
    return record


def build_models(seed: int) -> dict:
    seed_everything(seed)
    additive = ResidualTanhRNN(6, 32, 1, residual_scale=1.0)
    whole = MarkerResetAdditiveRNN(6, 32, 1, residual_scale=1.0, marker_index=3)
    addressed = AddressedResetAdditiveRNN(6, 32, 1, residual_scale=1.0)
    whole.load_state_dict(additive.state_dict())
    addressed.load_state_dict(additive.state_dict())
    models = {"additive_tanh_rnn": additive, "whole_reset_additive": whole, "addressed_reset_additive": addressed}
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()}
    if set(counts.values()) != {1313}:
        raise ImplementationError(f"parameter counts are not equal: {counts}")
    fingerprints = {name: parameter_fingerprint(model) for name, model in models.items()}
    if len(set(fingerprints.values())) != 1:
        raise ImplementationError("initial parameters do not match")
    return models


def structural_whole_reset(model) -> dict:
    data = make_two_memory(64, 701, 0)
    inputs = data["inputs"].clone()
    rows = torch.arange(64)
    inputs[rows, data["timing"], 0] = 1.0
    inputs[rows, data["timing"], 1] = 1.0
    inputs[rows, data["timing"], 3] = 1.0
    inputs[rows, data["timing"], 4] = 0.0
    inputs[rows, data["timing"], 5] = 1.0
    flipped = inputs.clone()
    flipped[:, 0, 0] = -inputs[:, 0, 0]
    first = model.recurrent_states(inputs)
    second = model.recurrent_states(flipped)
    gap = float((first[:, 1:] - second[:, 1:]).detach().abs().max())
    return {"max_abs_difference_from_second_write_onward": gap, "structural_independence": gap == 0.0}


def reset_equation_checks() -> None:
    models = build_models(0)
    hidden = torch.randn(4, 32)
    marked = torch.zeros(4, 6)
    marked[:, 3] = 1
    marked[:, 4] = 1
    whole_state, whole_candidate = models["whole_reset_additive"].candidate_step(marked, hidden)
    if not torch.equal(whole_state - whole_candidate, torch.zeros_like(hidden)):
        raise ImplementationError("whole reset did not clear every coordinate before the candidate")
    addressed = models["addressed_reset_additive"]
    carried = addressed.carried_state(marked, hidden)
    if not torch.equal(carried[:, :16], torch.zeros(4, 16)) or not torch.equal(carried[:, 16:], hidden[:, 16:]):
        raise ImplementationError("addressed reset did not clear only the selected half")
    state, candidate = addressed.candidate_step(marked, hidden)
    if float((state - (carried + candidate)).detach().abs().max()) > 1e-6:
        raise ImplementationError("the candidate was not added after the addressed reset")
    unmarked = marked.clone()
    unmarked[:, 3] = 0
    if not torch.equal(addressed.carried_state(unmarked, hidden), hidden):
        raise ImplementationError("an unmarked input triggered a reset")
    query = torch.zeros(4, 6)
    query[:, 2] = 1
    query[:, 5] = 1
    if not torch.equal(addressed.carried_state(query, hidden), hidden):
        raise ImplementationError("the query triggered a reset")
    noisy = unmarked.clone()
    noisy[:, 1] = 4
    if not torch.equal(models["whole_reset_additive"].candidate_step(noisy, hidden)[0], models["additive_tanh_rnn"].candidate_step(noisy, hidden)[0]):
        raise ImplementationError("the noisy write channel triggered a reset")
    report = structural_whole_reset(models["whole_reset_additive"])
    if not report["structural_independence"]:
        raise ImplementationError("whole-state reset still depended on a bit written before the wipe")
    return report


def preflight() -> dict:
    equation = structural_whole_reset(build_models(0)["whole_reset_additive"])
    reset_equation_checks()
    data = make_two_memory(512, 701, 2)
    groups = subgroup_masks(data["A"], data["B"], data["C"], data["F"], data["S"], data["Q"])
    report = {
        "structural_whole_reset": equation,
        "subgroup_sizes": {name: int(mask.sum()) for name, mask in groups.items()},
        "parameter_count": 1313,
        "clean_control_binary": True,
    }
    if not equation["structural_independence"]:
        raise ImplementationError("structural whole-reset check failed")
    return report


def candidate_diagnostics(model, inputs, timing, flag) -> dict:
    states = model.recurrent_states(inputs)
    rows = torch.arange(inputs.shape[0])
    before = states[rows, timing - 1]
    _state, candidate = model.candidate_step(inputs[rows, timing], before)
    marked = flag == 1
    def region(values):
        return {
            "slot0_rms": float(values[:, :16].pow(2).mean().sqrt()),
            "slot1_rms": float(values[:, 16:].pow(2).mean().sqrt()),
        }
    return {
        "before_update": region(before),
        "after_update": region(states[rows, timing]),
        "saturation_fraction": float(((1 - candidate.detach() ** 2) < 1e-6).float().mean()),
        "marked_saturation_fraction": None if int(marked.sum()) == 0 else float(((1 - candidate.detach()[marked] ** 2) < 1e-6).float().mean()),
    }


def train_job(job: dict, seconds_left: float) -> None:
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    seed = int(job["seed"])
    model_type = job["model_type"]
    bundle = {name: make_two_memory(count, seed, split) for count, split, name in ((1536, 0, "train"), (512, 1, "val"), (512, 2, "test"))}
    model = build_models(seed)[model_type]
    indices = minibatch_indices(seed, 128, 1536, 64, 400)
    append_jsonl(OUT_DIR / "fingerprints.jsonl", {
        "seed": seed, "model_type": model_type, "init_fingerprint": parameter_fingerprint(model),
        "train_x": file_sha256_tensor(bundle["train"]["inputs"]), "minibatch_fingerprint": index_fingerprint(indices),
    })
    started = time.perf_counter()
    deadline = started + min(PER_RUN, seconds_left)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {"experiment_id": "EXP-022", "model_type": model_type, "seed": seed, "attempt": 1, "status": "ok", "stop_reason": None, "code_revision": git_revision(), "parameter_count": 1313}
    loss_curve, checkpoints, grad_norms, clipped = [], [], [], 0
    try:
        loss, accuracy = evaluate(model, bundle["val"]["inputs"], bundle["val"]["targets"], 256)
        checkpoints.append({"update": 0, "val_loss": loss, "val_accuracy": accuracy})
        for update, rows in enumerate(indices):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"CPU budget exhausted at update {update}")
            logits, _hidden = model(bundle["train"]["inputs"][rows])
            loss = bce(logits[:, -1, 0], (bundle["train"]["targets"][rows] + 1) / 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            optimizer.zero_grad()
            loss.backward()
            clip_record = apply_clipping(list(model.parameters()), True)
            grad_norms.append(clip_record["raw_grad_norm_float64"])
            clipped += int(clip_record["clipping_applied"])
            optimizer.step()
            loss_curve.append(float(loss.detach()))
            if (update + 1) in (50, 100, 200, 400):
                val_loss, val_accuracy = evaluate(model, bundle["val"]["inputs"], bundle["val"]["targets"], 256)
                checkpoints.append({"update": update + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
        test = bundle["test"]
        test_loss, _accuracy = evaluate(model, test["inputs"], test["targets"], 256)
        predictions = terminal_predictions(model, test["inputs"])
        correct = int((predictions == test["targets"]).sum())
        groups = score_groups(predictions, test["targets"], test["A"], test["B"], test["C"], test["F"], test["S"], test["Q"])
        success = correct / 512 >= 0.95 and all(groups[name]["accuracy"] >= 0.95 for name in SUBGROUPS)
        magnitude = hidden_magnitude_stats(model, test["inputs"])
        record.update(
            updates_completed=400, test_loss=test_loss, test_accuracy=correct / 512, test_correct=correct,
            subgroups=groups, success=success, grad_clip_fraction=clipped / 400,
            mean_raw_grad_norm=float(np.mean(grad_norms)), max_abs_hidden=magnitude["max_abs_hidden"],
            runtime_seconds=round(time.perf_counter() - started, 3),
            **candidate_diagnostics(model, test["inputs"], test["timing"], test["F"]),
        )
        destination = OUT_DIR / "checkpoints" / f"{model_type}_seed{seed}.pt"
        record["checkpoint_sha256"] = atomic_torch_save(destination, {"experiment_id": "EXP-022", "model_type": model_type, "seed": seed, "state_dict": model.state_dict()})
        fitted = fit_diagnostic(model, bundle["train"]["inputs"], bundle["train"]["targets"], bundle["val"]["inputs"], bundle["val"]["targets"], test["inputs"], test["targets"])
        classifier_path = OUT_DIR / "classifiers" / f"{model_type}_seed{seed}.joblib"
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"]}, classifier_path)
        diagnostic_predictions = torch.from_numpy(refit_predictions(fitted["decision"](final_states(model, test["inputs"])))).float()
        record["diagnostic_test_accuracy"] = fitted["diagnostic_test_accuracy"]
        record["diagnostic_subgroups"] = score_groups(diagnostic_predictions, test["targets"], test["A"], test["B"], test["C"], test["F"], test["S"], test["Q"])
        record["classifier_sha256"] = file_sha256(classifier_path)
        record["audits"] = audit_measurement(model, seed)
    except (FloatingPointError, TimeoutError) as exc:
        record.update(status="budget_stop" if isinstance(exc, TimeoutError) else "numerical_failure", stop_reason=f"{type(exc).__name__}: {exc}", updates_completed=len(loss_curve), success=False, runtime_seconds=round(time.perf_counter() - started, 3))
    append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
    append_jsonl(OUT_DIR / "loss_curves.jsonl", {"seed": seed, "model_type": model_type, "loss_curve": loss_curve})
    for item in checkpoints:
        append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {"seed": seed, "model_type": model_type, **item})
    print(json.dumps({"seed": seed, "model_type": model_type, "status": record["status"], "test_accuracy": record.get("test_accuracy")}, sort_keys=True), flush=True)


def file_sha256_tensor(tensor: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rewrite_candidate(inputs, timing, bit, flag, slot):
    copied = inputs.clone()
    rows = torch.arange(copied.shape[0])
    copied[rows, timing, 0] = bit
    copied[rows, timing, 1] = flag
    copied[rows, timing, 2] = 0
    copied[rows, timing, 3] = flag
    copied[rows, timing, 4] = (slot == 0).float()
    copied[rows, timing, 5] = (slot == 1).float()
    return copied


def set_query(inputs, query):
    copied = inputs.clone()
    copied[:, 129, 3] = 0
    copied[:, 129, 4] = (query == 0).float()
    copied[:, 129, 5] = (query == 1).float()
    return copied


def audit_measurement(model, seed: int) -> dict:
    data = make_two_memory(256, seed, 3)
    inputs, timing = data["inputs"], data["timing"]
    rows = torch.arange(256)
    def both(left, right, target_left, target_right):
        pred_left = terminal_predictions(model, left)
        pred_right = terminal_predictions(model, right)
        return {
            "both_correct_fraction": float(((pred_left == target_left) & (pred_right == target_right)).float().mean()),
            "prediction_agreement_fraction": float((pred_left == pred_right).float().mean()),
            "final_state_equal_fraction": float(torch.all(model.recurrent_states(left)[:, -1] == model.recurrent_states(right)[:, -1], dim=-1).float().mean()),
        }
    marked_left = rewrite_candidate(inputs, timing, 1, 1, data["S"])
    marked_right = rewrite_candidate(inputs, timing, -1, 1, data["S"])
    marked_left = set_query(marked_left, data["S"])
    marked_right = set_query(marked_right, data["S"])
    untouched_left = rewrite_candidate(inputs, timing, data["C"], 1, torch.ones(256))
    untouched_right = untouched_left.clone()
    untouched_right[:, 0, 0] = -untouched_left[:, 0, 0]
    untouched_left = set_query(untouched_left, torch.zeros(256))
    untouched_right = set_query(untouched_right, torch.zeros(256))
    obsolete_left = rewrite_candidate(inputs, timing, torch.ones(256), 1, torch.zeros(256))
    obsolete_right = obsolete_left.clone()
    obsolete_right[:, 0, 0] = -obsolete_left[:, 0, 0]
    obsolete_left = set_query(obsolete_left, torch.zeros(256))
    obsolete_right = set_query(obsolete_right, torch.zeros(256))
    unmarked_left = rewrite_candidate(inputs, timing, 1, 0, data["S"])
    unmarked_right = rewrite_candidate(inputs, timing, -1, 0, data["S"])
    unmarked_left = set_query(unmarked_left, torch.zeros(256))
    unmarked_right = set_query(unmarked_right, torch.zeros(256))
    query_left = set_query(inputs, torch.zeros(256))
    query_right = set_query(inputs, torch.ones(256))
    differ = current_bits(data["A"], data["B"], data["C"], data["F"], data["S"])
    differing = differ[0] != differ[1]
    return {
        "marked_new_bit": both(marked_left, marked_right, torch.ones(256), -torch.ones(256)),
        "untouched_bit": both(untouched_left, untouched_right, untouched_left[:, 0, 0], untouched_right[:, 0, 0]),
        "obsolete_bit": both(obsolete_left, obsolete_right, torch.ones(256), torch.ones(256)),
        "unmarked_new_bit": both(unmarked_left, unmarked_right, data["A"], data["A"]),
        "query_switch": {
            **both(query_left, query_right, differ[0], differ[1]),
            "histories_with_different_current_values": int(differing.sum()),
        },
    }


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017", "018", "019", "020", "021"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def find_seed_conflicts() -> list[str]:
    hits = []
    needles = [f'"seed": {seed}' for seed in SEEDS]
    for folder in (ROOT / "results", ROOT / "configs", ROOT / "experiments"):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "EXP-022" in path.parts:
                continue
            if path.suffix.lower() not in {".jsonl", ".json", ".yaml", ".yml", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits.extend(f"{path.relative_to(ROOT).as_posix()}: {needle}" for needle in needles if needle in text)
    return hits


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-022 registration")
    conflicts = find_seed_conflicts()
    if conflicts:
        raise ImplementationError("seed conflict: " + "; ".join(conflicts[:8]))
    reset_equation_checks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-022", "seeds": SEEDS, "base_seed": BASE_SEED,
        "models": list(MODELS), "jobs": len(JOBS), "parameter_count": 1313,
        "budget_seconds_total": BUDGET, "seed_conflicts": [],
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print("registered EXP-022; seeds unused")


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
        raise RuntimeError("EXP-022 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-022")
    started = time.perf_counter()
    deadline = started + BUDGET
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nTraining code revision: {revision}\nParameter count: 1313\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-022", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds_total": BUDGET,
    }, indent=2, sort_keys=True), encoding="utf-8")
    report = preflight()
    (OUT_DIR / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("preflight passed", flush=True)
    unstarted = []
    stopped = False
    for job in JOBS:
        remaining = deadline - time.perf_counter()
        if remaining <= 1 or stopped:
            unstarted.append({"seed": job["seed"], "model_type": job["model_type"], "reason": "not started"})
            continue
        log_dir = OUT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{job['model_type']}_seed{job['seed']}.stdout.txt"
        stderr_path = log_dir / f"{job['model_type']}_seed{job['seed']}.stderr.txt"
        job_started = time.perf_counter()
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "start", "seed": job["seed"], "model_type": job["model_type"], "session_elapsed_seconds": round(job_started - started, 3)})
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--train-one", json.dumps({"job": job, "seconds_left": remaining})],
                    cwd=str(ROOT), stdout=stdout, stderr=stderr, timeout=remaining,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "finish", "seed": job["seed"], "model_type": job["model_type"], "exit_code": exit_code, "runtime_seconds": round(time.perf_counter() - job_started, 3), "session_elapsed_seconds": round(time.perf_counter() - started, 3)})
        print(json.dumps({"seed": job["seed"], "model_type": job["model_type"], "exit_code": exit_code}, sort_keys=True), flush=True)
        if exit_code != 0:
            stopped = True
    records = [json.loads(line) for line in (OUT_DIR / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT_DIR / "raw_metrics.jsonl").exists() else []
    summary = {"experiment_id": "EXP-022", "elapsed_seconds": round(time.perf_counter() - started, 3), "unstarted": unstarted, "records": records}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["seed,model_type,status,test_accuracy,test_correct,success,diagnostic_test_accuracy"]
    for row in records:
        lines.append(",".join(str(row.get(field, "")) for field in ("seed", "model_type", "status", "test_accuracy", "test_correct", "success", "diagnostic_test_accuracy")))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-022")
    print(f"EXP-022 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

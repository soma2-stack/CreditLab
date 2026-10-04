"""EXP-020: selective overwrite with an explicit clean write control.

Registration, before any accuracy:
    python experiments/run_exp020.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp020.py
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
from creditlab.models import MarkerResetAdditiveRNN, ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp001 import evaluate  # noqa: E402
from run_exp004 import file_sha256, index_fingerprint, minibatch_indices, parameter_fingerprint, tensor_fingerprint  # noqa: E402
from run_exp011 import apply_clipping  # noqa: E402
from run_exp012 import sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, atomic_torch_save, fit_diagnostic, git_revision  # noqa: E402
from run_exp017 import (  # noqa: E402
    SEEDS,
    make_audit_pair,
    make_latent,
    subgroup_record,
    task_views,
    terminal_predictions,
)

CONFIG_PATH = ROOT / "configs" / "exp020_clean_write.yaml"
OUT_DIR = ROOT / "results" / "EXP-020"
TOLERANCE = 1e-5
BUDGET = 900.0
PER_RUN = 300.0
BLOCKED = {"summary.json", "runtime_session.json", "raw_metrics.jsonl"}
JOBS = tuple(
    {"seed": seed, "model_type": model_type, "task": task, "attempt": 1}
    for seed in SEEDS
    for task in ("hold", "selective")
    for model_type in ("additive_tanh_rnn", "marker_reset_additive")
)


class ImplementationError(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def condition_name(model_type: str, task: str) -> str:
    family = "reset" if model_type.startswith("marker_reset") else "additive"
    return f"{family}_{task}"


def paired_models(seed: int) -> tuple[ResidualTanhRNN, MarkerResetAdditiveRNN]:
    seed_everything(seed)
    control = ResidualTanhRNN(4, 32, 1, residual_scale=1.0)
    reset = MarkerResetAdditiveRNN(4, 32, 1, residual_scale=1.0, marker_index=3)
    reset.load_state_dict(control.state_dict())
    return control, reset


def with_control(inputs: torch.Tensor, timing: torch.Tensor, candidate_control: torch.Tensor) -> torch.Tensor:
    control = torch.zeros(inputs.shape[0], inputs.shape[1], 1)
    control[:, 0, 0] = 1.0
    rows = torch.arange(inputs.shape[0])
    control[rows, timing, 0] = candidate_control
    return torch.cat([inputs, control], dim=-1)


def four_channel_bundle(seed: int, task: str) -> dict:
    bundle = {}
    for num_samples, split_index, name in ((1536, 0, "train"), (512, 1, "val"), (512, 2, "test")):
        latent = make_latent(num_samples, seed, split_index)
        hold_x, hold_y, selective_x, selective_y = task_views(latent)
        base, targets = (hold_x, hold_y) if task == "hold" else (selective_x, selective_y)
        candidate_control = torch.zeros_like(latent["F"]) if task == "hold" else latent["F"]
        bundle[f"{name}_x"] = with_control(base, latent["timing"], candidate_control)
        bundle[f"{name}_y"] = targets
        if not torch.equal(bundle[f"{name}_x"][:, :, :3], base):
            raise ImplementationError("the original three channels changed")
        if name == "test":
            bundle["A"], bundle["B"], bundle["F"], bundle["timing"] = latent["A"], latent["B"], latent["F"], latent["timing"]
            bundle["three_channel"] = base
    return bundle


def inspect_control(inputs: torch.Tensor, timing: torch.Tensor, candidate_control: torch.Tensor, targets: torch.Tensor, bit_a, bit_b) -> None:
    control = inputs[:, :, 3]
    values = set(control.unique().tolist())
    if not values <= {0.0, 1.0}:
        raise ImplementationError(f"clean control is not binary: {values}")
    rows = torch.arange(inputs.shape[0])
    if not torch.equal(control[:, 0], torch.ones(inputs.shape[0])):
        raise ImplementationError("the initial event is not marked in the clean control")
    if not torch.equal(control[:, 129], torch.zeros(inputs.shape[0])):
        raise ImplementationError("the query is marked in the clean control")
    if not torch.equal(control[rows, timing], candidate_control):
        raise ImplementationError("the candidate control is not the write flag")
    mask = torch.ones_like(control, dtype=torch.bool)
    mask[:, 0] = False
    mask[rows, timing] = False
    mask[:, 129] = False
    if int(control[mask].abs().sum()) != 0:
        raise ImplementationError("a distractor or competitor carries the clean control")
    candidate = control[rows, timing]
    for value in candidate.unique().tolist():
        selected = bit_a[candidate == value]
        if int((selected > 0).sum()) == 0 or int((selected < 0).sum()) == 0:
            raise ImplementationError("the clean control separates the sign of A")


def hold_match(seed: int = 503) -> dict:
    latent = make_latent(64, seed, 0)
    hold_x, hold_y, _selective_x, _selective_y = task_views(latent)
    inputs = with_control(hold_x, latent["timing"], torch.zeros(64))
    inspect_control(inputs, latent["timing"], torch.zeros(64), hold_y, latent["A"], latent["B"])
    control, reset = paired_models(seed)
    labels = hold_y
    control_states = control.recurrent_states(inputs)
    reset_states = reset.recurrent_states(inputs)
    state_gap = float((control_states - reset_states).detach().abs().max())
    logit_gap = float((control.readout(control_states[:, -1]) - reset.readout(reset_states[:, -1])).detach().abs().max())

    def grads(model):
        model.zero_grad(set_to_none=True)
        logits, _hidden = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1, 0], (labels + 1) / 2)
        loss.backward()
        return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])

    gradient_gap = float((grads(control) - grads(reset)).abs().max())
    return {
        "state_gap": state_gap,
        "logit_gap": logit_gap,
        "gradient_gap": gradient_gap,
        "matched": state_gap <= TOLERANCE and logit_gap <= TOLERANCE and gradient_gap <= TOLERANCE,
        "parameter_count": sum(parameter.numel() for parameter in control.parameters()),
    }


def local_rules() -> None:
    control, reset = paired_models(0)
    if parameter_fingerprint(control) != parameter_fingerprint(reset):
        raise ImplementationError("paired initialization differs")
    if sum(parameter.numel() for parameter in control.parameters()) != 1249:
        raise ImplementationError("parameter count is not 1249")
    hidden = torch.randn(4, 32)
    unmarked = torch.zeros(4, 4)
    unmarked[:, 0] = 0.4
    unmarked[:, 1] = 3.5
    if not torch.equal(control.candidate_step(unmarked, hidden)[0], reset.candidate_step(unmarked, hidden)[0]):
        raise ImplementationError("an unmarked step no longer matches additive recurrence")
    marked = unmarked.clone()
    marked[:, 3] = 1
    varied = hidden.clone().requires_grad_(True)
    state, _candidate = reset.candidate_step(marked, varied)
    (gradient,) = torch.autograd.grad(state.sum(), varied)
    if float(gradient.abs().max()) != 0.0:
        raise ImplementationError("the clean mark did not block the previous state")
    noisy = unmarked.clone()
    noisy[:, 1] = 1
    noisy_hidden = hidden.detach().clone().requires_grad_(True)
    noisy_state, _candidate = reset.candidate_step(noisy, noisy_hidden)
    (noisy_gradient,) = torch.autograd.grad(noisy_state.sum(), noisy_hidden)
    if float(noisy_gradient.abs().max()) == 0.0:
        raise ImplementationError("the old noisy write channel triggered a reset")


def local_rules() -> None:
    control, reset = paired_models(0)
    if parameter_fingerprint(control) != parameter_fingerprint(reset):
        raise ImplementationError("paired initialization differs")
    if sum(p.numel() for p in control.parameters()) != 1249:
        raise ImplementationError("parameter count is not 1249")
    hidden = torch.randn(4, 32)
    unmarked = torch.zeros(4, 4)
    unmarked[:, 0] = 0.4
    unmarked[:, 1] = 3.5
    same_control, _ = control.candidate_step(unmarked, hidden)
    same_reset, _ = reset.candidate_step(unmarked, hidden)
    if not torch.equal(same_control, same_reset):
        raise ImplementationError("an unmarked step no longer matches additive recurrence")
    marked = unmarked.clone()
    marked[:, 3] = 1
    varied = hidden.clone().requires_grad_(True)
    state, _ = reset.candidate_step(marked, varied)
    (gradient,) = torch.autograd.grad(state.sum(), varied)
    if float(gradient.abs().max()) != 0.0:
        raise ImplementationError("the clean mark did not block the previous state")
    noisy = unmarked.clone()
    noisy[:, 1] = 1
    noisy_hidden = hidden.detach().clone().requires_grad_(True)
    noisy_state, _ = reset.candidate_step(noisy, noisy_hidden)
    (noisy_gradient,) = torch.autograd.grad(noisy_state.sum(), noisy_hidden)
    if float(noisy_gradient.abs().max()) == 0.0:
        raise ImplementationError("the old noisy write channel triggered a reset")


def preflight() -> dict:
    local_rules()
    latent = make_latent(512, 503, 2)
    hold_x, hold_y, selective_x, selective_y = task_views(latent)
    again = task_views(make_latent(512, 503, 2))
    if not torch.equal(hold_x, again[0]) or not torch.equal(selective_y, again[3]):
        raise ImplementationError("the EXP-017 generator is not deterministic")
    hold_inputs = with_control(hold_x, latent["timing"], torch.zeros_like(latent["F"]))
    selective_inputs = with_control(selective_x, latent["timing"], latent["F"])
    inspect_control(hold_inputs, latent["timing"], torch.zeros_like(latent["F"]), hold_y, latent["A"], latent["B"])
    inspect_control(selective_inputs, latent["timing"], latent["F"], selective_y, latent["A"], latent["B"])
    report = hold_match(503)
    report["generator_fingerprints_saved_by_exp017"] = False
    report["three_channel_fingerprint"] = tensor_fingerprint(hold_x)
    if not report["matched"]:
        raise ImplementationError(
            f"hold trajectories do not match: state {report['state_gap']}, logit {report['logit_gap']}, gradient {report['gradient_gap']}"
        )
    return report


def build_model(model_type: str):
    if model_type == "additive_tanh_rnn":
        return ResidualTanhRNN(4, 32, 1, residual_scale=1.0)
    if model_type == "marker_reset_additive":
        return MarkerResetAdditiveRNN(4, 32, 1, residual_scale=1.0, marker_index=3)
    raise ImplementationError(model_type)


def candidate_diagnostics(model, inputs, timing, flag) -> dict:
    states = model.recurrent_states(inputs)
    rows = torch.arange(inputs.shape[0])
    before = states[rows, timing - 1]
    marker = inputs[rows, timing, 3:4]
    after_reset = (1.0 - marker) * before if isinstance(model, MarkerResetAdditiveRNN) else before
    _state, candidate = model.candidate_step(inputs[rows, timing], before)
    marked = flag == 1
    return {
        "mean_magnitude_before_update": float(before.detach().norm(dim=-1).mean()),
        "mean_magnitude_after_update": float(states[rows, timing].detach().norm(dim=-1).mean()),
        "mean_magnitude_after_reset": float(after_reset.detach().norm(dim=-1).mean()),
        "marked_magnitude_after_reset": None if int(marked.sum()) == 0 else float(after_reset[marked].detach().norm(dim=-1).mean()),
        "saturation_fraction": float(((1 - candidate.detach() ** 2) < 1e-6).float().mean()),
    }


def train_job(job: dict, seconds_left: float) -> None:
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    seed = int(job["seed"])
    model_type = job["model_type"]
    task = job["task"]
    name = condition_name(model_type, task)
    bundle = four_channel_bundle(seed, task)
    control, reset = paired_models(seed)
    model = control if model_type == "additive_tanh_rnn" else reset
    indices = minibatch_indices(seed, 128, 1536, 64, 400)
    append_jsonl(OUT_DIR / "fingerprints.jsonl", {
        "seed": seed, "condition": name,
        "init_fingerprint": parameter_fingerprint(model),
        "train_x": tensor_fingerprint(bundle["train_x"]),
        "train_y": tensor_fingerprint(bundle["train_y"]),
        "test_x": tensor_fingerprint(bundle["test_x"]),
        "minibatch_fingerprint": index_fingerprint(indices),
    })
    started = time.perf_counter()
    deadline = started + min(PER_RUN, seconds_left)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    record = {
        "experiment_id": "EXP-020", "model_type": model_type, "task": task, "condition": name,
        "seed": seed, "attempt": 1, "status": "ok", "stop_reason": None, "code_revision": git_revision(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    loss_curve, checkpoints, grad_norms = [], [], []
    clipped = 0
    try:
        loss, accuracy = evaluate(model, bundle["val_x"], bundle["val_y"], 256)
        checkpoints.append({"update": 0, "val_loss": loss, "val_accuracy": accuracy})
        for update, rows in enumerate(indices):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"CPU budget exhausted at update {update}")
            logits, _hidden = model(bundle["train_x"][rows])
            loss = bce(logits[:, -1, 0], (bundle["train_y"][rows] + 1) / 2)
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
                val_loss, val_accuracy = evaluate(model, bundle["val_x"], bundle["val_y"], 256)
                checkpoints.append({"update": update + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
        test_loss, test_accuracy = evaluate(model, bundle["test_x"], bundle["test_y"], 256)
        predictions = terminal_predictions(model, bundle["test_x"])
        correct = int((predictions == bundle["test_y"]).sum())
        magnitude = hidden_magnitude_stats(model, bundle["test_x"])
        record.update(
            updates_completed=400, test_loss=test_loss, test_accuracy=correct / 512, test_correct=correct,
            max_abs_hidden=magnitude["max_abs_hidden"], final_hidden_rms=magnitude["final_hidden_rms"],
            grad_clip_fraction=clipped / 400, mean_raw_grad_norm=float(np.mean(grad_norms)),
            runtime_seconds=round(time.perf_counter() - started, 3),
            **candidate_diagnostics(model, bundle["test_x"], bundle["timing"], bundle["F"] if task == "selective" else torch.zeros(512)),
        )
        if task == "selective":
            record["subgroups"] = subgroup_record(predictions, bundle["test_y"], bundle["A"], bundle["B"], bundle["F"])
            record["selective_success"] = bool(record["test_accuracy"] >= 0.95 and all(item["accuracy"] >= 0.95 for item in record["subgroups"].values()))
        else:
            record["hold_success"] = bool(record["test_accuracy"] >= 0.95)
        destination = OUT_DIR / "checkpoints" / f"{name}_seed{seed}.pt"
        record["checkpoint_sha256"] = atomic_torch_save(destination, {
            "experiment_id": "EXP-020", "model_type": model_type, "task": task, "seed": seed,
            "input_size": 4, "marker_index": 3, "state_dict": model.state_dict(),
        })
        fitted = fit_diagnostic(model, bundle["train_x"], bundle["train_y"], bundle["val_x"], bundle["val_y"], bundle["test_x"], bundle["test_y"])
        classifier_path = OUT_DIR / "classifiers" / f"{name}_seed{seed}.joblib"
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": fitted["classifier"], "mean": fitted["mean"], "scale": fitted["scale"]}, classifier_path)
        record["diagnostic_test_accuracy"] = fitted["diagnostic_test_accuracy"]
        record["classifier_sha256"] = file_sha256(classifier_path)
        if task == "selective":
            from run_exp012 import final_states, refit_predictions
            diagnostic_predictions = torch.from_numpy(refit_predictions(fitted["decision"](final_states(model, bundle["test_x"])))).float()
            record["diagnostic_subgroups"] = subgroup_record(diagnostic_predictions, bundle["test_y"], bundle["A"], bundle["B"], bundle["F"])
        record["entry"] = entry_measurement(model, seed)
        record["audits"] = audit_measurement(model, seed, task)
    except (FloatingPointError, TimeoutError) as exc:
        record.update(status="budget_stop" if isinstance(exc, TimeoutError) else "numerical_failure", stop_reason=f"{type(exc).__name__}: {exc}", updates_completed=len(loss_curve), runtime_seconds=round(time.perf_counter() - started, 3))
    append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
    append_jsonl(OUT_DIR / "loss_curves.jsonl", {"seed": seed, "condition": name, "loss_curve": loss_curve})
    for item in checkpoints:
        append_jsonl(OUT_DIR / "checkpoint_diagnostics.jsonl", {"seed": seed, "condition": name, **item})
    print(json.dumps({"seed": seed, "condition": name, "status": record["status"], "test_accuracy": record.get("test_accuracy")}, sort_keys=True), flush=True)


def audit_inputs(latent: dict, audit: str) -> tuple[torch.Tensor, torch.Tensor]:
    pair = make_audit_pair(latent, audit)
    timing = latent["timing"]
    if audit in ("replacement_bit", "ignore_obsolete_initial"):
        control_value = torch.ones(latent["A"].shape[0])
    else:
        control_value = torch.zeros(latent["A"].shape[0])
    return with_control(pair["positive"], timing, control_value), with_control(pair["negative"], timing, control_value), pair


def entry_measurement(model, seed: int) -> dict:
    latent = make_latent(256, seed, 3)
    timing = latent["timing"]
    rows = torch.arange(256)
    base = latent["hold_x"].clone()
    positive, negative = base.clone(), base.clone()
    positive[rows, timing, 0] = 1
    negative[rows, timing, 0] = -1
    positive[rows, timing, 1] = 1
    negative[rows, timing, 1] = 1
    ones = torch.ones(256)
    positive = with_control(positive, timing, ones)
    negative = with_control(negative, timing, ones)
    left = model.recurrent_states(positive)
    right = model.recurrent_states(negative)
    return {
        "mean_l2_after_update": float((left[rows, timing] - right[rows, timing]).norm(dim=-1).mean()),
        "mean_l2_at_query": float((left[:, -1] - right[:, -1]).norm(dim=-1).mean()),
        "equal_after_update_fraction": float(torch.all(left[rows, timing] == right[rows, timing], dim=-1).float().mean()),
        "equal_at_query_fraction": float(torch.all(left[:, -1] == right[:, -1], dim=-1).float().mean()),
    }


def audit_measurement(model, seed: int, task: str) -> dict:
    latent = make_latent(256, seed, 3)
    results = {}
    for audit in ("retained_initial", "replacement_bit", "ignore_unmarked_later", "ignore_obsolete_initial"):
        positive, negative, pair = audit_inputs(latent, audit)
        pred_positive = terminal_predictions(model, positive)
        pred_negative = terminal_predictions(model, negative)
        target_positive, target_negative = pair["selective_targets" if task == "selective" else "hold_targets"]
        both = (pred_positive == target_positive) & (pred_negative == target_negative)
        results[audit] = {
            "both_correct_fraction": float(both.float().mean()),
            "prediction_flip_fraction": float((pred_positive != pred_negative).float().mean()),
            "construction": "original candidate marker and clean control are set together",
        }
    return results


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017", "018", "019"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-020 registration")
    local_rules()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-020",
        "seeds": list(SEEDS),
        "input_size": 4,
        "parameter_count": 1249,
        "jobs": len(JOBS),
        "hold_match_tolerance": TOLERANCE,
        "budget_seconds_total": BUDGET,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print("registered EXP-020")


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
        raise RuntimeError("EXP-020 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-020")
    started = time.perf_counter()
    deadline = started + BUDGET
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nTraining code revision: {revision}\nParameter count: 1249\nEXP-017 parameter count: 1217\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-020", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds_total": BUDGET,
    }, indent=2, sort_keys=True), encoding="utf-8")
    report = preflight()
    (OUT_DIR / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("preflight passed", flush=True)
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
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "finish", "seed": job["seed"], "condition": name, "exit_code": exit_code, "runtime_seconds": round(time.perf_counter() - job_started, 3), "session_elapsed_seconds": round(time.perf_counter() - started, 3)})
        print(json.dumps({"seed": job["seed"], "condition": name, "exit_code": exit_code}, sort_keys=True), flush=True)
        if exit_code != 0:
            stopped = True
    summary = {
        "experiment_id": "EXP-020",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "unstarted": unstarted,
        "preflight": report,
        "records": [json.loads(line) for line in (OUT_DIR / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT_DIR / "raw_metrics.jsonl").exists() else [],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["seed,condition,status,test_accuracy,test_correct,hold_success,selective_success,diagnostic_test_accuracy"]
    for row in summary["records"]:
        lines.append(",".join(str(row.get(field, "")) for field in ("seed", "condition", "status", "test_accuracy", "test_correct", "hold_success", "selective_success", "diagnostic_test_accuracy")))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-020")
    print(f"EXP-020 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

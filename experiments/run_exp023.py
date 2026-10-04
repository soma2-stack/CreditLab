"""EXP-023: can both current memory values be read before the query?

Registration, before any new diagnostic score:
    python experiments/run_exp023.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp023.py
"""

from __future__ import annotations

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

from creditlab.models import AddressedResetAdditiveRNN, MarkerResetAdditiveRNN, ResidualTanhRNN  # noqa: E402
from run_exp001 import evaluate  # noqa: E402
from run_exp004 import file_sha256, parameter_fingerprint  # noqa: E402
from run_exp012 import apply_scale, bce_from_decision, binary_labels, fit_scale, make_classifier, refit_predictions, sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, git_revision  # noqa: E402
from run_exp017 import terminal_predictions  # noqa: E402
from run_exp022 import (  # noqa: E402
    MODELS,
    SEEDS,
    SUBGROUPS,
    current_bits,
    file_sha256_tensor,
    labels,
    make_two_memory,
    rewrite_candidate,
    score_groups,
    set_query,
)

CONFIG_PATH = ROOT / "configs" / "exp023_two_head_diagnostic.yaml"
OUT_DIR = ROOT / "results" / "EXP-023"
SOURCE_DIR = ROOT / "results" / "EXP-022"
BUDGET = 300.0
BLOCKED = {"summary.json", "runtime_session.json", "raw_metrics.jsonl"}
PREQUERY_INDEX = 128


class ImplementationError(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def blank_model(model_type: str):
    if model_type == "additive_tanh_rnn":
        return ResidualTanhRNN(6, 32, 1, residual_scale=1.0)
    if model_type == "whole_reset_additive":
        return MarkerResetAdditiveRNN(6, 32, 1, residual_scale=1.0, marker_index=3)
    if model_type == "addressed_reset_additive":
        return AddressedResetAdditiveRNN(6, 32, 1, residual_scale=1.0)
    raise ImplementationError(f"unknown model {model_type}")


def checkpoint_path(seed: int, model_type: str) -> Path:
    return SOURCE_DIR / "checkpoints" / f"{model_type}_seed{seed}.pt"


def load_checkpoint(seed: int, model_type: str):
    path = checkpoint_path(seed, model_type)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != "EXP-022" or int(payload.get("seed")) != seed or payload.get("model_type") != model_type:
        raise ImplementationError(f"checkpoint metadata mismatch for {model_type} seed {seed}")
    model = blank_model(model_type)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def saved_metrics() -> dict[tuple[int, str], dict]:
    rows = [json.loads(line) for line in (SOURCE_DIR / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(int(row["seed"]), row["model_type"]): row for row in rows}


def query_from_input(inputs: torch.Tensor) -> torch.Tensor:
    slot0 = inputs[:, 129, 4]
    slot1 = inputs[:, 129, 5]
    if not torch.equal(slot0 + slot1, torch.ones(inputs.shape[0])) or int(((slot0 != 0) & (slot0 != 1)).sum()) != 0:
        raise ImplementationError("the final query address is not one-hot")
    return torch.where(slot0 == 1, torch.zeros(inputs.shape[0]), torch.ones(inputs.shape[0]))


def assert_symbolic_labels() -> None:
    for bit_a in (-1.0, 1.0):
        for bit_b in (-1.0, 1.0):
            for bit_c in (-1.0, 1.0):
                for flag in (0.0, 1.0):
                    for slot in (0.0, 1.0):
                        for query in (0.0, 1.0):
                            a = torch.tensor([bit_a])
                            b = torch.tensor([bit_b])
                            c = torch.tensor([bit_c])
                            f = torch.tensor([flag])
                            s = torch.tensor([slot])
                            q = torch.tensor([query])
                            slot0, slot1 = current_bits(a, b, c, f, s)
                            expected0 = bit_c if flag == 1.0 and slot == 0.0 else bit_a
                            expected1 = bit_c if flag == 1.0 and slot == 1.0 else bit_b
                            if float(slot0) != expected0 or float(slot1) != expected1:
                                raise ImplementationError("V0 or V1 disagrees with the symbolic rule")
                            target = labels(a, b, c, f, s, q)
                            expected = expected0 if query == 0.0 else expected1
                            if float(target) != expected:
                                raise ImplementationError("the task label is not V_Q")


def assert_split_contract(data: dict) -> None:
    slot0, slot1 = current_bits(data["A"], data["B"], data["C"], data["F"], data["S"])
    query = query_from_input(data["inputs"])
    if not torch.equal(query, data["Q"]):
        raise ImplementationError("the stored query does not match the query input")
    if not torch.equal(data["targets"], torch.where(query == 0, slot0, slot1)):
        raise ImplementationError("the original target is not V_Q")
    addresses = data["inputs"][:, :129, 4:6].clone()
    addresses[:, 0] = 0
    addresses[:, 1] = 0
    rows = torch.arange(data["inputs"].shape[0])
    addresses[rows, data["timing"]] = 0
    if int(addresses.abs().sum()) != 0:
        raise ImplementationError("an address appears before its designated event")


def prequery_state(model, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.recurrent_states(inputs)[:, PREQUERY_INDEX]


def assert_extraction(model, inputs: torch.Tensor) -> None:
    with torch.no_grad():
        full = model.recurrent_states(inputs)
        prefix = model.recurrent_states(inputs[:, : PREQUERY_INDEX + 1])
        if not torch.equal(full[:, PREQUERY_INDEX], prefix[:, -1]):
            raise ImplementationError("pre-query extraction does not match the rollout through index 128")
        zeroed = inputs.clone()
        zeroed[:, 129] = 0
        if not torch.equal(full[:, PREQUERY_INDEX], model.recurrent_states(zeroed)[:, PREQUERY_INDEX]):
            raise ImplementationError("zero-bit negative-control inputs changed the pre-query state")
        cloned = inputs.clone()
        if not torch.equal(full[:, PREQUERY_INDEX], model.recurrent_states(cloned)[:, PREQUERY_INDEX]):
            raise ImplementationError("identical inputs produced different pre-query states")
        other = set_query(inputs, 1.0 - query_from_input(inputs))
        if not torch.equal(full[:, PREQUERY_INDEX], model.recurrent_states(other)[:, PREQUERY_INDEX]):
            raise ImplementationError("the query feature entered the pre-query state")


def features_from_state(state: torch.Tensor) -> np.ndarray:
    values = state.detach().cpu().numpy().astype(np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 32:
        raise ImplementationError("the diagnostic features are not the 32-dimensional state")
    if not np.isfinite(values).all():
        raise ImplementationError("a pre-query state is not finite")
    return values


def slot_breakdown(predictions: np.ndarray, truth: np.ndarray, flag: torch.Tensor, slot: torch.Tensor, which: float) -> dict:
    groups = {
        "no_replacement": flag == 0,
        "updated": (flag == 1) & (slot == which),
        "untouched": (flag == 1) & (slot != which),
    }
    record = {}
    covered = torch.zeros(flag.shape[0], dtype=torch.bool)
    for name, mask in groups.items():
        if bool((covered & mask).any()):
            raise ImplementationError("slot conditions overlap")
        covered |= mask
        chosen = mask.numpy()
        size = int(chosen.sum())
        correct = int((predictions[chosen] == truth[chosen]).sum()) if size else 0
        record[name] = {"size": size, "correct": correct, "accuracy": None if size == 0 else correct / size}
    if not bool(covered.all()):
        raise ImplementationError("slot conditions do not cover the split")
    return record


def score_split(predictions0, predictions1, decisions0, decisions1, data: dict) -> dict:
    slot0, slot1 = current_bits(data["A"], data["B"], data["C"], data["F"], data["S"])
    truth0 = slot0.numpy()
    truth1 = slot1.numpy()
    query = query_from_input(data["inputs"]).numpy()
    routed = np.where(query == 0, predictions0, predictions1)
    routed_decision = np.where(query == 0, decisions0, decisions1)
    targets = data["targets"].numpy()
    if not np.array_equal(targets, np.where(query == 0, truth0, truth1)):
        raise ImplementationError("routing comparison is not against V_Q")
    routed_torch = torch.from_numpy(routed).float()
    groups = score_groups(routed_torch, data["targets"], data["A"], data["B"], data["C"], data["F"], data["S"], data["Q"])
    both = (predictions0 == truth0) & (predictions1 == truth1)
    size = int(truth0.shape[0])
    return {
        "size": size,
        "slot0": {
            "correct": int((predictions0 == truth0).sum()),
            "loss": bce_from_decision(decisions0, binary_labels(slot0)),
            "conditions": slot_breakdown(predictions0, truth0, data["F"], data["S"], 0.0),
        },
        "slot1": {
            "correct": int((predictions1 == truth1).sum()),
            "loss": bce_from_decision(decisions1, binary_labels(slot1)),
            "conditions": slot_breakdown(predictions1, truth1, data["F"], data["S"], 1.0),
        },
        "both_correct": int(both.sum()),
        "routed_correct": int((routed == targets).sum()),
        "routed_loss": bce_from_decision(routed_decision, binary_labels(data["targets"])),
        "subgroups": groups,
    }


def fit_heads(train_features: np.ndarray, slot0: torch.Tensor, slot1: torch.Tensor) -> dict:
    if train_features.shape != (1536, 32) or train_features.dtype != np.float64:
        raise ImplementationError("training features are not the registered 1536 by 32 float64 state")
    mean, scale = fit_scale(train_features)
    scaled = apply_scale(train_features, mean, scale)
    heads = []
    for truth in (slot0, slot1):
        classifier = make_classifier()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(scaled, binary_labels(truth))
        iterations = int(np.max(classifier.n_iter_))
        warning_text = [str(item.message) for item in caught]
        converged = iterations < 2000 and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        heads.append({"classifier": classifier, "iterations": iterations, "converged": converged, "warnings": warning_text})
    return {"mean": mean, "scale": scale, "heads": heads, "scaled_train": scaled}


def predict_heads(fitted: dict, features: np.ndarray):
    scaled = apply_scale(features, fitted["mean"], fitted["scale"])
    decisions = [head["classifier"].decision_function(scaled) for head in fitted["heads"]]
    predictions = [refit_predictions(decision) for decision in decisions]
    return predictions[0], predictions[1], decisions[0], decisions[1]


def save_heads(fitted: dict, seed: int, model_type: str) -> str:
    destination = OUT_DIR / "classifiers" / f"{model_type}_seed{seed}.joblib"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".joblib.partial")
    joblib.dump({
        "heads": [head["classifier"] for head in fitted["heads"]],
        "mean": fitted["mean"],
        "scale": fitted["scale"],
        "seed": seed,
        "model_type": model_type,
    }, temporary)
    temporary.replace(destination)
    loaded = joblib.load(destination)
    replay = loaded["heads"][0].decision_function(apply_scale(fitted["mean"].reshape(1, -1), loaded["mean"], loaded["scale"]))
    original = fitted["heads"][0]["classifier"].decision_function(apply_scale(fitted["mean"].reshape(1, -1), fitted["mean"], fitted["scale"]))
    if not np.array_equal(replay, original):
        raise ImplementationError("reloaded heads did not preserve a decision")
    return file_sha256(destination)


def pair_record(model, fitted, left, right, query_left, query_right, target_left, target_right) -> dict:
    state_left = prequery_state(model, left)
    state_right = prequery_state(model, right)
    pred_l0, pred_l1, _, _ = predict_heads(fitted, features_from_state(state_left))
    pred_r0, pred_r1, _, _ = predict_heads(fitted, features_from_state(state_right))
    q_left = query_left.numpy()
    q_right = query_right.numpy()
    chosen_left = np.where(q_left == 0, pred_l0, pred_l1)
    chosen_right = np.where(q_right == 0, pred_r0, pred_r1)
    both = (chosen_left == target_left.numpy()) & (chosen_right == target_right.numpy())
    return {
        "pairs": int(both.shape[0]),
        "both_correct": int(both.sum()),
        "prequery_state_equal": int(torch.all(state_left == state_right, dim=-1).sum()),
    }


def audit_heads(model, fitted, seed: int) -> dict:
    data = make_two_memory(256, seed, 3)
    inputs, timing = data["inputs"], data["timing"]
    marked_left = set_query(rewrite_candidate(inputs, timing, 1, 1, data["S"]), data["S"])
    marked_right = set_query(rewrite_candidate(inputs, timing, -1, 1, data["S"]), data["S"])
    untouched_left = set_query(rewrite_candidate(inputs, timing, data["C"], 1, torch.ones(256)), torch.zeros(256))
    untouched_right = untouched_left.clone()
    untouched_right[:, 0, 0] = -untouched_left[:, 0, 0]
    obsolete_left = set_query(rewrite_candidate(inputs, timing, torch.ones(256), 1, torch.zeros(256)), torch.zeros(256))
    obsolete_right = obsolete_left.clone()
    obsolete_right[:, 0, 0] = -obsolete_left[:, 0, 0]
    unmarked_left = set_query(rewrite_candidate(inputs, timing, 1, 0, data["S"]), torch.zeros(256))
    unmarked_right = set_query(rewrite_candidate(inputs, timing, -1, 0, data["S"]), torch.zeros(256))
    slot0, slot1 = current_bits(data["A"], data["B"], data["C"], data["F"], data["S"])
    state = prequery_state(model, inputs)
    query_zero = prequery_state(model, set_query(inputs, torch.zeros(256)))
    query_one = prequery_state(model, set_query(inputs, torch.ones(256)))
    if not torch.equal(state, query_zero) or not torch.equal(state, query_one):
        raise ImplementationError("the query-switch audit did not share one pre-query state")
    pred0, pred1, _, _ = predict_heads(fitted, features_from_state(state))
    differing = (slot0 != slot1).numpy()
    if int(differing.sum()) == 0:
        raise ImplementationError("the query-switch audit has no history with two different current values")
    both_all = (pred0 == slot0.numpy()) & (pred1 == slot1.numpy())
    return {
        "marked_new_bit": pair_record(model, fitted, marked_left, marked_right, data["S"], data["S"], torch.ones(256), -torch.ones(256)),
        "untouched_bit": pair_record(model, fitted, untouched_left, untouched_right, torch.zeros(256), torch.zeros(256), untouched_left[:, 0, 0], untouched_right[:, 0, 0]),
        "obsolete_bit": pair_record(model, fitted, obsolete_left, obsolete_right, torch.zeros(256), torch.zeros(256), torch.ones(256), torch.ones(256)),
        "unmarked_new_bit": pair_record(model, fitted, unmarked_left, unmarked_right, torch.zeros(256), torch.zeros(256), data["A"], data["A"]),
        "query_switch": {
            "histories": 256,
            "histories_with_different_current_values": int(differing.sum()),
            "both_correct_on_different_values": int(both_all[differing].sum()),
            "both_correct_on_all_histories": int(both_all.sum()),
        },
    }


def verify_fingerprints() -> None:
    rows = [json.loads(line) for line in (SOURCE_DIR / "fingerprints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for seed in SEEDS:
        matched = [row for row in rows if int(row["seed"]) == seed]
        if len(matched) != 3:
            raise ImplementationError(f"seed {seed} does not have three saved fingerprints")
        digest = file_sha256_tensor(make_two_memory(1536, seed, 0)["inputs"])
        if len({row["train_x"] for row in matched}) != 1 or matched[0]["train_x"] != digest:
            raise ImplementationError(f"training inputs for seed {seed} do not match the saved fingerprint")
        if len({row["init_fingerprint"] for row in matched}) != 1 or len({row["minibatch_fingerprint"] for row in matched}) != 1:
            raise ImplementationError(f"seed {seed} does not have matching initialization and minibatches")


def reproduce_readout(model, seed: int, model_type: str, saved: dict) -> dict:
    test = make_two_memory(512, seed, 2)
    assert_split_contract(test)
    predictions = terminal_predictions(model, test["inputs"])
    correct = int((predictions == test["targets"]).sum())
    if correct != int(saved["test_correct"]):
        raise ImplementationError(f"{model_type} seed {seed} reproduced {correct}, saved {saved['test_correct']}")
    loss, accuracy = evaluate(model, test["inputs"], test["targets"], 256)
    return {"reproduced_correct": correct, "original_loss": loss, "original_accuracy": accuracy, "test": test}


def diagnose(seed: int, model_type: str, saved: dict) -> dict:
    model, payload = load_checkpoint(seed, model_type)
    weight_fingerprint = parameter_fingerprint(model)
    original_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    verified = reproduce_readout(model, seed, model_type, saved)
    test = verified.pop("test")
    assert_extraction(model, test["inputs"][:64])
    splits = {
        "train": make_two_memory(1536, seed, 0),
        "val": make_two_memory(512, seed, 1),
        "test": test,
    }
    for data in splits.values():
        assert_split_contract(data)
    states = {name: features_from_state(prequery_state(model, data["inputs"])) for name, data in splits.items()}
    if states["train"].shape[0] == states["test"].shape[0]:
        raise ImplementationError("training and test feature counts match")
    slot0, slot1 = current_bits(splits["train"]["A"], splits["train"]["B"], splits["train"]["C"], splits["train"]["F"], splits["train"]["S"])
    fitted = fit_heads(states["train"], slot0, slot1)
    if not np.array_equal(fitted["scaled_train"], apply_scale(states["train"], fitted["mean"], fitted["scale"])):
        raise ImplementationError("the two heads did not share one feature scaling")
    record = {
        "experiment_id": "EXP-023",
        "seed": seed,
        "model_type": model_type,
        "source_checkpoint_sha256": file_sha256(checkpoint_path(seed, model_type)),
        "weight_fingerprint": weight_fingerprint,
        "reproduced_correct": verified["reproduced_correct"],
        "original_loss": verified["original_loss"],
        "original_accuracy": verified["original_accuracy"],
        "fit_rows": 1536,
        "iterations": [head["iterations"] for head in fitted["heads"]],
        "converged": all(head["converged"] for head in fitted["heads"]),
        "convergence_warnings": [head["warnings"] for head in fitted["heads"]],
    }
    for name in ("val", "test"):
        truth0, truth1 = current_bits(splits[name]["A"], splits[name]["B"], splits[name]["C"], splits[name]["F"], splits[name]["S"])
        predictions = predict_heads(fitted, states[name])
        if not np.array_equal(predictions[0].shape, truth0.numpy().shape):
            raise ImplementationError("a head prediction does not cover the split")
        record[name] = score_split(*predictions, splits[name])
    test_groups = record["test"]["subgroups"]
    routed_ok = record["test"]["routed_correct"] / 512 >= 0.95 and all(test_groups[name]["accuracy"] >= 0.95 for name in SUBGROUPS)
    record["success"] = bool(record["converged"] and routed_ok)
    record["status"] = "ok" if record["converged"] else "inconclusive"
    record["audits"] = audit_heads(model, fitted, seed)
    record["classifier_sha256"] = save_heads(fitted, seed, model_type)
    if parameter_fingerprint(model) != weight_fingerprint:
        raise ImplementationError("diagnostic fitting changed a recurrent parameter")
    for name, value in model.state_dict().items():
        if not torch.equal(value, original_weights[name]):
            raise ImplementationError("a saved recurrent weight changed")
    if payload["state_dict"].keys() != original_weights.keys():
        raise ImplementationError("checkpoint keys changed")
    return record


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017", "018", "019", "020", "021", "022"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-023 registration")
    assert_symbolic_labels()
    saved = saved_metrics()
    if len(saved) != 30:
        raise ImplementationError("EXP-022 did not save 30 metrics rows")
    manifest = []
    for seed in SEEDS:
        for model_type in MODELS:
            path = checkpoint_path(seed, model_type)
            digest = file_sha256(path)
            row = saved[(seed, model_type)]
            if row.get("checkpoint_sha256") != digest:
                raise ImplementationError(f"checkpoint hash mismatch for {model_type} seed {seed}")
            manifest.append({
                "seed": seed,
                "model_type": model_type,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "saved_test_correct": int(row["test_correct"]),
                "parameter_count": int(row["parameter_count"]),
            })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-023",
        "source": "EXP-022",
        "checkpoints": 30,
        "classifier_fits": 60,
        "feature": "hidden state at index 128",
        "routing": "hard-coded selection of head Q from the final query input",
        "budget_seconds_total": BUDGET,
        "no_recurrent_training": True,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print("registered EXP-023; 30 checkpoints hashed")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if any((OUT_DIR / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-023 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to measure")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-023")
    manifest = json.loads((OUT_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 30:
        raise ImplementationError("the checkpoint manifest does not list 30 models")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\nDiagnostic code revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-023", "started_unix": time.time(), "code_revision": revision,
        "timing_method": "time.perf_counter inside this process", "budget_seconds_total": BUDGET,
    }, indent=2, sort_keys=True), encoding="utf-8")
    assert_symbolic_labels()
    verify_fingerprints()
    saved = saved_metrics()
    print("contracts and fingerprints verified", flush=True)
    records = []
    unstarted = []
    stopped = False
    for seed in SEEDS:
        remaining = BUDGET - (time.perf_counter() - started)
        if remaining <= 1 or stopped:
            unstarted.extend({"seed": seed, "model_type": model_type, "reason": "not started"} for model_type in MODELS)
            continue
        batch_started = time.perf_counter()
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {"event": "start", "seed": seed, "session_elapsed_seconds": round(batch_started - started, 3)})
        try:
            for model_type in MODELS:
                record = diagnose(seed, model_type, saved[(seed, model_type)])
                records.append(record)
                append_jsonl(OUT_DIR / "raw_metrics.jsonl", record)
                print(json.dumps({
                    "seed": seed, "model_type": model_type, "status": record["status"],
                    "routed_correct": record["test"]["routed_correct"], "both_correct": record["test"]["both_correct"],
                }, sort_keys=True), flush=True)
            exit_code = 0
        except (ImplementationError, FileExistsError, RuntimeError) as exc:
            exit_code = 1
            stopped = True
            append_jsonl(OUT_DIR / "raw_metrics.jsonl", {"seed": seed, "status": "stopped", "stop_reason": f"{type(exc).__name__}: {exc}"})
            print(f"stopped during seed {seed}: {exc}", flush=True)
        append_jsonl(OUT_DIR / "runtime_log.jsonl", {
            "event": "finish", "seed": seed, "exit_code": exit_code,
            "runtime_seconds": round(time.perf_counter() - batch_started, 3),
            "session_elapsed_seconds": round(time.perf_counter() - started, 3),
        })
    summary = {"experiment_id": "EXP-023", "elapsed_seconds": round(time.perf_counter() - started, 3), "unstarted": unstarted, "records": records}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    header = "seed,model_type,status,original_correct,slot0_correct,slot1_correct,both_correct,routed_correct,success"
    lines = [header]
    for row in records:
        lines.append(",".join(str(value) for value in (
            row["seed"], row["model_type"], row["status"], row["reproduced_correct"],
            row["test"]["slot0"]["correct"], row["test"]["slot1"]["correct"],
            row["test"]["both_correct"], row["test"]["routed_correct"], row["success"],
        )))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-023")
    print(f"EXP-023 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

"""EXP-021: independent replication of the EXP-020 clean-control comparison.

Registration, before any accuracy:
    python experiments/run_exp021.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp021.py
"""

from __future__ import annotations

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
from run_exp004 import file_sha256  # noqa: E402
from run_exp012 import sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, git_revision  # noqa: E402
from run_exp020 import (  # noqa: E402
    BUDGET,
    ImplementationError,
    condition_name,
    local_rules,
    paired_models,
    preflight,
    train_job,
)

CONFIG_PATH = ROOT / "configs" / "exp021_replication.yaml"
OUT_DIR = ROOT / "results" / "EXP-021"
SEEDS = [601, 607, 613, 617, 619, 631, 641, 643, 647, 653]
BASE_SEED = 8000
BLOCKED = {"summary.json", "runtime_session.json", "raw_metrics.jsonl"}
JOBS = tuple(
    {"seed": seed, "model_type": model_type, "task": task, "attempt": 1}
    for seed in SEEDS
    for task in ("hold", "selective")
    for model_type in ("additive_tanh_rnn", "marker_reset_additive")
)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def query_does_not_reset() -> None:
    _control, reset = paired_models(0)
    hidden = torch.randn(2, 32, requires_grad=True)
    query = torch.zeros(2, 4)
    query[:, 2] = 1.0
    state, _candidate = reset.candidate_step(query, hidden)
    (gradient,) = torch.autograd.grad(state.sum(), hidden)
    if float(gradient.abs().max()) == 0.0:
        raise ImplementationError("the query step reset the carried state")


def find_seed_conflicts() -> list[str]:
    hits = []
    needles = [f'"seed": {seed}' for seed in SEEDS]
    for folder in (ROOT / "results", ROOT / "configs", ROOT / "experiments"):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "EXP-021" in path.parts:
                continue
            if path.suffix.lower() not in {".jsonl", ".json", ".yaml", ".yml", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits.extend(f"{path.relative_to(ROOT).as_posix()}: {needle}" for needle in needles if needle in text)
    return hits


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017", "018", "019", "020"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-021 registration")
    conflicts = find_seed_conflicts()
    if conflicts:
        raise ImplementationError("seed conflict: " + "; ".join(conflicts[:8]))
    local_rules()
    query_does_not_reset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-021",
        "replicates": "EXP-020",
        "seeds": SEEDS,
        "base_seed": BASE_SEED,
        "jobs": len(JOBS),
        "parameter_count": 1249,
        "seed_conflicts": [],
        "budget_seconds_total": BUDGET,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    print("registered EXP-021; seeds unused")


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if "--train-one" in sys.argv:
        payload = json.loads(sys.argv[sys.argv.index("--train-one") + 1])
        train_job(payload["job"], float(payload["seconds_left"]), OUT_DIR, BASE_SEED, "EXP-021")
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if any((OUT_DIR / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-021 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to train")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-021")
    started = time.perf_counter()
    deadline = started + BUDGET
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\nscikit-learn: {sklearn_version()}\n"
        f"Training code revision: {revision}\nParameter count: 1249\nData base seed: {BASE_SEED}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-021", "replicates": "EXP-020", "started_unix": time.time(),
        "code_revision": revision, "base_seed": BASE_SEED, "seeds": SEEDS,
        "timing_method": "time.perf_counter inside this process", "budget_seconds_total": BUDGET,
    }, indent=2, sort_keys=True), encoding="utf-8")
    query_does_not_reset()
    report = preflight(SEEDS[0], BASE_SEED)
    report["base_seed"] = BASE_SEED
    (OUT_DIR / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("preflight passed", flush=True)
    unstarted = []
    stopped = False
    for job in JOBS:
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
    records = [json.loads(line) for line in (OUT_DIR / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT_DIR / "raw_metrics.jsonl").exists() else []
    summary = {"experiment_id": "EXP-021", "elapsed_seconds": round(time.perf_counter() - started, 3), "unstarted": unstarted, "preflight": report, "records": records}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["seed,condition,status,test_accuracy,test_correct,hold_success,selective_success,diagnostic_test_accuracy"]
    for row in records:
        lines.append(",".join(str(row.get(field, "")) for field in ("seed", "condition", "status", "test_accuracy", "test_correct", "hold_success", "selective_success", "diagnostic_test_accuracy")))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = preserved_manifest()
    (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    if after != before:
        raise RuntimeError("historical artifacts changed during EXP-021")
    print(f"EXP-021 finished in {summary['elapsed_seconds']}s; unstarted {len(unstarted)}", flush=True)


if __name__ == "__main__":
    main()

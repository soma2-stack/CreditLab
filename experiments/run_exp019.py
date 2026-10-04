"""EXP-019: hard-wired reset on the existing write marker.

Registration, before any accuracy:
    python experiments/run_exp019.py --write-registration

Run, after that registration is committed:
    python experiments/run_exp019.py

Training starts only if hold-task trajectories match within the
preregistered tolerance. Otherwise the mismatch is recorded and the
run stops before any training score.
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

from creditlab.models import MarkerResetAdditiveRNN, ResidualTanhRNN  # noqa: E402
from creditlab.reproducibility import seed_everything  # noqa: E402
from run_exp004 import file_sha256, parameter_fingerprint  # noqa: E402
from run_exp012 import sklearn_version  # noqa: E402
from run_exp016 import artifact_manifest, git_revision  # noqa: E402
from run_exp017 import SEEDS, make_latent, task_views  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "exp019_marker_reset.yaml"
OUT_DIR = ROOT / "results" / "EXP-019"
TOLERANCE = 1e-5
BLOCKED = {"summary.json", "runtime_session.json", "investigation.json"}


class ImplementationError(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def paired_models(seed: int) -> tuple[ResidualTanhRNN, MarkerResetAdditiveRNN]:
    seed_everything(seed)
    control = ResidualTanhRNN(3, 32, 1, residual_scale=1.0)
    reset = MarkerResetAdditiveRNN(3, 32, 1, residual_scale=1.0)
    reset.load_state_dict(control.state_dict())
    return control, reset


def hold_match_report(num_samples: int = 64) -> dict:
    rows = []
    worst_state = 0.0
    worst_logit = 0.0
    for seed in SEEDS:
        latent = make_latent(num_samples, seed, 0)
        hold_x, _hold_y, _selective_x, _selective_y = task_views(latent)
        control, reset = paired_models(seed)
        control_states = control.recurrent_states(hold_x)
        reset_states = reset.recurrent_states(hold_x)
        state_gap = float((control_states - reset_states).detach().abs().max())
        control_logits = control.readout(control_states[:, -1, :])
        reset_logits = reset.readout(reset_states[:, -1, :])
        logit_gap = float((control_logits - reset_logits).detach().abs().max())
        write = hold_x[:, 1:129, 1]
        exact_binary = (write == 0) | (write == 1)
        competitor = latent["competitor_mask"]
        mismatch = (control_states - reset_states).abs().amax(dim=-1) > 0
        first = []
        for row in range(num_samples):
            hits = torch.nonzero(mismatch[row], as_tuple=False)
            first.append(int(hits[0]) if len(hits) else -1)
        positive = [step for step in first if step >= 0]
        rows.append({
            "seed": seed,
            "max_abs_state_difference": state_gap,
            "max_abs_logit_difference": logit_gap,
            "earliest_mismatch_timestep": None if not positive else min(positive),
            "interior_write_channel_not_exactly_binary_fraction": float((~exact_binary).float().mean()),
            "interior_competitor_fraction": float(competitor.float().mean()),
        })
        worst_state = max(worst_state, state_gap)
        worst_logit = max(worst_logit, logit_gap)
    matched = worst_state <= TOLERANCE and worst_logit <= TOLERANCE
    return {
        "tolerance_max_abs": TOLERANCE,
        "matched": matched,
        "worst_state_difference": worst_state,
        "worst_logit_difference": worst_logit,
        "per_seed": rows,
        "reason": None if matched else (
            "Hold trajectories do not match. The EXP-017 generator puts competitor "
            "write markers and distractor noise on the same channel the reset reads, "
            "so the reset is not confined to the initial event."
        ),
    }


def local_equation_checks() -> None:
    control, reset = paired_models(0)
    if parameter_fingerprint(control) != parameter_fingerprint(reset):
        raise ImplementationError("initial parameters do not match")
    if sum(p.numel() for p in control.parameters()) != sum(p.numel() for p in reset.parameters()):
        raise ImplementationError("parameter counts differ")
    hidden = torch.randn(4, 32)
    unmarked = torch.tensor([[0.2, 0.0, 0.0]]).expand(4, -1).contiguous()
    control_state, _ = control.candidate_step(unmarked, hidden)
    reset_state, _ = reset.candidate_step(unmarked, hidden)
    if not torch.equal(control_state, reset_state):
        raise ImplementationError("unmarked step does not match the additive control")
    marked = unmarked.clone()
    marked[:, 1] = 1
    zero_hidden = torch.zeros_like(hidden)
    control_zero, _ = control.candidate_step(marked, zero_hidden)
    reset_zero, _ = reset.candidate_step(marked, zero_hidden)
    if not torch.equal(control_zero, reset_zero):
        raise ImplementationError("a marked step from zero does not match the additive control")
    varied = hidden.clone().requires_grad_(True)
    state, _candidate = reset.candidate_step(marked, varied)
    (gradient,) = torch.autograd.grad(state.sum(), varied)
    if float(gradient.abs().max()) != 0.0:
        raise ImplementationError("a marked reset did not block the previous state")
    query = torch.zeros(4, 3)
    query[:, 2] = 1
    query_hidden = hidden.detach().clone().requires_grad_(True)
    query_state, _ = reset.candidate_step(query, query_hidden)
    (query_gradient,) = torch.autograd.grad(query_state.sum(), query_hidden)
    if float(query_gradient.abs().max()) == 0.0:
        raise ImplementationError("the query marker triggered a reset")


def preserved_manifest() -> dict[str, str]:
    records = artifact_manifest()
    for name in ("016", "017", "018"):
        folder = ROOT / "results" / f"EXP-{name}"
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            records[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return records


def write_registration() -> None:
    destination = OUT_DIR / "continuation_registration.yaml"
    if destination.exists():
        raise FileExistsError("refusing to overwrite the EXP-019 registration")
    local_equation_checks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump({
        "experiment_id": "EXP-019",
        "seeds": SEEDS,
        "marker": "x_t channel 1 with no threshold",
        "hold_match_tolerance": TOLERANCE,
        "training_starts_only_if_hold_matches": True,
        "budget_seconds_total": 900,
    }, sort_keys=False), encoding="utf-8")
    (OUT_DIR / "historical_hashes_before.json").write_text(
        json.dumps(preserved_manifest(), indent=2, sort_keys=True), encoding="utf-8",
    )
    print("registered EXP-019")


def main() -> None:
    if "--write-registration" in sys.argv:
        write_registration()
        return
    if platform.python_version() != "3.11.9" or torch.__version__ != "2.13.0+cpu" or sklearn_version() != "1.9.1":
        raise RuntimeError("environment does not match the preregistered versions")
    if any((OUT_DIR / name).exists() for name in BLOCKED):
        raise RuntimeError("EXP-019 outputs already exist. Automatic resume is disabled.")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; refusing to run")
    before = json.loads((OUT_DIR / "historical_hashes_before.json").read_text(encoding="utf-8"))
    if preserved_manifest() != before:
        raise RuntimeError("preexisting artifacts changed before EXP-019")
    started = time.perf_counter()
    revision = git_revision()
    (OUT_DIR / "environment.txt").write_text(
        f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\n"
        f"scikit-learn: {sklearn_version()}\nCode revision: {revision}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime_session.json").write_text(json.dumps({
        "experiment_id": "EXP-019",
        "started_unix": time.time(),
        "code_revision": revision,
        "timing_method": "time.perf_counter inside this process",
        "training_started": False,
    }, indent=2, sort_keys=True), encoding="utf-8")
    local_equation_checks()
    report = hold_match_report()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["training_started"] = False
    (OUT_DIR / "investigation.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["matched"]:
        summary = {
            "experiment_id": "EXP-019",
            "status": "stopped_before_training",
            "reason": report["reason"],
            "elapsed_seconds": report["elapsed_seconds"],
            "training_runs": 0,
            "unstarted": "all 40 conditions",
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        after = preserved_manifest()
        (OUT_DIR / "historical_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
        if after != before:
            raise RuntimeError("historical artifacts changed during EXP-019")
        print("EXP-019 stopped before training; hold trajectories do not match", flush=True)
        return
    raise ImplementationError("hold trajectories matched; the training section was not reached because this run stops for review")


if __name__ == "__main__":
    main()

"""Checks for the EXP-016 completion manifest. These do not train."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp016_completion import COMPLETION, JOBS, ORIGINAL, REUSED, key_of  # noqa: E402


def test_completion_is_fourteen_jobs_and_twenty_six_reused():
    assert len(REUSED) == 26
    assert len(JOBS) == 14
    reused = {key_of(item["seed"], item["model_type"], item["noise"]) for item in REUSED}
    jobs = {key_of(item["seed"], item["model_type"], item["noise"]) for item in JOBS}
    assert reused.isdisjoint(jobs)
    restart = [job for job in JOBS if job["attempt"] == 2]
    assert restart == [{
        "seed": 439,
        "model_type": "vanilla_tanh_rnn",
        "noise": 1.0,
        "attempt": 2,
        "role": "restart_from_initialization",
    }]
    assert COMPLETION == ORIGINAL / "completion"
    assert "completion" not in ORIGINAL.name

"""Checks for the delay-128 continuation manifest. These do not train."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp009_delay128_completion import (  # noqa: E402
    PER_RUN_SECONDS,
    RESTART,
    REUSED,
    TOTAL_SECONDS,
    planned_jobs,
)


def test_manifest_has_thirty_eight_jobs_and_two_reused_conditions():
    jobs = planned_jobs()
    assert len(jobs) == 38
    assert REUSED == ((101, "full"), (101, "readout_only"))
    assert all((job["seed"], job["regime"]) not in REUSED for job in jobs)
    identities = {(job["seed"], job["regime"]) for job in jobs}
    assert len(identities) == 38


def test_only_the_unfinished_bias_run_is_restarted():
    jobs = planned_jobs()
    restarts = [job for job in jobs if job["attempt_id"] == "authorized_restart_1"]
    assert len(restarts) == 1
    assert (restarts[0]["seed"], restarts[0]["regime"]) == RESTART
    assert restarts[0]["prior_outcome"] == "unknown_no_saved_result"
    assert restarts[0]["starts_from"] == "initialization"
    assert all(job["attempt_id"] == "continuation_1" for job in jobs if job is not restarts[0])


def test_allowance_is_separate_from_the_original_cap():
    assert TOTAL_SECONDS == 600
    assert PER_RUN_SECONDS == 300

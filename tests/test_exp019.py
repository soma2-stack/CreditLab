"""Checks for the EXP-019 reset rule. These do not train."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_exp019 import hold_match_report, load_config, local_equation_checks  # noqa: E402


def test_reset_rule_and_registration_gate():
    cfg = load_config()
    assert cfg["marker"]["threshold"] == "none"
    assert cfg["hold_match_gate"]["tolerance_max_abs"] == 1e-5
    assert cfg["hold_match_gate"]["if_not_matched"].startswith("record the mismatch")
    local_equation_checks()


def test_hold_task_inputs_do_not_keep_a_clean_marker():
    report = hold_match_report(num_samples=64)
    assert report["matched"] is False
    assert report["worst_state_difference"] > 1e-5
    assert all(row["interior_write_channel_not_exactly_binary_fraction"] > 0.5 for row in report["per_seed"])
    assert all(row["interior_competitor_fraction"] > 0.2 for row in report["per_seed"])

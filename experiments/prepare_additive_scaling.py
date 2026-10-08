"""Preparation-only matrix validator. Does not import torch or train models."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

WIDTHS = (16, 32, 64, 128)
DELAYS = (128, 256, 512, 1024)
SEEDS = (307, 311, 313, 317, 331, 337, 347, 349, 353, 359)
ARCHITECTURES = ("additive", "vanilla", "gru")
TRAIN_DELAY = 128
SUCCESS_THRESHOLD = 0.95

@dataclass(frozen=True)
class ProposedRun:
    architecture: str
    width: int
    seed: int
    train_delay: int
    evaluation_delays: tuple[int, ...]

def build_matrix() -> list[ProposedRun]:
    return [
        ProposedRun(a, w, s, TRAIN_DELAY, DELAYS)
        for a, w, s in product(ARCHITECTURES, WIDTHS, SEEDS)
    ]

def validate_matrix(runs: list[ProposedRun]) -> None:
    assert len(runs) == len(ARCHITECTURES) * len(WIDTHS) * len(SEEDS)
    assert len({(r.architecture, r.width, r.seed) for r in runs}) == len(runs)
    assert all(r.train_delay == 128 and r.evaluation_delays == DELAYS for r in runs)
    assert all(r.width > 0 and r.seed >= 0 for r in runs)

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proposed study matrix; never train.")
    parser.add_argument("--output", type=Path, help="Optional local JSON manifest path")
    args = parser.parse_args()
    runs = build_matrix()
    validate_matrix(runs)
    manifest = {
        "status": "PROPOSAL_ONLY_NOT_EXECUTED",
        "training_enabled": False,
        "train_delay": TRAIN_DELAY,
        "success_threshold": SUCCESS_THRESHOLD,
        "runs": [asdict(r) for r in runs],
        "notes": [
            "No training or evaluation performed.",
            "Do not execute until source entrypoints, generator, checkpoints and budgets are reviewed.",
            "Keep EXP-001 through EXP-023 immutable.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(runs)} proposed training configurations; 0 executed.")

if __name__ == "__main__":
    main()

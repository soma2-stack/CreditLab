"""Read-only preflight for the proposed scaling study.

No training, checkpoint loading, GPU allocation or result writes occur here.
Run from repository root: python scripts/preflight_additive_scaling.py
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.prepare_additive_scaling import build_matrix, validate_matrix

HISTORIC = (ROOT / "configs/exp011_clipping_horizon.yaml",
            ROOT / "configs/exp015_length_generalization.yaml")
REQUIRED = (
    "full_history_clip5", "hidden_size: 32", "updates_per_run: 400",
    "no_test_accuracy_during_training: true",
)

def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def preflight() -> dict:
    matrix = build_matrix()
    validate_matrix(matrix)
    missing = [str(p.relative_to(ROOT)) for p in HISTORIC if not p.is_file()]
    if missing:
        raise RuntimeError("missing historical configurations: " + ", ".join(missing))
    config11, config15 = (p.read_text(encoding="utf-8") for p in HISTORIC)
    for needle in REQUIRED:
        if needle not in config11:
            raise RuntimeError("EXP-011 mismatch: " + needle)
    for needle in ("reference: 128", "primary: 256", "secondary: 512",
                   "run_mode: HARD_V2", "evaluation_split_index: 4"):
        if needle not in config15:
            raise RuntimeError("EXP-015 mismatch: " + needle)
    # Separation is checked on the proposed deterministic index ranges,
    # not a claim that an unknown generator implementation is leak-free.
    train, val, test = range(0,1536), range(1536,2048), range(2048,2560)
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError("overlapping planned split indices")
    return {
        "status": "PREFLIGHT_ONLY_NO_TRAINING",
        "matrix_configurations": len(matrix),
        "historical_config_sha256": {
            str(p.relative_to(ROOT)): fingerprint(p) for p in HISTORIC
        },
        "split_policy": "disjoint indices; actual generator parity still requires audit",
        "execution_allowed": False,
    }

if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2))

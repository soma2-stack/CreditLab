"""No-training checks for the proposed scaling matrix."""
from experiments.prepare_additive_scaling import (
    ARCHITECTURES, DELAYS, SEEDS, WIDTHS, build_matrix, validate_matrix
)

def test_matrix_is_complete_and_unique():
    runs = build_matrix()
    validate_matrix(runs)
    assert len(runs) == 120
    assert len({(r.architecture, r.width, r.seed) for r in runs}) == 120

def test_controls_and_evaluation_lengths():
    runs = build_matrix()
    assert set(r.architecture for r in runs) == set(ARCHITECTURES)
    assert set(r.width for r in runs) == set(WIDTHS)
    assert set(r.seed for r in runs) == set(SEEDS)
    assert all(r.evaluation_delays == DELAYS for r in runs)
    assert all(r.train_delay == 128 for r in runs)

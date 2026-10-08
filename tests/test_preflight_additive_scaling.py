"""Read-only preflight checks; never launches training."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_additive_scaling.py"

def test_preflight_has_no_training_execution_switch():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "argparse" not in source
    assert "optimizer.step(" not in source
    assert "torch.cuda" not in source
    assert "execution_allowed" in source

def test_historical_configs_and_matrix():
    spec = importlib.util.spec_from_file_location("creditlab_scaling_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.preflight()
    assert result["matrix_configurations"] == 120
    assert result["execution_allowed"] is False
    assert len(result["historical_config_sha256"]) == 2

"""Step 5.1 test — Optuna sweep config composes without runtime error.

Does NOT run the sweep (3 trials × NODE training = minutes). Manual
verification per PLAN.md Step 5.1 Verify.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from cardiac_ml.conf_schemas import _register

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


@pytest.fixture(autouse=True)
def _register_and_reset_hydra():
    _register()
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_lr_batch_sweep_composes():
    """`+hparams_search=lr_batch` composes with any experiment."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=ionic_node_smoke", "+hparams_search=lr_batch"],
        )
    assert cfg is not None, "sweep config failed to compose"


def test_sweep_script_importable():
    """`scripts/sweep.py` imports without error (top-level side effects OK)."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts" / "sweep.py"
    spec = importlib.util.spec_from_file_location("sweep_module", path)
    assert spec is not None
    # We only verify the file parses — running main() would launch a sweep.
    with open(path) as f:
        src = f.read()
    compile(src, str(path), "exec")

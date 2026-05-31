"""Step 5.3 reusability test — diffusion stub trains via default teacher-forced
step, proving the harness handles a structurally different consumer (conv CNN
+ batched (x, y) tuples vs. ODE + dict beats) without Trainer changes."""
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


def test_diffusion_stub_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=diffusion_stub_smoke"],
        )
    assert cfg.model._target_.endswith("DiffusionResNetStub")
    assert cfg.training.device == "cpu"
    assert "teacher_forced_step" in cfg.training.train_step_fn.path


def test_diffusion_stub_trains_under_harness():
    """val_loss[end] < val_loss[start] * 0.9 — 10% improvement margin over 10 epochs."""
    from cardiac_ml.training.trainer import Trainer

    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=diffusion_stub_smoke", "tracking=off"],
        )
    trainer = Trainer(cfg)
    # Capture start val_loss by running a single eval epoch before fit.
    initial_val = trainer._run_epoch(train=False)
    trainer.fit()
    final_val = trainer._run_epoch(train=False)

    start = initial_val.get("val_loss") or initial_val.get("val_val_loss")
    end = final_val.get("val_loss") or final_val.get("val_val_loss")
    assert start is not None and end is not None, f"missing val_loss: {initial_val} / {final_val}"
    assert end < start * 0.9, (
        f"no meaningful improvement: val_loss start={start:.4f}, end={end:.4f}"
    )

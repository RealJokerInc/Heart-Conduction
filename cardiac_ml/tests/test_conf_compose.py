"""Hydra composition + target-importability tests for the conf/ tree.

Round-2 H-2 + Round-3 MED-6: beyond compose(), we walk every `_target_`
string and `path` field (under `_target_: hydra.utils.get_method`) and
try `importlib.import_module` on the parent module. Deferred targets
listed in _deferred_targets.py are skipped.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from cardiac_ml.conf_schemas import _register
from cardiac_ml.tests._deferred_targets import DEFERRED

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


@pytest.fixture(autouse=True)
def _register_and_reset_hydra():
    _register()
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(overrides=None):
    overrides = overrides or []
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


def _walk_targets(cfg) -> Iterator[str]:
    """Yield every _target_ string and get_method path in the composed cfg."""
    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        items = cfg.items() if hasattr(cfg, "items") else []
        tgt = None
        if hasattr(cfg, "get"):
            tgt = cfg.get("_target_")
        if tgt:
            yield str(tgt)
        # If this is a hydra.utils.get_method spec, also yield `path`.
        if tgt == "hydra.utils.get_method":
            p = cfg.get("path") if hasattr(cfg, "get") else None
            if p:
                yield str(p)
        for _, v in items:
            yield from _walk_targets(v)
    elif isinstance(cfg, list) or (hasattr(cfg, "__iter__") and not isinstance(cfg, str)):
        try:
            for v in cfg:
                yield from _walk_targets(v)
        except TypeError:
            pass


def _is_deferred(path: str) -> bool:
    """True if path's parent module (or any ancestor) is in DEFERRED."""
    parent = path.rsplit(".", 1)[0]
    while parent:
        if parent in DEFERRED:
            return True
        if "." not in parent:
            return False
        parent = parent.rsplit(".", 1)[0]
    return False


def test_default_config_resolves():
    """Top-level config.yaml composes without error."""
    cfg = _compose()
    assert cfg is not None
    assert cfg.model._target_ == "surrogate.model.node.IonicNODE"


def test_experiment_override():
    """experiment=ionic_node_t1 composes and swaps model/data/training/tracking.

    Note: `experiment` IS in conf/config.yaml defaults (as null), so the
    override syntax is `experiment=<name>`, NOT `+experiment=<name>`.
    """
    cfg = _compose(overrides=["experiment=ionic_node_t1"])
    assert "node_train_step" in cfg.training.train_step_fn.path
    assert cfg.training.phase_name == "A1"  # placeholder, verified in Step 4.0


def test_all_targets_importable():
    """Every _target_ string's parent module imports, except deferred ones."""
    cfg = _compose()
    failures = []
    for target in _walk_targets(cfg):
        if _is_deferred(target):
            continue
        parent = target.rsplit(".", 1)[0]
        try:
            importlib.import_module(parent)
        except ImportError as e:
            failures.append(f"{target}: {e}")
    assert not failures, (
        "Un-importable targets in composed config:\n  " + "\n  ".join(failures)
    )


def test_tracking_off_disables():
    """tracking=off sets enabled=false."""
    cfg = _compose(overrides=["tracking=off"])
    assert cfg.tracking.enabled is False

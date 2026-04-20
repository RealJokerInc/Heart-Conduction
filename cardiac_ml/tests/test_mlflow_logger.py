"""Tests for cardiac_ml/training/mlflow_logger.py."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import torch
from omegaconf import OmegaConf

from cardiac_ml.training.mlflow_logger import (
    MLflowLoggerCallback,
    _flatten,
    _is_scalar,
)


# ----------------------------- _flatten ---------------------------------


def test_flatten_simple_dict():
    assert _flatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_nested_dict():
    out = _flatten({"a": {"b": 1, "c": 2}, "d": 3})
    assert out == {"a.b": 1, "a.c": 2, "d": 3}


def test_flatten_list_gets_indexed_keys():
    # MLflow param keys forbid `[` / `]`, so list indices flatten as `.N`.
    out = _flatten({"xs": [10, 20, 30]})
    assert out == {"xs.0": 10, "xs.1": 20, "xs.2": 30}


def test_flatten_stringifies_non_primitives():
    out = _flatten({"obj": object()})
    assert isinstance(out["obj"], str)


# ----------------------------- _is_scalar -------------------------------


def test_is_scalar_python_numbers():
    assert _is_scalar(1)
    assert _is_scalar(1.5)
    assert not _is_scalar(True)  # bool explicitly excluded
    assert not _is_scalar("str")


def test_is_scalar_zero_dim_tensor():
    assert _is_scalar(torch.tensor(1.0))


def test_is_scalar_multi_dim_tensor_rejected():
    assert not _is_scalar(torch.tensor([1.0, 2.0]))


# ----------------------------- MLflowLoggerCallback ---------------------


def _tiny_cfg_and_trainer(tracking_uri: str):
    """Stub a minimal DictConfig + trainer with model.state_dict()."""
    cfg = OmegaConf.create({
        "training": {"epochs": 2, "lr": 1e-3, "batch_size": 4},
        "tracking": {"enabled": True, "tracking_uri": tracking_uri,
                     "experiment_name": "cardiac_ml_tests"},
        "model": {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4},
        "experiment": "test_experiment",  # string form; hits isinstance(exp, str) branch
    })
    model = torch.nn.Linear(4, 4).double()
    trainer = SimpleNamespace(cfg=cfg, model=model)
    return cfg, trainer


def test_run_creation_sets_tags_and_logs_metrics(tmp_path, monkeypatch):
    """End-to-end: fit 2 epochs, verify tags + metrics landed in MLflow."""
    uri = f"file:{tmp_path}"
    cfg, trainer = _tiny_cfg_and_trainer(uri)
    cb = MLflowLoggerCallback(
        experiment_name="cardiac_ml_tests", tracking_uri=uri
    )
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, 0, {"train_loss": 1.5, "val_loss": 1.2})
    cb.on_epoch_end(trainer, 1, {"train_loss": 1.0, "val_loss": 0.8})
    cb.on_fit_end(trainer)

    # Query MLflow directly.
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(uri)
    exp = client.get_experiment_by_name("cardiac_ml_tests")
    assert exp is not None
    runs = client.search_runs(exp.experiment_id)
    assert len(runs) == 1
    run = runs[0]
    # Tags
    assert run.data.tags.get("git.sha")  # set, non-empty
    assert "git.dirty" in run.data.tags
    assert run.data.tags["python.version"].startswith("3.")
    assert run.data.tags["torch.version"]
    # Metrics have 2 history entries each
    history = client.get_metric_history(run.info.run_id, "train_loss")
    assert len(history) == 2


def test_null_branch_no_writes_via_base_callback(tmp_path):
    """Sanity: NullLogger (from callbacks.py) inherits log_artifact as no-op.
    Verifies the routing invariant — tracking=off means no mlflow side-effects."""
    from cardiac_ml.training.callbacks import NullLogger
    cb = NullLogger()
    cb.log_artifact(str(tmp_path / "nope.pt"))
    cb.log_figure(None, "nope.png")
    # No mlflow run, no file at tmp_path
    assert list(tmp_path.iterdir()) == []


def test_derive_run_name_fallback_without_hydra_context(tmp_path):
    """When cfg has no `experiment` field and Hydra context isn't set up,
    falls back to 'cardiac_ml_run'."""
    uri = f"file:{tmp_path}"
    cfg = OmegaConf.create({"training": {}, "tracking": {"enabled": True}})
    trainer = SimpleNamespace(cfg=cfg)
    cb = MLflowLoggerCallback(tracking_uri=uri)
    name = cb._derive_run_name(trainer)
    # Format: "<name>_<git_sha>"; when git is present, <git_sha> is short hex.
    parts = name.rsplit("_", 1)
    assert parts[0] == "cardiac_ml_run"


def test_is_scalar_numpy_float_accepted():
    """Round-4 MED-4 non-regression: numpy scalars pass _is_scalar since
    numpy.floating subclasses float."""
    import numpy as np
    assert _is_scalar(np.float64(1.5))
    assert _is_scalar(np.int32(7))

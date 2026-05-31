"""MLflow logger callback + helper utilities.

Single-file MLflow entry point. Trainer never imports mlflow — escape-
hatch calls (`trainer.log_artifact`, `trainer.log_figure`) dispatch through
the Callback-base proxy methods, which this class overrides.

Round-3 fixes:
- C-4: `_derive_run_name` falls back through cfg.experiment.name →
       HydraConfig runtime choice → "cardiac_ml_run" when experiment
       YAML doesn't declare a `name` field.
- LOW-3: `_flatten` and `_is_scalar` helpers are explicit (not referenced
         without definition).
- MED-2 (indirect): the override pattern means tracking=off callers never
        execute mlflow calls — NullLogger inherits base no-ops.
"""
from __future__ import annotations

import sys
from typing import Any

import mlflow
import torch
from omegaconf import OmegaConf

from cardiac_ml.training.callbacks import Callback
from cardiac_ml.utils.git import git_dirty, git_sha


def _flatten(d: Any, prefix: str = "", sep: str = ".") -> dict:
    """Flatten nested dict/list config for mlflow.log_params.

    OmegaConf.to_container returns dicts + lists + primitives. Tuples and
    sets should never appear; if they do, the non-container branch
    str-ifies them.
    """
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(_flatten(v, key, sep))
    elif isinstance(d, list):
        # MLflow param key regex permits [a-zA-Z0-9_\-./: ]; use `.N` not `[N]`.
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{sep}{i}" if prefix else str(i), sep))
    else:
        out[prefix] = d if (d is None or isinstance(d, (str, int, float, bool))) else str(d)
    return out


def _is_scalar(v: Any) -> bool:
    """True if v is a 0-d tensor or any numeric scalar (excluding bool).

    Round-4 MED-4: numpy.float64 subclasses float but numpy.int32 does NOT
    subclass int on 64-bit Linux. Explicitly check np.number to cover both.
    """
    import numpy as np
    if torch.is_tensor(v):
        return v.numel() == 1
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, np.number):
        return True
    return False


class MLflowLoggerCallback(Callback):
    """File-backed MLflow logger. One run per fit(), tagged with git state."""

    def __init__(self, experiment_name: str = "default", tracking_uri: str = "./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: Any = None

    def _derive_run_name(self, trainer: Any) -> str:
        """Round-3 C-4 fix: cfg.experiment may be a bare group without a
        .name field. Fall back through: cfg.experiment.name → Hydra
        runtime choice → default string.
        """
        exp = trainer.cfg.get("experiment") if hasattr(trainer.cfg, "get") else None
        name = None
        if isinstance(exp, str):
            name = exp
        elif exp is not None and hasattr(exp, "get"):
            name = exp.get("name")
        if not name:
            try:
                from hydra.core.hydra_config import HydraConfig
                choices = HydraConfig.get().runtime.choices
                name = choices.get("experiment") or "cardiac_ml_run"
            except Exception:
                name = "cardiac_ml_run"
        return f"{name}_{git_sha()}"

    # Lifecycle hooks --------------------------------------------------------
    def on_fit_start(self, trainer: Any) -> None:
        self._run = mlflow.start_run(run_name=self._derive_run_name(trainer))
        mlflow.set_tag("git.sha", git_sha())
        mlflow.set_tag("git.dirty", str(git_dirty()))
        mlflow.set_tag("python.version", sys.version.split()[0])
        mlflow.set_tag("torch.version", torch.__version__)
        params = _flatten(OmegaConf.to_container(trainer.cfg, resolve=True))
        # mlflow.log_params truncates silently past 500 keys and rejects
        # non-scalar values — drop Nones and stringify everything else.
        cleaned = {k: ("" if v is None else str(v)) for k, v in params.items()}
        # MLflow 2.x has a per-call limit of 100 params; chunk if needed.
        items = list(cleaned.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        scalar_metrics = {k: float(v) for k, v in metrics.items() if _is_scalar(v)}
        if scalar_metrics:
            mlflow.log_metrics(scalar_metrics, step=epoch)

    def on_fit_end(self, trainer: Any) -> None:
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    # Proxy overrides --------------------------------------------------------
    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_figure(self, fig: Any, name: str) -> None:
        mlflow.log_figure(fig, name)

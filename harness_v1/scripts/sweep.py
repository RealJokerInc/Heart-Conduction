"""Optuna sweep entry point — identical to scripts/train.py but primed
for `--multirun` with the Hydra Optuna sweeper plugin.

Usage:
    python scripts/sweep.py --multirun +hparams_search=lr_batch \
        experiment=ionic_node_smoke

The sweeper's objective is the `fit()` return value. `Trainer.fit()` returns
None, so the last-logged metric matching the sweep's `direction` / `study_name`
is picked up via MLflow (each trial is a separate MLflow run). For Optuna's
TPE sampler to build a search surface, we emit the sweep's target metric as
`return_value` from `main()` — Hydra captures scalar returns as trial outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SURROGATE = _ROOT / "Surrogate"
if _SURROGATE.is_dir() and str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))

import hydra
from omegaconf import DictConfig

from cardiac_ml import Trainer
from cardiac_ml.conf_schemas import _register

_register()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Return the best val_loss observed — Hydra Optuna sweeper uses this as
    the trial objective.
    """
    trainer = Trainer(cfg)
    trainer.fit()

    # Pull the best val_loss from MLflow — the sweep direction is `minimize`.
    import mlflow
    mlflow.set_tracking_uri(cfg.tracking.get("tracking_uri", "./mlruns"))
    client = mlflow.tracking.MlflowClient()
    # Grab the most recent run in the default experiment.
    exps = client.search_experiments()
    if not exps:
        return float("inf")
    runs = client.search_runs(
        [e.experiment_id for e in exps],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return float("inf")
    hist = client.get_metric_history(runs[0].info.run_id, "val_loss")
    if not hist:
        return float("inf")
    return min(h.value for h in hist)


if __name__ == "__main__":
    main()

# cardiac_ml

Project-wide ML training harness for the Heart-Conduction repo. Shared across the ionic surrogate, diffusion ResNet, bidomain cross-skip, and Optimizer V1 BayesOpt work.

**Status: scaffolding only.** No code yet. Direction settled 2026-04-16. See `REQUIREMENTS.md` for the full goal spec and `Research/Active/cardiac_ml_harness/` for research context (KNOWLEDGE, IDEALOG, decisions).

## What it is

Four pieces, one package:

| Layer | Tool | Role |
|-------|------|------|
| Config | Hydra | Compose `model` / `data` / `training` / `experiment` groups. Instantiate model code via `_target_`. |
| Tracking | MLflow (file-backed) | Per-epoch metrics, `log_artifact(state_dict)`, git SHA / dirty tag. |
| HPO | Optuna (`hydra-optuna-sweeper`) | `--multirun` sweeps, median pruning. |
| Interpretability | SHAP | Post-hoc `scripts/analyze.py`, not a training callback. |

Orchestrated by a single `cardiac_ml.Trainer` that takes a `train_step_fn` injected via Hydra `_target_` — a pure function `(trainer, batch) -> dict`, not a subclass method. Custom per-step metrics flow through the function's return dict (auto-logged to MLflow). Rare cases (figures, artifacts) use `trainer.log_artifact(path)` / `trainer.log_figure(fig, name)` escape hatches. User code never imports MLflow.

## What it is NOT

- Not a model library. Model code lives in `Surrogate/surrogate/model/`, future `Bidomain/...`, etc. This package references them via Hydra `_target_`.
- Not a PyTorch Lightning wrapper. Raw PyTorch Trainer — Lightning's `LightningModule` assumptions clash with NODE adjoint.
- Not a replacement for model-specific training logic. Things like `Surrogate/surrogate/training/node_rollout.py` stay near the model; they *use* the harness, they don't move into it.

## Target layout (not yet implemented)

```
cardiac_ml/
├── README.md              # this file
├── REQUIREMENTS.md        # goals, non-goals, interface contracts
├── __init__.py
├── training/
│   ├── trainer.py         # Trainer — takes train_step_fn via Hydra _target_
│   ├── callbacks.py       # EarlyStopping, ModelCheckpoint, MLflowLogger core set
│   └── mlflow_logger.py   # log_artifact(state_dict), git SHA tag, per-epoch metrics
├── analysis/
│   └── shap_utils.py      # post-hoc SHAP for trained checkpoints
└── utils/
    └── ...                # seed, device, git-tag helpers

conf/                      # Hydra config tree (project root, NOT inside cardiac_ml/)
├── config.yaml
├── model/
├── data/
├── training/
├── optimizer/
└── experiment/

scripts/
├── train.py               # @hydra.main — uses cardiac_ml.Trainer
├── sweep.py               # Optuna --multirun entry point
└── analyze.py             # SHAP over a trained checkpoint

mlruns/                    # gitignored — MLflow file-backed store
outputs/                   # gitignored — Hydra per-run working dirs
archive/runs_legacy/       # old Surrogate/runs/ moved here at cutover
```

## Installation (future)

Conda env is `heart-conduction` (Python 3.11, PyTorch 2.10 CUDA). Extra deps to add when implementation starts:

```
pip install hydra-core mlflow optuna hydra-optuna-sweeper shap
```

(Versions to be pinned once template survey picks a template baseline.)

## Pilot task

Ionic NODE training from `Surrogate/surrogate/training/node_rollout.py`. Parity target: reproduce the multi-BCL T1 val=0.008 result from surrogate_pipeline Session 25. See `REQUIREMENTS.md § Success Criteria`.

## References

- **Goals + interface contracts**: `REQUIREMENTS.md`
- **Research context**: `Research/Active/cardiac_ml_harness/` — README, KNOWLEDGE.md, IDEALOG.md
- **Origin decisions**: `Research/Active/surrogate_pipeline/IDEALOG.md` Session 26 (2026-04-16)
- **Pilot reference impl**: `Surrogate/surrogate/training/node_rollout.py`
- **First template to survey**: [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

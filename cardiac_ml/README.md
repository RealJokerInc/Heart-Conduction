# cardiac_ml

Project-wide ML training harness for the Heart-Conduction repo. Shared across the ionic surrogate, diffusion ResNet, bidomain cross-skip, and Optimizer V1 BayesOpt work.

**Status (2026-04-20): implemented, consumer-ready.** NODE pilot reproduced the Session-25 oracle (best val_loss=0.00835 ≤ 0.0088 threshold). Reusability proven via a diffusion-stub CNN trained on the default `teacher_forced_step` — zero Trainer changes. 80 tests pass. See `REQUIREMENTS.md` for the full spec and `Research/Active/cardiac_ml_harness/` for research context (KNOWLEDGE, IDEALOG, plans/).

## What it is

Four pieces, one package:

| Layer | Tool | Role |
|-------|------|------|
| Config | Hydra 1.3.2 | Compose `model` / `data` / `training` / `tracking` / `experiment` groups. Instantiate model code via `_target_`. |
| Tracking | MLflow 2.22.4 (file-backed) | Per-epoch metrics, `log_artifact(state_dict)`, git SHA / dirty tag. |
| HPO | Optuna 2.10.1 via `hydra-optuna-sweeper 1.2.0` | `--multirun` sweeps. (Pruner unavailable at this plugin version — see KNOWLEDGE.md.) |
| Interpretability | SHAP 0.51.0 | Post-hoc `scripts/analyze.py`, not a training callback. `KernelExplainer`-only (`DeepExplainer` incompatible with `torchdiffeq`). |

Orchestrated by a single `cardiac_ml.Trainer` that takes a `train_step_fn` injected via Hydra `_target_` — a pure function `(trainer, batch) -> dict`, not a subclass method. Custom per-step metrics flow through the function's return dict (auto-logged to MLflow). Rare cases (figures, artifacts) use `trainer.log_artifact(path)` / `trainer.log_figure(fig, name)` escape hatches. User code never imports MLflow.

Protocol keys the Trainer recognizes on a step's return dict:

- `loss` — required. Scalar tensor.
- `_backward_done: bool` — optional. If True, Trainer skips its default `loss.backward()` (adjoint case). Asserts `loss.requires_grad or loss.grad_fn is not None`.
- `_on_after_backward: Callable[[], None]` — optional. Post-backward cleanup (e.g., NODE's `clear_v_trajectory()`). Fires on both train AND val paths; wrapped in try/except that zero-grads on failure.
- Any other key — surfaced as an auto-logged MLflow metric (prefixed `train_` or `val_` by the Trainer).

## What it is NOT

- Not a model library. Model code lives in `Surrogate/surrogate/model/`, future `Bidomain/...`, etc. This package references them via Hydra `_target_`.
- Not a PyTorch Lightning wrapper. Raw PyTorch Trainer — Lightning's `LightningModule` assumptions clash with NODE adjoint.
- Not a replacement for model-specific training logic. Things like `Surrogate/surrogate/training/node_rollout.py` stay near the model; they *use* the harness, they don't move into it.
- Does NOT import `mlflow` inside `Trainer`. All MLflow calls route through `MLflowLoggerCallback` (or the no-op `NullLogger` when `tracking=off`).

## Layout

```
cardiac_ml/
├── README.md              # this file
├── REQUIREMENTS.md        # goals, non-goals, interface contracts
├── conf_schemas.py        # structured ConfigStore registration
├── __init__.py            # PEP 562 lazy Trainer import
├── training/
│   ├── trainer.py         # Trainer — takes train_step_fn via Hydra _target_
│   ├── callbacks.py       # EarlyStopping, ModelCheckpoint, GradNormMonitor, LRSchedulerStep, NullLogger
│   ├── mlflow_logger.py   # log_artifact(state_dict), git SHA tag, per-epoch metrics
│   └── default_steps.py   # teacher_forced_step / _val_step (default train_step_fn)
├── data/
│   └── multi_bcl_loader.py   # oracle-parity T1 beat loader (mirrors run_multi_bcl.py:27-81)
├── model/
│   └── ionic_node_factory.py # IonicNODE builder with optional stage1 warm-start
├── analysis/
│   └── shap_utils.py      # post-hoc KernelExplainer over trained NODE checkpoints
├── utils/
│   ├── git.py             # git_sha / git_dirty helpers
│   └── seed.py            # seed_everything
└── tests/                 # 80 tests — import smoke, compose, Trainer, MLflow, SHAP, diffusion stub

conf/                      # Hydra config tree (project root, NOT inside cardiac_ml/)
├── config.yaml
├── model/                 # ionic_node, synthetic_mlp, diffusion_resnet_stub
├── data/                  # t1_multi_bcl, synthetic_linear, synthetic_2d
├── training/              # node, teacher_forced
├── optimizer/             # adam
├── tracking/              # default, off
├── hparams_search/        # lr_batch
└── experiment/            # ionic_node_t1, ionic_node_smoke, synthetic_smoke, diffusion_stub_smoke

scripts/
├── train.py               # @hydra.main — uses cardiac_ml.Trainer
├── sweep.py               # Optuna --multirun entry point
└── analyze.py             # SHAP over a trained checkpoint (MLflow run_id input)

mlruns/                    # gitignored — MLflow file-backed store
outputs/                   # gitignored — Hydra per-run working dirs
archive/runs_legacy/       # old Surrogate/runs/ moved here at cutover (git-tracked via allowlist)
```

## Installation

Conda env is `heart-conduction` (Python 3.11, PyTorch 2.10 CUDA). Harness deps (pinned):

```
pip install hydra-core==1.3.2 hydra-optuna-sweeper==1.2.0 optuna==2.10.1 \
            mlflow==2.22.4 shap==0.51.0
```

Note: `hydra-optuna-sweeper 1.2.0` pins `optuna<3.0`. Upgrading requires a paired `optuna 3.x + hydra-optuna-sweeper >=1.3` move.

## Usage

```bash
# train the ionic NODE pilot end-to-end
python scripts/train.py experiment=ionic_node_t1

# warm-start from the Session-25 oracle checkpoint
WARM_START_CKPT=archive/runs_legacy/multi_bcl_002/best.pt \
  python scripts/train.py experiment=ionic_node_t1

# 2-epoch smoke (1 train BCL, 1 val BCL, ~4 min on Blackwell)
python scripts/train.py experiment=ionic_node_smoke

# 3-trial LR sweep
python scripts/sweep.py --multirun +hparams_search=lr_batch experiment=ionic_node_smoke

# post-hoc SHAP plot for a trained run
python scripts/analyze.py experiment=ionic_node_t1 run_id=<mlflow_run_id> output_dir=./shap_out

# disable MLflow entirely (debug iteration)
python scripts/train.py experiment=ionic_node_t1 tracking=off
```

## Writing a new `train_step_fn`

Any consumer wiring a new model pattern:

1. Write a pure function `(trainer, batch) -> dict` near the model code (e.g., `Surrogate/surrogate/training/my_step.py`). Use the protocol keys documented above.
2. Add a Hydra training YAML (`conf/training/my_model.yaml`) pointing `train_step_fn._target_: hydra.utils.get_method` at `path: my.pkg.my_step_fn`.
3. Add an experiment YAML (`conf/experiment/my_experiment.yaml`) composing `model`, `data`, `training`, `tracking`.
4. `python scripts/train.py experiment=my_experiment`.

Zero Trainer changes. Zero Python changes outside the step function itself.

## Pilot task (DONE)

Ionic NODE training from `Surrogate/surrogate/training/node_rollout.py`, adapted via `Surrogate/surrogate/training/node_step.py`. Parity vs. surrogate_pipeline Session 25: best val_loss=0.00835 < 0.0088 threshold (= 1.05 × oracle best 0.00838).

## References

- **Goals + interface contracts**: `REQUIREMENTS.md` (patched 2026-04-20 with production NODE shape at §7.2).
- **Research context**: `Research/Active/cardiac_ml_harness/` — README, KNOWLEDGE.md, IDEALOG.md, `plans/2026-04-20_*.md`.
- **Origin decisions**: `Research/Active/surrogate_pipeline/IDEALOG.md` Session 26 (2026-04-16).
- **Pilot reference impl**: `Surrogate/surrogate/training/node_rollout.py` + adapter `Surrogate/surrogate/training/node_step.py`.
- **Oracle parity log**: `archive/runs_legacy/multi_bcl_002/log.jsonl` (8-epoch Session 25 run).

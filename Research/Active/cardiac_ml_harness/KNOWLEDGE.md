# Cardiac ML Harness — Knowledge File

> Running synthesis of what's settled, what's open, and what we've learned about the harness.
> Promoted to `Research/Knowledge/` when the question is complete.

## Current Understanding

The harness doesn't exist yet. We have a settled direction from the surrogate_pipeline line of work (Session 26, 2026-04-16) and concrete lessons from five prior training strategies inside the ionic surrogate. The question isn't "what should the harness do" — that's settled. The questions are:

1. **What do we borrow from existing open-source templates** (ashleve/lightning-hydra-template and peers), and what do we reject?
2. **What's the minimum Trainer surface** that covers NODE adjoint + teacher-forced diffusion + BayesOpt objective evaluation without forcing any of them through an uncomfortable abstraction? (`train_step_fn` form settled 2026-04-19 — covers (a) and (b); BayesOpt likely needs a separate `evaluate()` entry.)
3. **What callbacks belong in `cardiac_ml/` core vs in model-specific training modules?** NODE NFE tracking is obviously model-specific; MLflow logging is obviously core; in between is fuzzy.

## Settled Decisions

From surrogate_pipeline IDEALOG Session 26 (2026-04-16):

| Decision | Rationale |
|----------|-----------|
| **Config framework: Hydra** | Composition + `_target_` instantiation + `--multirun` for Optuna + CLI overrides. Plain YAML+dataclass would force factory code. |
| **Experiment tracker: MLflow, file-backed** (`./mlruns/`) | No server dependency. Multi-machine collaboration isn't a current requirement. |
| **Artifact logging: `log_artifact(state_dict)`, NOT `log_model`** | `log_model` pickles the full nn.Module. Fragile for custom classes (B-spline KANs, torchdiffeq wrappers, future FNO layers). State dict is the portable minimum. |
| **HPO: Optuna via hydra-optuna-sweeper** | Native `--multirun` integration. No custom sweep code. |
| **Interpretability: SHAP as overlay, not core** | `scripts/analyze.py` consumes trained checkpoints. Not a training-time callback. |
| **Trainer: single flexible class** | Not a hierarchy of task-specific subclasses. One place to look when training behavior is wrong. |
| **Package scope: project root `cardiac_ml/`** | Reusable across ionic surrogate, diffusion ResNet, bidomain cross-skip, Optimizer V1. NOT `Surrogate/cardiac_ml/`. |
| **Model code location: unchanged** | `Surrogate/surrogate/model/` etc. stay put. Hydra references them via `_target_: path.to.Class`. |
| **Model-specific training logic: near the model** | e.g. `Surrogate/surrogate/training/node_step.py` for NODE adjoint specifics. Exposes a pure `train_step_fn(trainer, batch) -> dict` pulled in by the Trainer via Hydra `_target_`. |
| **Clean start, no compatibility layer** | Archive old `Surrogate/runs/` → `archive/runs_legacy/`. No wrapper shim around old training code. |
| **MLflow auto-tags** | git SHA + dirty flag on every run. Reproducibility minimum. |

Added 2026-04-19:

| Decision | Rationale |
|----------|-----------|
| **Override mechanism: pure function via Hydra `_target_`, not subclass** (2026-04-19) | `train_step_fn(trainer, batch) -> dict` passed into Trainer at construction. Keeps model-specific training near the model without spawning a class hierarchy. Named replacement for the subclass form forbidden in REQUIREMENTS §3. |
| **Custom metric channel: return dict** (2026-04-19) | Every non-`loss` key returned by `train_step_fn` is auto-logged as an MLflow metric. User code imports zero MLflow. |
| **Rare-case escape hatches: `trainer.log_artifact` / `trainer.log_figure`** (2026-04-19) | For figures, histograms, per-run files the return dict can't cover. Function signatures stay pure; these are called from inside the function only when needed. |

## Open Questions

1. **Template selection**: Which 2–3 existing templates do we borrow from? `ashleve/lightning-hydra-template` is the obvious first candidate but uses PyTorch Lightning, which may clash with NODE adjoint. Need survey.
2. **Lightning vs. raw PyTorch for the Trainer**: Lightning handles callbacks / loggers / devices nicely, but its `LightningModule` abstraction may not play well with `odeint_adjoint` and per-batch variable trajectory lengths. Current lean: raw PyTorch. Need a proof point before committing.
3. **`train_step_fn` signature for BayesOpt**: Pure-function form `(trainer, batch) -> dict` covers (a) ionic NODE adjoint and (b) teacher-forced diffusion cleanly — see REQUIREMENTS §7. Settled 2026-04-19. (c) BayesOpt objective — no grads, no minibatches, no epochs — still doesn't fit `train_step_fn`. Likely routes through a separate `Trainer.evaluate(config) -> dict` entry point. Resolves during blueprint.
4. **Callback surface**: `EarlyStopping`, `ModelCheckpoint`, `MLflowLogger` are core. `NFEMonitor`, `AdjointGradNorm`, `LossLandscape` are model-specific. Where's the line?
5. **Hydra structured configs (dataclass) vs free-form YAML**: structured gives validation + type-check at resolution time; free-form is faster to iterate on. Likely structured for the most-used entries (model, training), free-form for experiment composition.
6. **Optuna pruning vs NODE instability**: NODE val loss can spike before settling; aggressive pruners will kill viable trials. Median pruner with long warmup (≥20 epochs) is the obvious default, but needs empirical check against ionic NODE training curves.
7. **SHAP input for the ionic surrogate**: the model takes `(z, V)` and returns `dz/dt`. SHAP over what — `V` only (simplest), or `(z, V)` joint (more informative but higher-dim)?
8. **Float64 default**: the project uses float64. Does this break any MLflow / Optuna / SHAP assumption? (Should be fine — they're framework-agnostic — but needs confirming.)

## Connections

- **Parent context**: [surrogate_pipeline](../surrogate_pipeline/) — origin of the requirement. Session 26 in its IDEALOG.md has the settled decisions. Session 25 has the multi-BCL T1 val=0.008 result that the pilot must reproduce.
- **First consumer (pilot)**: ionic NODE training, currently in `Surrogate/surrogate/training/node_rollout.py`.
- **Second consumer (planned)**: whichever of diffusion ResNet or Optimizer V1 BayesOpt lands first.
- **Related engines**: none modified. The harness doesn't touch Bidomain / Monodomain / LBM source.
- **Related pipelines**: Optimizer V1 (`Optimizer/`) — second consumer via BayesOpt objective wrapping.

## Risks

- **HIGH — Trainer abstraction too tight**: if `_train_step` is over-specified, NODE adjoint or BayesOpt objective evaluation will need ugly overrides. Mitigation: write the NODE `_train_step` first; extract the abstraction only when the second consumer forces it.
- **MEDIUM — MLflow artifact bloat**: naive per-epoch `log_artifact(state_dict)` fills disk fast. Mitigation: log every N epochs + always best + always last.
- **MEDIUM — `_target_` errors surface late**: constructor typos / dtype mismatches don't show until Hydra resolves. Mitigation: structured configs for the most-used entries.
- **LOW — Optuna pruning fights NODE early instability**: median pruner with long warmup, or disable pruning for NODE phase.

## Design Maxims (to keep honest)

- **Ship the pilot before generalizing.** The NODE training must work end-to-end under the harness before the abstraction is extracted. Generalization without a second consumer is speculation.
- **One Trainer, not a hierarchy.** If `_train_step` isn't general enough, the signature changes — we don't spawn `NODETrainer` / `ResNetTrainer`.
- **Project-wide from day one.** `cardiac_ml/` at repo root, never `Surrogate/cardiac_ml/`. Hard commitment.
- **MLflow artifact, not model.** State dict only. Never call `mlflow.pytorch.log_model`.

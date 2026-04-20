# cardiac_ml — Requirements

Detailed goal spec for the project-wide ML training harness. Drives the template survey and the `/blueprint` step that follows.

**Status**: Draft 2026-04-19. Settled decisions are recorded here; open questions are flagged with `[OPEN]` and belong in `Research/Active/cardiac_ml_harness/KNOWLEDGE.md § Open Questions`.

---

## 1. Purpose

Replace the ad-hoc training scripts under `Surrogate/runs/` with a single shared harness used across every learned component in the repo. The scripts under `Surrogate/runs/` were written one-per-strategy during the ionic surrogate's five-strategy evolution (rollout curriculum → dt curriculum → TBPTT → warm restarts → NODE). Each reinvented config loading, logging, checkpointing, and sweep plumbing. Historical runs are uncomparable. The next three training workloads (diffusion ResNet, bidomain cross-skip, Optimizer V1 BayesOpt) cannot be allowed to repeat that pattern.

---

## 2. Goals (prioritized)

In priority order — earlier goals block later goals.

1. **Config reproducibility.** Every run is defined by a composed Hydra config. Re-running the same config produces the same result up to framework non-determinism.
2. **Experiment tracking.** Every run logs per-epoch metrics, final + best + periodic checkpoints, and git SHA + dirty flag to a file-backed MLflow store. No run is un-tagged.
3. **One Trainer.** A single `cardiac_ml.Trainer` class handles ionic NODE (adjoint), teacher-forced diffusion, and BayesOpt objective evaluation via a `train_step_fn` — a pure function `(trainer, batch) -> dict`, injected via Hydra `_target_: hydra.utils.get_method`. Not a hierarchy of subclasses.
4. **Model-code independence.** The harness never imports model code directly. Model classes are instantiated by Hydra `_target_` only. Adding a new model = new config file, zero harness changes.
5. **HPO via `--multirun`.** Optuna sweeps run through Hydra's sweeper plugin, not custom search code.
6. **Post-hoc interpretability.** SHAP runs against trained checkpoints from a separate script, not inside the training loop.
7. **Clean cutover.** `Surrogate/runs/` is archived intact at the cutover commit; no compatibility shim wraps the old code.

---

## 3. Non-Goals

Explicit — we will NOT do these, even if convenient.

- **No `mlflow.pytorch.log_model`.** Always `log_artifact(state_dict)`. Custom classes (B-spline KAN, torchdiffeq wrappers, future FNO layers) pickle poorly.
- **No PyTorch Lightning.** `LightningModule` assumptions clash with `odeint_adjoint` and variable-length landmark losses. Raw PyTorch Trainer.
- **No per-task Trainer subclass** (`NODETrainer`, `ResNetTrainer`, `BayesOptTrainer`). One Trainer, swappable `train_step_fn` passed via config. Override is a pure function, not a subclass method.
- **No remote MLflow server (yet).** File-backed `./mlruns/`. Revisit only if multi-machine collaboration becomes a requirement.
- **No W&B / Neptune / Aim.** Decision settled on MLflow. Revisit listed under Future Work in the research question README.
- **No plain YAML + dataclass configs.** Hydra only.
- **No wrapper around existing `Surrogate/runs/` scripts.** Clean start.
- **No model-code changes during harness construction.** `Surrogate/surrogate/model/` is frozen input.

---

## 4. Functional Requirements

### 4.1 Config (Hydra)

- **FR-C1** Config tree lives at project root in `conf/`, not inside `cardiac_ml/`.
- **FR-C2** Top-level `conf/config.yaml` composes groups: `model`, `data`, `training`, `optimizer`, `experiment`.
- **FR-C3** Model classes are instantiated via `_target_: path.to.Class` — no factory code in the harness.
- **FR-C4** CLI overrides work: `python scripts/train.py training.lr=1e-3 model.hidden=64`.
- **FR-C5** Structured configs (dataclasses) enforce types on the most-used entries (`model`, `training`). Free-form YAML for experiment composition.  **[OPEN]** exact split.
- **FR-C6** Per-run Hydra working directory under `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}/` — gitignored.

### 4.2 Tracking (MLflow)

- **FR-M1** File-backed MLflow store at `./mlruns/` (gitignored).
- **FR-M2** Every run auto-tags: git SHA, git dirty flag, Hydra config hash, python version, torch version, CUDA version.
- **FR-M3** Per-epoch metrics logged: train loss, val loss, + any model-specific metrics the `_train_step` emits.
- **FR-M4** Checkpoint policy: **always** log `best.pt` and `last.pt` via `log_artifact`. Periodic `epoch_{N}.pt` every N epochs (N configurable, default 50).
- **FR-M5** State dict only — never `log_model`.
- **FR-M6** Run name = Hydra experiment name + short SHA.

### 4.3 Trainer

- **FR-T1** Single class `cardiac_ml.training.Trainer`.
- **FR-T2** Override mechanism is a pure function `train_step_fn(trainer, batch) -> dict[str, Tensor]`, injected into the Trainer at construction time via Hydra `_target_: hydra.utils.get_method`. Not a subclass method. (Settled 2026-04-19.)
- **FR-T3** Separate `val_step_fn(trainer, batch) -> dict[str, Tensor]` with the same return shape. Both injected via config; either may be omitted to use the default.
- **FR-T4** Default `train_step_fn` / `val_step_fn` handle the common case: model forward, MSE loss, `loss.backward()`, optimizer step, grad-norm logging. Suitable out-of-the-box for teacher-forced diffusion.
- **FR-T5** NODE training supplies a `train_step_fn` that calls `odeint_adjoint` and computes landmark loss. Lives in `Surrogate/surrogate/training/node_step.py`; referenced from `conf/training/node.yaml` via `_target_: hydra.utils.get_method`.
- **FR-T6** BayesOpt objective evaluation likely does NOT fit the `train_step_fn` shape (no grads, no minibatches, no epochs). Likely routes through a separate `Trainer.evaluate(config) -> dict` entry point. **[OPEN-2]** exact shape — deferred to blueprint.
- **FR-T7** Supports float64 by default (project convention). Device selection from Hydra config, default `cuda`.
- **FR-T8** Return-dict contract: the function's return dict must contain `"loss"` (scalar tensor). Every other key is auto-logged as an MLflow metric at that step, without the function calling any MLflow API. This is the primary channel for custom per-step metrics. (Settled 2026-04-19.)
- **FR-T9** Rare-case escape hatches: Trainer exposes `trainer.log_artifact(path: str)` and `trainer.log_figure(fig, name: str)` for cases the return dict can't cover (per-run files, matplotlib figures, histograms). Function signature stays pure; these are called from inside the function when needed. (Settled 2026-04-19.)

### 4.4 Callbacks

- **FR-CB1** Core callbacks ship with the harness: `EarlyStopping`, `ModelCheckpoint`, `MLflowLogger`, `LRSchedulerStep`, `GradNormMonitor`.
- **FR-CB2** Model-specific callbacks live with the model (e.g., `Surrogate/surrogate/training/callbacks.py` for `NFEMonitor`, `AdjointGradNormMonitor`) and are loaded via Hydra `_target_`.
- **FR-CB3** Callback hooks: `on_fit_start`, `on_epoch_start/end`, `on_train_batch_start/end`, `on_val_batch_start/end`, `on_fit_end`.

### 4.5 HPO (Optuna)

- **FR-H1** Sweeps run via `scripts/sweep.py` using `hydra-optuna-sweeper` plugin.
- **FR-H2** Search space defined in `conf/hparams_search/*.yaml`, not in code.
- **FR-H3** Pruning: median pruner with ≥20-epoch warmup, configurable per sweep.
- **FR-H4** Each sweep trial is a full MLflow run, parented to a sweep-level tag.

### 4.6 Interpretability (SHAP)

- **FR-S1** `scripts/analyze.py --run-id <mlflow_run>` loads a state dict artifact, runs SHAP, writes plots to the Hydra output dir.
- **FR-S2** Not a training-time callback.
- **FR-S3** Input / output scope for the ionic surrogate: **[OPEN]** V-only vs (z, V) joint.

### 4.7 Scripts

- **FR-SC1** `scripts/train.py` — `@hydra.main`, instantiates model / data / trainer via `_target_`, calls `trainer.fit()`.
- **FR-SC2** `scripts/sweep.py` — Hydra `--multirun` with Optuna sweeper.
- **FR-SC3** `scripts/analyze.py` — post-hoc SHAP entry point.
- **FR-SC4** All three scripts work on the ionic NODE pilot before the package ships.

---

## 5. Non-Functional Requirements

- **NFR-1 Reproducibility.** Given the same config + same git SHA + fixed seed, two runs produce loss curves within framework non-determinism.
- **NFR-2 No silent data races.** Checkpointing and metric logging happen on the main process only.
- **NFR-3 No disk bloat.** Default artifact policy (best + last + every 50) keeps per-run artifact size under 1 GB for typical ionic surrogate checkpoints (~10 KB state dict × few hundred logs).
- **NFR-4 Fast iteration.** Startup time (Hydra resolve + MLflow init + model instantiate) under 5 s on the Blackwell GPU.
- **NFR-5 Env isolation.** All new deps installed into the existing `heart-conduction` conda env. No new env.
- **NFR-6 float64 default.** All tensor construction respects project-wide `torch.float64` default.

---

## 6. Dependencies

New Python deps (to be added to `heart-conduction` env):

| Package | Purpose | Version |
|---------|---------|---------|
| `hydra-core` | Config composition | pinned after template survey |
| `mlflow` | Tracking | file-backed mode |
| `optuna` | HPO | — |
| `hydra-optuna-sweeper` | Hydra ↔ Optuna bridge | — |
| `shap` | Interpretability | — |

Existing deps that must NOT be disturbed: `torch`, `torch_dct`, `scipy`, `torchdiffeq` (NODE adjoint).

---

## 7. Interface Contracts — `train_step_fn`

The Trainer's generality hinges on `train_step_fn` covering three consumer patterns. Each is a pure function `(trainer, batch) -> dict`, loaded by Hydra via `_target_: hydra.utils.get_method`. Sketch per consumer:

### 7.1 Teacher-forced (diffusion ResNet, default)

```python
def teacher_forced_step(trainer, batch):
    x_in, x_out = batch
    pred = trainer.model(x_in)
    loss = F.mse_loss(pred, x_out)
    return {"loss": loss}
```

### 7.2 NODE adjoint (ionic surrogate)

Production shape (matches `Surrogate/surrogate/training/node_step.py` landed in Step 4.1):

```python
# Surrogate/surrogate/training/node_step.py
def node_train_step(trainer, batch):
    """Pure adapter — delegates to the frozen node_rollout() oracle.
    phase_name is a REQUIRED cfg field (A1|A2|A3|A4|ionic_state|conc_only|
    B1|B2|B3|B4|ionic_state_and_conductance)."""
    result = node_rollout(
        node=trainer.model, segment=batch,
        phase_name=trainer.cfg.training.phase_name,
        method="dopri5", rtol=1e-3, atol=1e-3, adjoint=False,
    )
    loss = result["loss"]
    def _clear(): trainer.model.clear_v_trajectory()
    return {"loss": loss, "_on_after_backward": _clear,
            **{k: v.detach() for k, v in result.items() if k != "loss"}}
```

Key corrections vs. the original sketch (post-Session 25 reality check):
- ODE config is `dopri5` + `rtol=atol=1e-3`, NOT `dopri8 / rtol=1e-5` (Session 23 found the stricter tolerances diverged under warm-start).
- `adjoint=False` — oracle runs `odeint` for stability; adjoint memory savings weren't worth the accuracy loss at scale.
- `node.nfe` is a latent AttributeError bug in legacy `train_node.py` — IonicNODE has no `.nfe` attribute at pin `8f191f77`. Drop it.
- Per-component scaffold metrics (`ionic_state_mse`, `conc_mse`, `conductance_mse`) pass through as detached entries — auto-logged per the return-dict convention.
- `_on_after_backward` hook calls `clear_v_trajectory()` after backward completes (adjoint re-calls forward during backward; clearing earlier corrupts the V interpolation).

Wired into Hydra via:

```yaml
# conf/training/node.yaml
train_step_fn:
  _target_: hydra.utils.get_method
  path: surrogate.training.node_step.node_train_step
```

### 7.3 BayesOpt objective (Optimizer V1)

**[OPEN-2]** May not fit `train_step_fn` at all — no grads, no minibatches, no epochs. Likely routes through a separate `cardiac_ml.Trainer.evaluate(config) -> dict` entry point rather than through `train_step_fn`. To be resolved during blueprint.

### Contract summary

- **Signature**: `train_step_fn(trainer, batch) -> dict[str, Tensor | float]`. Pure function, not a method.
- **Input**: `batch` — whatever the Hydra-configured dataloader yields. No fixed shape.
- **Output**: dict with required key `"loss"` (scalar tensor). Every other key becomes a per-step MLflow metric automatically. No MLflow calls needed in user code.
- **Backward pass**: called by Trainer on `output["loss"]`. If the function needs non-standard backward (e.g. adjoint), it calls `loss.backward()` itself and returns an already-grad'd loss tensor. **[OPEN-6]** clarify signal — flag in return dict?
- **Escape hatches**: `trainer.log_artifact(path)` and `trainer.log_figure(fig, name)` for rare cases the return dict can't cover (per-run files, matplotlib figures, histograms). Return dict stays the primary channel.

---

## 8. Success Criteria

Mapped 1:1 with research question README completion criteria.

| # | Criterion | How measured |
|---|-----------|--------------|
| 1 | Template survey complete | `results/template_survey.md` exists with 2–3 templates reviewed and a decision note |
| 2 | Package skeleton exists | `cardiac_ml/training/{trainer,callbacks,mlflow_logger}.py` + `analysis/shap_utils.py` importable |
| 3 | Hydra tree exists | `conf/config.yaml` + 5 group dirs; `python scripts/train.py --help` shows composition |
| 4 | `train.py` end-to-end | `python scripts/train.py experiment=ionic_node_t1` produces MLflow run with metrics + `best.pt` artifact |
| 5 | Git tagging | MLflow run has `git.sha` and `git.dirty` tags |
| 6 | Ionic NODE pilot parity | val loss reaches 0.008 on multi-BCL T1 (parity with surrogate_pipeline Session 25) |
| 7 | Optuna sweep | `python scripts/sweep.py --multirun experiment=ionic_node_t1 training.lr=choice(1e-3,5e-4,1e-4)` runs 3 trials, all tracked in MLflow |
| 8 | SHAP plot | `python scripts/analyze.py run_id=<id>` produces PNG |
| 9 | Runs archived | `Surrogate/runs/` moved to `archive/runs_legacy/`, `.gitignore` includes `mlruns/` + `outputs/` |
| 10 | Second-consumer parity | One of: (a) diffusion ResNet stub trains end-to-end, (b) Optimizer V1 BayesOpt wrapper evaluates one objective |

---

## 9. Decisions

### 9.1 Settled

- **[SETTLED 2026-04-19]** Override mechanism. `train_step_fn` is a pure function `(trainer, batch) -> dict`, injected via Hydra `_target_: hydra.utils.get_method`. NOT a subclass method. Model-specific training code (e.g. NODE adjoint) lives as a pure function near the model (e.g. `Surrogate/surrogate/training/node_step.py`) and is pulled in by config. Subclass hierarchies were already forbidden in §3; this names the mechanism that replaces them.
- **[SETTLED 2026-04-19]** Custom-metric channel. Return dict is the primary channel — every key except `"loss"` is auto-logged as an MLflow metric. For rare cases (figures, artifacts, histograms) Trainer exposes `trainer.log_artifact(path)` and `trainer.log_figure(fig, name)`. Functions stay pure; MLflow API is never in user code.

### 9.2 Open

Blocked items — need the template survey and/or a short design pass before `/blueprint`.

- **[OPEN-1]** Template choice. `ashleve/lightning-hydra-template` is the first candidate but uses Lightning (which we've ruled out). Survey needs to find a non-Lightning counterpart or accept partial reuse (config tree and Hydra patterns, but not the Trainer).
- **[OPEN-2]** `train_step_fn` vs separate `Trainer.evaluate()` for BayesOpt. See §7.3.
- **[OPEN-3]** Structured config boundary. Which groups get dataclass schemas, which stay free-form YAML?
- **[OPEN-4]** SHAP input scope for ionic surrogate — V-only (simplest, cheapest) vs (z, V) joint (more informative).
- **[OPEN-5]** Pruner tuning for NODE — median pruner + ≥20-epoch warmup is the default; verify against actual ionic NODE val-loss curves before committing.
- **[OPEN-6]** How to surface `odeint_adjoint`'s loss.backward() call through the Trainer loop — let the function call backward itself and signal via return dict?
- **[OPEN-7]** Dep version pins — deferred until template survey picks a baseline.
- **[OPEN-8]** Tracking-disable shim for quick-debug iteration (`cfg.tracking.enabled=false` → null logger, zero code-path divergence). Recommended but not yet decided.

---

## 10. Out-of-Scope Reminders

- No modifications to `Surrogate/surrogate/model/` during harness construction.
- No diffusion ResNet implementation — harness ships first, diffusion uses it.
- No cross-skip ResNet implementation — same.
- No engine changes (Bidomain V1, Monodomain V5.4, LBM V1 untouched).
- No build-fix / fewer-permission-prompts / other skill work bundled in.

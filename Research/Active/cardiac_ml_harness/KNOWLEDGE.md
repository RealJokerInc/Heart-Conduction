# Cardiac ML Harness — Knowledge File

> Running synthesis of what's settled, what's open, and what we've learned about the harness.
> Promoted to `Research/Knowledge/` when the question is complete.

## Current Understanding (post-implementation, 2026-04-20)

Harness exists and is consumer-ready. NODE pilot reproduced the Session 25 oracle (best val_loss=0.00835 ≤ 0.0088 threshold). Reusability proven via a second-consumer (diffusion-stub CNN) trained via the default `teacher_forced_step` with zero Trainer changes. Cutover of legacy `Surrogate/runs/` to `archive/runs_legacy/` is done. 80 tests pass.

**What the harness is:**
- `cardiac_ml/` package at repo root — Trainer, callbacks, MLflow logger, SHAP utils, Hydra factories.
- `conf/` Hydra tree — model/data/training/optimizer/tracking/experiment/hparams_search groups.
- `scripts/train.py` (entry), `scripts/sweep.py` (Optuna `--multirun`), `scripts/analyze.py` (post-hoc SHAP).

**What changed in the codebase:**
- New: `Surrogate/surrogate/training/node_step.py` — pure adapter wrapping the frozen `node_rollout()` oracle.
- New: `cardiac_ml/data/multi_bcl_loader.py` — replicates `Surrogate/run_multi_bcl.py:27-81` for oracle-parity beat extraction.
- New: `cardiac_ml/model/ionic_node_factory.py` — Hydra factory with optional stage1 warm-start.
- Moved: `Surrogate/runs/` → `archive/runs_legacy/` (tracked subset via `git mv`, untracked via `git add`; gitignore allowlist `!archive/runs_legacy/` keeps them version-controlled).

**What stayed frozen (M-6):**
- `Surrogate/surrogate/model/` — IonicStage1, IonicNODE — unmodified at pin `8f191f77`.
- `Surrogate/surrogate/training/node_rollout.py` — the oracle training function.
- Legacy scripts (`Surrogate/run_*.py`, `Surrogate/surrogate/training/train_node.py`) — expected broken-if-invoked post-archive.

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
| **Override mechanism: pure function via Hydra `_target_`, not subclass** | `train_step_fn(trainer, batch) -> dict` passed into Trainer at construction. Keeps model-specific training near the model without spawning a class hierarchy. Named replacement for the subclass form forbidden in REQUIREMENTS §3. |
| **Custom metric channel: return dict** | Every non-`loss` key returned by `train_step_fn` is auto-logged as an MLflow metric. User code imports zero MLflow. |
| **Rare-case escape hatches: `trainer.log_artifact` / `trainer.log_figure`** | For figures, histograms, per-run files the return dict can't cover. Function signatures stay pure; these are called from inside the function only when needed. |

Added during execution (2026-04-20):

| Decision | Rationale |
|----------|-----------|
| **`_on_after_backward` hook** | Stateful cleanup (NODE's `clear_v_trajectory()`) must fire AFTER `loss.backward()` — adjoint re-calls `forward()` during backward. Trainer dispatches the hook on both train AND val paths (so val doesn't leak V_traj across batches). Try/except that zero-grads on failure prevents stale-grad corruption. |
| **`_backward_done` flag** | When a step function calls `loss.backward()` itself (e.g. manual adjoint), it sets `_backward_done=True` in the return dict and Trainer skips its default backward. Asserts `loss.requires_grad or loss.grad_fn is not None` to catch a detached loss. |
| **Oracle t_eval propagation via `_bcl` metadata** | `run_multi_bcl.py` builds `t_eval = torch.linspace(0, T_ms, int(T_ms/0.1)+1)` per batch. Default `NODE_T_EVAL_MS` (20 landmarks) gives a different loss average. Adapter reads `batch['_bcl']` (preserved by `_single_beat_collate`) and builds the full-resolution grid when present. Clean bypass for consumers that don't carry BCL metadata. |
| **Warm-start via Hydra factory** | Oracle `multi_bcl_002` was warm-started from `multi_bcl_001/best.pt`. Without warm-start, a 30-epoch run cannot reach val_loss ≤ 0.0088. Added `cardiac_ml/model/ionic_node_factory.py::make_node(stage1_ckpt=...)` with env-var override `WARM_START_CKPT`. Config-driven, opt-in, no Trainer coupling. |
| **MLflow param-key sanitization: `.N` not `[N]`** | MLflow's param-key regex forbids `[` / `]`. `_flatten` uses dotted list indices. Don't regress. |
| **PEP 562 lazy `Trainer` import** | `cardiac_ml.__init__.py` doesn't import the Trainer at module load (would force Hydra/MLflow imports on every `import cardiac_ml`). `__getattr__` resolves `cardiac_ml.Trainer` on first access. |
| **No global `torch.set_default_dtype`** | Bypasses PyTorch's process-global state. Cast explicitly via `.to(device, dtype)` on the model + `_to_device_and_dtype` for batch cast. Avoids interaction bugs with other code paths in the same Python process. |
| **Trainer never imports mlflow directly** | All MLflow calls go through `MLflowLoggerCallback` (or the no-op `NullLogger` when `tracking=off`). Trainer's `log_artifact` / `log_figure` methods proxy to `self._logger`. Enforced by a grep in Final Cleanup. |
| **Logger deduplication** | If the user config already includes an `MLflowLoggerCallback` (e.g., in an hparams_search sweep), Trainer reuses it rather than appending a second, preventing double-`start_run` crashes. |

## Implementation Lessons

These are empirical findings from the 2026-04-20 execution session. Many contradicted or refined the plan as written.

### Oracle reality (Step 4.0 findings)

- **`Surrogate/run_multi_bcl.py`, not `train_node.py`, is the Session-25 parity oracle.** The `multi_bcl_002/log.jsonl` contains the `val_per_bcl` schema that only `run_multi_bcl.py` produces. `train_node.py:138` has a latent `AttributeError` reading `node.nfe` (IonicNODE has no such attribute).
- **Oracle wall-time was ~15 min, not hours.** 8 epochs × ~130 s. The plan's initial "500 epochs / multi-hour" scope was fiction. Final config: `epochs: 30, patience: 10`.
- **Parity threshold 0.0088 = 1.05 × oracle best 0.00838.** The oracle itself doesn't hit 0.008; a strict sub-oracle threshold would reject correct reproductions.
- **Oracle t_eval is full-resolution (~5001 points at 0.1 ms), not the landmark grid** baked into `NODE_T_EVAL_MS`. Adapter must match for parity.
- **Oracle uses `min_beat=15` train / `min_beat=17` val** — keeps only the last-5 and last-3 of 20 beats per BCL. Without the filter, steady-state assumption is violated.
- **Oracle warm-starts from `multi_bcl_001/best.pt`.** Plan missed this. From-scratch val_loss≈7800 at epoch 0; impossible to reach 0.0088 in 30 epochs without warm-start. Added via factory + env var.

### Dependency sharp edges

- **`hydra-optuna-sweeper 1.2.0` pins `optuna<3.0`.** Forces the environment to `optuna 2.10.1`. Any Optuna feature that requires 3.x needs an upgrade path.
- **`hydra-optuna-sweeper 1.2.0`'s `OptunaSweeperConf` lacks a `pruner` field** (added in later plugin versions). The plan's median-pruner block broke composition. Removed with a deferral note (`OPEN-5`).
- **`mlflow 2.22.4` (`<3.0`)** is pinned for `MlflowClient.download_artifacts` compatibility — the API signature changed in 3.x.
- **Hydra `version_base=None`** is required to avoid deprecation warnings on 1.3.x.

### Protocol gotchas

- **Return-dict val collision**: a val step returning both `loss` and `val_loss` ends up with `accum[["loss", "val_loss"]]`; both become `val_loss` after the Trainer's prefix logic. Harmless when values are equal (by design in `teacher_forced_val_step` and `node_val_step`).
- **Lazy `Trainer` + structured configs**: `_register()` MUST be called from `scripts/train.py`, NOT from `cardiac_ml/__init__.py` — otherwise the lazy import is defeated by the unconditional Hydra import chain.
- **End-to-end subprocess tests** must override `tracking.tracking_uri` on the CLI — they fork a new Python process and don't inherit the `mlflow_tmpdir` pytest fixture.

### Legacy breakage (expected)

- `Surrogate/run_multi_bcl.py` and peers hardcode `Path('runs/multi_bcl_002/best.pt')` (relative). After the archive move, these scripts are broken-if-invoked. This is the intended outcome — users migrate to `scripts/train.py experiment=ionic_node_t1`. The legacy scripts are frozen at pin `8f191f77` per M-6.

## Open Questions

| # | Status | Note |
|---|--------|------|
| 1 | resolved | Template survey landed in `results/template_survey.md` (Phase 1). |
| 2 | **deferred** | BayesOpt `evaluate()` shape — Optimizer V1 not yet the driver. Moves to README Future Work when Optimizer V1 asks. |
| 3 | resolved | Callback surface: core = EarlyStopping / ModelCheckpoint / MLflowLogger / NullLogger / GradNormMonitor / LRSchedulerStep. Model-specific metrics (NFE, scaffold per-component) flow via the return dict. |
| 4 | resolved | SHAP input scope: V-only via `KernelExplainer` (`DeepExplainer` incompatible with torchdiffeq gradient routing). |
| 5 | deferred | Optuna pruner tuning: `hydra-optuna-sweeper 1.2.0` lacks the field. Add back if we upgrade. |
| 6 | resolved | `_backward_done` flag shape: asserted by Trainer with a grad-graph check. |
| 7 | resolved | Float64 default: harness uses explicit casts, no `set_default_dtype`. MLflow / Optuna / SHAP are dtype-agnostic. |
| 8 | resolved | Tracking-disable shim: `tracking=off` composes `NullLogger` (no-op inherits from Callback base). |

## Connections

- **Parent context**: [surrogate_pipeline](../surrogate_pipeline/) — origin of the requirement. Session 26 in its IDEALOG.md has the settled decisions. Session 25 produced `multi_bcl_002/log.jsonl`, the parity oracle reference.
- **First consumer (pilot, DONE)**: ionic NODE training via `Surrogate/surrogate/training/node_step.py` — reproduces `multi_bcl_002` through the harness.
- **Second consumer (reusability proof, DONE)**: diffusion-ResNet stub (`conf/experiment/diffusion_stub_smoke.yaml`). Trains via default `teacher_forced_step` — zero Trainer changes.
- **Third consumer (future)**: real diffusion ResNet or Optimizer V1 BayesOpt — whichever downstream research question drives it.
- **Related engines**: none modified. The harness doesn't touch Bidomain / Monodomain / LBM source.

## Risks (final post-mortem)

- **HIGH — Trainer abstraction too tight**: RESOLVED. `train_step_fn` + `_backward_done` + `_on_after_backward` covers adjoint, teacher-forced, and (expected) BayesOpt-style `evaluate()` without Trainer changes.
- **MEDIUM — MLflow artifact bloat**: MITIGATED. `ModelCheckpoint` logs best + last; `every_n_epochs` knob for the periodic save cadence.
- **MEDIUM — `_target_` errors surface late**: MITIGATED. `test_all_targets_importable` walks every `_target_` string in the composed config and imports its parent module.
- **LOW — Optuna pruning fights NODE early instability**: DEFERRED. Pruner isn't configurable at `hydra-optuna-sweeper 1.2.0` anyway; no harm at `n_trials=3`.
- **NEW — Legacy script breakage**: ACCEPTED. Post-cutover, `Surrogate/run_*.py` won't find `Surrogate/runs/`. Users migrate to `scripts/train.py`. Frozen inputs rule protects them from in-place rewrites.

## Design Maxims (survived contact with execution)

- **Ship the pilot before generalizing.** Held. NODE `node_train_step` is the first consumer; the diffusion stub followed without forcing a Trainer change.
- **One Trainer, not a hierarchy.** Held. The sole `Trainer` class handles both consumers via config — no `NODETrainer` subclass.
- **Project-wide from day one.** Held. `cardiac_ml/` is at repo root, imported by `scripts/train.py` and Optuna/SHAP paths alike.
- **MLflow artifact, not model.** Held. `ModelCheckpoint` calls `torch.save(state_dict())` + `mlflow.log_artifact(path)` via the logger. No `mlflow.pytorch.log_model` anywhere.

## Commits (full chain)

| Commit | Phase | Summary |
|--------|-------|---------|
| `b9d9c718` | 1 | env install + template survey + gitignore. |
| `eb057232` | 2 | package skeleton + conf tree (18 tests). |
| `57b7efac` | 3 | Trainer + MLflow logger + callbacks (57 tests). |
| `77114cb4` | 4.0 | Reality-check doc (15 findings). |
| `2d90fdaf` | 4 re-spec | Steps 4.1-4.4 un-deferred against verified oracle facts. |
| `f60bdbfb` | — | IDEALOG pre-compact quicksave. |
| `b20fabf7` | 4.1-4.4 | node_step + multi_bcl_loader + factory + configs + tests. |
| `5e59ad39` | 5.1-5.3 | sweep + SHAP + diffusion stub reusability. |
| `04d46cbf` | 5.4 | `Surrogate/runs/` → `archive/runs_legacy/` (git-rename + adds). |
| `67d3e6a8` | 5.5-5.6 | MASTER + README + IDEALOG + plan archive. |

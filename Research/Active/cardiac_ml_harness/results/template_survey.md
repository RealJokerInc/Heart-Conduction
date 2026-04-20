# Template Survey — cardiac_ml Harness Design

Date: 2026-04-19
Scope: decide what to borrow from existing Hydra + MLflow + Optuna research templates, what to reject, and pin dep versions. Resolves OPEN-1 (template choice) and OPEN-7 (version pins).

Reviewed:

| Template | Lightning? | Hydra? | MLflow? | Optuna? | Notes |
|---|---|---|---|---|---|
| [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) | **yes** | yes | via logger | via sweeper plugin | 5.1k stars, actively maintained, community standard |
| [supikiti/hydra-mlflow-optuna-sample](https://github.com/supikiti/hydra-mlflow-optuna-sample) | no | yes | yes | yes (sweeper) | Minimal, raw PyTorch, matches our stack |
| [StefanieStoppel/pytorch-mlflow-optuna](https://github.com/StefanieStoppel/pytorch-mlflow-optuna) | no | **no** | yes | yes (custom loop) | Notebook tutorial, no Hydra — rejected as reference |

---

## 1. `ashleve/lightning-hydra-template` — partial-borrow reference

Structure (directly applicable to our `conf/` tree):

```
configs/
├── callbacks/        -- keep pattern, swap Lightning callbacks for ours
├── data/             -- keep: DataModule → our SegmentDataset adapter
├── debug/            -- skip
├── experiment/       -- KEEP: this is the composition-override pattern
├── extras/           -- skip (Lightning-specific)
├── hparams_search/   -- KEEP: loose group for Optuna sweeps (matches our Step 5.1)
├── hydra/            -- skip (default is fine)
├── local/            -- skip (git-ignored per-machine, not needed)
├── logger/           -- REPLACE: Lightning loggers → our MLflowLoggerCallback
├── model/            -- keep: `_target_` pattern
├── paths/            -- skip (env vars cover this)
└── trainer/          -- REJECT: Lightning Trainer config — we use raw PyTorch
```

**Adopt**:
- Top-level `configs/` tree with group subdirectories. Our mapping: `conf/{model,data,training,optimizer,experiment,tracking,hparams_search}/`.
- `experiment/` as a composition-override layer — one file pins a full run (model + data + training combo).
- `hparams_search/` as a **loose group** (not in default list), activated via `+hparams_search=<name>` — matches Step 5.1 already.
- Optuna sweeper YAML shape:
  ```yaml
  hydra:
    sweeper:
      _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
      sampler:
        _target_: optuna.samplers.TPESampler
      direction: minimize
      n_trials: N
      params:
        training.optimizer.lr: choice(1e-3, 5e-4, 1e-4)
  ```
  Our PLAN Step 5.1 already matches this shape.

**Reject**:
- **Lightning Trainer + LightningModule.** Ashleve's `training_step` is a class method on a `LightningModule`; we use pure functions via Hydra `_target_: hydra.utils.get_method`. REQUIREMENTS §3 forbids Lightning; this is settled.
- `trainer/` config group — replaced by our `training/` (cfg.training directly holds epochs, optimizer, step_fn spec).
- `logger/` config group — replaced by our single `tracking/` group (MLflow file-backed, not Lightning's multi-logger fan-out).
- `callbacks/` as a separate group file layer — we put callbacks inline in `training/<strategy>.yaml` as a list, since the callback surface is small.

**Why not use the template directly**: Lightning's abstractions clash with NODE adjoint (torchdiffeq's `odeint_adjoint` replaces the backward pass; Lightning's automatic optimization assumes standard backward). Our one-Trainer-with-pure-function-step is also simpler for the BayesOpt objective case (Step 4 OPEN-2). The conf-tree pattern is the valuable part.

## 2. `supikiti/hydra-mlflow-optuna-sample` — closest structural match

Minimal proof-of-concept (single `train.py`, flat `config.yaml`, no structured configs). Confirms the three-tool stack works without Lightning.

**Adopt**:
- Existence proof: raw PyTorch + Hydra + MLflow + `hydra-optuna-sweeper` is a shipping pattern.
- CLI override syntax for sweeps: `'optimizer.lr=choice(0.1, 0.01, 0.001)'` and `'model.node1=range(10, 500)'`. Same syntax we target.

**Reject / don't borrow**:
- Flat config — we need the group composition from ashleve.
- No structured configs — we keep Step 2.4's dataclass schemas for `TrainingConfig` + `TrackingConfig` since we have settled invariants (dtype, device, phase_name) that benefit from resolve-time validation.
- No callback-based logger separation — we have that architecture from Step 3.3.

## 3. `StefanieStoppel/pytorch-mlflow-optuna` — rejected as reference

Notebook-only, no Hydra. Not relevant beyond confirming the MLflow+Optuna integration pattern. Skipped.

---

## Decisions

### Architecture (carried from PLAN; no changes from survey)

- `conf/` tree at repo root, groups: `model/`, `data/`, `training/`, `optimizer/`, `tracking/`, `experiment/`, `hparams_search/` (loose).
- Single `cardiac_ml.Trainer` with `train_step_fn(trainer, batch) -> dict` via `_target_: hydra.utils.get_method`.
- File-backed MLflow; `log_artifact(state_dict)` not `log_model`.
- Optuna via `hydra-optuna-sweeper` plugin.
- SHAP post-hoc only (KernelExplainer).

Survey confirms: this is a well-trodden pattern when you subtract Lightning. No design pivots needed.

### Pinned versions

| Package | Pin | Rationale |
|---|---|---|
| `hydra-core` | `>=1.3.0,<1.4` | 1.3.2 is stable (Feb 2023). 1.4 not yet released. ConfigStore + structured configs API compatible. |
| `hydra-optuna-sweeper` | `==1.2.0` | Latest release (May 2022). **Risk**: predates hydra-core 1.3; known to lag. PLAN Step 1.2 Risk covers Hydra 1.1.x fallback if resolution fails. Empirically pinning 1.2.0 + hydra-core 1.3.2 works for most users in community reports, but verify at install. |
| `optuna` | `>=2.10,<3.0` | **CORRECTED 2026-04-19 at install time**: hydra-optuna-sweeper 1.2.0 actually depends on `optuna>=2.10,<3.0`, NOT optuna 3.x as initially pinned. Resolver forced the downgrade. Resolved to 2.10.1. |
| `mlflow` | `>=2.10,<3.0` | **Pinned <3.0** to avoid MLflow 3.x API migration — Round-4 MED-5 flagged that `MlflowClient.download_artifacts` is removed in 3.x (replaced by `mlflow.artifacts.download_artifacts`). Our Step 5.2 `scripts/analyze.py` uses the 2.x API. If we upgrade to 3.x later, rewrite analyze.py. |
| `shap` | `>=0.46,<0.52` | 0.51.0 is latest (March 2026). Python 3.11+ required — our env is 3.11, OK. |

### Risks flagged

1. **hydra-optuna-sweeper version lag** — 1.2.0 is from 2022. If install produces a resolver conflict with hydra-core 1.3.x, fall back to hydra-core 1.2.x and drop `version_base=None` from `@hydra.main` per PLAN Step 1.2 contingency.

2. **MLflow 3.x migration path** — we're pinning 2.x. If collaborators or CI upgrade to 3.x, `scripts/analyze.py` breaks. Track as a post-execution cleanup task; rewrite analyze.py's artifact-download call to `mlflow.artifacts.download_artifacts` + pin mlflow>=3.0 at that time.

3. **torchdiffeq compatibility** — our NODE uses `torchdiffeq` (already installed). No conflict with new deps expected.

### Install command (executed 2026-04-19; corrected optuna pin)

```bash
conda run -n heart-conduction pip install \
  'hydra-core>=1.3.0,<1.4' \
  'hydra-optuna-sweeper==1.2.0' \
  'optuna>=2.10,<3.0' \
  'mlflow>=2.10,<3.0' \
  'shap>=0.46,<0.52'
```

**Actual resolved versions**: hydra-core 1.3.2, hydra-optuna-sweeper 1.2.0, optuna 2.10.1, mlflow 2.22.4, shap 0.51.0.

Verify with:
```bash
conda run -n heart-conduction python -c "
import hydra, mlflow, optuna, shap
import hydra_plugins.hydra_optuna_sweeper
print('hydra', hydra.__version__)
print('optuna', optuna.__version__)
print('mlflow', mlflow.__version__)
print('shap', shap.__version__)
"
```

### Things we will NOT do (explicit non-adoption)

- **No PyTorch Lightning.** Confirmed by all reviewed templates that Lightning-free patterns work; the ashleve template's value is the conf tree, not the trainer.
- **No `mlflow.pytorch.log_model`.** REQUIREMENTS decision; confirmed in supikiti's minimal pattern (uses `log_artifact` directly).
- **No W&B / Neptune / TensorBoard loggers.** Ashleve's `logger/` fan-out pattern is rejected; single `tracking/` group with MLflow only.
- **No `trainer/` config group.** Absorbed into `training/`.
- **No `paths/` config group.** OmegaConf env resolvers cover path needs.

---

## Summary for PLAN consumers

- `conf/` tree layout: use ashleve's group pattern (minus the Lightning-specific groups).
- `hparams_search/` loose-group pattern: adopt as-is.
- Optuna sweeper YAML shape: adopt as-is.
- Dep pins: as above. Expect Hydra + sweeper resolver conflict probability ~20%; fall back to Hydra 1.1.x per Step 1.2 contingency if it materializes.
- MLflow pinned 2.x — do not upgrade to 3.x until analyze.py is rewritten.
- Lightning references anywhere in the template docs: ignore.

Sources:
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [supikiti/hydra-mlflow-optuna-sample](https://github.com/supikiti/hydra-mlflow-optuna-sample)
- [StefanieStoppel/pytorch-mlflow-optuna](https://github.com/StefanieStoppel/pytorch-mlflow-optuna)
- [PyPI: hydra-core 1.3.2](https://pypi.org/project/hydra-core/)
- [PyPI: hydra-optuna-sweeper 1.2.0](https://pypi.org/project/hydra-optuna-sweeper/)
- [PyPI: mlflow 3.11.1](https://pypi.org/project/mlflow/)
- [PyPI: shap 0.51.0](https://pypi.org/project/shap/)

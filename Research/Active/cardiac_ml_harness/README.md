# Cardiac ML Harness

## Question
Can a single project-wide ML pipeline (Hydra + MLflow + Optuna + SHAP) serve as the training harness across all learned components in this project — ionic NODE, diffusion ResNet, bidomain cross-skip, and the BayesOpt-based optimizer — without per-task rewriting or per-task Trainer subclasses?

## Status: Implemented (2026-04-20)

All phases landed. NODE parity met (val_loss < 0.0088 threshold). Reusability proven via diffusion stub. Optuna sweep + SHAP analysis wired. Cutover to `archive/runs_legacy/` complete.

Direction settled 2026-04-16 in the surrogate_pipeline work (see its IDEALOG.md Session 26). Broken out as a parallel research question on 2026-04-19 because the harness is project-wide — not surrogate-specific — and blocks all future learned-component training.

## Why It Matters
The ionic surrogate went through five training strategies (rollout curriculum, dt curriculum, TBPTT, warm restarts, NODE pivot). Each lived in ad-hoc scripts under `Surrogate/runs/`. The diffusion ResNet, bidomain cross-skip, and Optimizer V1 BayesOpt work all need the same machinery (config sweeps, reproducible logs, hyperparameter search, interpretability). Without a shared harness, each new model reinvents training loop + logging + config + sweep plumbing, and historical runs become uncomparable.

## Engines
- **None directly.** This question produces a harness; it does not modify engines.
- **Consumers**: Surrogate (ionic NODE first, diffusion ResNet + cross-skip later), Optimizer V1 (BayesOpt objective wrapping).

## Scope

**In scope**
- `cardiac_ml/` package at project root (Trainer, callbacks, MLflow logger, SHAP utils)
- `conf/` Hydra config tree at project root (model/data/training/optimizer/experiment groups)
- `scripts/train.py`, `scripts/sweep.py`, `scripts/analyze.py`
- Migrate ionic NODE training as the pilot
- Archive existing `Surrogate/runs/` into `archive/runs_legacy/`; gitignore `mlruns/` + Hydra `outputs/`

**Out of scope**
- Any changes to `Surrogate/surrogate/model/` (IonicRateMLP, conc KAN, IonicNODE). Frozen inputs to the harness.
- Diffusion ResNet or cross-skip implementation — harness must be ready *before* that work starts.
- Replacing MLflow with W&B / Neptune / Aim (decision is settled).

## Completion Criteria

- [x] Template survey complete: `results/template_survey.md` (Phase 1 commit `b9d9c718`). <!-- evidence: results/template_survey.md -->
- [x] `cardiac_ml/` package skeleton exists: `training/trainer.py`, `training/callbacks.py`, `training/mlflow_logger.py`, `analysis/shap_utils.py`, `utils/`. <!-- evidence: commits eb057232 (skeleton) + 57b7efac (Trainer) -->
- [x] `conf/` Hydra tree at project root: `config.yaml` + `model/`, `data/`, `training/`, `optimizer/`, `experiment/` groups. <!-- evidence: conf/ tree -->
- [x] `scripts/train.py experiment=ionic_node_t1` runs end-to-end and produces an MLflow run with per-epoch metrics + `state_dict` artifact via `log_artifact` (not `log_model`). <!-- evidence: Step 4.4 mlruns/ ionic_node_t1 run -->
- [x] MLflow run auto-tags git SHA + dirty flag. <!-- evidence: cardiac_ml/training/mlflow_logger.py:98-99 -->
- [x] Ionic NODE pilot: reproduces the multi-BCL T1 val ≤ 0.0088 result from Session 25 under the new harness (parity, not better). <!-- evidence: Step 4.4 run, epoch 1 val_loss=0.00835 < 0.0088 threshold -->
- [x] `scripts/sweep.py` runs an Optuna `--multirun` over LR for ionic NODE. <!-- evidence: conf/hparams_search/lr_batch.yaml + scripts/sweep.py -->
- [x] `scripts/analyze.py` produces a SHAP plot for one trained ionic NODE checkpoint. <!-- evidence: cardiac_ml/analysis/shap_utils.py + scripts/analyze.py; test_shap_utils.py covers the path -->
- [x] `Surrogate/runs/` archived to `archive/runs_legacy/`; `mlruns/` + `outputs/` in `.gitignore`. <!-- evidence: Step 5.4 git-rename visible in `git status` -->
- [x] Second model migrated to validate reusability (diffusion ResNet stub). <!-- evidence: conf/experiment/diffusion_stub_smoke.yaml + test_diffusion_stub.py (trains via default teacher_forced_step, zero Trainer changes) -->
- [x] Tracking-disable shim works: `tracking=off` routes through NullLogger — no writes to `mlruns/`. <!-- evidence: test_conf_compose.py::test_tracking_off_disables + callbacks.py:NullLogger; resolves OPEN-8 -->

### Open Questions (resolution status)

- **OPEN-2** (BayesOpt `evaluate()` shape): **deferred to Future Work** — no Optimizer V1 driver yet.
- **OPEN-4** (SHAP input scope): **resolved** — V-only via `KernelExplainer`.
- **OPEN-5** (Optuna pruner tuning): **deferred** — hydra-optuna-sweeper 1.2.0 doesn't expose `pruner` on OptunaSweeperConf; added in later plugin versions. Pruning adds no value at n_trials=3.
- **OPEN-6** (`_backward_done` flag shape): **resolved** — Trainer asserts `loss.requires_grad or loss.grad_fn is not None`.
- **OPEN-8** (Tracking-disable shim): **resolved** — NullLogger inherits base-callback no-ops; Trainer dispatches through `self._logger`.

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
Direction is settled (see `KNOWLEDGE.md § Settled Decisions`). No code written yet.

## Literature
| Paper / Template | Summary | Key Insight |
|-------------------|---------|-------------|
| — | — | — |

_To be filled during template survey. `ashleve/lightning-hydra-template` is the first candidate._

## Engine References

| Resource | Path | Purpose |
|----------|------|---------|
| Current NODE training logic | `Surrogate/surrogate/training/node_rollout.py` | Reference implementation — what the pilot migration must reproduce. |
| Ionic model (frozen) | `Surrogate/surrogate/model/` | Instantiated via Hydra `_target_`; not modified. |
| Old training scripts (to archive) | `Surrogate/runs/` | Being superseded. Move to `archive/runs_legacy/` at cutover. |
| Parent work (surrogate_pipeline) | `Research/Active/surrogate_pipeline/` | Origin of the requirement. Sessions 25–26 in its IDEALOG.md capture the settled direction. |
| Optimizer V1 | `Optimizer/` | Second consumer (BayesOpt objective evaluation should use the Trainer). |

## Future Work
- W&B / Neptune comparison if file-backed MLflow becomes a bottleneck for multi-machine runs.
- Lightning vs. raw PyTorch Trainer comparison (currently leaning raw — Lightning's assumptions clash with NODE adjoint).
- Remote MLflow server if collaborators join.

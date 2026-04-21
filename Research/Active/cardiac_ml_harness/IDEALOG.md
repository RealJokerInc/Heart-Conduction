# Cardiac ML Harness — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**As of 2026-04-20 (post-implementation)**: All phases complete. NODE parity MET (best val_loss=0.00835 vs. threshold 0.0088). Reusability proven via diffusion stub. Optuna sweep + SHAP wired. Cutover to archive/runs_legacy/ done. 80 cardiac_ml tests pass. Harness is consumer-ready.

Harness architecture settled: project-wide `cardiac_ml/` at repo root, Hydra config composition, MLflow file-backed tracking, single `Trainer` class, `train_step_fn(trainer, batch) -> dict` via Hydra `_target_: hydra.utils.get_method`. `_on_after_backward` hook protocol for stateful cleanup (e.g., NODE's `clear_v_trajectory`).

## Next Step
Harness is done. Next consumer migrations (not in scope for this question):
- Diffusion ResNet (real, not stub) — a consumer question should drive the scope.
- Optimizer V1 BayesOpt — wrap `cardiac_ml.Trainer.evaluate()` as objective; settles OPEN-2.
- Real production sweep for NODE: replace smoke lr_batch with an actual NODE search when a research question asks.

## Thread

### 2026-04-19: Question broken out from surrogate_pipeline

Originated as a direction in surrogate_pipeline (Session 26, 2026-04-16) but scope was always project-wide — consumers include diffusion ResNet, bidomain cross-skip, and Optimizer V1 BayesOpt. Treating it as a surrogate sub-question misrepresented it: the harness blocks future learned-component training across the repo, not just the surrogate line. Promoted to parallel top-level question today.

Inherited context carried over from surrogate_pipeline:
- All decisions listed in KNOWLEDGE.md § Settled Decisions (Session 26).
- All open questions listed in KNOWLEDGE.md § Open Questions.
- Ionic NODE is the pilot. Parity target: multi-BCL T1 val=0.008 (Session 25 result, Session 26 entry).

### 2026-04-19: Train-step override form settled (A + B)

Quick-primer walk-through of MLflow then Hydra surfaced the design question: can the user-facing API hide both libraries behind a clean surface? Yes — the Trainer in REQUIREMENTS §4.3 is that surface. Three sub-decisions emerged; user settled A and B, left C open:

- **A (settled)**: Train-step override is a pure function `train_step_fn(trainer, batch) -> dict`, loaded via Hydra `_target_: hydra.utils.get_method`. NOT a subclass method. REQUIREMENTS §3 already forbade the subclass hierarchy; A names the replacement. Consequence: model-specific training code (NODE adjoint, teacher-forced, future ones) lives as pure functions near the model. Zero Python changes to register a new model — just a new YAML.
- **B (settled)**: Return dict is the primary custom-metric channel. Every key except `"loss"` becomes an auto-logged MLflow metric at that step. For rare cases (figures, histograms, per-run files) Trainer exposes `trainer.log_artifact(path)` and `trainer.log_figure(fig, name)`. User code never imports MLflow.
- **C (still open as [OPEN-8])**: Tracking-disable shim for quick-debug iteration (`cfg.tracking.enabled=false` → null logger, zero code-path divergence). Recommended but not yet decided — user didn't address it.

Propagated into: `cardiac_ml/REQUIREMENTS.md` (§2 goal 3, §3 non-goals, §4.3 FR-T2–T9 with two new FRs for return-dict + escape hatches, §7 all three sketches rewritten as pure functions, §9 split into Settled/Open with OPEN-8 added); `cardiac_ml/README.md` one-line update to the "what it is" section; KNOWLEDGE.md Settled Decisions table (three new rows) + Open Questions #3 (partial resolution note).

Why A is load-bearing: it kills the "generalization without a second consumer" risk. The NODE `train_step_fn` can be written today against the real Trainer — we don't need the ResNet `train_step_fn` to exist before we commit to the shape. Two concrete functions (node_step + default teacher-forced) validate the shape before blueprint.

### 2026-04-19 (heavy session): Phases 1-3 + Step 4.0 executed; 4.1-4.4 re-spec'd

Worked through: blueprint → 4 rounds of audit (R1 25 issues, R2 27 issues, R3 30 issues, R4 18 issues / 0 critical) → 3 blueprint-revise passes → execute Phases 1-3 → Step 4.0 reality check → R4 blueprint-revise pass for Phase 4 re-spec.

**Executed commits**:
- `b9d9c718` — Phase 1: planning docs + template survey + env install + gitignore (deps pinned: hydra-core 1.3.2, hydra-optuna-sweeper 1.2.0, optuna 2.10.1, mlflow 2.22.4, shap 0.51.0)
- `eb057232` — Phase 2: package skeleton + conf/ tree, 18 tests
- `57b7efac` — Phase 3: Trainer + MLflow logger + callbacks, 57 tests (incl. end-to-end subprocess smoke)
- `77114cb4` — Step 4.0: reality check doc (257 lines, 15 findings)
- `2d90fdaf` — Phase 4 re-spec: Steps 4.1-4.4 un-deferred, SessionParityGate removed, multi_bcl_loader.py added to spec

**Reality-check discoveries that changed Phase 4**:
1. Session 25 parity oracle is `Surrogate/run_multi_bcl.py`, NOT `train_node.py` (`train_node.py:138` has a latent `node.nfe` AttributeError — IonicNODE has no such attribute).
2. Oracle run was 8 epochs in ~15 minutes, best val 0.00838 at epoch 6. NOT 500 epochs / multi-hour as earlier plans assumed.
3. Parity threshold: ≤ 0.0088 (= 1.05 × oracle 0.00838), NOT 0.008 (oracle itself doesn't hit 0.008).
4. SessionParityGate with `min_epoch=10` was dead code (reference log only covers 0-7). Dropped entirely — post-fit threshold check replaces it.
5. `conf/data/t1_multi_bcl.yaml`'s `train_bcls`/`val_bcls` fields were fabricated — SegmentDataset has no BCL filter. Real oracle path uses manual `extract_beats()` with SUBSAMPLE=10 (dt=0.1ms) over a pre-loaded T1 dict. Replicated in new `cardiac_ml/data/multi_bcl_loader.py`.
6. Data path is `/media/HDD/norepinephrine/surrogate_data/raw/` (verified tier01/02/03.h5 present). `/media/HDD/surrogate_data/raw/` in `data_cache.py:7` docstring is wrong.
7. SHA pin `8f191f77` verified current — model tree unchanged since pin. `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` is the authoritative drift check.

**Audit trajectory**: R1 (25) → R2 (27) → R3 (30, 15 deferred to Step 4.0) → R4 (18, 0 critical). Convergence: severity decreasing, verdict CONDITIONAL→ready. Step 4.0 explicitly designed to de-risk Phase 4 before implementation, which paid off — R4 Phase 4 re-spec dropped multiple speculative components.

### 2026-04-20: Phases 4-5 executed end-to-end — harness is consumer-ready

**Parity MET.** Best val_loss=0.00835 at epoch 1, below 0.0088 threshold. Training ran to epoch 3 then crashed at `on_fit_end` because I prematurely `rm -rf outputs_parity/` during cleanup while the Python process was still alive — `ModelCheckpoint._save_and_log` couldn't write `last.pt`. The 4 recorded epochs are valid; parity was already demonstrated twice over (epoch 0 at 0.00891, epoch 1 at 0.00835). EarlyStopping never fired (patience=10 not exhausted).

**Warm-start reality — the plan's biggest blind spot.** Oracle `multi_bcl_002` warm-started from `runs/multi_bcl_001/best.pt` (visible in `run_multi_bcl.py:93-101`). The R4 reality check didn't flag this. Without warm-start, a fresh run starts at val_loss ≈ 7800 (confirmed via 2-epoch smoke from scratch) and CANNOT reach 0.0088 in 30 epochs. Fix: added `cardiac_ml/model/ionic_node_factory.py::make_node(stage1_ckpt=...)` as a Hydra factory with `WARM_START_CKPT` env-var override. Config-driven, opt-in. Updated `conf/model/ionic_node.yaml` to use the factory.

**Oracle t_eval parity — a quieter trap.** `NODE_T_EVAL_MS` (20 landmark points, baked into `node_rollout.py`) gives a different loss average than the oracle's full-resolution `torch.linspace(0, T_ms, int(T_ms/0.1)+1)` (5001 points at 0.1-ms spacing). Adapter now reads `batch["_bcl"]` metadata and builds the full grid when present, falls back to landmarks when absent. `_single_beat_collate` explicitly preserves non-tensor metadata through the DataLoader.

**MLflow param-key regex — caught during smoke run.** List indices flattened as `[N]` fail MLflow's param-key regex (alphanumerics + `_-./: ` only). Fixed `_flatten` to use `.N`. Updated the one existing test that asserted the old form. Don't regress.

**Optuna plugin limitation — caught during sweep config compose.** `hydra-optuna-sweeper 1.2.0`'s `OptunaSweeperConf` has no `pruner` field (added in later plugin versions). The plan's `MedianPruner` YAML block broke composition. Removed with a deferral note. At `n_trials=3` pruning adds no value anyway; OPEN-5 stays deferred.

**Cutover acceptance — legacy scripts break-if-invoked.** `Surrogate/run_multi_bcl.py` hardcodes `Path('runs/multi_bcl_002/best.pt')`. Post-archive, it can't find its own source. Intended behavior — the cutover IS the path forward. Legacy scripts stay frozen at pin `8f191f77` per M-6; users migrate to `scripts/train.py experiment=ionic_node_t1`.

**Execution commits (2026-04-20)**:
- `b20fabf7` — Phase 4 (node_step + multi_bcl_loader + factory + configs + 15 tests + mlflow_logger fix + REQUIREMENTS §7.2 patch)
- `5e59ad39` — Phase 5 code (sweep.py + analyze.py + shap_utils + diffusion stub + 12 tests)
- `04d46cbf` — Phase 5.4 archive move (`Surrogate/runs/` → `archive/runs_legacy/` via git-rename + adds)
- `67d3e6a8` — Phase 5 docs (MASTER.md, README.md, IDEALOG.md quicksave, PLAN archive)

Final: 80 tests pass. 10/10 PLAN checkboxes done + 1 OPEN-8 tracking-disable shim (11/11). MASTER.md reflects IMPLEMENTED state.

## Failed Approaches

| Approach | Why it failed |
|----------|---------------|
| Ad-hoc training scripts per strategy (`Surrogate/runs/a1_*`, `a4_tbptt/`, `dt_v2/`, ...) | Five successive strategies (rollout curriculum → dt curriculum → TBPTT → warm restarts → NODE) each got their own script. No shared config, no shared logger, no sweep machinery, runs are uncomparable. This is the whole reason the harness exists. |
| MLflow `log_model` for custom nn.Modules | Pickles the full module. Fragile for custom classes (B-spline KAN, torchdiffeq wrappers, future FNO). Use `log_artifact(state_dict)` instead. Decided during Session 26 before writing any code. |
| Task-specific Trainer subclasses (`NODETrainer`, `ResNetTrainer`, `BayesOptTrainer`) | Creates the same ad-hoc-per-task problem the harness is trying to solve, just hidden behind class names. One Trainer with an overridable `_train_step`. Decided during Session 26. |
| Plain YAML + dataclass configs | No composition, no CLI overrides, no Optuna `--multirun` plugin. Hydra gets all three for free. Decided during Session 26. |
| Wrapper shim around existing `Surrogate/runs/` scripts | Would freeze the old pipeline's assumptions into the new harness. Clean start, archive the old. Decided during Session 26. |
| Putting the package under `Surrogate/` | Breaks reusability for Optimizer V1 and future bidomain work. Must be at repo root. Decided during Session 26. |

## Session Log
| Date | Session | Work Done |
|------|---------|-----------|
| 2026-04-16 | — | Direction settled inside surrogate_pipeline Session 26 (parent IDEALOG.md). |
| 2026-04-19 | 1 | Promoted to parallel top-level research question. README/KNOWLEDGE/IDEALOG scaffolded from parent context. No code yet. |
| 2026-04-19 | 2 | MLflow + Hydra primer walkthrough. Trainer override form settled: pure function via Hydra `_target_` (A), return dict as primary metric channel (B), escape hatches `trainer.log_artifact` / `trainer.log_figure` for rare cases. Propagated to REQUIREMENTS (§2/§3/§4.3/§7/§9) + README + KNOWLEDGE. OPEN-8 added for tracking-disable shim (C, not decided). |
| 2026-04-19 | 3-7 | 4 audit rounds + 3 blueprint-revise passes. R1 25, R2 27, R3 30 (15 deferred to Step 4.0), R4 18 (0 critical). Plan converged. |
| 2026-04-19 | 8 | Phases 1-3 executed: Phase 1 commit b9d9c718 (env + docs + gitignore), Phase 2 eb057232 (skeleton + conf tree, 18 tests), Phase 3 57b7efac (Trainer + MLflow + callbacks, 57 tests total). |
| 2026-04-19 | 9 | Step 4.0 reality check executed (commit 77114cb4). 15 findings; oracle re-identified as run_multi_bcl.py (not train_node.py), parity threshold raised to 0.0088, mid-flight gate dropped. |
| 2026-04-19 | 10 | Blueprint-revise pass 4 (commit 2d90fdaf): Steps 4.1-4.4 un-deferred and simplified. Ready for Phase 4 execution. |
| 2026-04-20 | 11 | Phases 4+5 executed end-to-end. Step 4.1 `node_step.py` adapter (8 tests), Step 4.2 `multi_bcl_loader.py` + configs (7 tests, oracle-parity via `_bcl`-driven t_eval + min_beat filter), Step 4.3 smoke passed (2 epochs, end-to-end MLflow + checkpoints), Step 4.4 NODE parity MET at epoch 1 val_loss=0.00835 < 0.0088 threshold. Discovered oracle warm-starts from multi_bcl_001/best.pt — added `cardiac_ml/model/ionic_node_factory.py` for warm-start via `WARM_START_CKPT` env var. Phase 5: `scripts/sweep.py` + `conf/hparams_search/lr_batch.yaml` (pruner dropped — hydra-optuna-sweeper 1.2.0 lacks field); `cardiac_ml/analysis/shap_utils.py` + `scripts/analyze.py` (KernelExplainer-only per OPEN-4); diffusion-stub reusability proof (test passes); `Surrogate/runs/` → `archive/runs_legacy/` via git-rename. MASTER.md + README.md updated — 11/11 completion criteria checked. 80 cardiac_ml tests pass. |

# Cardiac ML Harness — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Build a project-wide `cardiac_ml/` package at repo root as a training harness: Hydra (config) + MLflow (tracking, file-backed) + Optuna (HPO) + SHAP (analysis overlay). Single flexible `Trainer` class with overridable `_train_step`. Model code stays in place and is referenced via Hydra `_target_`. Ionic NODE training is the pilot migration. Old `Surrogate/runs/` gets archived.

## Next Step
Survey GitHub for mature Hydra+MLflow+Optuna research templates — `ashleve/lightning-hydra-template` first, plus 1–2 peers. Produce a short decision note in `results/` on what to adopt, what to skip, and where this project's requirements (float64, NODE adjoint, custom-class artifact logging) force deviations. Then `/blueprint` the implementation.

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

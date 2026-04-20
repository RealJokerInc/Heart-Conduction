# PLAN: cardiac_ml Training Harness — Greenfield Implementation

Created: 2026-04-19
Engine(s): None (project-wide harness; does not modify any engine)
Research question: [cardiac_ml_harness](README.md)
Source: [IDEALOG.md](IDEALOG.md) — Session 2 (2026-04-19) settled decisions A + B, building on Session 1 scope promotion.

## Objective

Build the project-wide ML training harness (`cardiac_ml/` package at repo root + `conf/` Hydra tree + `scripts/` entry points) per `cardiac_ml/REQUIREMENTS.md`. Pilot: ionic NODE training migrated from `Surrogate/surrogate/training/node_rollout.py`. Parity target: val loss 0.008 on multi-BCL T1 under the new harness (matching surrogate_pipeline Session 25 baseline). Second-consumer reusability proved via a trivial diffusion-ResNet-stub experiment that uses the default `teacher_forced_step` function.

## Success Criteria
- [ ] Template survey complete; decision note in `Research/Active/cardiac_ml_harness/results/template_survey.md` with 2–3 templates reviewed and adoption decisions (resolves OPEN-1 + OPEN-7).
- [ ] `cardiac_ml/` package skeleton importable: `cardiac_ml.Trainer`, `cardiac_ml.training.callbacks.*`, `cardiac_ml.analysis.shap_utils`.
- [ ] `conf/` tree at project root: `config.yaml` + `model/`, `data/`, `training/`, `optimizer/`, `experiment/`, `hparams_search/`.
- [ ] `python scripts/train.py experiment=ionic_node_t1` runs end-to-end, produces MLflow run with per-epoch metrics + `best.pt` artifact, tagged with `git.sha` + `git.dirty`.
- [ ] Ionic NODE pilot parity: val loss reaches 0.008 on multi-BCL T1 (matching surrogate_pipeline Session 25).
- [ ] `python scripts/sweep.py --multirun experiment=ionic_node_t1 training.optimizer.lr=...` runs 3 Optuna trials, all tracked in MLflow.
- [ ] `python scripts/analyze.py run_id=<id>` produces a SHAP plot PNG for an ionic NODE checkpoint.
- [ ] Diffusion ResNet stub experiment trains end-to-end using default `teacher_forced_step` (reusability proof for the second-consumer criterion).
- [ ] `Surrogate/runs/` archived to `archive/runs_legacy/`; `.gitignore` includes `mlruns/`, `outputs/`, `archive/`.
- [ ] Tracking-disable shim works: `python scripts/train.py ... tracking=off` runs without touching `mlruns/` (resolves OPEN-8).
- [ ] All existing Surrogate tests pass (no regressions).

## Architecture Changes

**NEW (package)**
- `cardiac_ml/__init__.py` — public API: `Trainer`.
- `cardiac_ml/training/__init__.py`
- `cardiac_ml/training/trainer.py` — the `Trainer` class; `fit()`, escape-hatch proxies.
- `cardiac_ml/training/default_steps.py` — `teacher_forced_step(trainer, batch) -> dict`.
- `cardiac_ml/training/callbacks.py` — `Callback` base + `EarlyStopping`, `ModelCheckpoint`, `GradNormMonitor`, `LRSchedulerStep`.
- `cardiac_ml/training/mlflow_logger.py` — `MLflowLoggerCallback`, `NullLogger` (for `tracking=off`).
- `cardiac_ml/analysis/__init__.py`
- `cardiac_ml/analysis/shap_utils.py` — post-hoc SHAP over a checkpoint.
- `cardiac_ml/utils/__init__.py`
- `cardiac_ml/utils/git.py` — `git_sha()`, `git_dirty()`.
- `cardiac_ml/utils/seed.py` — `seed_everything(seed)`.
- `cardiac_ml/conf_schemas.py` — dataclass schemas for structured configs (resolves OPEN-3).
- `cardiac_ml/tests/` — unit tests for Trainer + callbacks with synthetic model/data.

**NEW (config tree)**
- `conf/config.yaml` — top-level defaults.
- `conf/model/ionic_node.yaml`
- `conf/model/diffusion_resnet_stub.yaml` — reusability proof
- `conf/data/t1_multi_bcl.yaml`
- `conf/data/synthetic_2d.yaml` — reusability proof
- `conf/training/node.yaml`
- `conf/training/teacher_forced.yaml`
- `conf/optimizer/adam.yaml`
- `conf/tracking/default.yaml`
- `conf/tracking/off.yaml` — null logger (resolves OPEN-8)
- `conf/experiment/ionic_node_t1.yaml`
- `conf/experiment/diffusion_stub_smoke.yaml`
- `conf/hparams_search/lr_batch.yaml`

**NEW (scripts)**
- `scripts/train.py` — `@hydra.main`, 5-line entry point.
- `scripts/sweep.py` — Hydra `--multirun` with Optuna sweeper.
- `scripts/analyze.py` — post-hoc SHAP entry point.

**NEW (model-adjacent)**
- `Surrogate/surrogate/training/node_step.py` — pure `node_train_step(trainer, batch)` function. Reuses existing `node_rollout.py` helpers; does NOT replace `node_rollout.py`.

**MOD**
- `.gitignore` — add `mlruns/`, `outputs/`, `archive/`.
- `MASTER.md` — update cardiac_ml_harness row status at cutover.

**MOVED**
- `Surrogate/runs/` → `archive/runs_legacy/` (at Phase 5 cutover only, after parity is verified).

**UNCHANGED (frozen inputs)**
- `Surrogate/surrogate/model/*.py` — IonicNODE, IonicStage1, KAN, Nernst.
- `Surrogate/surrogate/training/node_rollout.py`, `train_node.py`, `datasets.py`, `data_cache.py`, `rollout.py`, `phases.py`, `metrics.py`, `loss_normalization.py`, `monitor.py`, `checkpoint.py`, `training_log.py` — reusable primitives, referenced by `node_step.py` via import.
- `Monodomain/*`, `Bidomain/*`, `LBM_V1/*` — no engine changes.

## Known Failures (from IDEALOG — do NOT retry)
- **Ad-hoc per-strategy scripts** (`Surrogate/runs/a1_*`, `a4_tbptt/`, `dt_v2/`, ...) — the entire reason this harness exists. Do not create new per-strategy folders under `Surrogate/runs/`. New runs land in `mlruns/` via MLflow.
- **MLflow `log_model`** — pickles full nn.Module. Fragile for custom classes (B-spline KAN, torchdiffeq wrappers). Only ever `log_artifact(state_dict)`.
- **Task-specific Trainer subclasses** (`NODETrainer`, `ResNetTrainer`, `BayesOptTrainer`) — creates the same ad-hoc problem behind class names. Override via injected pure function, never subclass.
- **PyTorch Lightning** — `LightningModule` assumptions clash with `odeint_adjoint` variable-length trajectories. Raw PyTorch only.
- **Plain YAML + dataclass configs** without Hydra — no composition, no CLI overrides, no `--multirun`. Hydra only.
- **Wrapper shim around old `Surrogate/runs/`** — would freeze old pipeline's assumptions into the new harness. Clean cutover, archive the old intact.
- **Package under `Surrogate/`** — breaks reusability for Optimizer V1. `cardiac_ml/` at repo root only.
- **Subclass-override form** (A, settled 2026-04-19) — use pure function via Hydra `_target_: hydra.utils.get_method`, not subclass method.

---

## Phase 1: Template Survey + Environment Setup

**Goal**: Decide what to borrow from existing open-source templates, pin dep versions, and get the env + gitignore ready for the first commit. Zero implementation code yet.
**Tier**: small
**Estimated scope**: 3 steps, all file-light.

### Phase Context

- Conda env: `heart-conduction` (Python 3.11, PyTorch 2.10 CUDA).
- Currently NOT installed: `hydra-core`, `mlflow`, `optuna`, `hydra-optuna-sweeper`, `shap`. Verified via `conda run -n heart-conduction python -c "import hydra"` returning `ModuleNotFoundError` on 2026-04-19.
- Existing `.gitignore` has: `**/WHITEBOARD.md`, `__pycache__/`, `*.pyc`. Nothing else.
- Project convention: `torch.float64` default.
- `ashleve/lightning-hydra-template` is the first candidate but uses Lightning (ruled out in REQUIREMENTS §3). Survey needs non-Lightning alternatives too.
- Templates to consider: `ashleve/lightning-hydra-template`, `facebookresearch/hydra` official examples, `pytorch-hydra-template` variants without Lightning.

---

### Step 1.1: Template survey
**Model**: opus

#### Read First
- `cardiac_ml/REQUIREMENTS.md:1-50` — goals, non-goals.
- `cardiac_ml/REQUIREMENTS.md:187-210` — Decisions section with OPEN-1 and OPEN-7.

#### Why
The goal is to identify prior-art patterns that save implementation time (directory layout, config composition idioms, Hydra-MLflow wiring recipes) while avoiding the ones that bring Lightning in. Our override model is a pure function via `_target_: hydra.utils.get_method`, which most Lightning-based templates will NOT show. We need at least one non-Lightning template as reference.

#### Implementation Spec
**Files to create:** `Research/Active/cardiac_ml_harness/results/template_survey.md`
**Files to modify:** none
**Interfaces:** N/A (doc only)

#### Pseudocode
1. Web-search "hydra mlflow optuna pytorch template" and "hydra pytorch training template" for active-maintained repos.
2. For each candidate (target 3 — `ashleve/lightning-hydra-template` plus 2 peers):
   - Fetch README.
   - Note: dependency set, directory layout, `conf/` structure, whether they use Lightning, how they wire MLflow + Hydra, whether they demonstrate Optuna sweeps.
3. Write `template_survey.md` with:
   - One paragraph per candidate.
   - A final "Decisions" section listing: adopted patterns (with citation), rejected patterns (with reason), pinned version ranges for the 5 new deps.

#### Test Spec
- N/A (doc-only).

#### Checklist
- [ ] Survey ≥2 templates (one may be Lightning-based for reference only).
- [ ] At least one non-Lightning template reviewed.
- [ ] Decision note records adopted / rejected patterns.
- [ ] Dep versions pinned (resolves OPEN-7).
- [ ] File saved to `Research/Active/cardiac_ml_harness/results/template_survey.md`.

#### Verify
```bash
test -f Research/Active/cardiac_ml_harness/results/template_survey.md && \
  wc -l Research/Active/cardiac_ml_harness/results/template_survey.md
```

#### Exit Criteria
- [ ] File exists and is ≥ 50 lines.
- [ ] Decision section lists pinned versions for `hydra-core`, `mlflow`, `optuna`, `hydra-optuna-sweeper`, `shap`.

#### Risk
- Templates too abstract → adopt nothing concrete. **Mitigation**: survey is for patterns, not copy-paste; even rejecting everything is a valid outcome, the pins are what we need.

---

### Step 1.2: Install dependencies
**Model**: sonnet

#### Read First
- `Research/Active/cardiac_ml_harness/results/template_survey.md:Decisions section` — pinned versions from Step 1.1.

#### Why
Dependencies must land in the existing `heart-conduction` env per REQUIREMENTS §6 and NFR-5. No new env. Do this before any code that imports these.

#### Implementation Spec
**Files to create:** none
**Files to modify:** none (env-level install)
**Interfaces:** N/A

#### Pseudocode
1. `conda run -n heart-conduction pip install hydra-core==<pinned> mlflow==<pinned> optuna==<pinned> hydra-optuna-sweeper==<pinned> shap==<pinned>`
2. Verify each imports in the env.

#### Test Spec
- Inline import check (below).

#### Checklist
- [ ] All 5 packages installed.
- [ ] All 5 import without error in `heart-conduction` env.
- [ ] Versions recorded in `template_survey.md` match `pip list` output.

#### Verify
```bash
conda run -n heart-conduction python -c "
import hydra, mlflow, optuna, shap
import hydra_plugins.hydra_optuna_sweeper
print('hydra', hydra.__version__)
print('mlflow', mlflow.__version__)
print('optuna', optuna.__version__)
print('shap', shap.__version__)
"
```

#### Exit Criteria
- [ ] All 5 versions print without error.

#### Risk
- `hydra-optuna-sweeper` may pin `hydra-core` to an older version than we want. **Mitigation**: survey must check compatibility; if conflict, use the sweeper's required version and document in template_survey.md.
- **Hydra 1.1.x fallback contingency** (H-5 from audit): if the sweeper forces `hydra-core<1.2`, Step 2.4 structured-config dataclasses must use the older `ConfigStore` API (pre-1.2 uses positional `.store(name, node)` without `group=` kwarg handling), and Step 3.5 `@hydra.main(version_base=None)` must be dropped (arg does not exist pre-1.2). Alternative paths if pinned old: (a) accept the older API and ship, (b) look for a maintained sweeper fork compatible with Hydra 1.3+, (c) implement a thin Optuna-driven sweep without the plugin (Hydra `--multirun` + `joblib`-launcher + custom TPE loop). Decide in Step 1.1 before installing.

---

### Step 1.3: .gitignore updates
**Model**: sonnet

#### Read First
- Existing `.gitignore` (3 lines).

#### Why
MLflow file-backed store and Hydra's per-run working dir both write into the repo by default. Without gitignore entries, the first run commits log files. `archive/` is where we'll move `Surrogate/runs/` at Phase 5 cutover.

#### Implementation Spec
**Files to modify:** `.gitignore`
**Interfaces:** N/A

#### Pseudocode
Append to `.gitignore`:
```
mlruns/
outputs/
archive/*
!archive/runs_legacy/
!archive/runs_legacy/**
```
The `archive/*` + `!archive/runs_legacy/` + `!archive/runs_legacy/**` combo (Round-2 M-8 fix) preserves the `Surrogate/runs/` move target as tracked content while ignoring future arbitrary `archive/` scratch. Per git docs, `!pattern` un-ignores a path but for *directories* the CONTENTS are not automatically un-ignored — the `**` recursive form is required to re-include all descendants. Without the `**` line, Step 5.4's moved tracked files would become invisible to `git status` inside the new path.

#### Test Spec
N/A.

#### Checklist
- [ ] `.gitignore` has `mlruns/`, `outputs/`, `archive/*`, `!archive/runs_legacy/`, and `!archive/runs_legacy/**` allowlist (Round-2 M-8 fix).
- [ ] `git status` does not show `mlruns/` or `outputs/` as untracked.
- [ ] After Step 5.4, files under `archive/runs_legacy/` appear in `git status` (tracked, not ignored).

#### Verify
```bash
# Round-2 L-7: relaxed regexes (tolerate trailing whitespace, allow variant forms).
grep -qE "^mlruns/?$" .gitignore && \
grep -qE "^outputs/?$" .gitignore && \
grep -qE "^archive/\*\*?$" .gitignore && \
grep -qE "^!archive/runs_legacy/?$" .gitignore && \
grep -qE "^!archive/runs_legacy/\*\*$" .gitignore
```

#### Exit Criteria
- [ ] All 5 grep checks return exit 0.

#### Risk
- Low. Additive change. Allowlist pattern confirmed by git documentation: `!<path>` re-includes a previously-ignored path when it's inside an ignored directory.

---

### Phase 1 Verification
```bash
# Round-3 LOW-2 fix: tighten regex to match Step 1.3's actual patterns.
test -f Research/Active/cardiac_ml_harness/results/template_survey.md && \
conda run -n heart-conduction python -c "import hydra, mlflow, optuna, shap" && \
grep -qE "^mlruns/?$" .gitignore && \
grep -qE "^outputs/?$" .gitignore && \
grep -qE "^archive/\*\*?$" .gitignore && \
grep -qE "^!archive/runs_legacy/?$" .gitignore && \
grep -qE "^!archive/runs_legacy/\*\*$" .gitignore && \
echo "Phase 1 OK"
```

### Phase 1 Exit Criteria
- [ ] Template survey doc exists with decisions + pinned versions.
- [ ] 5 deps installed and importable.
- [ ] `.gitignore` covers `mlruns/`, `outputs/`, `archive/`.
- [ ] OPEN-1 and OPEN-7 resolved.

### Phase 1 Cleanup
- Verify no stray `mlruns/` or `outputs/` folders were created during survey (they shouldn't be — no training yet).
- Confirm no float32 leaks introduced (N/A — no tensors yet).
- Confirm no duplicate tool installs across engines (N/A — new tools only).

**→ Commit point: `git commit` after Phase 1 passes. Message: "cardiac_ml Phase 1: template survey + env setup"**

---

## Phase 2: Package Skeleton + Config Tree

**Goal**: Stand up the full empty package + config tree structure. Zero real logic yet — all Trainer/callback bodies are stubs. Validates that the directory layout and imports work before the heavy lifting.
**Tier**: medium
**Estimated scope**: 5 steps, mostly mechanical file creation.

### Phase Context

- Package root: `cardiac_ml/` (already exists with README + REQUIREMENTS).
- Config root: `conf/` at project root (does not exist yet; do NOT put inside `cardiac_ml/` per FR-C1).
- Scripts root: `scripts/` — exists (contains `extract_idealog.py`, unrelated). Add new scripts alongside.
- Structured configs: resolves OPEN-3 — `model` and `training` groups get dataclass schemas in `cardiac_ml/conf_schemas.py`; `data`, `experiment`, `tracking` stay free-form YAML.
- All Python files must preserve project convention `torch.float64` default (NFR-6).

---

### Step 2.1: Package __init__ files
**Model**: sonnet

#### Read First
- `cardiac_ml/README.md:29-58` — target layout.

#### Why
Python needs `__init__.py` at each package level to enable imports. Public API surface is deliberately small — only `Trainer` is re-exported from the top level.

#### Implementation Spec
**Files to create:**
- `cardiac_ml/__init__.py` — re-export `Trainer`.
- `cardiac_ml/training/__init__.py` — re-export callback classes (populated in Phase 3).
- `cardiac_ml/analysis/__init__.py` — empty stub.
- `cardiac_ml/utils/__init__.py` — empty stub.
- `cardiac_ml/tests/__init__.py` — empty stub.

#### Pseudocode
```python
# cardiac_ml/__init__.py — uses PEP 562 __getattr__ for lazy Trainer access.
# Round-2 H-6 fix: lazy access, not a bare try/except (which binds at load time).
"""cardiac_ml: project-wide ML training harness."""
__all__ = ["Trainer"]

def __getattr__(name):
    if name == "Trainer":
        # Deferred import: actual Trainer class lands in Step 3.4.
        # Before that, accessing cardiac_ml.Trainer raises ImportError with
        # a clear message pointing at the incomplete implementation.
        try:
            from cardiac_ml.training.trainer import Trainer
        except ImportError as e:
            raise ImportError(
                "cardiac_ml.Trainer is not yet implemented. "
                "See PLAN.md Step 3.4. Underlying error: " + str(e)
            ) from e
        return Trainer
    raise AttributeError(f"module 'cardiac_ml' has no attribute {name!r}")
```

Stubs for others: one-line module docstring, nothing else.

#### Test Spec
- `cardiac_ml/tests/test_imports.py::test_all_init_files_importable` — Setup: fresh Python. Expected: `importlib.import_module("cardiac_ml.training")`, `cardiac_ml.analysis`, `cardiac_ml.utils`, `cardiac_ml.tests` all succeed without error. Top-level `from cardiac_ml import Trainer` is DEFERRED until Step 3.4 (L-1 from audit — asserting an import *fails* because a feature is missing is a fragile test).

#### Checklist
- [ ] All 5 `__init__.py` files created.
- [ ] All 4 sub-packages (`training`, `analysis`, `utils`, `tests`) importable.
- [ ] Top-level `__init__.py` uses PEP 562 `__getattr__` for lazy `Trainer` access (Round-2 H-6 + Round-3 MED-5 fix — prior "try/except" language was stale). Access to `cardiac_ml.Trainer` before Step 3.4 lands raises a clear ImportError; sibling imports of `cardiac_ml.training`, `cardiac_ml.analysis`, etc. are unaffected.

#### Verify
```bash
find cardiac_ml -name __init__.py | sort
```

#### Exit Criteria
- [ ] Find command returns 5 paths.

#### Risk
- None.

---

### Step 2.2: Git utility
**Model**: sonnet

#### Read First
- FR-M2 in REQUIREMENTS.md — "Every run auto-tags: git SHA, git dirty flag".

#### Why
MLflow logger (Phase 3) needs `git_sha()` and `git_dirty()` to set as run tags. Isolate git shell-out here so the logger stays pure.

#### Implementation Spec
**Files to create:** `cardiac_ml/utils/git.py`
**Interfaces:**
```python
def git_sha(short: bool = True) -> str:
    """Return current HEAD SHA. Empty string if not a git repo."""

def git_dirty() -> bool:
    """True if working tree has uncommitted changes. False if not a git repo."""
```

#### Pseudocode
```python
import subprocess
def git_sha(short=True):
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""
def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False
```

#### Test Spec
- `cardiac_ml/tests/test_git_utils.py::test_git_sha_short` — Setup: run in repo. Expected: returns non-empty hex string of length 7–12.
- `test_git_dirty_returns_bool` — Setup: run in repo. Expected: returns `bool` (value depends on tree state).

#### Checklist
- [ ] `git_sha()` returns short SHA by default.
- [ ] `git_dirty()` returns `bool`.
- [ ] Both handle non-git-repo gracefully (empty string / False).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_git_utils.py -v
```

#### Exit Criteria
- [ ] Both tests pass.

#### Risk
- Low.

---

### Step 2.3: Seed utility
**Model**: sonnet

#### Read First
- NFR-1 in REQUIREMENTS — reproducibility.

#### Why
Trainer calls this at `fit()` start. Covers torch, numpy, python random, CUDA.

#### Implementation Spec
**Files to create:** `cardiac_ml/utils/seed.py`
**Interfaces:**
```python
def seed_everything(seed: int) -> None: ...
```

#### Pseudocode
```python
import random, numpy as np, torch
def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
```

#### Test Spec
- `cardiac_ml/tests/test_seed.py::test_seed_deterministic` — Setup: seed 42, sample torch.randn(10) twice (re-seeding between). Expected: two tensors bitwise-equal.

#### Checklist
- [ ] Function seeds all four RNG sources.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_seed.py -v
```

#### Exit Criteria
- [ ] Test passes.

#### Risk
- Low.

---

### Step 2.4: Structured config schemas
**Model**: opus

#### Read First
- REQUIREMENTS.md FR-C5 and Decision 9.2 OPEN-3.

#### Why
Dataclass schemas catch config typos at resolve time, not 3 hours into training. Apply only to `model` and `training` groups per OPEN-3 resolution (too-restrictive schemas would make `experiment` composition painful).

#### Implementation Spec
**Files to create:** `cardiac_ml/conf_schemas.py`
**Interfaces:**
```python
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class TrainingConfig:
    epochs: int
    optimizer: Any                      # _target_ spec — free-form within this
    train_step_fn: Optional[Any] = None # _target_: hydra.utils.get_method spec
    val_step_fn: Optional[Any] = None   # _target_: hydra.utils.get_method spec
    callbacks: List[Any] = field(default_factory=list)
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float64"
    # Round-3 MED-7 fix: NODE-specific ODE config fields. Use Optional so the
    # schema accepts either NODE configs (these fields present) or teacher-
    # forced configs (these absent).
    phase_name: Optional[str] = None   # A1/A2/.../ionic_state/conc_only/...
    ode_method: str = "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-3
    ode_adjoint: bool = False

@dataclass
class TrackingConfig:
    enabled: bool = True
    experiment_name: str = "default"
    tracking_uri: str = "./mlruns"
    checkpoint_every: int = 50
```

#### Pseudocode
```python
# cardiac_ml/conf_schemas.py
from hydra.core.config_store import ConfigStore

def _register():
    """Register structured configs with Hydra's ConfigStore. Called once
    at harness entry (scripts/train.py), NOT from cardiac_ml/__init__.py
    (which would force unconditional Hydra import, breaking the lazy PEP 562
    pattern in Step 2.1). Round-3 MED-6 fix."""
    cs = ConfigStore.instance()
    cs.store(group="training", name="schema", node=TrainingConfig)
    cs.store(group="tracking", name="schema", node=TrackingConfig)
```

And in `scripts/train.py`:
```python
from cardiac_ml.conf_schemas import _register
_register()  # call once before @hydra.main so ConfigStore has schemas

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg): ...
```

Note: `model` group gets NO dataclass schema at this step — model configs are user-written and too diverse for a single schema. Revisit if OPEN-3 proves too permissive.

#### Test Spec
- `cardiac_ml/tests/test_conf_schemas.py::test_training_schema_registered` — Setup: call `_register()`; query ConfigStore. Expected: `training/schema` entry exists.
- `test_invalid_training_config_raises` — Setup: Hydra resolves a YAML missing required `epochs`. Expected: `ConfigCompositionException` or equivalent.

#### Checklist
- [ ] `TrainingConfig` and `TrackingConfig` dataclasses defined.
- [ ] `_register()` registers both with Hydra ConfigStore.
- [ ] Tests verify schema enforcement.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_conf_schemas.py -v
```

#### Exit Criteria
- [ ] Both tests pass.
- [ ] OPEN-3 resolved (training + tracking get schemas; model + data + experiment stay free YAML).

#### Risk
- Hydra structured-config API changed between versions. **Mitigation**: pin from Step 1.1 survey; test against pinned version only.

---

### Step 2.5: Config tree skeleton
**Model**: opus

#### Read First
- REQUIREMENTS.md FR-C1 through FR-C6.
- Template survey `results/template_survey.md` — adopted `conf/` layout patterns.

#### Why
Establishes the config composition surface. Empty-ish YAMLs at this step — they just need to exist and compose without Hydra errors. Real content fills in during Phases 3 and 4.

#### Implementation Spec
**Files to create:**
- `conf/config.yaml` — defaults list referencing the group files below.
- `conf/model/ionic_node.yaml` — stub with just `_target_: surrogate.model.node.IonicNODE` + 2-3 required kwargs from IonicNODE signature.
- `conf/data/t1_multi_bcl.yaml` — stub referencing existing `surrogate.training.datasets.SegmentDataset`.
- `conf/training/teacher_forced.yaml` — defaults: `epochs: 10`, optimizer Adam lr=1e-3, train_step_fn pointing at `cardiac_ml.training.default_steps.teacher_forced_step`.
- `conf/training/node.yaml` — same shape, `train_step_fn` pointing at `surrogate.training.node_step.node_train_step` (file created in Phase 4).
- `conf/optimizer/adam.yaml`
- `conf/tracking/default.yaml` — `enabled: true`, `experiment_name: default`, `checkpoint_every: 50`.
- `conf/tracking/off.yaml` — `enabled: false`.
- `conf/experiment/ionic_node_t1.yaml` — composes NODE model + T1 data + node training + default tracking.
- `conf/hparams_search/lr_batch.yaml` — Optuna sweep spec (populated in Phase 5).

#### Pseudocode
`conf/config.yaml`:
```yaml
defaults:
  - model: ionic_node
  - data: t1_multi_bcl
  - training: teacher_forced
  - optimizer: adam
  - tracking: default
  - _self_
```

Group YAMLs follow Hydra idioms — `_target_`-based instantiation for model/data/optimizer; pure values for tracking; `_target_: hydra.utils.get_method` with a `path` field for `train_step_fn`.

#### Test Spec
- `cardiac_ml/tests/test_conf_compose.py::test_default_config_resolves` — Setup: `with initialize(config_path="../../conf"): cfg = compose("config")`. Expected: resolves without error, `cfg.model._target_ == "surrogate.model.node.IonicNODE"`.
- `test_experiment_override` — Setup: compose with `overrides=["experiment=ionic_node_t1"]`. Expected: `cfg.training.train_step_fn.path` contains `node_train_step`.
- `test_all_targets_importable` — Setup: compose default config; walk the DictConfig tree and collect every `_target_` string and every `path` field under a `_target_: hydra.utils.get_method` parent. For each, call `importlib.import_module(target.rsplit('.', 1)[0])`. Expected: all imports succeed, raising at the first typo. This covers H-2 from the audit — Hydra itself won't validate target paths until `instantiate()` runs, so we validate them here.

NOTE — some targets point at files not yet created (e.g. `surrogate.training.node_step.node_train_step` lands in Step 4.1). The importability walk needs a concrete exclusion mechanism (Round-2 M-6 fix):

**Mechanism**: a module-level set `cardiac_ml/tests/_deferred_targets.py::DEFERRED` lists module prefixes that are expected-missing during a phase. At Step 2.5 time it contains `{"surrogate.training.node_step"}`. At Step 4.1 exit (new checklist item below) this file is edited to remove `surrogate.training.node_step` — turning the Phase-4 check on.

```python
# cardiac_ml/tests/_deferred_targets.py
DEFERRED = {
    # Phase 4 deferred — remove at Step 4.1 exit to activate import checks.
    "surrogate.training.node_step",
}
```

Test walks all `_target_` strings, strips each to its module prefix, and skips import if the prefix (or any parent) is in `DEFERRED`. Resolves the broken cross-reference between Step 2.5 and Step 4.1 flagged in Round-2 M-6.

#### Checklist
- [ ] All 10 config files created.
- [ ] `conf/config.yaml` defaults list composes all groups.
- [ ] Three compose tests pass (one is the Phase-2-scoped importability check).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_conf_compose.py -v
```

#### Exit Criteria
- [ ] Compose tests pass.
- [ ] All `_target_` strings in the Phase 2–3 scope are importable; Phase-4-scoped targets (node_step, node configs) excluded until Step 4.1.

#### Risk
- `_target_` paths for Phase 4 files don't exist yet at Phase 2 close. **Mitigation**: split importability test by phase scope — Phase-2 check validates only the targets for files landing in Phase 2–3. Add the Phase-4 scope check as part of Step 4.1's exit.
- Hydra only resolves paths at `instantiate()` time; the importability test closes this gap manually (H-2 from audit).

---

### Phase 2 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/ -v && \
find cardiac_ml conf -name "*.py" -o -name "*.yaml" | sort && \
echo "Phase 2 OK"
```

### Phase 2 Exit Criteria
- [ ] All files from §2 Architecture Changes (phase 2 scope) exist.
- [ ] All unit tests in `cardiac_ml/tests/` pass.
- [ ] `hydra.compose("config")` resolves default config without error.
- [ ] OPEN-3 resolved.

### Phase 2 Cleanup
- Ensure no `torch.float32` tensors anywhere in new code (utils don't use tensors — verify no hidden casts).
- No duplication of utilities from existing `Surrogate/surrogate/training/` — we're NOT re-implementing data_cache, datasets, etc.
- Confirm no edits to `Monodomain/Engine_V5.3/` (validated baseline — read-only).

**→ Commit point: `git commit` after Phase 2 passes. Message: "cardiac_ml Phase 2: package skeleton + config tree"**

---

## Phase 3: Trainer + MLflow + Callbacks

**Goal**: Implement the Trainer and its callbacks end-to-end. Validate via the default `teacher_forced_step` on a synthetic MLP / synthetic data. The NODE pilot (Phase 4) must drop in with zero Trainer changes.
**Tier**: large
**Estimated scope**: 8 steps — this is the load-bearing phase.

### Phase Context

- Trainer signature is now settled per IDEALOG 2026-04-19 and REQUIREMENTS §4.3: pure-function `train_step_fn(trainer, batch) -> dict` via Hydra `_target_: hydra.utils.get_method`. NOT a subclass method.
- Return dict is the primary metric channel: every key except `"loss"` gets auto-logged to MLflow (FR-T8). Escape hatches `trainer.log_artifact(path)` and `trainer.log_figure(fig, name)` on the Trainer instance (FR-T9).
- Backward pass: Trainer calls `output["loss"].backward()` by default. If `train_step_fn` returns an already-backward-called loss (adjoint case), signal via return-dict flag `"_backward_done": True`. This resolves OPEN-6.
- MLflow mode: file-backed at `./mlruns/`. Never `log_model`; always `log_artifact(state_dict_path)`.
- Model parameters and tensors explicitly cast to `cfg.training.dtype` (default float64) via `.to(dtype=...)`. Trainer does NOT call `torch.set_default_dtype(...)` — that mutates process-global state and breaks tests for other engines running in the same pytest session (M-1 from audit).
- MLflow imports are isolated to `cardiac_ml/training/mlflow_logger.py` only. Trainer's `log_artifact` / `log_figure` escape hatches route through the logger callback (not direct mlflow calls) — M-7 / M-2 resolution.

---

### Step 3.1: Default teacher-forced step
**Model**: sonnet

#### Read First
- REQUIREMENTS.md §7.1 for the target shape.

#### Why
Default `train_step_fn` for the common case. Also serves as the reusability-proof target for the diffusion-ResNet-stub experiment in Phase 5.

#### Implementation Spec
**Files to create:** `cardiac_ml/training/default_steps.py`
**Interfaces:**
```python
def teacher_forced_step(trainer, batch) -> dict:
    """Default train_step_fn: forward → MSE → return dict."""

def teacher_forced_val_step(trainer, batch) -> dict:
    """Default val_step_fn: no-grad forward → MSE → return dict."""
```

#### Pseudocode
```python
import torch, torch.nn.functional as F
def teacher_forced_step(trainer, batch):
    x, y = batch
    pred = trainer.model(x)
    loss = F.mse_loss(pred, y)
    return {"loss": loss}

def teacher_forced_val_step(trainer, batch):
    with torch.no_grad():
        x, y = batch
        pred = trainer.model(x)
        loss = F.mse_loss(pred, y)
    return {"val_loss": loss, "loss": loss}  # "loss" required even for val
```

**Protocol keys recognized by Trainer** (all optional; omit if unused):
- `"_backward_done": bool` — set True if the step called `loss.backward()` itself (adjoint case). Trainer skips its default backward.
- `"_on_after_backward": Callable[[], None]` — post-backward cleanup (e.g., NODE's `clear_v_trajectory()`). Trainer invokes on both train and val paths.

Most `train_step_fn`s need neither flag. The default teacher-forced step above does not set either.

#### Test Spec
- `cardiac_ml/tests/test_default_steps.py::test_teacher_forced_returns_dict_with_loss` — Setup: mock model (identity), tensor pair batch. Expected: return dict has `"loss"` key, is scalar float64 tensor, requires_grad.
- `test_val_step_no_grad` — Setup: same. Expected: returned loss does NOT require grad.

#### Checklist
- [ ] Both functions implemented.
- [ ] Both return dicts with `"loss"` key.
- [ ] Val step uses `torch.no_grad()`.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_default_steps.py -v
```

#### Exit Criteria
- [ ] Both tests pass.

#### Risk
- Low.

---

### Step 3.2: Callback base class + simple callbacks
**Model**: opus

#### Read First
- REQUIREMENTS.md §4.4.

#### Why
Callback hooks isolate logger, checkpointing, early-stopping concerns from the Trainer core. Adding a new callback = new class, no Trainer changes.

#### Implementation Spec
**Files to create:** `cardiac_ml/training/callbacks.py`
**Interfaces:**
```python
class Callback:
    # Lifecycle hooks — 6 total, all no-ops in base class.
    def on_fit_start(self, trainer): ...
    def on_epoch_start(self, trainer, epoch): ...
    def on_train_batch_end(self, trainer, batch_idx, outputs): ...
    def on_val_batch_end(self, trainer, batch_idx, outputs): ...
    def on_epoch_end(self, trainer, epoch, metrics): ...
    def on_fit_end(self, trainer): ...
    # Logging proxies — called by Trainer.log_artifact / Trainer.log_figure.
    # No-op in base class; NullLogger inherits; MLflowLoggerCallback overrides.
    def log_artifact(self, path: str) -> None: ...
    def log_figure(self, fig, name: str) -> None: ...

class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min"): ...

class ModelCheckpoint(Callback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min",
                 every_n_epochs: int = 50, save_last: bool = True): ...

# NOTE: Callback base is extended with log_artifact / log_figure proxy hooks
# (no-op in the base class). See Step 3.3 Interfaces for the full base, and
# Step 3.3 MLflowLoggerCallback / NullLogger for concrete implementations.

class GradNormMonitor(Callback):
    """Logs grad_norm as a metric after each train batch."""

class LRSchedulerStep(Callback):
    def __init__(self, scheduler_target: str): ...
    # scheduler_target: Hydra _target_ path resolved lazily on first on_fit_start call.
    # The resolved scheduler is attached to trainer.optimizer and stepped in on_epoch_end.
    # If None (or omitted), callback is a no-op (L-6 resolution — scheduler optional).
```

#### Pseudocode
Base class has no-op hooks (subclasses override). `EarlyStopping` tracks best monitor value + counter; at `on_epoch_end`, if no improvement for `patience` epochs, sets `trainer.should_stop = True`. `ModelCheckpoint` writes `best.pt` / `last.pt` / periodic `epoch_{N}.pt` to the Hydra working dir, then calls `trainer.log_artifact(path)`.

#### Test Spec
- `cardiac_ml/tests/test_callbacks.py::test_early_stopping_triggers` — Setup: feed degrading val_loss for patience+1 epochs. Expected: `trainer.should_stop` becomes True.
- `test_model_checkpoint_best_on_improvement` — Setup: feed improving val_loss. Expected: `best.pt` written each time, `trainer.log_artifact` called.
- `test_grad_norm_logged` — Setup: backward on a toy model. Expected: `grad_norm` appears in metrics dict.

#### Checklist
- [ ] Base `Callback` with 6 hooks.
- [ ] 4 concrete callbacks.
- [ ] Tests for each.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_callbacks.py -v
```

#### Exit Criteria
- [ ] All callback tests pass.

#### Risk
- `on_fit_end` misses final checkpoint if early-stop fires. **Mitigation**: `ModelCheckpoint.on_fit_end` always writes `last.pt` regardless.

---

### Step 3.3: MLflow logger callback
**Model**: opus

#### Read First
- REQUIREMENTS.md §4.2 FR-M1 through FR-M6.
- `cardiac_ml/utils/git.py` (from Step 2.2).

#### Why
Single callback handles all MLflow concerns: start run at fit start, log params, log metrics per epoch, set git tags, end run at fit end. Trainer does NOT import mlflow directly — the callback is the only place.

Also implements `NullLogger` for the `tracking=off` case (resolves OPEN-8).

#### Implementation Spec
**Files to create:** `cardiac_ml/training/mlflow_logger.py`
**Interfaces:**
```python
class MLflowLoggerCallback(Callback):
    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"): ...
    # Hooks: on_fit_start (start_run, set_tags, log_params),
    #        on_epoch_end (log_metrics),
    #        on_fit_end (end_run).
    # Proxies (override base): log_artifact (→ mlflow.log_artifact),
    #                          log_figure   (→ mlflow.log_figure).

class NullLogger(Callback):
    """All Callback hooks AND the log_artifact / log_figure proxies are no-ops.
    Used when cfg.tracking.enabled=false so Trainer can dispatch uniformly
    without checking tracking state on every call (M-2 / M-7 resolution).
    Inherits all methods from Callback base; the base's log_artifact /
    log_figure are already no-ops, so NullLogger needs no overrides."""
```

#### Pseudocode
```python
import sys
import mlflow
import torch
from omegaconf import OmegaConf
from cardiac_ml.utils.git import git_sha, git_dirty

def _flatten(d, prefix="", sep="."):
    """Flatten nested dict/list config for mlflow.log_params."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{sep}{k}" if prefix else str(k), sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]", sep))
    else:
        out[prefix] = d if d is None or isinstance(d, (str, int, float, bool)) else str(d)
    return out

def _is_scalar(v):
    """Scalar metric test: 0-d tensor or plain number."""
    if torch.is_tensor(v):
        return v.numel() == 1
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class MLflowLoggerCallback(Callback):
    def __init__(self, experiment_name, tracking_uri="./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    def _derive_run_name(self, trainer) -> str:
        # Round-3 C-4 fix: cfg.experiment may be a bare group without a .name
        # field (Hydra experiment YAMLs often use `# @package _global_` and
        # lack top-level name). Fall back to the experiment group key if set,
        # then to the Hydra config name, then to a git-SHA-derived default.
        exp = trainer.cfg.get("experiment", {})
        name = None
        if isinstance(exp, str):
            name = exp
        elif hasattr(exp, "name"):
            name = exp.get("name") if hasattr(exp, "get") else getattr(exp, "name", None)
        if not name:
            # Try Hydra's HydraConfig for the experiment choice
            try:
                from hydra.core.hydra_config import HydraConfig
                choices = HydraConfig.get().runtime.choices
                name = choices.get("experiment") or "cardiac_ml_run"
            except Exception:
                name = "cardiac_ml_run"
        return f"{name}_{git_sha()}"

    def on_fit_start(self, trainer):
        self._run = mlflow.start_run(run_name=self._derive_run_name(trainer))
        mlflow.set_tag("git.sha", git_sha())
        mlflow.set_tag("git.dirty", str(git_dirty()))
        mlflow.set_tag("python.version", sys.version.split()[0])
        mlflow.set_tag("torch.version", torch.__version__)
        mlflow.log_params(_flatten(OmegaConf.to_container(trainer.cfg, resolve=True)))
    def on_epoch_end(self, trainer, epoch, metrics):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_scalar(v)}, step=epoch)
    def on_fit_end(self, trainer):
        mlflow.end_run()
    # Proxy overrides — Trainer.log_artifact / log_figure dispatch here.
    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)
    def log_figure(self, fig, name: str) -> None:
        mlflow.log_figure(fig, name)


class NullLogger(Callback):
    """All methods inherit base Callback no-ops (hooks + log_artifact + log_figure).
    No overrides needed — explicit empty subclass documents intent."""
    pass
```

#### Test Spec
- `cardiac_ml/tests/test_mlflow_logger.py::test_run_creation_sets_tags` — Setup: fresh `./mlruns` temp dir, tiny Trainer with MLflowLoggerCallback, fit 2 epochs. Expected: MLflow run exists with `git.sha`, `git.dirty`, `python.version` tags.
- `test_log_metrics_per_epoch` — Setup: same. Expected: MLflow run has `loss` metric with 2 data points.
- `test_null_logger_no_writes` — Setup: Trainer with NullLogger only; check `./mlruns` stays empty. Expected: no directory created.

#### Checklist
- [ ] `MLflowLoggerCallback` implemented.
- [ ] `NullLogger` implemented.
- [ ] Git SHA + dirty flag tagged.
- [ ] Per-epoch metrics logged.
- [ ] Tests pass using a temp `./mlruns` dir.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_mlflow_logger.py -v
```

#### Exit Criteria
- [ ] All 3 tests pass.
- [ ] No leaked `./mlruns/` in repo after tests.

#### Risk
- MLflow run leakage if tests crash before `end_run()`. **Mitigation**: use `pytest` fixtures with try/finally.

---

### Step 3.4: Trainer class
**Model**: opus

#### Read First
- REQUIREMENTS.md §4.3 — all 9 FR-Ts.
- IDEALOG.md 2026-04-19 Session 2 — decisions A + B.

#### Why
The central class. Must accept a Hydra DictConfig, instantiate model / data / optimizer / callbacks via `_target_`, load `train_step_fn` / `val_step_fn` via Hydra `hydra.utils.get_method`, and run the fit loop with callback hooks at the right points.

#### Implementation Spec
**Files to create:** `cardiac_ml/training/trainer.py`
**Interfaces:**
```python
class Trainer:
    def __init__(self, cfg: DictConfig): ...
        # Instantiates model, train/val loaders, optimizer, callbacks from cfg.
        # Loads train_step_fn (default teacher_forced) and val_step_fn via get_method.
        # Sets device, dtype (float64), seed.
    def fit(self) -> None: ...
        # Full epoch + batch loop. Calls callback hooks.
        # For each batch: out = self._train_step_fn(self, batch)
        #                 if not out.get("_backward_done"): out["loss"].backward()
        #                 self.optimizer.step(); self.optimizer.zero_grad()
    def log_artifact(self, path: str) -> None: ...  # escape hatch (FR-T9)
    def log_figure(self, fig, name: str) -> None: ...  # escape hatch (FR-T9)
    # Attributes: model, train_loader, val_loader, optimizer, callbacks,
    #            cfg, device, dtype, current_epoch, current_batch, should_stop
```

#### Pseudocode
```python
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg.training.seed)
        self.dtype = getattr(torch, cfg.training.dtype)  # float64 per NFR-6
        self.device = torch.device(cfg.training.device)
        # Cast model explicitly — do NOT mutate global dtype (M-1)
        self.model = instantiate(cfg.model).to(device=self.device, dtype=self.dtype)
        self.train_loader = instantiate(cfg.data.train)
        self.val_loader = instantiate(cfg.data.val) if cfg.data.get("val") else None
        self.optimizer = instantiate(cfg.training.optimizer, params=self.model.parameters())
        self._train_step_fn = instantiate(cfg.training.train_step_fn) \
                              if cfg.training.get("train_step_fn") \
                              else teacher_forced_step
        self._val_step_fn = instantiate(cfg.training.val_step_fn) \
                            if cfg.training.get("val_step_fn") \
                            else teacher_forced_val_step
        self.callbacks = [instantiate(c) for c in cfg.training.get("callbacks", [])]
        # Logger is ALWAYS a callback — MLflowLoggerCallback or NullLogger.
        # Keep a reference so log_artifact/log_figure can dispatch to it without
        # importing mlflow in Trainer (M-7).
        #
        # Round-3 HIGH-2 fix: DEDUP. If the user's config already registers
        # an MLflowLoggerCallback / NullLogger via cfg.training.callbacks,
        # don't double-attach — would cause two mlflow.start_run() calls.
        existing_logger = next(
            (cb for cb in self.callbacks
             if isinstance(cb, (MLflowLoggerCallback, NullLogger))),
            None)
        if existing_logger is not None:
            self._logger = existing_logger
        else:
            if cfg.tracking.enabled:
                self._logger = MLflowLoggerCallback(
                    cfg.tracking.get("experiment_name", "default"),
                    tracking_uri=cfg.tracking.get("tracking_uri", "./mlruns"))
            else:
                self._logger = NullLogger()
            self.callbacks.append(self._logger)
        self.current_epoch = 0
        self.should_stop = False

    def fit(self):
        for cb in self.callbacks: cb.on_fit_start(self)
        for epoch in range(self.cfg.training.epochs):
            if self.should_stop: break
            self.current_epoch = epoch
            for cb in self.callbacks: cb.on_epoch_start(self, epoch)
            train_metrics = self._run_epoch(train=True)
            val_metrics = self._run_epoch(train=False) if self.val_loader else {}
            metrics = {**train_metrics, **val_metrics}
            for cb in self.callbacks: cb.on_epoch_end(self, epoch, metrics)
        for cb in self.callbacks: cb.on_fit_end(self)

    def _run_epoch(self, train):
        step_fn = self._train_step_fn if train else self._val_step_fn
        self.model.train(train)
        accum = defaultdict(list)
        for batch_idx, batch in enumerate(self.train_loader if train else self.val_loader):
            # Round-2 M-4 + Round-3 HIGH-4: batch casting promotes BOTH device
            # AND dtype. Handles heterogeneous dict batches (SegmentDataset yields
            # dict of tensors; non-float tensors keep their dtype — only floating-
            # point tensors get promoted).
            #
            # def _to_device_and_dtype(batch, device, dtype):
            #     if torch.is_tensor(batch):
            #         return (batch.to(device=device, dtype=dtype)
            #                 if batch.is_floating_point()
            #                 else batch.to(device=device))
            #     if isinstance(batch, dict):
            #         return {k: _to_device_and_dtype(v, device, dtype) for k, v in batch.items()}
            #     if isinstance(batch, (list, tuple)):
            #         t = type(batch)
            #         return t(_to_device_and_dtype(v, device, dtype) for v in batch)
            #     return batch  # pass-through for non-tensor metadata
            batch = _to_device_and_dtype(batch, self.device, self.dtype)
            out = step_fn(self, batch)
            backward_done = out.get("_backward_done", False)
            if train:
                loss = out["loss"]
                if backward_done:
                    # Round-2 M-9 fix: clearer message. Catches the "user returned
                    # a fresh detached tensor instead of the real loss" case.
                    assert loss.requires_grad or loss.grad_fn is not None, \
                        "_backward_done=True but loss is not attached to any compute graph. " \
                        "Did you return a constant tensor instead of the loss used in backward()?"
                else:
                    assert loss.requires_grad, \
                        "train_step_fn returned loss with requires_grad=False"
                    loss.backward()
                # Step 4.1 H-1: post-backward hook for stateful cleanup (e.g., NODE's
                # clear_v_trajectory). Optional — most step fns omit this key.
                # Round-3 M-9 fix: wrap in try/except so a hook exception clears
                # grads + re-raises, instead of silently skipping optimizer.step()
                # and letting stale grads accumulate on the next batch.
                post_hook = out.get("_on_after_backward")
                if post_hook is not None:
                    try:
                        post_hook()
                    except Exception as e:
                        self.optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(
                            f"_on_after_backward hook raised at epoch "
                            f"{self.current_epoch} batch {batch_idx}: "
                            f"{type(e).__name__}: {e}") from e
                self.optimizer.step(); self.optimizer.zero_grad()
            else:
                assert not backward_done, \
                    "_backward_done=True returned from val_step_fn — flag is meaningless on val path"
                # Val path still honors the cleanup hook — V_traj / other stateful
                # caches must be cleared regardless of gradient mode.
                post_hook = out.get("_on_after_backward")
                if post_hook is not None:
                    post_hook()
            for k, v in out.items():
                if k.startswith("_"): continue  # skip protocol flags (_backward_done, _on_after_backward)
                # M-3: default reduction is mean across batches. Keys ending in "_sum"
                # or "_last" are reserved for future callback-based reduction override.
                accum[k].append(float(v) if torch.is_tensor(v) else v)
            for cb in self.callbacks:
                (cb.on_train_batch_end if train else cb.on_val_batch_end)(self, batch_idx, out)
        prefix = "train_" if train else "val_"
        return {f"{prefix}{k}": np.mean(v) for k, v in accum.items()}

    # M-2 / M-7: escape hatches route through logger callback. Trainer itself
    # does NOT import mlflow. NullLogger inherits log_artifact / log_figure as
    # no-ops from Callback base (Round-2 C-3 fix — methods defined on base class).
    # MLflowLoggerCallback overrides both. So both tracking=on and tracking=off
    # paths are a safe method dispatch.
    def log_artifact(self, path: str) -> None:
        self._logger.log_artifact(path)

    def log_figure(self, fig, name: str) -> None:
        self._logger.log_figure(fig, name)
```

#### Test Spec
- `cardiac_ml/tests/test_trainer_synthetic.py::test_fit_converges_on_identity_task` — Setup: synthetic MLP (2 hidden, 8 units, float64), 128 samples of `y = x @ W` for fixed W, Trainer with default `teacher_forced_step`, 20 epochs, Adam lr=1e-2. Expected: `val_loss < 1e-3`.
- `test_backward_done_skipped_by_trainer` — Setup: custom `train_step_fn` that calls `loss.backward()` itself and returns `{"loss": loss, "_backward_done": True}`. Expected: no double-backward error, weights update after step.
- `test_backward_done_with_no_grad_loss_raises` — Setup: `train_step_fn` returns `{"loss": torch.tensor(1.0), "_backward_done": True}` where loss has no grad. Expected: AssertionError with helpful message (H-1).
- `test_loss_requires_grad_on_train_path` — Setup: `train_step_fn` returns `{"loss": loss}` with `requires_grad=False`. Expected: AssertionError.
- `test_backward_done_on_val_path_raises` — Setup: `val_step_fn` returns `{"loss": loss, "_backward_done": True}`. Expected: AssertionError (val path doesn't backward).
- `test_model_dtype_is_float64_without_global_mutation` — Setup: record `torch.get_default_dtype()` before Trainer init; init Trainer; check global dtype unchanged AND `next(trainer.model.parameters()).dtype == torch.float64` (M-1).
- `test_log_artifact_routes_through_logger` — Setup: mock `_logger.log_artifact`, call `trainer.log_artifact("foo.pt")`. Expected: mock called once with "foo.pt"; no `mlflow.log_artifact` called directly (M-7).
- `test_log_artifact_with_null_logger_no_op` — Setup: `tracking.enabled=false` Trainer; call `trainer.log_artifact("foo.pt")`. Expected: returns None, no filesystem side effect, no exception (M-2).

#### Checklist
- [ ] Trainer class with `__init__`, `fit`, `_run_epoch`, `log_artifact`, `log_figure`.
- [ ] `_to_device_and_dtype` helper implemented with heterogeneous-batch handling (Round-3 HIGH-4 fix — float-only promotion).
- [ ] Handles `_backward_done` flag (OPEN-6 resolution).
- [ ] Handles `_on_after_backward` hook, wrapped in try/except that clears grads + re-raises on hook failure (Round-3 M-9 fix).
- [ ] Logger dedup: if user-config callbacks list already contains an MLflowLoggerCallback / NullLogger, Trainer reuses it instead of appending a second (Round-3 HIGH-2 fix).
- [ ] Default `teacher_forced_step` used when `train_step_fn` not provided.
- [ ] MLflowLogger auto-added when `tracking.enabled=true`, NullLogger when false — unless already present via config.
- [ ] All 8 tests pass (Round-3 MED-4 fix — prior "3 tests" count was stale).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_trainer_synthetic.py -v
```

#### Exit Criteria
- [ ] Tests pass.
- [ ] Convergence test hits `val_loss < 1e-3` in ≤ 20 epochs.

#### Risk
- **MEDIUM**: `_run_epoch` averages metrics across batches with `np.mean`. Documented as default reduction (M-3). Future work: callback-based override for sum/last reductions, keyed on metric name suffix (e.g. `*_sum`, `*_last`). Not implemented yet; log a warning if a reserved suffix is observed.
- **MEDIUM**: `MLflowLoggerCallback` is always constructed even in tracking=off? NO — Step 3.4 only instantiates `NullLogger` in the off path. Confirmed in pseudocode.
- **LOW**: `torch.set_default_dtype` removal means callers that rely on global float64 (e.g. ad-hoc `torch.tensor(...)` in a `train_step_fn` without explicit dtype) will get float32. Document in Step 4.1 / node_step.py that all tensor construction inside step functions must specify dtype.

---

### Step 3.5: scripts/train.py entry point
**Model**: sonnet

#### Read First
- REQUIREMENTS.md §4.7 FR-SC1.

#### Why
The 5-line user-facing entry point. This is the full interface users see.

#### Implementation Spec
**Files to create:** `scripts/train.py`
**Interfaces:** `@hydra.main(config_path="../conf", config_name="config", version_base=None)`

#### Pseudocode
```python
import hydra
from omegaconf import DictConfig
from cardiac_ml import Trainer

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    Trainer(cfg).fit()

if __name__ == "__main__":
    main()
```

#### Test Spec
- Manual: `conda run -n heart-conduction python scripts/train.py --help` should print Hydra help.
- `cardiac_ml/tests/test_scripts.py::test_train_script_help_exits_zero` — Setup: subprocess. Expected: exit 0.

#### Checklist
- [ ] File exists at `scripts/train.py`.
- [ ] `@hydra.main` decorator with `version_base=None`.
- [ ] `--help` exits 0.

#### Verify
```bash
# L-2: --help only checks argparse layer; also run --cfg job to verify the
# full Hydra composition resolves.
conda run -n heart-conduction python scripts/train.py --help && \
conda run -n heart-conduction python scripts/train.py --cfg job > /tmp/composed_cfg.yaml && \
test -s /tmp/composed_cfg.yaml && \
grep -q "_target_" /tmp/composed_cfg.yaml
```

#### Exit Criteria
- [ ] `--help` exits 0 with Hydra-composed help text.
- [ ] `--cfg job` dumps a non-empty composed config to stdout.
- [ ] The composed config contains at least one `_target_` key (proves instantiation targets resolved into the composition).

#### Risk
- Low.

---

### Step 3.6: End-to-end synthetic test
**Model**: opus

#### Read First
- `scripts/train.py` from Step 3.5.

#### Why
Proves the whole stack end-to-end before we hand it a real model. If a 4-param linear-regression-equivalent task can train under the harness, the NODE pilot's only remaining risk is model-specific.

#### Implementation Spec
**Files to create:**
- `cardiac_ml/tests/conftest.py` — pytest fixture `mlflow_tmpdir` with `@pytest.fixture(autouse=True, scope="session")` that sets `MLFLOW_TRACKING_URI` env var and `mlflow.set_tracking_uri(...)` to a per-session `tmp_path_factory.mktemp("mlruns")` (M-4 + Round-3 MED-10 fix — `autouse=True` is REQUIRED for automatic pickup, otherwise tests without explicit injection pollute real `./mlruns/`). Subprocess tests (Step 3.6 end-to-end) must still pass `tracking.tracking_uri=./mlruns_test` explicitly via CLI override — the subprocess is a separate Python process and doesn't inherit pytest fixtures.
- `cardiac_ml/tests/test_end_to_end.py` — spawns `scripts/train.py` as subprocess with a synthetic experiment.
- `conf/experiment/synthetic_smoke.yaml` — composes a trivial MLP + synthetic data + `teacher_forced` training for 5 epochs.
- `conf/model/synthetic_mlp.yaml` — 2-layer MLP, `_target_` points to a tiny test-only model class.
- `conf/data/synthetic_linear.yaml` — `_target_` points to test-only dataset.
- `cardiac_ml/tests/synthetic_harness_fixtures.py` — defines the tiny model + dataset classes referenced by the above YAMLs.

#### Pseudocode
Synthetic fixture: `y = x @ W + b` for fixed float64 `W`, `b`. Trainer fits for 5 epochs. Test asserts:
1. Exit code 0.
2. `./mlruns/` was created.
3. At least one run directory exists.
4. `best.pt` artifact exists in the run.
5. Per-epoch `loss` metric logged.

#### Test Spec
- `test_synthetic_end_to_end` — exercises the full script via `subprocess.run`.

#### Checklist
- [ ] Synthetic model + dataset classes exist.
- [ ] `synthetic_smoke.yaml` composes.
- [ ] `scripts/train.py experiment=synthetic_smoke` runs to completion.
- [ ] MLflow run produced with metrics + artifacts.

#### Verify
```bash
rm -rf mlruns_test && \
conda run -n heart-conduction python scripts/train.py experiment=synthetic_smoke \
  tracking.tracking_uri=./mlruns_test hydra.run.dir=./outputs_test/\${now:%H%M%S} && \
ls mlruns_test && rm -rf mlruns_test outputs_test
```

#### Exit Criteria
- [ ] Verify command exits 0.
- [ ] `mlruns_test/` contains a run with at least one artifact.

#### Risk
- Test pollutes top-level `./mlruns/` if override fails. **Mitigation**: always use `mlruns_test/` in this test; clean up at end.

---

### Phase 3 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/ -v && \
conda run -n heart-conduction python scripts/train.py --help > /dev/null && \
echo "Phase 3 OK"
```

### Phase 3 Exit Criteria
- [ ] All `cardiac_ml/tests/` pass.
- [ ] End-to-end synthetic training produces valid MLflow run.
- [ ] `import mlflow` appears ONLY in `cardiac_ml/training/mlflow_logger.py` (and optionally `conftest.py` for the tmpdir fixture). Trainer stays mlflow-free via logger dispatch.
- [ ] `torch.set_default_dtype` NOT called anywhere in `cardiac_ml/` (M-1).
- [ ] OPEN-6 resolved (`_backward_done` flag with assertions).
- [ ] OPEN-8 resolved (NullLogger + `tracking=off` path).

### Phase 3 Cleanup
- `grep -rn "torch.float32" cardiac_ml/` — expect zero hits.
- `grep -rn "import mlflow" cardiac_ml/` — expect hits ONLY in `mlflow_logger.py` and optionally `tests/conftest.py`. NOT in `trainer.py` (M-7).
- `grep -rn "set_default_dtype" cardiac_ml/` — expect zero hits (M-1).
- Confirm V5.3 untouched.
- No code copied from `Surrogate/surrogate/training/`; we import from it only if needed (shouldn't be needed for synthetic tests).

**→ Commit point: `git commit` after Phase 3 passes. Message: "cardiac_ml Phase 3: Trainer + MLflow + callbacks + synthetic smoke test"**

---

## Phase 4: Ionic NODE Pilot Migration

> **⚠ Phase 4 Steps 4.1–4.4 are DEFERRED pending Step 4.0 completion.** Round-3 audit (2026-04-19) found 4 critical and 3 high issues in the current Phase 4 specification, all rooted in assumptions about `SegmentDataset` / `make_dataloaders` / `phase_name` / `cfg.experiment.name` / parity-gate infrastructure that have NOT been verified against the actual codebase. Step 4.0 is an exploratory step that produces a reality-check note; 4.1–4.4 must be re-specified via `/blueprint-revise` after 4.0 lands.

**Goal**: Run the ionic NODE training under the new harness. Achieve parity with surrogate_pipeline Session 25 baseline (val ≤ 1.05 × oracle best = 0.0088 on multi-BCL T1). This is the hard success criterion — if parity fails, the harness design is suspect, not the training config.
**Tier**: large
**Estimated scope**: 1 exploratory step (4.0) + 4 implementation steps (4.1-4.4, to be re-specified). Depends entirely on Phase 3 being solid.

### Phase Context

- Reference implementation: `Surrogate/surrogate/training/train_node.py` (311 lines, argparse-based), with the rollout logic in `node_rollout.py` (193 lines).
- Ionic model class: `surrogate.model.node.IonicNODE`. See `Surrogate/surrogate/model/node.py` for constructor signature — required kwargs go in `conf/model/ionic_node.yaml`.
- **Actual node_rollout.py API** (re-verified 2026-04-19 against source):
  - `node_rollout(node, segment, phase_name="A1", device=None, t_eval_ms=None, z0_noise_sigma=0.0, adjoint=False, method="dopri5", rtol=1e-3, atol=1e-3) -> dict` at line 54.
  - **Returns a dict**, not a scalar: `{"loss": mean_loss, "ionic_state_mse": ..., "conc_mse": ..., "conductance_mse": ...}`. Per-component keys depend on `phase_name`. Adapter MUST extract `result["loss"]`.
  - **`phase_name` valid values**: `"A1" | "A2" | "A3" | "A4" | "ionic_state"` (ionic-only loss), `"conc_only"` (concentration-only), `"B1" | "B2" | "B3" | "B4" | "ionic_state_and_conductance"` (full), `"C" | "D" | "I_ion"` (NotImplementedError — requires Stage2 wiring). Default `"A1"`. This is the LOSS-COMPOSITION selector, NOT a train/val split. Session 25's multi-BCL parity run used `phase_name="A1"` (equivalent to `"ionic_state"`).
  - **Caller MUST call `node.clear_v_trajectory()` after `loss.backward()`** (per node_rollout.py:68-69, 81 docstring). The V_traj is set on the node object before integration and must be cleared after backward traverses the graph. Under `adjoint=False` this is mandatory even though the graph stays in-Python, because adjoint-mode backwards (future toggle) would re-enter the forward.
  - `node.integrate(z0, t, method=..., rtol=..., atol=..., adjoint=False)` at node.py:60 — the solver wrapper. Defaults match Session 23's working config.
  - **No `prepare_rollout_batch` helper, no `compute_landmark_loss` public helper, no `.nfe` attribute on IonicNODE** — all Round-1 and Round-2 audit findings that the plan must obey.
- Dataset class: `surrogate.training.datasets.SegmentDataset`. The raw data lives at `/media/HDD/norepinephrine/surrogate_data/raw/` (HDD, verified by `ls` 2026-04-19). SSD cache at `/tmp/surrogate_cache/` per MEMORY.md. Do NOT copy or move raw data.
- Parity target: val_loss ≤ 0.008 on multi-BCL T1 (train BCLs {300,500,700,1000,1500}, val BCLs {400,600,800,2000}). This is from Session 25 IDEALOG entry.
- NODE uses `dopri5`, `rtol=atol=1e-3`, `adjoint=False` per Session 23 IDEALOG (H-6 from audit — REQUIREMENTS §7.2 dopri8 sketch is misleading; real code uses dopri5). Parity target achieved WITHOUT the adjoint method.
- NODE `train_step_fn` must NOT modify `node_rollout.py` or `node.py`. Instead, `node_step.py` is a new file that calls the existing `node_rollout()` function and adapts the return shape to the Trainer's `{"loss": ...}` contract.

---

### Step 4.0: Phase 4 Reality Check
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/datasets.py` (full — ~200 lines)
- `Surrogate/surrogate/training/data_cache.py` (full)
- `Surrogate/surrogate/training/train_node.py` (full — 311 lines; pay special attention to `make_dataloaders`, argparse defaults, and the full `for epoch` loop)
- `Surrogate/surrogate/training/node_rollout.py` (full — 193 lines; verified in Round 2 but re-read under Phase-4 lens)
- `Surrogate/surrogate/training/phases.py` — phase constants, `_HALF1_PARAMS` / `_HALF2_PARAMS`
- `Surrogate/surrogate/training/training_log.py` — what JSONL fields get written, how
- `Surrogate/runs/multi_bcl_002/log.jsonl` — count epochs, inspect schema, note `elapsed_s` trajectory for wall-time estimation
- `Surrogate/surrogate/model/node.py` (full — IonicNODE with `integrate`, `set_v_trajectory`, `clear_v_trajectory`, `euler_step`)
- `Surrogate/surrogate/model/stage1.py` — verify scaffold decoders exist at pinned SHA

#### Why
Round-3 adversarial audit found that Phase 4's current Step 4.1–4.4 specification rests on multiple invented or incorrect assumptions:
- `conf/data/t1_multi_bcl.yaml`'s `train_bcls` / `val_bcls` fields don't exist in `SegmentDataset` (real pipeline uses `make_dataloaders(tier)` over pre-split `tier{N}_train.pt` / `tier{N}_val.pt` cache files).
- `conf/training/node.yaml` is missing the required `phase_name` field → KeyError at first batch.
- `SessionParityGate` reads a JSONL current-log that the new harness doesn't produce (Trainer logs to MLflow).
- Reference log only covers 8 epochs (0–7); gate's `min_epoch=10` means it never fires on real data.
- `cfg.experiment.name` is used in `MLflowLoggerCallback.on_fit_start` but no experiment YAML is spec'd to populate it.

Rather than patch these individually without checking the actual code, this step produces a single reality-check document that records verified facts + concrete recommendations. Steps 4.1–4.4 get re-specified (via `/blueprint-revise`) based on the output.

#### Implementation Spec
**Files to create:** `Research/Active/cardiac_ml_harness/results/phase4_reality_check.md`
**Files to modify:** none.

#### Pseudocode
Walk the NODE training pipeline and produce a document with one section per topic. Every finding cites a file:line. End with a "Revisions needed" section listing concrete changes to apply to Phase 4 before execution.

Sections:

1. **Dataset API.** `SegmentDataset.__init__(...)` signature and kwargs. `__getitem__` return shape (keys present in the dict, dtype of each). Does `Vm` come pre-cast to float64? Are `dt`, `ionic_states`, `concentrations`, `conductance_products` present, matching what `node_rollout` reads? Is there any BCL filtering?
2. **Dataloader construction.** `make_dataloaders(tier, ...)` — where does it live (`train_node.py`? separate module?), what tier files does it read, how does it split train/val. How does the Hydra config need to map onto this?
3. **Phase naming for multi_bcl_002.** Trace from `Surrogate/runs/multi_bcl_002/` — is there a `config.yaml`, `args.json`, or similar recording the invocation? If not, inspect `log.jsonl` header / per-epoch keys to infer what was being trained. If still ambiguous, flag that Step 4.2 must test `phase_name="A1"` vs `"ionic_state"` empirically.
4. **Parity gate source.** Does `train_node.py` write `log.jsonl`? Via what module? Can the harness reproduce the same format cheaply (shadow-log callback), or should `SessionParityGate` read from MLflow instead? Record which side of the choice is less code.
5. **Reference log coverage.** How many epochs in `multi_bcl_002/log.jsonl`? Does the reference cover enough range for a 3×/5× tolerance check beyond epoch 10? If not, what alternative gate design covers epochs 0–7 only (e.g., "val_loss at epoch 7 ≤ 1.2 × oracle")?
6. **`cfg.experiment.name` population.** Run `conda run -n heart-conduction python -c "from hydra import initialize, compose; ..."` once the conf tree exists (Phase 2), or reason from Hydra docs: does an experiment YAML in a `conf/experiment/` subgroup automatically get a `name` field, or must each YAML declare one? Specify in detail so Step 3.3 logger doesn't crash.
7. **Wall-time budget.** Extrapolate `elapsed_s` from `multi_bcl_002/log.jsonl` to 500 epochs (or to the epoch count at which oracle reached 0.00838). Document the total hours. Update Phase 4 Context scope accordingly.
8. **V-trajectory lifecycle.** Inspect `node.set_v_trajectory` / `clear_v_trajectory`. Must `clear_v_trajectory` run between *batches* (per the `node_rollout` docstring), or is it safe to defer to end-of-epoch? Confirm the `_on_after_backward` hook pattern matches the real requirement.
9. **SHA pin robustness.** Run `git log --oneline 8f191f77..HEAD -- Surrogate/surrogate/model/` — has the model tree changed since the pin? If the pin is stale against current HEAD, document what the actual correct pin is and update REQUIREMENTS references.
10. **Scaffold decoder presence at pinned SHA.** `git show 8f191f77:Surrogate/surrogate/model/stage1.py | grep -E "ionic_state_decoder|gate_conductance_decoder"` — confirm the decoders exist at 8f191f77 and produce the metric keys the adapter plans to surface.

Final section: "Revisions needed before Step 4.1 is executable." Enumerate specific changes to Phase 4 Context, Step 4.1 pseudocode, Step 4.2 YAML, Step 4.3 smoke config, Step 4.4 parity gate. This is the input to the next `/blueprint-revise` pass.

#### Test Spec
N/A (doc-producing step).

#### Checklist
- [ ] `phase4_reality_check.md` exists with all 10 topic sections.
- [ ] Each finding cites a concrete file:line reference.
- [ ] "Revisions needed" section lists ≥5 concrete changes to Phase 4 steps.
- [ ] SHA pin drift checked; if stale, new pin + rationale documented.
- [ ] Wall-time budget recorded (hours, not "reasonable").

#### Verify
```bash
test -f Research/Active/cardiac_ml_harness/results/phase4_reality_check.md && \
wc -l Research/Active/cardiac_ml_harness/results/phase4_reality_check.md && \
grep -qc "^## " Research/Active/cardiac_ml_harness/results/phase4_reality_check.md
```

#### Exit Criteria
- [ ] File exists and is ≥ 100 lines.
- [ ] ≥ 10 second-level `##` section headings (one per topic).
- [ ] "Revisions needed" section enumerates concrete Step-4.x changes.
- [ ] This step's findings drive a `/blueprint-revise` call BEFORE Step 4.1 is attempted. Do not proceed to 4.1 from the current (pre-reality-check) specification.

#### Risk
- **HIGH** — reality check may surface architectural mismatches that require more than a revise pass (e.g., if `node_rollout` can't be called from the harness without modifying `node.py`). Mitigation: document the mismatch, escalate to user, potentially split Phase 4 into Phase 4' (adapter + pilot) and Phase 4'' (parity run) if adapter work grows.
- **MEDIUM** — the exploratory note may conclude that parity against Session 25's marginal 0.00838 is unrealistic given natural training variance. If so, propose an oracle-independent success criterion (e.g., "val_loss ≤ 0.02 on multi-BCL T1 after 100 epochs") as the gate.

---

### Step 4.1: node_train_step function  [DEFERRED — pending Step 4.0]
**Model**: opus

> **⚠ This step's current specification has known critical issues** (Round-3 audit: `phase_name` KeyError in Step 4.2 YAML, phase-ordering violation with Step 3.4's addendum, duplicate logger risk). Do NOT execute as written. Re-spec via `/blueprint-revise` using the output of Step 4.0 before execution.


#### Read First
- `Surrogate/surrogate/training/node_rollout.py:54-193` — `node_rollout()` signature and return shape; `_compute_node_loss()` for landmark-loss internals (do NOT re-implement).
- `Surrogate/surrogate/model/node.py:60-98` — `IonicNODE.integrate()` signature and `euler_step()` inference path.
- `Surrogate/surrogate/training/train_node.py:50-180` — how `node_rollout()` is currently driven (batch shape, phase_name semantics).
- Session 23 in `Research/Active/surrogate_pipeline/IDEALOG.md` — `adjoint=False` with `dopri5/rtol=atol=1e-3` is the oracle config.

#### Why
The pure-function entry point that wires the existing NODE machinery into the new Trainer. Key constraint: `node_rollout()` IS the oracle — do not bypass it by reconstructing the time grid, V_traj set, or loss computation. Call it as-is, then unpack the `"loss"` key from its dict return, clear the V-trajectory on the node, and return `{"loss": loss}` to the Trainer.

Required protocol per `node_rollout()` docstring:
1. Call `node_rollout(...)` — returns dict containing `"loss"` plus per-component scaffold losses.
2. Extract `loss = result["loss"]`.
3. Hand loss to Trainer; Trainer calls `loss.backward()` in `_run_epoch` default path.
4. AFTER backward, caller MUST call `node.clear_v_trajectory()`. Adapter cannot do this inline because backward hasn't happened yet when the `train_step_fn` returns. **Solution**: adapter returns `{"loss": loss, "_on_after_backward": lambda: trainer.model.clear_v_trajectory()}` and Trainer invokes the callback after `loss.backward()`. This extends the Trainer's protocol — see Step 3.4 addendum.

NFE is NOT tracked (IonicNODE has no `.nfe` attribute; adding one violates frozen-model rule).

`phase_name` is a REQUIRED config field — NOT defaulted in the adapter. Valid values: `A1/A2/A3/A4/ionic_state/conc_only/B1-B4/ionic_state_and_conductance`. For Session 25 parity, use `phase_name: "A1"` (equivalent to `"ionic_state"`).

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/node_step.py`
**Files modified:**
- `cardiac_ml/training/trainer.py` — add `_on_after_backward` callback dispatch in `_run_epoch`. Small additive change to support the V_traj-clear contract. See Step 3.4 addendum below.
- none under `Surrogate/surrogate/` (M-6 resolution — `node_rollout.py`, `node.py` stay untouched).

**Interfaces:**
```python
def node_train_step(trainer, batch) -> dict:
    """Call node_rollout() with required phase_name from config.
    Returns {"loss": loss, "_on_after_backward": clear_v_callback}."""

def node_val_step(trainer, batch) -> dict:
    """No-grad variant. Same phase_name. Returns
    {"loss": loss, "val_loss": loss, "_on_after_backward": clear_v_callback}.
    Val path still clears V_traj — model state must be clean between epochs."""
```

#### Pseudocode
```python
# Surrogate/surrogate/training/node_step.py
"""Trainer-shape adapter for the existing node_rollout() function."""
from __future__ import annotations
import torch
from .node_rollout import node_rollout

def _phase_from_cfg(trainer) -> str:
    """Required config field. Valid: A1|A2|A3|A4|ionic_state|conc_only|B1|B2|B3|B4|
    ionic_state_and_conductance. Raises KeyError if missing — no silent default."""
    pn = trainer.cfg.training.get("phase_name")
    if pn is None:
        raise KeyError("cfg.training.phase_name is required for NODE training "
                       "(valid: A1|A2|A3|A4|ionic_state|conc_only|B1..B4|"
                       "ionic_state_and_conductance). See node_rollout.py:160-191.")
    return pn

def node_train_step(trainer, batch) -> dict:
    result = node_rollout(
        node=trainer.model,
        segment=batch,
        phase_name=_phase_from_cfg(trainer),
        method=trainer.cfg.training.get("ode_method", "dopri5"),
        rtol=trainer.cfg.training.get("ode_rtol", 1e-3),
        atol=trainer.cfg.training.get("ode_atol", 1e-3),
        adjoint=trainer.cfg.training.get("ode_adjoint", False),
    )
    loss = result["loss"]
    # clear_v_trajectory must run AFTER loss.backward(). Trainer's _run_epoch
    # pops _on_after_backward and invokes it post-backward (see Step 3.4).
    def _clear():
        trainer.model.clear_v_trajectory()
    # Surface per-component scaffold losses as detached metrics (auto-logged).
    extra = {k: v.detach() for k, v in result.items() if k != "loss"}
    return {"loss": loss, "_on_after_backward": _clear, **extra}

def node_val_step(trainer, batch) -> dict:
    with torch.no_grad():
        result = node_rollout(
            node=trainer.model,
            segment=batch,
            phase_name=_phase_from_cfg(trainer),
            method=trainer.cfg.training.get("ode_method", "dopri5"),
            rtol=trainer.cfg.training.get("ode_rtol", 1e-3),
            atol=trainer.cfg.training.get("ode_atol", 1e-3),
            adjoint=trainer.cfg.training.get("ode_adjoint", False),
        )
    loss = result["loss"]
    def _clear():
        trainer.model.clear_v_trajectory()
    extra = {k: v.detach() for k, v in result.items() if k != "loss"}
    return {"loss": loss, "val_loss": loss, "_on_after_backward": _clear, **extra}
```

**Step 3.4 addendum (callback dispatch for `_on_after_backward`)** — update `_run_epoch`'s train-path after `loss.backward()`:

```python
if train:
    if backward_done:
        assert loss.requires_grad or loss.grad_fn is not None, \
            "_backward_done=True but loss is not attached to any compute graph"
    else:
        assert loss.requires_grad, \
            "train_step_fn returned loss with requires_grad=False"
        loss.backward()
    # NEW: post-backward hook, used by node_step.py to call clear_v_trajectory().
    post_hook = out.get("_on_after_backward")
    if post_hook is not None:
        post_hook()
    self.optimizer.step(); self.optimizer.zero_grad()
else:
    # Val path: still honor post-backward hook even though no backward happened,
    # because clear_v_trajectory() is stateful cleanup regardless of gradient mode.
    post_hook = out.get("_on_after_backward")
    if post_hook is not None:
        post_hook()
```

#### Test Spec
- `cardiac_ml/tests/test_node_step.py::test_node_train_step_returns_loss_and_cleanup_hook` — Setup: instantiate IonicNODE on CPU with minimal config, build a 1-sample segment fixture, call `node_train_step(trainer, batch)`. Expected: return dict has `"loss"` scalar tensor with `.requires_grad == True`, `"_on_after_backward"` is callable, `"ionic_state_mse"` metric present (detached).
- `test_node_step_raises_without_phase_name` — Setup: cfg without `training.phase_name`. Expected: KeyError with message listing valid phases.
- `test_node_val_step_no_grad` — Setup: same, val variant. Expected: loss has `.requires_grad == False`, cleanup hook present.
- `test_node_step_does_not_import_torchdiffeq_directly` — Setup: parse `node_step.py` source. Expected: no `from torchdiffeq` import.
- `test_clear_v_trajectory_invoked_by_trainer` — Setup: run a 1-epoch fit via Trainer + `node_train_step` with a mock IonicNODE that spies on `clear_v_trajectory`. Expected: called once per batch, AFTER backward (ordering verifiable via side-effect counter).
- `test_node_source_files_unchanged` — Setup: `git diff --quiet 8f191f77 -- Surrogate/surrogate/training/node_rollout.py Surrogate/surrogate/model/node.py`. Expected: exit 0 (no changes). (No sha256-based test — git's own diff is the baseline, removing the baseline-less sha256 test from prior round.)

#### Checklist
- [ ] `node_step.py` created as a pure adapter importing only `node_rollout`.
- [ ] Adapter extracts `result["loss"]` from the dict return (C-1 fix).
- [ ] `phase_name` is REQUIRED — no silent default. Fails loud with valid-values list (C-2 fix).
- [ ] Per-component metrics (`ionic_state_mse`, `conc_mse`, `conductance_mse`) surfaced as detached extras in the return dict — auto-logged to MLflow for parity debugging.
- [ ] `_on_after_backward` cleanup hook calls `node.clear_v_trajectory()` (H-1 fix).
- [ ] Trainer.`_run_epoch` invokes `_on_after_backward` after backward on train path AND on val path.
- [ ] `git diff --quiet 8f191f77 -- Surrogate/surrogate/training/node_rollout.py Surrogate/surrogate/model/node.py` passes.
- [ ] ODE config fields (method/rtol/atol/adjoint) pulled from cfg with oracle defaults.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_node_step.py -v && \
git diff --quiet 8f191f77 -- Surrogate/surrogate/training/node_rollout.py Surrogate/surrogate/model/node.py
```

#### Exit Criteria
- [ ] All 6 tests pass.
- [ ] `git diff --quiet` against `8f191f77` shows no changes to `node_rollout.py` or `node.py`.
- [ ] NFE is NOT in the return dict; V_traj cleanup is wired; phase_name is required.
- [ ] Phase-4-scoped importability check (promised at Step 2.5 risk) runs here as `test_targets_importable_phase4` — verifies `surrogate.training.node_step.node_train_step` and `node_val_step` import successfully.

#### Risk
- **MEDIUM** — `SegmentDataset` yields a dict with keys `Vm`, `dt`, `ionic_states`, `concentrations`, `conductance_products` (per `node_rollout.py:86-87,162-181`). Verify `SegmentDataset.__getitem__` produces this exact shape; if the DataLoader's default `collate_fn` reshapes or drops keys, fix the config or wrap `collate_fn` — NOT `node_rollout.py`.
- **LOW** — `_on_after_backward` hook semantics are harness-wide (any `train_step_fn` can use it). Document in Step 3.1 (teacher_forced_step) that this hook exists but is optional; most steps won't need it.

---

### Step 4.2: NODE config files  [DEFERRED — pending Step 4.0]
**Model**: sonnet

> **⚠ This step's current YAML omits `phase_name` (required by node_step.py adapter) and fabricates `train_bcls`/`val_bcls` fields that SegmentDataset doesn't consume.** Re-spec via `/blueprint-revise` using Step 4.0 findings before execution.


#### Read First
- `Surrogate/surrogate/model/node.py` for IonicNODE constructor signature.
- `Surrogate/surrogate/training/datasets.py` for SegmentDataset constructor signature.
- `Surrogate/surrogate/training/train_node.py` for multi-BCL specs.

#### Why
Replace argparse flags with composable Hydra configs. Each YAML is read-once, machine-checkable, and version-controllable.

#### Implementation Spec
**Files to create/fill:**
- `conf/model/ionic_node.yaml` — `_target_: surrogate.model.node.IonicNODE` with all constructor kwargs.
- `conf/data/t1_multi_bcl.yaml` — train/val SegmentDataset configs with BCL splits.
- `conf/training/node.yaml` — `train_step_fn` / `val_step_fn` pointing at `node_step.py`, optimizer, epochs, ODE method/tolerances.
- `conf/experiment/ionic_node_t1.yaml` — composes the above + default tracking.

#### Pseudocode
```yaml
# conf/training/node.yaml
epochs: 500
seed: 42
device: cuda
dtype: float64
ode_method: dopri5
ode_rtol: 1e-3
ode_atol: 1e-3
optimizer:
  _target_: torch.optim.Adam
  lr: 5e-4
train_step_fn:
  _target_: hydra.utils.get_method
  path: surrogate.training.node_step.node_train_step
val_step_fn:
  _target_: hydra.utils.get_method
  path: surrogate.training.node_step.node_val_step
callbacks:
  - _target_: cardiac_ml.training.callbacks.EarlyStopping
    monitor: val_loss
    patience: 30
  - _target_: cardiac_ml.training.callbacks.ModelCheckpoint
    monitor: val_loss
    every_n_epochs: 50
  - _target_: cardiac_ml.training.callbacks.GradNormMonitor
```

#### Test Spec
- `cardiac_ml/tests/test_node_configs.py::test_ionic_node_t1_composes` — Setup: `compose("config", overrides=["experiment=ionic_node_t1"])`. Expected: resolves without error, all `_target_` paths importable.
- `test_ionic_node_t1_instantiates` — Setup: `instantiate(cfg.model)`. Expected: IonicNODE instance on CUDA, float64.

#### Checklist
- [ ] All 4 YAMLs filled in.
- [ ] All `_target_` paths point at real classes / methods.
- [ ] Compose + instantiate tests pass.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_node_configs.py -v
```

#### Exit Criteria
- [ ] Both tests pass.

#### Risk
- Data paths may need absolute paths if `/tmp/surrogate_cache/` isn't reliably populated (`/tmp` is volatile on many systems). **Mitigation**: use `${oc.env:SURROGATE_CACHE_DIR,/media/HDD/norepinephrine/surrogate_data/raw}` — OmegaConf env resolver with comma-separated default (Round-2 H-3 fix: correct real-path is `/media/HDD/norepinephrine/surrogate_data/raw/`, NOT `/media/HDD/surrogate_data/raw/` which doesn't exist). Verified 2026-04-19 via `ls /media/HDD/norepinephrine/surrogate_data/raw/` → `tier01.h5..tier05.h5`. If SSD cache at `/tmp/surrogate_cache/` is present, set `SURROGATE_CACHE_DIR=/tmp/surrogate_cache/` in the shell.

---

### Step 4.3: NODE smoke test  [DEFERRED — pending Step 4.0]
**Model**: opus

> **⚠ Depends on Step 4.2 YAML fixes.** Do not execute until Step 4.0 + revise pass land.


#### Read First
- `scripts/train.py`.
- `conf/experiment/ionic_node_t1.yaml` from Step 4.2.

#### Why
Run 3-epoch training on a small BCL subset to verify the full stack works (data loads → model runs → adjoint backward → metrics logged → artifact saved) before committing 500+ epochs of GPU time.

#### Implementation Spec
**Files to create:** `conf/experiment/ionic_node_smoke.yaml` — trims to 3 epochs, 1 train BCL, 1 val BCL.

#### Pseudocode
```yaml
# conf/experiment/ionic_node_smoke.yaml
defaults:
  - ionic_node_t1
training:
  epochs: 3
data:
  train_bcls: [500]
  val_bcls: [600]
```

Run with override.

#### Test Spec
Manual (not a pytest — too slow / GPU-heavy). Pass = criteria below.

#### Checklist
- [ ] Smoke run completes in < 5 minutes on Blackwell.
- [ ] MLflow run appears in `./mlruns_smoke/` (NOT top-level `./mlruns/` — tracking_uri override per Step 4.3 verify block).
- [ ] Run has `train_loss`, `val_loss`, plus per-component metrics (`train_ionic_state_mse`, `train_conc_mse`) logged for 3 epochs. (Round-2 M-2 fix: NFE removed from checklist since the adapter drops it.)
- [ ] Run has `best.pt` + `last.pt` artifacts.
- [ ] Run has `git.sha` + `git.dirty` tags.

#### Verify
```bash
# Use an isolated mlruns dir to avoid polluting top-level during smoke (M-4).
rm -rf mlruns_smoke && \
conda run -n heart-conduction python scripts/train.py \
  experiment=ionic_node_smoke \
  tracking.tracking_uri=./mlruns_smoke && \
conda run -n heart-conduction python -c "
import mlflow
mlflow.set_tracking_uri('./mlruns_smoke')
runs = mlflow.search_runs(experiment_names=['default'])
# NOTE: NFE metric dropped — IonicNODE has no nfe attribute (C-2 fix)
print(runs[['run_id','metrics.loss','tags.git.sha']].head())
" && rm -rf mlruns_smoke
```

#### Exit Criteria
- [ ] Verify commands produce expected output.
- [ ] `mlruns_smoke/` directory exists during run, cleaned up after.
- [ ] Real top-level `./mlruns/` was NOT written to (M-4).

#### Risk
- **HIGH** — backward may fail if V_traj doesn't persist through backward (Session 23 flagged this). **Mitigation**: since `node_step.py` delegates to `node_rollout()` unchanged, this issue would manifest in the oracle too — if the smoke test hits a gradient error, the bug is in how SegmentDataset yields batches, not in the adapter. Do NOT modify `node_rollout.py` to fix it; fix the upstream data path.
- **MEDIUM** — `segment` object type mismatch between DataLoader batch and what `node_rollout()` expects. **Mitigation**: during implementation, test with `collate_fn` override in the DataLoader config if the default `torch.utils.data` collation produces wrong shape.

---

### Step 4.4: NODE parity run  [DEFERRED — pending Step 4.0]
**Model**: opus

> **⚠ Current specification has critical infrastructure bugs**: (a) `SessionParityGate` utility reads a JSONL current-log the harness doesn't write; (b) reference log only covers 8 epochs but `min_epoch=10` locks gate out; (c) Phase 4 Verification block has a comment instead of a parity command. Re-spec via `/blueprint-revise` after Step 4.0.


#### Read First
- Session 25 IDEALOG entry for parity target (val=0.008).
- `Surrogate/runs/multi_bcl_002/log.jsonl` — Session 25's per-epoch val_loss trajectory. This is the oracle for the mid-flight validation gate.
- Smoke run's MLflow record.

#### Why
The hard success criterion. Full 500-epoch training on multi-BCL T1, matching Session 25 config exactly modulo the harness. Val_loss must reach ≤ 0.008.

**Architecture-drift precondition (M-8 from Round 1)**: Session 25 val=0.008 was achieved with the `IonicRateMLP` architecture (832 params, dense MLP replacing VoltageAttention — see `Surrogate/surrogate/model/stage1.py` as of SHA `8f191f77`). If `Surrogate/surrogate/model/` has diverged at Phase 4 start, parity may not be achievable. **Phase 4 MUST abort if the model tree at Phase 4 start differs from SHA `8f191f77`.**

**SHA pin (H-7 from Round 1; Round-2 H-5 fix applied)**: Phase 4 pins the model tree to commit `8f191f77` (verified real commit 2026-04-19 via `git log --oneline 8f191f77 -1` — "Surrogate: NODE pivot validated, dense MLP replaces VoltageAttention"). Drift is checked via `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/`, NOT via SHA-string comparison (the previous `[ "$MODEL_SHA" != "8f191f77..." ]` pseudocode had a literal-ellipsis bug and was dead code). The diff check is the sole authoritative precondition.

**Mid-flight validation gate (H-3 from Round 1; Round-2 fixes applied)**: every 50 epochs starting at epoch ≥ 10 (not epoch 0 — early-epoch transients have 1000× spread per multi_bcl_001 vs multi_bcl_002 comparison), compare current val_loss against Session 25's log.jsonl at the same epoch. Tolerance: 5× until epoch 50, 3× after. Reference log is copied to a stable location at Phase 4 start (`cardiac_ml/reference/session25_log.jsonl`) so post-cutover Phase 4 re-runs still find it (Round-2 H-4 fix).

**Parity threshold**: ≤ 1.05 × oracle best val_loss. Oracle best (multi_bcl_002) is **0.00838** — so pass threshold is 0.0088, not 0.008 as stated in prior rounds. Session 25's actual result is marginal at 0.008 and would fail a literal-0.008 gate by 5% (Round-2 M-7 fix).

#### Implementation Spec
**Files to create:**
- `cardiac_ml/reference/session25_log.jsonl` — copy of `Surrogate/runs/multi_bcl_002/log.jsonl`, committed as a stable reference that survives the Step 5.4 archive move (Round-2 H-4 fix).
- `cardiac_ml/analysis/parity_gate.py` — `compare_to_session25(current_log, reference_log, tolerance, min_epoch) -> tuple[bool, str]` utility. Reads both JSONL logs, aligns by epoch, returns False at first over-tolerance epoch ≥ `min_epoch`.
- A new Callback `SessionParityGate(Callback)` that reads the stable reference at `on_fit_start`, invokes `compare_to_session25` at `on_epoch_end` every 50 epochs starting from epoch ≥ 10. Sets `trainer.should_stop = True` if gate trips.

**Files to modify:**
- `conf/experiment/ionic_node_t1.yaml` — add the `SessionParityGate` callback with tolerance schedule.

#### Pseudocode
```bash
# Precondition — architecture tree matches SHA 8f191f77 (M-8, Round-2 H-5 fix).
# The diff check is authoritative; no fragile SHA-string comparison.
git diff --quiet 8f191f77 -- Surrogate/surrogate/model/ || {
  echo "ABORT: Surrogate/surrogate/model/ diverges from parity oracle SHA 8f191f77."
  echo "Run 'git diff 8f191f77 -- Surrogate/surrogate/model/' to inspect."
  echo "Checkout or stash changes under that path before running Phase 4."
  exit 1
}

# One-time: copy reference log to stable location (Round-2 H-4 fix).
# Idempotent — skip if already present.
mkdir -p cardiac_ml/reference
if [ ! -f cardiac_ml/reference/session25_log.jsonl ]; then
  cp Surrogate/runs/multi_bcl_002/log.jsonl cardiac_ml/reference/session25_log.jsonl
  git add cardiac_ml/reference/session25_log.jsonl
fi

# Record starting HEAD SHA for drift-detection during the run.
git rev-parse HEAD > .parity_start_sha

# Launch training.
conda run -n heart-conduction python scripts/train.py experiment=ionic_node_t1
# Monitor via: mlflow ui --backend-store-uri ./mlruns
```

```python
# cardiac_ml/analysis/parity_gate.py
import json, pathlib
def compare_to_session25(
    current_log_path: str,
    reference_log_path: str = "cardiac_ml/reference/session25_log.jsonl",
    tolerance_early: float = 5.0,   # epochs 10..50
    tolerance_late: float = 3.0,    # epochs >= 50
    min_epoch: int = 10,            # ignore transient early epochs
) -> tuple[bool, str]:
    """Returns (passes, diagnostic_msg). Gate PASSES if at every epoch >= min_epoch,
    current_val_loss / reference_val_loss <= tolerance (schedule above).
    log.jsonl schema: {epoch, train_loss, val_loss, val_per_bcl, lr, elapsed_s}
    (verified via cat Surrogate/runs/multi_bcl_002/log.jsonl 2026-04-19)."""
    def _load(path):
        return {e["epoch"]: e["val_loss"]
                for e in (json.loads(l) for l in pathlib.Path(path).read_text().splitlines())}
    current, reference = _load(current_log_path), _load(reference_log_path)
    for epoch in sorted(set(current) & set(reference)):
        if epoch < min_epoch:
            continue
        tol = tolerance_early if epoch < 50 else tolerance_late
        if reference[epoch] > 0 and current[epoch] / reference[epoch] > tol:
            return False, (f"epoch {epoch}: current={current[epoch]:.5f} > "
                           f"{tol}× reference={reference[epoch]:.5f}")
    return True, "OK"
```

#### Test Spec
- `cardiac_ml/tests/test_parity_gate.py::test_gate_passes_when_current_matches_reference` — Setup: synthetic logs where current==reference for epochs 10..100. Expected: `(True, "OK")`.
- `test_gate_skips_early_epochs` — Setup: current has val_loss=100× reference at epoch 5 (below min_epoch=10). Expected: `(True, "OK")` — gate does not fire on transient early spikes.
- `test_gate_fails_at_first_excess_epoch_late_window` — Setup: current is 4× reference at epoch 100 (late window, tolerance=3×). Expected: `(False, "epoch 100: ...")`.
- `test_gate_allows_5x_early_window` — Setup: current is 4× reference at epoch 30 (early window, tolerance=5×). Expected: `(True, "OK")`.
- `test_reference_log_schema_matches` — Setup: load `cardiac_ml/reference/session25_log.jsonl`. Expected: at least `epoch` and `val_loss` keys on every line.
- Real parity run: not a pytest, criteria below.

#### Checklist
- [ ] `parity_gate.py` + `SessionParityGate` callback implemented with epoch-schedule tolerances.
- [ ] `cardiac_ml/reference/session25_log.jsonl` exists, committed, survives Step 5.4 archive move (Round-2 H-4 fix).
- [ ] Pre-run `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` passes (Round-2 H-5 fix — no brittle SHA-string comparison).
- [ ] `conf/experiment/ionic_node_t1.yaml` includes the gate callback.
- [ ] Training runs to completion OR early-stops at good val_loss OR aborts via gate.
- [ ] If gate trips: abort is clean, diagnostic logged to MLflow, no further epochs consumed.
- [ ] If run completes: final val_loss ≤ 1.05 × 0.00838 = 0.0088 (Round-2 M-7 fix — marginal oracle means strict-0.008 gate would fail the oracle itself).
- [ ] Training wall time within 2× of Session 25 run (no major regression).
- [ ] Model tree unchanged during the run (`git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` still passes at end).

#### Verify
```bash
# Parity check — use oracle-relative threshold.
conda run -n heart-conduction python -c "
import mlflow
mlflow.set_tracking_uri('./mlruns')
# Use run_name filter via attribute (Round-2 L-5 fix — attribute path, not tags.mlflow.runName LIKE).
runs = mlflow.search_runs(experiment_names=['default'])
runs = runs[runs['tags.mlflow.runName'].fillna('').str.startswith('ionic_node_t1')]
best_val = runs['metrics.val_loss'].min()
threshold = 1.05 * 0.00838
print(f'Best val_loss: {best_val:.5f}, threshold: {threshold:.5f}')
assert best_val <= threshold, f'Parity failed: {best_val:.5f} > {threshold:.5f}'
print('PARITY OK')
"

# Drift check — use git diff not SHA comparison.
git diff --quiet 8f191f77 -- Surrogate/surrogate/model/ || {
  echo "WARNING: model tree diverged during Phase 4"
  exit 1
}
rm -f .parity_start_sha
```

#### Exit Criteria
- [ ] Parity met (val_loss ≤ 1.05 × oracle best = 0.0088), OR gate tripped cleanly with diagnostic.
- [ ] Wall time reasonable.
- [ ] Model-tree diff against `8f191f77` still empty at Phase 4 end.

#### Risk
- **HIGH** — if parity fails (and the gate hasn't already caught it), investigate in this order:
  1. **Architecture drift**: does `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` still pass? (M-8)
  2. Is the dataset batch shape / keys identical? `SegmentDataset` must yield `{Vm, dt, ionic_states, concentrations, conductance_products}` per `node_rollout.py` contract.
  3. Is `phase_name` correctly set to `"A1"` (or equivalent `"ionic_state"`) matching Session 25's multi-BCL config?
  4. Is the optimizer config identical? (Adam lr=5e-4 per Session 25.)
  5. Is the ODE method/tolerances identical? (dopri5, rtol=atol=1e-3, adjoint=False.)
  6. Is `node.clear_v_trajectory()` being called after each backward? (Round-2 H-1 — missing cleanup causes stale V_traj across batches.)
  If any diverges, fix the config, NOT the code. The old `train_node.py` + `node_rollout.py` combo is the oracle.
- **MEDIUM** — log.jsonl schema: verified 2026-04-19 as `{epoch, train_loss, val_loss, val_per_bcl, lr, elapsed_s}` per line. `compare_to_session25` only reads `epoch` + `val_loss`. Other keys are ignored. If multi_bcl_002 log.jsonl is ever regenerated with a different schema, `test_reference_log_schema_matches` catches it at test time.

---

### Phase 4 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_node_step.py cardiac_ml/tests/test_node_configs.py -v && \
# parity verification command from Step 4.4
echo "Phase 4 OK"
```

### Phase 4 Exit Criteria
- [ ] NODE smoke test passes.
- [ ] NODE parity met (val_loss ≤ 0.008 on multi-BCL T1).
- [ ] `node_rollout.py` and `train_node.py` unchanged (reusability via import, not modification).
- [ ] All existing Surrogate tests still pass.

### Phase 4 Cleanup
- Confirm `Surrogate/surrogate/model/*.py` unchanged (C-2: no NFE counter added, frozen input rule holds).
- Confirm `Surrogate/surrogate/training/node_rollout.py` and `train_node.py` unchanged (M-6).
- `git diff --stat 8f191f77 -- Surrogate/surrogate/model/ Surrogate/surrogate/training/node_rollout.py Surrogate/surrogate/training/train_node.py` — expect empty.
- Grep for any `scipy` usage in NEW files under `cardiac_ml/` (Round-2 L-1 — scipy is allowed project-wide per CLAUDE.md, only a concern if the harness itself starts depending on it). Expect zero hits inside `cardiac_ml/`.
- Verify `Surrogate/runs/` still exists untouched (it gets archived in Phase 5, NOT here).
- Verify `cardiac_ml/reference/session25_log.jsonl` exists and is git-tracked (Round-2 H-4 stable-path copy).

**→ Commit point: `git commit` after Phase 4 passes. Message: "cardiac_ml Phase 4: ionic NODE pilot migration — parity met"**

---

## Phase 5: Optuna Sweeps + SHAP + Cutover + Reusability

**Goal**: Close out the remaining success criteria: Optuna HPO, SHAP analysis, archiving old `Surrogate/runs/`, and a second-consumer reusability proof via a diffusion-ResNet stub. After this phase, the harness is complete per REQUIREMENTS.
**Tier**: medium
**Estimated scope**: 6 steps.

### Phase Context

- Optuna integration via `hydra-optuna-sweeper` plugin — configured in `conf/hparams_search/`.
- SHAP input scope for ionic NODE: **V-only** (resolves OPEN-4). Simpler baseline; joint `(z, V)` deferred to future work.
- Optuna pruner: median, 20-epoch warmup (settles OPEN-5 empirically based on NODE training curves from Phase 4).
- Diffusion ResNet stub: trivial 2-layer CNN on 32×32 random-pattern inputs. Exists solely to prove the harness handles a second, structurally different model. NOT a real diffusion implementation.

---

### Step 5.1: Optuna sweep config + sweep.py
**Model**: opus

#### Read First
- Template survey `results/template_survey.md` for Optuna sweep patterns.
- Phase 4 MLflow records — use the best-run's hparams as the sweep center.

#### Why
Demonstrates `--multirun` HPO. 3-trial sweep is the minimum to validate the plumbing without burning GPU time.

#### Implementation Spec
**Files to create:**
- `conf/hparams_search/lr_batch.yaml` — sweep over `training.optimizer.lr` ∈ {1e-3, 5e-4, 1e-4}.
- `scripts/sweep.py` — same as `train.py` but with sweeper config override.

#### Pseudocode
`conf/hparams_search/lr_batch.yaml`:
```yaml
# Round-3 LOW-4 fix: pruner block was missing; OPEN-5's "median + 20-epoch
# warmup" was a trailing comment, not actual YAML. Now an explicit directive.
defaults:
  - override /hydra/sweeper: optuna
hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
    pruner:
      _target_: optuna.pruners.MedianPruner
      n_startup_trials: 5
      n_warmup_steps: 20  # resolves OPEN-5
    direction: minimize
    study_name: lr_sweep
    n_trials: 3
    n_jobs: 1
    params:
      training.optimizer.lr: choice(1e-3, 5e-4, 1e-4)
```

`scripts/sweep.py` is `scripts/train.py` with sweep config attached.

Note on the `+hparams_search=lr_batch` syntax (L-8): `hparams_search` is a LOOSE config group — NOT included in `conf/config.yaml`'s defaults list. The leading `+` tells Hydra to add the group to the composition at runtime. This keeps non-sweep runs clean (they don't see sweep config) while letting sweep invocations opt-in. The `override /hydra/sweeper: optuna` line inside the YAML handles the sweeper swap at composition time.

#### Test Spec
- Manual: `python scripts/sweep.py --multirun +hparams_search=lr_batch experiment=ionic_node_smoke` runs 3 trials.

#### Checklist
- [ ] 3 trials launch in sequence.
- [ ] Each trial is a separate MLflow run.
- [ ] Best trial identified by Optuna matches MLflow best val_loss.

#### Verify
```bash
conda run -n heart-conduction python scripts/sweep.py --multirun +hparams_search=lr_batch experiment=ionic_node_smoke
```

#### Exit Criteria
- [ ] 3 runs in MLflow.
- [ ] Sweep completes without error.

#### Risk
- Pruning may kill viable trials — smoke test is only 3 epochs, below 20-epoch warmup, so pruner stays dormant (confirms the warmup protects NODE). Full pruner tuning via OPEN-5 deferred to first real sweep on the parity config.

---

### Step 5.2: SHAP analysis script
**Model**: opus

#### Read First
- OPEN-4 in REQUIREMENTS.md — V-only scope decision.
- `shap.KernelExplainer` docs — model-agnostic explainer.

#### Why
Post-hoc interpretability over a trained NODE checkpoint. V-only keeps the input space 1D, plot simple.

**Explainer choice (H-4 from audit)**: Use `shap.KernelExplainer` as the default (and only) path. `DeepExplainer` requires TF-style forward gradient hooks — incompatible with `torchdiffeq`'s `odeint_adjoint` (which replaces the backward entirely) and with `odeint` (which routes gradients through the integrator's own graph, not the module's `.forward()`). `KernelExplainer` is model-agnostic (treats the model as a black box function), works for any `(V) -> z_final` mapping, is slower but always works. No `DeepExplainer` fallback attempted.

#### Implementation Spec
**Files to create:**
- `cardiac_ml/analysis/shap_utils.py`
- `scripts/analyze.py`

**Interfaces:**
```python
# cardiac_ml/analysis/shap_utils.py
def kernel_shap_v_only(model, V_samples: torch.Tensor, baseline: torch.Tensor,
                       nsamples: int = 100) -> 'shap.Explanation':
    """V-only KernelExplainer over a trained NODE. Returns shap.Explanation."""

def plot_shap_summary(explanation: 'shap.Explanation', output_path: str) -> None: ...

# scripts/analyze.py — Round-3 MED-11 fix: concrete pseudocode below.
```

**`scripts/analyze.py` pseudocode**:
```python
"""Post-hoc SHAP over a trained checkpoint. Usage:
   python scripts/analyze.py run_id=<mlflow_run_id> output_dir=./shap_out
"""
import hydra
import mlflow
import torch
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
from cardiac_ml.analysis.shap_utils import kernel_shap_v_only, plot_shap_summary

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    assert cfg.get("run_id"), "Required: run_id=<mlflow_run_id>"
    output_dir = Path(cfg.get("output_dir", "./shap_out"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rebuild the model from the config that was used for the run.
    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.tracking.tracking_uri)
    run = client.get_run(cfg.run_id)
    # Model is rebuilt from the current cfg.model — caller must supply
    # the same experiment/model config used at training time (Hydra CLI
    # override pattern: `python scripts/analyze.py experiment=ionic_node_t1
    # run_id=<sha>`).
    model = instantiate(cfg.model).to(cfg.training.device, dtype=getattr(torch, cfg.training.dtype))

    # 2. Load the state dict from the run's best.pt artifact.
    artifact_path = client.download_artifacts(cfg.run_id, "best.pt", str(output_dir))
    state = torch.load(artifact_path, map_location=cfg.training.device)
    model.load_state_dict(state)
    model.eval()

    # 3. Build V samples + baseline (cfg-driven; small defaults for cost).
    V_samples = _sample_v_trajectories(cfg)  # shape (N, T)
    baseline = _resting_v_baseline(cfg)      # shape (M, T)

    # 4. Run SHAP + plot.
    explanation = kernel_shap_v_only(model, V_samples, baseline,
                                      nsamples=cfg.get("shap_nsamples", 50))
    plot_shap_summary(explanation, str(output_dir / "shap_v_only.png"))

if __name__ == "__main__":
    main()
```

#### Pseudocode
```python
import shap, numpy as np, torch
def kernel_shap_v_only(model, V_samples, baseline, nsamples=100):
    """V_samples: (N, T) numpy array of voltage trajectories.
    baseline: (M, T) reference set (e.g. resting voltage replicated M times)."""
    model.eval()
    def predict(V_batch_np):
        # V_batch_np: (batch, T) numpy. Convert, run NODE, return final z as flattened output.
        V_batch = torch.from_numpy(V_batch_np).double().to(next(model.parameters()).device)
        outputs = []
        with torch.no_grad():
            for V in V_batch:
                z_final = _run_node_to_final_z(model, V)  # wrap node.integrate() or node.euler_step()
                outputs.append(z_final.cpu().numpy())
        return np.stack(outputs)
    explainer = shap.KernelExplainer(predict, baseline.cpu().numpy())
    shap_values = explainer.shap_values(V_samples.cpu().numpy(), nsamples=nsamples)
    return shap.Explanation(values=shap_values, data=V_samples.cpu().numpy(),
                            feature_names=[f"V[t={t}]" for t in range(V_samples.shape[1])])
```

#### Test Spec
- `cardiac_ml/tests/test_shap_utils.py::test_kernel_shap_returns_correct_shape` — Setup: tiny NODE instance on CPU, small V_samples tensor (N=3, T=10), baseline (M=2, T=10). Expected: SHAP values array shape `(N, T, z_dim)` or `(N, T)` per SHAP's conventions.
- `test_kernel_shap_handles_torchdiffeq_model` — Setup: real IonicNODE with `integrate()`, but with **`nsamples=5`** and **`T=3` trajectory length** to cap cost (Round-2 M-3 fix — default nsamples=100 × long trajectory = minutes/test). Mark `@pytest.mark.slow` if test suite adopts markers. Expected: no gradient-related errors (KernelExplainer doesn't touch gradients); function returns shape-correct result.

#### Checklist
- [ ] `kernel_shap_v_only` implemented using `shap.KernelExplainer` only (H-4).
- [ ] `plot_shap_summary` implemented.
- [ ] `scripts/analyze.py` loads state_dict from MLflow run_id via the SDK.
- [ ] No `shap.DeepExplainer` import anywhere (H-4).
- [ ] Tests pass.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_shap_utils.py -v && \
# check for DeepExplainer leak
! grep -rn "DeepExplainer" cardiac_ml/ scripts/ && \
conda run -n heart-conduction python scripts/analyze.py run_id=<parity_run_id_from_phase_4>
```

#### Exit Criteria
- [ ] Tests pass.
- [ ] No `DeepExplainer` reference in codebase.
- [ ] Analyze script produces PNG.

#### Risk
- KernelExplainer is O(nsamples × (N + M)) NODE forward passes. For a NODE with ~300 epochs of integration, 100 samples × (N=20 + M=10) = 3000 forward passes per checkpoint — ~10 minutes wall time. **Mitigation**: keep N, M small (3, 2) for first plot; scale up only if signal is needed. Document the cost in `scripts/analyze.py` help text.

---

### Step 5.3: Diffusion ResNet stub (reusability proof)
**Model**: opus

#### Read First
- REQUIREMENTS.md §8 success criterion #10 — second-consumer reusability.

#### Why
Concrete proof that Trainer generalizes beyond NODE. Two different `train_step_fn`s (default + node) sharing the same Trainer is necessary; showing a third structurally different model (convolutional, teacher-forced) seals it.

#### Implementation Spec
**Files to create:**
- `conf/model/diffusion_resnet_stub.yaml` — 2-layer CNN, `_target_` pointing at a test-only class.
- `conf/data/synthetic_2d.yaml` — random 32×32 noise → target mapping.
- `conf/experiment/diffusion_stub_smoke.yaml` — composes + 10 epochs.
- `cardiac_ml/tests/stub_models.py` — defines the tiny CNN class.
- `cardiac_ml/tests/synthetic_2d_dataset.py` — synthetic 2D dataset.

#### Pseudocode
```python
# stub_models.py
class DiffusionResNetStub(nn.Module):
    def __init__(self, in_ch=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        ).double()
    def forward(self, x): return x + self.net(x)
```

#### Test Spec
- `cardiac_ml/tests/test_diffusion_stub.py::test_stub_trains_under_harness` — Setup: 10-epoch fit via Trainer + default `teacher_forced_step`. Expected: `val_loss[epoch=9] < val_loss[epoch=0] * 0.9` (Round-3 LOW-1 fix — strict monotonic over 10 epochs is brittle for randomly-initialized CNN; endpoint comparison with 10% improvement margin is the real signal).

#### Checklist
- [ ] Stub model + dataset exist.
- [ ] Experiment config composes.
- [ ] Trains under harness using default `teacher_forced_step` — zero Trainer changes.
- [ ] Test passes.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/test_diffusion_stub.py -v
```

#### Exit Criteria
- [ ] Test passes.
- [ ] Proves second-consumer reusability without modifying Trainer.

#### Risk
- Low — trivial model, synthetic data, proven harness.

---

### Step 5.4: Archive Surrogate/runs/
**Model**: sonnet

#### Read First
- REQUIREMENTS.md §2 goal 7.
- `.gitignore` — verify Step 1.3's `archive/*` + `!archive/runs_legacy/` allowlist is in place.
- Current git state of `Surrogate/runs/` — some subdirs tracked (`multi_bcl_001/`, `multi_bcl_002/`, etc.), some untracked (`a4_tbptt/`, `dt_v2/`). `git mv` would fail on untracked.

#### Why
Clean cutover per the decision in Session 26. Old training artifacts preserved but moved out of the way. Archive must remain git-tracked for the tracked subset so history is not lost (C-3 from audit).

#### Implementation Spec
**Files to move:** `Surrogate/runs/` → `archive/runs_legacy/`.
**Files to modify:** `.gitignore` already covers this via Step 1.3 allowlist — verify only.

#### Pseudocode
```bash
# Step A: grep for stale references (Round-2 H-2 fix — broader pattern).
# Matches both "Surrogate/runs" (absolute) and "runs/" (relative-from-Surrogate/).
# Excludes Surrogate/runs itself and the PLAN/docs that legitimately describe the path.
STALE=$(grep -rnE "(Surrogate/)?runs/[a-zA-Z0-9_]" \
  --include="*.py" --include="*.yaml" --include="*.sh" \
  --exclude-dir=runs --exclude-dir=archive \
  Surrogate/ scripts/ cardiac_ml/ conf/ 2>/dev/null || true)
if [ -n "$STALE" ]; then
  echo "ABORT: stale references to runs/ found:"
  echo "$STALE"
  echo "Fix them (update paths to archive/runs_legacy/ or remove obsolete refs) before moving."
  exit 1
fi

# Step B: verify gitignore allowlist is in place (defensive).
grep -qE "^archive/\*" .gitignore || { echo "ABORT: .gitignore missing archive/* rule from Step 1.3"; exit 1; }
grep -qE "^!archive/runs_legacy/" .gitignore || { echo "ABORT: .gitignore missing allowlist from Step 1.3"; exit 1; }

# Step C: mixed-tracking move.
mkdir -p archive
mv Surrogate/runs archive/runs_legacy

# Step D: update git index for the tracked subset.
git add -A archive/runs_legacy Surrogate/runs

# Step D': inspect what's about to be staged — abort if surprise files appear.
git status --porcelain archive/runs_legacy | head -20
# Prompt: manually inspect before committing. Look for .DS_Store, lock files, etc.

# Step E: verify tracked history preserved via git log --follow.
git log --follow --oneline archive/runs_legacy/multi_bcl_002/ | head -3
```

#### Test Spec
N/A (filesystem + git move).

#### Checklist
- [ ] Step A `grep` finds no stale references (exit 0 after scan — inversion of the `if grep` in pseudocode; i.e., the guard did NOT trigger).
- [ ] Step B confirms `.gitignore` allowlist.
- [ ] `Surrogate/runs/` no longer exists.
- [ ] `archive/runs_legacy/` exists and contains the old content (both previously-tracked and previously-untracked subdirs).
- [ ] `git status` shows the tracked-subset move (renames/deletes/adds).
- [ ] `git log --follow archive/runs_legacy/multi_bcl_002/` shows at least the Session 25 commit history (preserved).

#### Verify
```bash
test ! -d Surrogate/runs && \
test -d archive/runs_legacy && \
test -d archive/runs_legacy/multi_bcl_002 && \
# Round-2 H-2: broader stale-ref check catching relative runs/ paths too.
! grep -rnE "(Surrogate/)?runs/[a-zA-Z0-9_]" \
    --include="*.py" --include="*.yaml" --include="*.sh" \
    --exclude-dir=runs --exclude-dir=archive \
    Surrogate/ scripts/ cardiac_ml/ conf/ && \
git log --follow --oneline archive/runs_legacy/multi_bcl_002/ | head -1 && \
# Reference log for SessionParityGate still resolvable at stable path.
test -f cardiac_ml/reference/session25_log.jsonl
```

#### Exit Criteria
- [ ] Directory moved with both tracked and untracked content intact.
- [ ] No stale path references.
- [ ] Git history preserved via `--follow` for the previously-tracked subset.
- [ ] `archive/runs_legacy/` remains git-tracked (not caught by the `archive/*` ignore) thanks to the `!archive/runs_legacy/` allowlist.

#### Risk
- **MEDIUM** — something still imports from or references `Surrogate/runs/`. Step A grep covers `.py` and `.yaml`; extend to `.md` / `.sh` if any script orchestration exists there.
- **LOW** — `git add -A` may pick up transient files in the archive (lock files, `.DS_Store`). **Mitigation**: inspect `git status` diff before committing Phase 5; reject with `git restore --staged` if so.
- **LOW** — if `.parity_start_sha` from Step 4.4 leaked into the workspace, clean it up before commit.

---

### Step 5.5: MASTER.md status update
**Model**: sonnet

#### Read First
- `MASTER.md:15-17` — current cardiac_ml_harness row.

#### Why
Update the project dashboard to reflect the completed harness.

#### Implementation Spec
**Files to modify:** `MASTER.md` — update `cardiac_ml_harness` row status and next-step columns.

#### Pseudocode
Change from:
```
| [Cardiac ML harness](...) | None (project-wide) | Direction settled, no code yet | Template survey → blueprint |
```
to:
```
| [Cardiac ML harness](...) | None (project-wide) | IMPLEMENTED. NODE parity met. Reusability proved via diffusion stub. | Consumer migrations (diffusion ResNet, BayesOpt wrapper) |
```

#### Test Spec
N/A.

#### Checklist
- [ ] Status field updated.
- [ ] Next-step field updated.

#### Verify
```bash
# L-5: concrete verification beyond grep.
grep -A1 "Cardiac ML harness" MASTER.md | grep -qi "IMPLEMENTED" && \
grep -A1 "Cardiac ML harness" MASTER.md | grep -qi "parity met" && \
! grep "Direction settled, no code yet" MASTER.md
```

#### Exit Criteria
- [ ] Verify grep checks all pass.
- [ ] Old "Direction settled, no code yet" phrase is gone.
- [ ] New "IMPLEMENTED" + "parity met" markers present.

#### Risk
- None.

---

### Step 5.6: Completion criteria audit
**Model**: opus

#### Read First
- `Research/Active/cardiac_ml_harness/README.md` — completion criteria (9 rows).
- This PLAN.md Success Criteria (10 rows — the tracking-disable shim row is new per OPEN-8 absorption).

#### Why
Final pass: walk each completion criterion in the research README and confirm it's satisfied. Update checkboxes. Also reconcile the drift between PLAN (10 criteria) and README (9) — L-3 from audit: add the tracking-disable shim row to README.

#### Implementation Spec
**Files to modify:**
- `Research/Active/cardiac_ml_harness/README.md` — tick completion criteria boxes; add a 10th row for the tracking-disable shim to match PLAN's Success Criteria. Each checked box gets an evidence comment (`<!-- run_id: xxx / file: yyy / test: zzz -->`).

#### Pseudocode
1. Walk the 9 README completion criteria. For each:
   - If done: `[x]` + evidence comment.
   - If deferred: keep `[ ]` + add a `<!-- deferred: reason -->` comment.
2. Add missing 10th row for tracking-disable shim:
   ```markdown
   - [ ] Tracking-disable shim works: `python scripts/train.py ... tracking=off` runs without touching `mlruns/` (resolves OPEN-8).
   ```
3. Sync README's "Open" and "Settled" lists if any OPEN-N items were resolved during execution.

#### Test Spec
N/A (doc update).

#### Checklist
- [ ] All 9 original criteria have evidence or deferred-reason comments.
- [ ] 10th row added for tracking-disable shim (L-3).
- [ ] Any OPEN-N resolutions synced to README "Open / Settled" section.

#### Verify
```bash
# L-5: concrete grep checks.
grep -c "^- \[x\]" Research/Active/cardiac_ml_harness/README.md
# Should print at least 8 (could be 10 if everything's done).

grep -q "tracking=off" Research/Active/cardiac_ml_harness/README.md
# Verifies the 10th row was added.

# OPEN-N resolution audit.
grep -E "^\s*-\s*\[OPEN-[1-8]\]" Research/Active/cardiac_ml_harness/README.md || echo "No unresolved OPEN-N entries"
```

#### Exit Criteria
- [ ] ≥8 README criteria checked with evidence.
- [ ] Tracking-disable shim row present in README.
- [ ] No unresolved OPEN-N items except OPEN-2 (deferred to Future Work, moved there explicitly).

#### Risk
- None.

---

### Phase 5 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_ml/tests/ Surrogate/tests/ -v && \
test -d archive/runs_legacy && test ! -d Surrogate/runs && \
grep -q "IMPLEMENTED" MASTER.md && \
echo "Phase 5 OK"
```

### Phase 5 Exit Criteria
- [ ] Optuna 3-trial sweep works end-to-end.
- [ ] SHAP analysis produces a PNG for an ionic NODE checkpoint.
- [ ] Diffusion ResNet stub trains via default `teacher_forced_step` — proof of second-consumer reusability.
- [ ] `Surrogate/runs/` archived to `archive/runs_legacy/`.
- [ ] MASTER.md reflects completed status.
- [ ] README completion criteria audited.
- [ ] OPEN-4 resolved (V-only SHAP scope).
- [ ] OPEN-5 resolved (median pruner + 20-epoch warmup, verified not to kill viable NODE trials).

### Phase 5 Cleanup
- Grep for `torch.float32` anywhere in new code — zero hits.
- Grep for `import mlflow` outside `mlflow_logger.py` (Round-3 MED-3 fix: prior language allowed `trainer.py`, but R2-C3's dispatch-through-logger architecture means `trainer.py` should NOT import mlflow directly. Allowed: `mlflow_logger.py`, optionally `tests/conftest.py`, `scripts/analyze.py`).
- No duplicate data-cache / dataset implementations (reuse `surrogate.training.datasets.*`).
- Confirm no V5.3 changes.
- Verify `.gitignore` still covers `mlruns/`, `outputs/`, `archive/*`, `!archive/runs_legacy/`, `!archive/runs_legacy/**`.

**→ Commit point: `git commit` after Phase 5 passes. Message: "cardiac_ml Phase 5: Optuna + SHAP + cutover + reusability proof"**

---

## Final Cleanup

After all 5 phases land:

1. **Full regression**: `conda run -n heart-conduction python -m pytest -v` across `cardiac_ml/tests/`, `Surrogate/tests/`, and all engine test suites. No regressions allowed.
2. **float64 sweep**: `grep -rn "float32" cardiac_ml/ conf/ scripts/` — expect zero hits outside docstrings.
3. **MLflow isolation sweep**: `grep -rn "import mlflow" cardiac_ml/ conf/ scripts/` — expect hits ONLY in `mlflow_logger.py` (NOT `trainer.py` — Round-2 C-3 fix: Trainer dispatches via `self._logger`, never imports mlflow directly; optionally also `cardiac_ml/tests/conftest.py` for the tmpdir fixture).
4. **V5.3 untouched**: `git diff --stat main -- Monodomain/Engine_V5.3/` — expect empty.
5. **Cross-engine duplication check**: `grep -rn "from surrogate.training" cardiac_ml/` — expect zero hits (harness should not import from Surrogate; only the other direction).
6. **EXPERIMENT.md backlinks**: N/A (harness, not an engine experiment).
7. **Patch REQUIREMENTS.md §7.2** (Round-2 L-3) — update the NODE code sketch to match production config (`adjoint=False`, `dopri5`, `rtol=atol=1e-3`). The original `dopri8` / `rtol=1e-5` sketch is misleading and the new Step 4.1 pseudocode is the authoritative shape.
8. **Archive PLAN.md**:
   ```bash
   mkdir -p Research/Active/cardiac_ml_harness/plans
   cp Research/Active/cardiac_ml_harness/PLAN.md \
      "Research/Active/cardiac_ml_harness/plans/$(date +%Y-%m-%d)_cardiac-ml-harness-greenfield-implementation.md"
   ```
9. **Tmux pane revert**: SKIPPED — user disabled tmux panes for this research question.
10. **Update IDEALOG.md Session Log**: new row for "Plan executed — all phases complete, parity met, cutover done".
11. **Open decisions audit**: confirm all OPEN-1 through OPEN-8 are resolved. OPEN-2 stays deferred (BayesOpt shape — needs Optimizer V1 as driver) and should move to README Future Work.

---

## Mutation Log

Format: `**MUTATED YYYY-MM-DD**: Step X.Y {MODIFIED|SKIPPED|SPLIT|ADDED|INSERTED} — reason`.

### Blueprint-revise pass 2026-04-19 — audit findings (25 issues: 3C/7H/8M/7L)

**MUTATED 2026-04-19**: Step 1.2 Risk MODIFIED — added Hydra 1.1.x fallback contingency (H-5): if sweeper forces `hydra-core<1.2`, adjust Step 2.4 ConfigStore API and drop `version_base=None` from Step 3.5; alternatives (sweeper fork, custom TPE loop) listed.

**MUTATED 2026-04-19**: Step 1.3 Pseudocode + Checklist + Verify MODIFIED — added `archive/*` + `!archive/runs_legacy/` gitignore allowlist (M-5) so the Step 5.4 archived runs stay tracked. Plain `archive/` ignore would have silently hidden the moved Session 25 runs.

**MUTATED 2026-04-19**: Step 2.1 Test Spec + Checklist MODIFIED — replaced intentionally-failing `test_package_imports` with `test_all_init_files_importable` (L-1). Top-level `from cardiac_ml import Trainer` deferred to after Step 3.4; added lazy try/except shim in `__init__.py` to avoid breaking sibling imports during Phase 2.

**MUTATED 2026-04-19**: Step 2.5 Test Spec + Checklist + Exit + Risk MODIFIED — added `test_all_targets_importable` walking `_target_` strings and calling `importlib.import_module` per leaf (H-2). Closes the gap where `hydra.compose()` accepts non-existent target paths. Scoped to Phase 2–3 targets initially; Phase-4 targets checked at Step 4.1 exit.

**MUTATED 2026-04-19**: Step 3.2 Interfaces MODIFIED — clarified `LRSchedulerStep.scheduler_target` resolution: Hydra `_target_` string resolved lazily on `on_fit_start`, scheduler attached to `trainer.optimizer`, no-op if None (L-6).

**MUTATED 2026-04-19**: Phase 3 Context MODIFIED — removed "All tensors default to torch.float64" claim (it was the rationale for the removed `set_default_dtype` call); replaced with explicit-cast policy + MLflow isolation note (M-1, M-7).

**MUTATED 2026-04-19**: Step 3.4 Pseudocode MODIFIED — multiple changes:
  - Removed `torch.set_default_dtype(...)` global-state mutation (M-1). Dtype now applied via `.to(device=..., dtype=...)` on the model only.
  - Stored `MLflowLoggerCallback` or `NullLogger` as `self._logger` reference so Trainer's `log_artifact` / `log_figure` dispatch through it instead of importing mlflow directly (M-2, M-7).
  - Added precondition assertions in `_run_epoch`: `loss.requires_grad` on train path, `not _backward_done` on val path (H-1). Fails loud on protocol violations instead of silently skipping weight updates.
  - Documented per-batch mean as the default metric reduction with reserved `*_sum` / `*_last` suffixes for future callback-based override (M-3).

**MUTATED 2026-04-19**: Step 3.4 Test Spec MODIFIED — removed `test_float64_preserved` (no longer global); added 7 new tests covering `_backward_done` precondition violations (H-1), no-global-dtype-mutation (M-1), logger dispatch for escape hatches (M-2, M-7).

**MUTATED 2026-04-19**: Step 3.4 Risk MODIFIED — replaced MEDIUM concern about escape-hatch gating with M-3 metric-reduction note; noted LOW risk from removing global float64 default (step functions must specify dtype explicitly).

**MUTATED 2026-04-19**: Step 3.5 Verify + Exit Criteria MODIFIED — added `--cfg job` composition dump check beyond `--help` (L-2). `--help` only exercises argparse; `--cfg job` verifies the full Hydra defaults list resolves into a non-empty config with at least one `_target_`.

**MUTATED 2026-04-19**: Step 3.6 Implementation Spec ADDED — `cardiac_ml/tests/conftest.py` fixture that redirects MLflow tracking URI to a pytest tmpdir for all tests (M-4). Prevents Steps 3.6 and later from polluting `./mlruns/`.

**MUTATED 2026-04-19**: Phase 3 Exit Criteria + Cleanup MODIFIED — tightened mlflow-isolation check to exclude `trainer.py` from allowed imports (M-7). Added `set_default_dtype` grep guard (M-1).

**MUTATED 2026-04-19**: Phase 4 Context MODIFIED — added verified description of actual `node_rollout.py` / `node.py` API (C-1, C-2, H-6, LOW-7): monolithic `node_rollout()` function (no public helpers), `IonicNODE.integrate(..., adjoint=False)` with `dopri5/rtol=atol=1e-3`, no `.nfe` attribute. REQUIREMENTS §7.2 dopri8 sketch flagged as misleading.

**MUTATED 2026-04-19**: Step 4.1 FULL REWRITE (C-1, C-2, H-6, M-6, LOW-7) — replaced fictional `prepare_rollout_batch` / `compute_landmark_loss` / `odeint_adjoint` / NFE return-dict pseudocode with a pure adapter that calls the existing `node_rollout(node, segment, phase_name)` function unchanged. Return dict is `{"loss": loss}` only — no NFE (C-2: no counter exists in IonicNODE). Exit criteria now verify `node_rollout.py` + `node.py` source hashes unchanged (M-6).

**MUTATED 2026-04-19**: Step 4.2 Risk MODIFIED — fixed `oc.env:SURROGATE_CACHE_DIR` resolver syntax to `${oc.env:SURROGATE_CACHE_DIR,/media/HDD/surrogate_data/raw}` (L-4); changed default from volatile `/tmp` to HDD path.

**MUTATED 2026-04-19**: Step 4.3 Verify + Exit MODIFIED — added `tracking.tracking_uri=./mlruns_smoke` override and cleanup so the smoke test doesn't pollute top-level `./mlruns/` (M-4). Removed stale `metrics.nfe` reference from post-run query (C-2). Removed "Add the counter if absent" mitigation from Risk.

**MUTATED 2026-04-19**: Step 4.4 MAJOR MODIFICATION (H-3, H-7, M-8) — added:
  - Architecture-drift precondition: Phase 4 aborts if `Surrogate/surrogate/model/` diverges from SHA `8f191f77` (Session 25 oracle).
  - SHA pin + drift-detection: `git rev-parse HEAD` recorded at start, verified at end.
  - Mid-flight validation gate: new `SessionParityGate` callback reads `multi_bcl_002/log.jsonl` and aborts if current val_loss exceeds 3× reference at any 50-epoch checkpoint. New `cardiac_ml/tests/parity_gate.py` utility and tests added.

**MUTATED 2026-04-19**: Phase 4 Cleanup MODIFIED — removed "additive NFE counter allowed" language (C-2); added explicit `git diff --stat 8f191f77` check for model + rollout files (M-6).

**MUTATED 2026-04-19**: Step 5.1 Pseudocode ADDED a note — `hparams_search` is a loose config group NOT in the top-level defaults list; `+` prefix adds at runtime, keeping non-sweep runs clean (L-8).

**MUTATED 2026-04-19**: Step 5.2 MAJOR MODIFICATION (H-4) — demoted `shap.DeepExplainer` entirely. `shap.KernelExplainer` is now the sole explainer path (DeepExplainer is incompatible with torchdiffeq's adjoint/odeint gradient routing). New `kernel_shap_v_only()` interface; added `! grep DeepExplainer` verify guard.

**MUTATED 2026-04-19**: Step 5.4 MAJOR MODIFICATION (C-3) — replaced `git mv Surrogate/runs archive/runs_legacy` (fails on partially-tracked dir) with plain `mv` + subsequent `git add -A archive/runs_legacy Surrogate/runs` to let git see a rename for the tracked subset and a new-file add for the untracked subset. Depends on Step 1.3's `!archive/runs_legacy/` allowlist (M-5). Added pre-move stale-reference grep and post-move `git log --follow` history verification.

**MUTATED 2026-04-19**: Step 5.5 Verify + Exit MODIFIED — tightened from "manual review" to concrete grep checks for MASTER.md state markers (L-5).

**MUTATED 2026-04-19**: Step 5.6 Implementation + Verify + Exit MODIFIED — explicitly reconciles the 10-vs-9 completion-criteria drift between PLAN and README by adding the tracking-disable shim row to README (L-3). Tightened verify from "manual review" to concrete grep checks (L-5).

### Pass-2 design choice recorded in log

**M-2 / M-7 resolution (NullLogger routing)**: chosen over the two alternatives discussed during the audit response:
  - (rejected) Raise `RuntimeError` if `log_artifact` called while tracking=off — less flexible, surprises `train_step_fn` authors.
  - (rejected) Keep direct `mlflow.log_artifact` in Trainer with `if enabled` gate — violates the "Trainer does NOT import mlflow directly" invariant from Step 3.3.
  - (chosen) Route through `self._logger`. `NullLogger.log_artifact` is a no-op; `MLflowLoggerCallback.log_artifact` forwards to `mlflow.log_artifact`. Trainer stays MLflow-free; both tracking=on and tracking=off paths are a method dispatch (mockable, testable, silent on off).

### Decisions remaining open (carried into execution)

- OPEN-2: BayesOpt `evaluate()` shape. Deferred to Future Work — no Optimizer V1 driver yet.
- OPEN-4: SHAP input scope settled as V-only at Step 5.2 (simpler baseline).
- OPEN-5: Optuna pruner tuning — median + 20-epoch warmup, verified during Step 5.1 sweep.
- OPEN-6: `_backward_done` flag shape settled at Step 3.4 (with assertions).
- OPEN-8: Tracking-disable shim — NullLogger path at Step 3.4.

---

### Blueprint-revise pass 2 — 2026-04-19 — Round-2 audit findings (27 issues: 3C/7H/9M/8L)

**MUTATED 2026-04-19 (R2)**: Step 3.2 Interfaces + Step 3.3 Interfaces MODIFIED — added `log_artifact(path)` and `log_figure(fig, name)` proxy hooks to the `Callback` base class (no-op defaults); MLflowLoggerCallback overrides with real `mlflow.log_artifact`/`log_figure` forwarding; NullLogger inherits no-op base. (Round-2 C-3 fix — previous revision's Step 3.4 dispatched `self._logger.log_artifact(...)` but neither NullLogger nor the base Callback defined the method, causing AttributeError.)

**MUTATED 2026-04-19 (R2)**: Step 4.1 FULL REWRITE — replaced broken pseudocode (Round-2 C-1 / C-2 / H-1):
  - `node_rollout()` returns a DICT, not a scalar. Adapter now extracts `result["loss"]` and surfaces per-component scaffold metrics (`ionic_state_mse`, `conc_mse`, `conductance_mse`) as detached extras.
  - `phase_name` is now REQUIRED (no silent default). Valid values A1-A4 / B1-B4 / ionic_state / conc_only / ionic_state_and_conductance (NOT "train"/"val"). Helper `_phase_from_cfg` raises KeyError with valid-values list on missing config.
  - Added `_on_after_backward` callback protocol: adapter returns a `_clear` closure that calls `node.clear_v_trajectory()`; Trainer invokes it after `loss.backward()` on train path and after the step on val path (H-1 — required per `node_rollout.py:81` contract).
  - Trainer `_run_epoch` updated to dispatch `_on_after_backward` hooks.
  - Test spec reworked: added `test_node_step_raises_without_phase_name`, `test_clear_v_trajectory_invoked_by_trainer`; removed baseline-less `test_node_rollout_unchanged` sha256 test in favor of `git diff --quiet 8f191f77` (Round-2 H-7 fix).

**MUTATED 2026-04-19 (R2)**: Phase 4 Context MODIFIED — rewrote node_rollout API description with correct return-dict shape, phase_name semantics as loss-composition selector, and explicit cleanup-hook contract. Fixed data-path default from `/media/HDD/surrogate_data/raw/` (nonexistent) to `/media/HDD/norepinephrine/surrogate_data/raw/` (verified 2026-04-19) (Round-2 H-3).

**MUTATED 2026-04-19 (R2)**: Step 4.2 Risk MODIFIED — same data-path fix as Phase 4 Context (Round-2 H-3).

**MUTATED 2026-04-19 (R2)**: Step 4.3 Checklist MODIFIED — removed leftover "nfe metrics" bullet (Round-2 M-2); aligned with the harness-side metric set (`train_loss`, `val_loss`, per-component scaffold metrics from `node_rollout` return dict).

**MUTATED 2026-04-19 (R2)**: Step 4.4 REWRITE of the precondition, gate, and parity-check blocks:
  - Removed buggy `[ "$MODEL_SHA" != "8f191f77..." ]` literal-ellipsis comparison (Round-2 H-5). The `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` check is now the sole authoritative drift detector.
  - Added stable-location copy: `cardiac_ml/reference/session25_log.jsonl` (copied at Phase 4 start, committed) so post-cutover re-runs still find it (Round-2 H-4).
  - Updated `compare_to_session25` signature: tolerance schedule (5× for epochs 10..50, 3× for >=50), `min_epoch=10` skips transient early spikes (Round-2 M-5). Moved to `cardiac_ml/analysis/parity_gate.py`.
  - Parity threshold: `≤ 1.05 × oracle best = 0.0088`, NOT `≤ 0.008` (Round-2 M-7 — actual oracle best is 0.00838; a strict 0.008 gate would reject the oracle itself).
  - Verify command uses pandas `.fillna('').str.startswith(...)` filter on `tags.mlflow.runName` instead of the dialect-sensitive `LIKE` clause (Round-2 L-5).
  - Risk item about multi_bcl_002 log schema replaced with the actual verified schema note (Round-2 L-4).

**MUTATED 2026-04-19 (R2)**: Phase 4 Cleanup MODIFIED — scipy grep scoped to `cardiac_ml/` only (Round-2 L-1 — scipy is allowed project-wide). Added explicit check for `cardiac_ml/reference/session25_log.jsonl` presence.

**MUTATED 2026-04-19 (R2)**: Step 5.4 Pseudocode + Verify MODIFIED — broadened stale-reference grep pattern to `(Surrogate/)?runs/[a-zA-Z0-9_]` to catch relative `runs/...` references in `Surrogate/run_multi_bcl.py` etc. (Round-2 H-2). Added pre-commit `git status archive/runs_legacy` inspection for accidental .DS_Store / lock-file staging.

**MUTATED 2026-04-19 (R2)**: Step 1.3 Pseudocode + Checklist + Verify MODIFIED — added `!archive/runs_legacy/**` recursive allowlist pattern (Round-2 M-8). Per git docs, `!pattern/` un-ignores a directory entry but NOT its contents; the `**` form is required to re-include descendants. Also relaxed verify regex to tolerate both `archive/*` and `archive/**` spellings (Round-2 L-7).

**MUTATED 2026-04-19 (R2)**: Step 2.1 Pseudocode MODIFIED — replaced bare try/except with proper PEP 562 `__getattr__` for lazy `Trainer` access (Round-2 H-6). Pre-Step-3.4 access to `cardiac_ml.Trainer` now raises a clean ImportError pointing at the missing implementation instead of binding a dummy or silent failure.

**MUTATED 2026-04-19 (R2)**: Step 2.5 Pseudocode MODIFIED — concretized the Phase-2-scoped importability mechanism (Round-2 M-6). New file `cardiac_ml/tests/_deferred_targets.py` holds a module-prefix set; the test skips imports whose prefix is in the set. Step 4.1 exit now includes removing `surrogate.training.node_step` from the set to activate the Phase-4 check.

**MUTATED 2026-04-19 (R2)**: Step 3.1 Pseudocode ADDED — protocol-keys documentation block listing `_backward_done` and `_on_after_backward` as harness-wide optional hooks. Most `train_step_fn`s omit both; NODE uses `_on_after_backward`.

**MUTATED 2026-04-19 (R2)**: Step 3.4 Pseudocode MODIFIED:
  - Batch casting now includes DTYPE promotion: `_to_device_and_dtype(batch, device, dtype)` (Round-2 M-4) — prevents silent float32→float64 mixing when DataLoader yields default float32.
  - `_backward_done` assertion message clarified: "loss is not attached to any compute graph. Did you return a constant tensor instead of the loss used in backward()?" (Round-2 M-9).
  - Added `_on_after_backward` post-hook dispatch on both train and val paths.
  - Comment on `log_artifact` / `log_figure` clarified: base class defines them as no-ops; NullLogger inherits (Round-2 C-3 cross-ref).

**MUTATED 2026-04-19 (R2)**: Step 5.2 Test Spec MODIFIED — capped `test_kernel_shap_handles_torchdiffeq_model` at `nsamples=5`, `T=3` to keep unit test under a minute (Round-2 M-3). Added `@pytest.mark.slow` recommendation.

**MUTATED 2026-04-19 (R2)**: Final Cleanup MODIFIED — added REQUIREMENTS.md §7.2 patch step (Round-2 L-3: update the misleading dopri8/rtol=1e-5 sketch to match production dopri5/rtol=1e-3/adjoint=False). Clarified MLflow isolation check: `trainer.py` should NOT import mlflow (only `mlflow_logger.py`), matching the C-3 dispatch architecture.

### Round-2 design choices recorded (no new open items)

- **Clear-V-trajectory via `_on_after_backward` hook**: chosen over alternatives (add Trainer-side model-type check, add to MLflowLoggerCallback, modify `node_rollout` to auto-clear). The hook is the only option that (a) keeps `node_rollout.py` unchanged, (b) generalizes to any future stateful model, and (c) stays inside the Trainer protocol surface.
- **Per-component scaffold metrics surfaced in return dict**: adds `ionic_state_mse`, `conc_mse`, `conductance_mse` to MLflow as auto-logged metrics. These are invaluable for parity debugging (Session 25 tracked them) and cost nothing.
- **Parity threshold 0.0088 (oracle-relative)** rather than 0.008 (oracle-absolute): the oracle itself hits 0.00838. A 5% envelope around the oracle is fair; strict sub-oracle thresholds would reject correct reproductions.

---

### Blueprint-revise pass 3 — 2026-04-19 — Round-3 audit (30 issues: 4C/5H/12M/9L)

**Strategy**: Round 3 found 4 new CRITICALs and multiple HIGH issues concentrated in Phase 4 (NODE pilot migration). Rather than patch Phase 4 Steps 4.1-4.4 for a third time, this pass inserts **Step 4.0 — Phase 4 Reality Check** as an exploratory step. Steps 4.1-4.4 are marked `[DEFERRED — pending Step 4.0]` and must be re-specified via a subsequent `/blueprint-revise` pass once the reality check produces verified facts about `SegmentDataset` / `make_dataloaders` / `phase_name` / parity-gate infrastructure. Non-Phase-4 issues fixed in-place this pass.

**Applied — non-Phase-4 (safe to fix now)**:

**MUTATED 2026-04-19 (R3)**: Step 3.3 Pseudocode MODIFIED (Round-3 C-4) — `cfg.experiment.name` replaced with `_derive_run_name(trainer)` method that tries `cfg.experiment.name` → Hydra's `runtime.choices["experiment"]` → `"cardiac_ml_run"` fallback. Also added concrete implementations of `_flatten` and `_is_scalar` helpers that were previously referenced without definition (Round-3 LOW-3).

**MUTATED 2026-04-19 (R3)**: Step 3.4 Pseudocode MODIFIED:
  - Logger dedup: Trainer checks `cfg.training.callbacks` for existing `MLflowLoggerCallback`/`NullLogger` instances before appending `self._logger` (Round-3 HIGH-2 — otherwise hparams_search configs with an explicit logger double-start MLflow runs).
  - `_to_device_and_dtype` helper now has concrete pseudocode inline (Round-3 HIGH-4) — float-only promotion, dict/list recursion, non-tensor pass-through.
  - `_on_after_backward` dispatch wrapped in try/except that zero-grads and re-raises with context (Round-3 MED-9) — prevents silent optimizer-skip + stale-grad corruption on hook failure.

**MUTATED 2026-04-19 (R3)**: Step 3.4 Checklist MODIFIED — count updated from "3 tests" to "8 tests" matching actual Test Spec (Round-3 MED-4). Added checklist items for `_to_device_and_dtype`, `_on_after_backward` try/except, and logger dedup.

**MUTATED 2026-04-19 (R3)**: Step 2.1 Checklist MODIFIED — removed stale "try/except" language; replaced with PEP 562 `__getattr__` description matching the R2-updated Pseudocode (Round-3 MED-5).

**MUTATED 2026-04-19 (R3)**: Step 2.4 MAJOR MODIFICATION:
  - `TrainingConfig` dataclass expanded to include `callbacks`, `phase_name`, `ode_method`, `ode_rtol`, `ode_atol`, `ode_adjoint` as Optional fields (Round-3 MED-7) — previously the schema rejected valid NODE configs.
  - `TrackingConfig` added `tracking_uri` field.
  - `_register()` location clarified: called once from `scripts/train.py` before `@hydra.main`, NOT from `cardiac_ml/__init__.py` (Round-3 MED-6) — otherwise conflicts with Step 2.1's lazy `__getattr__` (would force unconditional Hydra import at module load).

**MUTATED 2026-04-19 (R3)**: Step 3.6 conftest.py spec MODIFIED — `mlflow_tmpdir` fixture now declared `@pytest.fixture(autouse=True, scope="session")` (Round-3 MED-10) — without `autouse=True`, tests don't auto-pick-up and pollute real `./mlruns/`. Added note that subprocess tests (end-to-end) need explicit `tracking.tracking_uri` CLI override because subprocesses don't inherit pytest fixtures.

**MUTATED 2026-04-19 (R3)**: Step 5.1 YAML MODIFIED — added explicit `pruner:` block with `MedianPruner`, `n_startup_trials: 5`, `n_warmup_steps: 20` (Round-3 LOW-4 — OPEN-5's "median + warmup" was a trailing comment, not a YAML directive, so pruner was never actually configured).

**MUTATED 2026-04-19 (R3)**: Step 5.2 Implementation Spec MODIFIED — added full `scripts/analyze.py` pseudocode (Round-3 MED-11 — script was only described prose-wise; Success Criterion #8 depends on it).

**MUTATED 2026-04-19 (R3)**: Step 5.3 Test Spec MODIFIED — relaxed "val_loss decreases monotonically" to `val_loss[epoch=9] < val_loss[epoch=0] * 0.9` (Round-3 LOW-1 — strict monotonicity is too brittle for randomly-initialized CNN).

**MUTATED 2026-04-19 (R3)**: Phase 5 Cleanup MODIFIED — mlflow-grep expectation now excludes `trainer.py` (Round-3 MED-3 — R2-C3's dispatch-through-logger architecture means `trainer.py` should NOT import mlflow directly).

**MUTATED 2026-04-19 (R3)**: Phase 1 Verification MODIFIED — tightened gitignore grep regex to match Step 1.3's actual patterns including the `!archive/runs_legacy/**` allowlist (Round-3 LOW-2).

**Applied — Phase 4 structural changes**:

**MUTATED 2026-04-19 (R3)**: Step 4.0 INSERTED — "Phase 4 Reality Check" exploratory step that produces `Research/Active/cardiac_ml_harness/results/phase4_reality_check.md`. Agent reads 10 specific files, documents 10 topics (Dataset API, Dataloader construction, Phase naming, Parity gate source, Reference log coverage, `cfg.experiment.name` population, Wall-time budget, V-trajectory lifecycle, SHA pin robustness, Scaffold decoder presence at pinned SHA), ends with concrete "Revisions needed before Step 4.1 is executable" section that drives a subsequent `/blueprint-revise` pass.

**MUTATED 2026-04-19 (R3)**: Phase 4 Goal MODIFIED — parity threshold updated from "0.008" to "≤ 1.05 × oracle best = 0.0088" to match R2 M-7 fix. Scope updated to "1 exploratory + 4 deferred implementation steps".

**MUTATED 2026-04-19 (R3)**: Steps 4.1, 4.2, 4.3, 4.4 each prefixed with `[DEFERRED — pending Step 4.0]` banner and a short warning describing which specific Round-3 finding blocks execution. Agents reading these steps must NOT execute as-written.

**Deferred to post-Step-4.0 `/blueprint-revise` pass**:

- **Round-3 CRITICAL-1** (parity gate dead code — reference log 0-7 epochs, min_epoch=10): Step 4.0 topics 4-5 decide whether to rewrite the gate to work with the actual 8-epoch range or replace with a different comparator.
- **Round-3 CRITICAL-2** (SessionParityGate utility reads JSONL but Trainer writes MLflow): Step 4.0 topic 4 decides shadow-log callback vs MLflow-reader refactor.
- **Round-3 CRITICAL-3** (`phase_name` missing from Step 4.2 YAML): Step 4.0 topic 3 identifies the correct phase_name for Session 25, then Step 4.2 re-spec adds it.
- **Round-3 HIGH-1** (`conf/data/t1_multi_bcl.yaml` fabricates train_bcls/val_bcls): Step 4.0 topics 1-2 capture the real SegmentDataset + make_dataloaders API; Step 4.2 re-spec uses that shape.
- **Round-3 HIGH-3** (Step 4.1 "Files modified: trainer.py" is phase-ordering violation): resolved by acknowledging that the `_on_after_backward` dispatch IS in Phase 3 commit; Step 4.1 just uses it. Re-spec will remove the "Step 3.4 addendum" block.
- **Round-3 HIGH-5** (SHA pin 8f191f77 no override path): Step 4.0 topic 9 verifies whether the pin is stale; adds escape-hatch config if needed.
- **Round-3 MED-1** (Phase 4 Verification has comment instead of command): fixed in the post-4.0 revise pass.
- **Round-3 MED-2** (success-criteria drift 0.008 vs 0.0088 at lines 17, 1176, 1623): fixed in post-4.0 revise.
- **Round-3 MED-8** (Step 4.1 Checklist missing `_deferred_targets.py` update): fixed in post-4.0 revise.
- **Round-3 MED-12** (Step 5.4 `--exclude-dir=runs` over-excludes top-level `runs/`): fix in post-4.0 revise.
- **Round-3 LOW-5** (`_on_after_backward` not in REQUIREMENTS.md §7): add to REQUIREMENTS patch in Final Cleanup after protocol shape stabilizes in Step 4.0.
- **Round-3 LOW-6** (Mutation Log heading punctuation inconsistency): cosmetic; defer.
- **Round-3 LOW-7** (NFR-4 startup time not validated anywhere): add timing assertion to Step 3.5 verify in post-4.0 revise.
- **Round-3 LOW-8** (/tmp/surrogate_cache/ missing): Step 4.0 topic 2 decides cache strategy.
- **Round-3 LOW-9** (gitignore allowlist promotes untracked subdirs): Step 5.4 re-spec post-4.0 considers explicit `git rm -f` before allowlist fires.

### Round-3 summary

- **15 issues fixed in-place** (all non-Phase-4: C-4 + HIGH-2 + HIGH-4 + 10 MEDIUMs/LOWs).
- **1 new step inserted** (Step 4.0 reality check).
- **4 steps marked deferred** (Steps 4.1-4.4, executable after Step 4.0 + revise pass).
- **15 Phase-4-scoped issues deferred** (tracked above, resolved by Step 4.0 output + subsequent `/blueprint-revise`).

### Execution order after this revision

1. Phase 1 + Phase 2 + Phase 3 can execute as currently specified. Round-3 in-place fixes apply to these and are safe.
2. Phase 4 begins with Step 4.0 (reality check only). Do NOT attempt Steps 4.1-4.4 from current text.
3. After Step 4.0 lands, run `/blueprint-revise` with the reality-check doc as input to re-specify Steps 4.1-4.4.
4. Phase 5 can then execute normally.

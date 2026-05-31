# PLAN: Consolidation Phase 1 — Unify ionic models into cardiac_core/ionic/

Created: 2026-05-30
Engine(s): All (cardiac_core + Monodomain V5.5 + Bidomain V1 + LBM V1)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-05-30 "Phase 1 scoped — ionic is ~unified across engines; full direct migration decided"

## Objective
Establish `cardiac_core/ionic/` as the **canonical copy** of the ionic models (TTP06, ORd, MHAS13, PHAS13, Paci + base/lut) and make `cardiac_core` a properly importable package — the go-forward home for shared ionic code. **COPY-ONLY: do NOT delete or rewire any engine's ionic.** Engines (V5.5, Bidomain V1, LBM V1) and downstream consumers (Surrogate, Optimizer) keep their existing `ionic` imports untouched. This sidesteps the repo-wide deletion blast radius the audit found, while creating the canonical core that future opportunistic migration will target.

> **Scope decision (2026-05-30, post-audit):** the original plan deleted each engine's `ionic/` and rewired imports. The audit found this breaks engine TEST suites, examples, AND active cross-project consumers (Surrogate data generators + Optimizer import `cardiac_sim.ionic` via the Bidomain path). User decision: **don't delete the originals — just copy them over.** Engine migration (rewire + delete + migrate downstream consumers) is **DEFERRED** to a later phase, to be done per-consumer and safely, not as a big-bang deletion.

## Success Criteria
- [ ] `cardiac_core/ionic/` is the canonical superset copy (one keyword fix); `import cardiac_core.ionic` works from any cwd (cardiac_core editable-installed), without triggering the engine-wrapping `api.py`
- [ ] cardiac_core's own suite green: 77 tests pass
- [ ] **NO engine or downstream consumer modified** — `git status` shows changes only under `cardiac_core/` + new `pyproject.toml`. V5.5/Bidomain/LBM/Surrogate/Optimizer all still import their existing ionic and still pass (spot-check at least one engine suite to confirm nothing was disturbed)
- [ ] Duplication is knowingly retained (engines keep their copies); the "zero-duplication / engines import from cardiac_core" end-state is recorded as DEFERRED, not done
- [ ] Partially advances README Completion Criteria "Phase 1: ionic models in cardiac_core/ionic/" (canonical copy exists; engine rewiring deferred)

## Architecture Changes
- NEW: `cardiac_core/ionic/` — canonical ionic package (copied from V5.5's `cardiac_sim/ionic/`, the superset; one keyword fix)
- NEW: `pyproject.toml` (repo root) — declares `cardiac_core` (+ subpackages) installable; `pip install -e .` into the `heart-conduction` env
- MOD: `cardiac_core/__init__.py` — make LAZY (PEP 562 `__getattr__`) so `import cardiac_core.ionic` does NOT trigger `api.py`/`_prepare_engine` (clean-import hygiene; also unblocks any future consumer that imports cardiac_core.ionic)
- DO NOT MODIFY (this phase): any engine's `ionic/` or its importers; `Monodomain/Engine_V5.3/`, `Monodomain/Engine_V5.4/`, `Bidomain/Engine_V1/`, `LBM/Engine_V1/`, `Surrogate/`, `Optimizer/`. They keep their current ionic.

## Known Failures (from IDEALOG + audit)
- **Delete engine ionic + rewire (the original Phases 2–3)** — REJECTED (2026-05-30): the audit found deletion breaks engine test suites (V5.5 `tests/`, `test_phase10`, examples; Bidomain `test_phase1_foundation`/`cv_shared`/experiments; all LBM `tests/`) AND active downstream consumers (`Surrogate/surrogate/data/*_generator.py`, `Optimizer/V1/tuner/tissue_runner_bidomain.py`) that import `cardiac_sim.ionic`/top-level `ionic` via sys.path-inserted engine paths. User: don't delete; copy only.
- **Re-export shim (Option A)** — rejected earlier (per-engine residue).
- **`sys.path` hack for cardiac_core importability** — rejected: use a real editable-installed package.
- **Cross-engine in-process imports** — impossible: V5.4/V5.5/Bidomain share the `cardiac_sim` package name. cardiac_core is a separate top-level package, unaffected.
- **Assuming engine ionic is byte-identical** — classical engines ARE identical; LBM differs by one keyword + namespace + model subset + dead cruft. Canonical source = V5.5 (superset).

---

## Phase 1: Canonical cardiac_core/ionic/ + packaging + lazy __init__

**Goal**: `cardiac_core/ionic/` exists as the canonical superset, cleanly importable everywhere without dragging in the engine-wrapping `api.py`. **This is the entire deliverable for now** — no engine is touched (copy-only). Fully additive and reversible.
**Tier**: large
**Estimated scope**: copy ionic tree, 1 keyword fix, lazy `__init__`, pyproject + editable install

### Phase Context
- **Canonical source = `Monodomain/Engine_V5.5/cardiac_sim/ionic/`** — the superset: has `paci/`, all models, and the `PaciModel = PHAS13Model` backward-compat alias in `__init__.py`. It is byte-identical to Bidomain V1 for the shared models.
- **The one reconciliation:** `ttp06/model.py:303` in V5.5 calls `self._lut.get_all_gating(V, cell_type_is_endo=...)`, but `lut.py:209 get_all_gating` is defined with `celltype_is_endo` (in ALL engines). This is a LATENT bug in V5.5/Bidomain's LUT-enabled path (only fires when LUT is on; default path is unaffected, which is why tests pass). The canonical copy MUST use `celltype_is_endo` (matches lut.py; LBM already does). This fixes the latent bug with no test regression.
- **Circular-import hazard:** `cardiac_core/__init__.py` currently eagerly imports `api`, `run`, `analysis`, `geometry`, `io`, `file_format`. `api.py` runs `_prepare_engine()` (flushes `cardiac_sim`, imports V5.4/Bidomain). Once engines import `cardiac_core.ionic`, Python runs `cardiac_core/__init__.py` first — so an eager `api` import would re-enter engine setup mid-import. The package import MUST be light. Fix: lazy `__init__` (PEP 562).
- **`cardiac_core.ionic` must stay import-light**: it may import only torch/stdlib internally (it already does). It must NOT import `cardiac_core.api`/engines.
- Do NOT touch V5.4 — it is the frozen baseline and keeps its own ionic.

### Step 1.1: Copy ionic tree into cardiac_core + reconcile keyword
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/cardiac_sim/ionic/__init__.py` — the superset exports (note `PaciModel` alias).
- `Monodomain/Engine_V5.5/cardiac_sim/ionic/ttp06/model.py:303` — the `cell_type_is_endo=` call to fix.
- `Monodomain/Engine_V5.5/cardiac_sim/ionic/lut.py:209` — confirms `celltype_is_endo` is the correct kwarg.

#### Why
V5.5 is the most complete, validated ionic tree. Copying it verbatim (then one keyword fix) guarantees the canonical copy is correct and the classical engines' behavior is preserved when they later import it.

#### Implementation Spec
**Files to create:** `cardiac_core/ionic/` = recursive copy of `Monodomain/Engine_V5.5/cardiac_sim/ionic/` (base.py, lut.py, __init__.py, ttp06/, ord/, mhas13/, phas13/, paci/, celltypes within models). Strip `__pycache__`.
**Files to modify:** `cardiac_core/ionic/ttp06/model.py` — change the single `cell_type_is_endo=` call to `celltype_is_endo=`.

#### Pseudocode
```
cp -r Monodomain/Engine_V5.5/cardiac_sim/ionic cardiac_core/ionic
find cardiac_core/ionic -name __pycache__ -prune -exec rm -rf {} +
# fix the latent LUT keyword (only differing line vs LBM's correct version):
sed -i 's/cell_type_is_endo=/celltype_is_endo=/' cardiac_core/ionic/ttp06/model.py
# confirm internal imports are all relative (within ionic) — no cardiac_sim refs:
grep -rn "cardiac_sim\|cardiac_core" cardiac_core/ionic   # expect NONE
```

#### Test Spec
- Import smoke: `python -c "from cardiac_core.ionic import IonicModel, TTP06Model, ORdModel, CellType, MHAS13Model, PHAS13Model, PaciModel"` succeeds.
- LUT path: instantiate TTP06 with LUT enabled, call a step — no `TypeError` on `get_all_gating` (the keyword fix).

#### Checklist
- [ ] `cardiac_core/ionic/` copied, caches stripped
- [ ] `ttp06/model.py` uses `celltype_is_endo=`
- [ ] `grep cardiac_sim cardiac_core/ionic` → nothing (ionic is namespace-clean)
- [ ] internal imports are relative (`from .base`, `from ..lut`, etc.)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -c "import sys; sys.path.insert(0,'.'); from cardiac_core.ionic import IonicModel,TTP06Model,ORdModel,CellType,MHAS13Model,PHAS13Model,PaciModel; print('ionic superset imports OK')"
grep -rn "cardiac_sim" cardiac_core/ionic --include=*.py | grep -v __pycache__ || echo "clean: no cardiac_sim refs"
```

#### Exit Criteria
- [ ] Canonical `cardiac_core/ionic/` imports cleanly; keyword fixed; namespace-clean.

#### Risk
A model submodule using an absolute `cardiac_sim.ionic...` import would break — mitigation: the grep catches it; convert any to relative.

### Step 1.2: Lazy cardiac_core/__init__.py (circular-import guard)
**Model**: opus

#### Read First
- `cardiac_core/__init__.py` — current eager imports (file_format, api, run, analysis, geometry, io).
- `cardiac_core/api.py:25-37` — `_prepare_engine()` (the engine-flush this must not trigger on a bare `cardiac_core.ionic` import).

#### Why
If `cardiac_core/__init__.py` eagerly imports `api`, then any engine doing `from cardiac_core.ionic import X` triggers `_prepare_engine()` mid-engine-import — flushing `cardiac_sim` and recursing. Making `__init__` lazy means importing `cardiac_core.ionic` runs only a light parent `__init__`.

#### Implementation Spec
**Files to modify:** `cardiac_core/__init__.py` — replace eager submodule imports with PEP 562 lazy loading. Keep the public names available (`monodomain`, `bidomain`, `lbm`, `run_monodomain`, `activation_time`, `create_cardiac_mesh`, …) but resolve them on first attribute access via `__getattr__`. `ionic` is a normal subpackage (import on demand). `file_format` may stay eager (light) or also be lazy.

#### Pseudocode
```python
# cardiac_core/__init__.py
import importlib
_LAZY = {                      # public name -> submodule that defines it
    'monodomain': 'api', 'bidomain': 'api', 'lbm': 'api',
    'CardiacSimulation': 'api', 'SimulationSnapshot': 'api', 'Distribution': 'api',
    'run_monodomain': 'run', 'run_bidomain': 'run', 'run_lbm': 'run',
    'simulate': 'run', 'SimulationResult': 'run',
    'activation_time': 'analysis', 'conduction_velocity': 'analysis', 'apd_at': 'analysis',
    'apd_map': 'analysis', 'dominant_frequency': 'analysis', 'wavefront_mask': 'analysis',
    'phase_map': 'analysis', 'phase_singularities': 'analysis', 'restitution_curve': 'analysis',
    'circle_mask': 'geometry', 'rectangle_mask': 'geometry', 'annulus_mask': 'geometry',
    'left_edge_mask': 'geometry', 'right_edge_mask': 'geometry', 'point_distance': 'geometry',
    'boundary_distance': 'geometry', 'fiber_field_uniform': 'geometry', 'fiber_field_transmural': 'geometry',
    'save_result': 'io', 'load_result': 'io',
    'CardiacMeshData': 'file_format', 'save_cardiac_mesh': 'file_format',
    'load_cardiac_mesh': 'file_format', 'create_cardiac_mesh': 'file_format',
}
def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f'.{_LAZY[name]}', __name__)
        return getattr(mod, name)
    raise AttributeError(name)
def __dir__():
    return sorted(_LAZY)
# NOTE: do NOT import .api/.run/.analysis at top level. `import cardiac_core.ionic`
# must not pull engine wrappers. ionic is a normal subpackage (no entry needed here).
```

#### Test Spec
- `python -c "import cardiac_core.ionic"` does NOT import `cardiac_sim` (assert `'cardiac_sim' not in sys.modules` afterward).
- `from cardiac_core import monodomain, activation_time, create_cardiac_mesh` still works (lazy resolves).
- cardiac_core's own 77 tests still pass.

#### Checklist
- [ ] `__init__` no longer eagerly imports api/run/analysis/geometry/io
- [ ] `__getattr__` lazy-resolves every previously-exported public name
- [ ] `import cardiac_core.ionic` leaves `cardiac_sim` out of `sys.modules`
- [ ] `cardiac_core/tests/` still pass (they use the public names)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -c "import sys; sys.path.insert(0,'.'); import cardiac_core.ionic; assert 'cardiac_sim' not in sys.modules, 'LEAK: api was triggered'; print('lazy OK — no engine import')"
conda run -n heart-conduction python -c "import sys; sys.path.insert(0,'.'); from cardiac_core import monodomain, activation_time, create_cardiac_mesh; print('lazy public API OK')"
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q 2>&1 | tail -3
```

#### Exit Criteria
- [ ] cardiac_core 77/77 pass; `cardiac_core.ionic` import is engine-free.

#### Risk
A lazy name missed in `_LAZY` → `AttributeError` for a consumer — mitigation: enumerate from the current `__init__` exports exactly; the 77-test run exercises the public surface.

### Step 1.3: Make cardiac_core an editable-installed package (no path hack)
**Model**: opus

#### Read First
- repo root `conftest.py` — current `sys.path` mechanism (adds Surrogate/). We are REPLACING the need for cwd-based cardiac_core discovery with a real install.
- `cardiac_core/__init__.py` (post 1.2).

#### Why
Engines run from their own dirs and flush `cardiac_sim`. For `from cardiac_core.ionic import X` to resolve in every engine/test process WITHOUT a per-engine `sys.path.insert`, cardiac_core must be on the env's import path as an installed package — and being a real top-level package makes it immune to `_prepare_engine`'s `cardiac_sim` surgery.

#### Implementation Spec
**Files to create:** `pyproject.toml` (repo root) declaring package `cardiac_core` (and its subpackages incl. `cardiac_core.ionic` and nested model dirs). Use an explicit package list or `setuptools.packages.find` scoped to `cardiac_core*` so the many non-package repo dirs are excluded.
**Install:** `pip install -e .` into the `heart-conduction` env.

#### Pseudocode
```toml
# pyproject.toml (repo root)
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "cardiac-core"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
include = ["cardiac_core*"]   # ONLY cardiac_core + subpackages; exclude engines/research
```
```bash
conda run -n heart-conduction pip install -e .
```

#### Test Spec
- From an unrelated cwd (e.g. `/tmp`): `python -c "import cardiac_core.ionic; print('importable anywhere')"` succeeds.

#### Checklist
- [ ] `pyproject.toml` includes ONLY `cardiac_core*` (verify it does not try to package Monodomain/Bidomain/LBM/Research)
- [ ] `pip install -e .` succeeds in the heart-conduction env
- [ ] `import cardiac_core.ionic` works from `/tmp` (cwd-independent)
- [ ] cardiac_core 77 tests still pass (now via installed package, not cwd)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction pip install -e . 2>&1 | tail -5
( cd /tmp && conda run -n heart-conduction python -c "import cardiac_core.ionic; print('cwd-independent import OK')" )
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q 2>&1 | tail -3
```

#### Exit Criteria
- [ ] `import cardiac_core.ionic` resolves from any cwd; 77 tests green.

#### Risk
`packages.find` accidentally packaging engine dirs → install bloat/conflict — mitigation: explicit `include = ["cardiac_core*"]`; inspect `pip show -f cardiac-core` lists only cardiac_core files.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -c "import sys; sys.path.insert(0,'.'); import cardiac_core.ionic; assert 'cardiac_sim' not in sys.modules; print('OK')"
( cd /tmp && conda run -n heart-conduction python -c "from cardiac_core.ionic import TTP06Model; print('OK')" )
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q 2>&1 | tail -3
```

### Phase 1 Exit Criteria
- [ ] Canonical `cardiac_core/ionic/` superset exists, keyword reconciled, namespace-clean
- [ ] `cardiac_core/__init__` lazy; `cardiac_core.ionic` import is engine-free
- [ ] cardiac_core editable-installed; importable from any cwd; 77 tests green
- [ ] NO engine or downstream consumer modified (`git status` shows only `cardiac_core/` + `pyproject.toml`)
- [ ] Spot-check: at least one engine suite still green (e.g. V5.5 `_regression/check_golden.py`) — confirms the copy/install disturbed nothing

### Phase 1 Cleanup
- float64: ionic copy unchanged numerically (only the keyword + locations).
- V5.3 / V5.4 / Bidomain / LBM / Surrogate / Optimizer not modified.
- Duplication is KNOWINGLY retained this phase (cardiac_core/ionic is the canonical go-forward copy; engines keep theirs until the deferred migration). This is the deliberate, audit-driven trade-off — not an oversight.

**-> Commit point: git commit after Phase 1 ("cardiac_core: canonical ionic/ superset + lazy __init__ + editable install (copy-only)")**

---

## Deferred: engine + downstream migration (NOT this phase)

The original Phases 2–3 (rewire each engine to `cardiac_core.ionic` and delete its local
`ionic/`) are **deferred** per the post-audit scope decision. They are recorded here as the
future target, with the audit-found constraints baked in so they aren't rediscovered late.

**Why deferred:** deleting an engine's `ionic/` breaks (verified by audit):
- the engine's OWN tests/examples (V5.5 `tests/`, `test_phase10_cm_scaling.py`, `examples/`;
  Bidomain `tests/test_phase1_foundation.py`, `tests/cv_shared.py`, `experiments/`; all LBM
  `tests/test_phase*.py` via `sys.path.insert(0,'.')` + `from ionic...`),
- active CROSS-PROJECT consumers: `Surrogate/surrogate/data/{single_cell,ord_single_cell,
  batch}_generator.py`, `voltage_clamp_ss.py` (insert Bidomain on path → `from
  cardiac_sim.ionic.ttp06.model import TTP06Model`); `Optimizer/V1/tuner/
  tissue_runner_bidomain.py` (`cardiac_sim.ionic.mhas13`).

**When undertaken, the migration MUST (audit requirements):**
1. Discover consumers REPO-WIDE, not just engine `simulation/` source — grep across each
   engine's `tests/`/`examples/`/root, plus `Surrogate/` and `Optimizer/`, for ALL of:
   `from cardiac_sim.ionic`, `from ...ionic` / `from ....ionic`, and bare `from ionic` /
   `import ionic` (LBM + `cv_shared.build_lbm_sim`).
2. Migrate downstream consumers (Surrogate datagen, Optimizer) in lockstep with any deletion,
   or keep the engine copy until they are migrated. Never delete out from under a live consumer.
3. Per-engine, process-isolated test gates (the `cardiac_sim` name collision forbids
   cross-engine in-process imports).
4. Keep V5.3 and V5.4 frozen (do not migrate or delete their ionic).
5. Final survivor check must exclude V5.3, V5.4, `Monodomain/_archive/`, and
   `Research/code_examples/torchcor/` — those legitimately keep their own `ionic/`.

Sequencing when resumed: (a) Surrogate + Optimizer → `cardiac_core.ionic`; (b) classical
engines (V5.5 + Bidomain V1) incl. their tests/examples; (c) LBM V1 (top-level `ionic`,
drop dead `ionic/ionic/`). Each step: rewire → run that consumer's suite → only then delete.

---

## Final Cleanup (cross-phase de-sloppify)
- [ ] `git status` shows changes ONLY under `cardiac_core/` + new `pyproject.toml` (NO engine, NO Surrogate/Optimizer edits — copy-only).
- [ ] float64: no float32 leaks introduced (ionic copied verbatim + one keyword).
- [ ] V5.3 / V5.4 / Bidomain / LBM / Surrogate / Optimizer untouched.
- [ ] No `sys.path.insert` added for cardiac_core anywhere (editable install is the mechanism).
- [ ] `cardiac_core/ionic/` exists as the canonical copy; duplication with engines is KNOWINGLY retained (engine migration deferred — see "Deferred" section).
- [ ] Update README Completion Criteria: mark "Phase 1: ionic models in cardiac_core/ionic/" as PARTIAL — canonical copy done, engine rewiring deferred (do NOT check it fully).
- [ ] Update KNOWLEDGE migration table: "Phase 1 (copy)" DONE; cardiac_core editable-installed + lazy `__init__`; engine rewire + downstream-consumer migration listed as DEFERRED with the audit constraints.
- [ ] IDEALOG: record that cardiac_core is now editable-installed; and that the deferred migration must include Surrogate datagen + Optimizer consumers of engine ionic (audit finding).
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md \
   "Research/Active/engine_consolidation/plans/2026-05-30_consolidation-phase-1-unify-ionic.md"
```

## Mutation Log
_(execution-time mutations: `**MUTATED {date}**: Step X.Y SKIPPED/SPLIT/INSERTED — reason`)_

### Revision 2026-05-30 — post-audit descope to COPY-ONLY (12 findings: 2 CRIT, 4 HIGH, 2 MED, 4 LOW)
- **MUTATED 2026-05-30**: Phases 2 & 3 (rewire engines + delete local ionic) DEFERRED — audit found deletion breaks engine test suites/examples AND active cross-project consumers (Surrogate data generators, Optimizer) that import `cardiac_sim.ionic` via sys.path-inserted engine paths. User decision: "don't delete the originals — just copy them over." Objective/Success Criteria/Architecture rewritten to copy-only; the deferred migration is preserved with the audit constraints in the new "Deferred" section. This MOOTS audit CRITICAL-1 (V5.5 test imports), CRITICAL-2 (LBM tests grep wrong dir), HIGH-1 (Bidomain test consumers), HIGH-2 (Surrogate/Optimizer breakage) — none can fire without deletion.
- **MUTATED 2026-05-30**: Success Criterion "find -name ionic → only cardiac_core" REMOVED (audit HIGH-3: unachievable — V5.3/V5.4/_archive/torchcor legitimately keep ionic). Replaced with "no engine modified" + the deferred-migration survivor-exclusion list.
- **MUTATED 2026-05-30**: LUT re-export "touch-up" hedge dropped from scope (audit MED-2: V5.5 `ionic/__init__` already exports get_ttp06_lut/clear_lut_cache/TTP06LUT/LookupTable — verified). Step 1.3 verify standardized to the env python (audit LOW-1). Phase 1 (the kept phase) was audit-verified SOUND: lazy-`__init__`/circular guard correct, `_LAZY` list matches exports, ionic import-light, keyword fix exact, superset holds, packaging fine.

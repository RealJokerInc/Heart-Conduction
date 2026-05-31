# PLAN: Consolidation Phase 1 — Unify ionic models into cardiac_core/ionic/

Created: 2026-05-30
Engine(s): All (cardiac_core + Monodomain V5.5 + Bidomain V1 + LBM V1)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-05-30 "Phase 1 scoped — ionic is ~unified across engines; full direct migration decided"

## Objective
Make `cardiac_core/ionic/` the single canonical copy of the ionic models (TTP06, ORd, MHAS13, PHAS13, Paci + base/lut), then EXACTLY migrate all three engines to import from it (Option B — direct rewire; NOT re-export shims, NOT `sys.path` hacks). After this phase the ionic code exists in exactly one place and every engine + cardiac_core depends on it.

## Success Criteria
- [ ] `cardiac_core/ionic/` is the canonical superset; `import cardiac_core.ionic` works from any cwd without `sys.path` manipulation (cardiac_core editable-installed)
- [ ] Monodomain V5.5, Bidomain V1, and LBM V1 import ionic from `cardiac_core.ionic` (absolute), with their local `ionic/` directories DELETED
- [ ] `find . -type d -name ionic ! -path '*__pycache__*'` returns ONLY `./cardiac_core/ionic` (zero duplicated ionic trees)
- [ ] Every suite green: cardiac_core 77; V5.5 (test_phase7/8, test_boundary_modes, tests/, test_phase10_cm_scaling, _regression/check_golden); Bidomain V1 pytest; LBM V1 tests
- [ ] Mirrors README Completion Criteria "Phase 1: ionic models in cardiac_core/ionic/"

## Architecture Changes
- NEW: `cardiac_core/ionic/` — canonical ionic package (copied from V5.5's `cardiac_sim/ionic/`, the superset; one keyword fix)
- NEW: `pyproject.toml` (repo root) — declares `cardiac_core` (+ subpackages) installable; `pip install -e .` into the `heart-conduction` env
- MOD: `cardiac_core/__init__.py` — make LAZY (PEP 562 `__getattr__`) so `import cardiac_core.ionic` does NOT trigger `api.py`/`_prepare_engine` (circular-import guard)
- MOD: `Monodomain/Engine_V5.5/cardiac_sim/...` — `from ...ionic` / `from .....ionic.base` → `from cardiac_core.ionic...`; DELETE `cardiac_sim/ionic/`
- MOD: `Bidomain/Engine_V1/cardiac_sim/...` — same; DELETE `cardiac_sim/ionic/`
- MOD: `LBM/Engine_V1/src/...` — `from ionic...` → `from cardiac_core.ionic...`; DELETE `LBM/Engine_V1/ionic/` (incl. the dead stray `ionic/ionic/`)
- DO NOT MODIFY: `Monodomain/Engine_V5.3/` (read-only); `Monodomain/Engine_V5.4/` (frozen baseline — leave its `cardiac_sim/ionic/` intact)

## Known Failures (from IDEALOG)
- **Re-export shim (Option A)** — rejected: leaves per-engine residue; user wants exact migration, not forwarders.
- **`sys.path` hack for cardiac_core importability** — rejected: user directive "not as path". Use a real editable-installed package so `import cardiac_core` is immune to the `_prepare_engine` `cardiac_sim` flush.
- **A-then-B staging** — rejected: user wants the clean end-state directly.
- **Assuming all engine ionic is byte-identical** — false-ish: classical engines ARE identical; LBM differs by one keyword + namespace + model subset + dead cruft. Canonical source = V5.5 (superset).
- **Cross-engine in-process imports** — impossible: V5.4/V5.5/Bidomain all use the `cardiac_sim` package name. Run each engine's suite in its OWN process (as in the V5.5 work). cardiac_core is separate (top-level package), unaffected.

---

## Phase 1: Canonical cardiac_core/ionic/ + packaging + lazy __init__

**Goal**: `cardiac_core/ionic/` exists as the canonical superset and is cleanly importable everywhere, WITHOUT dragging in the engine-wrapping `api.py`. No engine touched yet — fully reversible.
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
- [ ] NO engine modified yet (V5.5/Bidomain/LBM ionic still present — `git status` shows only cardiac_core/ + pyproject.toml)

### Phase 1 Cleanup
- float64: ionic copy unchanged numerically (only the keyword + locations).
- V5.3 not modified; V5.4 not modified.
- No code duplication yet introduced (cardiac_core/ionic is the future single copy; engine copies removed in Phases 2–3).

**-> Commit point: git commit after Phase 1 ("cardiac_core: canonical ionic/ superset + lazy __init__ + editable install")**

---

## Phase 2: Migrate classical engines (Monodomain V5.5 + Bidomain V1)

**Goal**: V5.5 and Bidomain V1 import ionic from `cardiac_core.ionic`; their local `cardiac_sim/ionic/` directories are deleted; both suites green.
**Tier**: large
**Estimated scope**: ~5–8 real import edits per engine (runtime + TYPE_CHECKING) + delete dir, per engine

### Phase Context
- These two engines' ionic is byte-identical to the canonical copy (except V5.5's superset extras, all included) — so importing cardiac_core.ionic is behavior-preserving.
- **Import sites** (relative → absolute):
  - V5.5 runtime: `cardiac_sim/simulation/classical/monodomain.py:45` `from ...ionic import IonicModel, TTP06Model, ORdModel, CellType`.
  - V5.5 TYPE_CHECKING: `.../ionic_time_stepping/{base,rush_larsen,forward_euler}.py` `from .....ionic.base import IonicModel`.
  - Bidomain runtime: `cardiac_sim/simulation/classical/bidomain.py:133,136` `from ...ionic import TTP06Model / ORdModel`.
  - Bidomain TYPE_CHECKING: `.../ionic_stepping/{base,rush_larsen,forward_euler}.py` `from ....ionic.base import IonicModel`.
  - All become `from cardiac_core.ionic import ...` / `from cardiac_core.ionic.base import IonicModel`.
  - Re-grep each engine for `ionic` imports before editing — do not rely on this list being exhaustive.
- **Tests run per-engine, in-process is fine within ONE engine** (cardiac_sim is that engine's own). Do NOT import two engines in one process.
- cardiac_core is installed (Phase 1) → `from cardiac_core.ionic` resolves regardless of cwd, even after `_prepare_engine` flushes `cardiac_sim`.
- Do NOT modify V5.4.

### Step 2.1: Migrate Monodomain V5.5
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/cardiac_sim/simulation/classical/monodomain.py:45` and the three `ionic_time_stepping/*.py` TYPE_CHECKING imports.
- `Monodomain/Engine_V5.5/cardiac_sim/ionic/__init__.py` — names that must resolve from cardiac_core.ionic (they do — same superset).

#### Why
V5.5 is the active monodomain engine and our Cm-fix work lives here; migrating it first (with its rich suite incl. the Cm-scaling tests) is the strongest early signal that the canonical import is behavior-preserving.

#### Implementation Spec
**Files to modify:** every V5.5 file importing `...ionic` (grep first). Rewrite relative ionic imports to `from cardiac_core.ionic[...] import ...`.
**Files to delete:** `Monodomain/Engine_V5.5/cardiac_sim/ionic/` (entire dir) AFTER edits.

#### Pseudocode
```
grep -rln "ionic" Engine_V5.5/cardiac_sim --include=*.py | grep -v /ionic/ | grep -v __pycache__
# for each: from ...ionic import X      -> from cardiac_core.ionic import X
#           from .....ionic.base import -> from cardiac_core.ionic.base import
rm -rf Engine_V5.5/cardiac_sim/ionic
grep -rn "from .*ionic import\|\.\.\.ionic\|ionic\.base" Engine_V5.5/cardiac_sim --include=*.py | grep -v cardiac_core   # expect none referencing local ionic
```

#### Test Spec
- Full V5.5 suite (process-isolated, cwd = Engine_V5.5): test_phase7 7/7, test_phase8 7/7, test_boundary_modes, tests/test_{mhas13,paci,phas13}, test_phase10_cm_scaling 3/3, _regression/check_golden (max|dV|=0).

#### Checklist
- [ ] All V5.5 ionic imports point to `cardiac_core.ionic`
- [ ] `cardiac_sim/ionic/` deleted
- [ ] No remaining reference to a local `cardiac_sim.ionic`
- [ ] Cm=1 golden still exact (the migrated ionic is the same code)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
PY=/home/norepinephrine/.conda/envs/heart-conduction/bin/python
$PY _regression/check_golden.py
for t in test_phase7.py test_phase8.py test_boundary_modes.py tests/test_mhas13.py tests/test_paci.py tests/test_phas13.py test_phase10_cm_scaling.py; do
  echo "== $t =="; $PY "$t" 2>&1 | tail -2
done
```

#### Exit Criteria
- [ ] V5.5 suite fully green; `cardiac_sim/ionic/` gone; golden exact.

#### Risk
A name exported by V5.5's ionic `__init__` but missing from cardiac_core.ionic `__init__` → ImportError — mitigation: canonical copy IS V5.5's ionic (same `__init__`), so exports match exactly. Verify with the import smoke + suite.

### Step 2.2: Migrate Bidomain V1
**Model**: opus

#### Read First
- `Bidomain/Engine_V1/cardiac_sim/simulation/classical/bidomain.py:133,136` and `ionic_stepping/*.py` TYPE_CHECKING imports.
- `Bidomain/Engine_V1/tests/` — how the suite is invoked (pytest); note `cv_shared.py` constructs TTP06 via the engine.

#### Why
Bidomain V1 is the ground-truth engine and the V5.5 cross-validation oracle; its ionic is byte-identical to canonical, so migration must be behavior-preserving (and keeps the bidomain reaction `/Cm` intact).

#### Implementation Spec
**Files to modify:** every Bidomain file importing `...ionic` (grep first) → `from cardiac_core.ionic...`.
**Files to delete:** `Bidomain/Engine_V1/cardiac_sim/ionic/`.

#### Pseudocode
```
grep -rln "ionic" Bidomain/Engine_V1/cardiac_sim --include=*.py | grep -v /ionic/ | grep -v __pycache__
# rewrite relative ionic imports -> cardiac_core.ionic
rm -rf Bidomain/Engine_V1/cardiac_sim/ionic
```

#### Test Spec
- Bidomain V1 pytest suite green (incl. the phase6 boundary CV tests). A CV spot-check via `cv_shared.run_bidomain` still yields ~54.3 cm/s at Cm=1 (ionic unchanged).

#### Checklist
- [ ] Bidomain ionic imports → `cardiac_core.ionic`
- [ ] `cardiac_sim/ionic/` deleted
- [ ] pytest suite green

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1
conda run -n heart-conduction python -m pytest tests/ -q 2>&1 | tail -5
```

#### Exit Criteria
- [ ] Bidomain V1 suite green; `cardiac_sim/ionic/` gone.

#### Risk
Bidomain's `ionic/__init__` exported a name not in canonical (it's a SUBSET of V5.5, so unlikely) — mitigation: canonical is the superset; verify via suite.

### Phase 2 Verification
```bash
# process-isolated, one engine per invocation
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5 && /home/norepinephrine/.conda/envs/heart-conduction/bin/python _regression/check_golden.py
cd /home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1 && conda run -n heart-conduction python -m pytest tests/ -q 2>&1 | tail -3
```

### Phase 2 Exit Criteria
- [ ] V5.5 + Bidomain V1 suites green; both `cardiac_sim/ionic/` dirs deleted; imports point to cardiac_core.

### Phase 2 Cleanup
- float64 intact; V5.3/V5.4 untouched.
- `git status`: deletions under the two engines' `cardiac_sim/ionic/` + import-line edits only.

**-> Commit point: git commit after Phase 2 ("Migrate Monodomain V5.5 + Bidomain V1 to cardiac_core.ionic; delete local copies")**

---

## Phase 3: Migrate LBM V1

**Goal**: LBM V1 imports ionic from `cardiac_core.ionic`; its top-level `ionic/` (incl. dead `ionic/ionic/`) is deleted; LBM suite green.
**Tier**: large
**Estimated scope**: rewire LBM's ionic consumers (top-level `ionic` namespace) + delete dir

### Phase Context
- LBM uses a TOP-LEVEL `ionic/` package (not `cardiac_sim.ionic`). Its consumers resolve `ionic` via `sys.path.insert(0,'.')` from `LBM/Engine_V1/`. The ionic model code is the same (ABC byte-identical; ttp06 differs only by the keyword we already fixed canonically; ord identical). LBM only uses TTP06/ORd (subset) — canonical superset covers it.
- The ionic consumer entry is `LBM/Engine_V1/src/solver/rush_larsen.py` (used by `src/simulation.py`); grep the whole `src/` for `ionic` imports — there may be a few.
- `LBM/Engine_V1/ionic/ionic/` is a dead nested duplicate (imported by nothing) — delete with the rest.
- LBM tests run from `LBM/Engine_V1/` with `sys.path.insert(0,'.')`; cardiac_core is installed so `from cardiac_core.ionic` resolves there too.

### Step 3.1: Rewire LBM V1 ionic consumers + delete local ionic
**Model**: opus

#### Read First
- `grep -rn "from ionic\|import ionic\|ionic\." LBM/Engine_V1/src --include=*.py` — enumerate the real consumer sites.
- `LBM/Engine_V1/ionic/__init__.py` — exported names (IonicModel, CellType, ORdModel, TTP06Model, LookupTable, TTP06LUT, get_ttp06_lut, clear_lut_cache).
- `LBM/Engine_V1/src/solver/rush_larsen.py` — how it imports/uses the models.

#### Why
Completes the consolidation: after this, ionic exists ONLY in cardiac_core. LBM's LUT helpers (`get_ttp06_lut`, etc.) must resolve from cardiac_core.ionic — confirm canonical `__init__` exports them (V5.5's lut.py is the same module), else add to canonical exports.

#### Implementation Spec
**Files to modify:** LBM `src/` files importing `ionic` → `from cardiac_core.ionic import ...`.
**Files to delete:** `LBM/Engine_V1/ionic/` (entire tree, incl. `ionic/ionic/`).
**Possible canonical touch-up:** if LBM imports LUT names (`get_ttp06_lut`, `clear_lut_cache`, `TTP06LUT`, `LookupTable`) that V5.5's `ionic/__init__.py` does not re-export, add them to `cardiac_core/ionic/__init__.py` (they exist in `lut.py`).

#### Pseudocode
```
grep -rn "from ionic\|import ionic" LBM/Engine_V1/src --include=*.py | grep -v __pycache__
# rewrite each -> from cardiac_core.ionic[...] import ...
# ensure cardiac_core/ionic/__init__ exports the LUT names LBM needs (compare with LBM ionic/__init__)
rm -rf LBM/Engine_V1/ionic
```

#### Test Spec
- LBM V1 suite green: run each `LBM/Engine_V1/tests/test_phase*.py` (they `sys.path.insert(0,'.')`, cwd=LBM/Engine_V1). CV/diffusion behavior unchanged (ionic identical).

#### Checklist
- [ ] All LBM `ionic` imports → `cardiac_core.ionic`
- [ ] `LBM/Engine_V1/ionic/` deleted (incl. dead `ionic/ionic/`)
- [ ] cardiac_core.ionic exports every LUT name LBM uses
- [ ] LBM tests green

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1
PY=/home/norepinephrine/.conda/envs/heart-conduction/bin/python
for t in tests/test_phase*.py; do echo "== $t =="; $PY "$t" 2>&1 | tail -2; done
grep -rn "from ionic\|import ionic" src --include=*.py | grep -v __pycache__ || echo "clean: no local ionic imports"
```

#### Exit Criteria
- [ ] LBM suite green; `LBM/Engine_V1/ionic/` gone; imports point to cardiac_core.

#### Risk
LBM consumer uses an ionic API detail absent from canonical → ImportError/AttributeError — mitigation: ABC + models are the same code; add any missing LUT re-export to canonical `__init__`; suite catches the rest.

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
echo "=== ZERO duplicated ionic trees? (expect only ./cardiac_core/ionic; V5.4 keeps its own) ==="
find . -type d -name ionic ! -path '*__pycache__*' ! -path './Monodomain/Engine_V5.4/*' ! -path './Monodomain/_archive/*'
```

### Phase 3 Exit Criteria
- [ ] LBM V1 suite green; LBM `ionic/` deleted
- [ ] Only `cardiac_core/ionic/` remains (plus the intentionally-frozen V5.4 copy)

### Phase 3 Cleanup
- float64 intact; V5.3/V5.4 untouched.
- Confirm `LBM/Engine_V1/ionic/ionic/` (dead cruft) is gone.

**-> Commit point: git commit after Phase 3 ("Migrate LBM V1 to cardiac_core.ionic; delete local ionic incl. dead nested copy")**

---

## Final Cleanup (cross-phase de-sloppify)
- [ ] `find . -type d -name ionic ! -path '*__pycache__*'` → `./cardiac_core/ionic` and `./Monodomain/Engine_V5.4/cardiac_sim/ionic` (frozen baseline) ONLY. No other engine has a local ionic.
- [ ] float64: no float32 leaks introduced (ionic copied verbatim + one keyword).
- [ ] V5.3 and V5.4 untouched (`git status` shows no changes under either).
- [ ] No `sys.path.insert` added for cardiac_core anywhere (editable install is the mechanism).
- [ ] Update README Completion Criteria: check "Phase 1: ionic models in cardiac_core/ionic/".
- [ ] Update KNOWLEDGE migration table: Phase 1 DONE; note cardiac_core is now editable-installed and `__init__` is lazy.
- [ ] Note the new dependency in IDEALOG: engines now require cardiac_core installed (`pip install -e .`) — record for future cold-starts and for the Optimizer/Surrogate.
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md \
   "Research/Active/engine_consolidation/plans/2026-05-30_consolidation-phase-1-unify-ionic.md"
```

## Mutation Log
_(execution-time mutations: `**MUTATED {date}**: Step X.Y SKIPPED/SPLIT/INSERTED — reason`)_

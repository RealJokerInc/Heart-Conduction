# PLAN: cardiac_core unified ground-up package (Approach A2 — dissolve cardiac_sim)

Created: 2026-06-25
Engine(s): cardiac_core (vendor-in); Monodomain V5.5 / Bidomain V1 / LBM V1 read as COPY SOURCES only (originals untouched)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-06-25 thread entry ("consolidation = unified ground-up package (Approach A2)")

## Objective
Make `cardiac_core/` a single self-contained package: dissolve the engines' `cardiac_sim`/`src`
packages INTO it as one flat unified layout, extract the shared `ionic`+`mesh`+`stimulus` once, and
**delete the `_prepare_engine()` sys.modules hack**. After this, no file under `cardiac_core/` imports
from `Monodomain/`, `Bidomain/`, or `LBM/`. Copy-only — the three engine folders stay exactly as they
are (their own dev/tests). 121 cardiac_core tests stay green throughout.

## Success Criteria
- [ ] `cardiac_core/` is import-self-contained: a guard asserts **no `cardiac_core/**` `.py` references `Monodomain/`/`Bidomain/`/`LBM/` paths, `_prepare_engine`, `cardiac_sim`, or `from src.`**.
- [ ] Shared `cardiac_core/{ionic,mesh,stimulus}` are the single home for those modules (ionic already exists; mesh is a bidomain+mono **superset** incl. `boundary_spec`; stimulus unified).
- [ ] `cardiac_core/{monodomain,bidomain,lbm}` hold the three solvers; `api.py` constructs them via `from cardiac_core.{monodomain,bidomain,lbm} import …` (no sys.path games).
- [ ] `_prepare_engine()` + `_V55_PATH`/`_BIDOMAIN_PATH`/`_LBM_PATH` deleted; both classical engines importable simultaneously (no `cardiac_sim` collision).
- [ ] `pip install -e .` still works (new subpackages discovered); the 3 engine originals are byte-unchanged.
- [ ] All 121 existing cardiac_core tests pass (no regressions) + new structure/guard tests.

## Architecture Changes
- NEW: `cardiac_core/mesh/` — copy of the mono `tissue_builder/mesh/` (base, loader, structured, triangular) **+ bidomain's `boundary.py`**, with `structured.py` reconciled to the **superset** (supports `boundary_spec`).
- NEW: `cardiac_core/stimulus/` — copy of `tissue_builder/stimulus/` (StimulusProtocol; already aligned across engines).
- NEW: `cardiac_core/monodomain/` — V5.5 solver subtree (`simulation/`, `utils/`, `tissue/` IsotropicTissue) + facade `__init__.py`.
- NEW: `cardiac_core/bidomain/` — Bidomain V1 solver subtree (`simulation/`, `utils/`, `tissue/` BidomainConductivity) + facade `__init__.py`.
- NEW: `cardiac_core/lbm/` — LBM V1 `src/` contents (simulation, collision, streaming, boundary, lattice, solver) + facade `__init__.py`.
- MOD: `cardiac_core/api.py` — replace each `_prepare_engine(... ) ; from cardiac_sim… import …` / `from src… import …` block with `from cardiac_core.{…} import …`; delete `_prepare_engine`, `_V55_PATH`, `_BIDOMAIN_PATH`, `_LBM_PATH` (final phase). **(audit HIGH) ALSO the two stimulus-helper imports** `cardiac_core/api.py:964` (`_build_stimulus_protocol_v54`) + `:1008` (`_build_stimulus_protocol_bidomain`), each `from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol` → `from cardiac_core.stimulus.protocol import StimulusProtocol` (these run inside the factory bodies after `_prepare_engine`; they break when it's deleted). 964 repointed in Phase 2, 1008 in Phase 3.
- MOD: `cardiac_core/grid.py` — `_structured_grid()` imports `from cardiac_core.mesh.structured import StructuredGrid` (drop the `_prepare_engine` call).
- MOD: solver cross-imports (≈16 lines total) — rewrite `from …ionic` → `from cardiac_core.ionic`, `from …tissue_builder.mesh…` → `from cardiac_core.mesh…`, `from …tissue_builder.stimulus…` → `from cardiac_core.stimulus…`. **Solver-internal relative imports (`from .state`, `from ..ionic_time_stepping`, `from .solver…`) are NOT touched** — the subtree moves intact.
- MOD: `cardiac_core/tests/*` — module-top `sys.path.insert(... Engine_*)` lines removed (**audit MED: this includes `test_run.py:10-12` and `test_integration.py:11-13`, not just `test_monodomain/test_bidomain/test_lbm`**); the `…MatchesDirect` tests import the "direct" classes from `cardiac_core.{monodomain,bidomain,lbm}` instead of `cardiac_sim`/`src`. **(audit CRITICAL) `test_monodomain.py::test_engine_module_under_v55` asserts `'Engine_V5.5' in rl.__file__` — repoint to assert the live `rush_larsen` module resolves under `cardiac_core/monodomain/` (Phase 2).**
- MOD: `pyproject.toml` — confirm `setuptools` discovers the new `cardiac_core.{ionic,mesh,stimulus,monodomain,bidomain,lbm}` subpackages.

## Known Failures (from IDEALOG — do NOT retry / reintroduce)
- **Approach B (relocate-but-keep-hack)** — rejected: leaves nested `cardiac_core/engines/X/cardiac_sim/…` blobs + a packaging-exclusion + the hack. User wants flat/unified, not nested.
- **Rename-only (A1)** — rejected: renames the `cardiac_sim` silos but leaves `ionic`/`mesh`/`stimulus` **triplicated** inside each engine. Not unified.
- **"~70 absolute-import rewrites" estimate** — FALSE. The engines are 100% relative-import internally (0 absolute `cardiac_sim`/`src`, 0 `sys.path`, 0 `importlib`, 0 `__file__`, 0 name-as-string). Only the ≈16 solver→shared cross-refs change; everything else survives the move untouched.
- **Flattening `simulation/classical/` to reduce depth** — do NOT, in this plan. Collapsing levels changes relative-import dot-counts and forces internal rewrites (the thing we're avoiding). Preserve the subtree; expose a clean surface via the package `__init__.py`. A deeper flatten is a SEPARATE optional follow-up.
- **Deleting the engine originals** — NO. Copy only; `Monodomain/Engine_V5.5/`, `Bidomain/Engine_V1/`, `LBM/Engine_V1/` stay byte-identical.
- **Entangling FEM/TriangularMesh removal** — do NOT. FEM-ditch is a confirmed-but-separate cleanup. Bring `fem.py`/`triangular.py` over as-is so nothing breaks; remove later.
- **Touching V5.3/V5.4/`_archive`** — read-only. **Modifying the reaction (`/Cm`) or any solver numerics** — out of scope; this is a code-move, byte-for-byte where not a cross-ref rewrite.
- **Deleting `_prepare_engine()` before ALL THREE engines are vendored** — it's shared; deleting it early breaks the not-yet-moved engines. Delete in Phase 4 only.

---

## Phase 1: Shared `cardiac_core/mesh/` + `cardiac_core/stimulus/`

**Goal**: The two shared leaf packages exist as canonical copies (ionic already does). Nothing consumes
them yet, so this phase is purely additive — 121 tests must still pass unchanged.
**Tier**: medium
**Estimated scope**: 2 new packages (~6 + ~2 files) + reconcile one superset file; update `grid.py`.

### Phase Context
- `cardiac_core/ionic/` already exists (Phase-1 canonical copy; `celltype_is_endo` reconciled). Reuse it; do not re-copy.
- Mono mesh files: `base.py loader.py structured.py triangular.py`. Bidomain mesh files: `base.py boundary.py structured.py`. The shared `mesh/` = **union**: `base, loader, structured(superset), triangular, boundary`.
- **`structured.py` superset (audit MED — bidomain IS a strict superset, take it as the base).** Bidomain's `structured.py` adds, beyond mono's: the `boundary_spec` attribute AND the `edge_masks`, `dirichlet_mask_phi_e`, `neumann_mask_phi_e` properties, plus `field`/`Dict` imports and a `from .boundary import …` line. No mono-only method is dropped, so **start from bidomain's `structured.py`** and confirm by diff that every mono public method/property survives; verify by running BOTH engines' suites (Phases 2–3).
- **`mesh/__init__.py` is a UNION, not a copy (audit LOW).** Mono exports `Mesh, TriangularMesh, StructuredGrid`; bidomain exports `Mesh, StructuredGrid, BoundarySpec, BCType, Edge, EdgeBC` (no TriangularMesh). The shared `__init__` re-exports the union: `StructuredGrid, TriangularMesh, Mesh, BoundarySpec, BCType, Edge, EdgeBC`. Neither engine's `__init__.py` works verbatim.
- **`stimulus/` is NOT just `protocol.py` (audit HIGH).** The stimulus `__init__.py` does `from .regions import rectangular_region, circular_region, left_edge_region, point_stimulus` — so copy `regions.py` too (or drop the regions re-export from the `__init__`). Copying both `protocol.py` + `regions.py` + `__init__.py` is safest.
- These packages are leaves: `mesh`/`stimulus` do not import `ionic` or any solver. Confirm with a grep before copying; if a stray cross-ref exists, rewrite it to `cardiac_core.*`.
- float64 throughout; copy verbatim except the superset reconciliation.

### Step 1.1: Create `cardiac_core/mesh/` (superset) + `cardiac_core/stimulus/`
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/cardiac_sim/tissue_builder/mesh/` (all files) and `Bidomain/Engine_V1/cardiac_sim/tissue_builder/mesh/structured.py` + `boundary.py` — to build the union + superset `structured.py`.
- `Monodomain/Engine_V5.5/cardiac_sim/tissue_builder/stimulus/protocol.py` and the Bidomain one — confirm they are equivalent (IDEALOG: stimulus aligned, all accumulate `+=`).
- `cardiac_core/grid.py:_structured_grid` — the one current consumer pattern to repoint.

#### Why
Extracting the shared geometry/stimulus ONCE is what makes the result "unified" rather than three
renamed silos. mesh is the only non-trivial merge (the `boundary_spec` superset); doing it first, in
isolation, de-risks Phases 2–3 which both consume it.

#### Implementation Spec
**Files to create:**
- `cardiac_core/mesh/__init__.py` — re-export `StructuredGrid`, `BoundarySpec`/`BCType`/`Edge` (from boundary), `TriangularMesh` (FEM, kept for now).
- `cardiac_core/mesh/{base,loader,structured,triangular,boundary}.py` — copies; `structured.py` = bidomain superset; `__init__.py` = union (incl. `TriangularMesh` + boundary types).
- `cardiac_core/stimulus/__init__.py` + `cardiac_core/stimulus/protocol.py` + **`cardiac_core/stimulus/regions.py`** (the `__init__` re-exports `.regions`).
**Files to modify:**
- `cardiac_core/grid.py` — `_structured_grid()`: `from cardiac_core.mesh.structured import StructuredGrid` (drop `_prepare_engine`/`_V55_PATH` use here).
**Interfaces:** unchanged class APIs (`StructuredGrid.create_rectangle/from_mask/coordinates/flat_to_grid`, `StimulusProtocol.add_stimulus`).

#### Pseudocode
```
copy mono tissue_builder/mesh/*  -> cardiac_core/mesh/
add  bidomain mesh/boundary.py   -> cardiac_core/mesh/boundary.py
structured.py: start from whichever is the superset; ensure boundary_spec attr + mask + fibers all present
fix any internal cross-ref inside mesh/ (e.g. structured importing boundary) to relative (within mesh/) or cardiac_core.mesh.*
copy stimulus/protocol.py -> cardiac_core/stimulus/
grid.py: _structured_grid -> from cardiac_core.mesh.structured import StructuredGrid
```

#### Test Spec
- `tests/test_mesh_shared.py::test_structured_superset` — `cardiac_core.mesh.StructuredGrid.create_rectangle(...)` builds; `.coordinates`, `.flat_to_grid`, and a `boundary_spec` assignment all work.
- `tests/test_mesh_shared.py::test_stimulus_import` — `from cardiac_core.stimulus import StimulusProtocol`; `add_stimulus` accumulates.
- existing `tests/test_grid.py` — must pass UNCHANGED (grid now builds via shared mesh).

#### Checklist
- [ ] `cardiac_core/mesh/` (5 files + union `__init__`) created; `structured.py` = bidomain superset (boundary_spec + edge_masks/dirichlet/neumann props).
- [ ] `cardiac_core/stimulus/` (`protocol.py` + `regions.py` + `__init__`) created; `import cardiac_core.stimulus` succeeds (regions re-export resolves).
- [ ] internal mesh cross-refs resolved (no `cardiac_sim`/`tissue_builder` strings remain in `cardiac_core/mesh`).
- [ ] `grid.py` builds via `cardiac_core.mesh`; `_prepare_engine` no longer called from grid.py.
- [ ] new tests + `test_grid.py` green.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_grid.py cardiac_core/tests/test_mesh_shared.py -v
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```

#### Exit Criteria
- [ ] Shared mesh/stimulus import cleanly; `grid.py` uses them; 120 (+live gate) tests green.

#### Risk
`structured.py` superset misses a mono-only or bidomain-only method → an engine breaks in Phase 2/3.
Mitigation: diff both originals line-by-line; the superset must contain the union of public methods; the
real proof is Phases 2–3 running each engine's suite against it.

### Phase 1 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
### Phase 1 Exit Criteria
- [ ] `cardiac_core/mesh` + `cardiac_core/stimulus` exist + tested; grid.py repointed; no regressions.
### Phase 1 Cleanup
- float64; V5.3/V5.4 untouched; engine originals untouched; no leftover `cardiac_sim`/`tissue_builder` strings in the new packages.

**-> Commit point: git commit after Phase 1** (`feat(cardiac_core): shared mesh + stimulus packages`)

---

## Phase 2: Vendor the monodomain solver → `cardiac_core/monodomain/`

**Goal**: The monodomain factory constructs from `cardiac_core.monodomain` (V5.5 solver), consuming the
shared `cardiac_core.{ionic,mesh,stimulus}`. No `_prepare_engine(_V55_PATH)` for monodomain.
**Tier**: large
**Estimated scope**: copy one solver subtree + rewrite ~7 cross-refs + repoint the monodomain factory + the MatchesDirect test.

### Phase Context
- **Move the subtree INTACT** to preserve solver-internal relative imports. Copy from V5.5
  `cardiac_sim/`: `simulation/` (**including the parent `simulation/__init__.py`** — audit LOW — and the whole `classical/` tree), `utils/`, and the **WHOLE** `tissue_builder/tissue/` dir (audit MED — `__init__.py` + `isotropic.py`; do NOT cherry-pick by class name) → `cardiac_core/monodomain/{simulation,utils,tissue}/`. Do NOT bring `ionic`, `tissue_builder/mesh`, `tissue_builder/stimulus` (those are now shared).
- **Rewrite ONLY these cross-refs** (verified 2026-06-25) to absolute `cardiac_core.*`:
  - `simulation/classical/monodomain.py:45` `from ...ionic import …` → `from cardiac_core.ionic import …`
  - `simulation/classical/monodomain.py:48` `from ...tissue_builder.stimulus.protocol import StimulusProtocol` → `from cardiac_core.stimulus.protocol import StimulusProtocol`
  - `simulation/classical/solver/ionic_time_stepping/{rush_larsen,base,forward_euler}.py` `from .....ionic.base import IonicModel` → `from cardiac_core.ionic.base import IonicModel` (TYPE_CHECKING blocks)
  - `simulation/classical/discretization_scheme/fdm.py:18`, `fvm.py:23` `from ....tissue_builder.mesh.structured import StructuredGrid` → `from cardiac_core.mesh.structured import StructuredGrid`
  - `simulation/classical/discretization_scheme/fem.py:15` `from ....tissue_builder.mesh.triangular import TriangularMesh` → `from cardiac_core.mesh.triangular import TriangularMesh`
- **Do NOT touch** `from ..ionic_time_stepping…`, `from .state`, `from .discretization_scheme…`, `from .solver…` — these are solver-internal (note: `ionic_time_stepping` ≠ `ionic`).
- `cardiac_core/monodomain/__init__.py` exposes the clean surface: `from .simulation.classical import MonodomainSimulation, SimulationState; from .simulation.classical.discretization_scheme import FDMDiscretization`.
- `cardiac_core/monodomain/tissue/` holds IsotropicTissue (mono-specific; NOT shared).
- The `_prepare_engine`/`_V55_PATH` STAY in api.py this phase (bidomain/lbm still need them); only the monodomain factory + the LBM-ionic import switch off V5.5. Actually keep LBM-ionic on the old path until Phase 4; only the monodomain ENGINE import switches here.

### Step 2.1: Copy the monodomain solver subtree + rewrite cross-refs
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/cardiac_sim/simulation/classical/monodomain.py` (full) — the construction entry + its imports.
- the cross-ref list above (each file/line).
- `cardiac_core/api.py` monodomain factory (the `_prepare_engine(_V55_PATH)` block + the `MonodomainSimulation/FDMDiscretization/StructuredGrid/StimulusProtocol/ionic` imports + grid build).

#### Why
This is the first real engine dissolve. Preserving the subtree (only cross-refs change) is what keeps
the ~85 files importing correctly with zero internal edits; tests are the proof.

#### Implementation Spec
**Files to create:** `cardiac_core/monodomain/` (copied `simulation/` incl. parent `__init__`, `utils/`, whole `tissue/` + facade `__init__.py`).
**Files to modify:** the ~7 cross-ref lines (above); `cardiac_core/api.py` monodomain factory; **`cardiac_core/api.py:964` `_build_stimulus_protocol_v54` — `from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol` → `from cardiac_core.stimulus.protocol import StimulusProtocol`** (audit HIGH — it runs inside the monodomain factory body).
**Interfaces:** `from cardiac_core.monodomain import MonodomainSimulation, FDMDiscretization`; the factory builds the engine grid via `from cardiac_core.mesh.structured import StructuredGrid` and stimulus via `from cardiac_core.stimulus.protocol import StimulusProtocol`.

#### Pseudocode
```
cp -r V5.5 simulation/ utils/ tissue_builder/tissue/  ->  cardiac_core/monodomain/{simulation,utils,tissue}
rewrite the 7 cross-ref lines -> cardiac_core.{ionic,mesh,stimulus}.*
write cardiac_core/monodomain/__init__.py (facade re-exports)
api.py monodomain(): delete _prepare_engine(_V55_PATH); imports become
    from cardiac_core.monodomain import MonodomainSimulation, FDMDiscretization
    from cardiac_core.mesh.structured import StructuredGrid
    (stimulus builder already builds StimulusProtocol -> point at cardiac_core.stimulus)
```

#### Test Spec
- existing `tests/test_monodomain.py` (incl. `TestEngineIsV55` Cm-behavioral gate + `TestMonodomainMatchesDirect`) — update the `MatchesDirect` "direct" imports from `cardiac_sim…` → `cardiac_core.monodomain…`/`cardiac_core.mesh…`; remove the module-top `sys.path.insert(... Engine_V5.4)`. All must pass.
- **(audit CRITICAL) `test_monodomain.py::test_engine_module_under_v55` (≈line 144-148)** asserts `'Engine_V5.5' in rl.__file__` for the live `rush_larsen` module — this FAILS after vendoring (module now resolves under `cardiac_core/monodomain/`). Repoint the assertion to `'cardiac_core' in rl.__file__ and 'monodomain' in rl.__file__` (and `'Engine_V5' not in rl.__file__`). The behavioral `test_reaction_divides_by_cm` gate already proves Cm-correctness and is unaffected.
- **(audit MED) `tests/test_run.py:10-12` and `tests/test_integration.py:11-13`** also have module-top `sys.path.insert(... Engine_V5.4 / Bidomain/Engine_V1 / LBM/Engine_V1)` — remove these too (they become dangling/dead and trip the Phase-5 guard).
- existing `tests/test_integration.py`, `test_run*.py`, `test_construction_api.py` monodomain paths — green.

#### Checklist
- [ ] subtree copied (simulation+parent `__init__`, utils, whole tissue); 7 cross-refs rewritten; `__init__` facade added; IsotropicTissue under `monodomain/tissue/`.
- [ ] monodomain factory imports from `cardiac_core.monodomain` + `cardiac_core.mesh`/`stimulus`; **`api.py:964` stimulus helper repointed to `cardiac_core.stimulus`**; no `_prepare_engine(_V55_PATH)` in the monodomain factory.
- [ ] `test_monodomain.py` MatchesDirect repointed; **`test_engine_module_under_v55` repointed to `cardiac_core/monodomain`**; module-top engine `sys.path.insert` removed (incl. `test_run.py:10-12`, `test_integration.py:11-13`).
- [ ] grep: `cardiac_core/monodomain/` has no `cardiac_sim`/`tissue_builder` import strings.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_monodomain.py cardiac_core/tests/test_integration.py cardiac_core/tests/test_run.py cardiac_core/tests/test_run_contract.py cardiac_core/tests/test_construction_api.py -v
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```

#### Exit Criteria
- [ ] monodomain runs from `cardiac_core.monodomain`; Cm-behavioral gate still passes; all tests green.

#### Risk
A missed cross-ref (e.g. a deeper `from ....tissue_builder`) → ImportError. Mitigation: after copy, `grep -rn "tissue_builder\|from \.\+ionic\b\|cardiac_sim" cardiac_core/monodomain` and rewrite every hit; tests catch the rest. Superset mesh missing a method → mono breaks here (fix in Phase 1's structured.py).

### Phase 2 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
- [ ] monodomain fully on `cardiac_core.monodomain`; no regressions; engine original untouched. float64. No `cardiac_sim` strings in `cardiac_core/monodomain`.

**-> Commit point: git commit after Phase 2** (`feat(cardiac_core): vendor monodomain solver, drop V5.5 path`)

---

## Phase 3: Vendor the bidomain solver → `cardiac_core/bidomain/`

**Goal**: bidomain factory constructs from `cardiac_core.bidomain`; no `_prepare_engine(_BIDOMAIN_PATH)`.
**Tier**: large
**Estimated scope**: same pattern as Phase 2; ~9 cross-refs; `BidomainConductivity` stays per-engine.

### Phase Context
- Copy from Bidomain V1 `cardiac_sim/`: `simulation/` (**incl. parent `simulation/__init__.py`** — audit LOW — and the `classical/` bidomain tree: `bidomain.py`, `state.py`, `discretization/`, `solver/ionic_stepping/`, `solver/splitting/`, elliptic solvers), `utils/`, the **WHOLE** `tissue_builder/tissue/` dir (audit MED — `__init__.py` + `isotropic.py` + `conductivity.py`; bidomain's `tissue/__init__` re-exports BOTH `IsotropicTissue` and `BidomainConductivity`, so don't cherry-pick) → `cardiac_core/bidomain/{simulation,utils,tissue}/`. Skip `simulation/lbm/` (dead), `ionic`, `tissue_builder/mesh`, `tissue_builder/stimulus` (shared).
- **Rewrite these cross-refs** (verified) → `cardiac_core.*`:
  - `simulation/classical/bidomain.py:133,136` `from ...ionic import TTP06Model/ORdModel` → `from cardiac_core.ionic import …`
  - `simulation/classical/solver/ionic_stepping/{base,rush_larsen,forward_euler}.py` `from ....ionic.base import IonicModel` → `from cardiac_core.ionic.base import IonicModel`
  - `simulation/classical/discretization/base.py:15`, `fdm.py:46` `from ....tissue_builder.mesh.structured import StructuredGrid` → `from cardiac_core.mesh.structured import StructuredGrid`
  - `simulation/classical/discretization/fdm.py:47` `from ....tissue_builder.mesh.boundary import BoundarySpec, BCType, Edge` → `from cardiac_core.mesh.boundary import BoundarySpec, BCType, Edge`
  - `simulation/classical/discretization/fdm.py:48` `from ....tissue_builder.tissue.conductivity import BidomainConductivity` → **`from ...tissue.conductivity import BidomainConductivity`** (stays per-engine — adjust the relative depth to the new `cardiac_core/bidomain/tissue/`, OR `from cardiac_core.bidomain.tissue.conductivity import …`).
- **Do NOT touch** `from ..ionic_stepping…`, `from .state`, `from .discretization…`, `from .solver…` (internal).
- `cardiac_core/bidomain/__init__.py` re-exports `BidomainSimulation`, `BidomainFDMDiscretization`, `BoundarySpec` (re-export from `cardiac_core.mesh`), `BidomainConductivity`.
- The api.py bidomain factory currently does its own σ→D math + `BidomainConductivity(...)`; only the import source changes (`from cardiac_core.bidomain import BidomainSimulation, BidomainFDMDiscretization, BidomainConductivity` + `from cardiac_core.mesh import StructuredGrid, BoundarySpec`).

### Step 3.1: Copy bidomain subtree + rewrite cross-refs + repoint factory
**Model**: opus

#### Read First
- `Bidomain/Engine_V1/cardiac_sim/simulation/classical/bidomain.py` + `discretization/fdm.py` (the cross-refs).
- `cardiac_core/api.py` bidomain factory (the `_prepare_engine(_BIDOMAIN_PATH)` block + `BidomainSimulation/BidomainFDMDiscretization/StructuredGrid/BoundarySpec/BidomainConductivity` imports + the σ→D block).
- `cardiac_core/tests/test_bidomain.py` (MatchesDirect + module-top sys.path inserts).

#### Why
Bidomain is the second classical engine sharing `cardiac_sim`; once it's on `cardiac_core.bidomain`, the
two no longer collide, unlocking the hack deletion in Phase 4.

#### Implementation Spec
**Files to create:** `cardiac_core/bidomain/` (`simulation/`, `utils/`, `tissue/` + `__init__.py`).
**Files to modify:** the ~9 cross-refs; `api.py` bidomain factory; **`api.py:1008` `_build_stimulus_protocol_bidomain` — `from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol` → `from cardiac_core.stimulus.protocol import StimulusProtocol`** (audit HIGH — runs inside the bidomain factory body); `test_bidomain.py`.
**Interfaces:** `from cardiac_core.bidomain import BidomainSimulation, BidomainFDMDiscretization, BidomainConductivity`; `from cardiac_core.mesh import StructuredGrid, BoundarySpec`.

#### Pseudocode
```
cp bidomain simulation/(classical only) utils/ tissue_builder/tissue/ -> cardiac_core/bidomain/{simulation,utils,tissue}
rewrite the 9 cross-refs (ionic+mesh -> cardiac_core.*; conductivity -> bidomain-local)
write cardiac_core/bidomain/__init__.py
api.py bidomain(): delete _prepare_engine(_BIDOMAIN_PATH); import from cardiac_core.bidomain + cardiac_core.mesh
```

#### Test Spec
- `tests/test_bidomain.py` — repoint MatchesDirect "direct" imports to `cardiac_core.bidomain`/`cardiac_core.mesh`; drop module-top engine `sys.path.insert`. All pass.
- bidomain paths in `test_run*.py`, `test_integration.py`, `test_construction_api.py` — green (incl. the declarative `test_bidomain_from_grid`).

#### Checklist
- [ ] subtree copied (no `simulation/lbm`); 9 cross-refs rewritten; BidomainConductivity under `bidomain/tissue/`; `__init__` facade.
- [ ] bidomain factory imports from `cardiac_core.bidomain` + `cardiac_core.mesh`; **`api.py:1008` stimulus helper repointed to `cardiac_core.stimulus`**; no `_prepare_engine(_BIDOMAIN_PATH)`.
- [ ] `test_bidomain.py` repointed; grep clean (`cardiac_core/bidomain` has no `cardiac_sim`/`tissue_builder` strings).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_bidomain.py cardiac_core/tests/test_integration.py cardiac_core/tests/test_construction_api.py -v
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```

#### Exit Criteria
- [ ] bidomain runs from `cardiac_core.bidomain`; monodomain + bidomain coexist without `_prepare_engine` between them; all tests green.

#### Risk
mesh superset must serve bidomain's `boundary_spec` path (the elliptic BC). Mitigation: `test_bidomain` boundary/bath tests exercise it; if a `BoundarySpec` method is mono-absent, it's already in the shared `boundary.py` (copied from bidomain). Relative-depth slip on the `tissue.conductivity` cross-ref → ImportError; mitigation: prefer the absolute `from cardiac_core.bidomain.tissue.conductivity import …`.

### Phase 3 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
- [ ] bidomain on `cardiac_core.bidomain`; no regressions; original untouched. float64. grep clean.

**-> Commit point: git commit after Phase 3** (`feat(cardiac_core): vendor bidomain solver, drop bidomain path`)

---

## Phase 4: Vendor LBM → `cardiac_core/lbm/` + DELETE the hack

**Goal**: LBM constructs from `cardiac_core.lbm`; `_prepare_engine()` and all three path constants are
deleted. cardiac_core is now import-self-contained.
**Tier**: large
**Estimated scope**: copy `src/`→`lbm/` (easiest — no internal ionic refs); repoint factory; ionic now from `cardiac_core.ionic`; delete hack.

### Phase Context
- Copy LBM `src/` contents → `cardiac_core/lbm/` directly (simulation, collision/, streaming/, boundary/, lattice/, solver/). Internal imports are relative (`from .solver.rush_larsen import ionic_step`) → survive untouched. LBM has **no internal ionic import** (it receives an ionic_model object), so zero cross-ref rewrites inside lbm/.
- The LBM factory's ionic instance currently comes from `_prepare_engine(_V55_PATH); from cardiac_sim.ionic import TTP06Model, …`. Switch to `from cardiac_core.ionic import TTP06Model, ORdModel, PHAS13Model, MHAS13Model`. Switch `from src.simulation import LBMSimulation` → `from cardiac_core.lbm.simulation import LBMSimulation`.
- After all three engines + ionic are off the hack: **delete `_prepare_engine`, `_V55_PATH`, `_BIDOMAIN_PATH`, `_LBM_PATH`** from api.py, plus any remaining `import sys`/`sys.path` use that existed only for the hack (verify before removing `import sys`).
- `cardiac_core/lbm/__init__.py` re-exports `LBMSimulation`.

### Step 4.1: Copy LBM + repoint factory + delete `_prepare_engine`
**Model**: opus

#### Read First
- `LBM/Engine_V1/src/simulation.py` head (its relative imports).
- `cardiac_core/api.py` lbm factory (the `_LBM_PATH` sys.path insert + `from src.simulation import LBMSimulation` + `_prepare_engine(_V55_PATH); from cardiac_sim.ionic import …`).
- `cardiac_core/api.py:25-37` (`_prepare_engine` + the 3 path constants — to delete).
- `cardiac_core/tests/test_lbm.py` (MatchesDirect uses `sim_direct.run` returning a tuple; module-top inserts).

#### Why
LBM is the last engine on the hack; removing it lets the hack and all cross-folder paths die — the
success criterion of the whole plan.

#### Implementation Spec
**Files to create:** `cardiac_core/lbm/` (copied `src/` contents + `__init__.py`).
**Files to modify:** `api.py` lbm factory + DELETE `_prepare_engine`/`_V55_PATH`/`_BIDOMAIN_PATH`/`_LBM_PATH`; `test_lbm.py`.
**Interfaces:** `from cardiac_core.lbm.simulation import LBMSimulation`; ionic from `cardiac_core.ionic`.

#### Pseudocode
```
cp -r LBM/Engine_V1/src/*  ->  cardiac_core/lbm/
write cardiac_core/lbm/__init__.py
api.py lbm(): from cardiac_core.lbm.simulation import LBMSimulation; from cardiac_core.ionic import (models)
DELETE _prepare_engine, _V54/_V55/_BIDOMAIN/_LBM_PATH; drop now-dead `import sys` if unused
```

#### Test Spec
- `tests/test_lbm.py` — repoint MatchesDirect "direct" import to `cardiac_core.lbm`; drop module-top engine inserts. Pass.
- `tests/test_run_contract.py::test_record_ionic_states_lbm_not_implemented` + LBM paths in `test_run/integration/construction_api` — green.

#### Checklist
- [ ] `cardiac_core/lbm/` copied; `__init__` facade; LBM factory imports from `cardiac_core.lbm` + `cardiac_core.ionic`.
- [ ] `_prepare_engine` + 3 path constants DELETED; `import sys` removed if now unused.
- [ ] `test_lbm.py` repointed.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q   # FULL suite incl. live gate
# Guard matches only REAL imports / hack calls — NOT docstring path-mentions (audit HIGH).
# conductivity.py:11 docstring references "Monodomain/Engine_V5.5/..." legitimately; do NOT flag it.
grep -rnE "^[[:space:]]*(from|import)[[:space:]]+(cardiac_sim|src)[. ]|_prepare_engine\(" \
  cardiac_core --include=*.py | grep -v "tests/_live_cv_gate_driver.py" \
  && echo "FOUND cross-folder import/hack — NOT self-contained" || echo "CLEAN — no cross-folder imports"
```

#### Exit Criteria
- [ ] All 121 tests pass with the hack GONE; the grep prints CLEAN.

#### Risk
A residual `import sys` / `sys.path` use elsewhere in api.py → NameError after deletion. Mitigation: grep `sys\.` in api.py before removing `import sys`. LBM `src` is a generic name — ensure nothing else imports top-level `src` after the move.

### Phase 4 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
- [ ] hack deleted; 121 green; grep CLEAN; 3 engine originals byte-unchanged (`git status` shows only `cardiac_core/` + docs).

**-> Commit point: git commit after Phase 4** (`feat(cardiac_core): vendor LBM, delete _prepare_engine hack — self-contained`)

---

## Phase 5: Seal — guard test, packaging, drift note

**Goal**: Lock self-containment with an automated guard; confirm the editable install; document drift.
**Tier**: medium
**Estimated scope**: 1 guard test + pyproject check + a SYNC note.

### Phase Context
- The guard is the durable success criterion — it should FAIL if any future edit re-introduces a cross-folder import or the hack.
- pyproject: the new subpackages must be discoverable. With proper `__init__.py` chains they are real `cardiac_core.*` subpackages; confirm `pip install -e .` re-resolves and `import cardiac_core.monodomain` works from an arbitrary cwd.

### Step 5.1: Guard test + packaging confirm + SYNC note
**Model**: opus

#### Read First
- `pyproject.toml` (the `cardiac_core*` package-find config).
- `cardiac_core/__init__.py` (lazy export map — add no engine internals).

#### Why
Without the guard, the `_prepare_engine` hack or a stray `Monodomain/` import can silently creep back.

#### Implementation Spec
**Files to create:** `cardiac_core/tests/test_self_contained.py`; `cardiac_core/engines_SOURCE.md` (a short re-vendor/drift note: which engine + commit each package was copied from, and the `cp` recipe to re-sync).
**Files to modify:** `pyproject.toml` if discovery misses the new subpackages.

#### Pseudocode
```
test_no_cross_folder_refs: walk cardiac_core/**/*.py
    # match ONLY real cross-folder IMPORTS or hack CALLS — never docstrings / path-string literals.
    bad = re.compile(r'^\s*(from|import)\s+(cardiac_sim|src)[.\s]') OR substring '_prepare_engine('
    for each .py line: assert not bad  (skip tests/_live_cv_gate_driver.py)
test_subpackages_importable: import cardiac_core.{ionic,mesh,stimulus,monodomain,bidomain,lbm}
```
> **Guard scope (audit HIGH + MED):** match the IMPORT-STATEMENT form, NOT bare `Engine_V5`/`Engine_V1`
> substrings. Otherwise it false-positives on legitimate path-MENTIONS: `conductivity.py:11` docstring
> ("…`Monodomain/Engine_V5.5/_probe…`"), the `_live_cv_gate_driver.py` path-string `_V55 = …Engine_V5.5`,
> and any remaining test path literals. Those are not imports → the import-line regex ignores them.
> `_prepare_engine(` is matched as a CALL (the def is deleted in Phase 4, so any call is a real bug; a
> stray comment wouldn't survive cleanup anyway). Still exclude `_live_cv_gate_driver.py` belt-and-suspenders.

#### Test Spec
- `tests/test_self_contained.py::test_no_cross_folder_refs` — passes (CLEAN).
- `tests/test_self_contained.py::test_subpackages_importable` — all 6 import.

#### Checklist
- [ ] guard test added + green; `_live_cv_gate_driver.py` excluded/handled.
- [ ] `pip install -e .` re-run; `python -c "import cardiac_core.monodomain"` from `/tmp` works.
- [ ] `engines_SOURCE.md` records source engine + re-vendor recipe.

#### Verify
```bash
conda run -n heart-conduction pip install -e . -q && cd /tmp && conda run -n heart-conduction python -c "import cardiac_core.monodomain, cardiac_core.bidomain, cardiac_core.lbm; print('self-contained OK')"; cd -
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```

#### Exit Criteria
- [ ] guard green; editable install resolves new subpackages; 121 tests pass.

#### Risk
setuptools flat-layout/duplicate-top-level complaints from the vendored trees. Mitigation: the trees are nested under `cardiac_core.*` (not top-level), so standard `find` includes them; if discovery misbehaves, set explicit `packages.find` include `cardiac_core*`.

### Phase 5 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
- [ ] self-contained guard green; full suite green; engine originals untouched.

**-> Commit point: git commit after Phase 5** (`test(cardiac_core): self-containment guard + packaging + drift note`)

---

## Final Cleanup (cross-phase)
- [ ] float64 consistency across all vendored/edited modules — no float32 leaks.
- [ ] V5.3/V5.4/`_archive` untouched; the 3 engine originals (`Engine_V5.5`, `Engine_V1` ×2) byte-identical (`git status` shows only `cardiac_core/` + research docs).
- [ ] No real cross-folder **imports** (`^(from|import) (cardiac_sim|src)[. ]`) or `_prepare_engine(` calls anywhere under `cardiac_core/` (except the deliberately-excluded `_live_cv_gate_driver.py`). Docstring/path-string MENTIONS of `Engine_V5.5` (e.g. `conductivity.py:11`, the driver's `_V55` path) are OK and must NOT be flagged — the guard matches import statements, not substrings.
- [ ] KNOWLEDGE.md migration table + IDEALOG session log updated with the unified layout that shipped; README status banner refreshed.
- [ ] Decide: keep `_live_cv_gate_driver.py` driving V5.5, or repoint it at `cardiac_core.monodomain` (then it too is self-contained). Note the choice.
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_cardiac-core-unified-ground-up-package.md"
```
- [ ] Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Out of Scope (follow-up plans)
- Deeper flatten (collapse `simulation/classical/` levels) — optional, requires internal relative-import rewrites.
- FEM / `TriangularMesh` removal (structured-only) — confirmed-but-separate.
- Form-A→B monodomain convergence + delete `ConductivityConfig.for_monodomain()`.
- Collapsing the per-engine `tissue/` (IsotropicTissue vs BidomainConductivity) or the duplicate ionic-inside-LBM stray.
- Goal-2 `SimulationSpec` / LLM wrapper.

## Mutation Log

Revision pass 2026-06-25 — adversarial audit (`/audit`, 14 findings: 1 critical, 4 high, 6 medium, 3 low). No steps completed; all changes are spec patches (missed edit-sites + guard scope). The auditor independently verified the solver cross-ref enumeration is EXACT/complete, the relative-import assumption holds, dependency ordering is sound, and all 9 sections are present.

**MUTATED 2026-06-25**: Phase 2 Test Spec MODIFIED — audit CRITICAL. `test_monodomain.py::test_engine_module_under_v55` asserts `'Engine_V5.5' in rl.__file__`, which FAILS after vendoring (module resolves under `cardiac_core/monodomain/`). Repoint the assertion to `cardiac_core`+`monodomain` (and `Engine_V5 not in`). Behavioral Cm gate unaffected.
**MUTATED 2026-06-25**: Phase 2 + Phase 3 Implementation Spec/Checklist MODIFIED — audit HIGH. Added the two api.py stimulus-helper edit sites as concrete: `api.py:964` `_build_stimulus_protocol_v54` (Phase 2) and `api.py:1008` `_build_stimulus_protocol_bidomain` (Phase 3), each `from cardiac_sim.tissue_builder.stimulus.protocol …` → `cardiac_core.stimulus.protocol`. They run inside the factory bodies and break when `_prepare_engine` is deleted. Mirrored in Architecture Changes.
**MUTATED 2026-06-25**: Phase 1 Context/Spec/Checklist MODIFIED — audit HIGH. `cardiac_core/stimulus/` must copy `regions.py` too (the stimulus `__init__` re-exports `.regions`), else `import cardiac_core.stimulus` raises ImportError.
**MUTATED 2026-06-25**: Phase 4 verify + Phase 5 guard + Final Cleanup MODIFIED — audit HIGH+MED. Re-scoped the self-containment guard to match only real IMPORT statements (`^(from|import) (cardiac_sim|src)[. ]`) and `_prepare_engine(` calls — NOT bare `Engine_V5`/`Engine_V1` substrings — so it no longer false-positives on the `conductivity.py:11` docstring path-mention, the `_live_cv_gate_driver.py` `_V55` path string, or test literals.
**MUTATED 2026-06-25**: Phase 1 Context MODIFIED — audit MED. `structured.py` superset carries (beyond `boundary_spec`) the `edge_masks`/`dirichlet_mask_phi_e`/`neumann_mask_phi_e` properties + `field`/`Dict` imports + `from .boundary import`; bidomain's IS the strict superset → take it as the base.
**MUTATED 2026-06-25**: Phase 2 + Phase 3 test-edit list MODIFIED — audit MED. Also remove module-top `sys.path.insert(... Engine_*)` from `test_run.py:10-12` and `test_integration.py:11-13` (not just the three per-engine test files).
**MUTATED 2026-06-25**: Phase 2 + Phase 3 copy spec MODIFIED — audit MED+LOW. Copy the WHOLE `tissue/` dir per engine (`__init__` + `isotropic.py` [+ bidomain `conductivity.py`]) — don't cherry-pick by class name; and include the parent `simulation/__init__.py` when copying `simulation/classical/`.
**MUTATED 2026-06-25**: Phase 1 Context/Spec MODIFIED — audit LOW. `mesh/__init__.py` is the UNION of mono+bidomain exports (`TriangularMesh` + boundary types `BoundarySpec/BCType/Edge/EdgeBC`) — neither engine's `__init__` works verbatim.
**MUTATED 2026-06-25**: (audit LOW, no change) `lbm/__init__.py` facade re-exporting `LBMSimulation` confirmed necessary — the copied `src/__init__.py` is docstring-only.

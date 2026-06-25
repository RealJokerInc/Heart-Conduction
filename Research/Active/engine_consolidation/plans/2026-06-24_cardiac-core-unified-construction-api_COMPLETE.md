# PLAN: cardiac_core unified construction API (Goal 1)

Created: 2026-06-24
Engine(s): cardiac_core (construction surface); uses Monodomain V5.5 read-only for the conductivity gate
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-06-24 thread (API_DESIGN + API_REFERENCE drafted; firewall gate closed; Form-B target; FEM ditched)

## Objective
Turn the `[design]` symbols in `API_REFERENCE.md` into real `cardiac_core` code: the unified,
engine-agnostic **construction API** (North-Star Goal 1). Build foundation-first — `ConductivityConfig`
(the χ/Cm + Formulation-A/B firewall, already gate-verified) and `Grid` (structured-only) — then
canonicalize `Vm`, wire the factories to construct from `(Grid, ionic_model, ConductivityConfig,
stimulus)`, and deliver the `run()` eager/batch contract returning `SimulationResult`.

This realizes the construction + run surface. The **declarative `SimulationSpec`/`create_simulation`
(Goal-2 LLM-intake bridge) is OUT OF SCOPE** here — it sits on top and gets its own plan.

> ⚠️ **Phase numbering note (audit MEDIUM).** The phases below are **API-track** (this plan only). They
> are NOT the consolidation-track phases in `KNOWLEDGE.md` ("Phase 4" there = the engine rewire +
> Form-A→B conversion). Don't conflate: this plan's "Phase 4" is API factory wiring and deliberately
> keeps the engines as-is (except the Phase-0 V5.4→V5.5 monodomain repoint below).

## Success Criteria  — ✅ ALL MET (2026-06-24; 121 cardiac_core tests pass, was 80)
- [x] **cardiac_core's `monodomain()` path runs Cm-correct V5.5** (not V5.4) — `_V55_PATH` repoint; behavioral `test_reaction_divides_by_cm` gate.
- [x] `ConductivityConfig` built (`.isotropic/.bidomain/.anisotropic`, `.sigma_eff`, `.D_eff`, `for_monodomain/for_bidomain/for_lbm`), with the firewall gate as a test (arithmetic to 1e-12 + live monodomain CV 54.35 cm/s, subprocess-isolated).
- [x] `Grid` built (structured-only; `Nx,Ny,dx,dy,mask`; `Lx/Ly/coordinates/n_dof`).
- [x] `Vm` is the canonical voltage field on `SimulationSnapshot` + `SimulationResult`, with a read-only `.V` alias (existing tests stayed green).
- [x] Factories `monodomain/bidomain/lbm` construct from `(Grid, ionic_model, ConductivityConfig, stimulus)`, preserving the legacy `mesh=` path. + `reset/with_/stimulate/introspection` + `Simulation` Protocol.
- [x] `sim.run(t_end, save_every, *, batch=None, record=("Vm",))` returns one `SimulationResult` (eager) or an iterator (batch); `SimulationResult` carries `Vm (T,Nx,Ny)` + `.cv()/.apd()/.lat()/.restitution()` hooks; `snapshots()` is the back-compat generator.
- [x] All existing cardiac_core tests pass (no regressions); new tests for each phase pass (121 total).

> **STATUS: IMPLEMENTATION COMPLETE (2026-06-24).** All 6 API-track phases shipped + tested. Remaining: doc polish (`API_REFERENCE.md` `[design]`→`[now]`), archive this plan, and commit (uncommitted on `main`). See KNOWLEDGE "Goal-1 Construction API — SHIPPED".

## Architecture Changes
- MOD: `cardiac_core/api.py:20,841,1085` — add `_V55_PATH`; repoint the `monodomain()` factory's `_prepare_engine(_V54_PATH)`→`_V55_PATH` so the delivered engine is Cm-correct (Phase 0). `run_monodomain` calls the factory (no own engine import) → no change needed there. V5.4 stays on disk, untouched.
- NEW: `cardiac_core/conductivity.py` — `ConductivityConfig` (firewall).
- NEW: `cardiac_core/grid.py` — `Grid` (structured-only geometry).
- NEW: `cardiac_core/simulation.py` — `Simulation` typing.Protocol (the engine-agnostic interface).
- MOD: `cardiac_core/api.py` — `SimulationSnapshot.V`→`Vm` (+`.V` alias); factories accept `Grid`+`ConductivityConfig`+`stimulus`; implement `reset/stimulate/with_/introspection`; new `run()` contract.
- MOD: `cardiac_core/run.py` — `SimulationResult.V`→`Vm` (+`.V` alias) + `phi_e/ionic_states/dx/dy` + `.cv()/.apd()/.lat()/.restitution()` hooks.
- MOD: `cardiac_core/__init__.py` — export `Grid`, `ConductivityConfig`, `Simulation`.
- MOD: `cardiac_core/io.py` — accept `Vm=`/`V=` (back-compat) in `save_result`.
- NEW tests: `tests/test_conductivity.py`, `tests/test_grid.py`, `tests/test_construction_api.py`, `tests/test_run_contract.py`.

## Known Failures (from IDEALOG — do NOT retry / do NOT reintroduce)
- **`for_monodomain()` feeding `chi=1, Cm=1`** — WRONG for Cm≠1 (pins the reaction `/Cm` to `/1`). The real `Cm` MUST reach the engine; feed monodomain Cm-**un**scaled `D = sigma_eff/chi` with `chi=1` and the **real** `Cm`. (Same Cm-trap family as the false time-dilation invariant.)
- **Cm time-dilation invariant** (`V(x,t;Cm=k)==V(x,t/k)`) — FALSE; never assert APD scales with Cm.
- **`sigma` = a pre-divided `D`** — units trap. `ConductivityConfig` `sigma_*` are raw conductivities (mS/cm); `D_eff = sigma_eff/(chi·Cm)` is derived. `0.00097` is the *D_eff*, not σ.
- **Merging solver internals into cardiac_core** — only construction/shared code is unified; solvers stay in engines (this plan wraps, does not move solver code).
- **Breaking the `cardiac_sim` namespace / removing `_prepare_engine()`** — out of scope; leave the hack.
- **Modifying V5.3 / V5.4 / `_archive`** — read-only. (V5.5 is read-only here too — used only as the gate oracle.)

---

## Phase 0: Repoint cardiac_core's monodomain path to V5.5 (CRITICAL — audit)

**Goal**: The cardiac_core `monodomain()` factory constructs the **Cm-correct V5.5** engine, not V5.4.
Without this, every later phase wires the firewall into V5.4's `/Cm`-less reaction — the gate passes in
the V5.5 harness but the *delivered* factory is wrong at Cm≠1.
**Tier**: medium
**Estimated scope**: a path constant + 1–2 `_prepare_engine` call-site swaps + re-green the suite.

### Phase Context
- **The bug** (audit CRITICAL): `cardiac_core/api.py:20` `_V54_PATH = …/Engine_V5.4`; the monodomain
  factory (`api.py:841`) and LBM ionic import (`api.py:1085`) call `_prepare_engine(_V54_PATH)`. V5.4's
  Rush-Larsen reaction is `state.V = V + dt*(-(Iion+Istim))` — **no `/Cm`** (`Engine_V5.4/.../rush_larsen.py:83`).
  V5.5 is `… / state.Cm` (`Engine_V5.5/.../rush_larsen.py:84`). KNOWLEDGE.md:231 mandates "build cardiac_core
  against **V5.5**."
- **Scope**: repoint the **monodomain ENGINE** (the stepper with the `/Cm` fix) to V5.5. The LBM ionic
  import (api.py:1085) imports only the ionic MODEL (Cm-independent; V5.4/V5.5 byte-identical) — repoint
  it too for consistency, but it is not Cm-correctness-critical. **The LBM *engine* is a separate package
  entirely (`LBM/Engine_V1`, already Form-B/Cm-correct) — this Phase does NOT and need NOT touch it; the
  api.py:1085 swap is the ionic-model import only.** Bidomain is already Form-B (Cm-correct) — leave it.
- V5.5 is "Cm=1 bit-identical to V5.4" (golden, max|dV|=0) → the 80 tests (all Cm=1) MUST still pass.
- V5.4 stays on disk untouched (frozen baseline). This is a path swap in cardiac_core, not an engine edit.

### Step 0.1: Add `_V55_PATH` and repoint the monodomain factory
**Model**: opus

#### Read First
- `cardiac_core/api.py:18-37` — `_V54_PATH`/`_BIDOMAIN_PATH` constants + `_prepare_engine()`.
- `cardiac_core/api.py:838-898` — the `monodomain()` body (the `_prepare_engine(_V54_PATH)` + imports of `MonodomainSimulation`/`FDMDiscretization`/`StructuredGrid`/`StimulusProtocol`/ionic).
- `cardiac_core/api.py:1082-1090` — the LBM factory's ionic import (`_prepare_engine(_V54_PATH)`).
- `cardiac_core/run.py:60-103` — `run_monodomain` (does it import the engine itself, or call the factory?).
- `Monodomain/Engine_V5.5/cardiac_sim/` — confirm the import paths match V5.4's (V5.5 removed the dead internal `lbm/` package; the classical monodomain layout is otherwise identical).

#### Why
This is the single change that makes the firewall's whole reason-for-being (Cm-correct reaction) real in
the shipped API. Every Phase 1 gate value (CV 54.35/28.09) was measured against V5.5 — the factory must
run the same engine for the gate to mean anything.

#### Implementation Spec
**Files to modify:** `cardiac_core/api.py` — add `_V55_PATH = str(_project_root / "Monodomain" / "Engine_V5.5")`; change the monodomain factory's `_prepare_engine(_V54_PATH)`→`_prepare_engine(_V55_PATH)` and the imports' source; change the LBM ionic-import `_prepare_engine(_V54_PATH)`→`_V55_PATH` (consistency). `run.py` if it imports the engine directly.
**Interfaces / Signatures:** unchanged (path swap only).

#### Pseudocode
```
_V55_PATH = str(_project_root / "Monodomain" / "Engine_V5.5")
# monodomain(): _prepare_engine(_V55_PATH); from cardiac_sim... import MonodomainSimulation, FDMDiscretization, ...
# lbm() ionic import: _prepare_engine(_V55_PATH)
```
If V5.5's module layout diverges from V5.4 for any imported symbol, adjust the import; do NOT touch V5.5 source.

#### Test Spec
- existing `tests/test_monodomain.py` (6) + `tests/test_integration.py` (5) — must pass UNCHANGED (V5.5 Cm=1 ≡ V5.4).
- Add `tests/test_monodomain.py::test_engine_is_v55` — assert the reaction is **Cm-SENSITIVE** (the behavioral V5.5 signature), NOT just a path/`state.Cm` check (re-audit LOW, corrected pass 3: a module-path/`__file__` substring is brittle against stale `sys.modules`, and an attribute check is the wrong target — the V5.5 fix is a *behavior* (reaction `/Cm`); in fact V5.4's `classical/state.py` has **no** `Cm` field at all, so only the behavioral check actually proves V5.5 is the live engine). Concretely: from an identical mid-AP state, one reaction step at `Cm=1` vs `Cm=2` gives `dV(Cm=2) ≈ dV(Cm=1)/2` (V5.4 would give equal dV). Optionally also assert the `rush_larsen` module's `__file__` resolves under `Engine_V5.5`.

#### Checklist
- [ ] `_V55_PATH` added; monodomain factory + run path repointed; LBM ionic import repointed.
- [ ] All monodomain/integration imports resolve against V5.5.
- [ ] New `test_engine_is_v55` asserts the live engine is V5.5.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_monodomain.py cardiac_core/tests/test_integration.py -v
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```

#### Exit Criteria
- [ ] 80 existing tests pass against V5.5; `test_engine_is_v55` passes.

#### Risk
V5.5's package/module layout differs from V5.4 (e.g. removed `lbm/`, different `__init__` exports) →
import errors. Mitigation: V5.5 is a fork — the classical paths match; fix imports at the cardiac_core
boundary only (never edit V5.5). If a Cm=1 test output drifts (it must not), STOP — that contradicts the
documented bit-identical golden and signals a deeper issue.

### Phase 0 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
### Phase 0 Exit Criteria
- [ ] cardiac_core `monodomain()` runs V5.5; 80 existing tests green (Cm=1 unchanged).
### Phase 0 Cleanup
- V5.4/V5.5 source untouched (path swap only). No float32. **Remove the now-dead `_V54_PATH` (api.py:20) with a one-line comment** (`# V5.4 frozen baseline — cardiac_core runs V5.5`), rather than leaving a dangling unused constant (re-audit LOW).

**-> Commit point: git commit after Phase 0** (`fix(cardiac_core): repoint monodomain path V5.4->V5.5 (Cm-correct)`)

---

## Phase 1: `ConductivityConfig` (the firewall)

**Goal**: A standalone, tested `cardiac_core/conductivity.py` that converts physics (`sigma_*`, `chi`,
`Cm`, `fiber_angle`) into per-engine diffusion inputs, with the gate locked as a test. Purely additive —
no existing code touched.
**Tier**: large
**Estimated scope**: one new module (~120 lines) + one new test file; verify against the existing probe.

### Phase Context
- The design + math is in `API_DESIGN.md` §4 and `API_REFERENCE.md` "Conductivity". Follow them exactly.
- The gate oracle already exists: `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` (passing:
  arithmetic to 1.1e-19; CV 54.35 @Cm=1, 28.09 @Cm=2). The Phase-1 test reuses its logic.
- `frozen=True` dataclass; all floats; isotropic case = scalar; anisotropic = `(Dxx,Dyy,Dxy)` tuple.
- The harmonic i/e collapse: `sigma_eff = sigma_i*sigma_e/(sigma_i+sigma_e)`; `D_eff = sigma_eff/(chi*Cm)`.
- **CRITICAL Cm rule** (Known Failures): `for_monodomain` returns `D = D_eff*Cm (= sigma_eff/chi)`,
  `chi=1.0`, `Cm=Cm` — Cm-UNscaled D, real Cm. `for_bidomain/for_lbm` return fully-scaled D + real Cm.
- **Layer note (audit HIGH — for_bidomain shape):** the `for_*` emitters return engine-level
  **diffusivities** (the post-rewire *direct-feed* contract). In THIS plan's wrapper era they flow
  through `CardiacMeshData`, whose schema stores `D_xx/D_yy/D_xy` (effective **D**, monodomain) but
  `sigma_i/sigma_e` as **conductivity tuples** `(xx,yy,xy)` (bidomain) — NOT D. So Phase 4 maps:
  monodomain via `for_monodomain()` → `D_xx=D_yy=D, D_xy=0, chi=1, Cm`; bidomain via the config's **raw σ**
  → `sigma_i=(σ_i,σ_i,0.0)`, `sigma_e=(σ_e,σ_e,0.0)`, `chi=cfg.chi, Cm=cfg.Cm` (the existing bidomain
  factory does σ→D internally). `for_bidomain()`'s D output is reserved for the post-rewire direct-feed and
  is NOT used through `CardiacMeshData`. Phase 1 still implements `for_bidomain` (D output) for that future path.
- Do NOT wire this into the factories yet (Phase 4). Phase 1 ends with a standalone, exported, tested class.

### Step 1.1: Implement `ConductivityConfig`
**Model**: opus

#### Read First
- `Research/Active/engine_consolidation/API_DESIGN.md` §4 — the emitter table + the "why Cm must reach the engine" note.
- `Research/Active/engine_consolidation/API_REFERENCE.md` "Conductivity" — the public surface (constructors, properties, emitters, units note).
- `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` — the reference `ConductivityConfig` stub (defines `sigma_eff`, `D_eff`, `for_monodomain`, `for_bidomain`). Mirror this arithmetic exactly.
- `LBM/Engine_V1/src/diffusion.py:9` — `sigma_to_D(sigma_l, sigma_t, fiber_angle, chi, Cm)` for the anisotropic tensor path.
- `cardiac_core/file_format.py:14-76` — `CardiacMeshData` conductivity fields (`D_xx/D_yy/D_xy`, `sigma_i/sigma_e`, `chi`, `Cm`); Phase 4 maps config → these.

#### Why
This is THE consolidation principle made concrete (decision #13: χ/Cm only here). The Cm-asymmetry is
subtle and was wrong on the first design pass — copying the *verified probe arithmetic* avoids
re-deriving it. The anisotropic path must keep tensor subscripts (P1).

#### Implementation Spec
**Files to create:** `cardiac_core/conductivity.py` — the `ConductivityConfig` dataclass.
**Files to modify:** `cardiac_core/__init__.py:1-47` — add `ConductivityConfig` to the lazy export map → `conductivity`.
**Interfaces / Signatures:**
```python
@dataclass(frozen=True)
class ConductivityConfig:
    sigma_i: float | tuple | None = None
    sigma_e: float | tuple | None = None
    sigma_eff: float | tuple | None = None
    chi: float = 1400.0
    Cm: float = 1.0
    fiber_angle: float = 0.0
    @classmethod
    def isotropic(cls, sigma, chi=1400.0, Cm=1.0) -> "ConductivityConfig": ...
    @classmethod
    def bidomain(cls, sigma_i, sigma_e, chi=1400.0, Cm=1.0) -> "ConductivityConfig": ...
    @classmethod
    def anisotropic(cls, sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0) -> "ConductivityConfig": ...
    @property
    def sigma_eff_value(self) -> float: ...     # resolves: explicit sigma_eff, or i/e harmonic collapse
    @property
    def D_eff(self): ...                         # sigma_eff_value/(chi*Cm)  (scalar or tensor)
    def for_monodomain(self) -> dict: ...         # {'D': D_eff*Cm, 'chi': 1.0, 'Cm': Cm}
    def for_bidomain(self) -> dict: ...           # {'D_i': sigma_i/(chi*Cm), 'D_e': sigma_e/(chi*Cm), 'Cm': Cm}
    def for_lbm(self) -> dict: ...                # {'D': D_eff, 'Cm': Cm}
```

#### Pseudocode
```
isotropic(sigma):    return cls(sigma_eff=sigma, chi=chi, Cm=Cm)
bidomain(si, se):    return cls(sigma_i=si, sigma_e=se, chi=chi, Cm=Cm)
anisotropic(sl,st,a):# store (sl,st) + fiber_angle; D_eff -> (Dxx,Dyy,Dxy) via sigma_to_D(sl,st,a,chi,Cm)
sigma_eff_value:     if sigma_eff is not None: return sigma_eff (scalar)
                     else: return sigma_i*sigma_e/(sigma_i+sigma_e)
D_eff:               return sigma_eff_value/(chi*Cm)          # scalar (iso) or (Dxx,Dyy,Dxy) (aniso)
for_monodomain:      return {'D': D_eff*Cm, 'chi': 1.0, 'Cm': Cm}   # D is Cm-INDEPENDENT = sigma_eff/chi
for_bidomain:        require sigma_i/sigma_e else raise; return {'D_i':.., 'D_e':.., 'Cm':Cm}
for_lbm:             return {'D': D_eff, 'Cm': Cm}
```
Keep the isotropic scalar path first and complete; anisotropic tensor path second.

#### Test Spec
(covered in Step 1.2)

#### Checklist
- [ ] `conductivity.py` created: frozen dataclass + 3 classmethods + `sigma_eff_value`/`D_eff` + 3 emitters.
- [ ] Scalar/isotropic path correct; anisotropic tensor path mirrors `sigma_to_D`.
- [ ] Clear `ValueError` when an emitter lacks data (e.g. `for_bidomain` with no `sigma_i/sigma_e`).
- [ ] Exported lazily from `__init__.py`.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -c "from cardiac_core import ConductivityConfig as C; \
c=C.bidomain(sigma_i=1.74, sigma_e=6.25, chi=1400); print(c.D_eff, c.for_monodomain())"
```
Expect `D_eff ≈ 0.000972` and `for_monodomain == {'D': ~0.000972, 'chi': 1.0, 'Cm': 1.0}`.

#### Exit Criteria
- [ ] Import + the one-liner prints the expected values.

#### Risk
Anisotropic reduction for unequal anisotropy ratios is non-trivial — mitigation: implement the
`sigma_to_D` rotation (matches LBM), add a `# TODO unequal-ratio` note; the gate + isotropic tests are
the load-bearing ones for now.

### Step 1.2: Lock the firewall gate as a test
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` (whole file) — reuse `run_cable_v55`, `_T_END_BY_CM`, the two gates.
- `Monodomain/Engine_V5.5/_regression/bidomain_cm_ref.json` — reference `D_EFF_input=0.0009721973895941354`, CVs 54.347826 (Cm=1) / 27.771606 (Cm=2).

#### Why
The arithmetic AND the live-engine CV are what make the firewall trustworthy at Cm≠1. A pure unit test
would miss the engine-coupling; reuse the validated cable harness.

#### Implementation Spec
**Files to create:** `cardiac_core/tests/test_conductivity.py`.

#### Pseudocode
```
test_arithmetic_gate:
    for Cm in {1,2}:
        c = ConductivityConfig.bidomain(1.74, 6.25, chi=1400, Cm=Cm)
        assert abs(c.for_monodomain()['D'] - 0.0009721973895941354) < 1e-12   # Cm-independent
        assert c.for_monodomain()['chi'] == 1.0 and c.for_monodomain()['Cm'] == Cm
        assert isclose(c.D_eff, 0.0009721973895941354/Cm)
        assert isclose(c.for_bidomain()['D_i'], 1.74/(1400*Cm))
test_live_cv_gate (mark slow/skippable):
    sys.path.insert V5.5 dir; from test_phase10_cm_scaling import run_cable_v55, _T_END_BY_CM
    for Cm in {1.0, 2.0}:
        mono = ConductivityConfig.bidomain(1.74,6.25,1400,Cm).for_monodomain()
        cv = run_cable_v55(Cm=mono['Cm'], t_end=_T_END_BY_CM[Cm], d_eff=mono['D'])
        assert abs(cv - CV_REF[Cm]) / CV_REF[Cm] <= 0.05
```
Guard the live-CV test with `pytest.mark.skipif` if the V5.5 dir / ref JSON is absent.

#### Test Spec
- `test_conductivity.py::test_arithmetic_gate` — Expected: `for_monodomain D = 0.0009721973895941354 ± 1e-12`, Cm-independent; `D_eff = D/Cm`.
- `test_conductivity.py::test_live_cv_gate` — assertion target is the **JSON reference** (`cases[Cm].cv_cm_per_s`: 54.347826 @Cm=1, **27.771606** @Cm=2), tolerance `rel ≤ 5%`. Do NOT hardcode 28.09 — that is the *V5.5 firewall* output (≈1.15% from the 27.77 bidomain reference, within tol); the comparison is `|cv_v55 − CV_REF[Cm]|/CV_REF[Cm] ≤ 0.05` with `CV_REF` read from `bidomain_cm_ref.json`.

#### Checklist
- [ ] Both tests written; live-CV test guarded/skippable.
- [ ] Constants read from `bidomain_cm_ref.json`, not hardcoded.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_conductivity.py -v
```

#### Exit Criteria
- [ ] `test_arithmetic_gate` passes; `test_live_cv_gate` passes (or SKIPs cleanly with a printed reason).

#### Risk
The live test runs a V5.5 cable (~few seconds) — mitigation: `@pytest.mark.slow`; the arithmetic gate is the always-on guard.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_conductivity.py -v
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q   # no regressions (additive phase)
```
### Phase 1 Exit Criteria
- [ ] `ConductivityConfig` importable + gate tests green.
- [ ] All 80 existing tests still pass (nothing else touched).
### Phase 1 Cleanup
- float64 (no float32). V5.3/V5.4/_archive untouched (V5.5 read-only). No duplicated shared code — `conductivity.py` is the new single home. Remove debug prints.

**-> Commit point: git commit after Phase 1** (`feat(cardiac_core): ConductivityConfig firewall + gate test`)

---

## Phase 2: `Grid` (structured-only geometry)

**Goal**: `cardiac_core/grid.py` — a single structured-grid descriptor wrapping the engines'
`StructuredGrid`. Additive; no existing code touched.
**Tier**: medium
**Estimated scope**: one module (~80 lines) + one test file.

### Phase Context
- FEM/unstructured dropped (decision 2026-06-24) — `Grid` is the ONLY geometry type. No `TriangularMesh`.
- `Grid` wraps `StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device, dtype)`.
- `(Nx, Ny)` `ij` convention; `dy` defaults to `dx`.
- Goal-1 fields/methods: `Nx, Ny, dx, dy, mask`; `Lx, Ly, coordinates -> (x,y)`, `n_dof`. The
  `to_mesh_data(...)` helper is added in Phase 4 — NOT here.

### Step 2.1: Implement `Grid`
**Model**: opus

#### Read First
- `cardiac_core/api.py:25-37` — `_prepare_engine()`; call it before importing `StructuredGrid` (both engines share `cardiac_sim`).
- `cardiac_core/geometry.py:1-50` — existing mask helpers (Grid should accept a `mask` from these).
- `cardiac_core/file_format.py:171-256` — `create_cardiac_mesh` (how `Lx,Ly,dx,mask` flow today).
- `API_REFERENCE.md` "Geometry" — the public surface.

#### Why
Consolidates LBM (inline `Nx/Ny/dx`) and the classical engines (`StructuredGrid`) behind one front, and
gives the factories a single geometry type to accept in Phase 4.

#### Implementation Spec
**Files to create:** `cardiac_core/grid.py`.
**Files to modify:** `cardiac_core/__init__.py` — export `Grid`.
**Interfaces / Signatures:**
```python
class Grid:
    def __init__(self, Nx, Ny, dx, dy=None, *, mask=None, device="cpu", dtype=torch.float64): ...
    Nx: int; Ny: int; dx: float; dy: float
    @property
    def Lx(self) -> float: ...        # dx*(Nx-1)
    @property
    def Ly(self) -> float: ...        # dy*(Ny-1)
    @property
    def coordinates(self): ...        # (x, y) meshgrid tensors (Nx,Ny), ij
    @property
    def n_dof(self) -> int: ...       # mask.sum() or Nx*Ny
    def _structured_grid(self): ...    # internal: build/cache the engine StructuredGrid
```

#### Pseudocode
```
__init__: store Nx,Ny,dx; dy = dy or dx; mask (default full True); device/dtype.
_structured_grid: prepare_engine(); StructuredGrid.create_rectangle(Lx,Ly,Nx,Ny,device,dtype); cache.
coordinates: from the structured grid (or torch.meshgrid); shape (Nx,Ny), ij indexing.
n_dof: int(mask.sum()) if mask given else Nx*Ny.
```

#### Test Spec
- `test_grid.py::test_dims` — `Grid(150,40,0.025)` → `Lx≈3.725`, `Ly≈0.975`, `dy==dx`, `n_dof==6000`.
- `test_grid.py::test_coordinates` — `coordinates` shapes `(150,40)`; `x[0,0]==0`, `x[-1,0]≈Lx`.
- `test_grid.py::test_mask_ndof` — with a circle mask, `n_dof == mask.sum()`.
- `test_grid.py::test_structured_roundtrip` — `_structured_grid()` returns an object with matching `Nx/Ny`.

#### Checklist
- [ ] `grid.py` created; `dy` defaulting; mask handling; coordinates `ij`.
- [ ] `_structured_grid()` caches (build once).
- [ ] Exported from `__init__.py`.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_grid.py -v
```

#### Exit Criteria
- [ ] All `test_grid.py` tests pass.

#### Risk
`_prepare_engine()` ordering — importing `StructuredGrid` without flushing `cardiac_sim` can grab the
wrong engine. Mitigation: route through api.py's import pattern; test the roundtrip.
**(audit LOW)** `coordinates` `ij` orientation — the `x[-1,0]≈Lx` assertion ASSUMES `StructuredGrid`'s
meshgrid convention. Mitigation: before asserting the corner value, confirm the engine's `coordinates`
property (`StructuredGrid` structured.py:~89) actual orientation rather than assuming; match the test to it.

### Phase 2 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_grid.py cardiac_core/tests/ -q
```
### Phase 2 Exit Criteria
- [ ] `Grid` importable + tests green; 80 existing tests still pass.
### Phase 2 Cleanup
- float64 default; no FEM/TriangularMesh references; V5.3/V5.4 untouched; no duplicated grid logic.

**-> Commit point: git commit after Phase 2** (`feat(cardiac_core): Grid structured-only geometry`)

---

## Phase 3: `Vm` canonicalization (alias-preserving)

**Goal**: `Vm` is the canonical voltage field on `SimulationSnapshot` and `SimulationResult`, with a
read-only `.V` alias so the 80 existing tests stay green. Glossary #2.
**Tier**: medium
**Estimated scope**: targeted edits to `api.py`, `run.py`, `io.py` + a couple of new assertions.

### Phase Context
- Glossary #2: rename `V`→`Vm` at the State/API layer; keep `.V` alias (then deprecate). Do NOT rename
  the ionic `IonicModel` ABC's positional `V` (out of scope, wide-blast).
- The `.V` alias is the key risk-reducer: existing tests use `.V` and must keep passing unchanged.
- Bidomain engine already exposes `state.Vm` (api.py:677) — only the cardiac_core-facing field renames.

### Step 3.1: Rename field on `SimulationSnapshot` + `SimulationResult`, add `.V` alias
**Model**: opus

#### Read First
- `cardiac_core/api.py:109-132` (`SimulationSnapshot`), `:197-217` (`.V`/`.t` props), `:660-700` (the three `_run_*` yield sites: `V=V_grid`).
- `cardiac_core/run.py:19-34` (`SimulationResult`), `:40-55` (`_collect`), `:60-243` (returns).
- `cardiac_core/io.py:16-92` (`save_result`/`load_result` `V` usage).

#### Why
Locks the canonical name before the factories/results surface is extended (Phases 4–5), so new code is
born using `Vm`. The alias keeps the migration non-breaking.

#### Implementation Spec
**Files to modify:**
- `api.py`: `SimulationSnapshot`: rename field `V`→`Vm`; add `@property def V(self): return self.Vm`. The three `_run_*` yield `Vm=...`. Wrapper `.V` property (line 198): rename to `.Vm` (reads `state.Vm`/`state.V`/engine `.V` as today) + add `.V` alias property delegating to `.Vm`.
- `run.py`: `SimulationResult`: rename field `V`→`Vm`; add `.V` alias property; add fields `dx`, `dy`, `ionic_states: Optional=None` (phi_e already present). `_collect`/returns populate `Vm`. **AUDIT HIGH — the `simulate()` constructor at `run.py:243` builds `SimulationResult(times=…, V=V, phi_e=…)` BY KEYWORD; after the field rename, `V=` is no longer an `__init__` parameter (it's a read-only `@property`) → TypeError. Change that call to `Vm=`.** Grep `SimulationResult(` across `run.py`/`io.py` for any other keyword construction site and fix all.
- `io.py`: `save_result(path, times, Vm=None, phi_e=None, *, V=None, **md)` — accept either voltage kw; prefer `Vm`, fall back to legacy `V=` with `DeprecationWarning`. **AUDIT HIGH (pass 3) — keep `phi_e` positional-or-keyword (BEFORE the `*` barrier). Existing callers pass it positionally as the 4th arg: `save_result(path, times, V, phi_e)` at `test_io.py:29`. Moving `phi_e` keyword-only (`*, V=None, phi_e=None`) would raise `TypeError: save_result() takes 3 positional args but 4 were given` and fail a "must-stay-green" test. The voltage in positional slot 3 binds to `Vm` (correct, since legacy positional callers put the voltage there); only the explicit legacy `V=` keyword need be keyword-only.** `load_result` tuple unchanged (positional unpack).

#### Pseudocode
```
@dataclass SimulationSnapshot: t; Vm; phi_e; Nx; Ny; dx; dy
    @property V(self): return self.Vm
SimulationResult: times; Vm; phi_e; dx; dy; ionic_states=None
    @property V(self): return self.Vm
```

#### Test Spec
- `tests/test_run.py` (existing) — must pass UNCHANGED via the `.V` alias.
- Add `tests/test_run.py::test_vm_is_canonical` — `result.Vm is result.V`; `snap.Vm` exists.

#### Checklist
- [ ] `Vm` field on both dataclasses; `.V` read-only alias property on both.
- [ ] `dx/dy` (+`ionic_states=None`) added to `SimulationResult`.
- [ ] All `_run_*` yield `Vm=`; `_collect` builds `Vm`.
- [ ] **`simulate()` keyword construction `SimulationResult(V=…)` at run.py:243 → `Vm=…`** (+ any other `SimulationResult(`/`SimulationSnapshot(` keyword sites found by grep).
- [ ] `io.save_result` accepts `Vm=`/`V=` (voltage positional slot 3 → `Vm`) AND keeps `phi_e` positional-or-keyword (the `test_io.py:29` positional `save_result(path, times, V, phi_e)` call stays green); `load_result` tuple unchanged.
- [ ] New `test_vm_is_canonical`.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```

#### Exit Criteria
- [ ] All 80 existing tests pass (alias works); `test_vm_is_canonical` passes.

#### Risk
A dataclass can't have both a `Vm` field and a `V` field — mitigation: field is `Vm`, `V` is a plain
`@property` (read-only alias; setting `.V` intentionally unsupported).

### Phase 3 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
### Phase 3 Exit Criteria
- [ ] 80 existing + new test green; `Vm` canonical, `.V` alias works.
### Phase 3 Cleanup
- Grep for stray `.V` in NON-test cardiac_core source that should be `.Vm` (leave the alias + tests). float64. V5.3/V5.4 untouched.

**-> Commit point: git commit after Phase 3** (`refactor(cardiac_core): canonical Vm with read-only .V alias`)

---

## Phase 4: Declarative factory construction + `Simulation` surface

**Goal**: Factories `monodomain/bidomain/lbm` construct from `(Grid, ionic_model, ConductivityConfig,
stimulus)` (assembling a `CardiacMeshData` internally), preserving the legacy `mesh=` path; implement
`reset/stimulate/with_/introspection`; define the `Simulation` Protocol.
**Tier**: large
**Estimated scope**: the integrative phase — api.py factory bodies + several `CardiacSimulation` methods + a Protocol module.

### Phase Context
- This is a WRAPPER extension — engines/solvers are NOT modified. New construction assembles a
  `CardiacMeshData` from `(Grid, ConductivityConfig, stimulus, ionic_model, dt)` then calls the existing
  engine-construction path. Conductivity flows via `ConductivityConfig` → `CardiacMeshData` fields.
- **Integration risk to resolve FIRST (Step 4.0):** verify how api.py currently maps
  `CardiacMeshData.D_xx/chi/Cm` into each engine's constructor, so the config's outputs land correctly.
  The gate proved the *engine* wants monodomain `D=sigma_eff/chi, chi=1, Cm`; confirm cardiac_core feeds
  that (not silently `chi=1400` to the monodomain engine, which would mis-scale 1400×).
- Signature compatibility: keep the legacy positional `mesh`. New shape:
  `monodomain(geometry=None, ionic_model=None, conductivity=None, stimulus=None, *, mesh=None, dt=..., …)`.
  Type-sniff: if `geometry` is a `CardiacMeshData`/`str`, treat it as `mesh` (back-compat).

### Step 4.0: Verify cardiac_core → engine conductivity mapping
**Model**: opus

#### Read First
- `cardiac_core/api.py:802-898` (`monodomain`), `:901-1044` (`bidomain`), `:1047-1143` (`lbm`) — the engine-construction bodies. Find where `data.D_xx`, `data.chi`, `data.Cm`, `data.sigma_i/e` are read and what is passed to `FDMDiscretization`/`BidomainConductivity`/LBM.
- `cardiac_core/file_format.py:171-256` — what `D_xx` actually holds today (effective vs raw) and the default `D=0.001`.

#### Why
If cardiac_core feeds the monodomain engine `chi=1400` with `D_xx`=effective-diffusivity, the existing
wrapper would mis-scale (D_phys = D_xx/1400) — OR it already passes `chi=1`. Either way `ConductivityConfig`
must feed whatever the existing path expects. This read prevents a silent CV bug in Step 4.1.

#### Implementation Spec
**Files to modify:** none (investigation). Produce a 3-line note (commit message / Mutation Log) stating, per engine: "cardiac_core passes D=___, chi=___, Cm=___ to the engine constructor."

#### Pseudocode
N/A — investigation/read-only step (no code produced; the deliverable is the documented `(D, chi, Cm)` mapping consumed by Steps 4.1/4.2).

#### Test Spec
N/A — investigation step (no new code to test). The mapping it produces is exercised end-to-end by Step 4.1's `test_construction_api.py` and Step 4.2's CV smoke test.

#### Checklist
- [ ] Documented, per engine, the exact `(D, chi, Cm)` (or `D_i/D_e`) values cardiac_core hands the engine today.
- [ ] Decided the mapping: `ConductivityConfig.for_monodomain()` → which `CardiacMeshData`/engine fields.
- [ ] **(audit HIGH) Bidomain branch:** documented how the bidomain factory consumes `CardiacMeshData.sigma_i`/`sigma_e` — confirmed it's a `(xx,yy,xy)` **conductivity** tuple (api.py:962-970), and how it derives `D_i/D_e` (σ/(χ·Cm)? sigma_ratio fallback when absent?). Decided: feed raw σ tuples `(σ,σ,0)` from the config (per Phase 1 Layer note), NOT `for_bidomain()`'s D output.
- [ ] **(audit MEDIUM) Reaction `/Cm`:** confirmed the repointed V5.5 monodomain stepper divides by `Cm` (Phase 0). Step 4.0 covers the *diffusion* mapping; the *reaction*-Cm correctness is delivered by Phase 0, NOT by this step — do not rely on the Cm=1 CV smoke (4.2) to catch a reaction-Cm error (it can't, at Cm=1).

#### Verify
```bash
conda run -n heart-conduction python -c "from cardiac_core import create_cardiac_mesh; \
m=create_cardiac_mesh(2.0,0.5,0.01); print('D_xx', m.D_xx.mean(), 'chi', m.chi, 'Cm', m.Cm)"
```

#### Exit Criteria
- [ ] The mapping is written down and consistent with the firewall gate (monodomain effectively sees `D_phys = sigma_eff/(chi·Cm)`).

#### Risk
The legacy default `D=0.001` with `chi=1400` may NOT be calibrated (nominal) — mitigation: new path uses `ConductivityConfig`; Step 4.2 CV smoke test catches mis-scaling.

### Step 4.1: Factories accept `(Grid, ionic_model, ConductivityConfig, stimulus)`
**Model**: opus

#### Read First
- Step 4.0 findings.
- `cardiac_core/api.py:708-712` (`_resolve_mesh`), `:802-1143` (factory bodies).
- `API_REFERENCE.md` "DECLARE — engine factories" (per-engine knob tables).
- `Bidomain/Engine_V1/cardiac_sim/tissue_builder/stimulus/protocol.py` + `Monodomain/Engine_V5.5/cardiac_sim/tissue_builder/stimulus/protocol.py` — the `StimulusProtocol` the engines consume (so `stimulus` maps to `CardiacMeshData.stimuli` / passes through).

#### Why
This is the construction idiom the whole API rests on. Assembling a `CardiacMeshData` reuses the
validated engine-construction path (low risk) while presenting the design's surface.

#### Implementation Spec
**Files to modify:** `cardiac_core/api.py` — three factory signatures + a `_build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, Cm)` helper; add a `Grid.to_mesh_data(...)` convenience in `grid.py`.
**Interfaces:** per `API_REFERENCE.md`. Back-compat: legacy `mesh` preserved via type-sniff + `mesh=` kwarg.

#### Pseudocode
```
def monodomain(geometry=None, ionic_model=None, conductivity=None, stimulus=None, *, mesh=None, dt=0.02, Cm=1.0, splitting=..., ...):
    if isinstance(geometry, (str, CardiacMeshData)): mesh, geometry = geometry, None
    if mesh is None:
        mesh = _build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, Cm)
    # ... existing construction from `mesh` unchanged ...
_build_mesh_data(geometry, ionic_model, conductivity, stimulus, dt, Cm):
    Grid -> dx, dy, mask
    MONODOMAIN branch: m = conductivity.for_monodomain();  D_xx=D_yy=m['D'], D_xy=0.0, chi=m['chi'](=1.0), Cm=m['Cm']
    BIDOMAIN branch:   sigma_i=(conductivity.sigma_i, conductivity.sigma_i, 0.0),
                       sigma_e=(conductivity.sigma_e, conductivity.sigma_e, 0.0),  # RAW σ tuples (audit HIGH)
                       chi=conductivity.chi, Cm=conductivity.Cm   # existing factory does σ->D internally; NOT for_bidomain()
    stimulus -> mesh.stimuli (accept callable region OR (Nx,Ny) mask; mirror CardiacMeshData.stimuli format)
```
> Scalar→tuple: the config stores scalar σ; `CardiacMeshData.sigma_i/e` need `(xx,yy,xy)` tuples → wrap as
> `(σ, σ, 0.0)` (isotropic). Anisotropic configs supply the real tensor. Confirm against Step 4.0 findings.

#### Test Spec
- `tests/test_construction_api.py::test_monodomain_from_grid` — `monodomain(Grid(...), 'ttp06', ConductivityConfig.bidomain(1.74,6.25), stim)`; `sim.Nx/Ny/dx` correct; `step()` runs.
- `::test_legacy_mesh_path` — `monodomain(create_cardiac_mesh(...))` still works.
- `::test_bidomain_from_grid`, `::test_lbm_from_grid` — analogous.

#### Checklist
- [ ] Three factories accept the new shape + preserve `mesh`.
- [ ] `_build_mesh_data` maps Grid + ConductivityConfig + stimulus → `CardiacMeshData`.
- [ ] Conductivity mapping matches Step 4.0 (no mis-scaling).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_construction_api.py -v
```

#### Exit Criteria
- [ ] New construction + legacy mesh path both green.

#### Risk
Conductivity mis-scaling (Step 4.0) — mitigation: Step 4.2 CV smoke test. Stimulus region (callable vs mask) mismatch — mitigation: accept both; mirror `CardiacMeshData.stimuli` format.

### Step 4.2: CV smoke test through the new construction path
**Model**: opus

#### Read First
- Step 4.0 findings (the documented per-engine `(D, chi, Cm)` mapping) + Step 4.1's `_build_mesh_data`.
- `cardiac_core/analysis.py` — `conduction_velocity(V, times, dx, x1, x2, y, ...)` signature (the exact CV-measurement call this test makes — don't guess the arg names).
- `cardiac_core/grid.py` — `Grid` constructor (from Step 2.1), for the geometry the test builds.

#### Why
Proves the assembled `CardiacMeshData` from `ConductivityConfig` propagates to the right physical
diffusivity end-to-end (catches a Step-4.0 mis-mapping).

#### Implementation Spec
**Files to modify:** `tests/test_construction_api.py` — add a CV test.

#### Pseudocode
```
sim = monodomain(Grid(200, 50, 0.01), 'ttp06', ConductivityConfig.bidomain(1.74, 6.25, 1400), left_edge_stim)
r   = sim.run(t_end=50)
cv  = analysis.conduction_velocity(r.Vm, r.times, r.dx, x1=.., x2=.., y=..)
assert 10 < cv < 100   # physiological cm/s; ~0.04 cm/s would mean an extra /chi double-scaling
```

#### Test Spec
- `::test_monodomain_cv_via_config` — `monodomain(Grid(200,50,0.01),'ttp06',ConductivityConfig.bidomain(1.74,6.25,1400), left-edge stim).run(t_end=50)`; CV via `analysis.conduction_velocity`. Expected: physiological (tens of cm/s; NOT ~0.04 cm/s = `/chi` double-scaling). Tolerance loose (±20%) — wiring check, not the precision gate.

#### Checklist
- [ ] CV test added + passing in a physiological band.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_construction_api.py::test_monodomain_cv_via_config -v
```

#### Exit Criteria
- [ ] CV physiological (mis-scaling would be ~1000× off — easy to catch).

#### Risk
Loose tolerance could mask a small error — acceptable; the precise gate is Phase 1's live-CV test.

### Step 4.3: Implement `reset/stimulate/with_/introspection` + `Simulation` Protocol
**Model**: opus

#### Read First
- `cardiac_core/api.py:189-191` (`reset` NotImplementedError), `:243-289` (stimulus stubs), `:631-658` (introspection props — **only `Nx/Ny/dx/dy/mask/engine_type` exist today; `dt/Cm/ionic_model` are MISSING — audit HIGH**).
- `cardiac_core/api.py:1082-1143` (LBM factory) + `LBM/Engine_V1/src/simulation.py:136` — LBM `add_stimulus(mask, start=…)` uses `start=`, not `start_time=` (audit LOW).
- `API_REFERENCE.md` "The `Simulation` object".

#### Why
These are the remaining idioms (CHANGE = `with_`, STIMULATE). `with_` must be functional (immutable) per
the 2026-06-24 decision. The Protocol gives Optimizer/Surrogate a typed target.

#### Implementation Spec
**Files to create:** `cardiac_core/simulation.py` — `Simulation` `typing.Protocol` (`runtime_checkable`): introspection + `run/step/reset/with_/stimulate`.
**Files to modify:** `cardiac_core/api.py` `CardiacSimulation`:
- **(audit HIGH) ADD the missing introspection properties** `dt`, `Cm`, `ionic_model` (read from the stored construction args / the underlying engine). Without them the `with_(dt=…)` test and `isinstance(sim, Simulation)` (runtime_checkable) both fail.
- store the construction args **and** the resolved `CardiacMeshData` at build time (so `with_`/`reset` can replay BOTH the declarative path and the legacy `mesh=` path).
- `reset()` — rebuild engine from the stored construction record → t=0.
- `with_(**overrides)` — copy the stored record, apply overrides, return a NEW sim via the factory. Original untouched.
- `stimulate(region, start_time, duration=1.0, amplitude=-52.0)` — append to the stimulus source + rebuild. **(audit MEDIUM) Must work for BOTH paths**: declarative → append to the stored `StimulusProtocol`; legacy mesh → append to `self._data.stimuli`. **(audit LOW) LBM path**: translate `start_time`→`start` when forwarding to LBM's `add_stimulus`.

#### Pseudocode
```
# build time: self._record = {'engine_type':…, 'args':{…}, 'data':CardiacMeshData}
@property dt(self):  return self._record['args'].get('dt') or self._data.dt
@property Cm(self):  return self._record['args'].get('Cm', getattr(self._engine, 'Cm', 1.0))
@property ionic_model(self): return self._record['args'].get('ionic_model') or self._data.ionic_model
reset:  self.__init__(*_FACTORY[self._engine_type]._construct(self._record))   # replay record -> t=0
with_:  r = deepcopy(self._record); r['args'].update(overrides); return _FACTORY[self._engine_type](**r['args'], mesh=r.get('data') if no declarative args)
stimulate(region, start_time, duration, amplitude):
    if self._record['args'].get('stimulus') is not None: self._record['args']['stimulus'].add_stimulus(region, start_time, duration, amplitude)
    else: self._data.stimuli.append(_make_stim(region, start_time, duration, amplitude))   # mesh path
    self.reset()
```

#### Test Spec
- `::test_introspection` — `sim.dt`, `sim.Cm`, `sim.ionic_model` all readable + correct (audit HIGH).
- `::test_with_is_functional` — `s2=s.with_(dt=0.01)`; `s.dt!=s2.dt`; `s` unchanged.
- `::test_reset` — run, `reset()`, `t==0`, `Vm` back to rest.
- `::test_stimulate_adds` — declarative-built sim: `stimulate(...)` then run → activation occurs.
- `::test_stimulate_mesh_path` — sim built via legacy `mesh=`: `stimulate(...)` appends to `data.stimuli` and reset/run works (audit MEDIUM).
- `::test_protocol_isinstance` — `isinstance(sim, Simulation)` (runtime_checkable) is True (needs `dt/Cm/ionic_model`).

#### Checklist
- [ ] Construction record (args + resolved `CardiacMeshData`) stored on the sim.
- [ ] **`dt`, `Cm`, `ionic_model` introspection properties added** (Protocol + `with_(dt=…)` test need them).
- [ ] `reset/with_/stimulate` implemented (no longer NotImplementedError); **work for BOTH declarative and legacy `mesh=` paths**.
- [ ] **`stimulate` LBM path translates `start_time`→`start`.**
- [ ] `Simulation` Protocol defined + exported; `runtime_checkable`; `isinstance(sim, Simulation)` true.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_construction_api.py -v
```

#### Exit Criteria
- [ ] All construction-api tests green; `with_` immutable.

#### Risk
`with_` copying live engine state — mitigation: rebuild from construction args (don't copy live engine). `stimulate` timing — mitigation: document "applies from next reset/run"; test the add-then-run path.

### Phase 4 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
### Phase 4 Exit Criteria
- [ ] New construction (Grid+ConductivityConfig+stimulus) + legacy mesh path both work.
- [ ] CV smoke physiological; `reset/with_/stimulate` implemented; Protocol defined.
- [ ] 80 existing tests still pass.
### Phase 4 Cleanup
- Remove the now-done `NotImplementedError` stubs (leave genuinely-out-of-scope ones — probes, clamps — with a clear note). float64. No solver-internal changes. V5.3/V5.4 untouched.

**-> Commit point: git commit after Phase 4** (`feat(cardiac_core): declarative factory construction + Simulation idioms`)

---

## Phase 5: `run()` eager/batch contract + `SimulationResult` analysis hooks

**Goal**: `sim.run(t_end, save_every, *, batch=None, record=("Vm",), callback=None)` returns one
`SimulationResult` (eager) or an `Iterator[SimulationResult]` (batch=k); `SimulationResult` carries
`Vm (T,Nx,Ny)`, `phi_e`, optional `ionic_states`, `dx/dy`, and `.cv()/.apd()/.lat()/.restitution()`.
**Tier**: large
**Estimated scope**: rework `CardiacSimulation.run` from snapshot-generator → the unified contract; add result hooks.

### Phase Context
- TODAY `CardiacSimulation.run()` yields `SimulationSnapshot`. The new contract returns an EAGER
  `SimulationResult` (not iterable as snapshots). Keep an internal `_iter_snapshots()` generator + a
  public `sim.snapshots()` alias, and BUILD `SimulationResult` from it.
- **AUDIT HIGH — the `snapshots()` alias does NOT auto-green callers.** ~25 call sites do
  `for snap in sim.run(...)` / `list(sim.run(...))`. Each MUST be edited to `sim.snapshots(...)` (or
  `sim.run(...).Vm`). The "no regressions / 80 tests pass" criterion therefore means **"pass AFTER
  migrating the run()-iterating sites"** — an intended, documented contract change, not a
  magically-preserved API.
- **AUDIT HIGH (re-audit) — the migration includes a PRODUCTION site, not just tests.**
  `cardiac_core/run.py:44` `_collect()` does `for snap in sim.run(t_end, save_every)` and is the engine
  behind `run_monodomain/run_bidomain/run_lbm/simulate` (run.py:102,151,194,242). If only test files are
  migrated, `_collect` breaks and `tests/test_run.py` (11 tests) fails. **Migrate `_collect` → `sim.snapshots(...)` too.**
- Migration grep is the WHOLE package, not just tests: `grep -rn "sim.run(\|\.run(" cardiac_core/` → for
  each iterating site (incl. `run.py:_collect`), swap `run`→`snapshots`. Sites that already call
  `simulate(...)` / `run_*(...)` (tuple / `SimulationResult` returns) are unaffected. `_collect` accesses
  `snap.V` — the `.V` alias (Phase 3) keeps that line working; only `.run`→`.snapshots` changes.
- `record=` controls which fields are collected; `phi_e` auto for bidomain; `ionic_states` opt-in.
- `batch=k` yields `SimulationResult` chunks of ≤k save-points (k=1 = frame-by-frame).
- `SimulationResult.cv()/apd()/...` are thin wrappers over `cardiac_core.analysis` (expects `(T,Nx,Ny)`).

### Step 5.1: Unified `run()` contract
**Model**: opus

#### Read First
- `cardiac_core/api.py:165-187` (current generator `run`/`step`).
- `cardiac_core/run.py:40-55` (`_collect` — reuse to assemble a `SimulationResult`).
- `API_DESIGN.md` §5–6 + `API_REFERENCE.md` "run(...)" (the batch/record table).

#### Why
This is the RUN idiom — the single public output type (`SimulationResult`). The batch knob folds
streaming into one verb (decision 2026-06-02).

#### Implementation Spec
**Files to modify:** `cardiac_core/api.py` — refactor `CardiacSimulation.run`:
- internal `_iter_snapshots(t_end, save_every, record, callback)` (the old generator).
- public `sim.snapshots(...)` = the old generator (back-compat alias).
- public `run(t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None)`:
  - `batch is None` → drain `_iter_snapshots`, stack into one `SimulationResult`.
  - `batch=k` → yield `SimulationResult` chunks of ≤k.
- **Return type (audit LOW, pass 3):** annotate `run(...) -> "SimulationResult | Iterator[SimulationResult]"` (Union) so the eager-vs-batch polymorphism is explicit for the typed Optimizer/Surrogate targets the plan elsewhere cares about.
- **`record=("ionic_states",)` wiring (audit LOW, pass 3) — make it real, not a silent pass.** Today the `_run_*` generators yield only `t/V/phi_e` (api.py:662-699), so `ionic_states` is never populated and the weak `is not None` test could pass on `None`. Wire it: `SimulationSnapshot` gains an optional `ionic_states` field; when `"ionic_states" in record`, `_iter_snapshots` attaches the live `state.ionic_states` (present on monodomain `SimulationState` + `BidomainState`); `_result_from` stacks → `SimulationResult.ionic_states (T, n_states, Nx, Ny)`. The LBM path (gates live on the sim object, no uniform container) raises `NotImplementedError("ionic_states recording not supported for LBM")` — an **explicit raise, NOT a silent `None`**.

#### Pseudocode
```
def run(self, t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None) -> "SimulationResult | Iterator[SimulationResult]":
    it = self._iter_snapshots(t_end, save_every, record, callback)
    if batch is None: return _result_from(list(it), record, self.dx, self.dy)
    def gen():
        buf=[]
        for s in it:
            buf.append(s)
            if len(buf)==batch: yield _result_from(buf, ...); buf=[]
        if buf: yield _result_from(buf, ...)
    return gen()
```

#### Test Spec
- `tests/test_run_contract.py::test_eager_returns_result` — `r=sim.run(t_end=20)`; `isinstance(r,SimulationResult)`; `r.Vm.shape==(T,Nx,Ny)`.
- `::test_batch_streams` — `list(sim.run(t_end=20, batch=5))` → all `SimulationResult`, each `T<=5`; concatenated equals eager.
- `::test_record_ionic_states` — monodomain `record=("Vm","ionic_states")` → `r.ionic_states` is a REAL tensor of shape `(T, n_states, Nx, Ny)` (assert the shape, NOT merely `is not None`); LBM `record=("ionic_states",)` → `pytest.raises(NotImplementedError)` (asserts the explicit raise — closes the silent-escape-hatch).
- existing snapshot-iterating tests migrated to `sim.snapshots()` or `sim.run().Vm`.

#### Checklist
- [ ] `_iter_snapshots` + new `run` (eager + batch).
- [ ] `sim.snapshots()` generator alias added.
- [ ] **Production `run.py:44` `_collect()` migrated `sim.run`→`sim.snapshots`** (re-audit HIGH — powers `run_*`/`simulate`).
- [ ] **ALL ~25 `for snap in sim.run(...)` / `list(sim.run(...))` sites migrated to `sim.snapshots(...)`** (grep `cardiac_core/` — package-wide, not just `tests/`; none missed).
- [ ] `record` honored; `phi_e` auto for bidomain.
- [ ] `run()` annotated `-> "SimulationResult | Iterator[SimulationResult]"` (Union — explicit eager/batch polymorphism).
- [ ] `ionic_states` recording wired for classical engines (real `(T, n_states, Nx, Ny)` tensor) + explicit `NotImplementedError` for LBM (no silent `None`).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_run_contract.py cardiac_core/tests/ -q
```

#### Exit Criteria
- [ ] Eager + batch contract green; existing tests green (migrated where needed).

#### Risk
Changing `run()`'s return type breaks `for snap in sim.run()` — mitigation: `sim.snapshots()` alias + migrate the few call sites; this is an intended, documented contract change.

### Step 5.2: `SimulationResult` analysis hooks
**Model**: opus

#### Read First
- `cardiac_core/run.py:19-34` (`SimulationResult` — now `Vm/phi_e/dx/dy/ionic_states` from Phase 3).
- `cardiac_core/analysis.py` — `conduction_velocity(V,times,dx,x1,x2,y,...)`, `apd_map`, `activation_time`, `restitution_curve` signatures.

#### Why
"Thin `.cv()/.apd()` hooks into analysis.py" (glossary #6) make results self-describing for the LLM/user.

#### Implementation Spec
**Files to modify:** `cardiac_core/run.py` — add methods to `SimulationResult` delegating to `analysis`.
**Interfaces:** `cv(**kw)`, `apd(**kw)`, `lat(**kw)`, `restitution(**kw)` — pass `self.Vm, self.times, self.dx`.

#### Pseudocode
```
def cv(self, **kw):  from cardiac_core import analysis; return analysis.conduction_velocity(self.Vm, self.times, self.dx, **kw)
def apd(self, **kw): return analysis.apd_map(self.Vm, self.times, **kw)
def lat(self, **kw): return analysis.activation_time(self.Vm, self.times, **kw)
def restitution(self, **kw): return analysis.restitution_curve(self.Vm, self.times, **kw)
```

#### Test Spec
- `::test_result_cv_hook` — `sim.run(t_end=50).cv(x1=.., x2=.., y=..)` equals a direct `analysis.conduction_velocity` call.
- `::test_result_apd_hook` — `.apd()` shape `(Nx,Ny)`.

#### Checklist
- [ ] `.cv/.apd/.lat/.restitution` on `SimulationResult`, delegating to `analysis`.
- [ ] Hooks match direct `analysis` calls (same numbers).

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_run_contract.py -v
```

#### Exit Criteria
- [ ] Hooks present + numerically equal to direct analysis calls.

#### Risk
`analysis.conduction_velocity` needs index args (`x1,x2,y`) — mitigation: pass through `**kw`; document required args in the hook docstring.

### Phase 5 Verification
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q
```
### Phase 5 Exit Criteria
- [ ] `run()` eager/batch contract + `SimulationResult` hooks green.
- [ ] All existing tests pass (migrated snapshot iterators).
### Phase 5 Cleanup
- float64. Update `API_REFERENCE.md` status tags `[design]`→`[now]` for delivered symbols. V5.3/V5.4 untouched. No duplicated analysis logic (hooks delegate).

**-> Commit point: git commit after Phase 5** (`feat(cardiac_core): unified run() eager/batch + SimulationResult hooks`)

---

## Final Cleanup (cross-phase)
- [ ] float64 consistency across all new modules — no float32 leaks.
- [ ] V5.3/V5.4/`_archive` untouched; V5.5 read-only (gate oracle only).
- [ ] No duplicated shared code — conductivity/grid live ONLY in `cardiac_core`.
- [ ] `API_REFERENCE.md` Appendix updated: delivered symbols `[design]`→`[now]`; note SimulationSpec/create_simulation still deferred; **fix the stale "77 tests"→current count (audit LOW)** in API_REFERENCE.md:411 (+ KNOWLEDGE/IDEALOG where they say 77).
- [ ] KNOWLEDGE.md migration-plan + IDEALOG session log updated with what shipped.
- [ ] Decide keep/relocate `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` (its logic now lives in `tests/test_conductivity.py`).
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_cardiac-core-unified-construction-api.md"
```
- [ ] Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Out of Scope (follow-up plans)
- `SimulationSpec` / `GridSpec` / `PacingSpec` / `create_simulation` — the declarative Goal-2 intake bridge.
- Monodomain Form-A→B convergence + delete `for_monodomain()` + drop FEM scheme — the Phase-4 *engine rewire* (consolidation track), distinct from this Goal-1 *API* track.
- Ionic ABC positional `V`→`Vm` rename (wide-blast, separate).

## Mutation Log

Revision pass 2026-06-24 — adversarial audit (`/audit`, 14 findings: 1 critical, 5 high, 4 medium, 4 low). No steps were completed; all changes are spec patches, not invalidations.

**MUTATED 2026-06-24**: Phase 0 ADDED — audit CRITICAL. cardiac_core `monodomain()` imports V5.4 (`api.py:841`, `/Cm`-less reaction); the firewall gate validates V5.5. New Phase 0 repoints the monodomain path V5.4→V5.5 so the delivered factory is Cm-correct (KNOWLEDGE.md:231 mandate). Added matching Success Criterion + Architecture-Changes entry + API-track phase-numbering banner (audit MEDIUM #4).
**MUTATED 2026-06-24**: Step 1.2 MODIFIED — audit MEDIUM. Test assertion target is the JSON reference (27.771606 @Cm=2), NOT the 28.09 V5.5-firewall output; read `CV_REF` from `bidomain_cm_ref.json`, don't hardcode.
**MUTATED 2026-06-24**: Phase 1 Context MODIFIED — audit HIGH. Added the for_bidomain layer note: `for_*` emitters return engine-level D (post-rewire direct-feed); the wrapper-era `CardiacMeshData` path feeds monodomain via `for_monodomain()` (D→D_xx, chi=1) and bidomain via RAW σ tuples `(σ,σ,0)` (factory does σ→D), NOT `for_bidomain()`.
**MUTATED 2026-06-24**: Step 3.1 MODIFIED — audit HIGH. Named the `simulate()` keyword construction `SimulationResult(V=…)` at run.py:243 (TypeError after field rename) + grep for all keyword construction sites; added to checklist.
**MUTATED 2026-06-24**: Step 4.0 MODIFIED — audit HIGH+MEDIUM. Expanded to document the bidomain σ-tuple mapping (api.py:962-970) and to note the reaction-`/Cm` correctness is delivered by Phase 0 (not catchable by the Cm=1 smoke test 4.2).
**MUTATED 2026-06-24**: Step 4.1 MODIFIED — audit HIGH. `_build_mesh_data` pseudocode now specifies the monodomain (for_monodomain→D_xx/chi=1/Cm) vs bidomain (raw σ tuples) mapping + scalar→tuple wrapping.
**MUTATED 2026-06-24**: Step 4.3 MODIFIED — audit HIGH+MEDIUM+LOW. Added the missing `dt/Cm/ionic_model` introspection properties (Protocol/`with_` need them); made `with_/reset/stimulate` work for BOTH declarative and legacy `mesh=` paths; LBM `start_time`→`start` translation; added `test_introspection` + `test_stimulate_mesh_path`; corrected the "introspection already working" Read-First claim.
**MUTATED 2026-06-24**: Step 5.1 MODIFIED — audit HIGH. Enumerated the ~25 `for snap in sim.run(...)`/`list(sim.run(...))` test sites that the `snapshots()` alias does NOT auto-green; made migration an explicit checklist item; reconciled the "80 tests pass" criterion as "pass AFTER migration" (documented contract change).
**MUTATED 2026-06-24**: Step 2.1 MODIFIED — audit LOW. Added the `coordinates` `ij`-orientation confirmation to Risk (don't assume `x[-1,0]≈Lx`).
**MUTATED 2026-06-24**: Final Cleanup MODIFIED — audit LOW. Added fixing the stale "77 tests"→current count in API_REFERENCE.md:411 + KNOWLEDGE/IDEALOG.

Revision pass 2 (2026-06-24) — RE-AUDIT (4 findings: 0 critical, 1 high, 2 medium, 1+1 low). Prior 14 confirmed CLOSED except #5 (PARTIAL → now fully closed below). All re-audit code/line claims verified true.

**MUTATED 2026-06-24 (pass 2)**: Step 5.1 MODIFIED — re-audit HIGH. The Phase-5 migration missed the PRODUCTION `_collect()` site at `run.py:44` (`for snap in sim.run(...)`), which powers `run_monodomain/bidomain/lbm/simulate` (run.py:102,151,194,242) and `test_run.py`. Verified at run.py:44. Broadened the migration grep to the WHOLE package (`cardiac_core/`, not just `tests/`) + named `_collect` explicitly in Context + Checklist. This fully closes prior finding #5.
**MUTATED 2026-06-24 (pass 2)**: Architecture-Changes MODIFIED — re-audit MEDIUM. Removed the misleading "(and `run_monodomain`)" repoint claim — `run_monodomain` calls the factory and imports no engine itself (run.py:97-101), so it needs no change.
**MUTATED 2026-06-24 (pass 2)**: Phase 0 Context MODIFIED — re-audit MEDIUM. Clarified the LBM *engine* is a separate package (`LBM/Engine_V1`, already Form-B); the api.py:1085 swap is the ionic-MODEL import only — no LBM reaction was ever at risk.
**MUTATED 2026-06-24 (pass 2)**: Step 0.1 Test Spec MODIFIED — re-audit LOW. Strengthened `test_engine_is_v55` from a brittle path/`state.Cm` check (both pass on V5.4) to a behavioral Cm-SENSITIVITY check (`dV(Cm=2)≈dV(Cm=1)/2`).
**MUTATED 2026-06-24 (pass 2)**: Phase 0 Cleanup MODIFIED — re-audit LOW. Remove the now-dead `_V54_PATH` with a comment instead of leaving it dangling.

Revision pass 3 (2026-06-24) — adversarial audit, final pre-implementation pass (5 findings: 0 critical, 1 high, 2 medium, 2 low). All cited line numbers + the Cm-firewall arithmetic re-verified true by the auditor. No steps completed; all changes are spec patches.

**MUTATED 2026-06-24 (pass 3)**: Step 3.1 MODIFIED — audit HIGH. The proposed `save_result(path, times, Vm=None, *, V=None, phi_e=None, **md)` put `phi_e` keyword-only, breaking the positional `save_result(path, times, V, phi_e)` call at `test_io.py:29` (`TypeError`, fails a "80-must-stay-green" test). Verified against `io.py:16-21` + `test_io.py:29`. Fixed the signature to `save_result(path, times, Vm=None, phi_e=None, *, V=None, **md)` (phi_e positional-or-keyword; only legacy `V=` keyword-only) + added the back-compat note to the checklist.
**MUTATED 2026-06-24 (pass 3)**: Step 4.0 MODIFIED — audit MEDIUM. Added explicit "N/A — investigation step" notes under the missing Pseudocode + Test Spec headers (9-section structural contract; mirrors Step 1.1's pattern so a cold-start agent doesn't read the absence as an oversight).
**MUTATED 2026-06-24 (pass 3)**: Step 4.2 MODIFIED — audit MEDIUM. Added the missing Read First (→ Step 4.0/4.1 findings + `analysis.conduction_velocity` signature + `Grid`) and Pseudocode (the CV-smoke call) so the step is cold-start-executable without guessing the measurement call.
**MUTATED 2026-06-24 (pass 3)**: Step 0.1 Test Spec MODIFIED — audit LOW. Corrected the inaccurate rationale: V5.4's `classical/state.py` has **no** `Cm` field (V5.5 adds `Cm: float = 1.0` at state.py:87), so the prior "`state.Cm` exists in V5.4 too" claim is false. The behavioral `dV(Cm=2)≈dV(Cm=1)/2` check stays (correct choice); only the stated reason was wrong.
**MUTATED 2026-06-24 (pass 3)**: Step 5.1 MODIFIED — audit LOW (×2). (a) Annotated `run() -> "SimulationResult | Iterator[SimulationResult]"` so the eager/batch polymorphism is explicit. (b) Specified the `record=("ionic_states",)` wiring (attach live `state.ionic_states` via `_iter_snapshots` → stack in `_result_from`; LBM raises `NotImplementedError`) and strengthened the test from the silent `is not None` escape-hatch to a real shape assertion + an explicit-raise assertion.

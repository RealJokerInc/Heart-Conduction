# Engine Consolidation — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

Three engines solve cardiac electrophysiology with 15+ duplicated files (ionic models, mesh, stimulus, solvers). They use two chi/Cm formulations, both correct but internally incompatible.

**Target architecture**: `cardiac_core/` is the unified codebase — it owns shared code (ionic models, mesh, stimulus, conductivity) and provides the public API. Engines import from `cardiac_core/`, not the other way around. Engine directories shrink to solver-specific code only.

```
cardiac_core/                  ← THE source of truth
  ionic/                       ← TTP06, ORd, PHAS13, MHAS13 (one copy)
  mesh/                        ← StructuredGrid, BoundarySpec (one copy)
  stimulus/                    ← StimulusProtocol (one copy)
  conductivity.py              ← ConductivityConfig (sigma → D, chi/Cm in one place)
  file_format.py               ← CardiacMeshData, save/load/create
  api.py                       ← monodomain(), bidomain(), lbm()

Bidomain/Engine_V1/            ← imports cardiac_core.ionic, cardiac_core.mesh
  solver/                      ← decoupled GS, 3-tier elliptic (unique to bidomain)
Monodomain/Engine_V5.4/        ← imports cardiac_core.ionic, cardiac_core.mesh
  solver/                      ← CN/BDF/RK diffusion solvers (unique to monodomain)
Monodomain/LBM_V1/             ← imports cardiac_core.ionic
  collision/, streaming/       ← BGK/MRT, D2Q5/D2Q9 (unique to LBM)
```

**Current state**: Phase 0 complete — API wrapper layer imports FROM engines (temporary). Engines still have their own copies. The `_prepare_engine()` function flushes `sys.modules` because both engines use the `cardiac_sim` namespace; eliminated in Phase 1.

### Chi/Cm Formulations (audited March 2026)

**Formulation A** (Monodomain V5.4): chi·Cm appears in mass/time-derivative term. Ionic solver does NOT divide by Cm.
```
Operator: A = chi·Cm · I ± θ·dt · L
Ionic:    V += dt · (-(I_ion + I_stim))          # no Cm division
LBM src:  S = -(I_ion + I_stim) / (chi·Cm)
```

**Formulation B** (Bidomain V1, LBM V1): D already contains chi·Cm. Mass term is 1/dt. Ionic solver divides by Cm.
```
Operator: A = 1/dt · I ± θ · L
Ionic:    V += dt · (-(I_ion + I_stim) / Cm)     # divides by Cm
LBM src:  R = -(I_ion + I_stim) / Cm
```

Both produce identical results when Cm = 1.0. V5.4's formulation is fragile: changing Cm from 1.0 would silently break the ionic step because the chi·Cm factor is absorbed by the diffusion mass term, coupling the two halves of the operator split.

### Engine-by-engine chi/Cm audit

**Bidomain V1** (Formulation B):
- `BidomainConductivity`: D_i, D_e pre-scaled (chi/Cm absorbed)
- FDM constructor: warns if chi ≠ 1.0, mass term = 1/dt
- RushLarsen: `V += dt·(-(I_ion+I_stim)/Cm)` with `Cm = getattr(state, 'Cm', 1.0)`
- `fft.py`: DEPRECATED, still uses Formulation A (not called by any solver)

**Monodomain V5.4** (Formulation A):
- `IsotropicTissue`: stores D, chi=1400, Cm=1.0 as separate fields
- FDM/FEM/FVM: chi·Cm in mass terms (`chi·Cm·I`, `chi·Cm·∫φ_iφ_j`, `chi·Cm·Vol`)
- DCT/FFT solvers: chi·Cm in spectral denominators
- RushLarsen: `V += dt·(-(I_ion + I_stim))` — NO Cm division
- V5.4 LBM: source = `-(I_ion+I_stim) / (chi·Cm)`

**LBM V1** (Formulation B):
- `sigma_to_D()`: only place chi appears in entire engine
- `tau_from_D()`: D has chi/Cm baked in, tau = 0.5 + D·dt/(cs²·dx²)
- `compute_source_term()`: `R = -(I_ion + I_stim) / Cm` — only Cm, NOT chi

### Diffusion tensor encoding across methods

| Method | What it receives | Dxx ≠ Dyy | Dxy ≠ 0 |
|--------|-----------------|:---------:|:-------:|
| FDM 5-point | Dxx, Dyy per node (harmonic mean) | Yes | No |
| FDM 9-point | Dxx, Dxy, Dyy per node | Yes | Yes |
| FDM Mehrstellen | Scalar D only (dx=dy required) | No | No |
| FEM P1 | D tensor per element | Yes | Yes |
| FVM (TPFA) | D per face | Yes | No |
| LBM D2Q5 BGK | Scalar D → single tau | No | No |
| LBM D2Q5 MRT | Dxx, Dyy → separate tau_x, tau_y | Yes | No |
| LBM D2Q9 MRT | Dxx, Dyy, Dxy → tau_x, tau_y, tau_xy (shear moment) | Yes | Yes |

D2Q5 cannot encode Dxy because it has no diagonal velocities → no p_xy shear stress moment. D2Q9 adds 4 diagonal velocities enabling the p_xy moment.

### Code duplication inventory

| Module | Copies | Notes |
|--------|--------|-------|
| IonicModel ABC + lut.py | 3 | Identical across all engines |
| TTP06 (5 files) | 3 | Identical |
| ORd (6 files) | 3 | Identical |
| PCG solver | 2 | Minor divergence (breakdown check, warm start) |
| Splitting strategies | 2 | Identical logic, different state type hints |
| Stimulus protocol | 2 | Bidomain uses += (accumulate), V5.4 uses = (overwrite) |
| StructuredGrid | 2 | Bidomain adds boundary_spec |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Formulation unification | **Keep both**, unify API | Converting V5.4 to B risks 77 tests; both correct |
| Shared code | `cardiac_core/` package | Extract ionic/mesh/stimulus; don't merge solver internals |
| ConductivityConfig | sigma → D in one place | chi/Cm appear only in ConductivityConfig |
| Simulation protocol | `Simulation` protocol + `create_simulation()` factory | Optimizer/Surrogate call any engine interchangeably |
| Target architecture | **Option B — unified core** | `cardiac_core/` owns shared code, engines import from it. Not a wrapper; the actual codebase. Eliminates `sys.modules` hack and code duplication. |
| Implementation order | API first, then extract | Phase 0 builds API as wrapper (low risk). Later phases move shared code in and rewire engine imports. |
| Canonical LBM | LBM V1 | More features: MRT, D2Q9, 3 BC types, compiled steps |
| FDM stencil (bidomain) | Face-based symmetric | SPD required for PCG on elliptic |
| Cm convention | Always 1.0 | All ionic models output pA/pF |
| Project structure | Research = writing, Engines = code | Experiments in engine `experiments/`, EXPERIMENT.md backlinks to research |

### Phase 0: API Layer (completed 2026-03-17)

`cardiac_core/` API currently wraps unchanged engines — engines still own their code, `cardiac_core` imports from them. The `_prepare_engine()` function flushes `sys.modules` to work around both engines using the `cardiac_sim` namespace. Eliminated in Phase 1 when shared code moves into `cardiac_core/`.

**File format** (`file_format.py`): `CardiacMeshData` dataclass + `save_cardiac_mesh()` / `load_cardiac_mesh()` / `create_cardiac_mesh()`. Format version 1 `.npz` stores grid, conductivity, stimulus regions, and optional bidomain sigma fields.

**API** (`api.py`): `monodomain()`, `bidomain()`, `lbm()` each accept `str | CardiacMeshData`, construct the engine, and return `CardiacSimulation` with a generator `.run()` interface yielding `SimulationSnapshot(t, V, phi_e, Nx, Ny, dx, dy)`. V is always (Nx, Ny) grid shape.

**Bidomain D_i/D_e derivation**: When file lacks sigma_i/sigma_e, derives from D_eff and sigma_ratio: `D_i = D_eff * (1+r)/r`, `D_e = D_eff * (1+r)`.

**34 tests** (10 file format, 6 monodomain, 6 LBM, 7 bidomain, 5 integration). Wrapper output matches direct engine construction exactly (verified with `torch.allclose` at atol=1e-10). All 93 V5.4 tests and 10 bidomain spot-check tests still pass.

### Migration plan (toward unified core)

| Phase | What | Status |
|-------|------|--------|
| Phase 0 | API layer + file format (wrapper, imports from engines) | **DONE** (2026-03-17) |
| Phase 1 | Move ionic models into `cardiac_core/ionic/` (one copy) | Not started |
| Phase 2 | Move mesh + stimulus into `cardiac_core/mesh/`, `cardiac_core/stimulus/` | Not started |
| Phase 3 | Add `ConductivityConfig` — sigma → D conversion in one place | Not started |
| Phase 4 | Rewire engines to import from `cardiac_core` (delete their copies) | Not started |
| Phase 5 | Remove `_prepare_engine()` hack, unify `cardiac_sim` namespace | Not started |
| Validation | All 149+ tests pass, no duplicated ionic/mesh/stimulus code | Not started |

## Open Questions

- Should V5.4 eventually convert to Formulation B? (Deferred, not blocking)
- V5.4 LBM source term `/(chi·Cm)` should switch to `/Cm` when using ConductivityConfig (chi absorbed into D)
- Stimulus overlap: V5.4 uses `=` (overwrite), Bidomain uses `+=` (accumulate) — which is correct?

## Connections
- **Engines**: All three + cardiac_core (target)
- **Related research**: All active questions depend on stable engines
- **Pipelines**: Optimizer V1 needs unified API for cross-engine validation (V2)
- **Key document**: [REVIEW.md §6](../../Bidomain/Engine_V1/REVIEW.md) — full technical proposal

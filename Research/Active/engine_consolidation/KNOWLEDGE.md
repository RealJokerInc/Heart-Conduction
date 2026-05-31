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
LBM/Engine_V1/                 ← imports cardiac_core.ionic
  collision/, streaming/       ← BGK/MRT, D2Q5/D2Q9 (unique to LBM)
```

**Current state**: Phase 0 complete (API wrapper imports FROM engines; `_prepare_engine()` flushes `sys.modules` for the shared `cardiac_sim` namespace — eliminated in Phase 1). **Prerequisite done (2026-05-30): Monodomain V5.5** — a Cm-correct fork of V5.4 (reaction `/Cm`, dead LBM path dropped) is now the canonical monodomain to consolidate; V5.4 frozen. See "V5.5 Cm-correct fork". The consolidation (Phases 1–5) has not started.

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
| Formulation unification | Keep V5.4 frozen; **fork V5.5** (Formulation-B reaction) | Converting V5.4 in place risked its 77 tests; instead forked V5.5 (2026-05-30) which divides the reaction by Cm. V5.5 is the Cm-correct monodomain to consolidate. See "V5.5 Cm-correct fork" below. |
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

### cardiac_core drift reconciled (2026-05-30)

Since Phase 0, `cardiac_core/` grew a **convenience/analysis layer** (all still wrapper-level — imports FROM engines, no shared-code packages yet). Current package = `file_format.py`, `api.py` **plus**:
- `run.py` — one-shot `run_monodomain/run_bidomain/run_lbm`, `simulate`, `SimulationResult` (call once → `(times, V)`, no generator).
- `analysis.py` — pure tensor analysis: `activation_time`, `conduction_velocity`, `apd_at`/`apd_map`, `dominant_frequency`, `wavefront_mask`, `phase_map`, `phase_singularities`, `restitution_curve`.
- `geometry.py` — mask/region/distance/fiber helpers (`circle_mask`, `annulus_mask`, `left_edge_mask`, `boundary_distance`, `fiber_field_transmural`, …).
- `io.py` — result `.npz` save/load (`save_result`/`load_result`).
- **Test count is now 77** (not 34) — the convenience layer is tested (`test_run`, `test_analysis`, `test_geometry`, `test_io`).

**Assessment:** the drift is benign and additive — it does NOT block consolidation Phase 1 (ionic extraction is orthogonal to these wrappers). The target architecture (ionic/mesh/stimulus/conductivity packages, engines importing from them) is still unstarted. Build Phase 1 against **V5.5**. Also fixed the stale `Engines/` symlink index: `Engines/cardiac_core` (was broken `../../cardiac_core` → `../cardiac_core`), restored `lbm_v1 → ../LBM/Engine_V1` (old target `../Monodomain/LBM_V1` was stale), added `monodomain_v5.5 → ../Monodomain/Engine_V5.5`.

### V5.5 Cm-correct fork (completed 2026-05-30)

Rather than convert V5.4 in place (risking its 77 tests), forked `Monodomain/Engine_V5.5` — a full copy of V5.4 with ONE functional change: the operator-split reaction divides by the tissue Cm, `dV = -(Iion + Istim)/state.Cm` (Formulation B, matching Bidomain V1 / LBM V1). V5.4 stays the frozen baseline.

- **Plumbing:** `SimulationState.Cm` (default 1.0) populated from `spatial.Cm` at construction (direct read, no `getattr` fallback → fails loud). All three schemes expose a `Cm` property; **FEM had to add `self._Cm`/`self._chi`** (it only baked them into `self.M`) — an audit-CRITICAL catch.
- **Diffusion untouched:** the `chi·Cm` mass term already handled arbitrary Cm; only the reaction was broken.
- **Dead LBM path removed:** V5.5's internal `cardiac_sim/simulation/lbm/` had zero importers/tests (boundary work uses the separate `LBM/Engine_V1`); deleted along with the now-dead `step_with_V`.
- **Validation** (`test_phase10_cm_scaling.py`): Cm=1 bit-identical to V5.4 (golden, max|dV|=0); exact 1/Cm reaction scaling to 3.55e-15; Bidomain V1 cross-check CV 54.35 vs 54.35 cm/s (Cm=1), 28.09 vs 27.77 cm/s (Cm=2, 1.1%).

**Physics correction (important):** there is NO Cm time-dilation invariant. Tissue Cm divides only the voltage update; gate time-constants and concentration rates carry no Cm. So scaling Cm changes AP **morphology**, not timescale: APD does NOT scale (218→292 ms at k=2, not 2×). CV *does* scale ~1/Cm, but by eikonal scaling (CV ∝ √(D_phys·upstroke_rate), both ∝1/Cm), not dilation. The original PLAN and both audit passes asserted the (false) dilation invariant; the empirical 0D test caught it. The fix was correct throughout — only the validation strategy was wrong.

### Migration plan (toward unified core)

| Phase | What | Status |
|-------|------|--------|
| Prereq | **Monodomain V5.5 fork** — Cm-correct reaction; drop dead LBM | **DONE** (2026-05-30) |
| Phase 0 | API layer + file format (wrapper, imports from engines) | **DONE** (2026-03-17) |
| Phase 1 (copy) | Canonical `cardiac_core/ionic/` superset copy + editable install + lazy `__init__` | **DONE** (2026-05-31, copy-only) |
| Phase 1 (migrate) | Rewire engines to import `cardiac_core.ionic` + delete local copies + migrate Surrogate/Optimizer consumers | **DEFERRED** — audit found big-bang deletion breaks engine tests/examples + active Surrogate datagen + Optimizer. Must be done per-consumer, repo-wide discovery, never delete out from under a live consumer. |
| Phase 2 | Move mesh + stimulus into `cardiac_core/mesh/`, `cardiac_core/stimulus/` | Not started |
| Phase 3 | Add `ConductivityConfig` — sigma → D conversion in one place | Not started |
| Phase 4 | Rewire engines to import from `cardiac_core` (delete their copies) | Not started |
| Phase 5 | Remove `_prepare_engine()` hack, unify `cardiac_sim` namespace | Not started |
| Validation | All 149+ tests pass, no duplicated ionic/mesh/stimulus code | Not started |

## Open Questions

- ~~Should V5.4 eventually convert to Formulation B?~~ **RESOLVED (2026-05-30):** done in the V5.5 fork (V5.4 stays frozen). See "V5.5 Cm-correct fork" above.
- ~~V5.4 LBM source term `/(chi·Cm)` should switch to `/Cm`~~ **MOOT for V5.5:** the dead internal LBM path was removed from V5.5. The canonical LBM (LBM V1) is already Formulation B. The `/(chi·Cm)` reconciliation only matters if cardiac_core ever revives a monodomain-LBM path under ConductivityConfig.
- When consolidating, build `cardiac_core` against **V5.5** (Cm-correct), not V5.4.
- Stimulus overlap: V5.4/V5.5 use `=` (overwrite), Bidomain uses `+=` (accumulate) — which is correct? (Still open.)

## Connections
- **Engines**: All three + cardiac_core (target)
- **Related research**: All active questions depend on stable engines
- **Pipelines**: Optimizer V1 needs unified API for cross-engine validation (V2)
- **Key document**: [REVIEW.md §6](../../Bidomain/Engine_V1/REVIEW.md) — full technical proposal

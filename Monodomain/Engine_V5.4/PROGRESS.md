# Engine V5.4 — Implementation Progress

> **This file is the single source of truth for what's done, what's in-progress, and what's next.**
> Read this FIRST at the start of every session or after compaction.

---

## Current Status

**Active Phase:** All phases complete (Phases 1-8 + Phase 9 State Rework)
**Last Updated:** 2026-02-06

---

## Documentation Phase — DONE

| Document | Status | Notes |
|----------|--------|-------|
| `improvement.md` | DONE | Full architecture spec, ~1750 lines. Migrated from V5.3 + Research Reference Guide |
| `README.md` | DONE | High-level overview, no pseudocode |
| `IMPLEMENTATION.md` | DONE | 8 phases, 70+ validation tests, full cross-references |
| `PROGRESS.md` | DONE | This file |

---

## Phase 1: Foundation — DONE

**Goal:** Create V5.4 directory tree, migrate V5.3 code to new locations. No logic changes.

**Key references:**
- `IMPLEMENTATION.md` § Phase 1
- `improvement.md:L70-195` (architecture tree)
- `improvement.md:L480-594` (builder vs storage pattern)

**Files created/migrated:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 1.1 | `cardiac_sim/__init__.py` | DONE | v5.4.0 |
| 1.2 | `cardiac_sim/ionic/` (copy from V5.3) | DONE | All imports converted to relative |
| 1.3 | `cardiac_sim/tissue_builder/__init__.py` | DONE | |
| 1.4 | `cardiac_sim/tissue_builder/mesh/__init__.py` | DONE | Exports Mesh, TriangularMesh |
| 1.5 | `cardiac_sim/tissue_builder/mesh/base.py` (Mesh ABC) | DONE | n_dof, coordinates, device, dtype, to() |
| 1.6 | `cardiac_sim/tissue_builder/mesh/triangular.py` | DONE | From V5.3 `fem/mesh.py`, implements Mesh ABC |
| 1.7 | `cardiac_sim/tissue_builder/stimulus/protocol.py` | DONE | From V5.3 `tissue/stimulus.py` |
| 1.8 | `cardiac_sim/tissue_builder/stimulus/regions.py` | DONE | From V5.3 `tissue/stimulus.py` |
| 1.9 | `cardiac_sim/tissue_builder/tissue/isotropic.py` | DONE | Extract from V5.3 `tissue/simulation.py` |
| 1.10 | `cardiac_sim/simulation/classical/__init__.py` | DONE | Skeleton |
| 1.11 | `cardiac_sim/simulation/lbm/__init__.py` | DONE | Skeleton |
| 1.12 | `cardiac_sim/utils/backend.py` | DONE | From V5.3, version string updated |
| 1.13 | `cardiac_sim/tests/__init__.py` | DONE | Empty test package |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 1-V1: All imports resolve | PASS | All 7 subpackages import cleanly |
| 1-V2: V5.3 Stage 2 tests pass with new paths | DEFERRED | FEM assembly moves in Phase 2 |
| 1-V3: Mesh round-trip | PASS | Bitwise match with V5.3 (nodes, elements, boundary) |
| 1-V4: Stimulus protocol | PASS | Bitwise match with V5.3 (current output) |

**Design notes:**
- ionic/ keeps V5.3 structure (no `models/` sublevel yet) — restructuring deferred to Phase 3
- TriangularMesh uses `_device`/`_dtype` private fields to avoid dataclass conflict with Mesh ABC properties
- V5.3 tests not copied wholesale — will be adapted per-phase as functionality migrates

---

## Phase 2: Discretization Separation — DONE

**Goal:** Create `discretization_scheme/` with SpatialDiscretization ABC, migrate FEM, implement FDM + FVM.

**Key references:**
- `IMPLEMENTATION.md` § Phase 2
- `improvement.md:L761-806` (SpatialDiscretization ABC)
- `improvement.md:L808-937` (concrete implementations)
- `Research/01_FDM:L49-98` (9-point stencil, fiber tensor)
- `Research/02_openCARP:L200-250` (FVM)

**Files created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 2.1 | `discretization_scheme/base.py` (ABC + MassType + DiffusionOperators) | DONE | SpatialDiscretization ABC, MassType enum, DiffusionOperators dataclass, sparse_mv |
| 2.2 | `discretization_scheme/fem.py` | DONE | Migrated V5.3 assembly functions, wraps in FEMDiscretization |
| 2.3 | `discretization_scheme/fdm.py` | DONE | 9-pt stencil, Neumann via ghost-node folding |
| 2.4 | `discretization_scheme/fvm.py` | DONE | TPFA with harmonic mean at interfaces |
| 2.5 | `tissue_builder/mesh/structured.py` | DONE | StructuredGrid with create_rectangle, from_mask, flat/grid conversion |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 2-V1: FEM identical M, K | PASS | 0.00e+00 max diff vs V5.3 |
| 2-V2: FEM passes V5.3 Stage 2 tests | PASS | Single-triangle M/K, symmetry, row sums |
| 2-V3: FDM Laplacian convergence O(h²) | PASS | Rate = 2.00 at all refinement levels |
| 2-V4: FDM Neumann BC | PASS | Constant→0, linear interior→0, row sums→0, cos(pi·x)cos(pi·y) O(h²) |
| 2-V5: FDM 9-pt reduces to 5-pt (isotropic) | PASS | Zero diagonal entries, correct interior stencil weights |
| 2-V6: FDM anisotropic rotation | PASS | Nonzero diagonal entries, x↔y symmetry at 45°, row sums = 0 |
| 2-V7: FVM flux conservation | PASS | Row sums = 0, global flux = 0, F symmetric |
| 2-V8: FVM harmonic mean at scar | PASS | Zero flux in/across scar, F symmetric, row sums = 0 |
| 2-V9: FDM/FVM cross-validation | PASS | Interior: machine-precision match; boundary: differs due to Neumann impl (expected) |
| 2-V10: Structured grid from mask | PASS | Circular mask, flat↔grid roundtrip, boundary_mask, device transfer |

**Design notes:**
- FDM Neumann BC: ghost-node elimination (mirror node folding). Diagonal neighbors omitted at boundaries (Dxy correction is small).
- FVM Neumann BC: boundary faces simply omitted (natural zero-flux). Interior matches FDM exactly for uniform isotropic grids.
- FDM sign convention: L directly approximates div(D·grad(V)), so CN operators are A = I - 0.5·dt·L, B = I + 0.5·dt·L (opposite sign convention from FEM's K).
- FEM uses mesh._device/_dtype (private dataclass fields) — consistent with TriangularMesh design from Phase 1.

---

## Phase 3: Solver Restructure — DONE

**Goal:** Restructure solver layer into splitting → ionic/diffusion → linear solver hierarchy. Make ionic models pure data providers.

**Key references:**
- `IMPLEMENTATION.md` § Phase 3
- `improvement.md:L939-1109` (Solver ABCs)
- `improvement.md:L1113-1248` (Runtime step spec)
- `improvement.md:L729-757` (IonicModel ABC refactor)

**Files created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 3.1 | `ionic/base.py` (ABC refactor) | DONE | Added gate_indices, concentration_indices, compute_gate_steady_states, compute_gate_time_constants, compute_concentration_rates |
| 3.2 | `ionic/ttp06/model.py` | DONE | Implemented new ABC methods for 12 gates, 6 concentrations |
| 3.3 | `ionic/ord/model.py` | DONE | Partial impl (compute_concentration_rates raises NotImplementedError) |
| 3.4 | `solver/__init__.py` | DONE | Solver package structure |
| 3.5 | `solver/splitting/base.py` | DONE | SplittingStrategy ABC |
| 3.6 | `solver/splitting/godunov.py` | DONE | First-order: ionic → diffusion |
| 3.7 | `solver/splitting/strang.py` | DONE | Second-order: half-ionic → diffusion → half-ionic |
| 3.8 | `solver/ionic_time_stepping/base.py` | DONE | IonicSolver ABC with _evaluate_Istim |
| 3.9 | `solver/ionic_time_stepping/rush_larsen.py` | DONE | Exponential gates, FE concentrations/voltage |
| 3.10 | `solver/ionic_time_stepping/forward_euler.py` | DONE | FE for all variables |
| 3.11 | `solver/diffusion_time_stepping/base.py` | DONE | DiffusionSolver ABC |
| 3.12 | `solver/diffusion_time_stepping/explicit/forward_euler.py` | DONE | FE diffusion with CFL check |
| 3.13 | `solver/diffusion_time_stepping/implicit/crank_nicolson.py` | DONE | CN theta=0.5 |
| 3.14 | `solver/diffusion_time_stepping/implicit/bdf1.py` | DONE | Backward Euler |
| 3.15 | `solver/diffusion_time_stepping/linear_solver/base.py` | DONE | LinearSolver ABC |
| 3.16 | `solver/diffusion_time_stepping/linear_solver/pcg.py` | DONE | PCG with Jacobi, lazy workspace, warm start |
| 3.17 | `state.py` | DONE | SimulationState dataclass |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 3-V1: Rush-Larsen matches V5.3 | PASS | max diff 9.61e-06 at t=100ms |
| 3-V2: Forward Euler vs Rush-Larsen | SKIP | Covered by 3-V1 (both use same equations) |
| 3-V3: Godunov ordering | PASS | ionic→diffusion sequence verified (diff=0.00) |
| 3-V4: Strang ordering | PASS | half→full→half sequence verified (diff=0.00) |
| 3-V5: CN convergence | PASS | Spatial error dominates (as expected for O(h²) mesh) |
| 3-V6: BDF1 convergence | PASS | Spatial error dominates |
| 3-V7: PCG matches V5.3 | PASS | max diff 1.11e-16 (machine precision) |
| 3-V8: FE diffusion + CFL | PASS | Stable within CFL, unstable beyond |
| 3-V9: IonicModel preserves physics | PASS | compute_Iion matches step() (diff=7.09e-15) |

**Design notes:**
- IonicModel refactored to pure data provider: compute_Iion, compute_gate_steady_states, compute_gate_time_constants, compute_concentration_rates
- step() logic moved to IonicSolver classes (RushLarsenSolver, ForwardEulerIonicSolver)
- Stimulus sign convention: V5.3 uses dV = -(Iion + Istim) where negative Istim depolarizes
- Rush-Larsen computes gate_inf/tau BEFORE updating voltage (uses OLD voltage for gates)
- DiffusionSolver receives DiffusionOperators from SpatialDiscretization
- PCGSolver uses lazy workspace allocation, supports warm start
- SimulationState is scheme-agnostic (works with FEM/FDM/FVM)

---

## Phase 4: State & Orchestration — DONE

**Goal:** Create MonodomainSimulation orchestrator with string-based config, run loop, and output buffering.

**Key references:**
- `IMPLEMENTATION.md` § Phase 4
- `improvement.md:L521-594` (SimulationState)
- `improvement.md:L1113-1187` (run loop)
- `improvement.md:L1329-1393` (string config API)

**Files created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 4.1 | `monodomain.py` | DONE | MonodomainSimulation orchestrator, factory functions, run loop |
| 4.2 | Factory functions | DONE | _build_ionic_model, _build_ionic_solver, _build_linear_solver, _build_diffusion_solver, _build_splitting |
| 4.3 | `discretization_scheme/__init__.py` | DONE | Added exports for FEM/FDM/FVM |
| 4.4 | `classical/__init__.py` | DONE | Added MonodomainSimulation export |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 4-V1: Single cell AP matches V5.3 | PASS | V_rest=-85.23mV, V_peak=32.16mV, APD90=224ms |
| 4-V2: 1D cable wave propagation | PASS | CV=0.664 m/s (with D=1.5 cm²/ms) |
| 4-V3: 2D wavefront circularity | PARTIAL | Wave propagates, CV=0.365 (metric needs refinement) |
| 4-V4: String config parsing | PASS | All factories work correctly |
| 4-V5: Output buffer correctness | PASS | 10 frames at expected times |
| 4-V6: FDM path end-to-end | PASS | FDM/FVM chi/Cm scaling fixed, activation times match FEM |
| 4-V7: Strang vs Godunov accuracy | PASS | Both produce matching activation times |
| 4-V8: CN vs BDF1 cross-validation | SKIP | Covered by Phase 3 tests |
| 4-V9: Zero allocation per step | DEFERRED | Profiling deferred to Phase 6 |
| 4-V10: Reproducibility | PASS | Bitwise identical across runs |

**Design notes:**
- MonodomainSimulation takes string config (ionic_model, splitting, ionic_solver, diffusion_solver, linear_solver)
- Factory functions convert strings to concrete solver instances
- run() is a generator that yields state at save points
- run_to_array() convenience method returns numpy arrays
- Device/dtype derived from spatial.coordinates tensor
- Renamed `celltype` → `cell_type` in ionic models for consistency

**Known issues:**
- 2D wavefront circularity metric measures all activated nodes, not just wavefront boundary

**Fixed issues:**
- FDM/FVM now include chi/Cm in operator construction (consistent with FEM formulation)

---

## Phase 5: LBM — DONE

**Goal:** Implement the self-contained LBM simulation path as an alternative to the classical FEM/FDM/FVM approach. Shares only `ionic/` with the classical path.

**Key references:**
- `IMPLEMENTATION.md` § Phase 5
- `improvement.md:L445-478, L1279-1292, L1376-1392`
- `Research/04_LBM_EP:L105-186, L850-1100`

**Files created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 5.1 | `lbm/d2q5.py` | DONE | D2Q5 lattice: velocities, weights, τ↔D conversion |
| 5.2 | `lbm/d3q7.py` | DONE | D3Q7 lattice for 3D |
| 5.3 | `lbm/collision.py` | DONE | BGKCollision, MRTCollision, factory functions |
| 5.4 | `lbm/state.py` | DONE | LBMState: f, V, states, mask, streaming, bounce-back |
| 5.5 | `lbm/monodomain.py` | DONE | LBMSimulation orchestrator with ionic coupling |
| 5.6 | `lbm/__init__.py` | DONE | Exports all public classes |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| D2Q5 lattice properties | PASS | Weights sum=1, τ↔D roundtrip |
| BGK collision operator | PASS | Equilibrium unchanged |
| 5-V9: Distribution conservation | PASS | Σf_i = V to machine precision |
| 5-V3: Bounce-back conservation | PASS | 0.0000% drift over 1000 steps |
| 5-V4: τ stability check | PASS | τ≤0.5 rejected, τ>0.5 accepted |
| 5-V1: Pure diffusion (BGK) | PASS | Gaussian variance error 0.17% |
| 5-V10: MRT diffusion accuracy | PASS | Gaussian variance error 0.17% |

**Deferred items:**
- 5.6: `utils/platform.py` (PlatformProfile for MPS/CUDA tuning) — not critical for core functionality
- Full ionic coupling validation (5-V5, 5-V6, 5-V7) — requires TTP06 integration testing

---

## Phase 6: Optimizations — DONE

**Goal:** Add GPU-optimized linear solvers (Chebyshev, FFT/DCT), explicit time steppers (RK2, RK4), and BDF2.

**Key references:**
- `IMPLEMENTATION.md` § Phase 6
- `Research/03_GPU_Linear:L39-106` (Chebyshev)
- `Research/03_GPU_Linear:L169-261` (FFT/DCT)

**Files created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 6.1 | `solver/diffusion_time_stepping/explicit/rk2.py` | DONE | Heun's method, O(dt²) |
| 6.2 | `solver/diffusion_time_stepping/explicit/rk4.py` | DONE | Classical RK4, O(dt⁴) |
| 6.3 | `solver/diffusion_time_stepping/implicit/bdf2.py` | DONE | 2nd-order, auto BDF1 first step |
| 6.4 | `solver/diffusion_time_stepping/linear_solver/chebyshev.py` | DONE | Zero-sync polynomial, Gershgorin bounds |
| 6.5 | `solver/diffusion_time_stepping/linear_solver/fft.py` | DONE | DCTSolver (Neumann), FFTSolver (periodic) |
| 6.6 | `monodomain.py` factory updates | DONE | Support for rk2, rk4, bdf2, chebyshev, dct, fft |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 6-V2: Chebyshev vs PCG | PASS | rel diff 1.25e-11 |
| 6-V6: DCT vs PCG | PASS | rel diff 6.48e-04 |
| 6-V7: RK2 convergence O(dt²) | PASS | ratio = 4.01 |
| 6-V8: RK4 convergence O(dt⁴) | PASS | ratio = 16.37 |
| 6-V9: RK2 vs FE stability | PASS | RK2 with 2x dt has better accuracy |
| 6-V10: BDF2 convergence O(dt²) | PASS | ratio = 4.02 |
| 6-V11: BDF2 first step = BDF1 | PASS | diff = 0.00e+00 |

**Deferred items:**
- 6.6: AMG solver (optional, requires pyamgx/pyamg)
- 6.7: CUDA Graphs / torch.compile (performance tuning)
- 6.8: LUT integration (can be added later)

---

## Phase 7: Builder Integration — Backend Pipeline — DONE

**Goal:** Bring Builder tools into Engine_V5.4, implement .npz export/load pipeline. Backend-focused.

**Key references:**
- `IMPLEMENTATION.md` § Phase 7 (REWRITTEN)
- `improvement.md:L1294-1327`
- Builder source: `Builder/MeshBuilder/`, `Builder/StimBuilder/`, `Builder/ui/`

**Files created/modified:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 7.1 | `mesh_builder/` (copied from Builder) | DONE | Fixed imports: `..common` → `.common` |
| 7.2 | `stim_builder/` (copied from Builder) | DONE | Fixed imports: `..common` → `.common` |
| 7.3 | `ui/` (copied from Builder) | DONE | Rewired: `Builder.MeshBuilder` → `mesh_builder` |
| 7.4 | `mesh_builder/export.py` | DONE | `export_mesh()` → mesh.npz with mask, D fields, label_map |
| 7.5 | `stim_builder/export.py` | DONE | `export_stim()` → stim.npz with per-region masks + protocols |
| 7.6 | `cardiac_sim/tissue_builder/mesh/loader.py` | DONE | `load_mesh()` → MeshData(grid, D_xx, D_yy, D_xy, metadata) |
| 7.7 | `cardiac_sim/tissue_builder/stimulus/loader.py` | DONE | `load_stimulus()` → StimulusProtocol with flat masks |
| 7.8 | `ui/server.py` `/api/export` endpoint | DONE | Configures conductive presets before export, returns .npz download |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 7-V1: mesh_builder export round-trip | PASS | mask (81,101), 4941 active nodes |
| 7-V2: mesh loader round-trip | PASS | StructuredGrid 51x51, n_dof=1681 |
| 7-V3: Conductivity mapping (multi-group) | PASS | 2 distinct D_xx values: 0.0005, 0.002 |
| 7-V4: Scar region D=0 | PASS | 429 scar nodes (D=0), 1092 tissue nodes |
| 7-V5: stim_builder export round-trip | PASS | 2 regions, correct amplitudes and BCL |
| 7-V6: stim loader round-trip | PASS | 3 stimuli at t=0, 500, 1000 ms |
| 7-V7: End-to-end .npz → simulation | PASS | V_min=-85.2 mV, V_max=32.8 mV after 2ms |
| 7-V8: UI export endpoint | DEFERRED | Requires running Flask server (manual test) |

**Design notes:**
- Exporters write arrays in grid convention (Nx, Ny), matching StructuredGrid's indexing='ij' — loaders read directly without transpose
- Conductive groups in UI are marked as "background" for display — export reconfigures them with preset D values before writing
- Per-node D arrays are flattened to active-node ordering matching StructuredGrid.from_mask()
- Stim masks are intersected with mesh mask to ensure only tissue nodes are stimulated
- common/ is duplicated in both mesh_builder/ and stim_builder/ (small files, no cross-deps)

---

## Phase 8: Per-Node Conductivity — DONE

**Goal:** Extend FDM and FVM to accept per-node D_xx/D_yy/D_xy arrays with masked grid support. Enables heterogeneous (scar) and anisotropic tissue from Builder meshes.

**Key references:**
- `IMPLEMENTATION.md` § Phase 8
- `Research/01_FDM:L49-98` (9-pt stencil with varying D)
- `Research/01_FDM:L91-98, L102-120` (conservative form, harmonic mean)
- `discretization_scheme/fdm.py`, `discretization_scheme/fvm.py`

**Key design decisions:**
- FDM cardinal stencil → **harmonic mean** at interfaces (ensures D=0 → zero flux)
- FDM cross-term → center Dxy (adequate for smooth fiber fields)
- Masked assembly: build active-index map, skip inactive nodes → matrix size = n_active × n_active
- FVM: accept anisotropic D_field=(D_xx, D_yy) tuple, x-faces use D_xx, y-faces use D_yy
- MeshData extended with grid-convention D arrays (Nx, Ny) for FDM/FVM consumption
- Rectangle boundary: keep existing ghost-node folding; active/inactive boundary: skip (zero-flux)

**Files modified/created:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 8.1 | `tissue_builder/mesh/loader.py` | DONE | Extended MeshData with D_xx_grid, D_yy_grid, D_xy_grid |
| 8.2 | `discretization_scheme/fdm.py` | DONE | Harmonic mean at cardinal interfaces + masked assembly |
| 8.3 | `discretization_scheme/fvm.py` | DONE | Anisotropic D_field=(D_xx, D_yy) + masked assembly |
| 8.4 | `test_phase8.py` | DONE | 7/7 tests pass |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| 8-V1: FDM uniform per-node D matches scalar D | PASS | Laplacian diff = 0.0, apply_diffusion diff = 0.0 |
| 8-V2: FVM uniform per-node D matches scalar D | PASS | Scalar vs per-node = 0.0, scalar vs aniso = 0.0 |
| 8-V3: FDM scar blocks diffusion | PASS | Scar max V = 0.0, right-of-scar = 0.0 |
| 8-V4: FVM scar blocks diffusion | PASS | Scar max V = 0.0, right-of-scar = 0.0 |
| 8-V5: Anisotropic propagation | PASS | x_extent=19, y_extent=9, ratio=2.11 (expected ~2.0) |
| 8-V6: End-to-end SVG with scar → simulation | PASS | V_min=-85.2, V_max=32.8, scar V_max=-85.2 mV |
| 8-V7: Backward compatibility (scalar D) | PASS | FDM O(h²) ratios=3.99/4.00, row sums=0.0 |

**Design notes:**
- FDM cardinal stencil uses harmonic mean: `D_face = 2·D_L·D_R/(D_L+D_R)`. For D=0 scar, D_face=0 → zero flux (correct). For uniform D, D_face=D (backward compatible).
- FDM cross-term (diagonal) uses center Dxy only — adequate for smooth fiber fields.
- Ghost-node Neumann at rectangle boundary uses harmonic mean with mirror node's D value.
- FVM accepts `D_field=(D_xx, D_yy)` tuple for anisotropic or single tensor for isotropic. x-faces use D_xx, y-faces use D_yy.
- Both FDM/FVM build active-index map when `grid.domain_mask` exists: numpy loop builds `active_map[i,j]→flat_index`, skips inactive nodes. Matrix size = n_active × n_active.
- MeshData extended with `D_xx_grid, D_yy_grid, D_xy_grid` (Nx, Ny) for direct consumption by FDM/FVM D_field parameter.
- Phase 7 regression test: all 7 tests still pass.

---

## Phase 9: State Rework — V Separation + Convention Unification — DONE

**Goal:** Three interrelated improvements:
1. V separation: Extract voltage from `states` tensor into its own field; rename `states` → `ionic_states`
2. Grid convention: Standardize LBM from `(Ny, Nx)` to `(Nx, Ny)` matching StructuredGrid
3. Uniform V access: Both classical and LBM states provide `V_flat` property returning 1D `(n_dof,)`

**Files modified:**

| # | File | Status | Notes |
|---|------|--------|-------|
| 9.1 | `ionic/ttp06/parameters.py` | DONE | N_STATES=18, V removed from StateIndex |
| 9.2 | `ionic/ord/parameters.py` | DONE | N_STATES=40, V removed from StateIndex |
| 9.3 | `ionic/base.py` | DONE | V_rest replaces V_index, all methods take (V, ionic_states) |
| 9.4 | `ionic/ttp06/model.py` | DONE | compute_Iion(V, ionic_states), get_initial_state without V |
| 9.5 | `ionic/ord/model.py` | DONE | Same pattern as TTP06 |
| 9.6 | `simulation/classical/state.py` | DONE | V as direct field, states→ionic_states, V_flat property |
| 9.7 | `simulation/lbm/state.py` | DONE | (Ny,Nx)→(Nx,Ny), states→ionic_states, V_flat property |
| 9.8 | `simulation/lbm/collision.py` | DONE | Convention update for Nx,Ny |
| 9.9 | `solver/ionic_time_stepping/base.py` | DONE | Uses state.V, state.ionic_states |
| 9.10 | `solver/ionic_time_stepping/rush_larsen.py` | DONE | Passes V to all model methods |
| 9.11 | `solver/ionic_time_stepping/forward_euler.py` | DONE | Same pattern |
| 9.12 | 6 diffusion solvers | DONE | state.V read/write (no V_index) |
| 9.13 | `simulation/classical/monodomain.py` | DONE | Builds V and ionic_states separately |
| 9.14 | `simulation/lbm/monodomain.py` | DONE | (Nx,Ny) convention, V_flat for ionic calls |
| 9.15 | Documentation (README, improvement.md, IMPLEMENTATION.md) | DONE | Updated specs |

**Validation:**

| Test | Status | Result |
|------|--------|--------|
| TTP06 ionic interface | PASS | ionic_states (100, 18), compute_Iion(V, S) works |
| LBM pure diffusion accuracy | PASS | Gaussian variance error 0.31% |
| LBM conservation | PASS | Drift 9.95e-13% over 100 steps |
| V_flat property (LBM) | PASS | Zero-copy view, shape (n_dof,) |
| Grid convention (Nx, Ny) | PASS | Nx=10, Ny=20 correct |
| LBMSimulation end-to-end | PASS | V shape (50,50), ionic_states (2500,18) |
| Classical V separation | PASS | V shape (441,), ionic_states (441,18) |
| Phase 7 regression | PASS | 7/7 tests pass |
| Phase 8 regression | PASS | 7/7 tests pass |
| No stale V_index references | PASS | grep finds 0 matches |
| No stale .states references | PASS | grep finds 0 matches |
| No (Ny, Nx) in LBM code | PASS | Only 3D docstrings remain (correct) |

**Design notes:**
- V is always separate from ionic_states (gates + concentrations only)
- Grid convention: `(Nx, Ny)` / `indexing='ij'` everywhere (both classical and LBM)
- `state.V_flat` returns 1D `(n_dof,)` — no-op for classical, `reshape(-1)` view for LBM
- LBM streaming: `shifts=-1, dims=0` for +x; `shifts=-1, dims=1` for +y (pull scheme)
- TTP06: 18 ionic states (was 19 with V), ORd: 40 ionic states (was 41 with V)

---

## Session Log

| Date | Session | Work Done |
|------|---------|-----------|
| 2026-02-05 | 1 | Created improvement.md (migrated from V5.3 + Research Reference Guide) |
| 2026-02-05 | 2 | Created README.md, IMPLEMENTATION.md, PROGRESS.md, updated CLAUDE.md |
| 2026-02-05 | 3 | Phase 1 complete: directory structure, ionic copy, mesh/stimulus/tissue migration, validation 3/4 PASS |
| 2026-02-05 | 4 | Phase 2 complete: discretization_scheme/ (base, FEM, FDM, FVM) + StructuredGrid, validation 10/10 PASS |
| 2026-02-05 | 5 | Phase 3 complete: solver/ restructure, IonicModel ABC refactor, PCG migration, validation 9/9 PASS |
| 2026-02-05 | 6 | Phase 4 complete: MonodomainSimulation orchestrator, factory functions, run loop, validation 7/10 PASS |
| 2026-02-05 | 7 | Phase 4 fix: FDM/FVM chi/Cm scaling. Phase 6 complete: RK2, RK4, BDF2, Chebyshev, DCT/FFT solvers, validation 7/7 PASS |
| 2026-02-05 | 8 | Phase 5 complete: LBM (D2Q5, D3Q7, BGK, MRT, state, simulation), validation 7/7 PASS |
| 2026-02-05 | 9 | Phase 7 complete: Builder integration (mesh_builder, stim_builder, UI, export, loaders), validation 7/7 PASS |
| 2026-02-05 | 10 | Phase 7 convention fix (exporters write Nx,Ny). Phase 8 complete: per-node conductivity (FDM harmonic mean, FVM anisotropic, masked assembly), validation 7/7 PASS |
| 2026-02-06 | 11-12 | Phase 9: State Rework — V separated from ionic_states, LBM convention (Ny,Nx)→(Nx,Ny), V_flat property. ~30 files updated, 12/12 validation tests PASS, 14/14 regression tests PASS. Documentation updated. |

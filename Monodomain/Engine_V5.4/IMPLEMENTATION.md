# Engine V5.4: Phase-by-Phase Implementation Guide

This document provides a concrete, ordered implementation plan for V5.4. Each phase lists action items, file placements, cross-references to design docs and research, parallel work opportunities, and validation criteria.

**Cross-Reference Key:**
- `improvement.md:L###` — Architecture specification (this repo)
- `V5.3/IMPLEMENTATION.md:L###` — Prior implementation log (validated, all 55+ tests PASS)
- `Research/01_FDM...md:L###` — FDM stencils and implementation research
- `Research/02_openCARP...md:L###` — openCARP architecture research
- `Research/03_GPU_Linear...md:L###` — GPU linear solver research
- `Research/04_LBM_EP...md:L###` — LBM-EP implementation research
- `Research/00_Research_Summary.md` — Research index and priority map

---

## Pre-Implementation: V5.3 Baseline

The V5.3 codebase is the starting point. Everything migrates forward; nothing is thrown away.

**V5.3 validated components (100% test pass rate):**

| Component | V5.3 Location | Tests | Status |
|-----------|--------------|-------|--------|
| IonicModel ABC + TTP06 + ORd | `ionic/` | Stage 1 (8 tests) + Stage 1.5 (13 tests) | PASS |
| FEM mesh + assembly | `fem/mesh.py`, `fem/assembly.py` | Stage 2 (9 tests) | PASS |
| PCG + Jacobi preconditioner | `solver/linear.py` | Stage 3 (7 tests) | PASS |
| CN + BDF1/BDF2 time stepping | `solver/time_stepping.py` | Stage 4 (8 tests) | PASS |
| MonodomainSimulation | `tissue/simulation.py` | Stage 5 (6 tests) | PASS |
| LUT optimization | `ionic/lut.py` | Stage 6 (6 tests) | PASS |
| Performance benchmarks | — | Stage 7 (6 tests) | PASS |

> Ref: `V5.3/IMPLEMENTATION.md:L1112-1797` for full test results.

---

## Phase 1: Foundation — Directory Structure & File Migration

**Goal:** Create the V5.4 directory tree and move existing V5.3 code into its new locations. No logic changes — pure reorganization.

### Action Items

| # | Task | Source (V5.3) | Destination (V5.4) |
|---|------|--------------|---------------------|
| 1.1 | Create top-level `cardiac_sim/` package | — | `cardiac_sim/__init__.py` |
| 1.2 | Copy `ionic/` wholesale | `ionic/` | `cardiac_sim/ionic/` |
| 1.3 | Create `tissue_builder/` structure | — | `cardiac_sim/tissue_builder/` |
| 1.4 | Create mesh subpackage | `fem/mesh.py` | `cardiac_sim/tissue_builder/mesh/` |
| 1.5 | Move triangular mesh | `fem/mesh.py` | `cardiac_sim/tissue_builder/mesh/triangular.py` |
| 1.6 | Create `mesh/base.py` ABC | — | `cardiac_sim/tissue_builder/mesh/base.py` |
| 1.7 | Create stimulus subpackage | `tissue/stimulus.py` | `cardiac_sim/tissue_builder/stimulus/` |
| 1.8 | Split stimulus into protocol + regions | `tissue/stimulus.py` | `cardiac_sim/tissue_builder/stimulus/protocol.py`, `regions.py` |
| 1.9 | Create tissue subpackage | — | `cardiac_sim/tissue_builder/tissue/` |
| 1.10 | Create `isotropic.py` from existing conductivity logic | `tissue/simulation.py` (conductivity parts) | `cardiac_sim/tissue_builder/tissue/isotropic.py` |
| 1.11 | Create `simulation/` skeleton | — | `cardiac_sim/simulation/` |
| 1.12 | Create `simulation/classical/` skeleton | — | `cardiac_sim/simulation/classical/` |
| 1.13 | Create `simulation/lbm/` skeleton (empty `__init__.py`) | — | `cardiac_sim/simulation/lbm/` |
| 1.14 | Create `utils/` | `utils/backend.py` | `cardiac_sim/utils/backend.py` |
| 1.15 | Copy tests | `tests/` | `cardiac_sim/tests/` |

### File Hierarchy After Phase 1

```
cardiac_sim/
├── __init__.py
├── ionic/                          # ← copied from V5.3 (unchanged)
│   ├── base.py
│   ├── lut.py
│   └── models/
│       ├── ord/
│       └── ttp06/
├── tissue_builder/
│   ├── __init__.py
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── base.py                 # ← NEW (Mesh ABC)
│   │   └── triangular.py           # ← from V5.3 fem/mesh.py
│   ├── tissue/
│   │   ├── __init__.py
│   │   └── isotropic.py            # ← extracted from V5.3 tissue/simulation.py
│   └── stimulus/
│       ├── __init__.py
│       ├── protocol.py             # ← from V5.3 tissue/stimulus.py
│       └── regions.py              # ← from V5.3 tissue/stimulus.py
├── simulation/
│   ├── __init__.py
│   ├── classical/
│   │   └── __init__.py             # ← skeleton
│   └── lbm/
│       └── __init__.py             # ← skeleton
├── utils/
│   ├── __init__.py
│   └── backend.py                  # ← from V5.3 utils/backend.py
└── tests/                          # ← copied from V5.3
```

### References

- Architecture tree: `improvement.md:L70-195`
- Builder vs Storage pattern: `improvement.md:L480-594`
- V5.3 mesh implementation: `V5.3/IMPLEMENTATION.md:L859-965` (Stage 2, FEM mesh)
- V5.3 stimulus: `V5.3/IMPLEMENTATION.md:L1021-1111` (MonodomainSimulation, stimulus handling)

### Parallel Work

Phase 1 is inherently sequential (directory creation), but three developers could work simultaneously:

| Track | Files | Independent? |
|-------|-------|-------------|
| A: Mesh migration | `mesh/base.py`, `mesh/triangular.py` | Yes |
| B: Stimulus split | `stimulus/protocol.py`, `stimulus/regions.py` | Yes |
| C: Tissue extraction | `tissue/isotropic.py` | Yes |

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 1-V1 | All imports resolve | `python -c "import cardiac_sim"` |
| 1-V2 | V5.3 Stage 2 tests pass with new paths | Run `test_stage2_fem.py` with updated imports |
| 1-V3 | Mesh round-trip | Create triangular mesh → verify `n_nodes`, `n_elements`, coordinates match V5.3 |
| 1-V4 | Stimulus protocol | Create StimulusProtocol → verify timing, amplitude, mask match V5.3 |

**Debug strategy:** If imports break, the issue is always a path problem. Use `grep -r "from fem" tests/` to find all old import paths that need updating.

---

## Phase 2: Discretization Separation — Spatial Operators

**Goal:** Create `discretization_scheme/` with the `SpatialDiscretization` ABC, migrate FEM assembly, and add FDM/FVM stubs.

### Action Items

| # | Task | Details |
|---|------|---------|
| 2.1 | Create `SpatialDiscretization` ABC | Abstract base with `n_dof`, `coordinates`, `mass_type`, `get_diffusion_operators()`, `apply_diffusion()` |
| 2.2 | Implement `FEMDiscretization` | Wrap existing V5.3 `fem/assembly.py` — mass matrix M, stiffness matrix K, operator construction |
| 2.3 | Implement `FDMDiscretization` | 9-point anisotropic stencil via sparse matrix or F.conv2d. Neumann BC via modified stencil |
| 2.4 | Implement `FVMDiscretization` | Cell-centered flux computation with harmonic mean at interfaces |
| 2.5 | Create `MassType` enum | `IDENTITY` (FDM), `DIAGONAL` (FVM), `SPARSE` (FEM) |
| 2.6 | Create `DiffusionOperators` dataclass | `A_lhs`, `B_rhs`, `apply_mass` — built by spatial, consumed by solver |
| 2.7 | Create `mesh/structured.py` | StructuredGrid: `(Nx, Ny, dx, dy)` + domain mask + fiber angle field |

### File Hierarchy

```
cardiac_sim/simulation/classical/
├── discretization_scheme/
│   ├── __init__.py
│   ├── base.py             # SpatialDiscretization ABC, MassType enum, DiffusionOperators
│   ├── fem.py              # FEMDiscretization (wraps V5.3 assembly.py)
│   ├── fdm.py              # FDMDiscretization (NEW)
│   └── fvm.py              # FVMDiscretization (NEW)

cardiac_sim/tissue_builder/mesh/
├── structured.py           # StructuredGrid (NEW — serves FDM/FVM/LBM)
```

### References

**ABC Design:**
- `SpatialDiscretization` ABC: `improvement.md:L761-806`
- `DiffusionOperators` dataclass: `improvement.md:L964-973`
- `MassType` enum: `improvement.md:L785-787`
- Key design — spatial vs temporal separation: `improvement.md:L229-301`

**FEM (migration from V5.3):**
- Concrete FEM spec: `improvement.md:L810-870`
- V5.3 FEM assembly code: `V5.3/IMPLEMENTATION.md:L909-965`
- V5.3 mass/stiffness matrices: `V5.3/IMPLEMENTATION.md:L861-907`
- V5.3 validation (9/9 PASS): `V5.3/IMPLEMENTATION.md:L1251-1335`

**FDM (new implementation):**
- Concrete FDM spec: `improvement.md:L871-900`
- 9-point stencil coefficients: `Research/01_FDM:L49-76`
  ```
  NW = +Dxy/(4·dx·dy)    N = Dyy/dy²     NE = -Dxy/(4·dx·dy)
  W  = Dxx/dx²            C = -(sum)       E  = Dxx/dx²
  SW = -Dxy/(4·dx·dy)    S = Dyy/dy²     SE = +Dxy/(4·dx·dy)
  ```
- Diffusion tensor from fiber angle: `Research/01_FDM:L91-98`
  ```python
  Dxx = D_fibre * cos²θ + D_cross * sin²θ
  Dyy = D_fibre * sin²θ + D_cross * cos²θ
  Dxy = (D_fibre - D_cross) * sinθ * cosθ
  ```
- M-matrix constraint: `Research/01_FDM:L80` — `|Dxy| ≤ min(Dxx·dy/(2·dx), Dyy·dx/(2·dy))`
- Neumann BC (modified stencil, preferred): `Research/01_FDM:L132-134`
- Sparse matrix assembly (COO→CSR): `Research/01_FDM:L140-161`
- CFL stability: `Research/01_FDM:L67` — `dt ≤ Cm·h²/(4·D_max)`

**FVM (new implementation):**
- Concrete FVM spec: `improvement.md:L901-937`
- Cell-centered FVM: `Research/02_openCARP:L200-250`
- Harmonic mean for interface conductivity: `Research/00_Research_Summary:L90`
  ```
  D_face = 2·D_left·D_right / (D_left + D_right)  → gives 0 at scar boundary
  ```
- Reference implementation: `Research/code_examples/MonoAlg3D_C/` (C/CUDA cell-centered FVM)

**Structured Grid:**
- Spec: `Research/00_Research_Summary:L82-83`
- Stores: `(Nx, Ny, dx, dy)` + domain mask (bool tensor) + fiber angle field

### Parallel Work

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: ABC + FEM migration | `base.py`, `fem.py` | None (uses existing V5.3 code) |
| B: FDM | `fdm.py`, `mesh/structured.py` | `base.py` (ABC interface only — agree on signatures first) |
| C: FVM | `fvm.py` | `base.py` + `mesh/structured.py` (share structured grid with track B) |

Tracks B and C both need `mesh/structured.py`. Either assign structured grid to Track B (FDM needs it first), or define the interface up front and implement in parallel.

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 2-V1 | FEM produces identical M, K | Compare `FEMDiscretization.get_diffusion_operators()` output against V5.3 `assembly.py` output matrix-by-matrix (max abs diff < 1e-12) |
| 2-V2 | FEM passes all V5.3 Stage 2 tests | Run V5.3 `test_stage2_fem.py` through new interface |
| 2-V3 | FDM Laplacian convergence | Apply FDM Laplacian to `V(x,y) = sin(πx)·sin(πy)` on unit square. Compare against analytic `-2π²·V`. L2 error should decrease as O(h²) |
| 2-V4 | FDM Neumann BC | Constant V field → Laplacian = 0 everywhere (including boundaries). Linear V field → Laplacian = 0 interior, zero-flux at boundary |
| 2-V5 | FDM 9-point stencil symmetry | For isotropic case (Dxy=0), 9-point stencil must reduce to standard 5-point. Verify by comparing outputs |
| 2-V6 | FDM anisotropic rotation | Rotate fibers 45° → off-diagonal terms appear. Verify diffusion is faster along fiber direction by applying Gaussian pulse and measuring spread ratio |
| 2-V7 | FVM flux conservation | Sum of all fluxes across domain = 0 (conservation). Apply constant V → zero flux |
| 2-V8 | FVM harmonic mean | At scar boundary (D_right=0), face conductivity = 0. Verify zero flux into scar region |
| 2-V9 | FDM/FVM cross-validation | On isotropic structured grid, FDM and FVM Laplacians should agree to O(h²) |
| 2-V10 | Structured grid from image mask | Create StructuredGrid from binary mask → verify n_dof, coordinates, boundary nodes |

**Debug strategy for FDM:** The most common bug is sign errors in the 9-point stencil. Test isotropic case first (Dxy=0, reduces to 5-point), then add anisotropy. Check that the stencil row sums to zero (conservation).

**Debug strategy for FVM:** The most common bug is arithmetic vs harmonic mean at interfaces. Use a test with a sharp conductivity discontinuity — arithmetic mean gives D/2 at scar boundary (unphysical leakage), harmonic mean gives 0 (correct).

---

## Phase 3: Solver Restructure — Time Stepping Hierarchy

**Goal:** Restructure the solver layer into the splitting → ionic/diffusion → linear solver hierarchy. Extract `step()` logic from ionic models into solvers. Ionic models become pure data providers.

### Action Items

| # | Task | Details |
|---|------|---------|
| 3.1 | Create `SplittingStrategy` ABC | `step(state, dt)` with `GodunovSplitting` and `StrangSplitting` |
| 3.2 | Create `IonicSolver` ABC | `step(state, dt)` — owns `IonicModel`, evaluates Istim from state |
| 3.3 | Implement `RushLarsenSolver` | Exponential integrator for gating + Forward Euler for concentrations |
| 3.4 | Implement `ForwardEulerIonicSolver` | Simple explicit for all ionic ODEs |
| 3.5 | Create `DiffusionSolver` ABC | `step(state, dt)` — owns `DiffusionOperators`, optionally owns `LinearSolver` |
| 3.6 | Implement `CrankNicolsonSolver` | Implicit, 2nd order — migrate from V5.3 `solver/time_stepping.py` |
| 3.7 | Implement `BDF1Solver` | Implicit, 1st order — migrate from V5.3 |
| 3.8 | Implement `ForwardEulerDiffusionSolver` | Explicit, no linear solve — `V += dt·L·V` |
| 3.9 | Create `LinearSolver` ABC | `solve(A, b) → x`, owns workspace buffers |
| 3.10 | Migrate `PCGSolver` | From V5.3 `solver/linear.py` — add lazy workspace allocation |
| 3.11 | Refactor `IonicModel` ABC | Remove `step()` method, keep only `compute_*(V, ionic_states)` functions. V is separate from ionic_states. |
| 3.12 | Refactor TTP06 model | Remove stepping logic, expose `compute_Iion(V, ionic_states)`, `compute_gate_steady_states(V, ionic_states)`, `compute_gate_time_constants(V, ionic_states)`, `compute_concentration_rates(V, ionic_states)` |
| 3.13 | Refactor ORd model | Same as TTP06 |

### File Hierarchy

```
cardiac_sim/simulation/classical/solver/
├── __init__.py
├── splitting/
│   ├── __init__.py
│   ├── godunov.py              # GodunovSplitting
│   └── strang.py               # StrangSplitting
├── ionic_time_stepping/
│   ├── __init__.py
│   ├── rush_larsen.py          # RushLarsenSolver (primary)
│   └── forward_euler.py        # ForwardEulerIonicSolver
└── diffusion_time_stepping/
    ├── __init__.py
    ├── explicit/
    │   ├── __init__.py
    │   └── forward_euler.py    # ForwardEulerDiffusionSolver
    ├── implicit/
    │   ├── __init__.py
    │   ├── crank_nicolson.py   # CrankNicolsonSolver
    │   └── bdf1.py             # BDF1Solver
    └── linear_solver/
        ├── __init__.py
        ├── base.py             # LinearSolver ABC
        └── pcg.py              # PCGSolver (from V5.3)
```

### References

**Solver ABCs:**
- Ownership chain: `improvement.md:L939-963`
- `SplittingStrategy` ABC: `improvement.md:L975-1004`
- `IonicSolver` ABC: `improvement.md:L1006-1031`
- `DiffusionSolver` ABC: `improvement.md:L1033-1074`
- `LinearSolver` ABC: `improvement.md:L1076-1109`
- Runtime step spec: `improvement.md:L1113-1248`

**Rush-Larsen Implementation:**
- Detailed step spec: `improvement.md:L1214-1248`
- Rush-Larsen theory: `V5.3/IMPLEMENTATION.md:L620-638`
  ```
  Gates:    x_new = x_inf - (x_inf - x_old) · exp(-dt/tau)
  Conc:     c_new = c_old + dt · dc/dt
  Voltage:  V_new = V_old + dt/Cm · (-Iion + Istim/chi)
  ```
- openCARP Rush-Larsen analysis: `Research/02_openCARP:L300-350`

**IonicModel Refactor:**
- New data-provider interface: `improvement.md:L729-757`
- Design principle "models are data providers": `improvement.md:L666-678`
- V5.3 current IonicModel: `V5.3/IMPLEMENTATION.md:L154-215`
- V5.3 TTP06 model code: `V5.3/IMPLEMENTATION.md:L273-585`
- V5.3 ORd model structure: `V5.3/IMPLEMENTATION.md:L217-271`

**Time Stepping (migration):**
- V5.3 CN implementation: `V5.3/IMPLEMENTATION.md:L759-778`
- V5.3 BDF1/BDF2: `V5.3/IMPLEMENTATION.md:L780-825`
- V5.3 convergence order validation: `V5.3/IMPLEMENTATION.md:L1409-1520` (Stage 4, 8/8 PASS)
- Diffusion time stepping hierarchy: `improvement.md:L612-632`

**PCG (migration):**
- V5.3 PCG code: `V5.3/IMPLEMENTATION.md:L966-1019`
- V5.3 validation: `V5.3/IMPLEMENTATION.md:L1336-1408` (Stage 3, 7/7 PASS)

**Forward Euler Diffusion:**
- CFL constraint: `Research/01_FDM:L67` — `dt ≤ Cm·h²/(4·D_max)`
- Typical cardiac parameters: `Research/00_Research_Summary:L166-176` — dt=0.01ms, h=0.025cm → CFL=0.064 (safe)

### Parallel Work

This phase has the most parallelism opportunity:

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: Splitting strategies | `godunov.py`, `strang.py` | IonicSolver + DiffusionSolver interfaces (ABCs only) |
| B: Ionic solvers + model refactor | `rush_larsen.py`, `forward_euler.py` (ionic), refactored `ionic/base.py` | None (self-contained) |
| C: Diffusion solvers | `crank_nicolson.py`, `bdf1.py`, `forward_euler.py` (diffusion) | `base.py` (DiffusionSolver ABC), Phase 2 operators |
| D: Linear solver migration | `pcg.py` | None (self-contained) |

**Critical path:** Track B (ionic model refactor) is the riskiest because it changes an existing validated interface. Do this first and re-validate against V5.3 Stage 1 tests before other tracks depend on it.

**Strategy for ionic model refactor:**
1. Add new `compute_*()` methods alongside existing `step()` (backward compatible)
2. Verify new methods produce same outputs
3. Build `RushLarsenSolver` that calls new methods
4. Verify `RushLarsenSolver` matches old `step()` output
5. Only then remove `step()` from model

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 3-V1 | Rush-Larsen matches V5.3 | Single cell 1000ms: compare V trace from `RushLarsenSolver(model)` vs V5.3 `model.step()`. Max diff < 0.01 mV |
| 3-V2 | Forward Euler ionic matches | Same as 3-V1 but with `ForwardEulerIonicSolver` at dt=0.001ms (10x smaller for stability) |
| 3-V3 | Godunov = ionic then diffusion | Step once with Godunov → verify identical to calling ionic.step() then diffusion.step() manually |
| 3-V4 | Strang = half-ionic, full-diffusion, half-ionic | Same verification pattern |
| 3-V5 | CN reproduces V5.3 Stage 4 | Richardson extrapolation → O(dt²) convergence. Compare ratios against V5.3 results (4.2: ratios = 0.250, 0.250) |
| 3-V6 | BDF1 convergence order | Richardson extrapolation → O(dt). Compare against V5.3 (4.3: ratios = 0.510, 0.505) |
| 3-V7 | PCG matches V5.3 | Solve same system → diff from scipy < 1e-12. Iteration count within ±1 of V5.3 |
| 3-V8 | Forward Euler diffusion | Heat equation on unit square: analytic decay rate matches numerical. Verify CFL violation causes instability |
| 3-V9 | IonicModel refactor preserves physics | `model.compute_Iion(V, ionic_states)` must match V5.3 ionic current for same input states (bitwise on same device) |
| 3-V10 | Strang vs Godunov | Strang should show O(dt²) splitting error vs Godunov O(dt). Measure by running both at decreasing dt and comparing against reference solution |

**Debug strategy for ionic model refactor:** The critical invariant is that `compute_Iion(V, ionic_states)` returns the **exact same tensor** as the current V5.3 code computes internally. Test this by injecting a print/assert inside the existing V5.3 `step()` method, capturing Iion, then comparing against the new standalone `compute_Iion(V, ionic_states)`.

**Debug strategy for splitting:** If Strang doesn't show 2nd-order convergence, the bug is almost always in the half-step: either `dt/2` isn't being passed correctly, or the ionic solver's internal state update is not idempotent for split steps.

---

## Phase 4: State & Orchestration — SimulationState + MonodomainSimulation

**Goal:** Create the unified `SimulationState` dataclass and the `MonodomainSimulation` orchestrator that wires all components together via string-based config.

### Action Items

| # | Task | Details |
|---|------|---------|
| 4.1 | Create `SimulationState` dataclass | Scheme-agnostic: `V`, `ionic_states`, `gate_indices`, `concentration_indices`, stimulus data, output buffer, spatial reference |
| 4.2 | Create `MonodomainSimulation` orchestrator | String-based config → construct spatial discretization, splitting strategy, ionic/diffusion solvers, linear solver |
| 4.3 | Implement solver factory | Parse config strings: `"CN+PCG"`, `"FE"`, `"BDF1+Chebyshev"` → instantiate correct solver chain |
| 4.4 | Implement output buffering | GPU-side buffer, copy to CPU only at save intervals |
| 4.5 | Implement run loop | Generator-based: `yield` at save points, handle stimulus timing |
| 4.6 | Wire stimulus evaluation | IonicSolver reads stimulus from state, evaluates timing internally |

### File Hierarchy

```
cardiac_sim/simulation/classical/
├── state.py                # SimulationState dataclass
└── monodomain.py           # MonodomainSimulation orchestrator
```

### References

**SimulationState:**
- Full dataclass spec: `improvement.md:L521-594`
- What moved out of state (into solvers): `improvement.md:L560-567`
- Scheme-agnostic design: `improvement.md:L575-582`

**MonodomainSimulation:**
- Run loop spec: `improvement.md:L1113-1187`
- Splitting step (Strang example): `improvement.md:L1189-1212`
- V5.3 MonodomainSimulation: `V5.3/IMPLEMENTATION.md:L1021-1111`

**User API (string config):**
- Config examples: `improvement.md:L1329-1393`
  ```python
  sim = MonodomainSimulation(
      mesh=mesh, tissue=tissue, stimulus=stim,
      ionic_model="ttp06", cell_type="EPI",
      spatial="fem",
      diffusion="CN", linear_solver="PCG",
      splitting="strang", ionic_stepping="rush_larsen",
      dt=0.02, save_interval=1.0,
  )
  for frame in sim.run(duration=500.0):
      process(frame)
  ```

**GPU Optimization:**
- Output buffering strategy: `improvement.md:L1394-1430`
- Platform-specific notes: `improvement.md:L1417-1430`

### Parallel Work

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: SimulationState | `state.py` | Phase 2 (SpatialDiscretization interface), Phase 3 (solver interfaces) |
| B: MonodomainSimulation | `monodomain.py` | `state.py`, all Phase 2-3 components |

Phase 4 is mostly serial — `monodomain.py` depends on everything. However, `state.py` can be written and tested independently with mock spatial/solver objects.

**Strategy:** Write `state.py` first, test it with mock objects, then write `monodomain.py` that wires real components.

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 4-V1 | Single cell AP matches V5.3 | FEM + TTP06 + CN + PCG + Godunov: Vrest ≈ -85 mV, Vpeak ≈ 32 mV, APD90 ≈ 224 ms. Compare against V5.3 Stage 5 test 5.1 |
| 4-V2 | 1D cable wave propagation | Stimulus at left end, measure activation at 3 positions. CV in 0.3-1.0 m/s range. Compare against V5.3 test 5.2-5.3 |
| 4-V3 | 2D wavefront circularity | Isotropic 2D sheet, center stimulus. Wavefront R² > 0.95. Compare against V5.3 test 5.6 |
| 4-V4 | String config parsing | `"CN+PCG"` → CrankNicolsonSolver with PCGSolver. `"FE"` → ForwardEulerDiffusionSolver (no linear solver). Verify correct types instantiated |
| 4-V5 | Output buffer correctness | Run 100 steps, save every 10th. Verify buffer contains exactly 10 frames with correct time stamps |
| 4-V6 | FDM path end-to-end | StructuredGrid + FDM + ForwardEuler + TTP06: single cell AP. Compare against FEM result (should agree within 5% for APD90 on fine grid) |
| 4-V7 | Strang vs Godunov accuracy | Run same tissue simulation with both. Strang should produce smaller splitting error at large dt (compare activation times) |
| 4-V8 | CN vs BDF2 cross-validation | Same as V5.3 test 5.5: both produce matching LAT, CV |
| 4-V9 | Zero allocation per step | Profile 100 steps with `torch.cuda.memory_stats()`. Verify no new allocations after warmup step |
| 4-V10 | Reproducibility | Same config → bitwise identical output (fixed seed). Reset and re-run → identical |

**Debug strategy:** The most common integration bug is shape mismatches between solver outputs. Add shape assertions at every interface boundary:
```
assert state.V.shape == (state.n_dof,), f"V shape {state.V.shape}, expected ({state.n_dof},)"
```

**Critical regression test:** V5.3 Stage 5 results are the gold standard. The V5.4 FEM path must reproduce them exactly (same config → same numbers to machine precision).

---

## Phase 5: LBM — Lattice-Boltzmann Method

**Goal:** Implement the self-contained LBM simulation path as an alternative to the classical FEM/FDM/FVM approach. Shares only `ionic/` with the classical path.

### Action Items

| # | Task | Details |
|---|------|---------|
| 5.1 | Create LBM state | Distribution tensor `f: (Q, Nx, Ny)`, voltage V `(Nx, Ny)`, ionic_states, domain mask |
| 5.2 | Implement D2Q5 lattice | Velocity vectors, weights, `c_s²`, opposite directions |
| 5.3 | Implement D3Q7 lattice | 3D extension |
| 5.4 | Implement BGK collision | Single relaxation time: `f* = f - (1/τ)(f - f_eq) + dt·w·S` |
| 5.5 | Implement MRT collision | Multi-relaxation with transformation matrix M, anisotropic diffusion via direction-dependent relaxation |
| 5.6 | Implement streaming | `torch.roll()` per direction for periodic domains |
| 5.7 | Implement bounce-back BC | No-flux: incoming = outgoing at domain boundary. Use domain mask |
| 5.8 | Implement voltage recovery | `V = Σ fᵢ` (macroscopic variable from distributions) |
| 5.9 | Implement ionic coupling | Source term in collision from `compute_Iion()`. NOT operator splitting — embedded in collision |
| 5.10 | Create LBM simulation orchestrator | `LBMSimulation.run()` — collide → stream → bounce-back → recover → ionic update |
| 5.11 | Create `utils/platform.py` | `PlatformProfile` for Apple Silicon (MPS) vs CUDA vs CPU tuning |

### File Hierarchy

```
cardiac_sim/simulation/lbm/
├── __init__.py
├── state.py                # LBMState: f, v, gating, mask
├── monodomain.py           # LBMSimulation orchestrator
├── collision.py            # BGKCollision, MRTCollision
├── d2q5.py                 # 2D lattice (4 moving + 1 rest)
└── d3q7.py                 # 3D lattice (6 moving + 1 rest)

cardiac_sim/utils/
├── platform.py             # PlatformProfile (NEW)
```

### References

**Architecture:**
- LBM paradigm overview: `improvement.md:L445-478`
- LBM independence: `improvement.md:L1477-1482`
- LBM component dependency: `improvement.md:L1279-1292`
- LBM user API: `improvement.md:L1376-1392`

**D2Q5 Lattice:**
- Velocity vectors and weights: `Research/04_LBM_EP:L105-125`
  ```
  e₀=(0,0), e₁=(1,0), e₂=(-1,0), e₃=(0,1), e₄=(0,-1)
  w₀=1/3, w₁₋₄=1/6
  ```
- Diffusion-relaxation relation: `Research/04_LBM_EP:L120-123`
  ```
  τ = 0.5 + 3·D·dt/dx²
  ```

**Collision Operators:**
- BGK: `Research/04_LBM_EP:L134-149`
- MRT (recommended for anisotropy): `Research/04_LBM_EP:L151-186`
  - Transformation matrix M: `Research/04_LBM_EP:L162-170`
  - Relaxation rates from diffusion tensor: `Research/04_LBM_EP:L172-175`
    ```
    τ_ij = δ_ij/2 + 4·D_ij·dt/dx²
    ```

**Streaming:**
- `torch.roll()` implementation: `Research/04_LBM_EP:L873-897`

**Bounce-Back BC:**
- Physics (Neumann condition): `Research/04_LBM_EP:L289-349`
- Mask-based implementation: `Research/04_LBM_EP:L899-934`
  ```python
  mask_px = torch.roll(mask, shifts=-1, dims=0)  # (Nx, Ny): x = dim 0
  bounce_px = (mask > 0) & (mask_px == 0)
  f[2][bounce_px] = f[1][bounce_px]  # +x → -x
  ```

**Stability:**
- τ > 0.5 required: `Research/00_Research_Summary:L179-181`
- Typical cardiac: τ_fiber = 0.548, τ_cross = 0.512 (both stable)

**Reference Implementations:**
- PyTorch LBM framework: `Research/code_examples/lettuce/` (BGK/MRT, D2Q9/D3Q19 patterns)
- Full LBM-EP PyTorch blueprint (~300 lines): `Research/04_LBM_EP:L850-1100`
- LBM-EP paper: Rapaka et al., MICCAI 2012

### Parallel Work

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: Lattice definitions | `d2q5.py`, `d3q7.py` | None (pure data) |
| B: Collision operators | `collision.py` | Lattice definitions (interfaces only) |
| C: State + orchestrator | `state.py`, `monodomain.py` | A, B, `ionic/` |
| D: Platform utilities | `utils/platform.py` | None |

Tracks A and D can start immediately. Track B needs lattice interfaces. Track C integrates everything.

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 5-V1 | Pure diffusion (no ionic) | Initialize Gaussian pulse, run LBM diffusion only. Compare against analytic Gaussian spreading: `σ²(t) = σ²(0) + 2Dt`. Verify D matches configured value within 1% |
| 5-V2 | Anisotropic diffusion | Fiber at 45°, Gaussian pulse. Spread should be 4:1 (fiber:cross) ratio. MRT should handle this; BGK should show errors |
| 5-V3 | Bounce-back no-flux | Rectangular domain with walls. Total voltage (sum of all V) must be conserved (< 0.01% drift over 1000 steps) |
| 5-V4 | τ stability check | Set D too high so τ < 0.5 → should warn/error. Set D at valid range → stable propagation |
| 5-V5 | Single cell AP (LBM) | 1×1 domain (no diffusion), couple with TTP06. AP shape must match classical single-cell within 1% APD90 |
| 5-V6 | 1D cable CV (LBM vs classical) | 200×1 cable, same conductivity. LBM CV should match FDM/FEM CV within 5% |
| 5-V7 | 2D wavefront (LBM) | Isotropic 2D sheet, center stimulus. Circular wavefront (R² > 0.95), CV matches classical |
| 5-V8 | LBM performance | Measure steps/sec for 200×200 grid. LBM should be faster than FEM (no linear solve). Target: 5-10x on GPU |
| 5-V9 | Distribution conservation | `sum(f_i)` over all directions at each cell should equal voltage V. Verify to machine precision after each step |
| 5-V10 | MRT vs BGK isotropic | On isotropic domain, MRT and BGK should produce identical results. Max diff < 1e-10 |

**Debug strategy for LBM:** The most common bugs are:
1. **Wrong streaming direction** — distributions shift the wrong way. Test: place a delta function in one distribution, stream once, check it moved to the correct neighbor.
2. **Bounce-back applied to wrong direction** — f_incoming and f_outgoing swapped. Test: 1D domain with walls, check voltage profile is symmetric.
3. **Source term sign** — ionic current should depolarize (increase V), check sign convention.
4. **τ calculation** — off-by-one in the `0.5 + 3D·dt/dx²` formula. Verify by measuring effective diffusivity from Gaussian spread test.

---

## Phase 6: Optimizations — Advanced Solvers & Performance

**Goal:** Add GPU-optimized linear solvers (Chebyshev, FFT/DCT), explicit time steppers (RK2, RK4), and performance tuning.

### Action Items

| # | Task | Details |
|---|------|---------|
| 6.1 | Implement Chebyshev linear solver | Zero-sync polynomial iteration. Needs eigenvalue bounds (Gershgorin or power iteration) |
| 6.2 | Implement FFT/DCT solver | Direct spectral solve for structured grids with Neumann BC. O(N log N) |
| 6.3 | Implement RK2 (Heun's method) | 2nd-order explicit diffusion. Two Laplacian evaluations per step |
| 6.4 | Implement RK4 | 4th-order explicit diffusion. Four Laplacian evaluations. Reference solution generator |
| 6.5 | Implement BDF2 diffusion solver | 2nd-order implicit. Two-step method, needs BDF1 for first step |
| 6.6 | Add AMG solver interface | Optional — via `pyamgx` (GPU) or `pyamg` (CPU). Setup once, reuse hierarchy |
| 6.7 | Add CUDA Graphs / torch.compile | Wrap hot loop in CUDA graph for reduced launch overhead |
| 6.8 | Add LUT integration with new solver chain | Wire `ionic/lut.py` into `RushLarsenSolver` for accelerated gate lookups |

### File Hierarchy

```
cardiac_sim/simulation/classical/solver/
├── diffusion_time_stepping/
│   ├── explicit/
│   │   ├── rk2.py              # Heun's method (NEW)
│   │   └── rk4.py              # Classical RK4 (NEW)
│   ├── implicit/
│   │   └── bdf2.py             # BDF2 (NEW)
│   └── linear_solver/
│       ├── chebyshev.py        # ChebyshevSolver (NEW)
│       ├── fft.py              # FFTSolver / DCTSolver (NEW)
│       └── amg.py              # AMGSolver wrapper (NEW, optional)
```

### References

**Chebyshev Solver:**
- Algorithm (3-term recurrence, zero sync): `Research/03_GPU_Linear:L39-65`
  ```
  θ = (λ_max + λ_min) / 2
  δ = (λ_max - λ_min) / 2
  d = z / θ;  x += d
  for k in 1..maxiter:
      ρ_new = 1 / (2σ - ρ)
      d = ρ_new·ρ·d + (2·ρ_new/δ)·z
      x += d
  ```
- Eigenvalue estimation — Gershgorin: `Research/03_GPU_Linear:L77-106`
- Eigenvalue estimation — Power iteration: `Research/03_GPU_Linear:L110-122`
- CG-based bounds (PETSc pattern): `Research/03_GPU_Linear:L124-126`
- Linear solver comparison: `improvement.md:L634-645`

**FFT/DCT Solver:**
- Periodic BC via FFT: `Research/03_GPU_Linear:L169-206`
- Neumann BC via DCT: `Research/03_GPU_Linear:L217-248`
  ```python
  # Laplacian eigenvalues (Neumann)
  λ_x = (2/dx²)(cos(πi/Nx) - 1)
  λ_y = (2/dy²)(cos(πj/Ny) - 1)
  # Solve: (1 - dt·D·Λ)·û = rhs_dct
  ```
- Singularity handling (zero frequency): `Research/03_GPU_Linear:L254-261`
- `torch-dct` package: `Research/code_examples/torch-dct/`
- Spectral Poisson reference: `Research/code_examples/shape_as_points/`

**AMG:**
- NVIDIA AmgX via pyamgx: `Research/03_GPU_Linear:L280-350`
- PyAMG for CPU prototyping: `Research/code_examples/pyamg/`
- Chebyshev smoother preferred over Gauss-Seidel on GPU: `Research/03_GPU_Linear:L300-310`

**Explicit Methods:**
- RK2 (Heun's): `Research/02_openCARP:L280-290` — 2x larger dt than FE for same accuracy
- RK4: `Research/00_Research_Summary:L68` — overkill for cardiac but useful for reference

**BDF2:**
- V5.3 implementation: `V5.3/IMPLEMENTATION.md:L780-825`
- Convergence validation: `V5.3/IMPLEMENTATION.md:L1409-1520` (Stage 4, test 4.4: ratios = 0.237, 0.247)

**LUT Integration:**
- V5.3 LUT code: `V5.3/IMPLEMENTATION.md:L640-741`
- V5.3 LUT validation: `V5.3/IMPLEMENTATION.md:L1639-1732` (Stage 6, 6/6 PASS, 1.46x speedup)

### Parallel Work

All items in Phase 6 are independent of each other (they all implement the same ABC interfaces):

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: Chebyshev | `chebyshev.py` | `LinearSolver` ABC (Phase 3) |
| B: FFT/DCT | `fft.py` | `LinearSolver` ABC, `mesh/structured.py` (Phase 2) |
| C: RK2/RK4 | `rk2.py`, `rk4.py` | `DiffusionSolver` ABC (Phase 3), `SpatialDiscretization.apply_diffusion()` (Phase 2) |
| D: BDF2 | `bdf2.py` | `DiffusionSolver` ABC, `LinearSolver` (Phase 3) |
| E: AMG | `amg.py` | `LinearSolver` ABC, external dependency (pyamgx/pyamg) |
| F: LUT wiring | Modify `rush_larsen.py` | `ionic/lut.py` (V5.3), `RushLarsenSolver` (Phase 3) |

**Maximum parallelism:** All 6 tracks are fully independent. This is the phase where the most developers can work simultaneously.

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 6-V1 | Chebyshev solves Laplacian | Same system as V5.3 PCG test 3.1. Relative error < 1e-6 |
| 6-V2 | Chebyshev vs PCG | Same system, same tolerance. Solutions agree to 1e-8. Chebyshev should use fewer GPU sync points (profile with `torch.cuda.Event`) |
| 6-V3 | Chebyshev eigenvalue bounds | Gershgorin bounds contain all eigenvalues. Verify against scipy.linalg.eigsh for small test matrix |
| 6-V4 | FFT/DCT pure diffusion | Isotropic structured grid, Gaussian decay. Compare against analytic. Error < 0.1% |
| 6-V5 | DCT Neumann BC | Constant field → no change. Linear gradient → correct steady-state flux |
| 6-V6 | FFT vs PCG cross-validation | Same structured grid problem. Solutions agree to 1e-6 |
| 6-V7 | RK2 convergence order | Richardson extrapolation → O(dt²). Compare error ratios |
| 6-V8 | RK4 convergence order | Richardson extrapolation → O(dt⁴). Use as reference for other methods |
| 6-V9 | RK2 allows larger dt than FE | Same accuracy threshold, RK2 should allow ~2x dt compared to Forward Euler |
| 6-V10 | BDF2 convergence order | Richardson extrapolation → O(dt²). Compare against V5.3 Stage 4 results (ratios ≈ 0.24) |
| 6-V11 | BDF2 first step = BDF1 | Verify automatic fallback. Compare against V5.3 test 4.6 |
| 6-V12 | LUT speedup preserved | LUT-backed RushLarsenSolver vs direct compute. Speedup ≥ 1.4x (V5.3 baseline: 1.46x) |
| 6-V13 | LUT accuracy preserved | LUT vs direct single cell: max V diff < 0.001 mV (V5.3 test 6.2: 0.0001 mV) |

**Debug strategy for Chebyshev:** If convergence is poor, the eigenvalue bounds are wrong. Check:
1. Gershgorin gives `[λ_min, λ_max]` — verify these bracket the true spectrum
2. Apply 10% safety margin: `λ_min *= 0.9`, `λ_max *= 1.1`
3. The condition number `λ_max/λ_min` determines iteration count

**Debug strategy for FFT/DCT:** The most common bug is incorrect eigenvalue formula for Neumann BC. The DCT eigenvalues are `λ_k = (2/dx²)(cos(πk/N) - 1)`, NOT `(2/dx²)(cos(2πk/N) - 1)` (that's periodic/FFT).

---

## Phase 7: Builder Integration — Backend Pipeline

**Goal:** Bring the Builder tools (MeshBuilder, StimBuilder, UI) into Engine_V5.4 and implement the export/load pipeline: SVG → `.npz` file → Engine objects. Backend-focused — UI polish is not in scope.

### Architecture

```
SVG File (draw.io export)
    │
    ├───────────────────────────────────────┐
    ▼                                       ▼
┌──────────────────────┐         ┌──────────────────────┐
│  mesh_builder/        │         │  stim_builder/        │
│                       │         │                       │
│  session.py           │         │  session.py           │
│    load_image()       │         │    load_image()       │
│    detect_colors()    │         │    detect_colors()    │
│    configure_group()  │         │    configure_region() │
│    set_dimensions()   │         │                       │
│                       │         │                       │
│  export.py ← NEW      │         │  export.py ← NEW      │
│    export_mesh()      │         │    export_stim()      │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           ▼                                ▼
    ┌──────────────┐                 ┌──────────────┐
    │  mesh.npz     │                 │  stim.npz     │
    └──────┬───────┘                 └──────┬───────┘
           │                                │
           ▼                                ▼
┌──────────────────────┐         ┌──────────────────────┐
│  tissue_builder/      │         │  tissue_builder/      │
│  mesh/loader.py ← NEW │         │  stimulus/loader.py   │
│                       │         │    ← NEW              │
│  load_mesh() →        │         │  load_stimulus() →    │
│    StructuredGrid     │         │    StimulusProtocol   │
│    + D_xx, D_yy, D_xy│         │                       │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           └────────────┬───────────────────┘
                        ▼
             ┌─────────────────────┐
             │ User script:        │
             │   spatial = FDM(…)  │
             │   sim = Monodomain… │
             │   sim.run(500.0)    │
             └─────────────────────┘
```

### Action Items

| # | Task | Details |
|---|------|---------|
| 7.1 | Copy `Builder/MeshBuilder/` → `Engine_V5.4/mesh_builder/` | Include `common/` for image processing. Fix imports. |
| 7.2 | Copy `Builder/StimBuilder/` → `Engine_V5.4/stim_builder/` | Include `common/` (duplicated). Fix imports. |
| 7.3 | Copy `Builder/ui/` → `Engine_V5.4/ui/` | Flask app, templates, static. Rewire imports to local `mesh_builder/`. |
| 7.4 | Implement `mesh_builder/export.py` | `export_mesh(session, path)` → writes `mesh.npz` |
| 7.5 | Implement `stim_builder/export.py` | `export_stim(session, path)` → writes `stim.npz` |
| 7.6 | Implement `tissue_builder/mesh/loader.py` | `load_mesh(path)` → `MeshData(grid, D_xx, D_yy, D_xy, metadata)` |
| 7.7 | Implement `tissue_builder/stimulus/loader.py` | `load_stimulus(path, mesh_mask)` → `StimulusProtocol` |
| 7.8 | Wire UI export button | Add `/api/export` endpoint to `ui/server.py` |

### File Hierarchy After Phase 7

```
Engine_V5.4/
├── cardiac_sim/                          # existing engine (Phases 1-6)
│   ├── tissue_builder/
│   │   ├── mesh/
│   │   │   ├── base.py                   # existing (Mesh ABC)
│   │   │   ├── triangular.py             # existing (FEM)
│   │   │   ├── structured.py             # existing (FDM/FVM/LBM)
│   │   │   └── loader.py                 # NEW — mesh.npz → StructuredGrid + D fields
│   │   ├── tissue/
│   │   │   └── isotropic.py              # existing
│   │   └── stimulus/
│   │       ├── protocol.py               # existing
│   │       ├── regions.py                # existing
│   │       └── loader.py                 # NEW — stim.npz → StimulusProtocol
│   └── ...
│
├── mesh_builder/                         # NEW — from Builder/MeshBuilder
│   ├── __init__.py
│   ├── models.py                         # CellGroup dataclass
│   ├── session.py                        # MeshBuilderSession
│   ├── export.py                         # NEW — session → mesh.npz
│   └── common/                           # from Builder/common (duplicated)
│       ├── __init__.py
│       ├── image.py
│       └── utils.py
│
├── stim_builder/                         # NEW — from Builder/StimBuilder
│   ├── __init__.py
│   ├── models.py                         # StimRegion, StimProtocol
│   ├── session.py                        # StimBuilderSession
│   ├── export.py                         # NEW — session → stim.npz
│   └── common/                           # from Builder/common (duplicated)
│       ├── __init__.py
│       ├── image.py
│       └── utils.py
│
├── ui/                                   # NEW — from Builder/ui
│   ├── server.py                         # Flask app (imports → mesh_builder/)
│   ├── templates/                        # base, start, upload, loading, workspace
│   └── static/                           # css, js
│
└── ...
```

### mesh.npz File Specification

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `mask` | bool | (Nx, Ny) | True = active tissue, False = background |
| `dx` | float | scalar | Grid spacing x (cm) |
| `dy` | float | scalar | Grid spacing y (cm) |
| `Lx` | float | scalar | Physical domain width (cm) |
| `Ly` | float | scalar | Physical domain height (cm) |
| `D_xx` | float64 | (Nx, Ny) | Conductivity tensor D_xx per pixel (cm²/ms) |
| `D_yy` | float64 | (Nx, Ny) | Conductivity tensor D_yy per pixel (cm²/ms) |
| `D_xy` | float64 | (Nx, Ny) | Conductivity tensor D_xy per pixel (cm²/ms) |
| `label_map` | int32 | (Nx, Ny) | Group index per pixel (-1 = background) |
| `group_labels` | str | (n_groups,) | Human label per tissue group |
| `group_cell_types` | str | (n_groups,) | Cell type per group (e.g., "myocardial") |

Background pixels have D_xx = D_yy = D_xy = 0 and label_map = -1.

Non-conductive tissue (scar) has label_map ≥ 0 but D = 0.

### stim.npz File Specification

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `n_regions` | int | scalar | Number of stimulus regions |
| `mask_{i}` | bool | (Nx, Ny) | Spatial mask for region i (same grid as mesh) |
| `label_{i}` | str | — | Region label (e.g., "S1_pacing") |
| `stim_type_{i}` | str | — | "current_injection" or "voltage_clamp" |
| `amplitude_{i}` | float | scalar | μA/cm² (current) or mV (clamp) |
| `duration_{i}` | float | scalar | Pulse duration (ms) |
| `start_time_{i}` | float | scalar | First pulse onset (ms) |
| `bcl_{i}` | float | scalar | Basic cycle length (ms), 0 = single pulse |
| `num_pulses_{i}` | int | scalar | Pulse count, 0 = infinite |

### Key Implementation Details

**`mesh_builder/export.py` — `export_mesh(session, output_path)`:**
1. Iterate session's `image_array` pixel-by-pixel
2. For each pixel, look up its color in `session.color_groups`
3. If group is background → mask=False, D=0
4. If group is tissue → mask=True, D from `CellGroup.conductivity_tensor`
5. Build D_xx/D_yy/D_xy arrays from each group's 2×2 tensor
6. Save with `np.savez(output_path, ...)`

**`tissue_builder/mesh/loader.py` — `load_mesh(path, device, dtype)`:**
1. `data = np.load(path)` → extract mask, dx, dy, D arrays (already in grid convention)
2. `grid = StructuredGrid.from_mask(torch.tensor(mask), dx, dy, device, dtype)`
3. Flatten D arrays to active-node ordering: `D_xx_flat = D_xx[mask]`
4. Convert to torch tensors on target device
5. Return `MeshData(grid, D_xx, D_yy, D_xy, labels, cell_types)`

**`tissue_builder/stimulus/loader.py` — `load_stimulus(path, mesh_mask, device, dtype)`:**
1. Load stim.npz, iterate over n_regions
2. For each region: intersect mask with mesh_mask, flatten to active-node ordering
3. Create `StimulusProtocol`, call `add_regular_pacing()` or `add_stimulus()` per region
4. Return configured `StimulusProtocol`

### References

- Builder integration spec: `improvement.md:L1294-1327`
- Builder API: `Builder/BACKEND.md`
- MeshBuilderSession: `Builder/MeshBuilder/session.py`
- StimBuilderSession: `Builder/StimBuilder/session.py`
- CellGroup dataclass: `Builder/MeshBuilder/models.py`
- StimRegion/StimProtocol: `Builder/StimBuilder/models.py`
- StructuredGrid.from_mask: `cardiac_sim/tissue_builder/mesh/structured.py`
- StimulusProtocol API: `cardiac_sim/tissue_builder/stimulus/protocol.py`

### Parallel Work

| Track | Files | Dependencies |
|-------|-------|-------------|
| A: Copy + fix imports | mesh_builder/, stim_builder/, ui/ | None |
| B: Export functions | mesh_builder/export.py, stim_builder/export.py | Track A (needs files in place) |
| C: Engine loaders | mesh/loader.py, stimulus/loader.py | Track B (needs .npz spec) |

Tracks B and C can be developed against the .npz spec in parallel if test .npz files are created manually.

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 7-V1 | mesh_builder export round-trip | Load SVG → `MeshBuilderSession` → `export_mesh()` → verify .npz contains expected keys, shapes, dtypes |
| 7-V2 | mesh loader round-trip | Load mesh.npz → `load_mesh()` → StructuredGrid. Verify n_dof = mask.sum(), dx matches, coordinates correct |
| 7-V3 | Conductivity mapping | Configure 2 tissue groups with different D → export → load → verify per-node D_xx/D_yy values match group config |
| 7-V4 | Scar region D=0 | Mark a group as non-conductive → export → load → verify D_xx=D_yy=D_xy=0 for those nodes |
| 7-V5 | stim_builder export round-trip | Configure 2 stim regions → `export_stim()` → verify .npz keys, mask shapes, protocol values |
| 7-V6 | stim loader round-trip | Load stim.npz + mesh mask → `load_stimulus()` → StimulusProtocol with correct region masks and timing |
| 7-V7 | End-to-end: .npz → simulation | Load mesh.npz + stim.npz → create FDMDiscretization(grid, D=scalar) → MonodomainSimulation → run 10ms → verify AP initiates at stimulus site |
| 7-V8 | UI export endpoint | POST to `/api/export` after configuring session → verify .npz file written and downloadable |

**Note:** Test 7-V7 uses scalar D (existing FDMDiscretization). Per-node D support is Phase 8.

**Debug strategy:** Most likely issues are:
1. Import path mismatches after copy (test with `python -c "from mesh_builder.session import MeshBuilderSession"`)
2. Array shape/orientation (image is (H, W, C) → mask is (Nx, Ny) where Ny=H, Nx=W)
3. Active-node flattening order must match `StructuredGrid.from_mask()` internal ordering

---

## Phase 8: Per-Node Conductivity — Heterogeneous & Anisotropic Tissue

**Goal:** Extend FDM and FVM discretizations to accept per-node conductivity tensor arrays (D_xx, D_yy, D_xy). This enables heterogeneous tissue (scar = D=0 regions) and anisotropic propagation from Builder-exported meshes.

### Action Items

| # | Task | Details |
|---|------|---------|
| 8.1 | Extend `FDMDiscretization` | Accept optional `D_xx, D_yy, D_xy` arrays (n_dof,). When provided, use per-node values in 9-pt stencil assembly instead of scalar D. Backward compatible. |
| 8.2 | Extend `FVMDiscretization` | Accept optional per-node D arrays. Compute per-face D via harmonic mean of neighboring cells. Backward compatible. |
| 8.3 | Update `monodomain.py` factory | Forward D arrays from loader output to discretization constructor. Accept pre-built `SpatialDiscretization` (already works) or add D array kwargs. |
| 8.4 | End-to-end heterogeneous simulation | SVG with scar region → mesh.npz → loader → FDM with per-node D → MonodomainSimulation → wave routes around scar |

### References

- 9-pt stencil with spatially-varying D: `Research/01_FDM:L49-98`
- Diffusion tensor from fiber angle: `Research/01_FDM:L91-98`
- FVM harmonic mean at interfaces: `Research/00_Research_Summary:L90`
- D=0 boundary handling: `Research/01_FDM:L100-120`
- Current FDM implementation: `cardiac_sim/simulation/classical/discretization_scheme/fdm.py`
- Current FVM implementation: `cardiac_sim/simulation/classical/discretization_scheme/fvm.py`

### Key Implementation Details

**FDM per-node extension:**
- Current: scalar D → uniform Dxx, Dyy, Dxy across grid
- New: optional D_xx(i,j), D_yy(i,j), D_xy(i,j) arrays → per-node stencil weights
- The 9-pt stencil math is already per-node capable:
  ```
  NW = +D_xy(i,j)/(4·dx·dy)     N = D_yy(i,j)/dy²
  W  = D_xx(i,j)/dx²             C = -(sum)
  ```
- Just replace scalar D with array indexing in the assembly loop
- At scar boundaries (D=0), stencil weights naturally become zero → no flux

**FVM per-node extension:**
- Face conductivity uses harmonic mean: `D_face = 2·D_L·D_R / (D_L + D_R)`
- When either neighbor has D=0: `D_face = 0` (correct scar behavior)
- Replace scalar D with per-cell D_xx/D_yy in flux computation

**Backward compatibility:**
- If only scalar D provided → broadcast to uniform arrays internally
- All Phase 1-6 tests continue to pass unchanged

### Validation & Debug Plan

| Test | Criteria | Method |
|------|----------|--------|
| 8-V1 | FDM uniform D matches scalar D | Per-node D_xx=D_yy=D (uniform) must produce identical Laplacian as scalar D. Max diff = 0. |
| 8-V2 | FVM uniform D matches scalar D | Same as 8-V1 for FVM. |
| 8-V3 | FDM scar blocks diffusion | Central scar strip (D=0). Apply Gaussian pulse one side. Verify zero voltage change inside scar after 100 steps. |
| 8-V4 | FVM scar blocks diffusion | Same as 8-V3 for FVM. Harmonic mean → D_face=0 at scar boundary. |
| 8-V5 | Anisotropic propagation | 45° fiber field → elliptical wavefront. Axis ratio ≈ 2:1 for D_fiber:D_cross = 4:1. |
| 8-V6 | End-to-end: SVG with scar → simulation | Load Builder mesh with scar → FDM per-node D → MonodomainSimulation → wave routes around scar, zero V inside scar |
| 8-V7 | Phase 1-6 regression | All existing tests still pass with scalar D path |

**Debug strategy:**
- Scar leakage: If voltage appears inside scar, check harmonic mean (FVM) or stencil weights (FDM) at the scar boundary. Arithmetic mean gives D/2 (wrong). Harmonic mean gives 0 (correct).
- Anisotropy: If wavefront isn't elliptical, verify fiber angle θ is in radians and that the tensor construction `Dxx = Df·cos²θ + Dc·sin²θ` uses the correct convention.

---

## Cross-Phase Dependency Summary

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
  │              │            │           │
  │              │            │           ▼
  │              │            │      End-to-end
  │              │            │      (classical)
  │              │            │
  │              ├────────────┼──→ Phase 6 (all tracks parallel)
  │              │            │
  │              └────────→ Phase 5 (LBM, semi-independent)
  │                                   │
  └─────────────────────────────→ Phase 7 (Builder pipeline)
                                      │
                                      └──→ Phase 8 (per-node D in FDM/FVM)
```

**Critical path:** Phase 1 → 2 → 3 → 4 (sequential).

**Parallelism opportunities:**
- Phase 5 (LBM) can start after Phase 2
- Phase 6 (optimizations) can start after Phase 3
- Phase 7 (Builder) can start after Phase 2 (needs mesh types)
- Phase 8 (per-node D) depends on Phase 7 (needs .npz loaders) + Phase 2 (FDM/FVM code)
- Within Phase 6, all tracks are fully independent

---

## V5.3 → V5.4 Test Migration Map

Every V5.3 test should have a V5.4 counterpart. New features add new tests.

| V5.3 Stage | V5.3 Tests | V5.4 Phase | V5.4 Tests | Notes |
|------------|-----------|------------|-----------|-------|
| 1 (IonicModel) | 1.1-1.8 | 3 | 3-V1, 3-V9 | Refactored interface, same physics |
| 1.5 (TTP06 detail) | gating, ERP | 3 | 3-V1 | Physics unchanged |
| 2 (FEM) | 2.1-2.9 | 2 | 2-V1, 2-V2 | FEM migrated, new FDM/FVM tests added |
| 3 (PCG) | 3.1-3.7 | 3 | 3-V7 | PCG migrated, new Chebyshev/FFT in Phase 6 |
| 4 (Time integration) | 4.1-4.8 | 3 | 3-V5, 3-V6 | CN/BDF migrated, new explicit in Phase 6 |
| 5 (Full integration) | 5.1-5.6 | 4 | 4-V1 to 4-V8 | End-to-end through new orchestrator |
| 6 (LUT) | 6.1-6.6 | 6 | 6-V12, 6-V13 | LUT wired into new solver chain |
| 7 (Performance) | 7.1-7.6 | 4 | 4-V9, 4-V10 | Zero-alloc, reproducibility |
| — (new) | — | 2 | 2-V3 to 2-V10 | FDM, FVM, structured grid (new) |
| — (new) | — | 5 | 5-V1 to 5-V10 | LBM (new paradigm) |
| — (new) | — | 6 | 6-V1 to 6-V11 | Chebyshev, FFT, RK2/4, BDF2 (new) |
| — (new) | — | 7 | 7-V1 to 7-V8 | Builder pipeline (new) |
| — (new) | — | 8 | 8-V1 to 8-V7 | Per-node conductivity, scar, anisotropy (new) |

**Total: 80+ validation tests across 8 phases.**

---

## Summary of Key Research References by Module

Quick-lookup table for implementation — which research doc to open for each file.

| File to Implement | Primary Research | Key Lines | Code Example |
|-------------------|-----------------|-----------|--------------|
| `fdm.py` | `01_FDM:L49-98` | 9-point stencil, fiber tensor | — |
| `fvm.py` | `02_openCARP:L200-250` | Cell-centered flux | `code_examples/MonoAlg3D_C/` |
| `mesh/structured.py` | `01_FDM:L140-161` | Grid + mask + fibers | — |
| `chebyshev.py` | `03_GPU_Linear:L39-106` | Algorithm + Gershgorin bounds | — |
| `fft.py` | `03_GPU_Linear:L169-261` | FFT (periodic) + DCT (Neumann) | `code_examples/torch-dct/` |
| `amg.py` | `03_GPU_Linear:L280-350` | AmgX/PyAMG wrapper | `code_examples/pyamg/`, `pyamgx/` |
| `rush_larsen.py` | `02_openCARP:L300-350` | openCARP pattern | — |
| `lbm/d2q5.py` | `04_LBM_EP:L105-125` | Lattice vectors, weights | `code_examples/lettuce/` |
| `lbm/collision.py` | `04_LBM_EP:L134-186` | BGK + MRT | `code_examples/lettuce/` |
| `lbm/monodomain.py` | `04_LBM_EP:L850-1100` | Full PyTorch blueprint | `code_examples/lettuce/` |
| `mesh_builder/export.py` | — | Builder session → .npz | `Builder/MeshBuilder/session.py` |
| `stim_builder/export.py` | — | Builder session → .npz | `Builder/StimBuilder/session.py` |
| `mesh/loader.py` | — | .npz → StructuredGrid | `tissue_builder/mesh/structured.py` |
| `stimulus/loader.py` | — | .npz → StimulusProtocol | `tissue_builder/stimulus/protocol.py` |
| `fdm.py` (Phase 8) | `01_FDM:L49-98` | Per-node D in 9-pt stencil | — |
| `fvm.py` (Phase 8) | `02_openCARP:L200-250` | Per-face harmonic mean | `code_examples/MonoAlg3D_C/` |

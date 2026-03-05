# Engine V1 Bidomain: Phase-by-Phase Implementation Guide

> **SUPERSEDED:** This document describes the original coupled MINRES, FEM-first, 10-phase plan.
> The current implementation plan is in **`FDM_CODING_PLAN.md`** (FDM-first, decoupled, 6 phases).
> This file is retained for reference only — do NOT use it for implementation guidance.
>
> Key differences from current plan:
> - This file: FEM-first, coupled 2N×2N MINRES, block preconditioners, 10 phases, ~61 files
> - Current: FDM-first, decoupled N×N SPD, three-tier spectral/GMG, 6 phases, ~28 files

This document provides a concrete, ordered implementation plan for the Bidomain Engine V1. Each phase lists action items, file placements, cross-references to design docs and research, and validation criteria.

**Cross-Reference Key:**
- `improvement.md:L###` -- Architecture specification (this repo)
- `V5.4/` -- Monodomain Engine V5.4 (validated baseline, source for reusable code)
- `Research/Bidomain/Discretization/` -- FEM/FDM/FVM bidomain discretization
- `Research/Bidomain/Solver_Methods/` -- Operator splitting, time stepping
- `Research/Bidomain/Linear_Solvers/` -- Block preconditioners, AMG, MINRES
- `Research/Bidomain/LBM_Bidomain/` -- Dual-lattice LBM approach
- `Research/Bidomain/Code_Examples/` -- Reference implementations (~3600 lines)

---

## Pre-Implementation: V5.4 Baseline

The Monodomain Engine V5.4 is the starting point. Reusable components are copied directly; bidomain-specific components are built new.

**V5.4 components to reuse (unchanged):**

| Component | V5.4 Location | Reuse Strategy |
|-----------|--------------|----------------|
| IonicModel ABC + TTP06 + ORd | `ionic/` | Direct copy |
| Mesh ABC + TriangularMesh + StructuredGrid | `tissue_builder/mesh/` | Direct copy |
| StimulusProtocol + regions | `tissue_builder/stimulus/` | Direct copy |
| IsotropicTissue | `tissue_builder/tissue/` | Direct copy |
| Backend/device abstraction | `utils/backend.py` | Direct copy |
| IonicSolver ABC + RushLarsen + ForwardEuler | `solver/ionic_time_stepping/` | Direct copy |
| PCGSolver | `solver/.../linear_solver/pcg.py` | Adapt as SubSolver |
| ChebyshevSolver | `solver/.../linear_solver/chebyshev.py` | Adapt as SubSolver |

**V5.4 components that need bidomain-specific rewrite:**

| Component | Why Rewrite |
|-----------|-------------|
| SimulationState | Add phi_e field |
| SpatialDiscretization ABC | Returns K_i, K_e, M (not just K, M) |
| DiffusionSolver | Coupled parabolic-elliptic (not single PDE) |
| LinearSolver | 2N x 2N block system (not N x N SPD) |
| MonodomainSimulation | New orchestrator (BidomainSimulation) |

---

## Phase 1: Foundation -- Directory Structure & Reusable Code

**Goal:** Create V1 directory tree, copy reusable V5.4 code, create bidomain-specific stubs.

**Key references:**
- `improvement.md` -- Architecture tree
- V5.4 source for all reusable code

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 1.1 | `cardiac_sim/__init__.py` | Create | Version v1.0.0 |
| 1.2 | `cardiac_sim/ionic/` | Copy from V5.4 | All files, fix relative imports |
| 1.3 | `cardiac_sim/tissue_builder/` | Copy from V5.4 | mesh/, stimulus/, tissue/ |
| 1.4 | `cardiac_sim/tissue_builder/tissue/conductivity.py` | Create | BidomainConductivity dataclass |
| 1.5 | `cardiac_sim/utils/backend.py` | Copy from V5.4 | Update version string |
| 1.6 | `cardiac_sim/simulation/__init__.py` | Create | Package init |
| 1.7 | `cardiac_sim/simulation/classical/__init__.py` | Create | Skeleton |
| 1.8 | `cardiac_sim/simulation/classical/state.py` | Create | BidomainState dataclass |
| 1.9 | `cardiac_sim/simulation/classical/solver/__init__.py` | Create | Package structure |
| 1.10 | `cardiac_sim/simulation/classical/solver/ionic_time_stepping/` | Copy from V5.4 | rush_larsen, forward_euler |
| 1.11 | `cardiac_sim/simulation/classical/solver/splitting/` | Create | base.py stub |
| 1.12 | `cardiac_sim/simulation/classical/discretization_scheme/__init__.py` | Create | Skeleton |
| 1.13 | `cardiac_sim/simulation/lbm/__init__.py` | Create | Skeleton (future) |
| 1.14 | `cardiac_sim/tests/__init__.py` | Create | Test package |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 1-V1: All imports resolve | `import cardiac_sim.ionic`, `import cardiac_sim.tissue_builder`, etc. | |
| 1-V2: IonicModel works | TTP06 `compute_Iion(V, S)` produces correct output | |
| 1-V3: Mesh works | TriangularMesh.create_rectangle() matches V5.4 bitwise | |
| 1-V4: BidomainState | Instantiate with Vm, phi_e, ionic_states; verify shapes | |
| 1-V5: BidomainConductivity | D_i, D_e scalar and fiber-based creation | |

---

## Phase 2: Bidomain Spatial Discretization (FEM)

**Goal:** Create BidomainSpatialDiscretization ABC and FEM implementation that assembles K_i, K_e, M.

**Key references:**
- `improvement.md` -- BidomainSpatialDiscretization ABC
- `Research/Bidomain/Discretization/BIDOMAIN_DISCRETIZATION.md` Sections 2.1-2.4 (FEM)
- V5.4 `discretization_scheme/fem.py` (single-conductivity reference)
- `Research/Bidomain/Code_Examples/bidomain_block_system.py`

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 2.1 | `discretization_scheme/base.py` | Create | BidomainSpatialDiscretization ABC, BidomainOperators, MassType |
| 2.2 | `discretization_scheme/fem.py` | Create | BidomainFEMDiscretization: K_i, K_e, M assembly |
| 2.3 | Tests | Create | Matrix symmetry, row sums, K_i vs V5.4 single-D |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 2-V1: FEM K_i matches V5.4 K | When D_i = D (V5.4), K_i must be bitwise identical to V5.4's K | |
| 2-V2: FEM K_e independent | K_e assembled with D_e, different from K_i | |
| 2-V3: K_i + K_e SPD | Sum is symmetric positive semi-definite (row sums = 0) | |
| 2-V4: M matches V5.4 | Mass matrix identical to V5.4 (chi*Cm scaling) | |
| 2-V5: BidomainOperators | A11, A12, A21, A22 have correct dimensions (N x N) | |
| 2-V6: RHS vector | build_rhs(Vm, phi_e, Iion) produces correct-shape (N,) vectors | |
| 2-V7: CN vs BDF1 operators | Different theta values produce different A11 | |

---

## Phase 3: Block Linear Solver (MINRES)

**Goal:** Implement MINRES for the 2N x 2N symmetric indefinite block system, with block diagonal preconditioner.

**Key references:**
- `improvement.md` -- BlockLinearSolver ABC, BlockPreconditioner ABC
- `Research/Bidomain/Linear_Solvers/BIDOMAIN_LINEAR_SOLVERS.md` Sections 1-2, 4
- `Research/Bidomain/Code_Examples/bidomain_block_preconditioner.py`

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 3.1 | `block_linear_solver/base.py` | Create | BlockLinearSolver ABC |
| 3.2 | `block_linear_solver/minres.py` | Create | MINRES iteration, workspace, convergence check |
| 3.3 | `block_linear_solver/preconditioner/base.py` | Create | BlockPreconditioner ABC |
| 3.4 | `block_linear_solver/preconditioner/block_diagonal.py` | Create | P = diag(A11, A22) |
| 3.5 | `block_linear_solver/preconditioner/subsolver/base.py` | Create | SubSolver ABC |
| 3.6 | `block_linear_solver/preconditioner/subsolver/pcg.py` | Create | Adapt V5.4 PCGSolver |
| 3.7 | Tests | Create | MINRES convergence, preconditioner effect |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 3-V1: MINRES solves small system | 4x4 block system, exact solution to tol=1e-10 | |
| 3-V2: MINRES + block diag precond | Converges in <80 iters for 32x32 grid | |
| 3-V3: Unpreconditioned MINRES | Converges (slowly, 200+ iters) for correctness check | |
| 3-V4: PCG sub-solver | Solves SPD A11 block to tolerance | |
| 3-V5: Residual monotonic | MINRES residual decreases monotonically | |
| 3-V6: Warm start | Second solve with same system converges faster | |
| 3-V7: Zero allocation | No tensor allocation after first solve | |

---

## Phase 4: Bidomain Diffusion Solver

**Goal:** Implement BidomainDiffusionSolver (Crank-Nicolson) that assembles the block system and solves via MINRES.

**Key references:**
- `improvement.md` -- BidomainDiffusionSolver ABC, BidomainCrankNicolsonSolver
- `Research/Bidomain/Solver_Methods/BIDOMAIN_SOLVER_METHODS.md` Sections 3-4
- `Research/Bidomain/Code_Examples/bidomain_operator_splitting.py`

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 4.1 | `diffusion_time_stepping/base.py` | Create | BidomainDiffusionSolver ABC |
| 4.2 | `diffusion_time_stepping/implicit/crank_nicolson.py` | Create | CN theta=0.5, block build |
| 4.3 | `diffusion_time_stepping/implicit/bdf1.py` | Create | BDF1 theta=1.0 |
| 4.4 | Null space pinning | Implement | phi_e pinning in A22 block |
| 4.5 | Tests | Create | Diffusion-only convergence |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 4-V1: Pure diffusion (Vm only) | Gaussian diffusion O(h^2) convergence (D_e=0 limit) | |
| 4-V2: Bidomain diffusion | Vm and phi_e both evolve correctly from impulse | |
| 4-V3: Phi_e pinning | phi_e(pin_node) = 0 maintained throughout | |
| 4-V4: CN 2nd order | O(dt^2) convergence in time | |
| 4-V5: BDF1 1st order | O(dt) convergence in time | |
| 4-V6: Monodomain limit | With D_e = lambda*D_i, phi_e -> 0, Vm matches monodomain | |
| 4-V7: Conservation | Total Vm unchanged by diffusion (Neumann BC) | |

---

## Phase 5: Orchestration (BidomainSimulation)

**Goal:** Create BidomainSimulation orchestrator with operator splitting, factory functions, and run loop.

**Key references:**
- `improvement.md` -- BidomainSimulation, SplittingStrategy, runtime step spec
- V5.4 `monodomain.py` (orchestrator pattern)

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 5.1 | `solver/splitting/base.py` | Create | SplittingStrategy ABC |
| 5.2 | `solver/splitting/godunov.py` | Create | ionic -> diffusion |
| 5.3 | `solver/splitting/strang.py` | Create | half -> full -> half |
| 5.4 | `simulation/classical/bidomain.py` | Create | Orchestrator, factory functions, run loop |
| 5.5 | `simulation/classical/__init__.py` | Update | Export BidomainSimulation |
| 5.6 | Tests | Create | End-to-end simulation |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 5-V1: Single cell AP | Vm matches V5.4 monodomain AP (D_i=D, D_e=0 limit) | |
| 5-V2: 1D cable propagation | Conduction velocity within 5% of expected | |
| 5-V3: Godunov ordering | ionic -> diffusion sequence verified | |
| 5-V4: Strang ordering | half -> full -> half sequence verified | |
| 5-V5: Phi_e field generated | phi_e non-zero during propagation | |
| 5-V6: String config | All factory functions work correctly | |
| 5-V7: Output buffer | Both Vm and phi_e saved at correct times | |
| 5-V8: Reproducibility | Bitwise identical across runs | |
| 5-V9: Strang vs Godunov | Strang more accurate (O(dt^2) vs O(dt)) | |
| 5-V10: Monodomain equivalence | With D_e = lambda*D_i, matches monodomain CV | |

---

## Phase 6: Advanced Preconditioners

**Goal:** Implement block triangular preconditioner with Schur complement for production-grade solver performance.

**Key references:**
- `improvement.md` -- BlockTriangularPreconditioner, Schur complement
- `Research/Bidomain/Linear_Solvers/BIDOMAIN_LINEAR_SOLVERS.md` Sections 2.2-2.3

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 6.1 | `preconditioner/block_triangular.py` | Create | P = [A11, 0; A21, S] |
| 6.2 | Schur complement approx | Implement | S_approx = A22 (zero) and Neumann |
| 6.3 | GMRES solver | Create | `block_linear_solver/gmres.py` for non-symmetric case |
| 6.4 | BDF2 solver | Create | `diffusion_time_stepping/implicit/bdf2.py` |
| 6.5 | Tests | Create | Iteration count reduction |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 6-V1: Block triangular iters | MINRES converges in <30 iters (vs 50-80 diagonal) | |
| 6-V2: Schur complement quality | Neumann approx reduces iters by 2x vs zero approx | |
| 6-V3: GMRES correctness | Matches MINRES solution to machine precision | |
| 6-V4: BDF2 convergence | O(dt^2) convergence | |
| 6-V5: BDF2 first step = BDF1 | First step uses BDF1 bootstrap | |
| 6-V6: Performance timing | Block triangular 2x faster than block diagonal (wall time) | |

---

## Phase 7: FDM/FVM Bidomain

**Goal:** Extend FDM and FVM discretizations to bidomain (dual conductivity L_i/L_e or F_i/F_e).

**Key references:**
- `improvement.md` -- BidomainFDMDiscretization, BidomainFVMDiscretization
- V5.4 `discretization_scheme/fdm.py`, `fvm.py` (single-conductivity reference)
- `Research/Bidomain/Discretization/` Sections 3-4 (FDM/FVM)
- `Research/Bidomain/Code_Examples/bidomain_fdm_assembly.py`

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 7.1 | `discretization_scheme/fdm.py` | Create | Bidomain FDM: L_i, L_e from D_i, D_e |
| 7.2 | `discretization_scheme/fvm.py` | Create | Bidomain FVM: F_i, F_e from D_i, D_e |
| 7.3 | Factory function updates | Update | Support fdm/fvm in config strings |
| 7.4 | Tests | Create | Cross-validation with FEM |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 7-V1: FDM L_i matches V5.4 L | With D_i = D (V5.4), L_i bitwise matches V5.4 Laplacian | |
| 7-V2: FDM Laplacian convergence | O(h^2) for both L_i and L_e | |
| 7-V3: FDM Neumann BC | Row sums = 0 for both L_i and L_e | |
| 7-V4: FVM flux conservation | Row sums = 0 for both F_i and F_e | |
| 7-V5: FDM/FVM cross-validation | Interior values match at grid points | |
| 7-V6: FDM end-to-end | Full bidomain simulation with FDM produces wave | |
| 7-V7: FVM end-to-end | Full bidomain simulation with FVM produces wave | |
| 7-V8: FDM per-node D | Anisotropic D_i_field and D_e_field with harmonic mean | |
| 7-V9: Scar blocks both domains | D_i=0 and D_e=0 at scar -> zero flux | |

---

## Phase 8: GPU Optimization

**Goal:** Add GPU-optimized sub-solvers (Chebyshev, AMG) for block preconditioner inner solves.

**Key references:**
- `improvement.md` -- GPU optimization strategy, SubSolver
- `Research/Bidomain/Linear_Solvers/` Section 7 (GPU-specific)
- V5.4 `linear_solver/chebyshev.py` (reference)

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 8.1 | `preconditioner/subsolver/chebyshev.py` | Create | Adapt V5.4 Chebyshev as SubSolver |
| 8.2 | `preconditioner/subsolver/amg.py` | Create | pyamg interface for CPU prototyping |
| 8.3 | Explicit diffusion solver | Create | Forward Euler for parabolic part |
| 8.4 | Factory updates | Update | Support chebyshev/amg in config |
| 8.5 | Tests | Create | GPU performance benchmarks |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 8-V1: Chebyshev sub-solver | Matches PCG solution to rel diff < 1e-6 | |
| 8-V2: AMG sub-solver | Converges in O(1) V-cycles | |
| 8-V3: Block triangular + Chebyshev | Total MINRES iters < 30 with Chebyshev inner | |
| 8-V4: Block triangular + AMG | Total MINRES iters < 25 with AMG inner | |
| 8-V5: Explicit FE diffusion | Stable within CFL, unstable beyond | |
| 8-V6: GPU vs CPU | Correct results on both devices | |
| 8-V7: Zero-sync inner solve | Chebyshev has no GPU-CPU sync per iteration | |

---

## Phase 9: Builder Integration

**Goal:** Extend Builder pipeline to support bidomain conductivity export (D_i and D_e per node/region).

**Key references:**
- V5.4 Phase 7 (Builder integration pattern)
- V5.4 `mesh_builder/export.py`, `tissue_builder/mesh/loader.py`

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 9.1 | `mesh_builder/` | Copy from V5.4 | Unchanged |
| 9.2 | `stim_builder/` | Copy from V5.4 | Unchanged |
| 9.3 | `mesh_builder/export.py` | Extend | Export D_i and D_e fields per region |
| 9.4 | `tissue_builder/mesh/loader.py` | Extend | Load D_i and D_e as BidomainConductivity |
| 9.5 | `tissue_builder/stimulus/loader.py` | Extend | Support I_stim_i and I_stim_e |
| 9.6 | `ui/server.py` | Extend | Bidomain conductivity presets in UI |
| 9.7 | Tests | Create | Round-trip export/load |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 9-V1: Mesh export round-trip | D_i and D_e arrays preserved | |
| 9-V2: Loader produces BidomainConductivity | Correct D_i_field, D_e_field shapes | |
| 9-V3: Multi-region D_i/D_e | Different conductivities per tissue group | |
| 9-V4: Scar D_i=D_e=0 | Scar nodes have zero in both domains | |
| 9-V5: Stim loader I_stim_i/e | Separate amplitudes loaded correctly | |
| 9-V6: End-to-end SVG -> bidomain | Full pipeline produces valid simulation | |

---

## Phase 10: LBM Bidomain (Future / Exploratory)

**Goal:** Implement dual-lattice LBM approach for bidomain as an alternative to classical FEM/FDM/FVM.

**Key references:**
- `improvement.md` -- LBM bidomain section
- `Research/Bidomain/LBM_Bidomain/LBM_BIDOMAIN.md` (full document)
- `Research/Bidomain/Code_Examples/bidomain_lbm_dual_lattice.py`
- V5.4 `simulation/lbm/` (monodomain LBM reference)

### Action Items

| # | File | Action | Notes |
|---|------|--------|-------|
| 10.1 | `simulation/lbm/state.py` | Create | Dual-lattice state (f_Vm, f_phi_e) |
| 10.2 | `simulation/lbm/d2q5.py` | Copy from V5.4 | Reuse lattice definition |
| 10.3 | `simulation/lbm/collision.py` | Create | BGK/MRT for dual lattices |
| 10.4 | `simulation/lbm/bidomain.py` | Create | Orchestrator with pseudo-time phi_e |
| 10.5 | Tests | Create | Convergence and feasibility |

### Validation

| Test | Criteria | Status |
|------|----------|--------|
| 10-V1: Vm diffusion | Gaussian diffusion matches analytical | |
| 10-V2: phi_e steady state | Pseudo-time converges to elliptic solution | |
| 10-V3: Coupled system | Vm and phi_e fields correct for known test case | |
| 10-V4: Conservation | Sum of distributions = potential to machine precision | |
| 10-V5: Pseudo-time iterations | Converges in <50 inner iterations per time step | |
| 10-V6: LBM vs FEM cross-validation | Matches FEM solution within discretization error | |

---

## Summary of Phases

| Phase | Goal | Files | Tests | Priority |
|-------|------|-------|-------|----------|
| 1 | Foundation | 14 | 5 | Critical |
| 2 | FEM Spatial Discretization | 3 | 7 | Critical |
| 3 | Block Linear Solver (MINRES) | 7 | 7 | Critical |
| 4 | Bidomain Diffusion Solver | 5 | 7 | Critical |
| 5 | Orchestration | 6 | 10 | Critical |
| 6 | Advanced Preconditioners | 5 | 6 | High |
| 7 | FDM/FVM | 4 | 9 | High |
| 8 | GPU Optimization | 5 | 7 | Medium |
| 9 | Builder Integration | 7 | 6 | Medium |
| 10 | LBM Bidomain | 5 | 6 | Low (exploratory) |
| **Total** | | **~61 files** | **~70 tests** | |

---

## Summary of Key Research References

| Module | Primary Research Doc | Key Sections |
|--------|---------------------|-------------|
| `discretization_scheme/fem.py` | Discretization doc | Sections 2.1-2.4 (FEM weak form) |
| `discretization_scheme/fdm.py` | Discretization doc | Sections 3.1-3.4 (FDM stencils) |
| `discretization_scheme/fvm.py` | Discretization doc | Sections 4.1-4.6 (FVM fluxes) |
| `block_linear_solver/minres.py` | Linear_Solvers doc | Sections 4.1-4.2 (MINRES) |
| `preconditioner/block_diagonal.py` | Linear_Solvers doc | Section 2.1 |
| `preconditioner/block_triangular.py` | Linear_Solvers doc | Sections 2.2-2.3 |
| `preconditioner/subsolver/amg.py` | Linear_Solvers doc | Section 3 (AMG) |
| `diffusion_time_stepping/implicit/` | Solver_Methods doc | Sections 3-4 (splitting) |
| `solver/splitting/` | Solver_Methods doc | Sections 3.1-3.3 |
| `lbm/` | LBM_Bidomain doc | Full document |
| `bidomain_block_system.py` | Code_Examples | Block assembly reference |
| `bidomain_block_preconditioner.py` | Code_Examples | Preconditioner reference |

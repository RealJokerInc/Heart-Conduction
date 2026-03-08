# Bidomain Engine V1 — Critical Review & Merger Proposal

Critical code review of the bidomain cardiac simulation engine.
Phases 1-5 complete (38/38 tests), Phase 6 cross-validation in progress.

---

## Table of Contents

1. [Bug List](#1-bug-list)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Cross-Cutting Concerns: Chi/Cm Formulation](#3-cross-cutting-concerns-chicm-formulation)
4. [Unit Convention Analysis](#4-unit-convention-analysis)
5. [Design Issues](#5-design-issues)
6. [Unified Merger Proposal: Bidomain + Monodomain + LBM](#6-unified-merger-proposal)

---

## 1. Bug List

### Critical

| # | Location | Description | Impact |
|---|----------|-------------|--------|
| C1 | `discretization/fdm.py:88,100` + `conductivity.py:29-30` | **Chi/Cm double-encoding.** `BidomainConductivity.D_i = 0.00124 = sigma_i/(chi*Cm)` — D values already contain chi*Cm scaling. But `BidomainFDMDiscretization.__init__` defaults to `chi=1400, Cm=1.0` and computes `self._chi_Cm = chi*Cm = 1400`. In `get_parabolic_operators()`, `A_para = chi_Cm/dt * I - theta * L_i` uses chi_Cm=1400 with L_i built from D (diffusivity), making diffusion 1400x too weak relative to the capacitive term. Workaround: set chi=1.0, Cm=1.0 (documented in MEMORY.md). But default constructor is **wrong**. | Simulation produces no wavefront propagation with default parameters. All Phase 6 tests use workaround. Phase 2-5 tests use chi=1400 but test operators in isolation (symmetry, eigenvalues) rather than coupled physics, so they pass. |
| C2 | `ionic_stepping/rush_larsen.py:83` | **Ionic step missing Cm division.** `state.V = V + dt * (-(Iion + Istim))` — the bidomain equation (divided by chi*Cm) gives `dVm/dt = -I_ion/Cm`. With Cm=1.0 this is numerically correct, but if Cm != 1.0, the ionic step produces wrong voltage change. Combined with C1, this creates a constraint: Cm must equal 1.0. | Forces Cm=1.0 always. Cannot use physiological Cm values other than 1.0 without code changes. |

### Major

| # | Location | Description | Impact |
|---|----------|-------------|--------|
| M1 | `discretization/fdm.py:278-440` | **O(N) Python loop for Laplacian assembly.** Each node is visited in a Python `for i in range(nx): for j in range(ny)` loop with per-element list appends. For a 150x40 grid this takes ~0.5s; for a 400x400 grid it would take ~30s. Should use vectorized numpy or torch operations. | Performance bottleneck at init time. Acceptable for V1 validation but blocks scaling. |
| M2 | `decoupled.py:83-84` | **Coupling term in parabolic RHS uses full L_i, not theta-weighted.** Code: `rhs = B_para * Vm + L_i * phi_e`. The CN discretization of `dVm/dt = L_i*(Vm + phi_e)` with phi_e lagged gives coupling `L_i * phi_e^n`. This is correct for the decoupled (Gauss-Seidel) approach where phi_e is lagged to time n. But the comment says "NOT theta * L_i * phi_e" which is potentially misleading because fully implicit would use `theta * L_i * phi_e^{n+1} + (1-theta) * L_i * phi_e^n`. | Not a bug per se for the decoupled approach (phi_e is lagged), but the comment should be clearer about why. |
| M3 | `bidomain.py:149,152,159` | **ForwardEulerSolver inline class modifies V without Cm.** The inline `ForwardEulerSolver` in `_build_ionic_solver` does `state.V = V + dt * (-(Iion + Istim))` — same issue as C2. Also calls `self._update_gates()` which is inherited from `IonicSolver` base (Forward Euler default). | Forward Euler ionic solver shares C2 bug. |
| M4 | `spectral.py:143-147` | **DST eigenvalue formula uses `n-2` interior points.** For Dirichlet BCs, the spectral solver strips boundary nodes: `m = n - 2`. The eigenvalues `(2/dx^2)(1 - cos(pi*(k+1)/(m+1)))` assume the grid has boundary nodes at positions 0 and n-1 that are Dirichlet-fixed. The solver **ignores the A matrix entirely** — it solves with precomputed eigenvalues. If A has different structure (e.g., anisotropic), the spectral solver silently gives wrong answers. | Spectral solver is correct only for isotropic, uniform-grid, constant-coefficient problems. No runtime validation. |
| M5 | `pcg.py:161` | **Warm start uses `x.abs().sum() > 0` check.** This computes the L1 norm of x just to check if it's zero — O(N) work every step. Should use `self._last_solution is not None and self.use_warm_start` check instead. | Minor performance waste, but called every time step. |

### Minor

| # | Location | Description | Impact |
|---|----------|-------------|--------|
| m1 | `fdm.py:88` | Default `chi=1400.0` in constructor contradicts the workaround chi=1.0. Default should be 1.0 to match the "D contains chi*Cm" convention, or the API should not accept chi at all. | API trap — new users will get wrong results. |
| m2 | `conductivity.py:33` | D_i_field tuple order documented as `(Dxx_i, Dyy_i, Dxy_i)` but `fdm.py:108-117` unpacks as `(Dxx, Dxy, Dyy)` via `_get_D_components` returning `(Dxx, Dxy, Dyy)`. If a user passes a D_field following the docstring order, Dxy and Dyy will be swapped. | Docstring/code mismatch. Latent bug if D_field is used with anisotropy. |
| m3 | `state.py:78-87` | `__post_init__` overwrites stim data when `stim_masks is None`. A caller who passes stim_masks=None but non-empty stim_starts would get them silently cleared. | Edge case, fragile. |
| m4 | `decoupled.py:101-129` | `_apply_pinning` uses `import torch` inside the method body. Module only imports torch under TYPE_CHECKING. Should have top-level import. | Works because torch is imported by callers, but incorrect module structure. |
| m5 | `bidomain.py:264` | `stim_amplitudes_e` always set to 0.0. No way to specify extracellular stimulus through the factory, even though `BidomainState` supports it. | Feature gap, not bug. |
| m6 | `pcg_spectral.py:90` | `self._precond.solve(A, r)` passes A to spectral solver which ignores it. Functionally harmless but semantically confusing. | API confusion only. |
| m7 | `boundary.py:136` | `edge.value in bath_edges` allows passing string values like `"left"` as well as Edge enums. Implicit type coercion. | Could lead to subtle bugs if mixed types are used. |
| m8 | `structured.py:~109` | **`StructuredGrid.to()` loses `boundary_spec`.** When moving a grid to a new device, `boundary_spec` is not passed to the new grid — the new grid gets default `BoundarySpec.insulated()` from `__post_init__`, silently losing custom BCs. | Any GPU migration of a non-default BC grid would break. |
| m9 | `backend.py:132` | Print banner says "Engine V5.4 Backend" — copy artifact. Should say "Bidomain Engine V1". | Cosmetic only. |
| m10 | `calcium.py:180` | `update_concentrations` return type hint lists 5 tensor types but function returns 6 values (includes `RR_new`). | Type hint bug. |
| m11 | `chebyshev.py:146-160` | Both branches of `if self.use_jacobi_precond` call `_gershgorin_bounds(A)` identically — eigenvalue bounds don't account for Jacobi preconditioning of the system. | Eigenvalue estimates may be too loose for preconditioned Chebyshev iteration. |
| m12 | `test_mixed_spectral.py:228` | `__main__` block calls `test_mixed_t5_analytical_convergence()` but the function is named `test_mixed_t5_self_consistency_and_convergence()`. Would crash with `NameError`. | Test runner broken. |
| m13 | `test_mixed_spectral.py:189` | `'dy' in dir()` hack always evaluates to False. Should just use `dx` directly. | Sloppy but harmless. |
| m14 | `PROGRESS.md:199` | Test 5-T5 description says "Mixed BCs -> 'pcg_gmg'" but the actual test expects 'spectral'. | Documentation/code mismatch. |

---

## 2. File-by-File Analysis

### 2.1 `tissue_builder/mesh/structured.py`
Grid class storing Nx, Ny, Lx, Ly, BoundarySpec. Computes dx, dy, coordinates, domain_mask. Provides `grid_to_flat()` and `flat_to_grid()` conversion.

**Bug m8**: `to()` method doesn't propagate `boundary_spec` to the new grid — new grid gets default `BoundarySpec.insulated()`, silently losing custom BCs.

`edge_masks` property recomputes 4 tensors every call (no caching). Only called during setup, so acceptable for V1.

### 2.2 `tissue_builder/mesh/boundary.py`
Well-designed BC protocol. `BoundarySpec` with per-edge, per-variable BC types. Factory methods (`insulated()`, `bath_coupled()`, `bath_coupled_edges()`). Spectral eligibility computed from BC topology. Clean, no bugs.

**Bug m7**: Allows string values in bath_edges list.

### 2.3 `tissue_builder/tissue/conductivity.py`
`BidomainConductivity` dataclass with three modes (scalar, per-node field, fiber-based). Default values correct for human ventricular tissue. `get_effective_monodomain_D()` computes harmonic mean.

**Bug m2**: D_field tuple order mismatch with FDM unpacking.

### 2.4 `tissue_builder/stimulus/protocol.py`, `regions.py`
Standard stimulus protocol. `left_edge_region()` creates a region mask. Clean. No issues.

### 2.5 `simulation/classical/state.py`
`BidomainState` dataclass. Clean separation of Vm and phi_e. V property alias for IonicSolver compatibility. Stimulus data as lists (not tensors) — acceptable for V1.

**Bug m3**: `__post_init__` overwrites stim data fragility.

### 2.6 `simulation/classical/discretization/base.py`
`BidomainSpatialDiscretization` ABC providing two Laplacians, parabolic/elliptic operators. Well-designed interface matching V5.4 pattern. Correct abstraction level.

### 2.7 `simulation/classical/discretization/fdm.py`
9-point anisotropic stencil with face-based symmetric construction. Good design decision vs. ghost-node approach (documented in module docstring — asymmetric L breaks bidomain elliptic solve). Harmonic mean at interfaces for heterogeneous D.

**Bug C1**: Chi/Cm double-encoding (critical).
**Bug M1**: O(N) Python loop assembly (performance).
**Bug m2**: D_field tuple order mismatch.

The diagonal stencil entries (NE/NW/SE/SW) use `d_xy * cxy` with `cxy = 1/(4*dx*dy)`. This is the standard 9-point stencil for cross-derivative `d^2/(dx*dy)`. The sign pattern `(-d_xy, +d_xy, +d_xy, -d_xy)` for (NE, NW, SE, SW) is correct.

Face-based approach at boundaries: only interior faces contribute. At boundary nodes this gives half the strong-form stiffness (half control volume). Physically correct in the variational sense, and cancels in bidomain because both LHS and RHS use the same stiffness form.

### 2.8 `simulation/classical/solver/linear_solver/pcg.py`
Standard PCG with Jacobi preconditioning. Lazy workspace allocation. Warm start. Extract diagonal from sparse COO. Clean implementation.

**Bug M5**: O(N) zero check for warm start.

### 2.9 `simulation/classical/solver/linear_solver/spectral.py`
Unified spectral solver supporting per-axis DCT/DST/FFT. DST-I implemented via odd-extension FFT — correct. Eigenvalue formulas verified for all three BC types.

**Bug M4**: No validation that problem matches spectral assumptions.

The null space handling (`eigenvalues[0,0] = 1.0`, `u_hat[0,0] = 0.0`) is correct for Neumann-Neumann and periodic BCs.

### 2.10 `simulation/classical/solver/linear_solver/pcg_spectral.py`
PCG with spectral preconditioner. Clean composition. Spectral inaccuracy in preconditioner just slows convergence — does not corrupt solution. Good design.

### 2.11 `simulation/classical/solver/linear_solver/multigrid.py`, `pcg_gmg.py`
**Stubs only.** Multigrid falls back to PCG. No actual multigrid implementation. `pcg_gmg` creates a plain PCGSolver with max_iters=500.

### 2.12 `simulation/classical/solver/linear_solver/chebyshev.py`
Chebyshev polynomial iteration solver. Uses Gershgorin circle theorem for eigenvalue bounds. Correct implementation matching V5.4.

### 2.13 `simulation/classical/solver/splitting/base.py`, `strang.py`, `godunov.py`
Standard splitting strategies. Strang: half-ionic → full-diffusion → half-ionic. Godunov: ionic → diffusion. Clean, minimal.

### 2.14 `simulation/classical/solver/ionic_stepping/base.py`
IonicSolver ABC with `_evaluate_Istim()` and `step_with_V()` for LBM compatibility. Forward Euler default `_update_gates()`. Clean.

### 2.15 `simulation/classical/solver/ionic_stepping/rush_larsen.py`
Rush-Larsen exponential integrator. Uses OLD voltage for gate computation (matches V5.3).

**Bug C2**: No Cm division in voltage update.

**Note**: `compute_concentration_rates` at line 95 receives `V` which is actually the NEW V from the ionic step (state.V was updated at line 83). Concentration rates are computed with post-update V. This matches V5.3 behavior but is theoretically questionable.

### 2.16 `simulation/classical/solver/diffusion_stepping/decoupled.py`
Decoupled Gauss-Seidel bidomain diffusion. Parabolic solve for Vm, then elliptic solve for phi_e. Null space pinning for Neumann phi_e. Post-subtract centering.

**Bug M2**: Coupling term comment is misleading.
**Bug m4**: Missing top-level torch import.

The pinning strategy (identity row at pin_node, post-subtract `phi_e - phi_e[pin_node]`) is correct for removing the null space mode.

### 2.17 `simulation/classical/bidomain.py`
Top-level orchestrator. Factory functions for all components. Auto-selects elliptic solver based on BC topology (spectral > pcg_spectral > pcg_gmg).

**Bug M3**: Inline ForwardEulerSolver.
**Bug m5**: No extracellular stimulus support in factory.

Good composition pattern. Follows V5.4's MonodomainSimulation.

### 2.18 `ionic/` (all files)
Shared with V5.3 and LBM_V1. TTP06 and ORd implementations. IonicModel ABC with `compute_Iion`, `compute_gate_steady_states`, etc.

No bidomain-specific bugs. Ionic models are domain-agnostic.

---

## 3. Cross-Cutting Concerns: Chi/Cm Formulation

### 3.1 The Bidomain Equations

Standard bidomain:
```
chi * Cm * dVm/dt = div(sigma_i * grad(Vm + phi_e)) - chi * I_ion    (parabolic)
0 = div(sigma_i * grad(Vm + phi_e)) + div(sigma_e * grad(phi_e))      (elliptic)
```

### 3.2 Two Valid Formulations

**Formulation A (conductivity-based)**: Build L with sigma. Keep chi*Cm in time derivative.
```
L_sigma_i * V = div(sigma_i * grad(V))
A_para = chi*Cm/dt * I - theta * L_sigma_i      -- CORRECT
```

**Formulation B (diffusivity-based)**: Build L with D = sigma/(chi*Cm). Remove chi*Cm from time derivative.
```
L_D_i * V = div(D_i * grad(V))     where D_i = sigma_i / (chi*Cm)
A_para = 1/dt * I - theta * L_D_i               -- CORRECT
```

Both are mathematically equivalent. **The code mixes them:**
- `BidomainConductivity` stores D (Formulation B)
- `get_parabolic_operators` uses chi*Cm/dt (Formulation A)

This is Bug C1. The mismatch factor is chi*Cm = 1400.

### 3.3 The Workaround

Setting chi=1.0, Cm=1.0 makes `self._chi_Cm = 1.0`, collapsing both formulations:
```
A_para = 1.0/dt * I - theta * L_D_i     -- CORRECT for Formulation B
```

All Phase 6 tests use `CHI_NUM = 1.0, CM_NUM = 1.0`.

### 3.4 Why the Elliptic Equation is Unaffected

The elliptic equation `-(L_i + L_e) * phi_e = L_i * Vm` uses the same L on both sides. Whether L contains sigma or D, the chi*Cm factor cancels:
```
-(D_i + D_e) * lap(phi_e) = D_i * lap(Vm)
```
is equivalent to:
```
-(sigma_i + sigma_e) * lap(phi_e) = sigma_i * lap(Vm)
```

### 3.5 Correct Fix (Two Options)

**Option 1 (recommended): Adopt Formulation B fully.**
- Change default chi=1.0 in FDM constructor (or remove chi/Cm from constructor entirely)
- Keep D values in conductivity
- Change parabolic operators to use `1/dt`, not `chi*Cm/dt`
- Add `/Cm` to ionic step voltage update
- This is what LBM already does

**Option 2: Adopt Formulation A fully.**
- Store sigma_i, sigma_e in conductivity (not D)
- Build L with sigma
- Keep chi*Cm/dt in parabolic operators
- This is the standard textbook approach but requires changing BidomainConductivity

### 3.6 Comparison with LBM and Monodomain V5.4

| Aspect | Bidomain V1 (current) | LBM V1 | Monodomain V5.4 |
|--------|----------------------|--------|-----------------|
| What L contains | D = sigma/(chi*Cm) | N/A (tau encodes D) | D (in FDM) or sigma (in FEM with mass matrix) |
| Time derivative scaling | chi*Cm/dt (wrong with D-based L) | N/A (LBM implicit) | chi*Cm/dt (but with sigma-based K in FEM) |
| Source term | -(I_ion + I_stim) no /Cm | -(I_ion + I_stim)/Cm | Applied via operators |
| Chi in source | No | No (absorbed into D) | In operators (chi*I_ion) |
| Works correctly when | chi=1.0, Cm=1.0 | Always | Always |

---

## 4. Unit Convention Analysis

| Quantity | Units | Location | Convention |
|----------|-------|----------|------------|
| Vm, phi_e | mV | state.py | Standard |
| I_ion, I_stim | pA/pF = mV/ms | ionic model | Already per-capacitance |
| sigma_i, sigma_e | mS/cm | cv_shared.py | Standard cardiac |
| D_i, D_e | cm^2/ms | conductivity.py | = sigma/(chi*Cm) |
| chi | cm^-1 | fdm.py | 1400 real, 1.0 numerical |
| Cm | uF/cm^2 | fdm.py | 1.0 (always) |
| dx, dy | cm | grid | Standard |
| dt | ms | everywhere | Standard |
| t | ms | state.t | Standard |

**Key issue**: Two chi values coexist (`CHI_REAL=1400` for physics in cv_shared.py, `CHI_NUM=1.0` for numerics). This is a workaround for Bug C1. In a clean design, there should be one chi and the equations should be consistent.

**Two Cm problem**: Cell-level Cm (0.185 uF in TTP06 `parameters.py`) vs. tissue-level Cm (1.0 uF/cm^2). These are physically different quantities — cell Cm converts membrane current to charge flux for concentration dynamics inside the ionic model; tissue Cm scales the PDE. Both used correctly but the distinction is undocumented.

---

## 5. Design Issues

### D1. No conductivity-vs-diffusivity clarity
`BidomainConductivity` stores "D" values but the constructor parameter is named `D_i`, `D_e`. It's unclear whether these are sigma or D without reading the docstring. The FDM constructor's `chi=1400` default further confuses.

### D2. Multigrid stub
`multigrid.py` and `pcg_gmg.py` are stubs that fall back to plain PCG. The "Tier 3" solver doesn't exist. Should be clearly marked as TODO or removed.

### D3. No LBM integration
The bidomain engine has `simulation/lbm/__init__.py` but it's empty. No LBM path for bidomain.

### D4. Assembly performance
O(N) Python loop for Laplacian assembly. Should be vectorized for grids > 200x200. The numpy-based approach in the loop body (converting to numpy, iterating in Python, then back to torch) is the worst of both worlds.

### D5. No adaptive time stepping
dt is fixed throughout. The decoupled solver has `rebuild_operators()` but it's never called.

### D6. Strang splitting accuracy
The decoupled Gauss-Seidel approach lags phi_e, introducing O(dt) splitting error in the diffusion step even with Strang splitting. True second-order requires either iterating the coupling or using a monolithic solve. This is acceptable for V1 but should be documented as a known limitation.

### D7. Spectral solver D parameter duplication
`_build_linear_solver('spectral', ...)` passes `D = D_i + D_e` from conductivity. This duplicates the knowledge that the elliptic operator uses D_i + D_e. If conductivity changes, this could get out of sync.

### D8. No LBM bidomain
Cardiac bidomain LBM would require two distribution function fields coupled through the bidomain constraint. This is an open research question — not a design defect but worth noting for the merger.

### D9. Double ionic current computation
`compute_Iion()` and `compute_concentration_rates()` both compute all 12 TTP06 ionic currents independently. When called sequentially from `RushLarsenSolver.step()`, ionic currents are computed twice per step. A cached or combined method would halve the ionic computation work.

### D10. Private attribute access for conductivity
`bidomain.py` factory functions access `spatial._conductivity` (private) to get D values for spectral solver setup. Should be a public property on `BidomainSpatialDiscretization`.

### D11. Unused legacy DCTSolver/FFTSolver
`fft.py` contains monodomain-specific `DCTSolver` and `FFTSolver` with baked-in chi, Cm, dt, scheme. Not used by the bidomain engine. Adds maintenance burden and confusion.

### D12. Sparse COO format throughout
PyTorch COO sparse tensors are used for all matrices. CSR would be faster for repeated SpMV. However, PyTorch's CSR support is still maturing.

### D13. Diagonal stencil uses local Dxy, not harmonic mean
Cardinal directions in `fdm.py` use harmonic mean `_harm(d_xx, dxx[i+1,j])` for heterogeneous D, but diagonal entries use the local node's `d_xy` without averaging with the neighbor. Standard 9-point stencil derivation typically uses local values, so this may be intentional, but it differs from the cardinal direction treatment.

---

## 6. Unified Merger Proposal: Bidomain + Monodomain + LBM

### 6.1 Motivation

Three separate codebases solve related equations:
- **Monodomain V5.4**: `chi*Cm * dV/dt = div(sigma*grad(V)) - chi*I_ion`
- **Bidomain V1**: `chi*Cm * dVm/dt = div(sigma_i*grad(Vm+phi_e)) - chi*I_ion`
- **LBM V1**: Same monodomain PDE via Lattice Boltzmann

They share ionic models but differ in state variables, spatial discretization, chi/Cm formulation, and solver architecture. The goal is a unified codebase with consistent equations and units.

### 6.2 Design Philosophy

Follow V5.4's proven patterns:
1. **IonicModel = pure data provider** (shared across all paradigms)
2. **Spatial discretization provides operators** (paradigm-specific)
3. **Solvers consume operators** (paradigm-specific)
4. **State is runtime data** (paradigm-specific shapes, unified interface via properties)
5. **Zero allocation per step** (solvers own workspace)
6. **Physics layer computes derived quantities once** (sigma -> D at init, never again)

### 6.3 Unified Equation Convention

**Choose Formulation B (diffusivity-based) for everything:**

```
Monodomain:   dV/dt  = D * lap(V) - I_ion/Cm + I_stim/Cm
Bidomain:     dVm/dt = D_i * lap(Vm + phi_e) - I_ion/Cm + I_stim/Cm
              0      = div(D_i * grad(Vm)) + div((D_i + D_e) * grad(phi_e))
LBM:          dV/dt  = D * lap(V) + R,     R = -(I_ion + I_stim)/Cm
```

Where:
- `D = sigma / (chi * Cm)` — computed ONCE from physical parameters
- `I_ion` in pA/pF (= mV/ms) — from ionic model, already per-capacitance
- `Cm` = 1.0 uF/cm^2 by convention (ionic models already normalize by cell capacitance)
- Chi appears ONLY in D computation, never in operators or source terms

**Implication**: All operators (Laplacians, parabolic matrices, LBM tau) use D directly. The identity/mass term in parabolic operators is `1/dt * I`, NOT `chi*Cm/dt * I`.

### 6.4 Unified Unit Convention

| Quantity | Units | Where Defined | Used By |
|----------|-------|---------------|---------|
| sigma_i, sigma_e | mS/cm | User input | ConductivityConfig only |
| chi | cm^-1 | User input | ConductivityConfig only |
| Cm_tissue | uF/cm^2 | User input (=1.0) | ConductivityConfig + source term |
| D = sigma/(chi*Cm) | cm^2/ms | ConductivityConfig | All operators, LBM tau |
| V, Vm, phi_e | mV | State | Everywhere |
| I_ion, I_stim | pA/pF = mV/ms | IonicModel | Source term R |
| R = -(I_ion+I_stim)/Cm | mV/ms | Source computation | Diffusion step, LBM collision |
| dx, dy | cm | Grid | Discretization |
| dt | ms | Solver | Time stepping |

### 6.5 Proposed Architecture

```
cardiac_sim/
|-- ionic/                          # SHARED (unchanged from V5.4)
|   |-- base.py                     # IonicModel ABC
|   |-- lut.py
|   |-- ttp06/
|   +-- ord/
|
|-- physics/                        # NEW: equation/parameter layer
|   |-- conductivity.py             # ConductivityConfig (sigma -> D)
|   +-- equations.py                # Enum: MONODOMAIN | BIDOMAIN
|
|-- tissue_builder/                 # SHARED (unchanged from V5.4)
|   |-- mesh/
|   |   |-- base.py
|   |   |-- structured.py           # StructuredGrid (with BoundarySpec)
|   |   +-- triangular.py
|   |-- tissue/
|   |   +-- isotropic.py
|   +-- stimulus/
|       |-- protocol.py
|       +-- regions.py
|
|-- simulation/
|   |-- state.py                    # NEW: unified state protocol
|   |
|   |-- classical/
|   |   |-- state.py                # ClassicalState (flat 1D arrays)
|   |   |-- discretization/
|   |   |   |-- base.py             # SpatialDiscretization ABC
|   |   |   |-- fdm.py              # FDMDiscretization (unified mono+bi)
|   |   |   |-- fvm.py
|   |   |   +-- fem.py
|   |   |-- solver/
|   |   |   |-- splitting/          # SplittingStrategy ABC, Godunov, Strang
|   |   |   |-- ionic/              # IonicSolver ABC, RushLarsen, ForwardEuler
|   |   |   |-- diffusion/
|   |   |   |   |-- monodomain/     # CN, BDF1, BDF2, FE, RK2, RK4
|   |   |   |   +-- bidomain/       # DecoupledSolver
|   |   |   +-- linear/             # PCG, Chebyshev, Spectral, PCGSpectral
|   |   |-- monodomain.py           # MonodomainSimulation
|   |   +-- bidomain.py             # BidomainSimulation
|   |
|   +-- lbm/
|       |-- state.py                # LBMState (grid-shaped f, ionic_states)
|       |-- lattice/
|       |   |-- base.py, d2q5.py, d2q9.py, d3q7.py
|       |-- collision/
|       |   |-- bgk.py
|       |   +-- mrt/ (d2q5.py, d2q9.py)
|       |-- streaming/ (d2q5.py, d2q9.py)
|       |-- boundary/ (neumann.py, dirichlet.py, absorbing.py, masks.py)
|       |-- step.py
|       |-- solver/
|       |   +-- rush_larsen.py      # LBM-specific ionic step
|       +-- monodomain.py           # LBMSimulation
|
+-- utils/
    +-- backend.py
```

### 6.6 Key Unification Points

#### 6.6.1 ConductivityConfig (replaces BidomainConductivity + diffusion.sigma_to_D)

```python
@dataclass
class ConductivityConfig:
    """Unified conductivity specification.

    User provides sigma (conductivity) + chi + Cm.
    D (diffusivity) is computed internally: D = sigma / (chi * Cm).

    For monodomain: sigma -> D
    For bidomain:   sigma_i, sigma_e -> D_i, D_e
    """
    sigma_i: float = 1.74     # mS/cm (intracellular)
    sigma_e: float = 6.25     # mS/cm (extracellular, ignored for monodomain)
    chi: float = 1400.0       # cm^-1 (surface-to-volume ratio)
    Cm: float = 1.0           # uF/cm^2 (membrane capacitance)

    # Computed in __post_init__:
    D_i: float = field(init=False)   # sigma_i / (chi * Cm)
    D_e: float = field(init=False)   # sigma_e / (chi * Cm)

    def __post_init__(self):
        self.D_i = self.sigma_i / (self.chi * self.Cm)
        self.D_e = self.sigma_e / (self.chi * self.Cm)

    @property
    def D_mono(self):
        """Effective monodomain diffusivity = D_i * D_e / (D_i + D_e)."""
        return self.D_i * self.D_e / (self.D_i + self.D_e)

    @property
    def D_sum(self):
        """D_i + D_e for elliptic operator."""
        return self.D_i + self.D_e
```

**Key**: The user provides physical parameters. D is computed once. All downstream code uses only D. Chi and Cm never appear in operators, source terms, or solver logic.

#### 6.6.2 Unified IonicSolver

The IonicSolver ABC is shared across monodomain, bidomain, and LBM:

```python
class IonicSolver(ABC):
    def step(self, V, ionic_states, dt, I_stim=None):
        """Update V (in-place) and ionic states."""
        ...

    def compute_source(self, V, ionic_states, I_stim=None, Cm=1.0):
        """Compute source term R = -(I_ion + I_stim) / Cm."""
        I_ion = self.ionic_model.compute_Iion(V, ionic_states)
        return -(I_ion + (I_stim if I_stim is not None else 0.0)) / Cm

    def step_ionic_only(self, V, ionic_states, dt):
        """Update only gates + concentrations (V unchanged). For LBM."""
        ...
```

For LBM, call `step_ionic_only()` (no V update) and `compute_source()` separately. This eliminates the current design where LBM calls `model.step()` and discards V_new.

#### 6.6.3 Unified FDM Discretization

```python
class FDMDiscretization(SpatialDiscretization):
    """Unified FDM for monodomain and bidomain.

    Builds Laplacians with D (diffusivity). NO chi or Cm anywhere.
    Parabolic operator: A = 1/dt * I - theta * L    (NOT chi*Cm/dt!)
    """

    def __init__(self, grid, conductivity: ConductivityConfig,
                 equation: Equation = Equation.MONODOMAIN):
        # NOTE: No chi or Cm parameters. D comes from ConductivityConfig.
        ...
```

#### 6.6.4 LBM Integration with ConductivityConfig

```python
# LBM uses same ConductivityConfig:
D = config.D_mono               # for monodomain
tau = 0.5 + D * dt / (cs2 * dx**2)
R = -(I_ion + I_stim) / config.Cm   # = -(I_ion + I_stim) since Cm=1.0
```

No chi in LBM source term. No chi in tau. Chi was absorbed into D via ConductivityConfig.

### 6.7 Migration Path

#### Phase 1: Unify physics layer
1. Create `physics/conductivity.py` with `ConductivityConfig`
2. Update all three codebases to use `ConductivityConfig` for D computation
3. Remove chi/Cm from FDM constructor — use ConductivityConfig instead
4. Fix parabolic operators: `1/dt * I` not `chi*Cm/dt * I`

#### Phase 2: Unify ionic layer
5. Adopt V5.4's IonicModel ABC as the single ionic interface
6. Add `step_ionic_only()` to IonicSolver for LBM use
7. Fix LBM ionic_step to not call `model.step()` (use compute functions directly)
8. Add `/Cm` to all ionic step voltage updates

#### Phase 3: Unify FDM discretization
9. Create unified `FDMDiscretization` supporting both mono and bidomain
10. The distinction is whether we build 1 Laplacian (mono) or 2 (bidomain)
11. Vectorize Laplacian assembly (eliminate O(N) Python loop)
12. Merge face-based stencil approach from bidomain into monodomain FDM

#### Phase 4: Unify solvers
13. Merge PCG, Chebyshev, Spectral solvers (nearly identical across codebases)
14. Unify splitting strategies (identical ABC, identical implementations)
15. Create BidomainDiffusionSolver as extension of DiffusionSolver pattern

#### Phase 5: Unify orchestrators
16. MonodomainSimulation and BidomainSimulation share factory pattern
17. LBMSimulation shares ConductivityConfig and ionic solver
18. Single `cardiac_sim` package with `simulation.classical.monodomain`, `simulation.classical.bidomain`, `simulation.lbm.monodomain`

#### Phase 6: LBM enhancements
19. Wire MRT into LBMSimulation orchestrator
20. Fix D2Q5 MRT equilibrium coefficient (e_eq should be -2/3, not -4/3)
21. Add irregular domain support to LBM orchestrator
22. Add off-diagonal diffusion (D_xy) support via s_pxy in MRT

### 6.8 Validation Strategy

Each phase must pass:
1. **Regression**: All existing tests from all three codebases (38 bidomain, 24 LBM, V5.4 suite)
2. **Cross-validation**: Monodomain FDM ~ LBM ~ Bidomain (D_e -> infinity limit)
3. **Unit consistency**: `ConductivityConfig` produces identical D values as current implementations
4. **Chi/Cm correctness**: Verify single source of truth, no double-encoding

### 6.9 Chi/Cm Consistency Checklist

After merger, verify these invariants:

- [ ] Chi appears ONLY in `ConductivityConfig.__post_init__` (D = sigma/(chi*Cm))
- [ ] Cm appears ONLY in `ConductivityConfig.__post_init__` AND source term R = -I_ion/Cm
- [ ] No chi or Cm in any Laplacian, operator matrix, or LBM tau computation
- [ ] `FDMDiscretization` does NOT take chi or Cm as parameters
- [ ] Parabolic operator uses `1/dt * I`, NOT `chi*Cm/dt * I`
- [ ] LBM source term uses Cm from ConductivityConfig, not a separate parameter
- [ ] Ionic step voltage update includes `/Cm` (even when Cm=1.0, for correctness)
- [ ] Default ConductivityConfig produces correct D from standard tissue parameters
- [ ] Elliptic operator `-(L_i + L_e)` is unaffected (chi*Cm cancels anyway)
- [ ] All three paradigms produce same CV for same physical parameters

### 6.10 Open Questions

1. **Should Cm always be 1.0?** TTP06/ORd already normalize currents by cell capacitance (pA/pF). Using Cm != 1.0 would require knowing whether the ionic model's output is already normalized. Safest: keep Cm=1.0 by convention, document that ionic models must provide pA/pF output.

2. **LBM bidomain?** Would require two distribution function fields (f_i, f_e) coupled through the bidomain constraint. This is an open research area — defer to future work.

3. **FVM in the merger?** V5.4 has FVMDiscretization. Include it — the `SpatialDiscretization` ABC already abstracts FEM/FDM/FVM.

4. **3D support?** V5.4's LBM has D3Q7. The merger should preserve this. FDM 9-point stencil generalizes to 27-point. Architecture should be dimension-agnostic where possible.

5. **Which FDM stencil for monodomain? (DEFERRED)**

   Two approaches exist in the codebase:
   - **Ghost-node (V5.4 monodomain)**: At boundary node i=N-1, ghost at i+1 mirrors to i-1, doubling the connection weight. Gives `L[N-1,N-2] = 2w` but `L[N-2,N-1] = w` — asymmetric. Produces correct strong-form Laplacian at boundaries.
   - **Face-based (Bidomain V1)**: Each interior face contributes equally to both adjacent nodes. Out-of-domain faces skipped (zero flux = Neumann). Symmetric L with zero row sum. Gives half stiffness at boundary nodes (half control volume).

   **Trade-offs**:
   | Property | Ghost-node | Face-based |
   |----------|-----------|------------|
   | Symmetry | Asymmetric | Symmetric (SPD) |
   | Boundary accuracy | Full strong-form | Half (variational) |
   | Monodomain safe? | Yes (identity dominates) | Yes |
   | Bidomain safe? | NO (non-SPD breaks PCG) | Yes |
   | Global convergence | O(h²) | O(h²) |

   **Recommendation**: Use face-based everywhere for consistency. The half-stiffness at boundaries is physically correct in the variational sense (half control volume) and cancels in bidomain because both LHS and RHS use the same form. For monodomain, the difference is negligible since the identity term dominates.

   **Decision**: Deferred to merger implementation. Document both approaches in the merged `SpatialDiscretization` and decide based on validation results.

# Bidomain Diffusion Splitting — Architecture & Implementation Design

## Table of Contents
1. [Background: Why Bidomain Diffusion is Harder](#1-background)
2. [The Six Strategies](#2-the-six-strategies)
3. [Architecture Design](#3-architecture-design)
4. [Strategy Specifications](#4-strategy-specifications)
5. [Linear Solver Compatibility Matrix](#5-linear-solver-compatibility-matrix)
6. [Implementation Order & Validation](#6-implementation-order--validation)
7. [Task List](#7-task-list)

---

## 1. Background

### Monodomain: One PDE, One Unknown

```
∂V/∂t = D·∇²V + R          (reaction-diffusion)
```

Diffusion step is a single N×N linear system. Well-conditioned (identity dominates).
Forward Euler, Crank-Nicolson, BDF — all straightforward.

### Bidomain: Two Coupled PDEs, Two Unknowns

```
∂Vm/∂t  = L_i·Vm + L_i·φ_e + R      [parabolic — has ∂/∂t]
      0 = L_i·Vm + (L_i+L_e)·φ_e     [elliptic  — NO ∂/∂t]
```

Where `L_i = D_i·∇²` and `L_e = D_e·∇²` (Laplacians with diffusivity baked in).

**The complication**: Vm and φ_e are coupled. The parabolic equation contains φ_e,
and the elliptic equation contains Vm. Different strategies handle this coupling
differently — that's what the six files encode.

### Key Properties

| Property | Parabolic (Vm) | Elliptic (φ_e) |
|----------|---------------|-----------------|
| Time derivative | Yes | **No** |
| Can use Forward Euler | Yes (CFL-limited) | **Never** (must solve system) |
| Identity term in operator | Yes (1/dt dominates) | **No** |
| Condition number | ~1 (easy) | O(1/h²) (hard) |
| Fraction of compute | 10-20% | **60-80%** |
| Null space | Never | Yes (all-Neumann) |

**Bottom line**: The elliptic solve for φ_e is always the bottleneck. Different
strategies optimize different aspects: simplicity, accuracy, parallelism, or
eliminating linear solves entirely.

---

## 2. The Six Strategies

### Overview

```
diffusion_stepping/
├── base.py                    # BidomainDiffusionSolver ABC (existing)
├── decoupled_gs.py            # [1] Gauss-Seidel sequential (current, renamed)
├── semi_implicit.py           # [2] Forward Euler parabolic + elliptic solve
├── decoupled_jacobi.py        # [3] Jacobi parallel splitting
├── coupled.py                 # [4] Monolithic 2N×2N block solve
├── imex_sbdf2.py              # [5] IMEX multistep (SBDF2, 2nd order)
└── explicit_rkc.py            # [6] Fully explicit Runge-Kutta-Chebyshev
```

### Comparison

| # | Strategy | Parabolic Solve | Elliptic Solve | Order | Linear Solves/Step | GPU Score |
|---|----------|----------------|----------------|-------|-------------------|-----------|
| 1 | Gauss-Seidel | Implicit (PCG) | Implicit (PCG/Spec) | 1st | 2 | Good |
| 2 | Semi-Implicit | **None** (SpMV) | Implicit (PCG/Spec) | 1st | **1** | Better |
| 3 | Jacobi | Implicit (PCG) | Implicit (parallel) | 1st | 2 (parallel) | Better |
| 4 | Coupled | N/A (2N×2N GMRES) | N/A (in block) | 2nd | 1 (but 2N×2N) | Poor |
| 5 | IMEX SBDF2 | Single combined solve | Built-in | **2nd** | 1 | Good |
| 6 | Explicit RKC | **None** | **None** | 2nd | **0** | **Best** |

### When to Use Each

| Strategy | Best For |
|----------|---------|
| **Gauss-Seidel** | Baseline, large dt, reference solutions |
| **Semi-Implicit** | Simplest code, small dt (CFL OK), GPU |
| **Jacobi** | GPU parallelism (both solves independent) |
| **Coupled** | Maximum accuracy, research validation |
| **IMEX SBDF2** | Production 2nd-order, no splitting error |
| **Explicit RKC** | Maximum GPU throughput, no linear solver dependencies |

---

## 3. Architecture Design

### Current Pattern (mirrors monodomain)

```
BidomainSimulation          (bidomain.py — orchestrator)
  └── SplittingStrategy     (splitting/ — Strang or Godunov)
        ├── IonicSolver      (ionic_stepping/ — Rush-Larsen or FE)
        └── BidomainDiffusionSolver   (diffusion_stepping/ — THIS IS WHAT WE'RE EXPANDING)
              ├── LinearSolver(s)      (linear_solver/ — PCG, Chebyshev, Spectral, etc.)
              └── SpatialDiscretization (discretization/ — provides L_i, L_e)
```

### ABC Update

The current `BidomainDiffusionSolver` ABC is minimal and sufficient. No changes needed:

```python
class BidomainDiffusionSolver(ABC):
    def __init__(self, spatial, dt):
        self._spatial = spatial
        self._dt = dt

    @abstractmethod
    def step(self, state: BidomainState, dt: float) -> None:
        """Advance Vm and phi_e by dt. Modifies state in-place."""

    def rebuild_operators(self, spatial, dt) -> None:
        """Rebuild when dt changes (adaptive stepping)."""
```

Each strategy implements `step()` differently. Some need linear solvers (injected
via constructor), some don't. Constructor signatures differ per strategy — this is
fine, the factory in `bidomain.py` handles construction.

### Factory Registration (bidomain.py)

```python
def _build_diffusion_solver(name, spatial, dt, para_ls, ellip_ls, theta):
    if name == 'decoupled_gs':
        return DecoupledGaussSeidelSolver(spatial, dt, para_ls, ellip_ls, theta)
    elif name == 'semi_implicit':
        return SemiImplicitSolver(spatial, dt, ellip_ls)
    elif name == 'decoupled_jacobi':
        return DecoupledJacobiSolver(spatial, dt, para_ls, ellip_ls, theta)
    elif name == 'coupled':
        return CoupledBlockSolver(spatial, dt, theta)
    elif name == 'imex_sbdf2':
        return IMEXSBDF2Solver(spatial, dt, ellip_ls)
    elif name == 'explicit_rkc':
        return ExplicitRKCSolver(spatial, dt, n_stages=40)
    # Legacy alias
    elif name == 'decoupled':
        return DecoupledGaussSeidelSolver(spatial, dt, para_ls, ellip_ls, theta)
```

---

## 4. Strategy Specifications

### Strategy 1: Gauss-Seidel Sequential (`decoupled_gs.py`)

**Status**: Exists as `decoupled.py`. Rename and keep as-is.

**Algorithm**:
```
Step 1 — Parabolic (implicit, Crank-Nicolson):
    A_para · Vm^{n+1} = B_para · Vm^n + L_i · φ_e^n
    where A_para = 1/dt · I - θ·L_i,  B_para = 1/dt · I + (1-θ)·L_i

Step 2 — Elliptic:
    A_ellip · φ_e^{n+1} = L_i · Vm^{n+1}    (uses JUST-COMPUTED Vm)
    where A_ellip = -(L_i + L_e)
```

**Constructor**: `(spatial, dt, parabolic_solver, elliptic_solver, theta=0.5, pin_node=0)`

**Properties**:
- Two sequential linear solves per step
- Parabolic is well-conditioned (converges in 2-5 iters)
- Using updated Vm in elliptic reduces splitting error vs Jacobi
- No CFL constraint (unconditionally stable)
- Splitting error: O(dt), coupling error from lagging φ_e

**Notes**: theta=0.5 gives Crank-Nicolson, theta=1.0 gives Backward Euler.

---

### Strategy 2: Semi-Implicit (`semi_implicit.py`)

**Status**: New file.

**Algorithm**:
```
Step 1 — Parabolic (explicit, Forward Euler — NO linear solve):
    Vm^{n+1} = Vm^n + dt · (L_i · Vm^n + L_i · φ_e^n + R^n)
    This is just: SpMV + SpMV + vector_add  (no solver needed)

Step 2 — Elliptic (implicit):
    A_ellip · φ_e^{n+1} = L_i · Vm^{n+1}
    where A_ellip = -(L_i + L_e)
```

**Constructor**: `(spatial, dt, elliptic_solver, pin_node=0)`

No parabolic solver needed! Only one linear solve per step.

**Properties**:
- **Simplest implementation** — one matrix-free step + one linear solve
- CFL constraint: dt < dx² / (4·D_i). For dx=0.025, D_i=0.00124: dt < 0.126 ms
- At dt=0.01 ms (our default), CFL ratio = 0.08 — well within limit
- Same elliptic solve as Gauss-Seidel (same bottleneck)
- Splitting error: O(dt)

**Implementation sketch**:
```python
class SemiImplicitSolver(BidomainDiffusionSolver):
    def __init__(self, spatial, dt, elliptic_solver, pin_node=0):
        super().__init__(spatial, dt)
        self.elliptic_solver = elliptic_solver
        self.A_ellip = spatial.get_elliptic_operator()
        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node
        if self._needs_pinning:
            self._apply_pinning(self.A_ellip, pin_node)

    def step(self, state, dt):
        # Step 1: Explicit parabolic (no linear solve)
        LiVm = self._spatial.apply_L_i(state.Vm)
        LiPhi = self._spatial.apply_L_i(state.phi_e)
        Vm_new = state.Vm + dt * (LiVm + LiPhi)

        # Step 2: Elliptic solve
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)
        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)
```

**CFL stability check** (in constructor):
```python
D_i = spatial.conductivity.D_i
dx = spatial.grid.dx
dt_cfl = dx**2 / (4 * D_i)
if dt > dt_cfl:
    raise ValueError(
        f"Semi-implicit CFL violated: dt={dt} > dt_max={dt_cfl:.4f}. "
        f"Use decoupled_gs (implicit) or reduce dt."
    )
```

---

### Strategy 3: Jacobi Parallel (`decoupled_jacobi.py`)

**Status**: New file.

**Algorithm**:
```
Step 1 — Parabolic (uses φ_e^n, NOT φ_e^{n+1}):
    A_para · Vm^{n+1} = B_para · Vm^n + L_i · φ_e^n

Step 2 — Elliptic (uses Vm^n, NOT Vm^{n+1}):     ← KEY DIFFERENCE FROM GS
    A_ellip · φ_e^{n+1} = L_i · Vm^n
```

Both solves use ONLY old values → can run in parallel on GPU.

**Constructor**: `(spatial, dt, parabolic_solver, elliptic_solver, theta=0.5, pin_node=0)`

**Properties**:
- Two parallel linear solves per step (GPU advantage)
- More splitting error than Gauss-Seidel (both lag their coupling variable)
- Same stability (Fernandez & Zemzemi 2010 proved energy stability preserved)
- Useful when GPU occupancy matters (two independent solves fill the GPU)

**Implementation sketch**:
```python
def step(self, state, dt):
    # Both use OLD values — can be parallelized
    rhs_para = sparse_mv(self.B_para, state.Vm) \
               + self._spatial.apply_L_i(state.phi_e)  # phi_e^n
    rhs_ellip = self._spatial.apply_L_i(state.Vm)       # Vm^n

    # These two solves are independent — GPU can run them concurrently
    Vm_new = self.parabolic_solver.solve(self.A_para, rhs_para)
    phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

    state.Vm.copy_(Vm_new)
    state.phi_e.copy_(phi_e_new)
```

---

### Strategy 4: Coupled Monolithic (`coupled.py`)

**Status**: New file. Research/validation only — not for production.

**Algorithm**:
Assemble the full 2N×2N block system and solve with GMRES or MINRES:

```
[A11  A12] [Vm^{n+1}  ]   [b1]
[A21  A22] [φ_e^{n+1} ] = [b2]

where:
  A11 = 1/dt · I - θ·L_i         (parabolic diagonal block)
  A12 = -θ·L_i                    (parabolic coupling)
  A21 = L_i                       (elliptic coupling)
  A22 = -(L_i + L_e)             (elliptic diagonal block)
  b1  = (1/dt · I + (1-θ)·L_i) · Vm^n + (1-θ)·L_i · φ_e^n
  b2  = 0
```

**Constructor**: `(spatial, dt, theta=0.5, max_iters=200, tol=1e-8)`

**Properties**:
- **No splitting error** — fully coupled, 2nd order in time with CN
- Matrix is **indefinite** (saddle-point) — cannot use CG
- Need GMRES or MINRES (with block-diagonal preconditioner)
- 2N×2N system — more memory, larger Krylov subspace
- Cervi & Bhatt (2017): Can be 80% faster than decoupled with good preconditioner
- Not GPU-friendly (GMRES requires many global syncs)

**Block preconditioner**:
```
M = [A11   0 ]
    [ 0   A22]
```
Each block solved by its own LinearSolver (Spectral for A22, PCG for A11).

**Implementation notes**:
- Use `torch.linalg.solve` or implement GMRES with block preconditioner
- This is primarily for validation — confirming that decoupled methods match
  the "exact" coupled solution within splitting error tolerance

---

### Strategy 5: IMEX SBDF2 (`imex_sbdf2.py`)

**Status**: New file.

**Algorithm**:
Second-order semi-implicit BDF (SBDF2). Treats diffusion implicitly, reaction
explicitly. No operator splitting — single unified step.

```
Time discretization:
  (3/(2dt)) · Vm^{n+1} - L_i · Vm^{n+1} - L_i · φ_e^{n+1}
      = (2/dt) · Vm^n - (1/(2dt)) · Vm^{n-1} + 2·R^n - R^{n-1}

  L_i · Vm^{n+1} + (L_i + L_e) · φ_e^{n+1} = 0

Decoupled form:
  Step 1: (3/(2dt) · I - L_i) · Vm^{n+1}
          = (2/dt)·Vm^n - (1/(2dt))·Vm^{n-1} + L_i·φ_e^n + 2·R^n - R^{n-1}
  Step 2: -(L_i + L_e) · φ_e^{n+1} = L_i · Vm^{n+1}
```

**Constructor**: `(spatial, dt, parabolic_solver, elliptic_solver, pin_node=0)`

**Properties**:
- **2nd order** in time (vs 1st order for GS/semi-implicit)
- Requires storing Vm^{n-1} and R^{n-1} (two extra N-vectors)
- Not self-starting — first step must use 1st-order method (backward Euler)
- No CFL constraint (fully implicit diffusion)
- No operator splitting error (reaction extrapolated, not split)
- L-stable: no spurious oscillations (unlike CN which can oscillate)

**Workspace**: `Vm_prev`, `R_prev` (two extra N-vectors, allocated once)

---

### Strategy 6: Explicit RKC (`explicit_rkc.py`)

**Status**: New file. Research/experimental.

**Algorithm**:
Runge-Kutta-Chebyshev with s stages. Extends explicit stability region via
Chebyshev polynomial recursion. **No linear solves at all.**

```
For s stages (s ≈ 30-50):
    W_0 = u^n
    W_1 = W_0 + (dt·μ_1)·F(W_0)
    For j = 2, ..., s:
        W_j = (1 - μ_j - ν_j)·W_0 + μ_j·W_{j-1} + ν_j·W_{j-2} + dt·μ̃_j·F(W_{j-1})
    u^{n+1} = W_s

where F(u) applies both L_i and L_e stencils (matrix-free).
μ, ν, μ̃ are Chebyshev recursion coefficients.
```

**Constructor**: `(spatial, dt, n_stages=40, damping=0.05)`

**Properties**:
- **Zero linear solves** — every stage is just stencil application
- Perfect GPU parallelism (no sync points, no Krylov iterations)
- Stability region: O(s²) larger than Forward Euler
- For s=40: stable for dt ≈ 40²·dt_FE = 1600·dt_FE (massive CFL extension)
- 2nd order in time
- Each stage costs one L_i + one L_e application (matrix-free on GPU)
- Total cost per step: ~40 SpMVs — comparable to 20 PCG iterations

**Trade-off**:
- 40 stencil evaluations vs 1 spectral solve (spectral wins for isotropic)
- But: no FFT, no preconditioner setup, no memory for Krylov vectors
- Scales better to 3D and very large grids

**Stability coefficients** (from Verwer et al. 2004):
```python
def rkc_coefficients(s, damping=0.05):
    """Compute RKC coefficients for s stages with damping."""
    w0 = 1.0 + damping / s**2
    w1 = chebyshev_T(s, w0) / chebyshev_Tp(s, w0)  # T_s(w0) / T'_s(w0)
    # ... Chebyshev recursion for mu, nu, mu_tilde, gamma_tilde
```

---

## 5. Linear Solver Compatibility Matrix

Not all strategies need all solver types:

| Strategy | Needs Parabolic Solver | Needs Elliptic Solver | Compatible Solvers |
|----------|----------------------|----------------------|-------------------|
| Gauss-Seidel | Yes | Yes | PCG, Chebyshev, Spectral, PCG+Spectral |
| Semi-Implicit | **No** | Yes | PCG, Chebyshev, Spectral, PCG+Spectral |
| Jacobi | Yes | Yes | PCG, Chebyshev, Spectral, PCG+Spectral |
| Coupled | N/A | N/A | GMRES (internal) |
| IMEX SBDF2 | Yes | Yes | PCG, Chebyshev, Spectral, PCG+Spectral |
| Explicit RKC | **No** | **No** | N/A (matrix-free) |

### Recommended Solver Pairings

| Strategy | Parabolic | Elliptic | Notes |
|----------|-----------|----------|-------|
| GS (default) | PCG | Spectral (Tier 1) | Standard baseline |
| GS (anisotropic) | PCG | PCG+Spectral (Tier 2) | For D_xx ≠ D_yy |
| Semi-Implicit | — | Spectral | Fastest for isotropic |
| Semi-Implicit (GPU) | — | Chebyshev | Zero-sync elliptic |
| Jacobi (GPU) | Chebyshev | Chebyshev | Both sync-free |
| SBDF2 | PCG | Spectral | Best accuracy/cost |
| Explicit RKC | — | — | Maximum GPU throughput |

---

## 6. Implementation Order & Validation

### Phase Order

Implement simplest first, use each as validation baseline for the next:

```
[1] semi_implicit.py    ← Simplest (one SpMV + one solve). Validates against GS.
[2] decoupled_gs.py     ← Rename existing decoupled.py. Already validated.
[3] decoupled_jacobi.py ← Minor variant of GS. Validates against GS.
[4] imex_sbdf2.py       ← 2nd order. Validates convergence rate vs GS (1st order).
[5] coupled.py          ← Reference solution. All others validate against this.
[6] explicit_rkc.py     ← Most complex. Validates against coupled.
```

### Validation Tests Per Strategy

Each strategy must pass:

| Test | Description | Criterion |
|------|-------------|-----------|
| S-T1 | Pure diffusion (no ionic): Gaussian variance growth | D_eff within 15% |
| S-T2 | Cross-check vs Gauss-Seidel baseline | Vm max-norm diff < 1e-3 at t=5ms |
| S-T3 | Cross-check vs coupled reference | Vm max-norm diff < splitting_order·dt |
| S-T4 | Full AP simulation: CV measurement | CV within 5% of reference |
| S-T5 | Neumann null space handling (phi_e bounded) | phi_e doesn't diverge |
| S-T6 | Dirichlet BC (bath-coupled, Kleber effect) | Boundary/center CV ratio > 1.0 |
| S-T7 | Conservation: total Vm integral preserved (no ionic) | < 1e-10 drift |

### Convergence Order Tests

| Strategy | Expected dt-convergence | Test |
|----------|------------------------|------|
| GS, Semi-Implicit, Jacobi | O(dt) — 1st order | Halve dt → halve error |
| SBDF2, Coupled, RKC | O(dt²) — 2nd order | Halve dt → quarter error |

---

## 7. Task List

### Phase 1: Rename & Validate Baseline

- [ ] **1.1**: Rename `decoupled.py` → `decoupled_gs.py`, update all imports
- [ ] **1.2**: Update `bidomain.py` factory to accept both `'decoupled'` (legacy) and `'decoupled_gs'`
- [ ] **1.3**: Verify all existing tests still pass with renamed file

### Phase 2: Semi-Implicit (simplest new strategy)

- [ ] **2.1**: Implement `semi_implicit.py` with CFL check in constructor
- [ ] **2.2**: Register in `bidomain.py` factory as `'semi_implicit'`
- [ ] **2.3**: Run S-T1 (Gaussian diffusion) — validates basic correctness
- [ ] **2.4**: Run S-T2 (cross-check vs GS) — validates splitting equivalence
- [ ] **2.5**: Run S-T4 (full AP, CV) — validates production readiness

### Phase 3: Jacobi Parallel

- [ ] **3.1**: Implement `decoupled_jacobi.py`
- [ ] **3.2**: Register in factory as `'jacobi'`
- [ ] **3.3**: Run S-T2 (cross-check vs GS) — expect slightly larger but bounded error
- [ ] **3.4**: Benchmark: GS sequential vs Jacobi parallel (GPU timing)

### Phase 4: IMEX SBDF2 (2nd order)

- [ ] **4.1**: Implement `imex_sbdf2.py` with backward Euler bootstrap step
- [ ] **4.2**: Register in factory as `'imex_sbdf2'`
- [ ] **4.3**: Run convergence order test — verify O(dt²) vs O(dt) for GS
- [ ] **4.4**: Run S-T4 (full AP) — verify accuracy improvement at same dt

### Phase 5: Coupled Monolithic (reference)

- [ ] **5.1**: Implement `coupled.py` with block GMRES
- [ ] **5.2**: Register in factory as `'coupled'`
- [ ] **5.3**: Run S-T3 for all strategies — quantify splitting error
- [ ] **5.4**: Document: coupled matches decoupled to O(dt·splitting_error)

### Phase 6: Explicit RKC (experimental)

- [ ] **6.1**: Implement RKC coefficient computation (Chebyshev recursion)
- [ ] **6.2**: Implement `explicit_rkc.py` with matrix-free stencil evaluation
- [ ] **6.3**: Validate stability: run at dt=0.1 ms with s=40 stages
- [ ] **6.4**: Run S-T4 (full AP) — validate accuracy
- [ ] **6.5**: Benchmark: RKC matrix-free vs Spectral direct solve (GPU timing)

### Phase 7: Fix Linear Solvers (parallel track)

- [ ] **7.1**: Fix Chebyshev solver (4 critical bugs from LINEAR_SOLVER_IMPLEMENTATION.md)
- [ ] **7.2**: Harden PCG (3 minor bugs)
- [ ] **7.3**: Deprecate fft.py
- [ ] **7.4**: Cross-validate: Chebyshev vs PCG vs Spectral on elliptic system

---

## Appendix A: Monodomain Parallel (Reference Pattern)

The monodomain Engine V5.4 uses the same directory pattern for its diffusion solvers:

```
Monodomain/Engine_V5.4/.../solver/
├── splitting/
│   ├── base.py        → SplittingStrategy ABC
│   ├── godunov.py     → GodunovSplitting
│   └── strang.py      → StrangSplitting
├── ionic_time_stepping/
│   ├── base.py        → IonicSolver ABC
│   ├── forward_euler.py
│   └── rush_larsen.py
└── diffusion_time_stepping/
    ├── base.py        → DiffusionSolver ABC
    ├── forward_euler.py
    ├── crank_nicolson.py
    ├── bdf1.py
    ├── bdf2.py
    ├── rk2.py
    └── rk4.py
```

Our bidomain `diffusion_stepping/` directly parallels monodomain's
`diffusion_time_stepping/`. Each file is a self-contained strategy implementing
the same ABC, selected by string config in the orchestrator.

## Appendix B: Interaction with Splitting Strategies

The splitting strategies (Strang/Godunov) are ORTHOGONAL to the diffusion
strategies. Splitting determines how ionic and diffusion steps interleave:

```
Godunov: ionic(dt) → diffusion(dt)
Strang:  ionic(dt/2) → diffusion(dt) → ionic(dt/2)
```

The diffusion strategy determines HOW the diffusion step solves the
parabolic-elliptic system. Any diffusion strategy works with any splitting:

```
StrangSplitting + SemiImplicitSolver       ← valid
GodunovSplitting + ExplicitRKCSolver       ← valid
StrangSplitting + CoupledBlockSolver       ← valid
```

## Appendix C: CFL Limits for Explicit Methods

| dx (cm) | D_i (cm²/ms) | dt_max (ms) | Our dt (ms) | Safety Ratio |
|---------|-------------|-------------|-------------|-------------|
| 0.025 | 0.00124 | 0.126 | 0.01 | 12.6x |
| 0.010 | 0.00124 | 0.020 | 0.01 | 2.0x |
| 0.005 | 0.00124 | 0.005 | 0.005 | 1.0x (limit!) |

At dx=0.025 (our default), Forward Euler is very stable. At dx=0.005 (fine mesh),
we'd need dt=0.005 or smaller — still feasible but tight. This is where IMEX or
implicit methods become attractive.

For Explicit RKC with s=40 stages, the effective CFL extends by s² ≈ 1600:
dt_max(RKC) ≈ 1600 × dx²/(4·D_i) ≈ 200 ms (essentially unlimited).

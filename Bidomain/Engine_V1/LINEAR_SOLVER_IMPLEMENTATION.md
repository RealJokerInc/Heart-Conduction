# Linear Solver Implementation Plan — Bidomain Engine V1

**Related**: See `DIFFUSION_SPLITTING_DESIGN.md` for the 6 diffusion splitting
strategies that consume these linear solvers.

## Table of Contents
1. [Current State Assessment](#1-current-state-assessment)
2. [Bug Catalog](#2-bug-catalog)
3. [Chebyshev Solver Rewrite](#3-chebyshev-solver-rewrite)
4. [PCG Hardening](#4-pcg-hardening)
5. [GPU Acceleration Strategy](#5-gpu-acceleration-strategy)
6. [Validation Plan](#6-validation-plan)
7. [Task List](#7-task-list)

---

## 1. Current State Assessment

### Files

| File | Role | Status |
|------|------|--------|
| `base.py` | LinearSolver ABC | Clean |
| `pcg.py` | Jacobi-preconditioned CG | Functional, 3 minor bugs |
| `chebyshev.py` | Chebyshev polynomial iteration | **Broken** — 4 critical bugs |
| `spectral.py` | Direct DCT/DST/FFT solver (Tier 1) | Functional, 1 major limitation |
| `pcg_spectral.py` | PCG + spectral preconditioner (Tier 2) | Functional, inherits PCG bugs |
| `fft.py` | Legacy DCT/FFT solver | **Broken** — chi/Cm baked in, wrong fallback |
| `multigrid.py` | GMG stub | NotImplementedError |
| `pcg_gmg.py` | PCG+GMG stub | NotImplementedError |

### Solver Architecture (Three Tiers)

```
Elliptic solve: -(L_i + L_e) · φ_e = L_i · Vm
                 ───────────────────
                 A_ellip (SPD, no identity term)

Tier 1: SpectralSolver     — O(N log N), zero iterations
         Requires: isotropic D, uniform grid, per-axis uniform BCs
         Transforms: DCT (Neumann), DST (Dirichlet), FFT (periodic)

Tier 2: PCGSpectralSolver  — PCG with spectral preconditioner, 1-5 iterations
         Requires: uniform grid, per-axis uniform BCs
         Handles: mild anisotropy (D_xx ≠ D_yy)

Tier 3: PCG / Chebyshev    — Iterative with Jacobi preconditioner
         Requires: nothing (any mesh, any BCs)
         Handles: arbitrary anisotropy, heterogeneity, mixed BCs
```

### Parabolic vs Elliptic

Both solves use the same LinearSolver interface. Key difference:

| Property | Parabolic | Elliptic |
|----------|-----------|----------|
| Matrix | `1/dt · I - θ·L_i` | `-(L_i + L_e)` |
| Identity term | Yes (dominates) | **No** |
| Condition number | ~1 (well-conditioned) | O(1/h²) (ill-conditioned) |
| Symmetry critical? | No (identity masks it) | **Yes** (non-SPD breaks PCG) |
| Null space | Never | Yes (all-Neumann phi_e) |
| Solver pressure | Low (few iterations) | **High** (many iterations or direct solve) |

The elliptic solve is the bottleneck (60-80% of compute). This is where Chebyshev / GPU acceleration matters most.

---

## 2. Bug Catalog

### 2.1 Chebyshev Solver — CRITICAL (4 bugs, solver is non-functional)

**Bug CH-1: Eigenvalue estimation ignores preconditioning** (L152-157)
```python
# CURRENT (broken): both branches identical
if self.use_jacobi_precond:
    self._lam_min, self._lam_max = _gershgorin_bounds(A, self.safety_margin)
else:
    self._lam_min, self._lam_max = _gershgorin_bounds(A, self.safety_margin)
```

**Problem**: When `use_jacobi_precond=True`, the iteration solves `D⁻¹Ax = D⁻¹b` where
`D = diag(A)`. The eigenvalue bounds must be for `D⁻¹A`, not `A`. Since `D⁻¹A` has
eigenvalues clustered around 1 (Jacobi normalizes the diagonal), Gershgorin bounds for
`D⁻¹A` are much tighter. Using bounds for `A` gives wildly wrong Chebyshev parameters.

**Fix**: Compute Gershgorin bounds for `D⁻¹A`:
```python
def _gershgorin_bounds_preconditioned(A, diag_inv, safety_margin=0.1):
    """Gershgorin bounds for D^{-1}A where D = diag(A)."""
    A = A.coalesce()
    indices, values = A.indices(), A.values()
    n = A.size(0)

    # Preconditioned row: row_i of D^{-1}A has entries a_{ij} / a_{ii}
    # Center = 1.0 (diagonal of D^{-1}A), radius = sum(|a_{ij}/a_{ii}|, j≠i)
    rows, cols = indices[0], indices[1]
    off_diag = rows != cols
    scaled_abs = values[off_diag].abs() * diag_inv[rows[off_diag]]
    radii = torch.zeros(n, device=A.device, dtype=A.dtype)
    radii.scatter_add_(0, rows[off_diag], scaled_abs)

    lam_min = max((1.0 - radii.max()).item() * (1 - safety_margin), 1e-10)
    lam_max = (1.0 + radii.max()).item() * (1 + safety_margin)
    return lam_min, lam_max
```

**Bug CH-2: Division by zero in first iteration** (L211)
```python
d.copy_(z / theta)  # theta can be 0 if lam_min = -lam_max
```
**Fix**: Guard `theta > 0` (guaranteed for SPD, but add assertion).

**Bug CH-3: Chebyshev coefficient overflow** (L227)
```python
rho_new = 1.0 / (2.0 * sigma - rho)  # Can overflow
```
**Fix**: Clamp `rho_new` and detect divergence.

**Bug CH-4: No warm start** (L194)
The solver always starts from `x = 0`. For time-stepping where the solution changes
slowly, warm-starting from the previous solution reduces iterations ~3-8x.

### 2.2 PCG Solver — MINOR (3 bugs, functionally correct)

**Bug PCG-1: Absolute pAp threshold** (L198)
```python
if pAp.abs() < 1e-30:  # Should be relative
```
**Fix**: `if pAp.abs() < 1e-14 * b_norm**2:` (scale-relative).

**Bug PCG-2: Redundant O(n) check** (L161)
```python
if x.abs().sum() > 0:  # O(n) just to check if x is nonzero
```
**Fix**: Track whether warm start was applied via a boolean flag.

**Bug PCG-3: Cache invalidation via `id(A)`** (L108)
```python
A_id = id(A)  # Python object ID, not content-based
```
If A is rebuilt (same structure, new object), preconditioner is unnecessarily recomputed.
The Chebyshev solver uses `data_ptr()` which is slightly better but still fragile.
**Fix**: Accept — this is minor. Both are rebuilt once per dt change, not per step.

### 2.3 Spectral Solver — MAJOR limitation

**Bug SP-1: Assumes isotropic D** (L120)
```python
self._eigenvalues = self.D * (LAM_X + LAM_Y)  # Only correct for scalar D
```
For anisotropic D: should be `D_xx * LAM_X + D_yy * LAM_Y`. Silently gives wrong
eigenvalues for anisotropic problems. Not a bug per se (the solver documents "isotropic
only"), but the code doesn't validate or fail.

**Fix**: Add runtime check `assert isinstance(D, (int, float))` or accept `(D_xx, D_yy)`.

### 2.4 FFT Solver (Legacy) — CRITICAL, recommend deprecation

**Bug FFT-1: chi/Cm baked into eigenvalues** (L86-131)
Ties the solver to specific formulation and time-stepping parameters. Inconsistent
with Formulation B (where chi doesn't appear in operators). The newer `spectral.py`
already replaces this correctly.

**Bug FFT-2: DCT fallback has wrong normalization** (L189-214)
The `_dct2_via_fft()` fallback uses custom normalization that doesn't match
`torch_dct.dct(norm='ortho')`. Gives silently wrong answers.

**Recommendation**: Mark `fft.py` as deprecated. All new code should use `spectral.py`.

---

## 3. Chebyshev Solver Rewrite

### 3.1 Algorithm

Standard 3-term Chebyshev iteration with Jacobi preconditioning:

```
Given: A (SPD), b (RHS), M = diag(A) (preconditioner)
       [λ_min, λ_max] = eigenvalue bounds of M⁻¹A

Setup:
    θ = (λ_max + λ_min) / 2
    δ = (λ_max - λ_min) / 2
    σ = θ / δ

Iteration:
    x₀ = warm_start or 0
    r₀ = b - A·x₀
    z₀ = M⁻¹·r₀

    d₀ = z₀ / θ
    x₁ = x₀ + d₀
    ρ₀ = 1/σ

    for k = 1, 2, ..., max_iters:
        rₖ = b - A·xₖ           # 1 SpMV (GPU kernel)
        zₖ = M⁻¹·rₖ             # 1 element-wise multiply
        ρₖ = 1/(2σ - ρₖ₋₁)
        dₖ = ρₖ·ρₖ₋₁·dₖ₋₁ + (2ρₖ/δ)·zₖ   # 2 AXPY (fused on GPU)
        xₖ₊₁ = xₖ + dₖ          # 1 AXPY

    return xₖ₊₁
```

**Key properties:**
- **Zero global reductions** per iteration (no dot products)
- Only SpMV + vector AXPY operations → perfectly GPU-parallel
- Fixed iteration count → no convergence check sync
- Cost per iteration ≈ 1 SpMV + 3 vector ops

### 3.2 Eigenvalue Estimation

Three strategies, selectable at construction:

| Strategy | Cost | Accuracy | When to Use |
|----------|------|----------|-------------|
| Gershgorin | O(nnz) | Conservative | Default (GPU-native, one pass) |
| Power iteration | ~15 SpMVs | Good λ_max | When Gershgorin too loose |
| Manual bounds | 0 | Exact | When known from problem structure |

**Gershgorin for preconditioned system** (M⁻¹A):
- Centers: 1.0 (diagonal of M⁻¹A = 1 when M = diag(A))
- Radii: `rᵢ = Σⱼ≠ᵢ |aᵢⱼ| / |aᵢᵢ|`
- Bounds: `λ_min ≈ 1 - max(rᵢ)`, `λ_max ≈ 1 + max(rᵢ)`
- Safety margin: 10% expansion (configurable)

**Power iteration** (for λ_max estimation):
```python
def power_iteration(A, M_inv, n_iters=15):
    """Estimate largest eigenvalue of M⁻¹A via power iteration."""
    x = torch.randn(A.shape[0], device=A.device, dtype=A.dtype)
    for _ in range(n_iters):
        y = M_inv * sparse_mv(A, x)  # M⁻¹A·x
        lam = torch.dot(x, y) / torch.dot(x, x)
        x = y / torch.norm(y)
    return lam.item()
```

### 3.3 Implementation Plan

```python
class ChebyshevSolver(LinearSolver):
    """
    Chebyshev polynomial linear solver.

    Zero-sync polynomial iteration for SPD systems. No inner products or
    global reductions during iteration — ideal for GPU execution.

    Parameters
    ----------
    max_iters : int
        Fixed number of iterations (no convergence check during iteration)
    eigenvalue_method : str
        'gershgorin' (default), 'power', or 'manual'
    safety_margin : float
        Gershgorin bounds expansion factor (default 10%)
    use_jacobi_precond : bool
        Apply Jacobi (diagonal) preconditioning (default True)
    use_warm_start : bool
        Use previous solution as initial guess (default True)
    """

    def __init__(self, max_iters=50, eigenvalue_method='gershgorin',
                 safety_margin=0.1, use_jacobi_precond=True,
                 use_warm_start=True):
        ...

    def solve(self, A, b):
        """Solve Ax = b using Chebyshev iteration."""
        ...

    def set_eigenvalue_bounds(self, lam_min, lam_max):
        """Manually set eigenvalue bounds (bypasses estimation)."""
        ...
```

### 3.4 When to Use Chebyshev vs PCG

| Scenario | Best Solver | Why |
|----------|------------|-----|
| Elliptic solve, GPU | **Chebyshev** | Zero sync points dominate GPU efficiency |
| Elliptic solve, CPU | PCG | Adaptive iteration count saves work |
| Parabolic solve | PCG | Few iterations needed (well-conditioned), adaptive |
| Isotropic elliptic | **Spectral (Tier 1)** | O(N log N) direct solve, beats both |
| Anisotropic elliptic, GPU | **Chebyshev** or PCG+Spectral (Tier 2) | Depends on anisotropy ratio |

---

## 4. PCG Hardening

### 4.1 Fixes

```python
# Fix PCG-1: Scale-relative pAp check
if pAp.abs() < 1e-14 * b_norm_sq:
    break

# Fix PCG-2: Boolean warm-start check instead of O(n) sum
_used_warm_start = (self.use_warm_start and self._last_solution is not None)
if _used_warm_start:
    r.sub_(sparse_mv(A, x))

# Fix PCG-3: No change needed (accept id()-based cache)
```

### 4.2 Enhancements

1. **Convergence logging**: Optional callback for iteration monitoring
2. **Stagnation detection**: Break if `|r_norm_new - r_norm_old| / r_norm_old < 1e-12`
3. **Return residual history**: For debugging convergence issues

---

## 5. GPU Acceleration Strategy

### 5.1 Sparse Format

PyTorch sparse COO → CSR conversion for GPU SpMV:

```python
# COO is stored but CSR is faster for SpMV on GPU
A_csr = A.to_sparse_csr()  # One-time conversion, cached
y = A_csr @ x  # Uses cuSPARSE internally
```

**Note**: `torch.sparse.mm()` with COO triggers internal CSR conversion anyway.
Explicit CSR avoids repeated conversion overhead.

### 5.2 Kernel Fusion Opportunities

For Chebyshev iteration, the per-step operations are:
```
r = b - A*x           # SpMV + vector subtract
z = diag_inv * r       # Element-wise multiply
d = α*d + β*z          # AXPY
x = x + d              # AXPY
```

With `torch.compile`:
```python
@torch.compile
def chebyshev_step(A, x, b, d, diag_inv, rho_rho_prev, two_rho_over_delta):
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    z = diag_inv * r
    d = rho_rho_prev * d + two_rho_over_delta * z
    x = x + d
    return x, d
```

This fuses the 3 vector operations into a single GPU kernel after the SpMV.

### 5.3 Matrix-Free Stencil Application

For FDM on structured grids, SpMV can be replaced by direct stencil application:

```python
@torch.compile
def apply_laplacian_5pt(V, D, dx, dy, nx, ny):
    """5-point Laplacian stencil, avoiding sparse matrix entirely."""
    V_grid = V.reshape(nx, ny)
    lap = torch.zeros_like(V_grid)
    # Interior: standard 5-point
    lap[1:-1, 1:-1] = D * (
        (V_grid[2:, 1:-1] + V_grid[:-2, 1:-1] - 2*V_grid[1:-1, 1:-1]) / dx**2 +
        (V_grid[1:-1, 2:] + V_grid[1:-1, :-2] - 2*V_grid[1:-1, 1:-1]) / dy**2
    )
    # Boundary: Neumann (replicate padding)
    ...
    return lap.flatten()
```

**Speedup**: 3-5x vs assembled sparse matrix SpMV (avoids indirect addressing).
**Trade-off**: Harder to handle anisotropy and irregular domains.

### 5.4 Warm Start for Time Stepping

The elliptic solution `φ_e^{n+1}` is close to `φ_e^n` between timesteps.
Linear extrapolation provides an even better initial guess:

```python
def warm_start_extrapolate(x_curr, x_prev):
    """Linear extrapolation: x_next ≈ 2*x_curr - x_prev."""
    return 2.0 * x_curr - x_prev
```

**Impact**: Reduces Chebyshev iterations from 50 → ~15 for typical cardiac problems.

---

## 6. Validation Plan

### 6.1 Unit Tests

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| CH-T1 | Chebyshev solves 5-pt Laplacian Poisson (Neumann) | `‖Ax-b‖/‖b‖ < 1e-6` after 50 iters |
| CH-T2 | Chebyshev solves elliptic operator (bidomain) | Matches PCG to `< 1e-6` |
| CH-T3 | Jacobi-preconditioned eigenvalue bounds | `λ_min(D⁻¹A) ≤ estimate ≤ λ_max(D⁻¹A)` |
| CH-T4 | Power iteration λ_max vs torch.linalg.eigvalsh | `< 5%` error |
| CH-T5 | Warm start reduces residual | `‖r₀_warm‖ < 0.1 * ‖r₀_cold‖` |
| PCG-T1 | PCG converges on elliptic (Neumann + pinning) | `converged=True`, `< 100` iters |
| PCG-T2 | PCG converges on elliptic (Dirichlet) | `converged=True`, `< 100` iters |
| PCG-T3 | PCG warm start reduces iterations | `iters_warm < iters_cold / 2` |

### 6.2 Cross-Validation

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| XV-T1 | Chebyshev vs PCG on same elliptic system | `‖x_cheb - x_pcg‖/‖x_pcg‖ < 1e-4` |
| XV-T2 | Chebyshev vs Spectral on isotropic elliptic | `‖x_cheb - x_spec‖/‖x_spec‖ < 1e-4` |
| XV-T3 | Full bidomain sim: Chebyshev elliptic vs PCG elliptic | CV within 5% |
| XV-T4 | GPU vs CPU: same solver, same result | Bitwise identical (float64) |

### 6.3 Performance Benchmarks

| Test | Metric | Target |
|------|--------|--------|
| PB-T1 | Chebyshev iteration time (150x40 grid) | < 0.5 ms/step |
| PB-T2 | PCG iteration time (150x40 grid) | < 1.0 ms/step |
| PB-T3 | Spectral solve time (150x40 grid) | < 0.2 ms/step |
| PB-T4 | Chebyshev GPU vs CPU speedup | > 5x on 150x40, > 20x on 500x500 |
| PB-T5 | Full bidomain sim throughput (150x40, 40ms) | < 30s |

---

## 7. Task List

### Phase A: Bug Fixes (existing solvers)

- [ ] **A1**: Fix Chebyshev eigenvalue estimation (CH-1): implement `_gershgorin_bounds_preconditioned()`
- [ ] **A2**: Fix Chebyshev first-iteration guard (CH-2): assert theta > 0
- [ ] **A3**: Fix Chebyshev overflow protection (CH-3): clamp rho_new
- [ ] **A4**: Add Chebyshev warm start (CH-4): track `_last_solution`
- [ ] **A5**: Fix PCG pAp threshold (PCG-1): scale-relative check
- [ ] **A6**: Fix PCG warm-start check (PCG-2): boolean flag
- [ ] **A7**: Add isotropic assertion to SpectralSolver (SP-1)
- [ ] **A8**: Deprecate `fft.py` — add deprecation warning, point to `spectral.py`

### Phase B: Chebyshev Rewrite

- [ ] **B1**: Implement `_gershgorin_bounds_preconditioned()` for M⁻¹A
- [ ] **B2**: Implement power iteration eigenvalue estimation
- [ ] **B3**: Add `eigenvalue_method` parameter ('gershgorin', 'power', 'manual')
- [ ] **B4**: Add warm start with linear extrapolation
- [ ] **B5**: Add optional convergence check (for debugging, not production)
- [ ] **B6**: Run validation tests CH-T1 through CH-T5

### Phase C: GPU Optimization

- [ ] **C1**: Add CSR conversion caching for sparse matrices
- [ ] **C2**: Implement `torch.compile` kernel fusion for Chebyshev step
- [ ] **C3**: Benchmark Chebyshev GPU vs CPU (PB-T4)
- [ ] **C4**: Implement matrix-free stencil application (optional, structured grids only)
- [ ] **C5**: Profile full bidomain sim to identify remaining bottlenecks

### Phase D: Cross-Validation

- [ ] **D1**: Chebyshev vs PCG on elliptic system (XV-T1)
- [ ] **D2**: Chebyshev vs Spectral (XV-T2)
- [ ] **D3**: Full bidomain sim with Chebyshev elliptic solver (XV-T3)
- [ ] **D4**: GPU vs CPU reproducibility (XV-T4)

### Phase E: Integration

- [ ] **E1**: Add 'chebyshev' to solver selection in `bidomain.py`
- [ ] **E2**: Update `_auto_select_elliptic_solver()` to prefer Chebyshev on GPU
- [ ] **E3**: Update PROGRESS.md and REVIEW.md
- [ ] **E4**: Performance benchmarks (PB-T1 through PB-T5)

---

## Appendix A: Reference Implementations

### pyAMG Chebyshev Smoother
`Research/code_examples/pyamg/pyamg/relaxation/chebyshev.py`
- Polynomial coefficient computation for interval [a, b]
- Used as multigrid smoother (not standalone solver)

### AMGCL Chebyshev Relaxation
`Research/code_examples/amgcl/amgcl/relaxation/chebyshev.hpp`
- Parameters: degree=5, higher=1.0, lower=1/30
- Gershgorin or power iteration for eigenvalue estimation
- Scale by Jacobi preconditioner

### V5.4 Chebyshev (current, buggy)
`Monodomain/Engine_V5.4/.../chebyshev.py` — same code as Bidomain V1 (shared bugs)

### TorchCor (2025)
FEM bidomain in PyTorch. Uses PCG with warm start (3-8x iteration reduction).
Reports `torch.set_default_dtype(torch.float64)` essential for stability.

## Appendix B: Formulation B Consistency Check

All solvers must use Formulation B:
- **Parabolic**: `A_para = 1/dt · I - θ·L_i` (NOT chi*Cm/dt)
- **Elliptic**: `A_ellip = -(L_i + L_e)` (L contains D = σ/(χCm))
- **Source term**: `R = -(I_ion + I_stim) / Cm` (NOT /chi/Cm)
- **fft.py**: Uses chi*Cm in eigenvalues → **INCONSISTENT**, must be deprecated

## Appendix C: Null Space Handling Summary

| Solver | Null Space Method | Notes |
|--------|------------------|-------|
| PCG | Node pinning (identity row) | Direct constraint, no post-processing |
| Chebyshev | Node pinning (identity row) | Same as PCG |
| Spectral (DCT) | Set û[0,0] = 0, post-subtract | Spectral has built-in null space |
| PCG+Spectral | Node pinning + spectral precond | Pinning in outer PCG, null space in precond |

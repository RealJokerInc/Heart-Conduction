# Bidomain Simulation — Knowledge File

## Spatial Discretization

### Which method for which problem

| Scenario | Method | Why |
|----------|--------|-----|
| Isotropic rectangle | FDM 5-point | Fastest, spectral-compatible |
| Anisotropic with fibers (Dxy ≠ 0) | FDM 9-point | Cross-derivative support |
| Isotropic, higher accuracy | FDM Mehrstellen (9-point compact) | 4th order, spectral-compatible, dx=dy required |
| Complex geometry | FEM P1 | Unstructured mesh handles anything |
| Conservation-critical | FVM (TPFA) | Flux-based, exact local conservation |
| GPU-parallel, regular grid | LBM D2Q5/D2Q9 | No assembly, no linear solve (see lbm_cardiac.md) |

### Stencil construction

**5-point** (cardinal only): `L = Dxx/dx² + Dyy/dy²`. Harmonic mean at cell faces for heterogeneous D. Supports Dxx ≠ Dyy but NOT Dxy ≠ 0.

**9-point** (cardinal + diagonal): Adds cross-derivative `∂²V/∂x∂y ≈ (V_NE - V_NW - V_SE + V_SW)/(4·dx·dy)`. Handles full 2×2 diffusion tensor. Reduces to 5-point when Dxy = 0.

**Mehrstellen**: `L = D·[1,4,1; 4,-20,4; 1,4,1]/(6h²)`. Higher-order compact, isotropic only, requires dx = dy. Eigenvalues separate per-axis → O(N log N) spectral solve.

### Boundary treatment

**Face-based (Bidomain V1)**: Each interior face contributes equally to both adjacent nodes. Symmetric SPD Laplacian. Required for PCG/Chebyshev on bidomain elliptic.

**Ghost-node (V5.4)**: Mirror at boundary doubles connection weight. Asymmetric L. Works for monodomain (identity term dominates) but breaks bidomain PCG.

### Known pitfalls
- Ghost-node FDM produces asymmetric L — breaks PCG on bidomain elliptic
- FVM TPFA fails for non-K-orthogonal meshes — need MPFA
- Harmonic mean ensures zero flux at D=0 interfaces (scar tissue)

## Linear Solvers

### Three-tier auto-selection

| Tier | Solver | When to use | Iterations |
|------|--------|-------------|------------|
| 1 | Spectral (DCT/DST/FFT) | Isotropic, uniform grid, rectangle | 0 (direct) |
| 2 | PCG + spectral preconditioner | Moderate anisotropy, mixed BCs | 1–3 |
| 3 | PCG + geometric multigrid | Arbitrary coefficients, complex geometry | 10–25 |

Selection is automatic from BoundarySpec: all-Neumann → DCT, all-Dirichlet → DST, periodic → FFT, mixed → PCG.

### Spectral solver details
- DCT via `torch_dct` (GPU-native)
- DST-I via custom odd-extension FFT (`dst1_2d`/`idst1_2d`)
- Periodic via `torch.fft.fft2`
- Null space handling: set eigenvalue[0,0] = 1, coefficient[0,0] = 0

### PCG details
- Jacobi preconditioning (diagonal of A)
- Flag-based warm start (reuse previous solution)
- Exact breakdown check: `pAp <= 0`

## Time Integration

### Operator splitting
**Strang** (2nd order): half-ionic → full-diffusion → half-ionic. Use when splitting error matters.
**Godunov** (1st order): ionic → diffusion. Simpler, sufficient when dt is small.

### Ionic step
**Rush-Larsen**: Exponential integration on gates (`x_new = x_inf - (x_inf - x_old)·exp(-dt/tau)`), Forward Euler on concentrations. Stable at dt = 0.02–0.1 ms for TTP06.

### Diffusion step
**Crank-Nicolson** (θ=0.5): Unconditionally stable, O(dt²). Default choice.
**BDF1** (backward Euler): O(dt), more dissipative. Use for debugging.
**BDF2**: O(dt²), A-stable. Better accuracy than CN for oscillatory solutions.
**Explicit (FE/RK2/RK4)**: CFL-limited. Only for small grids or educational use.

### Known pitfalls
- Fully explicit methods **fail** for bidomain (elliptic equation has no time derivative)
- Strang needs dt/2 passed correctly to ionic solver — common bug source
- The decoupled Gauss-Seidel approach lags phi_e → O(dt) splitting error even with Strang

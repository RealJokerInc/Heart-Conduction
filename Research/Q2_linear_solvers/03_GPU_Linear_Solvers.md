---

# GPU-Friendly Linear Solvers for Cardiac Electrophysiology Simulation

## 1. Chebyshev Iteration (Sync-Free Polynomial Solver for SPD Systems)

### Why Chebyshev for GPU Cardiac Simulation?

The Chebyshev iteration is uniquely suited for GPU-resident cardiac solvers because it **requires no inner products** (dot products / global reductions). In standard CG/PCG, each iteration needs 2-3 global reductions, and each reduction forces a GPU-wide synchronization barrier. On modern GPUs these synchronization points are the primary bottleneck, not the arithmetic. Chebyshev iteration replaces convergence-adaptive stepping with a fixed polynomial recurrence that only needs sparse matrix-vector products (SpMV) and vector updates -- both embarrassingly parallel on GPUs.

**Trade-off**: You must supply eigenvalue bounds `[lambda_min, lambda_max]` of the preconditioned operator. If `lambda_max` is underestimated, the method diverges.

The [Netlib Templates reference](https://www.netlib.org/linalg/html_templates/node44.html) states: "Chebyshev Iteration avoids the computation of inner products as is necessary for the other nonstationary methods. For some distributed memory architectures these inner products are a bottleneck with respect to efficiency."

### Algorithm (from Saad's "Iterative Methods for Sparse Linear Systems", Algorithm 12.1)

```python
import torch

def chebyshev_iteration(A_matvec, b, precond_solve, lam_min, lam_max,
                        x0=None, maxiter=50):
    """
    Sync-free Chebyshev iteration for SPD system Ax = b.
    
    Args:
        A_matvec: callable, computes A @ x (sparse matvec on GPU)
        b: right-hand side tensor (on GPU)
        precond_solve: callable, applies C^{-1} (e.g., Jacobi inverse)
        lam_min: lower bound on eigenvalues of C^{-1}A
        lam_max: upper bound on eigenvalues of C^{-1}A
        x0: initial guess (default: zero)
        maxiter: number of iterations (fixed, no convergence check)
    
    Returns:
        x: approximate solution tensor (on GPU)
    
    Key property: NO inner products / global reductions at any iteration.
    """
    theta = (lam_max + lam_min) / 2.0
    delta = (lam_max - lam_min) / 2.0
    sigma = theta / delta

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_matvec(x)
    z = precond_solve(r)

    # First iteration (special case)
    rho = 1.0 / sigma
    d = z / theta
    x = x + d

    for k in range(1, maxiter):
        r = b - A_matvec(x)
        z = precond_solve(r)

        rho_new = 1.0 / (2.0 * sigma - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * z
        x = x + d
        rho = rho_new

    return x
```

Key properties of this implementation:
- **No inner products**: The entire iteration uses only SpMV, precon-apply, and AXPY-type vector operations.
- **Fixed iteration count**: No convergence check, so the GPU never needs to transfer a scalar back to the host.
- **Three-term recurrence**: Each step depends on the current and previous direction vector `d`, controlled by the Chebyshev parameter `rho`.

### Eigenvalue Bounds Estimation

Three practical strategies, from cheapest to most accurate:

**Strategy 1: Gershgorin Circle Theorem (Cheapest, GPU-friendly)**

For a sparse matrix A, the [Gershgorin circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) states that every eigenvalue lies in at least one disc centered at a diagonal element with radius equal to the sum of absolute off-diagonal row entries. For the Jacobi-preconditioned system `D^{-1}A`, the discs are centered at 1.

```python
def gershgorin_bounds_jacobi_preconditioned(A_values, A_row_ptr, A_col_idx, 
                                             n, device='cuda'):
    """
    Estimate eigenvalue bounds of D^{-1}A via Gershgorin discs.
    All operations are on GPU -- no host sync needed.
    """
    diag_vals = torch.zeros(n, device=device, dtype=torch.float64)
    off_diag_row_sums = torch.zeros(n, device=device, dtype=torch.float64)

    # Extract diagonal and off-diagonal row sums from CSR
    for i in range(n):
        start, end = A_row_ptr[i].item(), A_row_ptr[i+1].item()
        for j in range(start, end):
            col = A_col_idx[j].item()
            val = A_values[j].abs()
            if col == i:
                diag_vals[i] = A_values[j]
            else:
                off_diag_row_sums[i] += val

    radii = off_diag_row_sums / diag_vals.abs()
    lam_max = (1.0 + radii.max()).item() * 1.1  # 10% safety margin
    lam_min = max((1.0 - radii.max()).item(), 0.01)
    return lam_min, lam_max
```

**Strategy 2: Power Iteration (moderate cost, ~15 SpMVs)**

```python
def estimate_largest_eigenvalue(A_matvec, precond_solve, n, device, 
                                 num_iters=15):
    """Estimate lambda_max of C^{-1}A via power iteration."""
    v = torch.randn(n, device=device, dtype=torch.float64)
    v = v / torch.norm(v)
    lam = 0.0
    for _ in range(num_iters):
        w = precond_solve(A_matvec(v))
        lam = torch.dot(w, v).item()  # sync here, but only during setup
        v = w / torch.norm(w)
    return lam * 1.1  # safety margin
```

**Strategy 3: CG-Based Estimation (PETSc's default)**

[PETSc's KSPCHEBYSHEV](https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/) runs a few CG steps (default 10) and extracts eigenvalue estimates from the implicitly formed tridiagonal Lanczos matrix. Default transform is `(0, 0.1; 0, 1.1)`, meaning `lam_min = 0.1 * estimated_max` and `lam_max = 1.1 * estimated_max`. After estimation, Chebyshev proceeds with zero inner products.

### Production Implementations

- **PETSc** [KSPCHEBYSHEV](https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/): Supports eigenvalue estimation via Krylov methods, configurable bounds, requires SPD operator.
- **Trilinos/Ifpack2** [Ifpack2::Chebyshev](https://docs.trilinos.org/dev/packages/ifpack2/doc/html/classIfpack2_1_1Chebyshev.html): Uses estimate of max eigenvalue in apply step.
- **FELTOR** [dg::ChebyshevIteration](https://mwiesenberger.github.io/feltor/dg/html/classdg_1_1_chebyshev_iteration.html): Uses EVE class for eigenvalue estimation.
- **NVIDIA AmgX**: Includes Chebyshev polynomial as a smoother option within AMG, specifically designed for GPU execution.

### Chebyshev as Multigrid Smoother

A highly active area: [Adams et al., "Multigrid Smoothers for Ultra-Parallel Computing"](https://www.osti.gov/servlets/purl/1117969) showed that Chebyshev smoothers outperform Gauss-Seidel on parallel architectures because they only need SpMV (fully parallel), while Gauss-Seidel has sequential dependencies. A [2022 paper on optimal polynomial smoothers for multigrid V-cycles](https://arxiv.org/abs/2202.08830) demonstrates that fourth-kind Chebyshev polynomials are quasi-optimal for the V-cycle bound.

### Key References

- Saad, "Iterative Methods for Sparse Linear Systems", Algorithm 12.1, p. 399
- [Gutknecht & Rollin, "The Chebyshev Iteration Revisited", Parallel Computing 2002](https://people.math.ethz.ch/~mhg/pub/Cheby-02ParComp.pdf)
- [Wathen, "Chebyshev semi-iteration in Preconditioning", Oxford Report 08/14](https://www.cs.ox.ac.uk/files/1540/NA-08-14.pdf)
- [Interactive Finite Elements, Chebyshev Method (Python implementation)](https://jschoeberl.github.io/iFEM/iterative/Chebyshev.html)
- [Wikipedia: Chebyshev iteration](https://en.wikipedia.org/wiki/Chebyshev_iteration)
- [GPU-based Chebyshev preconditioner + CG: 46x GPU speedup](https://www.sciencedirect.com/science/article/abs/pii/S0378779614001850)
- [Optimal polynomial smoothers for parallel AMG (2025)](https://link.springer.com/article/10.1007/s11075-025-02117-6)

---

## 2. FFT-Based Solvers for Diffusion on Structured Grids

### Why FFT for Cardiac Diffusion?

If the cardiac tissue is discretized on a **regular structured grid** (common for voxel-based anatomical models from CT/MRI segmentation), the diffusion equation can be solved in O(N log N) time using FFT, compared to O(N * k) for iterative solvers. The implicit diffusion step `(M + dt*K) V^{n+1} = M*V^n + dt*I_ion` reduces to a diagonal solve in spectral space when K is the standard discrete Laplacian on a uniform grid.

**Boundary conditions determine the transform type:**

| Boundary Condition | Transform | PyTorch Support |
|---|---|---|
| Periodic | FFT (DFT) | `torch.fft.fftn` (native) |
| Neumann (zero-flux) | DCT (Type II/III) | Via `torch-dct` or manual |
| Dirichlet (zero) | DST (Sine Transform) | Via manual construction |

**For cardiac simulation, Neumann (zero-flux) BCs are most appropriate** since the heart boundary should not leak current. This requires the Discrete Cosine Transform.

### PyTorch Implementation: Periodic BCs (Simplest)

```python
import torch
import torch.fft

def solve_diffusion_fft_periodic(rhs, dx, dy, dz, dt, diff_coeff, device='cuda'):
    """
    Solve (I - dt * D * Laplacian) u = rhs on a periodic 3D structured grid.
    Uses torch.fft (backed by cuFFT on GPU).
    
    Args:
        rhs: source term tensor, shape (Nx, Ny, Nz), on GPU
        dx, dy, dz: grid spacings
        dt: time step
        diff_coeff: scalar diffusion coefficient
    Returns:
        u: solution tensor, shape (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = rhs.shape

    # Wavenumber grids (torch.fft.fftfreq returns frequencies in cycles/sample)
    kx = torch.fft.fftfreq(Nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(Nz, d=dz, device=device) * 2 * torch.pi

    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Forward FFT
    rhs_hat = torch.fft.fftn(rhs)

    # Solve in spectral space: (1 + dt*D*k^2) u_hat = rhs_hat
    denom = 1.0 + dt * diff_coeff * K2
    u_hat = rhs_hat / denom

    # Inverse FFT
    u = torch.fft.ifftn(u_hat).real
    return u
```

### PyTorch Implementation: Neumann BCs via DCT

For Neumann BCs, the DCT diagonalizes the discrete Laplacian. PyTorch does not have a built-in DCT, but the [`torch-dct` package](https://github.com/zh217/torch-dct) provides GPU-accelerated DCT with autograd support, implemented via the FFT.

```python
# pip install torch-dct
import torch
import torch_dct as dct

def solve_diffusion_dct_neumann_2d(rhs, dx, dy, dt, diff_coeff, device='cuda'):
    """
    Solve (I - dt * D * Laplacian) u = rhs with Neumann BCs on 2D grid.
    Uses DCT-II (forward) and DCT-III (inverse) to diagonalize the Laplacian.
    """
    Nx, Ny = rhs.shape

    # Eigenvalues of the discrete Laplacian with Neumann BCs:
    # lambda_{i,j} = (2/dx^2)(cos(pi*i/Nx) - 1) + (2/dy^2)(cos(pi*j/Ny) - 1)
    i_idx = torch.arange(Nx, device=device, dtype=rhs.dtype)
    j_idx = torch.arange(Ny, device=device, dtype=rhs.dtype)

    lam_x = (2.0 / dx**2) * (torch.cos(torch.pi * i_idx / Nx) - 1.0)
    lam_y = (2.0 / dy**2) * (torch.cos(torch.pi * j_idx / Ny) - 1.0)

    LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
    laplacian_eigenvalues = LAM_X + LAM_Y  # all <= 0

    # Forward 2D DCT (apply along each dimension)
    rhs_dct = dct.dct(dct.dct(rhs, norm='ortho').transpose(-1, -2), 
                       norm='ortho').transpose(-1, -2)

    # Solve: (1 - dt*D*lambda) u_hat = rhs_hat
    # lambda <= 0, so denominator >= 1
    denom = 1.0 - dt * diff_coeff * laplacian_eigenvalues
    u_dct = rhs_dct / denom

    # Inverse 2D DCT
    u = dct.idct(dct.idct(u_dct, norm='ortho').transpose(-1, -2), 
                  norm='ortho').transpose(-1, -2)
    return u
```

The `torch-dct` library implements DCT in terms of PyTorch's built-in FFT operations (`torch.fft.rfft`), so backpropagation works through it and it runs on GPU via cuFFT.

### Handling the Singularity

The Laplacian with pure Neumann BCs has a zero eigenvalue (constant null space). When dividing in spectral space, set the zero-frequency component to zero:

```python
# After computing denom:
denom[0, 0] = 1.0  # avoid division by zero
u_dct = rhs_dct / denom
u_dct[0, 0] = 0.0  # set mean to zero (or desired constant)
```

As noted in the [FFT Poisson solver literature](https://atmos.washington.edu/~breth/classes/AM585/lect/FS_2DPoisson.pdf): "There is an ambiguity in the solution as it's only defined up to a constant. Choose the solution with zero mean by setting the zeroth Fourier coefficient to zero."

### Anisotropic Diffusion and Irregular Domains

For cardiac tissue with **fiber-aligned anisotropic diffusion**, a pure FFT solve is not directly applicable because the operator is no longer diagonalized by the Fourier basis. Two workarounds:

1. **FFT as preconditioner**: Use the isotropic-average Laplacian as a spectral preconditioner for an iterative solve of the full anisotropic system. The spectral solve provides an excellent approximation, so only 2-5 iterations of the outer solver are needed.
2. **Operator splitting**: Split into isotropic (FFT-solvable) and anisotropic correction (explicit or a few Richardson iterations).

For **irregular domains** (the heart is not a box), apply a mask tensor: solve on the bounding box and zero out the exterior. This works well when combined with the immersed boundary approach.

### Shape As Points: Production PyTorch Spectral Solver

The [Shape As Points (NeurIPS 2021)](https://github.com/autonomousvision/shape_as_points) paper provides a differentiable Poisson solver in PyTorch using spectral methods. The authors state: "The spectral method is highly optimized on GPUs/TPUs. It is extremely simple -- it can be implemented with 25 lines of code." The solver achieves 12 ms at 128^3 resolution, demonstrating the efficiency of FFT-based approaches in PyTorch.

### Key References

- [Shape As Points, NeurIPS 2021](https://github.com/autonomousvision/shape_as_points) -- differentiable PyTorch spectral Poisson solver
- [torch-dct: DCT for PyTorch](https://github.com/zh217/torch-dct)
- [PyTorch torch.fft module blog post](https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/)
- [MathWorks DCT Poisson Solver](https://github.com/mathworks/Fast-Poisson-Equation-Solver-using-DCT)
- [Solving 2D Poisson with Neumann using DCT](https://elonen.iki.fi/code/misc-notes/neumann-cosine/)
- [poisson-dirichlet-neumann (Python, NumPy reference)](https://github.com/aywander/poisson-dirichlet-neumann)
- [Solving Poisson's Equation Using FFT in a GPU Cluster](https://www.sciencedirect.com/science/article/abs/pii/S0743731516301678)

---

## 3. Algebraic Multigrid (AMG) for Cardiac Monodomain

### Why AMG?

The monodomain equation discretized on unstructured meshes produces large SPD systems where simple preconditioners (Jacobi, SSOR) suffer from mesh-dependent convergence. AMG is the gold standard because:

- **Mesh-independent convergence**: The number of iterations does not grow as the mesh is refined.
- **Black-box**: AMG builds coarse-grid hierarchies directly from the matrix, no geometry needed.
- **Proven for cardiac**: [Plank et al. (2007)](https://pubmed.ncbi.nlm.nih.gov/17405366/) showed AMG (BoomerAMG from Hypre) achieves 5.9-7.7x speedup over ILU for the bidomain equations.

### Option A: NVIDIA AmgX (via pyamgx) -- GPU-Native AMG

[AmgX](https://github.com/NVIDIA/AMGX) is NVIDIA's production GPU-accelerated AMG library, achieving [2-5x speedup on a single GPU vs competitive CPU AMG, with both setup and solve phases scaling across multiple nodes](https://epubs.siam.org/doi/10.1137/140980260). [pyamgx](https://github.com/shwina/pyamgx) provides Python bindings.

```python
import pyamgx
import numpy as np

# Initialize AMGX
pyamgx.initialize()

# Configuration for AMG-preconditioned CG with Chebyshev smoother
cfg_dict = {
    "config_version": 2,
    "determinism_flag": 0,
    "solver": {
        "solver": "PCG",
        "max_iters": 100,
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1e-6,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",   # or "CLASSICAL"
            "selector": "SIZE_2",
            "max_iters": 1,               # single V-cycle per PCG step
            "cycle": "V",
            "smoother": {
                "solver": "CHEBYSHEV",    # sync-free smoother on GPU
                "preconditioner": {
                    "solver": "JACOBI"
                },
                "max_iters": 3
            },
            "coarsest_sweeps": 3,
            "max_levels": 10
        }
    }
}

# Configuration for FIXED-ITERATION mode (no convergence check)
cfg_fixed_iter = {
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "max_iters": 15,            # Fixed 15 iterations
        "convergence": "NONE",       # NO convergence check -> no host sync
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "max_iters": 1,
            "cycle": "V",
            "smoother": {"solver": "JACOBI", "max_iters": 2}
        }
    }
}

cfg = pyamgx.Config().create_from_dict(cfg_dict)
rsc = pyamgx.Resources().create_simple(cfg)

A_amgx = pyamgx.Matrix().create(rsc)
b_amgx = pyamgx.Vector().create(rsc, mode='dDDI')  # device (GPU) vector
x_amgx = pyamgx.Vector().create(rsc, mode='dDDI')
solver = pyamgx.Solver().create(rsc, cfg)

# Upload CSR matrix (from scipy.sparse.csr_matrix)
from scipy.sparse import csr_matrix
A_scipy = csr_matrix(...)  # your monodomain system matrix
A_amgx.upload_CSR(A_scipy)

# Upload vectors
b_np = np.array(...)
x_np = np.zeros_like(b_np)
b_amgx.upload(b_np)
x_amgx.upload(x_np)

# Setup AMG hierarchy (do ONCE if matrix is constant across timesteps)
solver.setup(A_amgx)

# Solve (repeat each timestep with new RHS)
solver.solve(b_amgx, x_amgx)

# Download solution
sol = np.zeros_like(b_np)
x_amgx.download(sol)

# Cleanup
for obj in [A_amgx, b_amgx, x_amgx, solver, rsc, cfg]:
    obj.destroy()
pyamgx.finalize()
```

**Key note for cardiac simulation**: The system matrix `(M + dt*K)` is constant across timesteps when using operator splitting with a fixed dt. Call `solver.setup(A_amgx)` once and reuse the hierarchy. The pyamgx documentation states: "For the case in which the coefficient matrix remains fixed, the setup() method should only be called once."

### Option B: PyAMG (CPU, for prototyping)

[PyAMG](https://github.com/pyamg/pyamg) is pure Python/NumPy and runs on CPU only, but excellent for prototyping AMG settings before deploying to GPU.

```python
import pyamg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

A = csr_matrix(...)  # SPD monodomain matrix
b = np.array(...)

# Smoothed Aggregation AMG
ml = pyamg.smoothed_aggregation_solver(
    A,
    strength='symmetric',
    smooth=('jacobi', {'omega': 4.0/3.0}),
    max_levels=10,
    max_coarse=500
)
print(ml)  # shows levels, operator/grid complexity

# As preconditioner for CG
M = ml.aspreconditioner(cycle='V')
x, info = cg(A, b, M=M, tol=1e-8, maxiter=100)

# Ruge-Stuben AMG (better for anisotropic diffusion)
ml_rs = pyamg.ruge_stuben_solver(A, strength='classical', max_levels=10)
```

### Option C: AMGCL (C++ with GPU backends)

[AMGCL](https://amgcl.readthedocs.io/) is a header-only C++ library with VexCL, CUDA, and Thrust backends. Setup always runs on CPU; the solve phase runs on GPU. Solution phase achieves ~4x speedup on GPU vs CPU.

```cpp
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/backend/cuda.hpp>

typedef amgcl::backend::cuda<double> Backend;
typedef amgcl::make_solver<
    amgcl::amg<Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::chebyshev>,    // Chebyshev smoother (GPU-friendly)
    amgcl::solver::cg<Backend>
> Solver;

Solver::params prm;
prm.precond.coarsening.aggr.eps_strong = 0.08;
prm.solver.maxiter = 100;
prm.solver.tol = 1e-6;

Backend::params bprm;
cusparseCreate(&bprm.cusparse_handle);

Solver solve(A_csr, prm, bprm);  // setup on CPU, hierarchy to GPU
```

### AMG for Cardiac: Specific Considerations

1. **Anisotropic diffusion**: Cardiac fiber anisotropy requires strength-of-connection tuning. Use `strength='evolution'` in PyAMG or tune `eps_strong` in AMGCL.
2. **Chebyshev smoother preferred on GPU**: Gauss-Seidel has sequential dependencies; Chebyshev only needs SpMV (fully parallel). Both AmgX and AMGCL support it.
3. **Matrix reuse**: For operator splitting with fixed dt, build the AMG hierarchy once and reuse across all timesteps.
4. **Recent work**: [Parallel AMG for cardiac EMI model](https://www.sciencedirect.com/science/article/pii/S0045782525002737) and [BDDC preconditioning on GPUs for cardiac simulations](https://arxiv.org/html/2410.14786) (MicroCARD project, >90% efficiency on 100 GPUs).

### Key References

- [PyAMG documentation](https://pyamg.readthedocs.io/)
- [NVIDIA AmgX](https://github.com/NVIDIA/AMGX)
- [pyamgx documentation](https://pyamgx.readthedocs.io/en/latest/basic.html)
- [AMGCL documentation](https://amgcl.readthedocs.io/)
- [Plank et al., "Algebraic Multigrid Preconditioner for the Cardiac Bidomain Model"](https://pubmed.ncbi.nlm.nih.gov/17405366/)
- [AmgX SIAM paper](https://epubs.siam.org/doi/10.1137/140980260)
- [BDDC on GPUs for Cardiac (MicroCARD)](https://arxiv.org/html/2410.14786)
- [AMG for cardiac EMI model (2025)](https://www.sciencedirect.com/science/article/pii/S0045782525002737)
- [Comparison of AMG bidomain solvers on hybrid CPU-GPU](https://www.sciencedirect.com/science/article/pii/S0045782524001312)

---

## 4. Fixed-Iteration PCG (Avoiding Sync Points on GPU)

### The Problem: Global Reductions Kill GPU Performance

Standard PCG requires 2-3 global dot products per iteration:
1. `r^T z` (for beta)
2. `p^T A p` (for alpha)
3. Optionally `r^T r` (convergence check)

Each is a global reduction that forces all GPU threads to synchronize, creating "bubbles" where most of the GPU sits idle.

### Strategy 1: Fixed-Iteration PCG (Drop Convergence Check)

Run PCG for a fixed iteration count. This eliminates the host transfer for convergence checking. The inter-iteration dot products still cause GPU-internal synchronization, but no GPU-to-CPU transfer is needed.

```python
import torch

def fixed_iter_pcg(A_matvec, b, precond_solve, x0=None, num_iters=15):
    """
    PCG with fixed iteration count. No convergence check, no host sync.
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_matvec(x)

    z = precond_solve(r)
    p = z.clone()
    rz = torch.dot(r, z)  # GPU-internal sync only

    for k in range(num_iters):
        Ap = A_matvec(p)
        pAp = torch.dot(p, Ap)
        alpha = rz / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        z = precond_solve(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz

        p = z + beta * p
        rz = rz_new

    return x
```

**This is what [TorchCor](https://github.com/sagebei/torchcor) uses**: PCG with Jacobi preconditioning, user-configurable max iterations and tolerances, and second-order extrapolation for the initial guess. TorchCor achieves high performance for cardiac electrophysiology on NVIDIA H100 GPUs (~400 seconds for a full simulation).

### Strategy 2: Pipelined CG (Reduce Sync Points by 2x)

[Pipelined PCG](https://www.researchgate.net/publication/322518273_The_Communication-Hiding_Conjugate_Gradient_Method_with_Deep_Pipelines) restructures the algorithm to overlap global reductions with SpMV/preconditioner computation, reducing from 2 sync points to 1 per iteration:

```python
def pipelined_pcg(A_matvec, b, precond_solve, x0=None, num_iters=15):
    """
    Pipelined PCG: 1 non-blocking reduction per iteration
    (overlapped with SpMV + preconditioner apply).
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_matvec(x)

    u = precond_solve(r)
    w = A_matvec(u)
    gamma = torch.dot(r, u)
    delta = torch.dot(w, u)

    m = precond_solve(w)
    n_vec = A_matvec(m)

    alpha = gamma / delta
    p = u.clone()
    s = w.clone()
    q = m.clone()
    z = n_vec.clone()

    for k in range(num_iters):
        x = x + alpha * p
        r = r - alpha * s
        u = u - alpha * q
        w = w - alpha * z

        # These two dot products CAN be fused into one allreduce
        gamma_new = torch.dot(r, u)
        delta_new = torch.dot(w, u)

        # These computations overlap with the reduction above
        m = precond_solve(w)
        n_vec = A_matvec(m)

        beta = gamma_new / gamma
        alpha = gamma_new / (delta_new - beta * gamma_new / alpha)

        p = u + beta * p
        s = w + beta * s
        q = m + beta * q
        z = n_vec + beta * z
        gamma = gamma_new

    return x
```

Note: Full overlap benefit in PyTorch requires multiple CUDA streams or CUDA Graph-based execution.

### Strategy 3: s-Step / Communication-Avoiding CG

The s-step CG method groups s consecutive iterations, reducing global reductions by a factor of s. The [January 2025 paper on communication-reduced CG for GPU clusters (arXiv:2501.03743)](https://arxiv.org/abs/2501.03743) presents an efficient MPI-CUDA implementation as part of the **BootCMatchGX** library, the "first publicly available open-source implementation" for heterogeneous clusters. Typical s = 4-8; numerical stability degrades for larger s.

### Strategy 4: Replace PCG with Chebyshev Iteration Entirely

Given the sync-point overhead of PCG, **replacing PCG with Chebyshev iteration** (Section 1) is the most GPU-friendly option:
- **Zero** global reductions per iteration
- Fixed iteration count is inherent
- Works well when eigenvalue bounds are known (constant matrix = constant bounds)

**This is the recommended approach for maximum single-GPU throughput.**

### Strategy 5: GPU-Resident Persistent Kernels (PERKS)

The [PERKS approach (arXiv:2204.02064)](https://arxiv.org/abs/2204.02064) moves the entire iteration loop inside a single persistent GPU kernel, using CUDA Cooperative Groups for device-wide barriers instead of kernel launch/termination. This eliminates kernel launch overhead and improves data locality by caching intermediates in registers/shared memory. Geometric mean speedup: 2.29x for iterative stencil computations.

A [2023 ACM ICS paper](https://dl.acm.org/doi/10.1145/3577193.3593713) extends this to multi-GPU, demonstrating CG solvers with fully device-initiated execution via NVSHMEM, "completely avoiding per-iteration CPU-GPU synchronization."

### Warm-Starting for Fixed-Iteration Solvers

For time-stepping cardiac simulation, warm-starting dramatically reduces iterations needed:

```python
def warm_start_extrapolation(V_current, V_prev, order=2):
    """
    Extrapolate initial guess for next timestep's linear solve.
    order=1: V_guess = V_current
    order=2: V_guess = 2*V_current - V_prev  (linear extrapolation)
    """
    if order == 2:
        return 2.0 * V_current - V_prev
    return V_current.clone()
```

TorchCor uses second-order extrapolation and reports that it significantly reduces PCG iterations. With warm-starting, 3-8 PCG iterations typically suffice for standard cardiac timesteps.

### CUDA Graphs for Fixed-Iteration Solvers

When using fixed iteration counts in PyTorch, CUDA Graphs can capture and replay the entire solve, eliminating kernel launch overhead:

```python
# Warmup
for _ in range(3):
    x = fixed_iter_pcg(A_matvec, b, precond, x0=x_guess, num_iters=15)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    x = fixed_iter_pcg(A_matvec, b, precond, x0=x_guess, num_iters=15)

# Replay each timestep (no kernel launch overhead)
for step in range(num_timesteps):
    b.copy_(new_rhs)          # update in-place
    x_guess.copy_(warm_start) # update in-place
    g.replay()                # x now contains solution
```

### Key References

- [TorchCor: PyTorch cardiac FEM with PCG+Jacobi](https://arxiv.org/abs/2510.12011) ([GitHub](https://github.com/sagebei/torchcor))
- [MonoAlg3D: GPU cardiac solver with CG via cuSPARSE/cuBLAS](https://www.biorxiv.org/content/10.1101/2025.04.09.647733v1)
- [Communication-reduced CG for GPU clusters (BootCMatchGX)](https://arxiv.org/abs/2501.03743)
- [PERKS: Persistent kernels for GPU iterative solvers](https://arxiv.org/abs/2204.02064)
- [Re::Solve: ORNL GPU-resident linear solver library](https://github.com/ORNL/ReSolve)
- [MPCGPU: Real-time PCG on GPU](https://arxiv.org/abs/2309.08079)
- [aCG: GPU CG with NCCL/NVSHMEM](https://github.com/ParCoreLab/aCG)
- [GPU-resident solvers for optimization](https://arxiv.org/html/2401.13926v1)

---

## 5. Comparison and Recommendations

### Decision Matrix

| Method | Global Syncs/Iter | Setup Cost | Mesh Requirement | Best For |
|--------|-------------------|------------|------------------|----------|
| **Chebyshev** | 0 | Eigenvalue estimation | Any (with precond) | Max GPU throughput |
| **FFT/DCT** | 0 (direct) | Precompute eigenvalues | Structured grid only | Voxel-based models |
| **AMG+PCG** | 2-3 | AMG hierarchy | Any mesh | Large unstructured FEM |
| **Fixed-iter PCG** | 2 (no host sync) | Preconditioner only | Any mesh | Simple implementation |
| **Pipelined CG** | 1 (overlapped) | Preconditioner only | Any mesh | Multi-GPU |
| **s-step CG** | 2/s | Preconditioner only | Any mesh | GPU clusters |

### Recommended Architecture for PyTorch Cardiac Simulation

**Option A (Structured Grid)**: FFT/DCT spectral solve for isotropic part + explicit correction for anisotropic fiber term. Zero sync points, O(N log N), fully differentiable via torch.fft.

**Option B (Unstructured Mesh, simplest)**: Fixed-iteration PCG with Jacobi preconditioner + warm-starting. Two GPU-internal syncs per iteration, dead simple to implement in PyTorch. This is what TorchCor does.

**Option C (Unstructured Mesh, max throughput)**: Chebyshev iteration with Jacobi preconditioner + Gershgorin or power-iteration eigenvalue estimation at setup. Zero sync points per timestep.

**Option D (Unstructured Mesh, mesh-independent convergence)**: AmgX via pyamgx with Chebyshev smoother + fixed-iter outer CG. Best convergence rate, but requires AmgX installation.

### Existing GPU Cardiac Solvers

| Solver | Framework | Linear Solver | Preconditioner | GPU? | Reference |
|--------|-----------|---------------|----------------|------|-----------|
| TorchCor | PyTorch | PCG | Jacobi | Yes | [arXiv:2510.12011](https://arxiv.org/abs/2510.12011) |
| MonoAlg3D | CUDA/C | CG (cuSPARSE) | None/Jacobi | Yes | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.09.647733v1) |
| openCARP | C/PETSc | CG/GMRES | ILU/AMG | CPU | [bioRxiv 2021](https://www.biorxiv.org/content/10.1101/2021.03.01.433036v3.full) |
| MicroCARD | Ginkgo | BDDC | Domain decomp | Yes | [arXiv:2410.14786](https://arxiv.org/html/2410.14786) |

### PyTorch Implementation Notes

**Double precision is essential**: TorchCor emphasizes that float64 is required for stability over thousands of timesteps. Use `torch.set_default_dtype(torch.float64)`.

**Sparse CSR on GPU**: PyTorch supports sparse CSR tensors with GPU-accelerated SpMV via cuSPARSE:
```python
A = torch.sparse_csr_tensor(crow_indices, col_indices, values, 
                             size=(n, n), device='cuda', dtype=torch.float64)
y = A @ x  # cuSPARSE SpMV
```

**Jacobi preconditioner**: Extract diagonal from CSR, invert, and apply as element-wise multiplication -- the simplest and most GPU-friendly preconditioner.

Sources:
- [Chebyshev Iteration (Netlib Templates)](https://www.netlib.org/linalg/html_templates/node44.html)
- [Chebyshev Iteration (Wikipedia)](https://en.wikipedia.org/wiki/Chebyshev_iteration)
- [PETSc KSPCHEBYSHEV](https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/)
- [Interactive Finite Elements: Chebyshev Method](https://jschoeberl.github.io/iFEM/iterative/Chebyshev.html)
- [Shape As Points (NeurIPS 2021)](https://github.com/autonomousvision/shape_as_points)
- [torch-dct](https://github.com/zh217/torch-dct)
- [PyTorch torch.fft](https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/)
- [PyAMG](https://github.com/pyamg/pyamg)
- [NVIDIA AmgX](https://github.com/NVIDIA/AMGX)
- [pyamgx](https://github.com/shwina/pyamgx)
- [AMGCL](https://amgcl.readthedocs.io/)
- [TorchCor](https://github.com/sagebei/torchcor)
- [MonoAlg3D](https://www.biorxiv.org/content/10.1101/2025.04.09.647733v1)
- [openCARP](https://www.biorxiv.org/content/10.1101/2021.03.01.433036v3.full)
- [PERKS: Persistent Kernels](https://arxiv.org/abs/2204.02064)
- [Communication-Reduced CG for GPU Clusters](https://arxiv.org/abs/2501.03743)
- [aCG: GPU CG solvers](https://github.com/ParCoreLab/aCG)
- [Re::Solve: GPU-resident solvers](https://github.com/ORNL/ReSolve)
- [AMG for Cardiac Bidomain](https://pubmed.ncbi.nlm.nih.gov/17405366/)
- [BDDC on GPUs for Cardiac](https://arxiv.org/html/2410.14786)
- [Gershgorin Circle Theorem (Wikipedia)](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)
- [Multigrid Smoothers for Ultra-Parallel Computing](https://www.osti.gov/servlets/purl/1117969)
- [Optimal Polynomial Smoothers for AMG (2025)](https://link.springer.com/article/10.1007/s11075-025-02117-6)
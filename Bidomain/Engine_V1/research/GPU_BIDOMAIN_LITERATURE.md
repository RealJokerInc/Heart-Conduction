# GPU-Accelerated Bidomain Solvers: Literature Review

**Date:** 2026-03-04
**Purpose:** Inform FDM-first bidomain engine architecture decisions

---

## Key Papers (2024-2026)

### 1. Comparison of AMG Bidomain Solvers on Hybrid CPU-GPU (Centofanti, Scacchi et al., 2024)
- **Source:** Computer Methods in Applied Mechanics and Engineering (arXiv:2311.13914)
- **Method:** FEM + BoomerAMG (Hypre) vs GAMG (PETSc)
- **GPU Strategy:** Hybrid CPU-GPU via PETSc with Hypre/AmgX backends. Elliptic solve offloaded to GPU.
- **Finding:** GPUs yield best absolute solution time. Detailed AMG parameter tuning results.
- **Relevance:** Confirms AMG is the production standard for bidomain. Our block preconditioner should support AMG sub-solver.

### 2. BDDC Preconditioning on GPUs for Cardiac Simulations (MicroCARD, 2024)
- **Source:** Euro-Par Workshops (arXiv:2410.14786)
- **Method:** FEM + BDDC (Balancing Domain Decomposition by Constraints) via Ginkgo framework
- **GPU Strategy:** Fully GPU-resident BDDC solver. Poisson converges in single CG iteration.
- **Finding:** 90%+ parallel efficiency on 100 GPUs. Portable across NVIDIA/AMD/Intel.
- **Relevance:** Domain decomposition alternative to AMG. Not applicable to our single-GPU PyTorch approach.

### 3. TorchCor: FEM on GPUs via PyTorch (Zhou, Balmus, Corrado et al., 2025)
- **Source:** arXiv:2510.12011, SoftwareX 2026. GitHub: github.com/sagebei/torchcor
- **Method:** FEM entirely in PyTorch. Supports monodomain AND bidomain.
- **GPU Strategy:** Single-GPU, no custom CUDA kernels. Uses PyTorch sparse linear algebra.
- **Finding:** Significantly outperforms openCARP (64-core CPU) for meshes >100K nodes.
- **Relevance:** **DIRECTLY RELEVANT.** Same technology stack (PyTorch). Proves PyTorch sparse is viable for bidomain. Should study their block system handling and preconditioner choices.

### 4. Novel Bidomain Partitioned Strategies with SDC (Gopika, Bastian, Chamakuri, 2025)
- **Source:** arXiv:2510.27447
- **Method:** FEM + partitioned (semi-decoupled) approach + spectral deferred correction
- **GPU Strategy:** Not GPU-specific, but the decoupled strategy is GPU-friendly.
- **Finding:** Accuracy comparable to fully coupled at cost of decoupled. Validated on 2D/3D with bath coupling.
- **Relevance:** The partitioned approach could reduce our block system to sequential N x N solves instead of 2N x 2N.

### 5. Matrix-Free SBP-SAT FD + Multigrid on GPUs (ACM ICS '24, 2024)
- **Source:** ACM International Conference on Supercomputing
- **Method:** FDM (summation-by-parts) + matrix-free geometric multigrid
- **GPU Strategy:** Matrix-free GPU kernels apply stencil on-the-fly. No sparse matrix assembly.
- **Finding:** **5x faster than SpMV approach** for 67M DOF. 3x faster stencil application alone.
- **Relevance:** **CRITICAL for our FDM approach.** Matrix-free stencil application on GPU eliminates sparse matrix overhead entirely. Geometric multigrid maps naturally to structured grids.

### 6. GPU Code Generation for openCARP (INRIA CARMEN, 2024)
- **Source:** HAL hal-04206195
- **Method:** Auto-generated GPU kernels for ionic model ODEs
- **Finding:** A100: 3.17x over vectorized CPU, 7.4x over baseline. 185 GFLOP/s.
- **Relevance:** Ionic model is embarrassingly parallel. PyTorch already handles this well via torch.compile.

### 7. Smoothed Boundary Bidomain Model (Biasi et al., 2023)
- **Source:** PLOS One (PMC10256234)
- **Method:** Smoothed boundary method (SBM) for anatomically detailed geometries on Cartesian grids
- **GPU Strategy:** Phase-field mask multiplied with FDM operators — entirely GPU-resident
- **Finding:** Handles complex anatomy on regular grids without body-fitted meshing. Accuracy comparable to FEM.
- **Relevance:** Phase-field approach could simplify our FDM boundary handling for non-rectangular domains.

### 8. Explicit Multirate Runge-Kutta-Chebyshev (emRKC) (Abdulle et al., 2024)
- **Source:** Journal of Computational Physics
- **Method:** Fully explicit multirate time integration for bidomain — **NO linear solves at all**
- **GPU Strategy:** Eliminates the most GPU-hostile operation (implicit linear solve). Each stage is an explicit stencil application.
- **Finding:** Stability achieved via extended Chebyshev polynomials (s~30-50 stages per step). Cost per step higher but embarrassingly parallel. Competitive with implicit for moderate accuracy on GPU.
- **Relevance:** **HIGH.** Could eliminate PCG/GMG entirely for the elliptic solve. Worth implementing as an alternative to implicit decoupled approach. Perfect GPU parallelism — every stage is a matrix-free stencil application.

### 9. openCARP GPU Backend (Plank et al., 2021-2024)
- **Source:** openCARP documentation + HAL hal-04206195
- **Method:** CARP production solver with GPU offloading via PETSc/AmgX
- **GPU Strategy:** Ionic ODE on GPU (embarrassingly parallel), elliptic solve via AmgX (NVIDIA AMG on GPU)
- **Finding:** 2460x speedup reported for ionic model step. Overall 10-50x speedup with GPU elliptic solve.
- **Relevance:** Confirms ionic step is trivially parallel. Their architecture validates our decoupled approach.

### 10. FFT/DCT Direct Elliptic Solve on Regular Grids
- **Source:** Classical numerical methods (Hockney & Eastwood, Swarztrauber)
- **Method:** For constant-coefficient Laplacian on regular grid, transform to spectral domain, divide by eigenvalues, transform back. O(N log N) direct solve.
- **GPU Strategy:** `torch.fft.rfft2` / `torch.fft.irfft2` — single forward + inverse FFT, no iterations.
- **Finding:** For isotropic conductivity on uniform grid: **exact solve in 2 FFT calls**. For anisotropic: use as preconditioner (1-2 PCG iterations).
- **Relevance:** **CRITICAL for isotropic test cases.** Our boundary speedup validation uses isotropic conductivity — FFT solve would be essentially free. For anisotropic production cases, FFT preconditioner inside PCG dramatically reduces iteration count.

### 11. torch.compile for Ionic Model Kernel Fusion (PyTorch 2.x)
- **Source:** PyTorch documentation + openCARP GPU code generation approach
- **Method:** `torch.compile()` fuses element-wise ionic model operations into single GPU kernel
- **GPU Strategy:** Eliminates kernel launch overhead for Rush-Larsen exponential updates. Single kernel for all 18-40 state variables.
- **Finding:** 2-5x speedup over eager mode for element-wise operations. Zero code changes required.
- **Relevance:** Free performance for ionic step. Should apply `@torch.compile` to ionic model `compute_rates()` and Rush-Larsen `step()`.

### 12. Temporal Blocking / Cache Optimization for Stencil Codes
- **Source:** Various HPC literature (Datta et al., Strzodka et al.)
- **Method:** Fuse multiple time steps of stencil application into single GPU kernel pass to maximize cache reuse
- **Finding:** 2-3x speedup for explicit stencil codes by reducing global memory traffic
- **Relevance:** Applicable if we use explicit time stepping (emRKC). For implicit PCG, less relevant since each iteration requires global sync.

---

## Key Architectural Insights

### 1. Everyone Decouples First
No production GPU bidomain solver uses the fully coupled 2N x 2N system. All decouple into:
- **Parabolic solve** for Vm (N x N, SPD, easy)
- **Elliptic solve** for phi_e (N x N, SPD, hard)
- Coupling handled via operator splitting or fixed-point iteration

**Implication for us:** Our block_linear_solver with MINRES is the "correct" approach but may not be the fastest. A decoupled approach with separate PCG solves for each N x N block could be simpler and faster on GPU.

### 2. Matrix-Free is the GPU Path for FDM
For structured grids (our FDM case), assembling the sparse matrix is wasteful:
- The stencil is regular and can be applied on-the-fly
- Matrix-free approach saves 4x memory
- 5x faster than SpMV on GPU (Paper 5)

**Implication for us:** Instead of building sparse K_i and K_e matrices, apply the stencil directly via `torch.nn.functional.conv2d` or custom CUDA kernels. This is how V5.4's explicit FDM already works.

### 3. Geometric Multigrid is Natural for FDM
On structured grids, geometric multigrid (GMG) has:
- No setup phase (unlike AMG)
- No coarsening heuristics
- Simple restriction/prolongation (averaging/bilinear interpolation)
- Constant memory per level
- Zero sync in smoothing (Chebyshev smoother)

**Implication for us:** For FDM bidomain, GMG is superior to AMG. We should implement a simple 2-level or V-cycle GMG as the sub-solver inside our block preconditioner.

### 4. PyTorch Sparse Works for Bidomain (TorchCor)
TorchCor proves that PyTorch's sparse tensor infrastructure is sufficient for bidomain FEM. For FDM (structured grids), we have even better options:
- Dense 2D tensors with conv2d for stencil application
- No sparse matrix construction needed
- torch.linalg for direct sub-block solves at coarse levels

### 5. The Elliptic Solve Dominates
The phi_e elliptic equation accounts for 60-80% of bidomain computation time. Optimizing this one solve determines overall performance.

**Implication:** Focus GPU optimization effort on the elliptic solver. The parabolic (Vm) part can use the same approach as monodomain.

### 6. FFT as Direct Solver or Preconditioner
For constant-coefficient Laplacian on regular grids, FFT/DCT provides O(N log N) **direct** solve — no iterations needed. For variable coefficients (anisotropic conductivity), FFT of the constant-coefficient part serves as a near-perfect preconditioner, reducing PCG to 1-3 iterations.

**Implication:** For our isotropic test cases (boundary speedup validation), the elliptic solve becomes essentially free via `torch.fft.rfft2`. For anisotropic production cases, FFT preconditioner + PCG is likely faster than geometric multigrid.

### 7. Fully Explicit Methods Eliminate Linear Solves
The emRKC method (Paper 8) uses extended Chebyshev stability polynomials to make explicit time stepping stable for parabolic PDEs. Each "stage" is just a stencil application — no inner iteration, no sync points, no preconditioner.

**Implication:** Worth implementing as Phase 5+ optimization. Trade-off: more stages per step (~30-50) but each stage is trivially parallel. Could outperform implicit PCG on GPU where sync costs dominate.

### 8. torch.compile is Free Performance
Applying `@torch.compile` to ionic model computation fuses multiple element-wise GPU kernels into one. This is the PyTorch equivalent of openCARP's GPU code generation approach.

**Implication:** Apply to Rush-Larsen step and ionic model rate computation. Expected 2-5x speedup on ionic step with zero code changes.

---

## Recommended Architecture for FDM Bidomain on GPU

Based on the literature, the optimal FDM bidomain architecture is:

```
1. Ionic step (embarrassingly parallel, GPU-native)
   - Rush-Larsen or Forward Euler on Vm
   - Same as monodomain, no change

2. Parabolic step (Vm update, moderate difficulty)
   - Matrix-free stencil application for D_i * Laplacian(Vm)
   - Matrix-free stencil for D_i * Laplacian(phi_e) coupling
   - Implicit: PCG with GMG preconditioner (N x N, SPD)
   - OR explicit: Forward Euler (CFL-limited, simpler)

3. Elliptic step (phi_e solve, expensive)
   - Matrix-free stencil for (D_i + D_e) * Laplacian(phi_e)
   - RHS: -D_i * Laplacian(Vm) (from step 2's stencil)
   - Option A: FFT direct solve (isotropic, O(N log N), ~0 iterations)
   - Option B: PCG + FFT preconditioner (anisotropic, 1-3 iterations)
   - Option C: PCG + geometric multigrid (general, 10-25 iterations)
   - Null space pinning: phi_e(corner) = 0

Key: steps 2 and 3 are DECOUPLED N x N solves, not a coupled 2N x 2N system

Alternative: emRKC fully explicit approach (no linear solves at all)
   - 30-50 explicit stages per step, each = stencil application
   - Perfect GPU parallelism, no sync points
   - Trade-off: more FLOPs but zero solver overhead
```

### Why Decoupled > Coupled for GPU FDM

| Aspect | Coupled (MINRES 2N) | Decoupled (PCG N + PCG N) |
|--------|---------------------|---------------------------|
| Matrix size | 2N x 2N | 2 x (N x N) |
| Matrix structure | Indefinite (needs MINRES) | SPD (can use PCG) |
| Preconditioner | Block preconditioner (complex) | GMG on each block (simple) |
| GPU sync points | 2-3 per MINRES iter | 2-3 per PCG iter, but smaller |
| Matrix-free | Harder (block matvec) | Natural (separate stencils) |
| Memory | 4x matrix storage | 2x matrix storage |
| Implementation | Complex | Simpler |
| Iteration count | 15-30 (MINRES) | 10-25 (PCG each) |
| Accuracy | Exact coupling | O(dt) splitting error |

**Recommendation:** Implement decoupled approach first (simpler, GPU-friendly), add coupled MINRES as option for accuracy-critical applications.

---

## Sources

- [AMG Bidomain Solvers Comparison (arXiv)](https://arxiv.org/abs/2311.13914)
- [BDDC on GPUs for Cardiac (arXiv)](https://arxiv.org/abs/2410.14786)
- [TorchCor: PyTorch FEM Cardiac (arXiv)](https://arxiv.org/abs/2510.12011)
- [TorchCor GitHub](https://github.com/sagebei/torchcor)
- [Partitioned Bidomain with SDC (arXiv)](https://arxiv.org/abs/2510.27447)
- [Matrix-Free FD + Multigrid GPU (ACM ICS '24)](https://dl.acm.org/doi/10.1145/3650200.3656614)
- [GPU Code Gen for openCARP (INRIA)](https://inria.hal.science/hal-04206195v1/document)
- [Smoothed Boundary Bidomain (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256234/)
- [emRKC: Explicit Multirate for Bidomain (JCP 2024)](https://doi.org/10.1016/j.jcp.2024.112806)
- [openCARP GPU backend documentation](https://opencarp.org/documentation)
- [PyTorch torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)

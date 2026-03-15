# Research Summary: Spatial Discretization & Solvers for Cardiac Monodomain Simulation

## Overview

This research folder contains findings from investigating openCARP's architecture, FDM/FVM methods, LBM-EP, and GPU-friendly linear solvers -- mapped to the TODO items in `Engine_V5.3/improvement.md`.

---

## Downloaded Code Examples (`../code_examples/`)

### Cardiac Simulators
| Repo | Description | Language | Size |
|------|-------------|----------|------|
| `torchcor/` | PyTorch FEM cardiac EP (PCG+Jacobi, Crank-Nicolson) | Python/PyTorch | 72M |
| `MonoAlg3D_C/` | Cell-centered FVM cardiac EP on GPU (CG via cuSPARSE) | C/CUDA | 514M |

### LBM Frameworks
| Repo | Description | Language | Size |
|------|-------------|----------|------|
| `lettuce/` | PyTorch LBM framework (BGK/MRT, D2Q9/D3Q19) | Python/PyTorch | 4.5M |
| `lbm/` | Simple Python LBM reference implementation | Python | 276M |
| `list-lattice-Boltzmann-codes/` | Curated index of all open-source LBM codes | Markdown | 156K |

### Linear Solvers
| Repo | Description | Language | Size |
|------|-------------|----------|------|
| `pyamg/` | Algebraic multigrid (SA, RS) for prototyping | Python/NumPy | 9.3M |
| `pyamgx/` | Python bindings for NVIDIA AmgX (GPU-native AMG) | Python/C | 14M |
| `amgcl/` | Header-only C++ AMG with CUDA/VexCL backends | C++ | 8.5M |
| `ReSolve/` | ORNL GPU-resident linear solver library | C++/CUDA | 52M |

### FFT/Spectral Solvers
| Repo | Description | Language | Size |
|------|-------------|----------|------|
| `torch-dct/` | DCT/IDCT for PyTorch (Neumann BC FFT solver) | Python/PyTorch | 196K |
| `shape_as_points/` | Differentiable spectral Poisson solver in PyTorch | Python/PyTorch | 50M |
| `poisson-dirichlet-neumann/` | NumPy FFT Poisson solver (Dirichlet+Neumann reference) | Python/NumPy | 136K |

---

## Research Reports

| # | File | Topic | Size |
|---|------|-------|------|
| 01 | `01_FDM_Stencils_and_Implementation.md` | 5-point/9-point stencils, anisotropic diffusion, Neumann BC, sparse assembly, PyTorch construction | ~7K |
| 02 | `02_openCARP_FDM_FVM_Architecture.md` | openCARP architecture, monodomain equation, FDM vs FVM, operator splitting, existing tool comparison | ~24K |
| 03 | `03_GPU_Linear_Solvers.md` | Chebyshev iteration, FFT/DCT solvers, AMG (AmgX/PyAMG/AMGCL), fixed-iter PCG, pipelined CG | ~33K |
| 04 | `04_LBM_EP_Implementation.md` | LBM-EP paper analysis, D2Q5/D3Q7 lattices, BGK/MRT collision, Rush-Larsen coupling, full PyTorch blueprint | ~43K |

---

## TODO → Research Mapping

### Spatial Discretization

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `fdm.py` | Researched | 01, 02 | Use `F.conv2d` with 9-point kernel for anisotropic diffusion. Replicate padding for Neumann BC. For spatially varying fibers, use per-node tensor fields with element-wise ops or decompose into 3 convolutions. |
| `fvm.py` | Researched | 02 | Cell-centered FVM with face flux computation. Harmonic mean for interface conductivity at scar boundaries. Existing Engine_V5.1 `_apply_uniform_anisotropic()` is already a correct FVM implementation. Reference: MonoAlg3D. |
| Conductivity tensor (anisotropic D) | Researched | 01, 02 | `D = R * diag(D_fiber, D_cross) * R^T` giving `Dxx, Dxy, Dyy` from fiber angle. 9-point stencil required when `Dxy != 0`. M-matrix condition: `|Dxy| <= min(Dxx*dy/(2*dx), Dyy*dx/(2*dy))`. |

### Temporal Discretization

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `forward_euler.py` | Researched | 01, 02 | `V^{n+1} = V^n + dt/Cm * L*V^n`. CFL: `dt <= Cm*h^2/(4*D_max)`. Typical cardiac: dt=0.01ms, h=0.025cm gives CFL=0.064 (stable). Compatible with ionic model dt requirements. |
| `rk2.py` (Heun's) | Partially | 02 | 2nd-order explicit. Two Laplacian evaluations per step. Allows ~2x larger dt than FE for same accuracy. Standard implementation. |
| `rk4.py` | Partially | 02 | 4th-order explicit. Four evaluations per step. Overkill for cardiac diffusion but useful for reference solutions. |

### Linear Solvers

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `chebyshev.py` | Researched | 03 | **Zero global reductions** per iteration -- ideal for GPU. Needs eigenvalue bounds via Gershgorin or power iteration. 3-term recurrence: only SpMV + vector updates. Full PyTorch implementation provided. |
| `multigrid.py` (AMG) | Researched | 03 | NVIDIA AmgX via pyamgx for GPU-native AMG. PyAMG for CPU prototyping. Setup once, reuse hierarchy across timesteps. Chebyshev smoother preferred over Gauss-Seidel on GPU. |
| `fft.py` | Researched | 03 | Periodic BC: `torch.fft.fftn`. Neumann BC: DCT via `torch-dct` package. O(N log N). Only works for isotropic diffusion on structured grids; use as preconditioner for anisotropic case. Handle zero eigenvalue for pure Neumann. |

### Mesh Types

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `structured.py` | Researched | 01, 02 | StructuredGrid stores `(Nx, Ny, dx, dy)` + domain mask + fiber angle field. Voxel-based from segmented images. Enables FDM/FVM/LBM. |
| `tetrahedral.py` | Not researched | -- | Out of scope (FEM-specific, openCARP uses CARP format). |

### Tissue Types

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `anisotropic.py` | Researched | 01, 02 | Fiber angle field `theta(x,y)` → tensor components. Transmural rotation (epicardium to endocardium). Rule-based fiber assignment from Laplace-Dirichlet fields. |
| `heterogeneous.py` (scar) | Researched | 01, 02 | **Harmonic mean** for face conductivity at scar boundaries: `D_face = 2*D_left*D_right/(D_left+D_right)`. Gives zero flux at D=0 (scar). Arithmetic mean gives D/2 (unphysical). |

### LBM

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `lbm/state.py` | Researched | 04 | Distribution tensor `f: (Q, Ny, Nx)`, voltage `v: (Ny, Nx)`, gate `h: (Ny, Nx)`, domain mask, fiber field. SoA layout for coalesced GPU access. |
| `lbm/monodomain.py` | Researched | 04 | Full simulation loop: collide → stream → bounce-back → update voltage. Source term embedded in collision (not operator splitting). Full PyTorch blueprint provided. |
| `lbm/lattice.py` | Researched | 04 | Velocity vectors, weights, opposite directions, c_s^2. |
| `lbm/d2q7.py` / `d3q7.py` | Researched | 04 | D2Q5 for 2D diffusion-only (not D2Q9). D3Q7 for 3D. Weights: w0=1/4, w1-6=1/8 (3D). |
| `lbm/collision.py` | Researched | 04 | **MRT recommended** over BGK for anisotropic diffusion. 5x5 transformation matrix M. Relaxation rates tied to diffusion tensor: `s_i = 1/(0.5 + D_i*dt/(c_s^2*dx^2))`. Off-diagonal coupling for fiber rotation. |

### Builder Integration

| TODO | Status | Report | Key Findings |
|------|--------|--------|--------------|
| `mesh/from_image.py` | Not researched | -- | Needs Builder's segmented image → structured grid conversion. Pixel groups → domain mask + tissue labels. |
| `stimulus/from_image.py` | Not researched | -- | Needs Builder's stim regions → stimulus protocol mapping. |

---

## Key External Tools & References

### Open-Source Cardiac Simulators

| Tool | Method | Language | GPU | Link |
|------|--------|----------|-----|------|
| openCARP | FEM (Galerkin) | C++/PETSc | CPU | [opencarp.org](https://opencarp.org/) |
| MonoAlg3D | FVM (cell-centered) | C/CUDA | Yes | [github.com/rsachetto/MonoAlg3D_C](https://github.com/rsachetto/MonoAlg3D_C) |
| TorchCor | FEM (PyTorch CSR) | Python | Yes | [github.com/sagebei/torchcor](https://github.com/sagebei/torchcor) |

### LBM Frameworks (Adaptable for Cardiac)

| Tool | Language | GPU | Link |
|------|----------|-----|------|
| Lettuce | Python/PyTorch | Yes | [github.com/lettucecfd/lettuce](https://github.com/lettucecfd/lettuce) |
| XLB | Python/JAX | Yes | [arxiv.org/html/2311.16080v3](https://arxiv.org/html/2311.16080v3) |
| OpenLB | C++ | MPI | [openlb.net](https://www.openlb.net/) |

### GPU Linear Solver Libraries

| Tool | Language | AMG | Chebyshev | Link |
|------|----------|-----|-----------|------|
| NVIDIA AmgX | C/CUDA | Yes | Yes (smoother) | [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) |
| pyamgx | Python | Yes | Yes | [github.com/shwina/pyamgx](https://github.com/shwina/pyamgx) |
| PyAMG | Python/NumPy | Yes | No | [github.com/pyamg/pyamg](https://github.com/pyamg/pyamg) |
| AMGCL | C++ header-only | Yes | Yes | [amgcl.readthedocs.io](https://amgcl.readthedocs.io/) |
| torch-dct | Python/PyTorch | N/A | N/A | [github.com/zh217/torch-dct](https://github.com/zh217/torch-dct) |

---

## Recommended Implementation Priority

Based on the research and existing codebase:

### Phase 1: FDM on Structured Grid (builds on Engine_V5.1)
1. **`structured.py`** - StructuredGrid class with domain mask + fiber angles
2. **`fdm.py`** - 9-point anisotropic stencil via `F.conv2d`, replicate padding for Neumann BC
3. **`forward_euler.py`** - Explicit diffusion stepping (already effectively implemented in Engine_V5.1)

### Phase 2: GPU Solvers
4. **`chebyshev.py`** - Sync-free polynomial solver (zero dot products per iteration)
5. **`fft.py`** - DCT-based direct solver for structured grids with Neumann BC

### Phase 3: FVM + Heterogeneous Tissue
6. **`fvm.py`** - Cell-centered flux with harmonic mean at interfaces
7. **`heterogeneous.py`** - Scar tissue support with D=0 boundaries
8. **`anisotropic.py`** - Fiber orientation from Builder or rule-based assignment

### Phase 4: LBM
9. **`lbm/`** - Full LBM-EP implementation using D2Q5 lattice with MRT collision
10. Start from Lettuce framework patterns, add cardiac-specific components

---

## Key Numerical Parameters (2D Reference)

```
D_fiber  = 0.001   cm^2/ms    (along fibers)
D_cross  = 0.00025 cm^2/ms    (across fibers, ratio 4:1)
C_m      = 1.0     uF/cm^2    (membrane capacitance)
chi      = 1400    cm^-1      (surface-to-volume ratio)
h        = 0.025   cm         (250 um grid spacing)
dt       = 0.01    ms         (time step)
CFL      = 0.064              (well within stability limit < 1)
```

LBM parameters:
```
tau_fiber = 0.5 + 3*D_fiber*dt/dx^2 = 0.548   (> 0.5, stable)
tau_cross = 0.5 + 3*D_cross*dt/dx^2 = 0.512   (> 0.5, stable)
```

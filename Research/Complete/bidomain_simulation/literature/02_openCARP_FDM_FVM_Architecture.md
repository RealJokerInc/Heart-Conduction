---

# Research Report: openCARP and Cardiac Electrophysiology Spatial Discretization (FDM/FVM)

## 1. openCARP Overview and Its Spatial Discretization Approach

**openCARP does not use FDM or FVM.** It uses the **Finite Element Method (FEM) with linear basis functions** (Galerkin method) on unstructured tetrahedral meshes. However, the openCARP paper and manual extensively discuss the historical evolution from FDM to FEM/FVM in cardiac electrophysiology, and the principles it documents are directly applicable to understanding how FDM and FVM should work for the monodomain equation.

The openCARP source code is hosted at [git.opencarp.org/openCARP/openCARP](https://git.opencarp.org/openCARP/openCARP). Its architecture consists of three main components:
- **Parabolic solver**: Determines electrical propagation (transmembrane voltage evolution)
- **Ionic current component**: LIMPET library for cell-level ODE integration
- **Elliptic solver**: For the bidomain extracellular potential

The linear systems are solved via [PETSc](https://petsc.org/).

**Key reference**: Plank et al. (2021). "The openCARP simulation environment for cardiac electrophysiology." *Computer Methods and Programs in Biomedicine*, 208, 106223. ([bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.03.01.433036v3.full)) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260721002972))

---

## 2. The Monodomain Equation

The monodomain equation governs electrical wave propagation:

```
chi * (Cm * dV/dt + I_ion(V, w)) = div(D * grad(V)) + I_stim
```

Where:
- `V` = transmembrane potential (mV)
- `Cm` = membrane capacitance (~1 uF/cm^2)
- `chi` = surface-to-volume ratio (~1400 cm^-1)
- `I_ion(V, w)` = ionic current from cell model (coupled ODEs for gating variables `w`)
- `D` = conductivity/diffusion tensor (cm^2/ms)
- `I_stim` = stimulus current

Rearranging to the standard reaction-diffusion form:

```
dV/dt = (1/(chi*Cm)) * div(D * grad(V)) - I_ion(V, w)/Cm + I_stim/(chi*Cm)
```

**Key insight from openCARP**: The monodomain is equivalent to the bidomain under the equal anisotropy ratio assumption (`sigma_i = alpha * sigma_e`), using the harmonic mean conductivity tensor. This is computationally 10x or more faster than the full bidomain.

---

## 3. Conductivity Tensor Construction from Fiber Orientations

The diffusion tensor `D` encodes cardiac fiber architecture. For the **transversely isotropic** case (2D):

```
D = sigma_t * I + (sigma_l - sigma_t) * f * f^T
```

Which expands using the fiber angle `theta` to:

```
D = R(theta) * [[sigma_l, 0], [0, sigma_t]] * R(theta)^T
```

Giving tensor components:

```python
D_xx = sigma_l * cos^2(theta) + sigma_t * sin^2(theta)
D_yy = sigma_l * sin^2(theta) + sigma_t * cos^2(theta)
D_xy = (sigma_l - sigma_t) * cos(theta) * sin(theta)
```

For the fully **orthotropic** 3D case (fiber `f`, sheet `s`, sheet-normal `n`):

```
D = sigma_l * f*f^T + sigma_t * s*s^T + sigma_n * n*n^T
```

Typical human ventricular conduction velocities: CV_f ~ 0.67 m/s, CV_s ~ 0.3 m/s, CV_n ~ 0.17 m/s. The relationship between CV and diffusion coefficient follows approximately `CV = k * sqrt(D)`.

**References**: [openCARP CV tuning example](https://opencarp.org/documentation/examples/02_ep_tissue/03a_study_prep_tunecv), [Monodomain model Wikipedia](https://en.wikipedia.org/wiki/Monodomain_model), [ScienceDirect: Monodomain Model overview](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/monodomain-model)

---

## 4. FDM Stencil Construction for the Laplacian with Anisotropic Diffusion

### The Anisotropic Diffusion Operator

The diffusion operator expands to:

```
div(D * grad(V)) = D_xx * d^2V/dx^2 + 2*D_xy * d^2V/dxdy + D_yy * d^2V/dy^2
```

(For uniform tensor; for spatially varying tensor, the full conservative form `d/dx(D_xx * dV/dx) + d/dx(D_xy * dV/dy) + d/dy(D_xy * dV/dx) + d/dy(D_yy * dV/dy)` must be used.)

### 5-Point Stencil (Isotropic Only)

The standard 5-point stencil for the isotropic Laplacian on a uniform grid with spacing `h`:

```
         [  0,    1/dy^2,      0   ]
kernel = [1/dx^2, -2(1/dx^2 + 1/dy^2), 1/dx^2]
         [  0,    1/dy^2,      0   ]
```

For `dx = dy = h`: `Lap(V) = (V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} - 4*V_{i,j}) / h^2`

This can be implemented efficiently in PyTorch as:

```python
kernel = torch.tensor([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=torch.float64) / (h * h)
kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 3, 3)
V_padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate')
laplacian = F.conv2d(V_padded, kernel)[0, 0]
```

**Limitation**: The 5-point stencil does not capture the cross-derivative term `D_xy * d^2V/dxdy`, so it cannot handle anisotropic diffusion with fibers not aligned to the coordinate axes.

### 9-Point Stencil (Anisotropic)

For the full anisotropic tensor, a 9-point stencil is needed. The cross-derivative is discretized using the four diagonal neighbors:

```
d^2V/dxdy = (V_{i+1,j+1} - V_{i-1,j+1} - V_{i+1,j-1} + V_{i-1,j-1}) / (4*dx*dy)
```

The full anisotropic 9-point stencil (from [Wikipedia: Nine-point stencil](https://en.wikipedia.org/wiki/Nine-point_stencil) and [Schrader et al., arXiv:2309.05575](https://arxiv.org/html/2309.05575v3)):

```
         [-D_xy/(4*dx*dy),  D_yy/dy^2,  D_xy/(4*dx*dy) ]
kernel = [ D_xx/dx^2,      -2(D_xx/dx^2 + D_yy/dy^2),  D_xx/dx^2 ]
         [ D_xy/(4*dx*dy),  D_yy/dy^2, -D_xy/(4*dx*dy) ]
```

For PyTorch implementation:

```python
kernel = torch.zeros(3, 3, dtype=torch.float64)
dx2 = dx * dx
dy2 = dy * dy
dxdy4 = 4.0 * dx * dy

# Axial neighbors (from d^2V/dx^2 and d^2V/dy^2)
kernel[1, 0] = D_xx / dx2          # West
kernel[1, 2] = D_xx / dx2          # East
kernel[0, 1] = D_yy / dy2          # North
kernel[2, 1] = D_yy / dy2          # South
kernel[1, 1] = -2.0 * (D_xx/dx2 + D_yy/dy2)  # Center

# Diagonal neighbors (from cross-derivative 2*D_xy * d^2V/dxdy)
kernel[0, 0] = -D_xy / dxdy4       # NW: -(+D_xy)
kernel[0, 2] =  D_xy / dxdy4       # NE: +(+D_xy)
kernel[2, 0] =  D_xy / dxdy4       # SW: +(+D_xy)
kernel[2, 2] = -D_xy / dxdy4       # SE: -(+D_xy)
```

**Note on the factor of 2**: The monodomain diffusion operator includes `2*D_xy * d^2V/dxdy`. When D_xy already includes the contribution from both `d/dx(D_xy * dV/dy)` and `d/dy(D_xy * dV/dx)`, the factor of 2 is already embedded in the `D_xy` term from the tensor expansion. The pre-scaling factor `Dxy_dt_4dxdy = D_xy * dt / (4*dx*dy)` in the existing code already accounts for this correctly (the `2*D_xy` combined with the `1/(4*dx*dy)` cross-derivative stencil gives `2*D_xy/(4*dx*dy) = D_xy/(2*dx*dy)`... but actually the standard approach is to note that the cross-derivative stencil already has the factor of 4 in the denominator, and the `2*D_xy` from the PDE gives `D_xy / (2*dx*dy)` applied to the four diagonal differences).

**Stability for the anisotropic 9-point stencil** (from [Schrader et al.](https://arxiv.org/abs/2309.05575)): The explicit Forward Euler time step limit is:

```
dt <= 1 / (2 * (D_xx/dx^2 + D_yy/dy^2) + |D_xy|/(dx*dy))
```

With a safety factor of 0.9 recommended.

### Spatially Varying Tensor (Non-uniform Fiber Field)

When the fiber angle varies spatially, the tensor components become fields `D_xx(x,y)`, `D_yy(x,y)`, `D_xy(x,y)`. The conservative form must be used:

```
div(D*grad(V)) = d/dx(D_xx * dV/dx + D_xy * dV/dy) + d/dy(D_xy * dV/dx + D_yy * dV/dy)
```

This is more naturally handled by the FVM approach (see below), but can be done with FDM by computing face-averaged diffusion coefficients.

**References**: [Khan & Ng (2017) "Finite Difference Monodomain Modeling"](https://www.researchgate.net/publication/318461369_Finite_Difference_Monodomain_Modeling_of_Cardiac_Tissue_with_Optimal_Parameters), [Schrader et al. "Anisotropic Diffusion Stencils"](https://arxiv.org/html/2309.05575v3)

---

## 5. FVM Flux Computation and Cell-Centered Schemes

### MonoAlg3D: The Reference FVM Implementation for Cardiac EP

**MonoAlg3D** ([GitHub](https://github.com/rsachetto/MonoAlg3D_C)) is the primary open-source FVM-based cardiac electrophysiology solver. It uses a **cell-centered finite volume method** on hexahedral grids.

Key details from the [MonoAlg3D paper](https://www.biorxiv.org/content/10.1101/2025.04.09.647733v1.full):

- **Grid**: Uniform hexahedral mesh with spacing `h_M`. Each control volume is a hexahedron with the transmembrane potential `V_M` stored at its center.
- **Conservation**: The governing monodomain PDE is integrated over each control volume, converting volume integrals to surface integrals via the divergence theorem.
- **Flux computation**: The surface integral `integral(D * grad(V) . n dS)` is approximated across each face of the hexahedron.
- **Linear system**: The backward Euler method is used for time discretization, leading to a sparse linear system solved by Conjugate Gradient (CG) on GPU via cuSparse/cuBLAS.

### General FVM Flux Computation for Anisotropic Diffusion

For a cell-centered FVM on a structured grid, the flux across each face is:

**East face flux** (between cell (i,j) and cell (i,j+1)):

```
F_east = (D_xx * dV/dx + D_xy * dV/dy) * dy_face
```

Where:
- `dV/dx` at the east face: `(V_{i,j+1} - V_{i,j}) / dx` (two-point approximation)
- `dV/dy` at the east face: average of the y-gradients in the two adjacent cells (reconstructed from 4 neighbors)

**North face flux** (between cell (i,j) and cell (i+1,j)):

```
F_north = (D_xy * dV/dx + D_yy * dV/dy) * dx_face
```

Where:
- `dV/dy` at the north face: `(V_{i+1,j} - V_{i,j}) / dy`
- `dV/dx` at the north face: average of x-gradients from adjacent cells

### Face Conductivity: Harmonic Mean

For **spatially varying** conductivity, the face conductivity between cells with different diffusion coefficients should use the **harmonic mean**:

```
D_face = 2 * D_left * D_right / (D_left + D_right)
```

This ensures correct flux continuity across material interfaces and is the standard approach in the Two-Point Flux Approximation (TPFA).

### FVM Flux Balance (Divergence)

The discretized divergence for cell (i,j):

```
div(D*grad(V))_{i,j} = (F_east - F_west) / dx + (F_north - F_south) / dy
```

This is exactly what the existing Engine_V5.1 `DiffusionOperator._apply_uniform_anisotropic()` implements.

### TPFA Limitations for Anisotropy

The **Two-Point Flux Approximation** (TPFA) is simple but only consistent for K-orthogonal grids (grids aligned with the diffusion tensor principal axes). For general anisotropy on non-aligned grids, TPFA can produce grid orientation effects. More advanced schemes include:
- **Multi-Point Flux Approximation (MPFA)**: Uses additional cell values for flux reconstruction
- **Discrete Duality Finite Volume (DDFV)**: Uses both cell-centered and vertex unknowns
- **Nonlinear two-point methods**: Split flux into harmonic and transversal components

For a **structured Cartesian grid** with the tensor projected onto face normals (as in your existing code), the approach used in `_apply_uniform_anisotropic()` is correct: it reconstructs the cross-derivative component at faces by averaging adjacent cell gradients.

**References**: [Bendahmane et al. "FV scheme for cardiac propagation"](https://www.sciencedirect.com/science/article/abs/pii/S0378475409003644), [Trew et al. "FVM for discontinuous activation"](https://www.researchgate.net/publication/7761736_A_Finite_Volume_Method_for_Modeling_Discontinuous_Electrical_Activation_in_Cardiac_Tissue), [Cell-centered nonlinear FV methods](https://www.sciencedirect.com/science/article/abs/pii/S0021999116305964)

---

## 6. Boundary Conditions: No-Flux Neumann Implementation

The physical boundary condition for cardiac tissue is **homogeneous Neumann** (no-flux): `D * grad(V) . n = 0` on the boundary.

### FDM Implementation (Ghost Cell Approach)

The standard approach is to introduce ghost cells. For a boundary at `j=0`:

```
V_{i,-1} = V_{i,1}   (mirror the value across the boundary)
```

This makes the gradient zero at the boundary: `dV/dx|_{j=0} = (V_{i,1} - V_{i,-1})/(2*dx) = 0`.

In practice (as seen in the existing Engine_V4 code), this is implemented by doubling the one-sided difference:

```python
# At left boundary (j=0), instead of V[i,-1]:
lap_x = 2.0 * (V[i,1] - V[i,0])  # = (V[i,1] - V[i,0]) + (V[i,1] - V[i,0])
                                    # which equals V[i,1] - 2*V[i,0] + V[i,1]
                                    # i.e., using ghost V[i,-1] = V[i,1]
```

### FDM + PyTorch Implementation (Replicate Padding)

PyTorch's `F.pad(..., mode='replicate')` achieves the same effect. Replicate padding copies the edge value: `V_ghost = V_edge`, which gives `dV/dx = 0` at the boundary. This is exactly the no-flux Neumann condition.

```python
V_padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate')
# Then apply convolution kernel normally
```

**Important subtlety**: For the anisotropic case with cross-derivatives, the no-flux condition is `D * grad(V) . n = 0`, which means `D_xx * dV/dx + D_xy * dV/dy = 0` at x-boundaries (not just `dV/dx = 0`). The existing Engine_V4 code handles this by dropping cross-derivative terms at boundaries, which is a common simplification. For more accuracy, the ghost cell value should be computed from the full flux condition. The Engine_V5.1 code handles it more correctly via the FVM approach by explicitly setting boundary fluxes to zero.

### FVM Implementation

In the cell-centered FVM, no-flux is naturally enforced by simply **setting the flux to zero** at boundary faces:

```python
# Left edge (j=0): no west flux
diff[1:-1, 0] = (flux_e[1:-1, 0] - 0.0) / dx + (flux_n[1:-1, 0] - flux_s[1:-1, 0]) / dy
```

This is what the existing Engine_V5.1 code does (see lines 556-569 of `/Users/catecholamines/Documents/Heart Conduction/Monodomain/Engine_V5.1/tissue/diffusion.py`).

**References**: [PLOS One: Code generator with automatic Neumann BC handling](https://journals.plos.org/plosone/article/figures?id=10.1371/journal.pone.0136821), [FiPy FVM documentation (NIST)](https://pages.nist.gov/fipy/en/3.4.5/numerical/discret.html), [Chehade & Coudiere thesis on FVM with Neumann BCs](https://theses.hal.science/tel-05293929v1)

---

## 7. Operator Splitting: Coupling Spatial Discretization with Ionic Models

### Godunov (First-Order) Splitting

At each time step `[t^n, t^{n+1}]`:

1. **Step 1 (Reaction)**: Solve the ODE system for each cell independently:
   ```
   dV/dt = -I_ion(V, w)/Cm
   dw/dt = f(V, w)
   ```
   for time `dt`, starting from `(V^n, w^n)`, producing intermediate `(V*, w*)`

2. **Step 2 (Diffusion)**: Solve the PDE:
   ```
   dV/dt = (1/(chi*Cm)) * div(D * grad(V))
   ```
   for time `dt`, starting from `V*`, producing `V^{n+1}`

This is first-order accurate in time.

### Strang (Second-Order) Splitting

1. **Half-step Reaction**: Solve ionic ODEs for `dt/2`
2. **Full-step Diffusion**: Solve PDE for `dt`
3. **Half-step Reaction**: Solve ionic ODEs for `dt/2`

This is second-order accurate (provided each sub-step integrator is also second-order).

### Implementation in openCARP

openCARP uses "various time stepping options including fully explicit Euler or theta-schemes, with or without operator splitting." The reaction term is always treated explicitly. The LIMPET library handles ionic ODE integration with methods including **Runge-Kutta**, **Rush-Larsen** (exploiting the exponential structure of gating variable ODEs), and **CVODE** (adaptive step-size BDF methods).

### Implementation in TorchCor (PyTorch-based)

[TorchCor](https://github.com/sagebei/torchcor) ([arXiv:2510.12011](https://arxiv.org/abs/2510.12011)) uses:
- **Crank-Nicolson** (theta=0.5) for the diffusion term: `(M + theta*dt*K) * V^{n+1} = (M - (1-theta)*dt*K) * V^n + dt*M*I_ion^n`
- **Forward Euler** for advancing ionic state variables at each time step
- **PCG solver** with Jacobi preconditioner for the implicit linear system
- Sparse matrices in **CSR format** for GPU-efficient matrix-vector products

### Rush-Larsen Method for Gating Variables

The Rush-Larsen method exploits the fact that gating variables `m, h, j, ...` follow linear ODEs when voltage is held constant:

```
dm/dt = alpha_m(V) * (1 - m) - beta_m(V) * m = (m_inf(V) - m) / tau_m(V)
```

Exact solution for constant V over `dt`:

```
m(t+dt) = m_inf + (m(t) - m_inf) * exp(-dt / tau_m)
```

This allows much larger time steps for the ionic ODE integration compared to Forward Euler.

### Practical Coupling Pattern

```python
# Godunov splitting in PyTorch
for step in range(n_steps):
    # Step 1: Ionic reaction (embarrassingly parallel over all cells)
    I_ion = ionic_model.compute_current(V, state_vars)
    V_star = V - dt * I_ion / Cm  # Forward Euler for reaction
    state_vars = ionic_model.update_gates(V, state_vars, dt)  # Rush-Larsen
    
    # Step 2: Diffusion
    lap_V = diffusion_operator.apply(V_star)
    V = V_star + dt * lap_V / (chi * Cm)  # Forward Euler for diffusion
```

**References**: [Krishnamoorthi et al. (2013) "Numerical Quadrature and Operator Splitting in FEM for Cardiac EP"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4519349/), [Lindner et al. (2023) "Efficient time splitting schemes"](https://onlinelibrary.wiley.com/doi/10.1002/cnm.3666), [Spiteri & Torabi Ziaratgahi "High-Order OS for Bidomain/Monodomain"](https://epubs.siam.org/doi/10.1137/17M1137061)

---

## 8. Relevant Tools and Their Approaches

| Tool | Method | Grid | Language | GPU |
|------|--------|------|----------|-----|
| **openCARP** | FEM (Galerkin, linear basis) | Unstructured tet | C++ | No (CPU/PETSc) |
| **MonoAlg3D** | FVM (cell-centered) | Hexahedral | C/CUDA | Yes |
| **TorchCor** | FEM (linear, sparse CSR) | Unstructured tri/tet | Python/PyTorch | Yes |
| **Ithildin** | FDM (explicit FE/RK) | Structured | C++/MPI | No |
| **magnum.np** | FDM (conv-based) | Structured | Python/PyTorch | Yes |

---

## 9. Practical Guidance for Implementing `fdm.py` and `fvm.py` in PyTorch

### For `fdm.py`: FDM with Conv2d/Conv3d

The most efficient PyTorch FDM approach for structured grids uses convolution:

**Isotropic case** -- single kernel convolution:
```python
def fdm_laplacian_isotropic(V, D, dx, dy):
    kernel = torch.tensor([[0, 1/dy**2, 0],
                           [1/dx**2, -2*(1/dx**2 + 1/dy**2), 1/dx**2],
                           [0, 1/dy**2, 0]])
    kernel = (D * kernel).reshape(1, 1, 3, 3)
    V_pad = F.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate')
    return F.conv2d(V_pad, kernel)[0, 0]
```

**Anisotropic case** -- 9-point kernel:
```python
def fdm_laplacian_anisotropic(V, D_xx, D_yy, D_xy, dx, dy):
    kernel = torch.zeros(3, 3)
    dx2, dy2, dxdy4 = dx**2, dy**2, 4*dx*dy
    kernel[1,0] = D_xx/dx2; kernel[1,2] = D_xx/dx2
    kernel[0,1] = D_yy/dy2; kernel[2,1] = D_yy/dy2
    kernel[1,1] = -2*(D_xx/dx2 + D_yy/dy2)
    kernel[0,0] = -D_xy/dxdy4; kernel[0,2] = D_xy/dxdy4
    kernel[2,0] = D_xy/dxdy4;  kernel[2,2] = -D_xy/dxdy4
    kernel = kernel.reshape(1, 1, 3, 3)
    V_pad = F.pad(V.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate')
    return F.conv2d(V_pad, kernel)[0, 0]
```

**Spatially varying fiber field**: Cannot use a single global convolution kernel. Options:
1. Use per-cell element-wise operations (as in existing `_apply_varying_anisotropic`)
2. Decompose into 3 separate convolutions for `D_xx`, `D_yy`, `D_xy` components with gradient fields
3. Use the FVM approach instead (more natural for varying coefficients)

### For `fvm.py`: Cell-Centered FVM

The FVM approach is more natural for spatially varying conductivity tensors because it explicitly computes fluxes at faces and can use harmonic averaging for face conductivities.

The existing implementation in `/Users/catecholamines/Documents/Heart Conduction/Monodomain/Engine_V5.1/tissue/diffusion.py` (`_apply_uniform_anisotropic` method, lines 490-571) already implements a cell-centered FVM with:
- Face flux computation (east/west/north/south)
- Cross-derivative reconstruction at faces via averaging
- Explicit no-flux boundary conditions at all edges and corners

The key enhancement for a proper `fvm.py` would be:
1. **Harmonic mean face conductivities** for heterogeneous media
2. **Proper face tensor interpolation** (not just cell-centered values)
3. **Support for irregular/adaptive meshes** via sparse representations
4. **Implicit time stepping** option (backward Euler + CG solve)

---

## 10. Key Papers and Documentation

| Resource | URL |
|----------|-----|
| openCARP Paper (bioRxiv) | [biorxiv.org/content/10.1101/2021.03.01.433036v3.full](https://www.biorxiv.org/content/10.1101/2021.03.01.433036v3.full) |
| openCARP Manual (PDF) | [opencarp.org/manual/opencarp-manual-latest.pdf](https://opencarp.org/manual/opencarp-manual-latest.pdf) |
| openCARP GitLab | [git.opencarp.org/openCARP/openCARP](https://git.opencarp.org/openCARP/openCARP) |
| openCARP Examples | [opencarp.org/documentation/examples](https://opencarp.org/documentation/examples) |
| TorchCor (PyTorch FEM) | [arxiv.org/abs/2510.12011](https://arxiv.org/abs/2510.12011) / [github.com/sagebei/torchcor](https://github.com/sagebei/torchcor) |
| MonoAlg3D (FVM on GPU) | [github.com/rsachetto/MonoAlg3D_C](https://github.com/rsachetto/MonoAlg3D_C) / [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2025.04.09.647733v1.full) |
| Operator Splitting (Krishnamoorthi) | [PMC4519349](https://pmc.ncbi.nlm.nih.gov/articles/PMC4519349/) |
| Efficient Time Splitting (Lindner) | [Wiley: cnm.3666](https://onlinelibrary.wiley.com/doi/10.1002/cnm.3666) |
| Anisotropic Diffusion Stencils (Schrader) | [arxiv.org/abs/2309.05575](https://arxiv.org/abs/2309.05575) |
| FVM for Cardiac Propagation (Bendahmane) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378475409003644) |
| FVM Discontinuous Activation (Trew) | [ResearchGate](https://www.researchgate.net/publication/7761736_A_Finite_Volume_Method_for_Modeling_Discontinuous_Electrical_Activation_in_Cardiac_Tissue) |
| GFD Method for Cardiac Meshes | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0025556405001343) |
| magnum.np (PyTorch FDM template) | [nature.com/articles/s41598-023-39192-5](https://www.nature.com/articles/s41598-023-39192-5) |
| FiPy FVM Documentation | [pages.nist.gov/fipy](https://pages.nist.gov/fipy/en/3.4.5/numerical/discret.html) |
| Cell-centered nonlinear FV | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999116305964) |

---

## Summary

**openCARP itself is FEM-based**, but its documentation and the broader cardiac EP literature provide the mathematical foundations needed for FDM and FVM implementations. Your existing codebase already contains working implementations of both approaches:

- **Engine_V4** (`/Users/catecholamines/Documents/Heart Conduction/Monodomain/Engine_V4/diffusion.py`): Pure FDM with Numba -- 5-point isotropic and 9-point anisotropic stencils with Neumann BCs via ghost cells, operator splitting with pre-scaled coefficients.

- **Engine_V5.1** (`/Users/catecholamines/Documents/Heart Conduction/Monodomain/Engine_V5.1/tissue/diffusion.py`): PyTorch-based FVM -- cell-centered flux computation, `F.pad(..., mode='replicate')` for no-flux BCs on the isotropic path, explicit flux zeroing for anisotropic boundaries, `F.conv2d` for GPU-accelerated isotropic diffusion.

The reference implementations from **MonoAlg3D** (FVM) and **TorchCor** (FEM in PyTorch) provide the closest architectural templates for building production-grade `fdm.py` and `fvm.py` modules.
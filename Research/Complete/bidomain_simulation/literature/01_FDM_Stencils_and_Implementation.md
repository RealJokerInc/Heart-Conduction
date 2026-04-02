# Finite Difference Method for the Cardiac Monodomain Equation on Structured Grids

## Comprehensive Research Report

---

## 1. The Cardiac Monodomain Equation

The monodomain equation governing cardiac electrical propagation is:

```
C_m * dV/dt = div(D * grad(V)) - I_ion(V, w) + I_stim
```

where:
- `V(x, y, t)` is the transmembrane potential
- `C_m` is the membrane capacitance (typically 1.0 uF/cm^2)
- `D` is the conductivity tensor (possibly anisotropic, spatially varying)
- `I_ion` is the ionic current from the cell model
- `I_stim` is the external stimulus current

On a structured 2D grid with spacing `dx`, `dy`, the spatial discretization using FDM transforms the PDE into a system of ODEs.

---

## 2. Five-Point vs Nine-Point Stencils

### 2.1 Standard 5-Point Stencil (Isotropic or Axis-Aligned Anisotropy)

For a diagonal diffusion tensor `D = [[Dxx, 0], [0, Dyy]]`, the standard 5-point Laplacian stencil discretizes `div(D * grad(V))` as:

```
           Dyy/dy^2
              |
Dxx/dx^2 -- center -- Dxx/dx^2
              |
           Dyy/dy^2
```

The discrete operator at node `(i, j)`:

```
L*V[i,j] = Dxx*(V[i+1,j] - 2*V[i,j] + V[i-1,j]) / dx^2
          + Dyy*(V[i,j+1] - 2*V[i,j] + V[i,j-1]) / dy^2
```

This gives 2nd-order accuracy O(dx^2 + dy^2) and handles only axis-aligned anisotropy (no cross-derivative terms).

### 2.2 Nine-Point Stencil (Full Anisotropic Tensor)

When the diffusion tensor has off-diagonal terms (fibre orientation not aligned with grid axes):

```
D = [[Dxx, Dxy],
     [Dxy, Dyy]]
```

The full anisotropic diffusion `div(D * grad(V))` expands to:

```
Dxx * d^2V/dx^2 + 2*Dxy * d^2V/dxdy + Dyy * d^2V/dy^2
```

The cross-derivative term `d^2V/dxdy` requires a 9-point stencil:

```
d^2V/dxdy ~ (V[i+1,j+1] - V[i+1,j-1] - V[i-1,j+1] + V[i-1,j-1]) / (4*dx*dy)
```

Full 9-point stencil coefficients at node `(i,j)`:

```
NW = +Dxy / (4*dx*dy)          N = Dyy / dy^2           NE = -Dxy / (4*dx*dy)
W  = Dxx / dx^2                C = -2*(Dxx/dx^2 + Dyy/dy^2)   E  = Dxx / dx^2
SW = -Dxy / (4*dx*dy)          S = Dyy / dy^2           SE = +Dxy / (4*dx*dy)
```

**Positivity / M-matrix condition:**
```
|Dxy| <= min(Dxx * dy / (2*dx), Dyy * dx / (2*dy))
```

### 2.3 Rotated Anisotropic Stencil

Diffusion tensor from fibre orientation:
```
D = R^T * [[D_fibre, 0], [0, D_cross]] * R
```

```python
def diffusion_tensor_from_fibres(theta, D_fibre, D_cross):
    c = np.cos(theta)
    s = np.sin(theta)
    Dxx = D_fibre * c**2 + D_cross * s**2
    Dyy = D_fibre * s**2 + D_cross * c**2
    Dxy = (D_fibre - D_cross) * s * c
    return Dxx, Dxy, Dyy
```

---

## 3. Conservative Form Discretization (Spatially Varying D)

For spatially varying `D(x, y)`:
```
div(D * grad(V)) = d/dx(Dxx * dV/dx + Dxy * dV/dy) + d/dy(Dxy * dV/dx + Dyy * dV/dy)
```

Flux at x-interface `(i+1/2, j)`:
```
Fx[i+1/2, j] = Dxx[i+1/2,j] * (V[i+1,j] - V[i,j]) / dx
             + Dxy[i+1/2,j] * (V[i,j+1] + V[i+1,j+1] - V[i,j-1] - V[i+1,j-1]) / (4*dy)
```

Interface D values via **harmonic mean** (preferred for sharp heterogeneity):
```
Dxx[i+1/2, j] = 2 * Dxx[i,j] * Dxx[i+1,j] / (Dxx[i,j] + Dxx[i+1,j])
```

Center coefficient = -sum(off-diagonal) ensures conservation.

---

## 4. No-Flux (Neumann) Boundary Conditions

### Ghost Node Method

For left boundary at `i = 0` with `n = (-1, 0)`:
- Isotropic: `V[-1, j] = V[1, j]`
- Anisotropic: `V[-1, j] = V[1, j] + (Dxy/Dxx) * dx/dy * (V[0, j+1] - V[0, j-1])`

### Modified Stencil (Preferred)

Redistribute ghost node contributions to in-bounds neighbors. No extra storage needed.

---

## 5. Sparse Matrix Assembly

### Vectorized Assembly (Production Code)

```python
def assemble_laplacian_vectorized(Nx, Ny, h, Dxx, Dxy, Dyy):
    N = Nx * Ny
    h2 = h * h
    dxx, dxy, dyy = Dxx.ravel(), Dxy.ravel(), Dyy.ravel()
    idx = np.arange(N).reshape(Ny, Nx)
    rows_list, cols_list, vals_list = [], [], []

    def add_entry(r, c, v):
        mask = np.isfinite(v)
        rows_list.append(r[mask]); cols_list.append(c[mask]); vals_list.append(v[mask])

    # East/West/North/South + 4 diagonals (see full code)
    # ...
    # Diagonal: negative row sum (ensures conservation)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag_vals = -np.array(L.sum(axis=1)).ravel()
    L = L + sparse.diags(diag_vals, 0, shape=(N, N), format='csr')
    return L
```

### PyTorch Sparse Construction

```python
def build_laplacian_torch(Nx, Ny, h, Dxx, Dxy, Dyy, device='cuda'):
    # Build COO indices and values using vectorized operations
    # Convert to torch.sparse_coo_tensor with .coalesce()
    # Diagonal = negative row sum via scatter_add_
    pass
```

---

## 6. Heterogeneous Conductivity

| Method | Formula | Best For |
|--------|---------|----------|
| Arithmetic mean | `(D_i + D_{i+1}) / 2` | Smooth D(x) |
| **Harmonic mean** | `2*D_i*D_{i+1} / (D_i + D_{i+1})` | **Sharp interfaces, scar tissue** |

Key: Harmonic mean gives zero flux at D=0 boundaries (scar), arithmetic mean gives D/2 (unphysical).

---

## 7. Time Stepping

### Forward Euler (Explicit)
```
V^{n+1} = V^n + dt/C_m * L * V^n
CFL: dt <= C_m * h^2 / (4 * D_max)
```

### Crank-Nicolson (Implicit, 2nd order)
```
(I - dt/(2*C_m) * L) * V^{n+1} = (I + dt/(2*C_m) * L) * V^n
```

### Recommendation
Forward Euler with operator splitting for cardiac: CFL constraint (~0.01-0.1 ms) is compatible with ionic model dt requirements.

---

## 8. Summary of Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Stencil type | 9-point for anisotropic, 5-point if Dxy=0 | Cross-derivatives require diagonal neighbors |
| Interface averaging | Harmonic mean for Dxx, Dyy | Correct flux at scar boundaries |
| Cross-term averaging | Arithmetic mean for Dxy | Adequate for smooth fibre fields |
| Boundary conditions | Modified stencil (ghost elimination) | No extra storage |
| Sparse format | COO for assembly, CSR for computation | Standard practice |
| Time stepping | Forward Euler with operator splitting | Simple, CFL-compatible with ionic dt |
| Conservation | Center = -sum(off-diagonal) | Ensures constant field has zero Laplacian |

### Typical Cardiac Parameters (2D)
```
D_fibre  = 0.001   cm^2/ms
D_cross  = 0.00025 cm^2/ms
C_m      = 1.0     uF/cm^2
h        = 0.025   cm (250 um)
dt       = 0.01    ms
CFL      = 0.064 < 1 (stable)
```

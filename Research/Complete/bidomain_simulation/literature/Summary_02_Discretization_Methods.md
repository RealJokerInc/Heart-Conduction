# Discretization Methods — FDM, FVM, FEM for Cardiac EP

## Overview

Cardiac electrophysiology simulation requires spatial discretization of the bidomain equations, which govern voltage and current propagation through ventricular tissue. Three primary methods are employed: Finite Difference Method (FDM), Finite Volume Method (FVM), and Finite Element Method (FEM). Each approach offers different trade-offs in computational efficiency, ease of implementation, and capability to handle complex tissue geometry and anisotropy.

---

## 1. Finite Difference Method (FDM)

### 1.1 5-Point Stencil (Isotropic Case)

The simplest FDM stencil applies to isotropic diffusion (uniform conduction in all directions). The standard Laplacian approximation is:

```
Laplacian(V) = [V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} - 4·V_{i,j}] / h²
```

**Key characteristics:**
- Applies when fiber cross-diffusion is negligible (Dxy = 0)
- Handles only axis-aligned diffusion tensors
- Second-order accurate with spacing h
- Easily implemented via convolution (PyTorch `F.conv2d`)

**PyTorch Implementation:**
```python
# 3×3 Laplacian kernel
kernel = torch.tensor([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=torch.float32) / (h**2)

# Apply with replicate padding for Neumann BC
V_laplacian = F.conv2d(V.unsqueeze(0), kernel.view(1,1,3,3),
                       padding=1, padding_mode='replicate')
```

### 1.2 9-Point Stencil (Anisotropic Case)

When fibers are not aligned with the grid (Dxy ≠ 0), the 9-point stencil is required to accurately capture cross-derivative terms.

**Cross-Derivative Approximation:**
```
∂²V/∂x∂y ≈ [V_{i+1,j+1} - V_{i+1,j-1} - V_{i-1,j+1} + V_{i-1,j-1}] / (4·dx·dy)
```

**Full 9-Point Stencil Coefficients:**

```
        NW              N               NE
        |               |               |
     -Dxy/(4·dx·dy) | Dyy/dy²   | +Dxy/(4·dx·dy)
        |               |               |
    ----+----------- ----+----------- ----+-----
        |               |               |
    Dxx/dx²      | -2(Dxx/dx² + Dyy/dy²) | Dxx/dx²
    (W)          |        (C = Center)        | (E)
        |               |               |
    ----+----------- ----+----------- ----+-----
        |               |               |
     +Dxy/(4·dx·dy) | Dyy/dy²   | -Dxy/(4·dx·dy)
        |               |               |
       SW              S               SE
```

**M-Matrix Condition (Ensuring Stability):**
The 9-point stencil represents an M-matrix (monotone, non-oscillatory solutions) only if:
```
|Dxy| ≤ min(Dxx·dy/(2·dx), Dyy·dx/(2·dy))
```

For cardiac tissue with typical anisotropy ratios (4:1 to 10:1), this condition is usually satisfied on adequately refined grids.

**Explicit Time-Stepping CFL Constraint:**
```
dt ≤ 1 / [2·(Dxx/dx² + Dyy/dy²) + |Dxy|/(dx·dy)]
```

The cross-derivative term increases the explicit stability requirement.

### 1.3 Fiber Orientation to Diffusion Tensor

Cardiac tissue exhibits fiber-aligned conduction (fast) and cross-fiber conduction (slow). Given fiber angle θ and longitudinal/transverse diffusivities D_fiber and D_cross:

```
D = R^T · diag(D_fiber, D_cross) · R

where rotation matrix R = [cosθ, -sinθ; sinθ, cosθ]
```

**Resulting Tensor Components:**
```
Dxx = D_fiber·cos²θ + D_cross·sin²θ
Dyy = D_fiber·sin²θ + D_cross·cos²θ
Dxy = (D_fiber - D_cross)·sinθ·cosθ
```

For example:
- θ = 0° (fibers along x-axis): Dxx = D_fiber, Dyy = D_cross, Dxy = 0
- θ = 45°: Dxx = Dyy = (D_fiber + D_cross)/2, Dxy = (D_fiber - D_cross)/2

### 1.4 Conservative Form for Spatially Varying Diffusion

When conductivity varies in space (e.g., scar tissue with D = 0), the conservative divergence form is essential:

```
div(D·∇V) = ∂/∂x(Dxx·∂V/∂x + Dxy·∂V/∂y) + ∂/∂y(Dxy·∂V/∂x + Dyy·∂V/∂y)
```

**Interface Conductivity via Harmonic Mean:**
```
D_{i+1/2} = 2·D_i·D_{i+1} / (D_i + D_{i+1})
```

**Why Harmonic Mean?**
- Arithmetic mean: D_avg = (D_i + D_{i+1})/2 → at scar boundary with D_{i+1} = 0, gives D_avg = D_i/2 (unphysical flux)
- Harmonic mean: correctly yields zero flux when either side is zero
- Preserves conservation: sum of fluxes across all faces = 0

**Center Coefficient:**
The center stencil coefficient must equal the negative sum of all off-diagonal coefficients to ensure local conservation:
```
C = -(NW + N + NE + W + E + SW + S + SE)
```

### 1.5 Neumann Boundary Conditions

Neumann (zero-flux) BCs are natural for cardiac tissue domains.

**Ghost Node Method (Isotropic):**
```
V_{ghost} = V_{-1,j} = V_{1,j}
```
This makes the gradient at the boundary zero.

**Ghost Node Method (Anisotropic):**
When cross-diffusion is present, the ghost node formula becomes:
```
V_{-1,j} = V_{1,j} + (Dxy/Dxx)·(dx/dy)·(V_{0,j+1} - V_{0,j-1})
```

**PyTorch Implementation:**
```python
# Replicate padding automatically enforces Neumann BC
V_padded = F.pad(V, (1, 1, 1, 1), mode='replicate')
```

**Modified Stencil Approach (Preferred):**
Instead of ghost nodes, redistribute their contributions to in-bounds neighbors. This avoids extra storage and is more elegant for heterogeneous domains.

---

## 2. Finite Volume Method (FVM)

### 2.1 Cell-Centered Scheme Overview

The FVM integrates the PDE over a control volume (cell), converting spatial derivatives to surface integrals via the divergence theorem:

```
∫∫_cell div(D·∇V) dA = ∮_∂cell (D·∇V)·n̂ ds
```

Results in a stencil equation balancing fluxes across all four faces (2D).

### 2.2 Two-Point Flux Approximation (TPFA)

For each face, compute the flux using only the two adjacent cell centers.

**East Face Flux:**
```
F_east = D_xx_face·(V_{i+1,j} - V_{i,j})/dx + D_xy_face·(∂V/∂y)_face
```

**Cross-Gradient Reconstruction:**
The normal gradient at a face is computed directly; the tangential gradient (∂V/∂y for vertical face) is reconstructed from a 4-point average:
```
(∂V/∂y)_east_face ≈ [(V_{i+1,j+1} + V_{i+1,j-1} + V_{i,j+1} + V_{i,j-1}) / 4 - V_{i,j}] / dy
```

**Divergence Assembly:**
```
div(D·∇V)_{i,j} = (F_east - F_west)/dx + (F_north - F_south)/dy
```

### 2.3 Face Conductivity

**Diagonal Terms (Dxx, Dyy):** Use harmonic mean
```
D_xx_face = 2·D_xx_left·D_xx_right / (D_xx_left + D_xx_right)
```

**Cross-Derivative Terms (Dxy):** Use arithmetic mean
```
D_xy_face = (D_xy_left + D_xy_right) / 2
```

The harmonic mean for diagonal terms ensures zero flux at scar boundaries; the arithmetic mean for cross terms provides adequate accuracy.

### 2.4 No-Flux Boundary Condition

A key advantage of FVM: Neumann BCs are natural. Simply set boundary face flux = 0.

```
F_boundary = 0
```

No ghost nodes required; the stencil is immediately at domain boundaries.

### 2.5 TPFA Consistency and Extensions

**For K-Orthogonal Grids:** TPFA is consistent (accuracy maintained as mesh refines)

**For Non-Orthogonal Grids:** Extended TPFA with cross-gradient reconstruction extends consistency to moderate anisotropy

**Cardiac Application:** On structured grids with typical anisotropy (4:1 to 10:1) and moderate element distortion, TPFA with 4-point cross-gradient reconstruction is sufficient.

### 2.6 Reference Implementation

**MonoAlg3D (open-source):**
- Cell-centered FVM on hexahedral meshes
- CG solver via cuSPARSE (GPU acceleration)
- Cross-gradient flux handling for bidomain equations

---

## 3. Finite Element Method (FEM)

### 3.1 Weak Formulation

Rather than discretize the strong form PDE pointwise, FEM derives the weak form by multiplying by test functions and integrating by parts (Green's formula):

**Strong form:**
```
∂V/∂t = div(D·∇V) + I_ion
```

**Weak form (multiply by test function φ, integrate over domain Ω):**
```
∫_Ω φ·(∂V/∂t) dΩ = -∫_Ω (D·∇φ)·∇V dΩ + ∫_Ω φ·I_ion dΩ + ∮_∂Ω (D·∇V)·n̂·φ ds
```

For Neumann BC (zero-flux), the boundary integral vanishes—this is a **natural boundary condition** in FEM, a major advantage.

### 3.2 Matrix Form

Expanding V in basis functions: V ≈ Σ_k u_k·φ_k

**Mass Matrix:**
```
M_kl = ∫_Ω φ_k·C_m·φ_l dΩ
```
(C_m is membrane capacitance per unit area)

**Stiffness Matrix:**
```
K_kl = ∫_Ω (D·∇φ_k)·∇φ_l dΩ
```

**Semi-discrete system:**
```
M·(du/dt) = -K·u + r
```

where r represents source terms (I_ion, boundary fluxes).

### 3.3 Element Assembly

For **linear triangular elements** (2D):

1. **Element stiffness matrix** (local assembly):
   ```
   K_e = A_e · B_e^T · D_e · B_e
   ```
   where:
   - A_e = triangle area
   - B_e = gradient matrix from nodal coordinates
   - D_e = diffusion tensor at element (fiber-rotated if applicable)

2. **B matrix construction** (for triangle with vertices (x_1, y_1), (x_2, y_2), (x_3, y_3)):
   ```
   B = 1/(2A) · [y_23  0  y_31  0  y_12  0  ]
               [0  x_32  0  x_13  0  x_21]
   ```
   (where x_ij = x_i - x_j, etc.)

3. **Global assembly:** scatter local K_e into global matrix via element-to-node connectivity

### 3.4 Mass Matrix Options

**Consistent Mass Matrix** (preserves accuracy):
```
For linear triangle: M_e = A_e · C_m/12 · [2  1  1]
                                          [1  2  1]
                                          [1  1  2]
```
- Sparse structure, requires solver
- Stable for explicit RK methods with small enough dt

**Lumped Mass Matrix** (diagonal, more efficient):
```
M_e_diag = A_e · C_m/3 · [1, 0, 0; 0, 1, 0; 0, 0, 1]  (sum row 2 to node 1, etc.)
```
- M^{-1} is trivial (point-wise division)
- Simplifies explicit time integration
- Slight loss of accuracy, but acceptable for monodomain/bidomain

---

## 4. Bidomain Extensions

The bidomain model solves two coupled equations: intracellular voltage V_i and extracellular voltage V_e (or transmembrane voltage V_m = V_i - V_e).

Both FDM and FVM require **two separate spatial operators**:

1. **A_i operator** (intracellular conductivity σ_i):
   ```
   ∇·(σ_i·∇V_i)
   ```
   Stencil/flux computed with σ_i

2. **A_sum operator** (total conductivity σ_i + σ_e):
   ```
   ∇·((σ_i + σ_e)·∇V_m)
   ```
   Stencil/flux computed with σ_i + σ_e

Both operators use identical stencil/flux machinery; only the conductivity tensor changes. **Harmonic means are computed independently** for σ_i and σ_sum.

**Example (FVM face):**
```
D_sum_face = 2·(σ_i + σ_e)_left · (σ_i + σ_e)_right / [(σ_i + σ_e)_left + (σ_i + σ_e)_right]
```

---

## 5. Comparison Table

| Aspect | FDM | FVM | FEM |
|--------|-----|-----|-----|
| **Mass matrix** | Identity | Diagonal (cell volumes) | Sparse (consistent or lumped) |
| **Mesh type** | Structured grid only | Structured/unstructured | Unstructured (triangles, tetrahedra) |
| **Anisotropy handling** | 9-pt stencil with M-matrix check | TPFA + cross-gradient reconstruction | Natural via element assembly (B^T·D·B) |
| **Neumann BC** | Ghost nodes or modified stencil | Flux = 0 at boundary (natural) | Natural in weak form (boundary integral vanishes) |
| **Conservation** | Local: center = -sum(off-diag) | Flux balance per cell | Variational principle |
| **Heterogeneity (scar)** | Harmonic mean at interfaces | Harmonic mean at faces | Per-element assembly (D varies per element) |
| **Implementation complexity** | Low | Medium | High (FE basis, assembly, solvers) |
| **Computational cost** | Low (convolution-like) | Medium | High (sparse matrix operations) |
| **Flexibility** | Low (grid-bound) | Medium | High (arbitrary geometry) |
| **Reference code** | — | MonoAlg3D | openCARP, TorchCor, CARPentry |

---

## 6. Connection to V5.4 Architecture

The V5.4 cardiac simulation framework abstracts discretization via the **SpatialDiscretization ABC** (Abstract Base Class):

### 6.1 SpatialDiscretization Interface

```python
class SpatialDiscretization(ABC):
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (nodes/cells)"""
        pass

    @property
    def coordinates(self) -> np.ndarray:
        """Node/cell center positions (n_dof, ndim)"""
        pass

    @property
    def mass_type(self) -> MassType:
        """IDENTITY, DIAGONAL, or SPARSE"""
        pass

    def get_diffusion_operators(self, fiber_angles, D_fiber, D_cross):
        """Returns DiffusionOperators dataclass"""
        pass

    def apply_diffusion(self, V, operators):
        """Computes div(D·∇V) directly"""
        pass
```

### 6.2 MassType Enum

```python
class MassType(Enum):
    IDENTITY = "fdm"       # FDM: M·v = v (no-op for diffusion)
    DIAGONAL = "fvm"       # FVM: M is diagonal (cell volumes)
    SPARSE   = "fem"       # FEM: sparse consistent or lumped
```

### 6.3 DiffusionOperators Dataclass

```python
@dataclass
class DiffusionOperators:
    A_lhs: scipy.sparse.spmatrix   # Stiffness matrix K (or CG matrix)
    B_rhs: scipy.sparse.spmatrix   # Right-hand side operator
    apply_mass: Callable           # Function to apply M (depends on mass_type)
```

For explicit time integration:
```
M·(dV/dt) = -A_lhs·V + rhs
→ (dV/dt) = M^{-1}·(-A_lhs·V + rhs)
```

### 6.4 Concrete Implementations

**StructuredGrid (FDM, FVM, LBM):**
- Regular Cartesian grid with nx, ny, nz
- `mass_type = MassType.IDENTITY` (FDM) or `MassType.DIAGONAL` (FVM)
- Fast convolution-based diffusion operators

**TriangularMesh (FEM):**
- Unstructured triangulation
- Basis functions φ_k (piecewise linear)
- `mass_type = MassType.SPARSE`
- Element-by-element assembly of K and M

**FEM Example:**
```python
mesh = TriangularMesh.load_from_vtk("ventricles.vtk")
mesh.set_fiber_field(fiber_angles)
diffusion_ops = mesh.get_diffusion_operators(
    fiber_angles, D_fiber=1.2, D_cross=0.26
)
# diffusion_ops.A_lhs is global stiffness matrix K
# diffusion_ops.apply_mass handles M^{-1} via consistent or lumped approach
```

---

## 7. Practical Selection Criteria

### Use FDM when:
- High-performance computing required (GPU via PyTorch convolutions)
- Regular ventricular geometry adequately approximated by structured grid
- Anisotropy is moderate (< 10:1) and fiber angles vary smoothly
- Memory is critical (implicit stencil coefficients vs. explicit matrix)

### Use FVM when:
- Structured/semi-structured mesh (refined near scar, coarser in normal tissue)
- Explicit control over flux conservation per cell desired
- Ease of heterogeneous conductivity (scars, ischemia)
- Moderate mesh irregularity tolerable

### Use FEM when:
- Arbitrary tissue geometry (complex infarcts, fiber re-entry zones)
- High accuracy needed (higher-order elements possible)
- Natural handling of Neumann BCs and complex boundaries
- Unstructured mesh generation tools available (Gmsh, Salome)

---

## 8. Summary

| Feature | Implementation |
|---------|-----------------|
| **Isotropic diffusion** | 5-pt stencil (FDM), TPFA (FVM), element assembly (FEM) |
| **Anisotropic diffusion** | 9-pt stencil with M-matrix check (FDM), TPFA + cross-gradient (FVM), B^T·D·B per element (FEM) |
| **Scar tissue (D→0)** | Harmonic mean ensures zero flux |
| **Bidomain** | Two operators (σ_i, σ_i+σ_e); same stencil logic |
| **Neumann BC** | Ghost nodes (FDM), natural (FVM, FEM) |
| **Mass matrix** | Identity (FDM), diagonal (FVM), sparse (FEM) |
| **Framework** | V5.4 SpatialDiscretization ABC with StructuredGrid, TriangularMesh |

All three methods, when properly implemented with appropriate stability checks and heterogeneity handling, yield accurate cardiac action potential propagation on physiologically realistic time scales (1–500 ms).

